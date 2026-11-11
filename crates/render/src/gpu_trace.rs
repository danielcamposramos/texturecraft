//! GPU path tracing for Rendering mode: the CPU tracer in [`crate::trace`] (the oracle) ported to
//! compute shaders (`shaders/trace.wgsl`, `shaders/trace_post.wgsl`), same light transport and
//! the same random streams, so both converge to the same image.
//!
//! The mesh, its BVH and the environment's importance-sampling tables live in storage buffers;
//! materials are copies of the renderer's display textures (`compose::display` packing, one array
//! layer per texture set plus one for the default material). [`GpuTracer::pass`] adds samples per
//! pixel to float accumulation buffers; [`GpuTracer::resolve`] runs the à-trous denoiser and tone
//! maps into an RGBA8 texture, so the app shows renders without reading them back.

use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use bytemuck::{Pod, Zeroable};
use texturecraft_compose::DisplayKind;
use texturecraft_geom::glam::{Vec3, Vec4};
use texturecraft_geom::{Mesh, OrbitCamera};
use wgpu::util::DeviceExt;

use crate::Renderer;
use crate::ibl::EnvImage;
use crate::trace::{TraceDisplay, TraceScene, TraceSettings};

/// Texture sets the tracer takes (one more array layer holds the default material).
pub const MAX_SETS: usize = 255;
/// Depth of the shader's BVH traversal stack (`STACK` in trace.wgsl).
pub(crate) const STACK: usize = 64;
const KINDS: usize = DisplayKind::ALL.len();
/// Bytes of the trace uniform: [`Head`] then 256 texture-set entries (`MAX_SETS` in trace.wgsl).
const PARAMS_SIZE: u64 = (size_of::<Head>() + 256 * 16) as u64;
/// Post uniform slots: resolve and denoise init, then the five à-trous steps.
const STEPS: [u32; 6] = [0, 1, 2, 4, 8, 16];
const WORKGROUP: u32 = 8;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Head {
    inv_vp: [f32; 16],
    forward: [f32; 4],
    size: [u32; 4],
    cfg: [u32; 4],
    env: [f32; 4],
    sigma: [f32; 4],
    eye: [f32; 4],
    lens: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostRaw {
    size: [u32; 4],
    display: [f32; 4],
}

struct Geometry {
    mesh: Arc<Mesh>,
    tri_set: Arc<Vec<usize>>,
    nodes: wgpu::Buffer,
    tris: wgpu::Buffer,
    verts: wgpu::Buffer,
}

struct EnvGpu {
    image: Arc<EnvImage>,
    view: wgpu::TextureView,
    cdf: wgpu::Buffer,
    /// Whether `cdf` holds importance-sampling tables.
    light: bool,
}

/// Copies of the renderer's display textures, one array per [`DisplayKind`].
struct Materials {
    rev: u64,
    arrays: Vec<(wgpu::Texture, wgpu::TextureView)>,
    /// Per layer: width, height, mask of kinds that are 1×1 placeholders.
    info: Vec<[f32; 3]>,
}

struct Accum {
    width: u32,
    height: u32,
    /// Radiance + luminance², albedo + hits, normal + depth, environment of misses.
    acc: [wgpu::Buffer; 4],
    /// Ping-pong demodulated radiance + variance for the denoiser.
    illum: [wgpu::Buffer; 2],
}

/// Scene constants of the trace uniform.
struct SceneConst {
    spec_gloss: bool,
    eps: f32,
    sigma: Vec3,
    rotation: f32,
    heights: Vec<f32>,
    ior: f32,
    /// Focus depth (0: the orbit target) and lens radius in world units (0: pinhole).
    lens: [f32; 2],
}

/// The GPU path tracer: pipelines, the uploaded scene and one view's accumulation.
pub struct GpuTracer {
    trace_bgl: wgpu::BindGroupLayout,
    post_bgl: wgpu::BindGroupLayout,
    trace_pipe: wgpu::ComputePipeline,
    init_pipe: wgpu::ComputePipeline,
    step_pipe: wgpu::ComputePipeline,
    resolve_pipe: wgpu::ComputePipeline,
    params: wgpu::Buffer,
    post: wgpu::Buffer,
    post_stride: u32,
    geometry: Option<Geometry>,
    env: Option<EnvGpu>,
    materials: Option<Materials>,
    acc: Option<Accum>,
    /// Keep the preceding size so motion/full-resolution switches can reuse allocations.
    spare_acc: Option<Accum>,
    /// How many sets of accumulation buffers were allocated (for tests).
    #[cfg(test)]
    acc_allocations: u32,
    scene: Option<SceneConst>,
    bind: Option<wgpu::BindGroup>,
    samples: u32,
}

/// The error a scope caught, if wgpu reported it at once (native backends do).
fn caught(f: impl Future<Output = Option<wgpu::Error>>) -> Option<wgpu::Error> {
    let mut f = std::pin::pin!(f);
    match f.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
        Poll::Ready(e) => e,
        Poll::Pending => None,
    }
}

/// Run `f` with validation and out-of-memory errors turned into an `Err` instead of wgpu's
/// default handler (which panics).
pub(crate) fn scoped<T>(device: &wgpu::Device, what: &str, f: impl FnOnce() -> T) -> Result<T, String> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    let errors = [caught(invalid.pop()), caught(oom.pop())];
    match errors.into_iter().flatten().next() {
        Some(e) => Err(format!("{what}: {e}")),
        None => Ok(out),
    }
}

/// What [`checked`] knows right after the call.
pub(crate) enum Checked<T> {
    /// wgpu answered at once (native backends, WebGL): the value, or what went wrong.
    Done(Result<T, String>),
    /// The browser's WebGPU answers later: the value, possibly invalid, and where the answer
    /// arrives (`None` for success). Don't use the value until it arrives. (Native backends
    /// never answer late; the renderer's tests build this variant by hand.)
    #[cfg_attr(not(target_arch = "wasm32"), allow(dead_code))]
    Later(T, std::sync::mpsc::Receiver<Option<String>>),
}

/// [`scoped`] for work whose errors may arrive later: never takes an unanswered scope as success.
pub(crate) fn checked<T>(device: &wgpu::Device, what: &str, f: impl FnOnce() -> T) -> Checked<T> {
    let oom = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let invalid = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let out = f();
    let (invalid, oom) = (invalid.pop(), oom.pop());
    let what = what.to_string();
    let mut answer = Box::pin(async move {
        let invalid = invalid.await;
        let oom = oom.await;
        // An allocation that ran out of memory also makes later calls on it invalid: the
        // out-of-memory error is the cause.
        oom.or(invalid).map(|e| format!("{what}: {e}"))
    });
    #[cfg(not(target_arch = "wasm32"))]
    {
        Checked::Done(match pollster::block_on(answer.as_mut()) {
            Some(e) => Err(e),
            None => Ok(out),
        })
    }
    #[cfg(target_arch = "wasm32")]
    {
        match answer.as_mut().poll(&mut Context::from_waker(Waker::noop())) {
            Poll::Ready(Some(e)) => Checked::Done(Err(e)),
            Poll::Ready(None) => Checked::Done(Ok(out)),
            Poll::Pending => {
                let (tx, rx) = std::sync::mpsc::channel();
                wasm_bindgen_futures::spawn_local(async move {
                    let _ = tx.send(answer.await);
                });
                Checked::Later(out, rx)
            }
        }
    }
}

pub(crate) fn entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::COMPUTE, ty, count: None }
}

pub(crate) fn storage(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Storage { read_only }, has_dynamic_offset: false, min_binding_size: None })
}

fn texture(binding: u32, view_dimension: wgpu::TextureViewDimension) -> wgpu::BindGroupLayoutEntry {
    entry(binding, wgpu::BindingType::Texture { sample_type: wgpu::TextureSampleType::Float { filterable: false }, view_dimension, multisampled: false })
}

pub(crate) fn buffer(device: &wgpu::Device, label: &str, contents: &[u8]) -> wgpu::Buffer {
    device.create_buffer_init(&wgpu::util::BufferInitDescriptor { label: Some(label), contents, usage: wgpu::BufferUsages::STORAGE })
}

fn bits(v: Vec3, w: u32) -> [u32; 4] {
    [v.x.to_bits(), v.y.to_bits(), v.z.to_bits(), w]
}

/// Depth of the deepest leaf (the root is 1), or None for a malformed tree.
pub(crate) fn bvh_depth(nodes: &[(Vec3, Vec3, u32, u32)]) -> Option<usize> {
    let mut stack = vec![(0usize, 1usize)];
    let mut deepest = 0;
    let mut visited = 0usize;
    while let Some((i, d)) = stack.pop() {
        visited += 1;
        let &(_, _, first, count) = nodes.get(i)?;
        deepest = deepest.max(d);
        if count == 0 && visited <= nodes.len() {
            stack.push((first as usize, d + 1));
            stack.push((first as usize + 1, d + 1));
        }
    }
    (visited <= nodes.len()).then_some(deepest)
}

impl GpuTracer {
    /// Whether `device` can run the tracer (compute shaders and enough bindings: not WebGL2).
    pub fn supported(device: &wgpu::Device) -> bool {
        let l = device.limits();
        l.max_storage_buffers_per_shader_stage >= 8
            && l.max_storage_textures_per_shader_stage >= 1
            && l.max_sampled_textures_per_shader_stage > KINDS as u32
            && l.max_compute_invocations_per_workgroup >= WORKGROUP * WORKGROUP
            && l.max_compute_workgroup_size_x >= WORKGROUP
            && l.max_compute_workgroup_size_y >= WORKGROUP
            && l.max_uniform_buffer_binding_size >= PARAMS_SIZE
            && l.max_texture_array_layers >= 2
    }

    pub fn new(device: &wgpu::Device) -> Result<GpuTracer, String> {
        if !Self::supported(device) {
            return Err("GPU path tracing needs compute shaders (WebGPU, Vulkan, Metal or DirectX 12)".into());
        }
        scoped(device, "GPU path tracing: pipelines", || Self::create(device))
    }

    fn create(device: &wgpu::Device) -> GpuTracer {
        let uniform = |binding, dynamic| {
            entry(binding, wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: dynamic, min_binding_size: None })
        };
        let mut trace_entries = vec![uniform(0, false), storage(1, true), storage(2, true), storage(3, true), storage(4, true)];
        trace_entries.extend((5..9).map(|b| storage(b, false)));
        trace_entries.push(texture(9, wgpu::TextureViewDimension::D2));
        trace_entries.extend((10..10 + KINDS as u32).map(|b| texture(b, wgpu::TextureViewDimension::D2Array)));
        let trace_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("tc_trace"), entries: &trace_entries });
        let mut post_entries = vec![uniform(0, true)];
        post_entries.extend((1..6).map(|b| storage(b, true)));
        post_entries.push(storage(6, false));
        post_entries.push(entry(
            7,
            wgpu::BindingType::StorageTexture {
                access: wgpu::StorageTextureAccess::WriteOnly,
                format: wgpu::TextureFormat::Rgba8Unorm,
                view_dimension: wgpu::TextureViewDimension::D2,
            },
        ));
        let post_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some("tc_trace_post"), entries: &post_entries });
        let module =
            |label, src: &str| device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(src.into()) });
        let trace_module = module("tc_trace", concat!(include_str!("shaders/rays.wgsl"), include_str!("shaders/trace.wgsl")));
        let post_module = module("tc_trace_post", include_str!("shaders/trace_post.wgsl"));
        let pipe = |bgl: &wgpu::BindGroupLayout, m: &wgpu::ShaderModule, entry_point: &str| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(entry_point),
                bind_group_layouts: &[Some(bgl)],
                immediate_size: 0,
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(entry_point),
                layout: Some(&layout),
                module: m,
                entry_point: Some(entry_point),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let post_stride = (size_of::<PostRaw>() as u32).next_multiple_of(device.limits().min_uniform_buffer_offset_alignment.max(1));
        let uniform_buffer = |label, size| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        GpuTracer {
            trace_pipe: pipe(&trace_bgl, &trace_module, "trace_main"),
            init_pipe: pipe(&post_bgl, &post_module, "denoise_init"),
            step_pipe: pipe(&post_bgl, &post_module, "denoise_step"),
            resolve_pipe: pipe(&post_bgl, &post_module, "resolve"),
            trace_bgl,
            post_bgl,
            params: uniform_buffer("tc_trace_params", PARAMS_SIZE),
            post: uniform_buffer("tc_trace_post", u64::from(post_stride) * STEPS.len() as u64),
            post_stride,
            geometry: None,
            env: None,
            materials: None,
            acc: None,
            spare_acc: None,
            #[cfg(test)]
            acc_allocations: 0,
            scene: None,
            bind: None,
            samples: 0,
        }
    }

    /// Samples per pixel accumulated since the last [`GpuTracer::reset`].
    pub fn samples(&self) -> u32 {
        self.samples
    }

    /// Free the scene and accumulation memory, keeping the compiled pipelines.
    pub fn release(&mut self) {
        (self.geometry, self.env, self.materials, self.acc, self.scene, self.bind) = (None, None, None, None, None, None);
        self.spare_acc = None;
        self.samples = 0;
    }

    /// Upload what changed of `scene`: the mesh and BVH (when the mesh or its texture-set
    /// mapping changed), the environment and the shading constants.
    pub fn set_scene(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, scene: &TraceScene) -> Result<(), String> {
        let limits = device.limits();
        let max_buf = limits.max_storage_buffer_binding_size.min(limits.max_buffer_size);
        let fits = |what: &str, bytes: usize| {
            if bytes as u64 <= max_buf { Ok(()) } else { Err(format!("the {what} ({} MB) exceeds this GPU's buffer limit", bytes >> 20)) }
        };
        let same_geometry = self.geometry.as_ref().is_some_and(|g| Arc::ptr_eq(&g.mesh, &scene.mesh) && g.tri_set == scene.tri_set);
        if !same_geometry {
            let nodes: Vec<(Vec3, Vec3, u32, u32)> = scene.bvh.nodes().collect();
            if bvh_depth(&nodes).is_none_or(|d| d >= STACK) {
                return Err("the mesh's BVH is too deep for GPU traversal".into());
            }
            let (data, verts) = geometry_data(&scene.mesh, &scene.bvh, &scene.tri_set);
            let (nodes, tris) = data;
            for (what, b) in [("BVH", nodes.len()), ("triangle list", tris.len()), ("vertex list", verts.len())] {
                fits(what, b * 16)?;
            }
            self.geometry = Some(scoped(device, "GPU path tracing: mesh upload", || Geometry {
                mesh: scene.mesh.clone(),
                tri_set: scene.tri_set.clone(),
                nodes: buffer(device, "tc_trace_nodes", bytemuck::cast_slice(&nodes)),
                tris: buffer(device, "tc_trace_tris", bytemuck::cast_slice(&tris)),
                verts: buffer(device, "tc_trace_verts", bytemuck::cast_slice(&verts)),
            })?);
            self.bind = None;
        }
        if !self.env.as_ref().is_some_and(|e| Arc::ptr_eq(&e.image, &scene.env)) {
            let img = &scene.env;
            let (w, h) = (img.width as usize, img.height as usize);
            let valid = w > 0 && h > 0 && img.data.len() >= w * h && img.width.max(img.height) <= limits.max_texture_dimension_2d;
            let (tw, th, rgba): (u32, u32, Vec<[f32; 4]>) =
                if valid { (img.width, img.height, img.data.iter().take(w * h).map(|c| [c[0], c[1], c[2], 1.0]).collect()) } else { (1, 1, vec![[0.0; 4]]) };
            let mut cdf = Vec::new();
            if let Some(l) = scene.light.as_ref().filter(|l| valid && l.width == w && l.height == h) {
                cdf.extend_from_slice(&l.rows);
                cdf.extend_from_slice(&l.cols);
                cdf.extend_from_slice(&l.prob);
            }
            fits("environment table", cdf.len() * 4)?;
            let light = !cdf.is_empty();
            if !light {
                cdf.push(0.0);
            }
            let env = scoped(device, "GPU path tracing: environment upload", || {
                let tex = device.create_texture_with_data(
                    queue,
                    &wgpu::TextureDescriptor {
                        label: Some("tc_trace_env"),
                        size: wgpu::Extent3d { width: tw, height: th, depth_or_array_layers: 1 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba32Float,
                        usage: wgpu::TextureUsages::TEXTURE_BINDING,
                        view_formats: &[],
                    },
                    wgpu::util::TextureDataOrder::LayerMajor,
                    bytemuck::cast_slice(&rgba),
                );
                EnvGpu {
                    image: scene.env.clone(),
                    view: tex.create_view(&Default::default()),
                    cdf: buffer(device, "tc_trace_cdf", bytemuck::cast_slice(&cdf)),
                    light,
                }
            })?;
            self.env = Some(env);
            self.bind = None;
        }
        self.scene = Some(SceneConst {
            spec_gloss: scene.spec_gloss,
            eps: scene.eps,
            sigma: scene.scatter_sigma,
            rotation: scene.env_rotation,
            heights: scene.sets.iter().map(|s| s.height_strength).collect(),
            ior: scene.ior,
            lens: scene.lens.map_or([0.0; 2], |[focus, aperture]| [focus, aperture.max(0.0) * scene.radius]),
        });
        Ok(())
    }

    /// Copy the renderer's display textures when their content changed (a GPU-side copy);
    /// true when it did (samples taken before no longer match the scene).
    pub fn sync_materials(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
    ) -> Result<bool, String> {
        let rev = renderer.content_rev();
        if self.materials.as_ref().is_some_and(|m| m.rev == rev) {
            return Ok(false);
        }
        let sets = renderer.display_textures();
        if sets.len() > MAX_SETS || sets.iter().any(|s| s.len() != KINDS) {
            return Err(format!("GPU path tracing takes at most {MAX_SETS} texture sets"));
        }
        let layers = sets.len() as u32 + 1;
        let limits = device.limits();
        if layers > limits.max_texture_array_layers {
            return Err(format!("this GPU takes at most {} texture sets", limits.max_texture_array_layers.saturating_sub(1)));
        }
        let old = self.materials.take();
        let mut arrays = Vec::with_capacity(KINDS);
        for (k, kind) in DisplayKind::ALL.into_iter().enumerate() {
            let size = sets.iter().filter_map(|s| s.get(k)).fold((1, 1), |(w, h), t| (t.width().max(w), t.height().max(h)));
            let reuse =
                old.as_ref().and_then(|m| m.arrays.get(k)).filter(|(t, _)| (t.width(), t.height(), t.depth_or_array_layers()) == (size.0, size.1, layers));
            let (tex, view) = match reuse {
                Some(a) => a.clone(),
                None => scoped(device, "GPU path tracing: material textures", || {
                    let tex = device.create_texture(&wgpu::TextureDescriptor {
                        label: Some("tc_trace_material"),
                        size: wgpu::Extent3d { width: size.0, height: size.1, depth_or_array_layers: layers },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: crate::renderer::display_format(kind),
                        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    });
                    let view = tex.create_view(&wgpu::TextureViewDescriptor { dimension: Some(wgpu::TextureViewDimension::D2Array), ..Default::default() });
                    (tex, view)
                })?,
            };
            if reuse.is_none() {
                self.bind = None;
            }
            for (layer, src) in sets.iter().filter_map(|s| s.get(k)).enumerate() {
                encoder.copy_texture_to_texture(
                    src.as_image_copy(),
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: 0, y: 0, z: layer as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d { width: src.width(), height: src.height(), depth_or_array_layers: 1 },
                );
            }
            // The default material: one texel of the kind's neutral value.
            let texel = kind.default_texel();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &tex,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layers - 1 },
                    aspect: wgpu::TextureAspect::All,
                },
                &texel,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(texel.len() as u32), rows_per_image: Some(1) },
                wgpu::Extent3d { width: 1, height: 1, depth_or_array_layers: 1 },
            );
            arrays.push((tex, view));
        }
        let mut info: Vec<[f32; 3]> = sets
            .iter()
            .map(|s| {
                // The set's size is its largest texture's; any kind may still be a 1×1 placeholder.
                let dims = s.iter().fold((1, 1), |(w, h), t| (t.width().max(w), t.height().max(h)));
                let small = s.iter().enumerate().filter(|(_, t)| (t.width(), t.height()) != dims).fold(0u32, |m, (k, _)| m | 1 << k);
                [dims.0 as f32, dims.1 as f32, small as f32]
            })
            .collect();
        info.push([1.0, 1.0, 0.0]);
        self.materials = Some(Materials { rev, arrays, info });
        Ok(true)
    }

    /// Start over at `width × height`, keeping the active render intact if allocation fails.
    pub fn reset(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, width: u32, height: u32) -> Result<(), String> {
        let (width, height) = (width.clamp(1, 8192), height.clamp(1, 8192));
        let bytes = u64::from(width) * u64::from(height) * 16;
        let limits = device.limits();

        // Reject unsupported sizes before invalidating the displayed render
        if bytes > limits.max_storage_buffer_binding_size.min(limits.max_buffer_size) {
            return Err(format!("a {width}×{height} render exceeds this GPU's buffer limit"));
        }

        if let Some(accumulation) = self.acc.as_ref().filter(|accumulation| (accumulation.width, accumulation.height) == (width, height)) {
            for buffer in &accumulation.acc {
                encoder.clear_buffer(buffer, 0, None);
            }

            self.samples = 0;
            return Ok(());
        }

        // Reused buffers must not retain samples from the previous view
        if let Some(spare) = self.spare_acc.take_if(|accumulation| (accumulation.width, accumulation.height) == (width, height)) {
            for buffer in spare.acc.iter().chain(&spare.illum) {
                encoder.clear_buffer(buffer, 0, None);
            }

            self.spare_acc = self.acc.replace(spare);
            self.bind = None;
            self.samples = 0;
            return Ok(());
        }

        // Retire the obsolete spare before allocating, but keep the active render until success
        self.spare_acc = None;

        let create_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: bytes,
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            })
        };
        let accumulation = scoped(device, "GPU path tracing: accumulation buffers", || Accum {
            width,
            height,
            acc: ["tc_acc_hit", "tc_acc_albedo", "tc_acc_normal", "tc_acc_env"].map(create_buffer),
            illum: ["tc_illum_a", "tc_illum_b"].map(create_buffer),
        })?;
        self.spare_acc = self.acc.replace(accumulation);
        self.bind = None;
        self.samples = 0;
        #[cfg(test)]
        {
            self.acc_allocations += 1;
        }

        Ok(())
    }

    /// Add `spp` samples per pixel (call once per submission: the uniform is rewritten). False
    /// until a scene, materials and a size are set.
    pub fn pass(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        cam: &OrbitCamera,
        settings: &TraceSettings,
        spp: u32,
    ) -> bool {
        let (Some(acc), Some(geo), Some(env), Some(mat), Some(sc)) = (&self.acc, &self.geometry, &self.env, &self.materials, &self.scene) else { return false };
        let bind = self.bind.get_or_insert_with(|| {
            let mut entries = vec![
                wgpu::BindGroupEntry { binding: 0, resource: self.params.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: geo.nodes.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: geo.tris.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 3, resource: geo.verts.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: env.cdf.as_entire_binding() },
            ];
            entries.extend(acc.acc.iter().zip(5..).map(|(b, binding)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }));
            entries.push(wgpu::BindGroupEntry { binding: 9, resource: wgpu::BindingResource::TextureView(&env.view) });
            entries
                .extend(mat.arrays.iter().zip(10..).map(|((_, v), binding)| wgpu::BindGroupEntry { binding, resource: wgpu::BindingResource::TextureView(v) }));
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("tc_trace"), layout: &self.trace_bgl, entries: &entries })
        });
        let (w, h) = (acc.width, acc.height);
        let inv_vp = cam.view_proj(w as f32 / h as f32).inverse();
        let (sin, cos) = sc.rotation.sin_cos();
        let head = Head {
            inv_vp: inv_vp.to_cols_array(),
            forward: cam.forward().extend(sc.eps).to_array(),
            size: [w, h, self.samples, spp.max(1)],
            cfg: [settings.bounces, u32::from(sc.spec_gloss), u32::from(env.light), (mat.info.len() - 1) as u32],
            env: [sin, cos, 0.0, 0.0],
            sigma: sc.sigma.extend(0.0).to_array(),
            eye: cam.eye().extend(1.0).to_array(),
            lens: [if sc.lens[0] > 0.0 { sc.lens[0] } else { cam.distance }, sc.lens[1], sc.ior, 0.0],
        };
        let mut sets = [[0.0f32; 4]; 256];
        for (i, (slot, m)) in sets.iter_mut().zip(&mat.info).enumerate() {
            *slot = [m[0], m[1], m[2], sc.heights.get(i).copied().unwrap_or(0.0)];
        }
        let mut bytes = bytemuck::bytes_of(&head).to_vec();
        bytes.extend_from_slice(bytemuck::cast_slice(&sets));
        queue.write_buffer(&self.params, 0, &bytes);
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tc_trace"), timestamp_writes: None });
        cpass.set_pipeline(&self.trace_pipe);
        cpass.set_bind_group(0, &*bind, &[]);
        cpass.dispatch_workgroups(w.div_ceil(WORKGROUP), h.div_ceil(WORKGROUP), 1);
        drop(cpass);
        self.samples = self.samples.saturating_add(spp.max(1));
        true
    }

    /// Display pixels into `out` (an [`GpuTracer::output_texture`] of the render's size):
    /// denoised when `display.denoise`, tone mapped over the backdrop as [`crate::trace::Tracer::rgba8`].
    pub fn resolve(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        display: &TraceDisplay,
        out: &wgpu::TextureView,
    ) -> bool {
        let Some(acc) = &self.acc else { return false };
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let disp = [display.exposure, flag(display.aces), display.env_opacity, flag(display.denoise)];
        let mut bytes = vec![0u8; self.post_stride as usize * STEPS.len()];
        for (i, step) in STEPS.iter().enumerate() {
            let raw = PostRaw { size: [acc.width, acc.height, self.samples, *step], display: disp };
            if let Some(slot) = bytes.get_mut(i * self.post_stride as usize..).and_then(|s| s.get_mut(..size_of::<PostRaw>())) {
                slot.copy_from_slice(bytemuck::bytes_of(&raw));
            }
        }
        queue.write_buffer(&self.post, 0, &bytes);
        let post = wgpu::BufferBinding { buffer: &self.post, offset: 0, size: wgpu::BufferSize::new(size_of::<PostRaw>() as u64) };
        // a: illum[0] → illum[1], b: illum[1] → illum[0].
        let group = |src: &wgpu::Buffer, dst: &wgpu::Buffer| {
            let mut entries = vec![wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(post.clone()) }];
            entries.extend(acc.acc.iter().zip(1..).map(|(b, binding)| wgpu::BindGroupEntry { binding, resource: b.as_entire_binding() }));
            entries.push(wgpu::BindGroupEntry { binding: 5, resource: src.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 6, resource: dst.as_entire_binding() });
            entries.push(wgpu::BindGroupEntry { binding: 7, resource: wgpu::BindingResource::TextureView(out) });
            device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("tc_trace_post"), layout: &self.post_bgl, entries: &entries })
        };
        let [i0, i1] = &acc.illum;
        let (a, b) = (group(i0, i1), group(i1, i0));
        let groups = (acc.width.div_ceil(WORKGROUP), acc.height.div_ceil(WORKGROUP));
        let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("tc_trace_post"), timestamp_writes: None });
        let mut run = |pipe: &wgpu::ComputePipeline, bg: &wgpu::BindGroup, slot: usize| {
            cpass.set_pipeline(pipe);
            cpass.set_bind_group(0, bg, &[slot as u32 * self.post_stride]);
            cpass.dispatch_workgroups(groups.0, groups.1, 1);
        };
        if display.denoise {
            // The five steps alternate b, a, b, a, b, leaving the result in illum[0].
            run(&self.init_pipe, &a, 0);
            for slot in 1..STEPS.len() {
                run(&self.step_pipe, if slot % 2 == 1 { &b } else { &a }, slot);
            }
        }
        run(&self.resolve_pipe, &a, 0);
        true
    }

    /// The texture [`GpuTracer::resolve`] writes (RGBA8, sRGB-encoded values, as the CPU tracer's
    /// `rgba8`): also sampled by the UI and copied for read-back.
    pub fn output_texture(device: &wgpu::Device, width: u32, height: u32) -> wgpu::Texture {
        device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tc_trace_out"),
            size: wgpu::Extent3d { width: width.clamp(1, 8192), height: height.clamp(1, 8192), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        })
    }

    /// A whole render, blocking: `samples` passes at `size`, resolved and read back as RGBA8
    /// (sRGB) pixels. `renderer` holds the session's texture sets.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        renderer: &Renderer,
        scene: &TraceScene,
        cam: &OrbitCamera,
        settings: &TraceSettings,
        display: &TraceDisplay,
        size: [u32; 2],
        samples: u32,
    ) -> Result<(u32, u32, Vec<u8>), String> {
        self.set_scene(device, queue, scene)?;
        let encoder = || device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("tc_trace") });
        let mut enc = encoder();
        self.sync_materials(device, queue, &mut enc, renderer)?;
        self.reset(device, &mut enc, size[0], size[1])?;
        queue.submit([enc.finish()]);
        let (w, h) = self.acc.as_ref().map_or((1, 1), |a| (a.width, a.height));
        // Several samples per dispatch on small renders; about one at 1080p (short GPU jobs).
        let chunk = (16_000_000 / (w * h).max(1)).clamp(1, 16);
        while self.samples < samples {
            let mut enc = encoder();
            if !self.pass(device, queue, &mut enc, cam, settings, chunk.min(samples - self.samples)) {
                return Err("GPU path tracing: the scene is incomplete".into());
            }
            queue.submit([enc.finish()]);
        }
        let out = Self::output_texture(device, w, h);
        let mut enc = encoder();
        self.resolve(device, queue, &mut enc, display, &out.create_view(&Default::default()));
        queue.submit([enc.finish()]);
        crate::readback::Readback::blocking(device, queue, &out).ok_or_else(|| "GPU path tracing: could not read the render back".into())
    }
}

/// GPU buffers of the mesh: (BVH nodes, triangles in BVH order), vertices.
type GeometryData = ((Vec<[u32; 4]>, Vec<[u32; 4]>), Vec<[f32; 4]>);

fn geometry_data(mesh: &Mesh, bvh: &texturecraft_geom::Bvh, tri_set: &[usize]) -> GeometryData {
    let vcount = mesh.positions.len();
    let rows = bvh_rows(bvh, (0, 0), |o| {
        let idx = [0, 1, 2].map(|k| mesh.indices.get(o * 3 + k).copied().filter(|i| (*i as usize) < vcount).unwrap_or(0));
        let set = tri_set.get(o).copied().filter(|s| *s < MAX_SETS).map_or(u32::MAX, |s| s as u32);
        let sign: f32 = mesh.tangents.get(idx[0] as usize).map_or(1.0, |t| if t.w < 0.0 { -1.0 } else { 1.0 });
        ([set, sign.to_bits(), 0], [idx[0], idx[1], idx[2], o as u32])
    });
    let verts: Vec<[f32; 4]> = (0..vcount.max(1))
        .flat_map(|v| {
            let n = mesh.normals.get(v).copied().unwrap_or(Vec3::ZERO);
            let uv = mesh.uvs.get(v).copied().unwrap_or_default();
            let t = mesh.tangents.get(v).copied().unwrap_or(Vec4::X);
            [[n.x, n.y, n.z, uv.x], [t.x, t.y, t.z, uv.y]]
        })
        .collect();
    (rows, verts)
}

/// `bvh` in the layout of `shaders/rays.wgsl`, numbered from `base` (first node, first
/// triangle) so several BVHs can share one buffer: per node (lo, first), (hi, count); per
/// triangle (p0, w0), (p1, w1), (p2, w2), `last`, where `(w, last) = per_tri(original index)`.
/// An empty BVH gets one leaf over one degenerate triangle.
pub(crate) fn bvh_rows(
    bvh: &texturecraft_geom::Bvh,
    (node_base, tri_base): (u32, u32),
    per_tri: impl Fn(usize) -> ([u32; 3], [u32; 4]),
) -> (Vec<[u32; 4]>, Vec<[u32; 4]>) {
    let (tris, order) = bvh.triangles();
    if tris.is_empty() {
        return (vec![bits(Vec3::ZERO, tri_base), bits(Vec3::ZERO, 1)], vec![[0; 4]; 4]);
    }
    let nodes = bvh
        .nodes()
        .flat_map(|(lo, hi, first, count)| [bits(lo, first.saturating_add(if count > 0 { tri_base } else { node_base })), bits(hi, count)])
        .collect();
    let rows = tris
        .iter()
        .zip(order)
        .flat_map(|(t, &orig)| {
            let (w, last) = per_tri(orig as usize);
            [bits(t[0], w[0]), bits(t[1], w[1]), bits(t[2], w[2]), last]
        })
        .collect();
    (nodes, rows)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless::Headless;
    use crate::trace::{TraceSet, Tracer};
    use texturecraft_compose::{BakedMaps, Composite, all_tiles, tile_bytes};
    use texturecraft_core::Channel;
    use texturecraft_geom::{Bvh, primitives};

    fn camera() -> OrbitCamera {
        let mut cam = OrbitCamera::default();
        cam.distance = 4.0;
        cam
    }

    /// A sphere whose one texture set holds the uniform channel `values`, that set uploaded to
    /// the headless renderer.
    fn sphere_scene(h: &mut Headless, env: EnvImage, values: &[(Channel, [f32; 3])]) -> TraceScene {
        let chans: Vec<Channel> = values.iter().map(|(c, _)| *c).collect();
        let mut composite = Composite::new(&chans, 64);
        for (c, v) in values {
            let raw: Vec<u16> = v[..c.components()].iter().map(|x| texturecraft_texel::to_u16(*x)).collect();
            let img = composite.channels.get_mut(c).unwrap();
            for y in 0..64 {
                for x in 0..64 {
                    img.set(x, y, &raw);
                }
            }
        }
        h.renderer.set_texture_sets(&h.device, &h.queue, &[((64, 64), 0)], 1);
        for (tx, ty) in all_tiles(64, 64) {
            for k in DisplayKind::ALL {
                h.renderer.upload_tile(&h.device, &h.queue, 0, k, tx, ty, &tile_bytes(&composite, &BakedMaps::default(), k, tx, ty));
            }
        }
        let mesh = Arc::new(primitives::sphere(1.0, 32, 16));
        let bvh = Arc::new(Bvh::new(&mesh));
        TraceScene::new(mesh, bvh, vec![TraceSet { material: 0, composite, height_strength: 0.0 }], Arc::new(env), 0.0)
    }

    fn gpu() -> Option<(Headless, GpuTracer)> {
        let Ok(h) = Headless::new() else {
            eprintln!("skipping: no GPU adapter");
            return None;
        };
        let t = match GpuTracer::new(&h.device) {
            Ok(t) => t,
            Err(e) if !GpuTracer::supported(&h.device) => {
                eprintln!("skipping: {e}");
                return None;
            }
            Err(e) => panic!("{e}"),
        };
        Some((h, t))
    }

    /// The radiance accumulation, read back.
    fn accumulation(h: &Headless, t: &GpuTracer) -> Vec<[f32; 4]> {
        read_buffer(h, &t.acc.as_ref().unwrap().acc[0])
    }

    /// Read a completed buffer to check its contents across resets and reuse.
    fn read_buffer(headless: &Headless, buffer: &wgpu::Buffer) -> Vec<[f32; 4]> {
        let bytes = buffer.size();
        let staging = headless.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: bytes,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = headless.device.create_command_encoder(&Default::default());
        encoder.copy_buffer_to_buffer(buffer, 0, &staging, 0, bytes);
        headless.queue.submit([encoder.finish()]);
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        headless.device.poll(wgpu::PollType::Wait { submission_index: None, timeout: None }).unwrap();

        bytemuck::cast_slice(&staging.slice(..).get_mapped_range().unwrap()).to_vec()
    }

    /// Two cached sizes reuse cleared buffers, and leaving Rendering mode releases both.
    #[test]
    fn switching_between_two_sizes_reuses_the_accumulation_buffers() {
        let Some((headless, mut tracer)) = gpu() else { return };

        for (width, height) in [(64, 48), (32, 24), (64, 48), (32, 24), (64, 48)] {
            let mut encoder = headless.device.create_command_encoder(&Default::default());
            tracer.reset(&headless.device, &mut encoder, width, height).unwrap();
            headless.queue.submit([encoder.finish()]);

            let accumulation = tracer.acc.as_ref().unwrap();
            assert_eq!((accumulation.width, accumulation.height), (width, height));
            assert_eq!(tracer.samples(), 0);
            for buffer in accumulation.acc.iter().chain(&accumulation.illum) {
                assert!(read_buffer(&headless, buffer).iter().all(|pixel| *pixel == [0.0; 4]), "reused buffers must be cleared");

                // Leave nonzero contents to detect stale pixels on the next reuse
                let pixels = vec![[1.0f32; 4]; (width * height) as usize];
                headless.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&pixels));
            }
            tracer.samples = 7;
        }

        assert_eq!(tracer.acc_allocations, 2, "two sizes, two allocations, however often it switches");
        let mut encoder = headless.device.create_command_encoder(&Default::default());
        tracer.reset(&headless.device, &mut encoder, 16, 16).unwrap();
        headless.queue.submit([encoder.finish()]);
        assert_eq!(tracer.acc_allocations, 3, "a third size allocates (and drops the oldest set)");

        tracer.release();
        assert!(tracer.acc.is_none() && tracer.spare_acc.is_none(), "leaving Rendering mode frees both sizes");
    }

    /// Rejecting an oversized render preserves the image and lets tracing continue.
    #[test]
    fn failed_reset_retains_the_active_render() {
        let Some((mut headless, mut tracer)) = gpu() else { return };
        let scene = sphere_scene(&mut headless, crate::ibl::environment("Studio Soft", 32), &WHITE);
        let camera = camera();
        let settings = TraceSettings::default();
        let display = TraceDisplay::from_display(&Default::default());
        tracer.render(&headless.device, &headless.queue, &headless.renderer, &scene, &camera, &settings, &display, [32, 24], 2).unwrap();
        let before = accumulation(&headless, &tracer);

        let mut encoder = headless.device.create_command_encoder(&Default::default());
        let error = tracer.reset(&headless.device, &mut encoder, 8192, 8192).unwrap_err();

        assert!(error.contains("exceeds this GPU's buffer limit"));
        assert!(tracer.acc.is_some(), "a rejected resize must keep the active buffers");
        assert!(tracer.bind.is_some(), "a rejected resize must keep the active bindings");
        assert_eq!(tracer.samples(), 2, "a rejected resize must keep the accumulated samples");
        assert_eq!(accumulation(&headless, &tracer), before);
        assert!(tracer.pass(&headless.device, &headless.queue, &mut encoder, &camera, &settings, 1));
        let output = GpuTracer::output_texture(&headless.device, 32, 24);
        assert!(tracer.resolve(&headless.device, &headless.queue, &mut encoder, &display, &output.create_view(&Default::default())));
        headless.queue.submit([encoder.finish()]);
        assert!(crate::readback::Readback::blocking(&headless.device, &headless.queue, &output).is_some());
        assert_eq!(tracer.samples(), 3);
    }

    const WHITE: [(Channel, [f32; 3]); 3] = [(Channel::BaseColor, [1.0; 3]), (Channel::Metallic, [0.0; 3]), (Channel::Roughness, [0.5; 3])];

    #[test]
    fn white_furnace_on_the_gpu_stays_bounded_and_finite() {
        // White spheres in a uniform white environment reflect at most what arrives: at half
        // exposure the environment shows as 0.5 and the spheres between 0.25 and 0.5.
        use Channel::*;
        let Some((mut h, mut t)) = gpu() else { return };
        let everything = [(CoatWeight, [1.0; 3]), (SheenWeight, [1.0; 3]), (AnisotropyLevel, [1.0; 3]), (ScatterWeight, [1.0; 3])];
        for (name, extra) in [("white", &[][..]), ("every lobe", &everything[..])] {
            let values: Vec<_> = WHITE.iter().chain(extra).copied().collect();
            let env = EnvImage { width: 8, height: 4, data: vec![[1.0; 3]; 32] };
            let scene = sphere_scene(&mut h, env, &values);
            let d = TraceDisplay { exposure: 0.5, aces: false, env_opacity: 1.0, denoise: false };
            let (w, _, px) = t.render(&h.device, &h.queue, &h.renderer, &scene, &camera(), &TraceSettings { bounces: 8 }, &d, [64, 48], 256).unwrap();
            assert!(accumulation(&h, &t).iter().all(|v| v.iter().all(|c| c.is_finite())), "{name}: no NaN or infinity");
            let at = |x: u32, y: u32| texturecraft_texel::srgb_to_linear(px[((y * w + x) * 4 + 1) as usize] as f32 / 255.0);
            assert!((at(0, 0) - 0.5).abs() < 0.02, "{name}: background {}", at(0, 0));
            let block: Vec<f32> = (18..30).flat_map(|y| (26..38).map(move |x| (x, y))).map(|(x, y)| at(x, y)).collect();
            let mean = block.iter().sum::<f32>() / block.len() as f32;
            assert!(mean > 0.25 && mean <= 0.5 * 1.05, "{name}: sphere {mean}");
        }
    }

    #[test]
    fn gpu_matches_cpu_with_every_lobe_glass_and_lens() {
        use Channel::*;
        let Some((mut h, mut t)) = gpu() else { return };
        let values = [
            (BaseColor, [0.8, 0.5, 0.3]),
            (Roughness, [0.35; 3]),
            (CoatWeight, [0.8; 3]),
            (CoatRoughness, [0.15; 3]),
            (SheenWeight, [0.6; 3]),
            (SheenColor, [0.4, 0.6, 1.0]),
            (AnisotropyLevel, [0.8; 3]),
            (AnisotropyAngle, [0.2; 3]),
            (ScatterWeight, [0.7; 3]),
        ];
        let glass = [(BaseColor, [0.9, 0.95, 1.0]), (Roughness, [0.2; 3]), (Transmission, [0.9; 3]), (CoatWeight, [0.6; 3]), (CoatNormal, [0.65, 0.5, 0.977])];
        for (name, values, lens) in [("every lobe", &values[..], None), ("glass under a thin lens", &glass[..], Some([3.2, 0.2]))] {
            let mut scene = sphere_scene(&mut h, crate::ibl::environment("Studio Contrast", 256), values);
            (scene.lens, scene.ior) = (lens, 1.45);
            let d = TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.5, denoise: false };
            let settings = TraceSettings { bounces: 6 };
            let (_, _, gpu) = t.render(&h.device, &h.queue, &h.renderer, &scene, &camera(), &settings, &d, [48, 36], 1024).unwrap();
            let mut cpu = Tracer::new(48, 36);
            for _ in 0..1024 {
                cpu.pass(&scene, &camera(), &settings);
            }
            assert_region_means_match(name, &gpu, &cpu.rgba8(&d), 48, 36, 3.0);
        }
    }

    #[test]
    fn gpu_reads_full_size_channels_beside_placeholders() {
        // Base color stays at its default (a 1×1 placeholder texture) while roughness varies
        // across the surface (a full-size texture): the tracer must still read every texel of it.
        use Channel::*;
        let Some((mut h, mut t)) = gpu() else { return };
        let mut composite = Composite::new(&[Roughness, Metallic], 64);
        for y in 0..64 {
            for x in 0..64 {
                let rough = if x < 32 { 0.05 } else { 0.9 };
                composite.channels.get_mut(&Roughness).unwrap().set(x, y, &[texturecraft_texel::to_u16(rough)]);
                composite.channels.get_mut(&Metallic).unwrap().set(x, y, &[texturecraft_texel::to_u16(1.0)]);
            }
        }
        h.renderer.set_texture_sets(&h.device, &h.queue, &[((64, 64), 0)], 1);
        for (tx, ty) in all_tiles(64, 64) {
            for k in DisplayKind::ALL {
                h.renderer.upload_tile(&h.device, &h.queue, 0, k, tx, ty, &tile_bytes(&composite, &BakedMaps::default(), k, tx, ty));
            }
        }
        let sizes: Vec<u32> = h.renderer.display_textures()[0].iter().map(|t| t.width()).collect();
        assert_eq!(sizes[0], 1, "base color stays a placeholder");
        assert_eq!(sizes[1], 64, "roughness and metallic are full size");
        let mesh = Arc::new(primitives::sphere(1.0, 32, 16));
        let bvh = Arc::new(Bvh::new(&mesh));
        let env = crate::ibl::environment("Studio Contrast", 256);
        let scene = TraceScene::new(mesh, bvh, vec![TraceSet { material: 0, composite, height_strength: 0.0 }], Arc::new(env), 0.0);
        let d = TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.5, denoise: false };
        let settings = TraceSettings { bounces: 4 };
        let (_, _, gpu) = t.render(&h.device, &h.queue, &h.renderer, &scene, &camera(), &settings, &d, [48, 36], 1024).unwrap();
        let mut cpu = Tracer::new(48, 36);
        for _ in 0..1024 {
            cpu.pass(&scene, &camera(), &settings);
        }
        assert_region_means_match("varying roughness, default base color", &gpu, &cpu.rgba8(&d), 48, 36, 3.0);
    }

    /// Mean luminance of each 6×6 block of two RGBA8 renders agrees within `tol` 8-bit levels.
    fn assert_region_means_match(name: &str, a: &[u8], b: &[u8], w: usize, h: usize, tol: f32) {
        let lum = |img: &[u8], x: usize, y: usize| {
            let p = &img[(y * w + x) * 4..(y * w + x) * 4 + 3];
            0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
        };
        let mut worst = 0.0f32;
        for by in (0..h).step_by(6) {
            for bx in (0..w).step_by(6) {
                let (mut sa, mut sb, mut n) = (0.0, 0.0, 0.0);
                for y in by..(by + 6).min(h) {
                    for x in bx..(bx + 6).min(w) {
                        sa += lum(a, x, y);
                        sb += lum(b, x, y);
                        n += 1.0;
                    }
                }
                worst = worst.max(((sa - sb) / n).abs());
            }
        }
        assert!(worst < tol, "{name}: worst block differs by {worst} levels");
    }

    /// Samples per second at 1080p on the material ball, GPU against CPU:
    /// `cargo test -p texturecraft-render --release --lib -- --ignored --nocapture speed`.
    #[test]
    #[ignore]
    fn speed_at_1080p() {
        let Some((h, mut t)) = gpu() else { return };
        let mesh = Arc::new(primitives::material_ball());
        let (lo, hi) = mesh.bounds();
        let mut cam = OrbitCamera::default();
        cam.frame(lo, hi);
        let scene = TraceScene::new(mesh.clone(), Arc::new(Bvh::new(&mesh)), Vec::new(), Arc::new(crate::ibl::environment("Studio Soft", 1024)), 0.0);
        let (settings, d) = (TraceSettings::default(), TraceDisplay { exposure: 1.0, aces: true, env_opacity: 0.0, denoise: true });
        let run = |t: &mut GpuTracer, n| t.render(&h.device, &h.queue, &h.renderer, &scene, &cam, &settings, &d, [1920, 1080], n).unwrap();
        run(&mut t, 8);
        let t0 = std::time::Instant::now();
        run(&mut t, 8);
        let base = t0.elapsed().as_secs_f64();
        let t0 = std::time::Instant::now();
        run(&mut t, 520);
        let gpu = 512.0 / (t0.elapsed().as_secs_f64() - base);
        let mut cpu_t = Tracer::new(1920, 1080);
        cpu_t.pass(&scene, &cam, &settings);
        let t0 = std::time::Instant::now();
        for _ in 0..16 {
            cpu_t.pass(&scene, &cam, &settings);
        }
        let cpu = 16.0 / t0.elapsed().as_secs_f64();
        eprintln!("1080p material ball: GPU {gpu:.0} samples/s, CPU {cpu:.1} samples/s, {:.1}x", gpu / cpu);
    }

    #[test]
    fn bvh_depth_counts_levels_and_rejects_cycles() {
        let leaf = (Vec3::ZERO, Vec3::ONE, 0, 1);
        let inner = (Vec3::ZERO, Vec3::ONE, 1, 0);
        assert_eq!(bvh_depth(&[leaf]), Some(1));
        assert_eq!(bvh_depth(&[inner, leaf, leaf]), Some(2));
        assert_eq!(bvh_depth(&[(Vec3::ZERO, Vec3::ONE, 0, 0), leaf]), None, "a node that is its own child");
    }
}
