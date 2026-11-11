//! The wgpu viewport renderer. Independent of any UI toolkit: the shell calls [`Renderer::prepare`]
//! with an encoder (offscreen render into per-view targets) and [`Renderer::blit`] inside its
//! own render pass.
//!
//! Per view and frame: the scene pass (MSAA, linear HDR premultiplied by scene coverage, resolved)
//! → the scene effects (only when on: a G-buffer pass, the screen-space subsurface blur and the
//! depth of field, see fx.wgsl) → the bloom pyramid (only when bloom is on) → the post pass (flat
//! studio backdrop, exposure,
//! tone mapping, effects, sRGB) into the view's RGBA8 target that [`Renderer::blit`] and
//! [`Renderer::read_view`] use. The key-light shadow map is re-rendered only when the mesh, the
//! light direction or the visible texture sets change.
//!
//! Temporal anti-aliasing (idle accumulation): while a 3D view's frame and the scene content stay
//! the same, each new frame renders with a sub-pixel jitter and is averaged into the view's
//! accumulation target, up to [`TAA_SAMPLES`] frames (4× MSAA each); then the view stops
//! rendering until something changes. [`Renderer::accumulating`] tells the shell to keep redrawing.
//!
//! Screen-space refraction (sets with a Transmission channel): the scene pass first renders
//! without the transmissive surfaces into the view's `behind` target (with mips for rough
//! glass), then transmissive surfaces refract that image, falling back to the environment where
//! the refracted ray leaves the view or shows nothing. Displacement draws a dense copy of the
//! mesh ([`crate::subdivide`]) moved along its normals by the height channel in the vertex shader.

use std::collections::HashMap;

use bytemuck::{Pod, Zeroable};
use texturecraft_compose::{DisplayKind, ImageRes};
use texturecraft_core::{Channel, DisplaySettings, Project, ToneMapping};
use texturecraft_geom::glam::{Mat4, Vec3};
use texturecraft_geom::{Mesh, OrbitCamera};
use texturecraft_texel::TILE;

use crate::bake_visualization::BakeVisualization;
use crate::cage_gpu::CageGpu;
use crate::gpu_trace::{Checked, checked};
use crate::high_poly_gpu::HighPolyGpu;
use crate::ibl;
use crate::overlay_quad::OverlayQuad;
use crate::set_highlight::SetHighlight;
use crate::subdivide::{self, Vertex};
use crate::uv_seam_gpu::UvSeamGpu;

pub const MSAA: u32 = 4;
pub const COLOR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Scene (pre tone mapping) format where float targets are renderable.
pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
/// Prefiltered environment levels (roughness 0..1).
pub const ENV_LEVELS: u32 = 6;
const SHADOW_SIZE: u32 = 2048;
/// Bloom pyramid levels (1/2 .. 1/32 resolution).
const BLOOM_LEVELS: u32 = 5;
/// Bright-pass threshold on the exposed scene value, and the glow gain at strength 1.
const BLOOM_THRESHOLD: f32 = 1.0;
const BLOOM_GAIN: f32 = 0.15;
/// Frames averaged by idle accumulation (temporal anti-aliasing).
pub const TAA_SAMPLES: u32 = 16;
/// Largest depth of field blur radius, as a fraction of the view height.
const MAX_COC: f32 = 0.03;
/// Mip levels of the refraction source (the roughest glass reads 1/64 resolution).
const BEHIND_LEVELS: u32 = 7;
/// How far refracted rays travel before they look up the scene behind, as a fraction of the
/// mesh's bounding radius (no thickness channel: a constant, like a thin shell).
const REFRACTION_DEPTH: f32 = 0.2;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct GlobalsRaw {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    env: [f32; 4],
    sh: [[f32; 4]; 9],
    mode: [f32; 4],
    bg: [f32; 4],
    uv_xform: [f32; 4],
    light: [f32; 4],
    light_vp: [[f32; 4]; 4],
    shadow: [f32; 4],
    material: [f32; 4],
    scatter: [f32; 4],
    fwd: [f32; 4],
    disp: [f32; 4],
    refr: [f32; 4],
    highlight: [f32; 4],
    highlight_shift: [f32; 4],
}

/// The post pass uniform (see `Post` in post.wgsl).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct PostRaw {
    a: [f32; 4],
    b: [f32; 4],
    c: [f32; 4],
    d: [f32; 4],
    e: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct BlitRaw {
    rect: [f32; 4],
    dst: [f32; 4],
}

/// Width of the highlighted set's edge lines, in pixels
const HIGHLIGHT_WIDTH_PX: f32 = 1.5;

/// Opacity of the highlighted set's edge lines
const HIGHLIGHT_OPACITY: f32 = 0.9;

/// Copies of each highlight line, shifted diagonally to the four corners of its extra width (`vs_wire_highlight`)
const HIGHLIGHT_SHIFTS: u32 = 4;

/// How the material is shaded (from the project's display settings and the view mode).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Shade {
    /// Exposure in EV, applied in the post pass (not to data views).
    pub exposure_ev: f32,
    /// ACES filmic tone mapping (else linear, clipped).
    pub aces: bool,
    pub env_rotation_deg: f32,
    /// 0..1 → background blur level.
    pub env_blur: f32,
    pub env_opacity: f32,
    pub ao_intensity: f32,
    pub height_strength: f32,
    /// -1 for DirectX normal maps in the normal channel, 1 for OpenGL.
    pub normal_y: f32,
    /// 0 material, 1 channel, 2 mesh map (aux texture), 3 UV checker.
    pub view_mode: u32,
    /// Channel index for view mode 1 (channel order, see the shader).
    pub channel: u32,
    /// Direction to an optional key light and its intensity (0 = off). The direction is in
    /// environment space: it turns with `env_rotation_deg` like the lights painted into the
    /// environment, so rotating the environment sweeps the key light and its shadows too.
    pub light: [f32; 4],
    /// Draw the UV wireframe in the 2D view.
    pub wireframe: bool,
    /// Key-light shadow map (needs `light[3] > 0`; 3D material view only). Shadows dim the key
    /// light, not the image-based lighting.
    pub shadows: bool,
    /// Master switch of the effects below ("Post effects"); off costs no extra pass.
    /// Effects never apply to data views (channels, mesh maps, UV checker, 2D view).
    pub post_effects: bool,
    /// Effect strengths, 0..1.
    pub bloom: f32,
    pub vignette: f32,
    pub sharpen: f32,
    pub grain: f32,
    /// Draw the mesh edges over the shaded 3D view (depth tested).
    pub wireframe_3d: bool,
    /// One texture set's mesh edges in a highlight color over the 3D view (the set under a shelf drag)
    pub highlight: Option<SetHighlight>,
    /// Specular/glossiness shading: base color is the diffuse color and the Specular channel the
    /// F0 color (roughness arrives as 1 - glossiness from the display conversion).
    pub spec_gloss: bool,
    /// Parallax occlusion mapping from the height channel (3D view).
    pub parallax: bool,
    /// Idle accumulation of jittered frames in the 3D view.
    pub taa: bool,
    /// Per-channel diffuse wrap where the scattering weight is 1 (0 = plain Lambert), see
    /// [`scatter_wrap`].
    pub scatter: [f32; 3],
    /// Index of refraction of the Transmission channel.
    pub ior: f32,
    /// Depth of field (3D material view): focus depth along the view axis (0: the orbit
    /// target) and lens radius as a fraction of the mesh's bounding radius.
    pub dof: Option<[f32; 2]>,
    /// Screen-space subsurface blur of the scattering diffuse light (3D material view): mean
    /// free path per color channel as a fraction of the mesh's bounding radius. Set by
    /// [`Shade::from_project`] for projects with a Scattering weight channel.
    pub sss_blur: Option<[f32; 3]>,
    /// Transmissive surfaces refract the scene behind them (3D material view; an extra pass).
    /// Set by [`Shade::from_project`] for projects with a Transmission channel.
    pub refraction: bool,
    /// Move the surface by the height channel (3D view): a dense copy of the mesh, displaced in
    /// the vertex shader at the height-to-normal strength.
    pub displacement: bool,
    /// Bake mode's high poly, cage and UV seams over the 3D view
    pub bake: Option<BakeVisualization>,
}

/// The viewport's subsurface scattering approximation: the diffuse lighting wraps around the
/// terminator by `min(1, 4 × radius × color)` per channel (radius as a fraction of the mesh
/// size, about the softening angle on mesh-scale curvature; the path tracer walks the real
/// distance).
pub fn scatter_wrap(d: &DisplaySettings) -> [f32; 3] {
    d.scatter_color.map(|c| (4.0 * d.scatter_radius.max(0.0) * c.max(0.0)).min(1.0))
}

impl Shade {
    /// The project-level shading (display and shader settings); the caller adds the view mode,
    /// channel, height strength and wireframe toggles.
    pub fn from_display(d: &DisplaySettings) -> Shade {
        Shade {
            exposure_ev: d.env_exposure,
            aces: d.tone_mapping == ToneMapping::AcesFilmic,
            env_rotation_deg: d.env_rotation,
            env_blur: d.env_blur,
            env_opacity: d.env_opacity,
            ao_intensity: d.ao_intensity,
            // The key light (on with shadows), in environment space.
            light: if d.shadows { [0.5, 0.8, 0.4, 2.5] } else { [0.0; 4] },
            shadows: d.shadows,
            post_effects: d.post_effects,
            bloom: d.bloom,
            vignette: d.vignette,
            sharpen: d.sharpen,
            grain: d.grain,
            spec_gloss: d.spec_gloss,
            parallax: d.parallax,
            taa: d.taa,
            scatter: scatter_wrap(d),
            ior: d.ior,
            dof: d.dof.enabled.then_some([d.dof.focus_distance, d.dof.aperture]),
            displacement: d.displacement,
            ..Shade::default()
        }
    }

    /// [`Shade::from_display`] of the project's settings, plus the subsurface blur where a
    /// texture set has the Scattering weight channel and refraction where one has Transmission.
    pub fn from_project(p: &Project) -> Shade {
        let d = &p.display;
        let has = |c: Channel| p.texture_sets.iter().any(|t| t.has_channel(c));
        let scatters = d.scatter_radius > 0.0 && has(Channel::ScatterWeight);
        Shade {
            sss_blur: scatters.then(|| d.scatter_color.map(|c| c.max(0.0) * d.scatter_radius)),
            refraction: has(Channel::Transmission),
            ..Shade::from_display(d)
        }
    }
}

impl Default for Shade {
    fn default() -> Self {
        Shade {
            exposure_ev: 0.0,
            aces: true,
            env_rotation_deg: 0.0,
            env_blur: 0.35,
            env_opacity: 0.0,
            ao_intensity: 0.75,
            height_strength: 1.0,
            normal_y: 1.0,
            view_mode: 0,
            channel: 0,
            light: [0.0; 4],
            wireframe: false,
            shadows: false,
            post_effects: false,
            bloom: 0.0,
            vignette: 0.0,
            sharpen: 0.0,
            grain: 0.0,
            wireframe_3d: false,
            highlight: None,
            spec_gloss: false,
            parallax: false,
            taa: false,
            scatter: scatter_wrap(&DisplaySettings::default()),
            ior: 1.5,
            dof: None,
            sss_blur: None,
            refraction: false,
            displacement: false,
            bake: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ViewKind {
    /// The 3D viewport.
    Scene,
    /// The 2D UV view of texture set `ts`: zoom (1 = fit) and pan (clip-space offset).
    Uv { ts: usize, zoom: f32, pan: [f32; 2] },
}

/// One view to render this frame.
#[derive(Clone, Debug, PartialEq)]
pub struct Frame {
    pub view_id: u64,
    /// Target size in physical pixels.
    pub size: [u32; 2],
    pub kind: ViewKind,
    pub camera: OrbitCamera,
    pub shade: Shade,
    /// Visibility per texture set.
    pub visible: Vec<bool>,
    /// See-through quads in the 3D view, hidden behind the mesh
    pub overlay: Vec<OverlayQuad>,
}

struct Tex {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    format: wgpu::TextureFormat,
    mips: u32,
}

fn mip_count(size: u32) -> u32 {
    32 - size.max(1).leading_zeros()
}

/// Clip-space half extents of the 2D view's UV square in a view of `aspect` (width / height)
/// at `zoom`: fit into the view and stretched to the set's `size` (a UV tile atlas is wider or
/// taller), keeping texels square.
pub fn uv_view_scale(aspect: f32, zoom: f32, size: (u32, u32)) -> [f32; 2] {
    let z = zoom.clamp(0.05, 64.0) * 0.92;
    let (sx, sy) = if aspect >= 1.0 { (z / aspect, z) } else { (z, z * aspect) };
    let a = size.0.max(1) as f32 / size.1.max(1) as f32;
    if a >= 1.0 { [sx, sy / a] } else { [sx * a, sy] }
}

/// Display texture size of a set of `size` texels halved `shift` times.
fn display_dims(size: (u32, u32), shift: u32) -> (u32, u32) {
    ((size.0 >> shift).max(1), (size.1 >> shift).max(1))
}

/// Side of the square aux (mesh map / mask / quick mask) texture of a set of `size` texels; it
/// stretches over the set like its other textures.
pub fn aux_size(size: (u32, u32)) -> u32 {
    size.0.max(size.1).clamp(1, 2048)
}

impl Tex {
    fn new(device: &wgpu::Device, label: &str, w: u32, h: u32, format: wgpu::TextureFormat, mips: u32) -> Tex {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
            mip_level_count: mips,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            // COPY_SRC: the GPU path tracer copies the display textures into its arrays.
            usage: wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_DST
                | wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        Tex { texture, view, format, mips }
    }
}

struct SetGpu {
    /// Texels across and down (a UV-tiled set's whole atlas).
    size: (u32, u32),
    /// Display textures hold the set at `size >> shift`: halved until they fit the device.
    shift: u32,
    /// The display textures in [`DisplayKind::ALL`] order, then the aux (mesh map) texture.
    /// Each starts as a 1×1 neutral placeholder and gets its full size with the first tile (or
    /// aux image) that holds something else, so a mesh with many materials costs no GPU memory
    /// until its sets are painted.
    tex: Vec<Tex>,
    bind_group: wgpu::BindGroup,
    mips_dirty: Vec<bool>,
    /// Textures whose full-size allocation failed (out of GPU memory): they keep their
    /// placeholder until the set is re-created, instead of retrying on every tile.
    failed: Vec<bool>,
    /// Full-size textures the browser's WebGPU has not yet confirmed: tiles go into them while
    /// the placeholder stays bound (see [`Renderer::resolve_pending`]).
    pending: Vec<Option<PendingTex>>,
}

/// A full-size texture waiting for WebGPU's verdict on its allocation.
struct PendingTex {
    tex: Tex,
    /// `None` for success, else what went wrong. (A mutex: the renderer is shared with the
    /// paint callback, so it must be `Sync`.)
    answer: std::sync::Mutex<std::sync::mpsc::Receiver<Option<String>>>,
}

/// Swap `tex` in for slot `slot` of `set` (bound at once; its mips are rebuilt).
fn install(set: &mut SetGpu, slot: usize, tex: Tex, device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler) {
    if let Some(s) = set.tex.get_mut(slot) {
        *s = tex;
    }
    set.bind_group = set_bind_group(device, layout, sampler, &set.tex);
    if let Some(d) = set.mips_dirty.get_mut(slot) {
        *d = true;
    }
}

/// Give slot `slot` of `set` (texture set `ts`) the full-size texture `checked` allocated:
/// installed when wgpu accepted it, kept pending while WebGPU has not answered, or (out of GPU
/// memory) left as the placeholder with the failure recorded. False when there is nothing to
/// write into.
fn grow(
    set: &mut SetGpu,
    ts: usize,
    slot: usize,
    checked: Checked<Tex>,
    ctx: (&wgpu::Device, &wgpu::BindGroupLayout, &wgpu::Sampler),
    failures: &mut Vec<usize>,
) -> bool {
    match checked {
        Checked::Done(Ok(full)) => install(set, slot, full, ctx.0, ctx.1, ctx.2),
        Checked::Done(Err(e)) => {
            if let Some(f) = set.failed.get_mut(slot) {
                *f = true;
            }
            memory_failed(failures, ts, &e);
            return false;
        }
        Checked::Later(tex, answer) => {
            if let Some(p) = set.pending.get_mut(slot) {
                *p = Some(PendingTex { tex, answer: std::sync::Mutex::new(answer) });
            }
        }
    }
    true
}

/// Vertex and index buffers with their draws (first index, count, material).
struct Geometry {
    vb: wgpu::Buffer,
    ib: wgpu::Buffer,
    draws: Vec<(u32, u32, usize)>,
    triangles: usize,
}

struct MeshGpu {
    id: u64,
    geo: Geometry,
    /// The vertices and indices of a mesh coarse enough to subdivide for displacement, until
    /// its dense copy is built.
    coarse: Option<(Vec<Vertex>, Vec<u32>)>,
    /// The subdivided copy displacement draws (built on first use; None: the mesh is dense).
    dense: Option<Geometry>,
    /// World length per UV unit (displacement scale).
    world_per_uv: f32,
    /// Unique UV edges as a line list, per material: (first index, count, material).
    lines: wgpu::Buffer,
    line_draws: Vec<(u32, u32, usize)>,
    /// Bounding sphere (center, radius) for fitting the shadow map.
    center: Vec3,
    radius: f32,
}

/// The scene effect targets of a view (full resolution unless noted), created the first time
/// depth of field or the subsurface blur is used.
struct Fx {
    /// G-buffer: scattering diffuse light, view depth (see `fs_gbuf`), and its depth buffer.
    gbuf: wgpu::TextureView,
    gbuf_depth: wgpu::TextureView,
    /// The horizontal subsurface pass, then the depth of field result.
    a: wgpu::TextureView,
    /// The scene with the subsurface blur.
    b: wgpu::TextureView,
    /// Half resolution: the gathered blur and the near-field coverage.
    half: wgpu::TextureView,
    near: wgpu::TextureView,
}

/// The bloom pyramid of a view, created the first time bloom is used.
struct Bloom {
    /// 1/2, 1/4, ... resolution targets.
    levels: Vec<wgpu::TextureView>,
    /// Per level: reads the level above (the HDR scene for level 0).
    down: Vec<wgpu::BindGroup>,
    /// Per level but the last: reads the level below, added onto this one.
    up: Vec<wgpu::BindGroup>,
}

/// What a view's transmissive surfaces refract: the scene without them, with mips.
struct Behind {
    /// Mip 0 (the scene pass resolves into it), the whole chain (`refr_bg`).
    level0: wgpu::TextureView,
    bg: wgpu::BindGroup,
    /// Per mip from 1: its view and the bind group reading the level above.
    mips: Vec<(wgpu::TextureView, wgpu::BindGroup)>,
}

struct ViewTargets {
    size: [u32; 2],
    msaa: wgpu::TextureView,
    depth: wgpu::TextureView,
    /// The resolved HDR scene of a jittered accumulation frame.
    hdr: wgpu::TextureView,
    /// The HDR scene the post pass reads: resolved directly, or the running average of jittered
    /// frames (see [`TAA_SAMPLES`]).
    accum: wgpu::TextureView,
    /// Reads `hdr` for the accumulation blend.
    accum_bg: wgpu::BindGroup,
    /// What `accum` holds: the frame and content revision, and the sample count.
    history: Option<(Frame, u64)>,
    samples: u32,
    /// The display image (RGBA8, sRGB-encoded): post pass output, blit and read-back source.
    resolve: wgpu::Texture,
    resolve_view: wgpu::TextureView,
    globals: wgpu::Buffer,
    globals_bg: Option<wgpu::BindGroup>,
    post_buf: wgpu::Buffer,
    post_bg: wgpu::BindGroup,
    bloom: Option<Bloom>,
    fx: Option<Fx>,
    behind: Option<Behind>,
    blit_buf: wgpu::Buffer,
    blit_bg: wgpu::BindGroup,
    /// The frame's overlay triangles and how many vertices of the buffer they use
    overlay: Option<wgpu::Buffer>,
    overlay_vertices: u32,
    /// Set by [`Renderer::prepare`]: the view has content to blit.
    ready: bool,
}

/// Key-light shadow map, shared by all views (1×1 until shadows are first used).
struct ShadowMap {
    view: wgpu::TextureView,
    size: u32,
    /// What the map was rendered for: mesh id, light direction bits, displacement (its
    /// parameters' bits and, when on, the content revision), visible sets.
    key: Option<(u64, [u32; 3], [u32; 4], u64)>,
    visible: Vec<bool>,
    vp: Mat4,
    buf: wgpu::Buffer,
    bg: wgpu::BindGroup,
}

pub struct Renderer {
    target_format: wgpu::TextureFormat,
    /// [`HDR_FORMAT`], or [`COLOR_FORMAT`] where float targets are not renderable (WebGL2
    /// without float color buffers).
    hdr_format: wgpu::TextureFormat,
    globals_bgl: wgpu::BindGroupLayout,
    set_bgl: wgpu::BindGroupLayout,
    blit_bgl: wgpu::BindGroupLayout,
    mip_bgl: wgpu::BindGroupLayout,
    fx_bgl: wgpu::BindGroupLayout,
    /// The refraction source of the mesh pipelines (group 2), and that bind group reading black.
    refr_bgl: wgpu::BindGroupLayout,
    refr_black: wgpu::BindGroup,
    mesh_pipeline: wgpu::RenderPipeline,
    /// The scene behind transmissive surfaces: `fs_mesh` without them.
    opaque_pipeline: wgpu::RenderPipeline,
    uv_pipeline: wgpu::RenderPipeline,
    wire_pipeline: wgpu::RenderPipeline,
    wire3d_pipeline: wgpu::RenderPipeline,
    overlay_pipeline: wgpu::RenderPipeline,
    /// The highlighted set's mesh edges, wider and in the highlight color
    highlight_pipeline: wgpu::RenderPipeline,
    /// Bake mode: the high poly, the cage's faces and wireframe, the low poly's UV seams
    bake_high_pipeline: wgpu::RenderPipeline,
    cage_pipeline: wgpu::RenderPipeline,
    cage_wire_pipeline: wgpu::RenderPipeline,
    uv_seam_pipeline: wgpu::RenderPipeline,
    bg_pipeline: wgpu::RenderPipeline,
    blit_pipeline: wgpu::RenderPipeline,
    post_pipeline: wgpu::RenderPipeline,
    /// Bright pass + first downsample, downsample, additive upsample.
    bloom_pipelines: [wgpu::RenderPipeline; 3],
    /// The G-buffer of the scene effects (`fs_gbuf`).
    gbuf_pipeline: wgpu::RenderPipeline,
    /// Subsurface blur (horizontal, vertical), depth of field gather and composite (fx.wgsl).
    sss_pipelines: [wgpu::RenderPipeline; 2],
    dof_gather_pipeline: wgpu::RenderPipeline,
    dof_pipeline: wgpu::RenderPipeline,
    shadow_pipeline: wgpu::RenderPipeline,
    mip_pipelines: HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>,
    util_module: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    env_sampler: wgpu::Sampler,
    shadow_sampler: wgpu::Sampler,
    shadow: ShadowMap,
    /// 1×1 black texture for unused effect inputs.
    black: wgpu::TextureView,
    /// [`DisplayKind::default_texel`] in [`DisplayKind::ALL`] order.
    default_texels: Vec<Vec<u8>>,
    /// Texture sets whose display textures could not be allocated (out of GPU memory) since
    /// [`Renderer::take_memory_failures`].
    memory_failures: Vec<usize>,
    accum_pipeline: wgpu::RenderPipeline,
    env: Tex,
    env_name: String,
    sh: [[f32; 4]; 9],
    /// Bumped by every change of what the views show (uploads, mesh, environment): restarts
    /// accumulation.
    content_rev: u64,
    mesh: Option<MeshGpu>,
    bake_high: Option<HighPolyGpu>,
    /// Each texture set's cage, by set index
    cages: Vec<Option<CageGpu>>,
    uv_seams: Option<UvSeamGpu>,
    sets: Vec<SetGpu>,
    /// Viewport texture resolution: display textures at 1/2^shift of each set's size (at least
    /// as reduced as the device's texture limit needs).
    display_shift: u32,
    /// Texture set index per mesh material.
    material_sets: Vec<Option<usize>>,
    views: HashMap<u64, ViewTargets>,
}

fn fragment_entry(binding: u32, ty: wgpu::BindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry { binding, visibility: wgpu::ShaderStages::FRAGMENT, ty, count: None }
}

fn tex_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    fragment_entry(
        binding,
        wgpu::BindingType::Texture {
            multisampled: false,
            sample_type: wgpu::TextureSampleType::Float { filterable: true },
            view_dimension: wgpu::TextureViewDimension::D2,
        },
    )
}

fn sampler_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    fragment_entry(binding, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering))
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: false, min_binding_size: None },
        count: None,
    }
}

pub(crate) fn display_format(k: DisplayKind) -> wgpu::TextureFormat {
    match k {
        DisplayKind::BaseColor | DisplayKind::Emissive | DisplayKind::Sheen => wgpu::TextureFormat::Rgba8UnormSrgb,
        DisplayKind::Orm | DisplayKind::Normal | DisplayKind::Lobes | DisplayKind::Extra => wgpu::TextureFormat::Rgba8Unorm,
        DisplayKind::Height => wgpu::TextureFormat::Rgba16Float,
    }
}

/// Texel of the aux texture before a mesh map or mask is shown.
const AUX_NEUTRAL: [u8; 4] = [128, 128, 128, 255];

/// A `w × h` texture filled with `texel`.
fn neutral_tex(device: &wgpu::Device, queue: &wgpu::Queue, (w, h): (u32, u32), format: wgpu::TextureFormat, mips: u32, texel: &[u8]) -> Tex {
    let t = Tex::new(device, "tc_set", w, h, format, mips);
    let row = texel.repeat(w as usize);
    // Fill in bands of rows: an 8K texture's bytes at once would be hundreds of megabytes.
    let band = (NEUTRAL_BAND_BYTES / row.len().max(1)).clamp(1, h.max(1) as usize) as u32;
    let data = row.repeat(band as usize);
    let mut y = 0;
    while y < h {
        let rows = band.min(h - y);
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t.texture, mip_level: 0, origin: wgpu::Origin3d { x: 0, y, z: 0 }, aspect: wgpu::TextureAspect::All },
            data.get(..row.len() * rows as usize).unwrap_or_default(),
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(row.len() as u32), rows_per_image: Some(rows) },
            wgpu::Extent3d { width: w, height: rows, depth_or_array_layers: 1 },
        );
        y += rows;
    }
    t
}

/// Record that texture set `ts` could not get a display texture (`error` says which).
fn memory_failed(failures: &mut Vec<usize>, ts: usize, error: &str) {
    log::warn!("cannot allocate the display texture of {error}");
    if !failures.contains(&ts) {
        failures.push(ts);
    }
}

/// Bytes per write when filling a texture with one texel.
const NEUTRAL_BAND_BYTES: usize = 8 << 20;

/// The bind group of a set's textures (`set_bgl`): textures, then the sampler.
fn set_bind_group(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, sampler: &wgpu::Sampler, tex: &[Tex]) -> wgpu::BindGroup {
    let mut entries: Vec<wgpu::BindGroupEntry> =
        tex.iter().enumerate().map(|(i, t)| wgpu::BindGroupEntry { binding: i as u32, resource: wgpu::BindingResource::TextureView(&t.view) }).collect();
    entries.push(wgpu::BindGroupEntry { binding: tex.len() as u32, resource: wgpu::BindingResource::Sampler(sampler) });
    device.create_bind_group(&wgpu::BindGroupDescriptor { label: Some("tc_set"), layout, entries: &entries })
}

/// The color output of a pipeline: fragment entry point, format, blending. `fs_wire*` entry
/// points draw line lists.
type ColorOut<'a> = (&'a str, wgpu::TextureFormat, Option<wgpu::BlendState>);

const ADD: wgpu::BlendComponent =
    wgpu::BlendComponent { src_factor: wgpu::BlendFactor::One, dst_factor: wgpu::BlendFactor::One, operation: wgpu::BlendOperation::Add };
const ADDITIVE: wgpu::BlendState = wgpu::BlendState { color: ADD, alpha: ADD };

#[allow(clippy::too_many_arguments)]
fn pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::PipelineLayout,
    module: &wgpu::ShaderModule,
    vs: &str,
    buffers: &[Option<wgpu::VertexBufferLayout>],
    color: Option<ColorOut<'_>>,
    samples: u32,
    depth: Option<wgpu::DepthStencilState>,
) -> wgpu::RenderPipeline {
    let lines = color.is_some_and(|(fs, ..)| fs.starts_with("fs_wire"));
    let topology = if lines { wgpu::PrimitiveTopology::LineList } else { wgpu::PrimitiveTopology::TriangleList };
    let targets = color.map(|(_, format, blend)| [Some(wgpu::ColorTargetState { format, blend, write_mask: wgpu::ColorWrites::ALL })]);
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(label),
        layout: Some(layout),
        vertex: wgpu::VertexState { module, entry_point: Some(vs), buffers, compilation_options: Default::default() },
        primitive: wgpu::PrimitiveState { topology, cull_mode: None, ..Default::default() },
        depth_stencil: depth,
        multisample: wgpu::MultisampleState { count: samples, mask: !0, alpha_to_coverage_enabled: false },
        fragment: color.zip(targets.as_ref()).map(|((fs, ..), targets)| wgpu::FragmentState {
            module,
            entry_point: Some(fs),
            targets,
            compilation_options: Default::default(),
        }),
        multiview_mask: None,
        cache: None,
    })
}

fn depth_state(write: bool, compare: wgpu::CompareFunction) -> wgpu::DepthStencilState {
    wgpu::DepthStencilState {
        format: DEPTH_FORMAT,
        depth_write_enabled: Some(write),
        depth_compare: Some(compare),
        stencil: wgpu::StencilState::default(),
        bias: wgpu::DepthBiasState::default(),
    }
}

/// Whether [`HDR_FORMAT`] works as an MSAA render target. Guaranteed on WebGPU and the native
/// backends; WebGL2/GLES need float color buffer extensions, so probe there (wgpu-core reports the
/// validation error synchronously, so the future is ready at once).
fn hdr_renderable(device: &wgpu::Device) -> bool {
    if device.adapter_info().backend != wgpu::Backend::Gl {
        return true;
    }
    let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
    let _probe = texture(device, "tc_probe", [1, 1], HDR_FORMAT, MSAA, wgpu::TextureUsages::RENDER_ATTACHMENT);
    let mut error = std::pin::pin!(scope.pop());
    matches!(error.as_mut().poll(&mut std::task::Context::from_waker(std::task::Waker::noop())), std::task::Poll::Ready(None))
}

fn texture(device: &wgpu::Device, label: &str, size: [u32; 2], format: wgpu::TextureFormat, samples: u32, usage: wgpu::TextureUsages) -> wgpu::Texture {
    device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
        mip_level_count: 1,
        sample_count: samples,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage,
        view_formats: &[],
    })
}

fn target_view(device: &wgpu::Device, label: &str, size: [u32; 2], format: wgpu::TextureFormat, samples: u32) -> wgpu::TextureView {
    let usage = if samples > 1 || format == DEPTH_FORMAT {
        wgpu::TextureUsages::RENDER_ATTACHMENT
    } else {
        wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING
    };
    texture(device, label, size, format, samples, usage).create_view(&Default::default())
}

fn shadow_view(device: &wgpu::Device, size: u32) -> wgpu::TextureView {
    texture(device, "tc_shadow", [size, size], DEPTH_FORMAT, 1, wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING)
        .create_view(&Default::default())
}

/// A buffer of `usage` holding `data`.
fn upload_buffer(device: &wgpu::Device, queue: &wgpu::Queue, label: &str, data: &[u8], usage: wgpu::BufferUsages) -> wgpu::Buffer {
    let buf = device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: (data.len() as u64).max(4).next_multiple_of(4),
        usage: usage | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    queue.write_buffer(&buf, 0, data);
    buf
}

/// One fullscreen-triangle pass into `target`.
fn fullscreen(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipe: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
) {
    fullscreen_blend(encoder, label, target, load, pipe, bg, 1.0);
}

/// [`fullscreen`] with a blend constant (for pipelines that blend by it).
fn fullscreen_blend(
    encoder: &mut wgpu::CommandEncoder,
    label: &str,
    target: &wgpu::TextureView,
    load: wgpu::LoadOp<wgpu::Color>,
    pipe: &wgpu::RenderPipeline,
    bg: &wgpu::BindGroup,
    constant: f64,
) {
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view: target,
            resolve_target: None,
            depth_slice: None,
            ops: wgpu::Operations { load, store: wgpu::StoreOp::Store },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipe);
    pass.set_bind_group(0, bg, &[]);
    pass.set_blend_constant(wgpu::Color { r: constant, g: constant, b: constant, a: constant });
    pass.draw(0..3, 0..1);
}

/// Halton low-discrepancy value of `i` in `base` (0..1).
fn halton(mut i: u32, base: u32) -> f32 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= base as f32;
        r += f * (i % base) as f32;
        i /= base;
    }
    r
}

const CLEAR: wgpu::LoadOp<wgpu::Color> = wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT);

/// Depth bias of the shadow map, per unit of surface slope: keeps lit surfaces from shadowing themselves
const SHADOW_SLOPE_BIAS: f32 = 1.5;

/// Anisotropic filtering of the texture sets' sampler
const MAX_ANISOTROPY: u16 = 8;

/// Size of the environment texture before the first environment loads
const PLACEHOLDER_ENV_SIZE: [u32; 2] = [512, 256];

impl Renderer {
    /// `target_format`: the format of the pass [`Renderer::blit`] draws into.
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, target_format: wgpu::TextureFormat) -> Renderer {
        let shader = |label: &str, source: &str| {
            device.create_shader_module(wgpu::ShaderModuleDescriptor { label: Some(label), source: wgpu::ShaderSource::Wgsl(source.into()) })
        };
        let module =
            shader("tc_mesh", concat!(include_str!("shaders/displace.wgsl"), include_str!("shaders/mesh.wgsl"), "\n", include_str!("shaders/bake_view.wgsl")));
        let util_module = shader("tc_util", include_str!("shaders/util.wgsl"));
        let post_module = shader("tc_post", concat!(include_str!("shaders/util.wgsl"), include_str!("shaders/post.wgsl"), include_str!("shaders/fx.wgsl")));
        let shadow_module = shader("tc_shadow", concat!(include_str!("shaders/displace.wgsl"), include_str!("shaders/shadow.wgsl")));

        let bind_group_layout = |label: &str, entries: &[wgpu::BindGroupLayoutEntry]| {
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(label), entries })
        };
        let globals_bind_group_layout = bind_group_layout(
            "tc_globals",
            &[
                uniform_entry(0),
                tex_entry(1),
                sampler_entry(2),
                fragment_entry(
                    3,
                    wgpu::BindingType::Texture {
                        multisampled: false,
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                ),
                fragment_entry(4, wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison)),
            ],
        );
        // The display textures, the aux texture, the sampler
        let textures = DisplayKind::ALL.len() as u32 + 1;
        // Displacement reads the height (and the sampler) in the vertex shader
        let height = DisplayKind::ALL.iter().position(|kind| *kind == DisplayKind::Height).unwrap_or(0) as u32;
        let set_entries: Vec<_> = (0..textures)
            .map(tex_entry)
            .chain(std::iter::once(sampler_entry(textures)))
            .map(|mut entry| {
                if entry.binding == height || entry.binding == textures {
                    entry.visibility |= wgpu::ShaderStages::VERTEX;
                }
                entry
            })
            .collect();
        let set_bind_group_layout = bind_group_layout("tc_set", &set_entries);
        let refraction_bind_group_layout = bind_group_layout("tc_refr", &[tex_entry(0)]);
        let blit_bind_group_layout = bind_group_layout("tc_blit", &[tex_entry(0), sampler_entry(1), uniform_entry(2)]);
        let mip_bind_group_layout = bind_group_layout("tc_mip", &[tex_entry(0), sampler_entry(1)]);
        let effect_bind_group_layout =
            bind_group_layout("tc_fx", &[tex_entry(0), sampler_entry(1), uniform_entry(3), tex_entry(4), tex_entry(5), tex_entry(6)]);
        let shadow_bind_group_layout = bind_group_layout("tc_shadow", &[uniform_entry(0)]);

        let pipeline_layout = |label: &str, groups: &[Option<&wgpu::BindGroupLayout>]| {
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor { label: Some(label), bind_group_layouts: groups, immediate_size: 0 })
        };
        let layout = pipeline_layout("tc_uv", &[Some(&globals_bind_group_layout), Some(&set_bind_group_layout)]);
        let mesh_layout = pipeline_layout("tc_mesh", &[Some(&globals_bind_group_layout), Some(&set_bind_group_layout), Some(&refraction_bind_group_layout)]);
        let background_layout = pipeline_layout("tc_bg", &[Some(&globals_bind_group_layout)]);
        let blit_layout = pipeline_layout("tc_blit", &[Some(&blit_bind_group_layout)]);
        let effect_layout = pipeline_layout("tc_fx", &[Some(&effect_bind_group_layout)]);
        let shadow_layout = pipeline_layout("tc_shadow", &[Some(&shadow_bind_group_layout), Some(&set_bind_group_layout)]);

        let vertex_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x4, 3 => Float32x2],
        })];

        // shortcut: the RGBA8 fallback clips highlights above 1 and bands in the darks; encode (e.g. RGBM) if WebGL2 without float color buffers turns out to matter
        let hdr_format = if hdr_renderable(device) { HDR_FORMAT } else { COLOR_FORMAT };
        let hdr_target = |fragment_entry_point, blend| Some((fragment_entry_point, hdr_format, blend));
        let always = || Some(depth_state(false, wgpu::CompareFunction::Always));
        let mesh_pipeline = pipeline(
            device,
            "tc_mesh",
            &mesh_layout,
            &module,
            "vs_mesh",
            &vertex_layout,
            hdr_target("fs_mesh", None),
            MSAA,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
        );
        let opaque_pipeline = pipeline(
            device,
            "tc_opaque",
            &mesh_layout,
            &module,
            "vs_mesh",
            &vertex_layout,
            hdr_target("fs_opaque", None),
            MSAA,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
        );
        let uv_pipeline = pipeline(device, "tc_uv", &layout, &module, "vs_uv", &vertex_layout, hdr_target("fs_uv", None), MSAA, always());
        let blend = Some(wgpu::BlendState::ALPHA_BLENDING);
        let wire_pipeline = pipeline(device, "tc_wire", &layout, &module, "vs_uv", &vertex_layout, hdr_target("fs_wire", blend), MSAA, always());
        let wire3d_depth = Some(depth_state(false, wgpu::CompareFunction::LessEqual));
        let wire3d_pipeline =
            pipeline(device, "tc_wire3d", &layout, &module, "vs_wire3d", &vertex_layout, hdr_target("fs_wire3d", blend), MSAA, wire3d_depth.clone());
        let highlight_pipeline = pipeline(
            device,
            "tc_highlight",
            &layout,
            &module,
            "vs_wire_highlight",
            &vertex_layout,
            hdr_target("fs_wire_highlight", blend),
            MSAA,
            wire3d_depth.clone(),
        );

        let [mesh_vertex_layout] = vertex_layout.clone();
        let high_layout = [
            mesh_vertex_layout,
            Some(wgpu::VertexBufferLayout {
                array_stride: std::mem::size_of::<f32>() as u64,
                step_mode: wgpu::VertexStepMode::Vertex,
                attributes: &wgpu::vertex_attr_array![4 => Float32],
            }),
        ];
        let bake_high_pipeline = pipeline(
            device,
            "tc_bake_high",
            &background_layout,
            &module,
            "vs_bake_high",
            &high_layout,
            hdr_target("fs_bake_high", None),
            MSAA,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
        );
        let corner_layout = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<[f32; 3]>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &wgpu::vertex_attr_array![0 => Float32x3],
        })];
        let cage_pipeline = pipeline(
            device,
            "tc_cage",
            &background_layout,
            &module,
            "vs_bake_cage",
            &corner_layout,
            hdr_target("fs_bake_cage", blend),
            MSAA,
            wire3d_depth.clone(),
        );
        let cage_wire_pipeline = pipeline(
            device,
            "tc_cage_wire",
            &background_layout,
            &module,
            "vs_bake_cage_wire",
            &corner_layout,
            hdr_target("fs_wire_bake_cage", blend),
            MSAA,
            wire3d_depth.clone(),
        );
        let uv_seam_pipeline =
            pipeline(device, "tc_uv_seam", &layout, &module, "vs_wire3d", &vertex_layout, hdr_target("fs_wire_uv_seam", blend), MSAA, wire3d_depth.clone());
        let bg_pipeline = pipeline(device, "tc_bg", &background_layout, &module, "vs_bg", &[], hdr_target("fs_bg", None), MSAA, always());
        let overlay_pipeline =
            pipeline(device, "tc_overlay", &background_layout, &module, "vs_overlay", &vertex_layout, hdr_target("fs_overlay", blend), MSAA, wire3d_depth);
        let blit_pipeline = pipeline(device, "tc_blit", &blit_layout, &util_module, "vs_blit", &[], Some(("fs_blit", target_format, None)), 1, None);
        let post_pipeline = pipeline(device, "tc_post", &effect_layout, &post_module, "vs_full", &[], Some(("fs_post", COLOR_FORMAT, None)), 1, None);
        let bloom_pipelines = [("fs_bloom_pre", None), ("fs_bloom_down", None), ("fs_bloom_up", Some(ADDITIVE))].map(|(fragment_entry_point, blend)| {
            pipeline(device, "tc_bloom", &effect_layout, &post_module, "vs_full", &[], hdr_target(fragment_entry_point, blend), 1, None)
        });
        let gbuffer_pipeline = pipeline(
            device,
            "tc_gbuf",
            &mesh_layout,
            &module,
            "vs_mesh",
            &vertex_layout,
            hdr_target("fs_gbuf", None),
            1,
            Some(depth_state(true, wgpu::CompareFunction::Less)),
        );
        let sss_pipelines = ["fs_sss_h", "fs_sss_v"].map(|fragment_entry_point| {
            pipeline(device, "tc_sss", &effect_layout, &post_module, "vs_full", &[], hdr_target(fragment_entry_point, None), 1, None)
        });
        let dof_pipeline = pipeline(device, "tc_dof", &effect_layout, &post_module, "vs_full", &[], hdr_target("fs_dof", None), 1, None);
        let dof_target = Some(wgpu::ColorTargetState { format: hdr_format, blend: None, write_mask: wgpu::ColorWrites::ALL });
        let dof_gather_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("tc_dof_gather"),
            layout: Some(&effect_layout),
            vertex: wgpu::VertexState { module: &post_module, entry_point: Some("vs_full"), buffers: &[], compilation_options: Default::default() },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &post_module,
                entry_point: Some("fs_dof_gather"),
                targets: &[dof_target.clone(), dof_target],
                compilation_options: Default::default(),
            }),
            multiview_mask: None,
            cache: None,
        });

        let shadow_depth = wgpu::DepthStencilState {
            bias: wgpu::DepthBiasState { constant: 0, slope_scale: SHADOW_SLOPE_BIAS, clamp: 0.0 },
            ..depth_state(true, wgpu::CompareFunction::Less)
        };
        let shadow_pipeline = pipeline(device, "tc_shadow", &shadow_layout, &shadow_module, "vs_shadow", &vertex_layout, None, 1, Some(shadow_depth));

        // accum = hdr_target * k + accum * (1 - k) with k = 1 / (sample + 1): a running average
        let average = wgpu::BlendComponent {
            src_factor: wgpu::BlendFactor::Constant,
            dst_factor: wgpu::BlendFactor::OneMinusConstant,
            operation: wgpu::BlendOperation::Add,
        };
        let accum_layout = pipeline_layout("tc_accum", &[Some(&mip_bind_group_layout)]);
        let accum_blend = Some(wgpu::BlendState { color: average, alpha: average });
        let accum_pipeline = pipeline(device, "tc_accum", &accum_layout, &util_module, "vs_full", &[], hdr_target("fs_copy", accum_blend), 1, None);

        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tc_tex"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: MAX_ANISOTROPY,
            ..Default::default()
        });
        let env_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tc_env"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("tc_shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });

        let shadow_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tc_shadow"),
            size: std::mem::size_of::<[[f32; 4]; 5]>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let shadow = ShadowMap {
            view: shadow_view(device, 1),
            size: 1,
            key: None,
            visible: Vec::new(),
            vp: Mat4::IDENTITY,
            bg: device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tc_shadow"),
                layout: &shadow_bind_group_layout,
                entries: &[wgpu::BindGroupEntry { binding: 0, resource: shadow_buffer.as_entire_binding() }],
            }),
            buf: shadow_buffer,
        };

        let black = Tex::new(device, "tc_black", 1, 1, COLOR_FORMAT, 1).view;
        let refraction_black = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tc_refr_black"),
            layout: &refraction_bind_group_layout,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&black) }],
        });
        let env = Tex::new(device, "tc_env", PLACEHOLDER_ENV_SIZE[0], PLACEHOLDER_ENV_SIZE[1], wgpu::TextureFormat::Rgba16Float, ENV_LEVELS);

        let mut renderer = Renderer {
            target_format,
            hdr_format,
            globals_bgl: globals_bind_group_layout,
            set_bgl: set_bind_group_layout,
            blit_bgl: blit_bind_group_layout,
            mip_bgl: mip_bind_group_layout,
            fx_bgl: effect_bind_group_layout,
            refr_bgl: refraction_bind_group_layout,
            refr_black: refraction_black,
            mesh_pipeline,
            opaque_pipeline,
            uv_pipeline,
            wire_pipeline,
            wire3d_pipeline,
            overlay_pipeline,
            highlight_pipeline,
            bake_high_pipeline,
            cage_pipeline,
            cage_wire_pipeline,
            uv_seam_pipeline,
            bg_pipeline,
            blit_pipeline,
            post_pipeline,
            bloom_pipelines,
            gbuf_pipeline: gbuffer_pipeline,
            sss_pipelines,
            dof_gather_pipeline,
            dof_pipeline,
            shadow_pipeline,
            accum_pipeline,
            mip_pipelines: HashMap::new(),
            util_module,
            sampler,
            env_sampler,
            shadow_sampler,
            shadow,
            black,
            default_texels: DisplayKind::ALL.map(DisplayKind::default_texel).to_vec(),
            memory_failures: Vec::new(),
            env,
            env_name: String::new(),
            sh: [[0.0; 4]; 9],
            content_rev: 0,
            mesh: None,
            bake_high: None,
            cages: Vec::new(),
            uv_seams: None,
            sets: Vec::new(),
            display_shift: 0,
            material_sets: Vec::new(),
            views: HashMap::new(),
        };
        renderer.set_environment(device, queue, "Studio Soft", None);
        renderer
    }

    pub fn environment_name(&self) -> &str {
        &self.env_name
    }

    /// Each texture set's display textures in [`DisplayKind::ALL`] order (lazy kinds may still be
    /// 1×1 placeholders), for the GPU path tracer.
    pub fn display_textures(&self) -> Vec<Vec<&wgpu::Texture>> {
        self.sets.iter().map(|s| s.tex.iter().take(DisplayKind::ALL.len()).map(|t| &t.texture).collect()).collect()
    }

    /// Changes whenever what the views show changes (uploads, mesh, environment).
    pub fn content_rev(&self) -> u64 {
        self.content_rev
    }

    /// Use environment `name`: with `image`, that equirect image (an imported `.hdr`/`.exr`, or
    /// any image, linearized), prefiltered on the CPU (rayon); otherwise the built-in of that
    /// name (no-op when already current; unknown names give the default studio).
    pub fn set_environment(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, name: &str, image: Option<&ImageRes>) {
        let img = match image.and_then(|i| ibl::EnvImage::from_rgba(i.width, i.height, &i.rgba, i.srgb)) {
            Some(img) => img,
            None if self.env_name == name => return,
            None => ibl::environment(name, 512),
        };
        let levels = ibl::prefilter(&img, ENV_LEVELS);
        if self.env.texture.width() != img.width || self.env.texture.height() != img.height {
            self.env = Tex::new(device, "tc_env", img.width, img.height, wgpu::TextureFormat::Rgba16Float, ENV_LEVELS);
        }
        for (l, lv) in levels.iter().enumerate() {
            let bytes: Vec<u8> = lv.data.iter().flat_map(|c| [c[0], c[1], c[2], 1.0]).flat_map(|v| half::f16::from_f32(v).to_le_bytes()).collect();
            queue.write_texture(
                wgpu::TexelCopyTextureInfo { texture: &self.env.texture, mip_level: l as u32, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
                &bytes,
                wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(lv.width * 8), rows_per_image: Some(lv.height) },
                wgpu::Extent3d { width: lv.width, height: lv.height, depth_or_array_layers: 1 },
            );
        }
        let sh = ibl::sh9(&img);
        self.sh = sh.map(|c| [c[0], c[1], c[2], 0.0]);
        self.env_name = name.to_string();
        self.content_rev += 1;
        for v in self.views.values_mut() {
            v.globals_bg = None;
        }
    }

    /// Upload the mesh when `id` changed.
    pub fn set_mesh(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: &Mesh, id: u64) {
        if self.mesh.as_ref().is_some_and(|m| m.id == id) {
            return;
        }
        let verts = subdivide::vertices(mesh);
        let mk = |label: &str, data: &[u8], usage: wgpu::BufferUsages| upload_buffer(device, queue, label, data, usage);
        let mut idx_bytes = bytemuck::cast_slice::<u32, u8>(&mesh.indices).to_vec();
        idx_bytes.resize(idx_bytes.len().next_multiple_of(4), 0);
        let vb = mk("tc_vb", bytemuck::cast_slice(&verts), wgpu::BufferUsages::VERTEX);
        let ib = mk("tc_ib", &idx_bytes, wgpu::BufferUsages::INDEX);
        let draws: Vec<(u32, u32, usize)> = mesh.submeshes.iter().map(|s| (s.start, s.count, s.material)).collect();
        // Mesh edges per material (polygons, not their triangles), for the wireframes.
        let mut line_idx: Vec<u32> = Vec::new();
        let mut line_draws = Vec::new();
        for (m, edges) in mesh.wire_edges().into_iter().enumerate() {
            let start = line_idx.len() as u32;
            line_idx.extend(edges.into_iter().flat_map(|(a, b)| [a, b]));
            line_draws.push((start, line_idx.len() as u32 - start, m));
        }
        let lines = mk("tc_lines", bytemuck::cast_slice(&line_idx), wgpu::BufferUsages::INDEX);
        let (lo, hi) = mesh.bounds();
        let (center, radius) = ((lo + hi) * 0.5, ((hi - lo).length() * 0.5).max(1e-3));
        let triangles = mesh.indices.len() / 3;
        let world_per_uv = subdivide::world_per_uv(&verts, &mesh.indices).unwrap_or(radius);
        let coarse = (subdivide::level_for(triangles) > 0).then(|| (verts, mesh.indices.clone()));
        let geo = Geometry { vb, ib, draws, triangles };
        self.mesh = Some(MeshGpu { id, geo, coarse, dense: None, world_per_uv, lines, line_draws, center, radius });
        self.content_rev += 1;
    }

    /// Upload the high poly when `id` changes; remove it on `None` or when its buffers exceed the device limit
    pub fn set_bake_high(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh: Option<&Mesh>, id: u64) -> Result<(), String> {
        if self.bake_high.as_ref().map(|high| high.id) == mesh.map(|_| id) {
            return Ok(());
        }

        self.bake_high = None;
        self.content_rev += 1;
        let Some(mesh) = mesh else { return Ok(()) };

        // Imported high polys can outgrow a viewport buffer even when the CPU baker accepts them
        let limit = device.limits().max_buffer_size;
        let buffers = [
            (mesh.positions.len(), std::mem::size_of::<Vertex>()),
            (mesh.indices.len(), std::mem::size_of::<u32>()),
            (mesh.positions.len(), std::mem::size_of::<f32>()),
        ];
        if mesh.indices.len() > u32::MAX as usize || buffers.iter().any(|&(count, stride)| count.checked_mul(stride).is_none_or(|bytes| bytes as u64 > limit)) {
            return Err(format!("High-poly visualization exceeds the GPU buffer limit ({limit} bytes). Use a smaller preview mesh or hide the high poly."));
        }

        self.bake_high = Some(HighPolyGpu {
            id,
            vertices: upload_buffer(device, queue, "tc_bake_high_vb", bytemuck::cast_slice(&subdivide::vertices(mesh)), wgpu::BufferUsages::VERTEX),
            vertex_count: mesh.positions.len(),
            indices: upload_buffer(device, queue, "tc_bake_high_ib", bytemuck::cast_slice(&mesh.indices), wgpu::BufferUsages::INDEX),
            index_count: mesh.indices.len() as u32,
            missed: upload_buffer(device, queue, "tc_bake_high_missed", bytemuck::cast_slice(&vec![0f32; mesh.positions.len()]), wgpu::BufferUsages::VERTEX),
        });
        Ok(())
    }

    /// Which vertices of high poly `id` the bake's rays miss (1) or reach (0); ignored for another mesh or vertex count
    pub fn set_bake_high_missed(&mut self, queue: &wgpu::Queue, id: u64, missed: &[f32]) {
        let Some(high) = self.bake_high.as_ref().filter(|high| high.id == id && high.vertex_count == missed.len()) else { return };

        queue.write_buffer(&high.missed, 0, bytemuck::cast_slice(missed));
        self.content_rev += 1;
    }

    /// Texture set `texture_set`'s cage: `corners` three per triangle and `edges` as corner pairs; `None` removes it. The same `topology` as before only rewrites the corners
    pub fn set_bake_cage(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, texture_set: usize, cage: Option<(&[Vec3], &[[u32; 2]])>, topology: u64) {
        if self.cages.len() <= texture_set {
            self.cages.resize_with(texture_set + 1, || None);
        }
        let Some(slot) = self.cages.get_mut(texture_set) else { return };

        match (cage, slot.as_mut()) {
            (None, _) => *slot = None,
            (Some((corners, _)), Some(gpu)) if gpu.topology == topology && gpu.corner_count as usize == corners.len() => {
                queue.write_buffer(&gpu.corners, 0, bytemuck::cast_slice(corners));
            }
            (Some((corners, edges)), _) => {
                *slot = Some(CageGpu {
                    topology,
                    corners: upload_buffer(device, queue, "tc_cage_vb", bytemuck::cast_slice(corners), wgpu::BufferUsages::VERTEX),
                    corner_count: corners.len() as u32,
                    edges: upload_buffer(device, queue, "tc_cage_ib", bytemuck::cast_slice(edges), wgpu::BufferUsages::INDEX),
                    edge_index_count: (edges.len() * 2) as u32,
                });
            }
        }
        self.content_rev += 1;
    }

    /// Forget the cages of texture sets from `count` on (sets that were removed)
    pub fn truncate_bake_cages(&mut self, count: usize) {
        if self.cages.len() > count {
            self.cages.truncate(count);
            self.content_rev += 1;
        }
    }

    /// The low poly's UV seams as vertex pairs per material ([`Mesh::uv_seam_edges`]) for mesh `mesh_id`
    pub fn set_uv_seams(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, mesh_id: u64, seams: &[Vec<(u32, u32)>]) {
        let mut line_indices: Vec<u32> = Vec::new();
        let mut line_draws = Vec::new();
        for (material, edges) in seams.iter().enumerate() {
            let start = line_indices.len() as u32;
            line_indices.extend(edges.iter().flat_map(|&(from, to)| [from, to]));
            line_draws.push((start, line_indices.len() as u32 - start, material));
        }

        let lines = upload_buffer(device, queue, "tc_uv_seams", bytemuck::cast_slice(&line_indices), wgpu::BufferUsages::INDEX);
        self.uv_seams = Some(UvSeamGpu { mesh_id, lines, line_draws });
        self.content_rev += 1;
    }

    /// Build the mesh's dense copy for displacement (once per mesh).
    fn ensure_dense(&mut self, device: &wgpu::Device, queue: &wgpu::Queue) {
        let Some(m) = self.mesh.as_mut() else { return };
        let Some((verts, indices)) = m.coarse.take() else { return };
        let (v, i, draws) = subdivide::subdivide(&verts, &indices, &m.geo.draws, subdivide::level_for(m.geo.triangles));
        let vb = upload_buffer(device, queue, "tc_dense_vb", bytemuck::cast_slice(&v), wgpu::BufferUsages::VERTEX);
        let ib = upload_buffer(device, queue, "tc_dense_ib", bytemuck::cast_slice(&i), wgpu::BufferUsages::INDEX);
        m.dense = Some(Geometry { vb, ib, draws, triangles: i.len() / 3 });
    }

    /// Show texture sets at 1/2^`shift` of their size (0 = full, 1 = half, 2 = quarter…) from the
    /// next [`Renderer::set_texture_sets`], which re-creates the sets that change.
    pub fn set_display_shift(&mut self, shift: u32) {
        self.display_shift = shift.min(6);
    }

    /// Make the per-texture-set textures match `sets` (texels across and down per set: a
    /// UV-tiled set's atlas) and map mesh materials to sets. Re-created textures start neutral;
    /// returns the sets that were reset (the caller must upload all their tiles).
    pub fn set_texture_sets(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sets: &[((u32, u32), usize)], material_count: usize) -> Vec<usize> {
        let mut reset = Vec::new();
        self.content_rev += 1;
        self.sets.truncate(sets.len());
        let max = device.limits().max_texture_dimension_2d.max(1);
        for (i, &(size, _)) in sets.iter().enumerate() {
            // shortcut: a set larger than the device's textures shows every 2^shift-th texel;
            // a texture array per UV tile would keep full detail on such devices.
            let shift = (0..31).find(|s| size.0.max(size.1) >> s <= max).unwrap_or(0).max(self.display_shift);
            if self.sets.get(i).is_some_and(|s| s.size == size && s.shift == shift) {
                continue;
            }
            // Placeholders only: full-size textures come with the first non-neutral upload.
            let mut tex: Vec<Tex> =
                DisplayKind::ALL.iter().zip(&self.default_texels).map(|(k, texel)| neutral_tex(device, queue, (1, 1), display_format(*k), 1, texel)).collect();
            tex.push(neutral_tex(device, queue, (1, 1), wgpu::TextureFormat::Rgba8Unorm, 1, &AUX_NEUTRAL));
            let bind_group = set_bind_group(device, &self.set_bgl, &self.sampler, &tex);
            let n = tex.len();
            let gpu = SetGpu { size, shift, tex, bind_group, mips_dirty: vec![true; n], failed: vec![false; n], pending: (0..n).map(|_| None).collect() };
            if i < self.sets.len() {
                self.sets[i] = gpu;
            } else {
                self.sets.push(gpu);
            }
            reset.push(i);
        }
        self.material_sets = vec![None; material_count];
        for (i, &(_, m)) in sets.iter().enumerate() {
            if let Some(slot) = self.material_sets.get_mut(m) {
                *slot = Some(i);
            }
        }
        reset
    }

    /// Upload one 64×64 tile of a display texture (bytes from `compose::tile_bytes`).
    #[allow(clippy::too_many_arguments)]
    pub fn upload_tile(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ts: usize, kind: DisplayKind, tx: u32, ty: u32, bytes: &[u8]) {
        let Some(set) = self.sets.get_mut(ts) else { return };
        let Some(ki) = DisplayKind::ALL.iter().position(|k| *k == kind) else { return };
        let (Some(t), Some(texel)) = (set.tex.get(ki), self.default_texels.get(ki)) else { return };
        let bpt = kind.bytes_per_texel() as u32;
        let (x, y) = (tx * TILE, ty * TILE);
        if x >= set.size.0 || y >= set.size.1 || bytes.len() < (TILE * TILE * bpt) as usize {
            return;
        }
        let dims = display_dims(set.size, set.shift);
        let awaiting = set.pending.get(ki).is_some_and(Option::is_some);
        if !awaiting && (t.texture.width() < dims.0 || t.texture.height() < dims.1) {
            // A placeholder: neutral tiles change nothing, the first real one allocates it.
            if set.failed.get(ki).copied().unwrap_or(true) || bytes.chunks_exact(texel.len()).all(|px| px == texel.as_slice()) {
                return;
            }
            let what = format!("texture set {} ({kind:?}, {}×{})", ts + 1, dims.0, dims.1);
            let full = checked(device, &what, || neutral_tex(device, queue, dims, display_format(kind), mip_count(dims.0.max(dims.1)), texel));
            if !grow(set, ts, ki, full, (device, &self.set_bgl, &self.sampler), &mut self.memory_failures) {
                return;
            }
        }
        // While WebGPU has not confirmed the full-size texture, tiles go into it (its errors, if
        // it is invalid, are dropped with these scopes) and the placeholder stays bound.
        let pending = set.pending.get(ki).and_then(Option::as_ref).map(|p| &p.tex);
        let Some(t) = pending.or(set.tex.get(ki)) else { return };
        let oom = pending.is_some().then(|| device.push_error_scope(wgpu::ErrorFilter::OutOfMemory));
        let invalid = pending.is_some().then(|| device.push_error_scope(wgpu::ErrorFilter::Validation));
        // Downscaled sets take every `step`-th texel of the tile.
        let (step, n) = (1usize << set.shift.min(6), TILE >> set.shift.min(6));
        let small: Vec<u8>;
        let data = if step == 1 {
            bytes
        } else {
            let texel = |r: usize, c: usize| bytes.get((r * TILE as usize + c) * bpt as usize..).unwrap_or_default().iter().take(bpt as usize);
            small = (0..n as usize).flat_map(|r| (0..n as usize).flat_map(move |c| texel(r * step, c * step))).copied().collect();
            &small
        };
        let (ox, oy) = (x >> set.shift, y >> set.shift);
        let (w, h) = (n.min(dims.0.saturating_sub(ox)), n.min(dims.1.saturating_sub(oy)));
        if w == 0 || h == 0 {
            return;
        }
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t.texture, mip_level: 0, origin: wgpu::Origin3d { x: ox, y: oy, z: 0 }, aspect: wgpu::TextureAspect::All },
            data,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(n * bpt), rows_per_image: Some(n) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        // Innermost scope first.
        drop(invalid);
        drop(oom);
        self.content_rev += 1;
        if let Some(d) = set.mips_dirty.get_mut(ki) {
            *d = true;
        }
    }

    /// Upload the aux (mesh map / debug) texture of a set: RGBA8, `size × size` with `size` from
    /// [`aux_size`]. The first upload allocates it.
    pub fn upload_aux(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, ts: usize, size: u32, rgba: &[u8]) {
        let Some(set) = self.sets.get_mut(ts) else { return };
        let want = aux_size(set.size);
        let ai = set.tex.len().saturating_sub(1);
        let Some(t) = set.tex.last() else { return };
        if size != want || rgba.len() < (size as usize) * (size as usize) * 4 {
            return;
        }
        let awaiting = set.pending.get(ai).is_some_and(Option::is_some);
        if !awaiting && t.texture.width() != size {
            if set.failed.get(ai).copied().unwrap_or(true) {
                return;
            }
            let what = format!("texture set {} (mesh map view, {size}×{size})", ts + 1);
            let full = checked(device, &what, || neutral_tex(device, queue, (size, size), wgpu::TextureFormat::Rgba8Unorm, mip_count(size), &AUX_NEUTRAL));
            if !grow(set, ts, ai, full, (device, &self.set_bgl, &self.sampler), &mut self.memory_failures) {
                return;
            }
        }
        let pending = set.pending.get(ai).and_then(Option::as_ref).map(|p| &p.tex);
        let Some(t) = pending.or(set.tex.last()) else { return };
        let (w, h) = (t.texture.width(), t.texture.height());
        let oom = pending.is_some().then(|| device.push_error_scope(wgpu::ErrorFilter::OutOfMemory));
        let invalid = pending.is_some().then(|| device.push_error_scope(wgpu::ErrorFilter::Validation));
        queue.write_texture(
            wgpu::TexelCopyTextureInfo { texture: &t.texture, mip_level: 0, origin: wgpu::Origin3d::ZERO, aspect: wgpu::TextureAspect::All },
            rgba,
            wgpu::TexelCopyBufferLayout { offset: 0, bytes_per_row: Some(w * 4), rows_per_image: Some(h) },
            wgpu::Extent3d { width: w, height: h, depth_or_array_layers: 1 },
        );
        drop(invalid);
        drop(oom);
        self.content_rev += 1;
        if let Some(d) = set.mips_dirty.last_mut() {
            *d = true;
        }
    }

    /// Texture sets whose display textures could not be allocated since the last call (out of
    /// GPU memory: they show neutral values where they could not be allocated).
    pub fn take_memory_failures(&mut self) -> Vec<usize> {
        std::mem::take(&mut self.memory_failures)
    }

    /// Whether a full-size texture still waits for WebGPU's verdict: keep drawing frames, since
    /// [`Renderer::prepare`] installs it (or reports the failure) once the answer is in.
    pub fn awaiting_gpu(&self) -> bool {
        self.sets.iter().any(|s| s.pending.iter().any(Option::is_some))
    }

    /// Install the pending full-size textures WebGPU accepted; drop the ones it rejected (their
    /// placeholder stays and the failure is reported).
    fn resolve_pending(&mut self, device: &wgpu::Device) {
        for (ts, set) in self.sets.iter_mut().enumerate() {
            for slot in 0..set.pending.len() {
                let poll = |p: &PendingTex| p.answer.lock().unwrap_or_else(std::sync::PoisonError::into_inner).try_recv();
                let verdict = match set.pending.get(slot).and_then(Option::as_ref).map(poll) {
                    None | Some(Err(std::sync::mpsc::TryRecvError::Empty)) => continue,
                    Some(Ok(v)) => v,
                    Some(Err(std::sync::mpsc::TryRecvError::Disconnected)) => Some(format!("texture set {}: the GPU never answered", ts + 1)),
                };
                let Some(p) = set.pending.get_mut(slot).and_then(Option::take) else { continue };
                match verdict {
                    None => {
                        install(set, slot, p.tex, device, &self.set_bgl, &self.sampler);
                        self.content_rev += 1;
                    }
                    Some(e) => {
                        if let Some(f) = set.failed.get_mut(slot) {
                            *f = true;
                        }
                        memory_failed(&mut self.memory_failures, ts, &e);
                    }
                }
            }
        }
    }

    /// Size of the aux texture of a set (1 until a mesh map or mask was shown).
    pub fn aux_size(&self, ts: usize) -> Option<u32> {
        self.sets.get(ts).and_then(|s| s.tex.last()).map(|t| t.texture.width())
    }

    fn generate_mips(&mut self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder) {
        for si in 0..self.sets.len() {
            for ti in 0..self.sets[si].tex.len() {
                if !self.sets[si].mips_dirty.get(ti).copied().unwrap_or(false) {
                    continue;
                }
                let format = self.sets[si].tex[ti].format;
                self.ensure_mip_pipeline(device, format);
                let Some(pipe) = self.mip_pipelines.get(&format) else { continue };
                let t = &self.sets[si].tex[ti];
                for level in 1..t.mips {
                    let src = t.texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level - 1, mip_level_count: Some(1), ..Default::default() });
                    let dst = t.texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: level, mip_level_count: Some(1), ..Default::default() });
                    let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("tc_mip"),
                        layout: &self.mip_bgl,
                        entries: &[
                            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&src) },
                            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                        ],
                    });
                    fullscreen(encoder, "tc_mip", &dst, CLEAR, pipe, &bg);
                }
                if let Some(d) = self.sets[si].mips_dirty.get_mut(ti) {
                    *d = false;
                }
            }
        }
    }

    /// The pipeline that downsamples one mip level of `format` into the next.
    fn ensure_mip_pipeline(&mut self, device: &wgpu::Device, format: wgpu::TextureFormat) {
        if !self.mip_pipelines.contains_key(&format) {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("tc_mip"),
                bind_group_layouts: &[Some(&self.mip_bgl)],
                immediate_size: 0,
            });
            let p = pipeline(device, "tc_mip", &layout, &self.util_module, "vs_full", &[], Some(("fs_mip", format, None)), 1, None);
            self.mip_pipelines.insert(format, p);
        }
    }

    /// The refraction source of a view of `size`: the scene without transmissive surfaces, with
    /// a short mip chain (rough glass reads blurrier levels).
    fn create_behind(&self, device: &wgpu::Device, size: [u32; 2]) -> Behind {
        let levels = mip_count(size[0].max(size[1])).min(BEHIND_LEVELS);
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("tc_behind"),
            size: wgpu::Extent3d { width: size[0], height: size[1], depth_or_array_layers: 1 },
            mip_level_count: levels,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: self.hdr_format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let level = |l: u32| texture.create_view(&wgpu::TextureViewDescriptor { base_mip_level: l, mip_level_count: Some(1), ..Default::default() });
        let mip_bg = |src: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tc_behind_mip"),
                layout: &self.mip_bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                ],
            })
        };
        let mips = (1..levels).map(|l| (level(l), mip_bg(&level(l - 1)))).collect();
        let all = texture.create_view(&Default::default());
        let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tc_refr"),
            layout: &self.refr_bgl,
            entries: &[wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&all) }],
        });
        Behind { level0: level(0), bg, mips }
    }

    /// Effect bind group (`fx_bgl`): source texture, the post uniform, up to three extra
    /// textures (`fx_a`, `fx_b`, `fx_c`; missing ones read black).
    fn fx_bind_group(&self, device: &wgpu::Device, src: &wgpu::TextureView, extra: &[&wgpu::TextureView], post: &wgpu::Buffer) -> wgpu::BindGroup {
        let tex = |i: usize| wgpu::BindingResource::TextureView(extra.get(i).copied().unwrap_or(&self.black));
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tc_fx"),
            layout: &self.fx_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(src) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: post.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 4, resource: tex(0) },
                wgpu::BindGroupEntry { binding: 5, resource: tex(1) },
                wgpu::BindGroupEntry { binding: 6, resource: tex(2) },
            ],
        })
    }

    fn create_targets(&self, device: &wgpu::Device, size: [u32; 2]) -> ViewTargets {
        let hdr = target_view(device, "tc_hdr", size, self.hdr_format, 1);
        let accum = target_view(device, "tc_accum", size, self.hdr_format, 1);
        let accum_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tc_accum"),
            layout: &self.mip_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&hdr) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
            ],
        });
        let usage = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_SRC;
        let resolve = texture(device, "tc_resolve", size, COLOR_FORMAT, 1, usage);
        let resolve_view = resolve.create_view(&Default::default());
        let uniform = |label: &str, size: usize| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: size as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let post_buf = uniform("tc_post", std::mem::size_of::<PostRaw>());
        let blit_buf = uniform("tc_blit", std::mem::size_of::<BlitRaw>());
        let blit_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("tc_blit"),
            layout: &self.blit_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&resolve_view) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                wgpu::BindGroupEntry { binding: 2, resource: blit_buf.as_entire_binding() },
            ],
        });
        ViewTargets {
            size,
            msaa: target_view(device, "tc_msaa", size, self.hdr_format, MSAA),
            depth: target_view(device, "tc_depth", size, DEPTH_FORMAT, MSAA),
            post_bg: self.fx_bind_group(device, &accum, &[], &post_buf),
            hdr,
            accum,
            accum_bg,
            history: None,
            samples: 0,
            resolve,
            resolve_view,
            globals: uniform("tc_globals", std::mem::size_of::<GlobalsRaw>()),
            globals_bg: None,
            post_buf,
            bloom: None,
            fx: None,
            behind: None,
            blit_buf,
            blit_bg,
            overlay: None,
            overlay_vertices: 0,
            ready: false,
        }
    }

    /// The bloom pyramid of `v`, and the post bind group that reads it.
    fn create_bloom(&self, device: &wgpu::Device, v: &ViewTargets) -> (Bloom, wgpu::BindGroup) {
        let levels: Vec<wgpu::TextureView> =
            (1..=BLOOM_LEVELS).map(|i| target_view(device, "tc_bloom", [(v.size[0] >> i).max(1), (v.size[1] >> i).max(1)], self.hdr_format, 1)).collect();
        let fx = |src: &wgpu::TextureView| self.fx_bind_group(device, src, &[], &v.post_buf);
        let down = std::iter::once(&v.accum).chain(&levels).take(levels.len()).map(fx).collect();
        let up = levels.iter().skip(1).map(fx).collect();
        let post_bg = self.fx_bind_group(device, &v.accum, &levels.iter().take(1).collect::<Vec<_>>(), &v.post_buf);
        (Bloom { levels, down, up }, post_bg)
    }

    fn create_fx(&self, device: &wgpu::Device, size: [u32; 2]) -> Fx {
        let full = |label| target_view(device, label, size, self.hdr_format, 1);
        let half_size = [(size[0] / 2).max(1), (size[1] / 2).max(1)];
        Fx {
            gbuf: full("tc_gbuf"),
            gbuf_depth: target_view(device, "tc_gbuf_depth", size, DEPTH_FORMAT, 1),
            a: full("tc_fx_a"),
            b: full("tc_fx_b"),
            half: target_view(device, "tc_dof_half", half_size, self.hdr_format, 1),
            near: target_view(device, "tc_dof_near", half_size, self.hdr_format, 1),
        }
    }

    fn targets(&mut self, device: &wgpu::Device, id: u64, size: [u32; 2], bloom: bool, fx: bool, refraction: bool) -> Option<&mut ViewTargets> {
        let size = [size[0].clamp(1, 8192), size[1].clamp(1, 8192)];
        if self.views.get(&id).is_none_or(|v| v.size != size) {
            let v = self.create_targets(device, size);
            self.views.insert(id, v);
        }
        if refraction && self.views.get(&id).is_some_and(|v| v.behind.is_none()) {
            self.ensure_mip_pipeline(device, self.hdr_format);
            let b = self.create_behind(device, size);
            self.views.get_mut(&id)?.behind = Some(b);
        }
        if fx && self.views.get(&id).is_some_and(|v| v.fx.is_none()) {
            let f = self.create_fx(device, size);
            self.views.get_mut(&id)?.fx = Some(f);
        }
        if bloom
            && let Some(v) = self.views.get(&id)
            && v.bloom.is_none()
        {
            let (b, post_bg) = self.create_bloom(device, v);
            let v = self.views.get_mut(&id)?;
            v.bloom = Some(b);
            v.post_bg = post_bg;
        }
        let (env_view, shadow_view) = (&self.env.view, &self.shadow.view);
        let (bgl, env_sampler, shadow_sampler) = (&self.globals_bgl, &self.env_sampler, &self.shadow_sampler);
        let v = self.views.get_mut(&id)?;
        if v.globals_bg.is_none() {
            v.globals_bg = Some(device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("tc_globals"),
                layout: bgl,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: v.globals.as_entire_binding() },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(env_view) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(env_sampler) },
                    wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::TextureView(shadow_view) },
                    wgpu::BindGroupEntry { binding: 4, resource: wgpu::BindingResource::Sampler(shadow_sampler) },
                ],
            }));
        }
        Some(v)
    }

    /// Index ranges of `ranges` (first, count, material) whose texture set is visible (and is
    /// `only`, when given), with that set.
    fn drawable<'a>(
        &'a self,
        ranges: &'a [(u32, u32, usize)],
        visible: &'a [bool],
        only: Option<usize>,
    ) -> impl Iterator<Item = (std::ops::Range<u32>, &'a SetGpu)> + 'a {
        drawable_ranges(ranges, &self.material_sets, visible, only).filter_map(|(range, ts)| Some((range, self.sets.get(ts)?)))
    }

    /// Keep the key-light shadow map current; returns its view-projection. `light` points to
    /// the light (world space), `disp` is the shadow pass's displacement (see shadow.wgsl).
    /// Re-renders only when the mesh, light, displacement or visible sets changed (and, when it
    /// displaces, the set textures).
    /// shortcut: casters are fully opaque (the dithered opacity cut-out is ignored) and the map is
    /// one shared buffer, so two 3D views with different lights in one frame would share the last
    /// light; upgrade to a per-view map if split 3D views get their own lights.
    #[allow(clippy::too_many_arguments)]
    fn update_shadow(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        light: Vec3,
        disp: [f32; 4],
        visible: &[bool],
    ) -> Option<Mat4> {
        let l = light.try_normalize()?;
        let (id, center, radius) = self.mesh.as_ref().map(|m| (m.id, m.center, m.radius))?;
        if self.shadow.size != SHADOW_SIZE {
            self.shadow.view = shadow_view(device, SHADOW_SIZE);
            self.shadow.size = SHADOW_SIZE;
            self.shadow.key = None;
            for v in self.views.values_mut() {
                v.globals_bg = None;
            }
        }
        let displacing = disp[0] > 0.5;
        let key = (id, l.to_array().map(f32::to_bits), disp.map(f32::to_bits), if displacing { self.content_rev } else { 0 });
        if self.shadow.key == Some(key) && self.shadow.visible == visible {
            return Some(self.shadow.vp);
        }
        // An orthographic light frustum around the mesh's bounding sphere.
        let up = if l.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let vp = Mat4::orthographic_rh(-radius, radius, -radius, radius, radius * 0.5, radius * 3.5) * Mat4::look_at_rh(center + l * radius * 2.0, center, up);
        let mut raw = [[0.0f32; 4]; 5];
        raw[..4].copy_from_slice(&vp.to_cols_array_2d());
        raw[4] = disp;
        queue.write_buffer(&self.shadow.buf, 0, bytemuck::bytes_of(&raw));
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tc_shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow.view,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Store }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.shadow_pipeline);
        pass.set_bind_group(0, &self.shadow.bg, &[]);
        if let Some(m) = &self.mesh {
            let geo = m.dense.as_ref().filter(|_| displacing).unwrap_or(&m.geo);
            pass.set_vertex_buffer(0, geo.vb.slice(..));
            pass.set_index_buffer(geo.ib.slice(..), wgpu::IndexFormat::Uint32);
            for (range, set) in self.drawable(&geo.draws, visible, None) {
                pass.set_bind_group(1, &set.bind_group, &[]);
                pass.draw_indexed(range, 0, 0..1);
            }
        }
        drop(pass);
        self.shadow.key = Some(key);
        self.shadow.visible = visible.to_vec();
        self.shadow.vp = vp;
        Some(vp)
    }

    /// Whether view `id` is still accumulating anti-aliasing samples: the shell should draw it
    /// again even though nothing changed.
    pub fn accumulating(&self, id: u64) -> bool {
        self.views.get(&id).is_some_and(|v| v.history.as_ref().is_some_and(|(f, _)| f.shade.taa && f.kind == ViewKind::Scene) && v.samples + 1 < TAA_SAMPLES)
    }

    /// Render `frame` into its offscreen target (and regenerate dirty mips). A 3D frame with
    /// `shade.taa` that repeats the previous one adds a jittered sample to the average instead,
    /// and renders nothing once [`TAA_SAMPLES`] are in.
    pub fn prepare(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, encoder: &mut wgpu::CommandEncoder, frame: &Frame) {
        self.resolve_pending(device);
        self.generate_mips(device, encoder);
        let rev = self.content_rev;
        let sample = match self.views.get(&frame.view_id) {
            Some(v) if frame.shade.taa && frame.kind == ViewKind::Scene && v.history.as_ref().is_some_and(|(f, r)| *r == rev && f == frame) => v.samples + 1,
            _ => 0,
        };
        if sample >= TAA_SAMPLES {
            return;
        }
        let aspect = frame.size[0].max(1) as f32 / frame.size[1].max(1) as f32;
        let mut vp = frame.camera.view_proj(aspect);
        if sample > 0 {
            // Sub-pixel jitter in clip space: a box filter over the pixel once averaged.
            let (jx, jy) = (halton(sample, 2) - 0.5, halton(sample, 3) - 0.5);
            vp = Mat4::from_translation(Vec3::new(2.0 * jx / frame.size[0].max(1) as f32, 2.0 * jy / frame.size[1].max(1) as f32, 0.0)) * vp;
        }
        let s = &frame.shade;
        let (uv_view, uv_xform) = match frame.kind {
            ViewKind::Scene => (0.0, [1.0, 1.0, 0.0, 0.0]),
            ViewKind::Uv { ts, zoom, pan } => {
                let [sx, sy] = uv_view_scale(aspect, zoom, self.sets.get(ts).map_or((1, 1), |s| s.size));
                (1.0, [sx, sy, pan[0], pan[1]])
            }
        };
        // Data views (channels, mesh maps, UV checker, the 2D view) show values as they are: no
        // exposure, tone mapping, effects or shadows.
        let data = uv_view > 0.5 || s.view_mode != 0;
        let fx = |v: f32| if s.post_effects && !data { v.clamp(0.0, 1.0) } else { 0.0 };
        let bloom = fx(s.bloom) > 0.0;
        // The key light lives in the environment: turn it with it (the inverse of env_rotate in
        // mesh.wgsl maps environment directions to world directions).
        let (sin, cos) = s.env_rotation_deg.to_radians().sin_cos();
        let [lx, ly, lz, lw] = s.light;
        let light = Vec3::new(cos * lx + sin * lz, ly, cos * lz - sin * lx);
        let displace = s.displacement && frame.kind == ViewKind::Scene;
        if displace {
            self.ensure_dense(device, queue);
        }
        let disp = match self.mesh.as_ref().filter(|_| displace) {
            // Height-to-normal's slope: strength 1 tilts 45° for a height change of 1 over 1/16 of
            // the UV square; the vertex shader samples the mip whose texels match the vertex spacing.
            Some(m) => [1.0, m.world_per_uv / 16.0, 1.0 / (m.dense.as_ref().unwrap_or(&m.geo).triangles.max(1) as f32 * 0.5).sqrt(), 0.0],
            None => [0.0; 4],
        };
        let shadow_disp = [disp[0], disp[1] * s.height_strength, disp[2], 0.0];
        let shadow_vp = if !data && s.shadows && lw > 0.0 { self.update_shadow(device, queue, encoder, light, shadow_disp, &frame.visible) } else { None };
        let radius = self.mesh.as_ref().map_or(1.0, |m| m.radius);
        // Background rays come from a perspective camera even in orthographic views (an ortho
        // ray grid would show a single flat environment color).
        let bg_vp = OrbitCamera { ortho: false, ..frame.camera }.view_proj(aspect);
        let e = frame.camera.eye();
        // Scene effects (3D material view; they need float targets for depths and signed light).
        let scene_fx = !data && frame.kind == ViewKind::Scene && self.hdr_format == HDR_FORMAT && self.mesh.is_some();
        let dof = s.dof.filter(|_| scene_fx);
        let sss = s.sss_blur.filter(|m| scene_fx && m.iter().any(|v| *v > 0.0));
        let fx_on = dof.is_some() || sss.is_some();
        let refraction = s.refraction && !data && frame.kind == ViewKind::Scene && self.mesh.is_some();
        let cam = &frame.camera;
        let half_h = (cam.fov.clamp(5.0, 150.0).to_radians() * 0.5).tan();
        let height = frame.size[1].max(1) as f32;
        // Pixels per world unit at view depth z: k / z in perspective, k in orthographic views.
        let k = if cam.ortho { height / (2.0 * cam.distance.max(1e-6) * half_h) } else { height / (2.0 * half_h) };
        let ppw = |z: f32| if cam.ortho { k } else { k / z.max(1e-6) };
        let dof_c = dof.map_or([0.0; 4], |[focus, aperture]| {
            let f = if focus > 0.0 { focus } else { cam.distance };
            [aperture.max(0.0) * radius * ppw(f), f, MAX_COC * height, sample as f32]
        });
        let sss_d = sss.map_or([0.0; 4], |m| [m[0] * radius, m[1] * radius, m[2] * radius, k]);
        let fwd = cam.forward();
        let raw = GlobalsRaw {
            view_proj: vp.to_cols_array_2d(),
            inv_view_proj: bg_vp.inverse().to_cols_array_2d(),
            eye: [e.x, e.y, e.z, 0.0],
            env: [s.env_rotation_deg.to_radians(), (ENV_LEVELS - 1) as f32, s.ao_intensity, 0.0],
            sh: self.sh,
            mode: [s.view_mode as f32, s.channel as f32, s.height_strength, s.normal_y],
            bg: [s.env_opacity, s.env_blur * (ENV_LEVELS - 1) as f32, uv_view, 0.0],
            uv_xform,
            light: [light.x, light.y, light.z, lw],
            light_vp: shadow_vp.unwrap_or(Mat4::IDENTITY).to_cols_array_2d(),
            // Normal offset: 1.5 shadow texels of the light frustum (2 × radius wide).
            shadow: [if shadow_vp.is_some() { 1.0 } else { 0.0 }, 1.0 / SHADOW_SIZE as f32, radius * 3.0 / SHADOW_SIZE as f32, 0.0],
            material: [if s.spec_gloss { 1.0 } else { 0.0 }, if s.parallax { 1.0 } else { 0.0 }, sample as f32, s.ior],
            scatter: [s.scatter[0], s.scatter[1], s.scatter[2], if sss.is_some() { 1.0 } else { 0.0 }],
            fwd: [fwd.x, fwd.y, fwd.z, 0.0],
            disp,
            refr: [if refraction { 1.0 } else { 0.0 }, (BEHIND_LEVELS - 1) as f32, REFRACTION_DEPTH * radius, 0.0],
            highlight: highlight_globals(s, data),
            highlight_shift: highlight_shift(frame.size),
        };
        let flag = |b: bool| if b { 1.0 } else { 0.0 };
        let post = PostRaw {
            a: [2f32.powf(s.exposure_ev), flag(s.aces), flag(data), fx(s.bloom) * BLOOM_GAIN],
            b: [fx(s.vignette), fx(s.sharpen), fx(s.grain), BLOOM_THRESHOLD],
            c: dof_c,
            d: sss_d,
            e: [flag(cam.ortho), sample as f32, 0.0, 0.0],
        };
        let Some(v) = self.targets(device, frame.view_id, frame.size, bloom, fx_on, refraction) else { return };
        queue.write_buffer(&v.globals, 0, bytemuck::bytes_of(&raw));
        queue.write_buffer(&v.post_buf, 0, bytemuck::bytes_of(&post));
        upload_overlay(device, queue, v, &frame.overlay, (!data).then_some(s.exposure_ev));
        v.ready = true;
        v.history = Some((frame.clone(), rev));
        v.samples = sample;
        let Some(v) = self.views.get(&frame.view_id) else { return };
        let Some(globals_bg) = v.globals_bg.as_ref() else { return };
        if refraction && let Some(b) = &v.behind {
            self.behind_pass(encoder, v, b, globals_bg, frame);
        }
        let weight = 1.0 / f64::from(sample + 1);
        match v.fx.as_ref().filter(|_| fx_on) {
            // The effects run on every frame before it joins the average, so idle accumulation
            // also smooths the depth of field's sampling pattern (turned each frame). Their
            // G-buffer (unjittered) stays valid while the view accumulates.
            Some(f) => {
                self.scene_pass(encoder, v, &v.hdr, globals_bg, frame);
                if sample == 0 {
                    self.gbuf_pass(encoder, f, globals_bg, frame);
                }
                let out = self.scene_fx(device, encoder, v, f, sss.is_some(), dof.is_some());
                let bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("tc_accum_fx"),
                    layout: &self.mip_bgl,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(out) },
                        wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::Sampler(&self.sampler) },
                    ],
                });
                fullscreen_blend(encoder, "tc_accum", &v.accum, wgpu::LoadOp::Load, &self.accum_pipeline, &bg, weight);
            }
            None if sample == 0 => self.scene_pass(encoder, v, &v.accum, globals_bg, frame),
            None => {
                self.scene_pass(encoder, v, &v.hdr, globals_bg, frame);
                fullscreen_blend(encoder, "tc_accum", &v.accum, wgpu::LoadOp::Load, &self.accum_pipeline, &v.accum_bg, weight);
            }
        }
        if bloom && let Some(b) = &v.bloom {
            let [pre, down, up] = &self.bloom_pipelines;
            for (i, (level, bg)) in b.levels.iter().zip(&b.down).enumerate() {
                fullscreen(encoder, "tc_bloom_down", level, CLEAR, if i == 0 { pre } else { down }, bg);
            }
            for (level, bg) in b.levels.iter().zip(&b.up).rev() {
                fullscreen(encoder, "tc_bloom_up", level, wgpu::LoadOp::Load, up, bg);
            }
        }
        fullscreen(encoder, "tc_post", &v.resolve_view, CLEAR, &self.post_pipeline, &v.post_bg);
    }

    /// The scene effects' G-buffer: the mesh again, single-sampled, into `f.gbuf`.
    fn gbuf_pass(&self, encoder: &mut wgpu::CommandEncoder, f: &Fx, globals_bg: &wgpu::BindGroup, frame: &Frame) {
        let Some(m) = &self.mesh else { return };
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tc_gbuf"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &f.gbuf,
                resolve_target: None,
                depth_slice: None,
                ops: wgpu::Operations { load: CLEAR, store: wgpu::StoreOp::Store },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &f.gbuf_depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.gbuf_pipeline);
        pass.set_bind_group(0, globals_bg, &[]);
        pass.set_bind_group(2, &self.refr_black, &[]);
        let geo = self.geometry(m, frame);
        pass.set_vertex_buffer(0, geo.vb.slice(..));
        pass.set_index_buffer(geo.ib.slice(..), wgpu::IndexFormat::Uint32);
        for (range, set) in self.drawable(&geo.draws, &frame.visible, None) {
            pass.set_bind_group(1, &set.bind_group, &[]);
            pass.draw_indexed(range, 0, 0..1);
        }
    }

    /// The triangles a frame draws: the dense copy when it displaces.
    fn geometry<'a>(&self, m: &'a MeshGpu, frame: &Frame) -> &'a Geometry {
        m.dense.as_ref().filter(|_| frame.shade.displacement && frame.kind == ViewKind::Scene).unwrap_or(&m.geo)
    }

    /// The scene without transmissive surfaces into `b` (MSAA resolved into mip 0), then its mips.
    fn behind_pass(&self, encoder: &mut wgpu::CommandEncoder, v: &ViewTargets, b: &Behind, globals_bg: &wgpu::BindGroup, frame: &Frame) {
        self.draw_scene(encoder, v, &b.level0, globals_bg, frame, &self.opaque_pipeline, None, None, &self.refr_black);
        let Some(pipe) = self.mip_pipelines.get(&self.hdr_format) else { return };
        for (dst, bg) in &b.mips {
            fullscreen(encoder, "tc_behind_mip", dst, CLEAR, pipe, bg);
        }
    }

    /// Subsurface blur and depth of field over this frame's scene (`v.hdr`); returns the result.
    fn scene_fx<'a>(
        &self,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
        v: &'a ViewTargets,
        f: &'a Fx,
        sss: bool,
        dof: bool,
    ) -> &'a wgpu::TextureView {
        let bg = |src: &wgpu::TextureView, extra: &[&wgpu::TextureView]| self.fx_bind_group(device, src, extra, &v.post_buf);
        let mut scene = &v.hdr;
        if sss {
            let [h, vert] = &self.sss_pipelines;
            fullscreen(encoder, "tc_sss_h", &f.a, CLEAR, h, &bg(&f.gbuf, &[]));
            fullscreen(encoder, "tc_sss_v", &f.b, CLEAR, vert, &bg(&f.a, &[&v.hdr]));
            scene = &f.b;
        }
        if dof {
            let target = |view| {
                Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    depth_slice: None,
                    ops: wgpu::Operations { load: CLEAR, store: wgpu::StoreOp::Store },
                })
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("tc_dof_gather"),
                color_attachments: &[target(&f.half), target(&f.near)],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(&self.dof_gather_pipeline);
            pass.set_bind_group(0, &bg(scene, &[&f.gbuf]), &[]);
            pass.draw(0..3, 0..1);
            drop(pass);
            fullscreen(encoder, "tc_dof", &f.a, CLEAR, &self.dof_pipeline, &bg(scene, &[&f.half, &f.gbuf, &f.near]));
            scene = &f.a;
        }
        scene
    }

    /// Background, mesh and wireframe into the view's MSAA HDR target, resolved to `resolve`.
    fn scene_pass(&self, encoder: &mut wgpu::CommandEncoder, v: &ViewTargets, resolve: &wgpu::TextureView, globals_bg: &wgpu::BindGroup, frame: &Frame) {
        let uv_ts = match frame.kind {
            ViewKind::Scene => None,
            ViewKind::Uv { ts, .. } => Some(ts),
        };
        let (pipe, wire) = match uv_ts {
            None => (&self.mesh_pipeline, frame.shade.wireframe_3d.then_some(&self.wire3d_pipeline)),
            Some(_) => (&self.uv_pipeline, frame.shade.wireframe.then_some(&self.wire_pipeline)),
        };
        // Transmissive surfaces refract what the behind pass rendered this frame.
        let refr = v.behind.as_ref().filter(|_| frame.shade.refraction).map_or(&self.refr_black, |b| &b.bg);
        let highlight = frame.shade.highlight.filter(|_| uv_ts.is_none()).map(|highlight| (&self.highlight_pipeline, highlight.set));
        self.draw_scene(encoder, v, resolve, globals_bg, frame, pipe, wire, highlight, refr);
    }

    /// One scene pass: background, then the mesh with `mesh_pipeline`, the wireframe with `wire` and one set's highlighted edges with `highlight`
    #[allow(clippy::too_many_arguments)]
    fn draw_scene(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        targets: &ViewTargets,
        resolve: &wgpu::TextureView,
        globals_bind_group: &wgpu::BindGroup,
        frame: &Frame,
        mesh_pipeline: &wgpu::RenderPipeline,
        wire: Option<&wgpu::RenderPipeline>,
        highlight: Option<(&wgpu::RenderPipeline, usize)>,
        refraction_source: &wgpu::BindGroup,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("tc_view"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &targets.msaa,
                resolve_target: Some(resolve),
                depth_slice: None,
                ops: wgpu::Operations { load: CLEAR, store: wgpu::StoreOp::Discard },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &targets.depth,
                depth_ops: Some(wgpu::Operations { load: wgpu::LoadOp::Clear(1.0), store: wgpu::StoreOp::Discard }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, globals_bind_group, &[]);
        pass.set_pipeline(&self.bg_pipeline);
        pass.draw(0..3, 0..1);
        let Some(mesh) = &self.mesh else { return };

        let uv_set = match frame.kind {
            ViewKind::Scene => None,
            ViewKind::Uv { ts, .. } => Some(ts),
        };
        pass.set_bind_group(2, refraction_source, &[]);
        let geometry = self.geometry(mesh, frame);
        let bake = frame.shade.bake.filter(|_| uv_set.is_none());
        let low_poly_shown = bake.is_none_or(|bake| bake.low_poly);
        let low_poly_set = bake.and_then(|bake| bake.only_set).or(uv_set);
        let (highlight, highlight_set) = highlight.filter(|(_, set)| low_poly_shown && low_poly_set.is_none_or(|only| only == *set)).unzip();

        // The wireframe and the highlight draw the mesh's own edges (displaced in the vertex shader too)
        let passes = [
            (low_poly_shown.then_some(mesh_pipeline), &geometry.vb, &geometry.ib, &geometry.draws, low_poly_set, 1),
            (wire.filter(|_| low_poly_shown), &mesh.geo.vb, &mesh.lines, &mesh.line_draws, low_poly_set, 1),
            (highlight, &mesh.geo.vb, &mesh.lines, &mesh.line_draws, highlight_set, HIGHLIGHT_SHIFTS),
        ];
        for (pipeline, vertices, indices, ranges, only_set, instances) in passes {
            let Some(pipeline) = pipeline else { continue };

            pass.set_pipeline(pipeline);
            pass.set_vertex_buffer(0, vertices.slice(..));
            pass.set_index_buffer(indices.slice(..), wgpu::IndexFormat::Uint32);
            for (range, set) in self.drawable(ranges, &frame.visible, only_set) {
                pass.set_bind_group(1, &set.bind_group, &[]);
                pass.draw_indexed(range, 0, 0..instances);
            }
        }

        if let Some(bake) = bake {
            self.draw_bake(&mut pass, mesh, frame, bake);
        }

        // Overlay quads last, so the mesh's depth hides what is behind it
        if uv_set.is_none()
            && let Some(overlay) = targets.overlay.as_ref().filter(|_| targets.overlay_vertices > 0)
        {
            pass.set_pipeline(&self.overlay_pipeline);
            pass.set_vertex_buffer(0, overlay.slice(..));
            pass.draw(0..targets.overlay_vertices, 0..1);
        }
    }

    /// Bake mode's high poly, UV seams and cages over the low poly `low_poly`
    fn draw_bake(&self, pass: &mut wgpu::RenderPass<'_>, low_poly: &MeshGpu, frame: &Frame, bake: BakeVisualization) {
        if bake.high_poly
            && let Some(high) = &self.bake_high
        {
            pass.set_pipeline(&self.bake_high_pipeline);
            pass.set_vertex_buffer(0, high.vertices.slice(..));
            pass.set_vertex_buffer(1, high.missed.slice(..));
            pass.set_index_buffer(high.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..high.index_count, 0, 0..1);
        }

        if bake.uv_seams
            && let Some(seams) = self.uv_seams.as_ref().filter(|seams| seams.mesh_id == low_poly.id)
        {
            pass.set_pipeline(&self.uv_seam_pipeline);
            pass.set_vertex_buffer(0, low_poly.geo.vb.slice(..));
            pass.set_index_buffer(seams.lines.slice(..), wgpu::IndexFormat::Uint32);
            for (range, set) in self.drawable(&seams.line_draws, &frame.visible, bake.only_set) {
                pass.set_bind_group(1, &set.bind_group, &[]);
                pass.draw_indexed(range, 0, 0..1);
            }
        }

        if !bake.cage {
            return;
        }

        let shown = |texture_set: usize| bake.only_set.is_none_or(|only| only == texture_set) && frame.visible.get(texture_set).copied().unwrap_or(true);
        let cages = || self.cages.iter().enumerate().filter(move |(texture_set, _)| shown(*texture_set)).filter_map(|(_, cage)| cage.as_ref());
        // The translucent faces first, then the wireframe over them
        pass.set_pipeline(&self.cage_pipeline);
        for cage in cages() {
            pass.set_vertex_buffer(0, cage.corners.slice(..));
            pass.draw(0..cage.corner_count, 0..1);
        }
        pass.set_pipeline(&self.cage_wire_pipeline);
        for cage in cages() {
            pass.set_vertex_buffer(0, cage.corners.slice(..));
            pass.set_index_buffer(cage.edges.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..cage.edge_index_count, 0, 0..1);
        }
    }

    /// Draw view `id` into the current pass at `rect` (target pixels: x, y, w, h).
    pub fn blit(&self, queue: &wgpu::Queue, pass: &mut wgpu::RenderPass<'_>, id: u64, rect: [f32; 4], target_size: [u32; 2]) {
        let Some(v) = self.views.get(&id).filter(|v| v.ready) else { return };
        let raw = BlitRaw { rect, dst: [target_size[0] as f32, target_size[1] as f32, if self.target_format.is_srgb() { 1.0 } else { 0.0 }, 0.0] };
        queue.write_buffer(&v.blit_buf, 0, bytemuck::bytes_of(&raw));
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &v.blit_bg, &[]);
        pass.draw(0..6, 0..1);
    }

    /// Write the blit rectangle for view `id` (call from `prepare` when the pass can't write
    /// buffers itself).
    pub fn set_blit_rect(&self, queue: &wgpu::Queue, id: u64, rect: [f32; 4], target_size: [u32; 2]) {
        if let Some(v) = self.views.get(&id) {
            let raw = BlitRaw { rect, dst: [target_size[0] as f32, target_size[1] as f32, if self.target_format.is_srgb() { 1.0 } else { 0.0 }, 0.0] };
            queue.write_buffer(&v.blit_buf, 0, bytemuck::bytes_of(&raw));
        }
    }

    /// Draw view `id` with the rectangle set by [`Renderer::set_blit_rect`].
    pub fn blit_prepared(&self, pass: &mut wgpu::RenderPass<'_>, id: u64) {
        let Some(v) = self.views.get(&id).filter(|v| v.ready) else { return };
        pass.set_pipeline(&self.blit_pipeline);
        pass.set_bind_group(0, &v.blit_bg, &[]);
        pass.draw(0..6, 0..1);
    }

    /// Read back the rendered view as tightly packed RGBA8 (sRGB-encoded), blocking.
    pub fn read_view(&self, device: &wgpu::Device, queue: &wgpu::Queue, id: u64) -> Option<(u32, u32, Vec<u8>)> {
        crate::readback::Readback::blocking(device, queue, &self.views.get(&id)?.resolve)
    }

    /// Forget a view's targets (e.g. a closed panel).
    pub fn drop_view(&mut self, id: u64) {
        self.views.remove(&id);
    }
}

/// Index ranges of `ranges` (first, count, material) whose texture set (from `material_sets`) is visible and is `only`, when given, with that set's index
fn drawable_ranges<'a>(
    ranges: &'a [(u32, u32, usize)],
    material_sets: &'a [Option<usize>],
    visible: &'a [bool],
    only: Option<usize>,
) -> impl Iterator<Item = (std::ops::Range<u32>, usize)> + 'a {
    ranges.iter().filter_map(move |&(start, count, material)| {
        let ts = material_sets.get(material).copied().flatten()?;
        if only.is_some_and(|o| o != ts) || !visible.get(ts).copied().unwrap_or(true) {
            return None;
        }

        Some((start..start + count, ts))
    })
}

/// A quad's two triangles, as corner indices
const QUAD_TRIANGLES: [usize; 6] = [0, 1, 2, 0, 2, 3];

/// Write `quads` into the view's overlay buffer, colors divided by the exposure only (undoing ACES makes saturated colors overshoot)
fn upload_overlay(device: &wgpu::Device, queue: &wgpu::Queue, view: &mut ViewTargets, quads: &[OverlayQuad], exposure_ev: Option<f32>) {
    view.overlay_vertices = 0;
    let Some(vertex_count) = quads.len().checked_mul(QUAD_TRIANGLES.len()).and_then(|count| u32::try_from(count).ok()) else {
        log::warn!("Symmetry overlay has too many vertices; hide the intersection or use a simpler mesh");
        return;
    };

    // A valid imported mesh can produce more intersection vertices than the GPU accepts
    let byte_count = u64::from(vertex_count) * std::mem::size_of::<Vertex>() as u64;
    if byte_count > device.limits().max_buffer_size {
        log::warn!("Symmetry overlay exceeds the GPU buffer limit; hide the intersection or use a simpler mesh");
        return;
    }

    if vertex_count == 0 {
        return;
    }

    let vertices: Vec<Vertex> = quads
        .iter()
        .flat_map(|quad| {
            let [r, g, b] = exposure_ev.map_or(quad.color, |exposure_ev| crate::set_highlight::scene_color_showing(quad.color, exposure_ev, false));
            let vertex = |corner: Vec3| Vertex { pos: corner.to_array(), nrm: [0.0; 3], tan: [r, g, b, quad.opacity], uv: [0.0; 2] };
            QUAD_TRIANGLES.map(|corner| vertex(quad.corners[corner]))
        })
        .collect();
    let bytes: &[u8] = bytemuck::cast_slice(&vertices);
    if view.overlay.as_ref().is_none_or(|buffer| buffer.size() < bytes.len() as u64) {
        view.overlay = Some(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tc_overlay"),
            size: bytes.len() as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        }));
    }
    if let Some(buffer) = &view.overlay {
        queue.write_buffer(buffer, 0, bytes);
        view.overlay_vertices = vertex_count;
    }
}

/// Globals `highlight`: the highlight's color and opacity, all zero without one. Data views skip exposure and tone mapping, so they take the color as it is
fn highlight_globals(shade: &Shade, data: bool) -> [f32; 4] {
    let Some(highlight) = shade.highlight else { return [0.0; 4] };

    let [r, g, b] = if data { highlight.color } else { highlight.scene_color(shade.exposure_ev, shade.aces) };
    [r, g, b, HIGHLIGHT_OPACITY]
}

/// Globals `highlight_shift`: how far each of the highlight's line copies moves sideways, in clip units of a `size` pixel target
fn highlight_shift(size: [u32; 2]) -> [f32; 4] {
    let half_extra_width = (HIGHLIGHT_WIDTH_PX - 1.0) * 0.5;
    let clip_per_pixel = size.map(|pixels| 2.0 / pixels.max(1) as f32);

    [half_extra_width * clip_per_pixel[0], half_extra_width * clip_per_pixel[1], 0.0, 0.0]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::headless::Headless;

    #[test]
    fn oversized_overlays_leave_the_view_usable() {
        let Ok(mut headless) = Headless::new() else { return };
        let limit = headless.device.limits().max_buffer_size as usize;
        let count = limit / (QUAD_TRIANGLES.len() * std::mem::size_of::<Vertex>()) + 1;
        let quad = OverlayQuad { corners: [Vec3::ZERO; 4], color: [1.0, 0.0, 0.0], opacity: 0.5 };
        let quads = vec![quad; count];
        let view = headless.renderer.targets(&headless.device, 1, [32, 32], false, false, false).unwrap();
        upload_overlay(&headless.device, &headless.queue, view, &[quad], None);

        upload_overlay(&headless.device, &headless.queue, view, &quads, None);

        assert_eq!(view.overlay_vertices, 0, "skip overlays beyond the device limit");
        upload_overlay(&headless.device, &headless.queue, view, &[quad], None);
        assert_eq!(view.overlay_vertices, 6, "the next frame can draw its overlay");
    }

    /// A headless renderer with one 256² texture set, and the base color slot's full-size
    /// texture handed over as WebGPU would: allocated, with its verdict still to come.
    fn awaiting() -> Option<(Headless, std::sync::mpsc::Sender<Option<String>>)> {
        let Ok(mut h) = Headless::new() else {
            eprintln!("skipping: no GPU adapter");
            return None;
        };
        h.renderer.set_texture_sets(&h.device, &h.queue, &[((256, 256), 0)], 1);
        let (tx, rx) = std::sync::mpsc::channel();
        let full = neutral_tex(&h.device, &h.queue, (256, 256), display_format(DisplayKind::BaseColor), mip_count(256), &[0, 0, 0, 255]);
        let r = &mut h.renderer;
        let set = r.sets.get_mut(0).unwrap();
        assert!(grow(set, 0, 0, Checked::Later(full, rx), (&h.device, &r.set_bgl, &r.sampler), &mut r.memory_failures));
        Some((h, tx))
    }

    fn base_width(h: &Headless) -> u32 {
        h.renderer.display_textures()[0][0].width()
    }

    fn prepare(h: &mut Headless) {
        let mut enc = h.device.create_command_encoder(&Default::default());
        h.renderer.resolve_pending(&h.device);
        h.renderer.generate_mips(&h.device, &mut enc);
        h.queue.submit([enc.finish()]);
    }

    #[test]
    fn a_late_texture_stays_unbound_until_accepted() {
        let Some((mut h, tx)) = awaiting() else { return };
        assert!(h.renderer.awaiting_gpu());
        // Tiles go into the pending texture; the placeholder stays bound and nothing is retried.
        h.renderer.upload_tile(&h.device, &h.queue, 0, DisplayKind::BaseColor, 0, 0, &[200, 30, 30, 255].repeat(64 * 64));
        prepare(&mut h);
        assert_eq!(base_width(&h), 1, "no verdict yet: the placeholder stays bound");
        assert!(h.renderer.awaiting_gpu());
        tx.send(None).unwrap();
        prepare(&mut h);
        assert_eq!(base_width(&h), 256, "accepted: installed");
        assert!(!h.renderer.awaiting_gpu());
        assert!(h.renderer.take_memory_failures().is_empty());
    }

    #[test]
    fn a_late_rejection_keeps_the_placeholder_and_is_reported() {
        let Some((mut h, tx)) = awaiting() else { return };
        tx.send(Some("texture set 1 (BaseColor, 256×256): Out of Memory".into())).unwrap();
        prepare(&mut h);
        assert_eq!(base_width(&h), 1);
        assert!(!h.renderer.awaiting_gpu());
        assert_eq!(h.renderer.take_memory_failures(), vec![0]);
        // The failed kind is not allocated again by later tiles.
        h.renderer.upload_tile(&h.device, &h.queue, 0, DisplayKind::BaseColor, 1, 0, &[200, 30, 30, 255].repeat(64 * 64));
        prepare(&mut h);
        assert_eq!(base_width(&h), 1);
        assert!(!h.renderer.awaiting_gpu());
    }

    #[test]
    fn the_highlight_draws_only_the_highlighted_sets_edges() {
        let line_draws = [(0, 4, 0), (4, 6, 1), (10, 2, 2)];
        let material_sets = [Some(0), Some(1), Some(2)];
        let highlighted = |visible: &[bool]| drawable_ranges(&line_draws, &material_sets, visible, Some(1)).collect::<Vec<_>>();

        assert_eq!(highlighted(&[true; 3]), vec![(4..10, 1)]);
        assert_eq!(highlighted(&[true, false, true]), vec![], "a hidden set shows no highlight");
    }

    #[test]
    fn a_data_view_takes_the_highlight_color_as_it_is() {
        let color = [0.8, 0.25, 0.08];
        let shade = Shade { highlight: Some(SetHighlight { set: 0, color }), exposure_ev: 1.0, ..Shade::default() };

        assert_eq!(highlight_globals(&shade, true), [0.8, 0.25, 0.08, HIGHLIGHT_OPACITY]);
        assert_ne!(highlight_globals(&shade, false)[..3], color, "the material view undoes exposure and tone mapping");
    }

    #[test]
    fn a_verdict_that_never_comes_counts_as_a_failure() {
        let Some((mut h, tx)) = awaiting() else { return };
        drop(tx);
        prepare(&mut h);
        assert_eq!(base_width(&h), 1);
        assert_eq!(h.renderer.take_memory_failures(), vec![0]);
    }
}
