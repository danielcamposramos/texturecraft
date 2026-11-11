//! Windowless renders of a session: the desktop app's per-frame GPU sync (`ui-egui/src/gpu.rs`)
//! and viewport shading (`ui-egui/src/viewport.rs`), done in one shot.

use rayon::prelude::*;
use texturecraft_engine::compose::{DisplayKind, all_tiles, tile_bytes};
use texturecraft_engine::{Session, ViewMode};
use texturecraft_render::displace::trace_geometry;
use texturecraft_render::gpu_trace::GpuTracer;
use texturecraft_render::trace::{TraceDisplay, TraceScene, TraceSet, TraceSettings, Tracer};
use texturecraft_render::{Frame, Shade, ViewKind};

/// A headless GPU device that remembers what it already holds of the session.
pub struct Gpu {
    h: texturecraft_render::headless::Headless,
    mesh_id: u64,
    sets: Vec<((u32, u32), usize)>,
    env: String,
    /// The GPU path tracer, created by the first [`Gpu::path_trace`].
    tracer: Option<GpuTracer>,
}

impl Gpu {
    /// Errors when there is no GPU (or software) adapter.
    pub fn new() -> Result<Gpu, String> {
        Ok(Gpu { h: texturecraft_render::headless::Headless::new()?, mesh_id: 0, sets: Vec::new(), env: String::new(), tracer: None })
    }

    pub fn adapter(&self) -> &str {
        &self.h.adapter_name
    }

    /// Bring the GPU up to date with `s` (mesh, texture sets, environment, dirty tiles).
    fn sync(&mut self, s: &mut Session) {
        s.update();
        let h = &mut self.h;
        let mut reupload = false;
        if self.mesh_id != s.mesh_id {
            h.renderer.set_mesh(&h.device, &h.queue, &s.mesh, s.mesh_id);
            self.mesh_id = s.mesh_id;
            self.sets.clear();
            reupload = true;
        }
        let sets: Vec<((u32, u32), usize)> = s.doc.project.texture_sets.iter().map(|t| (t.size(), t.material)).collect();
        if sets != self.sets {
            h.renderer.set_texture_sets(&h.device, &h.queue, &sets, s.mesh.materials.len().max(1));
            self.sets = sets;
            reupload = true;
        }
        let d = &s.doc.project.display;
        if d.environment != self.env {
            let image = d.environment_resource().and_then(|n| s.images.get(n));
            h.renderer.set_environment(&h.device, &h.queue, &d.environment, image.map(|i| i.as_ref()));
            self.env.clone_from(&d.environment);
        }
        for ts in 0..s.caches.len() {
            let mut tiles = s.take_uploads(ts);
            let Some(cache) = s.caches.get(ts) else { continue };
            if reupload {
                tiles = all_tiles(cache.size.0, cache.size.1);
            }
            let empty = Default::default();
            let baked = s.doc.baked.get(&ts).unwrap_or(&empty);
            let comp = &cache.composite;
            let bytes: Vec<_> = tiles
                .par_iter()
                .flat_map_iter(|&(tx, ty)| DisplayKind::ALL.into_iter().map(move |k| (k, tx, ty)))
                .map(|(k, tx, ty)| (k, tx, ty, tile_bytes(comp, baked, k, tx, ty)))
                .collect();
            for (k, tx, ty, b) in bytes {
                h.renderer.upload_tile(&h.device, &h.queue, ts, k, tx, ty, &b);
            }
            // The aux texture behind the mesh-map and mask views (rebuilt every render).
            let res = texturecraft_render::aux_size(cache.size);
            let aux = match s.view {
                ViewMode::MeshMap(m) => Some(texturecraft_render::mesh_maps::mesh_map_rgba(s.doc.baked.get(&ts), m, res, cache.size, &comp.block_scales)),
                ViewMode::Mask(id) if s.doc.project.locate(id).is_some_and(|(t, _, _)| t == ts) => s.mask_view_rgba(id, res),
                _ => None,
            };
            if let Some(rgba) = aux {
                h.renderer.upload_aux(&h.device, &h.queue, ts, res, &rgba);
            }
        }
    }

    /// Render `s` with `kind` at `size` through `camera`; RGBA8 (sRGB) pixels.
    pub fn render(&mut self, s: &mut Session, kind: ViewKind, size: [u32; 2], camera: texturecraft_engine::geom::OrbitCamera) -> Option<(u32, u32, Vec<u8>)> {
        self.sync(s);
        let frame = Frame {
            view_id: 1,
            size,
            kind,
            camera,
            shade: shade(s),
            visible: s.doc.project.texture_sets.iter().map(|t| t.visible).collect(),
            overlay: Vec::new(),
        };
        self.h.render(&frame)
    }

    /// [`path_trace`] on the GPU (compute shaders); errors when the adapter or the scene does
    /// not allow it.
    pub fn path_trace(
        &mut self,
        s: &mut Session,
        size: [u32; 2],
        camera: texturecraft_engine::geom::OrbitCamera,
        samples: u32,
        settings: &TraceSettings,
    ) -> Result<(u32, u32, Vec<u8>), String> {
        self.sync(s);
        let (scene, display) = trace_inputs(s);
        let h = &self.h;
        let tracer = match &mut self.tracer {
            Some(t) => t,
            None => self.tracer.insert(GpuTracer::new(&h.device)?),
        };
        tracer.render(&h.device, &h.queue, &h.renderer, &scene, &camera, settings, &display, size, samples)
    }
}

/// The session's 3D view as the path tracers see it, and how Rendering mode shows it (denoised).
fn trace_inputs(s: &mut Session) -> (TraceScene, TraceDisplay) {
    s.update();
    let sets: Vec<TraceSet> = s
        .doc
        .project
        .texture_sets
        .iter()
        .zip(&s.caches)
        .map(|(ts, c)| TraceSet { material: ts.material, composite: c.composite.clone(), height_strength: ts.atlas_height_strength() })
        .collect();
    let d = &s.doc.project.display;
    let env_image = d.environment_resource().and_then(|n| s.images.get(n));
    let (mesh, bvh) = trace_geometry(&s.mesh, &s.bvh, &sets, d);
    let scene = TraceScene::for_display(mesh, bvh, sets, d, env_image.map(|i| i.as_ref()));
    (scene, TraceDisplay { denoise: true, ..TraceDisplay::from_display(d) })
}

/// A progressive CPU path trace of the session's 3D view (the app's Rendering mode), finished
/// after `samples` passes and denoised; RGBA8 (sRGB) pixels.
pub fn path_trace(
    s: &mut Session,
    size: [u32; 2],
    camera: texturecraft_engine::geom::OrbitCamera,
    samples: u32,
    settings: &TraceSettings,
) -> (u32, u32, Vec<u8>) {
    let (scene, display) = trace_inputs(s);
    let mut tracer = Tracer::new(size[0], size[1]);
    for _ in 0..samples {
        tracer.pass(&scene, &camera, settings);
    }
    (tracer.width, tracer.height, tracer.rgba8(&display))
}

/// How the viewport shades the session (the app's `viewport::shade`).
pub fn shade(s: &Session) -> Shade {
    let (view_mode, channel) = match s.view {
        ViewMode::Material => (0, 0),
        ViewMode::Channel(c) => (1, c.index() as u32),
        ViewMode::MeshMap(_) | ViewMode::Mask(_) => (2, 0),
        ViewMode::UvChecker => (3, 0),
    };
    Shade {
        height_strength: s.doc.project.texture_sets.get(s.ts()).map(|t| t.atlas_height_strength()).unwrap_or(1.0),
        view_mode,
        channel,
        // The app's default (only the 2D view draws it).
        wireframe: true,
        ..Shade::from_project(&s.doc.project)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use texturecraft_engine::execute;

    /// A 64×32 Radiance `.hdr` (flat RGBE scanlines): a hot orange sky over a dim blue ground.
    fn sky_hdr() -> Vec<u8> {
        let rgbe = |c: [f32; 3]| {
            let e = c.iter().copied().fold(0.0f32, f32::max).log2().floor() as i32 + 1;
            let k = 256.0 / 2f32.powi(e);
            [(c[0] * k) as u8, (c[1] * k) as u8, (c[2] * k) as u8, (e + 128) as u8]
        };
        let mut out = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y 32 +X 64\n".to_vec();
        for y in 0..32 {
            let px = rgbe(if y < 16 { [12.0, 4.0, 1.0] } else { [0.02, 0.05, 0.2] });
            out.extend((0..64).flat_map(|_| px));
        }
        out
    }

    #[test]
    fn renders_with_an_imported_hdr_environment() {
        let mesh = texturecraft_engine::geom::primitives::material_ball();
        let mut s = Session::new_project("t", mesh, "builtin:material_ball", None, 256).unwrap();
        let name = s.import_resource_bytes("sky.hdr", sky_hdr()).unwrap();
        assert!(s.images[&name].rgba[0][0] > 10.0, "the decoder keeps HDR values");
        let Ok(mut gpu) = Gpu::new() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let cam = s.camera;
        let (_, _, base) = gpu.render(&mut s, ViewKind::Scene, [160, 120], cam).unwrap();
        execute(&mut s, "display.set", &json!({"environment": format!("resource:{name}"), "specGloss": true, "parallax": true})).unwrap();
        let (_, _, lit) = gpu.render(&mut s, ViewKind::Scene, [160, 120], cam).unwrap();
        assert_ne!(base, lit, "the HDR environment changes the render");
        let warm = |px: &[u8]| px.chunks(4).map(|p| p[0] as f32 - p[2] as f32).sum::<f32>();
        assert!(warm(&lit) > warm(&base), "the orange sky warms the image");
    }

    #[test]
    fn gpu_path_trace_matches_the_cpu_on_the_material_ball() {
        let mesh = texturecraft_engine::geom::primitives::material_ball();
        let mut s = Session::new_project("t", mesh, "builtin:material_ball", None, 256).unwrap();
        execute(&mut s, "layer.add.fill", &json!({"color": [0.75, 0.42, 0.23], "roughness": 0.3, "metallic": 0.4})).unwrap();
        let Ok(mut gpu) = Gpu::new() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let (cam, settings, size) = (s.camera, TraceSettings { bounces: 4 }, [64, 48]);
        let g = match gpu.path_trace(&mut s, size, cam, 384, &settings) {
            Ok((_, _, px)) => px,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        let (_, _, c) = path_trace(&mut s, size, cam, 384, &settings);
        // Mean luminance per 8×8 block (8-bit levels) agrees.
        let lum = |img: &[u8], x: usize, y: usize| {
            let p = &img[(y * 64 + x) * 4..(y * 64 + x) * 4 + 3];
            0.2126 * p[0] as f32 + 0.7152 * p[1] as f32 + 0.0722 * p[2] as f32
        };
        let block = |img: &[u8], bx: usize, by: usize| (0..64).map(|i| lum(img, bx * 8 + i % 8, by * 8 + i / 8)).sum::<f32>() / 64.0;
        let worst = (0..6).flat_map(|by| (0..8).map(move |bx| (bx, by))).map(|(bx, by)| (block(&g, bx, by) - block(&c, bx, by)).abs()).fold(0.0, f32::max);
        assert!(worst < 4.0, "worst block differs by {worst} levels");
        assert!(g.chunks(4).any(|p| p[0] > p[2] + 40), "the orange fill shows");
    }

    #[test]
    fn uv_tile_atlas_renders_tiles_side_by_side() {
        let mesh = texturecraft_engine::geom::primitives::udim_plane();
        let mut s = Session::new_project("t", mesh, "builtin:udim_plane", None, 128).unwrap();
        execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
        execute(&mut s, "tool.brush", &json!({"size": 10, "hardness": 100})).unwrap();
        // Atlas UV 0.75 is the middle of tile 1002.
        execute(&mut s, "tool.strokeUv", &json!({"points": [[0.75, 0.5]]})).unwrap();
        let Ok(mut gpu) = Gpu::new() else {
            eprintln!("skipping: no GPU adapter");
            return;
        };
        let cam = s.camera;
        // A 2:1 atlas fills a 2:1 view's middle 92%: tile 1001 left of center, 1002 right.
        let (w, _, px) = gpu.render(&mut s, ViewKind::Uv { ts: 0, zoom: 1.0, pan: [0.0, 0.0] }, [200, 100], cam).unwrap();
        let at = |x: u32, y: u32| px.get(((y * w + x) * 4) as usize..((y * w + x) * 4 + 3) as usize).unwrap().to_vec();
        let red = |p: &[u8]| i32::from(p[0]) - i32::from(p[1]) > 60;
        let (left, right) = (at(77, 50), at(123, 50));
        assert!(red(&right), "tile 1002 shows the stroke: {right:?}");
        assert!(!red(&left), "tile 1001 does not: {left:?}");
        let (_, _, scene) = gpu.render(&mut s, ViewKind::Scene, [160, 120], cam).unwrap();
        assert!(scene.chunks(4).any(red), "the 3D view samples the atlas");
    }
}
