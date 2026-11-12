//! Bridge between the session and the wgpu renderer: per-frame scene sync (mesh, texture set
//! layout, dirty tiles, environment) and the egui paint callback that renders and blits a view.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, PoisonError};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use rayon::prelude::*;
use texturecraft_engine::Session;
use texturecraft_engine::compose::{DisplayKind, ImageRes, tile_bytes};
use texturecraft_engine::core::BakedMap;
use texturecraft_engine::geom::Mesh;
use texturecraft_render::{BakeVisualization, Frame, Renderer, wgpu};

use crate::bake_scene::{BakeSceneLink, CageUpload};

/// Content shown through the aux texture.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AuxView {
    MeshMap(BakedMap),
    Mask(texturecraft_engine::core::LayerId),
    /// The quick mask being edited, over the base color.
    QuickMask,
}

/// A texture set's texels across and down, and its mesh material.
pub type SetLayout = ((u32, u32), usize);

/// The low poly's UV seam edges (vertex pairs) per material
pub type UvSeamEdges = Vec<Vec<(u32, u32)>>;

/// What the renderer must catch up with before drawing.
#[derive(Default)]
pub struct SceneSync {
    pub mesh: Option<(Arc<Mesh>, u64)>,
    /// (texels across and down, material) per texture set, the mesh material count and the
    /// viewport texture resolution (display shift).
    pub sets: Option<(Vec<SetLayout>, usize, u32)>,
    /// The environment name and, for `resource:<name>`, its decoded image.
    pub env: Option<(String, Option<Arc<ImageRes>>)>,
    pub tiles: Vec<(usize, DisplayKind, u32, u32, Vec<u8>)>,
    pub aux: Vec<(usize, u32, Vec<u8>)>,
    /// Bake mode's high poly (`None` removes it) and the id it is sent under
    pub bake_high: Option<(Option<Arc<Mesh>>, u64)>,
    /// Which vertices of the high poly (by its id) the bake's rays miss (1) or reach (0)
    pub bake_high_missed: Option<(u64, Vec<f32>)>,
    /// Changed cages: texture set, the cage or `None` to remove it, its topology
    pub bake_cages: Vec<(usize, Option<CageUpload>, u64)>,
    /// Texture sets left after some were removed: their cages go too
    pub bake_cage_count: Option<usize>,
    /// The low poly's UV seams per material, for mesh id
    pub uv_seams: Option<(u64, UvSeamEdges)>,
}

/// Lives in egui's callback resources.
pub struct GpuState {
    pub renderer: Renderer,
    /// Raised when texture-set textures were recreated: the UI must upload every tile again.
    pub reset: Arc<AtomicBool>,
    pub reports: Arc<Mutex<GpuReports>>,
    /// Wakes the UI when the GPU side has something to report.
    pub ctx: egui::Context,
}

/// Problems the GPU side hands to the UI, which shows them as toasts.
#[derive(Default)]
pub struct GpuReports {
    /// Texture sets whose display textures could not be allocated (out of GPU memory).
    pub memory: Vec<usize>,
    /// wgpu errors no error scope caught (logged instead of wgpu's default panic).
    pub errors: Vec<String>,
}

/// Most distinct uncaptured GPU errors kept until the UI shows them.
const MAX_GPU_ERRORS: usize = 8;

/// UI-side bookkeeping of what the GPU already has.
pub struct GpuLink {
    pub reset: Arc<AtomicBool>,
    pub reports: Arc<Mutex<GpuReports>>,
    /// Raised when a view callback was dropped before the GPU applied its scene sync (the
    /// frame was never painted, e.g. a WebGPU surface not ready yet): send everything again.
    pub lost: Arc<AtomicBool>,
    mesh_id: u64,
    sets: Vec<SetLayout>,
    /// The display shift the GPU has (`Prefs::viewport_texture`).
    shift: u32,
    /// The environment the GPU has: name and image identity (a re-imported resource differs).
    env: (String, usize),
    /// What each set's aux texture holds: (mesh map or mask key, revision).
    aux: BTreeMap<usize, (String, u64)>,
    pub adapter: String,
    /// Most tiles waiting to be composited since compositing last caught up (progress).
    composite_total: usize,
    bake_scene: BakeSceneLink,
    scene_sync: Option<(u64, Arc<SceneSyncSlot>)>,
}

/// Compositing time per frame: a full recomposite of a large texture set spreads over frames.
const COMPOSITE_BUDGET_MS: f64 = 30.0;
/// The most a frame spends on the scene (with the work already done this frame, such as a
/// stroke's commit): compositing and display conversion leave the rest for the next frames, so
/// a large (8K) texture set never stalls one frame.
const FRAME_BUDGET_MS: f64 = 35.0;
/// Each frame still makes this much progress.
const MIN_SLICE_MS: f64 = 4.0;
/// Display tiles converted between budget checks.
const UPLOAD_CHUNK: usize = 64;
/// Estimated GPU upload cost of one display tile (write_texture of one 64×64 tile of one
/// display kind, measured at ~5.5 µs on a desktop GPU), counted against the frame budget.
const UPLOAD_MS_PER_TILE: f64 = 0.006;

pub fn init(rs: &egui_wgpu::RenderState, ctx: &egui::Context) -> GpuLink {
    let reset = Arc::new(AtomicBool::new(false));
    let reports = Arc::new(Mutex::new(GpuReports::default()));
    // wgpu's default handler panics on any error no scope caught (an out-of-memory texture, a
    // driver hiccup): report it instead, so the user keeps the session.
    let (sink, wake) = (reports.clone(), ctx.clone());
    rs.device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
        log::error!("GPU error: {e}");
        let mut r = sink.lock().unwrap_or_else(PoisonError::into_inner);
        let text = e.to_string();
        if r.errors.len() < MAX_GPU_ERRORS && !r.errors.contains(&text) {
            r.errors.push(text);
        }
        // The UI may be idle: draw a frame to show it.
        wake.request_repaint();
    }));
    let renderer = Renderer::new(&rs.device, &rs.queue, rs.target_format);
    rs.renderer.write().callback_resources.insert(GpuState { renderer, reset: reset.clone(), reports: reports.clone(), ctx: ctx.clone() });
    let info = rs.adapter.get_info();
    GpuLink {
        reset,
        reports,
        lost: Arc::new(AtomicBool::new(false)),
        mesh_id: 0,
        sets: Vec::new(),
        shift: 0,
        env: Default::default(),
        aux: BTreeMap::new(),
        adapter: format!("{} ({:?})", info.name, info.backend),
        composite_total: 0,
        bake_scene: BakeSceneLink::default(),
        scene_sync: None,
    }
}

impl GpuLink {
    /// What the GPU side reported since the last call (see [`GpuReports`]).
    pub fn take_reports(&self) -> GpuReports {
        std::mem::take(&mut *self.reports.lock().unwrap_or_else(PoisonError::into_inner))
    }

    /// Whether bake mode's visualization is still being worked out off the UI thread
    pub fn bake_busy(&self) -> bool {
        self.bake_scene.busy()
    }

    /// Compositing progress (0..1) while tiles wait to be composited.
    pub fn composite_progress(&self, s: &Session) -> Option<f32> {
        let pending = s.pending_tiles();
        (pending > 0 && self.composite_total > 0).then(|| 1.0 - pending as f32 / self.composite_total as f32)
    }

    /// Collect everything the GPU needs this frame, running the compositor first: `bake` is what bake mode's 3D view draws besides the low poly, `spent_ms` the time this frame already took
    pub fn sync(&mut self, session: &mut Session, aux_map: Option<AuxView>, bake: Option<BakeVisualization>, shift: u32, spent_ms: f64) -> SceneSync {
        let start_ms = crate::now_ms();
        let mut queued = 0usize;
        let time_left_ms = |queued: usize| FRAME_BUDGET_MS - spent_ms - (crate::now_ms() - start_ms) - queued as f64 * UPLOAD_MS_PER_TILE;
        let pending = session.pending_tiles();
        self.composite_total = if pending == 0 { 0 } else { self.composite_total.max(pending) };
        session.update_for(COMPOSITE_BUDGET_MS.min(time_left_ms(0)).max(MIN_SLICE_MS), crate::now_ms);

        let mut out = SceneSync::default();
        if self.lost.swap(false, Ordering::AcqRel) {
            self.mesh_id = 0;
            self.sets.clear();
            self.env = Default::default();
            self.bake_scene.resend();
            self.reset.store(true, Ordering::Release);
        }

        if self.mesh_id != session.mesh_id {
            out.mesh = Some((session.mesh.clone(), session.mesh_id));
            self.mesh_id = session.mesh_id;
            self.aux.clear();
        }

        let sets: Vec<SetLayout> = session.doc.project.texture_sets.iter().map(|texture_set| (texture_set.size(), texture_set.material)).collect();
        if sets != self.sets || shift != self.shift || out.mesh.is_some() {
            out.sets = Some((sets.clone(), session.mesh.materials.len().max(1), shift));
            self.sets = sets;
            self.shift = shift;
        }

        let display = &session.doc.project.display;
        let image = display.environment_resource().and_then(|name| session.images.get(name));
        let image_identity = image.map_or(0, |image| Arc::as_ptr(image) as usize);
        if self.env.0 != display.environment || self.env.1 != image_identity {
            self.env = (display.environment.clone(), image_identity);
            out.env = Some((display.environment.clone(), image.cloned()));
        }

        if self.reset.swap(false, Ordering::AcqRel) {
            for cache in &mut session.caches {
                cache.upload.extend(texturecraft_engine::compose::all_tiles(cache.size.0, cache.size.1));
            }
            self.aux.clear();
        }

        for texture_set in 0..session.caches.len() {
            let tiles = session.take_uploads(texture_set);
            if tiles.is_empty() {
                continue;
            }

            let mut done = 0;
            if let Some(cache) = session.caches.get(texture_set) {
                let empty = Default::default();
                let baked = session.doc.baked.get(&texture_set).unwrap_or(&empty);
                let composite = &cache.composite;
                // In chunks while the frame has time (at least one chunk); the rest waits
                for chunk in tiles.chunks(UPLOAD_CHUNK) {
                    let bytes: Vec<_> = chunk
                        .par_iter()
                        .flat_map_iter(|&(tile_x, tile_y)| DisplayKind::ALL.into_iter().map(move |kind| (kind, tile_x, tile_y)))
                        .map(|(kind, tile_x, tile_y)| (texture_set, kind, tile_x, tile_y, tile_bytes(composite, baked, kind, tile_x, tile_y)))
                        .collect();
                    queued += bytes.len();
                    out.tiles.extend(bytes);
                    done += chunk.len();
                    if time_left_ms(queued) < MIN_SLICE_MS {
                        break;
                    }
                }
            }
            if let (Some(rest), Some(cache)) = (tiles.get(done..), session.caches.get_mut(texture_set)) {
                cache.upload.extend(rest.iter().copied());
            }
        }

        match aux_map {
            Some(AuxView::MeshMap(map)) => {
                for texture_set in 0..session.caches.len() {
                    // While a bake shows, its streamed maps (neutral where not baked yet) replace the document's
                    let preview = session.bake_preview_baked(texture_set);
                    let baked = preview.or_else(|| session.doc.baked.get(&texture_set));
                    let have = baked.and_then(|baked| baked.maps.get(&map));
                    let revision = have.map(|data| Arc::as_ptr(data) as u64).unwrap_or(0);
                    let key = (format!("{map:?}{}", if preview.is_some() { "~" } else { "" }), revision);
                    if self.aux.get(&texture_set) == Some(&key) {
                        continue;
                    }
                    let Some(cache) = session.caches.get(texture_set) else { continue };

                    let resolution = texturecraft_render::aux_size(cache.size);
                    let rgba = if preview.is_some() && have.is_none() {
                        texturecraft_render::mesh_maps::neutral_rgba(map, resolution)
                    } else {
                        texturecraft_render::mesh_maps::mesh_map_rgba(baked, map, resolution, cache.size, &cache.composite.block_scales)
                    };
                    out.aux.push((texture_set, resolution, rgba));
                    self.aux.insert(texture_set, key);
                }
            }
            Some(AuxView::Mask(id)) => {
                if let Some((texture_set, _, _)) = session.doc.project.locate(id) {
                    let key = (format!("mask{}", id.0), session.revision);
                    if self.aux.get(&texture_set) != Some(&key) && !session.stroke_active() {
                        let resolution =
                            session.caches.get(texture_set).map_or(MASK_VIEW_FALLBACK_RESOLUTION, |cache| texturecraft_render::aux_size(cache.size));
                        if let Some(rgba) = session.mask_view_rgba(id, resolution) {
                            out.aux.push((texture_set, resolution, rgba));
                        }
                        self.aux.insert(texture_set, key);
                    }
                }
            }
            Some(AuxView::QuickMask) => {
                for texture_set in 0..session.caches.len() {
                    let Some(cache) = session.caches.get(texture_set) else { continue };

                    let key = (format!("quickmask{}", session.quick_mask.revision), cache.revision);
                    if self.aux.get(&texture_set) == Some(&key) {
                        continue;
                    }
                    let resolution = texturecraft_render::aux_size(cache.size);
                    if let Some(rgba) = session.quick_mask_rgba(texture_set, resolution) {
                        out.aux.push((texture_set, resolution, rgba));
                    }
                    self.aux.insert(texture_set, key);
                }
            }
            None => {}
        }

        self.bake_scene.sync(session, bake, &mut out);
        out
    }
}

/// Mask view resolution for a texture set without a composite cache
const MASK_VIEW_FALLBACK_RESOLUTION: u32 = 1024;

/// Paint callback for one view.
pub struct ViewCallback {
    pub frame: Frame,
    /// Scene changes, applied by the first view drawn this frame.
    pub sync: Option<Arc<SceneSyncSlot>>,
    pub rect: egui::Rect,
}

impl CallbackTrait for ViewCallback {
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = resources.get_mut::<GpuState>() else { return Vec::new() };
        let sync = self.sync.as_ref().and_then(|slot| slot.take());
        if let Some(sync) = sync {
            gpu.apply_sync(device, queue, sync);
        }

        // Also installs (or rejects) full-size textures WebGPU answered for since last frame
        gpu.renderer.prepare(device, queue, encoder, &self.frame);
        gpu.report_pending();

        let pixels_per_point = screen.pixels_per_point;
        let rect = self.rect;
        let blit_rect = [rect.min.x * pixels_per_point, rect.min.y * pixels_per_point, rect.width() * pixels_per_point, rect.height() * pixels_per_point];
        gpu.renderer.set_blit_rect(queue, self.frame.view_id, blit_rect, screen.size_in_pixels);
        Vec::new()
    }

    fn paint(&self, info: egui::PaintCallbackInfo, pass: &mut wgpu::RenderPass<'static>, resources: &CallbackResources) {
        let Some(gpu) = resources.get::<GpuState>() else { return };
        let [sw, sh] = info.screen_size_px;
        let clip = info.clip_rect_in_pixels();
        if clip.width_px <= 0 || clip.height_px <= 0 || sw == 0 || sh == 0 {
            return;
        }
        pass.set_viewport(0.0, 0.0, sw as f32, sh as f32, 0.0, 1.0);
        pass.set_scissor_rect(clip.left_px.max(0) as u32, clip.top_px.max(0) as u32, clip.width_px as u32, clip.height_px as u32);
        gpu.renderer.blit_prepared(pass, self.frame.view_id);
    }
}

/// Final-owner tracking: a clipped ordinary view can hand its sync to the native companion.
pub struct SceneSyncSlot {
    sync: Mutex<Option<SceneSync>>,
    lost: Arc<AtomicBool>,
}

impl SceneSyncSlot {
    pub fn take(&self) -> Option<SceneSync> {
        self.sync.lock().unwrap_or_else(PoisonError::into_inner).take()
    }
}

impl Drop for SceneSyncSlot {
    fn drop(&mut self) {
        if self.sync.get_mut().unwrap_or_else(PoisonError::into_inner).is_some() {
            self.lost.store(true, Ordering::Release);
        }
    }
}

impl GpuLink {
    pub fn current_sync(&mut self, frame: u64) -> Option<Arc<SceneSyncSlot>> {
        if self.scene_sync.as_ref().is_some_and(|(number, _)| *number != frame) {
            self.scene_sync = None;
        }
        self.scene_sync.as_ref().map(|(_, slot)| slot.clone())
    }

    pub fn remember_sync(&mut self, frame: u64, sync: SceneSync) -> Arc<SceneSyncSlot> {
        let slot = Arc::new(SceneSyncSlot { sync: Mutex::new(Some(sync)), lost: self.lost.clone() });
        self.scene_sync = Some((frame, slot.clone()));
        slot
    }
}

impl GpuState {
    pub fn apply_sync(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, sync: SceneSync) {
        if let Some((env, image)) = &sync.env {
            self.renderer.set_environment(device, queue, env, image.as_deref());
        }
        if let Some((mesh, id)) = &sync.mesh {
            self.renderer.set_mesh(device, queue, mesh, *id);
        }
        if let Some((sets, material_count, shift)) = &sync.sets {
            self.renderer.set_display_shift(*shift);
            let reset = self.renderer.set_texture_sets(device, queue, sets, *material_count);
            if !reset.is_empty() {
                self.reset.store(true, Ordering::Release);
            }
        }
        for (texture_set, kind, tile_x, tile_y, bytes) in &sync.tiles {
            self.renderer.upload_tile(device, queue, *texture_set, *kind, *tile_x, *tile_y, bytes);
        }
        for (texture_set, size, bytes) in &sync.aux {
            self.renderer.upload_aux(device, queue, *texture_set, *size, bytes);
        }
        if let Some((high, id)) = &sync.bake_high
            && let Err(error) = self.renderer.set_bake_high(device, queue, high.as_deref(), *id)
        {
            let mut reports = self.reports.lock().unwrap_or_else(PoisonError::into_inner);
            if reports.errors.len() < MAX_GPU_ERRORS && !reports.errors.contains(&error) {
                reports.errors.push(error);
            }
            self.ctx.request_repaint();
        }
        if let Some((id, missed)) = &sync.bake_high_missed {
            self.renderer.set_bake_high_missed(queue, *id, missed);
        }
        if let Some(count) = sync.bake_cage_count {
            self.renderer.truncate_bake_cages(count);
        }
        for (texture_set, cage, topology) in &sync.bake_cages {
            let cage = cage.as_ref().map(|upload| (upload.corners.as_slice(), upload.shell.edges.as_slice()));
            self.renderer.set_bake_cage(device, queue, *texture_set, cage, *topology);
        }
        if let Some((mesh_id, seams)) = &sync.uv_seams {
            self.renderer.set_uv_seams(device, queue, *mesh_id, seams);
        }
    }
}

impl GpuState {
    pub fn report_pending(&mut self) {
        let failed = self.renderer.take_memory_failures();
        if !failed.is_empty() {
            let mut reports = self.reports.lock().unwrap_or_else(PoisonError::into_inner);
            for texture_set in failed {
                if !reports.memory.contains(&texture_set) {
                    reports.memory.push(texture_set);
                }
            }
            // This frame's report drain already ran: the next frame shows the toast
            self.ctx.request_repaint();
        }
        if self.renderer.awaiting_gpu() {
            self.ctx.request_repaint();
        }
    }
}

#[cfg(test)]
mod sync_slot_tests {
    use super::*;
    #[test]
    fn only_the_final_unapplied_owner_marks_a_lost_sync() {
        let lost = Arc::new(AtomicBool::new(false));
        let slot = Arc::new(SceneSyncSlot { sync: Mutex::new(Some(SceneSync::default())), lost: lost.clone() });
        let native = slot.clone();
        drop(slot);
        assert!(!lost.load(Ordering::Acquire));
        assert!(native.take().is_some());
        assert!(native.take().is_none());
        drop(native);
        assert!(!lost.load(Ordering::Acquire));
        drop(SceneSyncSlot { sync: Mutex::new(Some(SceneSync::default())), lost: lost.clone() });
        assert!(lost.load(Ordering::Acquire));
    }
}
