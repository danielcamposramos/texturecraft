//! Rendering mode (F10): a progressive path-traced view of the current camera, restarted whenever
//! the scene, camera, view size or settings change.
//!
//! Where the device has compute shaders (desktop, WebGPU) it traces on the GPU: the tracer lives in
//! egui's callback resources next to the viewport renderer (whose texture sets it copies), a paint
//! callback adds samples each frame and resolves into a texture egui shows as it is, so nothing is
//! read back until the render is saved. Otherwise (WebGL2, the CPU setting, or a GPU error) the CPU
//! tracer runs on a worker thread (one slice per frame on the single-threaded web).

use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, PoisonError};

use eframe::egui_wgpu::{self, CallbackResources, CallbackTrait, ScreenDescriptor};
use egui::{Color32, ColorImage, Rect, RichText, TextureHandle, TextureOptions, Ui, vec2};
use texturecraft_engine::compose::ImageRes;
use texturecraft_engine::geom::{Bvh, Mesh, OrbitCamera};
use texturecraft_render::displace::trace_geometry;
use texturecraft_render::gpu_trace::GpuTracer;
use texturecraft_render::ibl::EnvImage;
use texturecraft_render::readback::Readback;
use texturecraft_render::trace::{TraceDisplay, TraceScene, TraceSet, TraceSettings, Tracer};
use texturecraft_render::{Renderer, wgpu};

use crate::TextureCraftApp;
use crate::i18n::{n_, tr, trf};
use crate::theme::Tokens;

mod frame_floor;
mod motion;
mod spp_control;
#[cfg(test)]
mod tests;

use motion::Motion;
use spp_control::{SppControl, rescale_spp};

/// A finished pass, ready to show.
struct Frame {
    width: u32,
    height: u32,
    rgba: Vec<u8>,
    samples: u32,
}

/// The tracer and everything it needs.
struct Work {
    scene: TraceScene,
    camera: OrbitCamera,
    settings: TraceSettings,
    display: TraceDisplay,
    max_samples: u32,
    tracer: Tracer,
}

impl Work {
    fn frame(&self) -> Frame {
        Frame { width: self.tracer.width, height: self.tracer.height, rgba: self.tracer.rgba8(&self.display), samples: self.tracer.samples }
    }

    /// One more sample per pixel, or None when done.
    #[cfg(not(target_arch = "wasm32"))]
    fn step(&mut self) -> Option<Frame> {
        if self.tracer.samples >= self.max_samples {
            return None;
        }
        self.tracer.pass(&self.scene, &self.camera, &self.settings);
        Some(self.frame())
    }

    /// Trace for about `budget_ms` of this frame (the web has one thread: the UI must keep
    /// running); a frame to show when a pass finished.
    #[cfg(target_arch = "wasm32")]
    fn step_for(&mut self, budget_ms: f64) -> Option<Frame> {
        let start = js_sys::Date::now();
        let mut rows = 4;
        while self.tracer.samples < self.max_samples {
            let t = js_sys::Date::now();
            if self.tracer.pass_rows(&self.scene, &self.camera, &self.settings, rows) {
                return Some(self.frame());
            }
            let now = js_sys::Date::now();
            if now - start >= budget_ms {
                break;
            }
            // Grow the slice while rows are cheap, staying inside the budget.
            if now - t < budget_ms / 8.0 {
                rows *= 2;
            }
        }
        None
    }
}

/// The running CPU render of one scene state.
struct Job {
    key: String,
    latest: Arc<Mutex<Option<Frame>>>,
    started: f64,
    #[cfg(not(target_arch = "wasm32"))]
    cancel: Arc<AtomicBool>,
    #[cfg(target_arch = "wasm32")]
    work: Work,
}

#[cfg(not(target_arch = "wasm32"))]
impl Drop for Job {
    fn drop(&mut self) {
        self.cancel.store(true, Ordering::Release);
    }
}

/// What the GPU callback reports back to the UI.
#[derive(Default)]
struct GpuStatus {
    samples: u32,
    error: Option<String>,
}

/// "Save render..." on the GPU: the copy of the shown render on its way to the CPU.
#[derive(Default)]
enum Read {
    #[default]
    Idle,
    /// The next callback copies the render.
    Wanted,
    /// Copied in a submitted frame: map it.
    Copied(Readback),
    Mapping(Readback),
}

/// The GPU render shown in the view.
struct GpuView {
    size: (u32, u32),
    /// What [`GpuView::scene`] was built from (everything but the camera, size and settings).
    scene_key: String,
    scene: Arc<TraceScene>,
    /// The restart key of the samples counted in `status`.
    key: String,
    started: f64,
    out: wgpu::Texture,
    out_view: wgpu::TextureView,
    texture: egui::TextureId,
    status: Arc<Mutex<GpuStatus>>,
    /// Samples per pixel per frame: grows while frames are quick, decreases when they are slow.
    spp: u32,
    /// What steers `spp`.
    spp_control: SppControl,
    read: Arc<Mutex<Read>>,
    /// An image of this render was resolved into `texture` at least once, so a restart keeps
    /// showing it (the previous camera's frame) until the new samples replace it.
    has_image: bool,
    /// The egui texture of the last image traced at the smaller moving size, shown instead of
    /// `texture` while the full-size render is too grainy (see [`holds_previous`]).
    held: Option<egui::TextureId>,
    /// When `held` started being shown (egui time, seconds): a restart while it is up (a display
    /// slider dragged) resets `started`, so the hold is timed from this instead.
    held_since: f64,
}

/// While the camera moves, the GPU render is traced at this fraction of the view size per axis
/// (the image is stretched to the view). A quarter of the pixels means about four times the samples
/// per pixel in the same frame time, so an orbit shows a smooth image instead of one or two noisy
/// samples; a faster GPU affords more samples (see `GpuView::spp`) at the same size.
const MOVING_SCALE: f32 = 0.5;

/// Seconds without a camera change after which the render returns to the full view size.
const MOTION_SETTLE_S: f64 = 0.2;

/// While textures load, new tiles are taken into the render at most this often (seconds). Each
/// time restarts the render, so taking them in every frame leaves it forever one or two samples
/// deep (raw noise) until loading ends; this lets about a third of a second of samples build up
/// between restarts, at the price of a little delay before a new tile shows.
const LOADING_RESTART_S: f64 = 0.35;

/// Most samples per frame while textures load (see [`LOADING_RESTART_S`]). The frame time is
/// then the compositor's, not the tracer's, so the count is not steered by it: it stays where it
/// was, at most this (cheap at the smaller loading size).
const LOADING_MAX_SPP: u32 = 8;

/// Whether textures are still loading: tiles wait to be composited, or composited ones to be
/// uploaded (the GPU render copies the uploaded display textures).
fn textures_loading(session: &texturecraft_engine::Session) -> bool {
    session.pending_tiles() > 0 || session.caches.iter().any(|cache| !cache.upload.is_empty())
}

/// Whether changed textures are taken into the render now: always, except while loading, when at
/// most every [`LOADING_RESTART_S`].
fn take_in_textures(loading: bool, now: f64, last_sync: f64) -> bool {
    !loading || now - last_sync >= LOADING_RESTART_S
}

/// When the render returns to full size after the camera stops, the soft image traced while it
/// moved stays on screen until the full-size render has this many samples (it starts from none,
/// and its first few samples are grainy)...
const SETTLE_HOLD_SAMPLES: u32 = 8;

/// ...or for this many seconds at most, so a slow GPU never leaves the soft image up for long.
const SETTLE_HOLD_MAX_S: f64 = 0.5;

/// Whether the image from before the view returned to full size is still shown: the full-size
/// render has `samples` samples after `elapsed` seconds.
fn holds_previous(samples: u32, elapsed: f64) -> bool {
    samples < SETTLE_HOLD_SAMPLES && elapsed < SETTLE_HOLD_MAX_S
}

/// Trace fewer pixels while the camera is moving, keeping tiny views at their original size.
fn traced_size(full: (u32, u32), moving: bool) -> (u32, u32) {
    if !moving {
        return full;
    }

    let scale = |dimension: u32| ((dimension as f32 * MOVING_SCALE) as u32).clamp(16.min(dimension), dimension);
    (scale(full.0), scale(full.1))
}

/// The GPU tracer in egui's callback resources.
struct TraceState {
    tracer: GpuTracer,
    scene: Option<Arc<TraceScene>>,
    key: String,
    /// Samples in the resolved image (None: resolve again).
    resolved: Option<u32>,
    /// When the tracer last took in changed textures (egui time, seconds).
    last_sync: f64,
}

/// Traced geometry and the key of what it was built from.
type Geometry = (String, Arc<Mesh>, Arc<Bvh>);

/// Rendering mode state kept by the app.
#[derive(Default)]
pub struct RenderView {
    job: Option<Job>,
    gpu: Option<GpuView>,
    /// Why the GPU tracer failed on the scene of this key (the CPU renders it instead).
    gpu_error: Option<(String, String)>,
    texture: Option<TextureHandle>,
    motion: Motion,
    shown: Option<(u32, u32, Vec<u8>)>,
    samples: u32,
    elapsed: f64,
    /// The traced geometry (displaced or not) and what it was built from: changes to the display
    /// alone don't rebuild it.
    geometry: Mutex<Option<Geometry>>,
    /// The traced environment and what it was built from (see [`EnvKey`]): building one takes a
    /// while, and the scene is rebuilt on every display change.
    env: Mutex<Option<(EnvKey, Arc<EnvImage>)>>,
}

/// The display's environment name and its retained imported image (None: built-in).
type EnvKey = (String, Option<Arc<ImageRes>>);

/// Viewport resolution of a CPU render (the web renders at half size: it has one thread).
fn render_size(rect: Rect) -> (u32, u32) {
    let scale = if cfg!(target_arch = "wasm32") { 0.5 } else { 1.0 };
    ((rect.width() * scale).max(16.0) as u32, (rect.height() * scale).max(16.0) as u32)
}

/// Build a scene while retaining unchanged geometry and the decoded environment.
fn trace_scene(app: &TextureCraftApp) -> TraceScene {
    let session = &app.session;
    let display = &session.doc.project.display;
    let sets: Vec<TraceSet> = session
        .doc
        .project
        .texture_sets
        .iter()
        .zip(&session.caches)
        .map(|(set, cache)| TraceSet { material: set.material, composite: cache.composite.clone(), height_strength: set.atlas_height_strength() })
        .collect();
    let env_image = display.environment_resource().and_then(|n| session.images.get(n));

    // Only displaced geometry depends on the composites (their heights)
    let revisions: Vec<u64> = if display.displacement { session.caches.iter().map(|cache| cache.revision).collect() } else { Vec::new() };
    let strengths: Vec<f32> = sets.iter().map(|set| set.height_strength).collect();
    let key = format!("{}|{revisions:?}|{strengths:?}|{}", session.mesh_id, display.displacement);
    let mut cached = app.render.geometry.lock().unwrap_or_else(PoisonError::into_inner);
    let (mesh, bvh) = match cached.as_ref() {
        Some((cached_key, mesh, bvh)) if *cached_key == key => (mesh.clone(), bvh.clone()),
        _ => {
            let (mesh, bvh) = trace_geometry(&session.mesh, &session.bvh, &sets, display);
            *cached = Some((key, mesh.clone(), bvh.clone()));
            (mesh, bvh)
        }
    };
    drop(cached);

    let mut env_cached = app.render.env.lock().unwrap_or_else(PoisonError::into_inner);
    let same_image = |source: &Option<Arc<ImageRes>>| match (source, env_image) {
        (Some(cached), Some(current)) => Arc::ptr_eq(cached, current),
        (None, None) => true,
        _ => false,
    };
    let env = match env_cached.as_ref() {
        Some(((name, source), environment)) if *name == display.environment && same_image(source) => environment.clone(),
        _ => {
            let environment = Arc::new(TraceScene::display_env(display, env_image.map(|image| image.as_ref())));

            // Retaining the source prevents a later image from reusing its cached identity
            let env_key = (display.environment.clone(), env_image.cloned());
            *env_cached = Some((env_key, environment.clone()));
            environment
        }
    };

    TraceScene::for_display_env(mesh, bvh, sets, display, env)
}

fn settings(app: &TextureCraftApp) -> TraceSettings {
    TraceSettings { bounces: app.ui.render.bounces }
}

fn display(app: &TextureCraftApp) -> TraceDisplay {
    TraceDisplay { denoise: app.ui.render.denoise, ..TraceDisplay::from_display(&app.session.doc.project.display) }
}

fn work(app: &TextureCraftApp, size: (u32, u32)) -> Work {
    Work {
        scene: trace_scene(app),
        camera: app.session.camera,
        settings: settings(app),
        display: display(app),
        max_samples: app.ui.render.max_samples.max(1),
        tracer: Tracer::new(size.0, size.1),
    }
}

/// What the traced scene depends on. `composites`: the texture-set composites too (their
/// revisions bump with every composited chunk, so while textures load this changes every frame).
fn scene_key(session: &texturecraft_engine::Session, composites: bool) -> String {
    let revisions: Vec<u64> = if composites { session.caches.iter().map(|cache| cache.revision).collect() } else { Vec::new() };
    let sets: Vec<_> = session.doc.project.texture_sets.iter().map(|set| (set.material, set.atlas_height_strength())).collect();

    format!("{}|{revisions:?}|{sets:?}|{:?}", session.mesh_id, session.doc.project.display)
}

/// [`scene_key`] of the GPU render. It takes texture changes in by copying the display textures
/// when their content changes (throttled while textures load, see [`take_in_textures`]), so the
/// composites only count when they displace the geometry. Restarting on every composited chunk
/// would rebuild the scene and reset the render every frame of a load.
fn gpu_scene_key(session: &texturecraft_engine::Session) -> String {
    scene_key(session, session.doc.project.display.displacement)
}

/// What the render of scene `scene_key` depends on; a change restarts it.
fn key(app: &TextureCraftApp, scene_key: &str, size: (u32, u32)) -> String {
    format!("{scene_key}|{:?}|{size:?}|{:?}", app.session.camera, app.ui.render)
}

fn start(app: &TextureCraftApp, ctx: &egui::Context, key: String, size: (u32, u32)) -> Job {
    let latest = Arc::new(Mutex::new(None));
    let started = ctx.input(|i| i.time);
    #[cfg(not(target_arch = "wasm32"))]
    {
        let cancel = Arc::new(AtomicBool::new(false));
        let mut w = work(app, size);
        let (out, stop, ctx) = (latest.clone(), cancel.clone(), ctx.clone());
        let spawned = std::thread::Builder::new().name("texturecraft-render".into()).spawn(move || {
            while !stop.load(Ordering::Acquire) {
                let Some(frame) = w.step() else { break };
                if stop.load(Ordering::Acquire) {
                    break;
                }
                *out.lock().unwrap_or_else(PoisonError::into_inner) = Some(frame);
                ctx.request_repaint();
            }
        });
        if let Err(e) = spawned {
            log::error!("could not start the render thread: {e}");
        }
        Job { key, latest, started, cancel }
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = ctx;
        Job { key, latest, started, work: work(app, size) }
    }
}

/// Draw Rendering mode over the 3D view in `rect` (navigation is handled by the 3D view).
pub fn show(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect) {
    let on_gpu = !app.ui.render.cpu && show_gpu(app, ui, rect);
    if !on_gpu {
        drop_gpu(app);
        show_cpu(app, ui, rect);
    }
    overlay(app, ui, rect, on_gpu);
}

/// Advance the CPU render and keep the last completed image visible during restarts.
fn show_cpu(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect) {
    let ctx = ui.ctx().clone();
    let size = render_size(rect);

    // The CPU tracer reads the composites themselves: every change restarts it
    let render_key = key(app, &scene_key(&app.session, true), size);
    if app.render.job.as_ref().is_none_or(|job| job.key != render_key) {
        app.render.job = Some(start(app, &ctx, render_key, size));
        app.render.samples = 0;
    }

    let now = ctx.input(|i| i.time);
    if let Some(job) = app.render.job.as_mut() {
        #[cfg(target_arch = "wasm32")]
        {
            if let Some(frame) = job.work.step_for(30.0) {
                *job.latest.lock().unwrap_or_else(PoisonError::into_inner) = Some(frame);
            }
            if job.work.tracer.samples < job.work.max_samples {
                ctx.request_repaint();
            }
        }
        if let Some(frame) = job.latest.lock().unwrap_or_else(PoisonError::into_inner).take() {
            let image = ColorImage::from_rgba_unmultiplied([frame.width as usize, frame.height as usize], &frame.rgba);
            match app.render.texture.as_mut() {
                Some(texture) => texture.set(image, TextureOptions::LINEAR),
                None => app.render.texture = Some(ctx.load_texture("render-view", image, TextureOptions::LINEAR)),
            }
            app.render.samples = frame.samples;
            app.render.elapsed = now - job.started;
            app.render.shown = Some((frame.width, frame.height, frame.rgba));
        }
    }

    let shown = app.render.texture.as_ref().map(TextureHandle::id);
    paint(ui, rect, shown);
}

/// Whether the view shows the render image rather than the "Starting render..." placeholder.
/// `has_image`: an earlier frame of this render is on screen. A restart (the camera moved) resets
/// the sample count to 0 but keeps that frame up, so orbiting doesn't flash the placeholder.
/// Only the very first frame, before anything was traced, has nothing to show.
fn show_image(has_image: bool, samples: u32) -> bool {
    has_image || samples > 0
}

/// The render, or a placeholder until its first samples.
fn paint(ui: &Ui, rect: Rect, texture: Option<egui::TextureId>) {
    let tk = Tokens::get(ui.ctx());
    let p = ui.painter_at(rect);
    match texture {
        Some(t) => {
            p.image(t, rect, Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        None => {
            p.rect_filled(rect, 0.0, tk.viewport);
            p.text(rect.center(), egui::Align2::CENTER_CENTER, tr("Starting render..."), egui::FontId::proportional(14.0), tk.viewport_text);
        }
    }
}

/// Trace on the GPU; false when it can't (no compute shaders, or it failed on this scene).
fn show_gpu(app: &mut TextureCraftApp, ui: &Ui, rect: Rect) -> bool {
    let Some(render_state) = app.render_state.clone() else { return false };
    let scene_key = gpu_scene_key(&app.session);
    if !GpuTracer::supported(&render_state.device) || app.render.gpu_error.as_ref().is_some_and(|(k, _)| *k == scene_key) {
        return false;
    }

    app.render.gpu_error = None;
    if app.render.job.take().is_some() {
        app.render.texture = None;
        app.render.shown = None;
    }

    let ctx = ui.ctx().clone();
    let now = ctx.input(|i| i.time);
    let full = (rect.width().max(16.0) as u32, rect.height().max(16.0) as u32);
    let loading = textures_loading(&app.session);
    let moving = app.render.motion.moving(&app.session.camera, now, loading);
    if moving {
        // Come back and restore the full size once the camera has been still, even with no input
        ctx.request_repaint_after(std::time::Duration::from_secs_f64(MOTION_SETTLE_S));
    }

    let size = traced_size(full, moving);
    let key = key(app, &scene_key, size);
    let mut previous = app.render.gpu.take();
    if let Some(view) = previous.as_mut() {
        // Read the completed callback before resizing replaces its status
        view.has_image |= view.status.lock().unwrap_or_else(PoisonError::into_inner).samples > 0;
    }

    let mut view = match previous {
        Some(view) if view.size == size => view,
        old => {
            let out = GpuTracer::output_texture(&render_state.device, size.0, size.1);
            let out_view = out.create_view(&Default::default());

            // A resized view keeps its scene and egui texture id
            let old_has_image = old.as_ref().is_some_and(|previous| previous.has_image);
            let old_spp = old.as_ref().map_or(1, |previous| rescale_spp(previous.spp, previous.size, size));
            let mut held_since = old.as_ref().map_or(now, |previous| previous.held_since);
            let growing = old.as_ref().is_some_and(|previous| u64::from(previous.size.0) * u64::from(previous.size.1) < u64::from(size.0) * u64::from(size.1));
            let mut held = None;
            let (texture, scene_key, scene, spp_control) = match old {
                // Returning to full size: the old image stays up under its own texture id
                Some(old) if growing && old.has_image => {
                    let mut renderer = render_state.renderer.write();
                    if let Some(texture) = old.held {
                        renderer.free_texture(&texture);
                    }

                    held = Some(old.texture);
                    held_since = now;
                    (renderer.register_native_texture(&render_state.device, &out_view, wgpu::FilterMode::Linear), old.scene_key, old.scene, old.spp_control)
                }
                Some(old) => {
                    let mut renderer = render_state.renderer.write();
                    if let Some(texture) = old.held {
                        renderer.free_texture(&texture);
                    }

                    renderer.update_egui_texture_from_wgpu_texture(&render_state.device, &out_view, wgpu::FilterMode::Linear, old.texture);
                    (old.texture, old.scene_key, old.scene, old.spp_control)
                }
                None => (
                    render_state.renderer.write().register_native_texture(&render_state.device, &out_view, wgpu::FilterMode::Linear),
                    scene_key.clone(),
                    Arc::new(trace_scene(app)),
                    SppControl::default(),
                ),
            };

            GpuView {
                size,
                scene_key,
                scene,
                key: String::new(),
                started: now,
                out,
                out_view,
                texture,
                status: Arc::default(),
                spp: old_spp,
                spp_control,
                read: Arc::default(),
                has_image: old_has_image,
                held,
                held_since,
            }
        }
    };

    let (samples, error) = {
        let mut status = view.status.lock().unwrap_or_else(PoisonError::into_inner);
        if view.key != key {
            status.samples = 0;
        }

        (status.samples, status.error.take())
    };

    if let Some(error) = error {
        log::warn!("GPU path tracing failed, rendering on the CPU: {error}");
        app.render.gpu_error = Some((scene_key, error));
        app.render.gpu = Some(view);
        return false;
    }

    if view.key != key {
        view.key = key.clone();
        view.started = now;
    }
    if view.scene_key != scene_key {
        view.scene = Arc::new(trace_scene(app));
        view.scene_key = scene_key;
    }

    let max_samples = app.ui.render.max_samples.max(1);
    if samples < max_samples {
        // Loading time includes compositing, so it cannot reliably steer the tracer's sample count
        view.spp = if loading { view.spp.min(LOADING_MAX_SPP) } else { view.spp_control.next(view.spp, f64::from(ctx.input(|i| i.stable_dt))) };
        app.render.elapsed = now - view.started;
        ctx.request_repaint();
    }

    app.render.samples = samples;
    if poll_read(app, &render_state.device, &view) {
        ctx.request_repaint();
    }

    let callback = TraceCallback {
        key,
        scene: view.scene.clone(),
        camera: app.session.camera,
        settings: settings(app),
        display: display(app),
        max_samples,
        spp: view.spp,
        loading,
        now,
        size,
        out: view.out.clone(),
        out_view: view.out_view.clone(),
        status: view.status.clone(),
        read: view.read.clone(),
    };
    ui.painter().add(egui_wgpu::Callback::new_paint_callback(rect, callback));
    if let Some(held) = view.held {
        let elapsed = now - view.held_since;
        if holds_previous(samples, elapsed) {
            // A render capped below the hold threshold still needs a frame to release the image
            ctx.request_repaint_after(std::time::Duration::from_secs_f64((SETTLE_HOLD_MAX_S - elapsed).max(0.0)));
        } else {
            render_state.renderer.write().free_texture(&held);
            view.held = None;
        }
    }

    paint(ui, rect, show_image(view.has_image, samples).then_some(view.held.unwrap_or(view.texture)));
    app.render.gpu = Some(view);
    true
}

/// Advance "Save render..." on the GPU: map the copy once its frame was submitted, then save.
/// True while it is pending.
fn poll_read(app: &mut TextureCraftApp, device: &wgpu::Device, g: &GpuView) -> bool {
    let mut read = g.read.lock().unwrap_or_else(PoisonError::into_inner);
    let pixels = match std::mem::take(&mut *read) {
        Read::Copied(r) => {
            r.map();
            *read = Read::Mapping(r);
            None
        }
        Read::Mapping(r) => {
            // Runs map callbacks on native backends (the browser runs them on its own).
            let _ = device.poll(wgpu::PollType::Poll);
            match r.take() {
                Some(px) => Some(px),
                None => {
                    *read = Read::Mapping(r);
                    None
                }
            }
        }
        other => {
            *read = other;
            None
        }
    };
    let pending = !matches!(*read, Read::Idle);
    drop(read);
    match pixels {
        Some(Some(px)) => {
            app.render.shown = Some(px);
            save(app);
        }
        Some(None) => app.toast(tr("Couldn't read the render back from the GPU"), true),
        None => {}
    }
    pending
}

/// Adds samples on the GPU and resolves them into the shown texture.
struct TraceCallback {
    key: String,
    scene: Arc<TraceScene>,
    camera: OrbitCamera,
    settings: TraceSettings,
    display: TraceDisplay,
    max_samples: u32,
    spp: u32,
    /// Textures are still loading (see [`LOADING_RESTART_S`]) and the egui time of this frame.
    loading: bool,
    now: f64,
    size: (u32, u32),
    out: wgpu::Texture,
    out_view: wgpu::TextureView,
    status: Arc<Mutex<GpuStatus>>,
    read: Arc<Mutex<Read>>,
}

impl TraceCallback {
    /// Synchronize scene data, add samples and resolve the frame for the UI.
    fn run(
        &self,
        state: &mut TraceState,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
    ) -> Result<(), String> {
        let tracer = &mut state.tracer;
        if !state.scene.as_ref().is_some_and(|scene| Arc::ptr_eq(scene, &self.scene)) {
            tracer.set_scene(device, queue, &self.scene)?;
            state.scene = Some(self.scene.clone());
        }

        let sync_textures = take_in_textures(self.loading, self.now, state.last_sync);
        if sync_textures {
            state.last_sync = self.now;
        }
        if (sync_textures && tracer.sync_materials(device, queue, encoder, renderer)?) || state.key != self.key {
            tracer.reset(device, encoder, self.size.0, self.size.1)?;
            state.key.clone_from(&self.key);
            state.resolved = None;
        }

        let samples = tracer.samples();
        if samples < self.max_samples {
            tracer.pass(device, queue, encoder, &self.camera, &self.settings, self.spp.min(self.max_samples - samples));
        }
        if state.resolved != Some(tracer.samples()) {
            tracer.resolve(device, queue, encoder, &self.display, &self.out_view);
            state.resolved = Some(tracer.samples());
        }

        let mut read = self.read.lock().unwrap_or_else(PoisonError::into_inner);
        if matches!(*read, Read::Wanted) {
            *read = Read::Copied(Readback::new(device, encoder, &self.out));
        }

        Ok(())
    }
}

impl CallbackTrait for TraceCallback {
    /// Submit tracing work and report its samples or failure to the next UI frame.
    fn prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        _screen: &ScreenDescriptor,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let state = match resources.remove::<TraceState>() {
            Some(st) => Ok(st),
            None => GpuTracer::new(device).map(|tracer| TraceState { tracer, scene: None, key: String::new(), resolved: None, last_sync: f64::MIN }),
        };
        let result = state.and_then(|mut st| {
            let r = match resources.get::<crate::gpu::GpuState>() {
                Some(gpu) => self.run(&mut st, device, queue, encoder, &gpu.renderer),
                None => Err("the viewport renderer is missing".into()),
            };
            let samples = st.tracer.samples();
            resources.insert(st);
            r.map(|()| samples)
        });
        let mut status = self.status.lock().unwrap_or_else(PoisonError::into_inner);
        match result {
            Ok(samples) => status.samples = samples,
            Err(e) => status.error = Some(e),
        }
        Vec::new()
    }

    fn paint(&self, _info: egui::PaintCallbackInfo, _pass: &mut wgpu::RenderPass<'static>, _resources: &CallbackResources) {}
}

/// Free the GPU render (its texture and the tracer's scene memory; the pipelines stay).
fn drop_gpu(app: &mut TextureCraftApp) {
    let Some(view) = app.render.gpu.take() else { return };
    if let Some(render_state) = &app.render_state {
        let mut renderer = render_state.renderer.write();
        renderer.free_texture(&view.texture);
        if let Some(held) = view.held {
            renderer.free_texture(&held);
        }

        if let Some(state) = renderer.callback_resources.get_mut::<TraceState>() {
            state.tracer.release();
            (state.scene, state.resolved) = (None, None);
            state.key.clear();

            // The materials went with the release: the next render takes textures in at once
            state.last_sync = f64::MIN;
        }
    }
}

/// The render status and settings card in the top-left corner.
fn overlay(app: &mut TextureCraftApp, ui: &mut Ui, rect: Rect, on_gpu: bool) {
    let tk = Tokens::get(ui.ctx());
    let card = Rect::from_min_size(rect.min + vec2(12.0, 12.0), vec2(250.0, 200.0));
    let mut child = ui.new_child(egui::UiBuilder::new().max_rect(card).layout(egui::Layout::top_down(egui::Align::Min)));
    egui::Frame::popup(child.style()).show(&mut child, |ui| {
        ui.set_width(226.0);
        let max = app.ui.render.max_samples.max(1);
        let running = app.render.samples < max;
        ui.label(RichText::new(tr("RENDERING")).size(11.0).strong());
        let template = match (running, on_gpu) {
            (true, true) => n_("Rendering on GPU: {n} / {max} samples, {s} s"),
            (true, false) => n_("Rendering on CPU: {n} / {max} samples, {s} s"),
            (false, true) => n_("Done on GPU: {n} / {max} samples, {s} s"),
            (false, false) => n_("Done on CPU: {n} / {max} samples, {s} s"),
        };
        let line = ui.label(
            RichText::new(trf(template, &[("n", &app.render.samples), ("max", &max), ("s", &format!("{:.1}", app.render.elapsed))]))
                .size(11.0)
                .color(tk.text_dim),
        );
        if let Some((_, e)) = app.render.gpu_error.as_ref().filter(|_| !on_gpu && !app.ui.render.cpu) {
            line.on_hover_text(trf("The GPU couldn't render this scene: {error}", &[("error", e)]));
        }
        ui.add(egui::ProgressBar::new(app.render.samples as f32 / max as f32).desired_height(4.0));
        ui.horizontal(|ui| {
            ui.label(tr("Samples"));
            ui.add(egui::DragValue::new(&mut app.ui.render.max_samples).range(1..=16384));
            ui.label(tr("Bounces"));
            ui.add(egui::DragValue::new(&mut app.ui.render.bounces).range(1..=32));
        });
        ui.checkbox(&mut app.ui.render.denoise, tr("Denoise")).on_hover_text(tr("Filter the remaining noise while keeping edges and texture detail"));
        if app.render_state.as_ref().is_some_and(|rs| GpuTracer::supported(&rs.device)) {
            ui.checkbox(&mut app.ui.render.cpu, tr("Render on the CPU")).on_hover_text(tr("Path trace on the processor instead of the graphics card (slower)"));
        }
        ui.horizontal(|ui| {
            let saving = app.render.gpu.as_ref().is_some_and(|g| !matches!(*g.read.lock().unwrap_or_else(PoisonError::into_inner), Read::Idle));
            let ready = if on_gpu { app.render.samples > 0 && !saving } else { app.render.shown.is_some() };
            if ui.add_enabled(ready, egui::Button::new(tr("Save render..."))).clicked() {
                match app.render.gpu.as_ref().filter(|_| on_gpu) {
                    Some(g) => *g.read.lock().unwrap_or_else(PoisonError::into_inner) = Read::Wanted,
                    None => save(app),
                }
            }
            if ui.button(tr("Back to painting")).clicked() {
                app.ui.mode = crate::state::Mode::Painting;
            }
        });
    });
}

/// Save the shown render as PNG.
fn save(app: &mut TextureCraftApp) {
    let Some((w, h, rgba)) = app.render.shown.clone() else { return };
    let Some(png) = crate::control::encode_png(w, h, rgba).map(|(_, _, png)| png) else {
        app.toast(tr("Couldn't encode the render"), true);
        return;
    };
    let name = format!("{}_render.png", texturecraft_engine::formats::export::sanitize_file_stem(&app.session.doc.project.name));
    #[cfg(not(target_arch = "wasm32"))]
    {
        use crate::file_dialogs::{Bound, FileDialog};
        app.ask_file(FileDialog::save().filter(tr("PNG image"), &["png"]).file_name(name), Bound::App, move |app, paths| {
            let Some(path) = paths.first() else { return };
            match std::fs::write(path, &png) {
                Ok(()) => app.toast(trf("Saved {path}", &[("path", &path.display())]), false),
                Err(e) => app.toast(trf("Couldn't save the render: {error}", &[("error", &e)]), true),
            }
        });
    }
    #[cfg(target_arch = "wasm32")]
    match crate::web_files::download(&name, &png) {
        Ok(()) => app.toast(trf("Downloaded {name}", &[("name", &name)]), false),
        Err(e) => app.toast(trf("Couldn't save the render: {error}", &[("error", &e)]), true),
    }
}

impl RenderView {
    pub fn is_active(&self) -> bool {
        self.job.is_some() || self.gpu.is_some() || self.texture.is_some()
    }
}

/// Stop rendering (leaving Rendering mode frees the worker, the GPU memory and the image).
pub fn stop(app: &mut TextureCraftApp) {
    drop_gpu(app);
    app.render = RenderView::default();
}
