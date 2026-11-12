//! A read-only model companion, sharing the ordinary viewport's uploaded world.
use std::sync::{Arc, Mutex, PoisonError};
use std::time::{Duration, Instant};

use craft_stereo_window::{Picture, StereoWindow};
use eframe::egui_wgpu::{CallbackResources, CallbackTrait};
use texturecraft_engine::core::MAX_VIEW_SIDE;
use texturecraft_render::{Frame, ViewKind, readback::Readback, stereo::Budget, wgpu};

use crate::TextureCraftApp;
use crate::gpu::{GpuState, SceneSyncSlot};
use crate::i18n::{tr, trf};

const EYES: [u64; 2] = [u64::MAX, u64::MAX - 1];
type Pixels = (u32, u32, Vec<u8>);

#[derive(Clone, PartialEq)]
struct Request {
    frame: Frame,
    budget: Budget,
    mesh_id: u64,
}

struct Pending {
    reads: [Readback; 2],
    mapped: bool,
    pixels: [Option<Pixels>; 2],
}

#[derive(Default)]
struct Mailbox {
    request: Option<Request>,
    encoded: Option<(Request, u64)>,
    pending: Option<Pending>,
    error: Option<String>,
}

#[derive(Default)]
pub struct ModelView {
    pub options_open: bool,
    window: Option<StereoWindow>,
    mailbox: Arc<Mutex<Mailbox>>,
    opened: Option<Instant>,
    warned: bool,
}

impl ModelView {
    pub fn tick(&mut self, app: &mut TextureCraftApp, ctx: &egui::Context) {
        if let Some(gpu) = &app.gpu {
            let reports = gpu.take_reports();
            crate::viewport::gpu_report_toasts(app, reports);
        }
        let result = self.update(app, ctx);
        if let Err(error) = result {
            self.close(app);
            app.run("viewport.stereo", serde_json::json!({"enabled":false}));
            app.toast(trf("Could not show the stereo view: {error}", &[("error", &error)]), true);
        }
        self.settings(app, ctx);
    }

    fn close(&mut self, app: &TextureCraftApp) {
        if self.window.is_some()
            && let Some(rs) = &app.render_state
            && let Some(gpu) = rs.renderer.write().callback_resources.get_mut::<GpuState>()
        {
            for id in EYES {
                gpu.renderer.drop_view(id);
            }
        }
        self.window = None;
        *self.mailbox.lock().unwrap_or_else(PoisonError::into_inner) = Mailbox::default();
        self.opened = None;
        self.warned = false;
    }

    fn update(&mut self, app: &mut TextureCraftApp, ctx: &egui::Context) -> Result<(), String> {
        if !app.session.stereo.enabled {
            self.close(app);
            return Ok(());
        }
        if ctx.cumulative_frame_nr() < 2 {
            ctx.request_repaint();
            return Ok(());
        }
        let size = render_size(app.session.stereo.resolution, app.viewport_rect, app.session.viewport.to_array(), ctx.pixels_per_point())?;
        if self.window.is_none() {
            self.window = Some(StereoWindow::open(tr("Stereo View"), "ai.storyteller.texturecraft.StereoView").map_err(|e| e.to_string())?);
            self.opened = Some(Instant::now());
        }
        let Some(window) = &self.window else { return Ok(()) };
        if !window.is_open() {
            self.close(app);
            app.run("viewport.stereo", serde_json::json!({"enabled":false}));
            return Ok(());
        }
        let Some(rs) = &app.render_state else { return Err(tr("No GPU renderer available").into()) };
        let mut mailbox = self.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(error) = mailbox.error.take() {
            return Err(error);
        }
        if let Some(pending) = &mut mailbox.pending {
            // The previous UI frame has submitted both copies. Never map a copy in prepare.
            if !pending.mapped {
                for read in &pending.reads {
                    read.map();
                }
                pending.mapped = true;
            }
            rs.device.poll(wgpu::PollType::Poll).map_err(|e| e.to_string())?;
            for (read, pixels) in pending.reads.iter().zip(&mut pending.pixels) {
                if pixels.is_none()
                    && let Some(ready) = read.take()
                {
                    *pixels = Some(ready.ok_or_else(|| tr("Could not read the stereo views from the GPU.").to_owned())?);
                }
            }
            if let [Some(left), Some(right)] = &pending.pixels {
                if left.0 != right.0 || left.1 != right.1 {
                    return Err(tr("The stereo views have different native dimensions.").into());
                }
                if !window.show(pack([left.0, left.1], &left.2, &right.2)?) {
                    return Err(tr("The stereo window closed before the pair was shown.").into());
                }
                mailbox.pending = None;
            }
        }
        let limits = window.depth_limits().map(|l| l.native_pixels(size[0]));
        let (sunk, popped) = limits.unwrap_or((0.0, 0.0));
        let bake = crate::viewport::bake_visualization(app);
        let mut shade = crate::viewport::shade(app);
        shade.bake = bake;
        mailbox.request = Some(Request {
            frame: Frame {
                view_id: crate::viewport::VIEW_3D,
                size,
                kind: ViewKind::Scene,
                camera: app.session.camera,
                shade,
                visible: app.session.doc.project.texture_sets.iter().map(|t| t.visible).collect(),
                overlay: Vec::new(),
            },
            budget: Budget { sunk: sunk as f32, popped: popped as f32 },
            mesh_id: app.session.mesh_id,
        });
        drop(mailbox);
        let aux = match app.session.view {
            _ if app.ui.mode == crate::state::Mode::Painting && app.session.quick_mask.editing && app.picking_id_color.is_none() => {
                Some(crate::gpu::AuxView::QuickMask)
            }
            texturecraft_engine::ViewMode::MeshMap(m) => Some(crate::gpu::AuxView::MeshMap(m)),
            texturecraft_engine::ViewMode::Mask(id) => Some(crate::gpu::AuxView::Mask(id)),
            _ => None,
        };
        let Some(gpu) = &mut app.gpu else { return Err(tr("No GPU renderer available").into()) };
        let frame_nr = ctx.cumulative_frame_nr();
        let sync = match gpu.current_sync(frame_nr) {
            Some(sync) => sync,
            None => {
                let sync = gpu.sync(&mut app.session, aux, bake, u32::from(app.ui.prefs.viewport_texture.min(2)), f64::from(app.profiler.logic_ms));
                gpu.remember_sync(frame_nr, sync)
            }
        };
        let rect = egui::Rect::from_min_size(ctx.content_rect().min, egui::vec2(1.0, 1.0));
        ctx.layer_painter(egui::LayerId::background())
            .add(eframe::egui_wgpu::Callback::new_paint_callback(rect, NativeCallback { sync, mailbox: self.mailbox.clone(), ctx: ctx.clone() }));
        if limits.is_none() && !self.warned && self.opened.is_some_and(|at| at.elapsed() >= Duration::from_secs(3)) {
            self.warned = true;
            app.toast(tr("Stereo depth is unavailable on this desktop; showing the flat model."), false);
        }
        if limits.is_some() {
            self.warned = false;
        }
        ctx.request_repaint_after(Duration::from_millis(16));
        Ok(())
    }

    fn settings(&mut self, app: &mut TextureCraftApp, ctx: &egui::Context) {
        if !self.options_open {
            return;
        }
        let mut open = true;
        egui::Window::new(tr("Render resolution per eye")).open(&mut open).resizable(false).show(ctx, |ui| {
            ui.label(tr("The model viewport size preserves its current sampling; the desktop scales once."));
            let Ok(mut size) = render_size(app.session.stereo.resolution, app.viewport_rect, app.session.viewport.to_array(), ctx.pixels_per_point()) else {
                return;
            };
            let changed = ui
                .horizontal(|ui| {
                    ui.label(tr("Width"));
                    let w = ui.add(egui::DragValue::new(&mut size[0]).range(1..=MAX_VIEW_SIDE)).changed();
                    ui.label(tr("Height"));
                    let h = ui.add(egui::DragValue::new(&mut size[1]).range(1..=MAX_VIEW_SIDE)).changed();
                    w || h
                })
                .inner;
            if changed {
                app.run("viewport.stereo", serde_json::json!({"resolution":size}));
            }
            if ui.button(tr("Use model viewport size")).clicked() {
                app.run("viewport.stereo", serde_json::json!({"resolution":null}));
            }
        });
        self.options_open = open;
    }
}

struct NativeCallback {
    sync: Arc<SceneSyncSlot>,
    mailbox: Arc<Mutex<Mailbox>>,
    ctx: egui::Context,
}

impl CallbackTrait for NativeCallback {
    fn finish_prepare(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        resources: &mut CallbackResources,
    ) -> Vec<wgpu::CommandBuffer> {
        let Some(gpu) = resources.get_mut::<GpuState>() else { return Vec::new() };
        if let Some(sync) = self.sync.take() {
            gpu.apply_sync(device, queue, sync);
        }
        let mut mailbox = self.mailbox.lock().unwrap_or_else(PoisonError::into_inner);
        let Some(request) = mailbox.request.clone() else {
            for id in EYES {
                gpu.renderer.drop_view(id);
            }
            return Vec::new();
        };
        if mailbox.pending.is_some() || gpu.renderer.mesh_id() != Some(request.mesh_id) {
            return Vec::new();
        }
        let key = (request.clone(), gpu.renderer.content_rev());
        if mailbox.encoded.as_ref() == Some(&key) && !EYES.into_iter().any(|id| gpu.renderer.accumulating(id)) && !gpu.renderer.awaiting_gpu() {
            return Vec::new();
        }
        let result = gpu.renderer.prepare_stereo(device, queue, encoder, &request.frame, EYES, request.budget).and_then(|_| {
            let left = gpu.renderer.copy_view(device, encoder, EYES[0]);
            let right = gpu.renderer.copy_view(device, encoder, EYES[1]);
            match (left, right) {
                (Some(left), Some(right)) => Ok(Pending { reads: [left, right], mapped: false, pixels: [None, None] }),
                _ => Err(tr("Could not read the stereo views from the GPU.").into()),
            }
        });
        match result {
            Ok(pending) => {
                mailbox.pending = Some(pending);
                mailbox.encoded = Some(key);
            }
            Err(error) => mailbox.error = Some(error),
        }
        gpu.report_pending();
        self.ctx.request_repaint();
        Vec::new()
    }

    fn paint(&self, _: egui::PaintCallbackInfo, _: &mut wgpu::RenderPass<'static>, _: &CallbackResources) {}
}

/// Choose native samples from the rounded viewport grid; an explicit choice stays fixed.
pub fn render_size(requested: Option<[u32; 2]>, rect: egui::Rect, fallback: [f32; 2], scale: f32) -> Result<[u32; 2], String> {
    if !scale.is_finite() || scale <= 0.0 {
        return Err(tr("The stereo view has invalid native dimensions.").into());
    }
    let size = requested.unwrap_or_else(|| {
        let edge = |a: f32, b: f32| ((b * scale).round() - (a * scale).round()).max(1.0) as u32;
        if rect.is_finite() && rect.is_positive() {
            [edge(rect.left(), rect.right()), edge(rect.top(), rect.bottom())]
        } else {
            fallback.map(|s| (s * scale).round().max(1.0) as u32)
        }
    });
    if size.into_iter().any(|s| !(1..=MAX_VIEW_SIDE).contains(&s)) {
        return Err(tr("The stereo view has invalid native dimensions.").into());
    }
    Ok(size)
}

/// Copy whole native rows left first. Renderer output is already opaque RGBA8.
pub fn pack(size: [u32; 2], left: &[u8], right: &[u8]) -> Result<Picture, String> {
    let [w, h] = size;
    let bytes = u64::from(w) * u64::from(h) * 4;
    if w == 0 || h == 0 || w > MAX_VIEW_SIDE || h > MAX_VIEW_SIDE || bytes != left.len() as u64 || bytes != right.len() as u64 {
        return Err(tr("The stereo views have different native dimensions.").into());
    }
    let mut rgba = Vec::with_capacity(left.len() * 2);
    for (l, r) in left.chunks_exact(w as usize * 4).zip(right.chunks_exact(w as usize * 4)) {
        rgba.extend_from_slice(l);
        rgba.extend_from_slice(r);
    }
    Picture::new(w as usize * 2, h as usize, rgba.into(), true).map_err(|e| e.to_string())
}

pub fn display_settings() -> Result<(), String> {
    // The child is reaped off the UI thread; startup errors are shown by the caller.
    let mut child =
        std::process::Command::new("kcmshell6").arg("kcm_kscreen").spawn().map_err(|e| trf("Could not open Display Settings: {error}", &[("error", &e)]))?;
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}
