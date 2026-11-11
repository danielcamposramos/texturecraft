//! # texturecraft-ui-egui
//!
//! The TextureCraft desktop and web shell: a dense dark professional layout (menu bar,
//! tool-settings bar, tool shelf, asset shelf, 3D + 2D viewports, texture set list, layer stack,
//! properties, display/shader/history windows, bake mode). It is a thin layer: every action goes
//! through `texturecraft-engine` commands, so the CLI, control channel and MCP can do the same.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod bake_job;
pub mod bake_mode;
mod bake_scene;
mod color_selection_editor;
pub mod control;
mod decal_drop;
pub mod dock;
pub mod docking;
pub mod export_templates;
#[cfg(not(target_arch = "wasm32"))]
pub mod file_dialogs;
mod filter_slot;
pub mod fonts;
pub mod gpu;
pub mod i18n;
pub mod id_color_pick;
pub mod keymap;
pub mod layer_panel;
mod levels_editor;
mod levels_handle;
mod levels_properties;
mod mask_preview_image;
mod material_preview_image;
pub mod menus;
pub mod new_project_advanced;
mod orb_shape;
pub mod parity;
mod path_hit;
mod path_surface_cache;
mod preview_image;
mod preview_request;
mod preview_worker;
pub mod profile;
pub mod properties_popup;
mod rail_popup;
pub mod render_mode;
mod resource_picker;
mod resource_picker_item;
mod running_bake;
pub mod shelf;
pub mod shelf_preview;
mod soft_box;
pub mod state;
pub mod symmetry;
pub mod theme;
pub mod thumbs;
pub mod titlebar;
pub mod toolbar;
pub mod ui_state_store;
#[cfg(not(target_arch = "wasm32"))]
pub(crate) mod updates_ui;
pub mod uv_paint_point;
pub mod viewport;
pub mod web_files;
pub mod widgets;
pub mod windows;
mod worker_preview;

use std::sync::Arc;
use std::sync::atomic::Ordering;

use serde_json::Value;
use texturecraft_engine::Session;
use texturecraft_engine::bake::Progress;

pub use bake_job::BakeJob;
pub use control::ControlRequest;
pub use running_bake::RunningBake;
pub use state::UiState;

/// The app as the updater knows it: its releases, and the icon and color of its update prompt.
#[cfg(not(target_arch = "wasm32"))]
pub const UPDATE_APP: craft_update::App = craft_update::App {
    icon_png: Some(include_bytes!("../../../assets/app-icon/hicolor/128x128/apps/ai.storyteller.texturecraft.png")),
    accent: Some([0xc2, 0x67, 0x2b]),
    ..craft_update::App::new("texturecraft", "TextureCraft", "3D texture painting", "storytold/texturecraft")
};

/// A texture set resize running on a worker thread (`Session::resize_task`).
pub struct ResizeJob {
    pub progress: Arc<Progress>,
    pub rx: std::sync::mpsc::Receiver<texturecraft_engine::Result<texturecraft_engine::ResizeOutput>>,
}

/// A MaterialCraft material re-cooking on a worker thread (one per layer at a time).
pub struct LinkCook {
    pub layer: texturecraft_engine::core::LayerId,
    /// The parameter values it cooks: a result whose values are no longer the layer's is redone.
    pub values: serde_json::Map<String, Value>,
    pub rx: std::sync::mpsc::Receiver<texturecraft_engine::Result<Vec<u8>>>,
}

/// The engine's refusal to paint while a bake shows (translated where it is toasted).
pub const PAINTING_PAUSED: &str = i18n::n_("Painting is paused while a bake is shown. Cancel the bake or wait for it to finish.");

/// The browser build's notice when WebGPU is unavailable and wgpu fell back to WebGL2.
pub const WEBGL_NOTICE: &str = i18n::n_(
    "WebGPU isn't available in this browser, so TextureCraft runs on WebGL2: Rendering mode path traces on the CPU (slower) and large texture sets show at reduced resolution. A current Chrome, Edge or Safari has WebGPU.",
);

pub struct Toast {
    pub text: String,
    pub error: bool,
    pub until: f64,
}

pub struct TextureCraftApp {
    pub session: Session,
    /// A smooth move to a saved camera in progress.
    camera_glide: Option<CameraGlide>,
    pub ui: UiState,
    /// Where `ui` is saved between runs
    ui_store: ui_state_store::UiStateStore,
    pub gpu: Option<gpu::GpuLink>,
    pub control: Option<std::sync::mpsc::Receiver<ControlRequest>>,
    pub toasts: Vec<Toast>,
    /// Texture sets the GPU could not hold, and the text of the one sticky toast naming them.
    pub(crate) gpu_memory: (Vec<String>, Option<String>),
    pub bake: Option<BakeJob>,
    pub resize: Option<ResizeJob>,
    pub link_cooks: Vec<LinkCook>,
    pub thumbs: thumbs::Thumbs,
    /// The Layers panel's thumbnails and folded groups.
    pub layer_panel: layer_panel::LayerPanel,
    /// Last frame's viewport rect (logical points), for scripted input.
    pub viewport_rect: egui::Rect,
    pub fps: f32,
    /// User smart materials found in the library folder (refreshed on demand).
    pub library: Option<Vec<std::path::PathBuf>>,
    /// User brush presets found in the brushes folder (refreshed on demand).
    pub brushes: Option<Vec<std::path::PathBuf>>,
    /// Rendering mode's path tracer.
    pub render: render_mode::RenderView,
    /// Smoothed CPU time spent building the UI per frame (ms).
    pub ui_ms: f32,
    /// Frame timing and the scripted paint test (`debug.*` control methods).
    pub profiler: profile::Profiler,
    /// Where the last 3D stroke ended (viewport pixels), for Shift+click straight lines.
    pub last_paint_point: Option<texturecraft_engine::geom::glam::Vec2>,
    /// The last finished 2D stroke's endpoint, scoped to its mesh and texture set for Shift+click lines
    pub last_paint_uv: Option<uv_paint_point::UvPaintPoint>,
    /// The decal layer the next click in the 3D view places.
    pub placing_decal: Option<texturecraft_engine::core::LayerId>,
    /// The Properties popup a right click in a view opens.
    pub properties_popup: Option<properties_popup::PropertiesPopup>,
    /// The color selection the next click in the 3D or 2D view picks an ID color for
    pub picking_id_color: Option<id_color_pick::IdColorPick>,
    /// The wgpu state (screenshots read the renderer's targets).
    pub render_state: Option<eframe::egui_wgpu::RenderState>,
    /// CJK fallback fonts for the UI language.
    pub fonts: fonts::UiFonts,
    /// A notice across the top of the window until dismissed (English key, shown translated).
    pub banner: Option<&'static str>,
    /// Who draws the window's title bar (decided at startup, before the window opens).
    pub title_bar: titlebar::TitleBar,
    last_time: f64,
    /// When the next autosave is due (seconds of app time; desktop only).
    #[cfg_attr(target_arch = "wasm32", allow(dead_code))]
    next_autosave: f64,
    /// The brightness and accent the style was last built for.
    styled: Option<(theme::Brightness, theme::Accent)>,
    /// Whether the window had focus last frame.
    focused: bool,
    /// The update check and prompt (desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) updates_rt: updates_ui::Runtime,
    /// Native file dialogs and their continuations (desktop only).
    #[cfg(not(target_arch = "wasm32"))]
    pub file_dialogs: file_dialogs::FileDialogs,
    /// The unsaved-changes dialog's Save is waiting on the Save dialog's answer.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) saving_for_prompt: bool,
    /// That Save dialog saved the project; the prompt goes on at its next frame.
    #[cfg(not(target_arch = "wasm32"))]
    pub(crate) prompt_saved: bool,
    /// The resolved keymap and its key dispatcher.
    pub(crate) bindings: keymap::Bindings,
}

impl TextureCraftApp {
    /// The app around `session`, with the saved UI state (or defaults in the OS language) and the GPU viewport.
    pub fn new(cc: &eframe::CreationContext<'_>, session: Session) -> TextureCraftApp {
        theme::install(&cc.egui_ctx);

        // Bakes cast their rays on a GPU device of their own (created on the first bake)
        #[cfg(not(target_arch = "wasm32"))]
        texturecraft_engine::bake::install_gpu_tracer(Arc::new(texturecraft_render::gpu_bake::GpuBaker::new()));
        let gpu = cc.wgpu_render_state.as_ref().map(|render_state| gpu::init(render_state, &cc.egui_ctx));

        // The first run speaks the OS (or browser) language
        let mut ui_store = ui_state_store::UiStateStore::new(ui_state_store::FILE.get().cloned());
        let ui = ui_store.load(cc.storage).unwrap_or_else(|| {
            let mut ui = UiState::default();
            ui.prefs.language = i18n::os_language().into();
            ui
        });
        let next_autosave = ui.prefs.autosave_minutes as f64 * 60.0;
        ui.prefs.apply_resource_limits();

        let mut app = TextureCraftApp {
            camera_glide: None,
            session,
            ui,
            ui_store,
            gpu,
            control: None,
            toasts: Vec::new(),
            gpu_memory: (Vec::new(), None),
            bake: None,
            resize: None,
            link_cooks: Vec::new(),
            thumbs: thumbs::Thumbs::default(),
            layer_panel: layer_panel::LayerPanel::default(),
            viewport_rect: egui::Rect::NOTHING,
            fps: 0.0,
            ui_ms: 0.0,
            profiler: profile::Profiler::default(),
            library: None,
            brushes: None,
            render: render_mode::RenderView::default(),
            last_paint_point: None,
            last_paint_uv: None,
            placing_decal: None,
            properties_popup: None,
            picking_id_color: None,
            render_state: cc.wgpu_render_state.clone(),
            fonts: fonts::UiFonts::default(),
            banner: None,
            title_bar: titlebar::TitleBar::default(),
            last_time: 0.0,
            next_autosave,
            styled: None,
            focused: true,
            #[cfg(not(target_arch = "wasm32"))]
            updates_rt: updates_ui::Runtime::default(),
            #[cfg(not(target_arch = "wasm32"))]
            file_dialogs: file_dialogs::FileDialogs::new(cc.egui_ctx.clone()),
            #[cfg(not(target_arch = "wasm32"))]
            saving_for_prompt: false,
            #[cfg(not(target_arch = "wasm32"))]
            prompt_saved: false,
            bindings: keymap::Bindings::default(),
        };
        app.ui.rename = None;
        app.ui.windows.new_project = false;
        i18n::set_language(&app.ui.prefs.language);
        app.report_ui_storage_warning();
        export_templates::load_library(&mut app, cc.storage);

        #[cfg(not(target_arch = "wasm32"))]
        {
            // The first window of the process fixes it; a new folder applies on restart
            let _ = DATA_FOLDER.set(data_folder(&app.ui.prefs.data_folder));

            // An export folder still at the default follows the data folder
            if app.ui.export.dir == data_folder("").join("export").display().to_string() {
                app.ui.export.dir = user_dir("export").display().to_string();
            }

            let skipped = app.session.load_templates(user_dir("templates"));
            app.session.log.extend(skipped.into_iter().map(|warning| format!("Project template skipped: {warning}")));
        }

        app
    }

    /// Run an engine command, reporting errors as toasts. Returns the result on success.
    pub fn run(&mut self, id: &str, params: Value) -> Option<Value> {
        self.ui.prefs.apply_resource_limits();

        // Going to a saved camera glides there; the control channel and CLI jump straight away
        let glide_from = (id.starts_with("camera.select") || id.starts_with("camera.bookmark")).then_some(self.session.camera);
        match texturecraft_engine::execute(&mut self.session, id, &params) {
            Ok(v) => {
                if let Some(from) = glide_from.filter(|f| *f != self.session.camera) {
                    self.camera_glide = Some(CameraGlide { from, to: self.session.camera, start: self.last_time });
                    self.session.camera = from;
                }
                Some(v)
            }
            Err(e) => {
                self.toast(e.to_string(), true);
                None
            }
        }
    }

    fn report_ui_storage_warning(&mut self) {
        if let Some(warning) = self.ui_store.take_warning() {
            self.session.log.push(warning.clone());
            self.toast_sticky(warning);
        }
    }

    pub fn toast(&mut self, text: impl Into<String>, error: bool) {
        self.toasts.push(Toast { text: text.into(), error, until: self.last_time + if error { 6.0 } else { 3.0 } });
    }

    /// An error toast that stays until the user dismisses it (for a problem that persists and
    /// is reported once, like a texture set the GPU could not hold).
    pub fn toast_sticky(&mut self, text: impl Into<String>) {
        self.toasts.push(Toast { text: text.into(), error: true, until: f64::INFINITY });
    }

    /// Start baking texture sets in the background, one after another. Invalid set indices
    /// show an error without starting a preview or queuing any work.
    pub fn start_bake(&mut self, sets: Vec<usize>) {
        if self.bake.is_some() || sets.is_empty() {
            return;
        }

        // Restore the picker's view before the preview starts following baked maps
        id_color_pick::IdColorPick::end(self);

        if let Err(error) = self.session.bake_preview_begin_for(&sets) {
            self.toast(error.to_string(), true);
            return;
        }

        let total = sets.len();
        self.bake = Some(BakeJob { queue: sets, current: None, total });
    }

    /// Re-cook MaterialCraft material layer `layer` in the background. While one cook of the layer
    /// runs no other starts: when it ends with stale values, it is redone with the current ones.
    pub fn cook_link(&mut self, layer: texturecraft_engine::core::LayerId) {
        if self.link_cooks.iter().any(|c| c.layer == layer) {
            return;
        }
        match self.session.link_job(layer, &serde_json::Map::new()) {
            Ok(job) => {
                let values = job.values.clone();
                let rx = spawn_worker("texturecraft-link", move || job.run());
                self.link_cooks.push(LinkCook { layer, values, rx });
            }
            Err(e) => self.toast(e.to_string(), true),
        }
    }

    fn poll_link_cooks(&mut self, ctx: &egui::Context) {
        if self.link_cooks.is_empty() {
            return;
        }
        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        let mut done = Vec::new();
        self.link_cooks.retain(|c| match c.rx.try_recv() {
            Err(std::sync::mpsc::TryRecvError::Empty) => true,
            Ok(r) => {
                done.push((c.layer, c.values.clone(), r));
                false
            }
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                done.push((c.layer, c.values.clone(), Err(texturecraft_engine::EngineError::Other("the MaterialCraft cook stopped".into()))));
                false
            }
        });
        for (layer, values, result) in done {
            let current = self.session.link_values(layer);
            match (result, current) {
                // The layer is gone (undone, deleted): nothing to show.
                (_, Err(_)) => {}
                (Ok(_), Ok(now)) if now != values => self.cook_link(layer),
                (Ok(bytes), Ok(_)) => {
                    if let Err(e) = self.session.apply_link_cook(layer, &bytes, true) {
                        self.toast(e.to_string(), true);
                    }
                }
                (Err(e), Ok(_)) => self.toast(e.to_string(), true),
            }
        }
    }

    /// Show at most one streamed map per frame, then apply the set as one undo step after
    /// its map channel closes. Closing the preview alone does not stop the background bake.
    fn poll_bake(&mut self, ctx: &egui::Context) {
        // Leaving Bake mode puts the view mode back; the bake goes on
        if self.session.bake_preview_active() && self.ui.mode != state::Mode::Baking {
            self.session.bake_preview_end();
        }

        let Some(job) = self.bake.as_mut() else {
            // A preview whose bake is gone must not keep the view
            self.session.bake_preview_end();
            return;
        };

        ctx.request_repaint_after(std::time::Duration::from_millis(50));
        if let Some(running) = &job.current {
            let set_index = running.set_index;

            // Show one map per poll so each reaches its own frame
            // An empty channel can gain a map before the result arrives: wait for the sender to close
            match running.maps.try_recv() {
                Ok(map) => {
                    self.session.bake_preview_map(set_index, map);
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {}
            }

            match running.receiver.try_recv() {
                Ok(Ok(output)) => {
                    job.current = None;
                    if let Err(error) = self.session.apply_streamed_bake(set_index, output) {
                        self.toast(error.to_string(), true);
                    }
                }
                Ok(Err(error)) => {
                    self.cancel_bake();
                    self.toast(error.to_string(), true);
                    return;
                }
                Err(std::sync::mpsc::TryRecvError::Empty) => return,
                Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                    self.cancel_bake();
                    self.toast(i18n::tr("The bake worker stopped"), true);
                    return;
                }
            }
        }

        let Some(job) = self.bake.as_mut() else { return };

        if job.queue.is_empty() {
            self.bake = None;
            self.session.bake_preview_end();
            self.toast(i18n::tr("Mesh maps baked"), false);
            return;
        }

        let set_index = job.queue.remove(0);
        self.session.bake_preview_set_current(set_index);
        match self.session.bake_task(set_index) {
            Ok(task) => {
                let progress = Arc::new(Progress::new());
                let worker_progress = progress.clone();
                let maps_total = task.maps().len();
                let maps_done = Arc::new(std::sync::atomic::AtomicUsize::new(0));
                let worker_done = maps_done.clone();
                let (sender, maps) = std::sync::mpsc::channel();
                let receiver = spawn_worker("texturecraft-bake", move || {
                    task.run_streaming(&worker_progress, &mut |map| {
                        worker_done.fetch_add(1, Ordering::Relaxed);
                        let _ = sender.send(map);
                    })
                });

                if let Some(job) = self.bake.as_mut() {
                    job.current = Some(RunningBake { set_index, progress, receiver, maps, maps_done, maps_total });
                }
            }
            Err(error) => self.toast(error.to_string(), true),
        }
    }

    /// Cancel the active worker, discard queued sets and restore the view. The current set's
    /// preview maps are discarded; previously applied sets keep their results and undo steps.
    pub fn cancel_bake(&mut self) {
        let Some(job) = self.bake.take() else { return };

        if let Some(running) = job.current {
            running.progress.cancel.store(true, Ordering::Relaxed);
        }

        // Only the current set is uncommitted; completed sets remain applied
        self.session.bake_preview_end();
    }

    /// Resize texture set `ts` to `resolution` on a worker thread (the document changes when
    /// it is done, as one undo step).
    pub fn start_resize(&mut self, ts: usize, resolution: u32) {
        if self.resize.is_some() {
            return;
        }
        match self.session.resize_task(ts, resolution) {
            Ok(task) => {
                let progress = Arc::new(Progress::new());
                let p = progress.clone();
                self.resize = Some(ResizeJob { progress, rx: spawn_worker("texturecraft-resize", move || task.run(&p)) });
            }
            Err(e) => self.toast(e.to_string(), true),
        }
    }

    /// Apply a finished resize (after any stroke in progress ends).
    fn poll_resize(&mut self, ctx: &egui::Context) {
        let Some(job) = &self.resize else { return };
        ctx.request_repaint_after(std::time::Duration::from_millis(30));
        if self.session.stroke_active() {
            return;
        }
        let result = match job.rx.try_recv() {
            Ok(r) => r,
            Err(std::sync::mpsc::TryRecvError::Empty) => return,
            Err(std::sync::mpsc::TryRecvError::Disconnected) => Err(texturecraft_engine::EngineError::Other(i18n::tr("The resize worker stopped").into())),
        };
        self.resize = None;
        if let Err(e) = result.and_then(|out| self.session.apply_resize(out)) {
            self.toast(e.to_string(), true);
        }
    }

    /// The map being baked and its position among the queued maps and texture sets.
    pub fn bake_status(&self) -> Option<String> {
        let job = self.bake.as_ref()?;
        let stage = self.session.bake_preview.as_ref()?.stage()?;
        let map = i18n::tr(i18n::baked_map_label(stage.map));
        let (n, total) = (stage.position, stage.total);
        if job.total > 1 {
            let set = job.total - job.queue.len();
            return Some(i18n::trf(
                "Baking {map} ({n}/{total}), texture set {set} of {sets}",
                &[("map", &map), ("n", &n), ("total", &total), ("set", &set.max(1)), ("sets", &job.total)],
            ));
        }

        Some(i18n::trf("Baking {map} ({n}/{total})", &[("map", &map), ("n", &n), ("total", &total)]))
    }

    /// Bake progress 0..1 across all queued sets.
    pub fn bake_progress(&self) -> Option<f32> {
        let job = self.bake.as_ref()?;
        let done = job.total - job.queue.len() - usize::from(job.current.is_some());

        // Each map is a bake of its own: its progress runs from 0 to 1 within the set's share
        let current = job
            .current
            .as_ref()
            .map(|running| (running.maps_done.load(Ordering::Relaxed) as f32 + running.progress.fraction()) / running.maps_total.max(1) as f32)
            .unwrap_or(0.0);
        Some(((done as f32 + current) / job.total.max(1) as f32).clamp(0.0, 1.0))
    }
}

/// Run `f` on a worker thread named `name`; its result arrives on the returned channel.
#[cfg(not(target_arch = "wasm32"))]
fn spawn_worker<T: Send + 'static>(name: &str, f: impl FnOnce() -> T + Send + 'static) -> std::sync::mpsc::Receiver<T> {
    let (tx, rx) = std::sync::mpsc::channel();
    let spawned = std::thread::Builder::new().name(name.into()).spawn(move || {
        let _ = tx.send(f());
    });
    if let Err(e) = spawned {
        // The receiver reports a stopped worker.
        log::warn!("could not start {name}: {e}");
    }
    rx
}

#[cfg(target_arch = "wasm32")]
fn spawn_worker<T: Send + 'static>(_name: &str, f: impl FnOnce() -> T + Send + 'static) -> std::sync::mpsc::Receiver<T> {
    // shortcut: no threads on the web; the work runs on the UI thread (the window pauses).
    let (tx, rx) = std::sync::mpsc::channel();
    let _ = tx.send(f());
    rx
}

impl TextureCraftApp {
    /// Every `prefs.autosave_minutes`, write a modified project to the autosave folder (encoded
    /// and written on a worker thread; the project data is shared copy-on-write).
    #[cfg(not(target_arch = "wasm32"))]
    fn autosave(&mut self, now: f64) {
        let period = self.ui.prefs.autosave_minutes as f64 * 60.0;
        // The clock starts at the first unsaved change.
        if period <= 0.0 || !self.session.modified {
            self.next_autosave = now + period;
            return;
        }
        if now < self.next_autosave {
            return;
        }
        self.next_autosave = now + period;
        let dir = user_dir("autosave");
        let path = dir.join(format!(
            "{}.{}",
            texturecraft_engine::formats::export::sanitize_file_stem(&self.session.doc.project.name),
            texturecraft_engine::core::PROJECT_EXT
        ));
        let pf = self.session.project_file();
        let spawned = std::thread::Builder::new().name("texturecraft-autosave".into()).spawn(move || {
            let result = std::fs::create_dir_all(&dir)
                .map_err(|e| e.to_string())
                .and_then(|_| texturecraft_engine::formats::save_project(&path, &pf).map_err(|e| e.to_string()));
            match result {
                Ok(()) => log::info!("autosaved {}", path.display()),
                Err(e) => log::warn!("autosave to {} failed: {e}", path.display()),
            }
        });
        if let Err(e) = spawned {
            log::warn!("could not start autosave: {e}");
        }
    }
}

/// A smooth move of the viewport camera (see [`TextureCraftApp::run`]).
struct CameraGlide {
    from: texturecraft_engine::geom::OrbitCamera,
    to: texturecraft_engine::geom::OrbitCamera,
    start: f64,
}

const GLIDE_SECONDS: f64 = 0.35;

impl TextureCraftApp {
    /// Advance the camera glide by one frame.
    fn step_camera_glide(&mut self, ctx: &egui::Context) {
        let Some(g) = &self.camera_glide else { return };
        let t = ((self.last_time - g.start) / GLIDE_SECONDS).clamp(0.0, 1.0) as f32;
        let e = t * t * (3.0 - 2.0 * t);
        let (a, b) = (g.from, g.to);
        let mix = |x: f32, y: f32| x + (y - x) * e;
        let mut c = b;
        c.target = a.target.lerp(b.target, e);
        c.yaw = mix(a.yaw, b.yaw);
        c.pitch = mix(a.pitch, b.pitch);
        c.distance = mix(a.distance, b.distance);
        c.fov = mix(a.fov, b.fov);
        // Orthographic switches at the end (the projections cannot be blended).
        c.ortho = if t >= 1.0 { b.ortho } else { a.ortho };
        self.session.camera = c;
        self.session.revision = self.session.revision.wrapping_add(1);
        if t >= 1.0 {
            self.camera_glide = None;
        }
        ctx.request_repaint();
    }
}

impl eframe::App for TextureCraftApp {
    /// The browser never tells eframe the page is closing, so save often there.
    fn auto_save_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(if cfg!(target_arch = "wasm32") { 5 } else { 30 })
    }

    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        self.ui_store.save(&self.ui, storage);
        self.report_ui_storage_warning();
        export_templates::save_library(self, storage);
    }

    fn logic(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        let t0 = now_ms();
        let now = ctx.input(|i| i.time);
        let dt = (now - self.last_time) as f32;
        if dt > 0.0 {
            self.fps = self.fps * 0.9 + (1.0 / dt) * 0.1;
        }
        self.last_time = now;
        let style = (self.ui.prefs.brightness, self.ui.prefs.accent);
        if self.styled != Some(style) {
            theme::apply(ctx, style.0, style.1);
            self.styled = Some(style);
        }
        self.step_camera_glide(ctx);
        self.ui.prefs.apply_resource_limits();
        control::poll(self, ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.run_file_dialog_answers();
        web_files::poll(self);
        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for f in &dropped {
            web_files::dropped(self, ctx, f);
        }
        self.poll_bake(ctx);
        self.poll_resize(ctx);
        self.poll_link_cooks(ctx);
        // The library shelf rescans when the window comes back (MaterialCraft may have sent a material)
        let focused = ctx.input(|i| i.viewport().focused).unwrap_or(true);
        if focused && !self.focused {
            self.library = None;
        }
        self.focused = focused;
        let scale = self.ui.prefs.ui_scale.clamp(0.5, 3.0);
        if (ctx.zoom_factor() - scale).abs() > 1e-3 {
            ctx.set_zoom_factor(scale);
        }
        #[cfg(not(target_arch = "wasm32"))]
        self.autosave(now);
        i18n::set_language(&self.ui.prefs.language);
        #[cfg(target_arch = "wasm32")]
        {
            keymap::sync_captured(&self.ui.keymap);
            web_files::sync_unsaved(self);
        }
        self.fonts.update(ctx, i18n::current().code, self.ui.windows.settings);
        #[cfg(not(target_arch = "wasm32"))]
        updates_ui::tick(self, ctx);
        self.bindings.sync(&self.ui.keymap, craft_keymap::egui::platform(ctx));
        menus::shortcuts(self, ctx);
        // The Path tool's curve repaints (a preview while placed) as its points, brush or material change
        if let Err(e) = self.session.path_live() {
            self.session.cancel_path();
            self.toast(e.to_string(), true);
        }
        profile::drive(self, ctx);
        self.toasts.retain(|t| t.until > now);
        self.profiler.logic_ms = (now_ms() - t0) as f32;
        // Last, so it sees every close request this frame's work made.
        windows::close_request(self, ctx);
    }

    #[cfg_attr(target_arch = "wasm32", allow(unused_variables))]
    fn ui(&mut self, ui: &mut egui::Ui, frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        let t0 = now_ms();
        if !self.ui.hide_ui {
            titlebar::show(self, ui);
            banner(self, ui);
            toolbar::status_bar(self, ui);
            toolbar::tool_shelf(self, ui);
            self.ui.sync_mode_dock();
            if self.ui.mode != state::Mode::Baking {
                toolbar::rail(self, ui);
            }
        }
        let hide = self.ui.hide_ui;
        let tk = theme::Tokens::get(&ctx);
        // The dock keeps the cards' gutter from the bars and rails around it
        let panel_frame = if hide { egui::Frame::NONE.fill(tk.viewport) } else { egui::Frame::NONE.fill(tk.chrome).inner_margin(theme::GUTTER) };
        egui::CentralPanel::default().frame(panel_frame).show(ui, |ui| if hide { viewport::central(self, ui) } else { docking::show(self, ui) });
        windows::show(self, &ctx);
        #[cfg(not(target_arch = "wasm32"))]
        updates_ui::show(self, &ctx);
        toasts(self, &ctx);
        #[cfg(not(target_arch = "wasm32"))]
        self.file_dialogs.show_pending(frame);
        let end = now_ms();
        let dt = (end - t0) as f32;
        self.ui_ms = if self.ui_ms == 0.0 { dt } else { self.ui_ms * 0.9 + dt * 0.1 };
        self.profiler.end_frame(end, dt);
    }
}

/// The data folder chosen in Settings, fixed at startup (unset: the default).
#[cfg(not(target_arch = "wasm32"))]
static DATA_FOLDER: std::sync::OnceLock<std::path::PathBuf> = std::sync::OnceLock::new();

/// The data folder for a `Prefs::data_folder` value: that path when it is absolute, otherwise
/// `Documents/TextureCraft` in the user's home folder.
#[cfg(not(target_arch = "wasm32"))]
pub fn data_folder(pref: &str) -> std::path::PathBuf {
    let chosen = std::path::Path::new(pref.trim());
    if chosen.is_absolute() {
        return chosen.to_path_buf();
    }
    let home = std::env::var("USERPROFILE").or_else(|_| std::env::var("HOME")).unwrap_or_else(|_| ".".into());
    std::path::Path::new(&home).join("Documents").join("TextureCraft")
}

/// `<sub>` in the data folder in use (Settings › Folders › Data folder).
#[cfg(not(target_arch = "wasm32"))]
pub fn user_dir(sub: &str) -> std::path::PathBuf {
    DATA_FOLDER.get().cloned().unwrap_or_else(|| data_folder("")).join(sub)
}

/// Folder holding the user's smart materials.
#[cfg(not(target_arch = "wasm32"))]
pub fn library_dir() -> std::path::PathBuf {
    user_dir("library")
}

/// Milliseconds since the first call (0 on the web, where `std::time` is unavailable).
pub fn now_ms() -> f64 {
    #[cfg(not(target_arch = "wasm32"))]
    {
        static START: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
        START.get_or_init(std::time::Instant::now).elapsed().as_secs_f64() * 1000.0
    }
    #[cfg(target_arch = "wasm32")]
    0.0
}

/// The dismissable notice under the menu bar.
fn banner(app: &mut TextureCraftApp, ui: &mut egui::Ui) {
    let Some(text) = app.banner else { return };
    let tk = theme::Tokens::get(ui.ctx());
    egui::Panel::top("banner").show_separator_line(false).frame(egui::Frame::NONE.fill(tk.row_selected).inner_margin(egui::Margin::symmetric(10, 5))).show(
        ui,
        |ui| {
            ui.horizontal(|ui| {
                ui.label(egui::RichText::new(egui_phosphor::regular::INFO).color(tk.accent));
                // The text wraps in the width the dismiss button leaves.
                let width = (ui.available_width() - 28.0).max(80.0);
                ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                    ui.add(egui::Label::new(egui::RichText::new(i18n::tr(text)).color(tk.text_strong)).wrap());
                });
                if widgets::icon_button(ui, egui_phosphor::regular::X, i18n::tr("Dismiss"), false, 18.0).clicked() {
                    app.banner = None;
                }
            });
        },
    );
}

fn toasts(app: &mut TextureCraftApp, ctx: &egui::Context) {
    if app.toasts.is_empty() {
        return;
    }
    // Expiring toasts need frames to disappear on time; sticky ones wait for a click.
    if app.toasts.iter().any(|t| t.until.is_finite()) {
        ctx.request_repaint_after(std::time::Duration::from_millis(250));
    }
    let tk = theme::Tokens::get(ctx);
    let mut dismissed = None;
    // Toasts fit the window: at most 360 points wide and, for their text, a part of its height
    // (longer text scrolls), so a sticky toast's dismiss button always stays on screen.
    let screen = ctx.content_rect().size();
    let max_width = (screen.x - 52.0).clamp(120.0, 360.0);
    let max_text_height = (screen.y * 0.35).clamp(48.0, 220.0);
    egui::Area::new(egui::Id::unique("toasts")).anchor(egui::Align2::RIGHT_BOTTOM, egui::vec2(-16.0, -36.0)).show(ctx, |ui| {
        for (i, t) in app.toasts.iter().enumerate() {
            egui::Frame::NONE
                .fill(tk.panel)
                .stroke(egui::Stroke::new(1.0, if t.error { tk.danger } else { tk.accent }))
                .corner_radius(crate::theme::CARD_RADIUS)
                .shadow(ui.visuals().popup_shadow)
                .inner_margin(egui::Margin::symmetric(10, 6))
                .show(ui, |ui| {
                    ui.set_max_width(max_width);
                    let sticky = t.until.is_infinite();
                    ui.horizontal_top(|ui| {
                        // The text wraps in the width the dismiss button leaves.
                        let width = if sticky { max_width - 28.0 } else { max_width };
                        ui.allocate_ui_with_layout(egui::vec2(width, 0.0), egui::Layout::top_down(egui::Align::Min), |ui| {
                            egui::ScrollArea::vertical().id_salt(("toast", i)).max_height(max_text_height).show(ui, |ui| {
                                ui.set_max_width(width);
                                ui.add(egui::Label::new(egui::RichText::new(&t.text).color(tk.text_strong)).wrap());
                            });
                        });
                        if sticky && widgets::icon_button(ui, egui_phosphor::regular::X, i18n::tr("Dismiss"), false, 18.0).clicked() {
                            dismissed = Some(i);
                        }
                    });
                });
            ui.add_space(4.0);
        }
    });
    if let Some(i) = dismissed.filter(|i| *i < app.toasts.len()) {
        app.toasts.remove(i);
    }
}
