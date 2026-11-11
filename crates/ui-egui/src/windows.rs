//! Floating windows and dockable display, shader, history and log panels

use egui::{Color32, RichText, Ui, vec2};
use egui_phosphor::regular as ic;
use serde_json::json;
use texturecraft_engine::core::ToneMapping;
use texturecraft_engine::core::project::ENV_RESOURCE_PREFIX;
use texturecraft_engine::geom::primitives::BUILTINS;
use texturecraft_render::ibl::ENVIRONMENTS;

use crate::TextureCraftApp;
#[cfg(not(target_arch = "wasm32"))]
use crate::file_dialogs::{Bound, FileDialog};
use crate::i18n::LANGUAGES;
use crate::i18n::{tr, trf};
use crate::new_project_advanced::COMBO_WIDTH;
use crate::state::Mode;
use crate::theme::{Accent, Brightness, Tokens, caption};
use crate::widgets::{color_swatch, combo, icon_button, primary_button, section, slider};

pub fn show(app: &mut TextureCraftApp, ctx: &egui::Context) {
    start(app, ctx);
    new_project(app, ctx);
    about(app, ctx);
    settings(app, ctx);
    crate::keymap::window(app, ctx);
    crate::windows::export(app, ctx);
    #[cfg(target_arch = "wasm32")]
    export_mesh(app, ctx);
    unsaved(app, ctx);
}

/// Hold `then` (quitting, opening or closing a project) behind the unsaved-changes dialog when
/// the project has unsaved changes; false when it can go ahead now.
pub(crate) fn ask_to_save(app: &mut TextureCraftApp, then: crate::state::Unsaved) -> bool {
    if app.session.modified {
        app.ui.windows.unsaved = Some(then);
    }
    app.session.modified
}

/// The close decision, once per frame in `logic` (which also runs while the window is minimized):
/// a close with unsaved changes waits on the dialog, in a window brought back so it can be
/// answered. A close nothing holds open quits through the platform's own path (craft-quit).
pub(crate) fn close_request(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let closing = ctx.input(|i| i.viewport().close_requested());
    // A stroke still open at the close, or released in this frame, is a change to ask about.
    if closing
        && app.session.stroke_active()
        && let Err(error) = app.session.end_stroke()
    {
        app.toast(error.to_string(), true);
    }
    if closing && ask_to_save(app, crate::state::Unsaved::Quit) {
        ctx.send_viewport_cmd(egui::ViewportCommand::CancelClose);
        if ctx.input(|i| i.viewport().minimized == Some(true)) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(false));
            ctx.send_viewport_cmd(egui::ViewportCommand::Focus);
        }
    }
    craft_quit::on_close_request(ctx);
}

/// Quitting, opening or closing a project with unsaved changes asks whether to save them first
/// (in the browser, Save downloads the project).
fn unsaved(app: &mut TextureCraftApp, ctx: &egui::Context) {
    use crate::state::Unsaved;

    let Some(pending) = app.ui.windows.unsaved.clone() else { return };

    #[cfg(not(target_arch = "wasm32"))]
    let (waiting_for_save, saved_by_dialog) = (app.saving_for_prompt, std::mem::take(&mut app.prompt_saved));
    #[cfg(target_arch = "wasm32")]
    let (waiting_for_save, saved_by_dialog) = (false, false);

    let mut open = true;
    let mut save_first = None;
    let mut cancel = false;
    let question_args: [(&str, &dyn std::fmt::Display); 1] = [("name", &app.session.doc.project.name)];
    let question = if pending == Unsaved::Quit {
        trf("Save the changes to {name} before closing?", &question_args)
    } else {
        trf("Save the changes to {name} first?", &question_args)
    };
    dialog("unsaved", tr("Unsaved changes"), &mut open).show(ctx, |ui| {
        ui.label(question);
        ui.add_space(8.0);

        // One row keeps the window to its contents; a bare right-to-left layout fills the screen height
        ui.horizontal(|ui| {
            // The answers wait while the Save dialog is open
            ui.add_enabled_ui(!waiting_for_save, |ui| {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(primary_button(ui, tr("Save"))).clicked() {
                        save_first = Some(true);
                    }
                    if ui.button(tr("Don't save")).clicked() {
                        save_first = Some(false);
                    }
                    if ui.button(tr("Cancel")).clicked() {
                        cancel = true;
                    }
                });
            });
        });
    });
    if !open || cancel {
        app.ui.windows.unsaved = None;
        #[cfg(not(target_arch = "wasm32"))]
        {
            app.saving_for_prompt = false;
        }
    }

    match save_first {
        Some(true) => {
            // An untitled project is saved from a dialog that answers on a later frame
            #[cfg(not(target_arch = "wasm32"))]
            {
                app.saving_for_prompt = app.session.path.is_none();
            }
            file_action(app, ctx, "file.save");
        }
        // Going ahead anyway: the changes are let go
        Some(false) => app.session.modified = false,
        None if saved_by_dialog => {}
        None => return,
    }

    // A canceled or failed save keeps the project and this dialog open
    if app.session.modified {
        return;
    }
    app.ui.windows.unsaved = None;
    match pending {
        Unsaved::Quit => ctx.send_viewport_cmd(egui::ViewportCommand::Close),
        Unsaved::Command(id) => crate::menus::invoke(app, ctx, &id),
        #[cfg(not(target_arch = "wasm32"))]
        Unsaved::Open(path) => open_path(app, &path),
        Unsaved::Load(name, bytes) => crate::web_files::load(app, &name, std::sync::Arc::unwrap_or_clone(bytes)),
    }
}

/// A titled tool window; `id` keeps its place across UI languages.
pub(crate) fn window<'a>(id: &str, title: &str, open: &'a mut bool) -> egui::Window<'a> {
    egui::Window::new(RichText::new(title).font(crate::theme::semibold(13.0)).strong())
        .id(egui::Id::unique(id))
        .open(open)
        .resizable(true)
        .collapsible(false)
        .default_width(300.0)
}

/// A centered dialog; `id` keeps its place across UI languages.
fn dialog<'a>(id: &str, title: &str, open: &'a mut bool) -> egui::Window<'a> {
    egui::Window::new(title.to_owned())
        .id(egui::Id::unique(id))
        .open(open)
        .collapsible(false)
        .resizable(false)
        .anchor(egui::Align2::CENTER_CENTER, vec2(0.0, 0.0))
}

/// The Display Settings panel body (a dock tab).
pub(crate) fn display(app: &mut TextureCraftApp, ui: &mut Ui) {
    let ctx = ui.ctx().clone();
    let mut d = app.session.doc.project.display.clone();
    let before = d.clone();
    ui.label(caption(tr("Environment settings")));
    let mut hdrs: Vec<&String> = app.session.images.iter().filter(|(_, i)| !i.srgb).map(|(k, _)| k).collect();
    hdrs.sort();
    let mut import_hdr = false;
    let shown = d.environment_resource().unwrap_or(&d.environment).to_string();
    combo(ui, tr("Environment map"), shown, |ui| {
        for e in ENVIRONMENTS {
            ui.selectable_value(&mut d.environment, e.to_string(), e);
        }
        ui.separator();
        for name in hdrs {
            ui.selectable_value(&mut d.environment, format!("{ENV_RESOURCE_PREFIX}{name}"), name.as_str());
        }
        import_hdr = ui
            .selectable_label(false, format!("{} {}", ic::FILE_ARROW_UP, tr("Import HDR...")))
            .on_hover_text(tr("Use an equirectangular .hdr or .exr image as the environment"))
            .clicked();
    });
    slider(ui, tr("Environment opacity"), &mut d.env_opacity, 0.0..=1.0, 2);
    slider(ui, tr("Environment exposure (EV)"), &mut d.env_exposure, -5.0..=5.0, 2);
    slider(ui, tr("Environment blur"), &mut d.env_blur, 0.0..=1.0, 2);
    slider(ui, tr("Environment rotation"), &mut d.env_rotation, 0.0..=360.0, 0);
    ui.checkbox(&mut d.shadows, tr("Key light (shadow-casting light, direct only)"));
    ui.add_space(6.0);
    ui.label(caption(tr("Camera settings")));
    slider(ui, tr("Field of view (deg)"), &mut d.fov, 5.0..=120.0, 0);
    let mut ortho = app.session.camera.ortho;
    if ui.checkbox(&mut ortho, tr("Orthographic")).changed() {
        app.run("camera.set", json!({"ortho": ortho}));
    }
    let tone_mapping_name = |tone_mapping: ToneMapping| match tone_mapping {
        ToneMapping::Linear => tr("Linear"),
        ToneMapping::AcesFilmic => tr("ACES filmic"),
    };
    combo(ui, tr("Tone mapping"), tone_mapping_name(d.tone_mapping), |ui| {
        for tone_mapping in [ToneMapping::Linear, ToneMapping::AcesFilmic] {
            ui.selectable_value(&mut d.tone_mapping, tone_mapping, tone_mapping_name(tone_mapping));
        }
    });
    cameras(app, ui);
    ui.add_space(6.0);
    ui.label(caption(tr("Depth of field")));
    ui.checkbox(&mut d.dof.enabled, tr("Depth of field (viewport and Rendering mode)"));
    ui.add_enabled_ui(d.dof.enabled, |ui| {
        let far = (app.session.camera.distance * 4.0).max(d.dof.focus_distance);
        slider(ui, tr("Focus distance"), &mut d.dof.focus_distance, 0.0..=far, 2)
            .on_hover_text(tr("Depth of the sharp plane along the view (0: the orbit center)"));
        slider(ui, tr("Aperture"), &mut d.dof.aperture, 0.0..=1.0, 3).on_hover_text(tr("Lens size: how quickly things blur away from the focus distance"));
        let picking = crate::viewport::focus_picking(ui.ctx());
        if ui.selectable_label(picking, format!("{}  {}", ic::CROSSHAIR_SIMPLE, tr("Pick focus in the viewport"))).clicked() {
            crate::viewport::set_focus_picking(ui.ctx(), !picking);
        }
    });
    ui.add_space(6.0);
    ui.label(caption(tr("Viewer settings")));
    ui.checkbox(&mut app.ui.wireframe_3d, tr("Show wireframe in the 3D view"));
    ui.checkbox(&mut app.ui.uv_wireframe, tr("Show UV wireframe in the 2D view"));
    ui.checkbox(&mut d.taa, tr("Temporal anti-aliasing (refines the 3D view while it is still)"));
    if d != before {
        if d.fov != before.fov {
            app.session.camera.fov = d.fov;
        }
        app.session.doc.project.display = d;
    }
    if import_hdr {
        crate::web_files::import_environment(app, &ctx);
    }
}

/// The camera list: rename (edit the name), view or delete saved and imported cameras, save the
/// current view.
fn cameras(app: &mut TextureCraftApp, ui: &mut Ui) {
    let current = texturecraft_engine::camera_cmds::current(&app.session);
    let names: Vec<String> = app.session.doc.project.cameras.iter().map(|c| c.name.clone()).collect();
    for (i, name) in names.iter().enumerate() {
        ui.horizontal(|ui| {
            if icon_button(ui, ic::VIDEO_CAMERA, tr("View through this camera"), current == Some(i), 20.0).clicked() {
                app.run("camera.select", json!({"index": i}));
            }
            // The name being typed lives in egui's memory until the field loses focus.
            let id = ui.scope_id().with(("camera_name", i));
            let mut text = ui.data(|d| d.get_temp::<String>(id)).unwrap_or_else(|| name.clone());
            let r = ui.add(egui::TextEdit::singleline(&mut text).desired_width(ui.available_width() - 26.0));
            if r.changed() {
                ui.data_mut(|d| d.insert_temp(id, text.clone()));
            }
            if r.lost_focus() {
                ui.data_mut(|d| d.remove::<String>(id));
                if text.trim() != name && !text.trim().is_empty() {
                    app.run("camera.rename", json!({"index": i, "to": text.trim()}));
                }
            }
            if icon_button(ui, ic::TRASH, tr("Delete this camera"), false, 20.0).clicked() {
                app.run("camera.remove", json!({"index": i}));
            }
        });
    }
    if ui.button(format!("{}  {}", ic::PLUS, tr("Save view as camera"))).clicked() {
        app.run("camera.add", json!({}));
    }
}

/// Wide enough for the longer shader model name
const SHADER_MODEL_COMBO_WIDTH: f32 = 260.0;

/// Width of the scatter color swatch
const SCATTER_SWATCH_WIDTH: f32 = 44.0;

/// Index of refraction slider range: air to well past diamond (2.42)
const INDEX_OF_REFRACTION_RANGE: std::ops::RangeInclusive<f32> = 1.0..=3.0;

/// The Shader Settings panel body (a dock tab)
pub(crate) fn shader(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tokens = Tokens::get(ui.ctx());
    let mut display = app.session.doc.project.display.clone();
    let before = display.clone();

    let model_name = |spec_gloss: bool| if spec_gloss { tr("TextureCraft PBR - Specular Glossiness") } else { tr("TextureCraft PBR - Metallic Roughness") };
    let channels = if display.spec_gloss {
        tr("Base color is the diffuse color, Specular the F0 color and Glossiness 1 - roughness (add these channels in the texture set settings).")
    } else {
        tr("Base color, Metallic, Roughness and Specular level.")
    };
    let model_text = tr(
        "GGX specular with height-correlated Smith visibility, Lambert diffuse, split-sum image-based lighting with spherical-harmonics irradiance and horizon fading.",
    );
    let optional = tr(
        "Optional channels add a clear coat (Coat weight, Coat roughness, Coat normal), sheen (Sheen weight, color, roughness), anisotropy (Anisotropy level, angle), subsurface scattering (Scattering weight) and transmission (Transmission).",
    );

    egui::ComboBox::from_id_salt("shader_model")
        .width(SHADER_MODEL_COMBO_WIDTH)
        .selected_text(model_name(display.spec_gloss))
        .show_ui(ui, |ui| {
            for spec_gloss in [false, true] {
                ui.selectable_value(&mut display.spec_gloss, spec_gloss, model_name(spec_gloss));
            }
        })
        .response
        .on_hover_text(format!("{model_text} {channels} {optional}"));
    ui.add_space(6.0);

    ui.checkbox(&mut display.parallax, tr("Parallax occlusion (height channel)"))
        .on_hover_text(tr("Offsets the textures by the height channel, at the height to normal strength"));
    ui.checkbox(&mut display.displacement, tr("Displacement (height channel)")).on_hover_text(tr(
        "Moves the surface by the height channel on a finer copy of the mesh, at the height to normal strength (viewport only: painting follows the base surface)",
    ));
    slider(ui, tr("AO intensity"), &mut display.ao_intensity, 0.0..=1.0, 2);
    ui.add_space(6.0);

    section(ui, "shader_subsurface", tr("Subsurface scattering"), true, |ui| {
        ui.horizontal(|ui| {
            ui.label(RichText::new(tr("Scatter color")).color(tokens.text));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                color_swatch(ui, "shader_scatter_color", &mut display.scatter_color, SCATTER_SWATCH_WIDTH)
            })
        })
        .response
        .on_hover_text(tr("Color of the light that travels under the surface (each channel goes as far as its value)"));
        slider(ui, tr("Scatter radius"), &mut display.scatter_radius, 0.0..=1.0, 3)
            .on_hover_text(tr("How far light travels under the surface, as a fraction of the mesh size (where the Scattering weight channel is painted)"));
    });
    section(ui, "shader_transmission", tr("Transmission"), true, |ui| {
        slider(ui, tr("Index of refraction"), &mut display.ior, INDEX_OF_REFRACTION_RANGE, 2).on_hover_text(tr(
            "How strongly transmissive surfaces bend light (where the Transmission channel is painted): water 1.33, glass 1.5, diamond 2.42",
        ));
    });

    if display != before {
        app.session.doc.project.display = display;
    }
}

/// The History panel body (a dock tab)
pub(crate) fn history(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tokens = Tokens::get(ui.ctx());
    let (done, undone) = (app.session.history_labels(), app.session.redo_labels());
    // A click on a step goes back (or forward) to it: steps to undo, or to redo when negative
    let mut jump = 0i64;

    // Bottom up: the Undo and Redo row keeps its place under a list that fills the rest of the panel
    ui.with_layout(egui::Layout::bottom_up(egui::Align::Min), |ui| {
        ui.horizontal(|ui| {
            if ui.add_enabled(app.session.can_undo(), egui::Button::new(format!("{} {}", ic::ARROW_COUNTER_CLOCKWISE, tr("Undo")))).clicked() {
                app.run("edit.undo", json!({}));
            }
            if ui.add_enabled(app.session.can_redo(), egui::Button::new(format!("{} {}", ic::ARROW_CLOCKWISE, tr("Redo")))).clicked() {
                app.run("edit.redo", json!({}));
            }
        });

        ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
            egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                let done_count = done.len() as i64;
                let rows = std::iter::once(RichText::new(tr("Project opened")).color(tokens.text_dim))
                    .chain(done.iter().map(RichText::new))
                    .chain(undone.iter().map(|label| RichText::new(trf("{action} (undone)", &[("action", label)])).color(tokens.text_dim)));
                for (row, text) in rows.enumerate() {
                    // Each row is the state after that many steps
                    let steps = done_count - row as i64;
                    if ui.selectable_label(steps == 0, text).on_hover_text(tr("Return to this step")).clicked() {
                        // Point history is consumed before the document rows shown here
                        jump = if steps >= 0 {
                            app.session.history_undo_steps(steps as usize) as i64
                        } else {
                            -(app.session.history_redo_steps((-steps) as usize) as i64)
                        };
                    }
                }
            });
        });
    });

    if jump > 0 {
        app.run("edit.undo", json!({ "steps": jump }));
    } else if jump < 0 {
        app.run("edit.redo", json!({ "steps": -jump }));
    }
}

/// What the start window was asked for.
enum Start {
    Menu(&'static str),
    #[cfg(not(target_arch = "wasm32"))]
    Recent(String),
    Close,
}

/// The start window: a new project, a project file or a recent one, or the sample that is open.
fn start(app: &mut TextureCraftApp, ctx: &egui::Context) {
    if !app.ui.windows.start {
        return;
    }

    #[cfg(not(target_arch = "wasm32"))]
    let tokens = Tokens::get(ctx);
    let mut open = true;
    let mut action = None;
    dialog("start", "TextureCraft", &mut open).min_width(380.0).show(ctx, |ui| {
        ui.horizontal(|ui| {
            if ui.add(primary_button(ui, tr("New project..."))).clicked() {
                action = Some(Start::Menu("ui.newProject"));
            }
            if ui.button(tr("Open...")).clicked() {
                action = Some(Start::Menu("file.open"));
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        if file_dialogs_available() {
            ui.add_space(8.0);
            ui.label(caption(tr("Recent projects")));
            if app.ui.recent.is_empty() {
                ui.label(RichText::new(tr("No recent projects yet")).color(tokens.text_dim));
            }
            for path in &app.ui.recent {
                let name = std::path::Path::new(path).file_name().and_then(|n| n.to_str()).unwrap_or(path);
                if ui
                    .add(egui::Button::selectable(false, format!("{}  {name}", ic::FILE)).min_size(vec2(ui.available_width(), 0.0)))
                    .on_hover_text(path)
                    .clicked()
                {
                    action = Some(Start::Recent(path.clone()));
                }
            }
        }
        ui.add_space(8.0);
        ui.separator();
        ui.horizontal(|ui| {
            ui.checkbox(&mut app.ui.prefs.start_window, tr("Show at startup"));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(tr("Continue with the sample project")).clicked() {
                    action = Some(Start::Close);
                }
            });
        });
    });

    if let Some(action) = action {
        app.ui.windows.start = false;
        match action {
            Start::Menu(id) => crate::menus::invoke(app, ctx, id),
            #[cfg(not(target_arch = "wasm32"))]
            Start::Recent(path) => open_path(app, &path),
            Start::Close => {}
        }
    }

    if !open {
        app.ui.windows.start = false;
    }
}

fn new_project(app: &mut TextureCraftApp, ctx: &egui::Context) {
    if !app.ui.windows.new_project {
        return;
    }

    let mut open = true;
    let mut create = false;
    dialog("new_project", tr("New project"), &mut open).show(ctx, |ui| {
        ui.label(caption(tr("Basic settings")));
        egui::Grid::new("np").num_columns(2).spacing(vec2(12.0, 8.0)).min_col_width(crate::new_project_advanced::form_label_width(ui)).show(ui, |ui| {
            ui.label(tr("3D mesh"));
            ui.horizontal(|ui| {
                let label = app.ui.new_project.file.clone().unwrap_or_else(|| {
                    BUILTINS
                        .iter()
                        .find(|(id, _)| *id == app.ui.new_project.mesh)
                        .map(|(_, builtin_label)| trf("Built-in: {mesh}", &[("mesh", &tr(builtin_label))]))
                        .unwrap_or_default()
                });
                egui::ComboBox::from_id_salt("np_mesh").width(COMBO_WIDTH).selected_text(label).show_ui(ui, |ui| {
                    for (id, builtin_label) in BUILTINS {
                        if ui
                            .selectable_label(
                                app.ui.new_project.mesh == id && app.ui.new_project.file.is_none(),
                                trf("Built-in: {mesh}", &[("mesh", &tr(builtin_label))]),
                            )
                            .clicked()
                        {
                            app.ui.new_project.mesh = id.to_string();
                            app.ui.new_project.file = None;
                        }
                    }
                });
                if file_dialogs_available() && ui.button(tr("Select...")).clicked() {
                    ask_mesh_file(app);
                }
            });
            ui.end_row();
            ui.label(tr("Project template"));
            let new_project = &mut app.ui.new_project;
            egui::ComboBox::from_id_salt("np_template").width(COMBO_WIDTH).selected_text(new_project.template.clone()).show_ui(ui, |ui| {
                let user_templates = app.session.templates.list.iter();
                for template in texturecraft_engine::template_cmds::builtin_templates().iter().chain(user_templates) {
                    if ui.selectable_label(template.name == new_project.template, &template.name).clicked() {
                        new_project.template = template.name.clone();
                        new_project.resolution = template.resolution;
                        new_project.normal_format = None;
                    }
                }
            });
            ui.end_row();
            ui.label("");
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut app.ui.new_project.template_name).hint_text(tr("Template name")).desired_width(120.0));
                let name = app.ui.new_project.template_name.trim().to_string();
                if ui.add_enabled(!name.is_empty(), egui::Button::new(tr("Save current project as template"))).clicked()
                    && app.run("project.template.save", json!({"name": name})).is_some()
                {
                    app.ui.new_project.template = name;
                    app.ui.new_project.template_name.clear();
                }
            });
            ui.end_row();
            ui.label(tr("Document resolution"));
            egui::ComboBox::from_id_salt("np_res").selected_text(format!("{0} x {0}", app.ui.new_project.resolution)).show_ui(ui, |ui| {
                for size in [512u32, 1024, 2048, 4096] {
                    ui.selectable_value(&mut app.ui.new_project.resolution, size, format!("{size} x {size}"));
                }
            });
            ui.end_row();
            ui.label(tr("UV tiles (UDIM)"));
            let tiles_label = |tiles: Option<bool>| match tiles {
                None => tr("Automatic (from the mesh UVs)"),
                Some(true) => tr("On"),
                Some(false) => tr("Off"),
            };
            egui::ComboBox::from_id_salt("np_tiles").selected_text(tiles_label(app.ui.new_project.uv_tiles)).show_ui(ui, |ui| {
                for tiles in [None, Some(true), Some(false)] {
                    ui.selectable_value(&mut app.ui.new_project.uv_tiles, tiles, tiles_label(tiles));
                }
            });
            ui.end_row();
            ui.label(tr("Name"));
            ui.text_edit_singleline(&mut app.ui.new_project.name);
            ui.end_row();
        });
        ui.add_space(8.0);
        crate::new_project_advanced::advanced_settings(app, ui);
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(primary_button(ui, tr("Create"))).clicked() {
                    create = true;
                }
                if ui.button(tr("Cancel")).clicked() {
                    app.ui.windows.new_project = false;
                }
            });
        });
    });
    if create {
        let new_project = app.ui.new_project.clone();
        let (command, mesh) = match &new_project.file {
            Some(path) => ("file.new", path.clone()),
            None => ("project.new", new_project.mesh.clone()),
        };
        let params = json!({"mesh": mesh, "resolution": new_project.resolution, "name": new_project.name, "uvTiles": new_project.uv_tiles, "template": new_project.template});
        if app.run(command, params).is_some() {
            app.ui.windows.new_project = false;
            app.ui.mode = Mode::Painting;
            crate::new_project_advanced::apply_after_create(app, &new_project);
        }
    }
    if !open {
        app.ui.windows.new_project = false;
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn settings_resources(ctx: &egui::Context) -> (u32, u64) {
    let id = egui::Id::unique("settings_resources");
    let now = ctx.input(|input| input.time);
    ctx.request_repaint_after(std::time::Duration::from_secs(1));

    // Query the operating system once a second while Settings is open, rather than every frame
    if let Some((sampled_at, cores, available_ram_bytes)) = ctx.data(|data| data.get_temp::<(f64, u32, u64)>(id))
        && (0.0..1.0).contains(&(now - sampled_at))
    {
        return (cores, available_ram_bytes);
    }

    let (cores, available_ram_bytes) = texturecraft_engine::formats::available_save_resources();
    let cores = u32::try_from(cores).unwrap_or(u32::MAX);
    ctx.data_mut(|data| data.insert_temp(id, (now, cores, available_ram_bytes)));
    (cores, available_ram_bytes)
}

fn settings(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let tk = Tokens::get(ctx);
    let mut open = app.ui.windows.settings;
    // A language whose script no font draws is listed by its English name and disabled.
    let drawable = LANGUAGES.each_ref().map(|l| crate::fonts::Script::of_language(l.code).is_none_or(|s| app.fonts.covers(s)));
    #[cfg(not(target_arch = "wasm32"))]
    let mut pick_data_folder = false;
    window("settings", tr("Settings"), &mut open).default_width(360.0).show(ctx, |ui| {
        let p = &mut app.ui.prefs;
        // Every section is a two-column form: labels on the left, controls aligned beside them.
        let form = |id: &str| egui::Grid::new(id).num_columns(2).spacing(vec2(12.0, 8.0)).min_col_width(120.0);
        ui.label(caption(tr("Interface")));
        form("settings_interface").show(ui, |ui| {
            ui.label(tr("Language"));
            let current = crate::i18n::current();
            egui::ComboBox::from_id_salt("language").width(180.0).height(320.0).selected_text(current.name).show_ui(ui, |ui| {
                for (l, ok) in LANGUAGES.iter().zip(drawable) {
                    let item = egui::Button::selectable(l.code == current.code, if ok { l.name } else { l.english });
                    if ui.add_enabled(ok, item).on_disabled_hover_text(tr("No font on this system can show this language")).clicked() {
                        p.language = l.code.to_owned();
                    }
                }
            });
            ui.end_row();
            ui.label(tr("Interface scale"));
            ui.add(egui::Slider::new(&mut p.ui_scale, 0.75..=2.0).fixed_decimals(2));
            ui.end_row();
            ui.label(tr("Brightness"));
            egui::ComboBox::from_id_salt("brightness").width(140.0).selected_text(tr(p.brightness.label())).show_ui(ui, |ui| {
                for b in Brightness::ALL {
                    ui.selectable_value(&mut p.brightness, b, tr(b.label()));
                }
            });
            ui.end_row();
            ui.label(tr("Accent color"));
            ui.horizontal(|ui| {
                for a in Accent::ALL {
                    let (rect, resp) = ui.allocate_exact_size(vec2(20.0, 20.0), egui::Sense::click());
                    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::RadioButton, true, p.accent == a, tr(a.label())));
                    let ring = if p.accent == a {
                        tk.text_strong
                    } else if resp.hovered() {
                        tk.text_dim
                    } else {
                        Color32::TRANSPARENT
                    };
                    ui.painter().circle(rect.center(), 6.5, a.colors(tk.dark).0, egui::Stroke::NONE);
                    ui.painter().circle_stroke(rect.center(), 9.0, egui::Stroke::new(1.5, ring));
                    if resp.on_hover_text(tr(a.label())).clicked() {
                        p.accent = a;
                    }
                }
            });
            ui.end_row();
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label(tr("Title bar"));
                ui.checkbox(&mut p.system_title_bar, tr("System title bar (applies on restart)"))
                    .on_hover_text(tr("Use the operating system's window frame and buttons instead of TextureCraft's own title bar"));
                ui.end_row();
            }
            ui.label(tr("Start window"));
            ui.checkbox(&mut p.start_window, tr("Show at startup"));
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.label(caption(tr("Viewport")));
        form("settings_viewport").show(ui, |ui| {
            ui.label(tr("Texture resolution"));
            ui.horizontal(|ui| {
                for (v, l) in [(0u8, tr("Full")), (1, tr("Half")), (2, tr("Quarter"))] {
                    ui.selectable_value(&mut p.viewport_texture, v, l);
                }
            })
            .response
            .on_hover_text(tr(
                "Size of the textures the viewport shows: lower uses less GPU memory with large texture sets. Painting and export keep full resolution.",
            ));
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.label(caption(tr("Project")));
        #[cfg(not(target_arch = "wasm32"))]
        let (available_cores, available_ram_bytes) = settings_resources(ctx);
        form("settings_project").show(ui, |ui| {
            #[cfg(not(target_arch = "wasm32"))]
            {
                ui.label(tr("Autosave every"));
                ui.add(egui::DragValue::new(&mut p.autosave_minutes).range(0..=120).suffix(format!(" {}", tr("min"))));
                ui.end_row();
                let label = ui.label(tr("Cores kept for the UI"));
                ui.add(egui::DragValue::new(&mut p.reserved_cores).range(0..=available_cores)).labelled_by(label.id);
                ui.end_row();
            }

            let label = ui.label(tr("Max project size on open"));
            ui.add(egui::DragValue::new(&mut p.project_open_limit_gib).range(1..=u32::MAX).suffix(" GiB"))
                .labelled_by(label.id)
                .on_hover_text(tr("Maximum decompressed project data. Increase this limit to open larger projects."));
            ui.end_row();

            #[cfg(not(target_arch = "wasm32"))]
            {
                let label = ui.label(tr("Max RAM while saving"));
                ui.horizontal(|ui| {
                    ui.add(egui::DragValue::new(&mut p.save_ram_percent).range(1..=100).suffix(" %")).labelled_by(label.id);
                    let budget_bytes = texturecraft_engine::formats::save_ram_budget(available_ram_bytes, p.save_ram_percent);
                    ui.label(format!("≈ {:.2} GiB", budget_bytes as f64 / (1_u64 << 30) as f64));
                });
                ui.end_row();
            }
        });
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.label(RichText::new(tr("Saving uses the other cores and at most this share of the free RAM. At least one core always works.")).small().color(tk.text_dim));
            ui.label(
                RichText::new(trf("0 turns autosave off. Copies go to {folder}", &[("folder", &crate::user_dir("autosave").display())]))
                    .small()
                    .color(tk.text_dim),
            );
            ui.add_space(8.0);
            ui.label(caption(tr("Folders")));
            form("settings_folders").show(ui, |ui| {
                ui.label(tr("Data folder"));
                ui.horizontal(|ui| {
                    let default = crate::data_folder("").display().to_string();
                    ui.add(egui::TextEdit::singleline(&mut p.data_folder).hint_text(default).desired_width(260.0));
                    if ui.button("...").clicked() {
                        pick_data_folder = true;
                    }
                });
                ui.end_row();
                for (label, dir) in [(tr("Smart materials"), crate::library_dir()), (tr("UI layouts"), crate::user_dir("layouts"))] {
                    ui.label(label);
                    ui.label(RichText::new(dir.display().to_string()).small().color(tk.text_dim));
                    ui.end_row();
                }
            });
            ui.label(
                RichText::new(tr(
                    "Autosaves, smart materials, presets, templates and UI layouts go to the data folder. A new folder applies on restart; files already saved stay where they are.",
                ))
                .small()
                .color(tk.text_dim),
            );
            ui.add_space(8.0);
        }
        ui.label(caption(tr("Keyboard")));
        if ui.button(tr("Edit keyboard shortcuts...")).on_hover_text(tr("Rebind any menu item, tool or command")).clicked() {
            app.ui.windows.shortcuts = true;
        }
    });
    app.ui.windows.settings = open;
    #[cfg(not(target_arch = "wasm32"))]
    if pick_data_folder {
        app.ask_file(FileDialog::folder().directory(crate::data_folder(&app.ui.prefs.data_folder)), Bound::App, |app, paths| {
            if let Some(d) = paths.first() {
                app.ui.prefs.data_folder = d.display().to_string();
            }
        });
    }
}

/// Browser File › Export Mesh: pick the download format
#[cfg(target_arch = "wasm32")]
fn export_mesh(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let formats = [
        ("glb", "glTF binary (.glb)", tr("Textures embedded; for game engines and most viewers")),
        ("usdz", "USDZ (.usdz)", tr("Textures packaged; for USD tools and AR Quick Look")),
        ("obj", "Wavefront OBJ (.obj)", tr("Geometry and UVs only")),
    ];
    if !app.ui.windows.export_mesh {
        return;
    }

    let mut open = true;
    let mut picked = None;
    dialog("export_mesh", tr("Export mesh"), &mut open).show(ctx, |ui| {
        ui.label(caption(tr("Download as")));
        for (extension, label, hint) in formats {
            if ui.add_sized(vec2(220.0, ui.spacing().interact_size.y), egui::Button::new(label)).on_hover_text(hint).clicked() {
                picked = Some(extension);
            }
        }
    });

    if let Some(extension) = picked {
        crate::web_files::export_mesh(app, extension);
        open = false;
    }
    app.ui.windows.export_mesh = open;
}

fn about(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let tk = Tokens::get(ctx);
    let mut open = app.ui.windows.about;
    dialog("about", tr("About TextureCraft"), &mut open).show(ctx, |ui| {
        ui.vertical_centered(|ui| {
            ui.label(RichText::new("TextureCraft").size(22.0).strong().color(tk.text_strong));
            ui.label(trf("Version {version}", &[("version", &texturecraft_engine::VERSION)]));
            ui.add_space(6.0);
            ui.label(tr("3D texture painting: layers, masks, smart materials, mesh map baking and export."));
            ui.label(tr("Free and open source (MIT OR Apache-2.0), written in Rust."));
            ui.add_space(6.0);
            ui.hyperlink_to(tr("Join the ArtCraft community on Discord"), "https://discord.gg/artcraft");
            ui.hyperlink_to("getartcraft.com/apps/texturecraft", "https://getartcraft.com/apps/texturecraft");
            ui.add_space(6.0);
            ui.label(RichText::new(tr("Made by the ArtCraft team and community.")).color(tk.text_dim));
        });
    });
    app.ui.windows.about = open;
}

/// The Log panel body (a dock tab)
pub(crate) fn log(app: &mut TextureCraftApp, ui: &mut Ui) {
    egui::ScrollArea::vertical().auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
        for line in &app.session.log {
            ui.label(RichText::new(line).monospace());
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Files (wired to texturecraft-formats once it lands)

pub fn file_dialogs_available() -> bool {
    cfg!(not(target_arch = "wasm32"))
}

#[cfg(not(target_arch = "wasm32"))]
fn ask_mesh_file(app: &mut TextureCraftApp) {
    let dialog = FileDialog::open().filter(tr("3D meshes"), &texturecraft_engine::formats::mesh_import::MESH_EXTENSIONS);
    app.ask_file(dialog, Bound::App, |app, paths| {
        if let Some(p) = paths.first() {
            app.ui.new_project.file = Some(p.display().to_string());
        }
    });
}

#[cfg(target_arch = "wasm32")]
fn ask_mesh_file(_app: &mut TextureCraftApp) {}

/// Remember a project path in File ▸ Recent projects.
pub fn remember(app: &mut TextureCraftApp, path: &str) {
    app.ui.recent.retain(|p| p != path);
    app.ui.recent.insert(0, path.to_string());
    app.ui.recent.truncate(10);
}

/// Window › Save / Load UI layout: panels, windows and view settings as a JSON file.
#[cfg(not(target_arch = "wasm32"))]
pub fn layout_file(app: &mut TextureCraftApp, save: bool) {
    let dir = crate::user_dir("layouts");
    let dialog = if save { FileDialog::save().file_name("layout.json") } else { FileDialog::open() };
    app.ask_file(dialog.filter(tr("UI layout"), &["json"]).directory(dir.clone()), Bound::App, move |app, paths| {
        let Some(path) = paths.first() else { return };
        let result = if save {
            let layout = crate::UiState { recent: Vec::new(), ..app.ui.clone() };
            std::fs::create_dir_all(&dir)
                .map_err(|e| e.to_string())
                .and_then(|_| serde_json::to_vec_pretty(&layout).map_err(|e| e.to_string()))
                .and_then(|bytes| std::fs::write(path, bytes).map_err(|e| e.to_string()))
                .map(|_| trf("Saved UI layout {path}", &[("path", &path.display())]))
        } else {
            std::fs::read(path).map_err(|e| e.to_string()).and_then(|bytes| serde_json::from_slice::<crate::UiState>(&bytes).map_err(|e| e.to_string())).map(
                |layout| {
                    app.ui.adopt_layout(layout);
                    trf("Loaded UI layout {path}", &[("path", &path.display())])
                },
            )
        };
        match result {
            Ok(msg) => app.toast(msg, false),
            Err(e) => app.toast(trf("UI layout: {error}", &[("error", &e)]), true),
        }
    });
}

/// The browser has no file paths; layouts persist with the rest of the UI state.
#[cfg(target_arch = "wasm32")]
pub fn layout_file(app: &mut TextureCraftApp, _save: bool) {
    app.toast(tr("Saving UI layouts to files is available in the desktop app"), false);
}

/// File menu actions that need dialogs. Returns true when handled.
#[cfg(not(target_arch = "wasm32"))]
pub fn file_action(app: &mut TextureCraftApp, _ctx: &egui::Context, id: &str) -> bool {
    match id {
        "file.open" => {
            app.ask_file(FileDialog::open().filter(tr("TextureCraft project or mesh"), crate::web_files::OPEN_EXTS), Bound::App, |app, paths| {
                if let Some(p) = paths.first() {
                    open_path(app, &p.display().to_string());
                }
            });
            true
        }
        "file.save" if app.session.path.is_some() => {
            if let Some(r) = app.run("file.save", json!({})) {
                let p = r.get("path").and_then(|p| p.as_str()).unwrap_or_default().to_string();
                app.toast(trf("Saved {path}", &[("path", &p)]), false);
            }
            true
        }
        "file.save" | "file.saveAs" => {
            ask_save_project(app, "file.saveAs");
            true
        }
        "file.saveCopy" => {
            ask_save_project(app, "file.saveCopy");
            true
        }
        "material.load" => {
            app.ask_file(FileDialog::open_many().filter(tr("Materials"), crate::web_files::MATERIAL_EXTS), Bound::Project, |app, paths| {
                for p in paths {
                    app.run("material.load", json!({"path": p.display().to_string()}));
                }
            });
            true
        }
        "file.importResources" | "resource.import" => {
            app.ask_file(FileDialog::open_many().filter(tr("Images"), crate::web_files::IMAGE_EXTS), Bound::Project, |app, paths| {
                if paths.is_empty() {
                    return;
                }
                for p in paths {
                    if let Some(r) = app.run("resource.import", json!({"path": p.display().to_string()})) {
                        let name = r.get("name").and_then(|n| n.as_str()).unwrap_or_default().to_string();
                        app.toast(trf("Imported {name}", &[("name", &name)]), false);
                    }
                }
                app.ui.dock.focus(crate::docking::Panel::Assets);
            });
            true
        }
        "file.importMaterialX" => {
            app.ask_file(FileDialog::open().filter(tr("MaterialX materials"), &["mtlx"]), Bound::Project, |app, paths| {
                if let Some(p) = paths.first()
                    && let Some(r) = app.run("file.importMaterialX", json!({"path": p.display().to_string()}))
                {
                    let name = r.get("material").and_then(|n| n.as_str()).unwrap_or_default().to_string();
                    for w in r.get("warnings").and_then(|w| w.as_array()).into_iter().flatten().filter_map(|w| w.as_str()) {
                        app.session.log.push(format!("MaterialX {name}: {w}"));
                    }
                    app.toast(trf("Imported {name}", &[("name", &name)]), false);
                    app.ui.dock.focus(crate::docking::Panel::Layers);
                }
            });
            true
        }
        "file.exportMesh" => {
            let name = format!("{}.obj", texturecraft_engine::formats::export::sanitize_file_stem(&app.session.doc.project.name));
            let dialog = FileDialog::save()
                .filter(tr("Wavefront OBJ with MTL, textures alongside"), &["obj"])
                .filter(tr("glTF binary with textures"), &["glb"])
                .filter(tr("USDZ with textures"), &["usdz"])
                .filter(tr("USD text, textures alongside"), &["usda"])
                .filter(tr("USD binary, textures alongside"), &["usdc"])
                .file_name(name);
            app.ask_file(dialog, Bound::Project, |app, paths| {
                if let Some(p) = paths.first()
                    && let Some(r) = app.run("file.exportMesh", json!({"path": p.display().to_string()}))
                {
                    let p = r.get("path").and_then(|p| p.as_str()).unwrap_or_default().to_string();
                    app.toast(trf("Exported mesh to {path}", &[("path", &p)]), false);
                }
            });
            true
        }
        "file.reimportMesh" => {
            // The file the mesh came from is read again from disk; ask only when it's gone.
            if let Some(path) = app.session.remembered_mesh_path().map(str::to_string) {
                reimport_mesh(app, path);
                return true;
            }
            let dialog = FileDialog::open().filter(tr("3D meshes"), &texturecraft_engine::formats::mesh_import::MESH_EXTENSIONS);
            app.ask_file(dialog, Bound::Project, |app, paths| {
                if let Some(p) = paths.first() {
                    reimport_mesh(app, p.display().to_string());
                }
            });
            true
        }
        _ => false,
    }
}

/// Asks where to save the project and runs `cmd` with the answer. Each caller is its own dialog
/// request, so Save As and Save a Copy queue separately.
#[cfg(not(target_arch = "wasm32"))]
#[track_caller]
fn ask_save_project(app: &mut TextureCraftApp, cmd: &'static str) {
    let name = crate::web_files::project_file_name(app);
    let dialog = FileDialog::save().filter(tr("TextureCraft project"), &[texturecraft_engine::core::PROJECT_EXT]).file_name(name);
    // Save a Copy leaves the project unsaved, so it never answers the unsaved-changes prompt.
    let for_prompt = app.saving_for_prompt && cmd == "file.saveAs";
    let asked = app.ask_file(dialog, Bound::Project, move |app, paths| {
        let mut saved = false;
        if let Some(p) = paths.first() {
            let path = p.display().to_string();
            saved = app.run(cmd, json!({"path": path})).is_some();
            if saved {
                remember(app, &path);
                app.toast(trf("Saved {path}", &[("path", &path)]), false);
            }
        }
        // The answer ends the wait. A canceled or failed save leaves the prompt open for an answer.
        if for_prompt {
            app.saving_for_prompt = false;
            app.prompt_saved = saved && !app.session.modified;
        }
    });
    // A Save As already on its way answers for itself: the prompt's buttons stay usable.
    if for_prompt && !asked {
        app.saving_for_prompt = false;
    }
}

/// Reimport the mesh; painted strokes follow the surface when the new mesh's UVs differ.
#[cfg(not(target_arch = "wasm32"))]
fn reimport_mesh(app: &mut TextureCraftApp, path: String) {
    app.run("file.reimportMesh", json!({"path": path, "preserveStrokes": true}));
}

/// Open a project or mesh by path (menus, Recent projects, drag and drop).
#[cfg(not(target_arch = "wasm32"))]
pub fn open_path(app: &mut TextureCraftApp, path: &str) {
    #[cfg(not(target_arch = "wasm32"))]
    if ask_to_save(app, crate::state::Unsaved::Open(path.into())) {
        return;
    }
    if app.run("file.open", json!({"path": path})).is_some() {
        remember(app, path);
        app.ui.mode = Mode::Painting;
    }
}

/// In the browser, Open and Import use the file picker and Save downloads the project.
#[cfg(target_arch = "wasm32")]
pub fn file_action(app: &mut TextureCraftApp, ctx: &egui::Context, id: &str) -> bool {
    match id {
        "file.open" => crate::web_files::open(ctx),
        "file.importResources" | "resource.import" => crate::web_files::import_images(ctx),
        "material.load" => crate::web_files::import_material(ctx),
        "file.save" | "file.saveAs" | "file.saveCopy" => crate::web_files::save_project(app),
        "file.exportMesh" => app.ui.windows.export_mesh = true,
        _ => return false,
    }
    true
}

/// Texture sets ticked in the export dialog.
fn export_selection(app: &TextureCraftApp) -> Vec<usize> {
    (0..app.session.set_count()).filter(|i| !app.ui.export.skip.contains(i)).collect()
}

/// Export the selected texture sets; false when a command failed (its error is already shown).
#[cfg(not(target_arch = "wasm32"))]
fn export_sets(app: &mut TextureCraftApp) -> bool {
    let mut written = 0usize;
    for ts in export_selection(app) {
        let mut p = export_params(app);
        p["dir"] = json!(app.ui.export.dir);
        p["ts"] = json!(ts);
        match app.run("file.exportTextures", p) {
            Some(r) => written += r.get("files").and_then(|f| f.as_array()).map(Vec::len).unwrap_or(0),
            None => return false,
        }
    }
    app.toast(trf("Exported {n} textures to {folder}", &[("n", &written), ("folder", &app.ui.export.dir)]), false);
    true
}

/// In the browser the textures download (one zip for several files).
#[cfg(target_arch = "wasm32")]
fn export_sets(app: &mut TextureCraftApp) -> bool {
    let sets = export_selection(app);
    crate::web_files::export_textures(app, &export_params(app), &sets);
    true
}

pub(crate) fn export_params(app: &TextureCraftApp) -> serde_json::Value {
    let e = &app.ui.export;
    let mut p = json!({"preset": e.preset, "padding": e.padding, "dilation": e.dilation});
    if !e.format.is_empty() {
        p["format"] = json!(e.format);
    }
    if e.bit_depth > 0 {
        p["bitDepth"] = json!(e.bit_depth);
    }
    if e.size > 0 {
        p["size"] = json!(e.size);
    }
    if !e.normal_format.is_empty() {
        p["normalFormat"] = json!(e.normal_format);
    }
    if !e.color_space.is_empty() {
        p["colorSpace"] = json!(e.color_space);
    }
    p
}

fn export_settings_tab(app: &mut TextureCraftApp, ui: &mut Ui, presets: &[texturecraft_engine::formats::ExportPreset]) {
    ui.label(caption(tr("General export parameters")));
    egui::Grid::new("exp_grid").num_columns(2).spacing(vec2(12.0, 8.0)).show(ui, |ui| {
        // The browser downloads the files instead.
        #[cfg(not(target_arch = "wasm32"))]
        {
            ui.label(tr("Output directory"));
            ui.horizontal(|ui| {
                ui.add(egui::TextEdit::singleline(&mut app.ui.export.dir).desired_width(360.0));
                if ui.button("...").clicked() {
                    app.ask_file(FileDialog::folder().directory(app.ui.export.dir.clone()), Bound::App, |app, paths| {
                        if let Some(d) = paths.first() {
                            app.ui.export.dir = d.display().to_string();
                        }
                    });
                }
            });
            ui.end_row();
        }
        ui.label(tr("Output template"));
        egui::ComboBox::from_id_salt("exp_preset").width(360.0).selected_text(app.ui.export.preset.clone()).show_ui(ui, |ui| {
            for p in presets {
                ui.selectable_value(&mut app.ui.export.preset, p.name.clone(), &p.name);
            }
            if !app.session.export_presets.list.is_empty() {
                ui.separator();
            }
            for p in &app.session.export_presets.list {
                ui.selectable_value(&mut app.ui.export.preset, p.name.clone(), &p.name);
            }
        });
        ui.end_row();
        ui.label(tr("File type"));
        ui.horizontal(|ui| {
            let label = if app.ui.export.format.is_empty() { tr("From template").to_string() } else { app.ui.export.format.clone() };
            egui::ComboBox::from_id_salt("exp_fmt").selected_text(label).show_ui(ui, |ui| {
                ui.selectable_value(&mut app.ui.export.format, String::new(), tr("From template"));
                for f in ["png", "jpeg", "tga", "tiff", "bmp", "webp", "exr", "hdr"] {
                    ui.selectable_value(&mut app.ui.export.format, f.to_string(), f);
                }
            });
            let bd = if app.ui.export.bit_depth == 0 { tr("Default").to_string() } else { trf("{n} bits", &[("n", &app.ui.export.bit_depth)]) };
            egui::ComboBox::from_id_salt("exp_bits").selected_text(bd).show_ui(ui, |ui| {
                for (v, l) in [
                    (0u8, tr("Default").to_string()),
                    (8, trf("{n} bits", &[("n", &8)])),
                    (16, trf("{n} bits", &[("n", &16)])),
                    (32, tr("32 bits (float)").to_string()),
                ] {
                    ui.selectable_value(&mut app.ui.export.bit_depth, v, l);
                }
            });
        });
        ui.end_row();
        ui.label(tr("Size"));
        let sz = if app.ui.export.size == 0 { tr("Based on each texture set's size").to_string() } else { format!("{0} x {0}", app.ui.export.size) };
        egui::ComboBox::from_id_salt("exp_size").width(360.0).selected_text(sz).show_ui(ui, |ui| {
            ui.selectable_value(&mut app.ui.export.size, 0, tr("Based on each texture set's size"));
            for r in [256u32, 512, 1024, 2048, 4096, 8192] {
                ui.selectable_value(&mut app.ui.export.size, r, format!("{r} x {r}"));
            }
        });
        ui.end_row();
        ui.label(tr("Padding"));
        ui.horizontal(|ui| {
            let label = match app.ui.export.padding.as_str() {
                "none" => tr("No padding (passthrough)"),
                "dilation" => tr("Dilation + transparent"),
                "dilationDefault" => tr("Dilation + default background color"),
                _ => tr("Dilation infinite"),
            };
            egui::ComboBox::from_id_salt("exp_pad").width(260.0).selected_text(label).show_ui(ui, |ui| {
                for (v, l) in [
                    ("infinite", tr("Dilation infinite")),
                    ("dilation", tr("Dilation + transparent")),
                    ("dilationDefault", tr("Dilation + default background color")),
                    ("none", tr("No padding (passthrough)")),
                ] {
                    ui.selectable_value(&mut app.ui.export.padding, v.to_string(), l);
                }
            });
            if app.ui.export.padding.starts_with("dilation") {
                ui.add(egui::DragValue::new(&mut app.ui.export.dilation).range(0..=256).suffix(" px"));
            }
        });
        ui.end_row();
        ui.label(tr("Normal map format"));
        let nf = match app.ui.export.normal_format.as_str() {
            "openGl" => "OpenGL",
            "directX" => "DirectX",
            _ => tr("From template / project"),
        };
        egui::ComboBox::from_id_salt("exp_nf").selected_text(nf).show_ui(ui, |ui| {
            for (v, l) in [("", tr("From template / project")), ("openGl", "OpenGL"), ("directX", "DirectX")] {
                ui.selectable_value(&mut app.ui.export.normal_format, v.to_string(), l);
            }
        });
        ui.end_row();
        ui.label(tr("Color space"));
        let shown = if app.ui.export.color_space.is_empty() { tr("Default") } else { app.ui.export.color_space.as_str() };
        egui::ComboBox::from_id_salt("exp_cs").selected_text(shown).show_ui(ui, |ui| {
            ui.selectable_value(&mut app.ui.export.color_space, String::new(), tr("Default"));
            for name in ["sRGB", "Linear Rec.709", "ACEScg", "ACES2065-1"] {
                ui.selectable_value(&mut app.ui.export.color_space, name.to_string(), name);
            }
        });
        ui.end_row();
    });
}

pub fn export(app: &mut TextureCraftApp, ctx: &egui::Context) {
    let tk = Tokens::get(ctx);
    if !app.ui.windows.export {
        return;
    }
    let mut open = true;
    let mut do_export = false;
    let presets = texturecraft_engine::formats::builtin_presets();
    dialog("export_textures", tr("Export textures"), &mut open).fixed_size(vec2(820.0, 430.0)).show(ctx, |ui| {
        ui.horizontal(|ui| {
            for (i, t) in [tr("Settings"), tr("Output templates"), tr("List of exports")].iter().enumerate() {
                if ui.selectable_label(app.ui.export.tab == i as u8, caption(t)).clicked() {
                    app.ui.export.tab = i as u8;
                }
            }
        });
        ui.separator();
        ui.allocate_ui_with_layout(vec2(ui.available_width(), 360.0), egui::Layout::left_to_right(egui::Align::Min), |ui| {
            ui.set_height(360.0);
            if app.ui.export.tab == 1 {
                crate::export_templates::tab(app, ui, &presets);
                return;
            }
            ui.vertical(|ui| {
                ui.set_width(180.0);
                ui.label(RichText::new(tr("Texture sets")).strong());
                for i in 0..app.session.set_count() {
                    let name = app.session.doc.project.texture_sets.get(i).map(|t| t.name.clone()).unwrap_or_default();
                    let mut on = !app.ui.export.skip.contains(&i);
                    if ui.checkbox(&mut on, name).changed() {
                        if on {
                            app.ui.export.skip.retain(|x| *x != i);
                        } else {
                            app.ui.export.skip.push(i);
                        }
                    }
                }
            });
            ui.separator();
            ui.vertical(|ui| match app.ui.export.tab {
                0 => export_settings_tab(app, ui, &presets),
                _ => {
                    let mut params = export_params(app);
                    params["ts"] = json!("all");
                    match texturecraft_engine::execute(&mut app.session, "export.plan", &params) {
                        Ok(v) => {
                            egui::ScrollArea::vertical().max_height(340.0).show(ui, |ui| {
                                for f in v.as_array().into_iter().flatten() {
                                    let ts = f.get("ts").and_then(|t| t.as_u64()).unwrap_or(0) as usize;
                                    if !app.ui.export.skip.contains(&ts) {
                                        ui.label(RichText::new(f.get("file").and_then(|x| x.as_str()).unwrap_or_default()).monospace());
                                    }
                                }
                            });
                        }
                        Err(e) => {
                            ui.label(RichText::new(e.to_string()).color(tk.danger));
                        }
                    }
                }
            });
        });
        ui.add_space(8.0);
        ui.separator();
        ui.horizontal(|ui| {
            if ui.button(tr("Reset all")).clicked() {
                app.ui.export = crate::state::ExportUi::default();
            }
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(primary_button(ui, tr("Export"))).clicked() {
                    do_export = true;
                }
                if ui.button(tr("Cancel")).clicked() {
                    app.ui.windows.export = false;
                }
            });
        });
    });
    if do_export && export_sets(app) {
        app.ui.windows.export = false;
    }
    if !open {
        app.ui.windows.export = false;
    }
}

#[cfg(test)]
mod tests {
    use egui_kittest::Harness;
    use egui_kittest::kittest::{NodeT as _, Queryable as _};

    use super::*;
    use crate::state::Unsaved;

    fn harness() -> Harness<'static, TextureCraftApp> {
        let session = texturecraft_engine::sample_session(256, 64).unwrap();
        let mut h = Harness::builder().with_size(vec2(1600.0, 960.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
        h.run_steps(2);
        h
    }

    #[test]
    fn a_canceled_save_dialog_does_not_answer_the_prompt() {
        let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut h = harness();
        let app = h.state_mut();
        app.file_dialogs.set_override(Some(Box::new(|_| crate::file_dialogs::Answer::Cancelled)));
        app.ui.windows.unsaved = Some(Unsaved::Quit);
        // Saved elsewhere, by a script, while the dialog was open.
        app.session.modified = false;
        app.saving_for_prompt = true;
        ask_save_project(app, "file.saveAs");
        h.run_steps(3);
        assert_eq!(h.state().ui.windows.unsaved, Some(Unsaved::Quit), "the prompt still waits for an answer");
    }

    #[test]
    fn the_prompts_buttons_wait_for_its_save_dialog() {
        let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut h = harness();
        let app = h.state_mut();
        app.ui.windows.unsaved = Some(Unsaved::Quit);
        app.session.modified = true;
        h.run_steps(2);
        for label in ["Save", "Don't save", "Cancel"] {
            assert!(!h.get_by_label(label).accesskit_node().is_disabled(), "{label} answers while no dialog is open");
        }
        h.state_mut().saving_for_prompt = true;
        h.run_steps(2);
        for label in ["Save", "Don't save", "Cancel"] {
            assert!(h.get_by_label(label).accesskit_node().is_disabled(), "{label} waits for the Save dialog");
        }
    }

    #[test]
    fn a_save_a_copy_answer_does_not_end_the_prompts_wait() {
        let mut h = harness();
        let app = h.state_mut();
        app.file_dialogs.set_override(Some(Box::new(|_| crate::file_dialogs::Answer::Cancelled)));
        app.saving_for_prompt = true;
        ask_save_project(app, "file.saveCopy");
        h.run_steps(3);
        assert!(h.state().saving_for_prompt, "only the prompt's own Save As ends the wait");
    }

    #[test]
    fn a_save_as_already_on_its_way_ends_the_prompts_wait() {
        let mut h = harness();
        let app = h.state_mut();
        // One call site, as File › Save As and the prompt's Save share one.
        for waiting in [false, true] {
            app.saving_for_prompt = waiting;
            ask_save_project(app, "file.saveAs");
        }
        assert_eq!(app.file_dialogs.pending(), 1);
        assert!(!app.saving_for_prompt, "the prompt's buttons work again");
    }

    /// A change that lands after the Save dialog saved (a bake finishing that frame) keeps the
    /// question open with buttons that still answer.
    #[test]
    fn a_change_after_the_dialog_saved_leaves_the_prompt_usable() {
        // The harness keeps the process-wide language at English while this asserts on labels.
        let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut h = harness();
        let app = h.state_mut();
        app.ui.windows.unsaved = Some(Unsaved::Quit);
        app.session.modified = true;
        app.prompt_saved = true;
        h.run_steps(2);
        assert_eq!(h.state().ui.windows.unsaved, Some(Unsaved::Quit), "the project has unsaved changes again");
        assert!(!h.state().saving_for_prompt);
        h.get_by_label("Cancel").click();
        h.run_steps(2);
        assert!(h.state().ui.windows.unsaved.is_none());
    }
}
