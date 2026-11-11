//! The asset shelf (the Assets dock tab): tabs (materials, smart materials, smart
//! masks, filters, alphas, procedurals, environments, textures), all built in and procedural.
//! Search filters across every tab. Materials (built in and saved), smart masks, filters, alphas
//! and textures drag onto the layer stack, a layer row, the viewport or the brush properties
//! ([`ShelfDrag`]); a texture dropped on the 3D view becomes a decal there, in the channel picked from a menu

use egui::{Color32, RichText, Sense, Stroke, Ui, vec2};
use egui_phosphor::regular as ic;
use serde_json::json;
use texturecraft_engine::core::ProceduralKind;
use texturecraft_engine::filter_lib::FILTERS;
#[cfg(not(target_arch = "wasm32"))]
use texturecraft_engine::formats::mcmat::MCMAT_EXT;
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::materials::MATERIALS;
use texturecraft_engine::orb_preview::MAX_PREVIEW_SIZE_PX;
use texturecraft_engine::paint::alphas;
use texturecraft_engine::paint::presets::{BrushPreset, PRESETS};
use texturecraft_engine::smart_masks::SMART_MASKS;
use texturecraft_render::ibl::ENVIRONMENTS;

use crate::TextureCraftApp;
use crate::decal_drop::DecalDrop;
use crate::i18n::{tr, trf};
use crate::shelf_preview::ShelfPreview;
use crate::theme::Tokens;
use crate::thumbs::Thumbs;
use crate::widgets::{dnd_release, icon_button};

/// A shelf tile's size in points
const TILE_WIDTH: f32 = 72.0;
const TILE_HEIGHT: f32 = 86.0;
/// Corner radius of a tile's hover and selection highlights, in points
const TILE_ROUNDING: f32 = 3.0;
/// The thumbnail's size in a tile and its center below the tile's top, in points
const THUMBNAIL_SIZE: f32 = 56.0;
const THUMBNAIL_CENTER_Y: f32 = 32.0;
/// Loading icon size relative to the thumbnail width, and its opacity over the dim text color
const LOADING_ICON_SCALE: f32 = 0.9;
const LOADING_ICON_OPACITY: f32 = 0.5;
/// A tile label's font size, its gap below the thumbnail and its total side margin, in points
const LABEL_FONT_SIZE: f32 = 9.5;
const LABEL_GAP: f32 = 3.0;
const LABEL_MARGIN: f32 = 4.0;
/// The hover preview's size and its gap above the label, in points
const BIG_PREVIEW_SIZE: f32 = 256.0;
const BIG_PREVIEW_LABEL_GAP: f32 = 4.0;
/// Dwell in seconds before rendering, shorter than the popup delay
const BIG_PREVIEW_DWELL: f32 = 0.1;

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Tab {
    Materials,
    Smart,
    Masks,
    Filters,
    Brushes,
    Alphas,
    Procedurals,
    Environments,
    Textures,
}

/// The shelf tabs: icon and (English, translated where shown) name.
pub(crate) const TABS: [(Tab, &str, &str); 9] = [
    (Tab::Materials, ic::SPHERE, "Materials"),
    (Tab::Smart, ic::STACK, "Smart materials"),
    (Tab::Masks, ic::CIRCLE_HALF, "Smart masks"),
    (Tab::Filters, ic::FUNNEL, "Filters"),
    (Tab::Brushes, ic::PAINT_BRUSH_BROAD, "Brushes"),
    (Tab::Alphas, ic::SHAPES, "Alphas"),
    (Tab::Procedurals, ic::CHECKERBOARD, "Procedurals"),
    (Tab::Environments, ic::SUN_HORIZON, "Environments"),
    (Tab::Textures, ic::IMAGE, "Textures"),
];

/// A shelf asset: built in (by id), a saved smart material (file path) or an image resource.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Asset {
    Material(&'static str),
    SmartMask(&'static str),
    Filter(&'static str),
    Alpha(&'static str),
    Library(String),
    Texture(String),
}

impl Asset {
    /// Assets that add layers (built-in and saved smart materials).
    pub fn is_material(&self) -> bool {
        matches!(self, Asset::Material(_) | Asset::Library(_))
    }

    /// Assets a 3D view drop puts on the texture set under the pointer
    pub fn goes_onto_set(&self) -> bool {
        self.is_material() || matches!(self, Asset::Texture(_))
    }
}

/// The drag-and-drop payload of a shelf tile.
#[derive(Clone, Debug)]
pub struct ShelfDrag {
    pub asset: Asset,
    pub label: String,
}

/// Apply `asset` the way a shelf click does: a material above the selection, a smart mask on
/// the selected layer, a filter on the selection (its mask when the mask is selected), an alpha
/// on the brush, a texture as an image fill.
pub fn apply(app: &mut TextureCraftApp, asset: &Asset) {
    let (cmd, params) = match asset {
        Asset::Material(id) => ("material.apply", json!({"material": id})),
        Asset::SmartMask(id) => ("mask.smart.apply", json!({"name": id})),
        Asset::Filter(id) => {
            let content = crate::dock::targets_content(app);
            if app.run("effect.add.filter", json!({"filter": id, "target": if content { "content" } else { "mask" }})).is_some()
                && content
                && let Some(name) = unpainted_paint_layer(app)
            {
                // Otherwise the filter seems to do nothing: it has nothing to change yet.
                app.toast(trf("\"{layer}\" has no paint yet: the filter changes what you paint on it.", &[("layer", &name)]), false);
            }
            return;
        }
        Asset::Alpha(id) => ("tool.brush", json!({"alpha": {"library": id}})),
        Asset::Library(path) => ("material.load", json!({"path": path})),
        Asset::Texture(name) => ("layer.add.image", json!({"name": name})),
    };
    app.run(cmd, params);
}

/// The selected layer's name when it is a paint layer with nothing painted (in itself or in its
/// content paint effects).
fn unpainted_paint_layer(app: &TextureCraftApp) -> Option<String> {
    use texturecraft_engine::core::{EffectKind, LayerKind};
    let s = &app.session;
    let layer = s.selected_layer().and_then(|id| s.doc.project.content_layer(id))?;
    if !matches!(layer.kind, LayerKind::Paint) {
        return None;
    }
    let ids: Vec<_> = std::iter::once(layer.id).chain(layer.content_effects.iter().filter(|e| matches!(e.kind, EffectKind::Paint)).map(|e| e.id)).collect();
    let painted = s.doc.pixels.layers.iter().any(|((id, _), img)| ids.contains(id) && img.allocated_count() > 0);
    (!painted).then(|| layer.name.clone())
}

/// What a view accepts from the shelf.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum DropZone {
    /// The brush properties: alphas.
    Brush,
    /// The 2D view: alphas and materials.
    View2d,
    /// The 3D view: alphas, materials (on the texture set they drop on) and textures (decals where they drop, on that set)
    View3d,
}

/// The texture set under the pointer while a shelf material or image is dragged over the 3D view `response`: the set a drop there goes to
pub fn drop_set(app: &TextureCraftApp, ui: &Ui, response: &egui::Response) -> Option<usize> {
    if app.ui.mode != crate::state::Mode::Painting {
        return None;
    }

    response.dnd_hover_payload::<ShelfDrag>().filter(|drag| drag.asset.goes_onto_set())?;
    set_under_pointer(app, ui, response)
}

/// The texture set of the surface under the pointer in the 3D view `response`
fn set_under_pointer(app: &TextureCraftApp, ui: &Ui, response: &egui::Response) -> Option<usize> {
    let local_position = ui.input(|input| input.pointer.interact_pos())? - response.rect.min;
    app.session.texture_set_at(Vec2::new(local_position.x, local_position.y), app.session.viewport)
}

/// Highlight accepted shelf assets on hover and apply them on drop. In the 3D view a material or image goes to the texture set under the pointer, highlighted by the view itself ([`drop_set`])
pub fn drop_target(app: &mut TextureCraftApp, ui: &Ui, response: &egui::Response, zone: DropZone) {
    if app.ui.mode != crate::state::Mode::Painting {
        return;
    }

    if zone == DropZone::View3d {
        DecalDrop::show_menu(app, ui, response);
    }

    let tokens = Tokens::get(ui.ctx());
    let accepts = |drag: &ShelfDrag| match drag.asset {
        Asset::Alpha(_) => true,
        Asset::Material(_) | Asset::Library(_) => zone != DropZone::Brush,
        Asset::Texture(_) => zone == DropZone::View3d,
        Asset::SmartMask(_) | Asset::Filter(_) => false,
    };
    let onto_set = |asset: &Asset| zone == DropZone::View3d && asset.goes_onto_set();
    if response.dnd_hover_payload::<ShelfDrag>().is_some_and(|drag| accepts(&drag) && !onto_set(&drag.asset)) {
        ui.painter().rect_stroke(response.rect.shrink(1.0), 3.0, Stroke::new(2.0, tokens.accent), egui::StrokeKind::Inside);
    }

    let pointer = ui.input(|input| input.pointer.interact_pos());
    let Some(drag) = dnd_release::<ShelfDrag>(response).filter(|drag| accepts(drag)) else { return };

    match (&drag.asset, pointer) {
        // The background takes nothing
        (Asset::Texture(name), Some(position)) => {
            let Some(texture_set) = set_under_pointer(app, ui, response) else { return };

            let drop = DecalDrop {
                image: name.clone(),
                texture_set,
                view_position: position - response.rect.min,
                view_size: response.rect.size(),
                camera: app.session.camera,
                screen_position: position,
            };
            drop.open(ui);
        }
        (asset, _) if onto_set(asset) => {
            let Some(ts) = set_under_pointer(app, ui, response) else { return };

            if app.run("textureSet.select", json!({"ts": ts})).is_some() {
                apply(app, asset);
            }
        }
        (asset, _) => apply(app, asset),
    }
}

/// The Assets panel body (a dock tab)
pub fn body(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tokens = Tokens::get(ui.ctx());
    let ctx = ui.ctx().clone();
    let mut tab = app.ui.shelf_tab.min(TABS.len() - 1);
    let searching = !app.ui.shelf_filter.is_empty();

    ui.horizontal_wrapped(|ui| {
        let button_size = ui.spacing().interact_size.y;
        ui.spacing_mut().item_spacing = vec2(2.0, 2.0);
        for (i, (_, icon, label)) in TABS.iter().enumerate() {
            if icon_button(ui, icon, tr(label), i == tab && !searching, button_size).clicked() {
                tab = i;
                app.ui.shelf_filter.clear();
            }
        }
    });
    app.ui.shelf_tab = tab;

    ui.horizontal(|ui| {
        ui.label(RichText::new(ic::MAGNIFYING_GLASS).color(tokens.text_dim));
        ui.add(egui::TextEdit::singleline(&mut app.ui.shelf_filter).hint_text(tr("Search all assets")).desired_width(f32::INFINITY));
    });
    ui.add_space(4.0);

    let filter = app.ui.shelf_filter.to_lowercase();
    egui::ScrollArea::vertical().id_salt("shelf_scroll").auto_shrink([false, false]).show(ui, |ui| {
        // Searching shows matches from every tab, each under its tab's name
        for (i, (t, _, title)) in TABS.iter().enumerate() {
            if filter.is_empty() && i != tab {
                continue;
            }

            let title = tr(title);
            match t {
                Tab::Materials => materials(app, ui, &ctx, &filter, false, title),
                Tab::Smart => {
                    materials(app, ui, &ctx, &filter, true, title);
                    library(app, ui, &filter);
                }
                Tab::Masks => smart_masks(app, ui, &ctx, &filter, title),
                Tab::Filters => filters(app, ui, &ctx, &filter, title),
                Tab::Brushes => brushes(app, ui, &ctx, &filter, title),
                Tab::Alphas => alpha_library(app, ui, &ctx, &filter, title),
                Tab::Procedurals => procedurals(app, ui, &ctx, &filter, title),
                Tab::Environments => environments(app, ui, &filter, title),
                Tab::Textures => textures(app, ui, &ctx, &filter, title),
            }
        }
    });
    drag_preview(&ctx);
}

/// The dragged asset's name next to the pointer.
fn drag_preview(ctx: &egui::Context) {
    let tk = Tokens::get(ctx);
    let (Some(d), Some(pos)) = (egui::DragAndDrop::payload::<ShelfDrag>(ctx), ctx.input(|i| i.pointer.hover_pos())) else { return };
    let p = ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::unique("shelf_drag")));
    let galley = p.layout_no_wrap(d.label.clone(), egui::FontId::proportional(11.0), tk.text_strong);
    let rect = egui::Rect::from_min_size(pos + vec2(14.0, 12.0), galley.size() + vec2(12.0, 6.0));
    p.rect_filled(rect, 3.0, tk.row_selected);
    p.rect_stroke(rect, 3.0, Stroke::new(1.0, tk.accent), egui::StrokeKind::Inside);
    p.galley(rect.min + vec2(6.0, 3.0), galley, tk.text_strong);
}

pub(crate) fn matches(filter: &str, texts: &[&str]) -> bool {
    filter.is_empty() || texts.iter().any(|t| t.to_lowercase().contains(filter))
}

fn title(ui: &mut Ui, text: &str) {
    let tk = Tokens::get(ui.ctx());
    ui.label(RichText::new(text).strong().color(tk.text_strong));
}

pub(crate) fn header(ui: &mut Ui, text: &str) {
    let tk = Tokens::get(ui.ctx());
    ui.add_space(2.0);
    ui.label(RichText::new(text).size(11.0).color(tk.text_dim));
}

pub(crate) fn tiles(ui: &mut Ui, body: impl FnOnce(&mut Ui)) {
    ui.horizontal_wrapped(|ui| {
        ui.spacing_mut().item_spacing = vec2(4.0, 4.0);
        body(ui);
    });
    ui.add_space(6.0);
}

/// The distinct categories of `items`, in order of first appearance.
pub(crate) fn categories<T>(items: &[T], category: impl Fn(&T) -> &'static str) -> Vec<&'static str> {
    let mut out: Vec<&'static str> = Vec::new();
    for c in items.iter().map(category) {
        if !out.contains(&c) {
            out.push(c);
        }
    }
    out
}

/// Built-in materials or smart materials, grouped by category
fn materials(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, smart: bool, name: &str) {
    let shown: Vec<_> =
        MATERIALS.iter().filter(|material| material.smart == smart && matches(filter, &[material.label, material.category, material.id])).collect();
    if shown.is_empty() {
        return;
    }

    title(ui, name);
    for category in categories(&shown, |material| material.category) {
        header(ui, category);
        tiles(ui, |ui| {
            for material in shown.iter().filter(|material| material.category == category) {
                let texture = app.thumbs.material(ctx, material.id);
                let asset = Asset::Material(material.id);
                if tile(ui, &mut app.thumbs, texture.as_ref(), || ShelfPreview::material(material.id), material.label, Some(&asset)).clicked() {
                    apply(app, &asset);
                }
            }
        });
    }
}

/// The user's saved smart materials (desktop library folder).
#[cfg(not(target_arch = "wasm32"))]
fn library(app: &mut TextureCraftApp, ui: &mut Ui, filter: &str) {
    let tk = Tokens::get(ui.ctx());
    if app.library.is_none() {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(crate::library_dir())
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == "tcmat" || x == MCMAT_EXT)).collect())
            .unwrap_or_default();
        files.sort();
        app.library = Some(files);
    }
    let files = app.library.clone().unwrap_or_default();
    let name = |f: &std::path::PathBuf| f.file_stem().and_then(|s| s.to_str()).unwrap_or("material").to_string();
    let shown: Vec<_> = files.iter().filter(|f| matches(filter, &[&name(f)])).collect();
    if !filter.is_empty() && shown.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        header(ui, tr("My smart materials"));
        if ui.small_button(ic::ARROW_CLOCKWISE).on_hover_text(tr("Rescan the library folder")).clicked() {
            app.library = None;
        }
    });
    if files.is_empty() {
        ui.label(RichText::new(tr("Right-click a layer > Create smart material")).size(10.0).color(tk.text_dim));
    }
    for f in shown {
        let (label, asset) = (name(f), Asset::Library(f.display().to_string()));
        // MaterialCraft materials (sent from MaterialCraft) show a graph.
        let glyph = if f.extension().is_some_and(|x| x == MCMAT_EXT) { ic::GRAPH } else { ic::SPHERE };
        let r = ui.add(egui::Button::selectable(false, format!("{glyph}  {label}")).sense(Sense::click_and_drag())).on_hover_ui(|ui| {
            ui.label(crate::i18n::trf(
                "{file}: add above the selected layer (or drag it into the layer stack or onto the viewport)",
                &[("file", &f.display())],
            ));
        });
        if r.drag_started() {
            r.dnd_set_drag_payload(ShelfDrag { asset: asset.clone(), label });
        }
        if r.clicked() {
            apply(app, &asset);
        }
    }
    ui.add_space(6.0);
}

#[cfg(target_arch = "wasm32")]
fn library(_: &mut TextureCraftApp, _: &mut Ui, _: &str) {}

fn smart_masks(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let shown: Vec<_> = SMART_MASKS.iter().filter(|m| matches(filter, &[m.label, m.id])).collect();
    if shown.is_empty() {
        return;
    }
    title(ui, name);
    tiles(ui, |ui| {
        for m in shown {
            let tex = app.thumbs.smart_mask(ctx, m.id);
            let asset = Asset::SmartMask(m.id);
            if tile(ui, &mut app.thumbs, tex.as_ref(), || ShelfPreview::SmartMask(m.id), m.label, Some(&asset)).clicked() {
                apply(app, &asset);
            }
        }
    });
}

/// The filter library, grouped: a click adds the filter to the selected layer's content (or its
/// mask when the mask is selected), a drop onto a layer to that layer's content.
fn filters(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let shown: Vec<_> = FILTERS.iter().filter(|f| matches(filter, &[f.label, f.group, f.id, tr(f.label), tr(f.group)])).collect();
    if shown.is_empty() {
        return;
    }
    title(ui, name);
    for group in categories(&shown, |f| f.group) {
        header(ui, tr(group));
        tiles(ui, |ui| {
            for f in shown.iter().filter(|f| f.group == group) {
                let tex = app.thumbs.filter(ctx, f.id);
                let asset = Asset::Filter(f.id);
                if tile(ui, &mut app.thumbs, tex.as_ref(), || ShelfPreview::Filter(f.id), tr(f.label), Some(&asset)).clicked() {
                    apply(app, &asset);
                }
            }
        });
    }
}

/// Brush presets, grouped by category, then the user's own: a click sets the brush's stroke
/// (alpha, spacing, flow, jitters…) and keeps its size.
fn brushes(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let shown: Vec<&BrushPreset> = PRESETS.iter().filter(|p| matches(filter, &[p.label, p.category, p.id])).collect();
    if !shown.is_empty() {
        title(ui, name);
        for cat in categories(&shown, |p| p.category) {
            header(ui, cat);
            tiles(ui, |ui| {
                for p in shown.iter().filter(|p| p.category == cat) {
                    let tex = app.thumbs.brush_preset(ctx, p);
                    if tile(ui, &mut app.thumbs, tex.as_ref(), || ShelfPreview::Brush(p), p.label, None).clicked() {
                        app.run("brush.preset.apply", json!({"preset": p.id}));
                    }
                }
            });
        }
    }
    my_brushes(app, ui, filter);
}

/// The user's saved brush presets (desktop brushes folder) and saving the current brush there.
#[cfg(not(target_arch = "wasm32"))]
fn my_brushes(app: &mut TextureCraftApp, ui: &mut Ui, filter: &str) {
    use texturecraft_engine::brush_preset_cmds::BRUSH_EXT;
    let dir = crate::user_dir("brushes");
    if app.brushes.is_none() {
        let mut files: Vec<std::path::PathBuf> = std::fs::read_dir(&dir)
            .map(|d| d.flatten().map(|e| e.path()).filter(|p| p.extension().is_some_and(|x| x == BRUSH_EXT)).collect())
            .unwrap_or_default();
        files.sort();
        app.brushes = Some(files);
    }
    let files = app.brushes.clone().unwrap_or_default();
    let name = |f: &std::path::PathBuf| f.file_stem().and_then(|s| s.to_str()).unwrap_or("brush").to_string();
    let shown: Vec<_> = files.iter().filter(|f| matches(filter, &[&name(f)])).collect();
    if !filter.is_empty() && shown.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        header(ui, tr("My brushes"));
        if ui.small_button(ic::ARROW_CLOCKWISE).on_hover_text(tr("Rescan the library folder")).clicked() {
            app.brushes = None;
        }
    });
    for f in shown {
        if ui
            .add(egui::Button::selectable(false, format!("{}  {}", ic::PAINT_BRUSH_BROAD, name(f))))
            .on_hover_text(tr("Use this brush (keeps its size)"))
            .clicked()
        {
            app.run("brush.preset.load", json!({"path": f.display().to_string()}));
        }
    }
    ui.horizontal(|ui| {
        let id = ui.make_persistent_id("brush_preset_name");
        let mut new_name: String = ui.data_mut(|d| d.get_temp(id)).unwrap_or_default();
        ui.add(egui::TextEdit::singleline(&mut new_name).hint_text(tr("Brush name")).desired_width(110.0));
        let file = texturecraft_engine::formats::export::sanitize_file_stem(new_name.trim());
        if ui.add_enabled(!new_name.trim().is_empty(), egui::Button::new(tr("Save current brush"))).clicked() {
            let path = dir.join(format!("{file}.{BRUSH_EXT}"));
            // A folder that can't be made fails the save, which reports it.
            let _ = std::fs::create_dir_all(&dir);
            if app.run("brush.preset.save", json!({"path": path.display().to_string()})).is_some() {
                new_name.clear();
                app.brushes = None;
            }
        }
        ui.data_mut(|d| d.insert_temp(id, new_name));
    });
    ui.add_space(6.0);
}

#[cfg(target_arch = "wasm32")]
fn my_brushes(_: &mut TextureCraftApp, _: &mut Ui, _: &str) {}

/// The built-in alpha library, grouped by category; the brush's alpha is outlined.
fn alpha_library(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let tk = Tokens::get(ctx);
    let shown: Vec<&'static alphas::LibraryAlpha> = alphas::library().iter().filter(|a| matches(filter, &[&a.label, a.category, &a.id])).collect();
    if shown.is_empty() {
        return;
    }
    title(ui, name);
    let brush = &app.session.tool.brush;
    let current = if brush.alpha_image.is_none() { brush.alpha_library.clone() } else { None };
    for cat in categories(&shown, |a| a.category) {
        header(ui, cat);
        tiles(ui, |ui| {
            for &a in shown.iter().filter(|a| a.category == cat) {
                let tex = app.thumbs.alpha(ctx, a);
                let asset = Asset::Alpha(&a.id);
                let r = tile(ui, &mut app.thumbs, tex.as_ref(), || ShelfPreview::Alpha(a), &a.label, Some(&asset));
                if current.as_deref() == Some(a.id.as_str()) {
                    ui.painter().rect_stroke(r.rect, TILE_ROUNDING, Stroke::new(1.0, tk.accent), egui::StrokeKind::Inside);
                }
                if r.clicked() {
                    apply(app, &asset);
                }
            }
        });
    }
}

fn procedurals(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let shown: Vec<ProceduralKind> = ProceduralKind::ALL.into_iter().filter(|k| matches(filter, &[k.label(), tr(k.label())])).collect();
    if shown.is_empty() {
        return;
    }
    title(ui, name);
    tiles(ui, |ui| {
        for k in shown {
            let tex = app.thumbs.procedural(ctx, k);
            if tile(ui, &mut app.thumbs, tex.as_ref(), || ShelfPreview::Procedural(k), tr(k.label()), None).clicked() {
                app.run("layer.add.fill", json!({"name": k.label()}));
                app.run("layer.fill.set", json!({"channel": "baseColor", "procedural": k.label(), "low": [0.1, 0.1, 0.1], "high": [0.9, 0.9, 0.9]}));
            }
        }
    });
}

fn environments(app: &mut TextureCraftApp, ui: &mut Ui, filter: &str, name: &str) {
    let shown: Vec<&&str> = ENVIRONMENTS.iter().filter(|e| matches(filter, &[e])).collect();
    if shown.is_empty() {
        return;
    }
    title(ui, name);
    for env in shown {
        let on = app.session.doc.project.display.environment == **env;
        if ui.selectable_label(on, format!("{}  {env}", ic::SUN_HORIZON)).clicked() {
            app.run("display.set", json!({"environment": env}));
        }
    }
    ui.add_space(6.0);
}

fn textures(app: &mut TextureCraftApp, ui: &mut Ui, ctx: &egui::Context, filter: &str, name: &str) {
    let tk = Tokens::get(ctx);
    let names: Vec<String> = app.session.doc.resources.keys().filter(|n| matches(filter, &[n])).cloned().collect();
    if !filter.is_empty() && names.is_empty() {
        return;
    }
    ui.horizontal(|ui| {
        title(ui, name);
        if ui.small_button(tr("Import...")).clicked() {
            crate::menus::invoke(app, ctx, "resource.import");
        }
    });
    if names.is_empty() {
        ui.label(RichText::new(tr("Import images to use them in fills and masks")).size(10.0).color(tk.text_dim));
    }
    tiles(ui, |ui| {
        for name in names {
            let Some(img) = app.session.images.get(&name).cloned() else { continue };

            let tex = app.thumbs.image(ctx, &name, &img);
            let asset = Asset::Texture(name.clone());
            let preview = || ShelfPreview::Image(name.clone(), img);
            let r = tile(ui, &mut app.thumbs, Some(&tex), preview, &name, Some(&asset));
            if r.clicked() {
                apply(app, &asset);
            }
            r.context_menu(|ui| {
                if ui.button(tr("Add as decal (view center)")).clicked() {
                    app.run("layer.add.decal", json!({"image": name}));
                    ui.close();
                }
                if ui.button(tr("Remove resource")).clicked() {
                    app.run("resource.remove", json!({"name": name}));
                    ui.close();
                }
            });
        }
    });
}

/// Asset tile with optional dragging and a larger hover preview
pub(crate) fn tile(
    ui: &mut Ui,
    thumbs: &mut Thumbs,
    tex: Option<&egui::TextureHandle>,
    preview: impl FnOnce() -> ShelfPreview,
    label: &str,
    drag: Option<&Asset>,
) -> egui::Response {
    let tk = Tokens::get(ui.ctx());
    let sense = if drag.is_some() { Sense::click_and_drag() } else { Sense::click() };
    let (rect, resp) = ui.allocate_exact_size(vec2(TILE_WIDTH, TILE_HEIGHT), sense);
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, label));
    if let Some(asset) = drag
        && resp.drag_started()
    {
        resp.dnd_set_drag_payload(ShelfDrag { asset: asset.clone(), label: label.to_string() });
    }

    let painter = ui.painter();
    if resp.hovered() || resp.dragged() {
        painter.rect_filled(rect, TILE_ROUNDING, tk.row_hover);
    }
    let image_rect = egui::Rect::from_center_size(rect.center_top() + vec2(0.0, THUMBNAIL_CENTER_Y), vec2(THUMBNAIL_SIZE, THUMBNAIL_SIZE));
    thumbnail(painter, tex, image_rect, &tk);
    let galley = painter.layout(label.to_string(), egui::FontId::proportional(LABEL_FONT_SIZE), tk.text, rect.width() - LABEL_MARGIN);
    painter.galley(egui::pos2(rect.center().x - galley.size().x * 0.5, image_rect.bottom() + LABEL_GAP), galley, tk.text);

    let big = if resp.hovered() { big_preview(ui, thumbs, preview) } else { None };
    resp.on_hover_ui(|ui| hover_preview(ui, tex, big.as_ref(), label))
}

/// Start rendering during the popup delay once the pointer rests
fn big_preview(ui: &Ui, thumbs: &mut Thumbs, preview: impl FnOnce() -> ShelfPreview) -> Option<egui::TextureHandle> {
    let ctx = ui.ctx();
    let preview = preview();
    let requested_size_px = (BIG_PREVIEW_SIZE * ctx.pixels_per_point()).round() as usize;

    // High display scales must respect the orb camera's allocation limit
    let on_the_orb = matches!(&preview, ShelfPreview::SmartMask(_) | ShelfPreview::SmartMaterial(_));
    let size_px = if on_the_orb { requested_size_px.clamp(1, MAX_PREVIEW_SIZE_PX) } else { requested_size_px };

    // Scrolling moves tiles beneath a stationary pointer
    let idle_seconds = ui.input(|input| if input.is_scrolling() { 0.0 } else { input.pointer.time_since_last_movement().min(input.time_since_last_scroll()) });
    if idle_seconds >= BIG_PREVIEW_DWELL {
        return thumbs.big(ctx, preview, size_px);
    }

    ctx.request_repaint_after_secs(BIG_PREVIEW_DWELL - idle_seconds);
    thumbs.rendered_big(ctx, &preview, size_px)
}

/// `texture` drawn into `rect`, or a dim loading icon while it renders
fn thumbnail(painter: &egui::Painter, texture: Option<&egui::TextureHandle>, rect: egui::Rect, tk: &Tokens) {
    match texture {
        Some(texture) => {
            painter.image(texture.id(), rect, egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)), Color32::WHITE);
        }
        None => {
            let font = egui::FontId::proportional(rect.width() * LOADING_ICON_SCALE);
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                egui_phosphor::regular::ARROWS_CLOCKWISE,
                font,
                tk.text_dim.gamma_multiply(LOADING_ICON_OPACITY),
            );
        }
    }
}

/// Use the thumbnail as a placeholder until the hover preview arrives
fn hover_preview(ui: &mut Ui, small: Option<&egui::TextureHandle>, big: Option<&egui::TextureHandle>, label: &str) {
    let tk = Tokens::get(ui.ctx());
    ui.set_max_width(BIG_PREVIEW_SIZE);
    ui.vertical_centered(|ui| {
        let (rect, _) = ui.allocate_exact_size(vec2(BIG_PREVIEW_SIZE, BIG_PREVIEW_SIZE), Sense::hover());
        thumbnail(ui.painter(), big.or(small), rect, &tk);
        ui.add_space(BIG_PREVIEW_LABEL_GAP);
        ui.label(label);
    });
}
