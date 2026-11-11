//! Panel bodies of the right column: Texture Sets, Layers, Texture Set Settings, Properties
//! (hosted as dock tabs by `docking`).

use egui::{Color32, Rect, RichText, Sense, Stroke, Ui, pos2, vec2};
use egui_phosphor::regular as ic;
use serde_json::{Value, json};
use texturecraft_engine::compose::ThumbTarget;
use texturecraft_engine::core::{
    BakedMap, BlendMode, Channel, CurvatureDetail, CurvatureMode, Decal, Effect, EffectKind, Fill, Filter, Generator, GeneratorKind, Layer, LayerId, LayerKind,
    LinkParam, MaterialLink, ProceduralKind, Projection, Source,
};
use texturecraft_engine::filter_lib::scalar_params;
use texturecraft_engine::paint::{BrushAlpha, BrushSettings, alphas};
use texturecraft_engine::{CloneFrom, StackTarget, Tool};

use crate::TextureCraftApp;
use crate::i18n::{n_, tr, trf};
use crate::layer_panel::{effect_preview_size, effect_row_height, layer_preview_size, layer_row_height};
use crate::shelf::{Asset, ShelfDrag};
use crate::state::PropsTab;
use crate::theme::{Tokens, caption};
use crate::toolbar::{MAX_BRUSH_SIZE, MIN_BRUSH_SIZE};
use crate::widgets::{chip, color_swatch, combo, dnd_release, elided, icon_button, section, section_with_reset, slider, slider_edit};

/// The filter Add filter adds: at its defaults it changes nothing, and its slot opens the picker to swap it
const NEW_FILTER: &str = "hsl";
const FILTER_SWATCH_WIDTH: f32 = 36.0;
const FILTER_HINT_SIZE: f32 = 10.5;
/// The most colors `effect.set {colors}` takes for a gradient map
const MAX_GRADIENT_STOPS: usize = 16;

const TAG_COLORS: [Color32; 8] = [
    Color32::from_rgb(0xc0, 0x40, 0x40),
    Color32::from_rgb(0xd0, 0x80, 0x30),
    Color32::from_rgb(0xc8, 0xb0, 0x30),
    Color32::from_rgb(0x58, 0xa0, 0x40),
    Color32::from_rgb(0x30, 0xa0, 0x90),
    Color32::from_rgb(0x40, 0x78, 0xd0),
    Color32::from_rgb(0x80, 0x58, 0xc8),
    Color32::from_rgb(0xb0, 0x50, 0xa0),
];

pub(crate) fn texture_set_list(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    let sets: Vec<(String, String, bool)> = app
        .session
        .doc
        .project
        .texture_sets
        .iter()
        .map(|t| {
            let res = t.resolution;
            let size = match t.uv_tiles.len() {
                0 => format!("{res}x{res}"),
                n => format!("{res}x{res}, {n} UDIM"),
            };
            (t.name.clone(), size, t.visible)
        })
        .collect();
    let active = app.session.ts();
    ui.horizontal(|ui| {
        ui.spacing_mut().item_spacing.x = 4.0;
        ui.menu_button(RichText::new(ic::EYE).size(14.0), |ui| {
            for (label, mode) in [(tr("Show all"), "showAll"), (tr("Hide all"), "hideAll"), (tr("Invert show/hide"), "invert")] {
                if ui.button(label).clicked() {
                    app.run("textureSet.visibility", json!({ "mode": mode }));
                    ui.close();
                }
            }
        })
        .response
        .on_hover_text(tr("Show or hide texture sets"));
        let solo = sets.iter().enumerate().all(|(i, s)| s.2 == (i == active));
        if icon_button(ui, ic::FRAME_CORNERS, tr("Focus: show only the active texture set"), solo && sets.len() > 1, 22.0).clicked() {
            app.run("textureSet.visibility", json!({ "mode": "solo" }));
        }
    });
    egui::ScrollArea::vertical().id_salt("ts_scroll").show(ui, |ui| {
        for (i, (name, size, visible)) in sets.into_iter().enumerate() {
            let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), 22.0), Sense::click());
            let p = ui.painter();
            let bg = if i == active {
                tk.row_selected
            } else if resp.hovered() {
                tk.row_hover
            } else {
                tk.panel
            };
            p.rect_filled(rect, 2.0, bg);
            if i == active {
                p.rect_stroke(rect, 2.0, Stroke::new(1.0, tk.accent), egui::StrokeKind::Inside);
            }
            let eye = Rect::from_min_size(rect.min + vec2(4.0, 3.0), vec2(16.0, 16.0));
            let eye_resp = ui.interact(eye, ui.scope_id().with(("ts_eye", i)), Sense::click()).on_hover_text(tr("Show or hide the texture set"));
            p.text(eye.center(), egui::Align2::CENTER_CENTER, if visible { ic::EYE } else { ic::EYE_SLASH }, egui::FontId::proportional(13.0), tk.text);
            // Narrow panels drop the shader label, then the resolution, before cutting the name.
            let cy = rect.center().y;
            let mut right = rect.right() - 6.0;
            if rect.width() >= 260.0 {
                let g = p.layout_no_wrap(tr("Main shader").into(), egui::FontId::proportional(11.0), tk.text_dim);
                right -= g.size().x;
                p.galley(pos2(right, cy - g.size().y * 0.5), g, tk.text_dim);
                right -= 10.0;
            }
            if rect.width() >= 170.0 {
                let g = p.layout_no_wrap(size, egui::FontId::proportional(11.0), tk.text_dim);
                let x = (rect.left() + rect.width() * 0.48).min(right - g.size().x);
                p.galley(pos2(x, cy - g.size().y * 0.5), g, tk.text_dim);
                right = x - 8.0;
            }
            let color = if i == active { tk.text_strong } else { tk.text };
            let g = elided(p, name, 12.0, color, right - rect.left() - 26.0);
            p.galley(pos2(rect.left() + 26.0, cy - g.size().y * 0.5), g, color);
            if eye_resp.clicked() {
                app.run("textureSet.visible", json!({"ts": i}));
            } else if resp.clicked() {
                app.run("textureSet.select", json!({"ts": i}));
            }
        }
    });
}

// ---------------------------------------------------------------------------------------------
// Layers

/// The add-mask commands, for the selected layer or (`row`) the layer whose mask slot was clicked.
fn add_mask_menu(app: &mut TextureCraftApp, ui: &mut Ui, row: Option<u64>) {
    let enabled = row.is_some() || app.session.selected_layer().is_some();
    for (label, id) in
        [(tr("Add white mask"), "mask.add.white"), (tr("Add black mask"), "mask.add.black"), (tr("Add mask with color selection"), "mask.add.colorSelection")]
    {
        if ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            if let Some(layer) = row {
                app.run("layer.select", json!({ "layer": layer }));
            }
            app.run(id, Value::Null);
            ui.close();
        }
    }
}

pub(crate) fn layers(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    // Wraps in a narrow column instead of widening the rows below past the panel.
    ui.horizontal_wrapped(|ui| {
        // The menu buttons as compact as the icon buttons beside them.
        ui.spacing_mut().button_padding.x = 4.0;
        ui.spacing_mut().item_spacing.x = 4.0;
        let chans = app.session.doc.project.texture_set(app.session.ts()).map(|t| t.channel_list()).unwrap_or_default();
        egui::ComboBox::from_id_salt("layer_channel").width(86.0).selected_text(tr(app.ui.layer_channel.label())).show_ui(ui, |ui| {
            for c in chans {
                ui.selectable_value(&mut app.ui.layer_channel, c, tr(c.label()));
            }
        });
        let selected = app.session.selected_layer().and_then(|id| app.session.doc.project.layers.get(&id)).cloned();
        ui.menu_button(RichText::new(ic::MAGIC_WAND).size(14.0), |ui| match &selected {
            Some(l) => effect_menu(app, ui, l),
            None => {
                ui.label(RichText::new(tr("Select a layer")).color(tk.text_dim));
            }
        })
        .response
        .on_hover_text(tr("Add effect"));
        ui.menu_button(RichText::new(ic::SELECTION_BACKGROUND).size(14.0), |ui| add_mask_menu(app, ui, None)).response.on_hover_text(tr("Add mask"));
        if icon_button(ui, ic::PAINT_BRUSH, tr("Add paint layer"), false, 22.0).clicked() {
            app.run("layer.add.paint", Value::Null);
        }
        if icon_button(ui, ic::PAINT_BUCKET, tr("Add fill layer"), false, 22.0).clicked() {
            app.run("layer.add.fill", Value::Null);
        }
        if icon_button(ui, ic::FOLDER_SIMPLE, tr("Add folder"), false, 22.0).clicked() {
            app.run("layer.add.group", Value::Null);
        }
        if icon_button(ui, ic::TRASH, &crate::keymap::tip(&app.ui.keymap, tr("Remove layer"), "layer.remove"), false, 22.0).clicked() {
            app.run("layer.remove", Value::Null);
        }
    });
    ui.add_space(2.0);
    let ts = app.session.ts();
    let stack: Vec<LayerId> = app.session.doc.project.texture_sets.get(ts).map(|t| t.stack.iter().rev().copied().collect()).unwrap_or_default();
    let now = ui.input(|i| i.time);
    app.layer_panel.begin(&app.session, now);
    egui::ScrollArea::vertical().id_salt("layers_scroll").auto_shrink([false, false]).show(ui, |ui| {
        let mut count = 0;
        for id in stack {
            layer_tree(app, ui, id, 0, &mut count);
        }
        // Click on empty space deselects nothing; the space below the rows drops at the bottom.
        let free = ui.available_height();
        let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), if free.is_finite() { free.max(24.0) } else { 24.0 }), Sense::hover());
        if let Some(dragged) = dnd_release::<u64>(&resp) {
            let bottom = app.session.doc.project.texture_sets.get(ts).and_then(|t| t.stack.first().copied());
            if bottom.map(|b| b.0) != Some(*dragged) {
                app.run("layer.move", json!({"layer": *dragged, "index": 0}));
            }
        }
        match dnd_release::<ShelfDrag>(&resp).map(|d| d.asset.clone()) {
            Some(Asset::Material(m)) => {
                app.run("material.apply", json!({"material": m, "index": 0}));
            }
            Some(Asset::Library(path)) => {
                app.run("material.load", json!({"path": path, "index": 0}));
            }
            _ => {}
        }
        let material = resp.dnd_hover_payload::<ShelfDrag>().is_some_and(|d| d.asset.is_material());
        if resp.dnd_hover_payload::<u64>().is_some() || material {
            ui.painter().line_segment([rect.left_top(), rect.right_top()], Stroke::new(2.0, tk.accent));
        }
    });
    app.layer_panel.end(ui.ctx(), &app.session);
}

/// Whether the selection targets the layer content (the content or one of its effects) rather
/// than the mask.
pub(crate) fn targets_content(app: &TextureCraftApp) -> bool {
    let p = &app.session.doc.project;
    match app.session.target {
        StackTarget::Content => true,
        StackTarget::Mask => false,
        StackTarget::Effect(e) => p.effect(e).and_then(|(owner, _)| p.layers.get(&owner)).is_some_and(|l| l.content_effects.iter().any(|x| x.id == e)),
    }
}

/// "Add effect" items for `layer`: content when it is the selected target, else the mask
fn effect_menu(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let content = !layer.is_group() && app.session.selected_layer() == Some(layer.id) && targets_content(app);
    let target = if content { "content" } else { "mask" };
    ui.label(caption(if content { tr("Add to the layer content") } else { tr("Add to the mask") }));

    // Generators and compare masks always go to the mask, anchor points to the content
    let items = [
        (tr("Add generator"), "effect.add.generator", json!({})),
        (tr("Add paint"), "effect.add.paint", json!({"target": target})),
        (tr("Add fill"), "effect.add.fill", json!({"target": target})),
        (tr("Add levels"), "effect.add.levels", json!({"target": target})),
        (tr("Add compare mask"), "effect.add.compare", json!({})),
        (tr("Add filter"), "effect.add.filter", json!({"target": target, "filter": NEW_FILTER})),
        (tr("Add anchor point"), "effect.add.anchor", json!({})),
    ];
    for (label, command, params) in items {
        let enabled = command != "effect.add.anchor" || !layer.is_group();
        if !ui.add_enabled(enabled, egui::Button::new(label)).clicked() {
            continue;
        }

        if app.session.selected_layer() != Some(layer.id) {
            app.run("layer.select", json!({"layer": layer.id.0}));
        }

        let added = app.run(command, with(&params, json!({"layer": layer.id.0})));
        // The neutral filter is a placeholder: its slot opens the picker to choose the real one
        if command == "effect.add.filter"
            && let Some(effect) = added.as_ref().and_then(|result| result["effect"].as_u64())
        {
            app.ui.dock.focus(crate::docking::Panel::Properties);
            crate::filter_slot::open_picker_when_shown(ui.ctx(), effect);
        }

        ui.close();
    }
}

/// Layer rows alternate between two shades so neighbours read as separate items.
fn row_shade(tk: &Tokens, index: usize) -> Color32 {
    if index.is_multiple_of(2) { tk.row } else { tk.row.lerp_to_gamma(tk.row_hover, 0.45) }
}

/// Width of the blend mode / opacity column at the right of a layer row.
const RIGHT_COLUMN: f32 = 58.0;
/// Indent per folder level, and per level of the effect tree.
const INDENT: f32 = 14.0;
/// Height of the "Content effects" / "Mask effects" header rows.
const GROUP_HEADER_H: f32 = 20.0;

/// Keep both thumbnail targets inside the row before spending space on indentation
fn layer_thumbnail_layout(width: f32, depth: usize, folder: bool, preferred_size: f32) -> (f32, f32) {
    let controls = 7.0 + 20.0 + if folder { 14.0 } else { 0.0 };
    let thumbnail_space = (width - controls - 4.0 - 4.0).max(0.0);
    let size = preferred_size.min(thumbnail_space * 0.5);
    let indent = (depth as f32 * INDENT).min((thumbnail_space - size * 2.0).max(0.0));
    (indent, size)
}

/// Drop secondary controls before allowing them to cover thumbnail targets or the name
fn layer_name_layout(rect: Rect, left: f32) -> (Rect, bool) {
    let right = rect.right() - 4.0;
    let show_blend = right - RIGHT_COLUMN - 4.0 - left >= 20.0;
    let name_right = if show_blend { right - RIGHT_COLUMN - 4.0 } else { right };
    (Rect::from_min_max(pos2(left, rect.top() + 4.0), pos2(name_right.max(left), rect.bottom() - 4.0)), show_blend)
}

/// Preserve the effect eye, preview and a readable name before spending space on nesting.
fn effect_row_indent(width: f32, level: usize, preview: f32) -> f32 {
    (level as f32 * INDENT).min((width - 7.0 - 22.0 - preview - 8.0 - 20.0 - 6.0).max(0.0))
}

/// X of the centre of the eye column at indent `level` in a row spanning from `left`.
fn level_x(left: f32, level: usize) -> f32 {
    left + 7.0 + level as f32 * INDENT + 8.0
}

fn layer_tree(app: &mut TextureCraftApp, ui: &mut Ui, id: LayerId, depth: usize, count: &mut usize) {
    let Some(layer) = app.session.doc.project.layers.get(&id).cloned() else { return };
    let index = *count;
    *count += 1;
    layer_row(app, ui, &layer, depth, index);
    let mask_effects = layer.mask.as_ref().map_or(0, |m| m.effects.len());
    if !layer.content_effects.is_empty() {
        effect_group(app, ui, &layer, depth, false, mask_effects > 0);
    }
    if mask_effects > 0 {
        effect_group(app, ui, &layer, depth, true, false);
    }
    // A line under the layer and its effects separates it from the next layer.
    let tk = Tokens::get(ui.ctx());
    let r = ui.available_rect_before_wrap();
    ui.painter().hline(r.x_range(), r.top() - 0.5, Stroke::new(1.0, tk.separator));
    if let LayerKind::Group { children, collapsed } = &layer.kind
        && !collapsed
        && depth < 12
    {
        for c in children.iter().rev() {
            layer_tree(app, ui, *c, depth + 1, count);
        }
    }
}

/// The picture for `target` inside `rect` (drawn over the well), whether there is one yet.
fn thumb_box(ui: &Ui, app: &mut TextureCraftApp, rect: Rect, target: ThumbTarget, dim: bool) -> bool {
    let tk = Tokens::get(ui.ctx());
    let p = ui.painter();
    p.rect_filled(rect, 2.0, tk.field);
    // A checkerboard shows through where the layer has nothing (transparent texels).
    let cells = 4;
    let cell = rect.width() / cells as f32;
    let check = tk.field.lerp_to_gamma(tk.row_hover, 0.6);
    for y in 0..cells {
        for x in (0..cells).filter(|x| (x + y) % 2 == 0) {
            p.rect_filled(Rect::from_min_size(rect.min + vec2(x as f32 * cell, y as f32 * cell), vec2(cell, cell)), 0.0, check);
        }
    }
    // Only rows on screen ask for a picture.
    if ui.is_rect_visible(rect)
        && let Some(tex) = app.layer_panel.texture(ui.ctx(), &app.session, target)
    {
        let tint = if dim { Color32::WHITE.gamma_multiply(0.4) } else { Color32::WHITE };
        ui.painter().image(tex, rect, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), tint);
        return true;
    }
    false
}

fn layer_row(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer, depth: usize, index: usize) {
    let tk = Tokens::get(ui.ctx());
    let selected = app.session.selected_layer() == Some(layer.id);
    let content_target = selected && targets_content(app);
    let size = layer_preview_size(app);
    let h = layer_row_height(size);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), h), Sense::click_and_drag());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::ListItem, true, selected, &layer.name));
    let p = ui.painter().clone();
    let bg = if selected {
        tk.row_selected
    } else if resp.hovered() {
        tk.row_hover
    } else {
        row_shade(&tk, index)
    };
    p.rect_filled(rect, 0.0, bg);
    let (indent, size) = layer_thumbnail_layout(rect.width(), depth, layer.is_group(), size);

    // Folder contents hang off lines down the left without crossing compact row controls
    let line = Stroke::new(1.0, tk.button_border.gamma_multiply(0.6));
    for level in 0..depth {
        let x = level_x(rect.left(), level);
        if level as f32 * INDENT + 8.0 < indent {
            p.vline(x, rect.y_range(), line);
        }
    }
    if depth > 0 && indent >= INDENT {
        let x = rect.left() + 15.0 + indent - INDENT;
        p.hline(x..=x + INDENT * 0.5, rect.center().y, line);
    }
    if selected {
        p.rect_filled(Rect::from_min_size(rect.min, vec2(3.0, h)), 0.0, tk.accent);
    }
    if let Some(tag) = layer.color_tag.and_then(|t| TAG_COLORS.get(t as usize)) {
        p.rect_filled(Rect::from_min_size(rect.min + vec2(3.0, 0.0), vec2(3.0, h)), 0.0, *tag);
    }
    let x0 = rect.left() + 7.0 + indent;
    // Eye.
    let eye = Rect::from_center_size(pos2(x0 + 8.0, rect.center().y), vec2(18.0, 18.0));
    let eye_resp = ui.interact(eye, ui.scope_id().with(("eye", layer.id.0)), Sense::click()).on_hover_text(tr("Show or hide the layer"));
    p.text(
        eye.center(),
        egui::Align2::CENTER_CENTER,
        if layer.visible { ic::EYE } else { ic::EYE_SLASH },
        egui::FontId::proportional(13.0),
        if layer.visible { tk.text } else { tk.text_dim },
    );
    if eye_resp.clicked() {
        app.run("layer.visible", json!({"layer": layer.id.0}));
    }
    // Group caret.
    let mut x = x0 + 20.0;
    if let LayerKind::Group { collapsed, .. } = layer.kind {
        let caret = Rect::from_center_size(pos2(x + 6.0, rect.center().y), vec2(14.0, 14.0));
        let cr = ui.interact(caret, ui.scope_id().with(("caret", layer.id.0)), Sense::click()).on_hover_text(tr("Expand or collapse the folder"));
        p.text(
            caret.center(),
            egui::Align2::CENTER_CENTER,
            if collapsed { ic::CARET_RIGHT } else { ic::CARET_DOWN },
            egui::FontId::proportional(11.0),
            tk.text_dim,
        );
        if cr.clicked()
            && let Some(LayerKind::Group { collapsed, .. }) = app.session.doc.project.layers.get_mut(&layer.id).map(|l| &mut l.kind)
        {
            *collapsed = !*collapsed;
        }
        x += 14.0;
    }
    // Content thumbnail: the layer's own result in the displayed channel.
    let ch = app.ui.layer_channel;
    let thumb = Rect::from_min_size(pos2(x, rect.center().y - size * 0.5), vec2(size, size));
    let drawn = thumb_box(ui, app, thumb, ThumbTarget::Content { layer: layer.id, channel: ch }, !layer.visible);
    if !drawn {
        content_glyph(&p, &tk, thumb, layer);
    }
    p.rect_stroke(
        thumb,
        2.0,
        Stroke::new(if content_target { 1.5 } else { 1.0 }, if content_target { tk.accent } else { tk.button_border }),
        egui::StrokeKind::Outside,
    );
    let content_resp = ui.interact(thumb, ui.scope_id().with(("content", layer.id.0)), Sense::click()).on_hover_text(tr("Select the layer content"));
    if content_resp.clicked() {
        app.run("layer.select", json!({"layer": layer.id.0, "target": "content"}));
    }
    x = thumb.right() + 4.0;
    // An instance draws its source's content, blending and (unless it has its own) mask.
    let project = &app.session.doc.project;
    let source = layer.instance_source().map(|src| {
        let set = project.locate(src).and_then(|(ts, _, _)| project.texture_sets.get(ts)).map(|t| t.name.as_str()).unwrap_or("?");
        project.layers.get(&src).map_or_else(|| tr("missing source").to_string(), |l| format!("{} · {set}", l.name))
    });
    let mask = project.layers.get(&project.mask_owner(layer.id)).and_then(|l| l.mask.clone());
    let blend = project.content_layer(layer.id).and_then(|l| l.blend(ch));
    // Mask thumbnail, selectable on its own; or the slot that adds one.
    let mask_rect;
    if let Some(m) = &mask {
        let mask_sel = selected && !targets_content(app);
        mask_rect = Rect::from_min_size(pos2(x, rect.center().y - size * 0.5), vec2(size, size));
        let drawn = thumb_box(ui, app, mask_rect, ThumbTarget::Mask { layer: layer.id }, !layer.visible || !m.enabled);
        if !drawn {
            p.rect_filled(
                mask_rect,
                2.0,
                if m.background == texturecraft_engine::core::MaskBackground::White { Color32::from_gray(225) } else { Color32::from_gray(18) },
            );
            if !m.effects.is_empty() {
                p.text(mask_rect.center(), egui::Align2::CENTER_CENTER, ic::MAGIC_WAND, egui::FontId::proportional(13.0), tk.text_dim);
            }
        }
        p.rect_stroke(
            mask_rect,
            2.0,
            Stroke::new(if mask_sel { 1.5 } else { 1.0 }, if mask_sel { tk.accent } else { tk.button_border }),
            egui::StrokeKind::Outside,
        );
        if !m.enabled {
            p.line_segment([mask_rect.left_top(), mask_rect.right_bottom()], Stroke::new(1.5, tk.danger));
        }
        let mr = ui
            .interact(mask_rect, ui.scope_id().with(("mask", layer.id.0)), Sense::click())
            .on_hover_text(tr("Select the mask (Alt+click: show it in the viewport)"));
        if mr.clicked() {
            app.run("layer.select", json!({"layer": layer.id.0, "target": "mask"}));
            if ui.input(|i| i.modifiers.alt) {
                let showing = app.session.view == texturecraft_engine::ViewMode::Mask(layer.id);
                let p = if showing { json!({"mode": "material"}) } else { json!({"mode": "mask", "layer": layer.id.0}) };
                app.run("viewport.view", p);
            }
        }
        if app.session.view == texturecraft_engine::ViewMode::Mask(layer.id) {
            p.rect_stroke(mask_rect.expand(2.0), 2.0, Stroke::new(1.5, tk.warning), egui::StrokeKind::Outside);
        }
        x = mask_rect.right() + 8.0;
    } else {
        // The empty slot adds a mask to this layer.
        let slot_size = size.min(18.0);
        mask_rect = Rect::from_min_size(pos2(x, rect.center().y - slot_size * 0.5), vec2(slot_size, slot_size));
        let slot = ui.interact(mask_rect, ui.scope_id().with(("mask_slot", layer.id.0)), Sense::click()).on_hover_text(tr("Add mask"));
        p.rect_stroke(mask_rect, 2.0, Stroke::new(1.0, if slot.hovered() { tk.accent } else { tk.field_border }), egui::StrokeKind::Inside);
        if slot.hovered() {
            p.text(mask_rect.center(), egui::Align2::CENTER_CENTER, ic::PLUS, egui::FontId::proportional(12.0), tk.text);
        }
        egui::Popup::menu(&slot).show(|ui| add_mask_menu(app, ui, Some(layer.id.0)));
        x = mask_rect.right() + 8.0;
    }
    // Blend mode and opacity of the displayed channel, stacked at the right.
    let blend_rect = Rect::from_min_size(pos2(rect.right() - RIGHT_COLUMN - 4.0, rect.center().y - 18.0), vec2(RIGHT_COLUMN, 17.0));
    let op_rect = Rect::from_min_size(pos2(rect.right() - RIGHT_COLUMN - 4.0, rect.center().y + 1.0), vec2(RIGHT_COLUMN, 17.0));
    // Name (or rename editor) with the quieter subtitle under it.
    let (name_rect, show_blend) = layer_name_layout(rect, x);
    let show_name = name_rect.width() >= 20.0;
    if !show_name {
        resp.clone().on_hover_text(&layer.name);
        if app.ui.rename.as_ref().is_some_and(|(id, _)| *id == layer.id.0) {
            app.ui.rename = None;
        }
    }
    let renaming = app.ui.rename.as_ref().is_some_and(|(r, _)| *r == layer.id.0);
    if renaming {
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(name_rect).layout(egui::Layout::left_to_right(egui::Align::Center)));
        if let Some((_, buf)) = app.ui.rename.as_mut() {
            // Escape reaches us instead of just unfocusing the field, so it can cancel.
            let filter = egui::EventFilter { horizontal_arrows: true, vertical_arrows: true, escape: true, ..Default::default() };
            let r = child.add(egui::TextEdit::singleline(buf).desired_width(name_rect.width()).event_filter(filter));
            if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                app.ui.rename = None;
            } else if r.lost_focus() {
                // Enter or a click elsewhere.
                let name = buf.trim().to_string();
                app.ui.rename = None;
                if !name.is_empty() && name != layer.name {
                    app.run("layer.rename", json!({"layer": layer.id.0, "name": name}));
                }
            } else if !r.has_focus() {
                // Only until the field has focus: requesting it every frame kept it from ever
                // losing focus, which trapped the editor open and blocked every shortcut.
                r.request_focus();
            }
        }
    } else if show_name {
        let galley = elided(&p, layer.name.clone(), 12.5, if layer.visible { tk.text_strong } else { tk.text_dim }, name_rect.width());
        p.galley(pos2(name_rect.left(), rect.center().y - 2.0 - galley.size().y), galley, tk.text);
        let kind = tr(match &layer.kind {
            LayerKind::Paint => "Paint layer",
            LayerKind::Fill(_) => "Fill layer",
            LayerKind::Group { .. } => "Folder",
            LayerKind::Instance { .. } => "Instance",
        });
        let effects = layer.content_effects.len() + mask.as_ref().map_or(0, |m| m.effects.len());
        let subtitle = match (source, effects) {
            (Some(src), _) => format!("{kind} · {} {src}", ic::LINK_SIMPLE),
            (None, 0) => kind.to_string(),
            (None, 1) => format!("{kind} · {}", tr("1 effect")),
            (None, n) => format!("{kind} · {}", trf("{count} effects", &[("count", &n)])),
        };
        let sub = elided(&p, subtitle, 10.0, tk.text_dim, name_rect.width());
        p.galley(pos2(name_rect.left(), rect.center().y + 2.0), sub, tk.text_dim);
    }
    if let Some(b) = blend.filter(|_| show_blend) {
        let br = ui.interact(blend_rect, ui.scope_id().with(("blend", layer.id.0)), Sense::click()).on_hover_text(tr("Blending mode of the displayed channel"));
        if br.hovered() {
            p.rect_filled(blend_rect, 2.0, tk.row_hover);
        }
        p.text(blend_rect.right_center() - vec2(10.0, 0.0), egui::Align2::RIGHT_CENTER, b.mode.short(), egui::FontId::proportional(11.0), tk.text);
        p.text(blend_rect.right_center(), egui::Align2::RIGHT_CENTER, ic::CARET_DOWN, egui::FontId::proportional(8.0), tk.text_dim);
        egui::Popup::menu(&br).show(|ui| {
            ui.set_min_width(190.0);
            for m in BlendMode::MENU {
                match m {
                    Some(m) => {
                        if ui.selectable_label(m == b.mode, tr(m.label())).clicked() {
                            app.run("layer.blend", json!({"layer": layer.id.0, "channel": ch.id(), "mode": format!("{m:?}")}));
                            ui.close();
                        }
                    }
                    None => {
                        ui.separator();
                    }
                }
            }
        });
        let mut op = (b.opacity * 100.0).round();
        let mut child = ui.new_child(egui::UiBuilder::new().max_rect(op_rect).layout(egui::Layout::right_to_left(egui::Align::Center)));
        // As tall as its slot in the row (the style's fields are taller), in the row's small print.
        child.spacing_mut().interact_size.y = op_rect.height();
        child.spacing_mut().button_padding = vec2(4.0, 0.0);
        child.style_mut().override_text_style = Some(egui::TextStyle::Small);
        let r = child.add(egui::DragValue::new(&mut op).range(0.0..=100.0).speed(0.5).max_decimals(0).suffix("%"));
        if r.changed() {
            app.run("layer.blend", json!({"layer": layer.id.0, "channel": ch.id(), "opacity": op / 100.0}));
        }
    } else if show_blend {
        p.text(blend_rect.right_center(), egui::Align2::RIGHT_CENTER, tr("off"), egui::FontId::proportional(11.0), tk.text_dim);
    }
    // Interaction: select, rename, drag & drop, context menu.
    let on_control = eye_resp.clicked() || content_resp.clicked();
    if resp.clicked() && !on_control && !renaming {
        app.run("layer.select", json!({"layer": layer.id.0}));
    }
    if resp.double_clicked() && !on_control && !renaming && show_name {
        app.ui.rename = Some((layer.id.0, layer.name.clone()));
    }
    if resp.drag_started() && !renaming {
        resp.dnd_set_drag_payload(layer.id.0);
    }
    // Drops: a layer or a shelf material lands above this row (inside a folder when dropped on
    // its middle); a shelf smart mask replaces this layer's mask.
    let inside = layer.is_group() && ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| (p.y - rect.center().y).abs() < h * 0.25);
    let shelf = resp.dnd_hover_payload::<ShelfDrag>().map(|d| d.asset.clone());
    let moving = resp.dnd_hover_payload::<u64>().is_some_and(|src| *src != layer.id.0);
    if moving || shelf.as_ref().is_some_and(Asset::is_material) {
        if inside {
            p.rect_stroke(rect, 2.0, Stroke::new(2.0, tk.accent), egui::StrokeKind::Inside);
        } else {
            p.line_segment([rect.left_top(), rect.right_top()], Stroke::new(2.0, tk.accent));
        }
    } else if matches!(shelf, Some(Asset::SmartMask(_))) {
        p.rect_stroke(mask_rect.expand(2.0), 2.0, Stroke::new(2.0, tk.accent), egui::StrokeKind::Outside);
    } else if matches!(shelf, Some(Asset::Filter(_))) && !layer.is_group() {
        p.rect_stroke(rect, 2.0, Stroke::new(2.0, tk.accent), egui::StrokeKind::Inside);
    }
    if let Some(src) = dnd_release::<u64>(&resp)
        && *src != layer.id.0
    {
        app.run("layer.move", json!({"layer": *src, "above": layer.id.0, "inside": inside}));
    }
    match dnd_release::<ShelfDrag>(&resp).map(|d| d.asset.clone()) {
        Some(Asset::Material(m)) => {
            app.run("material.apply", json!({"material": m, "above": layer.id.0, "inside": inside}));
        }
        Some(Asset::Library(path)) => {
            app.run("material.load", json!({"path": path, "above": layer.id.0, "inside": inside}));
        }
        Some(Asset::SmartMask(m)) => {
            app.run("mask.smart.apply", json!({"name": m, "layer": layer.id.0}));
        }
        Some(Asset::Filter(f)) => {
            app.run("layer.select", json!({"layer": layer.id.0}));
            app.run("effect.add.filter", json!({"layer": layer.id.0, "filter": f, "target": "content"}));
        }
        Some(Asset::Alpha(_) | Asset::Texture(_)) | None => {}
    }
    resp.context_menu(|ui| layer_context_menu(app, ui, layer));
}

/// What a content thumbnail shows until its picture arrives (or where the layer has none).
fn content_glyph(p: &egui::Painter, tk: &Tokens, thumb: Rect, layer: &Layer) {
    match &layer.kind {
        LayerKind::Fill(f) => {
            let c = match f.source(Channel::BaseColor) {
                Source::Uniform(v) => v,
                Source::Procedural { low, high, .. } => [(low[0] + high[0]) * 0.5, (low[1] + high[1]) * 0.5, (low[2] + high[2]) * 0.5],
                Source::Image { .. } | Source::Anchor { .. } => [0.5; 3],
            };
            crate::thumbs::paint_sphere(p, thumb.center(), thumb.width() * 0.34, c);
        }
        LayerKind::Paint => {
            p.text(thumb.center(), egui::Align2::CENTER_CENTER, ic::PAINT_BRUSH, egui::FontId::proportional(thumb.width() * 0.45), tk.text_dim);
        }
        LayerKind::Group { .. } => {
            p.text(thumb.center(), egui::Align2::CENTER_CENTER, ic::FOLDER_SIMPLE, egui::FontId::proportional(thumb.width() * 0.5), tk.text);
        }
        LayerKind::Instance { .. } => {
            p.text(thumb.center(), egui::Align2::CENTER_CENTER, ic::LINK, egui::FontId::proportional(thumb.width() * 0.45), tk.accent);
        }
    }
}

/// The "Content effects" or "Mask effects" header of `layer` and, unless folded, its effect rows
/// hung off tree lines. `more_below`: a second group follows, so the tree line goes on.
fn effect_group(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer, depth: usize, mask: bool, more_below: bool) {
    let tk = Tokens::get(ui.ctx());
    let stack: &[Effect] = if mask { layer.mask.as_ref().map_or(&[], |m| &m.effects) } else { &layer.content_effects };
    let key = (layer.id.0, mask);
    let folded = app.layer_panel.folded.contains(&key);
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), GROUP_HEADER_H), Sense::click());
    let label = if mask { tr("Mask effects") } else { tr("Content effects") };
    resp.widget_info(|| egui::WidgetInfo::labeled(egui::Role::Button, true, label));
    let p = ui.painter().clone();
    p.rect_filled(rect, 0.0, if resp.hovered() { tk.row_hover } else { tk.panel });
    let indent = ((depth + 1) as f32 * INDENT).min((rect.width() - 90.0).max(0.0));
    tree_lines(&p, &tk, rect, indent, &[], more_below);
    let x = rect.left() + 15.0 + indent;
    p.text(
        pos2(x, rect.center().y),
        egui::Align2::CENTER_CENTER,
        if folded { ic::CARET_RIGHT } else { ic::CARET_DOWN },
        egui::FontId::proportional(10.0),
        tk.text_dim,
    );
    let galley = elided(&p, label.to_string(), 10.5, tk.text_dim, (rect.right() - 28.0 - x - 10.0).max(0.0));
    p.galley(pos2(x + 10.0, rect.center().y - galley.size().y * 0.5), galley, tk.text_dim);
    p.text(pos2(rect.right() - 8.0, rect.center().y), egui::Align2::RIGHT_CENTER, stack.len().to_string(), egui::FontId::proportional(10.0), tk.text_dim);
    if resp.on_hover_text(tr("Fold or unfold these effects")).clicked() && !app.layer_panel.folded.remove(&key) {
        app.layer_panel.folded.insert(key);
    }
    if folded {
        return;
    }
    let trunks: Vec<usize> = if more_below { vec![depth] } else { Vec::new() };
    for (i, e) in stack.iter().rev().enumerate() {
        effect_row(app, ui, layer, stack, e, depth + 2, &trunks, i + 1 == stack.len(), mask);
    }
}

/// Tree connector of a row at the clamped `indent`: a line down from the parent's column (to the middle
/// of the row when `more` is false, the row's end otherwise) and a tick towards the row, with
/// lines straight through for the `trunks` of ancestors that continue.
fn tree_lines(p: &egui::Painter, tk: &Tokens, rect: Rect, indent: f32, trunks: &[usize], more: bool) {
    if indent < INDENT {
        return;
    }
    let stroke = Stroke::new(1.0, tk.button_border);
    for t in trunks {
        if *t as f32 * INDENT + 8.0 < indent {
            p.vline(level_x(rect.left(), *t), rect.y_range(), stroke);
        }
    }
    let x = rect.left() + 15.0 + indent - INDENT;
    let end = if more { rect.bottom() } else { rect.center().y };
    p.vline(x, rect.top()..=end, stroke);
    p.hline(x..=x + INDENT * 0.6, rect.center().y, stroke);
}

/// The drag-and-drop payload of an effect row (the effect id).
struct EffectDrag(u64);

/// Effect `e` of `stack` (a layer's content or, with `in_mask`, mask effects) at indent `level`.
#[allow(clippy::too_many_arguments)]
fn effect_row(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer, stack: &[Effect], e: &Effect, level: usize, trunks: &[usize], last: bool, in_mask: bool) {
    let tk = Tokens::get(ui.ctx());
    let (eid, visible) = (e.id, e.visible);
    let selected = app.session.selected_layer() == Some(layer.id) && app.session.target == StackTarget::Effect(eid);
    let size = effect_preview_size(layer_preview_size(app));
    let (rect, resp) = ui.allocate_exact_size(vec2(ui.available_width(), effect_row_height(layer_preview_size(app))), Sense::click_and_drag());
    resp.widget_info(|| egui::WidgetInfo::selected(egui::Role::ListItem, true, selected, &e.name));
    let p = ui.painter().clone();
    p.rect_filled(
        rect,
        0.0,
        if selected {
            tk.row_selected
        } else if resp.hovered() {
            tk.row_hover
        } else {
            tk.panel
        },
    );
    let indent = effect_row_indent(rect.width(), level, size);
    tree_lines(&p, &tk, rect, indent, trunks, !last);
    if selected {
        p.rect_filled(Rect::from_min_size(rect.min, vec2(3.0, rect.height())), 0.0, tk.accent);
    }
    let x0 = rect.left() + 7.0 + indent;
    let eye = Rect::from_center_size(pos2(x0 + 8.0, rect.center().y), vec2(16.0, 16.0));
    let er = ui.interact(eye, ui.scope_id().with(("eff_eye", eid.0)), Sense::click()).on_hover_text(tr("Enable or disable the effect"));
    p.text(eye.center(), egui::Align2::CENTER_CENTER, if visible { ic::EYE } else { ic::EYE_SLASH }, egui::FontId::proportional(12.0), tk.text_dim);
    // The stack as it is up to and including this effect; a hidden one shows what it bypasses.
    let thumb = Rect::from_min_size(pos2(x0 + 22.0, rect.center().y - size * 0.5), vec2(size, size));
    let ch = app.ui.layer_channel;
    if !thumb_box(ui, app, thumb, ThumbTarget::Stage { effect: eid, channel: ch }, false) {
        let glyph = if in_mask { ic::MAGIC_WAND } else { ic::SPHERE };
        p.text(thumb.center(), egui::Align2::CENTER_CENTER, glyph, egui::FontId::proportional(size * 0.5), tk.text_dim);
    }
    p.rect_stroke(thumb, 2.0, Stroke::new(1.0, tk.button_border), egui::StrokeKind::Outside);
    if !visible {
        p.line_segment([thumb.left_bottom(), thumb.right_top()], Stroke::new(1.5, tk.text_dim));
    }
    let name_x = thumb.right() + 8.0;
    let galley = elided(&p, e.name.clone(), 11.5, if visible { tk.text } else { tk.text_dim }, (rect.right() - name_x - 6.0).max(20.0));
    p.galley(pos2(name_x, rect.center().y - galley.size().y * 0.5), galley, tk.text);
    if er.clicked() {
        app.run("effect.set", json!({"effect": eid.0, "visible": !visible}));
    } else if resp.clicked() {
        app.run("layer.select", json!({"layer": layer.id.0, "target": eid.0}));
    }
    // Drag to reorder within the stack: a drop lands above this row, or below it from its lower half.
    if resp.drag_started() {
        resp.dnd_set_drag_payload(EffectDrag(eid.0));
    }
    let pos = |id: u64| stack.iter().position(|x| x.id.0 == id);
    let below = ui.input(|i| i.pointer.hover_pos()).is_some_and(|p| p.y > rect.center().y);
    if resp.dnd_hover_payload::<EffectDrag>().is_some_and(|src| src.0 != eid.0 && pos(src.0).is_some()) {
        let y = if below { rect.bottom() } else { rect.top() };
        p.line_segment([pos2(rect.left(), y), pos2(rect.right(), y)], Stroke::new(2.0, tk.accent));
    }
    if let Some(src) = dnd_release::<EffectDrag>(&resp)
        && let (Some(from), Some(to)) = (pos(src.0), pos(eid.0))
        && from != to
    {
        // Rows list the stack top first, so above this row is the next index up.
        let index = match (from < to, below) {
            (true, false) | (false, true) => to,
            (true, true) => to - 1,
            (false, false) => to + 1,
        };
        app.run("effect.move", json!({"effect": src.0, "index": index}));
    }
    resp.context_menu(|ui| {
        let keymap = app.ui.keymap.clone();
        // The row is selected before the paste runs, so a copied effect is all it needs.
        let paste = texturecraft_engine::effect_clipboard::clipboard_effect(&app.session).is_some();
        // The clipboard items show the layer clipboard's shortcuts, which act on a selected effect.
        for (label, id, keys, enabled) in [
            (tr("Copy effect"), "effect.copy", Some("layer.copy"), true),
            (tr("Cut effect"), "effect.cut", Some("layer.cut"), true),
            (tr("Paste effect"), "effect.paste", Some("layer.paste"), paste),
            (tr("Duplicate effect"), "effect.duplicate", None, true),
            (tr("Remove effect"), "effect.remove", None, true),
        ] {
            let mut b = egui::Button::new(label);
            if let Some(sc) = keys.and_then(|k| crate::keymap::shortcut_of(&keymap, k)) {
                b = b.shortcut_text(sc);
            }
            if ui.add_enabled(enabled, b).clicked() {
                // Selected first: a paste lands above it, a duplicate takes the selection.
                app.run("layer.select", json!({"layer": layer.id.0, "target": eid.0}));
                app.run(id, json!({"effect": eid.0}));
                ui.close();
            }
        }
    });
}

/// The Geometry mask tool: what a click does, and the selected layer's parts.
fn geometry_tool_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Geometry mask")));
    ui.label(RichText::new(tr("Click a mesh part to show or hide the selected layer on it.")).color(tk.text_dim));
    let Some(layer) = app.session.selected_layer().and_then(|id| app.session.doc.project.layers.get(&id).cloned()) else { return };
    let ts = app.session.doc.project.locate(layer.id).map_or(app.session.ts(), |(ts, _, _)| ts);
    let parts = texturecraft_engine::set_cmds::parts(&app.session, ts);
    for part in &parts {
        let on = layer.geometry_mask.as_ref().is_none_or(|m| m.contains(part));
        ui.label(RichText::new(format!("{} {part}", if on { ic::EYE } else { ic::EYE_SLASH })).color(if on { tk.text } else { tk.text_dim }));
    }
    if ui.add_enabled(layer.geometry_mask.is_some(), egui::Button::new(tr("Show on all parts"))).clicked() {
        app.run("layer.set", json!({"layer": layer.id.0, "geometryMask": null}));
    }
}

/// Geometry mask: the mesh parts of the layer's texture set it shows on (when the set has
/// several parts).
fn geometry_mask_menu(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let ts = app.session.doc.project.locate(layer.id).map_or(app.session.ts(), |(ts, _, _)| ts);
    let parts = texturecraft_engine::set_cmds::parts(&app.session, ts);
    if parts.len() < 2 {
        return;
    }
    let lid = layer.id.0;
    let shown = layer.geometry_mask.clone().unwrap_or_else(|| parts.clone());
    ui.menu_button(tr("Geometry mask"), |ui| {
        for part in &parts {
            let mut on = shown.contains(part);
            if ui.checkbox(&mut on, part.as_str()).changed() {
                let next: Vec<&String> = parts.iter().filter(|q| if *q == part { on } else { shown.contains(q) }).collect();
                let value = if next.len() == parts.len() { Value::Null } else { json!(next) };
                app.run("layer.set", json!({"layer": lid, "geometryMask": value}));
            }
        }
        ui.separator();
        if ui.add_enabled(layer.geometry_mask.is_some(), egui::Button::new(tr("Show on all parts"))).clicked() {
            app.run("layer.set", json!({"layer": lid, "geometryMask": null}));
            ui.close();
        }
    });
}

fn layer_context_menu(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let tk = Tokens::get(ui.ctx());
    ui.set_min_width(230.0);
    let lid = layer.id.0;
    let keymap = app.ui.keymap.clone();
    let run = |app: &mut TextureCraftApp, ui: &mut Ui, label: &str, id: &str, p: Value, enabled: bool| {
        let mut b = egui::Button::new(label);
        if let Some(sc) = crate::keymap::shortcut_of(&keymap, id) {
            b = b.shortcut_text(sc);
        }
        if ui.add_enabled(enabled, b).clicked() {
            app.run("layer.select", json!({"layer": lid}));
            app.run(id, p);
            ui.close();
        }
    };
    // An instance without a mask of its own edits its source's; adding one overrides it.
    let project = &app.session.doc.project;
    let has_mask = project.layers.get(&project.mask_owner(layer.id)).is_some_and(|l| l.mask.is_some());
    run(app, ui, tr("Add white mask"), "mask.add.white", json!({"layer": lid}), layer.mask.is_none());
    run(app, ui, tr("Add black mask"), "mask.add.black", json!({"layer": lid}), layer.mask.is_none());
    run(app, ui, tr("Add mask with color selection"), "mask.add.colorSelection", json!({"layer": lid}), true);
    ui.separator();
    run(app, ui, tr("Toggle mask"), "mask.toggle", json!({"layer": lid}), has_mask);
    run(app, ui, tr("Invert mask background"), "mask.invertBackground", json!({"layer": lid}), has_mask);
    run(app, ui, tr("Invert mask"), "mask.invert", json!({"layer": lid}), has_mask);
    run(app, ui, tr("Clear mask"), "mask.clear", json!({"layer": lid}), has_mask);
    run(app, ui, tr("Remove mask"), "mask.remove", json!({"layer": lid}), has_mask);
    geometry_mask_menu(app, ui, layer);
    ui.separator();
    run(app, ui, tr("Flatten layer"), "layer.flatten", json!({"layer": lid}), !layer.is_group());
    #[cfg(not(target_arch = "wasm32"))]
    if ui.button(tr("Create smart material")).clicked() {
        let file = crate::library_dir().join(format!("{}.tcmat", texturecraft_engine::formats::sanitize_resource_name(&layer.name)));
        if let Some(r) = app.run("material.save", json!({"layer": lid, "path": file.display().to_string(), "name": layer.name})) {
            let p = r.get("path").and_then(|p| p.as_str()).unwrap_or_default().to_string();
            app.toast(trf("Smart material saved to {path}", &[("path", &p)]), false);
            app.library = None;
        }
        ui.close();
    }
    ui.separator();
    run(app, ui, tr("Cut layer(s)"), "layer.cut", json!({"layer": lid}), true);
    run(app, ui, tr("Copy layer(s)"), "layer.copy", json!({"layer": lid}), true);
    let paste_layer = crate::menus::enabled(app, "layer.paste");
    let paste_effect = texturecraft_engine::effect_clipboard::clipboard_effect(&app.session).is_some();
    run(app, ui, tr("Paste layer(s)"), "layer.paste", json!({}), paste_layer);
    run(app, ui, tr("Paste effect"), "effect.paste", json!({}), paste_effect);
    let paste_instance = crate::menus::enabled(app, "layer.pasteInstance");
    run(app, ui, tr("Paste as instance"), "layer.pasteInstance", json!({}), paste_instance);
    run(app, ui, tr("Duplicate layer(s)"), "layer.duplicate", json!({"layer": lid}), true);
    run(app, ui, tr("Instantiate across texture sets"), "layer.instantiateAll", json!({"layer": lid}), app.session.set_count() > 1);
    run(app, ui, tr("Copy to all texture sets"), "layer.instantiateAll", json!({"layer": lid, "copy": true}), app.session.set_count() > 1);
    run(app, ui, tr("Remove layer(s)"), "layer.remove", json!({"layer": lid}), true);
    run(app, ui, tr("Group layer(s)"), "layer.group", json!({"layer": lid}), true);
    ui.separator();
    ui.horizontal(|ui| {
        let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
        ui.painter().rect_stroke(r, 2.0, Stroke::new(1.0, tk.text_dim), egui::StrokeKind::Inside);
        ui.painter().line_segment([r.left_bottom(), r.right_top()], Stroke::new(1.0, tk.danger));
        if resp.clicked() {
            app.run("layer.colorTag", json!({"layer": lid}));
        }
        for (i, c) in TAG_COLORS.iter().enumerate() {
            let (r, resp) = ui.allocate_exact_size(vec2(16.0, 16.0), Sense::click());
            ui.painter().rect_filled(r, 2.0, *c);
            if resp.clicked() {
                app.run("layer.colorTag", json!({"layer": lid, "tag": i}));
            }
        }
    });
    ui.separator();
    effect_menu(app, ui, layer);
}

// ---------------------------------------------------------------------------------------------
// Texture set settings

pub(crate) fn texture_set_settings(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    let ts = app.session.ts();
    let Some(set) = app.session.doc.project.texture_sets.get(ts).cloned() else { return };
    egui::ScrollArea::vertical().id_salt("tss").show(ui, |ui| {
        ui.label(caption(tr("General properties")));
        egui::Grid::new("tss_grid").num_columns(2).spacing(vec2(8.0, 6.0)).show(ui, |ui| {
            ui.label(tr("Name"));
            ui.label(RichText::new(&set.name).color(tk.text_strong));
            ui.end_row();
            ui.label(tr("Size"));
            egui::ComboBox::from_id_salt("ts_size").selected_text(format!("{}", set.resolution)).show_ui(ui, |ui| {
                for r in [256u32, 512, 1024, 2048, 4096, 8192] {
                    if ui.selectable_label(r == set.resolution, format!("{r}")).clicked() {
                        app.start_resize(ts, r);
                    }
                }
            });
            ui.end_row();
            ui.label(tr("UV tiles (UDIM)"));
            let mut tiled = !set.uv_tiles.is_empty();
            if ui.checkbox(&mut tiled, "").changed() {
                app.run("textureSet.uvTiles", json!({"ts": ts, "enabled": tiled}));
            }
            ui.end_row();
            // Each tile's own size (per-tile resolution).
            for &udim in &set.uv_tiles {
                let r = set.tile_resolution(udim);
                ui.label(RichText::new(format!("   {udim}")).color(tk.text_dim));
                egui::ComboBox::from_id_salt(("ts_tile", udim)).selected_text(format!("{r}")).show_ui(ui, |ui| {
                    for v in [256u32, 512, 1024, 2048, 4096, 8192] {
                        if ui.selectable_label(v == r, format!("{v}")).clicked() {
                            app.run("textureSet.tile.resolution", json!({"ts": ts, "tile": udim, "resolution": v}));
                        }
                    }
                });
                ui.end_row();
            }
            ui.label(tr("Shader instance"));
            ui.label(tr("Main shader"));
            ui.end_row();
        });
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(caption(tr("Channels")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.menu_button(tr("Add channel"), |ui| {
                    for c in Channel::ALL {
                        if !set.has_channel(c) && ui.button(tr(c.label())).clicked() {
                            app.run("textureSet.channel.add", json!({"ts": ts, "channel": c.id()}));
                            ui.close();
                        }
                    }
                });
            });
        });
        for (c, f) in &set.channels {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tr(c.label())).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if *c != Channel::BaseColor && icon_button(ui, ic::X, tr("Remove channel"), false, 18.0).clicked() {
                        app.run("textureSet.channel.remove", json!({"ts": ts, "channel": c.id()}));
                    }
                    ui.label(RichText::new(f.label()).color(tk.text_dim));
                });
            });
        }
        ui.add_space(6.0);
        let mut hs = set.height_strength;
        if let Some(merge) = slider_edit(ui, tr("Height to normal strength"), &mut hs, 0.0..=10.0, 2) {
            app.run("textureSet.settings", json!({"ts": ts, "heightStrength": hs, "merge": merge}));
        }
        let directx = app.session.doc.project.normal_format == texturecraft_engine::core::NormalFormat::DirectX;
        combo(ui, tr("Normal map format"), if directx { "DirectX" } else { "OpenGL" }, |ui| {
            for (id, label, on) in [("openGl", "OpenGL", !directx), ("directX", "DirectX", directx)] {
                if ui.selectable_label(on, label).clicked() && !on {
                    app.run("textureSet.settings", json!({"ts": ts, "normalFormat": id}));
                }
            }
        })
        .on_hover_text(tr(
            "For the whole project: how normal map images set as mesh maps are read (set them again after a change), and the default for export",
        ));
        ui.add_space(8.0);
        ui.horizontal(|ui| {
            ui.label(caption(tr("Mesh maps")));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.button(tr("Bake Mesh Maps")).clicked() {
                    app.ui.mode = crate::state::Mode::Baking;
                }
            });
        });
        for m in BakedMap::ALL {
            let present = app.session.doc.baked.get(&ts).is_some_and(|b| b.maps.contains_key(&m));
            let row = ui.horizontal(|ui| {
                let (r, _) = ui.allocate_exact_size(vec2(26.0, 26.0), Sense::hover());
                ui.painter().rect_filled(r, 2.0, if present { mesh_map_color(m) } else { tk.button });
                ui.vertical(|ui| {
                    ui.label(RichText::new(tr(crate::i18n::baked_map_label(m))).strong());
                    ui.label(
                        RichText::new(if present { format!("{}_{}", set.name, m.id()) } else { tr("Missing (bake to create)").into() })
                            .size(10.5)
                            .color(tk.text_dim),
                    );
                });
                // An image resource instead of a bake, or no map.
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.menu_button(RichText::new(ic::DOTS_THREE).size(13.0), |ui| {
                        if let Some(name) = image_menu(app, ui) {
                            app.run("textureSet.meshMap.set", json!({"ts": ts, "map": m.id(), "image": name}));
                        }
                        ui.separator();
                        if ui.add_enabled(present, egui::Button::new(tr("Clear"))).clicked() {
                            app.run("textureSet.meshMap.clear", json!({"ts": ts, "map": m.id()}));
                            ui.close();
                        }
                    });
                });
            });
            // A shelf image dropped on the row becomes this mesh map.
            let resp = row.response.interact(Sense::hover()).on_hover_text(tr("Drop an image from the Assets panel here to use it as this mesh map"));
            if let Some(name) = image_drop(ui, &resp) {
                app.run("textureSet.meshMap.set", json!({"ts": ts, "map": m.id(), "image": name}));
            }
        }
    });
}

/// Menu items listing the image resources, then Import...: the image picked, if any.
fn image_menu(app: &mut TextureCraftApp, ui: &mut Ui) -> Option<String> {
    ui.label(caption(tr("Images")));
    let mut picked = None;
    // Bounded, so Import... and the items after it stay on screen with many images.
    egui::ScrollArea::vertical().id_salt("image_menu").max_height(240.0).show(ui, |ui| {
        for name in app.session.doc.resources.keys() {
            if ui.button(name.as_str()).clicked() {
                picked = Some(name.clone());
            }
        }
    });
    if ui.button(tr("Import...")).clicked() {
        crate::menus::invoke(app, ui.ctx(), "resource.import");
        ui.close();
    }
    if picked.is_some() {
        ui.close();
    }
    picked
}

/// An image from the Assets panel dropped on `resp`, outlined while one hovers it. Other assets
/// are left for the drop targets around it.
fn image_drop(ui: &Ui, resp: &egui::Response) -> Option<String> {
    if !resp.dnd_hover_payload::<ShelfDrag>().is_some_and(|d| matches!(d.asset, Asset::Texture(_))) {
        return None;
    }
    ui.painter().rect_stroke(resp.rect.shrink(1.0), 3.0, Stroke::new(2.0, Tokens::get(ui.ctx()).accent), egui::StrokeKind::Inside);
    match dnd_release::<ShelfDrag>(resp).map(|d| d.asset.clone()) {
        Some(Asset::Texture(name)) => Some(name),
        _ => None,
    }
}

fn mesh_map_color(map: BakedMap) -> Color32 {
    match map {
        BakedMap::Normal => Color32::from_rgb(128, 128, 255),
        BakedMap::WorldNormal => Color32::from_rgb(120, 200, 140),
        BakedMap::Id => Color32::from_rgb(220, 60, 60),
        BakedMap::AmbientOcclusion => Color32::from_gray(200),
        BakedMap::Curvature => Color32::from_gray(128),
        BakedMap::Position => Color32::from_rgb(120, 180, 90),
        BakedMap::Thickness => Color32::from_gray(160),
        BakedMap::Height => Color32::from_gray(128),
        BakedMap::BentNormal => Color32::from_rgb(140, 170, 230),
        BakedMap::Opacity => Color32::from_gray(230),
    }
}

// ---------------------------------------------------------------------------------------------
// Properties

/// The Properties tab title: what the panel edits.
pub(crate) fn properties_title(app: &TextureCraftApp) -> String {
    let project = &app.session.doc.project;
    let what = match (app.session.selected_layer().and_then(|id| project.layers.get(&id)), app.session.target) {
        // Polygon fill keeps its own properties whatever is selected
        _ if app.session.tool.tool == Tool::PolygonFill => tr(app.session.tool.tool.label()).to_string(),
        (Some(_), StackTarget::Effect(effect)) => project.effect(effect).map(|(_, found)| found.name.clone()).unwrap_or_default(),
        (Some(layer), StackTarget::Content) if matches!(layer.kind, LayerKind::Fill(_)) => tr("Fill").to_string(),
        (Some(layer), StackTarget::Content) if layer.is_group() => tr("Folder").to_string(),
        (Some(layer), StackTarget::Content) if layer.instance_source().is_some() => tr("Instance").to_string(),
        _ => tr(app.session.tool.tool.label()).to_string(),
    };
    format!("{} - {what}", tr("Properties"))
}

pub(crate) fn properties(app: &mut TextureCraftApp, ui: &mut Ui) {
    // A floating scroll bar would cover the right-aligned slider values and section reset icons
    ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
    egui::ScrollArea::vertical().id_salt("props").auto_shrink([false, false]).show(ui, |ui| properties_contents(app, ui));
}

/// What the Properties panel edits: polygon fill's own settings, the selected effect, fill, folder or instance, else the tool
pub(crate) fn properties_contents(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tokens = Tokens::get(ui.ctx());
    let selected = app.session.selected_layer().and_then(|id| app.session.doc.project.layers.get(&id).cloned());

    match (&selected, app.session.target) {
        // Polygon fill shows its own settings and says what it fills into, whatever is selected
        _ if app.session.tool.tool == Tool::PolygonFill => polygon_fill_properties(app, ui),
        (Some(layer), StackTarget::Effect(effect)) => effect_properties(app, ui, layer, effect),
        (Some(layer), StackTarget::Content) if matches!(layer.kind, LayerKind::Fill(_)) => fill_properties(app, ui, layer),
        (Some(layer), StackTarget::Content) if layer.is_group() => {
            ui.label(RichText::new(tr("A folder blends its children (pass-through). Set per-channel opacity in the layer rows.")).color(tokens.text_dim));
        }
        (Some(layer), StackTarget::Content) if layer.instance_source().is_some() => instance_properties(app, ui, layer),
        (Some(layer), StackTarget::Mask) => {
            height_mask_section(app, ui, layer);
            ui.separator();
            tool_properties(app, ui);
        }
        _ => tool_properties(app, ui),
    }
}

/// An instance draws its source: point there for edits.
fn instance_properties(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let tk = Tokens::get(ui.ctx());
    let Some(src) = layer.instance_source() else { return };
    let p = &app.session.doc.project;
    let name = p.layers.get(&src).map_or_else(|| tr("a missing layer").to_string(), |l| format!("\"{}\"", l.name));
    let set = p.locate(src).and_then(|(ts, _, _)| p.texture_sets.get(ts)).map(|t| t.name.clone()).unwrap_or_default();
    ui.label(caption(tr("Instance")));
    ui.label(
        RichText::new(format!(
            "{} {}",
            ic::LINK,
            trf("Draws {name} of texture set \"{set}\". Edits to the source show in every instance.", &[("name", &name), ("set", &set)])
        ))
        .color(tk.text_dim),
    );
    if ui.button(tr("Select source")).clicked() {
        app.run("layer.select", json!({"layer": src.0}));
    }
}

/// Mask with height: the layer's height shapes its mask transitions.
fn height_mask_section(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let id = layer.id.0;
    let hm = app.session.doc.project.content_layer(layer.id).map(|l| l.height_mask).unwrap_or_default();
    section(ui, "sec_height_mask", tr("Mask with height"), true, |ui| {
        let mut on = hm.enabled;
        if ui
            .checkbox(&mut on, tr("Combine with the layer's height"))
            .on_hover_text(tr("Partial mask values follow the layer's height: high parts show first"))
            .changed()
        {
            app.run("layer.set", json!({"layer": id, "heightMask": {"enabled": on}}));
        }
        ui.add_enabled_ui(hm.enabled, |ui| {
            let (mut contrast, mut offset) = (hm.contrast, hm.offset);
            if let Some(merge) = slider_edit(ui, tr("Contrast"), &mut contrast, 0.0..=16.0, 2) {
                app.run("layer.set", json!({"layer": id, "heightMask": {"contrast": contrast}, "merge": merge}));
            }
            if let Some(merge) = slider_edit(ui, tr("Height offset"), &mut offset, -1.0..=1.0, 2) {
                app.run("layer.set", json!({"layer": id, "heightMask": {"offset": offset}, "merge": merge}));
            }
        });
    });
}

fn tool_properties(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tokens = Tokens::get(ui.ctx());
    // Brush preview: a stroke and a sphere with the tool's material
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), 88.0), Sense::hover());
    let painter = ui.painter();
    painter.rect_filled(rect, 2.0, tokens.viewport);
    let half_width = rect.width() * 0.5;
    let brush = &app.session.tool.brush;
    let mut points = Vec::new();
    for step in 0..=40 {
        let fraction = step as f32 / 40.0;
        points.push(pos2(rect.left() + 14.0 + fraction * (half_width - 28.0), rect.center().y + (fraction * 9.0).sin() * 18.0));
    }

    let stroke_width = (brush.size / 0.25 * 60.0).clamp(2.0, 30.0);
    let alpha = (brush.flow * brush.opacity * 255.0) as u8;
    painter.add(egui::Shape::line(points, Stroke::new(stroke_width, Color32::from_white_alpha(alpha.max(30)))));
    let base_color = app.session.tool.values.get(&Channel::BaseColor).copied().unwrap_or([0.8; 3]);
    crate::thumbs::paint_sphere(painter, pos2(rect.left() + half_width * 1.5, rect.center().y), 36.0, base_color);
    ui.add_space(4.0);

    ui.horizontal(|ui| {
        let button_size = ui.spacing().interact_size.y;
        for (tab, icon, tip) in
            [(PropsTab::Brush, ic::PAINT_BRUSH, tr("Brush")), (PropsTab::Alpha, ic::CIRCLE_HALF, tr("Alpha")), (PropsTab::Material, ic::SPHERE, tr("Material"))]
        {
            if icon_button(ui, icon, tip, app.ui.props_tab == tab, button_size).clicked() {
                app.ui.props_tab = tab;
            }
        }
    });
    ui.separator();

    if app.session.quick_mask.editing {
        ui.label(RichText::new(tr("Editing the quick mask: strokes paint where painting is allowed. Press U to finish.")).color(tokens.text_dim));
        ui.separator();
    }

    let tool_section: Option<fn(&mut TextureCraftApp, &mut Ui)> = match app.session.tool.tool {
        Tool::Projection => Some(stencil_section),
        Tool::Clone => Some(clone_section),
        Tool::Smudge => Some(smudge_section),
        Tool::Path => Some(path_section),
        Tool::GeometryMask => Some(geometry_tool_section),
        _ => None,
    };
    if let Some(section) = tool_section {
        section(app, ui);
        ui.separator();
    }

    match app.ui.props_tab {
        PropsTab::Brush => brush_section(app, ui),
        PropsTab::Alpha => alpha_section(app, ui),
        PropsTab::Material => material_section(app, ui),
    }
}

fn stencil_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Stencil")));
    let names: Vec<String> = app.session.doc.resources.keys().cloned().collect();
    let current = app.session.tool.stencil.as_ref().map(|s| s.image.clone()).unwrap_or_else(|| tr("None").into());
    ui.horizontal(|ui| {
        egui::ComboBox::from_id_salt("stencil_img").width(170.0).selected_text(current.clone()).show_ui(ui, |ui| {
            if ui.selectable_label(app.session.tool.stencil.is_none(), tr("None")).clicked() {
                app.run("tool.stencil", json!({"clear": true}));
            }
            for n in &names {
                if ui.selectable_label(*n == current, n).clicked() {
                    app.run("tool.stencil", json!({"image": n}));
                }
            }
        });
        import_button(app, ui);
    });
    if names.is_empty() {
        ui.label(RichText::new(tr("Import an image (File > Import resources) to project it onto the mesh.")).color(tk.text_dim));
        return;
    }
    if let Some(st) = app.session.tool.stencil.as_mut() {
        slider(ui, tr("Scale"), &mut st.scale, 0.05..=4.0, 2);
        slider(ui, tr("Rotation"), &mut st.rotation, 0.0..=360.0, 0);
        slider(ui, tr("Overlay opacity"), &mut st.opacity, 0.0..=1.0, 2);
        ui.checkbox(&mut st.tile, tr("Tile"));
        ui.label(RichText::new(tr("Hold S: left-drag rotates, right-drag scales, middle-drag moves the stencil.")).size(10.5).color(tk.text_dim));
    }
}

fn import_button(app: &mut TextureCraftApp, ui: &mut Ui) {
    if ui.small_button(tr("Import...")).clicked() {
        let ctx = ui.ctx().clone();
        crate::menus::invoke(app, &ctx, "resource.import");
    }
}

fn clone_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Clone")));
    ui.horizontal(|ui| {
        ui.label(tr("Copy from"));
        for (mode, label, tip, id) in [
            (CloneFrom::Composite, tr("All layers"), tr("Copy what the texture set shows"), "composite"),
            (CloneFrom::Layer, tr("Current layer"), tr("Copy the selected layer alone (its empty parts copy nothing)"), "layer"),
        ] {
            if chip(ui, label, app.session.tool.clone_from == mode).on_hover_text(tip).clicked() {
                app.run("tool.brush", json!({"cloneSource": id}));
            }
        }
    });
    let Some(src) = app.session.tool.clone_source else {
        ui.label(RichText::new(tr("Hold V and click the mesh to set the source, then paint elsewhere to copy it.")).color(tk.text_dim));
        return;
    };
    ui.horizontal(|ui| {
        ui.label(trf("Source  U {u}  V {v}", &[("u", &format!("{:.3}", src.uv[0])), ("v", &format!("{:.3}", src.uv[1]))]));
        if ui.small_button(tr("Clear")).clicked() {
            app.run("tool.cloneSource", json!({"clear": true}));
        }
    });
    let hint = if src.offset.is_some() {
        tr("Aligned: the offset set by the first stroke is kept. V + click sets a new source.")
    } else {
        tr("The first stroke sets the offset from the source.")
    };
    ui.label(RichText::new(hint).size(10.5).color(tk.text_dim));
}

fn smudge_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Smudge")));
    let mut v = app.session.tool.smudge_strength * 100.0;
    if slider(ui, tr("Strength"), &mut v, 0.0..=100.0, 0).changed() {
        app.session.tool.smudge_strength = (v / 100.0).clamp(0.0, 1.0);
    }
    ui.label(RichText::new(tr("Drags the layer stack's values along the stroke into the selected layer.")).size(10.5).color(tk.text_dim));
}

fn path_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Path")));
    ui.horizontal(|ui| crate::toolbar::path_controls(app, ui));
    let n = app.session.tool.path.points.len();
    let hint = trf(
        "{n} control points. Click the mesh (or the 2D view) to add points and drag them to adjust; click the first point to close the path. Enter paints it with the brush (a closed, filled path fills its inside), Esc cancels.",
        &[("n", &n)],
    );
    ui.label(RichText::new(hint).size(10.5).color(tk.text_dim));
}

/// Width of the label column in polygon fill's properties
const POLYGON_FILL_LABEL_WIDTH: f32 = 108.0;

/// Polygon fill's properties: how it selects and fills, its blending, then what it fills with (material, mask gray, or why it can't fill)
fn polygon_fill_properties(app: &mut TextureCraftApp, ui: &mut Ui) {
    use texturecraft_engine::FillHitTest;

    let button_size = ui.spacing().interact_size.y;
    ui.label(caption(&tr("Polygon fill").to_uppercase()));
    property_row(ui, tr("Selection mode"), tr("A click fills what is under it; a dragged rectangle fills what it selects as it grows"), |ui| {
        icon_button(ui, ic::SELECTION, tr("Rectangle fill"), true, button_size);
    });
    property_row(
        ui,
        tr("Fill mode"),
        tr("Triangle fills single triangles, Polygon whole quads and n-gons, Mesh the connected geometry, UV island the UV islands"),
        |ui| crate::toolbar::fill_mode_buttons(app, ui),
    );
    property_row(ui, tr("Hit-test"), tr("How a rectangle selects polygons"), |ui| {
        for (hit_test, icon, tooltip) in [
            (FillHitTest::Intersection, ic::INTERSECT_SQUARE, tr("Intersection: fill polygons the rectangle touches")),
            (FillHitTest::ContainedCenter, ic::SELECTION_FOREGROUND, tr("Contained center: fill polygons whose center is in the rectangle")),
        ] {
            if icon_button(ui, icon, tooltip, app.session.tool.fill_hit_test == hit_test, button_size).clicked() {
                app.session.tool.fill_hit_test = hit_test;
            }
        }
    });
    ui.horizontal(|ui| {
        ui.checkbox(&mut app.session.tool.fill_visible_only, tr("Visible polygons only"));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| info_icon(ui, tr("Fill only polygons the camera sees (3D view)")));
    });
    section(ui, "sec_polygon_fill_blending", tr("Blending"), false, |ui| {
        let mut opacity = app.session.tool.brush.opacity * 100.0;
        if slider(ui, tr("Opacity"), &mut opacity, 0.0..=100.0, 0).changed() {
            app.session.tool.brush.opacity = (opacity / 100.0).clamp(0.0, 1.0);
        }
    });
    ui.separator();

    match polygon_fill_paints_mask(app) {
        Ok(false) => {
            ui.label(caption(&tr("Material").to_uppercase()));
            material_controls(app, ui);
        }
        Ok(true) => {
            ui.label(caption(&tr("Grayscale").to_uppercase()));
            let mut gray = app.session.tool.mask_value;
            if slider(ui, tr("Value"), &mut gray, 0.0..=1.0, 2).changed() {
                app.session.tool.mask_value = gray;
            }
        }
        Err(note) => {
            ui.label(RichText::new(note).color(Tokens::get(ui.ctx()).text_dim));
        }
    }
}

/// One polygon fill setting: its label in a fixed column, its `controls`, and an info icon with `info` at the right edge
fn property_row(ui: &mut Ui, label: &str, info: &str, controls: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        let label_width = ui.label(label).rect.width();
        ui.add_space((POLYGON_FILL_LABEL_WIDTH - label_width - ui.spacing().item_spacing.x).max(0.0));
        controls(ui);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| info_icon(ui, info));
    });
}

/// A dim info icon explaining a setting on hover
fn info_icon(ui: &mut Ui, text: &str) {
    ui.label(RichText::new(ic::INFO).color(Tokens::get(ui.ctx()).text_dim)).on_hover_text(text);
}

/// What a polygon fill paints into now: Ok(true) a mask (gray), Ok(false) layer content (material), or a note on why it can't fill
fn polygon_fill_paints_mask(app: &TextureCraftApp) -> Result<bool, String> {
    if app.session.quick_mask.editing {
        return Err(tr("Editing the quick mask: fills paint where painting is allowed. Press U to finish.").to_string());
    }

    let project = &app.session.doc.project;
    let Some(layer) = app.session.selected_layer().and_then(|id| project.layers.get(&id)) else {
        return Err(tr("Select a paint layer or a layer's mask to fill").to_string());
    };

    match app.session.target {
        StackTarget::Mask => Ok(true),
        StackTarget::Effect(_) => Ok(!targets_content(app)),
        StackTarget::Content => {
            let content = project.content_layer(layer.id).unwrap_or(layer);
            if matches!(content.kind, LayerKind::Paint) {
                Ok(false)
            } else {
                Err(trf("\"{layer}\" can't be filled: select a paint layer or a layer's mask", &[("layer", &layer.name)]))
            }
        }
    }
}

/// Font size of the dim hint line under the Brush caption
const HINT_TEXT_SIZE: f32 = 10.5;

fn brush_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let brush = &mut app.session.tool.brush;
    let defaults = BrushSettings::default();
    ui.label(caption(tr("Brush")));
    let hint = tr("Ctrl+right-drag in a view sets the size and hardness, Ctrl+left-drag the flow and angle.");
    ui.label(RichText::new(hint).size(HINT_TEXT_SIZE).color(Tokens::get(ui.ctx()).text_dim));

    // Each section's reset puts back only the fields it shows
    let size_fields = |settings: &BrushSettings| (settings.size, settings.min_size, settings.screen_space);
    if section_with_reset(ui, "sec_size", tr("Size"), true, size_fields(brush) == size_fields(&defaults), |ui| {
        let mut size_percent = brush.size / MAX_BRUSH_SIZE * 100.0;
        if slider(ui, tr("Size"), &mut size_percent, 0.05..=100.0, 1).changed() {
            brush.size = (size_percent / 100.0 * MAX_BRUSH_SIZE).clamp(MIN_BRUSH_SIZE, MAX_BRUSH_SIZE);
        }

        let mut min_size_percent = brush.min_size * 100.0;
        if slider(ui, tr("Minimum size (%)"), &mut min_size_percent, 0.0..=100.0, 0).changed() {
            brush.min_size = min_size_percent / 100.0;
        }

        ui.checkbox(&mut brush.screen_space, tr("Size in screen space"));
    }) {
        (brush.size, brush.min_size, brush.screen_space) = size_fields(&defaults);
    }

    let flow_fields = |settings: &BrushSettings| (settings.flow, settings.opacity, settings.spacing, settings.hardness);
    if section_with_reset(ui, "sec_flow", tr("Flow and opacity"), true, flow_fields(brush) == flow_fields(&defaults), |ui| {
        let mut flow_percent = brush.flow * 100.0;
        if slider(ui, tr("Flow"), &mut flow_percent, 0.0..=100.0, 0).changed() {
            brush.flow = flow_percent / 100.0;
        }

        let mut opacity_percent = brush.opacity * 100.0;
        if slider(ui, tr("Stroke opacity"), &mut opacity_percent, 0.0..=100.0, 0).changed() {
            brush.opacity = opacity_percent / 100.0;
        }

        let mut spacing_percent = brush.spacing * 100.0;
        if slider(ui, tr("Spacing"), &mut spacing_percent, 1.0..=1000.0, 0).changed() {
            brush.spacing = spacing_percent / 100.0;
        }

        let mut hardness_percent = brush.hardness * 100.0;
        if slider(ui, tr("Hardness"), &mut hardness_percent, 0.0..=99.9, 0).changed() {
            brush.hardness = hardness_percent / 100.0;
        }
    }) {
        (brush.flow, brush.opacity, brush.spacing, brush.hardness) = flow_fields(&defaults);
    }

    let angle_fields = |settings: &BrushSettings| (settings.angle, settings.follow_path);
    if section_with_reset(ui, "sec_angle", tr("Angle"), false, angle_fields(brush) == angle_fields(&defaults), |ui| {
        slider(ui, tr("Angle"), &mut brush.angle, -180.0..=180.0, 0);
        ui.checkbox(&mut brush.follow_path, tr("Follow path"));
    }) {
        (brush.angle, brush.follow_path) = angle_fields(&defaults);
    }

    let jitter_fields = |settings: &BrushSettings| (settings.jitter_size, settings.jitter_angle, settings.jitter_flow, settings.jitter_position);
    if section_with_reset(ui, "sec_jitter", tr("Jitters"), false, jitter_fields(brush) == jitter_fields(&defaults), |ui| {
        let jitters = [(tr("Size jitter"), &mut brush.jitter_size), (tr("Angle jitter"), &mut brush.jitter_angle), (tr("Flow jitter"), &mut brush.jitter_flow)];
        for (label, jitter) in jitters {
            let mut jitter_percent = *jitter * 100.0;
            if slider(ui, label, &mut jitter_percent, 0.0..=100.0, 0).changed() {
                *jitter = jitter_percent / 100.0;
            }
        }

        let mut position_jitter_percent = brush.jitter_position * 100.0;
        if slider(ui, tr("Position jitter"), &mut position_jitter_percent, 0.0..=400.0, 0).changed() {
            brush.jitter_position = position_jitter_percent / 100.0;
        }
    }) {
        (brush.jitter_size, brush.jitter_angle, brush.jitter_flow, brush.jitter_position) = jitter_fields(&defaults);
    }

    let alignment_fields = |settings: &BrushSettings| (settings.alignment, settings.backface_culling, settings.angle_fade, settings.depth);
    if section_with_reset(ui, "sec_align", tr("Alignment and culling"), false, alignment_fields(brush) == alignment_fields(&defaults), |ui| {
        use texturecraft_engine::paint::Alignment;
        let alignments = [(Alignment::Tangent, tr("Tangent | Wrap")), (Alignment::Camera, tr("Camera"))];
        let shown = alignments.iter().find(|(alignment, _)| *alignment == brush.alignment).map_or("", |(_, label)| label);
        combo(ui, tr("Alignment"), shown, |ui| {
            for (alignment, label) in alignments {
                ui.selectable_value(&mut brush.alignment, alignment, label);
            }
        });

        ui.checkbox(&mut brush.backface_culling, tr("Backface culling"));
        slider(ui, tr("Angle fade from"), &mut brush.angle_fade.0, 0.0..=90.0, 0);
        slider(ui, tr("Angle fade to"), &mut brush.angle_fade.1, 0.0..=90.0, 0);
        slider(ui, tr("Stamp depth"), &mut brush.depth, 0.05..=4.0, 2);
    }) {
        (brush.alignment, brush.backface_culling, brush.angle_fade, brush.depth) = alignment_fields(&defaults);
    }

    if section_with_reset(ui, "sec_particles", tr("Particles"), false, brush.particles == defaults.particles, |ui| {
        let particles = &mut brush.particles;
        ui.checkbox(&mut particles.enabled, tr("Particle brush (slides down the surface)"));

        let mut count = particles.count as f32;
        if slider(ui, tr("Count per stamp"), &mut count, 1.0..=16.0, 0).changed() {
            particles.count = count as u32;
        }

        let mut lifetime_steps = particles.lifetime as f32;
        if slider(ui, tr("Lifetime (steps)"), &mut lifetime_steps, 1.0..=128.0, 0).changed() {
            particles.lifetime = lifetime_steps as u32;
        }

        slider(ui, tr("Gravity"), &mut particles.gravity, 0.0..=2.0, 2);
    }) {
        brush.particles = defaults.particles;
    }

    if section_with_reset(ui, "sec_lazy", tr("Lazy mouse"), false, brush.lazy == defaults.lazy, |ui| {
        slider(ui, tr("Lazy mouse distance"), &mut brush.lazy, 0.0..=200.0, 0);
    }) {
        brush.lazy = defaults.lazy;
    }

    *brush = brush.sanitized();
}

/// One alpha in the picker: a 52×64 tile with the shape drawn by `paint` and a label.
fn alpha_tile(ui: &mut Ui, on: bool, label: &str, paint: impl FnOnce(&egui::Painter, Rect)) -> egui::Response {
    let tk = Tokens::get(ui.ctx());
    let (rect, resp) = ui.allocate_exact_size(vec2(52.0, 64.0), Sense::click());
    let p = ui.painter();
    p.rect_filled(rect, 3.0, tk.viewport);
    if on {
        p.rect_stroke(rect, 3.0, Stroke::new(1.0, tk.accent), egui::StrokeKind::Inside);
    }
    paint(p, Rect::from_center_size(rect.center() - vec2(0.0, 7.0), vec2(40.0, 40.0)));
    let name = elided(p, label, 9.5, tk.viewport_text, rect.width() - 4.0);
    p.galley(rect.center_bottom() - vec2(name.size().x * 0.5, 6.0 + name.size().y * 0.5), name, tk.viewport_text);
    resp.on_hover_text(label)
}

fn alpha_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    let tk = Tokens::get(ui.ctx());
    ui.horizontal(|ui| {
        ui.label(caption(tr("Alpha")));
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| import_button(app, ui));
    });
    let brush = &app.session.tool.brush;
    let (current, image, hardness) = (brush.alpha, brush.alpha_image.clone(), brush.hardness);
    let library = if image.is_none() { brush.alpha_library.as_deref().and_then(alphas::find) } else { None };
    let mut pick = None;
    let zone = ui.horizontal_wrapped(|ui| {
        for a in BrushAlpha::ALL {
            let on = image.is_none() && library.is_none() && current == a;
            if alpha_tile(ui, on, tr(a.label()), |p, r| crate::thumbs::paint_alpha(p, r, a, hardness)).clicked() {
                pick = Some(json!(a.label()));
            }
        }
        // The library alpha in use (picked from the shelf's Alphas tab).
        if let Some(a) = library {
            let tex = app.thumbs.alpha(ui.ctx(), a);
            alpha_tile(ui, true, &a.label, |p, r| {
                if let Some(t) = tex {
                    p.image(t.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
                }
            });
        }
        // Imported images, sampled as grayscale.
        for name in app.session.doc.resources.keys() {
            let Some(img) = app.session.images.get(name) else { continue };
            let tex = app.thumbs.image(ui.ctx(), name, img);
            let on = image.as_deref() == Some(name.as_str());
            if alpha_tile(ui, on, name, |p, r| {
                p.image(tex.id(), r, Rect::from_min_max(pos2(0.0, 0.0), pos2(1.0, 1.0)), Color32::WHITE);
            })
            .clicked()
            {
                pick = Some(json!({ "image": name }));
            }
        }
    });
    if let Some(alpha) = pick {
        app.run("tool.brush", json!({ "alpha": alpha }));
    }
    ui.label(RichText::new(tr("More in Assets > Alphas: click one, or drag it here")).size(10.0).color(tk.text_dim));
    crate::shelf::drop_target(app, ui, &zone.response, crate::shelf::DropZone::Brush);
    section(ui, "sec_text", tr("Text"), true, |ui| text_section(app, ui));
}

/// Text drawn in a font, as the brush alpha or a decal (`resource.text`).
fn text_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    if app.ui.text.fonts.is_empty()
        && let Some(f) = app.run("text.fonts", Value::Null)
    {
        let installed = f["installed"].as_array().into_iter().flatten().map(|v| {
            let styles = v["styles"].as_array().into_iter().flatten().filter_map(Value::as_str).map(str::to_string).collect();
            (v["family"].as_str().unwrap_or_default().to_string(), styles)
        });
        let built_in = f["builtIn"].as_array().into_iter().flatten().filter_map(Value::as_str).map(|n| (n.to_string(), Vec::new()));
        app.ui.text.fonts = installed.chain(built_in).collect();
    }
    // A decal shows the text in the tool's base color; an alpha must stay white.
    let color = app.session.tool.values.get(&Channel::BaseColor).copied().unwrap_or([1.0; 3]);
    let t = &mut app.ui.text;
    ui.add(egui::TextEdit::multiline(&mut t.text).desired_rows(2).desired_width(f32::INFINITY).hint_text(tr("Text")));
    let shown = if t.font.is_empty() { tr("Default").to_string() } else { t.font.clone() };
    combo(ui, tr("Font"), shown, |ui| {
        if ui.selectable_label(t.font.is_empty(), tr("Default")).clicked() {
            (t.font, t.style) = (String::new(), String::new());
        }
        for (family, _) in &t.fonts {
            if ui.selectable_label(t.font == *family, family).clicked() {
                (t.font, t.style) = (family.clone(), String::new());
            }
        }
    });
    let styles = t.fonts.iter().find(|(f, _)| *f == t.font).map(|(_, s)| s.clone()).unwrap_or_default();
    if styles.len() > 1 {
        let shown = if t.style.is_empty() { tr("Default").to_string() } else { t.style.clone() };
        combo(ui, tr("Style"), shown, |ui| {
            for s in styles {
                ui.selectable_value(&mut t.style, s.clone(), s);
            }
        });
    }
    let mut params = json!({"text": t.text, "font": (!t.font.is_empty()).then_some(&t.font), "style": (!t.style.is_empty()).then_some(&t.style)});
    let ready = !t.text.trim().is_empty();
    ui.horizontal_wrapped(|ui| {
        let alpha = ui.add_enabled(ready, egui::Button::new(tr("Use as brush alpha"))).clicked();
        let decal = ui.add_enabled(ready, egui::Button::new(tr("Place as decal"))).clicked();
        if decal {
            params["color"] = json!(color);
        }
        if (alpha || decal)
            && let Some(name) = app.run("resource.text", params).and_then(|r| r["name"].as_str().map(str::to_string))
        {
            match alpha {
                true => app.run("tool.brush", json!({"alpha": {"image": name}})),
                false => app.run("layer.add.decal", json!({"image": name})),
            };
        }
    });
}

/// Tooltip of the material channel chips.
pub(crate) const SOLO_HINT: &str = "Click: turn this channel on or off. Alt+click: this channel only";

/// The tool's material under a "Material" heading
fn material_section(app: &mut TextureCraftApp, ui: &mut Ui) {
    ui.label(caption(tr("Material")));
    material_controls(app, ui);
}

/// The tool's material: channel chips, then each enabled channel's value (or the mask gray while painting a mask)
fn material_controls(app: &mut TextureCraftApp, ui: &mut Ui) {
    use texturecraft_engine::core::ChannelKind;

    let tokens = Tokens::get(ui.ctx());
    let channels = app.session.doc.project.texture_set(app.session.ts()).map(|set| set.channel_list()).unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        for channel in &channels {
            let on = app.session.tool.enabled.contains(channel);
            if chip(ui, tr(channel.short()), on).on_hover_text(tr(SOLO_HINT)).clicked() {
                let solo = ui.input(|input| input.modifiers.alt);
                app.run("tool.material", json!({"channel": channel.id(), "enabled": solo || !on, "solo": solo}));
            }
        }
    });
    ui.add_space(4.0);

    if !targets_content(app) {
        let mut gray = app.session.tool.mask_value;
        if slider(ui, tr("Mask value (gray)"), &mut gray, 0.0..=1.0, 2).changed() {
            app.session.tool.mask_value = gray;
        }
        return;
    }

    for channel in channels {
        if !app.session.tool.enabled.contains(&channel) {
            continue;
        }
        let mut value = app.session.tool.values.get(&channel).copied().unwrap_or(channel.default_value());
        match channel.kind() {
            ChannelKind::Color => {
                ui.horizontal(|ui| {
                    ui.label(tr(channel.label()));
                    if color_swatch(ui, &format!("tool_{}", channel.id()), &mut value, ui.available_width() - 4.0) {
                        app.session.tool.values.insert(channel, value);
                    }
                });
            }
            ChannelKind::NormalMap => {
                ui.label(RichText::new(tr("Normal: painting uses the flat normal")).color(tokens.text_dim));
            }
            ChannelKind::Signed => {
                if slider(ui, tr(channel.label()), &mut value[0], -1.0..=1.0, 2).changed() {
                    app.session.tool.values.insert(channel, [value[0]; 3]);
                }
            }
            ChannelKind::Scalar => {
                if slider(ui, tr(channel.label()), &mut value[0], 0.0..=1.0, 2).changed() {
                    app.session.tool.values.insert(channel, [value[0]; 3]);
                }
            }
        }
    }
    if app.session.tool.tool == Tool::Eraser {
        ui.label(RichText::new(tr("The eraser removes paint from the enabled channels.")).color(tokens.text_dim));
    }
}

fn fill_properties(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer) {
    let LayerKind::Fill(fill) = &layer.kind else { return };
    if let Some(link) = &fill.link {
        link_properties(app, ui, layer.id, link);
        ui.add_space(6.0);
    }
    let on: Vec<Channel> = layer.channels.keys().copied().collect();
    ui.label(caption(tr("Fill")));
    fill_editor(app, ui, fill, &json!({"layer": layer.id.0}), &on, layer.id);
}

/// A MaterialCraft material's presets and exposed inputs; edits store the value and re-cook the
/// maps on a worker thread (desktop only: cooking runs MaterialCraft's command line).
fn link_properties(app: &mut TextureCraftApp, ui: &mut Ui, layer: LayerId, link: &MaterialLink) {
    let tk = Tokens::get(ui.ctx());
    let base = json!({"layer": layer.0, "cook": false});
    ui.label(caption(tr("MaterialCraft material")));
    ui.horizontal(|ui| {
        ui.label(RichText::new(&link.name).strong());
        ui.label(RichText::new(trf("Cooked at {size} px", &[("size", &link.size)])).size(10.0).color(tk.text_dim));
        if app.link_cooks.iter().any(|c| c.layer == layer) {
            ui.spinner();
        }
    });
    let editable = cfg!(not(target_arch = "wasm32"));
    ui.add_enabled_ui(editable, |ui| {
        if !link.presets.is_empty() {
            combo(ui, tr("Preset"), "", |ui| {
                for p in &link.presets {
                    if ui.selectable_label(false, &p.name).clicked() && app.run("material.link.preset", with(&base, json!({"name": p.name}))).is_some() {
                        app.cook_link(layer);
                    }
                }
            });
        }
        let mut group = "";
        for p in &link.params {
            if p.group != group && !p.group.is_empty() {
                ui.label(RichText::new(p.group.to_uppercase()).size(10.0).strong().color(tk.text_dim));
            }
            group = &p.group;
            if let Some(v) = link_param_editor(ui, layer, p)
                && app.run("material.link.set", with(&base, json!({"id": p.id, "value": v}))).is_some()
            {
                app.cook_link(layer);
            }
        }
    });
    if !editable {
        ui.label(RichText::new(tr("Parameters change in the desktop app, with MaterialCraft installed")).size(10.0).color(tk.text_dim));
    }
    #[cfg(not(target_arch = "wasm32"))]
    ui.horizontal(|ui| {
        if ui.button(tr("Re-cook")).clicked() {
            app.cook_link(layer);
        }
        if ui.button(tr("Open in MaterialCraft")).clicked() {
            let path = std::env::temp_dir().join(format!("{}.mcraft", texturecraft_engine::formats::export::sanitize_file_stem(&link.name)));
            let opened = app
                .session
                .link_package(layer)
                .and_then(|b| std::fs::write(&path, b.as_slice()).map_err(|e| texturecraft_engine::EngineError::Other(e.to_string())))
                .and_then(|()| texturecraft_engine::link_cmds::open_in_materialcraft(&path));
            match opened {
                Ok(()) => app.toast(tr("Opened in MaterialCraft: use File > Send to TextureCraft there to bring your changes back"), false),
                Err(e) => app.toast(e.to_string(), true),
            }
        }
    });
}

/// The editor of one linked-material input; returns its new value when an edit completes.
fn link_param_editor(ui: &mut Ui, layer: LayerId, p: &LinkParam) -> Option<Value> {
    let t = p.default.get("t").and_then(Value::as_str).unwrap_or("");
    let v = p.value.get("v").unwrap_or(&Value::Null);
    let num = |x: &Value| x.as_f64().unwrap_or(0.0) as f32;
    let (lo, hi) = if p.max > p.min { (p.min, p.max) } else { (0.0, 1.0) };
    // A slider edit completes when the drag ends, on a click or a typed value.
    let edit = |ui: &mut Ui, label: &str, x: f32, decimals: usize| {
        let mut y = x;
        let r = slider(ui, label, &mut y, lo.min(x)..=hi.max(x), decimals);
        (r.drag_stopped() || (!r.dragged() && y != x)).then_some(y)
    };
    match t {
        "bool" => {
            let mut b = v.as_bool().unwrap_or(num(v) != 0.0);
            ui.checkbox(&mut b, &p.label).changed().then(|| json!(b))
        }
        "enum" if !p.options.is_empty() => {
            let cur = num(v).max(0.0) as usize;
            let mut pick = None;
            combo(ui, &p.label, p.options.get(cur).map_or("", String::as_str), |ui| {
                for (i, o) in p.options.iter().enumerate() {
                    if ui.selectable_label(i == cur, o).clicked() {
                        pick = Some(json!(i));
                    }
                }
            });
            pick
        }
        "float" | "int" | "enum" => edit(ui, &p.label, num(v), if t == "float" { 2 } else { 0 }).map(|y| json!(y)),
        "color" => {
            let a = v.as_array().map(|a| a.iter().map(num).collect::<Vec<f32>>()).unwrap_or_default();
            let mut rgb = [0, 1, 2].map(|i| a.get(i).copied().unwrap_or(1.0));
            let mut out = None;
            ui.horizontal(|ui| {
                ui.label(&p.label);
                if color_swatch(ui, &format!("link_{}_{}", layer.0, p.id), &mut rgb, ui.available_width() - 4.0) {
                    out = Some(json!([rgb[0], rgb[1], rgb[2], a.get(3).copied().unwrap_or(1.0)]));
                }
            });
            out
        }
        _ => {
            // Vectors: one slider per component.
            let mut a: Vec<f32> = v.as_array().map(|a| a.iter().map(num).collect()).unwrap_or_default();
            let decimals = if t == "int2" { 0 } else { 2 };
            let mut changed = false;
            for (x, axis) in a.iter_mut().zip(["x", "y", "z", "w"]) {
                if let Some(y) = edit(ui, &format!("{} {axis}", p.label), *x, decimals) {
                    *x = y;
                    changed = true;
                }
            }
            changed.then(|| json!(a))
        }
    }
}

/// `base` with the fields of `extra` added.
fn with(base: &Value, extra: Value) -> Value {
    let mut v = base.clone();
    if let (Some(o), Value::Object(e)) = (v.as_object_mut(), extra) {
        o.extend(e);
    }
    v
}

/// Anchor points below layer `reader` in its texture set: (effect id, "layer / anchor" label).
fn anchors_below(app: &TextureCraftApp, reader: LayerId) -> Vec<(LayerId, String)> {
    let p = &app.session.doc.project;
    let Some((ts, _, _)) = p.locate(reader) else { return Vec::new() };
    p.walk(ts)
        .into_iter()
        .map(|(id, _)| id)
        .take_while(|id| *id != reader)
        .filter_map(|id| p.layers.get(&id))
        .flat_map(|l| l.content_effects.iter().filter(|e| matches!(e.kind, EffectKind::Anchor)).map(move |e| (e.id, format!("{} / {}", l.name, e.name))))
        .collect()
}

fn anchor_label(app: &TextureCraftApp, anchor: LayerId, channel: Channel) -> String {
    let name = app.session.doc.project.effect(anchor).map(|(_, e)| e.name.clone()).unwrap_or_else(|| tr("missing").into());
    trf("Anchor: {name} ({channel})", &[("name", &name), ("channel", &tr(channel.label()))])
}

/// Space above a fill's Material caption, setting it apart from the projection
const FILL_MATERIAL_GAP: f32 = 6.0;

/// Space between a fill's channel chips and its channel sources
const FILL_CHANNELS_GAP: f32 = 4.0;

/// Projection and sources of a fill layer or fill effect (`base`) for its channels `on`; `reader` owns it
fn fill_editor(app: &mut TextureCraftApp, ui: &mut Ui, fill: &Fill, base: &Value, on: &[Channel], reader: LayerId) {
    let salt = base.to_string();
    if let Projection::Decal(decal) = &fill.projection {
        decal_editor(app, ui, decal, reader);
    } else {
        projection_editor(app, ui, fill, base, &salt);
    }

    ui.add_space(FILL_MATERIAL_GAP);
    ui.label(caption(tr("Material")));
    let channels = app.session.doc.project.texture_set(app.session.ts()).map(|texture_set| texture_set.channel_list()).unwrap_or_default();
    ui.horizontal_wrapped(|ui| {
        for channel in &channels {
            let enabled = on.contains(channel);
            if chip(ui, tr(channel.short()), enabled).on_hover_text(tr(SOLO_HINT)).clicked() {
                let solo = ui.input(|input| input.modifiers.alt);
                app.run("layer.fill.set", with(base, json!({"channel": channel.id(), "enabled": solo || !enabled, "solo": solo})));
            }
        }
    });

    ui.add_space(FILL_CHANNELS_GAP);
    let anchors = anchors_below(app, reader);
    for channel in channels {
        if on.contains(&channel) {
            fill_channel(app, ui, fill, base, channel, &anchors, &salt);
        }
    }
}

/// Slider range of a fill's scale or tiling
const FILL_SCALE_RANGE: std::ops::RangeInclusive<f32> = 0.01..=20.0;

/// Slider range of a UV fill's rotation, a full turn
const FILL_ROTATION_DEGREES_RANGE: std::ops::RangeInclusive<f32> = 0.0..=360.0;

/// Tri-planar hardness, the exponent weighting the three projections; `layer.fill.projection` clamps to the same range
const TRIPLANAR_HARDNESS_RANGE: std::ops::RangeInclusive<f32> = 1.0..=64.0;

/// A fill's projection kind, axis and hardness, then its collapsible UV transformations
fn projection_editor(app: &mut TextureCraftApp, ui: &mut Ui, fill: &Fill, base: &Value, salt: &str) {
    let projection_label = fill.projection.label();
    combo(ui, tr("Projection"), tr(projection_label), |ui| {
        for (id, label) in [
            ("uv", n_("UV projection")),
            ("triplanar", n_("Tri-planar projection")),
            ("planar", n_("Planar projection")),
            ("spherical", n_("Spherical projection")),
            ("cylindrical", n_("Cylindrical projection")),
        ] {
            if ui.selectable_label(projection_label == label, tr(label)).clicked() {
                app.run("layer.fill.projection", with(base, json!({"projection": id})));
            }
        }
    });

    // Each edit sends only the fields it changed; the command keeps the others
    let mut edit = |fields: Value| {
        app.run("layer.fill.projection", with(base, fields));
    };
    if let Projection::Planar { axis, .. } | Projection::Cylindrical { axis, .. } = &fill.projection
        && let Some(picked_axis) = axis_combo(ui, *axis)
    {
        edit(json!({"axis": picked_axis}));
    }
    if let Projection::Triplanar { sharpness, .. } = &fill.projection {
        let mut hardness = *sharpness;
        if let Some(merge) = slider_edit(ui, tr("Hardness"), &mut hardness, TRIPLANAR_HARDNESS_RANGE, 1) {
            edit(json!({"sharpness": hardness, "merge": merge}));
        }
    }

    // One id per fill so each keeps its own open or closed state
    section(ui, &format!("sec_fill_transform{salt}"), tr("UV transformations"), true, |ui| {
        projection_transform(ui, &fill.projection, &mut edit);
    });
}

/// Scale plus what the projection kind adds: UV rotation and offset, or a world offset
fn projection_transform(ui: &mut Ui, projection: &Projection, edit: &mut impl FnMut(Value)) {
    let mut scale = match projection {
        Projection::Uv { scale, .. } => scale[0],
        Projection::Triplanar { scale, .. } | Projection::Planar { scale, .. } | Projection::Spherical { scale } | Projection::Cylindrical { scale, .. } => {
            *scale
        }
        Projection::Decal(_) => return,
    };
    if let Some(merge) = slider_edit(ui, tr("Scale / tiling"), &mut scale, FILL_SCALE_RANGE, 2) {
        edit(json!({"scale": scale, "merge": merge}));
    }

    match projection {
        Projection::Uv { rotation, offset, .. } => {
            let mut rotation_degrees = *rotation;
            if let Some(merge) = slider_edit(ui, tr("Rotation"), &mut rotation_degrees, FILL_ROTATION_DEGREES_RANGE, 0) {
                edit(json!({"rotation": rotation_degrees, "merge": merge}));
            }
            if let Some((edited_offset, merge)) = offset_sliders(ui, ["U", "V"], *offset) {
                edit(json!({"offset": edited_offset, "merge": merge}));
            }
        }
        Projection::Triplanar { offset, .. } | Projection::Planar { offset, .. } => {
            if let Some((edited_offset, merge)) = offset_sliders(ui, ["X", "Y", "Z"], *offset) {
                edit(json!({"offset": edited_offset, "merge": merge}));
            }
        }
        Projection::Spherical { .. } | Projection::Cylindrical { .. } | Projection::Decal(_) => {}
    }
}

/// A slider per offset component ("Offset U", …): the whole offset once one is edited, with
/// [`slider_edit`]'s `merge`.
fn offset_sliders<const N: usize>(ui: &mut Ui, axes: [&str; N], mut v: [f32; N]) -> Option<([f32; N], bool)> {
    let mut edit = None;
    for (axis, x) in axes.iter().zip(v.iter_mut()) {
        edit = edit.or(slider_edit(ui, &trf("Offset {axis}", &[("axis", axis)]), x, -1.0..=1.0, 2));
    }
    edit.map(|merge| (v, merge))
}

/// The axis a planar or cylindrical projection runs along: the one picked, if any.
fn axis_combo(ui: &mut Ui, axis: u8) -> Option<u8> {
    let mut picked = None;
    combo(ui, tr("Axis"), ["X", "Y", "Z"].get(usize::from(axis)).copied().unwrap_or("Z"), |ui| {
        for (i, name) in (0u8..).zip(["X", "Y", "Z"]) {
            if ui.selectable_label(i == axis, name).clicked() && i != axis {
                picked = Some(i);
            }
        }
    });
    picked
}

/// A decal's frame: place it with a click in the 3D view, then size, turn and shift it.
fn decal_editor(app: &mut TextureCraftApp, ui: &mut Ui, d: &Decal, layer: LayerId) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Decal")));
    let placing = app.placing_decal == Some(layer);
    let label = format!("{} {}", ic::CROSSHAIR, if placing { tr("Click the mesh to place the decal") } else { tr("Place on the surface") });
    if ui.selectable_label(placing, label).on_hover_text(tr("The next click in the 3D view moves the decal there")).clicked() {
        app.placing_decal = if placing { None } else { Some(layer) };
    }
    let (lo, hi) = app.session.mesh.bounds();
    let diag = (hi - lo).length().max(1e-3);
    let mut v = [d.size[0], d.size[1], d.rotation, d.offset[0], d.offset[1], d.depth];
    let rows = [
        (tr("Width"), diag * 0.01..=diag * 2.0),
        (tr("Height"), diag * 0.01..=diag * 2.0),
        (tr("Rotation (degrees)"), 0.0..=360.0),
        (tr("Offset right"), -diag..=diag),
        (tr("Offset up"), -diag..=diag),
        (tr("Depth"), diag * 0.005..=diag),
    ];
    let mut edit = None;
    for ((label, range), x) in rows.into_iter().zip(v.iter_mut()) {
        edit = edit.or(slider_edit(ui, label, x, range, 3));
    }
    if let Some(merge) = edit {
        app.run("layer.decal.set", json!({"layer": layer.0, "size": [v[0], v[1]], "rotation": v[2], "offset": [v[3], v[4]], "depth": v[5], "merge": merge}));
    }
    ui.label(
        RichText::new(tr("The image's alpha cuts the decal's shape; faces turned away and surfaces beyond its depth stay clear."))
            .size(10.5)
            .color(tk.text_dim),
    );
}

/// The source of one fill channel: uniform, procedural, image or anchor.
fn fill_channel(app: &mut TextureCraftApp, ui: &mut Ui, fill: &Fill, base: &Value, c: Channel, anchors: &[(LayerId, String)], salt: &str) {
    let tk = Tokens::get(ui.ctx());
    use texturecraft_engine::core::ChannelKind;
    {
        let set = |extra: Value| with(base, with(&json!({"channel": c.id()}), extra));
        let group = ui.group(|ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tr(c.label())).strong());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.menu_button(RichText::new(ic::DOTS_THREE).size(13.0), |ui| {
                        if ui.button(tr("Uniform value")).clicked() {
                            app.run("layer.fill.set", set(json!({"value": c.default_value()})));
                            ui.close();
                        }
                        ui.separator();
                        for k in ProceduralKind::ALL {
                            if ui.button(tr(k.label())).clicked() {
                                let (lo, hi) = match c {
                                    Channel::BaseColor => ([0.15, 0.15, 0.15], [0.85, 0.85, 0.85]),
                                    Channel::Height => ([-0.05; 3], [0.05; 3]),
                                    _ => ([0.2; 3], [0.8; 3]),
                                };
                                app.run("layer.fill.set", set(json!({"procedural": k.label(), "low": lo, "high": hi})));
                                ui.close();
                            }
                        }
                        ui.separator();
                        if let Some(name) = image_menu(app, ui) {
                            app.run("layer.fill.set", set(json!({"image": name})));
                        }
                        if !anchors.is_empty() {
                            ui.separator();
                        }
                        for (a, label) in anchors {
                            if ui.button(trf("Anchor: {name}", &[("name", label)])).clicked() {
                                app.run("layer.fill.set", set(json!({"anchor": a.0})));
                                ui.close();
                            }
                        }
                    })
                    .response
                    .on_hover_text(tr("Channel source: uniform value, procedural, image or anchor"));
                });
            });
            match fill.source(c) {
                Source::Uniform(mut v) => {
                    if c.kind() == ChannelKind::Color {
                        if color_swatch(ui, &format!("fill_{salt}_{}", c.id()), &mut v, ui.available_width() - 4.0) {
                            app.run("layer.fill.set", set(json!({"color": v})));
                        }
                    } else {
                        let range = if c.kind() == ChannelKind::Signed { -1.0..=1.0 } else { 0.0..=1.0 };
                        if let Some(merge) = slider_edit(ui, tr("Value"), &mut v[0], range, 2) {
                            app.run("layer.fill.set", set(json!({"value": v[0], "merge": merge})));
                        }
                    }
                }
                Source::Procedural { pattern, low, high } => {
                    ui.label(
                        RichText::new(trf(
                            "{pattern} (scale {scale}, seed {seed})",
                            &[("pattern", &tr(pattern.kind.label())), ("scale", &format!("{:.1}", pattern.scale)), ("seed", &pattern.seed)],
                        ))
                        .color(tk.text_dim),
                    );
                    let mut s = pattern.scale;
                    let mut con = pattern.contrast;
                    let r1 = slider_edit(ui, tr("Pattern scale"), &mut s, 0.1..=100.0, 1);
                    let r2 = slider_edit(ui, tr("Contrast"), &mut con, 0.0..=8.0, 2);
                    let mut seed = pattern.seed as f32;
                    let r3 = slider_edit(ui, tr("Seed"), &mut seed, 0.0..=999.0, 0);
                    if let Some(merge) = r1.or(r2).or(r3) {
                        app.run(
                            "layer.fill.set",
                            set(json!({"procedural": pattern.kind.label(), "scale": s, "contrast": con, "seed": seed as u64, "low": low, "high": high, "merge": merge})),
                        );
                    }
                    if c.kind() == ChannelKind::Color {
                        let (mut lo, mut hi) = (low, high);
                        let mut changed = false;
                        ui.horizontal(|ui| {
                            ui.label(tr("Low"));
                            changed |= color_swatch(ui, &format!("lo_{salt}_{}", c.id()), &mut lo, 60.0);
                            ui.label(tr("High"));
                            changed |= color_swatch(ui, &format!("hi_{salt}_{}", c.id()), &mut hi, 60.0);
                        });
                        if changed {
                            app.run(
                                "layer.fill.set",
                                set(json!({"procedural": pattern.kind.label(), "scale": s, "contrast": con, "seed": pattern.seed, "low": lo, "high": hi})),
                            );
                        }
                    }
                }
                Source::Image { name, normal_format } => {
                    ui.label(trf("Image: {name}", &[("name", &name)]));
                    if c.kind() == ChannelKind::NormalMap {
                        let mut directx = normal_format == texturecraft_engine::core::NormalFormat::DirectX;
                        if ui
                            .checkbox(&mut directx, tr("DirectX normal map"))
                            .on_hover_text(tr("Flips the image's green channel, for normal maps made with Y pointing down"))
                            .changed()
                        {
                            app.run("layer.fill.set", set(json!({"normalFormat": if directx { "directX" } else { "openGl" }})));
                        }
                    }
                }
                Source::Anchor { anchor, channel } => {
                    ui.label(RichText::new(anchor_label(app, anchor, channel)).color(tk.text_dim));
                }
            }
        });
        // A shelf image dropped on the channel becomes its source.
        if let Some(name) = image_drop(ui, &group.response.interact(Sense::hover())) {
            app.run("layer.fill.set", set(json!({"image": name})));
        }
    }
}

fn effect_properties(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer, eid: LayerId) {
    let tk = Tokens::get(ui.ctx());
    let Some((_, effect)) = app.session.doc.project.effect(eid).map(|(owner, effect)| (owner, effect.clone())) else { return };

    let id = eid.0;
    ui.horizontal(|ui| {
        ui.label(tr("Blending"));
        egui::ComboBox::from_id_salt("eff_blend").selected_text(tr(effect.blend.label())).show_ui(ui, |ui| {
            for mode in BlendMode::all() {
                if ui.selectable_label(mode == effect.blend, tr(mode.label())).clicked() {
                    app.run("effect.set", json!({"effect": id, "blend": format!("{mode:?}")}));
                }
            }
        });
    });
    let mut opacity = effect.opacity;
    if let Some(merge) = slider_edit(ui, tr("Opacity"), &mut opacity, 0.0..=1.0, 2) {
        app.run("effect.set", json!({"effect": id, "opacity": opacity, "merge": merge}));
    }
    ui.separator();

    match &effect.kind {
        EffectKind::Generator(generator) => generator_properties(app, ui, id, generator),
        EffectKind::Levels(levels) => crate::levels_properties::levels_properties(app, ui, layer, eid, levels),
        EffectKind::Fill { source, .. } => {
            ui.label(caption(tr("Fill")));
            let anchors = anchors_below(app, layer.id);
            let current = match source {
                Source::Anchor { anchor, channel } => anchor_label(app, *anchor, *channel),
                Source::Uniform(_) => tr("Uniform value").to_string(),
                Source::Procedural { pattern, .. } => tr(pattern.kind.label()).to_string(),
                Source::Image { name, .. } => trf("Image: {name}", &[("name", name)]),
            };
            combo(ui, tr("Source"), current, |ui| {
                if ui.selectable_label(matches!(source, Source::Uniform(_)), tr("Uniform value")).clicked() {
                    app.run("effect.set", json!({"effect": id, "value": 1.0}));
                }
                for (anchor, label) in &anchors {
                    if ui.selectable_label(source.anchor() == Some(*anchor), trf("Anchor: {name}", &[("name", label)])).clicked() {
                        app.run("effect.set", json!({"effect": id, "anchor": anchor.0}));
                    }
                }
            });
            match source {
                Source::Uniform(color) => {
                    let mut value = color[0];
                    if let Some(merge) = slider_edit(ui, tr("Value"), &mut value, 0.0..=1.0, 2) {
                        app.run("effect.set", json!({"effect": id, "value": value, "merge": merge}));
                    }
                }
                Source::Anchor { channel, .. } => {
                    let channels = app.session.doc.project.texture_set(app.session.ts()).map(|set| set.channel_list()).unwrap_or_default();
                    combo(ui, tr("Channel"), tr(channel.label()), |ui| {
                        for option in channels {
                            if ui.selectable_label(option == *channel, tr(option.label())).clicked() {
                                app.run("effect.set", json!({"effect": id, "anchorChannel": option.id()}));
                            }
                        }
                    });
                    ui.label(RichText::new(tr("Colors read as gray. Add a levels or blur effect above to shape the result.")).color(tk.text_dim));
                }
                _ => {}
            }
            if anchors.is_empty() {
                ui.label(RichText::new(tr("Add an anchor point to a layer below to use its stack as a mask.")).color(tk.text_dim));
            }
        }
        EffectKind::Material(fill) => {
            ui.label(caption(tr("Fill")));
            let enabled: Vec<Channel> = fill.sources.keys().copied().collect();
            fill_editor(app, ui, fill, &json!({"effect": id}), &enabled, layer.id);
        }
        EffectKind::Anchor => {
            ui.label(caption(tr("Anchor point")));
            ui.label(
                RichText::new(tr("Records the stack up to this layer. Fill effects in masks (and fill layers) above can read it as a source."))
                    .color(tk.text_dim),
            );
        }
        EffectKind::Paint => {
            let hint = if layer.content_effects.iter().any(|content| content.id == eid) {
                tr("Paint on the mesh to edit the layer content.")
            } else {
                tr("Paint on the mesh to edit this mask. Use the Material tab's mask value for gray.")
            };
            ui.label(RichText::new(hint).color(tk.text_dim));
            tool_properties(app, ui);
        }
        EffectKind::Path(paths) => {
            ui.label(
                RichText::new(tr("Paths stay editable: pick one to move its points or change its brush and material, and the sublayer repaints at once. Add paths with the Path tool."))
                    .color(tk.text_dim),
            );
            let editing = app.session.tool.path.editing;
            for index in 0..paths.len() {
                ui.horizontal(|ui| {
                    let selected = editing == Some((eid, index));
                    if ui.selectable_label(selected, format!("{} {}", tr("Path"), index + 1)).on_hover_text(tr("Edit")).clicked() {
                        if selected {
                            app.run("tool.path", json!({"clear": true}));
                        } else {
                            app.run("path.edit", json!({"effect": id, "index": index}));
                        }
                    }
                    if icon_button(ui, ic::TRASH, tr("Remove"), false, 22.0).clicked() {
                        app.run("path.remove", json!({"effect": id, "index": index}));
                    }
                });
            }
            tool_properties(app, ui);
        }
        EffectKind::Filter(filter) => filter_properties(app, ui, id, filter),
        EffectKind::ColorSelection { colors, tolerance } => crate::color_selection_editor::color_selection_properties(app, ui, eid, colors, *tolerance),
        other => {
            ui.label(RichText::new(tr(other.name())).color(tk.text_dim));
        }
    }
}

/// The settings of generator effect `id`.
fn generator_properties(app: &mut TextureCraftApp, ui: &mut Ui, id: u64, generator: &Generator) {
    let tk = Tokens::get(ui.ctx());
    ui.label(caption(tr("Generator")));
    egui::ComboBox::from_id_salt("gen_kind").selected_text(tr(generator.kind.label())).show_ui(ui, |ui| {
        for kind in GeneratorKind::ALL {
            if ui.selectable_label(kind == generator.kind, tr(kind.label())).clicked() {
                // Replace the generator kind: remove and add keeps ordering simple
                app.run("effect.remove", json!({"effect": id}));
                app.run("effect.add.generator", json!({"generator": format!("{kind:?}")}));
            }
        }
    });

    let (mut level, mut softness, mut noise, mut noise_scale, mut invert) =
        (generator.level, generator.softness, generator.noise_amount, generator.noise_scale, generator.invert);
    let level_edit = slider_edit(ui, tr("Level"), &mut level, 0.0..=1.0, 2);
    let softness_edit = slider_edit(ui, tr("Softness"), &mut softness, 0.0..=1.0, 2);
    let noise_edit = slider_edit(ui, tr("Noise amount"), &mut noise, 0.0..=2.0, 2);
    let noise_scale_edit = slider_edit(ui, tr("Noise scale"), &mut noise_scale, 0.1..=64.0, 1);
    let invert_edit = ui.checkbox(&mut invert, tr("Invert")).changed().then_some(false);
    if let Some(merge) = level_edit.or(softness_edit).or(noise_edit).or(noise_scale_edit).or(invert_edit) {
        app.run(
            "effect.set",
            json!({"effect": id, "level": level, "softness": softness, "noise": noise, "noiseScale": noise_scale, "invert": invert, "merge": merge}),
        );
    }

    if matches!(generator.kind, GeneratorKind::Curvature | GeneratorKind::EdgeWear) {
        curvature_detail_properties(app, ui, id, generator.detail.as_ref());
    }
    if !app.session.doc.baked.contains_key(&app.session.ts()) {
        ui.label(RichText::new(format!("{} {}", ic::WARNING, tr("Bake mesh maps (F8) for curvature, AO and thickness based generators."))).color(tk.warning));
    }
}

/// The curvature mode and detail size weights of generator effect `id` (`detail` None: saved before detail sizes).
fn curvature_detail_properties(app: &mut TextureCraftApp, ui: &mut Ui, id: u64, detail: Option<&CurvatureDetail>) {
    let tk = Tokens::get(ui.ctx());
    let shown = detail.cloned().unwrap_or_default();
    combo(ui, tr("Mode"), curvature_mode_label(shown.mode), |ui| {
        for mode in CurvatureMode::ALL {
            if ui.selectable_label(mode == shown.mode, curvature_mode_label(mode)).clicked() {
                app.run("effect.set", json!({"effect": id, "mode": format!("{mode:?}")}));
            }
        }
    });

    ui.label(caption(tr("Detail sizes")));
    let mut weights = shown.weights;
    let mut edit = None;
    for (weight, size) in weights.iter_mut().zip(CurvatureDetail::SIZES) {
        edit = edit.or(slider_edit(ui, &format!("{size} px"), weight, 0.0..=1.0, 2));
    }
    if let Some(merge) = edit {
        app.run("effect.set", json!({"effect": id, "sizes": weights, "merge": merge}));
    }
    if detail.is_none() {
        ui.label(RichText::new(tr("Reads the curvature map as baked: changing the mode or a size turns detail sizes on.")).color(tk.text_dim));
    }
}

/// The translated name of curvature mode `mode`.
fn curvature_mode_label(mode: CurvatureMode) -> &'static str {
    match mode {
        CurvatureMode::Edges => tr("Edges"),
        CurvatureMode::Cavities => tr("Cavities"),
        CurvatureMode::Both => tr("Edges and cavities"),
        CurvatureMode::Raw => tr("Raw curvature"),
    }
}

/// The filter slot and the parameters of filter effect `id`
fn filter_properties(app: &mut TextureCraftApp, ui: &mut Ui, id: u64, filter: &Filter) {
    let tk = Tokens::get(ui.ctx());
    let set = |app: &mut TextureCraftApp, key: &str, value: Value, merge: bool| {
        let mut params = json!({ "effect": id, "merge": merge });
        params[key] = value;
        app.run("effect.set", params);
    };
    crate::filter_slot::show(app, ui, id, filter);

    if let Filter::Hsl { hue, .. } = filter {
        let mut degrees = hue * 360.0;
        if let Some(merge) = slider_edit(ui, tr("Hue shift (degrees)"), &mut degrees, -180.0..=180.0, 0) {
            set(app, "hue", json!(degrees / 360.0), merge);
        }
    }

    for (key, label, mut value, min, max) in scalar_params(filter) {
        let decimals = if matches!(key, "seed" | "levels") {
            0
        } else if max - min > 10.0 {
            1
        } else {
            2
        };
        if let Some(merge) = slider_edit(ui, tr(label), &mut value, min..=max, decimals) {
            set(app, key, json!(value), merge);
        }
    }

    match filter {
        Filter::ColorBalance { shadows, midtones, highlights } => {
            for (key, label, rgb) in [("shadows", tr("Shadows"), shadows), ("midtones", tr("Midtones"), midtones), ("highlights", tr("Highlights"), highlights)]
            {
                ui.label(RichText::new(label).strong());
                let mut balance = *rgb;
                let mut edit = None;
                for (component, name) in balance.iter_mut().zip([tr("Cyan / red"), tr("Magenta / green"), tr("Yellow / blue")]) {
                    edit = edit.or(slider_edit(ui, name, component, -1.0..=1.0, 2));
                }
                if let Some(merge) = edit {
                    set(app, key, json!(balance), merge);
                }
            }
        }
        Filter::ReplaceColor { from, to, .. } => {
            for (key, label, rgb) in [("from", tr("Color to replace"), from), ("to", tr("Replacement color"), to)] {
                let mut color = *rgb;
                ui.horizontal(|ui| {
                    if color_swatch(ui, &format!("replace_{key}_{id}"), &mut color, FILTER_SWATCH_WIDTH) {
                        set(app, key, json!(color), false);
                    }
                    ui.label(label);
                });
            }
            ui.label(RichText::new(tr("Colors near the first shift towards the second, keeping their shading.")).size(FILTER_HINT_SIZE).color(tk.text_dim));
        }
        Filter::GradientMap { colors } => {
            let mut stops = colors.clone();
            let mut changed = false;
            ui.horizontal_wrapped(|ui| {
                for (i, stop) in stops.iter_mut().enumerate() {
                    changed |= color_swatch(ui, &format!("gradient_{id}_{i}"), stop, FILTER_SWATCH_WIDTH);
                }
            });
            ui.horizontal(|ui| {
                if ui.add_enabled(stops.len() < MAX_GRADIENT_STOPS, egui::Button::new(format!("{} {}", ic::PLUS, tr("Add color")))).clicked() {
                    stops.push(stops.last().copied().unwrap_or([1.0; 3]));
                    changed = true;
                }
                if ui.add_enabled(stops.len() > 1, egui::Button::new(format!("{} {}", ic::MINUS, tr("Remove last")))).clicked() {
                    stops.pop();
                    changed = true;
                }
            });
            if changed {
                set(app, "colors", json!(stops), false);
            }
            ui.label(RichText::new(tr("Dark to light: the content's luminance picks a color along these stops.")).size(FILTER_HINT_SIZE).color(tk.text_dim));
        }
        Filter::Warp { .. }
        | Filter::DirectionalBlur { .. }
        | Filter::SlopeBlur { .. }
        | Filter::DirectionalWarp { .. }
        | Filter::HighPass { .. }
        | Filter::EdgeDetect { .. }
        | Filter::Bevel { .. }
        | Filter::Mosaic { .. }
        | Filter::Glow { .. } => {
            ui.label(RichText::new(tr("Works in texture space, like the blur: it does not cross UV seams.")).size(FILTER_HINT_SIZE).color(tk.text_dim));
        }
        _ => {}
    }
}

#[cfg(test)]
mod layer_layout_tests {
    use super::*;

    #[test]
    fn narrow_nested_rows_keep_thumbnail_and_secondary_controls_separate() {
        for width in [120.0, 180.0, 260.0, 400.0] {
            for depth in 0..=12 {
                for folder in [false, true] {
                    let row = Rect::from_min_size(pos2(100.0, 20.0), vec2(width, 48.0));
                    let (indent, size) = layer_thumbnail_layout(width, depth, folder, 36.0);
                    let content_left = row.left() + 7.0 + indent + 20.0 + if folder { 14.0 } else { 0.0 };
                    let content = Rect::from_min_size(pos2(content_left, row.top() + 6.0), vec2(size, size));
                    let mask = content.translate(vec2(size + 4.0, 0.0));
                    let (name, show_blend) = layer_name_layout(row, mask.right() + 8.0);

                    assert!(size >= 35.0, "minimum-width panels retain usable thumbnail targets");
                    assert!(row.contains_rect(content) && row.contains_rect(mask), "width={width}, depth={depth}, folder={folder}");
                    assert!(!content.intersects(mask));
                    if name.width() >= 20.0 {
                        assert!(row.contains_rect(name));
                        assert!(name.left() > mask.right());
                    }
                    if show_blend {
                        let blend_left = row.right() - RIGHT_COLUMN - 4.0;
                        assert!(name.right() + 4.0 <= blend_left);
                        assert!(mask.right() < blend_left);
                    }
                }
            }
        }
    }

    #[test]
    fn narrow_effect_rows_keep_the_eye_preview_and_name_inside() {
        for width in [120.0, 180.0, 260.0, 400.0] {
            for level in 0..=14 {
                for preview in [18.0, 21.6, 28.0] {
                    let indent = effect_row_indent(width, level, preview);
                    let eye_right = 7.0 + indent + 16.0;
                    let preview_left = 7.0 + indent + 22.0;
                    let name_left = preview_left + preview + 8.0;
                    assert!(eye_right < preview_left);
                    assert!(name_left + 20.0 <= width - 6.0 + 0.001);
                }
            }
        }
    }

    #[test]
    fn ordinary_rows_keep_preferred_sizes_and_secondary_controls() {
        let row = Rect::from_min_size(pos2(0.0, 0.0), vec2(300.0, 48.0));
        let (indent, size) = layer_thumbnail_layout(row.width(), 2, false, 36.0);
        let (_, show_blend) = layer_name_layout(row, 7.0 + indent + 20.0 + size * 2.0 + 4.0 + 8.0);

        assert_eq!(indent, 28.0);
        assert_eq!(size, 36.0);
        assert!(show_blend);
    }
}
