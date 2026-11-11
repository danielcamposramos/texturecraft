//! The rail's panels open in a popup beside their rail icon, one at a time; dragging its title docks or floats the panel

use egui::{Id, Pos2, Rect, RichText, Sense, Ui};
use egui_phosphor::regular as ic;

use crate::TextureCraftApp;
use crate::docking::Panel;
use crate::i18n::tr;
use crate::state::Mode;
use crate::theme::Tokens;
use crate::widgets::icon_button;

/// The rail's panels, top to bottom, with their icons and Window ▸ Views ids
pub(crate) const RAIL_PANELS: [(Panel, &str, &str); 5] = [
    (Panel::TextureSetSettings, ic::STACK, "ui.view.textureSetSettings"),
    (Panel::Display, ic::MONITOR, "ui.view.display"),
    (Panel::Shader, ic::SLIDERS_HORIZONTAL, "ui.view.shader"),
    (Panel::History, ic::CLOCK_COUNTER_CLOCKWISE, "ui.view.history"),
    (Panel::Log, ic::FILE_TEXT, "ui.view.log"),
];

/// The popup reaches this far into the viewport, so it covers the panels between the viewport and the rail
const VIEWPORT_OVERHANG: f32 = 48.0;

/// Narrowest the popup gets, when the viewport is not left of the rail's column
const POPUP_MIN_WIDTH: f32 = 320.0;

/// Settings panels' popups fit their contents up to this height, then scroll
const SETTINGS_MAX_HEIGHT: f32 = 640.0;

/// History and Log fill their popup, so it has one height
const LIST_HEIGHT: f32 = 400.0;

/// Gap between the popup and the rail
const RAIL_GAP: f32 = 4.0;

/// Size of the close button in the popup's title row
const TITLE_BUTTON_SIZE: f32 = 22.0;

/// The popup's area, whose rect egui remembers between frames
fn area_id() -> Id {
    Id::unique("rail_popup")
}

/// Where the rail drew its strip on this frame
fn strip_id() -> Id {
    Id::unique("rail_popup_strip")
}

/// Where the rail drew each panel's icon on this frame
fn icons_id() -> Id {
    Id::unique("rail_popup_icons")
}

/// The panel whose popup title a drag left, and the title's rect, until the dock takes the drag over
fn pending_drag_id() -> Id {
    Id::unique("rail_popup_pending_drag")
}

/// Where the dock drew the viewport on this frame
fn viewport_id() -> Id {
    Id::unique("rail_popup_viewport")
}

/// Whether `panel` opens from the rail rather than in the dock
pub(crate) fn is_rail_panel(panel: Panel) -> bool {
    RAIL_PANELS.iter().any(|(rail_panel, _, _)| *rail_panel == panel)
}

/// Whether the rail's open popup is visible in the current UI mode
pub(crate) fn is_visible(app: &TextureCraftApp) -> bool {
    app.ui.rail_panel.is_some() && !app.ui.hide_ui && app.ui.mode != Mode::Baking
}

/// Record the rail's strip before its icons, so the popup can line up with them
pub(crate) fn begin_rail(ctx: &egui::Context, strip: Rect) {
    ctx.data_mut(|data| {
        data.insert_temp(strip_id(), strip);
        data.insert_temp(icons_id(), Vec::<(Panel, Rect)>::new());
    });
}

/// Record where the rail drew `panel`'s icon
pub(crate) fn remember_icon(ctx: &egui::Context, panel: Panel, icon: Rect) {
    ctx.data_mut(|data| data.get_temp_mut_or_default::<Vec<(Panel, Rect)>>(icons_id()).push((panel, icon)));
}

/// Record where the dock drew the viewport, whose right edge the popup reaches past
pub(crate) fn remember_viewport(ctx: &egui::Context, viewport: Rect) {
    ctx.data_mut(|data| data.insert_temp(viewport_id(), viewport));
}

/// A rail icon or its Window ▸ Views item: open `panel` in the popup in place of the one there, or close it
pub(crate) fn toggle(app: &mut TextureCraftApp, panel: Panel) {
    // Bake mode has no rail, and a panel in the dock (which then has no rail icon) is toggled there
    if app.ui.mode == Mode::Baking || app.ui.dock.is_open(panel) {
        app.ui.dock.toggle(panel);
        return;
    }

    app.ui.rail_panel = if app.ui.rail_panel == Some(panel) { None } else { Some(panel) };
}

/// Open `panel` from the rail: in its popup, or where it is in the dock (bake mode has no rail)
pub(crate) fn open(app: &mut TextureCraftApp, panel: Panel) {
    if app.ui.mode == Mode::Baking || app.ui.dock.is_open(panel) {
        app.ui.dock.focus(panel);
    } else {
        app.ui.rail_panel = Some(panel);
    }
}

/// Hand a drag that left the popup's title to the dock, once the dock has drawn the panel it floated
pub(crate) fn hand_drag_to_dock(ctx: &egui::Context, dock_area: Id) {
    let Some((panel, title)) = ctx.data(|data| data.get_temp::<(Panel, Rect)>(pending_drag_id())) else { return };

    ctx.data_mut(|data| data.remove::<(Panel, Rect)>(pending_drag_id()));
    if ctx.input(|input| input.pointer.primary_down()) {
        craft_ui::docking::begin_drag(ctx, dock_area, panel, title);
    }
}

/// A rail popup and its dismissing press cannot start a paint stroke or move a decal
pub(crate) fn blocks_paint_input(app: &TextureCraftApp, ctx: &egui::Context) -> bool {
    let closing_id = area_id().with("closing");
    let pointer_active = ctx.input(|input| input.pointer.any_down() || input.pointer.any_released());
    if !pointer_active {
        ctx.data_mut(|data| data.remove::<bool>(closing_id));
    }

    // Both views must ignore the release, which would otherwise start a single-dab stroke
    let closing = ctx.data(|data| data.get_temp::<bool>(closing_id)).unwrap_or(false);

    // A shortcut may open the panel during a stroke; let that existing stroke finish
    !app.session.stroke_active() && (is_visible(app) || closing)
}

/// Draw the open popup beside its icon; a press outside it closes it, as for a menu (Escape does too, in `menus.rs`)
pub(crate) fn show(app: &mut TextureCraftApp, ctx: &egui::Context) {
    // The dock has drawn both views, so a released dismissal must not suppress the next click
    if !ctx.input(|input| input.pointer.any_down()) {
        ctx.data_mut(|data| data.remove::<bool>(area_id().with("closing")));
    }

    let Some(panel) = app.ui.rail_panel else { return };

    if app.ui.mode == Mode::Baking || app.ui.dock.is_open(panel) {
        app.ui.rail_panel = None;
        return;
    }

    let strip = ctx.data(|data| data.get_temp::<Rect>(strip_id())).unwrap_or(Rect::NOTHING);
    let icons = ctx.data(|data| data.get_temp::<Vec<(Panel, Rect)>>(icons_id())).unwrap_or_default();
    let Some(icon) = icons.iter().find(|(icon_panel, _)| *icon_panel == panel).map(|(_, rect)| *rect) else {
        app.ui.rail_panel = None;
        return;
    };

    let shown = ctx.memory(|memory| memory.area_rect(area_id()));
    if closed_by_input(ctx, shown, strip) {
        app.ui.rail_panel = None;
        let held = ctx.input(|input| input.pointer.any_down());
        ctx.data_mut(|data| data.insert_temp(area_id().with("closing"), held));
        return;
    }

    let tokens = Tokens::get(ctx);
    let frame = egui::Frame::popup(&ctx.global_style()).fill(tokens.panel).stroke(egui::Stroke::new(1.0, tokens.accent));
    let (top_right, width) = placement(ctx, strip, icon);
    let mut closed = false;
    let mut dragged_from = None;
    egui::Area::new(area_id()).order(egui::Order::Foreground).pivot(egui::Align2::RIGHT_TOP).fixed_pos(top_right).show(ctx, |ui| {
        frame.show(ui, |ui| {
            ui.set_width(width - frame.total_margin().sum().x);
            let title_top = ui.cursor().top();
            (closed, dragged_from) = title_row(ui, &tokens, panel);
            ui.separator();

            // Keep the title, frame and body on screen even when the window is shorter than the usual popup
            let title_height = ui.cursor().top() - title_top;
            let body_height = (ctx.content_rect().height() - frame.total_margin().sum().y - title_height).max(0.0);
            body(app, ui, panel, body_height);
        });
    });

    if closed {
        app.ui.rail_panel = None;
    } else if let Some(title) = dragged_from {
        app.ui.rail_panel = None;
        let rect = shown.unwrap_or(title);
        app.ui.dock.float_at(panel, [rect.left(), rect.top(), rect.width(), rect.height()]);
        ctx.data_mut(|data| data.insert_temp(pending_drag_id(), (panel, title)));
    }
}

/// The popup's top right corner, beside the rail and level with its icon, and its width (egui keeps the area on screen)
fn placement(ctx: &egui::Context, strip: Rect, icon: Rect) -> (Pos2, f32) {
    let right = strip.left() - RAIL_GAP;
    let viewport = ctx.data(|data| data.get_temp::<Rect>(viewport_id()));
    let width = viewport.map_or(POPUP_MIN_WIDTH, |viewport| right - (viewport.right() - VIEWPORT_OVERHANG)).max(POPUP_MIN_WIDTH);
    (Pos2::new(right, icon.top()), width)
}

/// A press outside the popup and the rail, unless in one of the popup's own dropdowns
fn closed_by_input(ctx: &egui::Context, shown: Option<Rect>, strip: Rect) -> bool {
    let pressed_at = ctx.input(|input| input.pointer.press_origin().filter(|_| input.pointer.any_pressed()));

    // A press on the rail is left to its icons, which switch or close the popup themselves
    let outside = pressed_at.is_some_and(|at| !strip.contains(at) && shown.is_some_and(|rect| !rect.contains(at)));
    outside && !egui::Popup::is_any_open(ctx)
}

/// The panel's contents: History and Log at one height, the settings panels as tall as their contents up to a limit
fn body(app: &mut TextureCraftApp, ui: &mut Ui, panel: Panel, available_height: f32) {
    // A child Ui, so the height limits start below the title row instead of moving the cursor back up to it
    ui.scope(|ui| {
        if matches!(panel, Panel::History | Panel::Log) {
            ui.set_height(LIST_HEIGHT.min(available_height));
        } else {
            // The area's remembered size would otherwise cap the height
            ui.set_max_height(SETTINGS_MAX_HEIGHT.min(available_height));
        }

        crate::docking::panel_body(app, ui, panel);
    });
}

/// The panel's name, which drags like a tab, and a close button: whether it was clicked, and where a drag started
fn title_row(ui: &mut Ui, tokens: &Tokens, panel: Panel) -> (bool, Option<Rect>) {
    let row = ui.horizontal(|ui| {
        ui.label(RichText::new(tr(panel.name())).strong().color(tokens.text_strong));
        ui.add_space((ui.available_width() - TITLE_BUTTON_SIZE).max(0.0));
        icon_button(ui, ic::X, tr("Close"), false, TITLE_BUTTON_SIZE)
    });

    let (row, close) = (row.response.rect, row.inner);
    let title = Rect::from_min_max(row.min, egui::pos2(close.rect.left(), row.bottom()));
    let drag = ui.interact(title, ui.scope_id().with("rail_popup_title"), Sense::drag()).on_hover_cursor(egui::CursorIcon::Grab);
    (close.clicked(), drag.drag_started().then_some(title))
}
