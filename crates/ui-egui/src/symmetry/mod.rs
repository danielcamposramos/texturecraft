//! Symmetry: the toolbar toggle and settings popup, and what the 3D view draws for it

mod ribbons;
mod settings_popup;
mod symmetry_display;
mod symmetry_drag;
mod symmetry_intersection;
mod symmetry_manipulator;
mod symmetry_manipulator_arrow;
mod view_overlay;

use egui::Ui;
use egui_phosphor::regular as ic;
use texturecraft_engine::Tool;
use texturecraft_engine::paint::Symmetry;

pub use symmetry_display::SymmetryDisplay;
pub(crate) use view_overlay::{manipulator_hovered, manipulator_input, overlay, scene_overlay};

use crate::TextureCraftApp;
use crate::i18n::tr;
use crate::widgets::{icon_badge_dropdown_button, icon_button};

/// Radial copies a brush gets when radial symmetry is enabled
const DEFAULT_RADIAL_COUNT: u8 = 6;

/// Whether `tool` paints symmetry copies
pub(crate) fn applies_to(tool: Tool) -> bool {
    matches!(tool, Tool::Paint | Tool::Eraser | Tool::Projection | Tool::Clone | Tool::Path)
}

/// Whether the 3D view shows symmetry now
pub(crate) fn shown(app: &TextureCraftApp) -> bool {
    app.ui.mode == crate::state::Mode::Painting && applies_to(app.session.tool.tool) && app.session.tool.brush.symmetry.any()
}

/// The world axes any active mirror plane or radial axis can move along, even while symmetry is disabled
pub(crate) fn movable_axes(symmetry: &Symmetry) -> [bool; 3] {
    let mirrors = [symmetry.x, symmetry.y, symmetry.z];
    let radial_axis = symmetry.axis_index();
    [0, 1, 2].map(|axis| mirrors[axis] || (symmetry.radial > 1 && axis != radial_axis))
}

/// The toolbar's Symmetry toggle: turning it on with nothing set mirrors on X
fn toggle(symmetry: &mut Symmetry) {
    if symmetry.any() {
        symmetry.enabled = false;
        return;
    }

    symmetry.enabled = true;
    if !symmetry.x && !symmetry.y && !symmetry.z && symmetry.radial <= 1 {
        symmetry.x = true;
    }
}

/// Apply changed symmetry settings through the same command used by scripts and agents
fn apply(app: &mut TextureCraftApp, symmetry: Symmetry) {
    if symmetry == app.session.tool.brush.symmetry {
        return;
    }

    app.run(
        "tool.brush",
        serde_json::json!({
            "symmetryOn": symmetry.enabled,
            "symmetryX": symmetry.x,
            "symmetryY": symmetry.y,
            "symmetryZ": symmetry.z,
            "symmetryRadial": symmetry.radial,
            "symmetryAxis": (["x", "y", "z"][symmetry.axis_index()]),
            "symmetryOffset": symmetry.offset,
            "symmetryAngleSpan": symmetry.angle_span,
        }),
    );
}

/// The Symmetry toggle and the settings button with its popup
pub(crate) fn toolbar_controls(app: &mut TextureCraftApp, ui: &mut Ui) {
    let button_size = ui.spacing().interact_size.y;
    if icon_button(ui, ic::FLIP_HORIZONTAL, tr("Symmetry"), app.session.tool.brush.symmetry.any(), button_size).clicked() {
        let mut symmetry = app.session.tool.brush.symmetry;
        toggle(&mut symmetry);
        apply(app, symmetry);
    }

    // No tooltip on the settings button; the label still names it for screen readers
    let popup_id = ui.make_persistent_id("symmetry_settings");
    let mut open = ui.data(|data| data.get_temp::<bool>(popup_id)).unwrap_or(false);
    let response = icon_badge_dropdown_button(ui, ic::FLIP_HORIZONTAL, ic::GEAR, tr("Symmetry settings"), open, button_size);
    if response.clicked() {
        open = !open;
    }

    // egui's single popup slot is needed by the color picker inside this popup
    let shown = egui::Popup::from_response(&response)
        .id(popup_id)
        .open_bool(&mut open)
        .close_behavior(egui::PopupCloseBehavior::IgnoreClicks)
        .show(|ui| settings_popup::show(app, ui));

    // A click outside the popup and the dropdowns it opens (foreground layers too) closes it
    if let Some(shown) = shown
        && !response.clicked()
        && ui.input(|input| input.pointer.any_click())
        && let Some(at) = ui.input(|input| input.pointer.interact_pos())
        && ui.ctx().layer_id_at(at).is_none_or(|layer| layer != shown.response.layer_id && layer.order != egui::Order::Foreground)
    {
        open = false;
    }

    ui.data_mut(|data| data.insert_temp(popup_id, open));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_symmetry_can_move_every_active_plane_and_axis() {
        for enabled in [true, false] {
            for axis in 0..3 {
                let symmetry = Symmetry { enabled, x: axis == 0, y: axis == 1, z: axis == 2, radial: 5, axis, ..Symmetry::default() };

                let movable = movable_axes(&symmetry);

                assert_eq!(movable, [true; 3], "the mirror plane still moves along the radial axis: {symmetry:?}");
            }
        }
    }
}
