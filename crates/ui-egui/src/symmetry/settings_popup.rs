//! The symmetry settings popup: independent mirror and radial copies, their position and display

use egui::{Align, Color32, Layout, Stroke, Ui, vec2};
use texturecraft_engine::paint::{FULL_TURN_DEGREES, MAX_RADIAL_COPIES, Symmetry};

use super::{DEFAULT_RADIAL_COUNT, movable_axes};
use crate::TextureCraftApp;
use crate::i18n::tr;
use crate::widgets::{color_swatch, section, slider};

/// The popup's width (points)
const POPUP_WIDTH: f32 = 340.0;

/// Width of the radial Axis row's label column
const LABEL_WIDTH: f32 = 64.0;

const AXIS_NAMES: [&str; 3] = ["X", "Y", "Z"];

/// Space between the mirror section and the radial toggle
const SECTIONS_GAP: f32 = 4.0;

/// The color square's width (points) and the Manipulator size slider's range
const COLOR_SWATCH_WIDTH: f32 = 48.0;
const MANIPULATOR_SIZES: std::ops::RangeInclusive<f32> = 0.1..=1.0;

/// Side of the radial copies preview beside the Count and Angle span sliders
const PREVIEW_SIZE: f32 = 96.0;

/// The popup's content, applying brush changes through the engine and display preferences in place
pub(super) fn show(app: &mut TextureCraftApp, ui: &mut Ui) {
    ui.set_width(POPUP_WIDTH);
    let (minimum, maximum) = app.session.mesh.bounds();
    let half_extent = (maximum - minimum) * 0.5;
    let display = &mut app.ui.symmetry;
    let mut edited = app.session.tool.brush.symmetry;
    let symmetry = &mut edited;
    section(ui, "symmetry_mirror", tr("Mirror symmetry"), true, |ui| {
        ui.horizontal(|ui| {
            let cell = vec2(ui.available_width() / AXIS_NAMES.len() as f32, ui.spacing().interact_size.y);
            for (name, enabled) in AXIS_NAMES.into_iter().zip([&mut symmetry.x, &mut symmetry.y, &mut symmetry.z]) {
                ui.allocate_ui_with_layout(cell, Layout::left_to_right(Align::Center), |ui| {
                    ui.checkbox(enabled, name);
                });
            }
        });
    });
    ui.add_space(SECTIONS_GAP);

    let mut radial = symmetry.radial > 1;
    if ui.checkbox(&mut radial, tr("Radial symmetry")).changed() {
        symmetry.radial = if radial { DEFAULT_RADIAL_COUNT } else { 0 };
    }

    if radial {
        section(ui, "symmetry_radial", tr("Radial settings"), true, |ui| {
            labelled_row(ui, tr("Axis"), |ui| {
                let cell = vec2(ui.available_width() / AXIS_NAMES.len() as f32, ui.spacing().interact_size.y);
                for (axis, name) in AXIS_NAMES.into_iter().enumerate() {
                    ui.allocate_ui_with_layout(cell, Layout::left_to_right(Align::Center), |ui| {
                        ui.radio_value(&mut symmetry.axis, axis as u8, name);
                    });
                }
            });
            ui.horizontal_top(|ui| {
                let sliders_width = ui.available_width() - PREVIEW_SIZE - ui.spacing().item_spacing.x;
                ui.allocate_ui_with_layout(vec2(sliders_width, PREVIEW_SIZE), Layout::top_down(Align::Min), |ui| {
                    let mut count = f32::from(symmetry.radial);
                    if slider(ui, tr("Count"), &mut count, 2.0..=f32::from(MAX_RADIAL_COPIES), 0).changed() {
                        symmetry.radial = count.round() as u8;
                    }
                    slider(ui, tr("Angle span"), &mut symmetry.angle_span, 0.0..=FULL_TURN_DEGREES, 0);
                });
                copies_preview(ui, symmetry, super::view_overlay::color(display.color, u8::MAX));
            });
        });
    }

    let movable = movable_axes(symmetry);
    section(ui, "symmetry_axis_position", tr("Axis position"), true, |ui| {
        for (axis, name) in AXIS_NAMES.into_iter().enumerate() {
            ui.add_enabled_ui(movable[axis], |ui| slider(ui, name, &mut symmetry.offset[axis], -half_extent[axis]..=half_extent[axis], 2));
        }
    });

    section(ui, "symmetry_display", tr("Display"), true, |ui| {
        ui.horizontal(|ui| {
            ui.label(tr("Symmetry color display"));
            ui.with_layout(Layout::right_to_left(Align::Center), |ui| color_swatch(ui, "symmetry_color", &mut display.color, COLOR_SWATCH_WIDTH));
        });
        if symmetry.x || symmetry.y || symmetry.z {
            ui.checkbox(&mut display.show_plane, tr("Show plane"));
        }
        if radial {
            ui.checkbox(&mut display.show_axis, tr("Show axis"));
        }
        ui.checkbox(&mut display.show_intersection, tr("Show intersection"));
        ui.checkbox(&mut display.show_cursor, tr("Show cursor"));
        ui.indent("symmetry_hide_while_painting", |ui| {
            ui.add_enabled(display.show_cursor, egui::Checkbox::new(&mut display.hide_while_painting, tr("Hide while painting")));
        });
        ui.checkbox(&mut display.show_manipulator, tr("Show axis position manipulator"));
        ui.add_enabled_ui(display.show_manipulator, |ui| slider(ui, tr("Manipulator size"), &mut display.manipulator_size, MANIPULATOR_SIZES, 1));
    });

    super::apply(app, edited);
}

/// A row with `label` in a fixed-width column and the control filling the rest
fn labelled_row(ui: &mut Ui, label: &str, control: impl FnOnce(&mut Ui)) {
    ui.horizontal(|ui| {
        let height = ui.spacing().interact_size.y;
        ui.allocate_ui_with_layout(vec2(LABEL_WIDTH, height), Layout::left_to_right(Align::Center), |ui| {
            ui.set_min_width(LABEL_WIDTH);
            ui.label(label);
        });
        control(ui);
    });
}

/// Where the radial copies go, on the view's backdrop: the stroke in white on the right, its copies in `copy_color`
fn copies_preview(ui: &mut Ui, symmetry: &Symmetry, copy_color: Color32) {
    const RING_FRACTION: f32 = 0.32;
    const DOT_RADIUS: f32 = 3.0;
    const CROSS_HALF: f32 = 4.0;
    const CORNER_RADIUS: f32 = 2.0;

    let tokens = crate::theme::Tokens::get(ui.ctx());
    let (rect, _) = ui.allocate_exact_size(vec2(PREVIEW_SIZE, PREVIEW_SIZE), egui::Sense::hover());
    let painter = ui.painter_at(rect);
    painter.rect_filled(rect, CORNER_RADIUS, tokens.viewport);

    let center = rect.center();
    let ring = PREVIEW_SIZE * RING_FRACTION;
    let cross = Stroke::new(1.0, tokens.viewport_text);
    painter.line_segment([center - vec2(CROSS_HALF, 0.0), center + vec2(CROSS_HALF, 0.0)], cross);
    painter.line_segment([center - vec2(0.0, CROSS_HALF), center + vec2(0.0, CROSS_HALF)], cross);
    for copy in 0..symmetry.radial {
        // Screen y points down: positive angles turn counter-clockwise
        let angle = symmetry.radial_angle(copy);
        let at = center + vec2(angle.cos(), -angle.sin()) * ring;
        painter.circle_filled(at, DOT_RADIUS, if copy == 0 { Color32::WHITE } else { copy_color });
    }
}
