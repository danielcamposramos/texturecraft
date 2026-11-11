//! Symmetry: the toolbar toggle, the settings popup and the 3D view's manipulator

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::{Queryable, by};
use serde_json::{Value, json};
use texturecraft_engine::Tool;
use texturecraft_engine::geom::glam::Vec3;
use texturecraft_ui_egui::TextureCraftApp;

/// The settings popup's width and its color square's (points)
const POPUP_WIDTH: f32 = 340.0;
const COLOR_SWATCH_WIDTH: f32 = 48.0;

/// How far along Y (world units) the test measures the arrow's direction on screen
const AXIS_PROBE: f32 = 0.05;

/// Points from the manipulator's center to grab the arrow (inside its length at the default size) and how far to drag it
const ON_THE_SHAFT: f32 = 40.0;
const DRAG_DISTANCE: f32 = 60.0;

/// A point inside the 3D view near its lower left corner, clear of the mesh and the popup (points from the corner)
const VIEW_CORNER_INSET: f32 = 40.0;

fn app_harness() -> Harness<'static, TextureCraftApp> {
    let session = texturecraft_engine::sample_session(1024, 256).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.state_mut().session.tool.tool = Tool::Paint;
    harness.run_steps(4);
    harness
}

/// Hold a primary-button drag from `from` to `to` without releasing it
fn hold_drag(harness: &mut Harness<'_, TextureCraftApp>, from: Pos2, to: Pos2) {
    harness.hover_at(from);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(2);

    for step in 1..=4 {
        harness.hover_at(from + (to - from) * (step as f32 / 4.0));
        harness.run_steps(1);
    }
}

fn click_at(harness: &mut Harness<'_, TextureCraftApp>, at: Pos2) {
    harness.hover_at(at);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    release(harness, at);
}

fn release(harness: &mut Harness<'_, TextureCraftApp>, at: Pos2) {
    harness.event(Event::PointerButton { pos: at, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(2);
}

/// The brush settings as agents read them
fn brush(harness: &mut Harness<'_, TextureCraftApp>) -> Value {
    texturecraft_engine::execute(&mut harness.state_mut().session, "tool.brush", &json!({})).unwrap()
}

/// The number field beside a uniquely labelled slider
fn number_field<'a>(harness: &'a Harness<'_, TextureCraftApp>, label: &str) -> egui_kittest::Node<'a> {
    let row = harness.get_by_label(label).rect().center().y;
    harness.get_all(by().role(egui::accesskit::Role::TextInput)).find(|node| (node.rect().center().y - row).abs() < 4.0).unwrap()
}

/// Type a complete value in the number field beside a uniquely labelled slider
fn set_number(harness: &mut Harness<'_, TextureCraftApp>, label: &str, value: &str) {
    let length = number_field(harness, label).value().unwrap().len();
    number_field(harness, label).focus();
    harness.run_steps(2);
    harness.key_press(egui::Key::End);
    harness.run_steps(1);
    for _ in 0..length {
        harness.key_press(egui::Key::Backspace);
        harness.run_steps(1);
    }
    number_field(harness, label).type_text(value);
    harness.run_steps(1);
    harness.key_press(egui::Key::Enter);
    harness.run_steps(2);
}

/// Where world `point` is in the 3D view
fn on_screen(harness: &Harness<'_, TextureCraftApp>, point: Vec3) -> Pos2 {
    let rect = harness.state().viewport_rect;
    let ndc = harness.state().session.camera.project(point, rect.width() / rect.height()).unwrap();
    egui::pos2(rect.left() + (ndc.x * 0.5 + 0.5) * rect.width(), rect.top() + (0.5 - ndc.y * 0.5) * rect.height())
}

#[test]
fn the_toolbar_popup_and_manipulator_change_the_brush_symmetry() {
    let mut harness = app_harness();
    let revision = harness.state().session.revision;
    let history = harness.state().session.history_labels();

    harness.get_by_label("Symmetry").click();
    harness.run_steps(2);
    assert!(harness.state().session.revision > revision, "the toggle runs the brush command");
    assert_eq!(brush(&mut harness)["symmetry"], json!([true, false, false]), "the toggle mirrors on X");

    harness.get_by_label("Symmetry settings").click();
    harness.run_steps(2);
    harness.get_by_role_and_label(egui::accesskit::Role::CheckBox, "X").click();
    harness.run_steps(2);
    harness.get_by_role_and_label(egui::accesskit::Role::CheckBox, "Y").click();
    harness.run_steps(2);
    assert_eq!(brush(&mut harness)["symmetry"], json!([false, true, false]), "mirror axes can be toggled independently");

    harness.get_by_label("Symmetry settings").click();
    harness.state_mut().ui.symmetry.show_manipulator = true;
    harness.run_steps(2);

    // Drag the arrow along Y, starting on its shaft
    let (lo, hi) = harness.state().session.mesh.bounds();
    let origin = (lo + hi) * 0.5;
    let center = on_screen(&harness, origin);
    let along = (on_screen(&harness, origin + Vec3::Y * AXIS_PROBE) - center).normalized();
    let from = center + along * ON_THE_SHAFT;
    let to = from + along * DRAG_DISTANCE;
    let revision = harness.state().session.revision;

    hold_drag(&mut harness, from, to);
    release(&mut harness, to);

    assert!(harness.state().session.revision > revision, "the manipulator runs the brush command");
    assert_eq!(harness.state().session.history_labels(), history, "symmetry settings do not add undo steps");
    let offset = brush(&mut harness)["symmetryOffset"].clone();
    assert!(offset[1].as_f64().unwrap() > 0.05, "the plane moved up: {offset}");
    assert_eq!((offset[0].as_f64(), offset[2].as_f64()), (Some(0.0), Some(0.0)), "only along Y: {offset}");

    harness.get_by_label("Symmetry").click();
    harness.run_steps(2);
    let settings = brush(&mut harness);
    assert_eq!(settings["symmetryOn"], json!(false), "the toggle turns it off");
    assert_eq!(settings["symmetry"], json!([false, true, false]), "and keeps the settings");
}

#[test]
fn the_color_picker_opens_inside_the_combined_symmetry_popup() {
    let mut harness = app_harness();
    harness.get_by_label("Symmetry").click();
    harness.run_steps(2);
    harness.get_by_label("Symmetry settings").click();
    harness.run_steps(2);

    harness.get_by_label("Radial symmetry").click();
    harness.run_steps(2);
    assert_eq!(brush(&mut harness)["radialSymmetry"], json!({"copies": 6, "axis": "y"}));
    assert_eq!(brush(&mut harness)["symmetry"], json!([true, false, false]));
    assert!(harness.query_by_label("Angle span").is_some(), "the popup stays open after enabling radial symmetry");

    // The color square sits at the right end of its row
    let row = harness.get_by_label("Symmetry color display").rect();
    let left = harness.get_by_label("Radial symmetry").rect().left();
    click_at(&mut harness, egui::pos2(left + POPUP_WIDTH - COLOR_SWATCH_WIDTH / 2.0, row.center().y));
    harness.run_steps(2);
    assert!(harness.query_by_label("Copy RGB values").is_some(), "the color picker opened");
    assert!(harness.query_by_label("Show cursor").is_some(), "the popup stays open with the color picker");

    // A click on the view closes it
    let view_corner = harness.state().viewport_rect.left_bottom() + egui::vec2(VIEW_CORNER_INSET, -VIEW_CORNER_INSET);
    click_at(&mut harness, view_corner);
    harness.run_steps(2);
    assert!(harness.query_by_label("Show cursor").is_none(), "a click elsewhere closes the popup");
}

#[test]
fn radial_axis_edits_preserve_combined_symmetry() {
    let mut harness = app_harness();
    harness
        .state_mut()
        .run("tool.brush", json!({"symmetryX": true, "symmetryZ": true, "symmetryRadial": 5, "symmetryAxis": "y", "symmetryAngleSpan": 215}))
        .unwrap();
    harness.get_by_label("Symmetry settings").click();
    harness.run_steps(2);
    let revision = harness.state().session.revision;
    let history = harness.state().session.history_labels();

    harness.get_by_role_and_label(egui::accesskit::Role::RadioButton, "Z").click();
    harness.run_steps(2);

    let settings = brush(&mut harness);
    assert_eq!(settings["symmetry"], json!([true, false, true]), "editing radial symmetry must preserve every mirror axis");
    assert_eq!(settings["radialSymmetry"], json!({"copies": 5, "axis": "z"}));
    assert_eq!(settings["symmetryAngleSpan"], json!(215.0));
    assert!(harness.state().session.revision > revision, "the edit runs the brush command");
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn the_popup_creates_and_edits_mirror_and_radial_copies_independently() {
    let mut harness = app_harness();
    harness.state_mut().run("tool.brush", json!({"symmetryOn": false})).unwrap();
    harness.get_by_label("Symmetry settings").click();
    harness.run_steps(2);
    let history = harness.state().session.history_labels();

    for axis in ["X", "Z"] {
        harness.get_by_role_and_label(egui::accesskit::Role::CheckBox, axis).click();
        harness.run_steps(2);
    }
    harness.get_by_label("Radial symmetry").click();
    harness.run_steps(2);

    let settings = brush(&mut harness);
    assert_eq!(settings["symmetry"], json!([true, false, true]));
    assert_eq!(settings["radialSymmetry"], json!({"copies": 6, "axis": "y"}));
    assert_eq!(settings["symmetryOn"], json!(false), "editing settings preserves the master toggle");
    assert!(harness.query_by_label("Show plane").is_some());
    assert!(harness.query_by_label("Show axis").is_some());

    set_number(&mut harness, "Count", "9");
    set_number(&mut harness, "Angle span", "180");

    let settings = brush(&mut harness);
    assert_eq!(settings["symmetry"], json!([true, false, true]), "radial edits keep both mirror axes");
    assert_eq!(settings["radialSymmetry"], json!({"copies": 9, "axis": "y"}));
    assert_eq!(settings["symmetryAngleSpan"], json!(180.0));

    for axis in ["X", "Y", "Z"] {
        harness.get_by_role_and_label(egui::accesskit::Role::CheckBox, axis).click();
        harness.run_steps(2);
    }

    let settings = brush(&mut harness);
    assert_eq!(settings["symmetry"], json!([false, true, false]));
    assert_eq!(settings["radialSymmetry"], json!({"copies": 9, "axis": "y"}), "mirror edits keep the radial settings");
    assert_eq!(settings["symmetryAngleSpan"], json!(180.0));

    harness.get_by_label("Symmetry").click();
    harness.run_steps(2);
    let combined = harness.state().session.tool.brush.symmetry;
    assert!(combined.enabled);
    assert_eq!(combined.transforms().len(), 18, "the engine receives both symmetry settings");
    harness.get_by_label("Symmetry").click();
    harness.run_steps(2);
    assert_eq!(harness.state().session.tool.brush.symmetry, texturecraft_engine::paint::Symmetry { enabled: false, ..combined });

    harness.get_by_label("Symmetry settings").click();
    harness.run_steps(2);
    harness.get_by_label("Radial symmetry").click();
    harness.run_steps(2);

    let settings = brush(&mut harness);
    assert_eq!(settings["symmetry"], json!([false, true, false]), "disabling radial symmetry keeps the mirror");
    assert_eq!(settings["radialSymmetry"], json!({"copies": 0, "axis": "y"}));
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn clicking_a_symmetry_handle_does_not_paint() {
    let mut harness = app_harness();
    harness.state_mut().run("tool.brush", json!({"symmetryY": true})).unwrap();
    harness.state_mut().ui.symmetry.show_manipulator = true;
    harness.run_steps(2);
    let (lo, hi) = harness.state().session.mesh.bounds();
    let center = on_screen(&harness, (lo + hi) * 0.5);
    let history = harness.state().session.history_labels();

    click_at(&mut harness, center);

    assert_eq!(harness.state().session.history_labels(), history, "clicking a handle must not add a paint dab");
    assert!(!harness.state().session.stroke_active());
}

#[test]
fn a_paint_stroke_can_cross_and_end_on_a_symmetry_handle() {
    let mut harness = app_harness();
    harness.state_mut().run("tool.brush", json!({"symmetryY": true})).unwrap();
    harness.state_mut().ui.symmetry.show_manipulator = true;
    harness.run_steps(2);
    let (lo, hi) = harness.state().session.mesh.bounds();
    let center = on_screen(&harness, (lo + hi) * 0.5);
    let from = center + egui::vec2(80.0, 0.0);
    let history = harness.state().session.history_labels().len();

    hold_drag(&mut harness, from, center);
    release(&mut harness, center);

    assert!(!harness.state().session.stroke_active(), "releasing over the handle must finish painting");
    assert_eq!(harness.state().session.history_labels().len(), history + 1);
    assert_eq!(harness.state().session.tool.brush.symmetry.offset, [0.0; 3]);
}

#[test]
fn ctrl_drag_on_a_symmetry_handle_only_adjusts_the_brush() {
    let mut harness = app_harness();
    harness.state_mut().run("tool.brush", json!({"symmetryY": true})).unwrap();
    harness.state_mut().ui.symmetry.show_manipulator = true;
    harness.run_steps(2);
    let (lo, hi) = harness.state().session.mesh.bounds();
    let from = on_screen(&harness, (lo + hi) * 0.5);
    let to = from + egui::vec2(-60.0, -40.0);
    let before = harness.state().session.tool.brush.clone();
    let history = harness.state().session.history_labels();

    harness.event(Event::ModifiersChanged(Modifiers::COMMAND));
    harness.hover_at(from);
    harness.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::COMMAND });
    harness.run_steps(2);
    for step in 1..=4 {
        harness.hover_at(from + (to - from) * (step as f32 / 4.0));
        harness.run_steps(1);
    }
    harness.event(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::COMMAND });
    harness.event(Event::ModifiersChanged(Modifiers::NONE));
    harness.run_steps(2);

    let after = &harness.state().session.tool.brush;
    assert!(after.flow < before.flow);
    assert_ne!(after.angle, before.angle);
    assert_eq!(after.symmetry, before.symmetry, "adjusting the brush must not also move symmetry");
    assert_eq!(harness.state().session.history_labels(), history);
}
