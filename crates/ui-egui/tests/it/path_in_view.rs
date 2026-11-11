//! The Path tool in the 2D view: placing previews the path, and points are edited where they are drawn

use egui::{Event, Key, Modifiers, PointerButton, Pos2, pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable};
use serde_json::json;
use texturecraft_engine::core::LayerId;
use texturecraft_engine::execute;
use texturecraft_ui_egui::TextureCraftApp;
use texturecraft_ui_egui::state::ViewLayout;

type AppHarness<'a> = Harness<'a, TextureCraftApp>;

// Three points left to right across the middle of the 2D view's texture
const A: Pos2 = pos2(600.0, 500.0);
const B: Pos2 = pos2(700.0, 500.0);
const C: Pos2 = pos2(800.0, 500.0);

/// A Path tool session whose texture fills the test pointer coordinates in the 2D view
fn harness() -> AppHarness<'static> {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    execute(&mut session, "layer.add.paint", &serde_json::Value::Null).unwrap();
    execute(&mut session, "tool.select", &json!({"tool": "path"})).unwrap();
    // Tall enough that the 2D view's texture square takes A, B and C
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1500.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    h.state_mut().ui.layout = ViewLayout::Only2D;
    h.run_steps(4);
    h
}

/// Advance one input frame, optionally changing the primary button's state
fn pointer(h: &mut AppHarness<'_>, pos: Pos2, pressed: Option<bool>) {
    let events = &mut h.input_mut().events;
    events.push(Event::PointerMoved(pos));
    if let Some(pressed) = pressed {
        events.push(Event::PointerButton { pos, button: PointerButton::Primary, pressed, modifiers: Modifiers::NONE });
    }
    h.step();
}

/// Send a complete pointer click and allow the resulting preview to update
fn click(h: &mut AppHarness<'_>, at: Pos2) {
    pointer(h, at, Some(true));
    pointer(h, at, Some(false));
    h.run_steps(2);
}

/// Move through several frames so the test exercises drag grouping as well as its final position
fn drag(h: &mut AppHarness<'_>, from: Pos2, to: Pos2) {
    pointer(h, from, Some(true));
    for k in 1..=6 {
        pointer(h, from + (to - from) * (k as f32 / 6.0), None);
    }
    pointer(h, to, Some(false));
    h.run_steps(2);
}

/// Dispatch an unmodified tool key and settle the resulting UI state
fn key(h: &mut AppHarness<'_>, key: Key) {
    h.input_mut().events.push(Event::Key { key, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::NONE });
    h.run_steps(2);
}

/// The path sublayers of the selected layer: (id, stored paths)
fn path_sublayers(h: &AppHarness<'_>) -> Vec<(LayerId, usize)> {
    let s = &h.state().session;
    let layer = s.doc.project.layer(s.selected_layer().unwrap()).unwrap();
    layer.content_effects.iter().filter_map(|e| s.paths(e.id).map(|p| (e.id, p.len()))).collect()
}

/// Channel images painted in path sublayer `eid`
fn painted_channels(h: &AppHarness<'_>, eid: LayerId) -> usize {
    h.state().session.doc.pixels.layers.iter().filter(|((id, _), _)| *id == eid).count()
}

#[test]
fn placing_points_previews_the_path_and_enter_applies_it_as_one_step() {
    let mut h = harness();
    let steps = h.state().session.history_labels().len();
    for at in [A, B, C] {
        click(&mut h, at);
    }
    assert_eq!(h.state().session.tool.path.points.len(), 3);
    let [(eid, 1)] = path_sublayers(&h)[..] else { panic!("the preview is a path in a path sublayer: {:?}", path_sublayers(&h)) };
    assert!(painted_channels(&h, eid) > 0, "the preview painted");
    assert_eq!(h.state().session.history_labels().len(), steps, "no undo step while placing");
    key(&mut h, Key::Enter);
    assert_eq!(h.state().session.history_labels().len(), steps + 1);
    assert_eq!(h.state().session.undo_label(), Some("Path"));
    assert!(h.state().session.tool.path.points.is_empty());
    assert_eq!(path_sublayers(&h), vec![(eid, 1)]);
}

/// Applying with the pointer commits the preview without leaving the current texture set
#[test]
fn apply_button_commits_the_path_in_the_current_texture_set() {
    let mut h = harness();
    assert!(h.get_all_by_label("Apply").next().unwrap().accesskit_node().is_disabled());
    let steps = h.state().session.history_labels().len();
    let texture_set = h.state().session.ts();
    for at in [A, B, C] {
        click(&mut h, at);
    }
    let paths = path_sublayers(&h);
    let preview = h.state().session.doc.pixels.layers.clone();

    // Path controls share the horizontally scrolling brush toolbar
    h.get_all_by_label("Apply").next().unwrap().scroll_to_me();
    h.run_steps(30);
    assert!(!h.get_all_by_label("Apply").next().unwrap().accesskit_node().is_disabled());
    h.get_all_by_label("Apply").next().unwrap().click();
    h.run_steps(2);

    assert_eq!(h.state().session.ts(), texture_set);
    assert_eq!(h.state().session.history_labels().len(), steps + 1);
    assert!(h.state().session.tool.path.points.is_empty());
    assert_eq!(path_sublayers(&h), paths);
    assert_eq!(h.state().session.doc.pixels.layers, preview);
    assert!(h.get_all_by_label("Apply").next().unwrap().accesskit_node().is_disabled());
}

#[test]
fn clicking_an_applied_path_edits_it_and_dragging_a_point_changes_it() {
    let mut h = harness();
    for at in [A, B, C] {
        click(&mut h, at);
    }
    key(&mut h, Key::Enter);
    let [(eid, 1)] = path_sublayers(&h)[..] else { panic!("one path") };
    let stored = |h: &AppHarness<'_>| h.state().session.paths(eid).unwrap()[0].points.clone();
    let before = stored(&h);
    let steps = h.state().session.history_labels().len();

    // A click on the path's curve opens it for editing without repainting it
    click(&mut h, A.lerp(B, 0.5));
    assert_eq!(h.state().session.tool.path.editing, Some((eid, 0)));
    assert_eq!(h.state().session.history_labels().len(), steps, "opening a path repaints nothing");

    drag(&mut h, B, B + egui::vec2(0.0, 60.0));
    let after = stored(&h);
    assert_eq!(after.len(), 3);
    assert_eq!((after[0], after[2]), (before[0], before[2]));
    assert!(after[1][1] > before[1][1], "the dragged point moved down: {:?} -> {:?}", before[1], after[1]);
    assert_eq!(h.state().session.history_labels().len(), steps + 1, "the edit is one step");
}

#[test]
fn clicking_the_2d_view_preserves_a_stored_3d_path_being_edited() {
    let mut session =
        texturecraft_engine::Session::new_project("t", texturecraft_engine::geom::primitives::sphere(1.0, 24, 12), "builtin:test", None, 128).unwrap();
    session.camera.yaw = 0.0;
    session.camera.pitch = 0.0;
    session.update();
    let report = execute(&mut session, "tool.path", &json!({"points": [[140, 210], [260, 210]], "viewport": [400, 400], "commit": true})).unwrap();
    execute(&mut session, "path.edit", &json!({"effect": report["committed"]["effect"], "index": 0})).unwrap();
    let before = session.doc.clone();
    let path = session.tool.path.clone();
    let history = session.history_labels();
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.layout = ViewLayout::Split;
    harness.run_steps(4);
    let view_3d = harness.state().viewport_rect;
    let view_2d_center = pos2(view_3d.right() + view_3d.width() * 0.5, view_3d.center().y);

    click(&mut harness, view_2d_center);

    assert_eq!(harness.state().session.tool.path, path);
    assert_eq!(harness.state().session.doc.project, before.project);
    assert_eq!(harness.state().session.doc.pixels.layers, before.pixels.layers);
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn clicking_the_curve_inserts_a_point_and_backspace_removes_the_selected_one() {
    let mut h = harness();
    for at in [A, B, C] {
        click(&mut h, at);
    }
    let path = |h: &AppHarness<'_>| h.state().session.tool.path.clone();
    let before = path(&h).points;
    click(&mut h, B.lerp(C, 0.5));
    let after = path(&h);
    assert_eq!(after.points.len(), 4, "inserted, not appended");
    assert_eq!(after.selected, Some(2));
    assert_eq!((after.points[1], after.points[3]), (before[1], before[2]));
    assert!(after.points[2].x > before[1].x && after.points[2].x < before[2].x);

    // B, not A: clicking the first point closes the path
    click(&mut h, B);
    assert_eq!(path(&h).selected, Some(1));
    key(&mut h, Key::Backspace);
    assert_eq!(path(&h).points, vec![after.points[0], after.points[2], after.points[3]]);

    // Undo brings the point back and keeps the path; again, the inserted point goes
    let undo = |h: &mut AppHarness<'_>| {
        h.input_mut().events.push(Event::Key { key: Key::Z, physical_key: None, pressed: true, repeat: false, modifiers: Modifiers::COMMAND });
        h.run_steps(2);
    };
    undo(&mut h);
    assert_eq!(path(&h).points, after.points);
    undo(&mut h);
    assert_eq!(path(&h).points, before);
    assert!(!path_sublayers(&h).is_empty(), "the path is still there");
}
