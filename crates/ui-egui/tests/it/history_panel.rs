//! A click on a step of the History panel goes back or forward to it.

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::json;
use texturecraft_engine::geom::glam::Vec3;
use texturecraft_engine::{Session, execute};
use texturecraft_ui_egui::TextureCraftApp;
use texturecraft_ui_egui::docking::Panel;

/// Show document history so row clicks exercise the same command dispatch as the application
fn history_harness(session: Session) -> Harness<'static, TextureCraftApp> {
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.dock.focus(Panel::History);
    harness.run_steps(4);
    harness
}

/// Three document rows provide an intermediate target that detects both short and excessive jumps
fn renamed_session() -> Session {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    for name in ["Step A", "Step B", "Step C"] {
        execute(&mut session, "layer.rename", &json!({"name": name})).unwrap();
    }
    session
}

/// Canceling a preview can be its own undo or part of undoing the first placed point
#[test]
fn history_jump_accounts_for_drafts_with_and_without_an_empty_checkpoint() {
    for placed in [false, true] {
        let mut session = renamed_session();
        let layer = session.selected_layer().unwrap();
        let effect_count = session.doc.project.layer(layer).unwrap().content_effects.len();
        if placed {
            execute(&mut session, "tool.select", &json!({"tool": "path"})).unwrap();
            for point in [Vec3::new(0.2, 0.5, 0.0), Vec3::new(0.5, 0.3, 0.0), Vec3::new(0.8, 0.5, 0.0)] {
                session.append_path_point(point, true);
                session.path_live().unwrap();
            }
        } else {
            execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]]})).unwrap();
            session.move_path_point(0, Vec3::new(0.2, 0.3, 0.0), true);
            session.path_live().unwrap();
        }
        session.path_checkpoint();
        let mut harness = history_harness(session);

        harness.get_all_by_label("Rename layer").next().unwrap().click();
        harness.run_steps(3);

        assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().name, "Step A");
        assert_eq!(harness.state().session.history_labels().len(), 1);
        assert_eq!(harness.state().session.redo_labels().len(), 2);
        assert!(harness.state().session.tool.path.points.is_empty());
        assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().content_effects.len(), effect_count);
    }
}

/// A row jump traverses point history, while the Undo and Redo buttons still take one point step
#[test]
fn history_jump_accounts_for_stored_point_edits_without_changing_single_steps() {
    let mut session = renamed_session();
    let layer = session.selected_layer().unwrap();
    let report = execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": true})).unwrap();
    execute(&mut session, "path.edit", &json!({"effect": report["committed"]["effect"], "index": 0})).unwrap();
    let original = session.tool.path.points.clone();
    session.move_path_point(0, Vec3::new(0.2, 0.3, 0.0), true);
    session.path_live().unwrap();
    let moved = session.tool.path.points.clone();
    session.move_path_point(0, original[0], true);
    session.path_live().unwrap();
    session.path_checkpoint();
    let done = session.history_labels().len();
    let mut harness = history_harness(session);

    harness.get_by_label(&format!("{} Undo", egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE)).click();
    harness.run_steps(3);
    assert_eq!(harness.state().session.tool.path.points, moved);
    assert_eq!(harness.state().session.history_labels().len(), done);
    harness.get_by_label(&format!("{} Redo", egui_phosphor::regular::ARROW_CLOCKWISE)).click();
    harness.run_steps(3);
    assert_eq!(harness.state().session.tool.path.points, original);
    assert_eq!(harness.state().session.history_labels().len(), done);

    harness.get_by_label("Edit path 1").click();
    harness.run_steps(3);
    assert_eq!(harness.state().session.tool.path.points, original);
    assert_eq!(harness.state().session.history_labels().len(), done);

    harness.get_all_by_label("Rename layer").next().unwrap().click();
    harness.run_steps(3);

    assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().name, "Step A");
    assert_eq!(harness.state().session.history_labels().len(), 1);
    assert_eq!(harness.state().session.redo_labels().len(), done - 1);
    assert!(harness.state().session.tool.path.points.is_empty());
}

/// Incomplete filled curves keep point redo separate from the document redo rows
#[test]
fn history_redo_jump_accounts_for_point_steps_without_changing_single_steps() {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    let layer = session.selected_layer().unwrap();
    let report =
        execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.5, 0.3], [0.8, 0.5]], "closed": true, "fill": true, "commit": true})).unwrap();
    for name in ["Step A", "Step B", "Step C"] {
        execute(&mut session, "layer.rename", &json!({"name": name})).unwrap();
    }
    let done = session.history_labels().len();
    execute(&mut session, "edit.undo", &json!({"steps": 3})).unwrap();
    execute(&mut session, "path.edit", &json!({"effect": report["committed"]["effect"], "index": 0})).unwrap();
    let original = session.tool.path.points.clone();

    // Partial filled shapes do not repaint; their undo/redo leaves document redo available
    for points in [json!([[0.2, 0.5], [0.8, 0.5]]), json!([[0.2, 0.5]]), json!([[0.2, 0.5], [0.8, 0.5]])] {
        session.path_checkpoint();
        execute(&mut session, "tool.path", &json!({"uvPoints": points})).unwrap();
    }
    for _ in 0..3 {
        session.undo().unwrap();
    }
    assert_eq!(session.redo_labels().len(), 3);
    let mut harness = history_harness(session);

    harness.get_by_label(&format!("{} Redo", egui_phosphor::regular::ARROW_CLOCKWISE)).click();
    harness.run_steps(3);
    assert_eq!(harness.state().session.tool.path.points.len(), 2);
    assert_eq!(harness.state().session.history_labels().len(), done - 3);
    harness.get_by_label(&format!("{} Undo", egui_phosphor::regular::ARROW_COUNTER_CLOCKWISE)).click();
    harness.run_steps(3);
    assert_eq!(harness.state().session.tool.path.points, original);
    assert_eq!(harness.state().session.history_labels().len(), done - 3);

    harness.get_all_by_label("Rename layer (undone)").last().unwrap().click();
    harness.run_steps(3);

    assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().name, "Step C");
    assert_eq!(harness.state().session.history_labels().len(), done);
    assert!(harness.state().session.redo_labels().is_empty());
    assert!(harness.state().session.tool.path.points.is_empty());
}

/// Returning to the current document row cancels a draft without undoing a document step
#[test]
fn clicking_the_current_history_row_cancels_a_draft() {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    let layer = session.selected_layer().unwrap();
    let effect_count = session.doc.project.layer(layer).unwrap().content_effects.len();
    execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]]})).unwrap();
    let mut harness = history_harness(session);

    harness.get_by_label("Project opened").click();
    harness.run_steps(3);

    assert!(harness.state().session.tool.path.points.is_empty());
    assert!(harness.state().session.history_labels().is_empty());
    assert!(harness.state().session.redo_labels().is_empty());
    assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().content_effects.len(), effect_count);
}

#[test]
fn clicking_a_history_step_returns_to_it() {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    let layer = session.selected_layer().unwrap();
    for name in ["Step A", "Step B", "Step C"] {
        texturecraft_engine::execute(&mut session, "layer.rename", &json!({"layer": layer.0, "name": name})).unwrap();
    }
    let done = session.history_labels().len();
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    h.state_mut().ui.dock.focus(Panel::History);
    h.run_steps(4);
    let name = |h: &Harness<'_, TextureCraftApp>| h.state().session.doc.project.layers[&layer].name.clone();
    // The first state: everything undone, and listed as undone.
    h.get_by_label("Project opened").click();
    h.run_steps(3);
    assert!(!h.state().session.can_undo());
    assert_eq!(h.state().session.redo_labels().len(), done);
    // The first undone row is the next step: one redo.
    h.get_all_by_label("Rename layer (undone)").next().unwrap().click();
    h.run_steps(3);
    assert_eq!(h.state().session.history_labels().len(), 1);
    assert_eq!(h.state().session.redo_labels().len(), done - 1);
    // The last undone row: all the way forward again.
    h.get_all_by_label("Rename layer (undone)").last().unwrap().click();
    h.run_steps(3);
    assert_eq!(name(&h), "Step C");
    assert!(!h.state().session.can_redo());
}
