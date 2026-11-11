//! The sliders of a selected effect (opacity, generator, levels…) apply what they were dragged
//! or clicked to.

use egui::accesskit::Role;
use egui_kittest::Harness;
use egui_kittest::kittest::{Queryable, by};
use serde_json::json;
use texturecraft_engine::core::{EffectKind, LayerId};
use texturecraft_ui_egui::TextureCraftApp;

/// The sample session with its first generator effect selected, and that effect's id.
fn harness() -> (Harness<'static, TextureCraftApp>, LayerId) {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    let (layer, effect) = session
        .doc
        .project
        .layers
        .values()
        .find_map(|l| {
            let mask = l.mask.iter().flat_map(|m| m.effects.iter());
            l.content_effects.iter().chain(mask).find(|e| matches!(e.kind, EffectKind::Generator(_))).map(|e| (l.id, e.id))
        })
        .unwrap();
    texturecraft_engine::execute(&mut session, "layer.select", &json!({"layer": layer.0, "target": effect.0})).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    h.run_steps(4);
    (h, effect)
}

fn opacity(h: &Harness<'_, TextureCraftApp>, effect: LayerId) -> f32 {
    h.state().session.doc.project.effect(effect).unwrap().1.opacity
}

/// The track under the slider labelled `label` (the row below the label).
fn track(h: &Harness<'_, TextureCraftApp>, label: &str) -> egui::Rect {
    let l = h.get_by_label(label).rect();
    egui::Rect::from_min_max(egui::pos2(l.left() + 6.0, l.bottom() + 6.0), egui::pos2(l.left() + 200.0, l.bottom() + 14.0))
}

#[test]
fn dragging_the_effect_opacity_applies_it() {
    let (mut h, effect) = harness();
    assert_eq!(opacity(&h, effect), 1.0);
    let t = track(&h, "Opacity");
    h.drag_at(egui::pos2(t.right(), t.center().y));
    h.run_steps(2);
    h.hover_at(egui::pos2(t.left() + 20.0, t.center().y));
    h.run_steps(2);
    h.drop_at(egui::pos2(t.left() + 20.0, t.center().y));
    h.run_steps(3);
    assert!(opacity(&h, effect) < 0.5, "the dragged opacity applies: {}", opacity(&h, effect));
}

/// A generator slider applies while it is dragged (the texture follows the pointer), and the
/// whole drag is one undo step.
#[test]
fn dragging_a_generator_slider_applies_live_as_one_undo_step() {
    let (mut h, effect) = harness();
    let level = |h: &Harness<'_, TextureCraftApp>| match &h.state().session.doc.project.effect(effect).unwrap().1.kind {
        EffectKind::Generator(g) => g.level,
        _ => panic!("a generator"),
    };
    let steps = h.state().session.history_labels().len();
    let t = track(&h, "Level");
    let at = |x: f32| egui::pos2(t.left() + x, t.center().y);
    h.drag_at(at(10.0));
    h.run_steps(2);
    let mut last = 0.0;
    for x in [60.0, 160.0] {
        h.hover_at(at(x));
        h.run_steps(2);
        assert!(level(&h) > last && level(&h) != 0.5, "the level follows the drag before it is released: {last} -> {}", level(&h));
        last = level(&h);
    }
    h.drop_at(at(160.0));
    h.run_steps(3);
    assert_eq!(level(&h), last);
    assert_eq!(h.state().session.history_labels().len(), steps + 1, "one undo step for the drag");
}

/// The combo box in the row of the levels editor's Reset button.
fn channel_picker<'a>(h: &'a Harness<'_, TextureCraftApp>) -> egui_kittest::Node<'a> {
    let row = h.get_by_label("Reset").rect().center().y;
    h.get_all(by().role(Role::ComboBox)).find(|node| (node.rect().center().y - row).abs() < 4.0).unwrap()
}

/// A content levels effect offers a channel for its histogram; picking one keeps it selected.
#[test]
fn content_levels_pick_the_histogram_channel() {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    texturecraft_engine::execute(&mut session, "layer.add.fill", &json!({"color": [0.5, 0.2, 0.1], "roughness": 0.6})).unwrap();
    texturecraft_engine::execute(&mut session, "effect.add.levels", &json!({"target": "content"})).unwrap();
    let mut h = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    h.run_steps(4);
    assert_eq!(channel_picker(&h).value().as_deref(), Some("Base color"), "the layer's first channel");
    channel_picker(&h).click();
    h.run_steps(2);
    h.get_by_label("Roughness").click();
    h.run_steps(3);
    assert_eq!(channel_picker(&h).value().as_deref(), Some("Roughness"), "the picked channel stays");
}

#[test]
fn clicking_the_effect_opacity_applies_it() {
    let (mut h, effect) = harness();
    let t = track(&h, "Opacity");
    let at = egui::pos2(t.left() + 20.0, t.center().y);
    h.drag_at(at);
    h.drop_at(at);
    h.run_steps(3);
    assert!(opacity(&h, effect) < 0.5, "a click on the track applies: {}", opacity(&h, effect));
}

/// The number field in the row of the slider labelled `label`.
fn number_field<'a>(h: &'a Harness<'_, TextureCraftApp>, label: &str) -> egui_kittest::Node<'a> {
    let row = h.get_by_label(label).rect().center().y;
    h.get_all(by().role(Role::TextInput)).find(|n| (n.rect().center().y - row).abs() < 4.0).unwrap()
}

/// Focus the opacity field and press `keys`, then type `text`, one key per frame like a person.
fn type_opacity(h: &mut Harness<'_, TextureCraftApp>, keys: &[egui::Key], text: &str) {
    number_field(h, "Opacity").focus();
    h.run_steps(2);
    for k in keys {
        h.key_press(*k);
        h.run_steps(1);
    }
    for c in text.chars() {
        number_field(h, "Opacity").type_text(&c.to_string());
        h.run_steps(1);
    }
}

#[test]
fn a_value_typed_in_the_number_field_applies() {
    let (mut h, effect) = harness();
    let steps = h.state().session.history_labels().len();
    type_opacity(&mut h, &[egui::Key::End, egui::Key::Backspace, egui::Key::Backspace, egui::Key::Backspace, egui::Key::Backspace], "0.25");
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(opacity(&h, effect), 0.25);
    assert_eq!(h.state().session.history_labels().len(), steps + 1);
}

/// Dragging the track while the number field has focus drops what was typed there: leaving the
/// field afterwards does not undo the drag.
#[test]
fn a_track_drag_drops_text_typed_in_the_number_field() {
    let (mut h, effect) = harness();
    type_opacity(&mut h, &[egui::Key::End, egui::Key::Backspace], "5");
    let t = track(&h, "Opacity");
    h.drag_at(egui::pos2(t.right(), t.center().y));
    h.run_steps(2);
    h.hover_at(egui::pos2(t.left() + 20.0, t.center().y));
    h.run_steps(2);
    h.drop_at(egui::pos2(t.left() + 20.0, t.center().y));
    h.run_steps(2);
    let dragged = opacity(&h, effect);
    assert!(dragged < 0.5, "the drag applies: {dragged}");
    // Leave whatever has focus: a click on empty space.
    h.drag_at(egui::pos2(5.0, 1390.0));
    h.drop_at(egui::pos2(5.0, 1390.0));
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(opacity(&h, effect), dragged, "the text typed before the drag stays dropped");
}

/// Leaving the number field without typing, or with Escape, keeps the value (not the rounded
/// text the field shows) and makes no undo step.
#[test]
fn leaving_the_number_field_unchanged_keeps_the_value() {
    let (mut h, effect) = harness();
    texturecraft_engine::execute(&mut h.state_mut().session, "effect.set", &json!({"effect": effect.0, "opacity": 0.333})).unwrap();
    h.run_steps(2);
    let steps = h.state().session.history_labels().len();
    type_opacity(&mut h, &[], "");
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(opacity(&h, effect), 0.333, "left without typing");
    type_opacity(&mut h, &[egui::Key::End, egui::Key::Backspace], "9");
    assert_eq!(number_field(&h, "Opacity").value().as_deref(), Some("0.39"));
    assert!(number_field(&h, "Opacity").is_focused());
    h.key_press(egui::Key::Escape);
    h.run_steps(3);
    assert_eq!(opacity(&h, effect), 0.333);
    // Typing the value it already has is no edit either.
    type_opacity(&mut h, &[egui::Key::End, egui::Key::Backspace, egui::Key::Backspace, egui::Key::Backspace, egui::Key::Backspace], "0.333");
    h.key_press(egui::Key::Enter);
    h.run_steps(3);
    assert_eq!(h.state().session.history_labels().len(), steps);
}
