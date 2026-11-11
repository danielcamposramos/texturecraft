//! The levels editor in Properties: handles, typed values and Reset, Auto and Invert edit the effect

use egui::accesskit::Role;
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable, by};
use serde_json::{Value, json};
use texturecraft_engine::core::{Channel, EffectKind, LayerId, Levels};
use texturecraft_engine::{Session, execute};
use texturecraft_ui_egui::TextureCraftApp;

/// Histogram buckets and Auto's share, as in levels_properties.rs
const HISTOGRAM_BUCKETS: usize = 64;
const AUTO_MIN_SHARE: f64 = 0.001;

/// Labels of the five handles
const HANDLE_LABELS: [&str; 5] = ["Input low", "Input high", "Gamma", "Output low", "Output high"];

/// Corners of a handle's pentagon pointer
const HANDLE_POINTER_CORNERS: usize = 5;

/// A clouds mask under a levels effect added by `command` with `params`, selected, and its id
fn levels_harness(command: &str, params: Value) -> (Harness<'static, TextureCraftApp>, LayerId) {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    for (id, params) in [
        ("layer.add.fill", json!({"color": [0.8, 0.3, 0.1]})),
        ("mask.add.black", Value::Null),
        ("effect.add.fill", json!({"target": "mask", "procedural": "clouds"})),
        (command, params),
    ] {
        execute(&mut session, id, &params).unwrap();
    }
    let effect = levels_effect(&session);
    open(session, effect)
}

fn open(mut session: Session, effect: LayerId) -> (Harness<'static, TextureCraftApp>, LayerId) {
    let (layer, _) = session.doc.project.effect(effect).unwrap();
    execute(&mut session, "layer.select", &json!({"layer": layer.0, "target": effect.0})).unwrap();

    // Frames of a 60 Hz screen: kittest's default quarter second spreads a double-click past egui's delay
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).with_step_dt(1.0 / 60.0).build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.run_steps(4);
    (harness, effect)
}

/// The newest levels effect in the project
fn levels_effect(session: &Session) -> LayerId {
    let layers = session.doc.project.layers.values();
    let effects = layers.flat_map(|layer| layer.content_effects.iter().chain(layer.mask.iter().flat_map(|mask| mask.effects.iter())));
    effects.filter(|effect| matches!(effect.kind, EffectKind::Levels(_))).map(|effect| effect.id).max_by_key(|id| id.0).unwrap()
}

fn levels(harness: &Harness<'_, TextureCraftApp>, effect: LayerId) -> Levels {
    match &harness.state().session.doc.project.effect(effect).unwrap().1.kind {
        EffectKind::Levels(levels) => *levels,
        other => panic!("a levels effect, found {other:?}"),
    }
}

fn undo_steps(harness: &Harness<'_, TextureCraftApp>) -> usize {
    harness.state().session.history_labels().len()
}

fn handle(harness: &Harness<'_, TextureCraftApp>, label: &str) -> egui::Pos2 {
    harness.get_by_role_and_label(Role::Slider, label).rect().center()
}

/// The button labelled `label` (an effect in the stack can share its name)
fn button<'a>(harness: &'a Harness<'_, TextureCraftApp>, label: &'a str) -> egui_kittest::Node<'a> {
    harness.get_by_role_and_label(Role::Button, label)
}

/// Presses at `from`, moves in a few steps to `to` (one frame each, like a hand) and releases
fn drag(harness: &mut Harness<'_, TextureCraftApp>, from: egui::Pos2, to: egui::Pos2) {
    harness.drag_at(from);
    harness.run_steps(2);
    for step in 1..=4 {
        harness.hover_at(from + (to - from) * (step as f32 / 4.0));
        harness.run_steps(1);
    }
    harness.drop_at(to);
    harness.run_steps(3);
}

fn set_levels(harness: &mut Harness<'_, TextureCraftApp>, effect: LayerId, mut params: Value) {
    params["effect"] = json!(effect.0);
    execute(&mut harness.state_mut().session, "effect.set", &params).unwrap();
    harness.run_steps(3);
}

#[test]
fn dragging_each_handle_edits_its_value_in_one_undo_step() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    for (label, offset_x) in [("Input low", 80.0), ("Input high", -80.0), ("Gamma", 30.0), ("Output low", 60.0), ("Output high", -60.0)] {
        let before = levels(&harness, effect);
        let steps = undo_steps(&harness);
        let at = handle(&harness, label);

        drag(&mut harness, at, at + egui::vec2(offset_x, 0.0));

        let after = levels(&harness, effect);
        let changed: Vec<&str> = [
            ("Input low", before.in_low != after.in_low),
            ("Input high", before.in_high != after.in_high),
            ("Gamma", before.gamma != after.gamma),
            ("Output low", before.out_low != after.out_low),
            ("Output high", before.out_high != after.out_high),
        ]
        .into_iter()
        .filter_map(|(name, changed)| changed.then_some(name))
        .collect();
        assert_eq!(changed, [label], "dragging {label} changes only it: {before:?} -> {after:?}");
        assert_eq!(undo_steps(&harness), steps + 1, "dragging {label} is one undo step");
    }

    let dragged = levels(&harness, effect);
    assert!(dragged.in_low > 0.0 && dragged.in_high < 1.0 && dragged.out_low > 0.0 && dragged.out_high < 1.0, "{dragged:?}");
    assert!(dragged.gamma < 1.0, "gamma right of the middle darkens: {}", dragged.gamma);
}

#[test]
fn handles_stop_at_their_neighbours() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "inLow": 0.2, "inHigh": 0.6, "outLow": 0.3, "outHigh": 0.7}));
    let far_right = egui::vec2(1000.0, 0.0);
    for label in ["Input low", "Gamma", "Output low"] {
        let at = handle(&harness, label);

        drag(&mut harness, at, at + far_right);
    }

    let edited = levels(&harness, effect);
    assert!((edited.in_low - 0.59).abs() < 1e-4, "input low stops below input high: {edited:?}");
    assert_eq!(edited.gamma, 0.01, "gamma stops at its lowest value");
    let gap = handle(&harness, "Input high").x - handle(&harness, "Gamma").x;
    assert!((0.0..1.0).contains(&gap), "the gamma handle reaches input high over this narrow range: {gap} pt apart");
    assert!((edited.out_low - 0.69).abs() < 1e-4, "output low stops below output high: {edited:?}");
}

#[test]
fn output_handles_keep_their_side_once_inverted() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "outLow": 0.2, "outHigh": 0.8}));
    button(&harness, "Invert").click();
    harness.run_steps(3);
    let inverted = levels(&harness, effect);
    assert_eq!((inverted.out_low, inverted.out_high), (0.8, 0.2));

    let at = handle(&harness, "Output high");
    drag(&mut harness, at, at + egui::vec2(1000.0, 0.0));

    let edited = levels(&harness, effect);
    assert!((edited.out_high - 0.79).abs() < 1e-4, "output high stops below output low: {edited:?}");
}

#[test]
fn reset_and_invert_are_one_undo_step_each() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    assert!(button(&harness, "Reset").accesskit_node().is_disabled(), "nothing to reset at the defaults");
    set_levels(&mut harness, effect, json!({"inLow": 0.1, "inHigh": 0.9, "gamma": 2.0, "outLow": 0.2, "outHigh": 0.7}));

    let steps = undo_steps(&harness);
    button(&harness, "Invert").click();
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect), Levels { in_low: 0.1, in_high: 0.9, gamma: 2.0, out_low: 0.7, out_high: 0.2 });
    assert_eq!(undo_steps(&harness), steps + 1);

    button(&harness, "Reset").click();
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect), Levels::default());
    assert_eq!(undo_steps(&harness), steps + 2);
}

#[test]
fn auto_fits_the_input_range_to_the_histogram() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "gamma": 1.5}));
    let counts = harness.state().session.effect_histogram(effect, Channel::BaseColor, HISTOGRAM_BUCKETS).unwrap();
    let total: u32 = counts.iter().sum();
    let holds = |count: &u32| f64::from(*count) > f64::from(total) * AUTO_MIN_SHARE;
    let low = counts.iter().position(holds).unwrap() as f32 / HISTOGRAM_BUCKETS as f32;
    let high = (counts.iter().rposition(holds).unwrap() + 1) as f32 / HISTOGRAM_BUCKETS as f32;
    assert!(low > 0.0 || high < 1.0, "the clouds don't fill the whole range: {low}..{high}");

    button(&harness, "Auto").click();
    harness.run_steps(3);

    let edited = levels(&harness, effect);
    assert_eq!((edited.in_low, edited.in_high), (low, high));
    assert_eq!(edited.gamma, 1.5, "Auto leaves gamma");

    let steps = undo_steps(&harness);
    button(&harness, "Auto").click();
    harness.run_steps(3);

    assert_eq!(undo_steps(&harness), steps, "Auto again changes nothing, so it adds no undo step");
}

/// Double-clicks `label`'s handle, then types `text` into its number field with the value selected
fn type_value(harness: &mut Harness<'_, TextureCraftApp>, label: &str, text: &str) {
    let at = handle(harness, label);
    harness.event(egui::Event::PointerMoved(at));
    for pressed in [true, false, true, false] {
        harness.event(egui::Event::PointerButton { pos: at, button: egui::PointerButton::Primary, pressed, modifiers: egui::Modifiers::NONE });
    }
    harness.run_steps(2);
    for character in text.chars() {
        harness.event(egui::Event::Text(character.to_string()));
        harness.run_steps(1);
    }
}

#[test]
fn a_value_typed_after_double_clicking_a_handle_applies() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let steps = undo_steps(&harness);

    type_value(&mut harness, "Gamma", "2.5");
    harness.key_press(egui::Key::Enter);
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect).gamma, 2.5);
    assert_eq!(undo_steps(&harness), steps + 1);
}

#[test]
fn focusing_exact_entry_moves_nothing_in_the_panel() {
    let (mut harness, _) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let buttons = |harness: &Harness<'_, TextureCraftApp>| ["Reset", "Auto", "Invert"].map(|label| button(harness, label).rect());
    let before = buttons(&harness);

    type_value(&mut harness, "Output low", "");

    assert!(harness.query_all(by().role(Role::TextInput)).any(|node| node.is_focused()), "the number field has focus");
    assert_eq!(buttons(&harness), before);
}

#[test]
fn escape_drops_a_typed_value() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let steps = undo_steps(&harness);

    type_value(&mut harness, "Output low", "0.4");
    harness.key_press(egui::Key::Escape);
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect), Levels::default());
    assert_eq!(undo_steps(&harness), steps);
    assert!(harness.query_by_value("0.4").is_none(), "the draft was discarded");
}

#[test]
fn a_typed_value_keeps_input_low_below_input_high() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "inHigh": 0.5}));
    type_value(&mut harness, "Input low", "0.9");
    harness.key_press(egui::Key::Enter);
    harness.run_steps(3);

    assert!((levels(&harness, effect).in_low - 0.49).abs() < 1e-4, "{:?}", levels(&harness, effect));
}

#[test]
fn the_levels_filter_shows_the_same_editor() {
    let (harness, _) = levels_harness("effect.add.filter", json!({"target": "mask", "filter": "levels"}));
    for label in ["Input low", "Gamma", "Input high", "Output low", "Output high"] {
        harness.get_by_role_and_label(Role::Slider, label);
    }
    for button in ["Reset", "Auto", "Invert"] {
        harness.get_by_role_and_label(Role::Button, button);
    }
}

/// Bounds of the handle pointers in paint order, the last one drawn on top
fn painted_handles(harness: &Harness<'_, TextureCraftApp>) -> Vec<egui::Rect> {
    let centres = HANDLE_LABELS.map(|label| handle(harness, label));
    let pointers = harness.output().shapes.iter().filter_map(|clipped| match &clipped.shape {
        egui::Shape::Path(path) if path.closed && path.points.len() == HANDLE_POINTER_CORNERS => Some(egui::Rect::from_points(&path.points)),
        _ => None,
    });
    pointers.filter(|bounds| centres.iter().any(|centre| bounds.expand(1.0).contains(*centre))).collect()
}

#[test]
fn the_hovered_handle_is_painted_on_top() {
    let (mut harness, _) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    assert_eq!(painted_handles(&harness).len(), HANDLE_LABELS.len(), "every handle is painted once");

    for label in HANDLE_LABELS {
        let at = handle(&harness, label);
        harness.hover_at(at);
        harness.run_steps(2);

        let on_top = painted_handles(&harness).last().copied().unwrap();
        assert!(on_top.expand(1.0).contains(at), "{label} is painted last while hovered: {on_top:?} vs {at:?}");
    }
}

#[test]
fn number_fields_are_labelled_and_stay_in_one_row() {
    let (harness, _) = levels_harness("effect.add.levels", json!({"target": "mask"}));

    let fields = HANDLE_LABELS.map(|label| harness.get_by_role_and_label(Role::TextInput, label).rect());
    let row = fields[0].center().y;
    assert!(fields.iter().all(|field| (field.center().y - row).abs() < 1.0), "number fields must share a row: {fields:?}");
}

#[test]
fn keyboard_and_accessibility_actions_edit_handles() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let steps = undo_steps(&harness);

    harness.get_by_role_and_label(Role::Slider, "Input low").focus();
    harness.run_steps(2);
    harness.key_press(egui::Key::ArrowRight);
    harness.run_steps(3);

    assert!(levels(&harness, effect).in_low > 0.0);
    assert_eq!(undo_steps(&harness), steps + 1);

    harness.get_by_role_and_label(Role::Slider, "Gamma").increment_accesskit();
    harness.run_steps(3);

    assert!(levels(&harness, effect).gamma > 1.0);

    harness.get_by_role_and_label(Role::Slider, "Output high").set_value_accesskit(0.75);
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect).out_high, 0.75);
    assert_eq!(undo_steps(&harness), steps + 3);
}

#[test]
fn invalid_typed_values_stay_open_with_feedback() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let steps = undo_steps(&harness);

    for text in ["invalid", "NaN", "inf"] {
        type_value(&mut harness, "Gamma", text);
        harness.key_press(egui::Key::Enter);
        harness.run_steps(3);

        let fields: Vec<_> = harness.query_all(by().role(Role::TextInput)).map(|node| (node.is_focused(), node.value())).collect();
        assert!(fields.iter().any(|(focused, value)| *focused && value.as_deref() == Some(text)), "{text}: {fields:?}");
        harness.get_by_label("Enter a finite number");
        assert_eq!(levels(&harness, effect), Levels::default());
        assert_eq!(undo_steps(&harness), steps);

        harness.key_press(egui::Key::Escape);
        // Keep the next double-click separate from this one
        harness.run_steps(30);
    }
}

#[test]
fn opening_exact_entry_does_not_round_the_saved_value() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "gamma": 1.23456}));
    let steps = undo_steps(&harness);

    type_value(&mut harness, "Gamma", "");
    harness.key_press(egui::Key::Enter);
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect).gamma, 1.23456);
    assert_eq!(undo_steps(&harness), steps);
}

#[test]
fn keyboard_entry_can_correct_an_invalid_value() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let steps = undo_steps(&harness);

    harness.get_by_role_and_label(Role::Slider, "Gamma").focus();
    harness.run_steps(2);
    harness.key_press(egui::Key::Enter);
    harness.run_steps(2);
    assert!(harness.get_by_role_and_label(Role::TextInput, "Gamma").is_focused());

    harness.get_by_role_and_label(Role::TextInput, "Gamma").type_text("bad");
    harness.key_press(egui::Key::Enter);
    harness.run_steps(3);
    harness.get_by_label("Enter a finite number");

    harness.get_by_role_and_label(Role::TextInput, "Gamma").set_text_value_accesskit("2.5");
    harness.run_steps(2);
    harness.key_press(egui::Key::Enter);
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect).gamma, 2.5);
    assert_eq!(undo_steps(&harness), steps + 1);
    assert!(harness.query_by_label("Enter a finite number").is_none());
}

#[test]
fn dragging_near_a_handle_uses_the_larger_target() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask"}));
    let at = handle(&harness, "Gamma") + egui::vec2(9.0, 9.0);
    let steps = undo_steps(&harness);

    drag(&mut harness, at, at + egui::vec2(30.0, 0.0));

    assert!(levels(&harness, effect).gamma < 1.0);
    assert_eq!(undo_steps(&harness), steps + 1);
}

#[test]
fn number_fields_commit_on_focus_loss() {
    let (mut harness, effect) = levels_harness("effect.add.levels", json!({"target": "mask", "inLow": 0.2, "inHigh": 0.8}));
    let steps = undo_steps(&harness);

    harness.get_by_role_and_label(Role::TextInput, "Input low").focus();
    harness.run_steps(2);
    harness.get_by_role_and_label(Role::TextInput, "Input low").set_text_value_accesskit("0.3");
    harness.run_steps(2);
    harness.get_by_role_and_label(Role::TextInput, "Gamma").focus();
    harness.run_steps(3);

    assert_eq!(levels(&harness, effect), Levels { in_low: 0.3, in_high: 0.8, ..Levels::default() });
    assert_eq!(undo_steps(&harness), steps + 1);
}

#[test]
fn replacing_effects_releases_old_histograms() {
    let (mut harness, _) = levels_harness("effect.add.levels", json!({"target": "mask"}));

    for _ in 0..20 {
        execute(&mut harness.state_mut().session, "effect.remove", &Value::Null).unwrap();
        execute(&mut harness.state_mut().session, "effect.add.levels", &json!({"target": "mask"})).unwrap();
        harness.run_steps(3);
    }

    let cached = harness.ctx.data(|data| data.count::<(u64, std::sync::Arc<Vec<u32>>)>());
    assert_eq!(cached, 1, "only the current effect's histogram remains cached");
}
