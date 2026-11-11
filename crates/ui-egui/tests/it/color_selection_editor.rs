//! The color selection editor in Properties: one row per ID color, remove, and the pick mode

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use serde_json::{Value, json};
use texturecraft_engine::core::{BakedMap, EffectKind, LayerId};
use texturecraft_engine::{StackTarget, ViewMode, execute};
use texturecraft_ui_egui::TextureCraftApp;

const FIRST: [f32; 3] = [200.0 / 255.0, 0.0, 0.0];
const SECOND: [f32; 3] = [10.0 / 255.0, 1.0, 0.0];
const THIRD: [f32; 3] = [30.0 / 255.0, 0.0, 1.0];
const ID_PART: [u8; 3] = [90, 160, 220];

/// An ID map of one color, as a PNG
fn id_map_png() -> Vec<u8> {
    let [red, green, blue] = ID_PART;
    let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([red, green, blue, 255]));
    let mut bytes = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
    bytes
}

/// A selected color selection of three colors on a fill layer's mask
fn color_selection_session(with_id_map: bool) -> (texturecraft_engine::Session, LayerId) {
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    if with_id_map {
        let image = session.import_resource_bytes("id.png", id_map_png()).unwrap();
        execute(&mut session, "textureSet.meshMap.set", &json!({"map": "id", "image": image})).unwrap();
    } else {
        // The sample bakes an ID map with the others
        execute(&mut session, "textureSet.meshMap.clear", &json!({"map": "id"})).unwrap();
    }

    let layer = LayerId(execute(&mut session, "layer.add.fill", &Value::Null).unwrap()["layer"].as_u64().unwrap());
    execute(&mut session, "mask.add.colorSelection", &Value::Null).unwrap();
    let effect = session.doc.project.layer(layer).unwrap().mask.as_ref().unwrap().effects[0].id;
    execute(&mut session, "effect.set", &json!({"effect": effect.0, "colors": [FIRST, SECOND, THIRD]})).unwrap();
    session.target = StackTarget::Effect(effect);
    (session, effect)
}

fn app_with_color_selection(with_id_map: bool) -> (Harness<'static, TextureCraftApp>, LayerId) {
    let (session, effect) = color_selection_session(with_id_map);
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 1400.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.run_steps(4);
    (harness, effect)
}

fn colors(harness: &Harness<'_, TextureCraftApp>, effect: LayerId) -> Vec<[f32; 3]> {
    match &harness.state().session.doc.project.effect(effect).unwrap().1.kind {
        EffectKind::ColorSelection { colors, .. } => colors.clone(),
        other => panic!("not a color selection: {other:?}"),
    }
}

fn click_at(harness: &mut Harness<'_, TextureCraftApp>, position: Pos2) {
    harness.event(Event::PointerMoved(position));
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(3);
}

/// Whether the last frame drew the eyedropper glyph with its tip at `pointer`
fn eyedropper_at(harness: &Harness<'_, TextureCraftApp>, pointer: Pos2) -> bool {
    harness.output().shapes.iter().any(|clipped| match &clipped.shape {
        egui::Shape::Text(text) => text.galley.text() == egui_phosphor::regular::EYEDROPPER && (text.pos.x - pointer.x).abs() < 1.0,
        _ => false,
    })
}

fn pick_color_label() -> String {
    format!("{} Pick color", egui_phosphor::regular::EYEDROPPER)
}

#[test]
fn each_color_gets_a_row_and_remove_sends_the_list_without_it() {
    let (mut harness, effect) = app_with_color_selection(false);
    for readout in ["R: 200", "R: 10", "R: 30"] {
        assert!(harness.query_by_label(readout).is_some(), "{readout} row");
    }

    let second_row = harness.get_by_label("R: 10").rect().center().y;
    let remove = harness
        .query_all_by_label("Remove")
        .find(|button| (button.rect().center().y - second_row).abs() < 8.0)
        .expect("the second row's remove button")
        .rect()
        .center();
    click_at(&mut harness, remove);

    assert_eq!(colors(&harness, effect), vec![FIRST, THIRD]);
}

#[test]
fn pick_mode_shows_the_id_map_and_a_click_on_the_mesh_adds_its_color() {
    let (mut harness, effect) = app_with_color_selection(true);
    harness.get_by_label(&pick_color_label()).click();
    harness.run_steps(3);
    assert!(harness.state().picking_id_color.is_some_and(|pick| pick.effect == effect && pick.index.is_none()));
    assert_eq!(harness.state().session.view, ViewMode::MeshMap(BakedMap::Id));

    let center = harness.state().viewport_rect.center();
    harness.event(Event::PointerMoved(center));
    harness.run_steps(2);
    assert!(eyedropper_at(&harness, center), "the pointer is an eyedropper");

    click_at(&mut harness, center);

    let picked = colors(&harness, effect);
    assert_eq!(picked.len(), 4, "the click adds a row");
    assert_eq!(picked[3], ID_PART.map(|channel| channel as f32 / 255.0));
    assert!(harness.state().picking_id_color.is_none(), "one click ends pick mode");
    assert_eq!(harness.state().session.view, ViewMode::Material, "the earlier view comes back");
}

#[test]
fn without_an_id_map_picking_is_off_and_a_warning_says_why() {
    let (mut harness, _) = app_with_color_selection(false);
    let warning = format!("{} Bake an ID map (F8) to pick colors from the mesh.", egui_phosphor::regular::WARNING);
    assert!(harness.query_by_label(&warning).is_some());

    harness.get_by_label(&pick_color_label()).click();
    harness.run_steps(3);
    assert!(harness.state().picking_id_color.is_none());
}

fn row_button(harness: &Harness<'_, TextureCraftApp>, readout: &str, label: &str) -> Pos2 {
    let row_y = harness.get_by_label(readout).rect().center().y;
    harness.query_all_by_label(label).find(|button| (button.rect().center().y - row_y).abs() < 8.0).unwrap().rect().center()
}

#[test]
fn removing_a_row_cancels_a_pending_replacement() {
    let (mut harness, effect) = app_with_color_selection(true);
    let repick = row_button(&harness, "R: 10", "Pick color");
    click_at(&mut harness, repick);
    assert_eq!(harness.state().picking_id_color.unwrap().index, Some(1));

    let remove = row_button(&harness, "R: 200", "Remove");
    click_at(&mut harness, remove);

    assert_eq!(colors(&harness, effect), vec![SECOND, THIRD]);
    assert!(harness.state().picking_id_color.is_none(), "removing a row must not leave a replacement targeting a different color");
    assert_eq!(harness.state().session.view, ViewMode::Material);
}

#[test]
fn escape_cancels_the_pick_without_editing_colors() {
    let (mut harness, effect) = app_with_color_selection(true);
    harness.get_by_label(&pick_color_label()).click();
    harness.run_steps(3);
    let history = harness.state().session.history_labels();

    harness.key_press(egui::Key::Escape);
    harness.run_steps(3);

    assert!(harness.state().picking_id_color.is_none());
    assert_eq!(harness.state().session.view, ViewMode::Material);
    assert_eq!(colors(&harness, effect), vec![FIRST, SECOND, THIRD]);
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn escape_cancels_id_picking_with_a_hidden_rail_popup() {
    let (mut harness, effect) = app_with_color_selection(true);
    harness.get_by_label(&pick_color_label()).click();
    harness.run_steps(3);
    let context = harness.ctx.clone();
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &context, "ui.view.log");
    harness.run_steps(3);
    harness.get_by_role_and_label(egui::Role::Button, "Close");
    harness.key_press(egui::Key::Tab);
    harness.run_steps(3);
    assert!(harness.state().ui.hide_ui);
    assert!(harness.query_by_role_and_label(egui::Role::Button, "Close").is_none());
    assert!(harness.state().picking_id_color.is_some());
    let history = harness.state().session.history_labels();

    harness.key_press(egui::Key::Escape);
    harness.run_steps(3);

    assert!(harness.state().picking_id_color.is_none(), "one Escape cancels ID picking despite the hidden popup");
    assert_eq!(harness.state().session.view, ViewMode::Material);
    assert_eq!(colors(&harness, effect), vec![FIRST, SECOND, THIRD]);
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn picking_from_the_popup_closes_it_before_the_viewport_click() {
    use texturecraft_ui_egui::properties_popup::{self, PropertiesPopup};

    let (mut harness, effect) = app_with_color_selection(true);
    let bounds = harness.state().viewport_rect;
    harness.state_mut().properties_popup = Some(PropertiesPopup::Open { at: bounds.center() - egui::vec2(100.0, 100.0), bounds });
    harness.run_steps(4);
    let popup = harness.ctx.memory(|memory| memory.area_rect(properties_popup::area_id())).unwrap();
    let pick = harness.query_all_by_label(&pick_color_label()).find(|button| popup.contains_rect(button.rect())).unwrap().rect().center();
    click_at(&mut harness, pick);

    assert!(harness.state().picking_id_color.is_some());
    let center = harness.state().viewport_rect.center();
    assert!(popup.contains(center), "pick a point the popup would otherwise cover");
    click_at(&mut harness, center);
    assert_eq!(colors(&harness, effect).len(), 4, "the next viewport click must pick, even under the former popup");
    assert_eq!(harness.state().properties_popup, None);
}

#[test]
fn a_2d_pick_replaces_the_chosen_row_and_undo_restores_it() {
    use texturecraft_ui_egui::state::ViewLayout;

    let (mut harness, effect) = app_with_color_selection(true);
    harness.state_mut().ui.layout = ViewLayout::Only3D;
    harness.run_steps(3);
    let center = harness.state().viewport_rect.center();
    harness.state_mut().ui.layout = ViewLayout::Only2D;
    harness.run_steps(3);
    let repick = row_button(&harness, "R: 10", "Pick color");
    click_at(&mut harness, repick);
    click_at(&mut harness, center);

    assert_eq!(colors(&harness, effect), vec![FIRST, ID_PART.map(|channel| channel as f32 / 255.0), THIRD]);
    assert_eq!(harness.state().session.view, ViewMode::Material);
    assert!(!harness.state().session.stroke_active());
    harness.state_mut().session.undo().unwrap();
    assert_eq!(colors(&harness, effect), vec![FIRST, SECOND, THIRD]);
}

#[test]
fn deselection_cancels_picking_and_preserves_a_view_chosen_meanwhile() {
    let (mut harness, effect) = app_with_color_selection(true);
    harness.get_by_label(&pick_color_label()).click();
    harness.run_steps(3);
    execute(&mut harness.state_mut().session, "viewport.view", &json!({"mode": "uvChecker"})).unwrap();
    harness.state_mut().session.target = StackTarget::Content;
    harness.run_steps(3);

    assert!(harness.state().picking_id_color.is_none());
    assert_eq!(harness.state().session.view, ViewMode::UvChecker);
    assert_eq!(colors(&harness, effect), vec![FIRST, SECOND, THIRD]);
}

#[test]
#[ignore = "requires a GPU adapter; set TC_UI_SNAPSHOTS to save the rendered views"]
fn color_selection_snapshots_and_quick_mask_pick() {
    let (session, _) = color_selection_session(true);
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|context| {
        let mut app = TextureCraftApp::new(context, session);
        app.ui.layout = texturecraft_ui_egui::state::ViewLayout::Only3D;
        app
    });
    let adapter = harness.state().render_state.as_ref().unwrap().adapter.get_info();
    eprintln!("Offscreen adapter: {adapter:?}");

    let save = |harness: &mut Harness<'_, TextureCraftApp>, name: &str| {
        harness.run_steps(8);
        let image = harness.render().unwrap();
        if let Some(directory) = std::env::var_os("TC_UI_SNAPSHOTS") {
            let path = std::path::Path::new(&directory).join(format!("{name}.png"));
            image.save(&path).unwrap();
            eprintln!("wrote {}", path.display());
        }
        image
    };

    save(&mut harness, "color-selection-editor");
    harness.get_by_label(&pick_color_label()).click();
    let center = harness.state().viewport_rect.center();
    harness.hover_at(center + egui::vec2(80.0, 0.0));
    let expected = save(&mut harness, "color-selection-picking");
    execute(&mut harness.state_mut().session, "viewport.quickMask.edit", &json!({"editing": true})).unwrap();
    let with_quick_mask = save(&mut harness, "color-selection-picking-quick-mask");
    assert_eq!(
        with_quick_mask.get_pixel(center.x as u32, center.y as u32),
        expected.get_pixel(center.x as u32, center.y as u32),
        "picking must display the ID map while quick-mask editing is enabled"
    );

    click_at(&mut harness, center);
    assert!(harness.state().session.quick_mask.editing, "picking preserves quick-mask editing");
    save(&mut harness, "color-selection-quick-mask-restored");
    execute(&mut harness.state_mut().session, "viewport.quickMask.edit", &json!({"editing": false})).unwrap();
    harness.state_mut().ui.layout = texturecraft_ui_egui::state::ViewLayout::Only2D;
    harness.get_by_label(&pick_color_label()).click();
    save(&mut harness, "color-selection-picking-2d");
    click_at(&mut harness, center);
    assert!(harness.state().picking_id_color.is_none());
    execute(&mut harness.state_mut().session, "textureSet.meshMap.clear", &json!({"map": "id"})).unwrap();
    save(&mut harness, "color-selection-no-id-map");
}
