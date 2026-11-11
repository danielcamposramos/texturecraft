//! A shelf material or image dropped on the 3D view goes to the texture set under the pointer; an image first asks what it becomes

use egui::{Event, Modifiers, PointerButton, Pos2};
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use texturecraft_engine::Tool;
use texturecraft_engine::core::Channel;
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_ui_egui::TextureCraftApp;
use texturecraft_ui_egui::shelf::{Asset, ShelfDrag};
use texturecraft_ui_egui::state::ViewLayout;

/// Distance between the view points searched for a set or the background
const SEARCH_STEP: f32 = 8.0;

/// How far from the drop point the drop menu's items can be
const MENU_REACH: f32 = 60.0;

/// The sample project (Ball = set 0, Base = set 1) with the Ball selected, at 1600 x 960
fn sample_app() -> Harness<'static, TextureCraftApp> {
    let session = texturecraft_engine::sample_session(256, 64).unwrap();
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.run_steps(4);
    assert_eq!(harness.state().session.ts(), 0);
    harness
}

/// The first point of the 3D view showing texture set `set`, or the background when `None`
fn point_over(harness: &Harness<'_, TextureCraftApp>, set: Option<usize>) -> Pos2 {
    let app = harness.state();
    let rect = app.viewport_rect;
    let steps = |length: f32| (1..(length / SEARCH_STEP) as usize).map(|step| step as f32 * SEARCH_STEP);

    steps(rect.height())
        .flat_map(|y| steps(rect.width()).map(move |x| (x, y)))
        .find(|&(x, y)| app.session.texture_set_at(Vec2::new(x, y), app.session.viewport) == set)
        .map(|(x, y)| rect.min + egui::vec2(x, y))
        .unwrap()
}

/// Drop a built-in shelf material at `position`, as a drag from the shelf ends there
fn drop_material(harness: &mut Harness<'_, TextureCraftApp>, position: Pos2) {
    drop_asset(harness, Asset::Material("iron_rusted"), position);
}

/// Drop shelf `asset` at `position`, as a drag from the shelf ends there
fn drop_asset(harness: &mut Harness<'_, TextureCraftApp>, asset: Asset, position: Pos2) {
    egui::DragAndDrop::set_payload(&harness.ctx, ShelfDrag { asset, label: "Dragged".into() });
    harness.hover_at(position);
    harness.run_steps(2);

    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(2);
}

/// Top-level layer count of each texture set
fn stack_sizes(harness: &Harness<'_, TextureCraftApp>) -> Vec<usize> {
    harness.state().session.doc.project.texture_sets.iter().map(|set| set.stack.len()).collect()
}

#[test]
fn a_material_dropped_on_another_set_goes_to_that_set_and_selects_it() {
    let mut harness = sample_app();
    let before = stack_sizes(&harness);

    let base = point_over(&harness, Some(1));
    drop_material(&mut harness, base);

    assert_eq!(harness.state().session.ts(), 1, "the set under the pointer is selected");
    assert_eq!(stack_sizes(&harness), vec![before[0], before[1] + 1], "only the set under the pointer gets the material");
}

#[test]
fn a_material_dropped_on_the_background_changes_nothing() {
    let mut harness = sample_app();
    let (stacks, history) = (stack_sizes(&harness), harness.state().session.history_labels().len());

    let background = point_over(&harness, None);
    drop_material(&mut harness, background);

    assert_eq!(harness.state().session.ts(), 0);
    assert_eq!(stack_sizes(&harness), stacks);
    assert_eq!(harness.state().session.history_labels().len(), history);
}

/// Import a small white image and return its resource name
fn import_white_image(harness: &mut Harness<'_, TextureCraftApp>) -> String {
    let mut png = Vec::new();
    image::RgbaImage::from_pixel(4, 4, image::Rgba([255, 255, 255, 255])).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let name = harness.state_mut().session.import_resource_bytes("logo.png", png).unwrap();
    harness.run_steps(2);
    name
}

/// The sample app in English with a white image dropped on texture set `position_over` (the background when `None`)
fn drop_image(position_over: Option<usize>) -> (Harness<'static, TextureCraftApp>, Pos2) {
    let mut harness = sample_app();
    harness.state_mut().ui.prefs.language = "en".into();
    let name = import_white_image(&mut harness);

    let position = point_over(&harness, position_over);
    drop_asset(&mut harness, Asset::Texture(name), position);
    // The menu's first frame only measures it
    harness.run_steps(2);
    (harness, position)
}

#[test]
fn an_image_dropped_on_another_set_asks_for_a_channel_then_goes_to_that_set() {
    let (mut harness, position) = drop_image(Some(1));
    let before = stack_sizes(&harness);

    // The menu item nearest the drop point (Properties may show the same name)
    let distance = |node: &egui_kittest::Node<'_>| node.rect().center().distance(position);
    harness.get_all_by_label("Roughness").min_by(|a, b| distance(a).total_cmp(&distance(b))).unwrap().click();
    harness.run_steps(2);

    let app = harness.state();
    assert_eq!(app.session.ts(), 1, "the set under the drop point is selected");
    assert_eq!(stack_sizes(&harness), vec![before[0], before[1] + 1]);
    let layer = app.session.selected_layer().and_then(|id| app.session.doc.project.layer(id).ok()).unwrap();
    assert_eq!(layer.channels.keys().copied().collect::<Vec<_>>(), [Channel::Roughness], "the image fills the picked channel");
}

/// Whether the drop menu is open at `position` (its Mask item near there)
fn menu_open(harness: &Harness<'_, TextureCraftApp>, position: Pos2) -> bool {
    harness.query_all_by_label("Mask").any(|node| node.rect().center().distance(position) < MENU_REACH)
}

/// Press and release the primary button at `position`
fn click(harness: &mut Harness<'_, TextureCraftApp>, position: Pos2) {
    harness.hover_at(position);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(2);
}

#[test]
fn dismissing_an_image_drop_in_the_2d_view_does_not_paint() {
    let (mut harness, position) = drop_image(Some(1));
    harness.state_mut().session.tool.tool = Tool::PolygonFill;
    let view_3d = harness.state().viewport_rect;
    let uv_center = egui::pos2(view_3d.right() + view_3d.width() * 0.5, view_3d.center().y);
    let history = harness.state().session.history_labels();
    assert!(menu_open(&harness, position));

    click(&mut harness, uv_center);

    assert!(!menu_open(&harness, position));
    assert_eq!(harness.state().session.history_labels(), history, "dismissing the menu must not fill the UV view");

    click(&mut harness, uv_center);

    assert_eq!(harness.state().session.history_labels().len(), history.len() + 1, "the next click paints normally");
    assert_eq!(harness.state().session.undo_label(), Some("Polygon fill"));
}

#[test]
fn a_pending_image_drop_does_not_drag_the_selected_decal() {
    let mut harness = sample_app();
    let image = import_white_image(&mut harness);
    let rect = harness.state().viewport_rect;
    let center = rect.center();
    let local = center - rect.min;
    harness.state_mut().run("layer.add.decal", serde_json::json!({"image": image, "x": local.x, "y": local.y})).unwrap();
    harness.run_steps(2);
    let drop_position = point_over(&harness, Some(1));
    drop_asset(&mut harness, Asset::Texture(image), drop_position);
    harness.run_steps(2);
    assert!(menu_open(&harness, drop_position));
    let layer = harness.state().session.selected_layer().unwrap();
    let before = harness.state().session.doc.project.layer(layer).unwrap().kind.clone();
    let history = harness.state().session.history_labels();

    harness.hover_at(center);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: center, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    let moved = center + egui::vec2(30.0, 0.0);
    harness.hover_at(moved);
    harness.run_steps(2);

    assert_eq!(harness.state().session.doc.project.layer(layer).unwrap().kind, before, "a pending drop owns the decal-drag input");
    assert_eq!(harness.state().session.history_labels(), history);
}

#[test]
fn an_image_drop_places_nothing_until_a_pick_and_nothing_on_the_background() {
    let (mut harness, position) = drop_image(Some(1));
    let (stacks, history) = (stack_sizes(&harness), harness.state().session.history_labels().len());
    assert!(menu_open(&harness, position));

    // A click elsewhere on the mesh closes the menu and fills nothing there (a polygon fill click records a step in this harness)
    harness.state_mut().session.tool.tool = Tool::PolygonFill;
    let elsewhere = point_over(&harness, Some(0));
    harness.hover_at(elsewhere);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: elsewhere, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: elsewhere, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(2);
    assert!(!menu_open(&harness, position));
    assert_eq!(stack_sizes(&harness), stacks);
    assert_eq!(harness.state().session.history_labels().len(), history);

    let (harness, position) = drop_image(None);
    assert!(!menu_open(&harness, position), "the background opens no menu");
}

#[test]
fn hiding_the_3d_view_lets_go_of_a_waiting_image_drop() {
    let (mut harness, position) = drop_image(Some(1));
    harness.state_mut().session.tool.tool = Tool::PolygonFill;
    let history = harness.state().session.history_labels().len();
    assert!(menu_open(&harness, position));

    // Only the 3D view shows the menu, so a 2D-only layout would otherwise leave the views blocked
    harness.state_mut().ui.layout = ViewLayout::Only2D;
    harness.run_steps(3);
    click(&mut harness, egui::pos2(700.0, 420.0));

    assert_eq!(harness.state().session.history_labels().len(), history + 1, "the hidden menu no longer blocks UV painting");
    assert_eq!(harness.state().session.undo_label(), Some("Polygon fill"));

    harness.state_mut().ui.layout = ViewLayout::Split;
    harness.run_steps(3);

    assert!(!menu_open(&harness, position), "the drop lapsed while its menu was hidden");
}

#[test]
fn changing_the_camera_cancels_a_drop_with_stale_view_coordinates() {
    let (mut harness, position) = drop_image(Some(1));
    let history = harness.state().session.history_labels();
    let stacks = stack_sizes(&harness);
    assert!(menu_open(&harness, position));

    harness.state_mut().session.camera.distance *= 2.0;
    harness.run_steps(3);

    let app = harness.state();
    let local = position - app.viewport_rect.min;
    assert_ne!(app.session.texture_set_at(Vec2::new(local.x, local.y), app.session.viewport), Some(1), "the saved pixel no longer hits the dropped-on set");
    assert!(!menu_open(&harness, position), "a stale drop must not offer placement on a different surface");
    assert_eq!(harness.state().session.history_labels(), history);
    assert_eq!(stack_sizes(&harness), stacks);
}

#[test]
fn the_click_that_closes_the_drop_menu_does_not_move_a_decal_being_placed() {
    let (mut harness, position) = drop_image(Some(1));
    let distance = |node: &egui_kittest::Node<'_>| node.rect().center().distance(position);
    harness.get_all_by_label("Roughness").min_by(|a, b| distance(a).total_cmp(&distance(b))).unwrap().click();
    harness.run_steps(2);
    let decal_layer = harness.state().session.selected_layer().unwrap();
    harness.state_mut().placing_decal = Some(decal_layer);

    let image = import_white_image(&mut harness);
    drop_asset(&mut harness, Asset::Texture(image), position);
    harness.run_steps(2);
    assert!(menu_open(&harness, position));
    let history = harness.state().session.history_labels().len();

    // A point on the same set, away from the menu
    let rect = harness.state().viewport_rect;
    let elsewhere = (1..(rect.width() / SEARCH_STEP) as usize)
        .map(|step| Pos2::new(rect.max.x - step as f32 * SEARCH_STEP, position.y))
        .find(|point| {
            let app = harness.state();
            let local = *point - rect.min;
            point.distance(position) > MENU_REACH * 2.0 && app.session.texture_set_at(Vec2::new(local.x, local.y), app.session.viewport) == Some(1)
        })
        .unwrap();
    harness.hover_at(elsewhere);
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: elsewhere, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: elsewhere, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(2);

    assert!(!menu_open(&harness, position));
    assert_eq!(harness.state().session.history_labels().len(), history, "the decal wasn't moved");
}

#[test]
#[ignore = "requires an offscreen GPU adapter and TC_UI_SNAPSHOTS"]
fn image_drop_menu_snapshots() {
    let Some(directory) = std::env::var_os("TC_UI_SNAPSHOTS").map(std::path::PathBuf::from) else { return };

    std::fs::create_dir_all(&directory).unwrap();
    let session = texturecraft_engine::sample_session(256, 64).unwrap();
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).wgpu().build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.prefs.language = "en".into();
    harness.run_steps(6);
    let image = import_white_image(&mut harness);
    let position = point_over(&harness, Some(1));
    drop_asset(&mut harness, Asset::Texture(image), position);
    harness.run_steps(6);

    assert!(menu_open(&harness, position));
    harness.render().unwrap().save(directory.join("image-drop-menu.png")).unwrap();

    let distance = |node: &egui_kittest::Node<'_>| node.rect().center().distance(position);
    harness.get_all_by_label("Mask").min_by(|left, right| distance(left).total_cmp(&distance(right))).unwrap().click();
    harness.run_steps(6);

    assert_eq!(harness.state().session.ts(), 1);
    assert!(matches!(harness.state().session.target, texturecraft_engine::StackTarget::Effect(_)));
    harness.render().unwrap().save(directory.join("image-drop-mask.png")).unwrap();
}
