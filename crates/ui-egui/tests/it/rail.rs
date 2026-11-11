//! The rail opens its panels in a popup beside their icons, one at a time; dragging the popup's title docks or floats the panel

use egui::{Event, Modifiers, PointerButton, Pos2, Role};
use egui_kittest::Harness;
use egui_kittest::kittest::{NodeT, Queryable, by};
use texturecraft_ui_egui::TextureCraftApp;
use texturecraft_ui_egui::docking::Panel;
use texturecraft_ui_egui::state::ViewLayout;

const WINDOW_SIZE: egui::Vec2 = egui::vec2(1600.0, 960.0);

/// The rail is the last 38 points of the window; the popup's own buttons sit left of it
const RAIL_LEFT_EDGE: f32 = WINDOW_SIZE.x - 38.0;

// Small texture and bake sizes keep the sample session fast; the rail doesn't depend on them
const TEXTURE_RESOLUTION: u32 = 256;
const BAKE_RESOLUTION: u32 = 64;

/// Frames for the first layout and fonts to settle
const STARTUP_FRAMES: usize = 4;

/// Where a drag out of the popup ends: in the viewport, clear of every panel
const INTO_THE_VIEWPORT: egui::Vec2 = egui::vec2(-500.0, 100.0);

/// Moves a drag takes, so the dock sees it pass its threshold and move
const DRAG_STEPS: usize = 6;

/// Frames between a drag's events: the press must land before the moves, and each move be seen
const FRAMES_PER_DRAG_EVENT: usize = 2;

/// Frames for a click to reach the rail, the popup and the dock
const SETTLE_FRAMES: usize = 3;

// A setting only Shader Settings shows, and one only Display Settings shows
const SHADER_SETTING: &str = "Index of refraction";
const DISPLAY_SETTING: &str = "Environment opacity";

fn app() -> Harness<'static, TextureCraftApp> {
    app_of_size(WINDOW_SIZE)
}

fn app_of_size(size: egui::Vec2) -> Harness<'static, TextureCraftApp> {
    let session = texturecraft_engine::sample_session(TEXTURE_RESOLUTION, BAKE_RESOLUTION).unwrap();
    let mut harness = Harness::builder().with_size(size).build_eframe(|cc| TextureCraftApp::new(cc, session));

    // The labels below are English; a first run would speak the system's language
    harness.state_mut().ui.prefs.language = "en".into();
    harness.run_steps(STARTUP_FRAMES);

    harness
}

/// The rail icon of the panel named `name`, if the rail shows one
fn rail_icon<'h>(harness: &'h Harness<'_, TextureCraftApp>, name: &str) -> Option<egui_kittest::Node<'h>> {
    // A rail icon's label is its tooltip, which adds the shortcut when the panel has one
    let is_panel = |label: &str| label == name || label.starts_with(&format!("{name} ("));
    harness
        .get_all_by_role(Role::Button)
        .find(|button| button.rect().min.x >= RAIL_LEFT_EDGE && button.accesskit_node().label().is_some_and(|label| is_panel(&label)))
}

fn click_rail(harness: &mut Harness<'_, TextureCraftApp>, name: &str) {
    rail_icon(harness, name).expect("the panel's icon on the rail").click();
    harness.run_steps(SETTLE_FRAMES);
}

fn click_at(harness: &mut Harness<'_, TextureCraftApp>, position: Pos2) {
    harness.hover_at(position);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(1);
    harness.event(Event::PointerButton { pos: position, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(SETTLE_FRAMES);
}

/// Drag with the primary button from `from` to `to` in a few moves
fn drag(harness: &mut Harness<'_, TextureCraftApp>, from: Pos2, to: Pos2) {
    harness.hover_at(from);
    harness.run_steps(FRAMES_PER_DRAG_EVENT);
    harness.event(Event::PointerButton { pos: from, button: PointerButton::Primary, pressed: true, modifiers: Modifiers::NONE });
    harness.run_steps(FRAMES_PER_DRAG_EVENT);

    for step in 1..=DRAG_STEPS {
        harness.hover_at(from + (to - from) * (step as f32 / DRAG_STEPS as f32));
        harness.run_steps(FRAMES_PER_DRAG_EVENT);
    }

    harness.event(Event::PointerButton { pos: to, button: PointerButton::Primary, pressed: false, modifiers: Modifiers::NONE });
    harness.run_steps(SETTLE_FRAMES);
}

/// The popup's Close button, left of the rail (the window's own Close sits above the rail)
fn popup_close_button(harness: &Harness<'_, TextureCraftApp>) -> Option<egui::Rect> {
    harness
        .query_all(by().role(Role::Button))
        .find(|button| button.rect().max.x < RAIL_LEFT_EDGE && button.accesskit_node().label().as_deref() == Some("Close"))
        .map(|button| button.rect())
}

/// The title of the popup open on the panel named `name`: that name, level with the popup's Close button
fn open_popup_title(harness: &Harness<'_, TextureCraftApp>, name: &str) -> Option<egui::Rect> {
    let close = popup_close_button(harness)?;
    harness
        .get_all_by_label(name)
        .map(|node| node.rect())
        .find(|title| title.right() < close.left() && (close.top()..=close.bottom()).contains(&title.center().y))
}

fn popup_title(harness: &Harness<'_, TextureCraftApp>, name: &str) -> egui::Rect {
    open_popup_title(harness, name).expect("the popup's title")
}

fn popup_shows(harness: &Harness<'_, TextureCraftApp>, name: &str) -> bool {
    open_popup_title(harness, name).is_some()
}

fn popup_closed(harness: &Harness<'_, TextureCraftApp>) -> bool {
    popup_close_button(harness).is_none()
}

#[test]
fn the_rail_shows_one_panel_at_a_time() {
    let mut harness = app();

    click_rail(&mut harness, "Shader Settings");

    assert!(popup_shows(&harness, "Shader Settings"));
    assert!(harness.query_by_label(SHADER_SETTING).is_some());

    click_rail(&mut harness, "Display Settings");

    assert!(popup_shows(&harness, "Display Settings"));
    assert!(harness.query_by_label(SHADER_SETTING).is_none(), "opening Display Settings closed Shader Settings");
    assert!(harness.query_by_label(DISPLAY_SETTING).is_some());
    assert!(!harness.state().ui.dock.is_open(Panel::Shader), "the rail's panels are not dock windows");

    click_rail(&mut harness, "Display Settings");

    assert!(popup_closed(&harness), "its own icon closes it");
}

#[test]
fn the_popup_lines_up_with_its_icon() {
    let mut harness = app();

    click_rail(&mut harness, "History");
    let icon = rail_icon(&harness, "History").expect("the History icon").rect();
    let title = popup_title(&harness, "History");

    assert!(title.top() >= icon.top() && title.top() <= icon.bottom(), "the title {title:?} is not level with the icon {icon:?}");
}

#[test]
fn the_close_button_a_press_outside_and_escape_close_it() {
    let mut harness = app();

    click_rail(&mut harness, "Display Settings");
    harness.get_by_role_and_label(Role::Button, "Close").click();
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_closed(&harness), "Close");

    click_rail(&mut harness, "Log");
    harness.get_by_label("Assets").click();
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_closed(&harness), "a press outside");

    click_rail(&mut harness, "Log");
    harness.key_press(egui::Key::Escape);
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_closed(&harness), "Escape");
}

#[test]
fn dragging_the_title_out_floats_the_panel_and_frees_the_rail() {
    let mut harness = app();

    click_rail(&mut harness, "History");
    let title = popup_title(&harness, "History").center();
    drag(&mut harness, title, title + INTO_THE_VIEWPORT);

    assert!(popup_closed(&harness));
    assert!(harness.state().ui.dock.is_visible(Panel::History), "History floats on its own");
    assert!(rail_icon(&harness, "History").is_none(), "a panel in the dock has no rail icon");

    click_rail(&mut harness, "Shader Settings");

    assert!(popup_shows(&harness, "Shader Settings"));
    assert!(harness.state().ui.dock.is_visible(Panel::History), "History stays open beside the rail's panel");
}

#[test]
fn dragging_the_title_onto_a_tab_strip_docks_the_panel_there() {
    let mut harness = app();

    click_rail(&mut harness, "Log");
    let title = popup_title(&harness, "Log").center();
    let properties_tab = harness.get_by_label("Properties - Paint").rect().center();
    drag(&mut harness, title, properties_tab);

    let layout = &harness.state().ui.dock.workspace().layout;
    assert!(layout.contains(&Panel::Log) && !layout.floating.iter().any(|window| window.contains(&Panel::Log)), "the Log is docked");
}

#[test]
fn a_panel_closed_in_the_dock_returns_to_the_rail() {
    let mut harness = app();
    assert!(rail_icon(&harness, "Texture Set Settings").is_none(), "Texture Set Settings starts in the dock, beside Layers");

    harness.state_mut().ui.dock.close(Panel::TextureSetSettings);
    harness.run_steps(SETTLE_FRAMES);
    click_rail(&mut harness, "Texture Set Settings");

    assert!(popup_shows(&harness, "Texture Set Settings"));
}

#[test]
fn window_views_open_the_rail_panels_popup() {
    let mut harness = app();

    let ctx = harness.ctx.clone();
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &ctx, "ui.view.shader");
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_shows(&harness, "Shader Settings"));
    assert!(!harness.state().ui.dock.is_open(Panel::Shader));

    // Project configuration opens Display Settings, in its popup too
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &ctx, "ui.projectConfig");
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_shows(&harness, "Display Settings"));
}

#[test]
fn a_popup_that_would_run_past_the_bottom_moves_up() {
    // Too short for Display Settings below its icon
    const SHORT_WINDOW: egui::Vec2 = egui::vec2(1600.0, 600.0);
    let mut harness = app_of_size(SHORT_WINDOW);

    click_rail(&mut harness, "Display Settings");
    let icon = rail_icon(&harness, "Display Settings").expect("the Display Settings icon").rect();
    let title = popup_title(&harness, "Display Settings");

    assert!(title.top() < icon.top(), "the popup at {title:?} did not move up from the icon at {icon:?}");
}

#[test]
fn display_settings_remain_reachable_in_a_short_window() {
    let mut harness = app_of_size(egui::vec2(1600.0, 600.0));
    click_rail(&mut harness, "Display Settings");

    let last_setting = "Temporal anti-aliasing (refines the 3D view while it is still)";
    harness.get_by_label(last_setting).scroll_to_me();
    harness.run_steps(SETTLE_FRAMES);

    let setting = harness.get_by_label(last_setting).rect();
    assert!(harness.ctx.content_rect().contains_rect(setting), "the last setting must be reachable: {setting:?}");
}

#[test]
fn history_actions_remain_reachable_in_a_short_window() {
    let mut harness = app_of_size(egui::vec2(1600.0, 400.0));
    click_rail(&mut harness, "History");

    for label in ["Undo", "Redo"] {
        let button = harness
            .get_all_by_role(Role::Button)
            .find(|button| button.accesskit_node().label().is_some_and(|text| text.ends_with(label)))
            .expect("the History action")
            .rect();
        assert!(harness.ctx.content_rect().contains_rect(button), "{label} must be reachable: {button:?}");
    }
}

#[test]
fn opening_the_window_menu_closes_the_rail_popup() {
    let mut harness = app();
    click_rail(&mut harness, "Shader Settings");

    harness.get_by_role_and_label(Role::Button, "Window").click();
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_closed(&harness));
}

#[test]
fn escape_closes_a_dropdown_before_its_rail_popup() {
    let mut harness = app();
    click_rail(&mut harness, "Display Settings");
    let environment_row = harness.get_by_label("Environment map").rect();
    harness
        .get_all_by_role(Role::ComboBox)
        .find(|combo| (combo.rect().center().y - environment_row.center().y).abs() < 1.0)
        .expect("the environment dropdown")
        .click();
    harness.run_steps(SETTLE_FRAMES);

    assert!(egui::Popup::is_any_open(&harness.ctx));

    harness.key_press(egui::Key::Escape);
    harness.run_steps(SETTLE_FRAMES);

    assert!(!egui::Popup::is_any_open(&harness.ctx));
    assert!(popup_shows(&harness, "Display Settings"));

    harness.key_press(egui::Key::Escape);
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_closed(&harness));
}

#[test]
fn dismissing_a_rail_popup_does_not_paint_the_viewport() {
    for layout in [ViewLayout::Only3D, ViewLayout::Only2D] {
        let mut harness = app();
        texturecraft_engine::execute(&mut harness.state_mut().session, "layer.add.paint", &serde_json::Value::Null).unwrap();
        harness.state_mut().ui.layout = layout;
        harness.run_steps(SETTLE_FRAMES);
        click_rail(&mut harness, "Log");
        let center = if layout == ViewLayout::Only2D { egui::pos2(700.0, 420.0) } else { harness.state().viewport_rect.center() };
        let history = harness.state().session.history_labels().len();

        click_at(&mut harness, center);

        assert!(popup_closed(&harness));
        assert_eq!(harness.state().session.history_labels().len(), history, "dismissing the popup must not paint in {layout:?}");

        click_at(&mut harness, center);

        assert_eq!(harness.state().session.history_labels().len(), history + 1, "the next click paints in {layout:?}");

        click_rail(&mut harness, "Log");
        drag(&mut harness, center, center + egui::vec2(40.0, 0.0));

        assert!(popup_closed(&harness));
        assert!(!harness.state().session.stroke_active());
        assert_eq!(harness.state().session.history_labels().len(), history + 1, "a dismissing drag must not paint in {layout:?}");
    }
}

#[test]
fn opening_a_rail_popup_during_a_stroke_still_finishes_the_stroke() {
    let mut harness = app();
    texturecraft_engine::execute(&mut harness.state_mut().session, "layer.add.paint", &serde_json::Value::Null).unwrap();
    harness.state_mut().ui.layout = ViewLayout::Only3D;
    harness.run_steps(SETTLE_FRAMES);
    let center = harness.state().viewport_rect.center();
    let history = harness.state().session.history_labels().len();
    harness.drag_at(center);
    harness.run_steps(FRAMES_PER_DRAG_EVENT);
    harness.hover_at(center + egui::vec2(20.0, 0.0));
    harness.run_steps(FRAMES_PER_DRAG_EVENT);
    assert!(harness.state().session.stroke_active());

    let context = harness.ctx.clone();
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &context, "ui.view.log");
    harness.run_steps(SETTLE_FRAMES);
    harness.drop_at(center + egui::vec2(30.0, 0.0));
    harness.run_steps(SETTLE_FRAMES);

    assert!(popup_shows(&harness, "Log"));
    assert!(!harness.state().session.stroke_active());
    assert_eq!(harness.state().session.history_labels().len(), history + 1);
}

#[test]
fn escape_cancels_a_stroke_with_a_hidden_rail_popup() {
    let mut harness = app();
    texturecraft_engine::execute(&mut harness.state_mut().session, "layer.add.paint", &serde_json::Value::Null).unwrap();
    harness.state_mut().ui.layout = ViewLayout::Only3D;
    harness.run_steps(SETTLE_FRAMES);
    click_rail(&mut harness, "Log");
    harness.key_press(egui::Key::Tab);
    harness.run_steps(SETTLE_FRAMES);
    assert!(harness.state().ui.hide_ui);
    assert!(popup_closed(&harness));
    let center = harness.state().viewport_rect.center();
    let history = harness.state().session.history_labels();
    harness.drag_at(center);
    harness.run_steps(FRAMES_PER_DRAG_EVENT);
    harness.hover_at(center + egui::vec2(20.0, 0.0));
    harness.run_steps(FRAMES_PER_DRAG_EVENT);
    assert!(harness.state().session.stroke_active());

    harness.key_press(egui::Key::Escape);
    harness.run_steps(SETTLE_FRAMES);

    assert!(!harness.state().session.stroke_active(), "one Escape cancels the stroke despite the hidden popup");
    harness.drop_at(center + egui::vec2(30.0, 0.0));
    harness.run_steps(SETTLE_FRAMES);
    assert_eq!(harness.state().session.history_labels(), history, "the canceled stroke must not be committed on release");
}

#[test]
fn escape_cancels_a_path_with_a_hidden_rail_popup() {
    for stored_path in [false, true] {
        let mut harness = app();
        texturecraft_engine::execute(&mut harness.state_mut().session, "layer.add.paint", &serde_json::Value::Null).unwrap();
        let path = texturecraft_engine::execute(
            &mut harness.state_mut().session,
            "tool.path",
            &serde_json::json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": stored_path}),
        )
        .unwrap();
        if stored_path {
            texturecraft_engine::execute(
                &mut harness.state_mut().session,
                "path.edit",
                &serde_json::json!({"effect": path["committed"]["effect"], "index": 0}),
            )
            .unwrap();
        }
        harness.run_steps(SETTLE_FRAMES);
        click_rail(&mut harness, "Log");
        harness.key_press(egui::Key::Tab);
        harness.run_steps(SETTLE_FRAMES);
        assert!(harness.state().ui.hide_ui);
        assert!(popup_closed(&harness));
        assert_eq!(harness.state().session.tool.path.points.len(), 2);

        // Drafts and committed paths both have a live preview target while edited
        let (effect, index) = harness.state().session.tool.path.editing.unwrap();
        let history = harness.state().session.history_labels();

        harness.key_press(egui::Key::Escape);
        harness.run_steps(SETTLE_FRAMES);

        assert!(harness.state().session.tool.path.points.is_empty(), "one Escape clears the path despite the hidden popup");
        assert!(harness.state().session.tool.path.editing.is_none());
        assert_eq!(harness.state().session.history_labels(), history);
        assert_eq!(
            harness.state().session.paths(effect).and_then(|paths| paths.get(index)).is_some(),
            stored_path,
            "Escape removes a draft but keeps a committed path"
        );

        harness.key_press(egui::Key::Tab);
        harness.run_steps(SETTLE_FRAMES);

        assert!(popup_shows(&harness, "Log"), "showing the UI restores its remembered popup");
    }
}
