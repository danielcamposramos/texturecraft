//! Offscreen UI snapshots at 1600x960, except the taller mesh-map panel
//! Requires a GPU adapter
//!
//! Run with TC_UI_SNAPSHOTS=<dir> cargo test -p texturecraft-ui-egui --release --test ui_snapshots -- --ignored

use std::path::{Path, PathBuf};

use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use texturecraft_engine::core::{BakeResolution, BakedMap, CurvatureMethod, HeightNormalization};
use texturecraft_ui_egui::TextureCraftApp;
use texturecraft_ui_egui::docking::Panel;
use texturecraft_ui_egui::state::Mode;
use texturecraft_ui_egui::theme::Brightness;
use texturecraft_ui_egui::titlebar::TitleBar;

/// Click a map's row in Mesh Map Bakers, which shows its settings
fn select_map(harness: &mut Harness<'_, TextureCraftApp>, name: &str) {
    harness.get_by_role_and_label(egui::Role::Button, name).click();
    harness.run_steps(3);
}

fn save(h: &mut Harness<'_, TextureCraftApp>, dir: &Path, name: &str) {
    h.run_steps(6);
    let img = h.render().unwrap_or_else(|e| panic!("{name}: {e}"));
    let path = dir.join(format!("{name}.png"));
    img.save(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    eprintln!("wrote {}", path.display());
}

/// Capture the moving preview and settled render after submitting real GPU callbacks.
#[test]
#[ignore]
fn rendering_mode_snapshots() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else { return };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session = texturecraft_engine::sample_session(256, 64).unwrap();
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1280.0, 800.0))
        .with_pixels_per_point(1.0)
        .wgpu()
        .build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.mode = Mode::Rendering;
    harness.state_mut().ui.render.max_samples = 16;
    eprintln!("Rendering-mode adapter: {}", harness.state().gpu.as_ref().unwrap().adapter);

    // Submit each frame so the snapshots exercise the tracer callback, including size switches
    for _ in 0..6 {
        harness.state_mut().session.camera.yaw += 0.08;
        harness.run_steps(1);
        harness.render().unwrap();
    }
    let moving_path = output_dir.join("rendering-moving.png");
    harness.render().unwrap().save(&moving_path).unwrap();
    eprintln!("wrote {}", moving_path.display());

    for _ in 0..45 {
        harness.run_steps(1);
        harness.render().unwrap();
    }
    save(&mut harness, &output_dir, "rendering-settled");
}

#[test]
#[ignore]
fn symmetry_snapshots() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else { return };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session = texturecraft_engine::sample_session(1024, 256).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.state_mut().ui.prefs.language = "en".into();
    harness.state_mut().run("tool.brush", serde_json::json!({"symmetryY": true, "symmetryOffset": [0, 0.1, 0]})).unwrap();
    harness.state_mut().ui.symmetry.show_plane = true;
    harness.state_mut().ui.symmetry.show_manipulator = true;
    harness.run_steps(4);
    harness.get_by_label("Symmetry settings").click();

    for (brightness, name) in [(Brightness::MediumDark, "medium"), (Brightness::Dark, "dark"), (Brightness::Light, "light")] {
        harness.state_mut().ui.prefs.brightness = brightness;
        save(&mut harness, &output_dir, &format!("symmetry-mirror-{name}"));
    }

    harness.state_mut().run("tool.brush", serde_json::json!({"symmetryY": false, "symmetryRadial": 6, "symmetryAxis": "y", "symmetryAngleSpan": 240})).unwrap();
    harness.state_mut().ui.symmetry.show_axis = true;
    harness.state_mut().ui.prefs.brightness = Brightness::MediumDark;
    save(&mut harness, &output_dir, "symmetry-radial");

    harness.state_mut().run("tool.brush", serde_json::json!({"symmetryX": true, "symmetryY": true})).unwrap();
    for (brightness, name) in [(Brightness::MediumDark, "medium"), (Brightness::Dark, "dark"), (Brightness::Light, "light")] {
        harness.state_mut().ui.prefs.brightness = brightness;
        save(&mut harness, &output_dir, &format!("symmetry-combined-{name}"));
    }
}

#[test]
#[ignore]
fn window_snapshots() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else {
        eprintln!("set TC_UI_SNAPSHOTS=<dir> to write the pictures");
        return;
    };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session = texturecraft_engine::sample_session(1024, 256).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    for (brightness, name) in [(Brightness::MediumDark, "main"), (Brightness::Dark, "brightness-dark"), (Brightness::Light, "brightness-light")] {
        harness.state_mut().ui.prefs.brightness = brightness;
        save(&mut harness, &output_dir, name);
    }
    harness.state_mut().ui.prefs.brightness = Brightness::MediumDark;
    harness.state_mut().title_bar = TitleBar::Integrated;
    save(&mut harness, &output_dir, "titlebar-macos");
    harness.state_mut().title_bar = TitleBar::App(craft_window::Layout::default());
    save(&mut harness, &output_dir, "titlebar-app");

    // Check hover feedback and fractional scaling
    harness.hover_at(egui::pos2(1600.0 - 23.0, 14.0));
    save(&mut harness, &output_dir, "titlebar-app-close-hover");
    harness.hover_at(egui::pos2(800.0, 500.0));
    harness.set_pixels_per_point(1.5);
    save(&mut harness, &output_dir, "titlebar-app-150");
    harness.set_pixels_per_point(1.0);
    harness.state_mut().title_bar = TitleBar::System;

    for texture_set in &mut harness.state_mut().session.doc.project.texture_sets {
        texture_set.bake = Default::default();
    }
    harness.state_mut().ui.mode = Mode::Baking;
    save(&mut harness, &output_dir, "mode-bake");
    select_map(&mut harness, "Ambient Occlusion");
    save(&mut harness, &output_dir, "mode-bake-maps");
    harness.state_mut().ui.prefs.brightness = Brightness::Light;
    save(&mut harness, &output_dir, "mode-bake-maps-light");
    harness.state_mut().ui.prefs.brightness = Brightness::MediumDark;
    select_map(&mut harness, "Opacity");
    save(&mut harness, &output_dir, "mode-bake-unchecked");
    harness.state_mut().ui.dock.focus(Panel::TextureSetList);
    save(&mut harness, &output_dir, "bake-texture-sets");
    harness.state_mut().ui.dock.focus(Panel::CommonSettings);
    save(&mut harness, &output_dir, "bake-settings");
    harness.state_mut().session.add_high_poly_bytes("High Poly.obj", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n".to_vec()).unwrap();
    save(&mut harness, &output_dir, "bake-settings-high-poly");
    harness.get_by_role_and_label(egui::Role::ComboBox, "Cage Type").scroll_to_me();
    save(&mut harness, &output_dir, "bake-settings-cage-skew");
    harness.get_by_label("Edge Contrast").scroll_to_me();
    save(&mut harness, &output_dir, "bake-settings-skew");
    harness.get_by_label("Output Size").scroll_to_me();
    harness.state_mut().session.doc.project.texture_sets[0].bake.common.resolution_height = Some(BakeResolution::R1024);
    save(&mut harness, &output_dir, "bake-settings-unlocked");
    harness.set_pixels_per_point(1.5);
    save(&mut harness, &output_dir, "bake-settings-150");
    harness.set_pixels_per_point(1.0);
    {
        let settings = &mut harness.state_mut().session.doc.project.texture_sets[0].bake;
        settings.maps.select([BakedMap::Curvature, BakedMap::Height]);
        settings.maps.curvature.parameters.method = CurvatureMethod::Mesh;
        settings.maps.height.parameters.normalization = HeightNormalization::Manual;
        settings.maps.height.parameters.scaling_divisor = 100.0;
    }
    select_map(&mut harness, "Curvature");
    save(&mut harness, &output_dir, "bake-curvature-mesh");
    select_map(&mut harness, "Height");
    save(&mut harness, &output_dir, "bake-height-manual");
    harness.state_mut().session.bake_meshes.high.clear();
    harness.state_mut().ui.mode = Mode::Painting;
    save(&mut harness, &output_dir, "painting-restored");

    harness.state_mut().ui.dock.focus(Panel::Display);
    save(&mut harness, &output_dir, "panel-display");
    harness.state_mut().ui.dock.close(Panel::Display);
    harness.state_mut().ui.windows.export = true;
    save(&mut harness, &output_dir, "window-export");
    harness.state_mut().ui.windows.export = false;
    harness.state_mut().ui.windows.settings = true;
    save(&mut harness, &output_dir, "window-settings");
    harness.state_mut().ui.windows.settings = false;

    // A taller window exposes every mesh-map slot
    harness.state_mut().ui.dock.focus(Panel::TextureSetSettings);
    harness.set_size(egui::vec2(1600.0, 1800.0));
    save(&mut harness, &output_dir, "panel-texture-set-settings");
    harness.set_size(egui::vec2(1600.0, 960.0));

    // Show the mask histogram in Properties
    let session = &mut harness.state_mut().session;
    for (id, params) in [
        ("layer.add.fill", serde_json::json!({"color": [0.8, 0.3, 0.1]})),
        ("mask.add.black", serde_json::Value::Null),
        ("effect.add.fill", serde_json::json!({"target": "mask", "procedural": "clouds"})),
        ("effect.add.levels", serde_json::json!({"target": "mask", "inLow": 0.3, "inHigh": 0.8})),
    ] {
        texturecraft_engine::execute(session, id, &params).unwrap();
    }
    save(&mut harness, &output_dir, "properties-levels");
    // Show the Layers panel thumbnails with a hidden content effect, a folder and a mask
    let session = &mut harness.state_mut().session;
    for (id, params) in [
        ("effect.add.filter", serde_json::json!({"target": "content", "filter": "hsl"})),
        ("effect.add.filter", serde_json::json!({"target": "content", "filter": "invert"})),
        ("layer.add.paint", serde_json::Value::Null),
        ("layer.group", serde_json::Value::Null),
    ] {
        texturecraft_engine::execute(session, id, &params).unwrap();
    }
    let hidden = harness
        .state()
        .session
        .doc
        .project
        .layers
        .values()
        .flat_map(|layer| layer.content_effects.iter())
        .find(|effect| effect.name == "Invert")
        .map(|effect| effect.id.0);
    if let Some(effect) = hidden {
        texturecraft_engine::execute(&mut harness.state_mut().session, "effect.set", &serde_json::json!({"effect": effect, "visible": false})).unwrap();
    }
    harness.state_mut().ui.dock.focus(Panel::Layers);
    harness.set_size(egui::vec2(1600.0, 1400.0));
    for _ in 0..400 {
        harness.run_steps(1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        if !harness.state().layer_panel.service().busy() {
            break;
        }
    }
    save(&mut harness, &output_dir, "panel-layers");
    harness.set_size(egui::vec2(1600.0, 960.0));
    // Show the second path selected in Properties
    let session = &mut harness.state_mut().session;
    for (id, params) in [
        ("layer.add.paint", serde_json::Value::Null),
        ("tool.path", serde_json::json!({"uvPoints": [[0.2, 0.5], [0.5, 0.4], [0.8, 0.5]], "commit": true})),
        ("tool.path", serde_json::json!({"uvPoints": [[0.3, 0.7], [0.7, 0.7], [0.7, 0.9]], "closed": true, "fill": true, "commit": true})),
    ] {
        texturecraft_engine::execute(session, id, &params).unwrap();
    }

    let effect = match session.target {
        texturecraft_engine::StackTarget::Effect(e) => e.0,
        _ => 0,
    };
    texturecraft_engine::execute(session, "path.edit", &serde_json::json!({"effect": effect, "index": 1})).unwrap();
    save(&mut harness, &output_dir, "properties-paths");

    // Show the brush cursor and readout during Ctrl+right drag
    let cursor_position = harness.state().viewport_rect.center();
    let (modifiers, button) = (egui::Modifiers::COMMAND, egui::PointerButton::Secondary);
    harness.event(egui::Event::ModifiersChanged(modifiers));
    harness.event(egui::Event::PointerButton { pos: cursor_position, button, pressed: true, modifiers });
    for step in 1..=4 {
        harness.hover_at(cursor_position + egui::vec2(15.0 * step as f32, -8.0 * step as f32));
        harness.run_steps(1);
    }
    save(&mut harness, &output_dir, "brush-ctrl-drag");
    harness.event(egui::Event::PointerButton { pos: cursor_position, button, pressed: false, modifiers });
    harness.event(egui::Event::ModifiersChanged(egui::Modifiers::NONE));
}

#[test]
#[ignore]
fn bake_uv_tiles_snapshot() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else {
        eprintln!("set TC_UI_SNAPSHOTS=<dir> to write the pictures");
        return;
    };

    std::fs::create_dir_all(&output_dir).unwrap();
    let mut session =
        texturecraft_engine::Session::new_project("UV tile baking", texturecraft_engine::geom::primitives::udim_plane(), "builtin:udim_plane", None, 256)
            .unwrap();
    texturecraft_engine::execute(&mut session, "textureSet.tile.resolution", &serde_json::json!({"tile": 1002, "resolution": 128})).unwrap();
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1600.0, 960.0))
        .with_pixels_per_point(1.0)
        .wgpu()
        .build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.mode = Mode::Baking;
    save(&mut harness, &output_dir, "bake-uv-tiles");

    harness.get_by_value("128 × 128*").click();
    save(&mut harness, &output_dir, "bake-tile-resolution-menu");
    harness.get_by_label("Remove resolution override").click();
    save(&mut harness, &output_dir, "bake-tile-resolution-inherited");
    assert!(harness.query_by_value("128 × 128*").is_none());
    assert!(harness.state().session.doc.project.texture_sets[0].tile_resolutions.is_empty());
    let inherited_resolution = harness.get_by_label("UV tile 1002 resolution").rect().center();
    harness.hover_at(inherited_resolution);
    save(&mut harness, &output_dir, "bake-tile-resolution-hover");
    assert_eq!(harness.get_all_by_value("256 × 256").count(), 2);
    harness.hover_at(egui::pos2(900.0, 500.0));
    harness.state_mut().session.undo().unwrap();
    harness.run_steps(3);

    harness.get_by_role_and_label(egui::Role::CheckBox, "1002").click();
    save(&mut harness, &output_dir, "bake-uv-tiles-partial");
    assert!(harness.state().session.doc.project.texture_sets[0].bake_excluded_tiles.contains(&1002));

    harness.get_by_label("Show or hide the texture set").click();
    save(&mut harness, &output_dir, "bake-uv-tiles-hidden");
    assert!(!harness.state().session.doc.project.texture_sets[0].visible);
    assert!(harness.state().session.doc.project.texture_sets[0].bake.common.is_enabled);
}

#[test]
#[ignore]
fn smart_mask_shelf_snapshot() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else {
        eprintln!("set TC_UI_SNAPSHOTS=<dir> to write the picture");
        return;
    };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session =
        texturecraft_engine::Session::new_project("Mask previews", texturecraft_engine::geom::primitives::cube(2.0), "builtin:cube", None, 128).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.state_mut().ui.shelf_tab = 2;
    let context = harness.ctx.clone();
    let started = std::time::Instant::now();
    loop {
        harness.run_steps(1);
        let thumbnails = &mut harness.state_mut().thumbs;
        if texturecraft_engine::smart_masks::SMART_MASKS.iter().all(|preset| thumbnails.smart_mask(&context, preset.id).is_some()) {
            break;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "smart mask thumbnails did not arrive");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    eprintln!("all smart mask thumbnails ready after {:.3} s", started.elapsed().as_secs_f64());
    save(&mut harness, &output_dir, "smart-mask-shelf");

    let mask = &texturecraft_engine::smart_masks::SMART_MASKS[0];
    harness.get_by_role_and_label(egui::Role::Button, mask.label).hover();
    let preview = texturecraft_ui_egui::shelf_preview::ShelfPreview::SmartMask(mask.id);
    let started = std::time::Instant::now();
    loop {
        harness.run_steps(1);
        if harness.state_mut().thumbs.rendered_big(&context, &preview, 256).is_some() {
            break;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "smart mask hover preview did not arrive");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    harness.run_steps(60);
    assert_eq!(harness.query_all_by_label(mask.label).count(), 2, "the popup shows the mask name");
    save(&mut harness, &output_dir, "smart-mask-hover");
}

#[test]
#[ignore]
fn smart_material_shelf_snapshot() {
    use texturecraft_engine::materials::MATERIALS;
    use texturecraft_ui_egui::shelf_preview::ShelfPreview;

    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else {
        eprintln!("set TC_UI_SNAPSHOTS=<dir> to write the pictures");
        return;
    };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session =
        texturecraft_engine::Session::new_project("Smart material previews", texturecraft_engine::geom::primitives::cube(2.0), "builtin:cube", None, 128)
            .unwrap();
    let mut harness = Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|context| {
        let render_state = context.wgpu_render_state.as_ref().expect("the snapshot uses wgpu");
        eprintln!("snapshot adapter: {:?}", render_state.adapter.get_info());
        TextureCraftApp::new(context, session)
    });
    harness.state_mut().ui.shelf_tab = 1;
    let context = harness.ctx.clone();
    let started = std::time::Instant::now();
    loop {
        harness.run_steps(1);
        let thumbnails = &mut harness.state_mut().thumbs;
        if MATERIALS.iter().filter(|material| material.smart).all(|material| thumbnails.material(&context, material.id).is_some()) {
            break;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(120), "smart material thumbnails did not arrive");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    eprintln!("all smart material thumbnails ready after {:.3} s", started.elapsed().as_secs_f64());
    save(&mut harness, &output_dir, "smart-material-shelf");

    let material = MATERIALS.iter().find(|material| material.id == "steel_painted_worn").unwrap();
    harness.get_by_role_and_label(egui::Role::Button, material.label).hover();
    let preview = ShelfPreview::material(material.id);
    let started = std::time::Instant::now();
    loop {
        harness.run_steps(1);
        if harness.state_mut().thumbs.rendered_big(&context, &preview, 256).is_some() {
            break;
        }
        assert!(started.elapsed() < std::time::Duration::from_secs(60), "smart material hover preview did not arrive");
        std::thread::sleep(std::time::Duration::from_millis(20));
    }
    harness.run_steps(60);
    assert_eq!(harness.query_all_by_label(material.label).count(), 2, "the popup shows the material name");
    save(&mut harness, &output_dir, "smart-material-hover");
}

/// Compact nested rows must keep their content, mask and effect controls in the panel.
#[test]
#[ignore]
fn nested_layers_snapshot() {
    use craft_ui::docking::Node;
    use craft_ui::layout::SplitSize;
    use serde_json::{Value, json};
    use texturecraft_engine::execute;
    use texturecraft_ui_egui::docking::Dock;

    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else { return };
    std::fs::create_dir_all(&output_dir).unwrap();
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    execute(&mut session, "layer.add.fill", &json!({"color": "#336699"})).unwrap();
    execute(&mut session, "mask.add.white", &Value::Null).unwrap();
    execute(&mut session, "effect.add.filter", &json!({"target": "content", "filter": "invert"})).unwrap();
    for _ in 0..6 {
        execute(&mut session, "layer.group", &Value::Null).unwrap();
    }
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 1600.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.run_steps(4);
    let mut workspace = harness.state().ui.dock.workspace().clone();
    let Some(Node::Split { second, .. }) = workspace.layout.root.as_mut() else { panic!("default columns") };
    let Node::Split { size, .. } = second.as_mut() else { panic!("viewport and right column") };
    *size = SplitSize::Ratio(0.99);
    harness.state_mut().ui.dock = Dock::from_value(serde_json::to_value(workspace).unwrap());
    for _ in 0..100 {
        harness.run_steps(1);
        std::thread::sleep(std::time::Duration::from_millis(10));
        if !harness.state().layer_panel.service().busy() {
            break;
        }
    }
    save(&mut harness, &output_dir, "panel-layers-narrow-nested");
}

#[test]
#[ignore]
fn path_tool_snapshots() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else { return };

    std::fs::create_dir_all(&output_dir).unwrap();
    let mut session = texturecraft_engine::sample_session(256, 64).unwrap();
    texturecraft_engine::execute(&mut session, "layer.add.paint", &serde_json::Value::Null).unwrap();
    texturecraft_engine::execute(&mut session, "tool.path", &serde_json::json!({"uvPoints": [[0.25, 0.5], [0.5, 0.3], [0.75, 0.5]]})).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.state_mut().ui.layout = texturecraft_ui_egui::state::ViewLayout::Only2D;
    harness.state_mut().session.select_path_point(1);
    harness.hover_at(egui::pos2(800.0, 600.0));
    save(&mut harness, &output_dir, "path-preview-2d");

    harness.state_mut().session.commit_path(texturecraft_engine::geom::glam::Vec2::splat(400.0)).unwrap();
    save(&mut harness, &output_dir, "path-stored-2d");
}

#[test]
#[ignore]
fn rail_window_snapshots() {
    let Some(output_dir) = std::env::var_os("TC_UI_SNAPSHOTS").map(PathBuf::from) else {
        eprintln!("set TC_UI_SNAPSHOTS=<dir> to write the pictures");
        return;
    };

    std::fs::create_dir_all(&output_dir).unwrap();
    let session = texturecraft_engine::sample_session(256, 64).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1600.0, 960.0)).with_pixels_per_point(1.0).wgpu().build_eframe(|cc| TextureCraftApp::new(cc, session));
    harness.state_mut().ui.prefs.language = "en".into();
    harness.state_mut().ui.dock.close(Panel::TextureSetSettings);
    let context = harness.ctx.clone();

    for (command, name) in [
        ("ui.view.textureSetSettings", "rail-texture-set-settings"),
        ("ui.view.display", "rail-display"),
        ("ui.view.shader", "rail-shader"),
        ("ui.view.history", "rail-history"),
        ("ui.view.log", "rail-log"),
    ] {
        texturecraft_ui_egui::menus::invoke(harness.state_mut(), &context, command);
        save(&mut harness, &output_dir, name);
    }

    harness.set_size(egui::vec2(1600.0, 600.0));
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &context, "ui.view.display");
    harness.run_steps(6);
    harness.get_by_label("Temporal anti-aliasing (refines the 3D view while it is still)").scroll_to_me();
    save(&mut harness, &output_dir, "rail-display-short");

    harness.set_size(egui::vec2(1600.0, 400.0));
    texturecraft_ui_egui::menus::invoke(harness.state_mut(), &context, "ui.view.history");
    save(&mut harness, &output_dir, "rail-history-short");
}
