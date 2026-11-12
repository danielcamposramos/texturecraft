//! Complete ordinary UI control, using the same harness and input in candidate and stock.
use egui_kittest::Harness;
use serde_json::json;
use texturecraft_engine::{execute, sample_session};
use texturecraft_ui_egui::{TextureCraftApp, state::ViewLayout};

#[test]
#[ignore = "requires an isolated software GPU, SCENE3_MODEL and SCENE3_STOCK_SNAPSHOT"]
fn ordinary_ui_with_controlled_timing() {
    let model = std::env::var("SCENE3_MODEL").expect("SCENE3_MODEL must name the original two-depth OBJ");
    let output = std::path::PathBuf::from(std::env::var_os("SCENE3_STOCK_SNAPSHOT").expect("SCENE3_STOCK_SNAPSHOT must name a fresh output directory"));
    std::fs::create_dir_all(&output).unwrap();
    let mut session = sample_session(128, 16).unwrap();
    execute(&mut session, "file.new", &json!({"mesh": model, "resolution": 128})).unwrap();
    for (index, color) in ["#ff0000", "#0000ff"].into_iter().enumerate() {
        execute(&mut session, "textureSet.select", &json!({"ts": index})).unwrap();
        execute(&mut session, "layer.add.fill", &json!({"color": color, "name": (["near", "far"][index])})).unwrap();
    }
    execute(&mut session, "camera.set", &json!({"yaw":0,"pitch":0,"distance":5,"fov":40,"target":[0,0,0]})).unwrap();
    execute(&mut session, "viewport.view", &json!({"mode":"baseColor"})).unwrap();
    execute(&mut session, "display.set", &json!({"taa":false,"shadows":false})).unwrap();
    let mut harness = Harness::builder()
        .with_size(egui::vec2(1000.0, 680.0))
        .with_pixels_per_point(1.0)
        .wgpu()
        .build_eframe(|context| TextureCraftApp::new(context, session));
    harness.state_mut().ui.layout = ViewLayout::Only3D;
    harness.state_mut().ui.prefs.language = "en".into();
    let info = harness.state().render_state.as_ref().unwrap().adapter.get_info();
    assert_eq!(info.backend, texturecraft_render::wgpu::Backend::Vulkan);
    assert_eq!(info.device_type, texturecraft_render::wgpu::DeviceType::Cpu);
    eprintln!("ordinary control adapter: {} ({:?}, {:?})", info.name, info.backend, info.device_type);
    for _ in 0..24 {
        // This public test input fixes the diagnostic number drawn in the status bar.
        // No output pixel is excluded, and production timing remains unchanged.
        harness.state_mut().ui_ms = 0.0;
        harness.run_steps(1);
        harness.render().unwrap();
    }
    assert_eq!(harness.state().session.pending_tiles(), 0);
    let first = harness.render().unwrap();
    first.save(output.join("whole-ui.png")).unwrap();
    harness.state_mut().ui_ms = 0.0;
    harness.run_steps(1);
    let second = harness.render().unwrap();
    assert_eq!(first.as_raw(), second.as_raw(), "the complete ordinary UI must be stable under the controlled timing input");
    second.save(output.join("whole-ui-stable.png")).unwrap();
    assert_eq!(first.dimensions(), (1000, 680));
}
