use eframe::App as _;
use egui_kittest::Harness;
use egui_kittest::kittest::Queryable;
use texturecraft_engine::{Session, formats, geom};
use texturecraft_ui_egui::{TextureCraftApp, UiState, control, state::Prefs};

#[path = "../src/ui_state_store/memory_storage.rs"]
mod memory_storage;
use memory_storage::MemoryStorage;

#[test]
fn resource_settings_follow_hardware_persist_and_apply_before_control_requests() {
    let legacy: Prefs = serde_json::from_str("{}").unwrap();
    assert_eq!(legacy.project_open_limit_gib, 8);
    let mut state = UiState::default();
    state.prefs.language = "en".into();
    state.prefs.project_open_limit_gib = 16;
    state.prefs.reserved_cores = u32::MAX;
    state.windows.settings = true;
    let mut storage = MemoryStorage::default();
    storage.values.insert("ui.json".into(), serde_json::to_string(&state).unwrap());
    let session = Session::new_project("Limits", geom::primitives::plane(2.0, 1), "builtin:plane", None, 128).unwrap();
    let available_cores = u32::try_from(std::thread::available_parallelism().unwrap().get()).unwrap();
    let mut harness =
        Harness::builder().with_size(egui::vec2(1200.0, 900.0)).with_step_dt(1.0 / 60.0).with_accessibility_check(false).build_eframe(|context| {
            context.egui_ctx.data_mut(|data| data.insert_temp(egui::Id::unique("settings_resources"), (0.0_f64, available_cores, 16_u64 << 30)));
            let mut creation = eframe::CreationContext::_new_kittest(context.egui_ctx.clone());
            creation.storage = Some(&storage);
            TextureCraftApp::new(&creation, session)
        });

    assert_eq!(formats::project_open_limit_gib(), 16, "saved preferences apply during startup");
    harness.run_steps(3);
    assert_eq!(harness.state().ui.prefs.reserved_cores, available_cores, "preferences from a larger machine are clamped");
    harness.get_by_role_and_label(egui::Role::SpinButton, "Cores kept for the UI").set_value_accesskit(f64::from(u32::MAX));
    harness.run_steps(3);
    assert_eq!(harness.state().ui.prefs.reserved_cores, available_cores, "editing cannot exceed this machine's CPU count");
    assert!(harness.query_by_label("≈ 8.00 GiB").is_some(), "50% of 16 GiB is shown beside the percentage");
    harness.get_by_role_and_label(egui::Role::SpinButton, "Max RAM while saving").set_value_accesskit(25.0);
    harness.run_steps(3);
    assert!(harness.query_by_label("≈ 4.00 GiB").is_some(), "the byte estimate follows the edited percentage");

    harness.get_by_role_and_label(egui::Role::SpinButton, "Max project size on open").set_value_accesskit(24.0);
    harness.run_steps(3);
    assert_eq!(formats::project_open_limit_gib(), 24, "the input field applies the edited value");

    let context = egui::Context::default();
    let reply = control::handle(harness.state_mut(), &context, "ui.set", &serde_json::json!({"prefs": {"project_open_limit_gib": 32}}));

    assert_eq!(reply["ok"], true, "{reply}");
    assert_eq!(formats::project_open_limit_gib(), 32, "no frame is needed before the next command");
    harness.state_mut().save(&mut storage);
    let restored: UiState = serde_json::from_str(&storage.values["ui.json"]).unwrap();
    assert_eq!(restored.prefs.project_open_limit_gib, 32);
}
