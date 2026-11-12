use serde_json::json;
use texturecraft_engine::{execute, sample_session};

#[test]
fn stereo_view_settings_are_inspectable_atomic_view_state() {
    let mut s = sample_session(128, 16).unwrap();
    let document = serde_json::to_value(&s.doc.project).unwrap();
    let before = texturecraft_engine::inspect::session(&s);
    let settings = execute(&mut s, "viewport.stereo", &json!({"enabled":true,"resolution":[640,360]})).unwrap();
    assert_eq!(settings, json!({"enabled":true,"resolution":[640,360]}));
    assert_eq!(texturecraft_engine::inspect::session(&s)["stereo"], settings);
    for bad in [
        json!({"resolution":[0,360]}),
        json!({"resolution":[9000,360]}),
        json!({"resolution":[640]}),
        json!({"enabled":"yes"}),
        json!({"enabled":false,"resolution":[1,-1]}),
    ] {
        assert!(execute(&mut s, "viewport.stereo", &bad).is_err());
        assert_eq!(execute(&mut s, "viewport.stereo", &json!({})).unwrap(), settings, "a rejected request changes nothing");
    }
    assert_eq!(execute(&mut s, "viewport.stereo", &json!({"resolution":null})).unwrap(), json!({"enabled":true,"resolution":null}));
    assert_eq!(serde_json::to_value(&s.doc.project).unwrap(), document);
    let after = texturecraft_engine::inspect::session(&s);
    assert_eq!(before["history"], after["history"]);
    assert_eq!(before["modified"], after["modified"]);
    assert_eq!(before["camera"], after["camera"]);
}

#[test]
fn stereo_choice_survives_a_project_switch_as_view_state() {
    let mut s = sample_session(128, 16).unwrap();
    let settings = execute(&mut s, "viewport.stereo", &json!({"enabled":true,"resolution":[640,360]})).unwrap();
    execute(&mut s, "project.new", &json!({"mesh":"cube","resolution":128})).unwrap();
    assert_eq!(execute(&mut s, "viewport.stereo", &json!({})).unwrap(), settings);
    assert!(!s.modified, "keeping a view choice must not mark the new document modified");
    assert!(texturecraft_engine::inspect::session(&s)["history"]["undo"].is_null());
}
