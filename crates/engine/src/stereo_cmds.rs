//! Inspectable view choices; scene, document and history are unchanged.
use serde::Serialize;
use serde_json::{Value, json};
use texturecraft_core::MAX_VIEW_SIDE;

use crate::commands::{CommandSpec, always, bad};
use crate::{Result, Session, cmd};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
pub struct Settings {
    pub enabled: bool,
    pub resolution: Option<[u32; 2]>,
}

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "viewport.stereo",
        "Stereo view",
        ["Viewport"],
        None,
        "{enabled?: bool, resolution?: [width, height] | null}; omitted fields stay; null uses the model viewport size",
        always,
        settings
    )]
}

fn settings(s: &mut Session, p: &Value) -> Result<Value> {
    let mut next = s.stereo;
    if let Some(value) = p.get("enabled") {
        next.enabled = value.as_bool().ok_or_else(|| bad("viewport.stereo", "enabled must be a boolean"))?;
    }
    if let Some(value) = p.get("resolution") {
        next.resolution = if value.is_null() {
            None
        } else {
            let invalid = || bad("viewport.stereo", format!("resolution must be null or [width, height], each between 1 and {MAX_VIEW_SIDE}"));
            let values = value.as_array().filter(|v| v.len() == 2).ok_or_else(invalid)?;
            let dimension = |index: usize| {
                values.get(index).and_then(Value::as_u64).filter(|v| (1..=u64::from(MAX_VIEW_SIDE)).contains(v)).map(|v| v as u32).ok_or_else(invalid)
            };
            Some([dimension(0)?, dimension(1)?])
        };
    }
    s.stereo = next;
    Ok(json!(next))
}
