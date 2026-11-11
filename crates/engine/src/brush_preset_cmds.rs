//! Brush presets: the built-in ones, and the user's own saved as `.tcbrush` files (the brush
//! settings as JSON). Applying or loading one changes the stroke and keeps the brush's size.

use serde_json::{Value, json};
use texturecraft_paint::BrushSettings;
#[cfg(not(target_arch = "wasm32"))]
use texturecraft_paint::presets::with_stroke_of;
use texturecraft_paint::presets::{PRESETS, find_preset};

use crate::commands::{CommandSpec, always, bad, strp};
use crate::tool_cmds::brush_json;
use crate::{Result, Session, cmd};

/// File extension of a saved brush preset.
pub const BRUSH_EXT: &str = "tcbrush";
/// Largest preset file read.
#[cfg(not(target_arch = "wasm32"))]
const MAX_BYTES: u64 = 1 << 20;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("brush.preset.list", "List brush presets", [], None, "{} — the built-in presets: id, label, category", always, |_, _| Ok(list())),
        cmd!(
            "brush.preset.apply",
            "Apply brush preset",
            [],
            None,
            "{preset: id or label (brush.preset.list)} — sets the stroke (alpha, spacing, flow, hardness, jitters…), keeps the size",
            always,
            apply
        ),
        cmd!("brush.preset.save", "Save brush preset", [], None, "{path} — the current brush as a .tcbrush file", always, save),
        cmd!("brush.preset.load", "Load brush preset", [], None, "{path} — a .tcbrush file's stroke on the brush, keeping the size", always, load),
    ]
}

fn list() -> Value {
    json!(PRESETS.iter().map(|p| json!({"id": p.id, "label": p.label, "category": p.category})).collect::<Vec<_>>())
}

fn apply(s: &mut Session, p: &Value) -> Result<Value> {
    let name = strp(p, "preset").ok_or_else(|| bad("brush.preset.apply", "`preset` is required"))?;
    let preset = find_preset(name).ok_or_else(|| bad("brush.preset.apply", format!("unknown brush preset `{name}` (see brush.preset.list)")))?;
    set(s, preset.applied(&s.tool.brush))
}

fn set(s: &mut Session, brush: BrushSettings) -> Result<Value> {
    s.tool.brush = brush;
    s.revision = s.revision.wrapping_add(1);
    Ok(brush_json(s))
}

#[cfg(not(target_arch = "wasm32"))]
fn save(s: &mut Session, p: &Value) -> Result<Value> {
    let path = strp(p, "path").ok_or_else(|| bad("brush.preset.save", "`path` is required"))?;
    let bytes = serde_json::to_vec_pretty(&s.tool.brush).map_err(|e| bad("brush.preset.save", e.to_string()))?;
    std::fs::write(path, bytes).map_err(|e| bad("brush.preset.save", format!("cannot write {path}: {e}")))?;
    Ok(json!({ "path": path }))
}

#[cfg(not(target_arch = "wasm32"))]
fn load(s: &mut Session, p: &Value) -> Result<Value> {
    use std::io::Read as _;
    let cmd = "brush.preset.load";
    let path = strp(p, "path").ok_or_else(|| bad(cmd, "`path` is required"))?;
    let mut bytes = Vec::new();
    std::fs::File::open(path).and_then(|f| f.take(MAX_BYTES + 1).read_to_end(&mut bytes)).map_err(|e| bad(cmd, format!("cannot read {path}: {e}")))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(bad(cmd, format!("{path} is too large for a brush preset")));
    }
    let mut from: BrushSettings = serde_json::from_slice(&bytes).map_err(|e| bad(cmd, format!("{path} is not a brush preset: {e}")))?;
    // An image alpha belongs to the project that saved it.
    if from.alpha_image.as_ref().is_some_and(|n| !s.images.contains_key(n)) {
        from.alpha_image = None;
    }
    set(s, with_stroke_of(&s.tool.brush, &from))
}

#[cfg(target_arch = "wasm32")]
fn save(_: &mut Session, _: &Value) -> Result<Value> {
    Err(bad("brush.preset.save", "file paths are not available in the browser"))
}

#[cfg(target_arch = "wasm32")]
fn load(_: &mut Session, _: &Value) -> Result<Value> {
    Err(bad("brush.preset.load", "file paths are not available in the browser"))
}
