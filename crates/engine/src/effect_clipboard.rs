//! Copy and paste effects between layers, masks and texture sets. The clipboard is the layer
//! clipboard: it holds a layer or an effect.

use serde_json::{Value, json};
use texturecraft_core::{Channel, EffectKind, Generator, LayerId, Mask, MaskBackground, Source};

use crate::commands::{CommandSpec, bad, has_layer, layer_param, strp};
use crate::layer_ops::{empty_texel, fit};
use crate::mask_cmds::{anchor_source, content_target, effect_id};
use crate::session::StackTarget;
use crate::{EngineError, Result, Session, cmd};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("effect.copy", "Copy effect", [], None, "{effect?: the selected effect}", has_layer, copy),
        cmd!("effect.cut", "Cut effect", [], None, "{effect?: the selected effect}", has_layer, cut),
        cmd!(
            "effect.paste",
            "Paste effect",
            [],
            None,
            "{layer?, target?: mask|content (default: the selection's stack)} → {effect} — above the selected effect, or on top of the stack",
            can_paste,
            paste
        ),
    ]
}

/// The copied effect, if the clipboard holds one.
pub fn clipboard_effect(s: &Session) -> Option<LayerId> {
    s.clipboard.as_ref().and_then(|(doc, id)| doc.project.effect(*id).map(|_| *id))
}

fn can_paste(s: &Session) -> std::result::Result<(), String> {
    match clipboard_effect(s) {
        Some(_) if s.selected_layer().is_some() => Ok(()),
        Some(_) => Err("select a layer to paste the effect on".into()),
        None => Err("the clipboard holds no effect".into()),
    }
}

fn copy(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "effect.copy";
    let eid = effect_id(s, p, CMD)?;
    if s.doc.project.effect(eid).is_none() {
        return Err(bad(CMD, format!("no effect {}", eid.0)));
    }
    s.clipboard = Some((s.doc.clone(), eid));
    Ok(json!({ "copied": eid.0 }))
}

fn cut(s: &mut Session, p: &Value) -> Result<Value> {
    let eid = copy(s, p)?["copied"].clone();
    crate::execute(s, "effect.remove", &json!({ "effect": eid }))?;
    Ok(json!({ "cut": eid }))
}

fn paste(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "effect.paste";
    let (src, eid) = s.clipboard.clone().ok_or_else(|| bad(CMD, "the clipboard holds no effect"))?;
    let effect = src.project.effect(eid).map(|(_, e)| e.clone()).ok_or_else(|| bad(CMD, "the clipboard holds no effect"))?;
    let layer = layer_param(s, CMD, p)?;
    // The selected effect's stack, or the content or mask the selection targets.
    let selected = match s.target {
        StackTarget::Effect(x) if s.selected_layer() == Some(layer) => Some(x),
        _ => None,
    };
    let in_content = |s: &Session, x: LayerId| {
        s.doc.project.effect(x).and_then(|(o, _)| s.doc.project.layers.get(&o)).is_some_and(|l| l.content_effects.iter().any(|e| e.id == x))
    };
    let asked = strp(p, "target").is_some();
    let content = match selected {
        _ if asked => content_target(p, CMD)?,
        Some(x) => in_content(s, x),
        None => s.target == StackTarget::Content && s.selected_layer() == Some(layer),
    };
    // Effects that fit one stack only go there, unless another was asked for.
    let content = match effect.kind.content_only() {
        Some(only) if only != content && asked => {
            return Err(bad(CMD, format!("{} effects go on the {}", effect.kind.name(), if only { "layer content" } else { "mask" })));
        }
        Some(only) => only,
        None => content,
    };
    // An instance's content and (unless it has its own) mask belong to its source.
    let owner = if content { s.doc.project.content_layer(layer).map_or(layer, |l| l.id) } else { s.doc.project.mask_owner(layer) };
    let owner_layer = s.doc.project.layer(owner)?;
    if content && owner_layer.is_group() {
        return Err(EngineError::Other(format!("\"{}\" is a folder: content effects go on paint and fill layers", owner_layer.name)));
    }
    // A fill reading an anchor point needs one below its new layer.
    if let EffectKind::Fill { source: Source::Anchor { anchor, channel }, .. } = &effect.kind {
        anchor_source(s, CMD, owner, *anchor, *channel)?;
    }
    // So does a generator reading one as micro detail.
    if let EffectKind::Generator(Generator { micro: Some(micro), .. }) = &effect.kind {
        anchor_source(s, CMD, owner, micro.anchor, Channel::Height)?;
    }
    let (ts, _, _) = s.doc.project.locate(owner).ok_or_else(|| bad(CMD, "the layer is not in a stack"))?;
    let (size, channels) = s.doc.project.texture_set(ts).map(|t| (t.size(), t.channel_list()))?;
    let new = s.edit("Paste effect", |s| {
        let new = s.doc.project.alloc_id();
        let l = s.doc.project.layer_mut(owner)?;
        let list = if content {
            &mut l.content_effects
        } else {
            let bg = if matches!(effect.kind, EffectKind::Generator(_) | EffectKind::Fill { .. }) { MaskBackground::Black } else { MaskBackground::White };
            &mut l.mask.get_or_insert_with(|| Mask::new(bg)).effects
        };
        let at = selected.and_then(|x| list.iter().position(|e| e.id == x)).map_or(list.len(), |i| i + 1);
        list.insert(at, texturecraft_core::Effect { id: new, ..effect.clone() });
        // Painted pixels come along, resampled to the set's size (a mask's paint stays a mask's).
        if let Some(img) = src.pixels.masks.get(&eid).filter(|_| !content) {
            s.doc.pixels.masks.insert(new, fit(img, size, &[0, 0]));
        }
        for c in channels.iter().copied().filter(|_| content) {
            if let Some(img) = src.pixels.layer(eid, c) {
                s.doc.pixels.layers.insert((new, c), fit(img, size, &empty_texel(c)));
            }
        }
        // The target belongs to the active set's selection: a paste into another set leaves it.
        if let Some((lts, _, _)) = s.doc.project.locate(layer) {
            s.selected.insert(lts, layer);
            if lts == s.ts() {
                s.target = StackTarget::Effect(new);
            }
        }
        Ok(new)
    })?;
    Ok(json!({ "effect": new.0 }))
}
