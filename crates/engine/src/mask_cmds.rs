//! Mask and effect commands (the layer context menu's mask section and "Add <effect>").

use serde_json::{Value, json};
use texturecraft_core::{
    BlendMode, Channel, CompareOp, CurvatureDetail, CurvatureMode, Effect, EffectKind, Filter, Generator, GeneratorKind, LayerId, Levels, Mask, MaskBackground,
    MicroDetail, Procedural, ProceduralKind, Project, Source,
};

use crate::commands::{CommandSpec, always, bad, blend_param, boolp, channel_param, color_param, f32p, has_layer, layer_param, strp, u64p};
use crate::layer_cmds::projection_from;
use crate::session::StackTarget;
use crate::{EngineError, Result, Session, cmd};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("mask.add.white", "Add white mask", ["Layers", "Mask"], None, "{layer?}", has_layer, |s, p| add_mask(s, p, MaskBackground::White)),
        cmd!("mask.add.black", "Add black mask", ["Layers", "Mask"], None, "{layer?}", has_layer, |s, p| add_mask(s, p, MaskBackground::Black)),
        cmd!(
            "mask.add.colorSelection",
            "Add mask with color selection",
            ["Layers", "Mask"],
            None,
            "{layer?, color?: [r,g,b] ID color, tolerance?}",
            has_layer,
            add_color_mask
        ),
        cmd!("mask.toggle", "Toggle mask", ["Layers", "Mask"], None, "{layer?, enabled?}", has_layer, toggle),
        cmd!("mask.invertBackground", "Invert mask background", ["Layers", "Mask"], None, "{layer?}", has_layer, invert_background),
        cmd!("mask.invert", "Invert mask", ["Layers", "Mask"], None, "{layer?}", has_layer, invert),
        cmd!("mask.clear", "Clear mask", ["Layers", "Mask"], None, "{layer?}", has_layer, clear),
        cmd!("mask.remove", "Remove mask", ["Layers", "Mask"], None, "{layer?}", has_layer, remove),
        cmd!("effect.add.paint", "Add paint", ["Layers", "Effects"], None, "{layer?, target?: mask|content, name?}", has_layer, |s, p| push_effect(
            s,
            p,
            EffectKind::Paint,
            BlendMode::Normal
        )),
        cmd!(
            "effect.add.fill",
            "Add fill",
            ["Layers", "Effects"],
            None,
            "{layer?, target?: mask|content, name?, mask: value?: 0..1 | procedural?: kind, scale?, seed? | anchor?: anchor effect id, anchorChannel?: channel (height); content: color?, roughness?, metallic?, height?, opacity? (default: every layer channel); projection?}",
            has_layer,
            add_fill
        ),
        cmd!(
            "effect.add.generator",
            "Add generator",
            ["Layers", "Effects"],
            None,
            "{layer?, name?, generator?: edgeWear|dirt|thinAreas|curvature|ambientOcclusion|position|direction, level?, softness?, noise?, invert?, mode?: edges|cavities|both|raw, sizes?: [7 weights 0..1, finest first] (curvature and edgeWear), microAnchor?: anchor effect id below (its height as micro detail: edges where raised, cavities where recessed; null removes it), microAmount?: 0..1 (0.5)}",
            has_layer,
            add_generator
        ),
        cmd!(
            "effect.add.levels",
            "Add levels",
            ["Layers", "Effects"],
            None,
            "{layer?, target?: mask|content, name?, inLow?, inHigh?, gamma?, outLow?, outHigh?}",
            has_layer,
            add_levels
        ),
        cmd!(
            "effect.add.filter",
            "Add filter",
            ["Layers", "Effects"],
            None,
            "{layer?, target?: mask|content, name?, filter?: id from filter.list (invert, threshold, blur, sharpen, hsl, levels, contrastLuminosity, colorBalance, gradientMap, posterize, clamp, histogramScan|Range|Shift, warp, blurDirectional, slopeBlur, matFinishGrind|Raw|Sanded|Hammered), level?, radius?, amount?, hue?: turns -1..1, saturation?, lightness?, contrast?, luminosity?, shadows?|midtones?|highlights?: [r,g,b] -1..1, colors?: [[r,g,b]|\"#rrggbb\", ...], levels?, min?, max?, position?, range?, offset?, intensity?, scale?, seed?, angle?} — material finishes go on the content",
            has_layer,
            add_filter
        ),
        cmd!(
            "effect.add.compare",
            "Add compare mask",
            ["Layers", "Effects"],
            None,
            "{layer?, name?, op?: lesser|within|greater, value?, tolerance?}",
            has_layer,
            add_compare
        ),
        cmd!(
            "effect.add.anchor",
            "Add anchor point",
            ["Layers", "Effects"],
            None,
            "{layer?, target?: content|mask (content), name?} (on the content it records the stack up to this layer, in a mask the mask value at that point; fills, masks and generators above read it: effect.add.fill {anchor}, effect.add.generator {microAnchor})",
            has_layer,
            add_anchor
        ),
        cmd!(
            "effect.set",
            "Edit effect",
            [],
            None,
            "{effect, visible?, blend?, opacity?, name?, filter?: id from filter.list (a filter effect only: swaps its type, keeping position, blend and opacity; its current id edits it in place), level?, softness?, noise?, noiseScale?, invert?, mode?, sizes?, microAnchor?: id|null, microAmount?, value?, anchor?, anchorChannel?, inLow?, inHigh?, gamma?, outLow?, outHigh?, tolerance?, colors?: [[r,g,b], ...] (a color selection's ID colors), filter parameters (see effect.add.filter)}",
            has_layer,
            set
        ),
        cmd!("effect.remove", "Remove effect", [], None, "{effect?: the selected effect}", has_layer, remove_effect),
        cmd!(
            "effect.duplicate",
            "Duplicate effect",
            [],
            None,
            "{effect?: the selected effect} → {effect} (the copy, just above it, pixels included)",
            has_layer,
            duplicate_effect
        ),
        cmd!("effect.move", "Move effect", [], None, "{effect, index: position in its stack, 0 = bottom (applied first)}", has_layer, move_effect),
        cmd!(
            "effect.histogram",
            "Effect input histogram",
            [],
            None,
            "{effect, channel?: for content effects (default: the layer's first channel), bins?: 2..256 (64)} → {target: mask|content, channel?, bins: [count per bucket over 0..1]}",
            always,
            histogram
        ),
    ]
}

fn with_mask<T>(s: &mut Session, p: &Value, cmd: &str, label: &str, f: impl FnOnce(&mut Session, LayerId) -> Result<T>) -> Result<T> {
    let id = layer_param(s, cmd, p)?;
    s.edit(label, |s| f(s, id))
}

fn add_mask(s: &mut Session, p: &Value, bg: MaskBackground) -> Result<Value> {
    let label = if bg == MaskBackground::White { "Add white mask" } else { "Add black mask" };
    with_mask(s, p, "mask.add", label, |s, id| {
        let l = s.doc.project.layer_mut(id)?;
        if l.mask.is_some() {
            return Err(EngineError::Other(format!("\"{}\" already has a mask", l.name)));
        }
        l.mask = Some(Mask::new(bg));
        s.target = StackTarget::Mask;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn add_color_mask(s: &mut Session, p: &Value) -> Result<Value> {
    let color = color_param(p, "color").unwrap_or([1.0, 0.0, 0.0]);
    let tolerance = f32p(p, "tolerance").unwrap_or(0.1).clamp(0.0, 1.0);
    with_mask(s, p, "mask.add.colorSelection", "Add mask with color selection", |s, id| {
        let eid = s.doc.project.alloc_id();
        let l = s.doc.project.layer_mut(s.doc.project.mask_owner(id))?;
        let m = l.mask.get_or_insert_with(|| Mask::new(MaskBackground::Black));
        m.effects.push(Effect::new(eid, EffectKind::ColorSelection { colors: vec![color], tolerance }));
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The mask layer `id` draws with (an instance's source's unless it has its own).
fn mask_of(s: &mut Session, id: LayerId) -> Result<&mut Mask> {
    let l = s.doc.project.layer_mut(s.doc.project.mask_owner(id))?;
    let name = l.name.clone();
    l.mask.as_mut().ok_or_else(|| EngineError::Other(format!("\"{name}\" has no mask")))
}

fn toggle(s: &mut Session, p: &Value) -> Result<Value> {
    let on = with_mask(s, p, "mask.toggle", "Toggle mask", |s, id| {
        let m = mask_of(s, id)?;
        m.enabled = boolp(p, "enabled").unwrap_or(!m.enabled);
        Ok(m.enabled)
    })?;
    Ok(json!({ "enabled": on }))
}

fn invert_background(s: &mut Session, p: &Value) -> Result<Value> {
    with_mask(s, p, "mask.invertBackground", "Invert mask background", |s, id| {
        let m = mask_of(s, id)?;
        m.background = if m.background == MaskBackground::White { MaskBackground::Black } else { MaskBackground::White };
        Ok(())
    })?;
    Ok(Value::Null)
}

fn invert(s: &mut Session, p: &Value) -> Result<Value> {
    with_mask(s, p, "mask.invert", "Invert mask", |s, id| {
        let eid = s.doc.project.alloc_id();
        let m = mask_of(s, id)?;
        // Inverting twice cancels out.
        if m.effects.last().is_some_and(|e| matches!(e.kind, EffectKind::Filter(Filter::Invert))) {
            m.effects.pop();
        } else {
            m.effects.push(Effect::new(eid, EffectKind::Filter(Filter::Invert)));
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn clear(s: &mut Session, p: &Value) -> Result<Value> {
    with_mask(s, p, "mask.clear", "Clear mask", |s, id| {
        let m = mask_of(s, id)?;
        let ids: Vec<LayerId> = m.effects.drain(..).map(|e| e.id).collect();
        for e in ids {
            s.doc.pixels.masks.remove(&e);
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    with_mask(s, p, "mask.remove", "Remove mask", |s, id| {
        let m = s.doc.project.layer_mut(s.doc.project.mask_owner(id))?.mask.take().ok_or_else(|| EngineError::Other("the layer has no mask".into()))?;
        for e in m.effects {
            s.doc.pixels.masks.remove(&e.id);
        }
        s.target = StackTarget::Content;
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Whether `effect.add.*` targets the layer content (`target: "content"`) or its mask (the
/// default).
pub(crate) fn content_target(p: &Value, cmd: &str) -> Result<bool> {
    match strp(p, "target") {
        None | Some("mask") => Ok(false),
        Some("content") => Ok(true),
        Some(other) => Err(bad(cmd, format!("unknown target `{other}`: use content or mask"))),
    }
}

/// A valid `name` param.
fn name_param(p: &Value) -> Option<&str> {
    strp(p, "name").filter(|n| !n.trim().is_empty() && n.chars().count() <= 256)
}

/// Add an effect to the selected layer's mask (creating a mask if there is none) or, with
/// `target: "content"` or a content-only effect, to its content stack.
fn push_effect(s: &mut Session, p: &Value, kind: EffectKind, blend: BlendMode) -> Result<Value> {
    let content = kind.content_only() == Some(true) || content_target(p, "effect.add")?;
    if kind.content_only().is_some_and(|c| c != content) {
        return Err(bad("effect.add", format!("{} effects go on masks, not on layer content", kind.name())));
    }
    let label = format!("Add {}", kind.name().to_ascii_lowercase());
    let eid = with_mask(s, p, "effect.add", &label, |s, id| {
        let eid = s.doc.project.alloc_id();
        // An instance's content and (unless it has its own) mask belong to its source.
        let id = if content { s.doc.project.content_layer(id).map_or(id, |l| l.id) } else { s.doc.project.mask_owner(id) };
        let l = s.doc.project.layer_mut(id)?;
        let bg = if matches!(kind, EffectKind::Generator(_) | EffectKind::Fill { .. }) { MaskBackground::Black } else { MaskBackground::White };
        let mut e = Effect::new(eid, kind);
        e.blend = blend;
        if let Some(n) = name_param(p) {
            e.name = n.to_string();
        }
        if !content {
            l.mask.get_or_insert_with(|| Mask::new(bg)).effects.push(e);
        } else if l.is_group() {
            return Err(EngineError::Other(format!("\"{}\" is a folder: content effects go on paint and fill layers", l.name)));
        } else {
            l.content_effects.push(e);
        }
        s.target = StackTarget::Effect(eid);
        Ok(eid)
    })?;
    Ok(json!({ "effect": eid.0 }))
}

/// A source reading anchor point `anchor` for layer `reader`: the anchor must sit on a layer
/// below the reader in the same texture set (the compositor walks bottom → top).
pub(crate) fn anchor_source(s: &Session, cmd: &str, reader: LayerId, anchor: LayerId, channel: Channel) -> Result<Source> {
    let owner = match s.doc.project.effect(anchor) {
        Some((owner, e)) if matches!(e.kind, EffectKind::Anchor) => owner,
        _ => return Err(bad(cmd, format!("{} is not an anchor point", anchor.0))),
    };
    let (ts, _, _) = s.doc.project.locate(reader).ok_or_else(|| bad(cmd, "layer not in a stack"))?;
    let order: Vec<LayerId> = s.doc.project.walk(ts).into_iter().map(|(id, _)| id).collect();
    let pos = |id: LayerId| order.iter().position(|x| *x == id);
    match (pos(owner), pos(reader)) {
        (Some(o), Some(r)) if o < r => Ok(Source::Anchor { anchor, channel }),
        _ => Err(bad(cmd, "the anchor point must be on a layer below this one in the same texture set")),
    }
}

fn add_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let projection = projection_from(p, "effect.add.fill", None)?;
    let blend = blend_param("effect.add.fill", p, "blend")?.unwrap_or(BlendMode::Normal);
    let id = layer_param(s, "effect.add.fill", p)?;
    if content_target(p, "effect.add.fill")? {
        let mut fill = crate::layer_cmds::fill_from(p);
        if fill.sources.is_empty() {
            fill.sources = s.doc.project.layer(id)?.channels.keys().map(|c| (*c, Source::uniform_of(*c))).collect();
        }
        fill.projection = projection;
        return push_effect(s, p, EffectKind::Material(fill), blend);
    }
    if let Some(a) = u64p(p, "anchor") {
        let channel = channel_param("effect.add.fill", p, "anchorChannel")?.unwrap_or(Channel::Height);
        let source = anchor_source(s, "effect.add.fill", id, LayerId(a), channel)?;
        return push_effect(s, p, EffectKind::Fill { source, projection }, blend);
    }
    let source = match strp(p, "procedural") {
        Some(k) => {
            let kind = ProceduralKind::parse(k).ok_or_else(|| bad("effect.add.fill", format!("unknown procedural `{k}`")))?;
            let mut pr = Procedural::new(kind);
            if let Some(v) = f32p(p, "scale") {
                pr.scale = v.clamp(0.001, 1000.0);
            }
            if let Some(v) = u64p(p, "seed") {
                pr.seed = v as u32;
            }
            Source::Procedural { pattern: pr, low: [0.0; 3], high: [1.0; 3] }
        }
        None => Source::Uniform([f32p(p, "value").unwrap_or(1.0).clamp(0.0, 1.0); 3]),
    };
    push_effect(s, p, EffectKind::Fill { source, projection }, blend)
}

/// Apply `params` to `generator`, using `command` in errors
fn generator_from(command: &str, params: &Value, generator: &mut Generator) -> Result<()> {
    if let Some(level) = f32p(params, "level") {
        generator.level = level.clamp(0.0, 1.0);
    }
    if let Some(softness) = f32p(params, "softness") {
        generator.softness = softness.clamp(0.0, 1.0);
    }
    if let Some(noise) = f32p(params, "noise") {
        generator.noise_amount = noise.clamp(0.0, 2.0);
    }
    if let Some(noise_scale) = f32p(params, "noiseScale") {
        generator.noise_scale = noise_scale.clamp(0.01, 1000.0);
    }
    if let Some(seed) = u64p(params, "seed") {
        generator.seed = seed as u32;
    }
    if let Some(invert) = boolp(params, "invert") {
        generator.invert = invert;
    }
    if let Some(direction) = color_param(params, "direction") {
        generator.direction = direction;
    }

    // A mode or sizes also give a curvature generator saved before detail sizes its detail
    let mode = params
        .get("mode")
        .map(|value| {
            let name = value.as_str().ok_or_else(|| bad(command, "`mode` must be a string"))?;
            CurvatureMode::parse(name).ok_or_else(|| bad(command, format!("unknown curvature mode `{name}`")))
        })
        .transpose()?;
    let weights = params.get("sizes").map(|value| size_weights(command, value)).transpose()?;
    if mode.is_none() && weights.is_none() {
        return Ok(());
    }
    if !matches!(generator.kind, GeneratorKind::Curvature | GeneratorKind::EdgeWear) {
        return Err(bad(command, "only curvature and edge wear generators have a mode and detail sizes"));
    }

    let detail = generator.detail.get_or_insert_with(CurvatureDetail::default);
    if let Some(mode) = mode {
        detail.mode = mode;
    }
    if let Some(weights) = weights {
        detail.weights = weights;
    }
    Ok(())
}

/// Read one weight (0..1) per size of [`CurvatureDetail::SIZES`], finest first
fn size_weights(command: &str, value: &Value) -> Result<[f32; CurvatureDetail::SIZES.len()]> {
    let invalid = || bad(command, format!("`sizes` takes {} numbers, finest first", CurvatureDetail::SIZES.len()));
    let items = value.as_array().filter(|items| items.len() == CurvatureDetail::SIZES.len()).ok_or_else(invalid)?;
    let mut weights = [0.0; CurvatureDetail::SIZES.len()];
    for (weight, item) in weights.iter_mut().zip(items) {
        *weight = item.as_f64().ok_or_else(invalid)?.clamp(0.0, 1.0) as f32;
    }
    Ok(weights)
}

/// An anchor point on the layer content (the default, as before anchors went on masks) or, with
/// `target: "mask"`, in its mask.
fn add_anchor(s: &mut Session, p: &Value) -> Result<Value> {
    // Params may be null or an object without a target: both default to the content.
    let mut params = p.as_object().cloned().unwrap_or_default();
    params.entry("target").or_insert_with(|| json!("content"));
    push_effect(s, &Value::Object(params), EffectKind::Anchor, BlendMode::Normal)
}

/// A generator's micro detail from `microAnchor` / `microAmount` for a generator on layer
/// `reader` (None: leave it as it is; `Some(None)`: remove it).
fn micro_param(s: &Session, cmd: &str, reader: LayerId, params: &Value, current: Option<MicroDetail>) -> Result<Option<Option<MicroDetail>>> {
    let amount = match params.get("microAmount") {
        Some(value) => Some(value.as_f64().filter(|v| v.is_finite()).ok_or_else(|| bad(cmd, "`microAmount` must be a number 0..1"))?.clamp(0.0, 1.0) as f32),
        None => None,
    };
    let anchor = match params.get("microAnchor") {
        None => current.map(|m| m.anchor),
        Some(Value::Null) => return Ok(Some(None)),
        Some(value) => Some(LayerId(value.as_u64().ok_or_else(|| bad(cmd, "`microAnchor` must be an anchor effect id or null"))?)),
    };
    let Some(anchor) = anchor else {
        return match amount {
            Some(_) => Err(bad(cmd, "give `microAnchor` before `microAmount`: the generator has no micro detail")),
            None => Ok(None),
        };
    };
    if params.get("microAnchor").is_none() && amount.is_none() {
        return Ok(None);
    }
    // The same rule as fills: an anchor below, in the same texture set.
    anchor_source(s, cmd, reader, anchor, Channel::Height)?;
    let amount = amount.or(current.map(|m| m.amount)).unwrap_or(MicroDetail::DEFAULT_AMOUNT);
    Ok(Some(Some(MicroDetail { anchor, amount })))
}

fn add_generator(session: &mut Session, params: &Value) -> Result<Value> {
    let kind = match strp(params, "generator") {
        Some(name) => GeneratorKind::parse(name).ok_or_else(|| bad("effect.add.generator", format!("unknown generator `{name}`")))?,
        None => GeneratorKind::EdgeWear,
    };
    let mut generator = Generator::new(kind);
    generator_from("effect.add.generator", params, &mut generator)?;
    let reader = layer_param(session, "effect.add.generator", params)?;
    if let Some(micro) = micro_param(session, "effect.add.generator", reader, params, None)? {
        generator.micro = micro;
    }
    let blend = blend_param("effect.add.generator", params, "blend")?.unwrap_or(BlendMode::Normal);
    push_effect(session, params, EffectKind::Generator(generator), blend)
}

pub(crate) fn levels_from(p: &Value, l: &mut Levels) {
    for (k, slot) in [("inLow", &mut l.in_low), ("inHigh", &mut l.in_high), ("gamma", &mut l.gamma), ("outLow", &mut l.out_low), ("outHigh", &mut l.out_high)] {
        if let Some(v) = f32p(p, k) {
            *slot = if k == "gamma" { v.clamp(0.01, 10.0) } else { v.clamp(0.0, 1.0) };
        }
    }
}

fn add_levels(s: &mut Session, p: &Value) -> Result<Value> {
    let mut l = Levels::default();
    levels_from(p, &mut l);
    push_effect(s, p, EffectKind::Levels(l), BlendMode::Normal)
}

fn add_filter(s: &mut Session, p: &Value) -> Result<Value> {
    let id = strp(p, "filter").unwrap_or("invert");
    let kind = crate::filter_lib::build(id, p)?;
    // Library filters are named after their shelf entry (material finishes are fill effects).
    let mut p = p.clone();
    if name_param(&p).is_none()
        && let (Some(d), Some(o)) = (crate::filter_lib::find(id), p.as_object_mut())
    {
        o.insert("name".into(), json!(d.label));
    }
    push_effect(s, &p, kind, BlendMode::Normal)
}

fn add_compare(s: &mut Session, p: &Value) -> Result<Value> {
    let op = match strp(p, "op").unwrap_or("greater") {
        "lesser" | "less" => CompareOp::LesserThan,
        "within" => CompareOp::WithinTolerance,
        "greater" => CompareOp::GreaterThan,
        other => return Err(bad("effect.add.compare", format!("unknown op `{other}`"))),
    };
    let kind =
        EffectKind::Compare { op, value: f32p(p, "value").unwrap_or(0.5).clamp(0.0, 1.0), tolerance: f32p(p, "tolerance").unwrap_or(0.1).clamp(0.0, 1.0) };
    push_effect(s, p, kind, BlendMode::Normal)
}

/// The `effect` param, or the selected effect.
pub(crate) fn effect_id(s: &Session, p: &Value, cmd: &str) -> Result<LayerId> {
    match (u64p(p, "effect"), s.target) {
        (Some(e), _) => Ok(LayerId(e)),
        (None, StackTarget::Effect(e)) => Ok(e),
        (None, _) => Err(bad(cmd, "`effect` is required (no effect is selected)")),
    }
}

/// The distribution of the values entering an effect (what a levels effect remaps).
fn histogram(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "effect.histogram";
    let eid = effect_id(s, p, CMD)?;
    let (owner, _) = s.doc.project.effect(eid).ok_or_else(|| bad(CMD, format!("no effect {}", eid.0)))?;
    let l = s.doc.project.layer(owner)?;
    let content = l.content_effects.iter().any(|e| e.id == eid);
    let channel = match channel_param(CMD, p, "channel")? {
        Some(c) => c,
        None => l.channels.keys().next().copied().unwrap_or(Channel::BaseColor),
    };
    let bins = u64p(p, "bins").unwrap_or(64).clamp(2, 256) as usize;
    let counts = s.effect_histogram(eid, channel, bins).ok_or_else(|| bad(CMD, "the texture set is not ready"))?;
    Ok(if content { json!({"target": "content", "channel": channel.id(), "bins": counts}) } else { json!({"target": "mask", "bins": counts}) })
}

/// Edit visibility, blending, name, effect parameters, or a filter's type
fn set(session: &mut Session, params: &Value) -> Result<Value> {
    const CMD: &str = "effect.set";
    let effect = effect_id(session, params, CMD)?;
    let blend = blend_param(CMD, params, "blend")?;

    // `anchor` picks a fill's anchor point, `anchorChannel` alone re-targets the current one
    let anchor = match session.doc.project.effect(effect) {
        Some((owner, Effect { kind: EffectKind::Fill { source, .. }, .. })) => {
            let (current, channel) = match source {
                Source::Anchor { anchor, channel } => (Some(*anchor), *channel),
                _ => (None, Channel::Height),
            };
            let channel_param = channel_param(CMD, params, "anchorChannel")?;
            match u64p(params, "anchor").map(LayerId).or(current.filter(|_| channel_param.is_some())) {
                Some(anchor) => Some(anchor_source(session, CMD, owner, anchor, channel_param.unwrap_or(channel))?),
                None => None,
            }
        }
        _ => None,
    };
    let micro = match session.doc.project.effect(effect) {
        Some((owner, Effect { kind: EffectKind::Generator(generator), .. })) => micro_param(session, CMD, owner, params, generator.micro)?,
        _ if params.get("microAnchor").is_some() || params.get("microAmount").is_some() => {
            return Err(bad(CMD, "`microAnchor` and `microAmount` edit generator effects only"));
        }
        _ => None,
    };

    // Naming the filter the effect already is edits its parameters like a plain effect.set
    let swapped = match strp(params, "filter") {
        Some(filter) if crate::filter_lib::find(filter).is_some_and(|def| current_filter_id(session, effect) == Some(def.id)) => None,
        Some(filter) => Some(swapped_filter(session, effect, filter, params)?),
        None => None,
    };
    let swaps_filter = swapped.is_some();

    session.edit("Edit effect", |session| {
        let edited = session.doc.project.effect_mut(effect).ok_or_else(|| bad(CMD, format!("no effect {}", effect.0)))?;
        if let Some((kind, label)) = swapped {
            if let EffectKind::Filter(old) = &edited.kind
                && edited.name == old.label()
            {
                edited.name = label.to_string();
            }
            edited.kind = kind;
        }

        if let Some(visible) = boolp(params, "visible") {
            edited.visible = visible;
        }
        if let Some(blend) = blend {
            edited.blend = blend;
        }
        if let Some(opacity) = f32p(params, "opacity") {
            edited.opacity = opacity.clamp(0.0, 1.0);
        }
        if let Some(name) = name_param(params) {
            edited.name = name.to_string();
        }

        match &mut edited.kind {
            EffectKind::Generator(generator) => {
                generator_from(CMD, params, generator)?;
                if let Some(micro) = micro {
                    generator.micro = micro;
                }
            }
            EffectKind::Levels(levels) => levels_from(params, levels),
            // The swapped filter was built with the parameters already
            EffectKind::Filter(_) if swaps_filter => {}
            EffectKind::Filter(filter) => crate::filter_lib::apply(params, filter, CMD)?,
            EffectKind::Fill { source, .. } => {
                if let Some(anchor) = anchor {
                    *source = anchor;
                } else if let Some(value) = f32p(params, "value")
                    && matches!(source, Source::Uniform(_) | Source::Anchor { .. })
                {
                    *source = Source::Uniform([value.clamp(0.0, 1.0); 3]);
                }
            }
            EffectKind::ColorSelection { colors, tolerance } => crate::color_selection_cmds::apply_params(params, colors, tolerance)?,
            EffectKind::Compare { tolerance, .. } => {
                if let Some(value) = f32p(params, "tolerance") {
                    *tolerance = value.clamp(0.0, 1.0);
                }
            }
            _ => {}
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The library id of the filter effect `effect` is, if it is one
fn current_filter_id(s: &Session, effect: LayerId) -> Option<&'static str> {
    match &s.doc.project.effect(effect)?.1.kind {
        EffectKind::Filter(filter) => crate::filter_lib::find(filter.label()).map(|def| def.id),
        _ => None,
    }
}

/// Filter `id` built with `p` and its label, to replace filter effect `effect`
fn swapped_filter(s: &Session, effect: LayerId, id: &str, p: &Value) -> Result<(EffectKind, &'static str)> {
    const CMD: &str = "effect.set";
    let (owner, current) = s.doc.project.effect(effect).ok_or_else(|| bad(CMD, format!("no effect {}", effect.0)))?;
    if !matches!(current.kind, EffectKind::Filter(_)) {
        return Err(bad(CMD, format!("`filter` changes filter effects only, not {}", current.kind.name().to_ascii_lowercase())));
    }

    let def = crate::filter_lib::find(id).ok_or_else(|| bad(CMD, format!("unknown filter `{id}` (see filter.list)")))?;
    let mut kind = crate::filter_lib::build(def.id, &Value::Null)?;
    let EffectKind::Filter(filter) = &mut kind else {
        return Err(bad(CMD, format!("{} is its own effect type: add it with effect.add.filter", def.label)));
    };

    let mask = s.doc.project.layer(owner)?.mask.as_ref();
    let on_mask = mask.is_some_and(|mask| mask.effects.iter().any(|masked| masked.id == effect));
    if on_mask && !def.mask {
        return Err(bad(CMD, format!("{} goes on layer content, not on masks", def.label)));
    }

    crate::filter_lib::apply(p, filter, CMD)?;
    Ok((kind, def.label))
}

/// The stack (a mask's or a layer's content effects) holding effect `eid`, and its position there.
fn stack_of(project: &mut Project, eid: LayerId) -> Option<(&mut Vec<Effect>, usize)> {
    project.layers.values_mut().find_map(|l| {
        l.mask.iter_mut().map(|m| &mut m.effects).chain(std::iter::once(&mut l.content_effects)).find_map(|list| {
            let i = list.iter().position(|e| e.id == eid)?;
            Some((list, i))
        })
    })
}

fn duplicate_effect(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "effect.duplicate";
    let eid = effect_id(s, p, CMD)?;
    let new = s.edit("Duplicate effect", |s| {
        let new = s.doc.project.alloc_id();
        let (list, i) = stack_of(&mut s.doc.project, eid).ok_or_else(|| bad(CMD, format!("no effect {}", eid.0)))?;
        let mut e = list.get(i).cloned().ok_or_else(|| bad(CMD, format!("no effect {}", eid.0)))?;
        e.id = new;
        e.name = format!("{} copy", e.name);
        list.insert(i + 1, e);
        // Painted effects keep their pixels (copy-on-write tiles: cheap).
        let px = &mut s.doc.pixels;
        if let Some(img) = px.masks.get(&eid).cloned() {
            px.masks.insert(new, img);
        }
        for c in Channel::ALL {
            if let Some(img) = px.layer(eid, c).cloned() {
                px.layers.insert((new, c), img);
            }
        }
        Ok(new)
    })?;
    if s.target == StackTarget::Effect(eid) {
        s.target = StackTarget::Effect(new);
    }
    Ok(json!({ "effect": new.0 }))
}

fn move_effect(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "effect.move";
    let eid = effect_id(s, p, CMD)?;
    let index = u64p(p, "index").ok_or_else(|| bad(CMD, "`index` is required"))?;
    s.edit("Move effect", |s| {
        let (list, i) = stack_of(&mut s.doc.project, eid).ok_or_else(|| bad(CMD, format!("no effect {}", eid.0)))?;
        let e = list.remove(i);
        let to = usize::try_from(index).unwrap_or(usize::MAX).min(list.len());
        list.insert(to, e);
        Ok(())
    })?;
    Ok(Value::Null)
}

fn remove_effect(s: &mut Session, p: &Value) -> Result<Value> {
    let eid = effect_id(s, p, "effect.remove")?;
    s.edit("Remove effect", |s| {
        let content = match s.doc.project.effect(eid) {
            Some((owner, _)) => s.doc.project.layer(owner)?.content_effects.iter().any(|e| e.id == eid),
            None => return Err(bad("effect.remove", format!("no effect {}", eid.0))),
        };
        for l in s.doc.project.layers.values_mut() {
            for list in l.mask.iter_mut().map(|m| &mut m.effects).chain(std::iter::once(&mut l.content_effects)) {
                list.retain(|e| e.id != eid);
            }
        }
        s.doc.pixels.masks.remove(&eid);
        s.doc.pixels.layers.retain(|(id, _), _| *id != eid);
        if s.target == StackTarget::Effect(eid) {
            s.target = if content { StackTarget::Content } else { StackTarget::Mask };
        }
        Ok(())
    })?;
    Ok(Value::Null)
}
