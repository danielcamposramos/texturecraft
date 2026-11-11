//! Layer stack commands: add, remove, duplicate, select, rename, visibility, reorder, group,
//! per-channel blending and fill sources.

use serde_json::{Value, json};
use texturecraft_core::{BlendMode, Channel, ChannelBlend, EffectKind, Fill, Layer, LayerId, LayerKind, Procedural, ProceduralKind, Projection, Source};

use crate::commands::{
    CommandSpec, always, bad, blend_param, boolp, channel_param, color_param, content_param, f32p, has_layer, layer_param, normal_format_param, strp, u64p,
};
use crate::{EngineError, Result, Session, cmd, session::StackTarget};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("layer.add.paint", "Add paint layer", ["Layers"], None, "{name?}", always, add_paint),
        cmd!("layer.add.fill", "Add fill layer", ["Layers"], None, "{name?, color?, roughness?, metallic?}", always, add_fill),
        cmd!("layer.add.group", "Add folder", ["Layers"], None, "{name?}", always, add_group),
        cmd!("layer.remove", "Remove layer(s)", ["Layers"], Some("Delete"), "{layer?}", has_layer, remove),
        cmd!("layer.duplicate", "Duplicate layer(s)", ["Layers"], Some("Ctrl+D"), "{layer?}", has_layer, duplicate),
        cmd!("layer.group", "Group layer(s)", ["Layers"], Some("Ctrl+G"), "{layer?}", has_layer, group),
        cmd!("layer.select", "Select layer", [], None, "{layer, target?: content|mask|<effect id>}", always, select),
        cmd!("layer.rename", "Rename layer", [], None, "{layer?, name}", has_layer, rename),
        cmd!("layer.visible", "Toggle layer visibility", [], None, "{layer?, visible?}", has_layer, visible),
        cmd!("layer.move", "Move layer", [], None, "{layer?, index?} | {layer?, above?: id|null, inside?: bool} | {layer?, delta: ±n}", has_layer, move_layer),
        cmd!("layer.blend", "Set channel blending", [], None, "{layer?, channel?: id|all, mode?, opacity?: 0..1, enabled?}", has_layer, blend),
        cmd!(
            "layer.fill.set",
            "Set fill source",
            [],
            None,
            "{layer? | effect?: content fill effect id, channel, color?|value?|procedural?: kind, scale?, seed?, contrast?, balance?, invert?, low?, high?|image?, normalFormat?: openGl|directX (the image's normal map convention; alone, it changes the channel's image)|anchor?: anchor effect id, anchorChannel?, enabled?, solo?: true (this channel on, every other one off)}",
            has_layer,
            fill_set
        ),
        cmd!(
            "layer.fill.projection",
            "Set fill projection",
            [],
            None,
            "{layer? | effect?: content fill effect id, projection?: uv|triplanar|planar|spherical|cylindrical (default: the current one), scale?: number | [u, v] (uv), rotation?: degrees (uv), offset?: [u, v] (uv) | [x, y, z] (triplanar, planar), sharpness?: 1..64 (triplanar), axis?: 0|1|2 (planar, cylindrical)} — what is not given keeps its value",
            has_layer,
            fill_projection
        ),
        cmd!("layer.colorTag", "Set layer color tag", [], None, "{layer?, tag?: 0..7 (absent clears)}", has_layer, color_tag),
        cmd!("layer.list", "List layers", [], None, "{ts?}", always, list),
    ]
}

/// Insert a layer above the selection (inside it when it is a group) and select it.
pub(crate) fn insert_selected(s: &mut Session, layer: Layer) -> Result<LayerId> {
    let ts = s.ts();
    let sel = s.selected_layer();
    let inside = sel.and_then(|id| s.doc.project.layer(id).ok()).is_some_and(|l| l.is_group());
    let id = s.doc.project.insert_layer(ts, layer, sel, inside)?;
    s.selected.insert(ts, id);
    s.target = StackTarget::Content;
    Ok(id)
}

fn channels_of(s: &Session) -> Vec<Channel> {
    s.doc.project.texture_set(s.ts()).map(|t| t.channel_list()).unwrap_or_default()
}

fn add_paint(s: &mut Session, p: &Value) -> Result<Value> {
    let ts = s.ts();
    let name = strp(p, "name").map(str::to_string).unwrap_or_else(|| s.doc.project.next_layer_name(ts, "Layer"));
    let id = s.edit("Add paint layer", |s| {
        let id = s.doc.project.alloc_id();
        insert_selected(s, Layer::new(id, name, LayerKind::Paint, &channels_of(s)))
    })?;
    Ok(json!({ "layer": id.0 }))
}

/// A uniform fill from the `color`, `roughness`, `metallic`, `height` and `opacity` params.
pub(crate) fn fill_from(p: &Value) -> Fill {
    let mut fill = Fill::default();
    if let Some(c) = color_param(p, "color") {
        fill.sources.insert(Channel::BaseColor, Source::Uniform(c));
    }
    for (key, ch) in [("roughness", Channel::Roughness), ("metallic", Channel::Metallic), ("height", Channel::Height), ("opacity", Channel::Opacity)] {
        if let Some(v) = f32p(p, key) {
            fill.sources.insert(ch, Source::Uniform([v; 3]));
        }
    }
    fill
}

fn add_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let ts = s.ts();
    let name = strp(p, "name").map(str::to_string).unwrap_or_else(|| s.doc.project.next_layer_name(ts, "Fill layer"));
    let fill = fill_from(p);
    let id = s.edit("Add fill layer", |s| {
        let id = s.doc.project.alloc_id();
        insert_selected(s, Layer::new(id, name, LayerKind::Fill(fill), &channels_of(s)))
    })?;
    Ok(json!({ "layer": id.0 }))
}

fn add_group(s: &mut Session, p: &Value) -> Result<Value> {
    let ts = s.ts();
    let name = strp(p, "name").map(str::to_string).unwrap_or_else(|| s.doc.project.next_layer_name(ts, "Folder"));
    let id = s.edit("Add folder", |s| {
        let id = s.doc.project.alloc_id();
        let mut l = Layer::new(id, name, LayerKind::Group { children: Vec::new(), collapsed: false }, &channels_of(s));
        for b in l.channels.values_mut() {
            b.mode = BlendMode::Passthrough;
        }
        insert_selected(s, l)
    })?;
    Ok(json!({ "layer": id.0 }))
}

fn remove(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.remove", p)?;
    let before = s.doc.clone();
    let removed = s.edit("Remove layer", |s| {
        let (ts, parent, i) = s.doc.project.locate(id).ok_or(EngineError::Other("layer not in a stack".into()))?;
        let removed = s.doc.project.remove_layer(id)?;
        // Instances of what was removed keep their look as independent layers.
        crate::instance_cmds::detach_orphans(s, &before)?;
        // Select the layer below (or above) the removed one.
        let list = match parent {
            Some(g) => s.doc.project.layer(g).map(|l| l.children().to_vec()).unwrap_or_default(),
            None => s.doc.project.texture_set(ts).map(|t| t.stack.clone()).unwrap_or_default(),
        };
        let next = list.get(i.saturating_sub(1)).or(list.first()).copied().or(parent);
        match next {
            Some(n) => s.selected.insert(ts, n),
            None => s.selected.remove(&ts),
        };
        Ok(removed)
    })?;
    Ok(json!({ "removed": removed.iter().map(|l| l.0).collect::<Vec<_>>() }))
}

fn duplicate(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.duplicate", p)?;
    let (ts, _, _) = s.doc.project.locate(id).ok_or(EngineError::Other("layer not in a stack".into()))?;
    let src = s.doc.clone();
    let new = s.edit("Duplicate layer", |s| {
        let new = crate::layer_ops::copy_tree(&src, id, &mut s.doc, ts, Some(id), false, 0)?;
        let name = format!("{} copy", src.project.layer(id)?.name);
        s.doc.project.layer_mut(new)?.name = name;
        Ok(new)
    })?;
    s.selected.insert(ts, new);
    Ok(json!({ "layer": new.0 }))
}

fn group(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.group", p)?;
    let g = s.edit("Group layers", |s| {
        let (ts, _, _) = s.doc.project.locate(id).ok_or(EngineError::Other("layer not in a stack".into()))?;
        let gid = s.doc.project.alloc_id();
        let name = s.doc.project.next_layer_name(ts, "Folder");
        let channels = s.doc.project.texture_set(ts)?.channel_list();
        let mut gl = Layer::new(gid, name, LayerKind::Group { children: Vec::new(), collapsed: false }, &channels);
        for b in gl.channels.values_mut() {
            b.mode = BlendMode::Passthrough;
        }
        s.doc.project.insert_layer(ts, gl, Some(id), false)?;
        s.doc.project.move_above(id, Some(gid), true)?;
        s.selected.insert(ts, gid);
        Ok(gid)
    })?;
    Ok(json!({ "layer": g.0 }))
}

fn select(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.select", p)?;
    let (ts, _, _) = s.doc.project.locate(id).ok_or_else(|| bad("layer.select", "layer not in a stack"))?;

    // Paths paint into the active set
    if ts != s.ts() {
        s.finish_path()?;
    }

    s.active_set = ts;
    s.selected.insert(ts, id);
    s.target = match p.get("target") {
        Some(Value::String(t)) if t == "mask" => {
            if s.doc.project.layer(s.doc.project.mask_owner(id))?.mask.is_none() {
                return Err(bad("layer.select", "the layer has no mask"));
            }
            StackTarget::Mask
        }
        Some(Value::String(t)) if t == "content" => StackTarget::Content,
        Some(v) => match v.as_u64() {
            Some(e) if s.doc.project.effect(LayerId(e)).is_some_and(|(owner, _)| owner == id) => StackTarget::Effect(LayerId(e)),
            _ => return Err(bad("layer.select", "target must be content, mask or an effect id of this layer")),
        },
        None => StackTarget::Content,
    };

    s.revision = s.revision.wrapping_add(1);
    Ok(json!({ "layer": id.0, "textureSet": ts }))
}

fn rename(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.rename", p)?;
    let name = strp(p, "name").map(str::trim).filter(|n| !n.is_empty()).ok_or_else(|| bad("layer.rename", "`name` is required"))?.to_string();
    if name.chars().count() > 256 {
        return Err(bad("layer.rename", "name is longer than 256 characters"));
    }
    s.edit("Rename layer", |s| {
        s.doc.project.layer_mut(id)?.name = name;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn visible(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.visible", p)?;
    let v = s.edit("Layer visibility", |s| {
        let l = s.doc.project.layer_mut(id)?;
        l.visible = boolp(p, "visible").unwrap_or(!l.visible);
        Ok(l.visible)
    })?;
    Ok(json!({ "visible": v }))
}

fn move_layer(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.move", p)?;
    s.edit("Move layer", |s| {
        if let Some(d) = p.get("delta").and_then(Value::as_i64) {
            let (_, _, i) = s.doc.project.locate(id).ok_or(EngineError::Other("layer not in a stack".into()))?;
            let to = (i as i64 + d).max(0) as usize;
            s.doc.project.move_layer(id, to)?;
        } else if let Some(i) = u64p(p, "index") {
            s.doc.project.move_layer(id, usize::try_from(i).unwrap_or(usize::MAX))?;
        } else if p.get("above").is_some() {
            let above = u64p(p, "above").map(LayerId);
            s.doc.project.move_above(id, above, boolp(p, "inside").unwrap_or(false))?;
        } else {
            return Err(bad("layer.move", "give `delta`, `index` or `above`"));
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

fn blend(s: &mut Session, p: &Value) -> Result<Value> {
    let id = content_param(s, "layer.blend", p)?;
    let all = strp(p, "channel") == Some("all");
    let ch = if all { None } else { channel_param("layer.blend", p, "channel")? };
    let mode = blend_param("layer.blend", p, "mode")?;
    let opacity = f32p(p, "opacity").map(|o| o.clamp(0.0, 1.0));
    let enabled = boolp(p, "enabled");
    let ts_channels = channels_of(s);
    s.edit("Layer blending", |s| {
        let l = s.doc.project.layer_mut(id)?;
        let targets: Vec<Channel> = match ch {
            Some(c) => vec![c],
            None => ts_channels.clone(),
        };
        for c in targets {
            match enabled {
                Some(false) => {
                    l.channels.remove(&c);
                    continue;
                }
                Some(true) => {
                    let mode = if l.is_group() { BlendMode::Passthrough } else { BlendMode::Normal };
                    l.channels.entry(c).or_insert(ChannelBlend { mode, opacity: 1.0 });
                }
                None => {}
            }
            if let Some(b) = l.channels.get_mut(&c) {
                if let Some(m) = mode {
                    b.mode = m;
                }
                if let Some(o) = opacity {
                    b.opacity = o;
                }
            }
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

/// Which fill `layer.fill.*` edits: a fill layer's (`layer`) or a content fill effect's
/// (`effect`).
#[derive(Clone, Copy)]
enum FillTarget {
    Layer(LayerId),
    Effect(LayerId),
}

fn fill_target(s: &Session, p: &Value, cmd: &str) -> Result<FillTarget> {
    match u64p(p, "effect").map(LayerId) {
        Some(e) => match s.doc.project.effect(e) {
            Some((_, x)) if matches!(x.kind, EffectKind::Material(_)) => Ok(FillTarget::Effect(e)),
            _ => Err(bad(cmd, format!("{} is not a content fill effect", e.0))),
        },
        None => Ok(FillTarget::Layer(content_param(s, cmd, p)?)),
    }
}

fn fill_mut(s: &mut Session, target: FillTarget) -> Result<&mut Fill> {
    match target {
        FillTarget::Layer(id) => match &mut s.doc.project.layer_mut(id)?.kind {
            LayerKind::Fill(f) => Ok(f),
            _ => Err(EngineError::Other("not a fill layer".into())),
        },
        FillTarget::Effect(e) => match s.doc.project.effect_mut(e).map(|x| &mut x.kind) {
            Some(EffectKind::Material(f)) => Ok(f),
            _ => Err(EngineError::Other("not a content fill effect".into())),
        },
    }
}

fn fill_set(s: &mut Session, p: &Value) -> Result<Value> {
    let target = fill_target(s, p, "layer.fill.set")?;
    let ch = channel_param("layer.fill.set", p, "channel")?.ok_or_else(|| bad("layer.fill.set", "`channel` is required"))?;
    // `solo`: this channel on, every other one off.
    let solo = boolp(p, "solo") == Some(true);
    let enabled = if solo { Some(true) } else { boolp(p, "enabled") };
    let normal_format = normal_format_param("layer.fill.set", p)?;
    let source = if let Some(kind) = strp(p, "procedural") {
        let kind = ProceduralKind::parse(kind).ok_or_else(|| bad("layer.fill.set", format!("unknown procedural `{kind}`")))?;
        let mut pr = Procedural::new(kind);
        if let Some(v) = f32p(p, "scale") {
            pr.scale = v.clamp(0.001, 1000.0);
        }
        if let Some(v) = u64p(p, "seed") {
            pr.seed = v as u32;
        }
        if let Some(v) = f32p(p, "contrast") {
            pr.contrast = v.clamp(0.0, 20.0);
        }
        if let Some(v) = f32p(p, "balance") {
            pr.balance = v.clamp(-1.0, 1.0);
        }
        pr.invert = boolp(p, "invert").unwrap_or(false);
        Some(Source::Procedural { pattern: pr, low: color_param(p, "low").unwrap_or([0.0; 3]), high: color_param(p, "high").unwrap_or([1.0; 3]) })
    } else if let Some(c) = color_param(p, "color").or_else(|| color_param(p, "value")) {
        Some(Source::Uniform(c))
    } else if let Some(name) = strp(p, "image") {
        Some(Source::Image { name: name.to_string(), normal_format: normal_format.unwrap_or_default() })
    } else if let Some(a) = u64p(p, "anchor") {
        let reader = match target {
            FillTarget::Layer(id) => Some(id),
            FillTarget::Effect(e) => s.doc.project.effect(e).map(|(owner, _)| owner),
        };
        let reader = reader.ok_or_else(|| bad("layer.fill.set", "the fill is not in a layer"))?;
        let anchor_channel = channel_param("layer.fill.set", p, "anchorChannel")?.unwrap_or(ch);
        Some(crate::mask_cmds::anchor_source(s, "layer.fill.set", reader, LayerId(a), anchor_channel)?)
    } else if enabled.is_some() || normal_format.is_some() {
        None
    } else {
        return Err(bad("layer.fill.set", "give `color`, `value`, `procedural`, `image`, `anchor`, `normalFormat`, `enabled` or `solo`"));
    };
    if normal_format.is_some() && source.as_ref().is_some_and(|src| !matches!(src, Source::Image { .. })) {
        return Err(bad("layer.fill.set", "`normalFormat` goes with `image` or alone"));
    }
    // A format-only edit leaves the channel on or off.
    let touches_channel = source.is_some() || enabled.is_some();
    s.edit("Set fill", |s| {
        let f = fill_mut(s, target)?;
        // `normalFormat` alone: the convention of the channel's image.
        if let (None, Some(nf)) = (&source, normal_format) {
            match f.sources.get_mut(&ch) {
                Some(Source::Image { normal_format, .. }) => *normal_format = nf,
                _ => return Err(bad("layer.fill.set", "`normalFormat` needs an image on the channel")),
            }
        }
        match target {
            // A fill effect's channel is on while it has a source.
            FillTarget::Effect(_) if enabled == Some(false) => {
                f.sources.remove(&ch);
            }
            FillTarget::Effect(_) => {
                let src = source.unwrap_or_else(|| f.source(ch));
                f.sources.insert(ch, src);
                if solo {
                    f.sources.retain(|c, _| *c == ch);
                }
            }
            FillTarget::Layer(id) => {
                if let Some(src) = source {
                    f.sources.insert(ch, src);
                }
                let l = s.doc.project.layer_mut(id)?;
                if enabled == Some(false) {
                    l.channels.remove(&ch);
                } else if touches_channel {
                    l.channels.entry(ch).or_default();
                }
                if solo {
                    l.channels.retain(|c, _| *c == ch);
                }
            }
        }
        Ok(())
    })?;
    Ok(Value::Null)
}

/// The projection `p` asks for. Fields it does not give keep their values from `current` when
/// that is of the same kind (or `p` names no kind), and the kind's defaults otherwise.
pub(crate) fn projection_from(p: &Value, cmd: &str, current: Option<&Projection>) -> Result<Projection> {
    let named = match strp(p, "projection") {
        None => None,
        Some("uv") => Some(Projection::default()),
        Some("triplanar" | "tri-planar") => Some(Projection::Triplanar { scale: 1.0, sharpness: 8.0, offset: [0.0; 3] }),
        Some("planar") => Some(Projection::Planar { scale: 1.0, axis: 2, offset: [0.0; 3] }),
        Some("spherical") => Some(Projection::Spherical { scale: 1.0 }),
        Some("cylindrical") => Some(Projection::Cylindrical { scale: 1.0, axis: 1 }),
        Some(other) => return Err(bad(cmd, format!("unknown projection `{other}`"))),
    };
    let mut proj = match (named, current) {
        (Some(n), Some(c)) if std::mem::discriminant(&n) == std::mem::discriminant(c) => c.clone(),
        (Some(n), _) => n,
        (None, c) => c.cloned().unwrap_or_default(),
    };
    let scale = floats::<2>(p, "scale", 0.001, 1000.0);
    let axis = u64p(p, "axis").map(|a| a.min(2) as u8);
    match &mut proj {
        Projection::Uv { scale: s, rotation, offset } => {
            *s = scale.unwrap_or(*s);
            *rotation = f32p(p, "rotation").map_or(*rotation, |r| r.rem_euclid(360.0));
            *offset = floats(p, "offset", -1000.0, 1000.0).unwrap_or(*offset);
        }
        Projection::Triplanar { scale: s, sharpness, offset } => {
            *s = scale.map_or(*s, |v| v[0]);
            *sharpness = f32p(p, "sharpness").map_or(*sharpness, |v| v.clamp(1.0, 64.0));
            *offset = floats(p, "offset", -1000.0, 1000.0).unwrap_or(*offset);
        }
        Projection::Planar { scale: s, axis: a, offset } => {
            *s = scale.map_or(*s, |v| v[0]);
            *a = axis.unwrap_or(*a);
            *offset = floats(p, "offset", -1000.0, 1000.0).unwrap_or(*offset);
        }
        Projection::Spherical { scale: s } => *s = scale.map_or(*s, |v| v[0]),
        Projection::Cylindrical { scale: s, axis: a } => {
            *s = scale.map_or(*s, |v| v[0]);
            *a = axis.unwrap_or(*a);
        }
        Projection::Decal(_) => {}
    }
    Ok(proj)
}

/// `key` as `N` finite numbers clamped to `lo..=hi`: an array of `N`, or one number for all.
fn floats<const N: usize>(p: &Value, key: &str, lo: f32, hi: f32) -> Option<[f32; N]> {
    let one = |x: &Value| x.as_f64().map(|f| f as f32).filter(|f| f.is_finite()).map(|f| f.clamp(lo, hi));
    match p.get(key)? {
        Value::Array(a) if a.len() == N => {
            let mut out = [0.0; N];
            for (o, x) in out.iter_mut().zip(a) {
                *o = one(x)?;
            }
            Some(out)
        }
        v => one(v).map(|f| [f; N]),
    }
}

fn fill_projection(s: &mut Session, p: &Value) -> Result<Value> {
    let target = fill_target(s, p, "layer.fill.projection")?;
    s.edit("Fill projection", |s| {
        let fill = fill_mut(s, target)?;
        fill.projection = projection_from(p, "layer.fill.projection", Some(&fill.projection))?;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn color_tag(s: &mut Session, p: &Value) -> Result<Value> {
    let id = layer_param(s, "layer.colorTag", p)?;
    let tag = u64p(p, "tag").map(|t| t.min(7) as u8);
    s.edit("Layer color tag", |s| {
        s.doc.project.layer_mut(id)?.color_tag = tag;
        Ok(())
    })?;
    Ok(Value::Null)
}

fn list(s: &mut Session, p: &Value) -> Result<Value> {
    let ts = u64p(p, "ts").map(|t| t as usize).unwrap_or(s.ts());
    Ok(crate::inspect::stack(s, ts))
}
