//! JSON views of the session for agents: everything an agent changed must be visible here
//! without a screenshot.

use serde_json::{Value, json};
use texturecraft_core::{Effect, EffectKind, Fill, Filter, Generator, Layer, LayerKind, NormalFormat, Projection, Source};

use crate::Session;
use crate::session::StackTarget;

fn source_json(s: &Source) -> Value {
    match s {
        Source::Uniform(v) => json!({ "uniform": v }),
        Source::Procedural { pattern, low, high } => json!({
            "procedural": pattern.kind.label(), "scale": pattern.scale, "seed": pattern.seed, "contrast": pattern.contrast, "low": low, "high": high
        }),
        Source::Image { name, normal_format: NormalFormat::OpenGl } => json!({ "image": name }),
        Source::Image { name, normal_format: NormalFormat::DirectX } => json!({ "image": name, "normalFormat": "directX" }),
        Source::Anchor { anchor, channel } => json!({ "anchor": anchor.0, "channel": channel.id() }),
    }
}

fn fill_json(f: &Fill) -> Value {
    let mut v = json!({
        "projection": f.projection.label(),
        "sources": f.sources.iter().map(|(c, s)| (c.id().to_string(), source_json(s))).collect::<serde_json::Map<_, _>>(),
    });
    if let Projection::Decal(d) = &f.projection {
        v["decal"] = json!(d);
    }
    if let Some(link) = &f.link {
        v["link"] = json!(link);
    }
    v
}

/// A mask or content (`content`) effect.
fn effect_json(session: &Session, effect: &Effect, content: bool) -> Value {
    let mut out = json!({ "id": effect.id.0, "name": effect.name, "kind": effect.kind.name(), "visible": effect.visible, "blend": effect.blend.label(), "opacity": effect.opacity });
    let mask_tiles = || session.doc.pixels.masks.get(&effect.id).map_or(0, |image| image.allocated_count());
    let content_tiles = || session.doc.pixels.layers.iter().filter(|((id, _), _)| *id == effect.id).map(|(_, image)| image.allocated_count()).sum::<usize>();

    match &effect.kind {
        EffectKind::Generator(generator) => out["generator"] = generator_json(generator),
        EffectKind::Levels(levels) => out["levels"] = json!([levels.in_low, levels.in_high, levels.gamma, levels.out_low, levels.out_high]),
        EffectKind::ColorSelection { colors, tolerance } => {
            out["colors"] = json!(colors);
            out["tolerance"] = json!(tolerance);
        }
        EffectKind::Path(paths) => {
            out["paths"] = json!(
                paths.iter().map(|path| json!({ "points": path.points.len(), "uv": path.uv, "closed": path.closed, "fill": path.fill })).collect::<Vec<_>>()
            );
            out["tiles"] = json!(if content { content_tiles() } else { mask_tiles() });
        }
        EffectKind::Paint if content => out["tiles"] = json!(content_tiles()),
        EffectKind::Paint => out["tiles"] = json!(mask_tiles()),
        EffectKind::Fill { source, .. } => out["source"] = source_json(source),
        EffectKind::Material(fill) => out["fill"] = fill_json(fill),
        EffectKind::Filter(filter) => {
            out["filter"] = json!(filter);
            match filter {
                Filter::Hsl { hue, saturation, lightness } => out["hsl"] = json!([hue, saturation, lightness]),
                Filter::Blur { radius } => out["radius"] = json!(radius),
                _ => {}
            }
        }
        _ => {}
    }
    out
}

/// A generator's settings, with its curvature mode and detail size weights when it has them.
fn generator_json(generator: &Generator) -> Value {
    let mut out = json!({ "type": format!("{:?}", generator.kind), "level": generator.level, "softness": generator.softness, "noise": generator.noise_amount, "invert": generator.invert });
    if let Some(detail) = &generator.detail {
        out["mode"] = json!(format!("{:?}", detail.mode));
        out["sizes"] = json!(detail.weights);
    }
    if let Some(micro) = &generator.micro {
        out["micro"] = json!({ "anchor": micro.anchor.0, "amount": micro.amount });
    }
    out
}

fn layer_json(s: &Session, l: &Layer, depth: usize) -> Value {
    let channels: serde_json::Map<String, Value> =
        l.channels.iter().map(|(c, b)| (c.id().to_string(), json!({ "mode": b.mode.label(), "opacity": b.opacity }))).collect();
    let mut v = json!({
        "id": l.id.0,
        "name": l.name,
        "kind": l.kind.name(),
        "visible": l.visible,
        "channels": channels,
        "colorTag": l.color_tag,
    });
    if let LayerKind::Fill(f) = &l.kind {
        v["fill"] = fill_json(f);
    }
    if let Some(src) = l.instance_source() {
        let source = s.doc.project.layers.get(&src);
        let set = s.doc.project.locate(src).map(|(ts, _, _)| ts);
        v["instanceOf"] = json!({ "layer": src.0, "name": source.map(|x| x.name.as_str()), "textureSet": set });
    }
    if l.height_mask.enabled {
        v["heightMask"] = json!({ "contrast": l.height_mask.contrast, "offset": l.height_mask.offset });
    }
    if let Some(parts) = &l.geometry_mask {
        v["geometryMask"] = json!(parts);
    }
    if matches!(l.kind, LayerKind::Paint) {
        // Sorted: the store is a hash map, and inspect output must be stable.
        let mut painted: Vec<(&str, usize)> =
            s.doc.pixels.layers.iter().filter(|((id, _), _)| *id == l.id).map(|((_, c), img)| (c.id(), img.allocated_count())).collect();
        painted.sort_unstable();
        v["painted"] = json!(painted.iter().map(|(c, n)| json!({ "channel": c, "tiles": n })).collect::<Vec<_>>());
    }
    if let Some(m) = &l.mask {
        v["mask"] = json!({
            "enabled": m.enabled,
            "background": format!("{:?}", m.background).to_lowercase(),
            "effects": m.effects.iter().map(|e| effect_json(s, e, false)).collect::<Vec<_>>(),
        });
    }
    if !l.content_effects.is_empty() {
        v["contentEffects"] = json!(l.content_effects.iter().map(|e| effect_json(s, e, true)).collect::<Vec<_>>());
    }
    if depth < texturecraft_core::project::MAX_DEPTH {
        let kids: Vec<Value> = l.children().iter().rev().filter_map(|c| s.doc.project.layers.get(c)).map(|c| layer_json(s, c, depth + 1)).collect();
        if l.is_group() {
            v["children"] = json!(kids);
        }
    }
    v
}

/// The layer stack of texture set `ts`, top → bottom (as displayed).
pub fn stack(s: &Session, ts: usize) -> Value {
    let Some(set) = s.doc.project.texture_sets.get(ts) else { return Value::Null };
    json!(set.stack.iter().rev().filter_map(|id| s.doc.project.layers.get(id)).map(|l| layer_json(s, l, 0)).collect::<Vec<_>>())
}

pub fn texture_sets(s: &Session) -> Value {
    json!((0..s.set_count()).map(|i| texture_set(s, i)).collect::<Vec<_>>())
}

/// One texture set, including UV tile layout and bake selection
pub fn texture_set(session: &Session, index: usize) -> Value {
    let Some(texture_set) = session.doc.project.texture_sets.get(index) else { return Value::Null };

    let grid = texture_set.tile_grid();
    let tiles: Vec<Value> = texture_set
        .uv_tiles
        .iter()
        .filter_map(|tile| {
            grid?.block(*tile).map(|(x, y)| {
                json!({"udim": tile, "block": [x, y], "resolution": texture_set.tile_resolution(*tile), "bakeEnabled": !texture_set.bake_excluded_tiles.contains(tile)})
            })
        })
        .collect();
    json!({
        "index": index,
        "name": texture_set.name,
        "resolution": texture_set.resolution,
        "size": texture_set.size(),
        "blockResolution": texture_set.block_resolution(),
        "uvTiles": texture_set.uv_tiles,
        "tiles": tiles,
        "channels": texture_set.channels.iter().map(|(channel, format)| json!({ "channel": channel.id(), "format": format.label() })).collect::<Vec<_>>(),
        "layers": session.doc.project.walk(index).len(),
        "parts": crate::set_cmds::parts(session, index),
        "baked": crate::project_cmds::baked_json(session.doc.baked.get(&index)),
        "visible": texture_set.visible,
        "active": index == session.ts(),
    })
}

/// The whole session.
pub fn session(s: &Session) -> Value {
    let ts = s.ts();
    json!({
        "project": s.doc.project.name,
        "mesh": { "source": s.doc.project.mesh, "triangles": s.mesh.triangle_count(), "vertices": s.mesh.vertex_count(), "materials": s.mesh.materials },
        "path": s.path,
        "modified": s.modified,
        "textureSets": texture_sets(s),
        "bakeMeshes": crate::bake::meshes_json(s),
        "activeTextureSet": ts,
        "selectedLayer": s.selected_layer().map(|l| l.0),
        "target": match s.target { StackTarget::Content => json!("content"), StackTarget::Mask => json!("mask"), StackTarget::Effect(e) => json!(e.0) },
        "layers": stack(s, ts),
        "tool": { "tool": s.tool.tool.id(), "brush": crate::tool_cmds::brush_json(s) },
        "view": s.view.label(),
        "bakePreview": crate::bake_preview::preview_json(s),
        "camera": crate::view_cmds::camera_json(s),
        "display": s.doc.project.display,
        "history": { "undo": s.undo_label(), "redo": s.redo_label(), "depth": s.history_labels().len() },
        "memoryBytes": s.memory().total(),
        "pendingTiles": s.pending_tiles(),
    })
}
