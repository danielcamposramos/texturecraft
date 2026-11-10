//! Content effects, HSL and anchor points through engine commands.

use serde_json::{Value, json};
use texturecraft_engine::core::{Channel, EffectKind, LayerId};
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, StackTarget, execute};

fn session() -> Session {
    let mut s = Session::new_project("t", primitives::sphere(1.0, 48, 24), "builtin:sphere", None, 256).unwrap();
    s.viewport = texturecraft_engine::geom::glam::Vec2::new(400.0, 400.0);
    s.camera.yaw = 0.0;
    s.camera.pitch = 0.0;
    s.update();
    s
}

/// Composited value of `c` under the viewport center.
fn center(s: &mut Session, c: Channel) -> [f32; 3] {
    s.update();
    let hit = s.pick(texturecraft_engine::geom::glam::Vec2::new(200.0, 200.0), s.viewport).unwrap();
    s.caches[0].composite.value(c, (hit.uv.x * 256.0) as u32, (hit.uv.y * 256.0) as u32)
}

fn close(a: [f32; 3], b: [f32; 3]) -> bool {
    (0..3).all(|i| (a[i] - b[i]).abs() < 2e-3)
}

fn id(v: Value, key: &str) -> u64 {
    v[key].as_u64().unwrap()
}

#[test]
fn content_effects_commands_inspect_and_round_trip() {
    let mut s = session();
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 0.0, 0.0], "roughness": 0.2})).unwrap(), "layer");
    let hsl = id(execute(&mut s, "effect.add.filter", &json!({"target": "content", "filter": "hsl", "hue": 1.0 / 3.0})).unwrap(), "effect");
    assert!(close(center(&mut s, Channel::BaseColor), [0.0, 1.0, 0.0]), "{:?}", center(&mut s, Channel::BaseColor));
    assert!(s.doc.project.layer(LayerId(layer)).unwrap().mask.is_none(), "content effects don't create a mask");
    assert_eq!(s.target, StackTarget::Effect(LayerId(hsl)));
    // Mask-only effects refuse the content, unknown targets are errors.
    assert!(execute(&mut s, "effect.add.generator", &json!({"target": "content"})).is_err());
    assert!(execute(&mut s, "effect.add.filter", &json!({"target": "nope"})).is_err());
    execute(&mut s, "effect.set", &json!({"effect": hsl, "hue": 0.0, "lightness": -1.0})).unwrap();
    assert!(close(center(&mut s, Channel::BaseColor), [0.0; 3]));
    execute(&mut s, "effect.set", &json!({"effect": hsl, "visible": false})).unwrap();

    // A content fill effect, edited through layer.fill.set.
    let fill = id(execute(&mut s, "effect.add.fill", &json!({"target": "content", "roughness": 0.7})).unwrap(), "effect");
    assert!(close(center(&mut s, Channel::Roughness), [0.7; 3]));
    assert!(close(center(&mut s, Channel::BaseColor), [1.0, 0.0, 0.0]), "channels without a source stay");
    execute(&mut s, "layer.fill.set", &json!({"effect": fill, "channel": "roughness", "value": 0.9})).unwrap();
    assert!(close(center(&mut s, Channel::Roughness), [0.9; 3]));
    execute(&mut s, "layer.fill.set", &json!({"effect": fill, "channel": "roughness", "enabled": false})).unwrap();
    assert!(close(center(&mut s, Channel::Roughness), [0.2; 3]));
    assert!(execute(&mut s, "layer.fill.set", &json!({"effect": hsl, "channel": "roughness", "value": 0.5})).is_err(), "not a fill effect");

    // Inspect shows the content stack; the project survives a save/load round trip.
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap();
    let row = inspect["layers"].as_array().unwrap().iter().find(|l| l["id"] == layer).unwrap().clone();
    let kinds: Vec<&str> = row["contentEffects"].as_array().unwrap().iter().map(|e| e["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["HSL Perceptive", "Fill"], "{row}");
    let back = Session::from_project_bytes(&s.to_project_bytes().unwrap()).unwrap();
    assert_eq!(back.doc.project, s.doc.project);

    execute(&mut s, "layer.select", &json!({"layer": layer, "target": fill})).unwrap();
    execute(&mut s, "effect.remove", &json!({"effect": fill})).unwrap();
    assert_eq!(s.target, StackTarget::Content);
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(s.doc.project.layer(LayerId(layer)).unwrap().content_effects.len(), 2);
}

#[test]
fn anchor_points_feed_masks_above() {
    let mut s = session();
    let low = id(execute(&mut s, "layer.add.fill", &json!({"height": 0.5})).unwrap(), "layer");
    let anchor = id(execute(&mut s, "effect.add.anchor", &json!({"name": "Height"})).unwrap(), "effect");
    assert_eq!(s.doc.project.effect(LayerId(anchor)).unwrap().1.name, "Height");
    execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 1.0, 1.0]})).unwrap();
    // A layer cannot read an anchor on itself or above it.
    assert!(execute(&mut s, "effect.add.fill", &json!({"layer": low, "anchor": anchor})).is_err());
    assert!(execute(&mut s, "effect.add.fill", &json!({"anchor": low})).is_err(), "not an anchor point");
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    let reader = id(execute(&mut s, "effect.add.fill", &json!({"anchor": anchor, "anchorChannel": "height"})).unwrap(), "effect");
    // Base color: the low fill's default gray, then white through a 0.5 mask.
    assert!(close(center(&mut s, Channel::BaseColor), [0.75; 3]), "{:?}", center(&mut s, Channel::BaseColor));
    // Reading the anchor's base color as gray instead.
    execute(&mut s, "effect.set", &json!({"effect": reader, "anchorChannel": "baseColor"})).unwrap();
    assert!(close(center(&mut s, Channel::BaseColor), [0.75; 3]));
    execute(&mut s, "effect.set", &json!({"effect": reader, "value": 1.0})).unwrap();
    assert!(close(center(&mut s, Channel::BaseColor), [1.0; 3]), "back to a uniform fill");
    // A blurred mask still reads the anchor (a flat height stays flat).
    execute(&mut s, "effect.set", &json!({"effect": reader, "anchor": anchor})).unwrap();
    execute(&mut s, "effect.add.filter", &json!({"filter": "blur", "radius": 8})).unwrap();
    assert!(close(center(&mut s, Channel::BaseColor), [0.75; 3]));
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap();
    assert!(inspect.to_string().contains(&format!("\"anchor\":{anchor}")), "{inspect}");
}

#[test]
fn anchor_points_in_masks_record_the_mask_for_layers_above() {
    let mut s = session();
    let low = id(execute(&mut s, "layer.add.fill", &json!({"color": [0.0, 0.0, 0.0]})).unwrap(), "layer");
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    execute(&mut s, "effect.add.fill", &json!({"value": 0.4})).unwrap();
    let anchor = id(execute(&mut s, "effect.add.anchor", &json!({"target": "mask", "name": "Wear"})).unwrap(), "effect");
    let layer = s.doc.project.layer(LayerId(low)).unwrap();
    assert!(layer.mask.as_ref().unwrap().effects.iter().any(|e| e.id.0 == anchor), "in the mask");
    assert!(layer.content_effects.is_empty(), "not on the content");
    // Without a target (or with no params at all) it still goes on the content, as before.
    let content = id(execute(&mut s, "effect.add.anchor", &json!({"layer": low})).unwrap(), "effect");
    assert!(s.doc.project.layer(LayerId(low)).unwrap().content_effects.iter().any(|e| e.id.0 == content));
    let bare = id(execute(&mut s, "effect.add.anchor", &Value::Null).unwrap(), "effect");
    assert!(s.doc.project.layer(LayerId(low)).unwrap().content_effects.iter().any(|e| e.id.0 == bare));

    // Folders take anchors in their masks only.
    let folder = id(execute(&mut s, "layer.add.group", &Value::Null).unwrap(), "layer");
    assert!(execute(&mut s, "effect.add.anchor", &json!({"layer": folder})).is_err(), "folder content");
    execute(&mut s, "effect.add.anchor", &json!({"layer": folder, "target": "mask"})).unwrap();

    execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 1.0, 1.0]})).unwrap();
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    execute(&mut s, "effect.add.fill", &json!({"anchor": anchor})).unwrap();
    // The low layer shows at 0.4 (black over the 0.5 default), the top one at the anchor's 0.4.
    let low_shown = 0.5 * 0.6;
    let expected = low_shown + (1.0 - low_shown) * 0.4;
    assert!(close(center(&mut s, Channel::BaseColor), [expected; 3]), "{:?}", center(&mut s, Channel::BaseColor));
}

#[test]
fn generators_read_an_anchor_height_as_micro_detail() {
    let mut s = session();
    let low = id(execute(&mut s, "layer.add.fill", &json!({"height": 0.5})).unwrap(), "layer");
    let anchor = id(execute(&mut s, "effect.add.anchor", &Value::Null).unwrap(), "effect");
    let top = id(execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 1.0, 1.0]})).unwrap(), "layer");

    // The anchor must be below; `microAmount` needs an anchor; ids must be numbers.
    assert!(execute(&mut s, "effect.add.generator", &json!({"layer": low, "microAnchor": anchor})).is_err());
    assert!(execute(&mut s, "effect.add.generator", &json!({"microAmount": 0.5})).is_err());
    assert!(execute(&mut s, "effect.add.generator", &json!({"microAnchor": "x"})).is_err());
    assert!(execute(&mut s, "effect.add.generator", &json!({"microAnchor": low})).is_err(), "not an anchor point");

    let generator = id(execute(&mut s, "effect.add.generator", &json!({"generator": "curvature", "noise": 0.0})).unwrap(), "effect");
    let plain = center(&mut s, Channel::BaseColor);
    execute(&mut s, "effect.set", &json!({"effect": generator, "microAnchor": anchor, "microAmount": 1.0})).unwrap();
    let raised = center(&mut s, Channel::BaseColor);
    assert!(raised[0] > plain[0] + 0.1, "{plain:?} -> {raised:?}");
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap();
    assert!(inspect.to_string().contains(&format!("\"micro\":{{\"amount\":1.0,\"anchor\":{anchor}}}")), "{inspect}");

    // The amount alone keeps the anchor; null removes the micro detail.
    execute(&mut s, "effect.set", &json!({"effect": generator, "microAmount": 0.0})).unwrap();
    assert!(close(center(&mut s, Channel::BaseColor), plain));
    execute(&mut s, "effect.set", &json!({"effect": generator, "microAnchor": null})).unwrap();
    assert!(s.doc.project.effect(LayerId(generator)).map(|(_, e)| matches!(&e.kind, EffectKind::Generator(g) if g.micro.is_none())).unwrap());
    assert!(execute(&mut s, "effect.set", &json!({"effect": anchor, "microAmount": 0.5})).is_err(), "generators only");

    // A copied generator with micro detail needs the anchor below its new layer.
    execute(&mut s, "effect.set", &json!({"effect": generator, "microAnchor": anchor})).unwrap();
    execute(&mut s, "layer.select", &json!({"layer": top, "target": generator})).unwrap();
    execute(&mut s, "effect.copy", &json!({"effect": generator})).unwrap();
    execute(&mut s, "layer.select", &json!({"layer": low})).unwrap();
    assert!(execute(&mut s, "effect.paste", &json!({"target": "mask"})).is_err());
    let back = Session::from_project_bytes(&s.to_project_bytes().unwrap()).unwrap();
    assert_eq!(back.doc.project, s.doc.project);
}

#[test]
fn painting_into_a_content_paint_effect() {
    let mut s = session();
    execute(&mut s, "layer.add.fill", &json!({"roughness": 0.3})).unwrap();
    let e = id(execute(&mut s, "effect.add.paint", &json!({"target": "content"})).unwrap(), "effect");
    execute(&mut s, "tool.material", &json!({"channel": "roughness", "value": 0.9})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 20, "hardness": 90})).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[180, 200], [220, 200]]})).unwrap();
    assert!(s.doc.pixels.layer(LayerId(e), Channel::Roughness).is_some());
    assert!(center(&mut s, Channel::Roughness)[0] > 0.85, "{:?}", center(&mut s, Channel::Roughness));
    // Flattening bakes the effect into a paint layer and drops its pixels.
    execute(&mut s, "layer.flatten", &Value::Null).unwrap();
    assert!(s.doc.pixels.layer(LayerId(e), Channel::Roughness).is_none());
    assert!(center(&mut s, Channel::Roughness)[0] > 0.85);
}

#[test]
fn effects_move_within_their_stack_and_duplicate_with_their_pixels() {
    let mut s = session();
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"roughness": 0.3})).unwrap(), "layer");
    let paint = id(execute(&mut s, "effect.add.paint", &json!({"target": "content"})).unwrap(), "effect");
    execute(&mut s, "tool.material", &json!({"channel": "roughness", "value": 0.9})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 20, "hardness": 90})).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[180, 200], [220, 200]]})).unwrap();
    let fill = id(execute(&mut s, "effect.add.fill", &json!({"target": "content", "roughness": 0.2})).unwrap(), "effect");
    let order = |s: &Session| s.doc.project.layer(LayerId(layer)).unwrap().content_effects.iter().map(|e| e.id.0).collect::<Vec<_>>();
    assert!(close(center(&mut s, Channel::Roughness), [0.2; 3]), "the fill covers the paint");
    // To the bottom, the paint shows again; an index past the end means the top.
    execute(&mut s, "effect.move", &json!({"effect": fill, "index": 0})).unwrap();
    assert_eq!(order(&s), [fill, paint]);
    assert!(center(&mut s, Channel::Roughness)[0] > 0.85, "{:?}", center(&mut s, Channel::Roughness));
    execute(&mut s, "effect.move", &json!({"effect": fill, "index": 99})).unwrap();
    assert_eq!(order(&s), [paint, fill]);
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(order(&s), [fill, paint]);
    assert!(execute(&mut s, "effect.move", &json!({"effect": fill})).is_err(), "index is required");
    assert!(execute(&mut s, "effect.move", &json!({"effect": 999_999, "index": 0})).is_err());
    // A duplicate sits just above its original with the painted pixels, and takes the selection.
    execute(&mut s, "layer.select", &json!({"layer": layer, "target": paint})).unwrap();
    let copy = id(execute(&mut s, "effect.duplicate", &json!({"effect": paint})).unwrap(), "effect");
    assert_eq!(order(&s), [fill, paint, copy]);
    assert_eq!(s.target, StackTarget::Effect(LayerId(copy)));
    assert!(s.doc.project.effect(LayerId(copy)).unwrap().1.name.ends_with(" copy"));
    // Undoing it leaves no missing effect selected, but the stack it was in.
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(s.target, StackTarget::Content);
    execute(&mut s, "edit.redo", &Value::Null).unwrap();
    execute(&mut s, "effect.remove", &json!({"effect": paint})).unwrap();
    assert!(center(&mut s, Channel::Roughness)[0] > 0.85, "the copy keeps the strokes");
    assert!(execute(&mut s, "effect.duplicate", &json!({"effect": 999_999})).is_err());
    // Mask effects move within the mask.
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    let a = id(execute(&mut s, "effect.add.fill", &json!({"target": "mask", "value": 0.3})).unwrap(), "effect");
    let b = id(execute(&mut s, "effect.add.levels", &json!({"target": "mask"})).unwrap(), "effect");
    execute(&mut s, "effect.move", &json!({"effect": b, "index": 0})).unwrap();
    let mask: Vec<u64> = s.doc.project.layer(LayerId(layer)).unwrap().mask.as_ref().unwrap().effects.iter().map(|e| e.id.0).collect();
    assert_eq!(mask, [b, a]);
}

#[test]
fn effects_copy_and_paste_between_layers_and_stacks() {
    let mut s = session();
    let low = id(execute(&mut s, "layer.add.fill", &json!({"roughness": 0.3})).unwrap(), "layer");
    let paint = id(execute(&mut s, "effect.add.paint", &json!({"target": "content"})).unwrap(), "effect");
    execute(&mut s, "tool.material", &json!({"channel": "roughness", "value": 0.9})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 20, "hardness": 90})).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[180, 200], [220, 200]]})).unwrap();
    assert_eq!(execute(&mut s, "effect.copy", &Value::Null).unwrap(), json!({"copied": paint}), "the selected effect");
    assert!(execute(&mut s, "layer.paste", &Value::Null).is_err(), "the clipboard holds an effect, not a layer");
    // Onto the content of a fill above that covers the first: its strokes come along.
    let high = id(execute(&mut s, "layer.add.fill", &json!({"roughness": 0.3})).unwrap(), "layer");
    let pasted = id(execute(&mut s, "effect.paste", &Value::Null).unwrap(), "effect");
    assert_eq!(s.target, StackTarget::Effect(LayerId(pasted)));
    assert!(s.doc.pixels.layer(LayerId(pasted), Channel::Roughness).is_some());
    execute(&mut s, "effect.remove", &json!({"effect": paint})).unwrap();
    assert!(center(&mut s, Channel::Roughness)[0] > 0.85, "{:?}", center(&mut s, Channel::Roughness));
    // Above the selected effect.
    let inv = id(execute(&mut s, "effect.add.filter", &json!({"target": "content", "filter": "invert"})).unwrap(), "effect");
    execute(&mut s, "effect.copy", &json!({"effect": inv})).unwrap();
    execute(&mut s, "layer.select", &json!({"layer": high, "target": pasted})).unwrap();
    let again = id(execute(&mut s, "effect.paste", &Value::Null).unwrap(), "effect");
    let order: Vec<u64> = s.doc.project.layer(LayerId(high)).unwrap().content_effects.iter().map(|e| e.id.0).collect();
    assert_eq!(order, [pasted, again, inv]);
    // A generator goes on a mask (made for it) even with the content selected; asking for the
    // content is an error. Cut removes the original.
    execute(&mut s, "layer.select", &json!({"layer": high})).unwrap();
    let generator = id(execute(&mut s, "effect.add.generator", &Value::Null).unwrap(), "effect");
    execute(&mut s, "effect.cut", &json!({"effect": generator})).unwrap();
    assert!(s.doc.project.effect(LayerId(generator)).is_none());
    execute(&mut s, "layer.select", &json!({"layer": low})).unwrap();
    assert!(execute(&mut s, "effect.paste", &json!({"target": "content"})).is_err());
    execute(&mut s, "effect.paste", &Value::Null).unwrap();
    let mask = &s.doc.project.layer(LayerId(low)).unwrap().mask;
    assert!(mask.as_ref().is_some_and(|m| m.effects.iter().any(|e| matches!(e.kind, EffectKind::Generator(_)))));
    // A fill reading an anchor point needs one below its new layer.
    execute(&mut s, "layer.select", &json!({"layer": low})).unwrap();
    let anchor = id(execute(&mut s, "effect.add.anchor", &Value::Null).unwrap(), "effect");
    execute(&mut s, "layer.select", &json!({"layer": high})).unwrap();
    let reader = id(execute(&mut s, "effect.add.fill", &json!({"target": "mask", "anchor": anchor})).unwrap(), "effect");
    execute(&mut s, "effect.copy", &json!({"effect": reader})).unwrap();
    execute(&mut s, "layer.select", &json!({"layer": low})).unwrap();
    assert!(execute(&mut s, "effect.paste", &Value::Null).is_err(), "the anchor is on the layer itself");
}

#[test]
fn pasting_an_effect_into_another_texture_set_keeps_the_active_target() {
    let mut s = texturecraft_engine::sample_session(256, 64).unwrap();
    execute(&mut s, "layer.add.fill", &Value::Null).unwrap();
    execute(&mut s, "effect.add.filter", &json!({"target": "content", "filter": "invert"})).unwrap();
    execute(&mut s, "effect.copy", &Value::Null).unwrap();
    execute(&mut s, "textureSet.select", &json!({"ts": 1})).unwrap();
    let other = id(execute(&mut s, "layer.add.fill", &Value::Null).unwrap(), "layer");
    execute(&mut s, "textureSet.select", &json!({"ts": 0})).unwrap();
    let before = s.target;
    let pasted = id(execute(&mut s, "effect.paste", &json!({"layer": other})).unwrap(), "effect");
    assert_eq!(s.target, before, "the active set's selection keeps its target");
    assert_eq!(s.doc.project.effect(LayerId(pasted)).map(|(owner, _)| owner), Some(LayerId(other)));
}

#[test]
fn histogram_counts_what_an_effect_reads() {
    let mut s = session();
    let hist = |s: &mut Session, p: Value| {
        let r = execute(s, "effect.histogram", &p).unwrap();
        let bins: Vec<u64> = r["bins"].as_array().unwrap().iter().map(|b| b.as_u64().unwrap()).collect();
        (r["target"].as_str().unwrap().to_string(), bins)
    };
    // Only the bucket holding the value is filled.
    let one = |bins: &[u64], k: usize| bins.iter().enumerate().all(|(i, b)| (*b > 0) == (i == k));
    execute(&mut s, "layer.add.fill", &json!({"color": [0.25, 0.25, 0.25], "roughness": 0.75})).unwrap();
    let lv = id(execute(&mut s, "effect.add.levels", &json!({"target": "content", "inLow": 0.5})).unwrap(), "effect");
    execute(&mut s, "effect.add.filter", &json!({"target": "content", "filter": "invert"})).unwrap();
    let (target, bins) = hist(&mut s, json!({"effect": lv, "channel": "roughness", "bins": 4}));
    assert_eq!(target, "content");
    assert!(one(&bins, 3), "roughness 0.75: {bins:?}");
    // The effect itself and those above it are left out; the default channel is the first.
    let (_, bins) = hist(&mut s, json!({"effect": lv, "bins": 4}));
    assert!(one(&bins, 1), "base color 0.25, components pooled: {bins:?}");
    assert_eq!(bins.iter().sum::<u64>() % 3, 0);
    // A mask's levels reads the mask below it.
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    execute(&mut s, "effect.add.fill", &json!({"target": "mask", "value": 0.3})).unwrap();
    let ml = id(execute(&mut s, "effect.add.levels", &json!({"target": "mask"})).unwrap(), "effect");
    let (target, bins) = hist(&mut s, json!({"effect": ml, "bins": 4}));
    assert_eq!(target, "mask");
    assert!(one(&bins, 1), "mask 0.3: {bins:?}");
    assert_eq!(hist(&mut s, json!({"effect": ml})).1.len(), 64);
    // Below a blur, every tile builds its own neighbourhood field; the texels counted stay the same.
    let texels: u64 = bins.iter().sum();
    execute(&mut s, "effect.remove", &json!({"effect": ml})).unwrap();
    execute(&mut s, "effect.add.filter", &json!({"target": "mask", "filter": "blur", "radius": 4})).unwrap();
    let ml = id(execute(&mut s, "effect.add.levels", &json!({"target": "mask"})).unwrap(), "effect");
    let (_, blurred) = hist(&mut s, json!({"effect": ml, "bins": 4}));
    assert_eq!(blurred.iter().sum::<u64>(), texels);
    assert_eq!(blurred.iter().enumerate().max_by_key(|(_, b)| **b).map(|(i, _)| i), Some(1), "{blurred:?}");
    // A paint layer with nothing painted yet still has its mask.
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    execute(&mut s, "mask.add.black", &Value::Null).unwrap();
    execute(&mut s, "effect.add.fill", &json!({"target": "mask", "value": 0.6})).unwrap();
    let pl = id(execute(&mut s, "effect.add.levels", &json!({"target": "mask"})).unwrap(), "effect");
    assert!(one(&hist(&mut s, json!({"effect": pl, "bins": 4})).1, 2), "an empty paint layer's mask");
    assert!(execute(&mut s, "effect.histogram", &json!({"effect": 999_999})).is_err());
    assert!(execute(&mut s, "effect.histogram", &json!({"effect": lv, "channel": "nope"})).is_err());
}

/// `merge: true` on any command folds its edit into the previous step of the same label (a
/// slider drag applies a value per frame and is one undo step); the next command starts afresh.
#[test]
fn merged_edits_are_one_undo_step() {
    let mut s = session();
    execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 0.0, 0.0]})).unwrap();
    let hsl = LayerId(id(execute(&mut s, "effect.add.filter", &json!({"target": "content", "filter": "hsl"})).unwrap(), "effect"));
    let opacity = |s: &Session| s.doc.project.effect(hsl).unwrap().1.opacity;
    let depth = s.history_labels().len();
    for (i, op) in [0.8, 0.6, 0.4].into_iter().enumerate() {
        execute(&mut s, "effect.set", &json!({"effect": hsl.0, "opacity": op, "merge": i > 0})).unwrap();
    }
    assert_eq!(s.history_labels().len(), depth + 1, "one step for the drag");
    execute(&mut s, "effect.set", &json!({"effect": hsl.0, "opacity": 0.2})).unwrap();
    assert_eq!(s.history_labels().len(), depth + 2, "a later edit without merge is its own step");
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(opacity(&s), 0.4);
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(opacity(&s), 1.0, "one undo reverts the whole drag");
}
