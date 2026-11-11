//! The filter library, geometry decals, cloning from the selected layer, placing saved smart
//! materials, anchors inside a content stack, instances across texture set resolutions, mesh
//! maps from images and fill projection transforms, through engine commands.

use serde_json::{Value, json};
use texturecraft_engine::core::{Channel, EffectKind, Filter, LayerId, LayerKind, Projection};
use texturecraft_engine::geom::NO_TRI;
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, execute};

fn session() -> Session {
    let mut s = Session::new_project("t", primitives::sphere(1.0, 48, 24), "builtin:sphere", None, 256).unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    s.camera.yaw = 0.0;
    s.camera.pitch = 0.0;
    s.update();
    s
}

fn id(v: Value, key: &str) -> LayerId {
    LayerId(v[key].as_u64().unwrap())
}

/// Composite value of `c` under a viewport point.
fn at(s: &mut Session, c: Channel, x: f32, y: f32) -> [f32; 3] {
    s.update();
    let hit = s.pick(Vec2::new(x, y), s.viewport).unwrap();
    let (w, h) = s.caches[0].size;
    s.caches[0].composite.value(c, (hit.uv.x * w as f32) as u32, (hit.uv.y * h as f32) as u32)
}

fn red(v: [f32; 3]) -> bool {
    v[0] > 0.85 && v[1] < 0.15
}

/// Distance between the view points searched for a texture set
const SEARCH_STEP: f32 = 10.0;

fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 4]) -> Vec<u8> {
    let mut out = Vec::new();
    image::RgbaImage::from_fn(w, h, |x, y| image::Rgba(f(x, y))).write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png).unwrap();
    out
}

#[test]
fn filter_library_adds_every_filter_and_edits_its_parameters() {
    let mut s = session();
    let list = execute(&mut s, "filter.list", &Value::Null).unwrap();
    let ids: Vec<&str> = list.as_array().unwrap().iter().map(|f| f["id"].as_str().unwrap()).collect();
    for want in [
        "warp",
        "blurDirectional",
        "sharpen",
        "hsl",
        "levels",
        "contrastLuminosity",
        "colorBalance",
        "gradientMap",
        "posterize",
        "clamp",
        "slopeBlur",
        "matFinishGrind",
        "highPass",
        "edgeDetect",
        "warpDirectional",
        "grayscale",
        "normalInvert",
        "bevel",
        "mosaic",
        "glow",
        "bakedLighting",
        "replaceColor",
    ] {
        assert!(ids.contains(&want), "{want} missing from {ids:?}");
    }
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"color": [0.8, 0.2, 0.1], "roughness": 0.4})).unwrap(), "layer");
    let before = at(&mut s, Channel::BaseColor, 200.0, 200.0);
    for f in list.as_array().unwrap() {
        let fid = f["id"].as_str().unwrap();
        let eid = id(execute(&mut s, "effect.add.filter", &json!({"layer": layer.0, "filter": fid, "target": "content"})).unwrap(), "effect");
        let (owner, e) = s.doc.project.effect(eid).unwrap();
        assert_eq!(owner, layer);
        assert_eq!(e.name, f["label"].as_str().unwrap(), "named after the shelf entry");
        assert!(s.doc.project.layer(layer).unwrap().content_effects.iter().any(|x| x.id == eid));
        execute(&mut s, "effect.remove", &json!({"effect": eid.0})).unwrap();
    }
    // A gradient map recolors the content; effect.set edits it and inspect shows it.
    let gm =
        id(execute(&mut s, "effect.add.filter", &json!({"filter": "gradientMap", "target": "content", "colors": [[0, 0, 1], [0, 0, 1]]})).unwrap(), "effect");
    let blue = at(&mut s, Channel::BaseColor, 200.0, 200.0);
    assert!(blue[2] > 0.95 && blue[0] < 0.05, "{before:?} -> {blue:?}");
    execute(&mut s, "effect.set", &json!({"effect": gm.0, "colors": ["#00ff00"]})).unwrap();
    assert!(at(&mut s, Channel::BaseColor, 200.0, 200.0)[1] > 0.95);
    assert!(execute(&mut s, "effect.set", &json!({"effect": gm.0, "colors": []})).is_err());
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap().to_string();
    assert!(inspect.contains("GradientMap"), "inspect shows filter parameters");
    execute(&mut s, "effect.remove", &json!({"effect": gm.0})).unwrap();
    // Posterize on the mask; its steps are clamped.
    let post = id(execute(&mut s, "effect.add.filter", &json!({"filter": "posterize", "levels": 3})).unwrap(), "effect");
    execute(&mut s, "effect.set", &json!({"effect": post.0, "levels": 500})).unwrap();
    assert!(s.doc.project.layer(layer).unwrap().mask.as_ref().is_some_and(|m| m.effects.iter().any(|e| e.id == post)));
    assert!(matches!(s.doc.project.effect(post).unwrap().1.kind, EffectKind::Filter(Filter::Posterize { levels }) if levels == 64.0));
    // Material finishes fill the content's roughness and height.
    let fin = id(execute(&mut s, "effect.add.filter", &json!({"filter": "MatFinish Hammered"})).unwrap(), "effect");
    assert!(s.doc.project.layer(layer).unwrap().content_effects.iter().any(|e| e.id == fin));
    assert!(matches!(&s.doc.project.effect(fin).unwrap().1.kind, EffectKind::Material(f) if f.sources.contains_key(&Channel::Height)));
    assert!(execute(&mut s, "effect.add.filter", &json!({"filter": "nope"})).is_err());
    // A directional blur, slope blur and warp composite without trouble on a sphere.
    for (fid, extra) in [
        ("blurDirectional", json!({"radius": 40, "angle": 45})),
        ("slopeBlur", json!({"radius": 30})),
        ("warp", json!({"intensity": 50})),
        ("warpDirectional", json!({"intensity": 50, "angle": 30})),
        ("highPass", json!({"radius": 20})),
        ("edgeDetect", json!({"width": 4})),
        ("bevel", json!({"width": 20})),
        ("mosaic", json!({"size": 40})),
        ("glow", json!({"radius": 20, "amount": 2})),
    ] {
        let mut p = json!({"filter": fid, "target": "content"});
        p.as_object_mut().unwrap().extend(extra.as_object().unwrap().clone());
        execute(&mut s, "effect.add.filter", &p).unwrap();
    }
    assert!(at(&mut s, Channel::BaseColor, 200.0, 200.0).iter().all(|v| v.is_finite()));
}

/// Two OBJ objects in one texture set: a fill shows on the parts its geometry mask names.
#[test]
fn geometry_masks_limit_layers_to_mesh_parts() {
    let obj = "o Lid\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 0.5 0\nvt 0 0.5\nf 1/1 2/2 3/3\n\
               o Body\nv 1 1 0\nv 2 1 0\nv 1 2 0\nvt 0.5 0.5\nvt 1 0.5\nvt 0.5 1\nf 4/4 5/5 6/6\n";
    let mut s = Session::from_mesh_bytes("parts.obj", obj.as_bytes(), 256, "parts").unwrap();
    assert_eq!(execute(&mut s, "textureSet.parts", &Value::Null).unwrap(), json!(["Lid", "Body"]));
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"color": [1, 0, 0]})).unwrap(), "layer");
    // Texels inside each triangle (OBJ v is flipped to the top-left origin).
    let texel = |s: &mut Session, u: f32, v: f32| {
        s.update();
        let (w, h) = s.caches[0].size;
        red(s.caches[0].composite.value(Channel::BaseColor, (u * w as f32) as u32, (v * h as f32) as u32))
    };
    assert!(texel(&mut s, 0.1, 0.9) && texel(&mut s, 0.6, 0.4));
    let r = execute(&mut s, "layer.set", &json!({"layer": layer.0, "geometryMask": ["Body"]})).unwrap();
    assert_eq!(r["geometryMask"], json!(["Body"]));
    assert!(!texel(&mut s, 0.1, 0.9) && texel(&mut s, 0.6, 0.4), "only on Body");
    assert!(execute(&mut s, "project.inspect", &Value::Null).unwrap().to_string().contains("geometryMask"));
    assert!(execute(&mut s, "layer.set", &json!({"layer": layer.0, "geometryMask": ["Nope"]})).is_err());
    assert!(execute(&mut s, "layer.set", &json!({"layer": layer.0, "geometryMask": "Body"})).is_err());
    assert!(execute(&mut s, "layer.set", &json!({"layer": layer.0})).is_err());
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(texel(&mut s, 0.1, 0.9), "undone");
    execute(&mut s, "layer.set", &json!({"layer": layer.0, "geometryMask": []})).unwrap();
    assert!(!texel(&mut s, 0.1, 0.9) && !texel(&mut s, 0.6, 0.4), "no parts: hidden");
    execute(&mut s, "layer.set", &json!({"layer": layer.0, "geometryMask": null})).unwrap();
    assert!(texel(&mut s, 0.1, 0.9) && texel(&mut s, 0.6, 0.4));
}

#[test]
fn solo_keeps_one_material_channel() {
    let mut s = session();
    let r = execute(&mut s, "tool.material", &json!({"channel": "roughness", "solo": true})).unwrap();
    assert_eq!(r["enabled"], json!(["roughness"]), "the brush paints roughness only");
    let fill = id(execute(&mut s, "layer.add.fill", &Value::Null).unwrap(), "layer");
    assert!(s.doc.project.layer(fill).unwrap().channels.len() > 1);
    execute(&mut s, "layer.fill.set", &json!({"channel": "metallic", "solo": true})).unwrap();
    assert_eq!(s.doc.project.layer(fill).unwrap().channels.keys().copied().collect::<Vec<_>>(), [Channel::Metallic]);
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(s.doc.project.layer(fill).unwrap().channels.len() > 1, "one undo step");
}

#[test]
fn decals_project_an_image_from_a_frame_placed_on_the_surface() {
    let mut s = session();
    // A red disc on a transparent background.
    let name = s
        .import_resource_bytes("logo.png", png(32, 32, |x, y| if (x as i32 - 16).pow(2) + (y as i32 - 16).pow(2) < 196 { [255, 0, 0, 255] } else { [0; 4] }))
        .unwrap();
    assert!(execute(&mut s, "layer.add.decal", &json!({"image": "missing.png"})).is_err());
    assert!(execute(&mut s, "layer.add.decal", &json!({"image": name, "x": 2, "y": 2})).is_err(), "off the mesh");
    let r = execute(&mut s, "layer.add.decal", &json!({"image": name, "x": 200, "y": 200, "size": 0.5})).unwrap();
    let layer = id(r.clone(), "layer");
    assert!((r["decal"]["size"][1].as_f64().unwrap() - 0.5).abs() < 1e-6);
    let LayerKind::Fill(f) = &s.doc.project.layer(layer).unwrap().kind else { panic!("a fill layer") };
    assert!(matches!(f.projection, Projection::Decal(_)));
    assert!(red(at(&mut s, Channel::BaseColor, 200.0, 200.0)), "the decal shows where it was placed");
    assert!(!red(at(&mut s, Channel::BaseColor, 120.0, 200.0)), "and only there");
    // The image's transparent corners leave the surface as it was.
    assert!(!red(at(&mut s, Channel::BaseColor, 200.0 + 30.0, 200.0 - 30.0)));
    // Bigger, then moved along its plane, then re-placed with a click.
    execute(&mut s, "layer.decal.set", &json!({"size": 1.2})).unwrap();
    assert!(red(at(&mut s, Channel::BaseColor, 160.0, 200.0)));
    execute(&mut s, "layer.decal.set", &json!({"size": 0.4, "offset": [0.6, 0.0], "rotation": 45})).unwrap();
    assert!(!red(at(&mut s, Channel::BaseColor, 200.0, 200.0)), "moved away");
    let r = execute(&mut s, "layer.decal.set", &json!({"x": 170, "y": 180})).unwrap();
    assert_eq!(r["decal"]["offset"], json!([0.0, 0.0]), "re-placing resets the offset");
    assert!(red(at(&mut s, Channel::BaseColor, 170.0, 180.0)));
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap().to_string();
    assert!(inspect.contains("Decal projection") && inspect.contains("\"rotation\":45"), "inspect shows the decal");
    // Not a decal; one undo step per edit.
    let plain = id(execute(&mut s, "layer.add.fill", &Value::Null).unwrap(), "layer");
    assert!(execute(&mut s, "layer.decal.set", &json!({"layer": plain.0, "size": 1})).is_err());
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(!red(at(&mut s, Channel::BaseColor, 170.0, 180.0)));
}

#[test]
fn a_decal_takes_its_up_from_the_params_or_a_rule_that_never_degenerates() {
    let mut s = session();
    let name = s.import_resource_bytes("logo.png", png(8, 4, |x, _| if x < 4 { [255, 0, 0, 255] } else { [0, 0, 255, 255] })).unwrap();
    let top = json!({"image": name, "position": [0.0, 1.0, 0.0], "normal": [0.0, 1.0, 0.0]});
    let up = |r: &Value| [0, 1, 2].map(|i| r["decal"]["up"][i].as_f64().unwrap() as f32);
    let mut with_up = top.clone();
    with_up["up"] = json!([0.0, 0.0, -1.0]);
    assert_eq!(up(&execute(&mut s, "layer.add.decal", &with_up).unwrap()), [0.0, 0.0, -1.0]);
    with_up["up"] = json!([0.0, 2.0, 0.0]);
    assert!(execute(&mut s, "layer.add.decal", &with_up).is_err(), "an up along the normal is refused");
    // A level camera's up is the top face's normal: the decal's up is where the camera looks.
    execute(&mut s, "camera.set", &json!({"yaw": 90.0, "pitch": 0.0})).unwrap();
    let forward = s.camera.view().inverse().transform_vector3(texturecraft_engine::geom::glam::Vec3::NEG_Z);
    let u = up(&execute(&mut s, "layer.add.decal", &top).unwrap());
    assert!((texturecraft_engine::geom::glam::Vec3::from(u) - forward).length() < 1e-4, "{u:?} vs {forward}");
    let r = execute(&mut s, "layer.decal.set", &json!({"up": [1.0, 0.0, 0.0]})).unwrap();
    assert_eq!(up(&r), [1.0, 0.0, 0.0]);
}

#[test]
fn clone_copies_the_selected_layer_alone_when_asked() {
    let mut s = session();
    execute(&mut s, "tool.paint", &Value::Null).unwrap();
    execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 20, "hardness": 90})).unwrap();
    let low = id(execute(&mut s, "layer.add.paint", &Value::Null).unwrap(), "layer");
    execute(&mut s, "tool.stroke", &json!({"points": [[160, 200]]})).unwrap();
    let top = id(execute(&mut s, "layer.add.paint", &Value::Null).unwrap(), "layer");
    execute(&mut s, "tool.clone", &Value::Null).unwrap();
    assert!(execute(&mut s, "tool.brush", &json!({"cloneSource": "nope"})).is_err());
    let r = execute(&mut s, "tool.brush", &json!({"size": 6, "cloneSource": "layer"})).unwrap();
    assert_eq!(r["cloneSource"], "layer");
    execute(&mut s, "tool.cloneSource", &json!({"x": 160, "y": 200})).unwrap();
    // The empty top layer has nothing to copy.
    execute(&mut s, "tool.stroke", &json!({"points": [[240, 200]]})).unwrap();
    assert!(!red(at(&mut s, Channel::BaseColor, 240.0, 200.0)));
    assert!(s.doc.pixels.layer(top, Channel::BaseColor).is_none_or(|i| i.allocated_count() == 0));
    // From the composite it copies what shows.
    execute(&mut s, "tool.brush", &json!({"cloneSource": "composite"})).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[240, 200]]})).unwrap();
    assert!(red(at(&mut s, Channel::BaseColor, 240.0, 200.0)));
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    // The painted layer itself copies its paint.
    execute(&mut s, "layer.select", &json!({"layer": low.0})).unwrap();
    execute(&mut s, "tool.brush", &json!({"cloneSource": "layer"})).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[240, 200]]})).unwrap();
    assert!(red(at(&mut s, Channel::BaseColor, 240.0, 200.0)));
    assert_eq!(s.tool.clone_from, texturecraft_engine::CloneFrom::Layer);
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn saved_smart_materials_load_where_they_are_dropped() {
    let mut s = session();
    let a = id(execute(&mut s, "layer.add.fill", &json!({"roughness": 0.9})).unwrap(), "layer");
    let b = id(execute(&mut s, "layer.add.paint", &Value::Null).unwrap(), "layer");
    let dir = std::env::temp_dir().join(format!("tc_gaps_{}", std::process::id()));
    let path = dir.join("rough.tcmat").display().to_string();
    execute(&mut s, "material.save", &json!({"layer": a.0, "path": path})).unwrap();
    let stack = |s: &Session| s.doc.project.texture_sets[0].stack.clone();
    let bottom = id(execute(&mut s, "material.load", &json!({"path": path, "index": 0})).unwrap(), "layer");
    assert_eq!(stack(&s).first(), Some(&bottom));
    let above = id(execute(&mut s, "material.load", &json!({"path": path, "above": a.0})).unwrap(), "layer");
    let st = stack(&s);
    assert_eq!(st.iter().position(|x| *x == above), st.iter().position(|x| *x == a).map(|i| i + 1));
    assert_eq!(st.last(), Some(&b));
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(!stack(&s).contains(&above), "one undo step");
    std::fs::remove_dir_all(dir).ok();
}

#[test]
fn anchors_inside_a_content_stack_read_the_content_there() {
    let mut s = session();
    let low = id(execute(&mut s, "layer.add.fill", &json!({"height": 0.4})).unwrap(), "layer");
    let anchor = id(execute(&mut s, "effect.add.anchor", &json!({"layer": low.0})).unwrap(), "effect");
    execute(&mut s, "effect.add.fill", &json!({"layer": low.0, "target": "content", "height": 0.8})).unwrap();
    let reader = id(execute(&mut s, "layer.add.fill", &Value::Null).unwrap(), "layer");
    execute(&mut s, "layer.blend", &json!({"layer": reader.0, "channel": "all", "enabled": false})).unwrap();
    execute(&mut s, "layer.fill.set", &json!({"layer": reader.0, "channel": "roughness", "anchor": anchor.0, "anchorChannel": "height"})).unwrap();
    assert!((at(&mut s, Channel::Roughness, 200.0, 200.0)[0] - 0.4).abs() < 2e-3, "the anchor reads the content before the fill above it");
    assert!((at(&mut s, Channel::Height, 200.0, 200.0)[0] - 0.8).abs() < 2e-3);
    // Moved to the top of the content stack, it records the layer's full result again.
    execute(&mut s, "effect.remove", &json!({"effect": anchor.0})).unwrap();
    let end = id(execute(&mut s, "effect.add.anchor", &json!({"layer": low.0})).unwrap(), "effect");
    execute(&mut s, "layer.fill.set", &json!({"layer": reader.0, "channel": "roughness", "anchor": end.0, "anchorChannel": "height"})).unwrap();
    assert!((at(&mut s, Channel::Roughness, 200.0, 200.0)[0] - 0.8).abs() < 2e-3);
}

#[test]
fn instances_show_their_source_in_sets_of_another_resolution() {
    let mut s = Session::new_project("t", primitives::material_ball(), "builtin:material_ball", None, 256).unwrap();
    s.update();
    let src = id(execute(&mut s, "layer.add.paint", &Value::Null).unwrap(), "layer");
    execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 100, "hardness": 100})).unwrap();
    let pts: Vec<[f32; 2]> = (0..=20).flat_map(|i| (0..=20).map(move |j| [i as f32 / 20.0, j as f32 / 20.0])).collect();
    execute(&mut s, "tool.strokeUv", &json!({"points": pts})).unwrap();
    execute(&mut s, "layer.instance", &json!({"layer": src.0, "ts": 1})).unwrap();
    let reds = |s: &mut Session| {
        s.update();
        let c = &s.caches[1];
        let (w, h) = c.size;
        (0..h)
            .flat_map(|y| (0..w).map(move |x| (x, y)))
            .filter(|&(x, y)| c.gbuf.tri[(y * w + x) as usize] != NO_TRI && red(c.composite.value(Channel::BaseColor, x, y)))
            .count()
    };
    let same = reds(&mut s);
    assert!(same > 1000, "{same}");
    execute(&mut s, "textureSet.resolution", &json!({"ts": 1, "resolution": 128})).unwrap();
    let half = reds(&mut s);
    assert!(half * 5 > same && half * 3 < same, "about a quarter of the texels: {same} -> {half}");
}

/// A decal dragged in the viewport (one update per frame with `merge`) is one undo step that
/// returns it to where the drag started.
#[test]
fn decal_drags_merge_into_one_undo_step() {
    let mut s = session();
    let name = s.import_resource_bytes("logo.png", png(8, 8, |_, _| [255, 0, 0, 255])).unwrap();
    let layer = id(execute(&mut s, "layer.add.decal", &json!({"image": name, "x": 200, "y": 200, "size": 0.3})).unwrap(), "layer");
    let depth = s.history_labels().len();
    let start = execute(&mut s, "project.inspect", &Value::Null).unwrap()["layers"].clone();
    let drag =
        [json!({"x": 190, "y": 200}), json!({"x": 180, "y": 195, "merge": true}), json!({"size": 0.5, "merge": true}), json!({"rotation": 30, "merge": true})];
    for p in drag {
        let mut p = p;
        p["layer"] = json!(layer.0);
        execute(&mut s, "layer.decal.set", &p).unwrap();
    }
    assert_eq!(s.history_labels().len(), depth + 1, "one step for the whole drag");
    assert!(red(at(&mut s, Channel::BaseColor, 180.0, 195.0)));
    // A merge never folds into a step of another kind.
    execute(&mut s, "layer.add.fill", &Value::Null).unwrap();
    execute(&mut s, "layer.decal.set", &json!({"layer": layer.0, "size": 0.2, "merge": true})).unwrap();
    assert_eq!(s.history_labels().len(), depth + 3);
    for _ in 0..3 {
        execute(&mut s, "edit.undo", &Value::Null).unwrap();
    }
    assert_eq!(execute(&mut s, "project.inspect", &Value::Null).unwrap()["layers"], start);
}

/// A decal dropped on another texture set lands there and selects it, in one undo step
#[test]
fn a_decal_on_another_texture_set_selects_that_set() {
    let mut s = Session::new_project("t", primitives::material_ball(), "builtin:material_ball", None, 256).unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    s.update();
    let name = s.import_resource_bytes("logo.png", png(8, 8, |_, _| [255, 0, 0, 255])).unwrap();
    let selected = s.ts();
    let (columns, rows) = ((s.viewport.x / SEARCH_STEP) as u32, (s.viewport.y / SEARCH_STEP) as u32);
    let other_set_pixel = (0..rows)
        .flat_map(|row| (0..columns).map(move |column| Vec2::new(column as f32, row as f32) * SEARCH_STEP))
        .find(|&pixel| s.texture_set_at(pixel, s.viewport).is_some_and(|ts| ts != selected))
        .expect("the material ball shows both of its texture sets");
    let other = s.texture_set_at(other_set_pixel, s.viewport).unwrap();
    let steps = s.history_labels().len();

    let drop = json!({"image": name, "x": other_set_pixel.x, "y": other_set_pixel.y, "viewport": [s.viewport.x, s.viewport.y]});
    let layer = id(execute(&mut s, "layer.add.decal", &drop).unwrap(), "layer");

    assert_eq!(s.ts(), other, "the set under the point is selected");
    assert_eq!(s.doc.project.locate(layer).map(|(ts, _, _)| ts), Some(other), "the decal is in that set");
    assert_eq!(s.selected_layer(), Some(layer));
    assert_eq!(s.history_labels().len(), steps + 1, "one undo step");

    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(s.doc.project.locate(layer).is_none(), "one undo takes the decal away");

    let background = json!({"image": name, "x": 2, "y": 2, "viewport": [s.viewport.x, s.viewport.y]});
    assert!(s.texture_set_at(Vec2::new(2.0, 2.0), s.viewport).is_none());
    assert!(execute(&mut s, "layer.add.decal", &background).is_err(), "the background still takes nothing");
}

/// A decal's image goes in the channel asked for, or masks a new fill layer
#[test]
fn a_decal_image_fills_a_chosen_channel_or_a_mask() {
    let mut s = session();
    let name = s.import_resource_bytes("logo.png", png(8, 8, |_, _| [255, 255, 255, 255])).unwrap();
    let place = |channel: &str| json!({"image": name, "channel": channel, "x": 200, "y": 200, "size": 0.5});

    let rough = id(execute(&mut s, "layer.add.decal", &place("roughness")).unwrap(), "layer");
    assert_eq!(s.doc.project.layer(rough).unwrap().channels.keys().copied().collect::<Vec<_>>(), [Channel::Roughness]);
    assert!(at(&mut s, Channel::Roughness, 200.0, 200.0)[0] > 0.95, "the white image is full roughness under the decal");
    assert!(execute(&mut s, "layer.add.decal", &place("sparkle")).is_err(), "unknown channel");
    let missing = Channel::ALL.into_iter().find(|channel| !s.doc.project.texture_sets[0].has_channel(*channel)).unwrap();
    assert!(execute(&mut s, "layer.add.decal", &place(missing.id())).is_err(), "a channel the set lacks");

    let added = execute(&mut s, "layer.add.decal", &place("mask")).unwrap();
    let masked = id(added.clone(), "layer");
    let mask = s.doc.project.layer(masked).unwrap().mask.as_ref().unwrap();
    assert!(matches!(mask.effects.as_slice(), [effect] if matches!(effect.kind, EffectKind::Fill { projection: Projection::Decal(_), .. })));
    assert_eq!(added["maskEffect"], json!(mask.effects[0].id.0));
    execute(&mut s, "layer.fill.set", &json!({"layer": masked.0, "channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
    assert!(red(at(&mut s, Channel::BaseColor, 200.0, 200.0)), "the mask shows the layer under the decal");
    assert!(!red(at(&mut s, Channel::BaseColor, 120.0, 200.0)), "and hides it elsewhere");

    let effect = &added["maskEffect"];
    execute(&mut s, "layer.decal.set", &json!({"effect": effect, "x": 150, "y": 200})).unwrap();
    assert!(red(at(&mut s, Channel::BaseColor, 150.0, 200.0)), "a mask decal moves like a decal layer");
    assert!(!red(at(&mut s, Channel::BaseColor, 230.0, 200.0)));
    assert!(execute(&mut s, "layer.decal.set", &json!({"effect": rough.0, "x": 150, "y": 200})).is_err(), "a layer id is not an effect");
    execute(&mut s, "layer.decal.set", &json!({"effect": effect, "layer": masked.0, "size": 0.4})).expect("an effect with its own layer");
    let mismatched = json!({"effect": effect, "layer": rough.0, "size": 0.4});
    assert!(execute(&mut s, "layer.decal.set", &mismatched).is_err(), "an effect with another layer is refused");
}

#[test]
fn editing_a_decal_checks_its_own_texture_set_after_selection_changes() {
    for channel in ["baseColor", "mask"] {
        let mut session = Session::new_project("t", primitives::material_ball(), "builtin:material_ball", None, 256).unwrap();
        session.viewport = Vec2::new(400.0, 400.0);
        session.update();
        let image = session.import_resource_bytes("logo.png", png(8, 8, |_, _| [255; 4])).unwrap();
        let (columns, rows) = ((session.viewport.x / SEARCH_STEP) as u32, (session.viewport.y / SEARCH_STEP) as u32);
        let point_on = |texture_set| {
            (0..rows)
                .flat_map(|row| (0..columns).map(move |column| Vec2::new(column as f32, row as f32) * SEARCH_STEP))
                .find(|&point| session.texture_set_at(point, session.viewport) == Some(texture_set))
                .expect("both material-ball sets are visible")
        };
        let original_point = point_on(0);
        let other_point = point_on(1);
        let added =
            execute(&mut session, "layer.add.decal", &json!({"image": image, "channel": channel, "x": original_point.x, "y": original_point.y})).unwrap();
        let layer = added["layer"].clone();
        let effect = added["maskEffect"].clone();
        execute(&mut session, "textureSet.select", &json!({"ts": 1})).unwrap();
        execute(&mut session, "layer.add.paint", &Value::Null).unwrap();
        let before = serde_json::to_value(&session.doc.project).unwrap();
        let history = session.history_labels();

        let rejected = execute(&mut session, "layer.decal.set", &json!({"layer": layer, "effect": effect, "x": other_point.x, "y": other_point.y}));

        assert!(rejected.is_err(), "{channel}: selecting another set must not allow moving the decal onto that set");
        assert_eq!(serde_json::to_value(&session.doc.project).unwrap(), before);
        assert_eq!(session.history_labels(), history);

        let replaced = execute(&mut session, "layer.decal.set", &json!({"layer": layer, "effect": effect, "x": original_point.x, "y": original_point.y}));

        assert!(replaced.is_ok(), "{channel}: the decal's own set remains a valid placement: {replaced:?}");
        let decal = replaced.unwrap()["decal"].clone();

        execute(&mut session, "layer.decal.set", &json!({"layer": layer, "effect": effect, "position": decal["center"], "normal": decal["normal"]})).unwrap();

        assert_eq!(session.ts(), 1, "editing an explicit decal does not change selection");
    }
}

/// A decal's new image stays in the channel the decal was placed in, read as gray there
#[test]
fn a_channel_decal_reads_its_image_as_gray_and_keeps_its_channel() {
    let mut s = session();
    let white = s.import_resource_bytes("white.png", png(8, 8, |_, _| [255, 255, 255, 255])).unwrap();
    let green = s.import_resource_bytes("green.png", png(8, 8, |_, _| [0, 255, 0, 255])).unwrap();
    let place = json!({"image": white, "channel": "roughness", "x": 200, "y": 200, "size": 0.5});
    let layer = id(execute(&mut s, "layer.add.decal", &place).unwrap(), "layer");

    execute(&mut s, "layer.decal.set", &json!({"layer": layer.0, "image": green})).unwrap();

    assert_eq!(s.doc.project.layer(layer).unwrap().channels.keys().copied().collect::<Vec<_>>(), [Channel::Roughness]);
    let roughness = at(&mut s, Channel::Roughness, 200.0, 200.0)[0];
    assert!((roughness - texturecraft_engine::core::lum([0.0, 1.0, 0.0])).abs() < 0.02, "green reads as its gray, not its red: {roughness}");
}

/// The Geometry mask tool: a click toggles the selected layer on the part under it.
#[test]
fn geometry_mask_tool_toggles_the_clicked_part() {
    let obj = "o Lid\nv 0 0 0\nv 1 0 0\nv 0 1 0\nvt 0 0\nvt 0.5 0\nvt 0 0.5\nf 1/1 2/2 3/3\n\
               o Body\nv 1 1 0\nv 2 1 0\nv 1 2 0\nvt 0.5 0.5\nvt 1 0.5\nvt 0.5 1\nf 4/4 5/5 6/6\n";
    let mut s = Session::from_mesh_bytes("parts.obj", obj.as_bytes(), 256, "parts").unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    (s.camera.yaw, s.camera.pitch) = (0.0, 0.0);
    let (lo, hi) = s.mesh.bounds();
    s.camera.frame(lo, hi);
    // Viewport pixels of a point inside each triangle.
    let screen = |s: &Session, p: [f32; 3]| {
        let n = s.camera.project(texturecraft_engine::geom::glam::Vec3::from(p), 1.0).unwrap();
        json!({"x": (n.x + 1.0) * 200.0, "y": (1.0 - n.y) * 200.0})
    };
    let (lid, body) = (screen(&s, [0.25, 0.25, 0.0]), screen(&s, [1.25, 1.25, 0.0]));
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"color": [1, 0, 0]})).unwrap(), "layer");
    execute(&mut s, "tool.geometryMask", &Value::Null).unwrap();
    let r = execute(&mut s, "tool.click", &lid).unwrap();
    assert_eq!((r["part"].as_str(), r["shown"].as_bool(), &r["geometryMask"]), (Some("Lid"), Some(false), &json!(["Body"])), "{r}");
    assert_eq!(s.doc.project.layer(layer).unwrap().geometry_mask, Some(vec!["Body".to_string()]));
    let r = execute(&mut s, "tool.click", &body).unwrap();
    assert_eq!(r["geometryMask"], json!([]), "both hidden");
    execute(&mut s, "tool.click", &lid).unwrap();
    let r = execute(&mut s, "tool.click", &body).unwrap();
    assert_eq!(r["geometryMask"], Value::Null, "shown on every part again: no mask");
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(s.doc.project.layer(layer).unwrap().geometry_mask, Some(vec!["Lid".to_string()]));
    assert!(execute(&mut s, "tool.click", &json!({"x": 1, "y": 1})).is_err(), "nothing under the point");
}

#[test]
fn images_fill_mesh_map_slots() {
    use texturecraft_engine::core::BakedMap;
    let mut s = session();
    // Dark on the left, light on the right.
    let ao = s.import_resource_bytes("ao.png", png(8, 8, |x, _| if x < 4 { [64, 64, 64, 255] } else { [192, 192, 192, 255] })).unwrap();
    let r = execute(&mut s, "textureSet.meshMap.set", &json!({"map": "ambientOcclusion", "image": ao})).unwrap();
    let res = s.doc.project.texture_sets[0].bake.common.resolution.pixels();
    assert_eq!(r, json!({"map": "ambientOcclusion", "resolution": res}));
    assert!(s.doc.project.texture_sets[0].baked.contains(&BakedMap::AmbientOcclusion));
    let sample = |s: &Session, m: BakedMap, x: u32| s.baked(0).unwrap().sample(m, x, 50, (100, 100)).unwrap();
    assert!((sample(&s, BakedMap::AmbientOcclusion, 10)[0] - 64.0 / 255.0).abs() < 1e-3, "raw file values, not linearized");
    assert!((sample(&s, BakedMap::AmbientOcclusion, 90)[0] - 192.0 / 255.0).abs() < 1e-3);
    // A normal map is read in the project's normal format (DirectX: green flipped).
    execute(&mut s, "textureSet.settings", &json!({"normalFormat": "directX"})).unwrap();
    let n = s.import_resource_bytes("n.png", png(2, 2, |_, _| [128, 64, 255, 255])).unwrap();
    execute(&mut s, "textureSet.meshMap.set", &json!({"map": "normal", "image": n})).unwrap();
    let v = sample(&s, BakedMap::Normal, 50);
    assert!((v[1] - (1.0 - 64.0 / 255.0)).abs() < 1e-3 && (v[2] - 1.0).abs() < 1e-3, "{v:?}");
    // Clearing empties the slot (one undo step); bad params are errors.
    execute(&mut s, "textureSet.meshMap.clear", &json!({"map": "ambientOcclusion"})).unwrap();
    assert!(s.baked(0).unwrap().sample(BakedMap::AmbientOcclusion, 0, 0, (100, 100)).is_none());
    assert!(!s.doc.project.texture_sets[0].baked.contains(&BakedMap::AmbientOcclusion));
    assert!(execute(&mut s, "textureSet.meshMap.clear", &json!({"map": "ambientOcclusion"})).is_err(), "already empty");
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(s.doc.project.texture_sets[0].baked.contains(&BakedMap::AmbientOcclusion));
    assert!(execute(&mut s, "textureSet.meshMap.set", &json!({"map": "nope", "image": ao})).is_err());
    assert!(execute(&mut s, "textureSet.meshMap.set", &json!({"map": "curvature", "image": "missing.png"})).is_err());
    assert!(execute(&mut s, "textureSet.meshMap.set", &json!({"map": "curvature"})).is_err());
    // The maps are saved with the project.
    let back = Session::from_project_bytes(&s.to_project_bytes().unwrap()).unwrap();
    assert_eq!(back.doc.baked, s.doc.baked);
}

#[test]
fn fill_projection_edits_keep_what_they_do_not_change() {
    let mut s = session();
    let layer = id(execute(&mut s, "layer.add.fill", &json!({"color": [0.5, 0.2, 0.1]})).unwrap(), "layer");
    let proj = |s: &Session| match &s.doc.project.layer(layer).unwrap().kind {
        LayerKind::Fill(f) => f.projection.clone(),
        _ => panic!("a fill layer"),
    };
    let set = |s: &mut Session, mut p: Value| {
        p["layer"] = json!(layer.0);
        execute(s, "layer.fill.projection", &p)
    };
    set(&mut s, json!({"rotation": 390, "offset": [0.25, 0.5]})).unwrap();
    set(&mut s, json!({"scale": 2})).unwrap();
    assert_eq!(proj(&s), Projection::Uv { scale: [2.0, 2.0], rotation: 30.0, offset: [0.25, 0.5] }, "a scale keeps the rotation and offset");
    set(&mut s, json!({"projection": "uv", "scale": [2, 3], "offset": "far"})).unwrap();
    assert_eq!(proj(&s), Projection::Uv { scale: [2.0, 3.0], rotation: 30.0, offset: [0.25, 0.5] }, "the same kind keeps its fields; a bad offset is ignored");
    set(&mut s, json!({"projection": "triplanar", "offset": [1, 2, 3]})).unwrap();
    set(&mut s, json!({"sharpness": 100})).unwrap();
    assert_eq!(proj(&s), Projection::Triplanar { scale: 1.0, sharpness: 64.0, offset: [1.0, 2.0, 3.0] }, "a new kind starts from its defaults");
    set(&mut s, json!({"projection": "planar", "axis": 7, "offset": [1e9, 0, 0]})).unwrap();
    assert_eq!(proj(&s), Projection::Planar { scale: 1.0, axis: 2, offset: [1000.0, 0.0, 0.0] }, "hostile values are clamped");
    assert!(set(&mut s, json!({"projection": "hexagonal"})).is_err());
    // The offset moves the pattern: a UV offset changes what the composite shows.
    set(&mut s, json!({"projection": "uv", "scale": 1})).unwrap();
    execute(
        &mut s,
        "layer.fill.set",
        &json!({"layer": layer.0, "channel": "baseColor", "procedural": "Checker", "scale": 8, "low": [0, 0, 0], "high": [1, 1, 1]}),
    )
    .unwrap();
    let before = at(&mut s, Channel::BaseColor, 200.0, 200.0);
    set(&mut s, json!({"offset": [0.125, 0.0]})).unwrap();
    let after = at(&mut s, Channel::BaseColor, 200.0, 200.0);
    assert!((before[0] - after[0]).abs() > 0.5, "one checker cell over: {before:?} -> {after:?}");
}

/// A fill's normal image is OpenGL unless marked DirectX, which flips its green; inspect shows
/// the mark, and a project saved before the mark existed reads its images as OpenGL.
#[test]
fn directx_normal_images_flip_green() {
    let mut s = session();
    let name = s.import_resource_bytes("n.png", png(4, 4, |_, _| [128, 204, 255, 255])).unwrap();
    let layer = id(execute(&mut s, "layer.add.fill", &Value::Null).unwrap(), "layer");
    let fill = |s: &mut Session, p: Value| {
        let mut p = p;
        p["layer"] = json!(layer.0);
        p["channel"] = json!("normal");
        execute(s, "layer.fill.set", &p)
    };
    let green = |s: &mut Session| at(s, Channel::Normal, 200.0, 200.0)[1];
    fill(&mut s, json!({"image": name})).unwrap();
    assert!((green(&mut s) - 0.8).abs() < 0.02, "OpenGL by default: {}", green(&mut s));
    fill(&mut s, json!({"normalFormat": "directX"})).unwrap();
    assert!((green(&mut s) - 0.2).abs() < 0.02, "DirectX flips green: {}", green(&mut s));
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap();
    let row = inspect["layers"].as_array().unwrap().iter().find(|l| l["id"] == layer.0).unwrap().clone();
    assert_eq!(row["fill"]["sources"]["normal"], json!({"image": name, "normalFormat": "directX"}));
    fill(&mut s, json!({"image": name, "normalFormat": "openGl"})).unwrap();
    assert!((green(&mut s) - 0.8).abs() < 0.02);
    assert!(fill(&mut s, json!({"normalFormat": "sideways"})).is_err());
    assert!(fill(&mut s, json!({"image": name, "normalFormat": 1})).is_err(), "not a string");
    assert!(fill(&mut s, json!({"value": [0.5, 0.5, 1.0], "normalFormat": "directX"})).is_err(), "not an image");
    // Marking an image on a channel that is off leaves it off.
    fill(&mut s, json!({"enabled": false})).unwrap();
    fill(&mut s, json!({"normalFormat": "directX"})).unwrap();
    assert!(!s.doc.project.layer(layer).unwrap().channels.contains_key(&Channel::Normal));
    fill(&mut s, json!({"enabled": true})).unwrap();
    fill(&mut s, json!({"value": [0.5, 0.5, 1.0]})).unwrap();
    assert!(fill(&mut s, json!({"normalFormat": "directX"})).is_err(), "no image on the channel");
    let old: texturecraft_engine::core::Source = serde_json::from_value(json!({"Image": {"name": "n"}})).unwrap();
    assert_eq!(old, texturecraft_engine::core::Source::image("n"));
}
