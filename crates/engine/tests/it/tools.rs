//! Clone, smudge, image and library alphas and the quick mask, driven through engine commands the way
//! agents paint.

use serde_json::{Value, json};
use texturecraft_engine::core::{Channel, LayerId};
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, StackTarget, execute};

const RES: f32 = 256.0;

fn session() -> Session {
    let mut s = Session::new_project("t", primitives::sphere(1.0, 48, 24), "builtin:sphere", None, 256).unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    s.camera.yaw = 0.0;
    s.camera.pitch = 0.0;
    s.update();
    s
}

/// Base color of the composite under a viewport point.
fn color_at(s: &Session, x: f32, y: f32) -> [f32; 3] {
    let hit = s.pick(Vec2::new(x, y), s.viewport).unwrap();
    s.caches[0].composite.value(Channel::BaseColor, (hit.uv.x * RES) as u32, (hit.uv.y * RES) as u32)
}

fn red(v: [f32; 3]) -> bool {
    v[0] > 0.85 && v[1] < 0.15
}

fn dab(s: &mut Session, x: f32, y: f32) {
    execute(s, "tool.stroke", &json!({"points": [[x, y]]})).unwrap();
}

fn paint_red(s: &mut Session, size: f32) {
    execute(s, "tool.paint", &Value::Null).unwrap();
    execute(s, "tool.material", &json!({"channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
    execute(s, "tool.brush", &json!({"size": size, "hardness": 90})).unwrap();
}

#[test]
fn clone_copies_values_from_the_source_offset() {
    let mut s = session();
    paint_red(&mut s, 20.0);
    dab(&mut s, 160.0, 200.0);
    assert!(red(color_at(&s, 160.0, 200.0)));
    assert!(!red(color_at(&s, 240.0, 200.0)));

    execute(&mut s, "tool.clone", &Value::Null).unwrap();
    assert!(execute(&mut s, "tool.stroke", &json!({"points": [[240, 200]]})).is_err(), "no source yet");
    execute(&mut s, "tool.cloneSource", &json!({"x": 160, "y": 200})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 6})).unwrap();
    dab(&mut s, 240.0, 200.0);
    let v = color_at(&s, 240.0, 200.0);
    assert!(red(v), "cloned red: {v:?}");
    // The offset stays for the next strokes: the source moved along with the destination.
    let off = s.tool.clone_source.unwrap().offset.unwrap();
    assert!(off[0].abs() > 0.01, "{off:?}");
    // One undo step per clone stroke.
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(!red(color_at(&s, 240.0, 200.0)));
}

#[test]
fn smudge_drags_color_along_the_stroke() {
    let mut s = session();
    paint_red(&mut s, 12.0);
    dab(&mut s, 170.0, 200.0);
    let before = color_at(&s, 205.0, 200.0);
    assert!(!red(before), "{before:?}");
    execute(&mut s, "tool.smudge", &Value::Null).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 10, "smudgeStrength": 100, "spacing": 5})).unwrap();
    let pts: Vec<[f32; 2]> = (0..=40).map(|i| [170.0 + i as f32, 200.0]).collect();
    execute(&mut s, "tool.stroke", &json!({"points": pts})).unwrap();
    let after = color_at(&s, 205.0, 200.0);
    assert!(after[0] - before[0] > 0.2 && after[1] < before[1], "red dragged right: {before:?} -> {after:?}");
}

#[test]
fn image_alpha_shapes_the_stamp() {
    let mut s = session();
    let mut png = Vec::new();
    // Top half white, bottom half black.
    let img = image::RgbaImage::from_fn(16, 16, |_, y| if y < 8 { image::Rgba([255; 4]) } else { image::Rgba([0, 0, 0, 255]) });
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let name = s.import_resource_bytes("half.png", png).unwrap();
    paint_red(&mut s, 20.0);
    assert!(execute(&mut s, "tool.brush", &json!({"alpha": {"image": "missing.png"}})).is_err());
    let r = execute(&mut s, "tool.brush", &json!({"alpha": {"image": name}})).unwrap();
    assert_eq!(r["alpha"]["image"], json!(name));
    execute(&mut s, "tool.strokeUv", &json!({"points": [[0.5, 0.5]]})).unwrap();
    // In the 2D view +v points down: the image's top half lands above the center.
    let at = |s: &Session, dy: i32| s.caches[0].composite.value(Channel::BaseColor, 128, (128 + dy) as u32);
    assert!(red(at(&s, -12)), "{:?}", at(&s, -12));
    assert!(!red(at(&s, 12)), "{:?}", at(&s, 12));
    // A procedural alpha clears the image.
    let r = execute(&mut s, "tool.brush", &json!({"alpha": "round"})).unwrap();
    assert_eq!(r["alpha"], json!("Round"));
}

/// Offsets (dx, dy) of red composite texels within `r` texels of (cx, cy), with their lengths.
fn red_offsets(s: &Session, cx: i32, cy: i32, r: i32) -> Vec<(i32, i32, f32)> {
    let mut out = Vec::new();
    for dy in -r..=r {
        for dx in -r..=r {
            let (x, y) = (cx + dx, cy + dy);
            if (0..RES as i32).contains(&x) && (0..RES as i32).contains(&y) && red(s.caches[0].composite.value(Channel::BaseColor, x as u32, y as u32)) {
                out.push((dx, dy, ((dx * dx + dy * dy) as f32).sqrt()));
            }
        }
    }
    out
}

#[test]
fn library_alphas_shape_the_stamp() {
    let mut s = session();
    paint_red(&mut s, 20.0);
    let list = execute(&mut s, "alpha.list", &Value::Null).unwrap();
    assert!(list["library"].as_array().is_some_and(|a| a.len() >= 60), "{list}");
    assert_eq!(list["procedural"][0], json!("Round"));
    assert!(execute(&mut s, "tool.brush", &json!({"alpha": {"library": "nope"}})).is_err());
    let r = execute(&mut s, "tool.brush", &json!({"alpha": {"library": "Ring"}})).unwrap();
    assert_eq!(r["alpha"], json!({"library": "ring"}));

    // 2D view: a ring with a clean middle.
    execute(&mut s, "tool.strokeUv", &json!({"points": [[0.5, 0.5]]})).unwrap();
    let ring = red_offsets(&s, 128, 128, 48);
    let outer = ring.iter().map(|t| t.2).fold(0.0, f32::max);
    assert!(outer > 6.0 && outer < 47.0, "{outer}");
    assert!(ring.iter().all(|t| t.2 > outer * 0.45), "the middle stays clean");
    let mid = (outer * 0.72 / 0.92).round() as i32;
    for (dx, dy) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
        assert!(ring.iter().any(|t| (t.0, t.1) == (dx * mid, dy * mid)), "ring at {:?}", (dx * mid, dy * mid));
    }

    // 3D view: the same hollow stamp on the sphere.
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    dab(&mut s, 200.0, 200.0);
    let hit = s.pick(Vec2::new(200.0, 200.0), s.viewport).unwrap();
    let (cx, cy) = ((hit.uv.x * RES) as i32, (hit.uv.y * RES) as i32);
    let ring = red_offsets(&s, cx, cy, 48);
    assert!(!ring.is_empty(), "painted");
    assert!(ring.iter().all(|t| t.2 > 2.0), "the middle stays clean");

    // A procedural alpha clears the library alpha.
    let r = execute(&mut s, "tool.brush", &json!({"alpha": "square"})).unwrap();
    assert_eq!(r["alpha"], json!("Square"));
    assert!(s.tool.brush.alpha_library.is_none());
}

#[test]
fn quick_mask_limits_painting() {
    let mut s = session();
    paint_red(&mut s, 20.0);
    let r = execute(&mut s, "viewport.quickMask.edit", &Value::Null).unwrap();
    assert_eq!(r["editing"], json!(true));
    assert_eq!(r["enabled"], json!(true), "editing enables the mask");
    let undo = s.history_labels().len();
    dab(&mut s, 150.0, 200.0);
    assert!(!red(color_at(&s, 150.0, 200.0)), "mask strokes do not paint the layer");
    assert_eq!(s.history_labels().len(), undo, "mask strokes stay out of the history");
    assert!(s.quick_mask_rgba(0, 64).is_some_and(|v| v.len() == 64 * 64 * 4));
    execute(&mut s, "viewport.quickMask.edit", &Value::Null).unwrap();

    execute(&mut s, "tool.brush", &json!({"size": 6})).unwrap();
    dab(&mut s, 150.0, 200.0);
    dab(&mut s, 250.0, 200.0);
    assert!(red(color_at(&s, 150.0, 200.0)), "inside the mask");
    assert!(!red(color_at(&s, 250.0, 200.0)), "outside the mask");

    execute(&mut s, "viewport.quickMask.invert", &Value::Null).unwrap();
    dab(&mut s, 250.0, 200.0);
    assert!(red(color_at(&s, 250.0, 200.0)), "inverted: outside is open");

    execute(&mut s, "viewport.quickMask", &json!({"enabled": false})).unwrap();
    execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [0.0, 1.0, 0.0]})).unwrap();
    dab(&mut s, 150.0, 200.0);
    assert!(color_at(&s, 150.0, 200.0)[1] > 0.85, "disabled: everything is open");
}

/// How much redder than green a texel must be to count as painted (faded edges included)
const PAINTED_TINT: f32 = 0.05;

/// Separate painted runs down texture column `x` (base color tinted red)
fn painted_runs(session: &Session, x: u32) -> usize {
    let painted = |y: u32| {
        let color = session.caches[0].composite.value(Channel::BaseColor, x, y);
        color[0] - color[1] > PAINTED_TINT
    };
    (0..RES as u32).filter(|&y| painted(y) && (y == 0 || !painted(y - 1))).count()
}

#[test]
fn a_stroke_to_the_edge_of_the_mesh_fades_without_breaking_into_dots() {
    let mut session = session();
    paint_red(&mut session, 4.0);
    // From the middle of the view to past the ball's right edge, above center
    let points: Vec<Value> = (0..40).map(|step| json!([200.0 + step as f32 * 6.0, 165.0])).collect();
    execute(&mut session, "tool.stroke", &json!({"points": points})).unwrap();
    session.update();

    // The stroke crosses each texture column once, also where the surface turns away from the camera
    let broken: Vec<u32> = (0..RES as u32).filter(|&x| painted_runs(&session, x) > 1).collect();
    assert!(broken.is_empty(), "columns with gaps: {broken:?}");
}

#[test]
fn swap_mask_value_inverts_only_while_a_mask_is_the_target() {
    let mut session = session();
    execute(&mut session, "layer.add.paint", &Value::Null).unwrap();
    let swap = |session: &mut Session| execute(session, "tool.swapMaskValue", &Value::Null);
    let effect_target = |result: Value| StackTarget::Effect(LayerId(result["effect"].as_u64().unwrap()));

    assert!(swap(&mut session).is_err(), "painting the layer content: nothing to invert");

    let content_effect = effect_target(execute(&mut session, "effect.add.paint", &json!({"target": "content"})).unwrap());
    session.target = content_effect;
    assert!(swap(&mut session).is_err(), "a content effect is not a mask");

    execute(&mut session, "mask.add.white", &Value::Null).unwrap();
    session.target = StackTarget::Mask;
    session.tool.mask_value = 0.2;

    swap(&mut session).unwrap();

    assert_eq!(session.tool.mask_value, 0.8);

    session.tool.mask_value = 1.0;

    swap(&mut session).unwrap();

    assert_eq!(session.tool.mask_value, 0.0);

    swap(&mut session).unwrap();

    assert_eq!(session.tool.mask_value, 1.0, "pressing it again goes back");

    session.target = effect_target(execute(&mut session, "effect.add.paint", &json!({"target": "mask"})).unwrap());

    swap(&mut session).unwrap();

    assert_eq!(session.tool.mask_value, 0.0, "a mask effect paints the mask too");
}
