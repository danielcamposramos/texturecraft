//! Layer-panel thumbnails: they equal a plain CPU evaluation of the same sub-stack at 64 px,
//! follow edits, drop stale and vanished work, and only render what was asked for.

use std::time::{Duration, Instant};

use serde_json::{Value, json};
use texturecraft_engine::compose::{Inputs, ThumbImage, ThumbTarget, all_tiles, flatten_layer, mask_image};
use texturecraft_engine::core::{Channel, ChannelKind, LayerId};
use texturecraft_engine::geom::UvGBuffer;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::texel::{from_u16, from_u16_signed, linear_to_srgb};
use texturecraft_engine::thumbs::{CACHE_ENTRIES, ThumbService};
use texturecraft_engine::{Session, execute};

const N: u32 = 64;

fn session() -> Session {
    Session::new_project("t", primitives::sphere(1.0, 32, 16), "builtin:sphere", None, 256).unwrap()
}

fn id(v: Value, key: &str) -> LayerId {
    LayerId(v[key].as_u64().unwrap())
}

/// A fill layer with two content effects (invert, then posterize) and a masked generator.
struct Stack {
    layer: LayerId,
    invert: LayerId,
    posterize: LayerId,
    gen_: LayerId,
}

fn stack(s: &mut Session) -> Stack {
    let layer = id(execute(s, "layer.add.fill", &json!({"color": "#336699", "roughness": 0.2})).unwrap(), "layer");
    let invert = id(execute(s, "effect.add.filter", &json!({"target": "content", "filter": "invert"})).unwrap(), "effect");
    let posterize = id(execute(s, "effect.add.filter", &json!({"target": "content", "filter": "posterize"})).unwrap(), "effect");
    execute(s, "mask.add.black", &Value::Null).unwrap();
    let gen_ = id(execute(s, "effect.add.generator", &json!({"generator": "position", "level": 0.5})).unwrap(), "effect");
    Stack { layer, invert, posterize, gen_ }
}

fn inputs<'a>(s: &'a Session, project: &'a texturecraft_engine::core::Project, gbuf: &'a UvGBuffer) -> Inputs<'a> {
    static EMPTY_BAKED: std::sync::LazyLock<texturecraft_engine::compose::BakedMaps> = std::sync::LazyLock::new(Default::default);
    let baked = s.doc.baked.get(&0).unwrap_or(&EMPTY_BAKED);
    Inputs {
        project,
        ts: 0,
        mesh: &s.mesh,
        gbuf,
        pixels: &s.doc.pixels,
        baked,
        block_scales: Default::default(),
        preview: None,
        images: &s.images,
        bounds: s.mesh.bounds(),
    }
}

/// The reference: the same sub-stack cut by hand and evaluated at 64 px with the ordinary CPU
/// evaluator (the mask with `mask_image`, the content with `flatten_layer`), then encoded.
fn oracle(s: &Session, target: ThumbTarget) -> ThumbImage {
    let gbuf = UvGBuffer::new(&s.mesh, 0, N, N);
    let mut p = s.doc.project.clone();
    let (layer, upto, channel) = match target {
        ThumbTarget::Content { layer, channel } => (layer, None, channel),
        ThumbTarget::Mask { layer } => (layer, None, Channel::BaseColor),
        ThumbTarget::Stage { effect, channel } => (p.effect(effect).unwrap().0, Some(effect), channel),
    };
    let l = p.layers.get_mut(&layer).unwrap();
    l.visible = true;
    let in_mask = upto.is_some_and(|e| l.mask.as_ref().is_some_and(|m| m.effects.iter().any(|x| x.id == e))) || matches!(target, ThumbTarget::Mask { .. });
    if in_mask {
        let m = l.mask.as_mut().unwrap();
        m.enabled = true;
        if let Some(e) = upto {
            let k = m.effects.iter().position(|x| x.id == e).unwrap();
            m.effects.truncate(k + 1);
        }
        let gray = mask_image(&inputs(s, &p, &gbuf), layer, N).unwrap();
        let rgba = gray.iter().enumerate().flat_map(|(i, g)| [*g, *g, *g, if gbuf.covered(i) { 255 } else { 0 }]).collect();
        return ThumbImage { width: N, height: N, rgba };
    }
    l.mask = None;
    l.channels.insert(channel, texturecraft_engine::core::ChannelBlend::default());
    if let Some(e) = upto {
        let k = l.content_effects.iter().position(|x| x.id == e).unwrap();
        l.content_effects.truncate(k + 1);
    }
    // A paint or fill layer: its own (premultiplied value, coverage), coverage as alpha.
    let flat = flatten_layer(&inputs(s, &p, &gbuf), layer, &all_tiles(N, N)).unwrap();
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0 + 0.5) as u8;
    let mut rgba = vec![0u8; (N * N * 4) as usize];
    let comps = channel.components();
    for (_, _, buf) in flat.iter().filter(|(c, _, _)| *c == channel) {
        for (i, px) in buf.chunks_exact(comps + 1).enumerate() {
            let a = from_u16(px[comps]);
            if a <= 0.0 {
                continue;
            }
            let v: [f32; 3] = std::array::from_fn(|k| {
                let raw = px[k.min(comps - 1)];
                (if channel.kind() == ChannelKind::Signed { from_u16_signed(raw) } else { from_u16(raw) }) / a
            });
            let px = match channel.kind() {
                ChannelKind::Color => v.map(|c| byte(linear_to_srgb(c.clamp(0.0, 1.0)))),
                ChannelKind::Signed => v.map(|c| byte(c * 0.5 + 0.5)),
                _ => v.map(byte),
            };
            rgba[i * 4..i * 4 + 4].copy_from_slice(&[px[0], px[1], px[2], byte(a)]);
        }
    }
    ThumbImage { width: N, height: N, rgba }
}

/// Frames of `svc` until every target is current (or a timeout).
fn settle(svc: &mut ThumbService, s: &Session, targets: &[ThumbTarget], clock: &mut f64) {
    let t0 = Instant::now();
    loop {
        *clock += 0.2;
        svc.begin_frame(s, *clock);
        for t in targets {
            svc.get(s, *t);
        }
        svc.poll(s);
        if targets.iter().all(|t| svc.is_fresh(*t)) {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(60), "thumbnails never settled");
        std::thread::sleep(Duration::from_millis(2));
    }
}

fn image(svc: &mut ThumbService, s: &Session, t: ThumbTarget) -> ThumbImage {
    (*svc.get(s, t).image.expect("a picture")).clone()
}

#[test]
fn thumbnails_equal_a_cpu_evaluation_of_the_same_sub_stack() {
    let mut session = session();
    let stack = stack(&mut session);
    let base_color = Channel::BaseColor;
    let targets = [
        ThumbTarget::Content { layer: stack.layer, channel: base_color },
        ThumbTarget::Content { layer: stack.layer, channel: Channel::Roughness },
        ThumbTarget::Mask { layer: stack.layer },
        ThumbTarget::Stage { effect: stack.invert, channel: base_color },
        ThumbTarget::Stage { effect: stack.posterize, channel: base_color },
        ThumbTarget::Stage { effect: stack.gen_, channel: base_color },
    ];
    let (mut service, mut clock) = (ThumbService::new(), 0.0);

    settle(&mut service, &session, &targets, &mut clock);

    for target in targets {
        assert_eq!(image(&mut service, &session, target), oracle(&session, target), "{target:?}");
    }

    // The first stage inverts #336699 in linear light
    let inverted = image(&mut service, &session, targets[3]);
    let pixel = inverted.rgba.as_chunks::<4>().0.iter().find(|pixel| pixel[3] == 255).unwrap();
    let expected = |byte: f32| (linear_to_srgb(1.0 - texturecraft_engine::texel::srgb_to_linear(byte / 255.0)) * 255.0 + 0.5) as i32;
    assert!((i32::from(pixel[0]) - expected(51.0)).abs() <= 1 && (i32::from(pixel[2]) - expected(153.0)).abs() <= 1, "{pixel:?}");

    // A position generator must produce a nonuniform mask
    let mask = image(&mut service, &session, targets[2]);
    let grays: Vec<u8> = mask.rgba.as_chunks::<4>().0.iter().filter(|pixel| pixel[3] == 255).map(|pixel| pixel[0]).collect();
    assert!(grays.iter().max() != grays.iter().min());
}

#[test]
fn earlier_stages_never_show_later_effects_and_hidden_effects_are_bypassed() {
    let mut s = session();
    let st = stack(&mut s);
    let base = Channel::BaseColor;
    let (first, second) = (ThumbTarget::Stage { effect: st.invert, channel: base }, ThumbTarget::Stage { effect: st.posterize, channel: base });
    let (mut svc, mut clock) = (ThumbService::new(), 0.0);
    settle(&mut svc, &s, &[first, second], &mut clock);
    let (a, b) = (image(&mut svc, &s, first), image(&mut svc, &s, second));
    // Hide the second effect: its stage now shows the first one's result; the first is unchanged.
    execute(&mut s, "effect.set", &json!({"effect": st.posterize.0, "visible": false})).unwrap();
    settle(&mut svc, &s, &[first, second], &mut clock);
    assert_eq!(image(&mut svc, &s, first), a);
    assert_eq!(image(&mut svc, &s, second), a);
    assert_ne!(a, b, "posterize should change something for the test to mean anything");
    // And the content thumbnail (everything) follows.
    let all = ThumbTarget::Content { layer: st.layer, channel: base };
    settle(&mut svc, &s, &[all], &mut clock);
    assert_eq!(image(&mut svc, &s, all), a);
}

#[test]
fn editing_invalidates_and_unrelated_changes_do_not() {
    let mut s = session();
    let st = stack(&mut s);
    let t = ThumbTarget::Content { layer: st.layer, channel: Channel::BaseColor };
    let (mut svc, mut clock) = (ThumbService::new(), 0.0);
    settle(&mut svc, &s, &[t], &mut clock);
    let before = image(&mut svc, &s, t);
    let rendered = svc.rendered;
    // A camera move changes the session revision but no thumbnail input: nothing is re-rendered.
    execute(&mut s, "camera.set", &json!({"yaw": 0.7})).ok();
    s.revision += 1;
    for _ in 0..3 {
        clock += 0.3;
        svc.begin_frame(&s, clock);
        svc.get(&s, t);
        svc.poll(&s);
    }
    assert!(svc.is_fresh(t));
    assert_eq!(svc.rendered, rendered, "an unrelated change re-rendered a thumbnail");
    // An edit of the layer re-renders it, and the new picture is the new oracle.
    execute(&mut s, "effect.set", &json!({"effect": st.invert.0, "visible": false})).unwrap();
    clock += 0.3;
    svc.begin_frame(&s, clock);
    assert!(!svc.is_fresh(t));
    settle(&mut svc, &s, &[t], &mut clock);
    let after = image(&mut svc, &s, t);
    assert_ne!(before, after);
    assert_eq!(after, oracle(&s, t));
    // Undo brings the first picture back.
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    settle(&mut svc, &s, &[t], &mut clock);
    assert_eq!(image(&mut svc, &s, t), before);
}

#[test]
fn deleting_a_layer_with_a_pending_thumbnail_is_harmless() {
    let mut s = session();
    let st = stack(&mut s);
    let (a, b) = (ThumbTarget::Content { layer: st.layer, channel: Channel::BaseColor }, ThumbTarget::Mask { layer: st.layer });
    let mut svc = ThumbService::new();
    // Queue the work and hand it to the worker, then delete the layer before collecting.
    svc.begin_frame(&s, 1.0);
    svc.get(&s, a);
    svc.get(&s, b);
    svc.poll(&s);
    execute(&mut s, "layer.remove", &json!({"layer": st.layer.0})).unwrap();
    let mut clock = 2.0;
    for _ in 0..200 {
        clock += 0.2;
        svc.begin_frame(&s, clock);
        // The row is gone, so nobody asks for it any more; the pending result is dropped.
        svc.poll(&s);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(!svc.is_fresh(a) && !svc.is_fresh(b));
    assert_eq!(svc.cached(), 0);
    assert!(!svc.busy());
    // Asking for a layer that no longer exists renders nothing and does not panic either.
    settle_gone(&mut svc, &s, a, &mut clock);
}

fn settle_gone(svc: &mut ThumbService, s: &Session, t: ThumbTarget, clock: &mut f64) {
    for _ in 0..20 {
        *clock += 0.2;
        svc.begin_frame(s, *clock);
        svc.get(s, t);
        svc.poll(s);
        std::thread::sleep(Duration::from_millis(2));
    }
    assert!(svc.get(s, t).image.is_none());
}

#[test]
fn a_result_of_an_older_document_never_replaces_a_newer_one() {
    let mut s = session();
    let st = stack(&mut s);
    let t = ThumbTarget::Content { layer: st.layer, channel: Channel::BaseColor };
    let mut svc = ThumbService::new();
    svc.begin_frame(&s, 1.0);
    svc.get(&s, t);
    svc.poll(&s); // the job of the first document is running
    execute(&mut s, "effect.set", &json!({"effect": st.invert.0, "visible": false})).unwrap();
    let mut clock = 2.0;
    settle(&mut svc, &s, &[t], &mut clock);
    // Whatever the first job returned, what is shown is the current document.
    assert_eq!(image(&mut svc, &s, t), oracle(&s, t));
}

#[test]
fn only_requested_rows_render_and_the_cache_is_bounded() {
    let mut session = session();
    let layers: Vec<LayerId> = (0..(CACHE_ENTRIES + 30)).map(|_| id(execute(&mut session, "layer.add.paint", &Value::Null).unwrap(), "layer")).collect();
    let target = |layer: LayerId| ThumbTarget::Content { layer, channel: Channel::BaseColor };
    let mut service = ThumbService::new();

    // A frame asks for six rows, then scrolling leaves only one visible
    service.begin_frame(&session, 1.0);
    for layer in &layers[..6] {
        service.get(&session, target(*layer));
    }
    service.poll(&session);
    let mut clock = 2.0;
    for _ in 0..300 {
        clock += 0.2;
        service.begin_frame(&session, clock);
        service.get(&session, target(layers[0]));
        service.poll(&session);
        std::thread::sleep(Duration::from_millis(1));
    }

    assert!(service.is_fresh(target(layers[0])));
    let completed = layers[..6].iter().filter(|layer| service.is_fresh(target(**layer))).count();
    assert!(completed <= 3, "{completed} of the scrolled-away rows were rendered");

    // Scrolling through every layer evicts pictures and releases their allocations
    let mut images = Vec::new();
    for chunk in layers.chunks(12) {
        let targets: Vec<ThumbTarget> = chunk.iter().map(|layer| target(*layer)).collect();
        settle(&mut service, &session, &targets, &mut clock);
        for target in targets {
            images.push(std::sync::Arc::downgrade(&service.get(&session, target).image.unwrap()));
        }
        assert!(service.cached() <= CACHE_ENTRIES, "{}", service.cached());
    }
    clock += 0.2;
    service.begin_frame(&session, clock);
    service.poll(&session);

    assert!(service.cached() <= CACHE_ENTRIES, "{}", service.cached());
    assert!(images.len() > CACHE_ENTRIES);
    assert!(images.first().unwrap().upgrade().is_none(), "the oldest image was evicted");
    assert_eq!(images.iter().filter(|image| image.strong_count() > 0).count(), service.cached());

    drop(service);

    assert!(images.iter().all(|image| image.upgrade().is_none()), "closing the cache releases every picture");
}

#[test]
fn an_empty_paint_layer_is_transparent_and_a_folder_composites_its_children() {
    let mut session = session();
    let paint = id(execute(&mut session, "layer.add.paint", &Value::Null).unwrap(), "layer");
    let fill = id(execute(&mut session, "layer.add.fill", &json!({"color": "#336699"})).unwrap(), "layer");
    let folder = id(execute(&mut session, "layer.group", &json!({"layer": fill.0})).unwrap(), "layer");
    let paint_target = ThumbTarget::Content { layer: paint, channel: Channel::BaseColor };
    let folder_target = ThumbTarget::Content { layer: folder, channel: Channel::BaseColor };
    let (mut service, mut clock) = (ThumbService::new(), 0.0);

    settle(&mut service, &session, &[paint_target, folder_target], &mut clock);

    assert!(image(&mut service, &session, paint_target).rgba.as_chunks::<4>().0.iter().all(|pixel| pixel[3] == 0), "nothing painted: nothing shown");
    let folder_image = image(&mut service, &session, folder_target);
    let pixel = folder_image.rgba.as_chunks::<4>().0.iter().find(|pixel| pixel[3] == 255).unwrap();
    assert!((i32::from(pixel[0]) - 51).abs() <= 1 && (i32::from(pixel[2]) - 153).abs() <= 1, "{pixel:?}");
}

#[test]
fn instances_preview_the_source_content_and_inherited_mask() {
    use texturecraft_engine::thumbs::{GbufCache, ThumbSnapshot};

    let mut session = session();
    let stack = stack(&mut session);
    let instance = id(execute(&mut session, "layer.instance", &json!({"layer": stack.layer.0})).unwrap(), "layer");
    let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();

    for channel in [Channel::BaseColor, Channel::Roughness] {
        let source = snapshot.render(ThumbTarget::Content { layer: stack.layer, channel }).unwrap();
        let preview = snapshot.render(ThumbTarget::Content { layer: instance, channel });

        assert!(preview == Some(source), "the instance inherits {channel:?} from its source");
    }

    assert_eq!(snapshot.render(ThumbTarget::Mask { layer: instance }), snapshot.render(ThumbTarget::Mask { layer: stack.layer }));
}

/// A lower height anchor and an independent upper fill that can read it.
fn anchor_stack() -> (Session, LayerId, LayerId) {
    let mut session = session();
    execute(&mut session, "layer.add.fill", &json!({"height": 0.25})).unwrap();
    let anchor = id(execute(&mut session, "effect.add.anchor", &Value::Null).unwrap(), "effect");
    let layer = id(execute(&mut session, "layer.add.fill", &Value::Null).unwrap(), "layer");
    (session, anchor, layer)
}

#[test]
fn thumbnails_keep_anchor_inputs_from_lower_layers() {
    use texturecraft_engine::thumbs::{GbufCache, ThumbSnapshot};

    let (mut session, anchor, layer) = anchor_stack();
    execute(&mut session, "layer.fill.set", &json!({"channel": "roughness", "anchor": anchor.0, "anchorChannel": "height"})).unwrap();
    execute(&mut session, "mask.add.black", &Value::Null).unwrap();
    let mask_effect = id(execute(&mut session, "effect.add.fill", &json!({"anchor": anchor.0, "anchorChannel": "height"})).unwrap(), "effect");
    let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();

    for target in [
        ThumbTarget::Content { layer, channel: Channel::Roughness },
        ThumbTarget::Mask { layer },
        ThumbTarget::Stage { effect: mask_effect, channel: Channel::BaseColor },
    ] {
        let image = snapshot.render(target).unwrap();
        let pixel = image.rgba.as_chunks::<4>().0.iter().find(|pixel| pixel[3] == 255).unwrap();

        assert!((i32::from(pixel[0]) - 64).abs() <= 1, "{target:?} must read the lower anchor's 0.25, got {pixel:?}");
    }
}

#[test]
fn folder_thumbnails_keep_anchors_outside_the_folder() {
    use texturecraft_engine::thumbs::{GbufCache, ThumbSnapshot};

    let (mut session, anchor, layer) = anchor_stack();
    execute(&mut session, "layer.fill.set", &json!({"channel": "roughness", "anchor": anchor.0, "anchorChannel": "height"})).unwrap();
    let folder = id(execute(&mut session, "layer.group", &json!({"layer": layer.0})).unwrap(), "layer");
    let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();

    let image = snapshot.render(ThumbTarget::Content { layer: folder, channel: Channel::Roughness }).unwrap();
    let pixel = image.rgba.as_chunks::<4>().0.iter().find(|pixel| pixel[3] == 255).unwrap();

    assert!((i32::from(pixel[0]) - 64).abs() <= 1, "the folder must read the external anchor's 0.25, got {pixel:?}");
}

#[test]
fn effect_only_anchor_reads_keep_the_lower_stack() {
    use texturecraft_engine::thumbs::{GbufCache, ThumbSnapshot};

    for target_kind in ["content", "mask"] {
        let (mut session, anchor, layer) = anchor_stack();
        if target_kind == "mask" {
            execute(&mut session, "mask.add.black", &Value::Null).unwrap();
        }
        let effect = id(execute(&mut session, "effect.add.fill", &json!({"target": target_kind})).unwrap(), "effect");
        if target_kind == "content" {
            execute(&mut session, "layer.fill.set", &json!({"effect": effect.0, "channel": "roughness", "anchor": anchor.0, "anchorChannel": "height"}))
                .unwrap();
        } else {
            execute(&mut session, "effect.set", &json!({"effect": effect.0, "anchor": anchor.0, "anchorChannel": "height"})).unwrap();
        }
        let target = if target_kind == "content" { ThumbTarget::Content { layer, channel: Channel::Roughness } } else { ThumbTarget::Mask { layer } };
        let snapshot = ThumbSnapshot::new(&session, 0, &mut GbufCache::default()).unwrap();

        for target in [target, ThumbTarget::Stage { effect, channel: Channel::Roughness }] {
            let image = snapshot.render(target).unwrap();
            let pixel = image.rgba.as_chunks::<4>().0.iter().find(|pixel| pixel[3] == 255).unwrap();

            assert!((i32::from(pixel[0]) - 64).abs() <= 1, "{target_kind} {target:?} must read the lower anchor's 0.25, got {pixel:?}");
        }
    }
}
