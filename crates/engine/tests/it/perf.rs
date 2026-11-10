//! Compositing benchmarks at 4K and 8K (run with `--ignored --nocapture` in release).

use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::{Value, json};
use texturecraft_engine::compose::{DisplayKind, tile_bytes};
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, execute};

/// A ~10 layer stack: procedural fills, generators, a blurred mask, a folder and painted layers.
fn stack(res: u32) -> Session {
    let mut s = Session::new_project("perf", primitives::sphere(1.0, 64, 32), "builtin:sphere", None, res).unwrap();
    s.viewport = Vec2::new(800.0, 800.0);
    let run = |s: &mut Session, id: &str, p: Value| execute(s, id, &p).unwrap();
    run(&mut s, "layer.add.fill", json!({"color": [0.6, 0.5, 0.4], "roughness": 0.6}));
    run(&mut s, "layer.fill.set", json!({"channel": "baseColor", "procedural": "clouds", "low": [0.2, 0.2, 0.2], "high": [0.8, 0.7, 0.6]}));
    run(&mut s, "layer.fill.set", json!({"channel": "height", "procedural": "cells", "low": [-0.2, -0.2, -0.2], "high": [0.2, 0.2, 0.2]}));
    run(&mut s, "layer.add.fill", json!({"color": [0.1, 0.3, 0.6]}));
    run(&mut s, "layer.fill.set", json!({"channel": "roughness", "procedural": "grunge"}));
    run(&mut s, "mask.add.black", Value::Null);
    run(&mut s, "effect.add.generator", json!({"generator": "position", "level": 0.5}));
    run(&mut s, "effect.add.filter", json!({"filter": "blur", "radius": 8}));
    run(&mut s, "layer.add.fill", json!({"roughness": 0.3}));
    run(&mut s, "layer.fill.set", json!({"channel": "baseColor", "procedural": "wood"}));
    run(&mut s, "layer.fill.projection", json!({"projection": "triplanar", "scale": 2}));
    run(&mut s, "mask.add.black", Value::Null);
    run(&mut s, "effect.add.fill", json!({"procedural": "scratches", "scale": 6}));
    run(&mut s, "layer.add.fill", json!({"color": [0.2, 0.2, 0.2], "metallic": 1.0}));
    run(&mut s, "mask.add.black", Value::Null);
    run(&mut s, "effect.add.generator", json!({"generator": "dirt"}));
    run(&mut s, "layer.add.fill", json!({"roughness": 0.8}));
    run(&mut s, "layer.fill.set", json!({"channel": "roughness", "procedural": "marble"}));
    run(&mut s, "layer.add.group", Value::Null);
    run(&mut s, "layer.add.fill", json!({"color": [0.9, 0.9, 0.2]}));
    run(&mut s, "mask.add.black", Value::Null);
    run(&mut s, "effect.add.generator", json!({"generator": "direction"}));
    run(&mut s, "effect.add.levels", json!({"inLow": 0.3, "inHigh": 0.7}));
    let top = s.doc.project.texture_sets[0].stack.last().copied().unwrap();
    run(&mut s, "layer.select", json!({"layer": top.0}));
    run(&mut s, "layer.add.paint", Value::Null);
    run(&mut s, "tool.brush", json!({"size": 6, "hardness": 80}));
    run(&mut s, "tool.stroke", json!({"points": [[200, 300], [600, 500]]}));
    run(&mut s, "layer.add.paint", Value::Null);
    s.update();
    s
}

fn upload_bytes(s: &mut Session) -> usize {
    let tiles = s.take_uploads(0);
    let baked = s.baked(0).cloned().unwrap_or_default();
    let comp = &s.caches[0].composite;
    // As the UI does it (in parallel).
    tiles.par_iter().flat_map_iter(|&(tx, ty)| DisplayKind::ALL.map(|k| tile_bytes(comp, &baked, k, tx, ty).len())).sum()
}

fn mb(b: usize) -> f64 {
    b as f64 / (1024.0 * 1024.0)
}

fn bench(res: u32) {
    let t = Instant::now();
    let mut s = stack(res);
    println!("[{res}²] setup {:.2?} ({} layers, {} covered tiles)", t.elapsed(), s.doc.project.walk(0).len(), s.caches[0].covered_tiles.len());
    s.caches[0].invalidate_all();
    let t = Instant::now();
    s.update();
    println!("[{res}²] full recomposite (Session::update) {:.2?}", t.elapsed());
    let t = Instant::now();
    let up = upload_bytes(&mut s);
    println!("[{res}²] full display tile conversion {:.2?} ({:.0} MB)", t.elapsed(), mb(up));

    // The UI's budgeted update: a full recomposite spread over frames.
    s.caches[0].invalidate_all();
    let clock = Instant::now();
    let ms = || clock.elapsed().as_secs_f64() * 1000.0;
    let (mut frames, mut worst) = (0, Duration::ZERO);
    while s.pending_tiles() > 0 {
        let t = Instant::now();
        s.update_for(30.0, ms);
        worst = worst.max(t.elapsed());
        frames += 1;
    }
    println!("[{res}²] budgeted full recomposite: {frames} frames, worst frame {worst:.2?}, total {:.2?}", clock.elapsed());

    // Strokes on the top paint layer, twice over the same path: per-frame dirty-tile composite
    // + display conversion, then the commit. Brush sizes are relative to the mesh: the largest
    // one covers a third of the texture per stamp.
    for (size, brush) in [(0.5, "largest brush"), (0.02, "default brush")] {
        s.tool.brush.size = size;
        for pass in ["1st stroke", "2nd stroke"] {
            let (mut comp, mut conv, mut worst, mut tiles) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, 0usize);
            s.begin_stroke(Vec2::new(250.0, 400.0), s.viewport, 1.0).unwrap();
            let frames = 40;
            for i in 0..frames {
                let p = Vec2::new(250.0 + i as f32 * 8.0, 400.0 + (i as f32 * 0.3).sin() * 40.0);
                s.continue_stroke(p, 1.0);
                tiles += s.caches[0].dirty.len();
                let t = Instant::now();
                s.update();
                let c = t.elapsed();
                let t = Instant::now();
                upload_bytes(&mut s);
                let u = t.elapsed();
                comp += c;
                conv += u;
                worst = worst.max(c + u);
            }
            s.end_stroke().unwrap();
            let t = Instant::now();
            s.update();
            let commit = t.elapsed();
            let f = frames as u32;
            println!(
                "[{res}²] {brush}, {pass}: frame composite {:.2?} + display {:.2?} avg, worst {worst:.2?} ({:.0} dirty tiles/frame); commit {commit:.2?}",
                comp / f,
                conv / f,
                tiles as f64 / frames as f64
            );
        }
    }
    let m = s.memory();
    println!("[{res}²] memory: layer pixels {:.1} MB, composites {:.1} MB, hot-layer cache {:.1} MB", mb(m.pixels), mb(m.composites), mb(m.cache));
}

#[test]
#[ignore]
fn bench_4k() {
    bench(4096);
}

#[test]
#[ignore]
fn bench_8k() {
    bench(8192);
}

/// What one interactive painting frame costs in the app, stage by stage, on the startup sample
/// project (2K material ball with a smart material folder): stamping, compositing the dirty
/// tiles, display conversion; then the commit at release. Painted twice: a new layer inside the
/// smart material folder (on top of its fills), then the bottom layer (everything above it).
#[test]
#[ignore]
fn bench_sample_painting() {
    let mut s = texturecraft_engine::project_cmds::sample_session(2048, 512).unwrap();
    s.viewport = Vec2::new(1000.0, 880.0);
    let bottom = s.doc.project.texture_sets[0].stack[0];
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    s.update();
    upload_bytes(&mut s);
    for place in ["in the smart material folder", "at the bottom of the stack"] {
        if place == "at the bottom of the stack" {
            execute(&mut s, "layer.select", &json!({"layer": bottom.0})).unwrap();
        }
        for (size, brush) in [(0.02, "default brush"), (0.08, "large brush")] {
            s.tool.brush.size = size;
            let (mut stamp, mut comp, mut conv, mut worst, mut tiles) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, 0usize);
            let t = Instant::now();
            s.begin_stroke(Vec2::new(350.0, 400.0), s.viewport, 1.0).unwrap();
            let begin = t.elapsed();
            let frames = 60;
            for i in 0..frames {
                let p = Vec2::new(350.0 + i as f32 * 5.0, 400.0 + (i as f32 * 0.2).sin() * 60.0);
                let t = Instant::now();
                s.continue_stroke(p, 1.0);
                let a = t.elapsed();
                tiles += s.caches.iter().map(|c| c.dirty.len()).sum::<usize>();
                let t = Instant::now();
                s.update();
                let b = t.elapsed();
                let t = Instant::now();
                upload_bytes(&mut s);
                let c = t.elapsed();
                stamp += a;
                comp += b;
                conv += c;
                worst = worst.max(a + b + c);
            }
            let t = Instant::now();
            s.end_stroke().unwrap();
            let end = t.elapsed();
            let committed = s.caches[0].dirty.len();
            let t = Instant::now();
            s.update();
            let after = t.elapsed();
            let f = frames as u32;
            let below = &s.caches[0].below;
            println!(
                "[sample] {place}, {brush}: begin {begin:.2?}; per frame stamp {:.2?} + composite {:.2?} ({:.2?}/tile) + display {:.2?}, worst {worst:.2?} ({:.0} dirty tiles/frame); end_stroke {end:.2?} + update {after:.2?} ({committed} tiles); cache {} tiles, {:.1} MB, {:.0} KB/tile",
                stamp / f,
                comp / f,
                comp / tiles.max(1) as u32,
                conv / f,
                tiles as f64 / frames as f64,
                below.len(),
                mb(below.bytes()),
                below.tile_bytes().unwrap_or(0) as f64 / 1024.0
            );
        }
    }
}

/// Per-texel overhead of the compositor on a one-paint-layer project.
#[test]
#[ignore]
fn bench_minimal_stack() {
    let mut s = Session::new_project("min", primitives::sphere(1.0, 64, 32), "builtin:sphere", None, 2048).unwrap();
    s.viewport = Vec2::new(800.0, 800.0);
    s.update();
    for label in ["empty set", "one paint layer"] {
        if label == "one paint layer" {
            execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
            execute(&mut s, "tool.stroke", &json!({"points": [[200, 300], [600, 500]]})).unwrap();
            s.update();
        }
        s.caches[0].invalidate_all();
        let n = s.caches[0].dirty.len();
        let t = Instant::now();
        s.update();
        println!("[min] {label}: {n} tiles in {:.2?} ({:.2?}/tile)", t.elapsed(), t.elapsed() / n.max(1) as u32);
    }
}

/// Full-composite cost of every built-in material on the sample ball (1K, baked), slowest first.
#[test]
#[ignore]
fn bench_materials() {
    let mut rows = Vec::new();
    for m in texturecraft_engine::materials::MATERIALS {
        let mut s = texturecraft_engine::project_cmds::sample_session(1024, 256).unwrap();
        let top = s.doc.project.texture_sets[0].stack.clone();
        for id in top {
            let _ = execute(&mut s, "layer.remove", &json!({"layer": id.0}));
        }
        execute(&mut s, "material.apply", &json!({"material": m.id})).unwrap();
        s.caches[0].invalidate_all();
        let n = s.caches[0].dirty.len();
        let t = Instant::now();
        s.update();
        rows.push((t.elapsed() / n.max(1) as u32, m.id));
    }
    rows.sort();
    rows.reverse();
    for (per_tile, id) in rows {
        println!("[materials] {id:<28} {per_tile:.2?}/tile");
    }
}

/// Per-tile cost of single-layer stacks, to find fixed per-texel overhead (run single-threaded).
#[test]
#[ignore]
fn bench_single_layers() {
    let cases: &[(&str, &[(&str, Value)])] = &[
        ("uniform fill", &[("layer.add.fill", json!({"color": [0.5, 0.5, 0.5], "roughness": 0.4}))]),
        ("fill + clouds", &[("layer.add.fill", json!({})), ("layer.fill.set", json!({"channel": "baseColor", "procedural": "clouds"}))]),
        (
            "fill + triplanar clouds",
            &[
                ("layer.add.fill", json!({})),
                ("layer.fill.set", json!({"channel": "baseColor", "procedural": "clouds"})),
                ("layer.fill.projection", json!({"projection": "triplanar"})),
            ],
        ),
        (
            "fill + black mask + edge wear",
            &[("layer.add.fill", json!({})), ("mask.add.black", Value::Null), ("effect.add.generator", json!({"generator": "edgeWear"}))],
        ),
        ("fill + black mask + dirt", &[("layer.add.fill", json!({})), ("mask.add.black", Value::Null), ("effect.add.generator", json!({"generator": "dirt"}))]),
    ];
    for (label, cmds) in cases {
        let mut s = texturecraft_engine::project_cmds::sample_session(1024, 256).unwrap();
        for id in s.doc.project.texture_sets[0].stack.clone() {
            let _ = execute(&mut s, "layer.remove", &json!({"layer": id.0}));
        }
        for (id, p) in cmds.iter() {
            execute(&mut s, id, p).unwrap();
        }
        s.update();
        s.caches[0].invalidate_all();
        let n = s.caches[0].dirty.len();
        let t = Instant::now();
        s.update();
        println!("[single] {label:<32} {:.2?}/tile ({n} tiles)", t.elapsed() / n.max(1) as u32);
    }
}

/// The frame where an 8K stroke ends: `end_stroke` and the first budgeted update after it.
#[test]
#[ignore]
fn bench_8k_stroke_end() {
    let mut s = texturecraft_engine::project_cmds::sample_session(2048, 256).unwrap();
    s.viewport = Vec2::new(1700.0, 1230.0);
    execute(&mut s, "textureSet.resolution", &json!({"resolution": 8192})).unwrap();
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    s.update();
    upload_bytes(&mut s);
    for size in [0.02, 0.08] {
        s.tool.brush.size = size;
        for rep in 0..3 {
            s.begin_stroke(Vec2::new(500.0, 500.0 + rep as f32 * 40.0), s.viewport, 1.0).unwrap();
            for i in 0..60 {
                let p = Vec2::new(500.0 + i as f32 * 10.0, 500.0 + rep as f32 * 40.0 + i as f32 * 3.3);
                s.continue_stroke(p, 1.0);
                s.update();
                upload_bytes(&mut s);
            }
            let t = Instant::now();
            s.end_stroke().unwrap();
            let end = t.elapsed();
            let tiles = s.caches[0].dirty.len();
            let t = Instant::now();
            s.update();
            let after = t.elapsed();
            let t = Instant::now();
            let up = upload_bytes(&mut s);
            println!("[8k] size {size}: end_stroke {end:.2?}, update {after:.2?} ({tiles} tiles), display {:.2?} ({:.1} MB)", t.elapsed(), mb(up));
        }
    }
}

#[test]
#[ignore]
fn bench_8k_resize() {
    let mut s = texturecraft_engine::project_cmds::sample_session(2048, 256).unwrap();
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    execute(&mut s, "tool.stroke", &json!({"points": [[300, 300], [700, 500]]})).unwrap();
    s.update();
    let t = Instant::now();
    execute(&mut s, "textureSet.resolution", &json!({"resolution": 8192})).unwrap();
    println!("[resize] 2K → 8K command {:.2?}", t.elapsed());
    // The UI's way: the worker thread's part, then what the frame that applies it pays.
    let t = Instant::now();
    let out = s.resize_task(0, 4096).unwrap().run(&texturecraft_engine::bake::Progress::new()).unwrap();
    let worker = t.elapsed();
    let t = Instant::now();
    s.apply_resize(out).unwrap();
    println!("[resize] 8K → 4K worker {worker:.2?}, apply {:.2?}", t.elapsed());
    let t = Instant::now();
    let out = s.resize_task(0, 8192).unwrap().run(&texturecraft_engine::bake::Progress::new()).unwrap();
    let worker = t.elapsed();
    let t = Instant::now();
    s.apply_resize(out).unwrap();
    println!("[resize] 4K → 8K worker {worker:.2?}, apply {:.2?}", t.elapsed());
    let t = Instant::now();
    let gbuf = texturecraft_engine::geom::UvGBuffer::new(&s.mesh, 0, 8192, 8192);
    println!("[resize] gbuffer {:.2?}", t.elapsed());
    let t = Instant::now();
    let d = texturecraft_engine::texel::DilationMap::in_blocks(&gbuf.coverage_mask(), 8192, 8192, Some(8), 8192);
    println!("[resize] dilation {:.2?}", t.elapsed());
    drop(d);
}

/// Layer-panel thumbnails on the 4K stack: the snapshot, each kind of picture, and a whole panel.
#[test]
#[ignore]
fn bench_layer_thumbs_4k() {
    use texturecraft_engine::compose::ThumbTarget;
    use texturecraft_engine::core::Channel;
    use texturecraft_engine::thumbs::{GbufCache, ThumbSnapshot};
    let s = stack(4096);
    let ts = s.ts();
    let mut gbufs = GbufCache::default();
    let t = Instant::now();
    let snap = ThumbSnapshot::new(&s, ts, &mut gbufs).unwrap();
    let first = t.elapsed();
    let t = Instant::now();
    let _again = ThumbSnapshot::new(&s, ts, &mut gbufs).unwrap();
    let cached = t.elapsed();
    let p = &s.doc.project;
    let mut targets = Vec::new();
    for l in p.layers.values() {
        targets.push(("content", ThumbTarget::Content { layer: l.id, channel: Channel::BaseColor }));
        if l.mask.is_some() {
            targets.push(("mask", ThumbTarget::Mask { layer: l.id }));
        }
        for e in l.mask.iter().flat_map(|m| m.effects.iter()).chain(l.content_effects.iter()) {
            targets.push(("stage", ThumbTarget::Stage { effect: e.id, channel: Channel::BaseColor }));
        }
    }
    let mut per: std::collections::BTreeMap<&str, (Duration, usize, Duration)> = Default::default();
    let all = Instant::now();
    for (kind, target) in &targets {
        let t = Instant::now();
        let _ = snap.render(*target);
        let d = t.elapsed();
        let e = per.entry(kind).or_default();
        e.0 += d;
        e.1 += 1;
        e.2 = e.2.max(d);
    }
    let total = all.elapsed();
    eprintln!("layer thumbnails, 4096 set, {} layers, 64 px:", p.layers.len());
    eprintln!("  snapshot: {:.2} ms first (with the 64 px G-buffer), {:.2} ms after", first.as_secs_f64() * 1e3, cached.as_secs_f64() * 1e3);
    for (kind, (sum, n, max)) in per {
        eprintln!("  {kind:8} {n:3} × avg {:.2} ms, max {:.2} ms", sum.as_secs_f64() * 1e3 / n as f64, max.as_secs_f64() * 1e3);
    }
    eprintln!("  every target once: {:.1} ms ({} pictures)", total.as_secs_f64() * 1e3, targets.len());
}
