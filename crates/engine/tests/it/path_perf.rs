//! Path preview timings and a repeatable memory soak (release, --ignored --nocapture).

use std::time::{Duration, Instant};

use serde_json::json;
use texturecraft_engine::execute;
use texturecraft_engine::geom::glam::Vec2;

/// Sort a nonempty timing sample in place and report its middle value in milliseconds
fn median_ms(samples: &mut [Duration]) -> f64 {
    samples.sort_unstable();
    samples[samples.len() / 2].as_secs_f64() * 1_000.0
}

/// Separate repaint cost from dirty-tile and full-set compositing across representative brush sizes
#[test]
#[ignore = "release benchmark: two texture resolutions and brush sizes"]
fn path_preview_timings() {
    for resolution in [2048, 4096] {
        let mut session = texturecraft_engine::sample_session(resolution, 64).unwrap();
        session.viewport = Vec2::splat(400.0);
        session.camera.yaw = 0.0;
        session.camera.pitch = 0.0;
        session.update();
        for size in [4, 20] {
            execute(&mut session, "tool.brush", &json!({"size": size})).unwrap();
            execute(&mut session, "tool.path", &json!({"points": [[150, 210], [175, 180], [200, 210], [225, 180], [250, 210]]})).unwrap();
            session.update();
            let original = session.tool.path.points[2];
            let (mut paint, mut composite, mut full) = (Vec::new(), Vec::new(), Vec::new());
            let mut dirty_tiles = Vec::new();

            for step in 0..20 {
                let point = original + texturecraft_engine::geom::glam::Vec3::Y * (step as f32 + 1.0) * 0.001;
                assert!(session.move_path_point(2, point, step == 0));
                let started = Instant::now();
                assert!(session.path_live().unwrap());
                paint.push(started.elapsed());
                dirty_tiles.push(session.pending_tiles());
                let started = Instant::now();
                session.update();
                composite.push(started.elapsed());
                // The same document with full invalidation isolates the compositor saving
                if step < 3 {
                    for cache in &mut session.caches {
                        cache.invalidate_all();
                    }
                    let started = Instant::now();
                    session.update();
                    full.push(started.elapsed());
                }
            }

            println!(
                "path {resolution}px brush={size}: paint {:.2} ms, dirty composite {:.2} ms, full composite {:.2} ms; dirty tiles {:?}",
                median_ms(&mut paint),
                median_ms(&mut composite),
                median_ms(&mut full),
                dirty_tiles
            );
            session.cancel_path();
            session.update();
        }
    }
}

/// Observe resident memory without treating allocator-retained pages as a leak
#[cfg(target_os = "linux")]
fn resident_kib() -> usize {
    let status = std::fs::read_to_string("/proc/self/status").unwrap();
    status.lines().find_map(|line| line.strip_prefix("VmRSS:").and_then(|rest| rest.split_whitespace().next()).and_then(|n| n.parse().ok())).unwrap()
}

/// Check that repeated canceled previews release tile buffers and leave no document history
#[cfg(target_os = "linux")]
#[test]
#[ignore = "release memory soak: 1,000 edits across 100 canceled previews"]
fn path_preview_memory_soak() {
    use texturecraft_engine::Session;
    use texturecraft_engine::geom::primitives;

    let mut session = Session::new_project("Path soak", primitives::sphere(1.0, 48, 24), "builtin:sphere", None, 512).unwrap();
    session.viewport = Vec2::splat(400.0);
    session.update();
    let history = session.history_labels().len();
    let mut resident = Vec::new();

    for cycle in 0..100 {
        execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.35, 0.45], [0.5, 0.5], [0.65, 0.45], [0.8, 0.5]]})).unwrap();
        for step in 0..10 {
            let point = texturecraft_engine::geom::glam::Vec3::new(0.5, 0.5 + (step as f32 + 1.0) * 0.002, 0.0);
            session.move_path_point(2, point, true);
            assert!(session.path_live().unwrap());
            session.update();
        }
        let preview_tiles: Vec<_> = session
            .doc
            .pixels
            .layers
            .values()
            .flat_map(|image| image.allocated().map(|(x, y)| std::sync::Arc::downgrade(image.tile_handle(x, y).unwrap())))
            .collect();
        assert!(!preview_tiles.is_empty());
        session.cancel_path();
        session.update();
        assert!(preview_tiles.iter().all(|tile| tile.upgrade().is_none()), "canceled previews release their painted tile buffers");
        assert_eq!(session.history_labels().len(), history, "canceled previews retain no document history");
        if (cycle + 1) % 10 == 0 {
            resident.push(resident_kib());
        }
    }

    println!("path soak VmRSS KiB after each 100 edits: {resident:?}");
    // RSS is an observation, not a portable leak assertion: allocators can retain freed pages
}
