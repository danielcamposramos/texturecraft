//! Rendering-mode cache and frame-lifecycle regressions.

use super::frame_floor::{FLOOR_FRAMES, FLOOR_RANGE_S, FrameFloor};
use super::spp_control::{MAX_SPP, SPP_GROW_PAUSE_FRAMES, SppControl, next_spp, rescale_spp};
use super::{
    LOADING_RESTART_S, MOTION_SETTLE_S, Motion, OrbitCamera, SETTLE_HOLD_MAX_S, SETTLE_HOLD_SAMPLES, gpu_scene_key, holds_previous, scene_key, show_image,
    take_in_textures, textures_loading, traced_size,
};
use texturecraft_engine::Session;
use texturecraft_engine::geom::primitives;

/// A small, fully composed sphere avoids unrelated texture-loading transitions.
fn session() -> Session {
    let mut session = Session::new_project("t", primitives::sphere(1.0, 16, 8), "builtin:sphere", None, 128).unwrap();
    session.update();

    session
}

/// Bump-strength edits invalidate GPU scene data even without displaced geometry.
#[test]
fn height_strength_changes_restart_the_gpu_scene_without_displacement() {
    let mut session = session();
    let before = gpu_scene_key(&session);
    assert!(!session.doc.project.display.displacement);

    texturecraft_engine::execute(&mut session, "textureSet.settings", &serde_json::json!({"heightStrength": 5.0})).unwrap();
    session.update();

    assert_ne!(gpu_scene_key(&session), before, "the GPU shader must receive the new bump strength");
}

/// Keeping an imported image alive prevents its cached identity from being reused for another image.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn environment_cache_retains_source_identity_until_replaced() {
    use std::sync::Arc;
    use texturecraft_engine::compose::ImageRes;

    // The harness draws once; parallel translation tests must not change its language mid-frame
    let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut app = egui_kittest::Harness::builder().build_eframe(|context| crate::TextureCraftApp::new(context, session())).into_state();
    app.session.doc.project.display.environment = "resource:environment".into();
    let source = Arc::new(ImageRes { width: 2, height: 1, rgba: vec![[1.0, 0.0, 0.0, 1.0]; 2], srgb: false });
    let original_source = Arc::downgrade(&source);
    app.session.images.insert("environment".into(), source);
    let first = super::trace_scene(&app).env;
    assert!(Arc::ptr_eq(&first, &super::trace_scene(&app).env), "unchanged images reuse the decoded environment");

    let replacement = Arc::new(ImageRes { width: 2, height: 1, rgba: vec![[0.0, 1.0, 0.0, 1.0]; 2], srgb: false });
    app.session.images.insert("environment".into(), replacement.clone());

    assert!(original_source.upgrade().is_some(), "the cache must retain its source so its address cannot be reused");
    let second = super::trace_scene(&app).env;
    assert!(!Arc::ptr_eq(&first, &second));
    assert_ne!(first.data, second.data, "replacing the image changes the lighting");
    assert!(Arc::ptr_eq(&second, &super::trace_scene(&app).env));
    assert!(original_source.upgrade().is_none(), "replacing the cache releases the previous source");

    app.session.images.insert("renamed".into(), replacement);
    app.session.doc.project.display.environment = "resource:renamed".into();
    assert!(!Arc::ptr_eq(&second, &super::trace_scene(&app).env), "the environment name is also part of the key");
}

/// Create one harness device, skipping unavailable or non-compute backends.
#[cfg(not(target_arch = "wasm32"))]
fn gpu_app() -> Option<crate::TextureCraftApp> {
    use eframe::egui_wgpu::{RenderState, RendererOptions, WgpuConfiguration};
    use egui_kittest::wgpu::{WgpuTestRenderer, default_wgpu_setup};

    let setup = default_wgpu_setup();
    let instance = pollster::block_on(setup.new_instance());
    let configuration = WgpuConfiguration { wgpu_setup: setup, ..Default::default() };
    let render_state = match pollster::block_on(RenderState::create(&configuration, &instance, None, RendererOptions::PREDICTABLE)) {
        Ok(render_state) => render_state,
        Err(error) => {
            eprintln!("skipping: could not create the harness GPU device: {error}");
            return None;
        }
    };

    let adapter = render_state.adapter.get_info();
    if !super::GpuTracer::supported(&render_state.device) {
        eprintln!("skipping: harness adapter {} ({:?}) cannot run the GPU tracer", adapter.name, adapter.backend);
        return None;
    }

    eprintln!("GPU lifecycle test adapter: {} ({:?})", adapter.name, adapter.backend);
    // The harness draws once; parallel translation tests must not change its language mid-frame
    let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
    let mut app = egui_kittest::Harness::builder()
        .with_size(egui::vec2(800.0, 600.0))
        .renderer(WgpuTestRenderer::from_render_state(render_state))
        .build_eframe(|context| crate::TextureCraftApp::new(context, session()))
        .into_state();
    app.session.update();
    for cache in &mut app.session.caches {
        cache.upload.clear();
    }

    Some(app)
}

/// Build a GPU rendering frame at a controlled time without executing its callback.
#[cfg(not(target_arch = "wasm32"))]
fn draw_gpu(app: &mut crate::TextureCraftApp, context: &egui::Context, time: f64) -> egui::FullOutput {
    context.run_ui(egui::RawInput { time: Some(time), ..Default::default() }, |ui| {
        assert!(super::show_gpu(app, ui, egui::Rect::from_min_size(egui::Pos2::ZERO, egui::vec2(128.0, 96.0))));
    })
}

/// A first completed callback must survive a slow frame's switch to full resolution.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn settling_after_the_first_gpu_frame_keeps_its_image() {
    let Some(mut app) = gpu_app() else { return };

    let context = egui::Context::default();
    draw_gpu(&mut app, &context, 0.0).drop_without_applying_deltas();
    let view = app.render.gpu.as_ref().unwrap();
    let previous_texture = view.texture;

    // The callback finishes the first frame before a slow frame returns to the UI
    view.status.lock().unwrap().samples = 1;
    assert!(!view.has_image);

    draw_gpu(&mut app, &context, MOTION_SETTLE_S + 0.1).drop_without_applying_deltas();

    let view = app.render.gpu.as_ref().unwrap();
    assert_eq!(view.size, (128, 96));
    assert!(view.has_image, "resizing must read the completed callback before replacing its status");
    assert_eq!(view.held, Some(previous_texture));
}

/// A finished low-sample render must wake again to release its retained preview.
#[cfg(not(target_arch = "wasm32"))]
#[test]
fn a_low_sample_limit_schedules_the_held_image_timeout() {
    let Some(mut app) = gpu_app() else { return };

    app.ui.render.max_samples = 1;
    let context = egui::Context::default();
    draw_gpu(&mut app, &context, 0.0).drop_without_applying_deltas();
    let view = app.render.gpu.as_mut().unwrap();
    view.has_image = true;
    view.status.lock().unwrap().samples = 1;
    let settled_at = MOTION_SETTLE_S + 0.1;
    draw_gpu(&mut app, &context, settled_at).drop_without_applying_deltas();
    app.render.gpu.as_ref().unwrap().status.lock().unwrap().samples = 1;

    for frame in 1..=5 {
        let output = draw_gpu(&mut app, &context, settled_at + f64::from(frame) * 0.01);
        let delay = output.viewport_output[&egui::ViewportId::ROOT].repaint_delay;
        output.drop_without_applying_deltas();
        assert!(delay.as_secs_f64() <= SETTLE_HOLD_MAX_S, "a finished render must wake to release its held image: {delay:?}");
    }

    draw_gpu(&mut app, &context, settled_at + SETTLE_HOLD_MAX_S + 0.01).drop_without_applying_deltas();
    assert!(app.render.gpu.as_ref().unwrap().held.is_none());
}

/// Every composited chunk bumps a revision. The CPU render restarts on it (it reads the
/// composites); the GPU render does not (it copies the display textures when they change,
/// throttled while loading) unless the heights displace its geometry.
#[test]
fn composited_chunks_restart_the_gpu_render_only_when_displaced() {
    let mut session = session();
    let (cpu, gpu) = (scene_key(&session, true), gpu_scene_key(&session));
    session.caches[0].revision += 1;

    assert_ne!(scene_key(&session, true), cpu);
    assert_eq!(gpu_scene_key(&session), gpu, "the GPU scene is kept while textures load");

    session.doc.project.display.displacement = true;
    let displaced = gpu_scene_key(&session);
    session.caches[0].revision += 1;

    assert_ne!(gpu_scene_key(&session), displaced, "displaced geometry follows the heights");
}

/// Loading lasts until the composited tiles are uploaded too (the GPU render reads them).
#[test]
fn textures_load_until_their_tiles_are_uploaded() {
    let mut session = session();
    session.caches[0].upload.clear();
    assert!(!textures_loading(&session));

    session.caches[0].upload.insert((0, 0));
    assert!(textures_loading(&session), "a tile waits to be uploaded");

    session.caches[0].upload.clear();
    session.caches[0].invalidate_all();
    assert!(textures_loading(&session), "tiles wait to be composited");
}

/// Orbiting restarts the render every frame (samples back to 0): the last image stays up.
#[test]
fn a_restart_keeps_the_previous_image_on_screen() {
    assert!(show_image(true, 0), "restart with an earlier frame must not show the placeholder");
    assert!(show_image(true, 5));
    assert!(show_image(false, 1));
}

/// Motion halves each dimension without shrinking already tiny views.
#[test]
fn a_moving_camera_traces_smaller_and_a_still_one_at_full_size() {
    assert_eq!(traced_size((1600, 900), false), (1600, 900));
    assert_eq!(traced_size((1600, 900), true), (800, 450));

    // Tiny views are never made smaller than they are or than 16 pixels
    assert_eq!(traced_size((10, 10), true), (10, 10));
    assert_eq!(traced_size((20, 400), true), (16, 200));
}

/// Camera changes and unfinished uploads keep the preview small until activity settles.
#[test]
fn motion_starts_on_a_camera_change_and_settles_after_a_moment() {
    let mut motion = Motion::default();
    let mut camera = OrbitCamera::default();
    assert!(motion.moving(&camera, 0.0, false), "a render that just started opens at the smaller size");
    assert!(!motion.moving(&camera, 1.0, false));

    camera.yaw += 0.1;
    assert!(motion.moving(&camera, 2.0, false));
    assert!(motion.moving(&camera, 2.0 + MOTION_SETTLE_S * 0.5, false));
    assert!(!motion.moving(&camera, 2.0 + MOTION_SETTLE_S * 1.5, false));

    assert!(motion.moving(&camera, 5.0, true), "textures still loading count as motion");
    assert!(motion.moving(&camera, 5.0 + MOTION_SETTLE_S * 0.5, false), "and settle a moment after they finish");
    assert!(!motion.moving(&camera, 5.0 + MOTION_SETTLE_S * 1.5, false));
}

/// Either sufficient samples or the deadline replaces the retained preview.
#[test]
fn the_soft_image_stays_until_the_full_size_render_has_samples_or_time_runs_out() {
    assert!(holds_previous(0, 0.0));
    assert!(holds_previous(SETTLE_HOLD_SAMPLES - 1, SETTLE_HOLD_MAX_S * 0.5));
    assert!(!holds_previous(SETTLE_HOLD_SAMPLES, 0.0), "enough samples: show the new render");
    assert!(!holds_previous(0, SETTLE_HOLD_MAX_S), "a slow GPU: don't hold it forever");
}

/// Sampling grows gradually, holds through jitter and cuts promptly on overload.
#[test]
fn samples_per_frame_grow_slowly_hold_in_between_and_drop_fast() {
    let (floor, free) = (1.0 / 144.0, SPP_GROW_PAUSE_FRAMES);

    assert_eq!(next_spp(4, floor, floor, free), 5, "frames at the refresh rate: one more");
    assert_eq!(next_spp(MAX_SPP, floor, floor, free), MAX_SPP);
    assert_eq!(next_spp(4, floor, floor, 0), 4, "just after a cut: no growth yet");
    assert_eq!(next_spp(6, floor * 1.4, floor, free), 6, "between the limits: steady");
    assert_eq!(next_spp(8, floor * 2.5, floor, free), 5, "slow frame: cut by the overshoot (8 * 1.75/2.5), more than a quarter");
    assert_eq!(next_spp(8, floor * 1.7, floor, free), 8, "a little slow: steady, not a cut (cuts are for real overload)");
    assert_eq!(next_spp(8, floor * 1.8, floor, free), 6, "slightly over the limit: a quarter fewer");
    assert_eq!(next_spp(8, floor * 20.0, floor, free), 1, "very slow frame: by the overshoot, at least one");
    assert_eq!(next_spp(1, floor * 20.0, floor, free), 1);
}

/// With vsync on a 60 Hz display every frame takes ~16.7 ms however light: that is the
/// refresh, not a slow GPU, so the samples grow; a missed refresh (two intervals) cuts them.
#[test]
fn samples_per_frame_grow_at_60_hz_and_drop_on_a_missed_refresh() {
    let refresh_seconds = 1.0 / 60.0;
    let mut control = SppControl::default();
    let mut spp = 2;

    for _ in 0..10 {
        spp = control.next(spp, refresh_seconds);
    }

    assert_eq!(spp, 12, "one more per frame at the refresh rate");

    let cut = control.next(spp, 2.0 * refresh_seconds);
    assert!(cut < spp && cut >= spp / 2, "a missed refresh cuts: {spp} -> {cut}");

    // No climb straight back to the frame that missed: steady for a while, then growth again
    let mut after = cut;
    for _ in 0..SPP_GROW_PAUSE_FRAMES - 1 {
        after = control.next(after, refresh_seconds);
    }

    assert_eq!(after, cut, "steady after the cut");

    for _ in 0..3 {
        after = control.next(after, refresh_seconds);
    }

    assert!(after > cut, "then it grows again");
}

/// Frames that are not paced by the display (or take almost no work) can be a few ms; a frame
/// that then does real work must not read as slow and cut the samples to one (the render was
/// left grainy for seconds).
#[test]
fn a_few_very_quick_frames_do_not_collapse_the_samples_per_frame() {
    let mut control = SppControl::default();
    for _ in 0..5 {
        control.next(4, 0.003);
    }

    let working_seconds = 0.012;
    let mut spp = 4;

    for _ in 0..10 {
        spp = control.next(spp, working_seconds);
    }

    assert!(spp >= 4, "12 ms frames after 3 ms ones are fine: {spp}");
}

/// Invalid timings are ignored and old timings expire within the bounded window.
#[test]
fn the_frame_floor_is_the_quickest_recent_frame_within_range() {
    let mut floor = FrameFloor::default();

    assert_eq!(floor.push(1.0 / 60.0), 1.0 / 60.0);
    assert_eq!(floor.push(0.05), 1.0 / 60.0, "a slow frame does not raise it");
    assert_eq!(floor.push(f64::NAN), 1.0 / 60.0, "a broken time is ignored");
    assert_eq!(floor.push(0.0001), FLOOR_RANGE_S.0, "never below the quick end of the range");

    for _ in 0..FLOOR_FRAMES {
        floor.push(0.2);
    }

    assert_eq!(floor.push(0.2), FLOOR_RANGE_S.1, "only recent frames count, and at most 1/30 s");
}

/// Resolution changes preserve approximate tracing work within the sample limits.
#[test]
fn samples_per_frame_follow_the_pixel_count_when_the_size_changes() {
    assert_eq!(rescale_spp(12, (500, 400), (1000, 800)), 3, "four times the pixels: a quarter of the samples");
    assert_eq!(rescale_spp(3, (1000, 800), (500, 400)), 12);
    assert_eq!(rescale_spp(1, (500, 400), (1000, 800)), 1, "never below one");
    assert_eq!(rescale_spp(16, (100, 100), (10, 10)), MAX_SPP, "never above the cap");
}

/// Loading throttles texture synchronization without delaying the first or settled update.
#[test]
fn loading_textures_are_taken_in_at_a_limited_rate() {
    assert!(take_in_textures(false, 1.0, 1.0), "not loading: every frame");
    assert!(!take_in_textures(true, 1.0 + LOADING_RESTART_S * 0.5, 1.0));
    assert!(take_in_textures(true, 1.0 + LOADING_RESTART_S, 1.0));
    assert!(take_in_textures(true, 0.0, f64::MIN), "the first frame always");
}

/// The placeholder appears only before the first image has been traced.
#[test]
fn nothing_traced_yet_shows_the_placeholder() {
    assert!(!show_image(false, 0));
}
