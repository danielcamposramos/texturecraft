use texturecraft_compose::{DisplayKind, ImageRes};
use texturecraft_geom::glam::Vec3;
use texturecraft_geom::{OrbitCamera, primitives};
use texturecraft_render::headless::Headless;
use texturecraft_render::renderer::TAA_SAMPLES;
use texturecraft_render::{BakeVisualization, Frame, Shade, ViewKind};

/// The UI's key light (`viewport::shade`).
const LIGHT: [f32; 4] = [0.5, 0.8, 0.4, 2.5];

struct Img {
    w: u32,
    px: Vec<u8>,
}

/// Two renders of the same image: equal within one 8-bit level per channel (software
/// adapters such as llvmpipe round some shader paths differently from hardware).
#[track_caller]
fn assert_same(a: &Img, b: &Img, msg: &str) {
    assert_eq!(a.px.len(), b.px.len(), "{msg}: sizes differ");
    let worst = a.px.iter().zip(&b.px).map(|(x, y)| x.abs_diff(*y)).max().unwrap_or(0);
    assert!(worst <= 1, "{msg}: pixels differ by up to {worst} levels");
}

impl Img {
    fn at(&self, x: u32, y: u32) -> [u8; 3] {
        let i = ((y * self.w + x) * 4) as usize;
        [self.px[i], self.px[i + 1], self.px[i + 2]]
    }
    fn luma(&self, x: u32, y: u32) -> f32 {
        let [r, g, b] = self.at(x, y);
        0.2126 * r as f32 + 0.7152 * g as f32 + 0.0722 * b as f32
    }
    /// Mean luma over a rectangle.
    fn mean(&self, x0: u32, y0: u32, x1: u32, y1: u32) -> f32 {
        let mut sum = 0.0;
        for y in y0..y1 {
            for x in x0..x1 {
                sum += self.luma(x, y);
            }
        }
        sum / ((x1 - x0) * (y1 - y0)) as f32
    }
}

/// The id `setup` uploads the material ball under
const MESH_ID: u64 = 1;

/// A headless renderer with the material ball framed in a 480×360 view (None without a GPU).
fn setup() -> Option<(Headless, Frame)> {
    let Ok(mut h) = Headless::new() else {
        eprintln!("skipping: no GPU adapter");
        return None;
    };
    eprintln!("headless adapter: {:?}", h.device.adapter_info());
    let mesh = primitives::material_ball();
    h.renderer.set_mesh(&h.device, &h.queue, &mesh, MESH_ID);
    h.renderer.set_texture_sets(&h.device, &h.queue, &[((256, 256), 0), ((256, 256), 1)], 2);
    let mut cam = OrbitCamera { yaw: 0.6, pitch: 0.25, ..OrbitCamera::default() };
    let (lo, hi) = mesh.bounds();
    cam.frame(lo, hi);
    let frame =
        Frame { view_id: 1, size: [480, 360], kind: ViewKind::Scene, camera: cam, shade: Shade::default(), visible: vec![true, true], overlay: Vec::new() };
    Some((h, frame))
}

fn render(h: &mut Headless, frame: &Frame, name: &str) -> Img {
    let (w, hgt, px) = h.render(frame).expect("readback");
    assert_eq!((w, hgt), (frame.size[0], frame.size[1]));
    if let Ok(dir) = std::env::var("TC_SNAPSHOT_DIR") {
        image::save_buffer(format!("{dir}/{name}.png"), &px, w, hgt, image::ColorType::Rgba8).expect("save snapshot");
    }
    Img { w, px }
}

fn shaded(frame: &Frame, f: impl FnOnce(&mut Shade)) -> Frame {
    let mut out = frame.clone();
    f(&mut out.shade);
    out
}

#[test]
fn renders_the_material_ball() {
    let Some((mut h, frame)) = setup() else { return };
    let img = render(&mut h, &frame, "headless_ball");
    assert_ne!(img.at(240, 140), img.at(5, 5), "mesh should differ from background");
    // UV view of set 0.
    let uv = Frame { view_id: 2, kind: ViewKind::Uv { ts: 0, zoom: 1.0, pan: [0.0, 0.0] }, ..frame.clone() };
    let uv_img = render(&mut h, &uv, "headless_uv");
    assert_eq!(uv_img.px.len(), img.px.len());
    let wire = render(&mut h, &shaded(&uv, |s| s.wireframe = true), "headless_uv_wire");
    assert_ne!(wire.px, uv_img.px, "the UV wireframe draws");
}

#[test]
fn exposure_brightens_the_material_but_not_data_views() {
    let Some((mut h, frame)) = setup() else { return };
    let base = render(&mut h, &frame, "exposure_0");
    let bright = render(&mut h, &shaded(&frame, |s| s.exposure_ev = 1.0), "exposure_plus1");
    let (b0, b1) = (base.mean(200, 100, 280, 180), bright.mean(200, 100, 280, 180));
    assert!(b1 > b0 + 10.0, "exposure +1 brightens the ball ({b0} → {b1})");
    assert_eq!(base.at(5, 5), bright.at(5, 5), "the flat backdrop ignores exposure");
    let ch = shaded(&frame, |s| {
        s.view_mode = 1;
        s.channel = 0;
    });
    let a = render(&mut h, &ch, "channel_base_color");
    let b = render(&mut h, &shaded(&ch, |s| s.exposure_ev = 2.0), "channel_base_color_ev2");
    assert_eq!(a.px, b.px, "data views ignore exposure");
    // Default base color 188 (sRGB) shows unchanged in the channel view.
    let [r, ..] = a.at(240, 140);
    assert!(r.abs_diff(188) <= 1, "base color channel shows the raw value, got {r}");
}

#[test]
fn post_effects() {
    let Some((mut h, frame)) = setup() else { return };
    let base = render(&mut h, &frame, "post_off");
    // Strengths without the master switch change nothing.
    let gated = render(
        &mut h,
        &shaded(&frame, |s| {
            s.vignette = 1.0;
            s.bloom = 1.0;
            s.grain = 1.0;
            s.sharpen = 1.0;
        }),
        "post_gated",
    );
    assert_eq!(base.px, gated.px, "effects are off without post_effects");
    let on = |f: fn(&mut Shade)| {
        shaded(&frame, |s| {
            s.post_effects = true;
            f(s);
        })
    };
    let vignette = render(&mut h, &on(|s| s.vignette = 1.0), "post_vignette");
    assert!(vignette.mean(0, 0, 20, 20) < base.mean(0, 0, 20, 20) - 5.0, "vignette darkens the corners");
    assert!((vignette.luma(240, 180) - base.luma(240, 180)).abs() < 1.5, "vignette keeps the center");
    // A bright environment and +2 EV so highlights pass the bloom threshold.
    let hot = |s: &mut Shade| {
        s.exposure_ev = 2.0;
        s.env_opacity = 1.0;
    };
    let hot_base = render(&mut h, &shaded(&frame, hot), "post_hot");
    let bloom = render(
        &mut h,
        &on(|s| {
            s.exposure_ev = 2.0;
            s.env_opacity = 1.0;
            s.bloom = 1.0;
        }),
        "post_bloom",
    );
    assert!(bloom.mean(0, 0, 480, 360) > hot_base.mean(0, 0, 480, 360) + 0.5, "bloom adds light");
    let sharpen = render(&mut h, &on(|s| s.sharpen = 1.0), "post_sharpen");
    assert_ne!(sharpen.px, base.px, "sharpen changes edges");
    let grain = render(&mut h, &on(|s| s.grain = 1.0), "post_grain");
    assert_ne!(grain.px, base.px, "grain adds noise");
    let all = render(
        &mut h,
        &on(|s| {
            s.bloom = 0.6;
            s.vignette = 0.5;
            s.sharpen = 0.4;
            s.grain = 0.3;
        }),
        "post_all",
    );
    assert_eq!(all.px.len(), base.px.len());
}

/// The shadow pass draws the displaced (dense) mesh when the view displaces: with flat height it
/// casts the same shadows as the plain mesh.
#[test]
fn displaced_shadows_match_flat_height() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| {
        s.light = LIGHT;
        s.shadows = true;
    });
    let plain = render(&mut h, &lit, "shadows_plain");
    let displaced = render(&mut h, &shaded(&lit, |s| s.displacement = true), "shadows_displaced");
    let off = plain.px.chunks(4).zip(displaced.px.chunks(4)).filter(|(a, b)| a.iter().zip(*b).any(|(x, y)| x.abs_diff(*y) > 8)).count();
    assert!(off < 480 * 360 / 100, "{off} pixels differ");
}

#[test]
fn shadows_darken_the_key_light() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let a = render(&mut h, &lit, "shadows_off");
    let b = render(&mut h, &shaded(&lit, |s| s.shadows = true), "shadows_on");
    let (mut darker, mut brighter) = (0, 0);
    for y in 0..360 {
        for x in 0..480 {
            let d = b.luma(x, y) - a.luma(x, y);
            darker += (d < -8.0) as u32;
            brighter += (d > 2.0) as u32;
        }
    }
    assert!(darker > 1000, "the ball shadows its pedestal ({darker} darker pixels)");
    assert_eq!(brighter, 0, "shadows never brighten");
    // From above, where the pedestal top shows the ball's shadow.
    let mut top = shaded(&lit, |s| s.shadows = true);
    top.camera.pitch = 0.9;
    render(&mut h, &top, "shadows_on_top");
    // Rotating the environment turns the key light and its shadow.
    let turned = render(
        &mut h,
        &shaded(&lit, |s| {
            s.shadows = true;
            s.env_rotation_deg = 120.0;
        }),
        "shadows_env_rotated",
    );
    assert_ne!(turned.px, b.px);
}

#[test]
fn wireframe_3d_overlays_the_mesh() {
    let Some((mut h, frame)) = setup() else { return };
    let a = render(&mut h, &frame, "wire3d_off");
    let b = render(&mut h, &shaded(&frame, |s| s.wireframe_3d = true), "wire3d_on");
    assert_ne!(a.mean(200, 100, 280, 180), b.mean(200, 100, 280, 180), "edges draw over the ball");
    assert_eq!(a.at(5, 5), b.at(5, 5), "the background stays clean");
}

#[test]
fn bake_visualization_draws_the_high_poly_and_the_cage() {
    // A high poly inside the material ball, a cage around it, and how far a color must lead the others to count as that color
    const HIGH_POLY_RADIUS_PER_SIZE: f32 = 0.3;
    const CAGE_SCALE: f32 = 2.5;
    const HIGH_POLY_ID: u64 = 7;
    const CAGE_TOPOLOGY: u64 = 1;
    const CAGE_EDGES: [[u32; 2]; 2] = [[0, 1], [1, 2]];
    const COLOR_LEAD: i32 = 30;

    let Some((mut h, frame)) = setup() else { return };
    let ball = primitives::material_ball();
    let (low_min, low_max) = ball.bounds();
    let ball_center = (low_min + low_max) * 0.5;
    let mut high = primitives::sphere((low_max - low_min).max_element() * HIGH_POLY_RADIUS_PER_SIZE, 32, 16);
    for position in &mut high.positions {
        *position += ball_center;
    }
    h.renderer.set_bake_high(&h.device, &h.queue, Some(&high), HIGH_POLY_ID).unwrap();
    let corners: Vec<Vec3> = high.indices.iter().map(|index| (high.positions[*index as usize] - ball_center) * CAGE_SCALE + ball_center).collect();
    h.renderer.set_bake_cage(&h.device, &h.queue, 0, Some((&corners, &CAGE_EDGES)), CAGE_TOPOLOGY);
    let bake = BakeVisualization { high_poly: false, cage: false, uv_seams: false, low_poly: true, only_set: None };
    let center = (frame.size[0] / 2, frame.size[1] / 2);

    let only = |h: &mut Headless, parts: BakeVisualization, name: &str| render(h, &shaded(&frame, |s| s.bake = Some(parts)), name);
    let low_only = only(&mut h, bake, "bake_low_only");
    let hidden = only(&mut h, BakeVisualization { low_poly: false, ..bake }, "bake_low_hidden");
    let high_poly = only(&mut h, BakeVisualization { low_poly: false, high_poly: true, ..bake }, "bake_high_poly");
    let caged = only(&mut h, BakeVisualization { cage: true, ..bake }, "bake_cage");
    assert_same(&low_only, &render(&mut h, &frame, "bake_off"), "every part off draws only the low poly");
    let background = render(&mut h, &Frame { visible: vec![false, false], ..frame.clone() }, "bake_background");
    assert_same(&hidden, &background, "without the low poly only the background is left");
    let [red, green, blue] = high_poly.at(center.0, center.1).map(i32::from);
    assert!(blue > red + COLOR_LEAD && blue > green, "the high poly inside the low poly draws blue: {:?}", [red, green, blue]);
    let ([low_red, low_green, low_blue], [caged_red, caged_green, caged_blue]) = (low_only.at(center.0, center.1), caged.at(center.0, center.1));
    assert!(
        caged_blue < low_blue && caged_red >= caged_blue && caged_green >= caged_blue,
        "the cage tints the ball tan: {:?} → {:?}",
        [low_red, low_green, low_blue],
        [caged_red, caged_green, caged_blue]
    );

    // Vertices the bake's rays miss turn red
    h.renderer.set_bake_high_missed(&h.queue, HIGH_POLY_ID, &vec![1.0; high.positions.len()]);
    let missed = only(&mut h, BakeVisualization { low_poly: false, high_poly: true, ..bake }, "bake_high_missed");
    let [red, green, blue] = missed.at(center.0, center.1).map(i32::from);
    assert!(red > blue + COLOR_LEAD && red > green + COLOR_LEAD, "missed vertices draw red: {:?}", [red, green, blue]);

    // Show for selected texture set only: the other set's low poly and the cage of set 0 drop out
    let first_set = only(&mut h, BakeVisualization { cage: true, only_set: Some(0), ..bake }, "bake_only_first_set");
    let second_set = only(&mut h, BakeVisualization { cage: true, only_set: Some(1), ..bake }, "bake_only_second_set");
    assert_ne!(first_set.px, caged.px, "the second set's low poly is left out");
    assert_ne!(second_set.px, caged.px, "the first set's low poly and cage are left out");
    assert_ne!(first_set.px, second_set.px);

    // UV seams draw over the low poly
    h.renderer.set_uv_seams(&h.device, &h.queue, MESH_ID, &ball.uv_seam_edges());
    let seams = only(&mut h, BakeVisualization { uv_seams: true, ..bake }, "bake_uv_seams");
    assert_ne!(seams.px, low_only.px, "the seams show");

    // A new distance with the same topology only moves the corners
    let grown: Vec<Vec3> = corners.iter().map(|corner| (*corner - ball_center) * 2.0 + ball_center).collect();
    h.renderer.set_bake_cage(&h.device, &h.queue, 0, Some((&grown, &CAGE_EDGES)), CAGE_TOPOLOGY);
    let bigger = only(&mut h, BakeVisualization { cage: true, ..bake }, "bake_cage_grown");
    assert_ne!(caged.px, bigger.px, "the cage grew");
}

#[test]
fn oversized_bake_high_poly_leaves_the_view_renderable() {
    let Some((mut headless, frame)) = setup() else { return };

    let high = primitives::sphere(0.5, 16, 8);
    headless.renderer.set_bake_high(&headless.device, &headless.queue, Some(&high), 1).unwrap();
    let limit = headless.device.limits().max_buffer_size;
    // The viewport vertex contains position, normal, tangent and UV: twelve f32 values
    let vertex_count = limit as usize / (12 * std::mem::size_of::<f32>()) + 1;
    let oversized = texturecraft_geom::Mesh { positions: vec![Vec3::ZERO; vertex_count], ..Default::default() };
    let error = headless.renderer.set_bake_high(&headless.device, &headless.queue, Some(&oversized), 2).unwrap_err();
    assert!(error.contains("GPU buffer limit"), "{error}");

    headless.renderer.set_bake_high_missed(&headless.queue, 2, &[]);
    let bake = BakeVisualization { high_poly: true, cage: false, uv_seams: false, low_poly: false, only_set: None };
    let cleared = render(&mut headless, &shaded(&frame, |shade| shade.bake = Some(bake)), "bake_oversized_cleared");
    let background = render(&mut headless, &Frame { visible: vec![false, false], ..frame.clone() }, "bake_oversized_background");
    assert_same(&cleared, &background, "an oversized replacement must not leave the previous high poly visible");
    headless.renderer.set_bake_high(&headless.device, &headless.queue, Some(&high), 3).unwrap();
    let recovered = render(&mut headless, &shaded(&frame, |shade| shade.bake = Some(bake)), "bake_oversized_recovered");
    assert_ne!(recovered.px, background.px, "a later valid high poly still draws");
}

#[test]
fn bake_low_poly_visibility_also_filters_its_wireframe() {
    let Some((mut headless, frame)) = setup() else { return };

    let bake = BakeVisualization { high_poly: false, cage: false, uv_seams: false, low_poly: false, only_set: None };
    let hidden = shaded(&frame, |shade| {
        shade.bake = Some(bake);
        shade.wireframe_3d = true;
        shade.highlight = Some(texturecraft_render::SetHighlight { set: 0, color: [1.0, 0.5, 0.0] });
    });
    let background = render(&mut headless, &Frame { visible: vec![false, false], ..frame.clone() }, "bake_wire_background");
    let hidden_wire = render(&mut headless, &hidden, "bake_wire_hidden");
    assert_same(&hidden_wire, &background, "hiding the project mesh also hides its wireframe and highlight");

    for texture_set in [0, 1] {
        let selected = shaded(&hidden, |shade| {
            shade.bake = Some(BakeVisualization { low_poly: true, only_set: Some(texture_set), ..bake });
            shade.highlight = Some(texturecraft_render::SetHighlight { set: 1 - texture_set, color: [1.0, 0.5, 0.0] });
        });
        let expected = Frame { visible: vec![texture_set == 0, texture_set == 1], ..selected.clone() };
        let actual = render(&mut headless, &selected, &format!("bake_wire_selected_{texture_set}"));
        let expected = render(&mut headless, &expected, &format!("bake_wire_expected_{texture_set}"));
        assert_same(&actual, &expected, "the wireframe and highlight respect the selected texture set");
    }
}

#[test]
fn ortho_background_matches_perspective() {
    let Some((mut h, frame)) = setup() else { return };
    let env = shaded(&frame, |s| s.env_opacity = 1.0);
    let persp = render(&mut h, &env, "env_perspective");
    let ortho = render(&mut h, &Frame { camera: OrbitCamera { ortho: true, ..env.camera }, ..env.clone() }, "env_ortho");
    for (x, y) in [(5, 5), (474, 5), (5, 354), (474, 354), (240, 5)] {
        assert_eq!(persp.at(x, y), ortho.at(x, y), "background at {x},{y}");
    }
    assert_ne!(ortho.at(5, 5), ortho.at(240, 5), "the environment varies across the view");
}

/// Frame cost at 1920×1080: `cargo test -p texturecraft-render --release -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_1080p() {
    let Some((mut h, frame)) = setup() else { return };
    h.renderer.set_texture_sets(&h.device, &h.queue, &[((2048, 2048), 0), ((2048, 2048), 1)], 2);
    let frame = Frame { size: [1920, 1080], ..frame };
    let run = |h: &mut Headless, frame: &Frame| {
        for _ in 0..5 {
            h.render(frame);
        }
        let t = std::time::Instant::now();
        for _ in 0..100 {
            let mut enc = h.device.create_command_encoder(&Default::default());
            h.renderer.prepare(&h.device, &h.queue, &mut enc, frame);
            h.queue.submit([enc.finish()]);
            let _ = h.device.poll(texturecraft_render::wgpu::PollType::Wait { submission_index: None, timeout: None });
        }
        t.elapsed().as_secs_f64() * 1000.0 / 100.0
    };
    println!("adapter: {}", h.adapter_name);
    println!("default: {:.3} ms/frame", run(&mut h, &frame));
    let effects = shaded(&frame, |s| {
        s.post_effects = true;
        s.bloom = 0.5;
        s.vignette = 0.5;
        s.sharpen = 0.5;
        s.grain = 0.5;
    });
    println!("post effects: {:.3} ms/frame", run(&mut h, &effects));
    let shadows = shaded(&frame, |s| {
        s.light = LIGHT;
        s.shadows = true;
    });
    println!("shadows (cached map): {:.3} ms/frame", run(&mut h, &shadows));
    let wire = shaded(&frame, |s| s.wireframe_3d = true);
    println!("3D wireframe: {:.3} ms/frame", run(&mut h, &wire));
    let all = shaded(&effects, |s| {
        s.light = LIGHT;
        s.shadows = true;
        s.wireframe_3d = true;
    });
    println!("everything: {:.3} ms/frame", run(&mut h, &all));
    println!("parallax occlusion: {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.parallax = true)));
    println!("depth of field: {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.dof = Some([0.0, 0.1]))));
    println!("subsurface blur: {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.sss_blur = Some([0.1, 0.04, 0.025]))));
    let both = shaded(&frame, |s| {
        s.dof = Some([0.0, 0.1]);
        s.sss_blur = Some([0.1, 0.04, 0.025]);
    });
    println!("depth of field + subsurface blur: {:.3} ms/frame", run(&mut h, &both));
    println!("displacement (dense copy): {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.displacement = true)));
    println!("screen-space refraction: {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.refraction = true)));
    // Every extra lobe painted (full-size lobe textures), with the key light.
    fill_res(&mut h, 2048, &[0, 1], DisplayKind::Lobes, &[255, 25, 200, 60]);
    fill_res(&mut h, 2048, &[0, 1], DisplayKind::Sheen, &[200, 150, 255, 200]);
    fill_res(&mut h, 2048, &[0, 1], DisplayKind::Emissive, &[0, 0, 0, 200]);
    println!("all lobes: {:.3} ms/frame", run(&mut h, &frame));
    println!("all lobes + key light: {:.3} ms/frame", run(&mut h, &shaded(&frame, |s| s.light = LIGHT)));
    let taa = shaded(&frame, |s| s.taa = true);
    h.render(&taa);
    let t = std::time::Instant::now();
    for _ in 1..TAA_SAMPLES {
        let mut enc = h.device.create_command_encoder(&Default::default());
        h.renderer.prepare(&h.device, &h.queue, &mut enc, &taa);
        h.queue.submit([enc.finish()]);
        let _ = h.device.poll(texturecraft_render::wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    println!("TAA accumulation: {:.3} ms/sample", t.elapsed().as_secs_f64() * 1000.0 / f64::from(TAA_SAMPLES - 1));
    // Shadow map re-rendered every frame (light moving).
    let t = std::time::Instant::now();
    for i in 0..100 {
        let f = shaded(&shadows, |s| s.env_rotation_deg = i as f32);
        let mut enc = h.device.create_command_encoder(&Default::default());
        h.renderer.prepare(&h.device, &h.queue, &mut enc, &f);
        h.queue.submit([enc.finish()]);
        let _ = h.device.poll(texturecraft_render::wgpu::PollType::Wait { submission_index: None, timeout: None });
    }
    println!("shadows, light moving: {:.3} ms/frame", t.elapsed().as_secs_f64() * 1000.0 / 100.0);
}

/// Fill the height texture of both 256² sets: `f(x, y)` → (height, F0 r, g, b).
fn upload_height(h: &mut Headless, f: impl Fn(u32, u32) -> [f32; 4]) {
    for ts in 0..2 {
        for (tx, ty) in (0..16).map(|i| (i % 4, i / 4)) {
            let bytes: Vec<u8> = (0..64 * 64).flat_map(|i| f(tx * 64 + i % 64, ty * 64 + i / 64)).flat_map(|v| half::f16::from_f32(v).to_le_bytes()).collect();
            h.renderer.upload_tile(&h.device, &h.queue, ts, DisplayKind::Height, tx, ty, &bytes);
        }
    }
}

/// Mean of red minus blue over the ball.
fn warmth(img: &Img) -> f32 {
    let mut sum = 0.0;
    for y in 100..180 {
        for x in 200..280 {
            let [r, _, b] = img.at(x, y);
            sum += r as f32 - b as f32;
        }
    }
    sum / 6400.0
}

#[test]
fn imported_hdr_environment_lights_the_ball() {
    let Some((mut h, frame)) = setup() else { return };
    let base = render(&mut h, &frame, "env_default");
    // A procedural HDR: a hot orange sky over a dim blue ground (values far above 1).
    let (w, hh) = (256u32, 128u32);
    let rgba = (0..w * hh).map(|i| if i / w < hh / 2 { [12.0, 4.0, 1.0, 1.0] } else { [0.02, 0.05, 0.2, 1.0] }).collect();
    let sky = ImageRes { width: w, height: hh, rgba, srgb: false };
    h.renderer.set_environment(&h.device, &h.queue, "resource:sky.hdr", Some(&sky));
    let lit = render(&mut h, &frame, "env_hdr_resource");
    assert_ne!(lit.px, base.px, "the imported environment changes the lighting");
    assert!(warmth(&lit) > warmth(&base) + 10.0, "the orange sky tints the ball");
    assert!(lit.mean(200, 100, 280, 180) > 20.0, "lit, finite shading (NaNs would render black)");
    h.renderer.set_environment(&h.device, &h.queue, "Studio Soft", None);
    assert_eq!(render(&mut h, &frame, "env_default_again").px, base.px, "built-ins come back");
}

#[test]
fn specular_glossiness_takes_f0_from_the_specular_color() {
    let Some((mut h, frame)) = setup() else { return };
    upload_height(&mut h, |_, _| [0.0, 1.0, 0.6, 0.1]);
    let mr = render(&mut h, &frame, "shader_metal_rough");
    let sg = render(&mut h, &shaded(&frame, |s| s.spec_gloss = true), "shader_spec_gloss_gold");
    assert!(warmth(&sg) > warmth(&mr) + 10.0, "a gold F0 tints the reflections ({} vs {})", warmth(&sg), warmth(&mr));
    assert!(sg.mean(200, 100, 280, 180) > 20.0);
}

#[test]
fn parallax_follows_the_height_channel() {
    let Some((mut h, frame)) = setup() else { return };
    let pom = shaded(&frame, |s| s.parallax = true);
    // A flat height field: parallax changes nothing.
    let (off, on) = (render(&mut h, &frame, "parallax_flat_off"), render(&mut h, &pom, "parallax_flat_on"));
    assert!(off.px.iter().zip(&on.px).all(|(a, b)| a.abs_diff(*b) <= 2), "flat heights keep the UVs");
    upload_height(&mut h, |x, y| [if (x / 16 + y / 16) % 2 == 0 { 0.8 } else { -0.8 }, 0.04, 0.04, 0.04]);
    let (off, on) = (render(&mut h, &frame, "parallax_off"), render(&mut h, &pom, "parallax_on"));
    assert_ne!(off.px, on.px, "a checker height field shifts the textures");
    assert!(on.mean(200, 100, 280, 180) > 20.0);
}

#[test]
fn idle_accumulation_converges_then_stops() {
    let Some((mut h, frame)) = setup() else { return };
    let taa = shaded(&frame, |s| s.taa = true);
    let first = render(&mut h, &taa, "taa_first");
    assert_eq!(first.px, render(&mut h, &frame, "taa_off").px, "the first sample is the plain frame");
    render(&mut h, &taa, "taa_first_again");
    for _ in 1..TAA_SAMPLES {
        assert!(h.renderer.accumulating(1));
        render(&mut h, &taa, "taa_sample");
    }
    assert!(!h.renderer.accumulating(1), "converged after {TAA_SAMPLES} samples");
    let last = render(&mut h, &taa, "taa_converged");
    assert_ne!(last.px, first.px, "jittered samples smooth the edges");
    assert_eq!(render(&mut h, &taa, "taa_idle").px, last.px, "a converged view keeps its image");
    render(&mut h, &shaded(&taa, |s| s.env_rotation_deg = 30.0), "taa_changed");
    assert!(h.renderer.accumulating(1), "a change restarts accumulation");
}

// ---- Extra lobes (clear coat, sheen, anisotropy, subsurface scattering) -----------------------

/// Fill a display texture of texture sets `sets` (256², see `setup`) with one texel.
fn fill(h: &mut Headless, sets: &[usize], kind: DisplayKind, texel: &[u8]) {
    fill_res(h, 256, sets, kind, texel);
}

fn fill_res(h: &mut Headless, res: u32, sets: &[usize], kind: DisplayKind, texel: &[u8]) {
    let bytes = texel.repeat(64 * 64);
    let n = res / 64;
    for &ts in sets {
        for (tx, ty) in (0..n * n).map(|i| (i % n, i / n)) {
            h.renderer.upload_tile(&h.device, &h.queue, ts, kind, tx, ty, &bytes);
        }
    }
}

/// Pixels showing texture sets `sets`: those a change of their base color reaches. Leaves the
/// base color at its default.
fn mesh_mask(h: &mut Headless, frame: &Frame, sets: &[usize]) -> Vec<bool> {
    fill(h, sets, DisplayKind::BaseColor, &[255, 0, 0, 255]);
    let a = render(h, frame, "mask_red");
    fill(h, sets, DisplayKind::BaseColor, &[0, 255, 0, 255]);
    let b = render(h, frame, "mask_green");
    fill(h, sets, DisplayKind::BaseColor, &DisplayKind::BaseColor.default_texel());
    a.px.chunks(4).zip(b.px.chunks(4)).map(|(x, y)| x != y).collect()
}

/// Mean of `f(x, y)` over the pixels in `region`.
fn region_mean(region: &[bool], f: impl Fn(u32, u32) -> f32) -> f32 {
    let (mut sum, mut n) = (0.0, 0.0);
    for (i, _) in region.iter().enumerate().filter(|(_, r)| **r) {
        sum += f(i as u32 % 480, i as u32 / 480);
        n += 1.0;
    }
    sum / f32::max(n, 1.0)
}

/// Mesh pixels within `r` pixels of the silhouette (`near`), or farther from it.
fn rim(mask: &[bool], r: usize, near: bool) -> Vec<bool> {
    let (w, h) = (480, mask.len() / 480);
    let edge = |x: usize, y: usize| (y.saturating_sub(r)..(y + r + 1).min(h)).any(|yy| (x.saturating_sub(r)..(x + r + 1).min(w)).any(|xx| !mask[yy * w + xx]));
    (0..mask.len()).map(|i| mask[i] && edge(i % w, i / w) == near).collect()
}

/// Σ|∂luma/∂y| / Σ|∂luma/∂x| inside `mask`: above 1 where the shading streaks sideways.
fn gradient_ratio(img: &Img, mask: &[bool]) -> f32 {
    let (mut gx, mut gy) = (0.0, 0.0);
    for y in 0..359 {
        for x in 0..479 {
            let at = |dx: u32, dy: u32| mask[((y + dy) * 480 + x + dx) as usize];
            if at(0, 0) && at(1, 0) && at(0, 1) {
                gx += (img.luma(x + 1, y) - img.luma(x, y)).abs();
                gy += (img.luma(x, y + 1) - img.luma(x, y)).abs();
            }
        }
    }
    gy / f32::max(gx, 1e-3)
}

#[test]
fn lobe_channels_at_their_defaults_change_nothing() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let base = render(&mut h, &lit, "lobes_absent");
    // Paint every lobe, then go back to the defaults: full-size textures shade like the placeholders.
    fill(&mut h, &[0, 1], DisplayKind::Lobes, &[255, 200, 255, 90]);
    fill(&mut h, &[0, 1], DisplayKind::Sheen, &[255, 0, 0, 255]);
    fill(&mut h, &[0, 1], DisplayKind::Emissive, &[0, 0, 0, 255]);
    fill(&mut h, &[0, 1], DisplayKind::Extra, &[255, 0, 255, 0]);
    assert_ne!(render(&mut h, &lit, "lobes_painted").px, base.px);
    for k in [DisplayKind::Lobes, DisplayKind::Sheen, DisplayKind::Emissive, DisplayKind::Extra] {
        fill(&mut h, &[0, 1], k, &k.default_texel());
    }
    assert_same(&render(&mut h, &lit, "lobes_defaults"), &base, "default lobes change nothing");
}

#[test]
fn clear_coat_adds_a_sharp_highlight_over_a_rough_base() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let mask = mesh_mask(&mut h, &lit, &[0, 1]);
    fill(&mut h, &[0, 1], DisplayKind::Orm, &[255, 255, 0, 128]);
    let rough = render(&mut h, &lit, "coat_off");
    fill(&mut h, &[0, 1], DisplayKind::Lobes, &[255, 13, 0, 0]);
    let coated = render(&mut h, &lit, "coat_on");
    let peak = |img: &Img| (0..mask.len()).filter(|i| mask[*i]).map(|i| img.luma(i as u32 % 480, i as u32 / 480)).fold(0.0, f32::max);
    let (p0, p1) = (peak(&rough), peak(&coated));
    assert!(p1 > p0 + 25.0, "the coat adds a highlight: peak {p0} -> {p1}");
    // Elsewhere a thin layer of reflection replaces a little of the base.
    let (m0, m1) = (region_mean(&mask, |x, y| rough.luma(x, y)), region_mean(&mask, |x, y| coated.luma(x, y)));
    assert!((m1 - m0).abs() < 15.0, "mean {m0} -> {m1}");
}

#[test]
fn sheen_brightens_grazing_angles() {
    let Some((mut h, frame)) = setup() else { return };
    let mask = mesh_mask(&mut h, &frame, &[0, 1]);
    fill(&mut h, &[0, 1], DisplayKind::BaseColor, &[30, 30, 30, 255]);
    let plain = render(&mut h, &frame, "sheen_off");
    fill(&mut h, &[0, 1], DisplayKind::Sheen, &[255, 255, 255, 255]);
    let velvet = render(&mut h, &frame, "sheen_on");
    let gain = |region: &[bool]| region_mean(region, |x, y| velvet.luma(x, y) - plain.luma(x, y));
    let (g_edge, g_inner) = (gain(&rim(&mask, 4, true)), gain(&rim(&mask, 25, false)));
    assert!(g_edge > 10.0 && g_edge > 1.5 * g_inner, "rim +{g_edge}, facing +{g_inner}");
}

#[test]
fn anisotropy_stretches_highlights_along_the_tangent() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let ball = mesh_mask(&mut h, &lit, &[0]);
    // A smooth metal ball.
    fill(&mut h, &[0], DisplayKind::Orm, &[255, 90, 255, 128]);
    let iso = gradient_ratio(&render(&mut h, &lit, "aniso_off"), &ball);
    fill(&mut h, &[0], DisplayKind::Lobes, &[0, 25, 255, 0]);
    let along_u = gradient_ratio(&render(&mut h, &lit, "aniso_along_u"), &ball);
    fill(&mut h, &[0], DisplayKind::Lobes, &[0, 25, 255, 128]);
    let along_v = gradient_ratio(&render(&mut h, &lit, "aniso_along_v"), &ball);
    // The tangent (+u) runs around the ball, across the screen: angle 0 streaks the reflections
    // sideways, a quarter turn (angle 0.5) up and down.
    assert!(along_u > 1.15 * iso && along_v < 0.6 * iso, "vertical/horizontal gradients: isotropic {iso}, angle 0 {along_u}, angle 0.5 {along_v}");
}

#[test]
fn subsurface_scattering_softens_the_terminator() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let mask = mesh_mask(&mut h, &lit, &[0, 1]);
    fill(&mut h, &[0, 1], DisplayKind::BaseColor, &[230, 230, 230, 255]);
    let lambert = render(&mut h, &lit, "sss_off");
    fill(&mut h, &[0, 1], DisplayKind::Emissive, &[0, 0, 0, 255]);
    let skin = render(&mut h, &lit, "sss_on");
    // Quartiles of the Lambert shading: the shadowed side brightens and warms, contrast drops.
    let luma_of = |i: usize| lambert.luma(i as u32 % 480, i as u32 / 480);
    let mut lumas: Vec<f32> = (0..mask.len()).filter(|i| mask[*i]).map(luma_of).collect();
    lumas.sort_by(f32::total_cmp);
    let (q1, q3) = (lumas[lumas.len() / 4], lumas[lumas.len() * 3 / 4]);
    let dark: Vec<bool> = (0..mask.len()).map(|i| mask[i] && luma_of(i) <= q1).collect();
    let bright: Vec<bool> = (0..mask.len()).map(|i| mask[i] && luma_of(i) >= q3).collect();
    let luma = |img: &Img, region: &[bool]| region_mean(region, |x, y| img.luma(x, y));
    let warm = |img: &Img| region_mean(&dark, |x, y| img.at(x, y)[0] as f32 - img.at(x, y)[2] as f32);
    assert!(luma(&skin, &dark) > luma(&lambert, &dark) + 3.0, "shadow side {} -> {}", luma(&lambert, &dark), luma(&skin, &dark));
    assert!(luma(&skin, &bright) - luma(&skin, &dark) < luma(&lambert, &bright) - luma(&lambert, &dark), "contrast drops");
    assert!(warm(&skin) > warm(&lambert) + 2.0, "the scatter color tints the falloff ({} -> {})", warm(&lambert), warm(&skin));
}

// ---- Transmission, coat normal, depth of field, subsurface blur --------------------------------

/// A base color checker (8-texel squares) on both sets, for detail to blur.
fn checker(h: &mut Headless) {
    for ts in 0..2 {
        for (tx, ty) in (0..16).map(|i| (i % 4, i / 4)) {
            let bytes: Vec<u8> =
                (0..64 * 64).flat_map(|i| if ((i % 64) / 8 + (i / 64) / 8) % 2 == 0 { [230, 230, 230, 255] } else { [20, 20, 20, 255] }).collect();
            h.renderer.upload_tile(&h.device, &h.queue, ts, DisplayKind::BaseColor, tx, ty, &bytes);
        }
    }
}

/// Σ|∂luma/∂x| over `mask`: how much detail the image shows there.
fn detail(img: &Img, mask: &[bool]) -> f32 {
    (0..mask.len())
        .filter(|i| mask[*i] && i % 480 < 479 && mask[i + 1])
        .map(|i| (img.luma(i as u32 % 480 + 1, i as u32 / 480) - img.luma(i as u32 % 480, i as u32 / 480)).abs())
        .sum()
}

/// Mean |luma difference| over a square of side 2r around (x, y).
fn patch_diff(a: &Img, b: &Img, (x, y): (u32, u32), r: u32) -> f32 {
    let mut sum = 0.0;
    for yy in y - r..y + r {
        for xx in x - r..x + r {
            sum += (a.luma(xx, yy) - b.luma(xx, yy)).abs();
        }
    }
    sum / (4 * r * r) as f32
}

/// View depth of the mesh under the view's center pixel.
fn center_depth(frame: &Frame) -> f32 {
    let mesh = primitives::material_ball();
    let bvh = texturecraft_geom::Bvh::new(&mesh);
    let cam = frame.camera;
    let ray = cam.ray(texturecraft_geom::glam::Vec2::ZERO, frame.size[0] as f32 / frame.size[1] as f32);
    let hit = bvh.closest(ray.origin, ray.dir, f32::MAX).expect("the ball is in the middle");
    (ray.origin + ray.dir * hit.t - cam.eye()).dot(cam.forward())
}

#[test]
fn depth_of_field_blurs_away_from_the_focus_plane() {
    let Some((mut h, frame)) = setup() else { return };
    let mask = mesh_mask(&mut h, &frame, &[0, 1]);
    checker(&mut h);
    let sharp = render(&mut h, &frame, "dof_off");
    let dof = |focus: f32, aperture: f32| shaded(&frame, |s| s.dof = Some([focus, aperture]));
    assert_same(&render(&mut h, &dof(0.0, 0.0), "dof_pinhole"), &sharp, "a pinhole lens changes nothing");
    let z = center_depth(&frame);
    let focused = render(&mut h, &dof(z, 0.15), "dof_focused");
    let center = (frame.size[0] / 2, frame.size[1] / 2);
    assert!(patch_diff(&focused, &sharp, center, 5) < 3.0, "the focus plane stays sharp: {}", patch_diff(&focused, &sharp, center, 5));
    let far = render(&mut h, &dof(z * 10.0, 0.3), "dof_far_focus");
    let (d_sharp, d_far) = (detail(&sharp, &mask), detail(&far, &mask));
    assert!(d_far < 0.5 * d_sharp, "out of focus, the checker blurs: detail {d_sharp} -> {d_far}");
    assert!(patch_diff(&far, &sharp, center, 5) > 5.0);
    // Focused on the ball, it keeps more of its detail than focused far behind it.
    assert!(detail(&focused, &mask) > d_far);
    // Idle accumulation averages frames with the sampling spiral turned each time.
    let taa = shaded(&dof(z * 10.0, 0.3), |s| s.taa = true);
    assert_eq!(render(&mut h, &taa, "dof_taa_first").px, far.px, "the first sample is the plain frame");
    for _ in 1..TAA_SAMPLES {
        render(&mut h, &taa, "dof_taa_sample");
    }
    assert!(!h.renderer.accumulating(1));
    assert_ne!(render(&mut h, &taa, "dof_taa_converged").px, far.px);
}

#[test]
fn subsurface_blur_spreads_only_scattering_light() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let mask = mesh_mask(&mut h, &lit, &[0, 1]);
    checker(&mut h);
    let plain = render(&mut h, &lit, "sss_blur_off");
    let blur = shaded(&lit, |s| s.sss_blur = Some([0.06, 0.024, 0.015]));
    assert_same(&render(&mut h, &blur, "sss_blur_no_weight"), &plain, "nothing scatters without the scattering weight");
    fill(&mut h, &[0, 1], DisplayKind::Emissive, &[0, 0, 0, 255]);
    let wrap = render(&mut h, &lit, "sss_wrap_only");
    let blurred = render(&mut h, &blur, "sss_blur_on");
    let (d0, d1) = (detail(&wrap, &mask), detail(&blurred, &mask));
    assert!(d1 < 0.85 * d0, "the scattering light blurs the checker: detail {d0} -> {d1}");
    // On a plain color the blur only moves light around the smooth shading (the checker's mean
    // display value rises: averaging happens before the sRGB curve).
    fill(&mut h, &[0, 1], DisplayKind::BaseColor, &[150, 150, 150, 255]);
    let mean = |img: &Img| region_mean(&mask, |x, y| img.luma(x, y));
    let (m0, m1) = (mean(&render(&mut h, &lit, "sss_plain_wrap")), mean(&render(&mut h, &blur, "sss_plain_blur")));
    assert!((m1 - m0).abs() < 2.0, "the blur moves light, it adds none: {m0} -> {m1}");
}

/// An environment that is bright only around `dir` (world space, rotation 0).
fn spot_environment(dir: texturecraft_geom::glam::Vec3) -> ImageRes {
    let (w, hh) = (256u32, 128u32);
    let rgba = (0..w * hh)
        .map(|i| {
            let d = texturecraft_render::ibl::dir_of(((i % w) as f32 + 0.5) / w as f32, ((i / w) as f32 + 0.5) / hh as f32);
            if d.dot(dir) > 0.5 { [6.0, 6.0, 6.0, 1.0] } else { [0.02, 0.02, 0.02, 1.0] }
        })
        .collect();
    ImageRes { width: w, height: hh, rgba, srgb: false }
}

#[test]
fn transmission_shows_the_environment_behind_the_surface() {
    let Some((mut h, frame)) = setup() else { return };
    h.renderer.set_environment(&h.device, &h.queue, "resource:behind.hdr", Some(&spot_environment(frame.camera.forward())));
    fill(&mut h, &[0, 1], DisplayKind::BaseColor, &[255, 255, 255, 255]);
    fill(&mut h, &[0, 1], DisplayKind::Orm, &[255, 10, 0, 128]);
    let center = (frame.size[0] / 2, frame.size[1] / 2);
    let opaque = render(&mut h, &frame, "glass_off");
    fill(&mut h, &[0, 1], DisplayKind::Extra, &[128, 128, 255, 0]);
    let glass = render(&mut h, &frame, "glass_on");
    let luma = |img: &Img| img.mean(center.0 - 4, center.1 - 4, center.0 + 4, center.1 + 4);
    assert!(luma(&glass) > luma(&opaque) + 60.0, "the bright environment behind shows through: {} -> {}", luma(&opaque), luma(&glass));
}

#[test]
fn coat_normal_turns_the_coat_reflection() {
    let Some((mut h, frame)) = setup() else { return };
    let lit = shaded(&frame, |s| s.light = LIGHT);
    let mask = mesh_mask(&mut h, &lit, &[0, 1]);
    fill(&mut h, &[0, 1], DisplayKind::Lobes, &[255, 13, 0, 0]);
    let flat = render(&mut h, &lit, "coat_normal_flat");
    fill(&mut h, &[0, 1], DisplayKind::Extra, &[128, 128, 0, 0]);
    assert_same(&render(&mut h, &lit, "coat_normal_default"), &flat, "the flat coat normal changes nothing");
    fill(&mut h, &[0, 1], DisplayKind::Extra, &[230, 128, 0, 0]);
    let tilted = render(&mut h, &lit, "coat_normal_tilted");
    let diff = region_mean(&mask, |x, y| (tilted.luma(x, y) - flat.luma(x, y)).abs());
    assert!(diff > 2.0, "a tilted coat normal moves the coat highlight: {diff}");
}

// ---- Displacement and screen-space refraction --------------------------------------------------

#[test]
fn displacement_moves_the_silhouette_by_the_height_channel() {
    let Some((mut h, frame)) = setup() else { return };
    let disp = shaded(&frame, |s| s.displacement = true);
    // Flat heights: the dense copy draws the same surface (but for a few outline pixels where
    // the split edges cover other MSAA samples).
    let (off, on) = (render(&mut h, &frame, "disp_flat_off"), render(&mut h, &disp, "disp_flat_on"));
    let moved = off.px.chunks(4).zip(on.px.chunks(4)).filter(|(a, b)| a.iter().zip(b.iter()).any(|(x, y)| x.abs_diff(*y) > 2)).count();
    assert!(moved < 40, "flat heights leave the surface where it was ({moved} pixels differ)");
    let area = |h: &mut Headless, f: &Frame| mesh_mask(h, f, &[0, 1]).iter().filter(|m| **m).count();
    let base = area(&mut h, &frame);
    upload_height(&mut h, |_, _| [0.5, 0.04, 0.04, 0.04]);
    render(&mut h, &disp, "disp_raised");
    let grown = area(&mut h, &disp);
    assert!(grown > base + base / 50, "raised heights grow the silhouette: {base} -> {grown} pixels");
    upload_height(&mut h, |_, _| [-0.5, 0.04, 0.04, 0.04]);
    render(&mut h, &disp, "disp_lowered");
    let shrunk = area(&mut h, &disp);
    assert!(shrunk + base / 50 < base, "lowered heights shrink it: {base} -> {shrunk} pixels");
    assert_eq!(area(&mut h, &frame), base, "without displacement heights only shade");
    // A checker of heights reaches the shading and the outline.
    upload_height(&mut h, |x, y| [if (x / 16 + y / 16) % 2 == 0 { 0.6 } else { -0.6 }, 0.04, 0.04, 0.04]);
    let bumpy = render(&mut h, &shaded(&disp, |s| s.light = LIGHT), "disp_checker");
    assert!(bumpy.mean(200, 100, 280, 180) > 20.0, "lit, finite shading");
}

#[test]
fn refraction_shows_the_scene_behind_transmissive_surfaces() {
    let Some((mut h, frame)) = setup() else { return };
    let refr = shaded(&frame, |s| s.refraction = true);
    // The ball away from its outline (outline pixels mix in the pedestal behind them).
    let ball = rim(&mesh_mask(&mut h, &frame, &[0]), 2, false);
    assert_same(&render(&mut h, &refr, "refr_opaque"), &render(&mut h, &frame, "refr_opaque_off"), "without transmission nothing changes");
    // A clear glass ball (set 0) on its pedestal (set 1), painted red, then green.
    fill(&mut h, &[0], DisplayKind::BaseColor, &[255, 255, 255, 255]);
    fill(&mut h, &[0], DisplayKind::Orm, &[255, 10, 0, 128]);
    fill(&mut h, &[0], DisplayKind::Extra, &[128, 128, 255, 0]);
    let seen = |h: &mut Headless, f: &Frame, name: &str| {
        fill(h, &[1], DisplayKind::BaseColor, &[255, 0, 0, 255]);
        let red = render(h, f, &format!("{name}_red"));
        fill(h, &[1], DisplayKind::BaseColor, &[0, 255, 0, 255]);
        let green = render(h, f, &format!("{name}_green"));
        (0..ball.len()).filter(|i| ball[*i] && red.at(*i as u32 % 480, *i as u32 / 480) != green.at(*i as u32 % 480, *i as u32 / 480)).count()
    };
    assert_eq!(seen(&mut h, &frame, "refr_env"), 0, "refracting only the environment, the pedestal never shows through");
    let through = seen(&mut h, &refr, "refr_scene");
    let total = ball.iter().filter(|b| **b).count();
    println!("refraction: the pedestal shows through {through} of {total} ball pixels");
    assert!(through > total / 20, "the pedestal shows through the glass: {through} of {total} pixels");
    // Rough glass blurs what it shows: fewer sharp edges across the ball.
    fill(&mut h, &[1], DisplayKind::BaseColor, &[255, 0, 0, 255]);
    checker(&mut h);
    fill(&mut h, &[0], DisplayKind::BaseColor, &[255, 255, 255, 255]);
    let clear = render(&mut h, &refr, "refr_clear_checker");
    fill(&mut h, &[0], DisplayKind::Orm, &[255, 200, 0, 128]);
    let frosted = render(&mut h, &refr, "refr_frosted_checker");
    assert!(detail(&frosted, &ball) < detail(&clear, &ball), "frosted glass blurs the scene behind");
}

/// Kinds in [`DisplayKind::ALL`] order with each set's display texture widths, then the aux width.
fn widths(h: &Headless) -> Vec<Vec<u32>> {
    h.renderer.display_textures().iter().map(|s| s.iter().map(|t| t.width()).collect()).collect()
}

#[test]
fn texture_sets_take_no_gpu_memory_until_painted() {
    // A mesh with many materials (one texture set each) must not reserve full-size textures
    // for every set up front: that ran a 4 GB GPU out of memory, and wgpu's handler aborted.
    let Some((mut h, _)) = setup() else { return };
    let sets: Vec<((u32, u32), usize)> = (0..40).map(|m| ((2048, 2048), m)).collect();
    h.renderer.set_texture_sets(&h.device, &h.queue, &sets, 40);
    assert!(widths(&h).iter().flatten().all(|w| *w == 1), "every display texture starts as a 1×1 placeholder");
    assert!((0..40).all(|ts| h.renderer.aux_size(ts) == Some(1)), "so does every aux texture");
    // A neutral tile changes nothing; the first painted one allocates that kind alone.
    let kind = DisplayKind::BaseColor;
    let neutral = kind.default_texel().repeat(64 * 64);
    h.renderer.upload_tile(&h.device, &h.queue, 3, kind, 0, 0, &neutral);
    assert_eq!(widths(&h)[3][0], 1);
    h.renderer.upload_tile(&h.device, &h.queue, 3, kind, 0, 0, &[200, 30, 30, 255].repeat(64 * 64));
    let w = widths(&h);
    assert_eq!(w[3][0], 2048, "the painted kind is full size");
    assert!(w[3][1..].iter().all(|w| *w == 1) && w[2].iter().all(|w| *w == 1), "nothing else was allocated");
    // The aux texture is allocated by its first upload, at the set's aux size only.
    h.renderer.upload_aux(&h.device, &h.queue, 3, 512, &[0; 512 * 512 * 4]);
    assert_eq!(h.renderer.aux_size(3), Some(1), "a wrong aux size is ignored");
    h.renderer.upload_aux(&h.device, &h.queue, 3, 2048, &[90; 2048 * 2048 * 4]);
    assert_eq!(h.renderer.aux_size(3), Some(2048));
    assert!(h.renderer.take_memory_failures().is_empty());
}

#[test]
fn painted_placeholders_render_like_full_textures() {
    // A set painted in one kind renders that paint whether the other kinds are placeholders.
    let Some((mut h, frame)) = setup() else { return };
    let before = render(&mut h, &frame, "placeholders");
    fill(&mut h, &[0, 1], DisplayKind::BaseColor, &[220, 40, 40, 255]);
    let red = render(&mut h, &frame, "painted_base_color");
    let (w, hgt) = (frame.size[0], frame.size[1]);
    let [r0, g0, _] = before.at(w / 2, hgt / 2);
    let [r1, g1, _] = red.at(w / 2, hgt / 2);
    assert!(r1 > r0 && g1 < g0, "the painted base color shows: {:?} → {:?}", before.at(w / 2, hgt / 2), red.at(w / 2, hgt / 2));
}
