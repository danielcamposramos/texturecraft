//! From pointer samples to stamps: picking, lazy mouse, spacing along the surface, stamp
//! frames, jitter and symmetry.

use texturecraft_geom::glam::{Mat3, Quat, Vec2, Vec3};
use texturecraft_geom::{Bvh, Mesh, OrbitCamera, surface_at};
use texturecraft_texel::noise::pcg3d;

use crate::brush::{Alignment, BrushSettings};
use crate::stamp_source::StampSource;

/// Scene access for picking.
pub struct PickCtx<'a> {
    pub mesh: &'a Mesh,
    pub bvh: &'a Bvh,
    pub camera: &'a OrbitCamera,
    /// Viewport size in pixels.
    pub viewport: Vec2,
}

/// A point on the mesh under the cursor.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SurfaceHit {
    pub pos: Vec3,
    pub nrm: Vec3,
    pub tri: u32,
    pub uv: Vec2,
    pub material: usize,
    pub screen: Vec2,
}

/// Pick the mesh at a viewport pixel (origin top-left).
pub fn pick(ctx: &PickCtx, screen: Vec2) -> Option<SurfaceHit> {
    let vp = ctx.viewport.max(Vec2::ONE);
    let ndc = Vec2::new(screen.x / vp.x * 2.0 - 1.0, 1.0 - screen.y / vp.y * 2.0);
    let ray = ctx.camera.ray(ndc, vp.x / vp.y);
    let hit = ctx.bvh.closest(ray.origin, ray.dir, f32::MAX)?;
    let s = surface_at(ctx.mesh, hit.tri, hit.u, hit.v)?;
    Some(SurfaceHit { pos: s.pos, nrm: s.nrm, tri: hit.tri, uv: s.uv, material: ctx.mesh.material_of(hit.tri as usize).unwrap_or(0), screen })
}

/// Random channels of a stamp's jitters (each draws its own number from the stamp's seed)
const JITTER_ANGLE: u32 = 1;
const JITTER_SIZE: u32 = 2;
const JITTER_ACROSS: u32 = 3;
const JITTER_ALONG: u32 = 4;
const JITTER_FLOW: u32 = 5;

/// Full angle jitter turns a stamp this far either way (degrees)
const HALF_TURN_DEGREES: f32 = 180.0;

/// Smallest size jitter leaves a stamp, as a fraction of its radius
const MIN_JITTERED_SIZE: f32 = 0.05;

/// One brush dab in world space
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Stamp {
    pub center: Vec3,
    pub normal: Vec3,
    pub tangent: Vec3,
    pub bitangent: Vec3,
    pub radius: f32,
    pub flow: f32,
    pub hardness: f32,
    /// Half-depth of the stamp volume relative to the radius
    pub depth: f32,
    /// Varies procedural alphas per stamp
    pub seed: u32,
    /// A symmetry copy's way back to its source: it paints where the source would, mirrored
    pub source: Option<StampSource>,
}

/// An in-progress stroke.
#[derive(Clone, Debug)]
pub struct Stroke {
    pub settings: BrushSettings,
    /// Brush radius in world units at full pressure.
    pub radius: f32,
    center: Vec3,
    last: Option<SurfaceHit>,
    dist: f32,
    lazy: Option<Vec2>,
    count: u32,
    last_dir: Option<Vec3>,
    /// Every stamp emitted so far (for replay, inspect and tests).
    pub stamps: Vec<Stamp>,
}

fn rand01(n: u32, k: u32) -> f32 {
    pcg3d(n, k, 0x5EED)[0] as f32 / u32::MAX as f32
}

impl Stroke {
    pub fn new(settings: &BrushSettings, mesh: &Mesh) -> Stroke {
        let s = settings.sanitized();
        let (lo, hi) = mesh.bounds();
        let diag = (hi - lo).length().max(1e-4);
        Stroke { radius: s.size * diag, settings: s, center: (lo + hi) * 0.5, last: None, dist: 0.0, lazy: None, count: 0, last_dir: None, stamps: Vec::new() }
    }

    /// Feed a pointer position (pixels) with pen pressure (1 for a mouse). Returns new stamps.
    pub fn add(&mut self, ctx: &PickCtx, screen: Vec2, pressure: f32) -> Vec<Stamp> {
        let mut p = screen;
        if self.settings.lazy > 0.0 {
            let l = *self.lazy.get_or_insert(screen);
            let d = screen - l;
            let len = d.length();
            let nl = if len > self.settings.lazy { l + d * (1.0 - self.settings.lazy / len) } else { l };
            self.lazy = Some(nl);
            if self.last.is_some() && nl == l {
                return Vec::new();
            }
            p = nl;
        }
        let mut out = Vec::new();
        self.feed(ctx, p, pressure, 0, &mut out);
        if self.settings.particles.enabled {
            out = out.iter().flat_map(|s| crate::particles::emit(ctx, &self.settings.particles, s)).collect();
        }
        let mut all = Vec::with_capacity(out.len() * 2);
        for s in out {
            all.push(s);
            all.extend(self.mirrors(&s));
        }
        self.stamps.extend_from_slice(&all);
        all
    }

    fn radius_at(&self, ctx: &PickCtx, pos: Vec3, pressure: f32) -> f32 {
        let p = pressure.clamp(0.0, 1.0);
        let scale = self.settings.min_size + (1.0 - self.settings.min_size) * p;
        let r = self.radius * if pressure < 1.0 { scale } else { 1.0 };
        if self.settings.screen_space { r * (pos.distance(ctx.camera.eye()) / ctx.camera.distance.max(1e-4)) } else { r }
    }

    fn feed(&mut self, ctx: &PickCtx, screen: Vec2, pressure: f32, depth: u32, out: &mut Vec<Stamp>) {
        let Some(hit) = pick(ctx, screen) else {
            self.last = None;
            return;
        };
        let radius = self.radius_at(ctx, hit.pos, pressure);
        let Some(last) = self.last else {
            out.push(self.stamp(ctx, &hit, radius));
            self.last = Some(hit);
            self.dist = 0.0;
            return;
        };
        let seg = hit.pos - last.pos;
        let len = seg.length();
        if len < 1e-9 {
            return;
        }
        let step = (self.settings.spacing * 2.0 * radius).max(1e-6);
        if len > 4.0 * radius && depth < 4 {
            // Jumped across a silhouette or gap: resample in screen space, never through air.
            let n = ((len / radius).ceil() as u32).clamp(2, 64);
            let a = last.screen;
            for k in 1..=n {
                let s = a + (screen - a) * (k as f32 / n as f32);
                self.feed(ctx, s, pressure, depth + 1, out);
            }
            return;
        }
        self.last_dir = Some(seg / len);
        let mut t = step - self.dist;
        let mut emitted = 0;
        while t <= len && emitted < 4096 {
            let p = last.pos + seg * (t / len);
            // Snap to the surface by re-picking through the projected point.
            let aspect = ctx.viewport.x.max(1.0) / ctx.viewport.y.max(1.0);
            let snapped = ctx
                .camera
                .project(p, aspect)
                .map(|ndc| Vec2::new((ndc.x * 0.5 + 0.5) * ctx.viewport.x, (0.5 - ndc.y * 0.5) * ctx.viewport.y))
                .and_then(|s| pick(ctx, s))
                .filter(|h| h.pos.distance(p) < radius * 2.0);
            let h = snapped.unwrap_or(SurfaceHit { pos: p, ..hit });
            out.push(self.stamp(ctx, &h, radius));
            emitted += 1;
            t += step;
        }
        self.dist = len - (t - step);
        self.last = Some(hit);
    }

    /// The next stamp at `hit`: oriented to the surface or the camera, then jittered
    fn stamp(&mut self, ctx: &PickCtx, hit: &SurfaceHit, radius: f32) -> Stamp {
        let settings = &self.settings;
        let seed = self.count;
        self.count = self.count.wrapping_add(1);
        let jitter = |channel: u32, amount: f32| if amount > 0.0 { (rand01(seed, channel) * 2.0 - 1.0) * amount } else { 0.0 };

        // Frame: the normal, and an up direction from the camera or the stroke's path
        let normal = match settings.alignment {
            Alignment::Tangent => hit.nrm,
            Alignment::Camera => -ctx.camera.forward(),
        };
        let camera_up = {
            let forward = ctx.camera.forward();
            let right = forward.cross(Vec3::Y).try_normalize().unwrap_or(Vec3::X);
            right.cross(forward)
        };
        let up_source = if settings.follow_path { self.last_dir.unwrap_or(camera_up) } else { camera_up };
        let up = (up_source - normal * normal.dot(up_source)).try_normalize().unwrap_or_else(|| normal.any_orthonormal_vector());
        let unturned_tangent = up.cross(normal).try_normalize().unwrap_or_else(|| normal.any_orthonormal_vector());

        // Jitter: angle, size, position and flow
        let angle = (settings.angle + jitter(JITTER_ANGLE, settings.jitter_angle) * HALF_TURN_DEGREES).to_radians();
        let turn = Quat::from_axis_angle(normal, angle);
        let tangent = turn * unturned_tangent;
        let bitangent = normal.cross(tangent);
        let jittered_radius = radius * (1.0 + jitter(JITTER_SIZE, settings.jitter_size)).max(MIN_JITTERED_SIZE);
        let offset = (tangent * jitter(JITTER_ACROSS, 1.0) + bitangent * jitter(JITTER_ALONG, 1.0)) * settings.jitter_position * radius;
        let flow = (settings.flow * (1.0 - jitter(JITTER_FLOW, settings.jitter_flow).abs())).clamp(0.0, 1.0);

        Stamp {
            center: hit.pos + offset,
            normal,
            tangent,
            bitangent,
            radius: jittered_radius,
            flow,
            hardness: settings.hardness,
            depth: settings.depth,
            seed,
            source: None,
        }
    }

    /// Mirror copies of a stamp for the enabled symmetry axes, then radial copies of it and its mirrors
    fn mirrors(&self, stamp: &Stamp) -> Vec<Stamp> {
        let symmetry = self.settings.symmetry;
        if !symmetry.any() {
            return Vec::new();
        }

        let origin = symmetry.origin(self.center);
        let mut copies = Vec::new();
        let mut maps = Vec::new();
        for sign in symmetry.mirror_signs().skip(1) {
            let normal = stamp.normal * sign;
            let tangent = stamp.tangent * sign;
            let map = Mat3::from_diagonal(sign);
            let source = Some(StampSource { origin, to_source: map.transpose() });
            copies.push(Stamp { center: origin + (stamp.center - origin) * sign, normal, tangent, bitangent: normal.cross(tangent), source, ..*stamp });
            maps.push(map);
        }

        if symmetry.radial_on() {
            let axis = symmetry.radial_axis();
            let to_turn: Vec<(Stamp, Mat3)> = std::iter::once((*stamp, Mat3::IDENTITY)).chain(copies.iter().copied().zip(maps)).collect();

            for copy in 1..symmetry.radial {
                let turn = Quat::from_axis_angle(axis, symmetry.radial_angle(copy));
                copies.extend(to_turn.iter().map(|(base, map)| Stamp {
                    center: origin + turn * (base.center - origin),
                    normal: turn * base.normal,
                    tangent: turn * base.tangent,
                    bitangent: turn * base.bitangent,
                    source: Some(StampSource { origin, to_source: (Mat3::from_quat(turn) * *map).transpose() }),
                    ..*base
                }));
            }
        }

        copies
    }

    pub fn last_hit(&self) -> Option<SurfaceHit> {
        self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_geom::primitives;

    fn ctx_parts() -> (Mesh, Bvh, OrbitCamera) {
        let m = primitives::sphere(1.0, 64, 32);
        let b = Bvh::new(&m);
        let mut c = OrbitCamera::default();
        c.pitch = 0.0;
        c.frame(Vec3::splat(-1.0), Vec3::splat(1.0));
        (m, b, c)
    }

    #[test]
    fn spacing_controls_stamp_count() {
        let (m, b, c) = ctx_parts();
        let ctx = PickCtx { mesh: &m, bvh: &b, camera: &c, viewport: Vec2::new(400.0, 400.0) };
        let count = |spacing: f32| {
            let mut s = Stroke::new(&BrushSettings { spacing, ..BrushSettings::default() }, &m);
            let mut n = 0;
            for i in 0..=40 {
                n += s.add(&ctx, Vec2::new(150.0 + i as f32 * 2.5, 200.0), 1.0).len();
            }
            n
        };
        let dense = count(0.05);
        let sparse = count(0.5);
        assert!(dense > sparse * 5, "{dense} vs {sparse}");
        assert!(sparse >= 2);
    }

    #[test]
    fn misses_produce_nothing_and_stamps_lie_on_the_surface() {
        let (m, b, c) = ctx_parts();
        let ctx = PickCtx { mesh: &m, bvh: &b, camera: &c, viewport: Vec2::new(400.0, 400.0) };
        let mut s = Stroke::new(&BrushSettings::default(), &m);
        assert!(s.add(&ctx, Vec2::new(2.0, 2.0), 1.0).is_empty());
        let st = s.add(&ctx, Vec2::new(200.0, 200.0), 1.0);
        assert_eq!(st.len(), 1);
        assert!((st[0].center.length() - 1.0).abs() < 0.01);
        assert!((st[0].tangent.dot(st[0].normal)).abs() < 1e-4);
    }

    #[test]
    fn symmetry_mirrors_stamps() {
        let (m, b, c) = ctx_parts();
        let ctx = PickCtx { mesh: &m, bvh: &b, camera: &c, viewport: Vec2::new(400.0, 400.0) };
        let mut st = BrushSettings::default();
        st.symmetry.x = true;
        let mut s = Stroke::new(&st, &m);
        let out = s.add(&ctx, Vec2::new(250.0, 200.0), 1.0);
        assert_eq!(out.len(), 2);
        assert!((out[0].center.x + out[1].center.x).abs() < 1e-4);

        st.symmetry.y = true;
        st.symmetry.z = true;
        let mut s = Stroke::new(&st, &m);
        assert_eq!(s.add(&ctx, Vec2::new(250.0, 180.0), 1.0).len(), 8);
    }

    #[test]
    fn symmetry_mirrors_about_the_moved_plane() {
        let (mesh, bvh, camera) = ctx_parts();
        let ctx = PickCtx { mesh: &mesh, bvh: &bvh, camera: &camera, viewport: Vec2::new(400.0, 400.0) };
        let mut settings = BrushSettings::default();
        settings.symmetry.x = true;
        settings.symmetry.offset = [0.25, 0.0, 0.0];
        let mut stroke = Stroke::new(&settings, &mesh);
        let stamps = stroke.add(&ctx, Vec2::new(250.0, 200.0), 1.0);
        assert_eq!(stamps.len(), 2, "one stamp and its copy");
        let (painted, copy) = (stamps[0], stamps[1]);

        // Mirrored about x = 0.25 instead of the mesh center (x = 0)
        assert!((painted.center.x + copy.center.x - 0.5).abs() < 1e-4, "{} {}", painted.center, copy.center);
        assert!((painted.center.y - copy.center.y).abs() < 1e-4 && (painted.center.z - copy.center.z).abs() < 1e-4);
    }

    #[test]
    fn default_symmetry_copies_match_the_mesh_centered_formula_bit_for_bit() {
        let (mesh, bvh, camera) = ctx_parts();
        let ctx = PickCtx { mesh: &mesh, bvh: &bvh, camera: &camera, viewport: Vec2::new(400.0, 400.0) };
        let mut settings = BrushSettings::default();
        settings.symmetry = crate::brush::Symmetry { x: true, y: true, radial: 3, axis: 1, ..settings.symmetry };
        let mut stroke = Stroke::new(&settings, &mesh);
        let stamps = stroke.add(&ctx, Vec2::new(250.0, 180.0), 1.0);

        // Copies as painting computed them before the plane could move: mirrors, then radial turns of each
        let (lo, hi) = mesh.bounds();
        let center = (lo + hi) * 0.5;
        let source = stamps[0].center;
        let mirrored = [Vec3::new(-1.0, 1.0, 1.0), Vec3::new(1.0, -1.0, 1.0), Vec3::new(-1.0, -1.0, 1.0)].map(|sign| center + (source - center) * sign);
        let to_turn: Vec<Vec3> = std::iter::once(source).chain(mirrored).collect();
        let turned = (1..3u8).flat_map(|copy| {
            let turn = Quat::from_axis_angle(Vec3::Y, f32::from(copy) * std::f32::consts::TAU / 3.0);
            to_turn.iter().map(move |point| center + turn * (*point - center)).collect::<Vec<_>>()
        });
        let expected: Vec<Vec3> = mirrored.into_iter().chain(turned).collect();

        let copies: Vec<Vec3> = stamps[1..].iter().map(|stamp| stamp.center).collect();
        assert_eq!(copies, expected);
    }

    #[test]
    fn a_zero_angle_span_paints_no_radial_copies() {
        let (mesh, bvh, camera) = ctx_parts();
        let ctx = PickCtx { mesh: &mesh, bvh: &bvh, camera: &camera, viewport: Vec2::new(400.0, 400.0) };
        let mut settings = BrushSettings::default();
        settings.symmetry.radial = 6;
        settings.symmetry.angle_span = 0.0;
        let mut stroke = Stroke::new(&settings, &mesh);

        // Six copies with no angle between them would all paint over the stroke
        assert_eq!(stroke.add(&ctx, Vec2::new(250.0, 180.0), 1.0).len(), 1);
    }

    #[test]
    fn radial_symmetry_spreads_copies_around_the_axis() {
        let (m, b, c) = ctx_parts();
        let ctx = PickCtx { mesh: &m, bvh: &b, camera: &c, viewport: Vec2::new(400.0, 400.0) };
        let mut st = BrushSettings::default();
        st.symmetry.radial = 4;
        let mut s = Stroke::new(&st, &m);
        let out = s.add(&ctx, Vec2::new(250.0, 180.0), 1.0);
        assert_eq!(out.len(), 4);
        // Same height and distance from the Y axis, a quarter turn apart: they sum to the axis.
        let sum = out.iter().fold(Vec3::ZERO, |a, x| a + x.center);
        assert!(sum.x.abs() < 1e-3 && sum.z.abs() < 1e-3, "{sum}");
        assert!(out.iter().all(|x| (x.center.y - out[0].center.y).abs() < 1e-4 && (x.normal.dot(x.tangent)).abs() < 1e-4));
        // With mirror X as well: each mirror is spread too.
        st.symmetry.x = true;
        let mut s = Stroke::new(&st, &m);
        assert_eq!(s.add(&ctx, Vec2::new(250.0, 180.0), 1.0).len(), 8);
        // Out-of-range settings are clamped.
        st.symmetry = crate::brush::Symmetry { radial: 200, axis: 9, ..st.symmetry };
        let clean = st.sanitized().symmetry;
        assert_eq!((clean.radial, clean.axis), (32, 2));
    }
}
