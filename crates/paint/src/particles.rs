//! Particle brush: each stamp emits a few particles that slide down the surface under gravity
//! (world −Y) and fall off overhangs, painting a small dab per simulation step. Deterministic:
//! particles are seeded by the stamp seed.

use texturecraft_geom::glam::Vec3;
use texturecraft_geom::{Surface, surface_at};
use texturecraft_texel::noise::pcg3d;

use crate::brush::Particles;
use crate::stroke::{PickCtx, Stamp};

/// Particle dab radius relative to the brush stamp.
const PARTICLE_SIZE: f32 = 0.35;

/// The surface hit by a ray, if any within `tmax`.
fn cast(ctx: &PickCtx, o: Vec3, d: Vec3, tmax: f32) -> Option<Surface> {
    let hit = ctx.bvh.closest(o, d, tmax)?;
    surface_at(ctx.mesh, hit.tri, hit.u, hit.v)
}

/// Project `p` onto the surface along the normal `n` (within `reach`).
fn snap(ctx: &PickCtx, p: Vec3, n: Vec3, reach: f32) -> Option<Surface> {
    cast(ctx, p + n * reach, -n, reach * 2.0)
}

/// Particle dabs for one brush stamp.
pub fn emit(ctx: &PickCtx, p: &Particles, s: &Stamp) -> Vec<Stamp> {
    let size = s.radius * PARTICLE_SIZE;
    let step = size * p.gravity;
    let fall = s.radius * 20.0;
    let unit = |v: u32| v as f32 / u32::MAX as f32;
    let mut out = Vec::new();
    for k in 0..p.count {
        let r = pcg3d(s.seed, k, 0xD057);
        let (a, d) = (unit(r[0]) * std::f32::consts::TAU, unit(r[1]).sqrt());
        let start = s.center + (s.tangent * a.cos() + s.bitangent * a.sin()) * d * s.radius;
        let Some(mut at) = snap(ctx, start, s.normal, s.radius) else { continue };
        for life in 0..p.lifetime {
            let tangent = at.nrm.any_orthonormal_vector();
            out.push(Stamp {
                center: at.pos,
                normal: at.nrm,
                tangent,
                bitangent: at.nrm.cross(tangent),
                radius: size,
                flow: s.flow * (1.0 - life as f32 / p.lifetime as f32),
                seed: s.seed.wrapping_mul(17).wrapping_add(k),
                ..*s
            });
            if step <= 0.0 {
                break;
            }
            let slide = Vec3::NEG_Y - at.nrm * at.nrm.dot(Vec3::NEG_Y);
            let next = if slide.length() > 0.05 {
                // Slide downhill; past an edge the particle drops onto what lies below.
                let to = at.pos + slide.normalize() * step;
                snap(ctx, to, at.nrm, step).or_else(|| cast(ctx, to + at.nrm * step, Vec3::NEG_Y, fall))
            } else if at.nrm.y < 0.0 {
                // Hanging under a ceiling: fall straight down.
                cast(ctx, at.pos + at.nrm * step, Vec3::NEG_Y, fall)
            } else {
                None // resting on flat ground
            };
            match next {
                Some(h) => at = h,
                None => break,
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_geom::glam::Vec2;
    use texturecraft_geom::{Bvh, OrbitCamera, primitives};

    #[test]
    fn particles_slide_down_a_wall_deterministically() {
        // A vertical wall: the plane rotated to face +Z
        let mut mesh = primitives::plane(2.0, 8);
        for position in &mut mesh.positions {
            *position = Vec3::new(position.x, -position.z, 0.0);
        }
        mesh.normals.fill(Vec3::Z);
        let bvh = Bvh::new(&mesh);
        let camera = OrbitCamera::default();
        let context = PickCtx { mesh: &mesh, bvh: &bvh, camera: &camera, viewport: Vec2::new(100.0, 100.0) };
        let stamp = Stamp {
            center: Vec3::ZERO,
            normal: Vec3::Z,
            tangent: Vec3::X,
            bitangent: Vec3::Y,
            radius: 0.1,
            flow: 1.0,
            hardness: 0.5,
            depth: 1.0,
            seed: 3,
            source: None,
        };
        let particles = Particles { enabled: true, count: 4, lifetime: 20, gravity: 1.0 };

        let dabs = emit(&context, &particles, &stamp);

        assert_eq!(dabs, emit(&context, &particles, &stamp));
        assert_eq!(dabs.len(), 80);
        let lowest = dabs.iter().map(|d| d.center.y).fold(f32::MAX, f32::min);
        assert!(lowest < -0.4, "{lowest}");
        assert!(dabs.iter().all(|d| d.center.z.abs() < 1e-3));
    }
}
