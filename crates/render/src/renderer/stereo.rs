//! Parallel eye projections of the existing uploaded world.
use super::*;
use craft_stereo_camera::{Budget, Camera, CameraPair, DepthRange, ProjectionKind};
use texturecraft_core::MAX_VIEW_SIDE;

#[derive(Default)]
pub(super) struct HeightBounds {
    tiles: HashMap<(u32, u32), f32>,
    peak: f32,
}

impl HeightBounds {
    pub(super) fn record(&mut self, tx: u32, ty: u32, w: u32, h: u32, stride: u32, data: &[u8]) {
        let mut peak = 0f32;
        for y in 0..h {
            for x in 0..w {
                let start = (u64::from(y) * u64::from(stride) + u64::from(x)) * 8;
                let value = usize::try_from(start).ok().and_then(|i| data.get(i..i + 2));
                let value = value.map_or(f32::INFINITY, |p| half::f16::from_le_bytes([p[0], p[1]]).to_f32().abs());
                peak = if value.is_finite() { peak.max(value) } else { f32::INFINITY };
            }
        }
        let old = self.tiles.insert((tx, ty), peak);
        if peak >= self.peak {
            self.peak = peak;
        } else if old == Some(self.peak) {
            self.peak = self.tiles.values().copied().fold(0.0, f32::max);
        }
    }
}

pub(super) fn bounds(points: &[Vec3]) -> Option<(Vec3, Vec3)> {
    let first = *points.first()?;
    let (mut lo, mut hi) = (first, first);
    for p in points {
        if !p.is_finite() {
            return Some((Vec3::NAN, Vec3::NAN));
        }
        lo = lo.min(*p);
        hi = hi.max(*p);
    }
    Some((lo, hi))
}

impl Renderer {
    pub fn mesh_id(&self) -> Option<u64> {
        self.mesh.as_ref().map(|m| m.id)
    }

    /// Derive cameras from conservative uploaded geometry bounds and signed vertex displacement.
    /// The host supplies live disparity limits in native eye samples. Camera units, target,
    /// clipping and mono projection remain those of the existing OrbitCamera.
    /// Bake high-poly and cage geometry contribute when visible.
    pub fn camera_pair(&self, frame: &Frame, budget: Budget) -> Result<CameraPair, String> {
        if frame.kind != ViewKind::Scene || frame.size.iter().any(|s| !(1..=MAX_VIEW_SIDE).contains(s)) {
            return Err("stereo needs a scene view with supported native dimensions".into());
        }
        let cam = frame.camera;
        let eye = cam.eye();
        let direction = (cam.target - eye).try_normalize();
        if !cam.target.is_finite()
            || !eye.is_finite()
            || !cam.distance.is_finite()
            || cam.distance <= 0.0
            || !cam.fov.is_finite()
            || !cam.scene_radius.is_finite()
            || direction.is_none_or(|f| f.cross(Vec3::Y).try_normalize().is_none())
        {
            return Err("the model camera has no finite pose or projection".into());
        }
        let r = cam.scene_radius.max(1e-3);
        let near = (cam.distance - r * 4.0).max(r * 0.01).max(1e-4);
        let far = cam.distance + r * 50.0;
        let aspect = frame.size[0] as f32 / frame.size[1] as f32;
        let half_h = cam.distance * (cam.fov.to_radians() * 0.5).tan();
        if !far.is_finite() || near >= far || (cam.ortho && (!half_h.is_finite() || half_h <= 0.0 || !(half_h * aspect).is_finite() || half_h * aspect == 0.0))
        {
            return Err("the model camera clip range or orthographic extent is invalid".into());
        }
        let view = cam.view();
        let mut world: Option<(Vec3, Vec3)> = None;
        let mut include = |candidate: Option<(Vec3, Vec3)>| -> Result<(), String> {
            if let Some((lo, hi)) = candidate {
                if !lo.is_finite() || !hi.is_finite() || !lo.cmple(hi).all() {
                    return Err("the uploaded model or displacement bounds are not finite".into());
                }
                world = Some(world.map_or((lo, hi), |(a, b)| (a.min(lo), b.max(hi))));
            }
            Ok(())
        };
        if frame.shade.bake.is_none_or(|b| b.low_poly)
            && let Some(mesh) = &self.mesh
        {
            let peak = if frame.shade.displacement {
                let slot = DisplayKind::ALL.iter().position(|k| *k == DisplayKind::Height);
                let height = self
                    .sets
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| frame.visible.get(*i).copied().unwrap_or(true))
                    .map(|(_, s)| {
                        let dims = display_dims(s.size, s.shift);
                        // Pending allocations still have a neutral placeholder bound. Reserve
                        // only the currently rendered texture; resolve_pending precedes a pair.
                        let active = slot.and_then(|i| s.tex.get(i)).is_some_and(|t| t.texture.width() >= dims.0 && t.texture.height() >= dims.1);
                        if active { s.height.peak } else { 0.0 }
                    })
                    .fold(0.0, f32::max);
                height * frame.shade.height_strength.abs() * mesh.world_per_uv / 16.0
            } else {
                0.0
            };
            if !peak.is_finite() {
                return Err("the uploaded displacement envelope is not finite".into());
            }
            include(Some((mesh.bounds.0 - Vec3::splat(peak), mesh.bounds.1 + Vec3::splat(peak))))?;
        }
        if let Some(bake) = frame.shade.bake {
            if bake.high_poly {
                include(self.bake_high.as_ref().and_then(|h| h.bounds))?;
            }
            if bake.cage {
                for (i, cage) in self.cages.iter().enumerate() {
                    if bake.only_set.is_none_or(|only| only == i) && frame.visible.get(i).copied().unwrap_or(true) {
                        include(cage.as_ref().and_then(|c| c.bounds))?;
                    }
                }
            }
        }
        let (lo, hi) = world.unwrap_or((cam.target, cam.target));
        let depths = DepthRange::from_aabb(view, lo, hi, if cam.ortho { near.min(0.01) } else { near }).map_err(|e| e.to_string())?;
        Camera {
            eye,
            view,
            projection: cam.proj(aspect),
            infinity_projection_x: OrbitCamera { ortho: false, ..cam }.proj(aspect).x_axis.x,
            kind: if cam.ortho { ProjectionKind::Orthographic } else { ProjectionKind::Perspective },
            zero_depth: cam.distance,
            depths,
            width: frame.size[0],
            budget,
        }
        .pair()
        .map_err(|e| e.to_string())
    }

    /// Resolve uploads once, then draw both cameras of one frame into private targets.
    /// Apply the scene sync first. The two IDs must differ from each other and the ordinary
    /// frame's ID; each owns its screen-space effects and temporal history.
    pub fn prepare_stereo(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        frame: &Frame,
        ids: [u64; 2],
        budget: Budget,
    ) -> Result<CameraPair, String> {
        if ids[0] == ids[1] || ids.contains(&frame.view_id) {
            return Err("stereo needs two private targets distinct from the ordinary view".into());
        }
        self.resolve_pending(device);
        self.generate_mips(device, encoder);
        let pair = self.camera_pair(frame, budget)?;
        for (id, eye) in ids.into_iter().zip(&pair.eyes) {
            let frame = Frame { view_id: id, ..frame.clone() };
            self.prepare_ready(device, queue, encoder, &frame, Some(eye));
        }
        Ok(pair)
    }

    /// Encode a nonblocking copy. Map once after submission, then poll/take it on the next frame.
    pub fn copy_view(&self, device: &wgpu::Device, encoder: &mut wgpu::CommandEncoder, id: u64) -> Option<crate::readback::Readback> {
        let view = self.views.get(&id).filter(|v| v.ready)?;
        Some(crate::readback::Readback::new(device, encoder, &view.resolve))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn height_bounds_follow_actual_channels_extents_and_overwrites() {
        let mut bytes = vec![0u8; 64 * 64 * 8];
        for pixel in bytes.chunks_exact_mut(8) {
            pixel[..2].copy_from_slice(&half::f16::from_f32(1000.0).to_le_bytes());
        }
        bytes[..2].copy_from_slice(&half::f16::from_f32(-3.0).to_le_bytes());
        bytes[64 * 8..64 * 8 + 2].copy_from_slice(&half::f16::from_f32(5.0).to_le_bytes());
        let mut bounds = HeightBounds::default();
        bounds.record(0, 0, 1, 2, 64, &bytes);
        assert_eq!(bounds.peak, 5.0, "padding outside the actual upload is not rendered");
        bounds.record(1, 0, 1, 1, 64, &half::f16::from_f32(-7.0).to_le_bytes());
        assert_eq!(bounds.peak, 7.0);
        bounds.record(1, 0, 1, 1, 64, &[0; 8]);
        assert_eq!(bounds.peak, 5.0, "a shrinking maximum is recomputed from cached tile bounds");
        bounds.record(0, 0, 1, 1, 64, &[0; 8]);
        assert_eq!(bounds.peak, 0.0);
        bounds.record(0, 0, 1, 1, 64, &half::f16::NAN.to_le_bytes());
        assert!(bounds.peak.is_infinite(), "a non-finite uploaded height makes bounds invalid");
        bounds.record(0, 0, 1, 1, 64, &[0; 8]);
        assert_eq!(bounds.peak, 0.0, "repairing the tile clears its invalid envelope");
    }
}
