//! Paired tracing of one immutable [`TraceScene`], without a window-system dependency.
//!
//! [`TracePairFrame`] keeps the existing orbit target as the zero-parallax plane and derives
//! cameras from the actual traced geometry. The caller supplies depth limits in native eye
//! samples and retains the scene's `Arc` until its source revision changes.
//! [`CpuTracePair`] publishes only complete equal-sample boundaries, including when work is
//! sliced into rows. [`GpuTracePair`] shares uploaded geometry, maps and environment data while
//! each eye owns its uniforms, accumulation, sample counter, denoising and output image.
//!
//! Before GPU tracing, finish pending scene uploads with [`Renderer::prepare_trace_resources`].
//! An `advance` encodes both views into one submission; its counters describe scheduled work
//! until that submission completes. Read both outputs after submission before publishing a pair.
//! Scene, camera, quality, native size and depth changes invalidate both histories together.
use std::sync::Arc;

use craft_stereo_camera::{Budget, Camera, CameraPair, DepthRange, ProjectionKind};
use texturecraft_geom::{OrbitCamera, glam::Vec3};

use crate::{
    Renderer,
    gpu_trace::GpuTracer,
    trace::{TraceDisplay, TraceScene, TraceSettings, Tracer},
};

#[derive(Clone)]
pub struct TracePairFrame {
    pub scene: Arc<TraceScene>,
    pub camera: OrbitCamera,
    pub size: [u32; 2],
    pub budget: Budget,
    pub settings: TraceSettings,
    pub display: TraceDisplay,
    pub max_samples: u32,
    /// The owner's composition revision, never a camera or history buffer.
    pub epoch: String,
}

impl TracePairFrame {
    pub fn cameras(&self) -> Result<CameraPair, String> {
        let c = self.camera;
        let eye = c.eye();
        if self.size.iter().any(|s| !(1..=8192).contains(s))
            || !eye.is_finite()
            || !c.target.is_finite()
            || !c.distance.is_finite()
            || c.distance <= 0.0
            || !c.fov.is_finite()
            || !c.scene_radius.is_finite()
            || (c.target - eye).try_normalize().is_none_or(|f| f.cross(Vec3::Y).try_normalize().is_none())
        {
            return Err("the traced stereo camera or native dimensions are invalid".into());
        }
        let aspect = self.size[0] as f32 / self.size[1] as f32;
        let near = (c.distance - c.scene_radius.max(1e-3) * 4.0).max(c.scene_radius.max(1e-3) * 0.01).max(1e-4);
        let (lo, hi) = self.scene.mesh.bounds();
        let view = c.view();
        Camera {
            eye,
            view,
            projection: c.proj(aspect),
            kind: if c.ortho { ProjectionKind::Orthographic } else { ProjectionKind::Perspective },
            // Orthographic tracing has parallel miss rays, not a perspective sky image.
            infinity_projection_x: if self.display.env_opacity > 0.0 && !c.ortho { c.proj(aspect).x_axis.x } else { 0.0 },
            zero_depth: c.distance,
            depths: DepthRange::from_aabb(view, lo, hi, if c.ortho { near.min(0.01) } else { near }).map_err(|e| e.to_string())?,
            width: self.size[0],
            budget: self.budget,
        }
        .pair()
        .map_err(|e| e.to_string())
    }

    pub fn key(&self) -> String {
        format!("{}|{:?}|{:?}|{:?}|{:?}|{:?}|{}", self.epoch, self.camera, self.size, self.budget, self.settings, self.display, self.max_samples)
    }
}

impl PartialEq for TracePairFrame {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.scene, &other.scene)
            && self.camera == other.camera
            && self.size == other.size
            && self.budget == other.budget
            && self.settings == other.settings
            && self.display == other.display
            && self.max_samples == other.max_samples
            && self.epoch == other.epoch
    }
}

/// CPU accumulation and denoising are private to each eye. A frame is returned only at equal N.
pub struct CpuTracePair {
    frame: TracePairFrame,
    cameras: CameraPair,
    eyes: [Tracer; 2],
    building: Option<u32>,
}

impl CpuTracePair {
    pub fn new(frame: TracePairFrame) -> Result<Self, String> {
        let cameras = frame.cameras()?;
        let eyes = [Tracer::try_new(frame.size[0], frame.size[1])?, Tracer::try_new(frame.size[0], frame.size[1])?];
        Ok(Self { frame, cameras, eyes, building: None })
    }

    pub fn samples(&self) -> [u32; 2] {
        self.eyes.each_ref().map(|t| t.samples)
    }

    pub fn advance(&mut self) -> Result<bool, String> {
        if self.building.is_none() && self.eyes[0].samples >= self.frame.max_samples.max(1) {
            return Ok(false);
        }
        while !self.advance_rows(self.frame.size[1])? {}
        Ok(true)
    }

    /// True at a completed equal-N boundary. The caller can cancel between small row slices.
    pub fn advance_rows(&mut self, rows: u32) -> Result<bool, String> {
        let target = match self.building {
            Some(target) => target,
            None => {
                if self.eyes[0].samples != self.eyes[1].samples {
                    return Err("the CPU trace eyes lost their shared sample boundary".into());
                }
                if self.eyes[0].samples >= self.frame.max_samples.max(1) {
                    return Ok(false);
                }
                let target = self.eyes[0].samples.saturating_add(1);
                self.building = Some(target);
                target
            }
        };
        for (tracer, eye) in self.eyes.iter_mut().zip(self.cameras.eyes) {
            if tracer.samples < target {
                tracer.pass_rows_eye(&self.frame.scene, &self.frame.camera, eye, &self.frame.settings, rows)?;
            }
        }
        if self.eyes.iter().all(|t| t.samples == target) {
            self.building = None;
            Ok(true)
        } else {
            Ok(false)
        }
    }

    #[cfg(test)]
    pub(crate) fn test_linear_means(&self) -> Result<[Vec<[f32; 3]>; 2], String> {
        if self.building.is_some() || self.eyes[0].samples == 0 || self.eyes[0].samples != self.eyes[1].samples {
            return Err("linear means require two complete equal-sample views".into());
        }
        Ok(self.eyes.each_ref().map(Tracer::test_linear_mean))
    }

    pub fn rgba8(&self) -> Result<[Vec<u8>; 2], String> {
        if self.building.is_some() || self.eyes[0].samples == 0 || self.eyes[0].samples != self.eyes[1].samples {
            return Err("a CPU stereo pair needs two completed equal sample counts".into());
        }
        Ok(self.eyes.each_ref().map(|t| t.rgba8(&self.frame.display)))
    }
}

/// GPU views share only prepared world resources, retaining private uniforms, histories and N.
pub struct GpuTracePair {
    eyes: [GpuTracer; 2],
    scene: Option<Arc<TraceScene>>,
    key: Option<String>,
    outputs: Option<[wgpu::Texture; 2]>,
    materials: Option<u64>,
}

impl GpuTracePair {
    pub fn new(device: &wgpu::Device) -> Result<Self, String> {
        Ok(Self { eyes: [GpuTracer::new(device)?, GpuTracer::new(device)?], scene: None, key: None, outputs: None, materials: None })
    }

    pub fn samples(&self) -> [u32; 2] {
        self.eyes.each_ref().map(|t| t.samples())
    }

    #[cfg(test)]
    pub(crate) fn test_views(&self) -> &[GpuTracer; 2] {
        &self.eyes
    }

    pub fn material_revision(&self) -> Option<u64> {
        self.materials
    }

    pub fn outputs(&self) -> Option<&[wgpu::Texture; 2]> {
        self.outputs.as_ref()
    }

    pub fn advance(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        renderer: &Renderer,
        frame: &TracePairFrame,
    ) -> Result<bool, String> {
        let cameras = frame.cameras()?;
        let key = frame.key();
        let scene_changed = self.scene.as_ref().is_none_or(|scene| !Arc::ptr_eq(scene, &frame.scene));
        let [left, right] = &mut self.eyes;
        if scene_changed {
            left.set_scene(device, queue, &frame.scene)?;
        }
        let materials_changed = left.sync_materials(device, queue, encoder, renderer)?;
        self.materials = Some(renderer.content_rev());
        right.share_scene_from(left);
        if scene_changed || materials_changed || self.key.as_ref() != Some(&key) {
            left.reset(device, encoder, frame.size[0], frame.size[1])?;
            right.reset(device, encoder, frame.size[0], frame.size[1])?;
            if self.outputs.as_ref().is_none_or(|o| (o[0].width(), o[0].height()) != (frame.size[0], frame.size[1])) {
                self.outputs = Some(crate::gpu_trace::scoped(device, "GPU stereo trace output allocation", || {
                    std::array::from_fn(|_| GpuTracer::output_texture(device, frame.size[0], frame.size[1]))
                })?);
            }
            self.scene = Some(frame.scene.clone());
            self.key = Some(key);
        }
        if left.samples() != right.samples() {
            self.key = None;
            return Err("the GPU trace eyes lost their shared sample boundary".into());
        }
        if left.samples() >= frame.max_samples.max(1) {
            return Ok(false);
        }
        for (tracer, eye) in self.eyes.iter_mut().zip(&cameras.eyes) {
            if !tracer.pass_eye(device, queue, encoder, &frame.camera, eye, &frame.settings, 1) {
                self.key = None;
                return Err("the GPU stereo trace has no complete prepared world".into());
            }
        }
        let Some(outputs) = &self.outputs else {
            return Err("the GPU stereo trace has no private output targets".into());
        };
        for (tracer, out) in self.eyes.iter().zip(outputs) {
            if !tracer.resolve(device, queue, encoder, &frame.display, &out.create_view(&Default::default())) {
                self.key = None;
                return Err("the GPU stereo trace could not resolve both completed views".into());
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::Tracer;
    use texturecraft_geom::{Bvh, primitives};

    #[test]
    fn stereo_cpu_zero_budget_preserves_every_mono_sample_with_lens_and_denoising() {
        let mesh = Arc::new(primitives::sphere(1.0, 16, 8));
        let bvh = Arc::new(Bvh::new(&mesh));
        let mut scene = TraceScene::new(mesh, bvh, vec![], Arc::new(crate::ibl::environment("Studio Contrast", 32)), 0.0);
        scene.lens = Some([4.0, 0.1]);
        let frame = TracePairFrame {
            scene: Arc::new(scene),
            camera: OrbitCamera { distance: 4.0, ..Default::default() },
            size: [24, 16],
            budget: Budget { sunk: 0.0, popped: 0.0 },
            settings: TraceSettings::default(),
            display: TraceDisplay { exposure: 1.0, aces: true, env_opacity: 1.0, denoise: true },
            max_samples: 2,
            epoch: "flat".into(),
        };
        let mut pair = CpuTracePair::new(frame.clone()).unwrap();
        let mut mono = Tracer::new(24, 16);
        assert!(pair.rgba8().is_err(), "a partial pair is not publishable");
        for n in 1..=2 {
            mono.pass(&frame.scene, &frame.camera, &frame.settings);
            if n == 2 {
                assert!(!pair.advance_rows(3).unwrap());
                assert_eq!(pair.samples(), [1, 1]);
                assert!(pair.rgba8().is_err(), "equal old counts do not make a partial next pass publishable");
            }
            assert!(pair.advance().unwrap());
            assert_eq!(pair.samples(), [n, n]);
            let pixels = pair.rgba8().unwrap();
            for eye in pixels {
                assert_eq!(eye, mono.rgba8(&frame.display));
            }
        }
        assert!(!pair.advance().unwrap());
    }
}
