//! Camera and texture-loading activity used to select render resolution.

use texturecraft_engine::geom::OrbitCamera;

use super::MOTION_SETTLE_S;

/// Whether the camera is moving, for [`super::MOVING_SCALE`].
#[derive(Default)]
pub(super) struct Motion {
    /// The camera as last seen (compared, never parsed).
    camera: String,
    /// When it last changed. The first frame counts as a change: a render that has just started
    /// begins at the smaller size too (four times the samples at once, so it opens smooth
    /// instead of as raw one-sample noise) and grows to full size like after any move.
    last_change: Option<f64>,
}

impl Motion {
    /// Record `camera` at time `now` (seconds); true while it changed within [`MOTION_SETTLE_S`].
    /// `loading`: textures are still being composited. Each finished tile restarts the render, so
    /// like a moving camera it keeps being redone from no samples, and is traced small for the
    /// same reason (four times the samples per restart).
    pub(super) fn moving(&mut self, camera: &OrbitCamera, now: f64, loading: bool) -> bool {
        if loading {
            self.last_change = Some(now);
        }

        let camera_key = format!("{camera:?}");
        if self.camera != camera_key {
            self.last_change = Some(now);
            self.camera = camera_key;
        }

        self.last_change.is_some_and(|last_change| now - last_change < MOTION_SETTLE_S)
    }
}
