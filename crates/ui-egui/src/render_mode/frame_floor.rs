//! Recent frame-time baseline for adaptive sampling.

/// Frames over which [`FrameFloor`] keeps the quickest frame time.
pub(super) const FLOOR_FRAMES: usize = 30;

/// The range the quickest frame time is held to (seconds): from 75 frames a second to 30, so a
/// run of slow frames never makes slow look normal. The quick end matters as much: frames that
/// are not paced by the display (or tiny ones while little is traced) can take 3 ms, and taking
/// that as the baseline cuts the samples per frame to one at the first frame that does real work,
/// which left the render grainy for seconds. Below this, frames count as quick, not as the target.
pub(super) const FLOOR_RANGE_S: (f64, f64) = (1.0 / 75.0, 1.0 / 30.0);

/// The quickest of the last [`FLOOR_FRAMES`] frame times: about the display's refresh interval
/// while the GPU keeps up.
#[derive(Default)]
pub(super) struct FrameFloor {
    recent: std::collections::VecDeque<f64>,
}

impl FrameFloor {
    /// Record frame time `frame_seconds` (seconds); the quickest recent one, within [`FLOOR_RANGE_S`].
    pub(super) fn push(&mut self, frame_seconds: f64) -> f64 {
        if frame_seconds.is_finite() && frame_seconds > 0.0 {
            if self.recent.len() >= FLOOR_FRAMES {
                self.recent.pop_front();
            }

            self.recent.push_back(frame_seconds);
        }

        self.recent.iter().copied().fold(FLOOR_RANGE_S.1, f64::min).clamp(FLOOR_RANGE_S.0, FLOOR_RANGE_S.1)
    }
}
