//! Adaptive samples per pixel for the GPU render.

use super::frame_floor::FrameFloor;

/// Most samples per pixel one frame adds.
pub(super) const MAX_SPP: u32 = 16;

/// A frame quicker than this many times the quickest recent one adds one more sample next time.
/// With vsync, frames come at the display's refresh rate however light they are, so the limits
/// are relative to it (a fixed 1/90 s never let a 60 Hz display grow at all).
const SPP_GROW_BELOW: f64 = 1.25;

/// A frame at least this many times the quickest recent one (a missed refresh with vsync is two;
/// wide enough that ordinary variation in a busy frame does not cut) cuts the samples per frame (by a
/// quarter, by the same ratio the frame overshoots when that is larger), so a slow frame is
/// followed by a short one, not a long one. With vsync a frame that misses the refresh takes two.
const SPP_CUT_ABOVE: f64 = 1.75;

/// Frames after a cut during which the samples per frame do not grow again, so they settle
/// below the limit instead of climbing straight back to the frame that missed it (a hitch every
/// few frames).
pub(super) const SPP_GROW_PAUSE_FRAMES: u32 = 6;

/// Samples per pixel for the next frame, from the last frame's time `frame_seconds` and the quickest recent
/// one `floor` (seconds; see [`FrameFloor`]): one more while frames are quick, fewer when slow,
/// steady in between and for [`SPP_GROW_PAUSE_FRAMES`] after a cut (`since_cut` frames ago).
/// Steady between the two limits is what stops the count from climbing until the GPU backs up,
/// jumping down, and climbing again.
pub(super) fn next_spp(spp: u32, frame_seconds: f64, floor: f64, since_cut: u32) -> u32 {
    let cut_above = floor * SPP_CUT_ABOVE;
    if frame_seconds >= cut_above {
        let by_ratio = (f64::from(spp) * cut_above / frame_seconds) as u32;
        by_ratio.min(spp * 3 / 4).max(1)
    } else if frame_seconds < floor * SPP_GROW_BELOW && since_cut >= SPP_GROW_PAUSE_FRAMES {
        (spp + 1).min(MAX_SPP)
    } else {
        spp
    }
}

/// What steers the GPU render's samples per pixel per frame (see [`next_spp`]).
pub(super) struct SppControl {
    floor: FrameFloor,
    /// Frames since the samples per frame were last cut.
    since_cut: u32,
}

impl Default for SppControl {
    /// A new render may grow from its first frame.
    fn default() -> SppControl {
        SppControl { floor: FrameFloor::default(), since_cut: SPP_GROW_PAUSE_FRAMES }
    }
}

impl SppControl {
    /// The samples per frame after a frame of `frame_seconds` seconds traced with `spp`.
    pub(super) fn next(&mut self, spp: u32, frame_seconds: f64) -> u32 {
        let floor = self.floor.push(frame_seconds);
        let next = next_spp(spp, frame_seconds, floor, self.since_cut);
        self.since_cut = if next < spp { 0 } else { self.since_cut.saturating_add(1) };

        next
    }
}

/// Samples per frame after the traced size changed from `old` to `new` pixels (frame time is
/// about proportional to samples times pixels).
pub(super) fn rescale_spp(spp: u32, old: (u32, u32), new: (u32, u32)) -> u32 {
    let old_pixels = u64::from(old.0) * u64::from(old.1);
    let new_pixels = (u64::from(new.0) * u64::from(new.1)).max(1);

    ((u64::from(spp) * old_pixels / new_pixels).max(1) as u32).min(MAX_SPP)
}
