//! A texture set's mesh edges drawn over the 3D view in a highlight color (the set under a shelf drag)

/// Brightest on-screen value the tone curve can reach without an infinite scene value
const MAX_ACES_SCREEN_VALUE: f32 = 0.99;

/// Scene scale the post pass applies before its ACES curve (`aces` in post.wgsl)
const ACES_INPUT_SCALE: f32 = 0.6;

/// Draw texture set `set`'s mesh edges over the 3D view in `color`
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SetHighlight {
    pub set: usize,
    /// The lines' color on screen, linear RGB
    pub color: [f32; 3],
}

impl SetHighlight {
    /// The scene color the lines are drawn with so that exposure `exposure_ev` and the tone curve (ACES when `aces`) show them in [`Self::color`]
    pub fn scene_color(&self, exposure_ev: f32, aces: bool) -> [f32; 3] {
        scene_color_showing(self.color, exposure_ev, aces)
    }
}

/// The scene color that exposure `exposure_ev` and the tone curve (ACES when `aces`) show as `screen` (linear RGB)
pub(crate) fn scene_color_showing(screen: [f32; 3], exposure_ev: f32, aces: bool) -> [f32; 3] {
    let exposure = 2f32.powf(exposure_ev).max(f32::MIN_POSITIVE);

    screen.map(|screen| {
        let screen = screen.max(0.0);
        let exposed = if aces { inverse_aces(screen.min(MAX_ACES_SCREEN_VALUE)) } else { screen };

        exposed / exposure
    })
}

/// The input of post.wgsl's `aces` that gives `screen`: the positive root of its rational curve solved for the input
fn inverse_aces(screen: f32) -> f32 {
    let (a, b, c) = (2.43 * screen - 2.51, 0.59 * screen - 0.03, 0.14 * screen);
    let scaled = (-b - (b * b - 4.0 * a * c).max(0.0).sqrt()) / (2.0 * a);

    scaled / ACES_INPUT_SCALE
}

#[cfg(test)]
mod tests {
    use super::*;

    /// post.wgsl's `aces` with exposure `exposure_ev`
    fn tone_mapped(scene: f32, exposure_ev: f32) -> f32 {
        let v = scene * 2f32.powf(exposure_ev) * ACES_INPUT_SCALE;
        ((v * (2.51 * v + 0.03)) / (v * (2.43 * v + 0.59) + 0.14)).clamp(0.0, 1.0)
    }

    #[test]
    fn the_scene_color_shows_the_highlight_color_on_screen() {
        let highlight = SetHighlight { set: 0, color: [0.9, 0.3, 0.05] };
        for exposure_ev in [-2.0, 0.0, 1.5] {
            let scene = highlight.scene_color(exposure_ev, true);
            for (scene, screen) in scene.iter().zip(highlight.color) {
                assert!((tone_mapped(*scene, exposure_ev) - screen).abs() < 1e-4, "{exposure_ev} EV: {scene} → {screen}");
            }
        }

        assert_eq!(highlight.scene_color(1.0, false), [0.45, 0.15, 0.025], "linear: exposure only");
    }
}
