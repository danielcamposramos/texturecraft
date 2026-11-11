//! Brush settings and alpha shapes.

use serde::{Deserialize, Serialize};
use texturecraft_compose::ImageRes;
use texturecraft_geom::glam::{Mat3, Vec3};
use texturecraft_texel::noise;

use crate::alphas::AlphaImage;

/// How the stamp plane is oriented.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Alignment {
    /// Along the surface (tangent to the hit normal).
    #[default]
    Tangent,
    /// Facing the camera (stable on noisy geometry).
    Camera,
}

/// The stamp shape (multiplied by the hardness falloff).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum BrushAlpha {
    #[default]
    Round,
    Square,
    /// Speckled spray.
    Spray,
    /// Organic grunge blotches.
    Grunge,
    /// Soft cloud.
    Cloud,
    /// Five-point star.
    Star,
    /// Horizontal streaks (dry brush).
    Streaks,
}

impl BrushAlpha {
    pub const ALL: [BrushAlpha; 7] =
        [BrushAlpha::Round, BrushAlpha::Square, BrushAlpha::Spray, BrushAlpha::Grunge, BrushAlpha::Cloud, BrushAlpha::Star, BrushAlpha::Streaks];

    pub fn label(self) -> &'static str {
        match self {
            BrushAlpha::Round => "Round",
            BrushAlpha::Square => "Square",
            BrushAlpha::Spray => "Spray",
            BrushAlpha::Grunge => "Grunge",
            BrushAlpha::Cloud => "Cloud",
            BrushAlpha::Star => "Star",
            BrushAlpha::Streaks => "Dry streaks",
        }
    }

    pub fn parse(s: &str) -> Option<BrushAlpha> {
        let n = |x: &str| x.to_ascii_lowercase().replace([' ', '_', '-'], "");
        BrushAlpha::ALL.into_iter().find(|a| n(a.label()) == n(s) || n(&format!("{a:?}")) == n(s))
    }

    /// Shape value at local stamp coordinates `l` (unit disc/square, −1..1), seeded per stamp.
    pub fn sample(self, lx: f32, ly: f32, hardness: f32, seed: u32) -> f32 {
        let r = (lx * lx + ly * ly).sqrt();
        let fall = |d: f32| 1.0 - smoothstep(hardness.min(0.999), 1.0, d);
        match self {
            BrushAlpha::Round => fall(r),
            BrushAlpha::Square => fall(lx.abs().max(ly.abs())),
            BrushAlpha::Spray => {
                let n = noise::value([lx * 14.0, ly * 14.0, seed as f32 * 0.37], seed);
                fall(r) * if n > 0.72 { 1.0 } else { 0.0 }
            }
            BrushAlpha::Grunge => {
                let n = noise::fbm([lx * 2.5 + seed as f32 * 0.13, ly * 2.5, 0.5], 5, seed, noise::perlin);
                fall(r) * ((n - 0.42) * 5.0).clamp(0.0, 1.0)
            }
            BrushAlpha::Cloud => fall(r) * noise::fbm([lx * 1.8, ly * 1.8, seed as f32 * 0.21], 4, seed, noise::value),
            BrushAlpha::Star => {
                let a = ly.atan2(lx);
                let k = (a * 5.0 / 2.0).cos().abs();
                let edge = 0.45 + 0.55 * k * k;
                fall(r / edge)
            }
            BrushAlpha::Streaks => {
                let n = noise::value([lx * 0.8, ly * 18.0, seed as f32 * 0.7], seed);
                fall(r) * ((n - 0.35) * 3.0).clamp(0.0, 1.0)
            }
        }
    }
}

/// A stamp shape ready to sample: an image alpha when one is set, else a library alpha when one
/// is set, else the procedural alpha.
#[derive(Clone, Copy)]
pub struct Shape<'a> {
    pub alpha: BrushAlpha,
    pub image: Option<&'a ImageRes>,
    pub library: Option<&'a AlphaImage>,
    /// Mip level of `library` to sample (set per stamp by [`Shape::at_radius`]).
    pub level: usize,
}

impl Shape<'_> {
    /// Shape value at local stamp coordinates (see [`BrushAlpha::sample`]).
    pub fn sample(&self, lx: f32, ly: f32, hardness: f32, seed: u32) -> f32 {
        match (self.image, self.library) {
            (Some(img), _) => image_alpha(img, lx, ly),
            (None, Some(lib)) => lib.sample(self.level, lx, ly),
            (None, None) => self.alpha.sample(lx, ly, hardness, seed),
        }
    }

    /// This shape for a stamp `radius` texels wide (picks the library alpha's mip level).
    pub fn at_radius(self, radius: f32) -> Self {
        Shape { level: self.library.map_or(0, |l| l.level_for(radius)), ..self }
    }
}

/// An image as a brush alpha: luminance × alpha, its longer side across the stamp square (−1..1,
/// y up) so it keeps its proportions (a line of text stays a line), 0 outside.
pub fn image_alpha(img: &ImageRes, lx: f32, ly: f32) -> f32 {
    let aspect = img.width as f32 / img.height.max(1) as f32;
    let (x, y) = if aspect >= 1.0 { (lx, ly * aspect) } else { (lx / aspect.max(1e-6), ly) };
    if !(x.abs() <= 1.0 && y.abs() <= 1.0) {
        return 0.0;
    }
    let [r, g, b, a] = img.sample(x * 0.5 + 0.5, 0.5 - y * 0.5);
    ((0.2126 * r + 0.7152 * g + 0.0722 * b) * a).clamp(0.0, 1.0)
}

pub(crate) fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0).max(1e-6)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// Largest symmetry offset kept from a command or a file (world units), far beyond any mesh
const MAX_SYMMETRY_OFFSET: f32 = 1.0e6;

/// Mirror and radial symmetry through the mesh center, optionally moved by `offset`
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Symmetry {
    /// The toolbar's Symmetry toggle: off keeps the settings below but paints no copies
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    pub x: bool,
    pub y: bool,
    pub z: bool,
    /// Radial copies around `axis`, spread over `angle_span` (0 or 1: off; at most [`MAX_RADIAL_COPIES`])
    #[serde(default)]
    pub radial: u8,
    /// Axis of radial symmetry (0 = X, 1 = Y, 2 = Z), independent of the mirror axes
    #[serde(default = "radial_axis")]
    pub axis: u8,
    /// Where the mirror planes and the radial axis cross, relative to the mesh center (world units)
    #[serde(default)]
    pub offset: [f32; 3],
    /// Degrees the radial copies spread over: a full turn spaces them evenly, less puts the first and last copy at its ends
    #[serde(default = "full_turn")]
    pub angle_span: f32,
}

fn enabled_by_default() -> bool {
    true
}

fn radial_axis() -> u8 {
    1
}

fn full_turn() -> f32 {
    FULL_TURN_DEGREES
}

/// [`Symmetry::angle_span`] of a full turn
pub const FULL_TURN_DEGREES: f32 = 360.0;

/// Most radial copies a brush paints
pub const MAX_RADIAL_COPIES: u8 = 32;

/// [`Symmetry::axis`] of the last axis, Z
const LAST_AXIS: u8 = 2;

impl Default for Symmetry {
    fn default() -> Self {
        Symmetry { enabled: enabled_by_default(), x: false, y: false, z: false, radial: 0, axis: radial_axis(), offset: [0.0; 3], angle_span: full_turn() }
    }
}

impl Symmetry {
    /// Whether symmetry is on with an axis to mirror on or radial copies
    pub fn any(&self) -> bool {
        self.enabled && (self.x || self.y || self.z || self.radial > 1)
    }

    /// Whether radial symmetry is on (see [`Symmetry::radial_on`] for whether it paints copies)
    pub fn radial_mode(&self) -> bool {
        self.enabled && self.radial > 1
    }

    /// Whether radial copies are painted: with no angle to spread over they would all land on the stroke
    pub fn radial_on(&self) -> bool {
        self.radial_mode() && self.angle_span > 0.0
    }

    /// Rotation (radians) of radial copy number `copy` (0 = the stroke itself) around the axis
    pub fn radial_angle(&self, copy: u8) -> f32 {
        if self.angle_span >= FULL_TURN_DEGREES {
            f32::from(copy) * std::f32::consts::TAU / f32::from(self.radial.max(1))
        } else {
            f32::from(copy) * self.angle_span.to_radians() / f32::from(self.radial.saturating_sub(1).max(1))
        }
    }

    /// Mirror scales in stroke order, including identity, without allocating
    pub(crate) fn mirror_signs(&self) -> impl Iterator<Item = Vec3> {
        [
            Vec3::ONE,
            Vec3::new(-1.0, 1.0, 1.0),
            Vec3::new(1.0, -1.0, 1.0),
            Vec3::new(-1.0, -1.0, 1.0),
            Vec3::new(1.0, 1.0, -1.0),
            Vec3::new(-1.0, 1.0, -1.0),
            Vec3::new(1.0, -1.0, -1.0),
            Vec3::NEG_ONE,
        ]
        .into_iter()
        .filter(|sign| *sign == Vec3::ONE || (self.enabled && (sign.x > 0.0 || self.x) && (sign.y > 0.0 || self.y) && (sign.z > 0.0 || self.z)))
    }

    /// The point the mirror planes and the radial axis pass through, for a mesh centered at `center`
    pub fn origin(&self, center: Vec3) -> Vec3 {
        center + Vec3::from(self.offset)
    }

    /// [`Symmetry::axis`] as an index (0 = X, 1 = Y, 2 = Z), out-of-range values read as Z
    pub fn axis_index(&self) -> usize {
        usize::from(self.axis.min(LAST_AXIS))
    }

    /// Axis of radial symmetry as a unit vector
    pub fn radial_axis(&self) -> Vec3 {
        [Vec3::X, Vec3::Y, Vec3::Z][self.axis_index()]
    }

    /// Linear maps `m` taking a point to its copies as `origin + m * (p - origin)`: identity, mirrors, then their radial turns
    pub fn transforms(&self) -> Vec<Mat3> {
        let mut maps: Vec<Mat3> = self.mirror_signs().map(Mat3::from_diagonal).collect();

        if self.radial_on() {
            let mirrors = maps.clone();
            maps.reserve(mirrors.len().saturating_mul(usize::from(self.radial - 1)));

            for copy in 1..self.radial {
                let turn = Mat3::from_axis_angle(self.radial_axis(), self.radial_angle(copy));
                maps.extend(mirrors.iter().map(|mirror| turn * *mirror));
            }
        }

        maps
    }
}

/// Particle brush: every stamp emits particles that slide down the surface under gravity
/// (world −Y), painting small dabs along their tracks instead of the stamp itself.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Particles {
    pub enabled: bool,
    /// Particles per stamp.
    pub count: u32,
    /// Simulation steps per particle (one dab each).
    pub lifetime: u32,
    /// Distance slid per step, relative to the particle size (0 = particles stay put).
    pub gravity: f32,
}

impl Default for Particles {
    fn default() -> Self {
        Particles { enabled: false, count: 4, lifetime: 24, gravity: 1.0 }
    }
}

/// Brush parameters (the "Brush" properties section plus tool-bar values).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct BrushSettings {
    /// Radius as a fraction of the mesh bounding-box diagonal (the UI shows 0–100).
    pub size: f32,
    /// Pen pressure may shrink the size down to this fraction.
    pub min_size: f32,
    /// Per-stamp alpha 0..1.
    pub flow: f32,
    /// Caps a whole stroke 0..1.
    pub opacity: f32,
    /// Distance between stamps as a fraction of the diameter.
    pub spacing: f32,
    /// Start of the edge falloff (0 soft … 1 hard).
    pub hardness: f32,
    /// Stamp rotation, degrees.
    pub angle: f32,
    pub follow_path: bool,
    pub alignment: Alignment,
    /// Surfaces facing away more than these angles (degrees) receive less paint.
    pub angle_fade: (f32, f32),
    /// Lazy mouse distance in screen pixels (0 = off).
    pub lazy: f32,
    pub jitter_size: f32,
    pub jitter_angle: f32,
    pub jitter_flow: f32,
    /// Position jitter as a fraction of the radius.
    pub jitter_position: f32,
    pub alpha: BrushAlpha,
    /// Image resource used as the stamp shape instead of `alpha` (grayscale, rotated with the stamp).
    #[serde(default)]
    pub alpha_image: Option<String>,
    /// Built-in library alpha id used as the stamp shape (see [`crate::alphas`]).
    #[serde(default)]
    pub alpha_library: Option<String>,
    /// Only paint surfaces facing the camera.
    pub backface_culling: bool,
    /// Half-depth of the stamp volume relative to the radius (thinner = less bleed into folds).
    pub depth: f32,
    pub symmetry: Symmetry,
    /// Size in screen space (constant on screen) instead of world space.
    pub screen_space: bool,
    #[serde(default)]
    pub particles: Particles,
}

impl Default for BrushSettings {
    fn default() -> Self {
        BrushSettings {
            size: 0.02,
            min_size: 0.05,
            flow: 1.0,
            opacity: 1.0,
            spacing: 0.1,
            hardness: 0.5,
            angle: 0.0,
            follow_path: false,
            alignment: Alignment::Tangent,
            angle_fade: (70.0, 85.0),
            lazy: 0.0,
            jitter_size: 0.0,
            jitter_angle: 0.0,
            jitter_flow: 0.0,
            jitter_position: 0.0,
            alpha: BrushAlpha::Round,
            alpha_image: None,
            alpha_library: None,
            backface_culling: true,
            depth: 1.0,
            symmetry: Symmetry::default(),
            screen_space: false,
            particles: Particles::default(),
        }
    }
}

impl BrushSettings {
    /// Clamp every value into its valid range (input from UI, CLI or agents)
    pub fn sanitized(&self) -> BrushSettings {
        let clamp_finite = |value: f32, min: f32, max: f32, default: f32| if value.is_finite() { value.clamp(min, max) } else { default };

        BrushSettings {
            size: clamp_finite(self.size, 0.0005, 0.5, 0.02),
            min_size: clamp_finite(self.min_size, 0.0, 1.0, 0.05),
            flow: clamp_finite(self.flow, 0.0, 1.0, 1.0),
            opacity: clamp_finite(self.opacity, 0.0, 1.0, 1.0),
            spacing: clamp_finite(self.spacing, 0.01, 10.0, 0.1),
            hardness: clamp_finite(self.hardness, 0.0, 0.999, 0.5),
            angle: clamp_finite(self.angle, -360.0, 360.0, 0.0),
            angle_fade: (clamp_finite(self.angle_fade.0, 0.0, 90.0, 70.0), clamp_finite(self.angle_fade.1, 0.0, 90.0, 85.0)),
            lazy: clamp_finite(self.lazy, 0.0, 500.0, 0.0),
            jitter_size: clamp_finite(self.jitter_size, 0.0, 1.0, 0.0),
            jitter_angle: clamp_finite(self.jitter_angle, 0.0, 1.0, 0.0),
            jitter_flow: clamp_finite(self.jitter_flow, 0.0, 1.0, 0.0),
            jitter_position: clamp_finite(self.jitter_position, 0.0, 4.0, 0.0),
            depth: clamp_finite(self.depth, 0.05, 4.0, 1.0),
            symmetry: Symmetry {
                radial: self.symmetry.radial.min(MAX_RADIAL_COPIES),
                axis: self.symmetry.axis.min(LAST_AXIS),
                offset: self.symmetry.offset.map(|value| clamp_finite(value, -MAX_SYMMETRY_OFFSET, MAX_SYMMETRY_OFFSET, 0.0)),
                angle_span: clamp_finite(self.symmetry.angle_span, 0.0, FULL_TURN_DEGREES, FULL_TURN_DEGREES),
                ..self.symmetry
            },
            particles: Particles {
                enabled: self.particles.enabled,
                count: self.particles.count.clamp(1, 16),
                lifetime: self.particles.lifetime.clamp(1, 128),
                gravity: clamp_finite(self.particles.gravity, 0.0, 2.0, 1.0),
            },
            ..self.clone()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alphas_are_bounded_and_zero_outside() {
        for a in BrushAlpha::ALL {
            for i in 0..40 {
                let t = i as f32 / 20.0 - 1.0;
                let v = a.sample(t, t * 0.5, 0.5, 3);
                assert!((0.0..=1.0).contains(&v), "{a:?} {v}");
            }
            assert_eq!(a.sample(1.5, 1.5, 0.5, 1), 0.0, "{a:?}");
            assert_eq!(BrushAlpha::parse(a.label()), Some(a));
        }
        assert!(BrushAlpha::Round.sample(0.0, 0.0, 0.5, 0) > 0.99);
    }

    #[test]
    fn image_alpha_follows_the_image_and_is_zero_outside() {
        // Left half white, right half black (2×1 image, sampled at texel centers).
        let img = ImageRes { width: 2, height: 1, rgba: vec![[1.0; 4], [0.0, 0.0, 0.0, 1.0]], srgb: true };
        let shape = Shape { alpha: BrushAlpha::Round, image: Some(&img), library: None, level: 0 };
        assert!(shape.sample(-0.5, 0.0, 0.5, 0) > 0.99);
        assert!(shape.sample(0.5, 0.0, 0.5, 0) < 0.01);
        assert_eq!(shape.sample(1.5, 0.0, 0.5, 0), 0.0);
        assert_eq!(shape.sample(f32::NAN, 0.0, 0.5, 0), 0.0);
    }

    #[test]
    fn sanitize_fixes_hostile_values() {
        let b = BrushSettings { size: f32::NAN, flow: 9.0, hardness: 1.0, spacing: -1.0, ..BrushSettings::default() }.sanitized();
        assert_eq!(b.size, 0.02);
        assert_eq!(b.flow, 1.0);
        assert!(b.hardness < 1.0);
        assert!(b.spacing > 0.0);
    }
}
