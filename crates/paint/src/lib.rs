//! # texturecraft-paint
//!
//! The brush engine. A [`Stroke`] turns pointer samples into world-space [`Stamp`]s (picking,
//! lazy mouse, spacing along the surface, jitter, symmetry); [`raster::rasterize`] accumulates
//! them into a texture-space stroke alpha over the UV G-buffer, hiding surfaces the camera can't
//! see ([`EyeDepth`]); [`commit`] blends the finished stroke into layer pixels. Clone and smudge
//! strokes carry channel values ([`pickup`]) instead of painting the tool's material.
//! [`path`] samples the Path tool's curves and lasso coverage; [`particles`] runs the particle
//! brush; [`alphas`] is the built-in library of procedural brush alphas and [`presets`] the built-in
//! brush presets; [`text`] draws text for text alphas and decals.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod alphas;
pub mod brush;
pub mod commit;
pub mod particles;
pub mod path;
pub mod pickup;
pub mod presets;
pub mod raster;
pub mod stamp_source;
pub mod stroke;
pub mod text;
pub mod visibility;

pub use brush::{Alignment, BrushAlpha, BrushSettings, FULL_TURN_DEGREES, MAX_RADIAL_COPIES, Particles, Shape, Symmetry, image_alpha};
pub use commit::{CommitSource, Staging, TexelFn, commit_coverage, commit_layer, commit_mask};
pub use path::{Lasso, spline, spline_segments};
pub use pickup::{Pickup, PickupMode};
pub use raster::{RasterCtx, StrokeCanvas, UvStamp, fill_coverage, fill_triangles, rasterize, rasterize_uv};
pub use stamp_source::StampSource;
pub use stroke::{PickCtx, Stamp, Stroke, SurfaceHit, pick};
pub use visibility::{EyeDepth, screen_point};
