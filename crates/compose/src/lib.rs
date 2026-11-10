//! # texturecraft-compose
//!
//! The CPU layer-stack evaluator: [`eval::composite_tiles`] turns a texture set's layers, masks,
//! fills and painted pixels into one image per channel, tile by tile in parallel, including the
//! live preview of a stroke being drawn. [`display`] converts composited tiles into the texture
//! formats the viewport samples; [`pattern`] evaluates procedurals through projections.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

mod baked_maps;
pub mod below;
mod curvature_size_level;
pub mod curvature_sizes;
pub mod display;
pub mod eval;
pub mod pattern;
pub mod store;
pub mod thumb;

pub use below::{BelowCache, StackKey};
pub use display::{DisplayKind, tile_bytes};
pub use eval::{
    Composite, Inputs, PaintTarget, Stencil, StrokePreview, all_tiles, composite_tiles, composite_tiles_cached, covered_tiles, effect_histogram, flatten_layer,
    mask_at, mask_image, pad_tiles,
};
pub use store::{BakedMaps, ImageRes, ImageStore, PixelStore};
pub use thumb::{THUMB_TEXELS, ThumbImage, ThumbTarget};
