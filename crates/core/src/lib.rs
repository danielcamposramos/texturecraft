//! # texturecraft-core
//!
//! The TextureCraft document model: a [`Project`] holds texture sets (one per mesh material),
//! each with a stack of [`Layer`]s (paint, fill, group) whose channels blend with one of
//! the [`BlendMode`]s. Pure data with serde; pixels live in `texturecraft-texel` images
//! owned by the engine. [`blend`] is the CPU reference for every blend mode.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bake;
pub mod blend;
mod block_scales;
pub mod channel;
pub mod layer;
pub mod project;

pub use bake::{
    AmbientOcclusionAttenuation, AmbientOcclusionSettings, BackfaceMode, BakeAntialiasing, BakePaintTarget, BakeResolution, BakeSettings, BakedMap,
    BentNormalSettings, CageSettings, CageType, CommonBakeSettings, CurvatureMethod, CurvatureSettings, GroundPlaneSettings, HeightNormalization,
    HeightSettings, HighPolySettings, IdColorGenerator, IdColorSource, IdSettings, MapSettings, MeshCurvatureSettings, MeshMapSettings, MeshMatchMode,
    NormalMapCurvatureSettings, PositionAxis, PositionMode, PositionNormalizationScale, PositionSettings, RayDistribution, RaySettings, SelfOcclusion,
    SkewCorrectionMethod, SkewSettings, ThicknessNormalization, ThicknessSettings,
};
pub use blend::{BlendMode, blend1, blend3, lum, mix3};
pub use channel::{Channel, ChannelFormat, ChannelKind};
pub use layer::{
    ChannelBlend, CompareOp, CurvatureDetail, CurvatureMode, Decal, Effect, EffectKind, Fill, Filter, Generator, GeneratorKind, HeightMask, Layer, LayerId,
    LayerKind, Levels, LinkParam, LinkPreset, Mask, MaskBackground, MaterialLink, MicroDetail, PathShape, Procedural, ProceduralKind, Projection, Source,
    gradient, hsl_adjust, per_unit,
};
pub use project::{
    AoMixing, BlockScales, Camera, DepthOfField, DisplaySettings, FORMAT_VERSION, MAX_RESOLUTION, MIN_RESOLUTION, NormalFormat, NormalMixing, Project,
    TextureSet, TileGrid, ToneMapping, UDIM_RANGE, udim_at,
};

#[derive(Debug, thiserror::Error, Clone, PartialEq)]
pub enum CoreError {
    #[error("no layer with id {}", .0.0)]
    NoLayer(LayerId),
    #[error("no texture set {0}")]
    NoTextureSet(usize),
    #[error("{0}")]
    Invalid(String),
}

pub type Result<T> = std::result::Result<T, CoreError>;

/// Product name for user-facing text.
pub const APP_NAME: &str = "TextureCraft";
/// Native project file extension.
pub const PROJECT_EXT: &str = "tcraft";
