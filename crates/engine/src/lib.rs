//! # texturecraft-engine
//!
//! The TextureCraft session: a [`Session`] owns the document (project + copy-on-write pixels +
//! baked maps), the mesh with its BVH and per-texture-set caches (UV G-buffer, padding map,
//! composite), snapshot-based undo, the tool state and the active stroke. Every user-visible
//! action is a command in the [`commands`] registry, dispatched by id from the UI, the CLI,
//! the control channel and the MCP server alike.
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bake;
pub mod bake_preview;
pub mod brush_preset_cmds;
pub mod camera_cmds;
pub mod color_selection_cmds;
pub mod commands;
pub mod decal_cmds;
pub mod effect_clipboard;
pub mod file_cmds;
pub mod fill_hit_test;
pub mod fill_mode;
pub mod filter_lib;
pub mod inspect;
pub mod instance_cmds;
pub mod layer_cmds;
pub mod layer_ops;
pub mod link_cmds;
pub mod mask_cmds;
mod mask_preview_pixel;
pub mod material_lib;
mod material_preview_pixel;
pub mod materials;
pub mod materialx_cmds;
pub mod meshmap_cmds;
mod meshmap_file_name;
pub mod meshmap_import_cmds;
mod orb_camera;
pub mod orb_preview;
pub mod painting;
pub mod path_tool;
mod polygon_fill;
mod polygon_fill_drag;
// Mesh reimport reads files: native only.
#[cfg(not(target_arch = "wasm32"))]
mod preserve;
pub mod preset_cmds;
pub mod preview;
pub mod project_cmds;
pub mod quick_mask;
pub mod resource_cmds;
pub mod session;
pub mod set_cmds;
pub mod smart_masks;
pub mod template_cmds;
pub mod text_cmds;
pub mod thumbs;
pub mod tool_cmds;
mod triangle_partition;
mod uv_stroke_state;
pub mod uv_tile_cmds;
pub mod view_cmds;

pub use bake::{BakeMesh, BakeMeshes, BakeTask, StreamedBake, StreamedMap};
pub use bake_preview::{BakePreview, PreviewShow};
pub use commands::{CommandSpec, command_specs, execute, find};
pub use fill_hit_test::FillHitTest;
pub use fill_mode::FillMode;
pub use path_tool::PathState;
pub use project_cmds::sample_session;
pub use session::{CloneFrom, CloneSource, Document, Session, SetCache, StackTarget, StencilState, Tool, ToolState, ViewMode};
pub use set_cmds::{ResizeOutput, ResizeTask};

pub use texturecraft_compose as compose;
pub use texturecraft_core as core;
pub use texturecraft_formats as formats;
pub use texturecraft_geom as geom;
pub use texturecraft_paint as paint;
pub use texturecraft_texel as texel;

#[derive(Debug, thiserror::Error)]
pub enum EngineError {
    #[error("unknown command `{0}`")]
    UnknownCommand(String),
    #[error("`{cmd}` is not available: {why}")]
    Disabled { cmd: String, why: String },
    #[error("`{cmd}`: {msg}")]
    BadParams { cmd: String, msg: String },
    #[error(transparent)]
    Core(#[from] texturecraft_core::CoreError),
    #[error("{0}")]
    Other(String),
}

pub type Result<T> = std::result::Result<T, EngineError>;

/// Version string shown in About, `--version` and window titles.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
