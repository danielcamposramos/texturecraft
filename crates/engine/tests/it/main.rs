//! One integration-test executable avoids linking the same dependencies for every feature.
//! Keep feature tests in their own modules; the standalone session.rs target remains available.

mod bake;
mod bake_resolution;
mod bake_settings;
mod bake_tiles;
mod color_selection;
mod content_effects;
mod export_colour_space;
mod filter_swap;
mod filters;
mod high_poly_missed;
mod instances;
mod materialcraft_link;
mod materialx;
mod materialx_import;
mod meshmap_import;
mod painter_gaps;
mod path_tool;
mod perf;
mod polygon_fill;
mod project_recovery;
mod sample;
mod thumbs;
mod tools;
mod udim;
mod uv_strokes;
