//! # texturecraft-render
//!
//! The wgpu viewport renderer: metal/roughness or specular/glossiness PBR with image-based
//! lighting (procedural environments or imported HDR images, prefiltered on the CPU in [`ibl`]),
//! height-to-normal, optional parallax occlusion, 4× MSAA plus idle temporal accumulation, channel / mesh-map / UV
//! checker debug views, the 2D UV view, a linear HDR scene with a post pass (exposure, Linear /
//! ACES tone mapping, bloom, sharpen, vignette, grain), key-light shadow maps, a 3D wireframe
//! overlay, and a [`headless`] device for renders
//! without a window. No UI toolkit dependency; the egui shell drives it through
//! [`Renderer::prepare`] and [`Renderer::blit`]. Fragment passes only, so it runs on WebGPU and
//! WebGL2 too. Rendering mode's path tracer runs on the CPU ([`trace`]) or, where compute shaders
//! exist (not WebGL2), on the GPU ([`gpu_trace`]).
#![forbid(unsafe_code)]
#![deny(clippy::unwrap_used, clippy::expect_used, clippy::panic, clippy::unimplemented, clippy::todo, clippy::unreachable)]

pub mod bake_visualization;
mod cage_gpu;
pub mod displace;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu_bake;
pub mod gpu_trace;
#[cfg(not(target_arch = "wasm32"))]
pub mod headless;
mod high_poly_gpu;
pub mod ibl;
pub mod mesh_maps;
pub mod overlay_quad;
pub mod readback;
pub mod renderer;
pub mod set_highlight;
mod subdivide;
pub mod trace;
mod uv_seam_gpu;

pub use bake_visualization::BakeVisualization;
pub use overlay_quad::OverlayQuad;
pub use renderer::{COLOR_FORMAT, Frame, MSAA, Renderer, Shade, ViewKind, aux_size, uv_view_scale};
pub use set_highlight::SetHighlight;
pub use wgpu;
