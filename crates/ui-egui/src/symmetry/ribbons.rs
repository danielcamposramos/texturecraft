//! Lines drawn into the 3D view as thin quads

use egui::Rect;
use texturecraft_engine::geom::glam::{Mat4, Vec3};
use texturecraft_render::OverlayQuad;

use crate::TextureCraftApp;

/// Height of the view in normalized device coordinates (−1 to 1)
const NDC_HEIGHT: f32 = 2.0;

/// Lines drawn as thin quads turned to face the camera, a steady width on screen
pub(crate) struct Ribbons {
    eye: Vec3,
    view_proj: Mat4,
    /// World units per screen point at clip-space w = 1
    world_per_point: f32,
}

impl Ribbons {
    pub(crate) fn new(app: &TextureCraftApp, rect: Rect) -> Self {
        let camera = &app.session.camera;
        let view_proj = camera.view_proj(rect.width() / rect.height().max(1.0));
        let ndc_per_unit = view_proj.row(1).truncate().length();
        Ribbons { eye: camera.eye(), view_proj, world_per_point: NDC_HEIGHT / (rect.height() * ndc_per_unit).max(f32::EPSILON) }
    }

    /// The line from `from` to `to`, `width` points wide
    pub(crate) fn quad(&self, from: Vec3, to: Vec3, width: f32, color: [f32; 3], opacity: f32) -> OverlayQuad {
        let middle = (from + to) * 0.5;
        let clip_w = (self.view_proj * middle.extend(1.0)).w.max(0.0);
        let side = (to - from).cross(self.eye - middle).normalize_or_zero() * (clip_w * self.world_per_point * width * 0.5);
        OverlayQuad { corners: [from - side, to - side, to + side, from + side], color, opacity }
    }
}
