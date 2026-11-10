//! State behind the Layers panel: the thumbnail service, the GPU textures made from its pictures,
//! which effect groups are folded, and the one place the preview size is decided.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use egui::{Context, TextureHandle, TextureId, TextureOptions};
use texturecraft_engine::Session;
use texturecraft_engine::compose::ThumbTarget;
use texturecraft_engine::thumbs::{CACHE_ENTRIES, ThumbService};

use crate::TextureCraftApp;

/// Edge (points) of a layer's content and mask thumbnails. The Layers "Preview size" setting
/// plugs in by changing [`layer_preview_size`]; every row measure below derives from it.
pub(crate) const LAYER_PREVIEW_SIZE: f32 = 36.0;

/// The thumbnail edge in points, read once per row.
pub(crate) fn layer_preview_size(_app: &TextureCraftApp) -> f32 {
    LAYER_PREVIEW_SIZE
}

/// Height of a layer row for thumbnails of `size`.
pub(crate) fn layer_row_height(size: f32) -> f32 {
    (size + 12.0).max(48.0)
}

/// Edge of an effect's stage thumbnail for layer thumbnails of `size` (18 to 28 points).
pub(crate) fn effect_preview_size(size: f32) -> f32 {
    (size * 0.6).clamp(18.0, 28.0)
}

/// Height of an effect row.
pub(crate) fn effect_row_height(size: f32) -> f32 {
    (effect_preview_size(size) + 4.0).max(26.0)
}

#[derive(Default)]
pub struct LayerPanel {
    service: ThumbService,
    textures: HashMap<ThumbTarget, (u64, TextureHandle, u64)>,
    frame: u64,
    /// Folded effect groups: (layer, true for the mask's). A view setting; the document is not touched.
    pub(crate) folded: HashSet<(u64, bool)>,
}

impl LayerPanel {
    /// Start drawing the panel for this frame.
    pub(crate) fn begin(&mut self, session: &Session, now: f64) {
        self.frame += 1;
        self.service.begin_frame(session, now);

        // Deleted layer IDs must not accumulate in panel state
        self.folded.retain(|(layer, _)| session.doc.project.layers.contains_key(&texturecraft_engine::core::LayerId(*layer)));
    }

    /// The texture of `target`, queuing its render (call only for rows that are on screen).
    pub(crate) fn texture(&mut self, ctx: &Context, session: &Session, target: ThumbTarget) -> Option<TextureId> {
        let view = self.service.get(session, target);
        let image = view.image?;
        let frame = self.frame;
        if let Some((version, tex, used)) = self.textures.get_mut(&target)
            && *version == view.version
        {
            *used = frame;
            return Some(tex.id());
        }
        let pixels = egui::ColorImage::from_rgba_unmultiplied([image.width as usize, image.height as usize], &image.rgba);
        let tex = ctx.load_texture("layer-thumb", pixels, TextureOptions::LINEAR);
        let id = tex.id();
        self.textures.insert(target, (view.version, tex, frame));
        Some(id)
    }

    /// Finish the frame: collect pictures and ask for another frame while work is outstanding.
    pub(crate) fn end(&mut self, ctx: &Context, session: &Session) {
        if self.service.poll(session) {
            ctx.request_repaint_after(Duration::from_millis(40));
        }
        if self.textures.len() > CACHE_ENTRIES * 2 {
            let frame = self.frame;
            self.textures.retain(|_, (_, _, used)| *used + 1 >= frame);
        }
    }

    /// The service, for tests.
    pub fn service(&self) -> &ThumbService {
        &self.service
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_engine::core::{Channel, LayerId};
    use texturecraft_engine::geom::primitives;

    #[test]
    fn deleted_layers_release_their_fold_state() {
        let mut session = Session::new_project("thumb", primitives::plane(1.0, 1), "builtin:plane", None, 128).unwrap();
        let layer = session.selected_layer().unwrap();
        let mut panel = LayerPanel::default();
        panel.folded.extend([(layer.0, false), (layer.0, true)]);
        panel.begin(&session, 0.0);
        assert_eq!(panel.folded.len(), 2, "existing layers keep their view settings");

        texturecraft_engine::execute(&mut session, "layer.remove", &serde_json::json!({"layer": layer.0})).unwrap();
        panel.begin(&session, 1.0);

        assert!(panel.folded.is_empty(), "deleted layer IDs must not accumulate in view state");
    }

    #[test]
    fn evicted_and_dropped_thumbnail_handles_release_textures() {
        let session = Session::new_project("thumb", primitives::plane(1.0, 1), "builtin:plane", None, 128).unwrap();
        let context = Context::default();
        let initial_textures = context.tex_manager().read().num_allocated();
        let mut panel = LayerPanel::default();
        let mut evicted = Vec::new();
        panel.begin(&session, 0.0);

        // Simulate scrolling through more textures than the UI cache retains
        for index in 0..CACHE_ENTRIES * 2 + 1 {
            let target = ThumbTarget::Content { layer: LayerId(index as u64), channel: Channel::BaseColor };
            let image = egui::ColorImage::filled([64, 64], egui::Color32::WHITE);
            let texture = context.load_texture("layer-thumb", image, TextureOptions::LINEAR);
            evicted.push(texture.id());
            panel.textures.insert(target, (1, texture, panel.frame));
        }
        let kept = evicted.pop().unwrap();
        panel.begin(&session, 1.0);
        panel.begin(&session, 2.0);
        panel.textures.values_mut().find(|(_, texture, _)| texture.id() == kept).unwrap().2 = panel.frame;
        panel.end(&context, &session);

        assert_eq!(context.tex_manager().read().num_allocated(), initial_textures + 1);
        assert!(evicted.iter().all(|id| context.tex_manager().read().meta(*id).is_none()));
        assert!(context.tex_manager().read().meta(kept).is_some());

        drop(panel);

        assert_eq!(context.tex_manager().read().num_allocated(), initial_textures);
    }
}
