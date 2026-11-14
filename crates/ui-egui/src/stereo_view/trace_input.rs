//! One retained source snapshot; pose and sampling choices remain view inputs.
use crate::TextureCraftApp;
use std::sync::Arc;
use texturecraft_engine::compose::ImageRes;
use texturecraft_render::{stereo::Budget, trace::TraceScene, trace_pair::TracePairFrame};

#[derive(Default)]
pub(super) struct TraceInput {
    base: String,
    image: Option<Arc<ImageRes>>,
    scene: Option<Arc<TraceScene>>,
    revision: u64,
}

impl TraceInput {
    pub fn frame(&mut self, app: &TextureCraftApp, size: [u32; 2], budget: Budget) -> TracePairFrame {
        let base = crate::render_mode::scene_key(&app.session, true);
        let image = app.session.doc.project.display.environment_resource().and_then(|name| app.session.images.get(name)).cloned();
        let same_image = match (&self.image, &image) {
            (Some(a), Some(b)) => Arc::ptr_eq(a, b),
            (None, None) => true,
            _ => false,
        };
        let scene = match &self.scene {
            Some(scene) if self.base == base && same_image => scene.clone(),
            _ => {
                let scene = Arc::new(crate::render_mode::trace_scene(app));
                self.scene = Some(scene.clone());
                self.base = base;
                // Holding the source prevents its address from being reused by a later image.
                self.image = image;
                self.revision = self.revision.wrapping_add(1);
                scene
            }
        };
        TracePairFrame {
            scene,
            camera: app.session.camera,
            size,
            budget,
            settings: crate::render_mode::settings(app),
            display: crate::render_mode::display(app),
            max_samples: app.ui.render.max_samples.max(1),
            epoch: format!("{}|source:{}", self.base, self.revision),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_engine::{Session, geom::primitives};
    fn app() -> TextureCraftApp {
        let session = Session::new_project("paired world", primitives::sphere(1.0, 16, 8), "builtin:sphere", None, 128).unwrap();
        egui_kittest::Harness::builder().build_eframe(|ctx| TextureCraftApp::new(ctx, session)).into_state()
    }

    #[test]
    fn replacing_an_environment_under_the_same_name_replaces_both_eye_inputs() {
        let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut app = app();
        app.session.doc.project.display.environment = "resource:sky".into();
        let image = Arc::new(ImageRes { width: 2, height: 1, rgba: vec![[1.0, 0.0, 0.0, 1.0]; 2], srgb: false });
        let old_source = Arc::downgrade(&image);
        app.session.images.insert("sky".into(), image);
        let mut input = TraceInput::default();
        let budget = Budget { sunk: 24.0, popped: 12.0 };
        let first = input.frame(&app, [640, 360], budget);
        let again = input.frame(&app, [640, 360], budget);
        assert!(Arc::ptr_eq(&first.scene, &again.scene));
        assert_eq!(first.key(), again.key());
        app.session.images.insert("sky".into(), Arc::new(ImageRes { width: 2, height: 1, rgba: vec![[0.0, 0.0, 1.0, 1.0]; 2], srgb: false }));
        assert!(old_source.upgrade().is_some(), "the previous source stays retained until replacement");
        let replaced = input.frame(&app, [640, 360], budget);
        assert!(!Arc::ptr_eq(&first.scene, &replaced.scene), "one name must not hide a new environment's source identity");
        assert_ne!(first.key(), replaced.key());
        assert_ne!(first.scene.env.data, replaced.scene.env.data, "both new eyes must actually see the replacement environment");
        assert!(old_source.upgrade().is_none(), "the retired source image is not retained forever");
    }

    #[test]
    fn camera_and_depth_changes_keep_one_world_but_invalidate_both_view_histories() {
        let _language = crate::i18n::TEST_LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut app = app();
        let mut input = TraceInput::default();
        let first = input.frame(&app, [640, 360], Budget { sunk: 24.0, popped: 12.0 });
        app.session.camera.target.x += 0.2;
        let moved = input.frame(&app, [640, 360], Budget { sunk: 24.0, popped: 12.0 });
        assert!(Arc::ptr_eq(&first.scene, &moved.scene));
        assert_ne!(first.key(), moved.key());
        let resized = input.frame(&app, [320, 180], Budget { sunk: 48.0, popped: 24.0 });
        assert!(Arc::ptr_eq(&moved.scene, &resized.scene));
        assert_ne!(moved.key(), resized.key());
    }
}
