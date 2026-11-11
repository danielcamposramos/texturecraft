//! Sampled path curves retained only while the viewport uses them.

use std::collections::HashMap;
use std::sync::Arc;

use egui::cache::CacheTrait;
use texturecraft_engine::Session;
use texturecraft_engine::core::{LayerId, PathShape};
use texturecraft_engine::geom::glam::Vec3;

pub(crate) type SurfaceCurve = Arc<Vec<Vec<Option<Vec3>>>>;

#[derive(Default)]
pub(crate) struct PathSurfaceCache {
    // Mesh, effect and path index identify entries; the flag records use during this pass
    curves: HashMap<(u64, LayerId, usize), (PathShape, SurfaceCurve, bool)>,
}

impl PathSurfaceCache {
    /// Reuse an unchanged curve and retain it for this pass; UV or missing paths have no surface curve
    pub(crate) fn get(&mut self, session: &Session, effect: LayerId, index: usize) -> Option<SurfaceCurve> {
        let shape = session.paths(effect)?.get(index)?;
        let key = (session.mesh_id, effect, index);
        if let Some((cached_shape, surface, used)) = self.curves.get_mut(&key)
            && cached_shape == shape
        {
            *used = true;
            return Some(Arc::clone(surface));
        }

        let surface = Arc::new(session.stored_path_surface(effect, index)?);
        self.curves.insert(key, (shape.clone(), Arc::clone(&surface), true));
        Some(surface)
    }
}

impl CacheTrait for PathSurfaceCache {
    fn update(&mut self) {
        // egui calls this even when paths stop drawing, so hidden or deleted paths release their buffers
        self.curves.retain(|_, (_, _, used)| std::mem::take(used));
    }

    fn len(&self) -> usize {
        self.curves.len()
    }
}
