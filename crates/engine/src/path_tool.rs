//! The Path tool: control points placed on the surface (3D view) or in texture space (2D view)
//! and a smooth curve through them, painted with the brush, or, when closed and filled, as an
//! antialiased lasso fill of the visible surface (3D) or of the UV region (2D).
//!
//! Paths are non-destructive: a committed path is stored in a path sublayer
//! ([`EffectKind::Path`]) that is repainted from its paths whenever one is added, edited
//! (`path.edit`, live while the points, brush or material change) or removed (`path.remove`).

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use texturecraft_compose::{PaintTarget, PixelStore};
use texturecraft_core::{Channel, Effect, EffectKind, LayerId, Mask, MaskBackground, PathShape};
use texturecraft_geom::glam::{Mat3, Vec2, Vec3};
use texturecraft_geom::{OrbitCamera, surface_at};
use texturecraft_paint::{BrushSettings, Lasso, PickCtx, fill_coverage, pick, spline, spline_segments};
use texturecraft_texel::TileImage;

use crate::commands::{CommandSpec, always, bad, boolp, has_layer, u64p};
use crate::path_draft::PathDraft;
use crate::session::{StackTarget, Tool};
use crate::tool_cmds::{points_param, set_tool, viewport_of};
use crate::{Document, EngineError, Result, Session, cmd};

/// Control points kept at most.
pub const MAX_PATH_POINTS: usize = 4096;
/// The undo step that adds a new path
pub(crate) const NEW_PATH_LABEL: &str = "Path";
/// Curve sampling step in viewport pixels (3D view) and texture units (2D view).
const STEP_PX: f32 = 2.0;
const STEP_UV: f32 = 0.001;
/// A stored path's viewport is clamped to this range (pixels per side) before painting from it
const MIN_PAINT_VIEWPORT_PX: f32 = 16.0;
const MAX_PAINT_VIEWPORT_PX: f32 = 16384.0;
/// Sampling step of a 3D path's curve on the mesh for the views, in its paint viewport pixels
const SURFACE_STEP_PX: f32 = 4.0;
const MIN_PROJECTIVE_W: f32 = 1e-6;
// Match the minimum world-space scale used by brush stamps
const MIN_BRUSH_SCALE: f32 = 1e-4;
// Allow ray hits to differ slightly from transformed texel positions
const RAY_HIT_RELATIVE_TOLERANCE: f32 = 0.001;

/// The curve being edited with the Path tool
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct PathState {
    /// Control points: surface positions (3D view) or texture coordinates with z = 0 (2D view)
    pub points: Vec<Vec3>,
    /// The points are texture coordinates (placed in the 2D view)
    pub uv: bool,
    /// The curve loops back to its first point
    pub closed: bool,
    /// A closed path fills its interior instead of stroking the curve
    pub fill: bool,
    /// The stored path being edited: (path sublayer, index). Changes repaint it live
    #[serde(default)]
    pub editing: Option<(LayerId, usize)>,
    /// This edit has already repainted: later changes join its undo step, not an earlier edit's
    #[serde(default)]
    pub edited: bool,
    /// The selected control point (drawn in the accent color; Backspace or Remove point removes it)
    #[serde(default)]
    pub selected: Option<usize>,
}

impl PathState {
    /// Append a control point; a point placed in the other view starts a new path.
    pub fn push(&mut self, point: Vec3, uv: bool) {
        if uv != self.uv {
            self.points.clear();
            self.uv = uv;
        }
        if self.points.len() < MAX_PATH_POINTS {
            self.points.push(point);
        }
    }

    /// Insert a control point before point `at` (on the segment ending there) and select it; None when it can't
    pub fn insert(&mut self, at: usize, point: Vec3) -> Option<usize> {
        if at > self.points.len() || self.points.len() >= MAX_PATH_POINTS {
            return None;
        }

        self.points.insert(at, point);
        self.selected = Some(at);
        self.selected
    }

    /// The same curve (points, open or closed, filled, view) as `other`
    fn same_curve(&self, other: &PathState) -> bool {
        (&self.points, self.closed, self.fill, self.uv) == (&other.points, other.closed, other.fill, other.uv)
    }

    /// Empty the tool for the next path: no points, open, nothing selected or edited
    pub fn clear(&mut self) {
        self.points.clear();
        self.closed = false;
        self.selected = None;
        self.editing = None;
        self.edited = false;
    }

    /// Remove the selected control point, or the last one when none is selected
    pub fn remove_selected(&mut self) {
        let at = self.selected.take().filter(|i| *i < self.points.len()).or(self.points.len().checked_sub(1));
        if let Some(i) = at {
            self.points.remove(i);
        }
        if self.closed && self.points.len() < 3 {
            self.closed = false;
        }
    }
}

/// A path sublayer's pixels after painting its first paths (`below`), with the mesh and
/// texture size they were painted for.
#[derive(Clone)]
// shortcut: the key doesn't cover image resources a path's brush alpha reads; replacing one
// leaves cached paths with the old alpha until a path below the edited one changes.
pub(crate) struct PathCache {
    eid: LayerId,
    key: (u64, Option<(u32, u32)>),
    below: Vec<PathShape>,
    mask: Option<texturecraft_texel::TileImage>,
    layers: Vec<((LayerId, Channel), texturecraft_texel::TileImage)>,
}

/// What a stored path is painted with ([`PathShape::paint`]).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
struct PathPaint {
    brush: BrushSettings,
    values: BTreeMap<Channel, [f32; 3]>,
    enabled: BTreeSet<Channel>,
    mask_value: f32,
    camera: OrbitCamera,
}

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "tool.path",
            "Path",
            ["Tools"],
            Some("7"),
            "{points?: [[x,y],...] viewport pixels on the mesh | uvPoints?: [[u,v],...], viewport?: [w,h], closed?, fill?, commit?: bool, clear?: bool} — selects the Path tool and edits its curve; commit adds the path to the selected path sublayer (or a new one), painted with the current brush or filled when closed and filled. While a stored path is edited (path.edit), changes repaint it at once",
            always,
            path
        ),
        cmd!(
            "path.edit",
            "Edit path",
            ["Tools"],
            None,
            "{effect: path sublayer id, index: path index} — loads a stored path into the Path tool (with its brush and material); changes to it repaint the sublayer live until commit or clear",
            has_layer,
            edit_path
        ),
        cmd!(
            "path.remove",
            "Remove path",
            ["Tools"],
            None,
            "{effect: path sublayer id, index: path index} — removes a stored path and repaints its sublayer",
            has_layer,
            remove_path
        ),
    ]
}

/// Set the curve from command parameters, then preview or commit it in the supplied viewport
fn path(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "tool.path";
    if s.tool.tool != Tool::Path {
        set_tool(s, Tool::Path)?;
    }

    let viewport = viewport_of(s, p);
    if boolp(p, "clear") == Some(true) {
        s.cancel_path();
    }

    let placed = match (points_param(CMD, p, "points")?, points_param(CMD, p, "uvPoints")?) {
        (Some(pixels), _) => Some((
            pixels
                .iter()
                .enumerate()
                .map(|(i, pixel)| s.pick(*pixel, viewport).map(|hit| hit.pos).ok_or_else(|| bad(CMD, format!("point {i} misses the mesh"))))
                .collect::<Result<Vec<_>>>()?,
            false,
        )),
        (None, Some(uvs)) => Some((uvs.iter().map(|uv| uv.clamp(Vec2::ZERO, Vec2::ONE).extend(0.0)).collect(), true)),
        (None, None) => None,
    };
    if let Some((points, uv)) = placed {
        if points.len() > MAX_PATH_POINTS {
            return Err(bad(CMD, format!("at most {MAX_PATH_POINTS} control points")));
        }

        s.tool.path.points = points;
        s.tool.path.uv = uv;
    }
    if let Some(v) = boolp(p, "closed") {
        s.tool.path.closed = v;
    }
    if let Some(v) = boolp(p, "fill") {
        s.tool.path.fill = v;
    }
    s.revision = s.revision.wrapping_add(1);

    let committed = if boolp(p, "commit") == Some(true) {
        let report = s.commit_path(viewport)?;
        s.update();
        report
    } else {
        if s.path_live_in_view(viewport)? {
            s.update();
        }
        Value::Null
    };

    let path = &s.tool.path;
    let editing = path.editing.map(|(e, i)| json!({ "effect": e.0, "index": i }));
    Ok(json!({ "points": path.points.len(), "uv": path.uv, "closed": path.closed, "fill": path.fill, "editing": editing, "committed": committed }))
}

/// The `effect` and `index` params naming a stored path, with the path.
fn path_ref(s: &Session, cmd: &str, p: &Value) -> Result<(LayerId, usize, PathShape)> {
    let eid = LayerId(u64p(p, "effect").ok_or_else(|| bad(cmd, "`effect` (a path sublayer id) is required"))?);
    let index = u64p(p, "index").ok_or_else(|| bad(cmd, "`index` is required"))?;
    let shapes = s.paths(eid).ok_or_else(|| bad(cmd, format!("{} is not a path sublayer", eid.0)))?;
    let i =
        usize::try_from(index).ok().filter(|i| *i < shapes.len()).ok_or_else(|| bad(cmd, format!("no path {index} (the sublayer has {})", shapes.len())))?;
    let shape = shapes.get(i).cloned().ok_or_else(|| bad(cmd, format!("no path {index}")))?;
    Ok((eid, i, shape))
}

/// Finish any draft and load a stored path's geometry and paint settings for live editing
fn edit_path(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.edit";
    let (eid, index, shape) = path_ref(s, CMD, p)?;
    let paint = read_paint(&shape)?;

    s.finish_path()?;
    set_tool(s, Tool::Path)?;
    s.tool.brush = paint.brush;
    s.tool.values = paint.values;
    s.tool.enabled = paint.enabled;
    s.tool.mask_value = paint.mask_value;
    s.tool.path = PathState { editing: Some((eid, index)), ..state_of(&shape) };
    s.path_steps = Default::default();
    s.target = StackTarget::Effect(eid);
    Ok(json!({ "effect": eid.0, "index": index, "points": s.tool.path.points.len() }))
}

/// Remove one stored curve and repaint its sublayer as an atomic document edit
fn remove_path(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "path.remove";
    let (eid, index, _) = path_ref(s, CMD, p)?;
    s.edit("Remove path", |s| {
        if let Some(Effect { kind: EffectKind::Path(shapes), .. }) = s.doc.project.effect_mut(eid)
            && index < shapes.len()
        {
            shapes.remove(index);
        }
        s.render_paths(eid, index)
    })?;
    if s.tool.path.editing.is_some_and(|(e, _)| e == eid) {
        s.stop_path_edit();
    }
    s.update();
    Ok(json!({ "effect": eid.0, "paths": s.paths(eid).map_or(0, <[_]>::len) }))
}

/// A stored path's settings, with the brush clamped like the tool's own (they come from the file).
fn read_paint(shape: &PathShape) -> Result<PathPaint> {
    let mut paint: PathPaint = serde_json::from_value(shape.paint.clone()).map_err(|e| EngineError::Other(format!("unreadable path settings: {e}")))?;
    paint.brush = paint.brush.sanitized();
    Ok(paint)
}

/// The Path tool state showing a stored path (finite points only, at most [`MAX_PATH_POINTS`])
fn state_of(shape: &PathShape) -> PathState {
    let points = shape.points.iter().filter(|p| p.iter().all(|v| v.is_finite())).take(MAX_PATH_POINTS).map(|p| Vec3::from_array(*p)).collect();
    PathState { points, uv: shape.uv, closed: shape.closed, fill: shape.fill, ..PathState::default() }
}

/// Tiles (x, y) where sublayer `eid`'s pixels differ between `before` and `after` (copy-on-write: an unchanged tile is the same allocation)
fn changed_tiles(before: &PixelStore, after: &PixelStore, eid: LayerId) -> BTreeSet<(u32, u32)> {
    let (before_images, after_images) = (sublayer_images(before, eid), sublayer_images(after, eid));
    let channels: BTreeSet<Option<Channel>> = before_images.keys().chain(after_images.keys()).copied().collect();

    let mut tiles = BTreeSet::new();
    for channel in channels {
        let (before_image, after_image) = (before_images.get(&channel), after_images.get(&channel));
        let Some(image) = before_image.or(after_image) else { continue };
        for ty in 0..image.tiles_y() {
            for tx in 0..image.tiles_x() {
                let same = match (before_image.and_then(|i| i.tile_handle(tx, ty)), after_image.and_then(|i| i.tile_handle(tx, ty))) {
                    (None, None) => true,
                    (Some(before_tile), Some(after_tile)) => Arc::ptr_eq(before_tile, after_tile),
                    _ => false,
                };
                if !same {
                    tiles.insert((tx, ty));
                }
            }
        }
    }
    tiles
}

/// Sublayer `eid`'s images by channel (None: its mask image)
fn sublayer_images(pixels: &PixelStore, eid: LayerId) -> BTreeMap<Option<Channel>, &TileImage> {
    let layers = pixels.layers.iter().filter(|((id, _), _)| *id == eid).map(|((_, channel), image)| (Some(*channel), image));
    layers.chain(pixels.masks.get(&eid).map(|mask| (None, mask))).collect()
}

/// The camera and viewport a stored path paints through (the views draw its curve the same way)
fn paint_view(shape: &PathShape, paint: &PathPaint) -> (OrbitCamera, Vec2) {
    (paint.camera, Vec2::from_array(shape.viewport).max(Vec2::splat(MIN_PAINT_VIEWPORT_PX)).min(Vec2::splat(MAX_PAINT_VIEWPORT_PX)))
}

/// NDC (y up) to viewport pixels (origin top-left).
fn to_pixels(ndc: Vec2, viewport: Vec2) -> Vec2 {
    Vec2::new((ndc.x * 0.5 + 0.5) * viewport.x, (0.5 - ndc.y * 0.5) * viewport.y)
}

impl Session {
    /// The path's control points in view space: viewport pixels for a 3D path (None behind the
    /// camera), texture coordinates for a 2D one.
    pub fn path_view_points(&self, viewport: Vec2) -> Vec<Option<Vec2>> {
        let path = &self.tool.path;
        if path.uv {
            return path.points.iter().map(|p| Some(p.truncate())).collect();
        }

        let aspect = viewport.x.max(1.0) / viewport.y.max(1.0);
        path.points.iter().map(|p| self.camera.project(*p, aspect).map(|n| to_pixels(n.truncate(), viewport))).collect()
    }

    /// The stored paths of path sublayer `eid` as Path tool states, for the views to draw
    pub fn stored_path_states(&self, eid: LayerId) -> Vec<PathState> {
        self.paths(eid).unwrap_or_default().iter().map(state_of).collect()
    }

    /// The stored paths of path sublayer `eid` (None if it isn't one).
    pub fn paths(&self, eid: LayerId) -> Option<&[PathShape]> {
        match self.doc.project.effect(eid) {
            Some((_, Effect { kind: EffectKind::Path(shapes), .. })) => Some(shapes),
            _ => None,
        }
    }

    /// The Path tool's curve with the current brush and material, as painted from `camera`.
    fn path_shape(&self, camera: OrbitCamera, viewport: Vec2) -> PathShape {
        let (t, path) = (&self.tool, &self.tool.path);
        let paint = PathPaint { brush: t.brush.clone(), values: t.values.clone(), enabled: t.enabled.clone(), mask_value: t.mask_value, camera };
        PathShape {
            points: path.points.iter().map(|p| p.to_array()).collect(),
            uv: path.uv,
            closed: path.closed,
            fill: path.fill,
            viewport: viewport.to_array(),
            paint: serde_json::to_value(paint).unwrap_or_default(),
        }
    }

    /// Store the Path tool's curve (new, or the path being edited) and repaint its sublayer as one undo step, then clear the tool
    pub fn commit_path(&mut self, viewport: Vec2) -> Result<Value> {
        if let Some((effect, index)) = self.tool.path.editing
            && self.paths(effect).and_then(|paths| paths.get(index)).is_some_and(|stored| self.same_as_stored(stored))
        {
            // Camera navigation alone must not replace the pixels the user just previewed
            let mut report = match self.path_draft.take() {
                Some(draft) => {
                    self.push_step(NEW_PATH_LABEL, draft.base, false);
                    draft.painted
                }
                None => json!({}),
            };
            report["effect"] = json!(effect.0);
            report["index"] = json!(index);
            self.target = StackTarget::Effect(effect);
            self.cancel_path();
            return Ok(report);
        }

        if let Some(draft) = self.path_draft.take() {
            // Changed settings paint again from the document before the preview
            let preview = std::mem::replace(&mut self.doc, draft.base);
            self.changed_without_history(&preview);
            self.tool.path.editing = None;
        }

        let shape = self.path_shape(self.camera, viewport);
        let editing = self.tool.path.editing;
        let label = editing.map_or(NEW_PATH_LABEL.to_string(), |(_, i)| format!("Edit path {}", i + 1));
        let out = self.edit_merging(&label, editing.is_some() && self.tool.path.edited, |s| {
            let (eid, index) = match editing {
                Some(at) => at,
                None => (s.path_effect()?, usize::MAX),
            };
            s.put_path(eid, index, shape)
        })?;

        if let Some(eid) = out.get("effect").and_then(Value::as_u64) {
            self.target = StackTarget::Effect(LayerId(eid));
        }
        self.tool.path.clear();

        Ok(out)
    }

    /// Repaint the edited path (one merged undo step) or a new path's preview (no step) when its points, brush or material changed; called every frame
    pub fn path_live(&mut self) -> Result<bool> {
        self.path_live_in_view(self.viewport)
    }

    /// Compare curve and typed paint settings through the stored camera and viewport
    fn same_as_stored(&self, stored: &PathShape) -> bool {
        // JSON round trips can change float representation without changing the paint settings
        read_paint(stored).is_ok_and(|paint| {
            let current = self.path_shape(paint.camera, Vec2::from_array(stored.viewport));
            (&current.points, current.uv, current.closed, current.fill) == (&stored.points, stored.uv, stored.closed, stored.fill)
                && read_paint(&current).is_ok_and(|current_paint| current_paint == paint)
        })
    }

    /// Preview a command's points in its supplied viewport, or the UI's current viewport
    fn path_live_in_view(&mut self, viewport: Vec2) -> Result<bool> {
        let stored = match self.tool.path.editing {
            Some((eid, index)) => match self.paths(eid).and_then(|v| v.get(index)).cloned() {
                Some(shape) => Some((eid, index, shape)),
                None => {
                    self.stop_path_edit();
                    return Ok(false);
                }
            },
            None => None,
        };

        if self.tool.tool != Tool::Path || self.stroke.is_some() {
            return Ok(false);
        }

        if self.tool.path.points.is_empty() {
            if let Some((effect, index, _)) = stored {
                if self.path_draft.is_some() {
                    self.cancel_path();
                } else {
                    self.edit_merging(&format!("Edit path {}", index + 1), self.tool.path.edited, |session| {
                        if let Some(Effect { kind: EffectKind::Path(paths), .. }) = session.doc.project.effect_mut(effect) {
                            paths.remove(index);
                        }
                        session.render_paths(effect, index)
                    })?;
                    self.cancel_path();
                }
                return Ok(true);
            }
            return Ok(false);
        }

        let path = &self.tool.path;
        if path.closed && path.fill && path.points.len() < 3 {
            return Ok(false);
        }

        let Some((eid, index, stored)) = stored else { return self.start_path_draft(viewport) };

        if self.same_as_stored(&stored) {
            return Ok(false);
        }

        let shape = self.path_shape(self.camera, viewport);
        let (painted, before) = self.repaint_path(eid, index, shape)?;
        match &mut self.path_draft {
            Some(draft) => draft.painted = painted,
            None => {
                // The path's own repaint keeps its point changes undoable, unlike other steps
                let steps = std::mem::take(&mut self.path_steps);
                self.push_step(&format!("Edit path {}", index + 1), before, self.tool.path.edited);
                self.path_steps = steps;
                self.tool.path.edited = true;
            }
        }
        Ok(true)
    }

    /// Paint a new path's first preview where [`Session::commit_path`] would add it, keeping the document before it for apply and cancel
    fn start_path_draft(&mut self, viewport: Vec2) -> Result<bool> {
        let base = self.doc.clone();
        let shape = self.path_shape(self.camera, viewport);
        let placed = self.path_effect().and_then(|eid| {
            let index = self.paths(eid).map_or(0, <[_]>::len);
            self.put_path(eid, index, shape).map(|painted| ((eid, index), painted))
        });
        match placed {
            Ok((at, painted)) => {
                // Placing a path is a new action: what was undone before can't be redone over it
                self.redo.clear();
                self.tool.path.editing = Some(at);
                self.changed_without_history(&base);
                self.path_draft = Some(PathDraft { base, painted });
                Ok(true)
            }
            Err(e) => {
                self.doc = base;
                Err(e)
            }
        }
    }

    /// Remember the curve before a point change from a view, so undo steps back through those changes while the path is edited
    pub fn path_checkpoint(&mut self) {
        if !self.path_steps.0.last().is_some_and(|s| s.same_curve(&self.tool.path)) {
            self.path_steps.0.push(self.tool.path.clone());
            // Each snapshot can contain thousands of points; bound point history like document history
            let excess = self.path_steps.0.len().saturating_sub(crate::session::MAX_HISTORY);
            self.path_steps.0.drain(..excess);
            self.path_steps.1.clear();
        }
    }

    /// Select a point without changing its geometry or clearing redo
    pub fn select_path_point(&mut self, index: usize) {
        self.tool.path.selected = (index < self.tool.path.points.len()).then_some(index);
    }

    /// Append and select a point, checkpointing the curve before it changes
    pub fn append_path_point(&mut self, point: Vec3, uv: bool) -> Option<usize> {
        if !point.is_finite() || (self.tool.path.uv == uv && self.tool.path.points.len() >= MAX_PATH_POINTS) {
            return None;
        }

        // A click in the other view must not replace a committed path's coordinate space
        if self.tool.path.editing.is_some() && self.path_draft.is_none() && self.tool.path.uv != uv {
            return None;
        }

        self.path_checkpoint();
        self.tool.path.push(point, uv);
        self.tool.path.selected = self.tool.path.points.len().checked_sub(1);
        self.tool.path.selected
    }

    /// Insert and select a point as one undoable point edit
    pub fn insert_path_point(&mut self, index: usize, point: Vec3) -> Option<usize> {
        if !point.is_finite() || index > self.tool.path.points.len() || self.tool.path.points.len() >= MAX_PATH_POINTS {
            return None;
        }

        self.path_checkpoint();
        self.tool.path.insert(index, point)
    }

    /// Move a point, checkpointing only the first actual movement in a drag
    pub fn move_path_point(&mut self, index: usize, point: Vec3, begin_drag: bool) -> bool {
        if !point.is_finite() || self.tool.path.points.get(index).is_none_or(|current| *current == point) {
            return false;
        }

        if begin_drag {
            self.path_checkpoint();
        }
        if let Some(current) = self.tool.path.points.get_mut(index) {
            *current = point;
        }
        true
    }

    /// Remove the selected point, or the last one, as one undoable point edit
    pub fn remove_selected_path_point(&mut self) {
        if self.tool.path.points.is_empty() {
            return;
        }

        self.path_checkpoint();
        self.tool.path.remove_selected();
    }

    /// Change whether the path loops back to its first point, with point undo
    pub fn set_path_closed(&mut self, closed: bool) {
        if self.tool.path.closed != closed {
            self.path_checkpoint();
            self.tool.path.closed = closed;
        }
    }

    /// Change whether a closed path fills its interior, with point undo
    pub fn set_path_fill(&mut self, fill: bool) {
        if self.tool.path.fill != fill {
            self.path_checkpoint();
            self.tool.path.fill = fill;
        }
    }

    /// Whether [`Session::step_path`] would undo (`back`) or redo a point change
    pub(crate) fn path_step_pending(&self, back: bool) -> bool {
        let steps = if back { &self.path_steps.0 } else { &self.path_steps.1 };
        self.tool.tool == Tool::Path && steps.iter().any(|s| !s.same_curve(&self.tool.path))
    }

    /// Undo calls needed to move back through document history, including pending path operations.
    /// With zero document steps, only an uncommitted draft is canceled; stored edits stay applied.
    pub fn history_undo_steps(&self, document_steps: usize) -> usize {
        if document_steps == 0 && self.path_draft.is_none() {
            return 0;
        }

        let (point_steps, clears_path) = self.path_steps_before_document(true);
        let cancel_draft = self.path_draft.is_some() && !clears_path;
        document_steps.saturating_add(point_steps).saturating_add(usize::from(cancel_draft))
    }

    /// Redo calls needed to move forward through document history, including pending point edits.
    /// Unlike undo, document redo cancels any remaining draft in the same call.
    pub fn history_redo_steps(&self, document_steps: usize) -> usize {
        if document_steps == 0 {
            return 0;
        }

        document_steps.saturating_add(self.path_steps_before_document(false).0)
    }

    /// Count pending point steps and whether their last step clears the path
    fn path_steps_before_document(&self, back: bool) -> (usize, bool) {
        if self.tool.tool != Tool::Path {
            return (0, false);
        }

        let states = if back { &self.path_steps.0 } else { &self.path_steps.1 };
        let mut steps = 0usize;
        let mut current = &self.tool.path;
        for state in states.iter().rev() {
            if state.same_curve(current) {
                continue;
            }

            steps = steps.saturating_add(1);

            // Restoring an empty curve cancels the draft and clears point history in the same step
            if state.points.is_empty() {
                return (steps, true);
            }

            current = state;
        }

        (steps, false)
    }

    /// Undo (`back`) or redo a point change and repaint before returning; failure preserves the curve and history
    pub(crate) fn step_path(&mut self, back: bool) -> Result<bool> {
        if self.tool.tool != Tool::Path {
            return Ok(false);
        }

        // Skip no-change checkpoints without consuming history until the repaint succeeds
        let from = if back { &self.path_steps.0 } else { &self.path_steps.1 };
        let Some((index, state)) = from.iter().enumerate().rfind(|(_, state)| !state.same_curve(&self.tool.path)) else { return Ok(false) };

        let path = &mut self.tool.path;
        let previous = std::mem::replace(path, PathState { editing: path.editing, edited: path.edited, ..state.clone() });
        if let Err(error) = self.path_live() {
            self.tool.path = previous;
            return Err(error);
        }

        if self.tool.path.points.is_empty() {
            self.stop_path_edit();
        } else {
            let (from, to) = if back { (&mut self.path_steps.0, &mut self.path_steps.1) } else { (&mut self.path_steps.1, &mut self.path_steps.0) };
            from.truncate(index);
            to.push(previous);
        }
        Ok(true)
    }

    /// Apply a new path being placed (as Enter would) and end the edit of a stored one, e.g. before the active texture set changes
    pub fn finish_path(&mut self) -> Result<()> {
        if self.path_draft.is_some() {
            let viewport = self.viewport;
            self.commit_path(viewport)?;
        }
        self.stop_path_edit();
        Ok(())
    }

    /// Cancel the Path tool's path: a new one's preview is removed, a stored one keeps its applied changes
    pub fn cancel_path(&mut self) {
        self.stop_path_edit();
        self.tool.path.clear();
    }

    /// Stop editing a stored path (its changes are already applied) and clear the tool; a new path's preview is removed
    pub fn stop_path_edit(&mut self) {
        if let Some(draft) = self.path_draft.take() {
            let preview = std::mem::replace(&mut self.doc, draft.base);
            self.changed_without_history(&preview);
        }
        if self.tool.path.editing.is_some() {
            self.tool.path.clear();
        }
        self.path_steps = Default::default();
    }

    /// The selected path sublayer, or a new one on the selected layer's content or mask
    /// (whichever is selected).
    fn path_effect(&mut self) -> Result<LayerId> {
        let id = self.selected_layer().ok_or_else(|| EngineError::Other("select a layer to add the path to".into()))?;
        let p = &self.doc.project;
        let mask = match self.target {
            StackTarget::Effect(e) => match p.effect(e) {
                Some((_, x)) if matches!(x.kind, EffectKind::Path(_)) => return Ok(e),
                Some((owner, _)) => !p.layer(owner)?.content_effects.iter().any(|x| x.id == e),
                None => false,
            },
            StackTarget::Mask => true,
            StackTarget::Content => false,
        };
        // An instance's content and (unless it has its own) mask belong to its source.
        let owner = if mask { p.mask_owner(id) } else { p.content_layer(id).map_or(id, |l| l.id) };
        let eid = self.doc.project.alloc_id();
        let l = self.doc.project.layer_mut(owner)?;
        let e = Effect::new(eid, EffectKind::Path(Vec::new()));
        if mask {
            l.mask.get_or_insert_with(|| Mask::new(MaskBackground::Black)).effects.push(e);
        } else if l.is_group() {
            return Err(EngineError::Other(format!("\"{}\" is a folder: add the path to a layer or a mask", l.name)));
        } else {
            l.content_effects.push(e);
        }
        Ok(eid)
    }

    /// [`Session::put_path`] recompositing only the tiles it changed, with no undo step: returns the paint report and the document before
    fn repaint_path(&mut self, eid: LayerId, index: usize, shape: PathShape) -> Result<(Value, Document)> {
        let before = self.doc.clone();
        let painted = match self.put_path(eid, index, shape) {
            Ok(v) => v,
            Err(e) => {
                self.doc = before;
                return Err(e);
            }
        };

        // Paths paint into the active set; a sublayer in another set recomposites in full
        let project = &self.doc.project;
        let set = project.effect(eid).and_then(|(owner, _)| project.locate(owner)).map(|(ts, _, _)| ts);
        if set == Some(self.ts()) {
            // Content or mask, the sublayer's owner decides which sets redraw
            let tiles = changed_tiles(&before.pixels, &self.doc.pixels, eid);
            self.mark_painted(self.ts(), Some(PaintTarget::Layer(eid)), tiles);
        } else {
            self.invalidate_changed(&before);
        }
        self.revision = self.revision.wrapping_add(1);
        self.modified = true;
        Ok((painted, before))
    }

    /// Replace path `index` of sublayer `eid` (append when out of range) and repaint the
    /// sublayer. Returns what painting that path reported, with the sublayer and index.
    fn put_path(&mut self, eid: LayerId, index: usize, shape: PathShape) -> Result<Value> {
        let index = match self.doc.project.effect_mut(eid) {
            Some(Effect { kind: EffectKind::Path(shapes), .. }) => match shapes.get_mut(index) {
                Some(slot) => {
                    *slot = shape;
                    index
                }
                None => {
                    shapes.push(shape);
                    shapes.len() - 1
                }
            },
            _ => return Err(EngineError::Other(format!("{} is not a path sublayer", eid.0))),
        };
        let mut out = self.render_paths(eid, index)?.get(index).cloned().unwrap_or_else(|| json!({}));
        out["effect"] = json!(eid.0);
        out["index"] = json!(index);
        Ok(out)
    }

    /// Repaint path sublayer `eid` from its stored paths, bottom to top, where paths before
    /// `from` are unchanged: their pixels come from [`PathCache`] when it holds them, and are
    /// cached for the next repaint otherwise, so editing a path repaints it and the paths above
    /// only. Runs inside the caller's undo step. Returns each painted path's report (null for
    /// the cached ones).
    fn render_paths(&mut self, eid: LayerId, from: usize) -> Result<Vec<Value>> {
        let shapes = self.paths(eid).ok_or_else(|| EngineError::Other(format!("{} is not a path sublayer", eid.0)))?.to_vec();
        let key = (self.mesh_id, self.caches.get(self.ts()).map(|c| c.size));
        self.doc.pixels.masks.remove(&eid);
        self.doc.pixels.layers.retain(|(id, _), _| *id != eid);
        let start = match &self.path_cache {
            Some(c) if c.eid == eid && c.key == key && c.below.len() <= from && shapes.get(..c.below.len()) == Some(&c.below[..]) => {
                if let Some(m) = &c.mask {
                    self.doc.pixels.masks.insert(eid, m.clone());
                }
                self.doc.pixels.layers.extend(c.layers.iter().cloned());
                c.below.len()
            }
            _ => 0,
        };
        // Each path goes through the regular stroke and fill code with its own settings; their
        // own history entries are dropped (the caller records one step).
        let saved = (self.tool.clone(), self.camera, self.target, self.quick_mask.editing);
        let history = (std::mem::take(&mut self.undo), std::mem::take(&mut self.redo));
        self.target = StackTarget::Effect(eid);
        self.quick_mask.editing = false;
        self.tool.tool = Tool::Path;
        self.painting_paths = true;
        let mut out = vec![Value::Null; start];
        let mut result = Ok(());
        for i in start..=shapes.len() {
            if i == from.min(shapes.len()) && i > start {
                let pixels = &self.doc.pixels;
                let layers = pixels.layers.iter().filter(|((id, _), _)| *id == eid).map(|(k, v)| (*k, v.clone())).collect();
                self.path_cache = Some(PathCache { eid, key, below: shapes[..i].to_vec(), mask: pixels.masks.get(&eid).cloned(), layers });
            }
            let Some(sh) = shapes.get(i) else { break };
            match self.paint_shape(sh) {
                Ok(v) => out.push(v),
                Err(e) => {
                    result = Err(e);
                    break;
                }
            }
        }
        self.painting_paths = false;
        (self.tool, self.camera, self.target, self.quick_mask.editing) = saved;
        (self.undo, self.redo) = history;
        result.map(|_| out)
    }

    fn paint_shape(&mut self, shape: &PathShape) -> Result<Value> {
        let paint = read_paint(shape)?;
        let (camera, viewport) = paint_view(shape, &paint);
        self.tool.brush = paint.brush;
        self.tool.values = paint.values;
        self.tool.enabled = paint.enabled;
        self.tool.mask_value = paint.mask_value;
        self.camera = camera;
        self.tool.path = state_of(shape);
        self.paint_path(viewport)
    }

    /// Where a 3D path's curve lands on the mesh painted through `camera`, per segment as [`spline_segments`] (None off the mesh)
    pub fn path_surface(&self, points: &[Vec3], closed: bool, camera: OrbitCamera, viewport: Vec2) -> Vec<Vec<Option<Vec3>>> {
        let aspect = viewport.x.max(1.0) / viewport.y.max(1.0);
        let Some(screen) = points.iter().map(|p| camera.project(*p, aspect).map(|n| to_pixels(n.truncate(), viewport))).collect::<Option<Vec<_>>>() else {
            return Vec::new();
        };

        let ctx = PickCtx { mesh: &self.mesh, bvh: &self.bvh, camera: &camera, viewport };
        spline_segments(&screen, closed, SURFACE_STEP_PX)
            .into_iter()
            .zip(&screen)
            .map(|(samples, first)| std::iter::once(*first).chain(samples).map(|q| pick(&ctx, q).map(|hit| hit.pos)).collect())
            .collect()
    }

    /// [`Session::path_surface`] of a stored 3D path through the camera and viewport it paints with
    pub fn stored_path_surface(&self, eid: LayerId, index: usize) -> Option<Vec<Vec<Option<Vec3>>>> {
        let shape = self.paths(eid)?.get(index)?;
        if shape.uv {
            return None;
        }

        let (camera, viewport) = paint_view(shape, &read_paint(shape).ok()?);
        let path = state_of(shape);
        Some(self.path_surface(&path.points, path.closed, camera, viewport))
    }

    /// Paint the Path tool's curve with the current brush (or fill it when closed and filled)
    /// into the paint target.
    fn paint_path(&mut self, viewport: Vec2) -> Result<Value> {
        let (uv, closed) = (self.tool.path.uv, self.tool.path.closed);
        let fill = closed && self.tool.path.fill;
        let pts: Vec<Vec2> = self.path_view_points(viewport).into_iter().flatten().collect();
        if pts.is_empty() {
            return Err(EngineError::Other("the path has no points: click the surface to add some".into()));
        }
        if fill && pts.len() < 3 {
            return Err(EngineError::Other("a filled path needs at least three points".into()));
        }
        let curve = spline(&pts, closed, if uv { STEP_UV } else { STEP_PX });
        Ok(if fill {
            json!({ "filledTiles": self.lasso_fill(curve, uv, viewport)? })
        } else if uv {
            json!({ "tiles": self.stroke_uv_points(&curve)? })
        } else {
            json!({ "stamps": self.stroke_points(&curve, viewport)? })
        })
    }

    /// Fill the inside of `polygon` (viewport pixels over the visible surface, or texture coordinates); returns the touched tiles
    fn lasso_fill(&mut self, polygon: Vec<Vec2>, uv: bool, viewport: Vec2) -> Result<usize> {
        let texture_set = self.ts();
        let (uv_buffer, mut canvas) = self.fill_canvas(texture_set)?;
        let width = uv_buffer.width.max(1);

        if uv {
            let lasso = Lasso::new(polygon.iter().map(|point| *point * Vec2::new(width as f32, uv_buffer.height as f32)).collect());
            fill_coverage(&mut canvas, &uv_buffer, |texel, _| {
                lasso.coverage(Vec2::new((texel as u32 % width) as f32 + 0.5, (texel as u32 / width) as f32 + 0.5))
            });
        } else {
            let lasso = Lasso::new(polygon);
            let view_projection = self.camera.view_proj(viewport.x.max(1.0) / viewport.y.max(1.0));
            let inverse_view_projection = view_projection.inverse();
            let (mesh, bvh) = (&self.mesh, &self.bvh);
            let brush = self.tool.brush.sanitized();

            // A visible source inside the lasso can paint each symmetry copy
            let (bounds_min, bounds_max) = mesh.bounds();
            let symmetry = brush.symmetry;
            let symmetry_origin = symmetry.origin((bounds_min + bounds_max) * 0.5);
            let inverses: Vec<_> = symmetry.transforms().iter().map(|transform| transform.inverse()).collect();

            // Copies can leave the source surface, within the same depth allowed for brush stamps
            let copy_depth = brush.size * (bounds_max - bounds_min).length().max(MIN_BRUSH_SCALE) * brush.depth;
            let eye = self.camera.eye();
            let camera_distance = self.camera.distance.max(MIN_BRUSH_SCALE);

            // Bound the search using the farthest bounding-box corner; acceptance still uses the actual hit
            let max_copy_depth =
                if brush.screen_space { copy_depth * (bounds_min - eye).abs().max((bounds_max - eye).abs()).length() / camera_distance } else { copy_depth };

            fill_coverage(&mut canvas, &uv_buffer, |texel, _| {
                let Some(surface) = uv_buffer.surface(mesh, texel) else { return 0.0 };
                let mut best = 0.0f32;

                for inverse in &inverses {
                    let position = symmetry_origin + *inverse * (surface.pos - symmetry_origin);
                    let clip = view_projection * position.extend(1.0);
                    if clip.w <= MIN_PROJECTIVE_W {
                        continue;
                    }

                    let normalized_screen = Vec2::new(clip.x, clip.y) / clip.w;
                    let coverage = lasso.coverage(to_pixels(normalized_screen, viewport));
                    if coverage <= best {
                        continue;
                    }

                    // Use the source surface normal; a copy may face away from the camera
                    let near_clip = inverse_view_projection * normalized_screen.extend(0.0).extend(1.0);
                    let origin = near_clip.truncate() / near_clip.w;
                    let to_surface = position - origin;
                    let distance = to_surface.length();
                    let Some(direction) = to_surface.try_normalize() else { continue };
                    let is_copy = *inverse != Mat3::IDENTITY;
                    let max_distance = distance * (1.0 + RAY_HIT_RELATIVE_TOLERANCE);
                    let search_depth = if is_copy { max_copy_depth } else { 0.0 };
                    let Some(hit) = bvh.closest(origin, direction, max_distance + search_depth) else { continue };

                    // Keep identity copies exact and reject occluders beyond the bounded displacement
                    let off_surface = if !is_copy {
                        0.0
                    } else if brush.screen_space {
                        copy_depth * (origin + direction * hit.t).distance(eye) / camera_distance
                    } else {
                        copy_depth
                    };
                    if hit.t < distance * (1.0 - RAY_HIT_RELATIVE_TOLERANCE) - off_surface || hit.t > max_distance + off_surface {
                        continue;
                    }

                    let Some(source) = surface_at(mesh, hit.tri, hit.u, hit.v) else { continue };

                    if brush.backface_culling && source.nrm.dot(direction) >= 0.0 {
                        continue;
                    }

                    best = coverage;
                }

                best
            });
        }

        let tiles = canvas.dirty.len();
        self.commit_fill("Path fill", texture_set, &canvas)?;
        Ok(tiles)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    use crate::{Session, execute};

    /// The path sublayer's pixels, comparable.
    fn pixels(s: &Session, eid: texturecraft_core::LayerId) -> Vec<String> {
        let mut v: Vec<String> = s.doc.pixels.layers.iter().filter(|((id, _), _)| *id == eid).map(|(k, img)| format!("{k:?}{img:?}")).collect();
        v.sort();
        v
    }

    #[test]
    fn cached_repaint_matches_a_full_repaint() {
        let mut s = Session::new_project("t", texturecraft_geom::primitives::sphere(1.0, 24, 12), "builtin:test", None, 128).unwrap();
        s.update();
        execute(&mut s, "tool.brush", &json!({"size": 6})).unwrap();
        let mut eid = 0;
        for v in [0.2, 0.5, 0.8] {
            let r = execute(&mut s, "tool.path", &json!({"uvPoints": [[0.1, v], [0.9, v + 0.05]], "commit": true})).unwrap();
            eid = r["committed"]["effect"].as_u64().unwrap();
        }
        let eid = texturecraft_core::LayerId(eid);
        // Live edits of the middle path reuse the bottom path's pixels.
        execute(&mut s, "path.edit", &json!({"effect": eid.0, "index": 1})).unwrap();
        for v in [0.4, 0.45] {
            execute(&mut s, "tool.path", &json!({"uvPoints": [[0.1, v], [0.5, v], [0.9, v]]})).unwrap();
        }
        assert_eq!(s.path_cache.as_ref().map(|c| c.below.len()), Some(1));
        let r = s.render_paths(eid, 1).unwrap();
        assert!(r[0].is_null() && !r[1].is_null(), "the bottom path came from the cache: {r:?}");
        let cached = pixels(&s, eid);
        s.path_cache = None;
        s.render_paths(eid, 0).unwrap();
        assert_eq!(cached, pixels(&s, eid), "same pixels as painting every path");
        assert!(!cached.is_empty());
    }

    #[test]
    fn point_insertion_and_removal_respect_boundaries() {
        let mut path = PathState { points: vec![Vec3::ZERO, Vec3::X], ..Default::default() };

        assert_eq!(path.insert(2, Vec3::Y), Some(2));
        assert_eq!(path.points, vec![Vec3::ZERO, Vec3::X, Vec3::Y]);
        assert_eq!(path.insert(4, Vec3::Z), None);
        path.closed = true;
        path.selected = Some(usize::MAX);
        path.remove_selected();
        assert_eq!(path.points, vec![Vec3::ZERO, Vec3::X]);
        assert!(!path.closed);
        path.remove_selected();
        path.remove_selected();
        path.remove_selected();
        assert!(path.points.is_empty());
        assert_eq!(path.selected, None);

        path.points = vec![Vec3::ZERO; MAX_PATH_POINTS];
        assert_eq!(path.insert(MAX_PATH_POINTS, Vec3::X), None);
        assert_eq!(path.points.len(), MAX_PATH_POINTS);
    }

    #[test]
    fn changed_tiles_include_mask_and_channel_creation_and_deletion() {
        let effect = LayerId(7);
        let mut before = PixelStore::default();
        let mut image = TileImage::new(192, 128, 1);
        image.tile_mut(0, 0, &[1]).unwrap();
        before.masks.insert(effect, image.clone());
        before.layers.insert((effect, Channel::Roughness), image.clone());
        let mut after = before.clone();
        after.masks.get_mut(&effect).unwrap().set_tile(0, 0, None);
        after.masks.get_mut(&effect).unwrap().tile_mut(2, 1, &[1]).unwrap();
        after.layers.remove(&(effect, Channel::Roughness));
        image.set_tile(0, 0, None);
        image.tile_mut(1, 0, &[1]).unwrap();
        after.layers.insert((effect, Channel::Metallic), image);

        assert_eq!(changed_tiles(&before, &before.clone(), effect), BTreeSet::new());
        assert_eq!(changed_tiles(&before, &after, effect), BTreeSet::from([(0, 0), (1, 0), (2, 1)]));
        assert_eq!(changed_tiles(&before, &after, LayerId(8)), BTreeSet::new());
    }

    #[test]
    fn point_checkpoints_skip_noops_and_bound_retained_points() {
        let mut session = Session::new_project("t", texturecraft_geom::primitives::plane(1.0, 1), "builtin:test", None, 128).unwrap();
        session.tool.tool = Tool::Path;
        session.tool.path.points = vec![Vec3::ZERO; MAX_PATH_POINTS];
        session.path_checkpoint();
        assert!(!session.path_step_pending(true));
        assert!(!session.step_path(true).unwrap());

        for index in 0..crate::session::MAX_HISTORY + 10 {
            session.path_checkpoint();
            session.tool.path.points[0].x = index as f32 + 1.0;
        }

        assert_eq!(session.path_steps.0.len(), crate::session::MAX_HISTORY);
        assert_eq!(session.path_steps.0.iter().map(|state| state.points.len()).sum::<usize>(), crate::session::MAX_HISTORY * MAX_PATH_POINTS);
        session.path_checkpoint();
        assert!(session.step_path(true).unwrap(), "a no-change checkpoint does not consume an undo");
        assert_eq!(session.tool.path.points[0].x, (crate::session::MAX_HISTORY + 9) as f32);
    }

    #[test]
    fn stored_surface_uses_bounded_viewports_and_skips_uv_paths() {
        let mut session = Session::new_project("t", texturecraft_geom::primitives::sphere(1.0, 24, 12), "builtin:test", None, 128).unwrap();
        execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": true})).unwrap();
        let StackTarget::Effect(effect) = session.target else { panic!("path sublayer") };
        assert!(session.stored_path_surface(effect, 0).is_none());
        let shape = session.paths(effect).unwrap()[0].clone();
        let paint = read_paint(&shape).unwrap();
        let mut shape = shape;
        shape.viewport = [-1.0, f32::MAX];
        assert_eq!(paint_view(&shape, &paint).1, Vec2::new(MIN_PAINT_VIEWPORT_PX, MAX_PAINT_VIEWPORT_PX));

        let points = [Vec3::new(-0.2, 0.0, 1.0), Vec3::new(0.0, 0.2, 1.0), Vec3::new(0.2, 0.0, 1.0)];
        session.camera.yaw = 0.0;
        session.camera.pitch = 0.0;
        let surface = session.path_surface(&points, true, session.camera, Vec2::splat(400.0));
        assert_eq!(surface.len(), 3);
        let first = surface.first().unwrap().first().unwrap().unwrap();
        let last = surface.last().unwrap().last().unwrap().unwrap();
        assert!(first.distance(last) < 1e-5, "the closed curve returns to its starting surface point");
        let off_mesh = session.path_surface(&[Vec3::new(10.0, 0.0, 0.0), Vec3::new(11.0, 0.0, 0.0)], false, session.camera, Vec2::splat(400.0));
        assert!(!off_mesh.is_empty());
        assert!(off_mesh.iter().flatten().all(Option::is_none));
    }
}
