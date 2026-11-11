//! The Path tool (stroked and filled paths, 3D and 2D) and the particle brush, driven through
//! engine commands the way agents paint.

use serde_json::{Value, json};
use texturecraft_engine::core::Channel;
use texturecraft_engine::geom::glam::{Vec2, Vec3, Vec4};
use texturecraft_engine::geom::{Mesh, primitives};
use texturecraft_engine::{Session, execute};

const RES: f32 = 256.0;

fn session(mesh: Mesh) -> Session {
    let mut s = Session::new_project("t", mesh, "builtin:test", None, 256).unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    s.camera.yaw = 0.0;
    s.camera.pitch = 0.0;
    s.update();
    execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [1.0, 0.0, 0.0]})).unwrap();
    execute(&mut s, "tool.brush", &json!({"size": 6, "hardness": 90})).unwrap();
    s
}

fn red_uv(s: &Session, u: f32, v: f32) -> bool {
    let c = s.caches[0].composite.value(Channel::BaseColor, (u * RES) as u32, (v * RES) as u32);
    c[0] > 0.85 && c[1] < 0.15
}

/// Is the surface under a viewport point painted red?
fn red_at(s: &Session, x: f32, y: f32) -> bool {
    let hit = s.pick(Vec2::new(x, y), s.viewport).unwrap();
    red_uv(s, hit.uv.x, hit.uv.y)
}

#[test]
fn path_stroke_paints_along_the_curve() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]]})).unwrap();
    assert_eq!(r["points"], 3);
    assert_eq!(s.tool.tool, texturecraft_engine::Tool::Path);
    assert!(red_at(&s, 200.0, 180.0), "the path is previewed before it is applied");
    let r = execute(&mut s, "tool.path", &json!({"commit": true})).unwrap();
    assert!(r["committed"]["stamps"].as_u64().unwrap() > 10, "{r}");
    assert_eq!(r["points"], 0, "commit clears the path");
    for (x, y) in [(140.0, 210.0), (200.0, 180.0), (260.0, 210.0), (170.0, 191.0)] {
        assert!(red_at(&s, x, y), "on the curve at {x},{y}");
    }
    assert!(!red_at(&s, 200.0, 240.0), "below the curve");
    assert!(!red_at(&s, 200.0, 130.0), "above the curve");
    // One undo step
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(!red_at(&s, 200.0, 180.0));
    // Errors: no points, points off the mesh
    assert!(execute(&mut s, "tool.path", &json!({"commit": true})).is_err());
    assert!(execute(&mut s, "tool.path", &json!({"points": [[2, 2]]})).is_err());
}

#[test]
fn uv_path_strokes_in_texture_space() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.5, 0.45], [0.8, 0.5]], "commit": true})).unwrap();
    assert!(red_uv(&s, 0.5, 0.45) && red_uv(&s, 0.2, 0.5) && red_uv(&s, 0.8, 0.5));
    assert!(!red_uv(&s, 0.5, 0.6));
}

#[test]
fn closed_filled_path_fills_the_inside_only() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let square = json!([[160, 160], [240, 160], [240, 240], [160, 240]]);
    execute(&mut s, "tool.path", &json!({"points": square, "closed": true, "fill": true, "commit": true})).unwrap();
    for (x, y) in [(200.0, 200.0), (170.0, 170.0), (230.0, 230.0)] {
        assert!(red_at(&s, x, y), "inside at {x},{y}");
    }
    for (x, y) in [(200.0, 125.0), (275.0, 200.0), (125.0, 200.0), (200.0, 275.0)] {
        assert!(!red_at(&s, x, y), "outside at {x},{y}");
    }
    // The far side of the sphere, hidden from the camera, stays clean.
    s.camera.yaw = std::f32::consts::PI;
    assert!(!red_at(&s, 200.0, 200.0), "hidden back side");

    // A 2D lasso fills the UV region.
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.path", &json!({"uvPoints": [[0.3, 0.3], [0.7, 0.3], [0.7, 0.7], [0.3, 0.7]], "closed": true, "fill": true, "commit": true})).unwrap();
    assert!(red_uv(&s, 0.5, 0.5) && red_uv(&s, 0.35, 0.65));
    assert!(!red_uv(&s, 0.2, 0.5) && !red_uv(&s, 0.5, 0.85));
    // Options persist between paths; a filled path needs three points.
    assert!(execute(&mut s, "tool.path", &json!({"uvPoints": [[0.1, 0.1], [0.2, 0.2]], "closed": false, "commit": true})).is_ok());
    assert!(execute(&mut s, "tool.path", &json!({"uvPoints": [[0.1, 0.1], [0.2, 0.2]], "closed": true, "fill": true, "commit": true})).is_err());
}

#[test]
fn closed_filled_path_is_symmetrized() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.brush", &json!({"symmetryX": true})).unwrap();

    let square = json!([[230, 160], [270, 160], [270, 240], [230, 240]]);
    execute(&mut s, "tool.path", &json!({"points": square, "closed": true, "fill": true, "commit": true})).unwrap();

    assert!(red_at(&s, 250.0, 200.0), "the drawn side");
    assert!(red_at(&s, 150.0, 200.0), "the mirrored side");
    assert!(!red_at(&s, 200.0, 200.0), "between the two");
}

#[test]
fn closed_filled_path_copies_from_visible_source_to_backfacing_destination() {
    let mut mesh = wall();
    for (position, normal) in mesh.positions.iter().zip(&mut mesh.normals) {
        *normal = if position.x < 0.0 { Vec3::NEG_Z } else { Vec3::Z };
    }

    let mut s = session(mesh.clone());
    execute(&mut s, "tool.brush", &json!({"symmetryX": true})).unwrap();

    let square = json!([[230, 170], [270, 170], [270, 230], [230, 230]]);
    execute(&mut s, "tool.path", &json!({"points": square, "closed": true, "fill": true, "commit": true})).unwrap();

    assert!(red_at(&s, 250.0, 200.0), "visible source");
    assert!(red_at(&s, 150.0, 200.0), "backfacing symmetry copy");
    assert!(!red_at(&s, 200.0, 200.0), "outside the lasso and its copy");

    let mut s = session(mesh);
    execute(&mut s, "tool.brush", &json!({"symmetryX": true})).unwrap();

    let square = json!([[130, 170], [170, 170], [170, 230], [130, 230]]);
    execute(&mut s, "tool.path", &json!({"points": square, "closed": true, "fill": true, "commit": true})).unwrap();

    assert!(!red_at(&s, 150.0, 200.0), "backfacing source");
    assert!(!red_at(&s, 250.0, 200.0), "no copy from a hidden source");
}

/// A 2×2 wall facing the camera (+Z): v grows downward (world −Y).
fn wall() -> Mesh {
    let mut m = primitives::plane(2.0, 8);
    for p in &mut m.positions {
        *p = Vec3::new(p.x, -p.z, 0.0);
    }
    m.normals.iter_mut().for_each(|n| *n = Vec3::Z);
    m.tangents.iter_mut().for_each(|t| *t = Vec4::new(1.0, 0.0, 0.0, 1.0));
    m
}

/// Painted texel rows (v) after one particle dab at the wall's center.
fn particle_rows() -> Vec<u32> {
    let mut s = session(wall());
    execute(&mut s, "tool.brush", &json!({"size": 10, "particles": {"enabled": true, "count": 6, "lifetime": 40, "gravity": 1.5}})).unwrap();
    let hit = s.pick(Vec2::new(200.0, 200.0), s.viewport).unwrap();
    assert!((hit.uv.y - 0.5).abs() < 0.02, "{hit:?}");
    execute(&mut s, "tool.stroke", &json!({"points": [[200, 200]]})).unwrap();
    let mut rows = Vec::new();
    for y in 0..RES as u32 {
        for x in 0..RES as u32 {
            let c = s.caches[0].composite.value(Channel::BaseColor, x, y);
            if c[0] > 0.85 && c[1] < 0.5 {
                rows.push(y);
            }
        }
    }
    rows
}

#[test]
fn particle_brush_paints_below_the_start_and_is_deterministic() {
    let rows = particle_rows();
    assert!(!rows.is_empty());
    let start = (0.5 * RES) as u32;
    let lowest = rows.iter().max().copied().unwrap();
    let highest = rows.iter().min().copied().unwrap();
    assert!(lowest > start + 40, "trails run down the wall: {lowest}");
    assert!(highest + 20 > start, "nothing well above the start: {highest}");
    assert_eq!(rows, particle_rows());
}

#[test]
fn radial_symmetry_paints_copies_around_the_axis() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    assert!(execute(&mut s, "tool.brush", &json!({"symmetryRadial": 2, "symmetryAxis": "w"})).is_err());
    let r = execute(&mut s, "tool.brush", &json!({"symmetryRadial": 2, "symmetryAxis": "y"})).unwrap();
    assert_eq!(r["radialSymmetry"], json!({"copies": 2, "axis": "y"}));
    execute(&mut s, "tool.stroke", &json!({"points": [[200, 200]]})).unwrap();
    s.update();
    assert!(red_at(&s, 200.0, 200.0));
    // Half a turn around Y: the copy faces the other way.
    s.camera.yaw = std::f32::consts::PI;
    assert!(red_at(&s, 200.0, 200.0), "the radial copy on the back");
}

/// The selected layer's content path sublayers: (effect id, stored paths).
fn path_sublayers(s: &Session) -> Vec<(u64, usize)> {
    let l = s.doc.project.layer(s.selected_layer().unwrap()).unwrap();
    l.content_effects.iter().filter_map(|e| s.paths(e.id).map(|p| (e.id.0, p.len()))).collect()
}

#[test]
fn committed_paths_stay_editable_in_a_path_sublayer() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let eid = r["committed"]["effect"].as_u64().unwrap();
    assert_eq!(r["committed"]["index"], 0);
    assert_eq!(path_sublayers(&s), vec![(eid, 1)]);
    // The next path goes into the same (selected) sublayer.
    let r = execute(&mut s, "tool.path", &json!({"uvPoints": [[0.2, 0.9], [0.8, 0.9]], "commit": true})).unwrap();
    assert_eq!((r["committed"]["effect"].as_u64(), r["committed"]["index"].as_u64()), (Some(eid), Some(1)));
    assert_eq!(path_sublayers(&s), vec![(eid, 2)]);
    assert!(red_at(&s, 200.0, 180.0) && red_uv(&s, 0.5, 0.9));
    let inspect = execute(&mut s, "project.inspect", &Value::Null).unwrap().to_string();
    assert!(inspect.contains("\"paths\""), "inspect lists the paths");
    // Paths are saved with the project; editing a loaded path repaints only once it changes.
    let saved = serde_json::to_string(&s.doc.project).unwrap();
    s.doc.project = serde_json::from_str(&saved).unwrap();
    assert_eq!(path_sublayers(&s), vec![(eid, 2)]);
    execute(&mut s, "path.edit", &json!({"effect": eid, "index": 0})).unwrap();
    assert!(!s.path_live().unwrap(), "a loaded path matches its settings");
    execute(&mut s, "tool.path", &json!({"clear": true})).unwrap();

    // Manual strokes can't paint over a path sublayer.
    let err = execute(&mut s, "tool.stroke", &json!({"points": [[200, 240]]})).unwrap_err().to_string();
    assert!(err.contains("path sublayer"), "{err}");

    // Edit the first path: moving its points repaints it live, the other path stays.
    let undo_before = s.history_labels().len();
    let r = execute(&mut s, "path.edit", &json!({"effect": eid, "index": 0})).unwrap();
    assert_eq!(r["points"], 3);
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 250], [200, 230], [260, 250]]})).unwrap();
    assert_eq!(r["editing"]["index"], 0, "{r}");
    assert!(red_at(&s, 200.0, 230.0), "the edited curve shows at once");
    assert!(!red_at(&s, 200.0, 180.0), "the old curve is gone");
    assert!(red_uv(&s, 0.5, 0.9), "the other path is kept");
    // Brush changes repaint too, merged into the same undo step.
    execute(&mut s, "tool.material", &json!({"channel": "baseColor", "color": [0.0, 0.0, 1.0]})).unwrap();
    execute(&mut s, "tool.path", &json!({})).unwrap();
    assert!(!red_at(&s, 200.0, 230.0), "repainted with the new color");
    assert_eq!(s.history_labels().len(), undo_before + 1);
    execute(&mut s, "tool.path", &json!({"commit": true})).unwrap();
    assert_eq!(s.history_labels().len(), undo_before + 1, "one step for the whole edit");
    assert!(s.tool.path.editing.is_none() && s.tool.path.points.is_empty());
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(red_at(&s, 200.0, 180.0) && !red_at(&s, 200.0, 230.0), "undo restores the original path");

    // Remove a path: its paint goes, the sublayer keeps the rest.
    execute(&mut s, "path.remove", &json!({"effect": eid, "index": 0})).unwrap();
    assert_eq!(path_sublayers(&s), vec![(eid, 1)]);
    assert!(!red_at(&s, 200.0, 180.0) && red_uv(&s, 0.5, 0.9));
    assert!(execute(&mut s, "path.edit", &json!({"effect": eid, "index": 5})).is_err());
    assert!(execute(&mut s, "path.remove", &json!({"effect": 999999, "index": 0})).is_err());
}

#[test]
fn undo_while_editing_a_path_stops_the_edit() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let eid = r["committed"]["effect"].as_u64().unwrap();
    execute(&mut s, "path.edit", &json!({"effect": eid, "index": 0})).unwrap();
    execute(&mut s, "tool.path", &json!({"points": [[140, 250], [200, 230], [260, 250]]})).unwrap();
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(s.tool.path.editing.is_none());
    assert!(!s.path_live().unwrap(), "nothing repaints over the undone state");
    s.update();
    assert!(red_at(&s, 200.0, 180.0));
}

#[test]
fn each_path_and_each_edit_of_one_is_its_own_undo_step() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let steps = |s: &Session| s.history_labels().len();
    let start = steps(&s);
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let eid = r["committed"]["effect"].as_u64().unwrap();
    execute(&mut s, "tool.path", &json!({"uvPoints": [[0.2, 0.9], [0.8, 0.9]], "commit": true})).unwrap();
    assert_eq!(steps(&s), start + 2, "two paths, two steps");
    for y in [250, 230] {
        execute(&mut s, "path.edit", &json!({"effect": eid, "index": 0})).unwrap();
        execute(&mut s, "tool.path", &json!({"points": [[140, y], [200, y - 20], [260, y]]})).unwrap();
        execute(&mut s, "tool.path", &json!({"commit": true})).unwrap();
    }
    assert_eq!(steps(&s), start + 4, "a second edit of the same path doesn't join the first");
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(red_at(&s, 200.0, 230.0), "undo brings back the first edit");
}

/// The pixels of every path sublayer of the selected layer, comparable
fn path_pixels(s: &Session) -> Vec<String> {
    let ids: Vec<u64> = path_sublayers(s).into_iter().map(|(id, _)| id).collect();
    let mut v: Vec<String> = s.doc.pixels.layers.iter().filter(|((id, _), _)| ids.contains(&id.0)).map(|(k, img)| format!("{k:?}{img:?}")).collect();
    v.sort();
    v
}

#[test]
fn placing_points_previews_the_path_with_no_undo_step_until_it_is_applied() {
    let points = json!([[140, 210], [200, 180], [260, 210]]);
    let mut direct = session(primitives::sphere(1.0, 48, 24));
    execute(&mut direct, "tool.path", &json!({"points": points, "commit": true})).unwrap();

    let mut s = session(primitives::sphere(1.0, 48, 24));
    let start = s.history_labels();
    execute(&mut s, "tool.path", &json!({"points": [[140, 280], [260, 280]]})).unwrap();
    assert!(red_at(&s, 200.0, 280.0));
    execute(&mut s, "tool.path", &json!({"points": points})).unwrap();
    assert!(red_at(&s, 200.0, 180.0) && !red_at(&s, 200.0, 280.0), "the preview follows the points");
    assert_eq!(s.history_labels(), start, "no undo step while placing");
    execute(&mut s, "tool.path", &json!({"commit": true})).unwrap();
    assert_eq!(s.history_labels().len(), start.len() + 1, "applying is one undo step");
    assert_eq!(s.undo_label(), Some("Path"));
    assert_eq!(path_pixels(&s), path_pixels(&direct), "the same pixels as applying without a preview");

    // Cancelling removes the preview
    execute(&mut s, "tool.path", &json!({"points": [[140, 260], [200, 250], [260, 260]]})).unwrap();
    assert!(red_at(&s, 200.0, 250.0));
    execute(&mut s, "tool.path", &json!({"clear": true})).unwrap();
    s.update();
    assert!(!red_at(&s, 200.0, 250.0) && red_at(&s, 200.0, 180.0));
    assert_eq!(s.history_labels().len(), start.len() + 1);
}

#[test]
fn undo_while_placing_removes_the_preview_only() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let steps = s.history_labels().len();
    execute(&mut s, "tool.path", &json!({"points": [[140, 260], [200, 250], [260, 260]]})).unwrap();
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(!red_at(&s, 200.0, 250.0), "the preview is gone");
    assert!(red_at(&s, 200.0, 180.0), "the applied path stays");
    assert_eq!(s.history_labels().len(), steps);
    assert!(s.tool.path.points.is_empty() && s.tool.path.editing.is_none());
}

#[test]
fn another_change_while_placing_makes_the_path_its_own_step_first() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let steps = s.history_labels().len();
    execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]]})).unwrap();
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    let labels = s.history_labels();
    assert_eq!(labels.len(), steps + 2);
    assert_eq!(labels.get(steps).map(String::as_str), Some("Path"));
    // Undoing the other change keeps the path; undoing again removes it
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(red_at(&s, 200.0, 180.0));
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    s.update();
    assert!(!red_at(&s, 200.0, 180.0));
}

#[test]
fn editing_another_path_while_placing_applies_the_new_one() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let eid = r["committed"]["effect"].as_u64().unwrap();
    execute(&mut s, "tool.path", &json!({"points": [[140, 260], [200, 250], [260, 260]]})).unwrap();
    execute(&mut s, "path.edit", &json!({"effect": eid, "index": 0})).unwrap();
    assert_eq!(path_sublayers(&s), vec![(eid, 2)], "the new path was applied");
    assert_eq!(s.undo_label(), Some("Path"));
    assert_eq!(s.tool.path.editing, Some((texturecraft_engine::core::LayerId(eid), 0)));
}

#[test]
fn the_next_path_starts_open_after_a_closed_one() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "closed": true, "commit": true})).unwrap();
    assert!(!s.tool.path.closed, "applying a closed path leaves the tool open for the next one");
    execute(&mut s, "tool.path", &json!({"points": [[140, 260], [200, 250], [260, 260]], "closed": true})).unwrap();
    execute(&mut s, "tool.path", &json!({"clear": true})).unwrap();
    assert!(!s.tool.path.closed, "cancelling does too");
}

/// A short path's repaint marks under 1/N of a 1024-texel set's 256 tiles
const SHORT_PATH_TILE_SHARE_DIVISOR: usize = 4;

/// Move the middle point of the Path tool's curve down to viewport row `y` and repaint, returning the (dirty, all) tile counts
fn move_middle_point(s: &mut Session, y: f32) -> (usize, usize) {
    s.tool.path.points[1] = s.pick(Vec2::new(200.0, y), s.viewport).unwrap().pos;
    assert!(s.path_live().unwrap());
    (s.caches[0].dirty.len(), s.caches[0].covered_tiles.len())
}

#[test]
fn moving_a_point_recomposites_only_the_paths_tiles() {
    // 1024 texels: 256 tiles, so a short path covers few of them
    let mut s = Session::new_project("t", primitives::sphere(1.0, 48, 24), "builtin:test", None, 1024).unwrap();
    s.viewport = Vec2::new(400.0, 400.0);
    (s.camera.yaw, s.camera.pitch) = (0.0, 0.0);
    execute(&mut s, "tool.brush", &json!({"size": 6})).unwrap();
    execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]]})).unwrap();
    s.update();

    let (dirty, all) = move_middle_point(&mut s, 190.0);
    assert!(dirty > 0 && dirty * SHORT_PATH_TILE_SHARE_DIVISOR < all, "a new path: {dirty} of {all} tiles to recomposite");
    s.update();

    let r = execute(&mut s, "tool.path", &json!({"commit": true})).unwrap();
    execute(&mut s, "path.edit", &json!({"effect": r["committed"]["effect"], "index": 0})).unwrap();
    s.update();
    // Another row than the applied one, so the edit changes the path
    let (dirty, all) = move_middle_point(&mut s, 200.0);
    assert!(dirty > 0 && dirty * SHORT_PATH_TILE_SHARE_DIVISOR < all, "an applied path being edited: {dirty} of {all} tiles to recomposite");
}

#[test]
fn another_change_ends_the_point_undo_of_an_edited_path() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    let r = execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    let eid = texturecraft_engine::core::LayerId(r["committed"]["effect"].as_u64().unwrap());
    execute(&mut s, "path.edit", &json!({"effect": eid.0, "index": 0})).unwrap();
    s.path_checkpoint();
    s.tool.path.points[1] = s.pick(Vec2::new(200.0, 230.0), s.viewport).unwrap().pos;
    assert!(s.path_live().unwrap());
    let moved: Vec<[f32; 3]> = s.tool.path.points.iter().map(|p| p.to_array()).collect();
    let layers = s.doc.project.layers.len();

    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert_eq!(s.doc.project.layers.len(), layers, "undo removes the new layer");
    assert_eq!(s.paths(eid).unwrap()[0].points, moved, "the path keeps its moved point");
}

#[test]
fn placing_a_path_ends_redo() {
    let mut s = session(primitives::sphere(1.0, 48, 24));
    execute(&mut s, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    execute(&mut s, "edit.undo", &Value::Null).unwrap();
    assert!(s.can_redo());
    execute(&mut s, "tool.path", &json!({"points": [[140, 260], [200, 250], [260, 260]]})).unwrap();
    assert!(!s.can_redo(), "redo would throw the path being placed away");
}

#[test]
fn switching_texture_sets_applies_the_path_being_placed() {
    let mut s = texturecraft_engine::sample_session(256, 64).unwrap();
    execute(&mut s, "layer.add.paint", &Value::Null).unwrap();
    let steps = s.history_labels().len();
    execute(&mut s, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.5, 0.45], [0.8, 0.5]]})).unwrap();

    execute(&mut s, "textureSet.select", &json!({"ts": 1})).unwrap();
    assert_eq!(s.history_labels().len(), steps + 1);
    assert_eq!(s.undo_label(), Some("Path"));
    assert!(s.tool.path.points.is_empty() && s.tool.path.editing.is_none(), "the path no longer paints into the other set");
}

#[test]
fn applying_after_orbit_preserves_preview_pixels_and_storage() {
    let mut session = session(primitives::sphere(1.0, 48, 24));
    execute(&mut session, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]]})).unwrap();
    let preview = session.doc.clone();
    let steps = session.history_labels().len();
    session.camera.yaw += 0.6;
    session.viewport = Vec2::new(600.0, 400.0);
    assert!(!session.path_live().unwrap());

    execute(&mut session, "tool.path", &json!({"commit": true})).unwrap();

    assert_eq!(session.doc.project, preview.project, "apply must keep the preview's camera and viewport");
    assert_eq!(session.doc.pixels.layers, preview.pixels.layers, "apply must keep the preview pixels");
    for (key, image) in &preview.pixels.layers {
        for (x, y) in image.allocated() {
            assert!(std::sync::Arc::ptr_eq(image.tile_handle(x, y).unwrap(), session.doc.pixels.layers[key].tile_handle(x, y).unwrap()));
        }
    }
    assert_eq!(session.history_labels().len(), steps + 1);
}

#[test]
fn applying_an_unchanged_stored_path_after_orbit_adds_no_history() {
    let mut session = session(primitives::sphere(1.0, 48, 24));
    let report = execute(&mut session, "tool.path", &json!({"points": [[140, 210], [200, 180], [260, 210]], "commit": true})).unwrap();
    execute(&mut session, "path.edit", &json!({"effect": report["committed"]["effect"], "index": 0})).unwrap();
    let before = session.doc.clone();
    let history = session.history_labels();
    session.camera.yaw += 0.6;
    assert!(!session.path_live().unwrap());

    execute(&mut session, "tool.path", &json!({"commit": true})).unwrap();

    assert_eq!(session.history_labels(), history, "accepting an unchanged path is not an edit");
    assert_eq!(session.doc.project, before.project);
    assert_eq!(session.doc.pixels.layers, before.pixels.layers);
    assert!(session.tool.path.editing.is_none());
}

#[test]
fn removing_the_final_preview_point_removes_its_pixels() {
    let mut session = session(wall());
    let before = session.doc.clone();
    let history = session.history_labels();
    execute(&mut session, "tool.path", &json!({"uvPoints": [[0.5, 0.5]]})).unwrap();
    assert!(!path_sublayers(&session).is_empty());

    session.path_checkpoint();
    session.tool.path.remove_selected();
    assert!(session.path_live().unwrap());
    session.update();

    assert_eq!(session.doc.project, before.project);
    assert_eq!(session.doc.pixels.layers, before.pixels.layers);
    assert_eq!(session.history_labels(), history);
    assert!(session.tool.path.editing.is_none());
}

#[test]
fn preview_uses_the_commands_explicit_viewport() {
    let mut session = session(primitives::sphere(1.0, 48, 24));
    let params = json!({"points": [[170, 210], [200, 180], [230, 210]], "viewport": [400, 600]});
    execute(&mut session, "tool.path", &params).unwrap();
    let preview = path_pixels(&session);
    let (effect, index) = session.tool.path.editing.unwrap();

    let mut direct = self::session(primitives::sphere(1.0, 48, 24));
    let mut params = params;
    params["commit"] = json!(true);
    execute(&mut direct, "tool.path", &params).unwrap();

    assert_eq!(session.paths(effect).unwrap()[index].viewport, [400.0, 600.0]);
    assert_eq!(preview, path_pixels(&direct));
}

/// An opposite-view placement must preserve the stored curve, painted pixels and point redo
#[test]
fn stored_paths_ignore_point_placement_from_the_other_view() {
    for uv in [false, true] {
        let mut session = session(wall());
        let params =
            if uv { json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": true}) } else { json!({"points": [[140, 210], [260, 210]], "commit": true}) };
        let report = execute(&mut session, "tool.path", &params).unwrap();
        execute(&mut session, "path.edit", &json!({"effect": report["committed"]["effect"], "index": 0})).unwrap();
        let (effect, index) = session.tool.path.editing.unwrap();
        let uv_point = Vec3::new(0.5, 0.6, 0.0);
        let world_point = session.pick(Vec2::new(200.0, 240.0), session.viewport).unwrap().pos;
        assert_eq!(session.append_path_point(if uv { uv_point } else { world_point }, uv), Some(2));
        session.path_live().unwrap();
        let extended = session.tool.path.points.clone();
        session.undo().unwrap();
        let before = session.doc.clone();
        let path = session.tool.path.clone();
        let history = session.history_labels();

        let placed = session.append_path_point(if uv { world_point } else { uv_point }, !uv);
        session.path_live().unwrap();

        assert_eq!(session.paths(effect).unwrap()[index].uv, uv, "the stored path must keep its coordinate space");
        assert_eq!(placed, None);
        assert_eq!(session.tool.path, path);
        assert_eq!(session.doc.project, before.project);
        assert_eq!(session.doc.pixels.layers, before.pixels.layers);
        assert_eq!(session.doc.pixels.masks, before.pixels.masks);
        assert_eq!(session.history_labels(), history);
        assert!(session.can_redo(), "ignored placement must preserve point redo");

        session.redo().unwrap();
        assert_eq!(session.tool.path.points, extended);
    }
}

/// A new, uncommitted curve can still switch views and cancel without changing the document
#[test]
fn draft_paths_can_switch_to_the_other_view() {
    for uv in [false, true] {
        let mut session = session(wall());
        let before = session.doc.clone();
        let history = session.history_labels();
        let params = if uv { json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]]}) } else { json!({"points": [[140, 210], [260, 210]]}) };
        execute(&mut session, "tool.path", &params).unwrap();
        let point = if uv { session.pick(Vec2::new(200.0, 240.0), session.viewport).unwrap().pos } else { Vec3::new(0.5, 0.6, 0.0) };

        assert_eq!(session.append_path_point(point, !uv), Some(0));
        session.path_live().unwrap();

        assert_eq!(session.tool.path.points, vec![point]);
        assert_eq!(session.tool.path.uv, !uv);
        assert_eq!(session.history_labels(), history);
        session.cancel_path();
        assert_eq!(session.doc.project, before.project);
        assert_eq!(session.doc.pixels.layers, before.pixels.layers);
    }
}

/// Headless history commands must restore geometry and pixels before any frame or subsequent edit
#[test]
fn stored_point_edits_undo_and_redo_in_order_without_selection_steps() {
    let mut session = session(wall());
    let report = execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": true})).unwrap();
    let effect = texturecraft_engine::core::LayerId(report["committed"]["effect"].as_u64().unwrap());
    execute(&mut session, "path.edit", &json!({"effect": effect.0, "index": 0})).unwrap();
    let original = session.tool.path.points.clone();
    let original_pixels = session.doc.pixels.clone();
    let history = session.history_labels().len();
    session.insert_path_point(1, Vec3::new(0.5, 0.5, 0.0));
    session.path_live().unwrap();
    let inserted = session.tool.path.points.clone();
    let inserted_pixels = session.doc.pixels.clone();
    session.move_path_point(1, Vec3::new(0.5, 0.6, 0.0), true);
    session.move_path_point(1, Vec3::new(0.5, 0.7, 0.0), false);
    session.path_live().unwrap();
    let moved = session.tool.path.points.clone();
    let moved_pixels = session.doc.pixels.clone();
    session.remove_selected_path_point();
    session.path_live().unwrap();
    let removed = session.tool.path.points.clone();
    let removed_pixels = session.doc.pixels.clone();

    for (expected, pixels) in [(&moved, &moved_pixels), (&inserted, &inserted_pixels), (&original, &original_pixels)] {
        execute(&mut session, "edit.undo", &Value::Null).unwrap();
        assert_eq!(&session.tool.path.points, expected);
        assert_eq!(session.paths(effect).unwrap()[0].points, expected.iter().map(|point| point.to_array()).collect::<Vec<_>>());
        assert_eq!(session.doc.pixels.layers, pixels.layers);
        assert_eq!(session.doc.pixels.masks, pixels.masks);
    }

    session.select_path_point(0);
    assert!(session.can_redo(), "selection must preserve point redo");
    for (expected, pixels) in [(&inserted, &inserted_pixels), (&moved, &moved_pixels), (&removed, &removed_pixels)] {
        execute(&mut session, "edit.redo", &Value::Null).unwrap();
        assert_eq!(&session.tool.path.points, expected);
        assert_eq!(session.paths(effect).unwrap()[0].points, expected.iter().map(|point| point.to_array()).collect::<Vec<_>>());
        assert_eq!(session.doc.pixels.layers, pixels.layers);
        assert_eq!(session.doc.pixels.masks, pixels.masks);
    }
    assert_eq!(session.history_labels().len(), history + 1, "all point edits share one document step");

    execute(&mut session, "edit.undo", &Value::Null).unwrap();
    execute(&mut session, "layer.add.paint", &Value::Null).unwrap();
    execute(&mut session, "edit.undo", &Value::Null).unwrap();
    assert_eq!(session.paths(effect).unwrap()[0].points, moved.iter().map(|point| point.to_array()).collect::<Vec<_>>());
    assert_eq!(session.doc.pixels.layers, moved_pixels.layers, "a subsequent command keeps the undone path pixels");
}

/// A bad path above the edited one must not consume undo or redo when repainting fails
#[test]
fn point_history_repaint_errors_preserve_the_edit_and_can_be_retried() {
    use texturecraft_engine::core::{EffectKind, LayerId};

    let mut session = session(wall());
    let report = execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.5], [0.8, 0.5]], "commit": true})).unwrap();
    let effect = LayerId(report["committed"]["effect"].as_u64().unwrap());
    execute(&mut session, "tool.path", &json!({"uvPoints": [[0.2, 0.8], [0.8, 0.8]], "commit": true})).unwrap();
    let paint = session.paths(effect).unwrap()[1].paint.clone();
    execute(&mut session, "path.edit", &json!({"effect": effect.0, "index": 0})).unwrap();
    let original = session.tool.path.points.clone();
    session.move_path_point(0, Vec3::new(0.2, 0.3, 0.0), true);
    session.path_live().unwrap();
    let moved = session.tool.path.points.clone();

    for (command, expected) in [("edit.undo", &original), ("edit.redo", &moved)] {
        let EffectKind::Path(paths) = &mut session.doc.project.effect_mut(effect).unwrap().kind else { panic!("path sublayer") };
        paths[1].paint = Value::Null;
        let before = session.doc.clone();
        let path = session.tool.path.clone();
        let history = (session.history_labels(), session.redo_labels());

        let error = execute(&mut session, command, &Value::Null).unwrap_err();

        assert!(error.to_string().contains("unreadable path settings"), "{error}");
        assert_eq!(session.tool.path, path);
        assert_eq!(session.doc.project, before.project);
        assert_eq!(session.doc.pixels.layers, before.pixels.layers);
        assert_eq!((session.history_labels(), session.redo_labels()), history);

        let EffectKind::Path(paths) = &mut session.doc.project.effect_mut(effect).unwrap().kind else { panic!("path sublayer") };
        paths[1].paint = paint.clone();
        execute(&mut session, command, &Value::Null).unwrap();
        assert_eq!(&session.tool.path.points, expected, "the failed step remains available");
        assert_eq!(session.paths(effect).unwrap()[0].points, expected.iter().map(|point| point.to_array()).collect::<Vec<_>>());
    }
}

/// Point history repaints an uncommitted preview without creating document undo steps
#[test]
fn draft_point_history_repaints_immediately_and_undoing_all_points_cancels() {
    let mut session = session(wall());
    execute(&mut session, "tool.select", &json!({"tool": "path"})).unwrap();
    let before = session.doc.clone();
    let history = session.history_labels();
    session.append_path_point(Vec3::new(0.2, 0.5, 0.0), true);
    session.path_live().unwrap();
    let one_point = session.doc.clone();
    session.append_path_point(Vec3::new(0.8, 0.5, 0.0), true);
    session.path_live().unwrap();
    let two_points = session.doc.clone();

    for (command, expected) in [("edit.undo", &one_point), ("edit.redo", &two_points), ("edit.undo", &one_point), ("edit.undo", &before)] {
        execute(&mut session, command, &Value::Null).unwrap();

        assert_eq!(session.doc.project, expected.project);
        assert_eq!(session.doc.pixels.layers, expected.pixels.layers);
        assert_eq!(session.history_labels(), history);
    }
    assert!(session.tool.path.points.is_empty());
    assert!(session.tool.path.editing.is_none());
}

#[test]
fn removing_the_final_stored_point_removes_the_path_and_undo_restores_it() {
    let mut session = session(wall());
    let report = execute(&mut session, "tool.path", &json!({"uvPoints": [[0.5, 0.5]], "commit": true})).unwrap();
    let effect = texturecraft_engine::core::LayerId(report["committed"]["effect"].as_u64().unwrap());
    execute(&mut session, "path.edit", &json!({"effect": effect.0, "index": 0})).unwrap();
    let before = session.doc.clone();

    session.remove_selected_path_point();
    assert!(session.path_live().unwrap());

    assert!(session.paths(effect).unwrap().is_empty());
    assert!(session.doc.pixels.layers.keys().all(|(id, _)| *id != effect));
    assert!(session.tool.path.editing.is_none());
    session.undo().unwrap();
    assert_eq!(session.doc.project, before.project);
    assert_eq!(session.doc.pixels.layers, before.pixels.layers);
}
