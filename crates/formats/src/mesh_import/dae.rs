//! COLLADA (.dae): `triangles`, `polylist` and `polygons` with VERTEX / NORMAL / TEXCOORD inputs,
//! materials bound through `instance_geometry` (and skinned `instance_controller`), the visual
//! scene's node transforms (matrix, translate, rotate, scale, lookat), instanced cameras, and the
//! asset's unit and up axis (converted to meters, Y up, like glTF).

use std::collections::HashMap;
use std::rc::Rc;

use quick_xml::events::{BytesStart, Event};
use texturecraft_geom::glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use super::{Corner, PolygonSink, RawCamera, RawMesh, check_vertices};
use crate::{FormatError, Result, malformed};

const KIND: &str = "COLLADA";
/// Deepest element nesting and node hierarchy followed.
const MAX_DEPTH: usize = 256;
/// Elements read from a document.
const MAX_ELEMENTS: usize = 2_000_000;
/// Node instances visited (bounds exponential instancing through `instance_node`).
const MAX_NODE_VISITS: usize = 1_000_000;

#[derive(Default)]
struct El {
    name: String,
    attrs: Vec<(String, String)>,
    text: String,
    children: Vec<El>,
}

impl El {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
    }

    fn child(&self, name: &str) -> Option<&El> {
        self.children.iter().find(|c| c.name == name)
    }

    fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a El> {
        self.children.iter().filter(move |c| c.name == name)
    }

    fn floats(&self) -> Result<Vec<f32>> {
        self.text.split_whitespace().map(|w| w.parse::<f32>().map_err(|_| malformed(KIND, format!("bad number \"{w}\" in <{}>", self.name)))).collect()
    }

    fn indices(&self) -> Result<Vec<u32>> {
        self.text.split_whitespace().map(|w| w.parse::<u32>().map_err(|_| malformed(KIND, format!("bad index \"{w}\" in <{}>", self.name)))).collect()
    }
}

fn element(e: &BytesStart) -> El {
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    let attrs = e.attributes().with_checks(false).flatten().map(|a| (text(a.key.local_name().as_ref()), text(&a.value))).collect();
    El { name: text(e.local_name().as_ref()), attrs, ..El::default() }
}

/// The document as an element tree (the root is an unnamed element holding the top level).
fn parse_xml(bytes: &[u8]) -> Result<El> {
    let mut reader = quick_xml::Reader::from_reader(bytes);
    let mut stack = vec![El::default()];
    let mut count = 0usize;
    loop {
        let event = reader.read_event().map_err(|e| malformed(KIND, e))?;
        if matches!(event, Event::Start(_) | Event::Empty(_)) {
            count += 1;
            if count > MAX_ELEMENTS || stack.len() > MAX_DEPTH {
                return Err(FormatError::TooLarge(format!("the COLLADA document has more than {MAX_ELEMENTS} elements or is nested too deeply")));
            }
        }
        match event {
            Event::Start(e) => stack.push(element(&e)),
            Event::Empty(e) => {
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(element(&e));
                }
            }
            Event::End(_) => {
                let el = stack.pop().filter(|_| !stack.is_empty()).ok_or_else(|| malformed(KIND, "unbalanced end tag"))?;
                if let Some(parent) = stack.last_mut() {
                    parent.children.push(el);
                }
            }
            Event::Text(t) => push_text(&mut stack, &t),
            Event::CData(t) => push_text(&mut stack, &t),
            Event::Eof => break,
            _ => {}
        }
    }
    match (stack.pop(), stack.is_empty()) {
        (Some(root), true) => Ok(root),
        _ => Err(malformed(KIND, "unclosed elements at the end of the document")),
    }
}

fn push_text(stack: &mut [El], t: &[u8]) {
    if let Some(el) = stack.last_mut() {
        el.text.push_str(&String::from_utf8_lossy(t));
        // Whitespace keeps numbers split across text events apart.
        el.text.push(' ');
    }
}

/// A `<source>`'s float array read through its accessor.
struct Source {
    data: Vec<f32>,
    stride: usize,
    offset: usize,
}

impl Source {
    fn get<const W: usize>(&self, i: u32) -> Option<[f32; W]> {
        let start = (i as usize).checked_mul(self.stride)?.checked_add(self.offset)?;
        self.data.get(start..start.checked_add(W)?)?.try_into().ok()
    }
}

struct Doc<'a> {
    ids: HashMap<&'a str, &'a El>,
    sources: HashMap<&'a str, Rc<Source>>,
}

impl<'a> Doc<'a> {
    /// The element a local URL (`#id`) points at.
    fn get(&self, url: Option<&str>) -> Option<&'a El> {
        url.and_then(|u| u.strip_prefix('#')).and_then(|id| self.ids.get(id)).copied()
    }

    fn source(&mut self, url: Option<&str>) -> Result<Rc<Source>> {
        let el = self.get(url).ok_or_else(|| malformed(KIND, format!("missing source {}", url.unwrap_or("(none)"))))?;
        let id = el.attr("id").unwrap_or_default();
        if let Some(s) = self.sources.get(id) {
            return Ok(s.clone());
        }
        let acc = el.child("technique_common").and_then(|t| t.child("accessor"));
        let array = acc.and_then(|a| self.get(a.attr("source"))).or_else(|| el.child("float_array"));
        let array = array.ok_or_else(|| malformed(KIND, format!("source \"{id}\" has no float_array")))?;
        let num = |k: &str, default: usize| acc.and_then(|a| a.attr(k)).and_then(|v| v.parse().ok()).unwrap_or(default);
        let data = array.floats()?;
        check_vertices(data.len() / 3)?;
        let s = Rc::new(Source { data, stride: num("stride", 1).max(1), offset: num("offset", 0) });
        self.sources.insert(id, s.clone());
        Ok(s)
    }
}

pub(crate) fn read(bytes: &[u8]) -> Result<RawMesh> {
    crate::guard(KIND, || read_inner(bytes))
}

fn read_inner(bytes: &[u8]) -> Result<RawMesh> {
    let doc = parse_xml(bytes)?;
    let root = doc.child("COLLADA").ok_or_else(|| malformed(KIND, "not a COLLADA document"))?;
    let mut ids = HashMap::new();
    let mut stack = vec![root];
    while let Some(el) = stack.pop() {
        if let Some(id) = el.attr("id") {
            ids.entry(id).or_insert(el);
        }
        stack.extend(&el.children);
    }
    let mut d = Doc { ids, sources: HashMap::new() };
    let asset = root.child("asset");
    let meter = asset.and_then(|a| a.child("unit")).and_then(|u| u.attr("meter")).and_then(|m| m.trim().parse::<f32>().ok());
    let meter = meter.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(1.0);
    let up = match asset.and_then(|a| a.child("up_axis")).map(|u| u.text.trim()) {
        Some("Z_UP") => Mat4::from_cols(Vec4::X, -Vec4::Z, Vec4::Y, Vec4::W),
        Some("X_UP") => Mat4::from_cols(Vec4::Y, -Vec4::X, Vec4::Z, Vec4::W),
        _ => Mat4::IDENTITY,
    };
    let global = up * Mat4::from_scale(Vec3::splat(meter));
    let mut sink = PolygonSink::default();
    let mut cameras = Vec::new();
    let scene = root.child("scene").and_then(|s| s.child("instance_visual_scene")).and_then(|i| d.get(i.attr("url")));
    let scene = scene.or_else(|| root.child("library_visual_scenes").and_then(|l| l.child("visual_scene")));
    let Some(scene) = scene else {
        // No scene: every geometry as stored.
        for g in root.all("library_geometries").flat_map(|l| l.all("geometry")) {
            add_geometry(&mut d, &mut sink, g, global, &HashMap::new())?;
        }
        return Ok(sink.finish());
    };
    let mut stack: Vec<(&El, Mat4, usize)> = scene.all("node").map(|n| (n, global, 0)).collect();
    stack.reverse();
    let mut visits = 0usize;
    while let Some((node, parent, depth)) = stack.pop() {
        visits += 1;
        if visits > MAX_NODE_VISITS || depth > MAX_DEPTH {
            return Err(FormatError::TooLarge("the COLLADA node hierarchy is too large, too deep or cyclic".into()));
        }
        let m = parent * node_transform(node)?;
        sink.part = node.attr("name").or_else(|| node.attr("id")).unwrap_or_default().to_string();
        for inst in &node.children {
            if inst.name == "instance_camera" {
                cameras.extend(d.get(inst.attr("url")).and_then(|c| camera(c, m, &sink.part)));
                continue;
            }
            let (geom, bind_shape) = match inst.name.as_str() {
                "instance_geometry" => (d.get(inst.attr("url")), Mat4::IDENTITY),
                "instance_controller" => {
                    let skin = d.get(inst.attr("url")).and_then(|c| c.child("skin"));
                    let bind = skin.and_then(|s| s.child("bind_shape_matrix")).map(El::floats).transpose()?.and_then(|v| matrix(&v));
                    (skin.and_then(|s| d.get(s.attr("source"))), bind.unwrap_or(Mat4::IDENTITY))
                }
                _ => continue,
            };
            let Some(geom) = geom else {
                sink.warn("an instanced geometry is missing from the file".into());
                continue;
            };
            let bind = bindings(&d, inst);
            add_geometry(&mut d, &mut sink, geom, m * bind_shape, &bind)?;
        }
        let children = node.all("node").chain(node.all("instance_node").filter_map(|i| d.get(i.attr("url"))));
        let children: Vec<&El> = children.collect();
        stack.extend(children.into_iter().rev().map(|c| (c, m, depth + 1)));
    }
    let mut raw = sink.finish();
    raw.cameras = cameras;
    Ok(raw)
}

/// A `<camera>` instanced by node `name` at `m`: COLLADA cameras look down -Z with +Y up.
fn camera(cam: &El, m: Mat4, name: &str) -> Option<RawCamera> {
    let tc = cam.child("optics")?.child("technique_common")?;
    let num = |e: &El, k: &str| e.child(k).and_then(|v| v.text.trim().parse::<f32>().ok()).filter(|v| v.is_finite() && *v > 0.0);
    let name = if name.is_empty() { cam.attr("name").or_else(|| cam.attr("id")).unwrap_or_default() } else { name };
    if let Some(p) = tc.child("perspective") {
        let x_to_y = |x: f32| Some(2.0 * ((x * 0.5).to_radians().tan() / num(p, "aspect_ratio")?).atan().to_degrees());
        let fov = num(p, "yfov").or_else(|| num(p, "xfov").and_then(x_to_y)).or_else(|| num(p, "xfov")).unwrap_or(40.0);
        return Some(RawCamera::placed(name.into(), m, Vec3::NEG_Z, fov, None));
    }
    let o = tc.child("orthographic")?;
    let half = num(o, "ymag").or_else(|| Some(num(o, "xmag")? / num(o, "aspect_ratio")?)).or_else(|| num(o, "xmag")).unwrap_or(1.0);
    Some(RawCamera::placed(name.into(), m, Vec3::NEG_Z, 40.0, Some(half)))
}

/// Row-major 4×4 floats to a matrix.
fn matrix(v: &[f32]) -> Option<Mat4> {
    let a: [f32; 16] = v.get(..16)?.try_into().ok()?;
    Some(Mat4::from_cols_array(&a).transpose())
}

/// A node's transform elements, applied in document order.
fn node_transform(node: &El) -> Result<Mat4> {
    let mut m = Mat4::IDENTITY;
    for t in &node.children {
        let v = match t.name.as_str() {
            "matrix" | "translate" | "rotate" | "scale" | "lookat" => t.floats()?,
            _ => continue,
        };
        let step = match (t.name.as_str(), v.as_slice()) {
            ("matrix", _) => matrix(&v),
            ("translate", [x, y, z, ..]) => Some(Mat4::from_translation(Vec3::new(*x, *y, *z))),
            ("scale", [x, y, z, ..]) => Some(Mat4::from_scale(Vec3::new(*x, *y, *z))),
            ("rotate", [x, y, z, deg, ..]) => Vec3::new(*x, *y, *z).try_normalize().map(|axis| Mat4::from_axis_angle(axis, deg.to_radians())),
            // Eye, interest point and up: the node sits at the eye looking at the interest.
            ("lookat", [ex, ey, ez, ix, iy, iz, ux, uy, uz, ..]) => {
                Some(Mat4::look_at_rh(Vec3::new(*ex, *ey, *ez), Vec3::new(*ix, *iy, *iz), Vec3::new(*ux, *uy, *uz)).inverse()).filter(|m| m.is_finite())
            }
            _ => None,
        };
        m *= step.unwrap_or(Mat4::IDENTITY);
    }
    Ok(m)
}

/// Material symbol → material name, from an instance's `bind_material`.
fn bindings(d: &Doc, inst: &El) -> HashMap<String, String> {
    let list = inst.child("bind_material").and_then(|b| b.child("technique_common"));
    list.into_iter()
        .flat_map(|t| t.all("instance_material"))
        .filter_map(|im| {
            let symbol = im.attr("symbol")?;
            let mat = d.get(im.attr("target"));
            let name = mat.and_then(|m| m.attr("name").or_else(|| m.attr("id"))).unwrap_or(symbol);
            Some((symbol.to_string(), name.to_string()))
        })
        .collect()
}

/// An input's index offset and source.
struct Input {
    offset: usize,
    source: Rc<Source>,
}

fn add_geometry(d: &mut Doc, sink: &mut PolygonSink, geom: &El, m: Mat4, bind: &HashMap<String, String>) -> Result<()> {
    let Some(mesh) = geom.child("mesh") else { return Ok(()) };
    let nm = Mat3::from_mat4(m).inverse().transpose();
    let flip = m.determinant() < 0.0;
    for prim in &mesh.children {
        if !matches!(prim.name.as_str(), "triangles" | "polylist" | "polygons") {
            if matches!(prim.name.as_str(), "lines" | "linestrips" | "trifans" | "tristrips") {
                sink.warn(format!("skipped COLLADA <{}> (only triangles and polygons are supported)", prim.name));
            }
            continue;
        }
        let (mut position, mut normal, mut uv): (Option<Input>, Option<Input>, Option<(u32, Input)>) = (None, None, None);
        let mut stride = 1usize;
        for input in prim.all("input") {
            let offset: usize = input.attr("offset").and_then(|o| o.parse().ok()).unwrap_or(0);
            stride = stride.max(offset.saturating_add(1));
            // VERTEX names a <vertices>, whose own inputs share the VERTEX index.
            let nested = match input.attr("semantic") {
                Some("VERTEX") => d.get(input.attr("source")).map(|v| v.all("input").collect()).unwrap_or_default(),
                _ => vec![input],
            };
            for i in nested {
                let set: u32 = input.attr("set").or_else(|| i.attr("set")).and_then(|s| s.parse().ok()).unwrap_or(0);
                match i.attr("semantic") {
                    Some("POSITION") => position = Some(Input { offset, source: d.source(i.attr("source"))? }),
                    Some("NORMAL") if normal.is_none() => normal = Some(Input { offset, source: d.source(i.attr("source"))? }),
                    Some("TEXCOORD") if uv.as_ref().is_none_or(|(s, _)| set < *s) => uv = Some((set, Input { offset, source: d.source(i.attr("source"))? })),
                    _ => {}
                }
            }
        }
        let Some(position) = position else {
            sink.warn("skipped COLLADA primitives without positions".into());
            continue;
        };
        let material = prim.attr("material").map(|s| bind.get(s).map_or(s, String::as_str)).unwrap_or("Default");
        // Each <p> with the corner count of every polygon in it.
        let lists: Vec<(Vec<u32>, Vec<usize>)> = match prim.name.as_str() {
            "polygons" => prim
                .all("p")
                .map(|p| {
                    let p = p.indices()?;
                    let n = p.len() / stride;
                    Ok((p, vec![n]))
                })
                .collect::<Result<_>>()?,
            _ => {
                let p = prim.child("p").map(El::indices).transpose()?.unwrap_or_default();
                let sizes = match prim.child("vcount") {
                    Some(v) if prim.name == "polylist" => v.indices()?.into_iter().map(|c| c as usize).collect(),
                    _ => vec![3; p.len() / stride / 3],
                };
                vec![(p, sizes)]
            }
        };
        let mut corners = Vec::new();
        for (p, sizes) in &lists {
            let mut next = 0usize;
            for &size in sizes {
                corners.clear();
                for c in next..next.saturating_add(size) {
                    let idx = |input: &Input| c.checked_mul(stride).and_then(|b| b.checked_add(input.offset)).and_then(|k| p.get(k)).copied();
                    let at = idx(&position).and_then(|i| position.source.get::<3>(i));
                    let pos = at.ok_or_else(|| malformed(KIND, "polygon index out of range"))?;
                    let n = normal.as_ref().and_then(|n| idx(n).and_then(|i| n.source.get::<3>(i)));
                    let t = uv.as_ref().and_then(|(_, t)| idx(t).and_then(|i| t.source.get::<2>(i)));
                    corners.push(Corner {
                        p: m.transform_point3(Vec3::from(pos)),
                        n: n.map(|n| nm * Vec3::from(n)),
                        uv: t.map(|[u, v]| Vec2::new(u, 1.0 - v)),
                        color: None,
                    });
                }
                next = next.saturating_add(size);
                if flip {
                    corners.reverse();
                }
                sink.polygon(material, &corners)?;
            }
        }
    }
    Ok(())
}
