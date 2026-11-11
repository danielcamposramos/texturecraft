//! USD: `.usda` (text), `.usdc` (Crate binary), `.usd` (either, by magic) and `.usdz`
//! (zip package; its first layer is read). Both encodings parse into one [`Layer`]; the stage is
//! then composed (internal references, prim hierarchy, xform ops) and every `Mesh` prim becomes
//! polygons in world space: points, faceVertexCounts/Indices (holes skipped), normals
//! (`primvars:normals` or `normals`), UVs (`primvars:st`, `st0`, `UVMap` or the first
//! `texCoord2f[]` primvar, indexed or not, any interpolation), `orientation`, and material names
//! from `material:binding` (inherited down the hierarchy) and `materialBind` GeomSubsets. Stage
//! `metersPerUnit` (USD's fallback is 0.01) and `upAxis` become meters, Y up. `Camera` prims
//! (focalLength, verticalAperture, projection) become the file's cameras. A material's
//! UsdPreviewSurface gives its base color: the `diffuseColor` value, or the UsdUVTexture it is
//! connected to when the texture file is inside the `.usdz` package (for color map bakes). A loose
//! layer is packaged with its texture files first ([`package`]).
//!
//! Not composed: external references, payloads, sublayers and variant sets (each warns once),
//! inherits and specializes. Time-sampled values use their first sample.

pub(crate) mod codec;
#[cfg(test)]
mod tests;
mod usda;
mod usdc;

use std::collections::HashMap;

use texturecraft_geom::glam::{DMat4, DQuat, DVec3, DVec4, Mat3, Mat4, Vec2, Vec3, Vec4};

use super::external::below;
use super::{Corner, MAX_MESH_FILE_BYTES, MAX_TEXTURE_BYTES, MAX_ZIP_ENTRIES, PolygonSink, RawBaseColor, RawCamera, RawMesh, lens_fov, zip_entry};
use crate::{FormatError, Result, malformed};

pub(crate) const KIND: &str = "USD";
/// Memory budget for decoded values (numbers are held as f64).
const MAX_DECODED_BYTES: usize = 1 << 31;
/// Deepest prim nesting and reference chain followed.
const MAX_DEPTH: usize = 128;
/// Composed prims visited (bounds exponential instancing through references).
const MAX_PRIM_VISITS: usize = 1_000_000;

/// A decoded value: numbers (scalars, vectors, quaternions as w x y z, matrices row-major and
/// bools all flattened), tokens / strings / asset paths, or paths.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Val {
    Num(Vec<f64>),
    Tok(Vec<String>),
    Path(Vec<String>),
}

#[derive(Clone, Debug, Default)]
pub(crate) struct Prop {
    /// Declared type (`point3f[]`, `texCoord2f[]`, `rel`, ...).
    pub type_name: String,
    pub value: Option<Val>,
    pub interpolation: Option<String>,
    /// The first connection source (`name.connect`): an attribute path.
    pub connect: Option<String>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub(crate) enum Specifier {
    #[default]
    Def,
    Over,
    Class,
}

/// One prim spec of a layer.
#[derive(Debug, Default)]
pub(crate) struct PrimSpec {
    pub name: String,
    pub path: String,
    pub type_name: String,
    pub specifier: Specifier,
    pub active: Option<bool>,
    pub props: HashMap<String, Prop>,
    pub children: Vec<usize>,
    /// Internal references: absolute prim paths in this layer.
    pub references: Vec<String>,
}

/// A parsed layer: prim specs (index 0 is the pseudo-root) and stage metadata.
#[derive(Debug)]
pub(crate) struct Layer {
    pub prims: Vec<PrimSpec>,
    pub meters_per_unit: Option<f64>,
    pub up_axis: Option<String>,
    pub warnings: Vec<String>,
}

impl Layer {
    pub fn new() -> Layer {
        Layer { prims: vec![PrimSpec { path: "/".into(), ..PrimSpec::default() }], meters_per_unit: None, up_axis: None, warnings: Vec::new() }
    }

    /// Add a prim spec named `name` under prim `parent`.
    pub fn add_prim(&mut self, parent: usize, name: &str) -> usize {
        let path = match self.prims.get(parent) {
            Some(p) if p.path == "/" => format!("/{name}"),
            Some(p) => format!("{}/{name}", p.path),
            None => format!("/{name}"),
        };
        let id = self.prims.len();
        self.prims.push(PrimSpec { name: name.into(), path, ..PrimSpec::default() });
        if let Some(p) = self.prims.get_mut(parent) {
            p.children.push(id);
        }
        id
    }

    pub fn warn(&mut self, msg: &str) {
        if !self.warnings.iter().any(|w| w == msg) {
            self.warnings.push(msg.into());
        }
    }
}

/// Charges decoded data against [`MAX_DECODED_BYTES`].
pub(crate) struct Budget(usize);

impl Budget {
    pub fn new() -> Budget {
        Budget(MAX_DECODED_BYTES)
    }

    pub fn take(&mut self, bytes: usize) -> Result<()> {
        self.0 =
            self.0.checked_sub(bytes).ok_or_else(|| FormatError::TooLarge(format!("the USD file decodes to more than {} MiB", MAX_DECODED_BYTES >> 20)))?;
        Ok(())
    }
}

pub(crate) fn read(file_name: &str, bytes: &[u8]) -> Result<RawMesh> {
    crate::guard(KIND, || {
        if super::extension(file_name) == "usdz" {
            let (layer, mut package) = read_usdz(bytes)?;
            compose(&layer, Some(&mut package as &mut dyn Assets))
        } else {
            compose(&read_layer(bytes)?, None)
        }
    })
}

/// A `.usda` / `.usdc` / `.usd` layer, by content.
fn read_layer(bytes: &[u8]) -> Result<Layer> {
    let mut budget = Budget::new();
    if bytes.starts_with(codec::MAGIC) {
        usdc::read(bytes, &mut budget)
    } else if bytes.trim_ascii_start().starts_with(b"#usda") {
        usda::read(bytes, &mut budget)
    } else {
        Err(malformed(KIND, "neither a USD text layer (#usda) nor a Crate binary (PXR-USDC)"))
    }
}

/// A `.usdz` package: its files, and the folder of the layer read (asset paths are relative
/// to it).
struct Package<'a> {
    zip: zip::ZipArchive<std::io::Cursor<&'a [u8]>>,
    dir: String,
}

impl Package<'_> {
    /// Entry `name`'s bytes, at most `cap` (None when missing, larger or unreadable).
    fn read(&mut self, name: &str, cap: u64) -> Option<Vec<u8>> {
        zip_entry(&mut self.zip, name, cap)
    }
}

/// Where a stage's texture files come from.
trait Assets {
    /// The file asset path `asset` names (relative to the layer's folder), at most
    /// [`MAX_TEXTURE_BYTES`].
    fn asset(&mut self, asset: &str) -> Option<Vec<u8>>;
}

impl Assets for Package<'_> {
    fn asset(&mut self, asset: &str) -> Option<Vec<u8>> {
        let rel = asset.trim_start_matches("./").trim_start_matches('/');
        let path = if self.dir.is_empty() { rel.to_string() } else { format!("{}/{rel}", self.dir) };
        self.read(&path, MAX_TEXTURE_BYTES)
            .or_else(|| self.read(rel, MAX_TEXTURE_BYTES))
            .or_else(|| self.read(&below(&format!("{}/{asset}", self.dir))?, MAX_TEXTURE_BYTES))
    }
}

/// The texture files next to a loose layer, kept under their paths below its folder.
struct Loose<'f> {
    found: &'f mut dyn FnMut(&str) -> Option<Vec<u8>>,
    files: Vec<(String, Vec<u8>)>,
    /// Bytes of the texture files kept so far, plus the layer.
    total: u64,
    max_bytes: u64,
}

impl Assets for Loose<'_> {
    fn asset(&mut self, asset: &str) -> Option<Vec<u8>> {
        // Once over budget, later materials must not trigger more file reads or clones
        if self.total > self.max_bytes {
            return None;
        }

        let name = below(asset)?;
        if let Some((_, data)) = self.files.iter().find(|(stored, _)| *stored == name) {
            return Some(data.clone());
        }

        let data = (self.found)(asset).filter(|d| d.len() as u64 <= MAX_TEXTURE_BYTES)?;
        self.total = self.total.saturating_add(data.len() as u64);
        if self.total > self.max_bytes {
            return None;
        }

        self.files.push((name, data.clone()));
        Some(data)
    }
}

/// A loose layer whose materials use texture files, as a `.usdz` package holding it (first) and
/// those files under their paths below its folder; None when `found` has none of them.
pub(crate) fn package(file_name: &str, bytes: &[u8], found: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Result<Option<Vec<u8>>> {
    package_with_limit(file_name, bytes, found, MAX_MESH_FILE_BYTES)
}

fn package_with_limit(file_name: &str, bytes: &[u8], found: &mut dyn FnMut(&str) -> Option<Vec<u8>>, max_bytes: u64) -> Result<Option<Vec<u8>>> {
    crate::guard(KIND, || {
        let layer = read_layer(bytes)?;
        let mut loose = Loose { found, files: Vec::new(), total: bytes.len() as u64, max_bytes };
        let mut stage = Stage::new(&layer);
        for p in layer.prims.iter().filter(|p| p.type_name == "Material") {
            stage.base_color(&p.name, Some(&mut loose));
        }
        if loose.total > max_bytes {
            return Err(FormatError::TooLarge("the USD layer and its textures exceed 2 GiB".into()));
        }

        if loose.files.is_empty() {
            return Ok(None);
        }

        let leaf = std::path::Path::new(file_name).file_name().and_then(|n| n.to_str()).unwrap_or("layer.usd");
        crate::mesh_export::usd::usdz(leaf, bytes, &loose.files).map(Some)
    })
}

/// The first USD layer of a `.usdz` package, and the package.
fn read_usdz(bytes: &[u8]) -> Result<(Layer, Package<'_>)> {
    let zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| malformed(KIND, format!("usdz package: {e}")))?;
    if zip.len() > MAX_ZIP_ENTRIES {
        return Err(FormatError::TooLarge(format!("the usdz package has more than {MAX_ZIP_ENTRIES} entries")));
    }
    let name = zip.file_names().flatten().filter(|n| matches!(super::extension(n).as_str(), "usda" | "usdc" | "usd")).min_by_key(|n| zip.index_for_name(n));
    let name = name.ok_or_else(|| malformed(KIND, "the usdz package contains no USD layer"))?.into_owned();
    let dir = name.rsplit_once('/').map_or("", |(d, _)| d).to_string();
    let mut package = Package { zip, dir };
    let data = package.read(&name, MAX_MESH_FILE_BYTES).ok_or_else(|| FormatError::TooLarge("the usdz layer is unreadable or larger than 2 GiB".into()))?;
    Ok((read_layer(&data)?, package))
}

/// Composes the layer's stage into polygons.
struct Stage<'a> {
    layer: &'a Layer,
    by_path: HashMap<&'a str, usize>,
    sink: PolygonSink,
    cameras: Vec<RawCamera>,
    /// Up axis and units: stage space to meters, Y up.
    global: Mat4,
    visits: usize,
}

/// A composed prim: its specs, strongest first (the prim, then what it references).
type Composed = Vec<usize>;

fn compose(layer: &Layer, mut assets: Option<&mut (dyn Assets + '_)>) -> Result<RawMesh> {
    let mut stage = Stage::new(layer);
    for w in &layer.warnings {
        stage.sink.warn(w.clone());
    }
    stage.visit(&[0], stage.global, None, 0)?;
    let mut raw = stage.sink.finish();
    raw.cameras = std::mem::take(&mut stage.cameras);
    let mut names: Vec<String> = raw.groups.iter().map(|(m, _, _)| m.clone()).collect();
    names.sort();
    names.dedup();
    let mut stage = Stage { sink: PolygonSink::default(), ..stage };
    raw.base_colors = names.iter().filter_map(|n| stage.base_color(n, assets.as_deref_mut())).collect();
    Ok(raw)
}

/// Last element of a prim path: the material name of a binding.
fn leaf(path: &str) -> &str {
    path.rsplit('/').next().unwrap_or(path)
}

impl<'a> Stage<'a> {
    fn new(layer: &'a Layer) -> Stage<'a> {
        let mpu = layer.meters_per_unit.filter(|m| m.is_finite() && *m > 0.0).unwrap_or(0.01) as f32;
        let up = match layer.up_axis.as_deref() {
            Some("Z" | "z") => Mat4::from_cols(Vec4::X, -Vec4::Z, Vec4::Y, Vec4::W),
            _ => Mat4::IDENTITY,
        };
        let by_path = layer.prims.iter().enumerate().map(|(i, p)| (p.path.as_str(), i)).collect();
        Stage { layer, by_path, sink: PolygonSink::default(), cameras: Vec::new(), global: up * Mat4::from_scale(Vec3::splat(mpu)), visits: 0 }
    }

    fn spec(&self, i: usize) -> Option<&'a PrimSpec> {
        self.layer.prims.get(i)
    }

    fn specs<'s>(&'s self, c: &'s [usize]) -> impl Iterator<Item = &'a PrimSpec> + 's {
        c.iter().filter_map(|&i| self.spec(i))
    }

    /// `specs` and, recursively, the prims their internal references name.
    fn expand(&mut self, specs: &[usize], out: &mut Composed, depth: usize) {
        for &s in specs {
            if out.contains(&s) || depth > MAX_DEPTH {
                continue;
            }
            out.push(s);
            let Some(spec) = self.spec(s) else { continue };
            let targets: Vec<usize> = spec
                .references
                .iter()
                .filter_map(|r| {
                    let t = self.by_path.get(r.as_str()).copied();
                    if t.is_none() {
                        self.sink.warn(format!("reference to missing prim {r} was skipped"));
                    }
                    t
                })
                .collect();
            self.expand(&targets, out, depth + 1);
        }
    }

    fn prop(&self, c: &[usize], name: &str) -> Option<&'a Prop> {
        self.specs(c).filter_map(|s| s.props.get(name)).find(|p| p.value.is_some())
    }

    fn nums(&self, c: &[usize], name: &str) -> Option<&'a [f64]> {
        match self.prop(c, name)?.value.as_ref()? {
            Val::Num(v) => Some(v),
            _ => None,
        }
    }

    /// The prim path an attribute is connected to (`/Looks/Red/Tex.outputs:rgb` → `/Looks/Red/Tex`).
    fn connected_prim(&mut self, c: &[usize], name: &str) -> Option<Composed> {
        let source = self.specs(c).filter_map(|s| s.props.get(name)).find_map(|p| p.connect.as_deref())?;
        self.prim_at(source.split('.').next().unwrap_or(source))
    }

    /// The composed prim at `path`.
    fn prim_at(&mut self, path: &str) -> Option<Composed> {
        let i = *self.by_path.get(path)?;
        let mut out = Vec::new();
        self.expand(&[i], &mut out, 0);
        Some(out)
    }

    /// Base color of the Material prim named `name` (bindings name materials by their last path
    /// element): its UsdPreviewSurface's `diffuseColor` value, or the texture of the UsdUVTexture
    /// connected to it when `assets` holds the file (white × the texture's `scale`).
    fn base_color(&mut self, name: &str, assets: Option<&mut (dyn Assets + '_)>) -> Option<RawBaseColor> {
        let path = self.layer.prims.iter().find(|p| p.type_name == "Material" && p.name == name)?.path.clone();
        let material = self.prim_at(&path)?;
        let surface = self.connected_prim(&material, "outputs:surface").filter(|s| self.token(s, "info:id") == Some("UsdPreviewSurface"))?;
        let rgb = |v: &[f64]| match v {
            [r, g, b, ..] => Some(Vec4::new(*r as f32, *g as f32, *b as f32, 1.0)).filter(|c| c.is_finite()),
            _ => None,
        };
        let texture = self.connected_prim(&surface, "inputs:diffuseColor").filter(|t| self.token(t, "info:id") == Some("UsdUVTexture"));
        let image = texture.as_ref().and_then(|t| assets?.asset(self.token(t, "inputs:file")?));
        let factor = match (&image, texture) {
            (Some(_), Some(t)) => self.nums(&t, "inputs:scale").and_then(rgb).unwrap_or(Vec4::ONE),
            _ => self.nums(&surface, "inputs:diffuseColor").and_then(rgb).unwrap_or(Vec4::ONE),
        };
        (image.is_some() || factor != Vec4::ONE).then(|| RawBaseColor { material: name.to_string(), factor, image })
    }

    fn token(&self, c: &[usize], name: &str) -> Option<&'a str> {
        match self.prop(c, name)?.value.as_ref()? {
            Val::Tok(v) => v.first().map(String::as_str),
            _ => None,
        }
    }

    fn type_name(&self, c: &[usize]) -> &'a str {
        self.specs(c).map(|s| s.type_name.as_str()).find(|t| !t.is_empty()).unwrap_or_default()
    }

    /// Material bound to the prim itself (not inherited).
    fn binding(&self, c: &[usize]) -> Option<String> {
        ["material:binding", "material:binding:full", "material:binding:preview"].iter().find_map(|name| match self.prop(c, name)?.value.as_ref()? {
            Val::Path(p) => p.first().map(|p| leaf(p).to_string()),
            _ => None,
        })
    }

    /// Composed children in namespace order, each with its references expanded.
    fn children(&mut self, c: &[usize], depth: usize) -> Vec<Composed> {
        let mut order: Vec<Vec<usize>> = Vec::new();
        let mut at: HashMap<&str, usize> = HashMap::new();
        for spec in self.specs(c) {
            for &k in &spec.children {
                let Some(child) = self.spec(k) else { continue };
                match at.get(child.name.as_str()) {
                    Some(&i) => order.get_mut(i).into_iter().for_each(|o| o.push(k)),
                    None => {
                        at.insert(&child.name, order.len());
                        order.push(vec![k]);
                    }
                }
            }
        }
        order
            .into_iter()
            .map(|specs| {
                let mut out = Vec::new();
                self.expand(&specs, &mut out, depth);
                out
            })
            .collect()
    }

    /// Prims that are defined, active, visible and not guides.
    fn shown(&self, c: &[usize]) -> bool {
        let first = c.first().and_then(|&i| self.spec(i));
        let defined = self.specs(c).any(|s| s.specifier == Specifier::Def);
        let active = self.specs(c).find_map(|s| s.active).unwrap_or(true);
        defined
            && active
            && first.is_none_or(|s| s.specifier != Specifier::Class)
            && self.token(c, "visibility") != Some("invisible")
            && self.token(c, "purpose") != Some("guide")
    }

    fn visit(&mut self, c: &[usize], parent: Mat4, binding: Option<String>, depth: usize) -> Result<()> {
        self.visits += 1;
        if self.visits > MAX_PRIM_VISITS || depth > MAX_DEPTH {
            return Err(FormatError::TooLarge("the USD prim hierarchy is too large, too deep or instanced too often".into()));
        }
        let (local, reset) = self.xform(c);
        let world = if reset { self.global } else { parent } * local;
        let binding = self.binding(c).or(binding);
        match self.type_name(c) {
            "Mesh" => self.mesh(c, world, binding.as_deref(), depth)?,
            "Camera" => self.camera(c, world),
            "PointInstancer" => self.sink.warn("PointInstancer prims were skipped".into()),
            _ => {}
        }
        for child in self.children(c, depth + 1) {
            if self.shown(&child) {
                self.visit(&child, world, binding.clone(), depth + 1)?;
            }
        }
        Ok(())
    }

    /// A camera prim (USD's lens fallbacks: 50 mm over 15.2908 mm; apertures are in tenths of
    /// a scene unit, which orthographic views measure).
    fn camera(&mut self, c: &[usize], world: Mat4) {
        let name = self.spec(c.first().copied().unwrap_or_default()).map(|s| s.name.clone()).unwrap_or_default();
        let num = |k: &str, fallback: f32| self.nums(c, k).and_then(|v| v.first()).map(|v| *v as f32).filter(|v| v.is_finite() && *v > 0.0).unwrap_or(fallback);
        let (focal, aperture) = (num("focalLength", 50.0), num("verticalAperture", 15.2908));
        let ortho = (self.token(c, "projection") == Some("orthographic")).then_some(aperture * 0.05);
        self.cameras.push(RawCamera::placed(name, world, Vec3::NEG_Z, lens_fov(focal, aperture), ortho));
    }

    /// The prim's local transform from `xformOpOrder`, and whether it resets the parent's.
    fn xform(&mut self, c: &[usize]) -> (Mat4, bool) {
        let order: Vec<String> = match self.prop(c, "xformOpOrder").and_then(|p| p.value.as_ref()) {
            Some(Val::Tok(t)) => t.clone(),
            _ => return (Mat4::IDENTITY, false),
        };
        let (mut m, mut reset) = (DMat4::IDENTITY, false);
        for op in &order {
            if op == "!resetXformStack!" {
                (m, reset) = (DMat4::IDENTITY, true);
                continue;
            }
            let (invert, name) = match op.strip_prefix("!invert!") {
                Some(n) => (true, n),
                None => (false, op.as_str()),
            };
            let kind = name.split(':').nth(1).unwrap_or_default();
            let step = self.nums(c, name).and_then(|v| op_matrix(kind, v));
            let Some(mut step) = step else {
                self.sink.warn(format!("xform op \"{name}\" is missing or not supported and was ignored"));
                continue;
            };
            if invert {
                step = if step.determinant().abs() > 1e-300 { step.inverse() } else { DMat4::IDENTITY };
            }
            m *= step;
        }
        let m = m.as_mat4();
        (if m.is_finite() { m } else { Mat4::IDENTITY }, reset)
    }

    /// Material per face from `materialBind` GeomSubsets: (names, index per face or MAX).
    fn subsets(&mut self, c: &[usize], faces: usize, depth: usize) -> Option<(Vec<String>, Vec<u32>)> {
        let mut names: Vec<String> = Vec::new();
        let mut per_face: Vec<u32> = Vec::new();
        for child in self.children(c, depth + 1) {
            if self.type_name(&child) != "GeomSubset" || !self.shown(&child) {
                continue;
            }
            let family_ok = self.token(&child, "familyName").is_none_or(|f| f == "materialBind");
            let faces_ok = self.token(&child, "elementType").is_none_or(|e| e == "face");
            let (Some(material), true) = (self.binding(&child), family_ok && faces_ok) else { continue };
            let id = names.iter().position(|n| *n == material).unwrap_or(names.len());
            if id == names.len() {
                names.push(material);
            }
            if per_face.is_empty() {
                per_face = vec![u32::MAX; faces];
            }
            for &f in self.nums(&child, "indices").unwrap_or_default() {
                if let Some(slot) = index(f).and_then(|f| per_face.get_mut(f)) {
                    *slot = id as u32;
                }
            }
        }
        (!names.is_empty()).then_some((names, per_face))
    }

    fn primvar(&self, c: &[usize], name: &str, width: usize, counts: (usize, usize, usize)) -> Option<Primvar<'a>> {
        let values = self.nums(c, name)?;
        let indices = self.nums(c, &format!("{name}:indices"));
        let n = indices.map_or(values.len() / width, <[f64]>::len);
        let (points, corners, faces) = counts;
        let interp = match self.prop(c, name).and_then(|p| p.interpolation.as_deref()) {
            Some("constant") => Interp::Constant,
            Some("uniform") => Interp::Uniform,
            Some("vertex" | "varying") => Interp::Vertex,
            Some("faceVarying") => Interp::FaceVarying,
            _ if n == corners => Interp::FaceVarying,
            _ if n == points => Interp::Vertex,
            _ if n == faces => Interp::Uniform,
            _ => return None,
        };
        Some(Primvar { values, indices, interp })
    }

    fn uv_primvar(&self, c: &[usize], counts: (usize, usize, usize)) -> Option<Primvar<'a>> {
        let named = ["primvars:st", "primvars:st0", "primvars:UVMap", "primvars:uv"].into_iter().find_map(|n| self.primvar(c, n, 2, counts));
        named.or_else(|| {
            let mut names: Vec<&str> = self
                .specs(c)
                .flat_map(|s| s.props.iter())
                .filter(|(k, p)| k.starts_with("primvars:") && p.type_name.starts_with("texCoord2"))
                .map(|(k, _)| k.as_str())
                .collect();
            names.sort_unstable();
            names.into_iter().find_map(|n| self.primvar(c, n, 2, counts))
        })
    }

    fn mesh(&mut self, c: &[usize], m: Mat4, binding: Option<&str>, depth: usize) -> Result<()> {
        let name = self.spec(c.first().copied().unwrap_or_default()).map(|s| s.name.clone()).unwrap_or_default();
        let (Some(points), Some(counts), Some(indices)) = (self.nums(c, "points"), self.nums(c, "faceVertexCounts"), self.nums(c, "faceVertexIndices")) else {
            self.sink.warn(format!("mesh \"{name}\" has no points or faces and was skipped"));
            return Ok(());
        };
        let total: f64 = counts.iter().map(|n| n.max(0.0)).sum();
        if total > indices.len() as f64 {
            self.sink.warn(format!("mesh \"{name}\" has fewer face vertex indices than its face sizes need and was skipped"));
            return Ok(());
        }
        let sizes = (points.len() / 3, indices.len(), counts.len());
        let normals = self.primvar(c, "primvars:normals", 3, sizes).or_else(|| self.primvar(c, "normals", 3, sizes));
        let uvs = self.uv_primvar(c, sizes);
        let holes: std::collections::HashSet<usize> = self.nums(c, "holeIndices").unwrap_or_default().iter().filter_map(|&h| index(h)).collect();
        let left = self.token(c, "orientation") == Some("leftHanded");
        let subsets = self.subsets(c, counts.len(), depth);
        let default = binding.unwrap_or("Default").to_string();
        let nm = Mat3::from_mat4(m).inverse().transpose();
        let flip = (m.determinant() < 0.0) != left;
        self.sink.part = name;
        let mut corners = Vec::new();
        let mut next = 0usize;
        for (face, &n) in counts.iter().enumerate() {
            let n = index(n).unwrap_or(0);
            let (start, end) = (next, next.saturating_add(n));
            next = end;
            if n < 3 || holes.contains(&face) {
                continue;
            }
            corners.clear();
            for corner in start..end {
                let point = indices.get(corner).copied().and_then(index);
                let p = point.and_then(|i| get::<3>(points, i)).ok_or_else(|| malformed(KIND, "face vertex index out of range"))?;
                let at = (point.unwrap_or_default(), corner, face);
                corners.push(Corner {
                    p: m.transform_point3(Vec3::from(p)),
                    n: normals.as_ref().and_then(|pv| pv.get::<3>(at)).map(|v| nm * Vec3::from(v)),
                    uv: uvs.as_ref().and_then(|pv| pv.get::<2>(at)).map(|[u, v]| Vec2::new(u, 1.0 - v)),
                });
            }
            if flip {
                corners.reverse();
            }
            let material =
                subsets.as_ref().and_then(|(names, per_face)| per_face.get(face).and_then(|&k| names.get(k as usize))).map_or(default.as_str(), String::as_str);
            self.sink.polygon(material, &corners)?;
        }
        Ok(())
    }
}

/// A non-negative integral number as an index.
fn index(v: f64) -> Option<usize> {
    (v >= 0.0 && v.fract() == 0.0 && v < usize::MAX as f64).then_some(v as usize)
}

/// Element `i` of a flat array of `W`-wide tuples, as f32.
fn get<const W: usize>(values: &[f64], i: usize) -> Option<[f32; W]> {
    let s = values.get(i.checked_mul(W)?..i.checked_mul(W)?.checked_add(W)?)?;
    let mut out = [0f32; W];
    for (o, v) in out.iter_mut().zip(s) {
        *o = *v as f32;
    }
    Some(out)
}

#[derive(Clone, Copy)]
enum Interp {
    Constant,
    Uniform,
    Vertex,
    FaceVarying,
}

struct Primvar<'a> {
    values: &'a [f64],
    indices: Option<&'a [f64]>,
    interp: Interp,
}

impl Primvar<'_> {
    /// The value at (point, corner, face).
    fn get<const W: usize>(&self, (point, corner, face): (usize, usize, usize)) -> Option<[f32; W]> {
        let e = match self.interp {
            Interp::Constant => 0,
            Interp::Uniform => face,
            Interp::Vertex => point,
            Interp::FaceVarying => corner,
        };
        let e = match self.indices {
            Some(ix) => index(*ix.get(e)?)?,
            None => e,
        };
        get::<W>(self.values, e).filter(|v| v.iter().all(|x| x.is_finite()))
    }
}

/// The matrix of one xform op of `kind` (`translate`, `rotateXYZ`, ...) from its value.
fn op_matrix(kind: &str, v: &[f64]) -> Option<DMat4> {
    let v3 = || v.get(..3).map(DVec3::from_slice);
    let rot = |axis: u8, deg: f64| match axis {
        b'X' => DMat4::from_rotation_x(deg.to_radians()),
        b'Y' => DMat4::from_rotation_y(deg.to_radians()),
        _ => DMat4::from_rotation_z(deg.to_radians()),
    };
    Some(match kind {
        "translate" => DMat4::from_translation(v3()?),
        "scale" => DMat4::from_scale(v3()?),
        "rotateX" | "rotateY" | "rotateZ" => rot(kind.as_bytes().get(6).copied()?, *v.first()?),
        "orient" => {
            let [w, x, y, z]: [f64; 4] = v.get(..4)?.try_into().ok()?;
            DMat4::from_quat(DQuat::from_vec4(DVec4::new(x, y, z, w).try_normalize()?))
        }
        "transform" => {
            // USD matrices are row-major with row vectors: as columns they are glam's.
            let a: [f64; 16] = v.get(..16)?.try_into().ok()?;
            DMat4::from_cols_array(&a)
        }
        _ => {
            // rotateABC: the value holds (x, y, z) degrees; A is applied first.
            let axes = kind.strip_prefix("rotate").filter(|a| a.len() == 3 && a.bytes().all(|b| matches!(b, b'X' | b'Y' | b'Z')))?;
            let angle = |axis: u8| v.get(usize::from(axis - b'X')).copied();
            let mut m = DMat4::IDENTITY;
            for axis in axes.bytes() {
                m = rot(axis, angle(axis)?) * m;
            }
            m
        }
    })
}
