//! FBX 6.x and 7.x, binary (32- and 64-bit node records, zlib-compressed arrays) and
//! ASCII: mesh geometry with normals, UVs and per-polygon materials from the layer elements, every
//! Model's transform and parent hierarchy, and the file's unit and axis system (converted to
//! meters, Y up, like glTF). FBX 6 keeps the geometry inside its Model, names objects instead of
//! numbering them and lists properties in `Properties60`; both read into the same scene.
//! Materials bring their diffuse color and, when the file embeds it (a Video's `Content`), the
//! diffuse texture (for high-poly color map bakes); a texture kept as a file beside the FBX comes
//! along when the FBX is packaged with it ([`package`]: a stored zip of the FBX first and the
//! files under their paths below its folder, which [`read`] opens like the FBX). Camera Models become the file's cameras (lens
//! from their camera NodeAttribute, or the Model's own properties in FBX 6).

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::Read as _;

use base64::Engine as _;
use texturecraft_geom::glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use super::external::below;
use super::{
    Corner, MAX_MESH_FILE_BYTES, MAX_TEXTURE_BYTES, MAX_ZIP_ENTRIES, PolygonSink, RawBaseColor, RawCamera, RawMesh, check_vertices, lens_fov, zip_entry,
};
use crate::{FormatError, Result, malformed};

const KIND: &str = "FBX";
const MAGIC: &[u8] = b"Kaydara FBX Binary  \0";
/// A zip package of an FBX and its texture files ([`package`]).
const ZIP_MAGIC: &[u8] = b"PK\x03\x04";
/// First FBX version with the object/connection layout read here (6.0).
const MIN_VERSION: i64 = 6000;
/// Deepest node nesting (and Model parent chain) followed.
const MAX_DEPTH: usize = 64;
/// Memory budget for parsed nodes and properties (arrays are held as f64).
const MAX_DECODED_BYTES: usize = 1 << 31;
/// Approximate memory charged per node and per scalar property.
const NODE_COST: usize = 96;
const PROP_COST: usize = 32;

enum Prop {
    I(i64),
    F(f64),
    S(String),
    /// Binary data (`R`), such as an embedded image.
    Raw(Vec<u8>),
    Arr(Vec<f64>),
}

impl Prop {
    fn num(&self) -> Option<f64> {
        match self {
            Prop::I(v) => Some(*v as f64),
            Prop::F(v) => Some(*v),
            _ => None,
        }
    }
}

struct Node {
    name: String,
    props: Vec<Prop>,
    children: Vec<Node>,
}

impl Node {
    fn child(&self, name: &str) -> Option<&Node> {
        self.children.iter().find(|c| c.name == name)
    }

    fn all<'a>(&'a self, name: &'a str) -> impl Iterator<Item = &'a Node> {
        self.children.iter().filter(move |c| c.name == name)
    }

    fn int_at(&self, i: usize) -> Option<i64> {
        match self.props.get(i) {
            Some(Prop::I(v)) => Some(*v),
            _ => None,
        }
    }

    /// Object id: the number (7.x) or the `Class::Name` string (6.x).
    fn id_at(&self, i: usize) -> Option<String> {
        match self.props.get(i) {
            Some(Prop::I(v)) => Some(v.to_string()),
            Some(Prop::S(s)) if !s.is_empty() => Some(s.clone()),
            _ => None,
        }
    }

    /// An object's name: after its id (7.x) or the id itself (6.x).
    fn object_name(&self) -> &str {
        clean_name(if self.int_at(0).is_some() { self.str_at(1) } else { self.str_at(0) })
    }

    /// An object's class (`Mesh`, `Camera`, …): after its name.
    fn class(&self) -> &str {
        self.str_at(if self.int_at(0).is_some() { 2 } else { 1 })
    }

    fn str_at(&self, i: usize) -> &str {
        match self.props.get(i) {
            Some(Prop::S(s)) => s,
            _ => "",
        }
    }

    fn child_str(&self, name: &str) -> &str {
        self.child(name).map_or("", |c| c.str_at(0))
    }

    /// Embedded file bytes of a Video: binary data, or base64 text in ASCII files.
    fn content(&self) -> Option<Vec<u8>> {
        let c = self.child("Content")?;
        let bytes = match c.props.first()? {
            Prop::Raw(b) => b.clone(),
            _ => {
                let text: String = c.props.iter().filter_map(|p| if let Prop::S(s) = p { Some(s.as_str()) } else { None }).collect();
                base64::engine::general_purpose::STANDARD.decode(text.trim()).ok()?
            }
        };
        (!bytes.is_empty()).then_some(bytes)
    }

    /// Array payload: a binary array, an ASCII `*N { a: ... }` block or a bare value list.
    fn arr(&self) -> Cow<'_, [f64]> {
        if let Some(Prop::Arr(a)) = self.props.first() {
            return Cow::Borrowed(a);
        }
        if let Some(a) = self.child("a") {
            return a.arr();
        }
        Cow::Owned(self.props.iter().filter_map(Prop::num).collect())
    }

    /// The numbers of entry `name` in Properties70 (`P: name, type, subtype, flags, values`) or
    /// Properties60 (`Property: name, type, flags, values`).
    fn property(&self, name: &str) -> Option<Vec<f64>> {
        let list = self.child("Properties70").or_else(|| self.child("Properties60"))?;
        list.children.iter().find(|p| p.str_at(0) == name).map(|p| p.props.iter().filter_map(Prop::num).collect())
    }

    fn vec3(&self, name: &str, default: Vec3) -> Vec3 {
        match self.property(name).as_deref() {
            Some([x, y, z, ..]) => Vec3::new(*x as f32, *y as f32, *z as f32),
            _ => default,
        }
    }

    fn number(&self, name: &str) -> Option<f64> {
        self.property(name)?.first().copied()
    }
}

/// Charges parsed data against [`MAX_DECODED_BYTES`].
struct Budget(usize);

impl Budget {
    fn charge(&mut self, bytes: usize) -> Result<()> {
        self.0 = self.0.checked_sub(bytes).ok_or_else(|| FormatError::TooLarge(format!("the FBX file decodes to more than {MAX_DECODED_BYTES} bytes")))?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------
// Binary.

struct Binary<'a> {
    b: &'a [u8],
    pos: usize,
    /// 64-bit node records (version 7500 and later).
    wide: bool,
    budget: Budget,
}

impl<'a> Binary<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let s = self.pos.checked_add(n).and_then(|end| self.b.get(self.pos..end)).ok_or_else(|| malformed(KIND, "unexpected end of file"))?;
        self.pos += n;
        Ok(s)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }

    fn u32(&mut self) -> Result<usize> {
        Ok(u32::from_le_bytes(self.bytes()?) as usize)
    }

    fn uint(&mut self) -> Result<u64> {
        Ok(if self.wide { u64::from_le_bytes(self.bytes()?) } else { u32::from_le_bytes(self.bytes()?) as u64 })
    }

    /// The next node ending at or before `limit`, or None at a null record / the end.
    fn node(&mut self, limit: usize, depth: usize) -> Result<Option<Node>> {
        let header = if self.wide { 25 } else { 13 };
        if limit.saturating_sub(self.pos) < header {
            return Ok(None);
        }
        let (end, nprops, plen) = (self.uint()?, self.uint()?, self.uint()?);
        let [name_len] = self.bytes()?;
        if end == 0 {
            return Ok(None);
        }
        if depth > MAX_DEPTH {
            return Err(malformed(KIND, "nodes are nested too deeply"));
        }
        let end = usize::try_from(end).ok().filter(|e| *e > self.pos && *e <= limit).ok_or_else(|| malformed(KIND, "node end offset out of range"))?;
        self.budget.charge(NODE_COST)?;
        let name = String::from_utf8_lossy(self.take(name_len as usize)?).into_owned();
        let props_end = usize::try_from(plen).ok().and_then(|p| self.pos.checked_add(p)).filter(|e| *e <= end);
        let props_end = props_end.ok_or_else(|| malformed(KIND, format!("properties of \"{name}\" overrun the node")))?;
        let mut props = Vec::new();
        for _ in 0..nprops {
            if self.pos >= props_end {
                return Err(malformed(KIND, format!("\"{name}\" claims more properties than it holds")));
            }
            props.push(self.prop()?);
        }
        if self.pos > props_end {
            return Err(malformed(KIND, format!("properties of \"{name}\" overrun their list")));
        }
        self.pos = props_end;
        let mut children = Vec::new();
        while let Some(c) = self.node(end, depth + 1)? {
            children.push(c);
        }
        self.pos = end;
        Ok(Some(Node { name, props, children }))
    }

    fn prop(&mut self) -> Result<Prop> {
        self.budget.charge(PROP_COST)?;
        let [ty] = self.bytes()?;
        Ok(match ty {
            b'Y' => Prop::I(i16::from_le_bytes(self.bytes()?) as i64),
            b'C' | b'B' => Prop::I(self.bytes::<1>()?[0] as i64),
            b'I' => Prop::I(i32::from_le_bytes(self.bytes()?) as i64),
            b'L' => Prop::I(i64::from_le_bytes(self.bytes()?)),
            b'F' => Prop::F(f32::from_le_bytes(self.bytes()?) as f64),
            b'D' => Prop::F(f64::from_le_bytes(self.bytes()?)),
            b'S' => {
                let n = self.u32()?;
                Prop::S(String::from_utf8_lossy(self.take(n)?).into_owned())
            }
            b'R' => {
                let n = self.u32()?;
                self.budget.charge(n)?;
                Prop::Raw(self.take(n)?.to_vec())
            }
            b'f' | b'd' | b'l' | b'i' | b'b' => Prop::Arr(self.array(ty)?),
            other => return Err(malformed(KIND, format!("unknown property type 0x{other:02x}"))),
        })
    }

    fn array(&mut self, value_type: u8) -> Result<Vec<f64>> {
        let (count, encoding, stored) = (self.u32()?, self.u32()?, self.u32()?);
        let value_bytes = match value_type {
            b'd' | b'l' => 8,
            b'f' | b'i' => 4,
            _ => 1,
        };
        let decoded_bytes = count.checked_mul(value_bytes).ok_or_else(|| malformed(KIND, "array too large"))?;
        self.budget.charge(count.saturating_mul(8))?;

        let data = self.take(stored)?;
        let raw: Cow<[u8]> = match encoding {
            0 if stored == decoded_bytes => Cow::Borrowed(data),
            1 => {
                let mut out = Vec::new();
                flate2::read::ZlibDecoder::new(data)
                    .take(decoded_bytes as u64 + 1)
                    .read_to_end(&mut out)
                    .map_err(|e| malformed(KIND, format!("bad compressed array: {e}")))?;
                if out.len() != decoded_bytes {
                    return Err(malformed(KIND, "compressed array has the wrong length"));
                }

                Cow::Owned(out)
            }
            _ => return Err(malformed(KIND, format!("array encoding {encoding} with {stored} bytes for {count} items"))),
        };

        Ok(match value_type {
            b'd' => raw.as_chunks::<8>().0.iter().map(|c| f64::from_le_bytes(*c)).collect(),
            b'l' => raw.as_chunks::<8>().0.iter().map(|c| i64::from_le_bytes(*c) as f64).collect(),
            b'f' => raw.as_chunks::<4>().0.iter().map(|c| f32::from_le_bytes(*c) as f64).collect(),
            b'i' => raw.as_chunks::<4>().0.iter().map(|c| i32::from_le_bytes(*c) as f64).collect(),
            _ => raw.iter().map(|b| *b as f64).collect(),
        })
    }
}

fn parse_binary(b: &[u8]) -> Result<(Vec<Node>, i64)> {
    let mut r = Binary { b, pos: MAGIC.len(), wide: false, budget: Budget(MAX_DECODED_BYTES) };
    let _reserved: [u8; 2] = r.bytes()?;
    let version = u32::from_le_bytes(r.bytes()?);
    r.wide = version >= 7500;
    let mut nodes = Vec::new();
    while let Some(n) = r.node(b.len(), 0)? {
        nodes.push(n);
    }
    Ok((nodes, version as i64))
}

// ---------------------------------------------------------------------------------------------
// ASCII.

#[derive(Clone, Copy)]
enum Tok<'a> {
    /// `name:`
    Key(&'a str),
    Str(&'a [u8]),
    Word(&'a str),
    Comma,
    Open,
    Close,
}

struct Ascii<'a> {
    s: &'a [u8],
    pos: usize,
    budget: Budget,
}

impl<'a> Ascii<'a> {
    /// The next token and the position after it (the position is not advanced).
    fn lex(&self) -> Result<Option<(Tok<'a>, usize)>> {
        let mut i = self.pos;
        loop {
            match self.s.get(i) {
                None => return Ok(None),
                Some(c) if c.is_ascii_whitespace() => i += 1,
                Some(b';') => i += self.s.get(i..).unwrap_or_default().iter().position(|c| *c == b'\n').unwrap_or(self.s.len() - i),
                Some(_) => break,
            }
        }
        let rest = self.s.get(i..).unwrap_or_default();
        Ok(Some(match rest.first() {
            Some(b',') => (Tok::Comma, i + 1),
            Some(b'{') => (Tok::Open, i + 1),
            Some(b'}') => (Tok::Close, i + 1),
            Some(b'"') => {
                let len = rest.get(1..).unwrap_or_default().iter().position(|c| *c == b'"').ok_or_else(|| malformed(KIND, "unterminated string"))?;
                (Tok::Str(rest.get(1..1 + len).unwrap_or_default()), i + len + 2)
            }
            _ => {
                let len = rest.iter().position(|c| c.is_ascii_whitespace() || b",{}\":;".contains(c)).unwrap_or(rest.len());
                if len == 0 {
                    return Err(malformed(KIND, format!("unexpected character at byte {i}")));
                }
                let word = std::str::from_utf8(rest.get(..len).unwrap_or_default()).map_err(|_| malformed(KIND, "text is not UTF-8"))?;
                if rest.get(len) == Some(&b':') { (Tok::Key(word), i + len + 1) } else { (Tok::Word(word), i + len) }
            }
        }))
    }

    /// Nodes until `}` (left for the caller) or the end.
    fn nodes(&mut self, depth: usize) -> Result<Vec<Node>> {
        if depth > MAX_DEPTH {
            return Err(malformed(KIND, "nodes are nested too deeply"));
        }
        let mut out = Vec::new();
        while let Some((tok, after)) = self.lex()? {
            match tok {
                Tok::Close => break,
                Tok::Key(name) => {
                    self.pos = after;
                    out.push(self.node(name, depth)?);
                }
                _ => return Err(malformed(KIND, format!("expected a property name at byte {}", self.pos))),
            }
        }
        Ok(out)
    }

    fn node(&mut self, name: &str, depth: usize) -> Result<Node> {
        self.budget.charge(NODE_COST)?;
        let mut props = Vec::new();
        // An empty first value (`Content: , "..."`).
        if let Some((Tok::Comma, after)) = self.lex()? {
            self.pos = after;
        }
        while let Some((tok, after)) = self.lex()? {
            let prop = match tok {
                Tok::Str(s) => Some(Prop::S(String::from_utf8_lossy(s).into_owned())),
                // `*N` announces an array's length.
                Tok::Word(w) if w.starts_with('*') => None,
                Tok::Word(w) => Some(w.parse::<i64>().map(Prop::I).or_else(|_| w.parse::<f64>().map(Prop::F)).unwrap_or_else(|_| Prop::S(w.to_string()))),
                _ => break,
            };
            self.pos = after;
            if let Some(p) = prop {
                self.budget.charge(PROP_COST)?;
                props.push(p);
            }
            match self.lex()? {
                Some((Tok::Comma, after)) => self.pos = after,
                _ => break,
            }
        }
        let mut children = Vec::new();
        if let Some((Tok::Open, after)) = self.lex()? {
            self.pos = after;
            children = self.nodes(depth + 1)?;
            match self.lex()? {
                Some((Tok::Close, after)) => self.pos = after,
                _ => return Err(malformed(KIND, format!("\"{name}\" is missing its closing brace"))),
            }
        }
        // Long value lists (arrays) are stored compactly.
        if props.len() > 1 && props.iter().all(|p| p.num().is_some()) {
            props = vec![Prop::Arr(props.iter().filter_map(Prop::num).collect())];
        }
        Ok(Node { name: name.to_string(), props, children })
    }
}

fn parse_ascii(s: &[u8]) -> Result<(Vec<Node>, i64)> {
    let mut a = Ascii { s, pos: 0, budget: Budget(MAX_DECODED_BYTES) };
    let nodes = a.nodes(0)?;
    if a.lex()?.is_some() {
        return Err(malformed(KIND, "unbalanced closing brace"));
    }
    let version = nodes.iter().find(|n| n.name == "FBXHeaderExtension").and_then(|h| h.child("FBXVersion")).and_then(|v| v.int_at(0));
    let version = version.ok_or_else(|| malformed(KIND, "not an FBX file (no FBXHeaderExtension / FBXVersion)"))?;
    Ok((nodes, version))
}

// ---------------------------------------------------------------------------------------------
// Scene.

/// The bytes of the first of a texture's candidate file paths (relative to the FBX's folder)
/// that exists.
type Files<'a> = dyn FnMut(&[&str]) -> Option<Vec<u8>> + 'a;

/// An FBX, or a package of one and its texture files.
pub(crate) fn read(bytes: &[u8]) -> Result<RawMesh> {
    if !bytes.starts_with(ZIP_MAGIC) {
        return crate::guard(KIND, || read_inner(bytes, &mut |_| None));
    }
    crate::guard(KIND, || {
        let mut zip = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(|e| malformed(KIND, format!("FBX package: {e}")))?;
        if zip.len() > MAX_ZIP_ENTRIES {
            return Err(FormatError::TooLarge(format!("the FBX package has more than {MAX_ZIP_ENTRIES} entries")));
        }
        let name = zip.file_names().flatten().filter(|n| super::extension(n) == "fbx").min_by_key(|n| zip.index_for_name(n)).map(Cow::into_owned);
        let name = name.ok_or_else(|| malformed(KIND, "the package contains no FBX file"))?;
        let fbx = zip_entry(&mut zip, &name, MAX_MESH_FILE_BYTES)
            .ok_or_else(|| FormatError::TooLarge("the packaged FBX is unreadable or larger than 2 GiB".into()))?;
        if fbx.starts_with(ZIP_MAGIC) {
            return Err(malformed(KIND, "the package holds another package"));
        }
        read_inner(&fbx, &mut |paths| paths.iter().find_map(|p| zip_entry(&mut zip, &below(p)?, MAX_TEXTURE_BYTES)))
    })
}

/// An FBX whose base color textures are files, as a package holding it (first) and the files
/// `found` has, under their paths below its folder; None when it found none (or `bytes` is a
/// package already). Also the texture paths it could not find.
pub(crate) fn package(file_name: &str, bytes: &[u8], found: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Result<(Option<Vec<u8>>, Vec<String>)> {
    package_with_limit(file_name, bytes, found, MAX_MESH_FILE_BYTES)
}

pub(super) fn package_with_limit(
    file_name: &str,
    bytes: &[u8],
    found: &mut dyn FnMut(&str) -> Option<Vec<u8>>,
    max_bytes: u64,
) -> Result<(Option<Vec<u8>>, Vec<String>)> {
    if bytes.starts_with(ZIP_MAGIC) {
        return Ok((None, Vec::new()));
    }

    crate::guard(KIND, || {
        let (mut files, mut missing): (Vec<(String, Vec<u8>)>, Vec<String>) = (Vec::new(), Vec::new());
        let mut total = bytes.len() as u64;
        read_inner(bytes, &mut |paths| {
            // Once over budget, later materials must not trigger more file reads or clones
            if total > max_bytes {
                return None;
            }

            if let Some((_, data)) = files.iter().find(|(n, _)| paths.iter().any(|p| below(p).as_ref() == Some(n))) {
                return Some(data.clone());
            }

            for path in paths {
                let Some(name) = below(path) else { continue };
                if let Some(data) = found(path).filter(|data| data.len() as u64 <= MAX_TEXTURE_BYTES) {
                    total = total.saturating_add(data.len() as u64);
                    if total > max_bytes {
                        return None;
                    }

                    files.push((name, data.clone()));
                    return Some(data);
                }
            }

            // A texture is missing when none of its paths was found; named by its first
            if let Some(first) = paths.first().filter(|f| !missing.iter().any(|m| m == *f)) {
                missing.push(first.to_string());
            }
            None
        })?;
        if total > max_bytes {
            return Err(FormatError::TooLarge("the FBX and its textures exceed 2 GiB".into()));
        }

        if files.is_empty() {
            return Ok((None, missing));
        }

        let leaf = std::path::Path::new(file_name).file_name().and_then(|n| n.to_str()).unwrap_or("mesh.fbx");
        Ok((Some(crate::mesh_export::usd::usdz(leaf, bytes, &files)?), missing))
    })
}

fn read_inner(bytes: &[u8], files: &mut Files) -> Result<RawMesh> {
    let (nodes, version) = if bytes.starts_with(MAGIC) { parse_binary(bytes)? } else { parse_ascii(bytes)? };
    if version < MIN_VERSION {
        return Err(FormatError::Unsupported(format!("FBX version {version} (re-export as FBX 2006 / 6.1 or newer)")));
    }
    let root = |name: &str| nodes.iter().find(|n| n.name == name);
    let objects = root("Objects").ok_or_else(|| malformed(KIND, "the file has no Objects"))?;
    let (mut geometries, mut models, mut materials) = (HashMap::new(), HashMap::new(), HashMap::new());
    let (mut textures, mut videos, mut attributes) = (HashMap::new(), HashMap::new(), HashMap::new());
    // (geometry, model) instances; an FBX 6 Model holds its geometry itself.
    let mut instances: Vec<(String, Option<String>)> = Vec::new();
    for o in &objects.children {
        let Some(id) = o.id_at(0) else { continue };
        match o.name.as_str() {
            "Geometry" if o.child("Vertices").is_some() => geometries.insert(id, o),
            "Model" => {
                if o.child("Vertices").is_some() {
                    geometries.insert(id.clone(), o);
                    instances.push((id.clone(), Some(id.clone())));
                }
                models.insert(id, o)
            }
            "Material" => materials.insert(id, o),
            "Texture" => textures.insert(id, o),
            "Video" => videos.insert(id, o),
            "NodeAttribute" => attributes.insert(id, o),
            _ => None,
        };
    }
    // Model parents, and each Model's materials in slot order.
    let mut parents: HashMap<String, String> = HashMap::new();
    let mut slots: HashMap<String, Vec<String>> = HashMap::new();
    let mut attribute_of: HashMap<String, &Node> = HashMap::new();
    let connections: Vec<&Node> = root("Connections").into_iter().flat_map(|c| c.all("C").chain(c.all("Connect"))).collect();
    for c in &connections {
        let (Some(child), Some(parent)) = (c.id_at(1), c.id_at(2)) else { continue };
        if c.str_at(0) != "OO" || child == parent || !models.contains_key(&parent) {
            continue;
        }
        if models.contains_key(&child) {
            parents.insert(child, parent);
        } else if geometries.contains_key(&child) {
            instances.push((child, Some(parent)));
        } else if let Some(m) = materials.get(&child) {
            slots.entry(parent).or_default().push(material_name(&child, m));
        } else if let Some(a) = attributes.get(&child) {
            attribute_of.insert(parent, *a);
        }
    }
    let placed: HashSet<&String> = instances.iter().map(|(g, _)| g).collect();
    let mut unplaced: Vec<String> = geometries.keys().filter(|g| !placed.contains(g)).cloned().collect();
    unplaced.sort_unstable();
    instances.extend(unplaced.into_iter().map(|g| (g, None)));

    let base_colors = base_colors(&materials, &textures, &videos, &connections, files);
    let global = global_transform(root("GlobalSettings").or_else(|| objects.child("GlobalSettings")));
    let mut sink = PolygonSink::default();
    let no_slots = Vec::new();
    for (geom, model) in instances {
        let Some(g) = geometries.get(&geom) else { continue };
        let m = match model.as_ref().and_then(|m| models.get(m).map(|n| (m, n))) {
            Some((id, node)) => {
                sink.part = node.object_name().to_string();
                world(&models, &parents, id)? * geometric(node)
            }
            None => {
                sink.part.clear();
                Mat4::IDENTITY
            }
        };
        let names = model.as_ref().and_then(|m| slots.get(m)).unwrap_or(&no_slots);
        add_geometry(&mut sink, g, global * m, names)?;
    }
    let mut raw = sink.finish();
    raw.base_colors = base_colors;
    // In file order; FBX 6's built-in "Producer" views are not the scene's cameras.
    for o in objects.children.iter().filter(|o| o.name == "Model" && o.class() == "Camera" && !o.object_name().starts_with("Producer ")) {
        let Some(id) = o.id_at(0) else { continue };
        if let Ok(m) = world(&models, &parents, &id) {
            raw.cameras.push(camera(o, attribute_of.get(&id).copied().unwrap_or(o), global * m));
        }
    }
    Ok(raw)
}

/// A camera Model at `m` with its lens in `lens`: FBX cameras look down +X with +Y up. The
/// vertical field of view comes from the focal length (mm) over the film height (inches).
/// shortcut: `OrthoZoom` is taken as the orthographic half view height in scene units; check it
/// against an exporter's orthographic camera when one is at hand.
fn camera(model: &Node, lens: &Node, m: Mat4) -> RawCamera {
    let num = |k: &str| lens.number(k).map(|v| v as f32).filter(|v| v.is_finite() && *v > 0.0);
    let fov = match num("FocalLength") {
        Some(f) => lens_fov(f, num("FilmHeight").unwrap_or(0.612) * 25.4),
        None => num("FieldOfViewY").or_else(|| num("FieldOfView")).unwrap_or(40.0),
    };
    let ortho = (lens.number("CameraProjectionType") == Some(1.0)).then(|| num("OrthoZoom").unwrap_or(1.0));
    RawCamera::placed(model.object_name().to_string(), m, Vec3::X, fov, ortho)
}

/// A material's name in the mesh: its object name, or `Material_<id>` when it has none.
fn material_name(id: &str, m: &Node) -> String {
    let name = m.object_name();
    if name.is_empty() { format!("Material_{id}") } else { name.to_string() }
}

/// Preserve material colors independently of textures for ID baking
fn base_colors(
    materials: &HashMap<String, &Node>,
    textures: &HashMap<String, &Node>,
    videos: &HashMap<String, &Node>,
    connections: &[&Node],
    files: &mut Files,
) -> Vec<RawBaseColor> {
    // Texture → its Video, material → its diffuse texture (a property connection wins).
    let (mut video_of, mut diffuse) = (HashMap::new(), HashMap::new());
    for c in connections {
        let (Some(child), Some(parent)) = (c.id_at(1), c.id_at(2)) else { continue };
        if videos.contains_key(&child) && textures.contains_key(&parent) {
            video_of.insert(parent, child);
        } else if textures.contains_key(&child) && materials.contains_key(&parent) {
            match (c.str_at(0), c.str_at(3)) {
                ("OP", "DiffuseColor" | "Maya|baseColor") => {
                    diffuse.insert(parent, child);
                }
                ("OO", _) => {
                    diffuse.entry(parent).or_insert(child);
                }
                _ => {}
            }
        }
    }
    let mut out: Vec<RawBaseColor> = materials
        .iter()
        .filter_map(|(id, m)| {
            let texture = diffuse.get(id);
            let (video, tex) = (texture.and_then(|t| video_of.get(t)).and_then(|v| videos.get(v)), texture.and_then(|t| textures.get(t)));
            // Embedded content first, then the file the texture (or its video) names.
            let image = video.or(tex).and_then(|n| n.content()).or_else(|| [tex, video].into_iter().flatten().find_map(|n| texture_file(n, files)));
            let diffuse_color = m.property("DiffuseColor");
            let factor = match diffuse_color.as_deref() {
                Some([r, g, b, ..]) => {
                    let f = m.number("DiffuseFactor").unwrap_or(1.0);
                    Vec4::new((r * f) as f32, (g * f) as f32, (b * f) as f32, 1.0)
                }
                _ => Vec4::ONE,
            };
            (factor.is_finite() && (image.is_some() || diffuse_color.is_some())).then(|| RawBaseColor { material: material_name(id, m), factor, image })
        })
        .collect();
    out.sort_by(|a, b| a.material.cmp(&b.material));
    out
}

/// The file a Texture or Video names: its relative path, else its absolute path's file name
/// (looked for in the FBX's folder).
fn texture_file(n: &Node, files: &mut Files) -> Option<Vec<u8>> {
    let relative = n.child_str("RelativeFilename");
    let absolute = [n.child_str("FileName"), n.child_str("Filename")].into_iter().find(|s| !s.is_empty()).unwrap_or_default();
    let leaf = absolute.rsplit(['/', '\\']).next().unwrap_or_default();
    let mut paths: Vec<&str> = Vec::new();
    for p in [relative, leaf] {
        if !p.is_empty() && !paths.contains(&p) {
            paths.push(p);
        }
    }
    if paths.is_empty() { None } else { files(&paths) }
}

/// `Name\0\1Class` (binary) or `Class::Name` (ASCII) to `Name`.
fn clean_name(s: &str) -> &str {
    match s.split_once("\u{0}\u{1}") {
        Some((name, _)) => name,
        None => s.split_once("::").map_or(s, |(_, name)| name),
    }
}

/// Unit scale to meters and the file's axis system to Y up (+Z front, +X right).
fn global_transform(settings: Option<&Node>) -> Mat4 {
    let get = |name: &str, default: f64| settings.and_then(|s| s.number(name)).filter(|v| v.is_finite()).unwrap_or(default);
    let unit = get("UnitScaleFactor", 1.0);
    let unit = if unit > 0.0 { unit / 100.0 } else { 0.01 };
    let axis = |a: &str, s: &str, default: f64| (get(a, default) as i64, if get(s, 1.0) < 0.0 { -1.0 } else { 1.0 });
    let [(right, rs), (up, us), (front, fs)] =
        [axis("CoordAxis", "CoordAxisSign", 0.0), axis("UpAxis", "UpAxisSign", 1.0), axis("FrontAxis", "FrontAxisSign", 2.0)];
    let basis = |a: i64| [Vec3::X, Vec3::Y, Vec3::Z].get(a as usize).copied().filter(|_| a >= 0);
    let rows = match (basis(right), basis(up), basis(front)) {
        (Some(r), Some(u), Some(f)) if r.cross(u).dot(f).abs() > 0.5 => Mat3::from_cols(r * rs, u * us, f * fs).transpose(),
        _ => Mat3::IDENTITY,
    };
    Mat4::from_mat3(rows) * Mat4::from_scale(Vec3::splat(unit as f32))
}

/// Euler rotation in degrees, `order` as FBX RotationOrder (0 = XYZ: X applied first).
fn euler(deg: Vec3, order: i64) -> Mat4 {
    use Vec3 as V;
    let axes = match order {
        1 => [V::X, V::Z, V::Y],
        2 => [V::Y, V::Z, V::X],
        3 => [V::Y, V::X, V::Z],
        4 => [V::Z, V::X, V::Y],
        5 => [V::Z, V::Y, V::X],
        _ => [V::X, V::Y, V::Z],
    };
    axes.iter().fold(Mat4::IDENTITY, |m, a| Mat4::from_axis_angle(*a, deg.dot(*a).to_radians()) * m)
}

/// A Model's local transform: T · Roff · Rp · Rpre · R · Rpost⁻¹ · Rp⁻¹ · Soff · Sp · S · Sp⁻¹.
fn local(m: &Node) -> Mat4 {
    let t = |name: &str| Mat4::from_translation(m.vec3(name, Vec3::ZERO));
    let order = m.number("RotationOrder").unwrap_or(0.0) as i64;
    let (rp, sp) = (m.vec3("RotationPivot", Vec3::ZERO), m.vec3("ScalingPivot", Vec3::ZERO));
    t("Lcl Translation")
        * t("RotationOffset")
        * Mat4::from_translation(rp)
        * euler(m.vec3("PreRotation", Vec3::ZERO), 0)
        * euler(m.vec3("Lcl Rotation", Vec3::ZERO), order)
        * euler(m.vec3("PostRotation", Vec3::ZERO), 0).inverse()
        * Mat4::from_translation(-rp)
        * t("ScalingOffset")
        * Mat4::from_translation(sp)
        * Mat4::from_scale(m.vec3("Lcl Scaling", Vec3::ONE))
        * Mat4::from_translation(-sp)
}

/// The geometry offset of a Model (not inherited by its children).
fn geometric(m: &Node) -> Mat4 {
    Mat4::from_translation(m.vec3("GeometricTranslation", Vec3::ZERO))
        * euler(m.vec3("GeometricRotation", Vec3::ZERO), 0)
        * Mat4::from_scale(m.vec3("GeometricScaling", Vec3::ONE))
}

fn world(models: &HashMap<String, &Node>, parents: &HashMap<String, String>, id: &str) -> Result<Mat4> {
    let mut m = Mat4::IDENTITY;
    let mut cur = Some(id);
    for _ in 0..=MAX_DEPTH {
        let Some(node) = cur.and_then(|c| models.get(c)) else { return Ok(m) };
        m = local(node) * m;
        cur = cur.and_then(|c| parents.get(c)).map(String::as_str);
    }
    Err(malformed(KIND, "the Model hierarchy is nested too deeply (or cyclic)"))
}

#[derive(Clone, Copy)]
enum Mapping {
    PolygonVertex,
    ControlPoint,
    Polygon,
    AllSame,
}

/// One LayerElement's items, optionally through an index array.
struct Layer<'a> {
    mapping: Mapping,
    data: Cow<'a, [f64]>,
    index: Option<Cow<'a, [f64]>>,
}

impl Layer<'_> {
    /// Element `index` names the index array (empty: the data is never indexed).
    fn read<'a>(geom: &'a Node, element: &str, data: &str, index: &str) -> Option<Layer<'a>> {
        let el = geom.children.iter().filter(|c| c.name == element).min_by_key(|e| e.int_at(0) != Some(0))?;
        let mapping = match el.child_str("MappingInformationType") {
            "ByPolygonVertex" => Mapping::PolygonVertex,
            "ByVertice" | "ByVertex" | "ByControlPoint" => Mapping::ControlPoint,
            "ByPolygon" => Mapping::Polygon,
            "AllSame" => Mapping::AllSame,
            _ => return None,
        };
        let indexed = !index.is_empty() && matches!(el.child_str("ReferenceInformationType"), "IndexToDirect" | "Index");
        let index = if indexed { Some(el.child(index)?.arr()) } else { None };
        Some(Layer { mapping, data: el.child(data)?.arr(), index })
    }

    /// The `W` numbers of polygon vertex `k` (control point `cp`, polygon `poly`).
    fn get<const W: usize>(&self, k: usize, cp: usize, poly: usize) -> Option<[f64; W]> {
        let i = match self.mapping {
            Mapping::PolygonVertex => k,
            Mapping::ControlPoint => cp,
            Mapping::Polygon => poly,
            Mapping::AllSame => 0,
        };
        let i = match &self.index {
            Some(ix) => Some(*ix.get(i)?).filter(|v| *v >= 0.0)? as usize,
            None => i,
        };
        let start = i.checked_mul(W)?;
        self.data.get(start..start.checked_add(W)?)?.try_into().ok()
    }
}

fn add_geometry(sink: &mut PolygonSink, geometry: &Node, transform: Mat4, slots: &[String]) -> Result<()> {
    let vertices = geometry.child("Vertices").map(Node::arr).unwrap_or_default();
    check_vertices(vertices.len() / 3)?;
    let points: Vec<Vec3> = vertices.as_chunks::<3>().0.iter().map(|c| transform.transform_point3(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))).collect();
    let polygon_indices = geometry.child("PolygonVertexIndex").map(Node::arr).unwrap_or_default();
    let normals = Layer::read(geometry, "LayerElementNormal", "Normals", "NormalsIndex");
    let uvs = Layer::read(geometry, "LayerElementUV", "UV", "UVIndex");
    let materials = Layer::read(geometry, "LayerElementMaterial", "Materials", "");
    let normal_transform = Mat3::from_mat4(transform).inverse().transpose();
    let flip = transform.determinant() < 0.0;
    let (mut corners, mut polygon_index, mut bad_normals, mut bad_uvs) = (Vec::new(), 0usize, 0usize, 0usize);

    for (k, &v) in polygon_indices.iter().enumerate() {
        // The last corner of each polygon is stored as -(index + 1)
        let (control_point, last) = if v < 0.0 { (-v - 1.0, true) } else { (v, false) };
        let p = points
            .get(control_point as usize)
            .filter(|_| control_point.fract() == 0.0)
            .ok_or_else(|| malformed(KIND, format!("polygon vertex index {v} out of range")))?;
        let control_point = control_point as usize;
        let n = normals.as_ref().and_then(|l| {
            let n = l.get(k, control_point, polygon_index).map(|[x, y, z]| normal_transform * Vec3::new(x as f32, y as f32, z as f32));
            bad_normals += usize::from(n.is_none());
            n
        });

        // A UV layer with holes keeps its layout: missing items read as 0, 0
        let uv = uvs.as_ref().map(|l| match l.get(k, control_point, polygon_index) {
            Some([u, v]) => Vec2::new(u as f32, 1.0 - v as f32),
            None => {
                bad_uvs += 1;
                Vec2::ZERO
            }
        });
        corners.push(Corner { p: *p, n, uv });
        if last || k + 1 == polygon_indices.len() {
            if flip {
                corners.reverse();
            }
            let slot = materials.as_ref().and_then(|l| l.get(polygon_index, polygon_index, polygon_index)).map_or(0.0, |[s]| s);
            let name = slots.get(slot as usize).filter(|_| slot >= 0.0).map_or("Default", String::as_str);
            sink.polygon(name, &corners)?;
            corners.clear();
            polygon_index += 1;
        }
    }

    if bad_normals > 0 {
        sink.warn(format!("{bad_normals} polygon corners have no normal in the FBX normal layer; normals were recomputed"));
    }
    if bad_uvs > 0 {
        sink.warn(format!("{bad_uvs} polygon corners have no UV in the FBX UV layer (set to 0, 0)"));
    }

    Ok(())
}
