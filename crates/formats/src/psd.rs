//! Layered PSD documents (16-bit RGB, version 1) for an image editor, and the layers read back
//! after the editor saved them.
//!
//! Written from the public PSD file format specification: a header, empty colour mode data,
//! image resources (the layer comp list, resource 1065), the layer and mask section (layer
//! records bottom to top, groups as `lsct` section dividers, layer masks as channel −2,
//! Unicode names in `luni`, layer ids in `lyid`, per-layer comp visibility in `shmd`/`cmls`),
//! and the merged image. Layer channels are written with ZIP-with-prediction compression; the
//! reader also takes raw, RLE and plain ZIP, 8 and 16 bits, which is what editors save.
//!
//! Every read is bounded: canvas size, layer count, nesting and decoded bytes are capped, and
//! malformed input is an error, never a panic.

use std::io::{Read, Write};

use crate::ora::Rgba16;
use crate::{FormatError, Result};

/// File extension.
pub const PSD_EXT: &str = "psd";
/// Largest canvas side written or read.
pub const MAX_SIDE: u32 = 16384;
/// Most layer records (groups count twice) read.
const MAX_RECORDS: usize = 8192;
/// Deepest group nesting written or read.
pub const MAX_DEPTH: usize = 64;
/// Most decoded channel bytes read from one file.
const MAX_DECODED: u64 = 6 << 30;
/// Image resource id of the layer comp list.
const LAYER_COMPS: u16 = 1065;

fn invalid(msg: impl Into<String>) -> FormatError {
    FormatError::Invalid(msg.into())
}

fn malformed(msg: impl Into<String>) -> FormatError {
    FormatError::Malformed { kind: "PSD", msg: msg.into() }
}

// ------------------------------------------------------------------------------------------
// Model

/// A rectangle as the format stores it: top, left, bottom, right.
type Rect = (i32, i32, i32, i32);

/// A grayscale layer mask the size of the canvas (0 hides, 65535 shows).
#[derive(Clone, Debug, PartialEq)]
pub struct Mask16 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
}

/// One layer or group of a document to write, children top-first.
#[derive(Clone, Debug, PartialEq)]
pub struct Layer {
    /// The layer's id (`lyid`); 0 picks one above every id given. Ids should be unique.
    pub id: u32,
    pub name: String,
    pub visible: bool,
    /// 0..1.
    pub opacity: f32,
    /// A PSD blend mode key (`norm`, `mul `, `pass` for a pass-through group, ...).
    pub blend: [u8; 4],
    pub mask: Option<EncodedMask>,
    /// Visibility in each of the document's layer comps (missing entries: `visible`).
    pub comps: Vec<bool>,
    pub kind: Kind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum Kind {
    /// Pixels, compressed as soon as they are made ([`encode_pixels`]), so a document of many
    /// large layers never holds them all uncompressed.
    Pixels(EncodedPixels),
    Group(Vec<Layer>),
}

/// A layer's channels, cropped to its pixels and compressed.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedPixels {
    canvas: (u32, u32),
    rect: (i32, i32, i32, i32),
    data: Vec<(i16, Vec<u8>)>,
}

/// A layer mask, cropped to where it is not 0 and compressed.
#[derive(Clone, Debug, PartialEq)]
pub struct EncodedMask {
    canvas: (u32, u32),
    rect: (i32, i32, i32, i32),
    data: Vec<u8>,
}

/// A document to write: layers top-first, the merged image, layer comp names.
#[derive(Clone, Debug, PartialEq)]
pub struct Document {
    pub layers: Vec<Layer>,
    pub merged: Rgba16,
    pub comps: Vec<String>,
}

// ------------------------------------------------------------------------------------------
// Writing

trait Put {
    fn u8(&mut self, v: u8);
    fn u16(&mut self, v: u16);
    fn i16(&mut self, v: i16);
    fn u32(&mut self, v: u32);
    fn i32(&mut self, v: i32);
    fn bytes(&mut self, v: &[u8]);
}

impl Put for Vec<u8> {
    fn u8(&mut self, v: u8) {
        self.push(v);
    }
    fn u16(&mut self, v: u16) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn i16(&mut self, v: i16) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.extend_from_slice(&v.to_be_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.extend_from_slice(v);
    }
}

fn len32(n: usize) -> Result<u32> {
    u32::try_from(n).map_err(|_| invalid("the document is larger than a PSD can hold (4 GB sections); use smaller texture sets"))
}

/// Action descriptor values (the subset layer comps need).
enum Value {
    Long(i32),
    Bool(bool),
    Text(String),
    List(Vec<Value>),
    Object(&'static str, Vec<(&'static str, Value)>),
}

/// A descriptor key or class id: four-character ids as length 0 plus the id, others with their length.
fn put_id(out: &mut Vec<u8>, id: &str) {
    if id.len() == 4 {
        out.u32(0);
    } else {
        out.u32(id.len() as u32);
    }
    out.bytes(id.as_bytes());
}

fn put_unicode(out: &mut Vec<u8>, s: &str, nul: bool) {
    let mut units: Vec<u16> = s.encode_utf16().collect();
    if nul {
        units.push(0);
    }
    out.u32(units.len() as u32);
    for u in units {
        out.u16(u);
    }
}

fn put_object(out: &mut Vec<u8>, class: &str, items: &[(&'static str, Value)]) {
    put_unicode(out, "", true);
    put_id(out, class);
    out.u32(items.len() as u32);
    for (k, v) in items {
        put_id(out, k);
        put_value(out, v);
    }
}

fn put_value(out: &mut Vec<u8>, v: &Value) {
    match v {
        Value::Long(i) => {
            out.bytes(b"long");
            out.i32(*i);
        }
        Value::Bool(b) => {
            out.bytes(b"bool");
            out.u8(u8::from(*b));
        }
        Value::Text(t) => {
            out.bytes(b"TEXT");
            put_unicode(out, t, true);
        }
        Value::List(items) => {
            out.bytes(b"VlLs");
            out.u32(items.len() as u32);
            for i in items {
                put_value(out, i);
            }
        }
        Value::Object(class, items) => {
            out.bytes(b"Objc");
            put_object(out, class, items);
        }
    }
}

fn versioned(class: &str, items: &[(&'static str, Value)]) -> Vec<u8> {
    let mut out = Vec::new();
    out.u32(16);
    put_object(&mut out, class, items);
    out
}

/// Additional layer information: signature, key, length, data padded to even.
fn tagged(out: &mut Vec<u8>, key: &[u8; 4], data: &[u8]) -> Result<()> {
    out.bytes(b"8BIM");
    out.bytes(key);
    let padded = data.len() + data.len() % 2;
    out.u32(len32(padded)?);
    out.bytes(data);
    if data.len() % 2 == 1 {
        out.u8(0);
    }
    Ok(())
}

/// One plane of `w`×`h` 16-bit samples, ZIP with prediction (row deltas, big-endian, zlib).
fn zip_predicted(plane: &[u16], w: usize) -> Result<Vec<u8>> {
    let mut bytes = Vec::with_capacity(plane.len() * 2);
    for row in plane.chunks(w.max(1)) {
        let mut prev = 0u16;
        for &v in row {
            bytes.extend_from_slice(&v.wrapping_sub(prev).to_be_bytes());
            prev = v;
        }
    }
    let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::new(6));
    z.write_all(&bytes)?;
    Ok(z.finish()?)
}

/// A layer's channels for its record: rectangle, (channel id, compression code + data), mask.
struct Channels {
    rect: (i32, i32, i32, i32),
    data: Vec<(i16, Vec<u8>)>,
    mask: Option<(Rect, Vec<u8>)>,
}

/// A channel with nothing in it: just the raw compression code.
fn empty_channel() -> Vec<u8> {
    vec![0, 0]
}

fn coded(plane: &[u16], w: usize) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    out.u16(3);
    out.bytes(&zip_predicted(plane, w)?);
    Ok(out)
}

/// The smallest rectangle (top, left, bottom, right) holding a sample for which `keep` is true.
fn bounds(w: usize, n: usize, keep: impl Fn(usize) -> bool) -> Option<(i32, i32, i32, i32)> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
    for i in (0..n).filter(|i| keep(*i)) {
        let (x, y) = (i % w.max(1), i / w.max(1));
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    (x0 != usize::MAX).then(|| (y0 as i32, x0 as i32, y1 as i32 + 1, x1 as i32 + 1))
}

fn crop_plane(src: &[u16], stride: usize, comps: usize, comp: usize, (t, l, b, r): (i32, i32, i32, i32)) -> Vec<u16> {
    let mut out = Vec::with_capacity(((b - t) * (r - l)).max(0) as usize);
    for y in t..b {
        for x in l..r {
            out.push(src.get((y as usize * stride + x as usize) * comps + comp).copied().unwrap_or(0));
        }
    }
    out
}

fn check_size(what: &str, (w, h): (u32, u32), len: usize, comps: usize) -> Result<()> {
    if w == 0 || h == 0 || w > MAX_SIDE || h > MAX_SIDE || len != w as usize * h as usize * comps {
        return Err(invalid(format!("{what}: {w}×{h} with {len} samples is not a size a PSD is written at")));
    }
    Ok(())
}

/// Compress a layer's straight-alpha pixels, cropped to where they have alpha.
pub fn encode_pixels(img: &Rgba16) -> Result<EncodedPixels> {
    check_size("layer", (img.width, img.height), img.data.len(), 4)?;
    let w = img.width as usize;
    let rect = bounds(w, w * img.height as usize, |i| img.data.get(i * 4 + 3).copied().unwrap_or(0) > 0);
    let mut data = Vec::new();
    match rect {
        Some(rect) => {
            let rw = (rect.3 - rect.1) as usize;
            for (id, comp) in [(-1i16, 3usize), (0, 0), (1, 1), (2, 2)] {
                data.push((id, coded(&crop_plane(&img.data, w, 4, comp, rect), rw)?));
            }
        }
        None => data.extend([-1i16, 0, 1, 2].map(|id| (id, empty_channel()))),
    }
    Ok(EncodedPixels { canvas: (img.width, img.height), rect: rect.unwrap_or((0, 0, 0, 0)), data })
}

/// Compress a mask over the rectangle where it is not 0 (outside it reads the default, 0).
pub fn encode_mask(m: &Mask16) -> Result<EncodedMask> {
    check_size("layer mask", (m.width, m.height), m.data.len(), 1)?;
    let w = m.width as usize;
    // An all-hidden mask still needs a rectangle to say so: one texel.
    let rect = bounds(w, m.data.len(), |i| m.data.get(i).copied().unwrap_or(0) > 0).unwrap_or((0, 0, 1, 1));
    let rw = (rect.3 - rect.1) as usize;
    Ok(EncodedMask { canvas: (m.width, m.height), rect, data: coded(&crop_plane(&m.data, w, 1, 0, rect), rw)? })
}

fn mask_channel(m: Option<&EncodedMask>, canvas: (u32, u32)) -> Result<Option<(Rect, Vec<u8>)>> {
    match m {
        Some(m) if m.canvas != canvas => Err(invalid("a layer mask is not the canvas size")),
        Some(m) => Ok(Some((m.rect, m.data.clone()))),
        None => Ok(None),
    }
}

/// Layer records and channel data, bottom to top.
struct Records {
    records: Vec<u8>,
    channels: Vec<u8>,
    count: usize,
    next_id: u32,
}

impl Records {
    fn layer(&mut self, l: &Layer, section: Option<u32>, ch: Channels, comps: usize) -> Result<()> {
        self.count += 1;
        let id = if l.id != 0 && section != Some(3) {
            l.id
        } else {
            self.next_id = self.next_id.saturating_add(1);
            self.next_id
        };
        let r = &mut self.records;
        let (t, left, b, right) = ch.rect;
        r.i32(t);
        r.i32(left);
        r.i32(b);
        r.i32(right);
        let mut chans = ch.data;
        if let Some((_, data)) = &ch.mask {
            chans.push((-2, data.clone()));
        }
        r.u16(chans.len() as u16);
        for (cid, data) in &chans {
            r.i16(*cid);
            r.u32(len32(data.len())?);
        }
        r.bytes(b"8BIM");
        r.bytes(&l.blend);
        r.u8((l.opacity.clamp(0.0, 1.0) * 255.0).round() as u8);
        r.u8(0); // clipping: base
        // Flags: bit 1 hides the layer; bit 3 says bit 4 is meaningful (not set: pixel data matters).
        r.u8(if l.visible { 0 } else { 2 } | 8);
        r.u8(0);

        let mut extra = Vec::new();
        match &ch.mask {
            Some(((mt, ml, mb, mr), _)) => {
                extra.u32(20);
                extra.i32(*mt);
                extra.i32(*ml);
                extra.i32(*mb);
                extra.i32(*mr);
                extra.u8(0); // default colour: hidden outside the rectangle
                extra.u8(0); // flags: position relative to layer, enabled
                extra.u16(0);
            }
            None => extra.u32(0),
        }
        extra.u32(0); // blending ranges
        // Pascal name, padded to a multiple of 4 (ASCII; the full name is in `luni`).
        let ascii: Vec<u8> = l.name.chars().map(|c| if c.is_ascii() && !c.is_ascii_control() { c as u8 } else { b'?' }).take(255).collect();
        extra.u8(ascii.len() as u8);
        extra.bytes(&ascii);
        extra.bytes(&[0; 3][..(4 - (ascii.len() + 1) % 4) % 4]);
        let mut luni = Vec::new();
        put_unicode(&mut luni, &l.name, false);
        tagged(&mut extra, b"luni", &luni)?;
        tagged(&mut extra, b"lyid", &id.to_be_bytes())?;
        if let Some(kind) = section {
            let mut s = Vec::new();
            s.u32(kind);
            if kind != 3 {
                s.bytes(b"8BIM");
                s.bytes(&l.blend);
            }
            tagged(&mut extra, b"lsct", &s)?;
        }
        if comps > 0 && section != Some(3) {
            tagged(&mut extra, b"shmd", &cmls(l, id, comps)?)?;
        }
        r.u32(len32(extra.len())?);
        r.bytes(&extra);
        for (_, data) in chans {
            self.channels.bytes(&data);
        }
        Ok(())
    }

    fn nodes(&mut self, nodes: &[Layer], canvas: (u32, u32), comps: usize, depth: usize) -> Result<()> {
        if depth > MAX_DEPTH {
            return Err(invalid(format!("groups are nested more than {MAX_DEPTH} deep")));
        }
        for l in nodes.iter().rev() {
            match &l.kind {
                Kind::Pixels(px) => {
                    if px.canvas != canvas {
                        return Err(invalid(format!("layer \"{}\" is not the canvas size", l.name)));
                    }
                    let ch = Channels { rect: px.rect, data: px.data.clone(), mask: mask_channel(l.mask.as_ref(), canvas)? };
                    self.layer(l, None, ch, comps)?;
                }
                Kind::Group(children) => {
                    let end = Layer {
                        id: 0,
                        name: "</Layer group>".into(),
                        visible: true,
                        opacity: 1.0,
                        blend: *b"norm",
                        mask: None,
                        comps: Vec::new(),
                        kind: Kind::Group(Vec::new()),
                    };
                    self.layer(&end, Some(3), empty(), comps)?;
                    self.nodes(children, canvas, comps, depth + 1)?;
                    self.layer(l, Some(1), Channels { mask: mask_channel(l.mask.as_ref(), canvas)?, ..empty() }, comps)?;
                }
            }
        }
        Ok(())
    }
}

fn empty() -> Channels {
    Channels { rect: (0, 0, 0, 0), data: [-1i16, 0, 1, 2].map(|id| (id, empty_channel())).to_vec(), mask: None }
}

/// The `shmd` block holding a layer's visibility in each comp (`cmls`).
fn cmls(l: &Layer, id: u32, comps: usize) -> Result<Vec<u8>> {
    let visible_in = |i: usize| l.comps.get(i).copied().unwrap_or(l.visible);
    let mut settings: Vec<Value> = (0..comps)
        .map(|i| Value::Object("null", vec![("enab", Value::Bool(visible_in(i))), ("compList", Value::List(vec![Value::Long(i as i32 + 1)]))]))
        .collect();
    // Comp 0 is the last document state: the layer as saved.
    settings.push(Value::Object("null", vec![("enab", Value::Bool(l.visible)), ("compList", Value::List(vec![Value::Long(0)]))]));
    let data = versioned("null", &[("LyrI", Value::Long(id as i32)), ("layerSettings", Value::List(settings))]);
    let mut out = Vec::new();
    out.u32(1);
    out.bytes(b"8BIM");
    out.bytes(b"cmls");
    out.u8(1);
    out.bytes(&[0, 0, 0]);
    out.u32(len32(data.len())?);
    out.bytes(&data);
    Ok(out)
}

fn comps_resource(names: &[String]) -> Vec<u8> {
    let list = names
        .iter()
        .enumerate()
        .map(|(i, n)| Value::Object("Comp", vec![("Nm  ", Value::Text(n.clone())), ("compID", Value::Long(i as i32 + 1)), ("capturedInfo", Value::Long(1))]))
        .collect();
    versioned("CompList", &[("list", Value::List(list)), ("lastAppliedComp", Value::Long(1))])
}

/// Write `doc` as a 16-bit RGB PSD. Every layer and mask must be the merged image's size.
pub fn write_psd(doc: &Document) -> Result<Vec<u8>> {
    let m = &doc.merged;
    if m.width == 0 || m.height == 0 || m.width > MAX_SIDE || m.height > MAX_SIDE || m.data.len() != m.width as usize * m.height as usize * 4 {
        return Err(invalid(format!("{}×{} is not a size a PSD is written at (at most {MAX_SIDE} a side)", m.width, m.height)));
    }
    let canvas = (m.width, m.height);
    let mut out = Vec::new();
    out.bytes(b"8BPS");
    out.u16(1);
    out.bytes(&[0; 6]);
    out.u16(4); // RGB + the merged transparency
    out.u32(m.height);
    out.u32(m.width);
    out.u16(16);
    out.u16(3); // RGB
    out.u32(0); // colour mode data

    let mut resources = Vec::new();
    if !doc.comps.is_empty() {
        let data = comps_resource(&doc.comps);
        resources.bytes(b"8BIM");
        resources.u16(LAYER_COMPS);
        resources.bytes(&[0, 0]); // empty Pascal name, padded
        resources.u32(len32(data.len())?);
        resources.bytes(&data);
        if data.len() % 2 == 1 {
            resources.u8(0);
        }
    }
    out.u32(len32(resources.len())?);
    out.bytes(&resources);

    let mut recs = Records { records: Vec::new(), channels: Vec::new(), count: 0, next_id: max_id(&doc.layers, 0) };
    recs.nodes(&doc.layers, canvas, doc.comps.len(), 1)?;
    if recs.count > MAX_RECORDS {
        return Err(invalid(format!("{} layers and groups; a document holds at most {MAX_RECORDS}", recs.count)));
    }
    let mut info = Vec::new();
    // Negative: the merged image's first alpha channel is its transparency.
    info.i16(-(recs.count as i16));
    info.bytes(&recs.records);
    info.bytes(&recs.channels);
    if info.len() % 2 == 1 {
        info.u8(0);
    }
    let mut section = Vec::new();
    section.u32(len32(info.len())?);
    section.bytes(&info);
    section.u32(0); // global layer mask info
    out.u32(len32(section.len())?);
    out.bytes(&section);

    // The merged image, planar R, G, B, A, uncompressed: readers only show it as a preview.
    out.u16(0);
    for comp in [0usize, 1, 2, 3] {
        for px in m.data.as_chunks::<4>().0 {
            out.u16(px[comp]);
        }
    }
    Ok(out)
}

/// The largest id given in `layers`.
fn max_id(layers: &[Layer], depth: usize) -> u32 {
    layers
        .iter()
        .map(|l| match &l.kind {
            Kind::Group(children) if depth < MAX_DEPTH => l.id.max(max_id(children, depth + 1)),
            _ => l.id,
        })
        .max()
        .unwrap_or(0)
}

// ------------------------------------------------------------------------------------------
// Reading

/// A layer or group read back, children top-first.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadLayer {
    /// The layer's id (`lyid`), when the file has one; editors keep it across saves.
    pub id: Option<u32>,
    pub name: String,
    pub visible: bool,
    pub opacity: f32,
    pub blend: [u8; 4],
    /// The layer mask, when enabled.
    pub mask: Option<ReadMask>,
    pub kind: ReadKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum ReadKind {
    /// Straight-alpha pixels of the layer's rectangle, whose top-left corner is at `(x, y)` on the
    /// canvas (possibly off it).
    Pixels {
        x: i32,
        y: i32,
        image: Rgba16,
    },
    Group(Vec<ReadLayer>),
}

/// A layer mask over its rectangle; outside it, the mask is `default` (0 or 65535).
#[derive(Clone, Debug, PartialEq)]
pub struct ReadMask {
    pub x: i32,
    pub y: i32,
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
    pub default: u16,
}

impl ReadMask {
    /// The mask at canvas texel (`x`, `y`), 0..65535.
    pub fn at(&self, x: i32, y: i32) -> u16 {
        let (mx, my) = (i64::from(x) - i64::from(self.x), i64::from(y) - i64::from(self.y));
        if mx < 0 || my < 0 || mx >= i64::from(self.width) || my >= i64::from(self.height) {
            return self.default;
        }
        self.data.get(my as usize * self.width as usize + mx as usize).copied().unwrap_or(self.default)
    }
}

/// A document read back.
#[derive(Clone, Debug, PartialEq)]
pub struct ReadDocument {
    pub width: u32,
    pub height: u32,
    pub layers: Vec<ReadLayer>,
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self.p.checked_add(n).filter(|e| *e <= self.b.len()).ok_or_else(|| malformed("truncated file"))?;
        let s = self.b.get(self.p..end).ok_or_else(|| malformed("truncated file"))?;
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?.first().copied().unwrap_or(0))
    }
    fn u16(&mut self) -> Result<u16> {
        let s = self.take(2)?;
        Ok(u16::from_be_bytes([s[0], s[1]]))
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(self.u16()? as i16)
    }
    fn u32(&mut self) -> Result<u32> {
        let s = self.take(4)?;
        Ok(u32::from_be_bytes([s[0], s[1], s[2], s[3]]))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(self.u32()? as i32)
    }
    fn skip(&mut self, n: usize) -> Result<()> {
        self.take(n).map(drop)
    }
}

struct Record {
    name: String,
    rect: (i32, i32, i32, i32),
    channels: Vec<(i16, usize)>,
    blend: [u8; 4],
    opacity: u8,
    visible: bool,
    mask: Option<(Rect, u8, bool)>,
    section: Option<u32>,
    id: Option<u32>,
}

fn rect_size((t, l, b, r): (i32, i32, i32, i32)) -> Result<(usize, usize)> {
    let w = i64::from(r) - i64::from(l);
    let h = i64::from(b) - i64::from(t);
    if w < 0 || h < 0 || w > i64::from(MAX_SIDE) * 2 || h > i64::from(MAX_SIDE) * 2 {
        return Err(malformed("a layer rectangle is out of range"));
    }
    Ok((w as usize, h as usize))
}

fn read_record(r: &mut Reader) -> Result<Record> {
    let rect = (r.i32()?, r.i32()?, r.i32()?, r.i32()?);
    let n = r.u16()? as usize;
    if n > 56 {
        return Err(malformed(format!("a layer has {n} channels")));
    }
    let mut channels = Vec::with_capacity(n);
    for _ in 0..n {
        let id = r.i16()?;
        let len = r.u32()? as usize;
        channels.push((id, len));
    }
    if r.take(4)? != b"8BIM" {
        return Err(malformed("bad blend mode signature"));
    }
    let mut blend = [0u8; 4];
    blend.copy_from_slice(r.take(4)?);
    let opacity = r.u8()?;
    let _clipping = r.u8()?;
    let flags = r.u8()?;
    let _ = r.u8()?;
    let extra_len = r.u32()? as usize;
    let extra = r.take(extra_len)?;
    let mut e = Reader { b: extra, p: 0 };
    let mask_len = e.u32()? as usize;
    let mask = if mask_len >= 18 {
        let m = e.take(mask_len)?;
        let mut mr = Reader { b: m, p: 0 };
        let mrect = (mr.i32()?, mr.i32()?, mr.i32()?, mr.i32()?);
        let default = mr.u8()?;
        let mflags = mr.u8()?;
        // Bit 1 disables the mask.
        Some((mrect, default, mflags & 2 == 0))
    } else {
        e.skip(mask_len)?;
        None
    };
    let ranges = e.u32()? as usize;
    e.skip(ranges)?;
    let name_len = e.u8()? as usize;
    let mut name: String = e.take(name_len)?.iter().map(|b| *b as char).collect();
    let pad = (4 - (name_len + 1) % 4) % 4;
    e.skip(pad.min(e.b.len().saturating_sub(e.p)))?;
    let mut section = None;
    let mut id = None;
    while e.b.len().saturating_sub(e.p) >= 12 {
        let sig = e.take(4)?;
        if sig != b"8BIM" && sig != b"8B64" {
            break;
        }
        let mut key = [0u8; 4];
        key.copy_from_slice(e.take(4)?);
        let len = e.u32()? as usize;
        let data = e.take(len.min(e.b.len().saturating_sub(e.p)))?;
        match &key {
            b"luni" => {
                let mut d = Reader { b: data, p: 0 };
                let units = d.u32()? as usize;
                let raw = d.take(units.saturating_mul(2).min(data.len().saturating_sub(4)))?;
                let units: Vec<u16> = raw.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).take_while(|u| *u != 0).collect();
                name = String::from_utf16_lossy(&units);
            }
            b"lyid" if data.len() >= 4 => id = Some(u32::from_be_bytes([data[0], data[1], data[2], data[3]])),
            b"lsct" | b"lsdk" => {
                let mut d = Reader { b: data, p: 0 };
                section = Some(d.u32()?);
                if data.len() >= 12 {
                    blend.copy_from_slice(data.get(8..12).unwrap_or(b"norm"));
                }
            }
            _ => {}
        }
    }
    Ok(Record { name, rect, channels, blend, opacity, visible: flags & 2 == 0, mask, section, id })
}

/// Decode one channel of `w`×`h` samples at `depth` bits into 16-bit samples.
fn decode_channel(data: &[u8], w: usize, h: usize, depth: u16, budget: &mut u64) -> Result<Vec<u16>> {
    let n = w.checked_mul(h).ok_or_else(|| malformed("channel too large"))?;
    if n == 0 {
        return Ok(Vec::new());
    }
    let bps = usize::from(depth / 8);
    let need = (n * bps) as u64;
    *budget = budget.checked_sub(need).ok_or_else(|| FormatError::TooLarge("the document's layers are too large to read back".into()))?;
    let mut r = Reader { b: data, p: 0 };
    let code = r.u16()?;
    let rest = data.get(2..).unwrap_or(&[]);
    let row = w * bps;
    let bytes: Vec<u8> = match code {
        0 => rest.get(..n * bps).ok_or_else(|| malformed("raw channel truncated"))?.to_vec(),
        1 => {
            let mut counts = Vec::with_capacity(h);
            for _ in 0..h {
                counts.push(r.u16()? as usize);
            }
            let mut out = Vec::with_capacity(n * bps);
            for c in counts {
                let src = r.take(c)?;
                unpack_bits(src, row, &mut out)?;
            }
            out
        }
        2 | 3 => {
            let mut out = Vec::with_capacity(n * bps);
            flate2::read::ZlibDecoder::new(rest).take(need + 1).read_to_end(&mut out).map_err(|e| malformed(format!("ZIP channel: {e}")))?;
            if out.len() as u64 != need {
                return Err(malformed("ZIP channel has the wrong size"));
            }
            if code == 3 {
                unpredict(&mut out, row, bps)?;
            }
            out
        }
        c => return Err(malformed(format!("unknown compression {c}"))),
    };
    Ok(match bps {
        1 => bytes.iter().map(|b| u16::from(*b) * 257).collect(),
        _ => bytes.as_chunks::<2>().0.iter().map(|c| u16::from_be_bytes(*c)).collect(),
    })
}

fn unpack_bits(src: &[u8], row: usize, out: &mut Vec<u8>) -> Result<()> {
    let start = out.len();
    let mut i = 0;
    while i < src.len() && out.len() - start < row {
        let n = src[i] as i8;
        i += 1;
        if n >= 0 {
            let k = n as usize + 1;
            out.extend_from_slice(src.get(i..i + k).ok_or_else(|| malformed("RLE run truncated"))?);
            i += k;
        } else if n != -128 {
            let v = *src.get(i).ok_or_else(|| malformed("RLE run truncated"))?;
            out.extend(std::iter::repeat_n(v, (1 - n as isize) as usize));
            i += 1;
        }
    }
    if out.len() - start != row {
        return Err(malformed("RLE row has the wrong length"));
    }
    Ok(())
}

fn unpredict(bytes: &mut [u8], row: usize, bps: usize) -> Result<()> {
    for r in bytes.chunks_mut(row.max(1)) {
        match bps {
            1 => {
                for i in 1..r.len() {
                    r[i] = r[i].wrapping_add(r[i - 1]);
                }
            }
            2 => {
                let mut prev = 0u16;
                for c in r.as_chunks_mut::<2>().0 {
                    let v = u16::from_be_bytes(*c).wrapping_add(prev);
                    c.copy_from_slice(&v.to_be_bytes());
                    prev = v;
                }
            }
            _ => return Err(FormatError::Unsupported("32-bit PSD layers".into())),
        }
    }
    Ok(())
}

/// Read the layers of a PSD (version 1, 8 or 16-bit RGB). Groups come back nested.
pub fn read_psd(bytes: &[u8]) -> Result<ReadDocument> {
    let mut r = Reader { b: bytes, p: 0 };
    if r.take(4)? != b"8BPS" {
        return Err(malformed("not a PSD file"));
    }
    match r.u16()? {
        1 => {}
        2 => return Err(FormatError::Unsupported("large documents (PSB); save as PSD".into())),
        v => return Err(malformed(format!("version {v}"))),
    }
    r.skip(6)?;
    let _channels = r.u16()?;
    let height = r.u32()?;
    let width = r.u32()?;
    let depth = r.u16()?;
    let mode = r.u16()?;
    if width == 0 || height == 0 || width > MAX_SIDE || height > MAX_SIDE {
        return Err(FormatError::TooLarge(format!("{width}×{height}")));
    }
    if depth != 8 && depth != 16 {
        return Err(FormatError::Unsupported(format!("{depth}-bit documents; save as 8 or 16-bit")));
    }
    if mode != 3 {
        return Err(FormatError::Unsupported("documents that are not RGB; convert to RGB first".into()));
    }
    let n = r.u32()? as usize;
    r.skip(n)?;
    let n = r.u32()? as usize;
    r.skip(n)?;
    let section_len = r.u32()? as usize;
    let section = r.take(section_len)?;
    let mut s = Reader { b: section, p: 0 };
    let info_len = s.u32()? as usize;
    let info = if info_len > 0 { s.take(info_len)? } else { &[][..] };
    let info = if info.is_empty() { find_lr16(section, s.p)? } else { info };
    let mut budget = MAX_DECODED;
    let mut flat: Vec<(Record, Option<ReadKind>, Option<ReadMask>)> = Vec::new();
    if !info.is_empty() {
        let mut ir = Reader { b: info, p: 0 };
        let count = ir.i16()?.unsigned_abs() as usize;
        if count > MAX_RECORDS {
            return Err(FormatError::TooLarge(format!("{count} layers")));
        }
        let mut records = Vec::with_capacity(count);
        for _ in 0..count {
            records.push(read_record(&mut ir)?);
        }
        for rec in records {
            let (rw, rh) = rect_size(rec.rect)?;
            let mut planes: [Option<Vec<u16>>; 4] = [None, None, None, None];
            let mut mask_plane = None;
            for &(id, len) in &rec.channels {
                let data = ir.take(len)?;
                match id {
                    -1..=2 => {
                        let idx = if id == -1 { 3 } else { id as usize };
                        if let Some(slot) = planes.get_mut(idx) {
                            *slot = Some(decode_channel(data, rw, rh, depth, &mut budget)?);
                        }
                    }
                    -2 => {
                        if let Some((mrect, _, _)) = rec.mask {
                            let (mw, mh) = rect_size(mrect)?;
                            mask_plane = Some(decode_channel(data, mw, mh, depth, &mut budget)?);
                        }
                    }
                    _ => {}
                }
            }
            let pixels = if rec.section.is_none() || rec.section == Some(0) {
                // The interleaved image is allocated whatever channels the record has.
                let need = (rw as u64).saturating_mul(rh as u64).saturating_mul(8);
                budget = budget.checked_sub(need).ok_or_else(|| FormatError::TooLarge("the document's layers are too large to read back".into()))?;
                Some(ReadKind::Pixels { x: rec.rect.1, y: rec.rect.0, image: interleave(&planes, (rw, rh)) })
            } else {
                None
            };
            let mask = match rec.mask {
                Some((mrect, default, true)) => {
                    let (mw, mh) = rect_size(mrect)?;
                    let data = mask_plane.filter(|p| p.len() == mw * mh).unwrap_or_default();
                    let (mw, mh) = if data.is_empty() { (0, 0) } else { (mw as u32, mh as u32) };
                    Some(ReadMask { x: mrect.1, y: mrect.0, width: mw, height: mh, data, default: u16::from(default) * 257 })
                }
                _ => None,
            };
            flat.push((rec, pixels, mask));
        }
    }
    let mut it = flat.into_iter().rev().peekable();
    let layers = nest(&mut it, 0)?;
    Ok(ReadDocument { width, height, layers })
}

/// The `Lr16` block after the (empty) layer info, where some writers put 16-bit layers.
fn find_lr16(section: &[u8], mut p: usize) -> Result<&[u8]> {
    // Skip the global layer mask info.
    let mut r = Reader { b: section, p };
    if let Ok(n) = r.u32() {
        p = r.p + n as usize;
    }
    let mut r = Reader { b: section, p };
    while section.len().saturating_sub(r.p) >= 12 {
        let sig = r.take(4)?;
        if sig != b"8BIM" && sig != b"8B64" {
            break;
        }
        let key = r.take(4)?;
        let len = r.u32()? as usize;
        let data = r.take(len)?;
        if key == b"Lr16" {
            return Ok(data);
        }
        if !len.is_multiple_of(4) {
            r.skip((4 - len % 4).min(section.len().saturating_sub(r.p)))?;
        }
    }
    Ok(&[])
}

/// Planes of a `rw`×`rh` rectangle as straight RGBA (a missing alpha plane is opaque).
fn interleave(planes: &[Option<Vec<u16>>; 4], (rw, rh): (usize, usize)) -> Rgba16 {
    let n = rw * rh;
    let mut data = Vec::with_capacity(n * 4);
    for i in 0..n {
        for (k, plane) in planes.iter().enumerate() {
            data.push(match plane {
                Some(p) => p.get(i).copied().unwrap_or(0),
                None if k == 3 => 65535,
                None => 0,
            });
        }
    }
    Rgba16 { width: rw as u32, height: rh as u32, data }
}

/// Records top-first into a tree: a folder record opens a group that its divider closes.
fn nest(it: &mut std::iter::Peekable<impl Iterator<Item = (Record, Option<ReadKind>, Option<ReadMask>)>>, depth: usize) -> Result<Vec<ReadLayer>> {
    if depth > MAX_DEPTH {
        return Err(malformed(format!("groups nested more than {MAX_DEPTH} deep")));
    }
    let mut out = Vec::new();
    while let Some((rec, pixels, mask)) = it.next() {
        let base = |kind| ReadLayer {
            id: rec.id,
            name: rec.name.clone(),
            visible: rec.visible,
            opacity: f32::from(rec.opacity) / 255.0,
            blend: rec.blend,
            mask: mask.clone(),
            kind,
        };
        match rec.section {
            Some(3) => return Ok(out),
            Some(1) | Some(2) => {
                let children = nest(it, depth + 1)?;
                out.push(base(ReadKind::Group(children)));
            }
            _ => out.push(base(pixels.unwrap_or(ReadKind::Pixels { x: 0, y: 0, image: Rgba16 { width: 0, height: 0, data: Vec::new() } }))),
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: u32, h: u32, rect: (u32, u32, u32, u32), px: [u16; 4]) -> Rgba16 {
        let mut img = Rgba16::transparent(w, h);
        for y in rect.1..rect.1 + rect.3 {
            for x in rect.0..rect.0 + rect.2 {
                let i = ((y * w + x) * 4) as usize;
                img.data[i..i + 4].copy_from_slice(&px);
            }
        }
        img
    }

    fn px(img: Rgba16) -> Kind {
        Kind::Pixels(encode_pixels(&img).unwrap())
    }

    fn layer(name: &str, kind: Kind) -> Layer {
        Layer { id: 0, name: name.into(), visible: true, opacity: 1.0, blend: *b"norm", mask: None, comps: Vec::new(), kind }
    }

    #[test]
    fn groups_masks_names_and_blends_round_trip() {
        let (w, h) = (16, 8);
        let mut mask = Mask16 { width: w, height: h, data: vec![0; (w * h) as usize] };
        for x in 0..8 {
            mask.data[(2 * w + x) as usize] = 40000;
        }
        let base =
            Layer { opacity: 0.5, blend: *b"mul ", comps: vec![true, false], ..layer("Base color", px(solid(w, h, (1, 1, 4, 3), [100, 200, 300, 65535]))) };
        let rough = Layer { visible: false, comps: vec![false, true], ..layer("Roughness", px(solid(w, h, (0, 0, 16, 8), [5, 5, 5, 65535]))) };
        let group = Layer { id: 42, blend: *b"pass", mask: Some(encode_mask(&mask).unwrap()), ..layer("Dirt ✓", Kind::Group(vec![base, rough])) };
        let doc = Document {
            layers: vec![group, layer("Layer 1", px(Rgba16::transparent(w, h)))],
            merged: solid(w, h, (0, 0, 16, 8), [1, 2, 3, 65535]),
            comps: vec!["Base color".into(), "Roughness".into()],
        };
        let bytes = write_psd(&doc).unwrap();
        let back = read_psd(&bytes).unwrap();
        assert_eq!((back.width, back.height), (w, h));
        assert_eq!(back.layers.len(), 2);
        let g = &back.layers[0];
        assert_eq!(g.name, "Dirt ✓");
        assert_eq!(g.id, Some(42), "a given id is kept");
        assert_eq!(&g.blend, b"pass");
        let m = g.mask.as_ref().unwrap();
        for y in 0..h as i32 {
            for x in 0..w as i32 {
                assert_eq!(m.at(x, y), mask.data[(y * w as i32 + x) as usize], "mask at {x},{y}");
            }
        }
        let ReadKind::Group(children) = &g.kind else { panic!("{g:?}") };
        assert_eq!(children.iter().map(|c| c.name.as_str()).collect::<Vec<_>>(), ["Base color", "Roughness"]);
        assert_eq!(&children[0].blend, b"mul ");
        assert!((children[0].opacity - 0.5).abs() < 0.01);
        assert!(!children[1].visible);
        // Layers without an id get ones above every id given.
        let ids = [children[0].id, children[1].id, back.layers[1].id];
        assert!(ids.iter().all(|i| i.is_some_and(|i| i > 42)) && ids[0] != ids[1] && ids[1] != ids[2], "{ids:?}");
        let ReadKind::Pixels { x, y, image } = &children[0].kind else { panic!() };
        assert_eq!((*x, *y, image.width, image.height), (1, 1, 4, 3), "cropped to the layer's pixels");
        assert_eq!(&image.data[..4], &[100, 200, 300, 65535]);
        // The comp list is in the resources.
        assert!(bytes.windows(8).any(|x| x == b"CompList"));
        assert!(bytes.windows(4).any(|x| x == b"cmls"));
    }

    #[test]
    fn rle_and_8_bit_layers_read() {
        // A hand-made 8-bit 2×1 document with one RLE layer.
        let mut f = Vec::new();
        f.bytes(b"8BPS");
        f.u16(1);
        f.bytes(&[0; 6]);
        f.u16(3);
        f.u32(1);
        f.u32(2);
        f.u16(8);
        f.u16(3);
        f.u32(0);
        f.u32(0);
        let mut rec = Vec::new();
        rec.i32(0);
        rec.i32(0);
        rec.i32(1);
        rec.i32(2);
        rec.u16(4);
        // Each channel: compression 1, one row count (2 bytes), a run "two of v": [-1, v].
        for id in [-1i16, 0, 1, 2] {
            rec.i16(id);
            rec.u32(6);
        }
        rec.bytes(b"8BIMnorm");
        rec.u8(255);
        rec.u8(0);
        rec.u8(0);
        rec.u8(0);
        rec.u32(4 + 4 + 4);
        rec.u32(0);
        rec.u32(0);
        rec.bytes(&[1, b'A', 0, 0]);
        for v in [255u8, 10, 20, 30] {
            rec.u16(1);
            rec.u16(2);
            rec.u8(0xFF);
            rec.u8(v);
        }
        let mut info = Vec::new();
        info.i16(1);
        info.bytes(&rec);
        let mut section = Vec::new();
        section.u32(info.len() as u32);
        section.bytes(&info);
        section.u32(0);
        f.u32(section.len() as u32);
        f.bytes(&section);
        let doc = read_psd(&f).unwrap();
        let ReadKind::Pixels { image, .. } = &doc.layers[0].kind else { panic!() };
        assert_eq!(doc.layers[0].name, "A");
        assert_eq!(&image.data[4..8], &[10 * 257, 20 * 257, 30 * 257, 65535]);
    }

    #[test]
    fn a_huge_layer_without_channels_is_refused_before_it_is_allocated() {
        // One pixel layer claiming 30000×30000 texels but storing no channels: nothing to decode,
        // yet its image would take 7 GB.
        let mut rec = Vec::new();
        for v in [0, 0, 30000, 30000] {
            rec.i32(v);
        }
        rec.u16(0);
        rec.bytes(b"8BIMnorm");
        rec.bytes(&[255, 0, 8, 0]);
        rec.u32(12);
        rec.u32(0); // no mask
        rec.u32(0); // no blending ranges
        rec.bytes(&[0, 0, 0, 0]); // empty name, padded
        let mut info = Vec::new();
        info.i16(1);
        info.bytes(&rec);
        let mut section = Vec::new();
        section.u32(info.len() as u32);
        section.bytes(&info);
        section.u32(0);
        let mut f = Vec::new();
        f.bytes(b"8BPS");
        f.u16(1);
        f.bytes(&[0; 6]);
        f.u16(4);
        f.u32(1);
        f.u32(1);
        f.u16(16);
        f.u16(3);
        f.u32(0);
        f.u32(0);
        f.u32(section.len() as u32);
        f.bytes(&section);
        f.u16(0);
        f.bytes(&[0; 8]);
        assert!(matches!(read_psd(&f), Err(FormatError::TooLarge(_))), "{:?}", read_psd(&f).map(|d| d.layers.len()));
    }

    #[test]
    fn hostile_input_is_an_error() {
        assert!(read_psd(b"").is_err());
        assert!(read_psd(b"8BPS\x00\x02").is_err());
        assert!(read_psd(b"not a psd at all, really not").is_err());
        let doc = Document { layers: vec![], merged: Rgba16::transparent(4, 4), comps: vec![] };
        let mut bytes = write_psd(&doc).unwrap();
        bytes.truncate(bytes.len() / 2);
        let _ = read_psd(&bytes);
        let bad = Document { layers: vec![], merged: Rgba16 { width: 0, height: 0, data: vec![] }, comps: vec![] };
        assert!(write_psd(&bad).is_err());
        assert!(encode_pixels(&Rgba16 { width: 2, height: 2, data: vec![0; 3] }).is_err());
        assert!(encode_mask(&Mask16 { width: 0, height: 0, data: vec![] }).is_err());
        let wrong = Document { layers: vec![layer("x", px(Rgba16::transparent(2, 2)))], merged: Rgba16::transparent(4, 4), comps: vec![] };
        assert!(write_psd(&wrong).is_err());
        let mut deep = layer("l", px(Rgba16::transparent(4, 4)));
        for _ in 0..=MAX_DEPTH {
            deep = layer("g", Kind::Group(vec![deep]));
        }
        assert!(write_psd(&Document { layers: vec![deep], merged: Rgba16::transparent(4, 4), comps: vec![] }).is_err());
        // Every prefix of a valid file is an error or a document, never a panic.
        let full = write_psd(&Document {
            layers: vec![Layer {
                mask: Some(encode_mask(&Mask16 { width: 4, height: 4, data: vec![65535; 16] }).unwrap()),
                ..layer("g", Kind::Group(vec![layer("p", px(solid(4, 4, (0, 0, 4, 4), [1, 2, 3, 65535])))]))
            }],
            merged: Rgba16::transparent(4, 4),
            comps: vec!["a".into()],
        })
        .unwrap();
        for n in 0..full.len() {
            let _ = read_psd(&full[..n]);
        }
    }
}
