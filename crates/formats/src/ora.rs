//! OpenRaster (`.ora`) documents: a layer tree written for an image editor, and the merged image
//! read back after the editor saved it.
//!
//! Written from the OpenRaster specification (openraster.org): a zip whose first entry is the
//! stored `mimetype` (`image/openraster`), a `stack.xml` describing nested `<stack>` and
//! `<layer>` elements top-first, one PNG per layer, `mergedimage.png` and
//! `Thumbnails/thumbnail.png`. Layers are cropped to their pixels and placed with `x`/`y`.

use std::io::{Cursor, Read, Write};

use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use crate::{FormatError, Result};

/// The MIME type an OpenRaster archive declares in its first entry.
pub const MIMETYPE: &str = "image/openraster";
/// The file extension.
pub const ORA_EXT: &str = "ora";
/// Deepest group nesting written (deeper trees are an error, never a stack overflow).
pub const MAX_DEPTH: usize = 64;
/// Largest canvas side written or read back.
pub const MAX_SIDE: u32 = 1 << 15;
/// Largest `mergedimage.png` read back.
const MAX_MERGED_BYTES: u64 = 512 << 20;
/// Longest side of the thumbnail.
const THUMBNAIL_SIDE: u32 = 256;

/// Straight-alpha RGBA, 16 bits per component, row by row.
#[derive(Clone, Debug, PartialEq)]
pub struct Rgba16 {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u16>,
}

impl Rgba16 {
    /// A fully transparent image.
    pub fn transparent(width: u32, height: u32) -> Rgba16 {
        Rgba16 { width, height, data: vec![0; width as usize * height as usize * 4] }
    }

    fn check(&self, what: &str) -> Result<()> {
        if self.width == 0 || self.height == 0 || self.width > MAX_SIDE || self.height > MAX_SIDE {
            return Err(FormatError::Invalid(format!("{what}: {}×{} is not a size OpenRaster can be written at", self.width, self.height)));
        }
        if self.data.len() != self.width as usize * self.height as usize * 4 {
            return Err(FormatError::Invalid(format!("{what}: {} components for a {}×{} image", self.data.len(), self.width, self.height)));
        }
        Ok(())
    }

    /// The smallest rectangle (x, y, w, h) holding every pixel with alpha; None when empty.
    fn bounds(&self) -> Option<(u32, u32, u32, u32)> {
        let w = self.width as usize;
        let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0usize, 0usize);
        for (i, px) in self.data.as_chunks::<4>().0.iter().enumerate() {
            if px[3] == 0 {
                continue;
            }
            let (x, y) = (i % w, i / w);
            x0 = x0.min(x);
            y0 = y0.min(y);
            x1 = x1.max(x);
            y1 = y1.max(y);
        }
        (x0 != usize::MAX).then(|| (x0 as u32, y0 as u32, (x1 - x0 + 1) as u32, (y1 - y0 + 1) as u32))
    }

    fn crop(&self, (x, y, w, h): (u32, u32, u32, u32)) -> Rgba16 {
        let mut out = Vec::with_capacity(w as usize * h as usize * 4);
        for row in y..y + h {
            let start = (row as usize * self.width as usize + x as usize) * 4;
            out.extend_from_slice(self.data.get(start..start + w as usize * 4).unwrap_or(&[]));
        }
        Rgba16 { width: w, height: h, data: out }
    }
}

/// One element of the layer tree.
#[derive(Clone, Debug, PartialEq)]
pub struct Node {
    pub name: String,
    pub visible: bool,
    /// 0..1.
    pub opacity: f32,
    /// An OpenRaster `composite-op` (`svg:src-over`, `svg:multiply`, ...).
    pub composite_op: &'static str,
    pub kind: NodeKind,
}

#[derive(Clone, Debug, PartialEq)]
pub enum NodeKind {
    /// Pixels the size of the canvas (cropped when written).
    Layer(Rgba16),
    /// A group, children top-first. `pass_through` composites its children with what is below.
    Stack { children: Vec<Node>, pass_through: bool },
}

/// Write an OpenRaster document of `merged`'s size: `nodes` top-first, `merged` as the merged
/// image (and, scaled down, the thumbnail). Every layer must be the canvas size.
pub fn write_ora(nodes: &[Node], merged: &Rgba16) -> Result<Vec<u8>> {
    merged.check("merged image")?;
    let mut zip = ZipWriter::new(Cursor::new(Vec::new()));
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    // PNGs are already compressed.
    let deflate = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated);
    entry(&mut zip, "mimetype", MIMETYPE.as_bytes(), stored)?;

    let mut xml = format!(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n<image version=\"0.0.6\" w=\"{}\" h=\"{}\" xres=\"72\" yres=\"72\">\n<stack>\n",
        merged.width, merged.height
    );
    let mut next = 0usize;
    write_nodes(&mut zip, &mut xml, nodes, merged, 1, &mut next, stored)?;
    xml.push_str("</stack>\n</image>\n");
    entry(&mut zip, "stack.xml", xml.as_bytes(), deflate)?;
    entry(&mut zip, "mergedimage.png", &png16(merged)?, stored)?;
    entry(&mut zip, "Thumbnails/thumbnail.png", &thumbnail(merged)?, stored)?;
    let out = zip.finish().map_err(zip_err)?;
    Ok(out.into_inner())
}

fn write_nodes(
    zip: &mut ZipWriter<Cursor<Vec<u8>>>,
    xml: &mut String,
    nodes: &[Node],
    canvas: &Rgba16,
    depth: usize,
    next: &mut usize,
    opts: SimpleFileOptions,
) -> Result<()> {
    if depth > MAX_DEPTH {
        return Err(FormatError::Invalid(format!("groups are nested more than {MAX_DEPTH} deep")));
    }
    for node in nodes {
        let attrs = format!(
            "name=\"{}\" visibility=\"{}\" opacity=\"{:.4}\" composite-op=\"{}\"",
            escape(&node.name),
            if node.visible { "visible" } else { "hidden" },
            if node.opacity.is_finite() { node.opacity.clamp(0.0, 1.0) } else { 1.0 },
            node.composite_op
        );
        match &node.kind {
            NodeKind::Layer(img) => {
                img.check(&node.name)?;
                if (img.width, img.height) != (canvas.width, canvas.height) {
                    return Err(FormatError::Invalid(format!("layer \"{}\" is not the canvas size", node.name)));
                }
                let rect = img.bounds().unwrap_or((0, 0, 1, 1));
                let src = format!("data/layer{next}.png");
                *next += 1;
                entry(zip, &src, &png16(&img.crop(rect))?, opts)?;
                xml.push_str(&format!("<layer {attrs} src=\"{src}\" x=\"{}\" y=\"{}\"/>\n", rect.0, rect.1));
            }
            NodeKind::Stack { children, pass_through } => {
                let isolation = if *pass_through { "auto" } else { "isolate" };
                xml.push_str(&format!("<stack {attrs} isolation=\"{isolation}\">\n"));
                write_nodes(zip, xml, children, canvas, depth + 1, next, opts)?;
                xml.push_str("</stack>\n");
            }
        }
    }
    Ok(())
}

fn entry(zip: &mut ZipWriter<Cursor<Vec<u8>>>, name: &str, bytes: &[u8], opts: SimpleFileOptions) -> Result<()> {
    zip.start_file(name, opts).map_err(zip_err)?;
    zip.write_all(bytes)?;
    Ok(())
}

fn zip_err(e: zip::result::ZipError) -> FormatError {
    FormatError::Malformed { kind: "OpenRaster", msg: e.to_string() }
}

/// XML attribute text: markup characters escaped, control characters dropped.
fn escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    out
}

/// `img` as a 16-bit RGBA PNG.
pub fn png16(img: &Rgba16) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let bytes: Vec<u8> = img.data.iter().flat_map(|v| v.to_ne_bytes()).collect();
    let mut out = Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(&bytes, img.width, img.height, image::ExtendedColorType::Rgba16)
        .map_err(|e| FormatError::Invalid(format!("PNG: {e}")))?;
    Ok(out.into_inner())
}

/// An 8-bit thumbnail at most [`THUMBNAIL_SIDE`] a side (nearest texel).
fn thumbnail(img: &Rgba16) -> Result<Vec<u8>> {
    use image::ImageEncoder;
    let scale = (img.width.max(img.height) as f32 / THUMBNAIL_SIDE as f32).max(1.0);
    let (w, h) = (((img.width as f32 / scale) as u32).max(1), ((img.height as f32 / scale) as u32).max(1));
    let mut px = Vec::with_capacity(w as usize * h as usize * 4);
    for y in 0..h {
        let sy = ((y as f32 * scale) as u32).min(img.height - 1);
        for x in 0..w {
            let sx = ((x as f32 * scale) as u32).min(img.width - 1);
            let i = (sy as usize * img.width as usize + sx as usize) * 4;
            px.extend(img.data.get(i..i + 4).unwrap_or(&[0; 4]).iter().map(|v| (v >> 8) as u8));
        }
    }
    let mut out = Cursor::new(Vec::new());
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(&px, w, h, image::ExtendedColorType::Rgba8)
        .map_err(|e| FormatError::Invalid(format!("PNG: {e}")))?;
    Ok(out.into_inner())
}

/// The `mergedimage.png` of an OpenRaster archive, as PNG bytes. Errors when the archive is not
/// OpenRaster, has no merged image or holds one larger than the read cap.
pub fn read_merged(bytes: &[u8]) -> Result<Vec<u8>> {
    let malformed = |msg: String| FormatError::Malformed { kind: "OpenRaster", msg };
    let mut z = ZipArchive::new(Cursor::new(bytes)).map_err(|e| malformed(e.to_string()))?;
    let mut mime = String::new();
    z.by_name("mimetype").map_err(|_| malformed("no mimetype entry".into()))?.take(64).read_to_string(&mut mime).map_err(|e| malformed(e.to_string()))?;
    if mime.trim() != MIMETYPE {
        return Err(malformed(format!("mimetype is `{}`, not {MIMETYPE}", mime.trim())));
    }
    let f = z.by_name("mergedimage.png").map_err(|_| malformed("no mergedimage.png".into()))?;
    let mut out = Vec::new();
    f.take(MAX_MERGED_BYTES + 1).read_to_end(&mut out).map_err(|e| malformed(format!("mergedimage.png: {e}")))?;
    if out.len() as u64 > MAX_MERGED_BYTES {
        return Err(FormatError::TooLarge(format!("mergedimage.png is larger than {} MB", MAX_MERGED_BYTES >> 20)));
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

    fn entries(bytes: &[u8]) -> (Vec<String>, String) {
        let mut z = ZipArchive::new(Cursor::new(bytes)).unwrap();
        let names: Vec<String> = (0..z.len()).map(|i| z.by_index(i).unwrap().name().unwrap().to_string()).collect();
        let mut xml = String::new();
        z.by_name("stack.xml").unwrap().read_to_string(&mut xml).unwrap();
        (names, xml)
    }

    #[test]
    fn writes_nested_groups_top_first_with_attributes() {
        let merged = solid(8, 8, (0, 0, 8, 8), [65535; 4]);
        let nodes = vec![
            Node {
                name: "Dirt & <grime>".into(),
                visible: false,
                opacity: 0.5,
                composite_op: "svg:multiply",
                kind: NodeKind::Layer(solid(8, 8, (2, 3, 2, 1), [100, 200, 300, 65535])),
            },
            Node {
                name: "Folder".into(),
                visible: true,
                opacity: 1.0,
                composite_op: "svg:src-over",
                kind: NodeKind::Stack {
                    children: vec![Node {
                        name: "Base".into(),
                        visible: true,
                        opacity: 1.0,
                        composite_op: "svg:src-over",
                        kind: NodeKind::Layer(merged.clone()),
                    }],
                    pass_through: true,
                },
            },
        ];
        let bytes = write_ora(&nodes, &merged).unwrap();
        let (names, xml) = entries(&bytes);
        assert_eq!(names.first().map(String::as_str), Some("mimetype"), "mimetype must be the first entry");
        for n in ["stack.xml", "mergedimage.png", "Thumbnails/thumbnail.png", "data/layer0.png", "data/layer1.png"] {
            assert!(names.iter().any(|x| x == n), "{n} missing: {names:?}");
        }
        assert!(xml.contains("name=\"Dirt &amp; &lt;grime&gt;\" visibility=\"hidden\" opacity=\"0.5000\" composite-op=\"svg:multiply\" src=\"data/layer0.png\" x=\"2\" y=\"3\""), "{xml}");
        assert!(xml.contains("<stack name=\"Folder\"") && xml.contains("isolation=\"auto\""), "{xml}");
        assert!(xml.find("Dirt").unwrap() < xml.find("Folder").unwrap());
        // The layer PNG is cropped to its pixels.
        let mut z = ZipArchive::new(Cursor::new(&bytes[..])).unwrap();
        let mut png = Vec::new();
        z.by_name("data/layer0.png").unwrap().read_to_end(&mut png).unwrap();
        let img = image::load_from_memory(&png).unwrap().into_rgba16();
        assert_eq!((img.width(), img.height()), (2, 1));
        assert_eq!(img.get_pixel(0, 0).0, [100, 200, 300, 65535]);
    }

    #[test]
    fn merged_image_reads_back() {
        let merged = solid(4, 2, (0, 0, 4, 2), [1000, 2000, 3000, 65535]);
        let bytes = write_ora(&[], &merged).unwrap();
        let png = read_merged(&bytes).unwrap();
        let img = image::load_from_memory(&png).unwrap().into_rgba16();
        assert_eq!(img.get_pixel(3, 1).0, [1000, 2000, 3000, 65535]);
    }

    #[test]
    fn hostile_input_is_an_error() {
        assert!(read_merged(b"not a zip").is_err());
        assert!(write_ora(&[], &Rgba16 { width: 0, height: 0, data: vec![] }).is_err());
        assert!(write_ora(&[], &Rgba16 { width: 2, height: 2, data: vec![0; 3] }).is_err());
        let merged = Rgba16::transparent(4, 4);
        let wrong = Node { name: "x".into(), visible: true, opacity: f32::NAN, composite_op: "svg:src-over", kind: NodeKind::Layer(Rgba16::transparent(2, 2)) };
        assert!(write_ora(&[wrong], &merged).is_err());
        let mut deep = Node { name: "l".into(), visible: true, opacity: 1.0, composite_op: "svg:src-over", kind: NodeKind::Layer(merged.clone()) };
        for _ in 0..=MAX_DEPTH {
            deep = Node {
                name: "g".into(),
                visible: true,
                opacity: 1.0,
                composite_op: "svg:src-over",
                kind: NodeKind::Stack { children: vec![deep], pass_through: false },
            };
        }
        assert!(write_ora(&[deep], &merged).is_err());
        // An empty layer still gets a (1×1) image, and a NaN opacity is written as 1.
        let empty = Node { name: "e".into(), visible: true, opacity: f32::NAN, composite_op: "svg:src-over", kind: NodeKind::Layer(merged.clone()) };
        let (_, xml) = entries(&write_ora(&[empty], &merged).unwrap());
        assert!(xml.contains("opacity=\"1.0000\""), "{xml}");
    }
}
