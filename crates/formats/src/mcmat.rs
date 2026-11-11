//! MaterialCraft portable materials (`.mcmat`, read only): a zip with `material.json` (the
//! manifest: maps by usage, exposed parameters, image inputs, presets), `maps/<usage>.png`
//! (8-bit sRGB for color usages, 16-bit linear otherwise; OpenGL normals) and the source
//! `.mcraft` package. TextureCraft keeps its own reader (the apps share the format, not code).
//!
//! Also the 16-bit PNG encoder that hands baked mesh maps to MaterialCraft as graph inputs.

use std::io::{Cursor, Read};

use serde::Deserialize;
use texturecraft_core::{LinkParam, LinkPreset};

use crate::{FormatError, Result};

mod map_entries_visitor;

/// The portable material extension.
pub const MCMAT_EXT: &str = "mcmat";
/// MaterialCraft's package extension.
pub const MCRAFT_EXT: &str = "mcraft";
const FORMAT: &str = "craft-material";
const VERSION: u32 = 1;
/// Largest entry read (maps are at most 8192² × 16 bits, compressed).
const MAX_ENTRY: u64 = 512 << 20;
/// Most maps a manifest may list (there are 9 usages; generous headroom).
const MAX_MAPS: usize = 64;
/// Most decompressed bytes read from one `.mcmat` in total (a zip bomb may list one entry many times).
const MAX_TOTAL: u64 = 4 << 30;
const KIND: &str = "MaterialCraft material";

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Manifest {
    pub format: String,
    pub version: u32,
    pub name: String,
    pub graph: String,
    pub size: [u32; 2],
    #[serde(deserialize_with = "bounded_maps")]
    pub maps: Vec<MapEntry>,
    #[serde(default)]
    pub parameters: Vec<LinkParam>,
    #[serde(default)]
    pub inputs: Vec<ImageInput>,
    #[serde(default)]
    pub presets: Vec<LinkPreset>,
    pub source: String,
}

const MAPS_LIMIT_MSG: &str = "too many maps";

/// Reads `maps` while enforcing `MAX_MAPS`, so a hostile manifest cannot make serde allocate every entry first.
fn bounded_maps<'de, D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Vec<MapEntry>, D::Error> {
    deserializer.deserialize_seq(map_entries_visitor::MapEntriesVisitor)
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MapEntry {
    /// `baseColor`, `normal`, `roughness`, `metallic`, `height`, `ambientOcclusion`, `emissive`,
    /// `opacity` or `specular`.
    pub usage: String,
    pub file: String,
}

#[derive(Clone, Debug, PartialEq, Deserialize)]
pub struct ImageInput {
    pub identifier: String,
}

/// A read `.mcmat`: the manifest, each map's PNG bytes by usage, the source package.
#[derive(Clone, Debug)]
pub struct PortableMaterial {
    pub manifest: Manifest,
    pub maps: Vec<(String, Vec<u8>)>,
    pub source: Vec<u8>,
}

fn malformed(msg: impl Into<String>) -> FormatError {
    FormatError::Malformed { kind: KIND, msg: msg.into() }
}

/// Read `.mcmat` bytes.
pub fn read_mcmat(bytes: &[u8]) -> Result<PortableMaterial> {
    read_with_limits(bytes, MAX_ENTRY, MAX_TOTAL)
}

fn read_with_limits(bytes: &[u8], max_entry: u64, max_total: u64) -> Result<PortableMaterial> {
    let mut archive = zip::ZipArchive::new(Cursor::new(bytes)).map_err(|error| malformed(error.to_string()))?;
    let mut total = 0u64;
    let mut read = |name: &str| -> Result<Vec<u8>> {
        let entry = archive.by_name(name).map_err(|_| malformed(format!("{name} is missing")))?;
        let remaining = max_entry.min(max_total.saturating_sub(total));
        if entry.size() > remaining {
            return Err(FormatError::TooLarge(name.to_string()));
        }

        // Declared sizes are untrusted; enforce the same allowance while decompressing
        let mut data = Vec::new();
        entry.take(remaining.saturating_add(1)).read_to_end(&mut data).map_err(|error| malformed(format!("{name}: {error}")))?;
        if data.len() as u64 > remaining {
            return Err(FormatError::TooLarge(name.to_string()));
        }

        total += data.len() as u64;
        Ok(data)
    };
    let manifest: Manifest = serde_json::from_slice(&read("material.json")?).map_err(|e| {
        let msg = e.to_string();
        if msg.contains(MAPS_LIMIT_MSG) { FormatError::TooLarge(format!("more than {MAX_MAPS} maps")) } else { malformed(format!("material.json: {msg}")) }
    })?;
    if manifest.format != FORMAT {
        return Err(malformed(format!("format `{}`", manifest.format)));
    }

    if manifest.version > VERSION {
        return Err(FormatError::Unsupported(format!("{KIND} version {} (this version reads up to {VERSION})", manifest.version)));
    }

    let maps = manifest.maps.iter().map(|m| Ok((m.usage.clone(), read(&m.file)?))).collect::<Result<Vec<_>>>()?;
    let source = read(&manifest.source)?;
    Ok(PortableMaterial { manifest, maps, source })
}

/// Whether `bytes` are a zip holding `entry` (`material.json`: a portable material,
/// `package.json`: a MaterialCraft package).
fn zip_has(bytes: &[u8], entry: &str) -> bool {
    zip::ZipArchive::new(Cursor::new(bytes)).is_ok_and(|mut z| z.by_name(entry).is_ok())
}

pub fn is_mcmat(bytes: &[u8]) -> bool {
    zip_has(bytes, "material.json")
}

pub fn is_mcraft(bytes: &[u8]) -> bool {
    zip_has(bytes, "package.json")
}

/// A baked mesh map (`components` floats per texel, 0..1, rows top first) as a 16-bit PNG
/// (grayscale for one component, RGB for three).
pub fn mesh_map_png(width: u32, components: usize, data: &[f32]) -> Result<Vec<u8>> {
    let per_row = (width as usize).saturating_mul(components);
    let height = data.len().checked_div(per_row).unwrap_or(0) as u32;
    if height == 0 || !matches!(components, 1 | 3) {
        return Err(FormatError::Invalid(format!("mesh map of {} values, {width} wide, {components} components", data.len())));
    }
    let px: Vec<u16> = data.iter().take(per_row * height as usize).map(|v| (v.clamp(0.0, 1.0) * 65535.0).round() as u16).collect();
    let img = if components == 1 {
        image::ImageBuffer::<image::Luma<u16>, _>::from_raw(width, height, px).map(image::DynamicImage::ImageLuma16)
    } else {
        image::ImageBuffer::<image::Rgb<u16>, _>::from_raw(width, height, px).map(image::DynamicImage::ImageRgb16)
    }
    .ok_or_else(|| FormatError::Invalid("mesh map size".into()))?;
    let mut out = Vec::new();
    img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png).map_err(|e| FormatError::Invalid(e.to_string()))?;
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    /// A `.mcmat` as MaterialCraft writes it (see its `formats/src/material.rs`).
    pub(crate) fn sample(version: u32) -> Vec<u8> {
        let manifest = serde_json::json!({
            "format": "craft-material", "version": version, "name": "Brick wall", "graph": "Brick_wall", "size": [4, 4],
            "maps": [{"usage": "baseColor", "file": "maps/baseColor.png", "colorSpace": "srgb"}],
            "parameters": [{"id": "wear", "label": "Wear", "default": {"t": "float", "v": 0.5}, "min": 0.0, "max": 1.0, "value": {"t": "float", "v": 0.25}}],
            "inputs": [{"identifier": "curvature", "kind": "gray"}],
            "presets": [{"name": "Worn", "values": {"wear": {"t": "float", "v": 1.0}}}],
            "source": "source.mcraft", "futureField": true
        });
        let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let o = zip::write::SimpleFileOptions::default();
        z.start_file("material.json", o).unwrap();
        z.write_all(manifest.to_string().as_bytes()).unwrap();
        z.start_file("maps/baseColor.png", o).unwrap();
        z.write_all(&mesh_map_png(4, 3, &[0.5; 48]).unwrap()).unwrap();
        z.start_file("source.mcraft", o).unwrap();
        z.write_all(b"pkg").unwrap();
        z.finish().unwrap().into_inner()
    }

    #[test]
    fn reads_manifest_maps_and_source() {
        let bytes = sample(1);
        assert!(is_mcmat(&bytes) && !is_mcraft(&bytes));
        let m = read_mcmat(&bytes).unwrap();
        assert_eq!(m.manifest.graph, "Brick_wall");
        assert_eq!(m.manifest.parameters[0].id, "wear");
        assert_eq!(m.manifest.parameters[0].value, serde_json::json!({"t": "float", "v": 0.25}));
        assert_eq!(m.manifest.presets[0].name, "Worn");
        assert_eq!(m.manifest.inputs[0].identifier, "curvature");
        assert_eq!(m.maps[0].0, "baseColor");
        assert_eq!(crate::decode_image(&m.maps[0].1).unwrap().width, 4);
        assert_eq!(m.source, b"pkg");
        assert!(matches!(read_mcmat(&sample(2)), Err(FormatError::Unsupported(_))));
        assert!(read_mcmat(b"junk").is_err());
    }

    #[test]
    fn rejects_a_manifest_listing_too_many_maps() {
        let maps: Vec<_> = (0..=MAX_MAPS).map(|_| serde_json::json!({"usage": "baseColor", "file": "maps/baseColor.png"})).collect();
        let manifest =
            serde_json::json!({"format": "craft-material", "version": 1, "name": "n", "graph": "g", "size": [4, 4], "maps": maps, "source": "source.mcraft"});
        let mut z = zip::ZipWriter::new(Cursor::new(Vec::new()));
        z.start_file("material.json", zip::write::SimpleFileOptions::default()).unwrap();
        z.write_all(manifest.to_string().as_bytes()).unwrap();
        let bytes = z.finish().unwrap().into_inner();

        assert!(matches!(read_mcmat(&bytes), Err(FormatError::TooLarge(_))));
    }

    #[test]
    fn manifest_map_list_is_bounded_while_parsing() {
        let entry = r#"{"usage":"a","file":"b"}"#;
        let head = r#"{"format":"craft-material","version":1,"name":"n","graph":"g","size":[4,4],"source":"s","maps":["#;
        let json = |n: usize| format!("{head}{}]}}", vec![entry; n].join(","));

        assert_eq!(serde_json::from_str::<Manifest>(&json(MAX_MAPS)).unwrap().maps.len(), MAX_MAPS);
        let err = serde_json::from_str::<Manifest>(&json(100_000)).unwrap_err().to_string();
        assert!(err.contains(MAPS_LIMIT_MSG), "{err}");
    }

    #[test]
    fn repeated_entries_and_source_share_the_total_budget() {
        let manifest = serde_json::json!({
            "format": FORMAT, "version": VERSION, "name": "Repeated", "graph": "g", "size": [4, 4],
            "maps": [{"usage": "baseColor", "file": "map"}, {"usage": "roughness", "file": "map"}], "source": "source"
        })
        .to_string();
        let mut archive = zip::ZipWriter::new(Cursor::new(Vec::new()));
        let options = zip::write::SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (name, data) in [("material.json", manifest.as_bytes()), ("map", &[0; 4096][..]), ("source", &[0; 128][..])] {
            archive.start_file(name, options).unwrap();
            archive.write_all(data).unwrap();
        }
        let bytes = archive.finish().unwrap().into_inner();
        let total = manifest.len() as u64 + 2 * 4096 + 128;

        let material = read_with_limits(&bytes, 4096, total).unwrap();
        let repeated_map_error = read_with_limits(&bytes, 4096, total - 4096);
        let source_error = read_with_limits(&bytes, 4096, total - 1);
        let entry_error = read_with_limits(&bytes, 4095, total);

        assert_eq!(material.maps.len(), 2);
        assert!(bytes.len() < 1024, "the fixture must exercise compression amplification");
        assert!(matches!(repeated_map_error, Err(FormatError::TooLarge(name)) if name == "map"));
        assert!(matches!(source_error, Err(FormatError::TooLarge(name)) if name == "source"));
        assert!(matches!(entry_error, Err(FormatError::TooLarge(name)) if name == "map"));
    }

    #[test]
    fn mesh_maps_encode_as_16_bit() {
        let png = mesh_map_png(2, 1, &[0.0, 0.25, 0.5, 1.0]).unwrap();
        let d = crate::decode_image(&png).unwrap();
        assert_eq!((d.width, d.height), (2, 2));
        assert!((d.rgba[1][0] - 0.25).abs() < 1e-4);
        assert!(mesh_map_png(0, 1, &[]).is_err());
        assert!(mesh_map_png(2, 2, &[0.0; 8]).is_err());
    }
}
