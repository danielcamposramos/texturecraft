//! Meshes that keep part of their data in other files: a `.gltf`'s buffers and images, a loose
//! USD layer's textures, an FBX's texture files. [`self_contained_mesh`] folds those files in, so a project stores the
//! mesh as one file and imports it again from its bytes alone.

use base64::Engine as _;
use serde_json::Value;

use super::MAX_MESH_FILE_BYTES;
use crate::{FormatError, Result, malformed};

/// A mesh file ready to import from its bytes.
#[derive(Clone, Debug, PartialEq)]
pub struct MeshFile {
    /// The name to import it by: the original, or `.usdz` for a loose USD layer packaged with its
    /// textures.
    pub name: String,
    pub bytes: Vec<u8>,
    /// Referenced files that could not be read, as messages for the log.
    pub warnings: Vec<String>,
}

/// `bytes` of mesh `name` with the files it references folded in: a `.gltf`'s external buffers
/// and images become data URIs, a `.usd`/`.usda`/`.usdc` layer whose materials use texture files
/// becomes a `.usdz` package holding them, and an FBX whose base color textures are files becomes
/// a package of it and them (still named `.fbx`). `read` gets a referenced file by its path below
/// the mesh's folder (`/`-separated); files outside that folder (absolute paths, URLs, `..` above
/// it) are not read. Other formats, and meshes that reference nothing, come back unchanged.
pub fn self_contained_mesh(name: &str, bytes: Vec<u8>, read: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Result<MeshFile> {
    let mut warnings = Vec::new();
    let mut found = |asset: &str| {
        let data = below(asset).and_then(|p| read(&p));
        let msg = format!("{name}: {asset} was not found next to the mesh");
        if data.is_none() && !warnings.contains(&msg) {
            warnings.push(msg);
        }
        data
    };
    let (name, bytes) = match super::extension(name).as_str() {
        "gltf" => (name.to_string(), inline_gltf(bytes, &mut found)?),
        "fbx" => {
            let (package, missing) = super::fbx::package(name, &bytes, &mut |asset| below(asset).and_then(|p| read(&p)))?;
            warnings.extend(missing.into_iter().map(|asset| format!("{name}: {asset} was not found next to the mesh")));
            (name.to_string(), package.unwrap_or(bytes))
        }
        "usd" | "usda" | "usdc" => match super::usd::package(name, &bytes, &mut found)? {
            Some(usdz) => (std::path::Path::new(name).with_extension("usdz").to_string_lossy().into_owned(), usdz),
            None => (name.to_string(), bytes),
        },
        _ => (name.to_string(), bytes),
    };
    Ok(MeshFile { name, bytes, warnings })
}

/// Read mesh file `path` with the files it references from its folder (see
/// [`self_contained_mesh`]), each at most [`MAX_MESH_FILE_BYTES`].
#[cfg(not(target_arch = "wasm32"))]
pub fn read_mesh_file(path: &std::path::Path) -> Result<MeshFile> {
    let bytes = crate::read_capped(path, MAX_MESH_FILE_BYTES)?;
    let dir = path.parent().unwrap_or_else(|| std::path::Path::new(""));
    self_contained_mesh(&path.to_string_lossy(), bytes, &mut |rel| crate::read_capped(&dir.join(rel), MAX_MESH_FILE_BYTES).ok())
}

/// `asset` as a path below the mesh's folder (`/`-separated, `.` and `..` resolved), or None when
/// it is absolute, has a URL scheme or drive letter, or leaves the folder.
pub(crate) fn below(asset: &str) -> Option<String> {
    let s = asset.replace('\\', "/");
    if s.starts_with('/') || s.contains(':') {
        return None;
    }
    let mut parts = Vec::new();
    for c in s.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                parts.pop()?;
            }
            c => parts.push(c),
        }
    }
    (!parts.is_empty()).then(|| parts.join("/"))
}

/// A `.gltf` with its external buffers and images as base64 data URIs (unchanged when it has none).
fn inline_gltf(bytes: Vec<u8>, found: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Result<Vec<u8>> {
    let mut doc: Value = serde_json::from_slice(&bytes).map_err(|e| malformed("glTF", e))?;
    let mut total = 0u64;
    let mut changed = false;
    for key in ["buffers", "images"] {
        let Some(items) = doc.get_mut(key).and_then(Value::as_array_mut) else { continue };
        for item in items {
            let Some(uri) = item.get("uri").and_then(Value::as_str) else { continue };
            if uri.starts_with("data:") {
                continue;
            }
            let path = percent_decode(uri).ok_or_else(|| malformed("glTF", format!("URI \"{uri}\" is not valid UTF-8")))?;
            let Some(data) = found(&path) else { continue };
            // Base64 grows the data by a third; the result must stay importable.
            total = total.saturating_add(data.len() as u64);
            if total.saturating_mul(4) / 3 > MAX_MESH_FILE_BYTES {
                return Err(FormatError::TooLarge("the glTF and the files it references exceed 2 GiB".into()));
            }
            let mime = match item.get("mimeType").and_then(Value::as_str) {
                Some(m) => m.to_string(),
                None if key == "buffers" => "application/octet-stream".into(),
                None => image_mime(&path).into(),
            };
            item["uri"] = Value::String(format!("data:{mime};base64,{}", base64::engine::general_purpose::STANDARD.encode(&data)));
            changed = true;
        }
    }
    if !changed {
        return Ok(bytes);
    }
    serde_json::to_vec(&doc).map_err(|e| FormatError::Invalid(e.to_string()))
}

/// Percent-decoded asset paths must be UTF-8 before they can be resolved below the mesh folder.
fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while let Some(&c) = b.get(i) {
        let hex = |k: usize| b.get(k).and_then(|h| (*h as char).to_digit(16));
        match (c, hex(i + 1), hex(i + 2)) {
            (b'%', Some(h), Some(l)) => {
                out.push((h * 16 + l) as u8);
                i += 3;
            }
            _ => {
                out.push(c);
                i += 1;
            }
        }
    }
    String::from_utf8(out).ok()
}

fn image_mime(path: &str) -> &'static str {
    match super::extension(path).as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "webp" => "image/webp",
        _ => "image/png",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn below_keeps_paths_inside_the_folder() {
        assert_eq!(below("tex/a.png").as_deref(), Some("tex/a.png"));
        assert_eq!(below("./tex\\b.png").as_deref(), Some("tex/b.png"));
        assert_eq!(below("tex/../c.png").as_deref(), Some("c.png"));
        for outside in ["../a.png", "/etc/passwd", "C:/x.png", "C:x.png", "http://host/a.png", "file:///a.png", "", ".", "tex/../.."] {
            assert_eq!(below(outside), None, "{outside}");
        }
    }

    #[test]
    fn other_formats_and_self_contained_files_are_unchanged() {
        let mut read = |_: &str| -> Option<Vec<u8>> { panic!("nothing to read") };
        let obj = b"v 0 0 0\n".to_vec();
        assert_eq!(self_contained_mesh("a.obj", obj.clone(), &mut read).unwrap(), MeshFile { name: "a.obj".into(), bytes: obj, warnings: vec![] });
        let gltf = br#"{"asset": {"version": "2.0"}, "buffers": [{"uri": "data:application/octet-stream;base64,AAAA", "byteLength": 3}]}"#.to_vec();
        assert_eq!(self_contained_mesh("a.gltf", gltf.clone(), &mut read).unwrap().bytes, gltf);
    }
}
