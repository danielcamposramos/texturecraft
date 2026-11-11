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

/// The most bytes of one file a mesh may reference: FBX and USD reference textures, glTF also larger geometry buffers.
pub fn reference_cap(mesh_name: &str) -> u64 {
    if matches!(super::extension(mesh_name).as_str(), "fbx" | "usd" | "usda" | "usdc") { super::MAX_TEXTURE_BYTES } else { MAX_MESH_FILE_BYTES }
}

/// Read mesh file `path` with the files it references from its folder (see
/// [`self_contained_mesh`]), each at most [`MAX_MESH_FILE_BYTES`].
///
/// The folder is opened first and held: the mesh file and every reference are then opened beneath
/// that one handle, so a folder replaced by a link while the import runs cannot redirect the reads.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_mesh_file(path: &std::path::Path) -> Result<MeshFile> {
    let directory = mesh_directory(path);
    let root = cap_std::fs::Dir::open_ambient_dir(directory, cap_std::ambient_authority()).ok();
    let canonical_root = directory.canonicalize().ok();
    // beneath the held folder; a mesh file that is a link out of it (or a folder that cannot be held) is read by its path, as before
    let beneath = root.as_ref().zip(path.file_name()).and_then(|(root, name)| open_beneath(root, std::path::Path::new(name)));
    let bytes = match beneath {
        Some(file) => crate::file_read::read_open_file(file, path, MAX_MESH_FILE_BYTES)?,
        None => crate::read_capped(path, MAX_MESH_FILE_BYTES)?,
    };
    let cap = reference_cap(&path.to_string_lossy());
    self_contained_mesh(&path.to_string_lossy(), bytes, &mut |relative| read_reference(root.as_ref()?, canonical_root.as_deref()?, relative, cap))
}

/// Read `relative` (a path below the mesh's folder `directory`, as [`self_contained_mesh`] asks for it), at most `cap`
/// bytes, from a regular file that resolves inside that folder: links out of it are refused, and the read goes through
/// a handle opened on the folder itself.
#[cfg(not(target_arch = "wasm32"))]
pub fn read_mesh_reference(directory: &std::path::Path, relative: &str, cap: u64) -> Option<Vec<u8>> {
    let root = cap_std::fs::Dir::open_ambient_dir(directory, cap_std::ambient_authority()).ok()?;
    read_reference(&root, &directory.canonicalize().ok()?, relative, cap)
}

/// `relative` opened for reading beneath `root` (a FIFO is refused rather than waited on).
#[cfg(not(target_arch = "wasm32"))]
fn open_beneath(root: &cap_std::fs::Dir, relative: &std::path::Path) -> Option<std::fs::File> {
    let mut options = cap_std::fs::OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    cap_std::fs::OpenOptionsExt::custom_flags(&mut options, rustix::fs::OFlags::NONBLOCK.bits() as i32);
    root.open_with(relative, &options).ok().map(cap_std::fs::File::into_std)
}

#[cfg(not(target_arch = "wasm32"))]
fn mesh_directory(path: &std::path::Path) -> &std::path::Path {
    path.parent().filter(|parent| !parent.as_os_str().is_empty()).unwrap_or_else(|| std::path::Path::new("."))
}

#[cfg(not(target_arch = "wasm32"))]
fn read_reference(root: &cap_std::fs::Dir, canonical_root: &std::path::Path, relative: &str, cap: u64) -> Option<Vec<u8>> {
    // Canonicalization preserves absolute symlinks whose targets are inside the mesh folder
    let target = canonical_root.join(relative).canonicalize().ok()?;
    let relative_target = target.strip_prefix(canonical_root).ok()?;

    // Resolve again beneath the held directory so replacement links cannot escape during open
    let file = open_beneath(root, relative_target)?;
    crate::file_read::read_open_file(file, &target, cap).ok()
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
            let path = super::gltf::percent_decode(uri).ok_or_else(|| malformed("glTF", format!("URI \"{uri}\" is not valid UTF-8")))?;
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

#[cfg(all(test, unix))]
mod symlink_tests {
    use super::*;
    use std::path::{Path, PathBuf};

    fn test_directory() -> std::io::Result<PathBuf> {
        use std::hash::{BuildHasher, Hasher};
        let nonce = std::collections::hash_map::RandomState::new().build_hasher().finish();
        let directory = std::env::temp_dir().join(format!("tc-mesh-reference-{nonce:016x}"));
        std::fs::create_dir(&directory)?;
        // canonical, as production passes it (the temporary folder may itself be reached through a link)
        directory.canonicalize()
    }

    #[test]
    fn a_symlink_out_of_the_mesh_folder_is_not_read() {
        let base = test_directory().unwrap();
        let (mesh_dir, outside) = (base.join("mesh"), base.join("outside"));
        std::fs::create_dir_all(&mesh_dir).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("secret.png"), b"secret").unwrap();
        std::os::unix::fs::symlink(&outside, mesh_dir.join("tex")).unwrap();
        let gltf = br#"{"asset":{"version":"2.0"},"images":[{"uri":"tex/secret.png"}]}"#;
        std::fs::write(mesh_dir.join("m.gltf"), gltf).unwrap();

        let file = super::read_mesh_file(&mesh_dir.join("m.gltf")).unwrap();

        assert!(!String::from_utf8_lossy(&file.bytes).contains("c2VjcmV0"), "symlinked file was embedded");
        assert_eq!(file.warnings.len(), 1);
        std::fs::remove_dir_all(base).unwrap();
    }

    #[test]
    fn references_support_in_folder_symlinks_and_enforce_the_cap() {
        let directory = test_directory().unwrap();
        std::fs::write(directory.join("texture"), b"pixels").unwrap();
        std::os::unix::fs::symlink("texture", directory.join("relative")).unwrap();
        std::os::unix::fs::symlink(directory.join("texture"), directory.join("absolute")).unwrap();
        let root = cap_std::fs::Dir::open_ambient_dir(&directory, cap_std::ambient_authority()).unwrap();

        for name in ["texture", "relative", "absolute"] {
            assert_eq!(read_reference(&root, &directory, name, 6), Some(b"pixels".to_vec()));
            assert_eq!(read_reference(&root, &directory, name, 5), None);
        }

        assert_eq!(read_reference(&root, &directory, "missing", 6), None);
        assert_eq!(read_reference(&root, &directory, ".", 6), None);
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn replacing_the_root_path_cannot_redirect_an_open_directory() {
        let directory = test_directory().unwrap();
        let original = directory.join("mesh");
        let moved = directory.join("moved");
        std::fs::create_dir(&original).unwrap();
        std::fs::write(original.join("texture"), b"inside").unwrap();
        let root = cap_std::fs::Dir::open_ambient_dir(&original, cap_std::ambient_authority()).unwrap();

        std::fs::rename(&original, &moved).unwrap();
        std::fs::create_dir(&original).unwrap();
        std::fs::write(original.join("texture"), b"outside").unwrap();
        let bytes = read_reference(&root, &original, "texture", 16);

        assert_eq!(bytes, Some(b"inside".to_vec()));
        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn bare_mesh_filename_keeps_neighboring_references() {
        const CHILD: &str = "TC_TEST_BARE_MESH_PATH";
        if std::env::var_os(CHILD).is_some() {
            let file = read_mesh_file(Path::new("mesh.gltf")).unwrap();
            assert!(String::from_utf8_lossy(&file.bytes).contains("cGl4ZWxz"));
            assert!(file.warnings.is_empty());
            return;
        }

        let directory = test_directory().unwrap();
        std::fs::write(directory.join("mesh.gltf"), br#"{"asset":{"version":"2.0"},"images":[{"uri":"texture"}]}"#).unwrap();
        std::fs::write(directory.join("texture"), b"pixels").unwrap();
        let status = std::process::Command::new(std::env::current_exe().unwrap())
            .args(["--exact", "mesh_import::external::symlink_tests::bare_mesh_filename_keeps_neighboring_references"])
            .env(CHILD, "1")
            .current_dir(&directory)
            .status()
            .unwrap();

        std::fs::remove_dir_all(directory).unwrap();
        assert!(status.success());
    }
}
