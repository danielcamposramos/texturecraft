//! `file.importMaterialX`: a MaterialX (`.mtlx`) material becomes one fill layer.
//!
//! The document is read by [`texturecraft_formats::materialx_read`]; here its image files are
//! loaded (only from the document's own folder and below: a document is untrusted, and its
//! `file` values must not reach other files), added to the project as resources and set as the
//! fill's sources, with constants as uniform sources. The whole import is one undo step. Channels
//! the texture set doesn't have, images that can't be read and anything the reader skipped are
//! reported in `warnings`; the import goes ahead with the rest.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde_json::{Value, json};
use texturecraft_core::{Channel, Fill, Layer, LayerKind, Projection, Source};
use texturecraft_formats::materialx_read::{Material, Value as Mtlx};

use crate::commands::{CommandSpec, always};
#[cfg(not(target_arch = "wasm32"))]
use crate::commands::{bad, strp};
use crate::{EngineError, Result, Session, cmd};

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "file.importMaterialX",
        "Import MaterialX material...",
        ["File"],
        None,
        "{path: a .mtlx file, material?: the surfacematerial to read (default: the first), name?: layer name} → {layer, material, channels, resources, warnings}: one fill layer with the material's textures (read from the file's folder) and values; open_pbr_surface, standard_surface and gltf_pbr are read",
        always,
        import
    )]
}

/// Loads an image file named in the document (relative to it): its bytes, or why not.
pub type Loader<'a> = &'a dyn Fn(&str) -> std::result::Result<Vec<u8>, String>;

impl Session {
    /// Add `material` as a fill layer on the active texture set, loading its images with `load`.
    /// Returns the reply of `file.importMaterialX`.
    pub fn import_material(&mut self, material: Material, name: Option<&str>, load: Loader) -> Result<Value> {
        let mut warnings = material.warnings.clone();
        let ts = self.ts();
        let available = self.doc.project.texture_set(ts).map(|t| t.channel_list()).unwrap_or_default();
        let mut fill = Fill { projection: Projection::Uv { scale: material.uv_tiling, rotation: 0.0, offset: [0.0, 0.0] }, ..Fill::default() };
        // Resources to add (name and encoded bytes), and their decoded images for the session.
        let mut added: Vec<(String, Arc<Vec<u8>>)> = Vec::new();
        let mut decoded = Vec::new();
        let mut names: BTreeMap<String, String> = BTreeMap::new();
        let mut channels: Vec<Channel> = Vec::new();
        for (channel, value) in &material.channels {
            if !available.contains(channel) {
                warnings.push(format!("{} was skipped: this texture set has no such channel", channel.id()));
                continue;
            }
            let source = match value {
                Mtlx::Constant(c) => Source::Uniform(*c),
                Mtlx::Image(file) => {
                    let resource = match names.get(file) {
                        Some(n) => n.clone(),
                        None => {
                            let loaded =
                                load(file).and_then(|bytes| texturecraft_formats::decode_image(&bytes).map(|image| (bytes, image)).map_err(|e| e.to_string()));
                            let (bytes, image) = match loaded {
                                Ok(v) => v,
                                Err(e) => {
                                    warnings.push(format!("{} was skipped: {file}: {e}", channel.id()));
                                    continue;
                                }
                            };
                            let leaf = file.rsplit(['/', '\\']).next().unwrap_or(file);
                            let base = texturecraft_formats::sanitize_resource_name(leaf);
                            let (mut unique, mut n) = (base.clone(), 2);
                            while self.doc.resources.contains_key(&unique) || added.iter().any(|(a, _)| *a == unique) {
                                unique = format!("{n}_{base}");
                                n += 1;
                            }
                            added.push((unique.clone(), Arc::new(bytes)));
                            decoded.push((unique.clone(), crate::resource_cmds::decoded(image)));
                            names.insert(file.clone(), unique.clone());
                            unique
                        }
                    };
                    Source::image(resource)
                }
            };
            fill.sources.insert(*channel, source);
            channels.push(*channel);
        }
        if channels.is_empty() {
            let why = if warnings.is_empty() { String::new() } else { format!(" ({})", warnings.join("; ")) };
            return Err(EngineError::Other(format!("file.importMaterialX: nothing to import from {:?}{why}", material.name)));
        }
        let layer_name = name.map(str::to_string).unwrap_or_else(|| self.doc.project.next_layer_name(ts, &material.name));
        let resource_names: Vec<String> = added.iter().map(|(n, _)| n.clone()).collect();
        let id = self.edit("Import MaterialX material", |s| {
            for (n, bytes) in &added {
                s.doc.resources.insert(n.clone(), bytes.clone());
            }
            let id = s.doc.project.alloc_id();
            crate::layer_cmds::insert_selected(s, Layer::new(id, layer_name, LayerKind::Fill(fill), &channels))
        })?;
        self.images.extend(decoded);
        Ok(json!({
            "layer": id.0,
            "material": material.name,
            "channels": channels.iter().map(|c| c.id()).collect::<Vec<_>>(),
            "resources": resource_names,
            "warnings": warnings,
        }))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn import(s: &mut Session, p: &Value) -> Result<Value> {
    use std::path::{Component, Path};
    let cmd = "file.importMaterialX";
    let path = strp(p, "path").filter(|s| !s.trim().is_empty()).ok_or_else(|| bad(cmd, "`path` is required"))?;
    let meta = std::fs::metadata(path).map_err(|e| bad(cmd, format!("cannot read {path}: {e}")))?;
    if !meta.is_file() || meta.len() > texturecraft_formats::materialx_read::MAX_BYTES as u64 {
        return Err(bad(cmd, format!("{path} is not a MaterialX file of a readable size")));
    }
    let bytes = std::fs::read(path).map_err(|e| bad(cmd, format!("cannot read {path}: {e}")))?;
    let material = texturecraft_formats::materialx_read::read_materialx(&bytes, strp(p, "material")).map_err(|e| bad(cmd, format!("{path}: {e}")))?;
    let folder = Path::new(path).parent().filter(|d| !d.as_os_str().is_empty()).unwrap_or(Path::new("."));
    let dir = folder.canonicalize().map_err(|e| bad(cmd, format!("cannot read the folder of {path}: {e}")))?;
    let load = |file: &str| -> std::result::Result<Vec<u8>, String> {
        // `<UDIM>` stands for a tile number; the first tile is read.
        let file = file.replace("<UDIM>", "1001");
        let rel = Path::new(&file);
        let outside = "the file is outside the material's folder";
        if rel.is_absolute() || rel.components().any(|c| matches!(c, Component::ParentDir | Component::Prefix(_) | Component::RootDir)) {
            return Err(outside.into());
        }
        let full = dir.join(rel).canonicalize().map_err(|e| e.to_string())?;
        if !full.starts_with(&dir) {
            return Err(outside.into());
        }
        let meta = std::fs::metadata(&full).map_err(|e| e.to_string())?;
        // A pipe or device in the folder would block the read (a pipe reports no length).
        if !meta.is_file() {
            return Err("it is not a regular file".into());
        }
        let len = meta.len();
        if len > crate::resource_cmds::MAX_BYTES as u64 {
            return Err(format!("the file is over {} MB", crate::resource_cmds::MAX_BYTES >> 20));
        }
        std::fs::read(&full).map_err(|e| e.to_string())
    };
    s.import_material(material, strp(p, "name"), &load)
}

#[cfg(target_arch = "wasm32")]
fn import(_: &mut Session, _: &Value) -> Result<Value> {
    Err(EngineError::Other("file paths are not available in the browser".into()))
}
