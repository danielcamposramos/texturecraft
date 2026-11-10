//! PhotoCraft hand-off: a texture set's layer stack as layered OpenRaster documents (one per
//! channel) opened in PhotoCraft, and PhotoCraft's saves shown live.
//!
//! Every layer is rendered alone (content with its fill sources, generators, filters and
//! content effects, times its mask: [`Session::flatten`]) into a layer of the same name, so the
//! mask becomes the layer's transparency; folders become groups. Each sent file is remembered
//! as a [`Link`]: when PhotoCraft saves it, its merged image becomes the image source of a
//! `PhotoCraft: <channel>` fill at the top of the set, which leaves the layers below untouched.
//! The apps share the file format and the command line, not code.

use serde_json::{Value, json};
use texturecraft_core::{BlendMode, Channel, ChannelKind, LayerId, LayerKind};
use texturecraft_formats::ora::{self, Node, NodeKind, Rgba16};
use texturecraft_texel::{TILE, TILE_AREA, from_u16, from_u16_signed, linear_to_srgb, to_u16};

use crate::commands::{CommandSpec, always, bad};
use crate::{EngineError, Result, Session, cmd};

const SEND: &str = "file.sendToPhotoCraft";
/// The undo label of a live update.
const UPDATE: &str = "Update from PhotoCraft";
/// Shortest time between two looks at the linked files' modification times.
pub const POLL_MS: f64 = 400.0;

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "file.sendToPhotoCraft",
            "Send layers to PhotoCraft...",
            ["File"],
            None,
            "{dir, ts?: index|\"all\" (default: the active set), channels?: [channel ids] (default: the set's channels), open?: true} → {files, warnings} — writes <mesh>_<set>_<channel>.ora per set and channel (every layer rendered with its mask as transparency, folders as groups) and opens them in PhotoCraft; saves made there show live (see photocraft.links)",
            always,
            send
        ),
        cmd!("photocraft.links", "List PhotoCraft links", [], None, "{} → [{path, ts, channel, layer?}]", always, |s, _| Ok(links_json(s))),
        cmd!(
            "photocraft.sync",
            "Update from PhotoCraft",
            [],
            None,
            "{} → {updated: [paths]} — reads the linked files PhotoCraft saved since the last look",
            always,
            |s, _| sync_now(s)
        ),
        cmd!("photocraft.unlink", "Stop PhotoCraft links", [], None, "{path?} — stops watching one sent file, or all of them", always, unlink),
    ]
}

/// A file sent to PhotoCraft, watched for its saves.
#[derive(Clone, Debug, PartialEq)]
pub struct Link {
    pub path: std::path::PathBuf,
    pub ts: usize,
    pub channel: Channel,
    /// The file's modification time when it was last read or written.
    pub modified: Option<std::time::SystemTime>,
    /// The `PhotoCraft: <channel>` fill showing the file's merged image, once there is one.
    pub layer: Option<LayerId>,
}

/// The session's PhotoCraft links (not saved with the project).
#[derive(Clone, Debug, Default)]
pub struct Links {
    pub links: Vec<Link>,
    /// When `Session::poll_photocraft` last looked (its caller's clock, milliseconds).
    last_poll: Option<f64>,
}

/// The links as `photocraft.links` and `session.inspect` show them.
pub(crate) fn links_json(s: &Session) -> Value {
    json!(
        s.photocraft
            .links
            .iter()
            .map(|l| json!({"path": l.path.display().to_string(), "ts": l.ts, "channel": l.channel.id(), "layer": l.layer.map(|id| id.0)}))
            .collect::<Vec<_>>()
    )
}

fn unlink(s: &mut Session, p: &Value) -> Result<Value> {
    let before = s.photocraft.links.len();
    match p.get("path").and_then(Value::as_str) {
        Some(path) => s.photocraft.links.retain(|l| l.path != std::path::Path::new(path)),
        None => s.photocraft.links.clear(),
    }
    Ok(json!({ "removed": before - s.photocraft.links.len() }))
}

/// The OpenRaster `composite-op` for `mode`, with a warning when it is only close or missing.
fn composite_op(mode: BlendMode) -> (&'static str, Option<&'static str>) {
    use BlendMode as B;
    match mode {
        B::Normal | B::Passthrough => ("svg:src-over", None),
        B::Multiply => ("svg:multiply", None),
        B::Screen => ("svg:screen", None),
        B::Overlay => ("svg:overlay", None),
        B::Darken => ("svg:darken", None),
        B::Lighten => ("svg:lighten", None),
        B::ColorDodge => ("svg:color-dodge", None),
        B::ColorBurn => ("svg:color-burn", None),
        B::HardLight => ("svg:hard-light", None),
        B::SoftLight => ("svg:soft-light", None),
        B::Difference => ("svg:difference", None),
        B::Exclusion => ("svg:exclusion", None),
        B::LinearDodge => ("svg:plus", None),
        B::Saturation => ("svg:saturation", None),
        B::Color => ("svg:color", None),
        B::Value => ("svg:luminosity", Some("Value is sent as Luminosity")),
        _ => ("svg:src-over", Some("this blend mode has no OpenRaster name; it is sent as Normal")),
    }
}

/// One straight-alpha texel of channel `c` from its stored, premultiplied components and coverage.
fn texel(c: Channel, px: &[u16]) -> Option<[u16; 4]> {
    let comps = c.components();
    let a = from_u16(*px.get(comps)?);
    if a <= 0.0 {
        return None;
    }
    let v = |k: usize| px.get(k).map_or(0.0, |v| from_u16(*v) / a);
    let rgb = match c.kind() {
        ChannelKind::Color => [0, 1, 2].map(|k| linear_to_srgb(v(k).clamp(0.0, 1.0))),
        ChannelKind::NormalMap => [0, 1, 2].map(v),
        ChannelKind::Scalar => [v(0); 3],
        ChannelKind::Signed => [px.first().map_or(0.0, |s| from_u16_signed(*s) / a) * 0.5 + 0.5; 3],
    };
    Some([to_u16(rgb[0]), to_u16(rgb[1]), to_u16(rgb[2]), to_u16(a)])
}

/// The canvas-sized image of channel `c` in flattened tiles.
fn flat_image(c: Channel, (w, h): (u32, u32), tiles: &[texturecraft_compose::eval::FlatTile]) -> Rgba16 {
    let mut img = Rgba16::transparent(w, h);
    let stride = c.components() + 1;
    for (_, (tx, ty), buf) in tiles.iter().filter(|(ch, _, _)| *ch == c) {
        for o in 0..TILE_AREA {
            let (x, y) = (tx * TILE + o as u32 % TILE, ty * TILE + o as u32 / TILE);
            if x >= w || y >= h {
                continue;
            }
            let Some(px) = buf.get(o * stride..(o + 1) * stride).and_then(|px| texel(c, px)) else { continue };
            let i = (y as usize * w as usize + x as usize) * 4;
            if let Some(dst) = img.data.get_mut(i..i + 4) {
                dst.copy_from_slice(&px);
            }
        }
    }
    img
}

/// The set's composite of channel `c`: opaque where the mesh covers the texture.
fn merged_image(s: &Session, ts: usize, c: Channel) -> Result<Rgba16> {
    let cache = s.caches.get(ts).ok_or_else(|| EngineError::Other(format!("no texture set {ts}")))?;
    let (w, h) = cache.size;
    let coverage = cache.gbuf.coverage_mask();
    let mut img = Rgba16::transparent(w, h);
    for y in 0..h {
        for x in 0..w {
            let i = y as usize * w as usize + x as usize;
            if !coverage.get(i).copied().unwrap_or(false) {
                continue;
            }
            let v = cache.composite.value(c, x, y);
            let rgb = match c.kind() {
                ChannelKind::Color => v.map(|v| linear_to_srgb(v.clamp(0.0, 1.0))),
                ChannelKind::NormalMap => v,
                ChannelKind::Scalar => [v[0]; 3],
                ChannelKind::Signed => [v[0] * 0.5 + 0.5; 3],
            };
            if let Some(dst) = img.data.get_mut(i * 4..i * 4 + 4) {
                dst.copy_from_slice(&[to_u16(rgb[0]), to_u16(rgb[1]), to_u16(rgb[2]), 65535]);
            }
        }
    }
    Ok(img)
}

/// Builds the OpenRaster tree of one texture set and channel.
struct Tree<'s> {
    s: &'s Session,
    ts: usize,
    size: (u32, u32),
    warnings: Vec<String>,
}

impl Tree<'_> {
    fn warn(&mut self, layer: &str, msg: &str) {
        let line = format!("\"{layer}\": {msg}");
        if !self.warnings.contains(&line) {
            self.warnings.push(line);
        }
    }

    /// `ids` (bottom → top) as nodes top-first; layers with `c` disabled are left out.
    fn nodes(&mut self, ids: &[LayerId], c: Channel, flat: &mut LayerCache, depth: usize) -> Result<Vec<Node>> {
        if depth > ora::MAX_DEPTH {
            return Err(EngineError::Other(format!("folders are nested more than {} deep", ora::MAX_DEPTH)));
        }
        let mut out = Vec::new();
        for &id in ids.iter().rev() {
            let layer = self.s.doc.project.layer(id).map_err(|e| EngineError::Other(e.to_string()))?;
            let Some(blend) = layer.blend(c).filter(|b| b.mode != BlendMode::Disable) else { continue };
            let (op, note) = composite_op(blend.mode);
            if let Some(note) = note {
                self.warn(&layer.name, note);
            }
            let kind = match &layer.kind {
                LayerKind::Group { children, .. } => {
                    if layer.mask.is_some() {
                        self.warn(&layer.name, "a folder's mask can't be sent; its layers are sent without it");
                    }
                    NodeKind::Stack { children: self.nodes(children, c, flat, depth + 1)?, pass_through: blend.mode == BlendMode::Passthrough }
                }
                _ => NodeKind::Layer(flat.image(self.s, self.ts, id, c, self.size)),
            };
            out.push(Node { name: layer.name.clone(), visible: layer.visible, opacity: blend.opacity, composite_op: op, kind });
        }
        Ok(out)
    }
}

/// The flattened tiles of the last layer asked for (each layer is evaluated once per channel
/// pass, never all of them at once).
#[derive(Default)]
struct LayerCache {
    last: Option<(LayerId, Vec<texturecraft_compose::eval::FlatTile>)>,
}

impl LayerCache {
    fn image(&mut self, s: &Session, ts: usize, id: LayerId, c: Channel, size: (u32, u32)) -> Rgba16 {
        if self.last.as_ref().is_none_or(|(l, _)| *l != id) {
            self.last = Some((id, s.flatten(ts, id).unwrap_or_default()));
        }
        self.last.as_ref().map(|(_, tiles)| flat_image(c, size, tiles)).unwrap_or_else(|| Rgba16::transparent(size.0, size.1))
    }
}

impl Session {
    /// Texture set `ts`'s channel `c` as an OpenRaster document, plus what it couldn't keep.
    pub fn photocraft_document(&mut self, ts: usize, c: Channel) -> Result<(Vec<u8>, Vec<String>)> {
        self.update();
        let set = self.doc.project.texture_set(ts).map_err(|e| EngineError::Other(e.to_string()))?;
        if !set.channel_list().contains(&c) {
            return Err(bad(SEND, format!("texture set \"{}\" has no {} channel", set.name, c.id())));
        }
        let stack = set.stack.clone();
        let size = self.caches.get(ts).map(|c| c.size).ok_or_else(|| EngineError::Other(format!("no texture set {ts}")))?;
        let mut tree = Tree { s: self, ts, size, warnings: Vec::new() };
        let nodes = tree.nodes(&stack, c, &mut LayerCache::default(), 1)?;
        let mut warnings = tree.warnings;
        if matches!(c.kind(), ChannelKind::NormalMap) {
            warnings.push(format!("{}: PhotoCraft blends normal layers as colours, not as normals", c.label()));
        }
        let merged = merged_image(self, ts, c)?;
        let bytes = ora::write_ora(&nodes, &merged).map_err(|e| EngineError::Other(e.to_string()))?;
        Ok((bytes, warnings))
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn channels_param(s: &Session, ts: usize, p: &Value) -> Result<Vec<Channel>> {
    let set = s.doc.project.texture_set(ts).map_err(|e| bad(SEND, e.to_string()))?;
    match p.get("channels") {
        None | Some(Value::Null) => Ok(set.channel_list()),
        Some(Value::Array(ids)) => ids
            .iter()
            .map(|v| v.as_str().and_then(Channel::parse).ok_or_else(|| bad(SEND, format!("unknown channel {v}"))))
            .collect::<Result<Vec<_>>>()
            .map(|cs| cs.into_iter().filter(|c| set.channel_list().contains(c)).collect()),
        Some(_) => Err(bad(SEND, "`channels` must be a list of channel ids")),
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn sets_param(s: &Session, p: &Value) -> Result<Vec<usize>> {
    match p.get("ts") {
        None | Some(Value::Null) => Ok(vec![s.ts()]),
        Some(Value::String(a)) if a == "all" => Ok((0..s.set_count()).collect()),
        Some(v) => match v.as_u64().and_then(|i| usize::try_from(i).ok()) {
            Some(i) if i < s.set_count() => Ok(vec![i]),
            _ => Err(bad(SEND, format!("no texture set {v}"))),
        },
    }
}

#[cfg(not(target_arch = "wasm32"))]
fn send(s: &mut Session, p: &Value) -> Result<Value> {
    let dir = p.get("dir").and_then(Value::as_str).filter(|d| !d.trim().is_empty()).ok_or_else(|| bad(SEND, "`dir` is required"))?;
    let dir = std::path::PathBuf::from(dir);
    let sets = sets_param(s, p)?;
    let open = p.get("open").and_then(Value::as_bool).unwrap_or(true);
    let mut jobs = Vec::new();
    for &ts in &sets {
        for c in channels_param(s, ts, p)? {
            jobs.push((ts, c));
        }
    }
    if jobs.is_empty() {
        return Err(bad(SEND, "none of the channels asked for is in the texture set"));
    }
    std::fs::create_dir_all(&dir).map_err(|e| bad(SEND, format!("cannot create {}: {e}", dir.display())))?;
    let mesh = std::path::Path::new(s.doc.project.mesh.trim_start_matches("builtin:")).file_stem().and_then(|n| n.to_str()).unwrap_or("Untitled").to_string();
    let mut files = Vec::new();
    let mut warnings = Vec::new();
    for (ts, c) in jobs {
        let set_name = s.doc.project.texture_set(ts).map(|t| t.name.clone()).unwrap_or_default();
        let stem = texturecraft_formats::export::sanitize_file_stem(&format!("{mesh}_{set_name}_{}", c.id()));
        let path = dir.join(format!("{stem}.{}", ora::ORA_EXT));
        if files.contains(&path) {
            warnings.push(format!("texture set \"{set_name}\" sends as {} like an earlier set, so it was skipped; rename one of them", path.display()));
            continue;
        }
        let (bytes, w) = s.photocraft_document(ts, c)?;
        warnings.extend(w.into_iter().map(|w| format!("{set_name} {}: {w}", c.label())));
        texturecraft_formats::write_atomic(&path, &bytes).map_err(|e| bad(SEND, format!("cannot write {}: {e}", path.display())))?;
        let modified = std::fs::metadata(&path).and_then(|m| m.modified()).ok();
        let layer = s.photocraft.links.iter().find(|l| l.path == path).and_then(|l| l.layer);
        s.photocraft.links.retain(|l| l.path != path);
        s.photocraft.links.push(Link { path: path.clone(), ts, channel: c, modified, layer });
        files.push(path);
    }
    s.log.push(format!("Sent {} documents to {}", files.len(), dir.display()));
    if open {
        open_in_photocraft(&files)?;
    }
    Ok(json!({ "files": files.iter().map(|p| p.display().to_string()).collect::<Vec<_>>(), "warnings": warnings }))
}

#[cfg(target_arch = "wasm32")]
fn send(_: &mut Session, _: &Value) -> Result<Value> {
    Err(bad(SEND, "the browser can't start PhotoCraft or watch its files; use the desktop app"))
}

/// Open `files` in PhotoCraft (`$PHOTOCRAFT`, next to this program, or `photocraft` on `PATH`),
/// which hands them to a PhotoCraft already running.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_in_photocraft(files: &[std::path::PathBuf]) -> Result<()> {
    let app = crate::link_cmds::materialcraft_tool("photocraft", "PHOTOCRAFT");
    std::process::Command::new(&app).arg("--").args(files).spawn().map(drop).map_err(|e| {
        EngineError::Other(format!("PhotoCraft ({}) did not start: {e}. Install PhotoCraft next to TextureCraft or set PHOTOCRAFT", app.display()))
    })
}

/// Largest linked file read back.
#[cfg(not(target_arch = "wasm32"))]
const MAX_LINKED_BYTES: u64 = 2 << 30;

#[cfg(not(target_arch = "wasm32"))]
fn sync_now(s: &mut Session) -> Result<Value> {
    let updated = s.sync_photocraft();
    Ok(json!({ "updated": updated.iter().map(|p| p.display().to_string()).collect::<Vec<_>>() }))
}

#[cfg(target_arch = "wasm32")]
fn sync_now(_: &mut Session) -> Result<Value> {
    Ok(json!({ "updated": [] }))
}

impl Session {
    /// Look at the linked files at most every [`POLL_MS`] of `now_ms` (the caller's clock) and
    /// show what PhotoCraft saved; true when something changed. The UI calls this every frame.
    pub fn poll_photocraft(&mut self, now_ms: f64) -> bool {
        // Never in the middle of a stroke: the update is an edit of its own.
        if self.photocraft.links.is_empty() || self.stroke_active() || self.photocraft.last_poll.is_some_and(|t| now_ms - t < POLL_MS && now_ms >= t) {
            return false;
        }
        self.photocraft.last_poll = Some(now_ms);
        #[cfg(not(target_arch = "wasm32"))]
        let changed = !self.sync_photocraft().is_empty();
        #[cfg(target_arch = "wasm32")]
        let changed = false;
        changed
    }

    /// Read every linked file whose modification time changed; returns the files shown.
    /// A file that can't be read is logged and retried when it changes again.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn sync_photocraft(&mut self) -> Vec<std::path::PathBuf> {
        let mut updated = Vec::new();
        for i in 0..self.photocraft.links.len() {
            let Some(link) = self.photocraft.links.get(i).cloned() else { continue };
            let Ok(modified) = std::fs::metadata(&link.path).and_then(|m| m.modified()) else { continue };
            if link.modified == Some(modified) {
                continue;
            }
            if let Some(l) = self.photocraft.links.get_mut(i) {
                l.modified = Some(modified);
            }
            match self.show_photocraft_file(&link) {
                Ok(layer) => {
                    if let Some(l) = self.photocraft.links.get_mut(i) {
                        l.layer = Some(layer);
                    }
                    self.log.push(format!("Updated {} from PhotoCraft", link.path.display()));
                    updated.push(link.path.clone());
                }
                Err(e) => self.log.push(format!("PhotoCraft: {}: {e}", link.path.display())),
            }
        }
        updated
    }

    /// Put the merged image of `link`'s file into its `PhotoCraft: <channel>` fill (created at the
    /// top of the set when missing), as one undo step.
    #[cfg(not(target_arch = "wasm32"))]
    fn show_photocraft_file(&mut self, link: &Link) -> Result<LayerId> {
        use std::io::Read;
        let io = |e: std::io::Error| EngineError::Other(e.to_string());
        let file = std::fs::File::open(&link.path).map_err(io)?;
        let mut bytes = Vec::new();
        file.take(MAX_LINKED_BYTES + 1).read_to_end(&mut bytes).map_err(io)?;
        if bytes.len() as u64 > MAX_LINKED_BYTES {
            return Err(EngineError::Other(format!("larger than {} MB", MAX_LINKED_BYTES >> 20)));
        }
        let png = ora::read_merged(&bytes).map_err(|e| EngineError::Other(e.to_string()))?;
        self.show_photocraft_image(link.ts, link.channel, link.layer, &png)
    }

    /// Show `png` as channel `c` of texture set `ts` through the fill `layer` (when it is still a
    /// fill of that set) or a new `PhotoCraft: <channel>` fill at the top. The previous image
    /// resource is removed when nothing else uses it.
    pub fn show_photocraft_image(&mut self, ts: usize, c: Channel, layer: Option<LayerId>, png: &[u8]) -> Result<LayerId> {
        let (w, h) = self.caches.get(ts).map(|c| c.size).ok_or_else(|| EngineError::Other(format!("no texture set {ts}")))?;
        let set_name = self.doc.project.texture_set(ts).map_err(|e| EngineError::Other(e.to_string()))?.name.clone();
        self.edit(UPDATE, |s| {
            let resource = texturecraft_formats::export::sanitize_file_stem(&format!("photocraft_{set_name}_{}", c.id()));
            let name = s.add_image_resource(&format!("{resource}.png"), png.to_vec(), u64::from(w) * u64::from(h) * 4)?;
            let source = texturecraft_core::Source::image(name);
            let existing = layer.filter(|id| s.doc.project.locate(*id).is_some_and(|(t, _, _)| t == ts));
            let old = existing.and_then(|id| match s.doc.project.layer(id).ok().map(|l| &l.kind) {
                Some(LayerKind::Fill(f)) => Some(f.sources.get(&c).cloned()),
                _ => None,
            });
            let id = match (existing, old) {
                (Some(id), Some(old)) => {
                    if let Ok(l) = s.doc.project.layer_mut(id)
                        && let LayerKind::Fill(f) = &mut l.kind
                    {
                        f.sources.insert(c, source);
                    }
                    if let Some(texturecraft_core::Source::Image { name, .. }) = old {
                        s.remove_resource_if_unused(&name);
                    }
                    id
                }
                _ => {
                    let id = s.doc.project.alloc_id();
                    let mut fill = texturecraft_core::Fill::default();
                    fill.sources.insert(c, source);
                    let layer = texturecraft_core::Layer::new(id, format!("PhotoCraft: {}", c.label()), LayerKind::Fill(fill), &[c]);
                    s.doc.project.insert_layer(ts, layer, None, false).map_err(|e| EngineError::Other(e.to_string()))?
                }
            };
            Ok(id)
        })
    }

    /// Drop image resource `name` when no fill source uses it any more.
    fn remove_resource_if_unused(&mut self, name: &str) {
        let used = self.doc.project.layers.values().any(|l| match &l.kind {
            LayerKind::Fill(f) => f.sources.values().any(|s| matches!(s, texturecraft_core::Source::Image { name: n, .. } if n == name)),
            _ => false,
        });
        if !used {
            self.doc.resources.remove(name);
            self.images.remove(name);
        }
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
mod tests {
    use super::*;
    use crate::execute;
    use std::io::Read;
    use texturecraft_geom::primitives;

    /// A fresh folder under the system temp dir, removed when dropped.
    struct TempDir(std::path::PathBuf);

    impl TempDir {
        fn path(&self) -> &std::path::Path {
            &self.0
        }
    }

    impl Drop for TempDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn tempdir(tag: &str) -> TempDir {
        let dir = std::env::temp_dir().join(format!("tc_photocraft_{tag}_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        TempDir(dir)
    }

    fn session() -> Session {
        let mut s = Session::new_project("t", primitives::plane(2.0, 2), "builtin:plane", None, 128).unwrap();
        s.update();
        s
    }

    fn stack_xml(path: &std::path::Path) -> String {
        let mut z = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
        let mut xml = String::new();
        z.by_name("stack.xml").unwrap().read_to_string(&mut xml).unwrap();
        xml
    }

    fn layer_png(path: &std::path::Path, src: &str) -> image::Rgba<u16> {
        let mut z = zip::ZipArchive::new(std::fs::File::open(path).unwrap()).unwrap();
        let mut png = Vec::new();
        z.by_name(src).unwrap().read_to_end(&mut png).unwrap();
        let img = image::load_from_memory(&png).unwrap().into_rgba16();
        *img.get_pixel(img.width() / 2, img.height() / 2)
    }

    #[test]
    fn sends_every_layer_with_its_mask_as_transparency_and_folders_as_groups() {
        let dir = tempdir("groups");
        let mut s = session();
        execute(&mut s, "layer.add.fill", &json!({"color": [1.0, 0.0, 0.0], "roughness": 0.2})).unwrap();
        let fill = s.selected_layer().unwrap();
        execute(&mut s, "layer.rename", &json!({"layer": fill.0, "name": "Red"})).unwrap();
        execute(&mut s, "mask.add.black", &Value::Null).unwrap();
        execute(&mut s, "layer.group", &json!({"layer": fill.0})).unwrap();
        let r = execute(&mut s, SEND, &json!({"dir": dir.path().display().to_string(), "channels": ["baseColor", "roughness"], "open": false})).unwrap();
        let files: Vec<String> = r["files"].as_array().unwrap().iter().map(|f| f.as_str().unwrap().to_string()).collect();
        assert_eq!(files.len(), 2, "{r}");
        let base = std::path::PathBuf::from(files.iter().find(|f| f.ends_with("_baseColor.ora")).unwrap());
        let xml = stack_xml(&base);
        // The fill sits in its folder, above the project's first paint layer (top-first).
        let (group, red) = (xml.find("<stack name=").unwrap(), xml.find("name=\"Red\"").unwrap());
        let paint = xml.find("name=\"Layer 1\"").unwrap();
        assert!(group < red && red < xml.find("</stack>").unwrap() && xml.find("</stack>").unwrap() < paint, "{xml}");
        assert!(xml.contains("isolation=\"auto\""), "a new folder passes through: {xml}");
        // A black mask hides the whole fill: its layer is empty (written 1×1, transparent).
        let src = xml[red..].split("src=\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        assert_eq!(layer_png(&base, &src).0[3], 0);
        // The links are remembered.
        assert_eq!(s.photocraft.links.len(), 2);
        assert_eq!(links_json(&s).as_array().unwrap().len(), 2);
        assert_eq!(crate::inspect::session(&s)["photocraftLinks"].as_array().unwrap().len(), 2);
    }

    #[test]
    fn a_visible_fill_is_sent_in_srgb_and_a_generator_mask_shapes_its_alpha() {
        let dir = tempdir("srgb");
        let mut s = session();
        execute(&mut s, "layer.add.fill", &json!({"color": [0.5, 0.5, 0.5]})).unwrap();
        let r = execute(&mut s, SEND, &json!({"dir": dir.path().display().to_string(), "channels": ["baseColor"], "open": false})).unwrap();
        let path = std::path::PathBuf::from(r["files"][0].as_str().unwrap());
        let xml = stack_xml(&path);
        let src = xml.split("src=\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        let px = layer_png(&path, &src);
        let want = (linear_to_srgb(0.5) * 65535.0) as i32;
        assert!((px.0[0] as i32 - want).abs() < 300 && px.0[3] == 65535, "{px:?} vs {want}");

        // A generator in the mask: the layer's alpha is the mask the compositor evaluates.
        let fill = s.selected_layer().unwrap();
        execute(&mut s, "mask.add.black", &Value::Null).unwrap();
        execute(&mut s, "effect.add.generator", &json!({"generator": "position"})).unwrap();
        s.update();
        let r = execute(&mut s, SEND, &json!({"dir": dir.path().display().to_string(), "channels": ["baseColor"], "open": false})).unwrap();
        let path = std::path::PathBuf::from(r["files"][0].as_str().unwrap());
        let xml = stack_xml(&path);
        let src = xml.split("src=\"").nth(1).unwrap().split('"').next().unwrap().to_string();
        let mut z = zip::ZipArchive::new(std::fs::File::open(&path).unwrap()).unwrap();
        let mut png = Vec::new();
        z.by_name(&src).unwrap().read_to_end(&mut png).unwrap();
        let img = image::load_from_memory(&png).unwrap().into_rgba16();
        let (x0, y0): (u32, u32) = {
            let attr = |k: &str| xml.split(&format!(" {k}=\"")).nth(1).unwrap().split('"').next().unwrap().parse::<u32>().unwrap();
            (attr("x"), attr("y"))
        };
        let (x, y) = (64u32, 64u32);
        let mask = s.with_inputs(0, |inp| texturecraft_compose::mask_at(inp, fill, &[(x, y)])).unwrap()[0];
        let alpha = img.get_pixel(x - x0, y - y0).0[3] as f32 / 65535.0;
        assert!(mask > 0.0, "the generator shows part of the fill: {mask}");
        assert!((alpha - mask).abs() < 2.0 / 255.0, "alpha {alpha} vs mask {mask}");
    }

    #[test]
    fn a_save_in_photocraft_shows_live_and_undoes() {
        let dir = tempdir("live");
        let mut s = session();
        let r = execute(&mut s, SEND, &json!({"dir": dir.path().display().to_string(), "channels": ["baseColor"], "open": false})).unwrap();
        let path = std::path::PathBuf::from(r["files"][0].as_str().unwrap());
        // Nothing changed since the send.
        assert!(s.sync_photocraft().is_empty());
        // PhotoCraft saves: a green merged image.
        let (w, h) = s.caches[0].size;
        let mut merged = Rgba16::transparent(w, h);
        for px in merged.data.as_chunks_mut::<4>().0 {
            *px = [0, 65535, 0, 65535];
        }
        std::fs::write(&path, ora::write_ora(&[], &merged).unwrap()).unwrap();
        let later = std::time::SystemTime::now() + std::time::Duration::from_secs(5);
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later).unwrap();
        assert_eq!(s.sync_photocraft(), vec![path.clone()]);
        s.update();
        let v = s.caches[0].composite.value(Channel::BaseColor, w / 2, h / 2);
        assert!(v[1] > 0.9 && v[0] < 0.1, "{v:?}");
        let layer = s.photocraft.links[0].layer.unwrap();
        assert!(s.doc.project.layer(layer).unwrap().name.starts_with("PhotoCraft"));
        // A second save reuses the layer and its old image resource goes.
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later + std::time::Duration::from_secs(5)).unwrap();
        assert_eq!(s.sync_photocraft().len(), 1);
        assert_eq!(s.photocraft.links[0].layer, Some(layer));
        assert_eq!(s.doc.resources.keys().filter(|k| k.contains("photocraft")).count(), 1, "{:?}", s.doc.resources.keys().collect::<Vec<_>>());
        // A broken file is logged, not fatal.
        std::fs::write(&path, b"broken").unwrap();
        std::fs::File::options().write(true).open(&path).unwrap().set_modified(later + std::time::Duration::from_secs(10)).unwrap();
        assert!(s.sync_photocraft().is_empty());
        assert!(s.log.iter().any(|l| l.starts_with("PhotoCraft:")));
        // Each update is one undo step: undoing it brings back the stack as it was sent.
        execute(&mut s, "edit.undo", &Value::Null).unwrap();
        execute(&mut s, "edit.undo", &Value::Null).unwrap();
        s.sync_images();
        assert!(s.doc.project.layer(layer).is_err());
        // Unlinking stops the watch.
        execute(&mut s, "photocraft.unlink", &Value::Null).unwrap();
        assert!(s.photocraft.links.is_empty());
    }

    #[test]
    fn bad_params_are_errors() {
        let mut s = session();
        assert!(execute(&mut s, SEND, &json!({})).is_err());
        assert!(execute(&mut s, SEND, &json!({"dir": "  "})).is_err());
        let dir = tempdir("bad");
        let d = dir.path().display().to_string();
        assert!(execute(&mut s, SEND, &json!({"dir": d, "ts": 99, "open": false})).is_err());
        assert!(execute(&mut s, SEND, &json!({"dir": d, "channels": ["nope"], "open": false})).is_err());
        assert!(execute(&mut s, SEND, &json!({"dir": d, "channels": "baseColor", "open": false})).is_err());
        assert!(execute(&mut s, SEND, &json!({"dir": d, "channels": ["coatNormal"], "open": false})).is_err());
        assert!(s.show_photocraft_image(0, Channel::BaseColor, None, b"not a png").is_err());
        assert!(s.doc.project.walk(0).iter().all(|(id, _)| !s.doc.project.layer(*id).unwrap().name.starts_with("PhotoCraft")));
    }
}
