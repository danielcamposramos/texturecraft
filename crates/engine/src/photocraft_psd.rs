//! The PhotoCraft hand-off as one layered PSD per texture set.
//!
//! The stack keeps its shape: every folder is a group and every layer a pass-through group
//! named after it, carrying the layer's mask as an editable group mask; inside it, one pixel
//! layer per channel (`Base color`, `Roughness`, ...) holds the layer's content for that
//! channel with the channel's blend mode and opacity. One layer comp per channel shows only
//! that channel's layers, so the whole material is one document and a mask edited once
//! shapes every channel.
//!
//! Every node gets a layer id (`lyid`, which PhotoCraft keeps across saves) from
//! [`FIRST_ID`] up, and the send remembers what each id stands for ([`Role`]) and how it looked
//! ([`Sent`]). When PhotoCraft saves the file, [`crate::photocraft_sync`] compares the saved
//! layers with that record and mirrors the changes into the stack.

use std::collections::BTreeMap;
use std::hash::{DefaultHasher, Hasher};

use texturecraft_core::{BlendMode, Channel, ChannelKind, LayerId, LayerKind};
use texturecraft_formats::psd::{self, EncodedMask, Kind, Mask16, ReadKind, ReadLayer, ReadMask};
use texturecraft_texel::{TILE, TILE_AREA, from_u16, srgb_to_linear};

use crate::photocraft_cmds::{flat_image, merged_image};
use crate::{EngineError, Result, Session};

/// The first layer id a send gives; PhotoCraft numbers the layers it adds from 1.
pub const FIRST_ID: u32 = 0x4000_0000;

/// What a node of a sent document stands for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A folder, as a group.
    Folder(LayerId),
    /// A layer, as a pass-through group holding its channels.
    Layer(LayerId),
    /// One channel of a layer, as a pixel layer inside the layer's group.
    Channel(LayerId, Channel),
    /// A layer brought in from PhotoCraft, as the one pixel layer it came from.
    Pixels(LayerId),
}

impl Role {
    pub fn layer(self) -> LayerId {
        match self {
            Role::Folder(id) | Role::Layer(id) | Role::Channel(id, _) | Role::Pixels(id) => id,
        }
    }
}

/// How a node looked when it was sent or last read back.
#[derive(Clone, Debug, PartialEq)]
pub struct Sent {
    pub role: Role,
    pub name: String,
    pub visible: bool,
    /// 0..255, as stored.
    pub opacity: u8,
    pub blend: [u8; 4],
    /// Fingerprints of the node's mask and pixels (0: none), independent of compression and
    /// cropping.
    pub mask: u64,
    pub pixels: u64,
}

/// A sent document's nodes by layer id.
pub type SentMap = BTreeMap<u32, Sent>;

/// The record of `node` as `role`.
pub(crate) fn record(node: &ReadLayer, role: Role) -> Sent {
    Sent {
        role,
        name: node.name.clone(),
        visible: node.visible,
        opacity: (node.opacity.clamp(0.0, 1.0) * 255.0).round() as u8,
        blend: node.blend,
        mask: mask_print(node.mask.as_ref()),
        pixels: pixel_print(&node.kind),
    }
}

fn mask_print(m: Option<&ReadMask>) -> u64 {
    let Some(m) = m else { return 0 };
    let mut h = DefaultHasher::new();
    h.write_u16(m.default);
    for (i, v) in m.data.iter().enumerate() {
        if *v != m.default {
            let (x, y) = (i64::from(m.x) + (i % m.width.max(1) as usize) as i64, i64::from(m.y) + (i / m.width.max(1) as usize) as i64);
            h.write_i64(x);
            h.write_i64(y);
            h.write_u16(*v);
        }
    }
    h.finish() | 1
}

fn pixel_print(kind: &ReadKind) -> u64 {
    let ReadKind::Pixels { x, y, image } = kind else { return 0 };
    let mut h = DefaultHasher::new();
    let w = image.width.max(1) as usize;
    for (i, px) in image.data.as_chunks::<4>().0.iter().enumerate() {
        if px[3] > 0 {
            h.write_i64(i64::from(*x) + (i % w) as i64);
            h.write_i64(i64::from(*y) + (i / w) as i64);
            for v in px {
                h.write_u16(*v);
            }
        }
    }
    h.finish() | 1
}

/// The PSD blend mode key for `mode`, with a warning when it is only close or missing.
pub(crate) fn psd_key(mode: BlendMode) -> ([u8; 4], Option<&'static str>) {
    use BlendMode as B;
    match mode {
        B::Normal => (*b"norm", None),
        B::Passthrough => (*b"pass", None),
        B::Multiply => (*b"mul ", None),
        B::Screen => (*b"scrn", None),
        B::Overlay => (*b"over", None),
        B::Darken => (*b"dark", None),
        B::Lighten => (*b"lite", None),
        B::ColorDodge => (*b"div ", None),
        B::ColorBurn => (*b"idiv", None),
        B::HardLight => (*b"hLit", None),
        B::SoftLight => (*b"sLit", None),
        B::Difference => (*b"diff", None),
        B::Exclusion => (*b"smud", None),
        B::LinearDodge => (*b"lddg", None),
        B::LinearBurn => (*b"lbrn", None),
        B::VividLight => (*b"vLit", None),
        B::LinearLight => (*b"lLit", None),
        B::PinLight => (*b"pLit", None),
        B::Subtract => (*b"fsub", None),
        B::Divide => (*b"fdiv", None),
        B::Saturation => (*b"sat ", None),
        B::Color => (*b"colr", None),
        B::Value => (*b"lum ", Some("Value is sent as Luminosity")),
        _ => (*b"norm", Some("this blend mode has no PSD equivalent; it is sent as Normal")),
    }
}

/// The blend mode a PSD key stands for (unknown keys blend as Normal).
pub(crate) fn mode_of(key: &[u8; 4]) -> BlendMode {
    use BlendMode as B;
    match key {
        b"mul " => B::Multiply,
        b"scrn" => B::Screen,
        b"over" => B::Overlay,
        b"dark" => B::Darken,
        b"lite" => B::Lighten,
        b"div " => B::ColorDodge,
        b"idiv" => B::ColorBurn,
        b"hLit" => B::HardLight,
        b"sLit" => B::SoftLight,
        b"diff" => B::Difference,
        b"smud" => B::Exclusion,
        b"lddg" => B::LinearDodge,
        b"lbrn" => B::LinearBurn,
        b"vLit" => B::VividLight,
        b"lLit" => B::LinearLight,
        b"pLit" => B::PinLight,
        b"fsub" => B::Subtract,
        b"fdiv" => B::Divide,
        b"sat " => B::Saturation,
        b"colr" => B::Color,
        b"lum " => B::Value,
        _ => B::Normal,
    }
}

/// Builds the PSD tree of one texture set.
struct Tree<'s> {
    s: &'s Session,
    ts: usize,
    size: (u32, u32),
    channels: Vec<Channel>,
    covered: Vec<(u32, u32)>,
    warnings: Vec<String>,
    roles: BTreeMap<u32, Role>,
}

impl Tree<'_> {
    /// The next node id, standing for `role`.
    fn id(&mut self, role: Role) -> u32 {
        let id = FIRST_ID.saturating_add(self.roles.len() as u32);
        self.roles.insert(id, role);
        id
    }

    fn warn(&mut self, layer: &str, msg: &str) {
        let line = format!("\"{layer}\": {msg}");
        if !self.warnings.contains(&line) {
            self.warnings.push(line);
        }
    }

    /// `ids` (bottom → top) as PSD layers top-first.
    fn nodes(&mut self, ids: &[LayerId], depth: usize) -> Result<Vec<psd::Layer>> {
        if depth > psd::MAX_DEPTH {
            return Err(EngineError::Other(format!("folders are nested more than {} deep", psd::MAX_DEPTH)));
        }
        let mut out = Vec::new();
        for &id in ids.iter().rev() {
            let layer = self.s.doc.project.layer(id).map_err(|e| EngineError::Other(e.to_string()))?.clone();
            let enabled: Vec<(Channel, texturecraft_core::ChannelBlend)> =
                self.channels.iter().filter_map(|c| layer.blend(*c).filter(|b| b.mode != BlendMode::Disable).map(|b| (*c, b))).collect();
            if enabled.is_empty() {
                continue;
            }
            let node = match &layer.kind {
                LayerKind::Group { children, .. } => {
                    // A TextureCraft folder blends its children with the layers below it and then mixes
                    // the result in by its opacity, whatever its mode: a pass-through group. An
                    // isolated PSD group would render a Multiply child differently. The PSD has one
                    // opacity per group; TextureCraft has one per channel.
                    let first = enabled.first().map(|(_, b)| b.opacity).unwrap_or(1.0);
                    let opacity = if enabled.iter().all(|(_, b)| (b.opacity - first).abs() < 1e-4) {
                        first
                    } else {
                        self.warn(&layer.name, "the folder's channels have different opacities; it is sent at 100%");
                        1.0
                    };
                    let blend = *b"pass";
                    let id_ = self.id(Role::Folder(id));
                    let children = self.nodes(children, depth + 1)?;
                    psd::Layer {
                        id: id_,
                        name: layer.name.clone(),
                        visible: layer.visible,
                        opacity,
                        blend,
                        mask: self.mask(id, &[])?,
                        comps: Vec::new(),
                        kind: Kind::Group(children),
                    }
                }
                _ => {
                    let (content, masks) = self.s.flatten_apart(self.ts, id).unwrap_or_default();
                    let mask = self.mask(id, &masks)?;
                    let mut children = Vec::new();
                    for (c, b) in &enabled {
                        let (key, note) = psd_key(b.mode);
                        if let Some(note) = note {
                            self.warn(&format!("{} › {}", layer.name, c.label()), note);
                        }
                        let image = psd::encode_pixels(&flat_image(*c, self.size, &content)).map_err(|e| EngineError::Other(e.to_string()))?;
                        let comps: Vec<bool> = self.channels.iter().map(|x| x == c).collect();
                        let first_comp = self.channels.first() == Some(c);
                        children.push(psd::Layer {
                            id: self.id(Role::Channel(id, *c)),
                            name: c.label().to_string(),
                            visible: first_comp,
                            opacity: b.opacity,
                            blend: key,
                            mask: None,
                            comps,
                            kind: Kind::Pixels(image),
                        });
                    }
                    psd::Layer {
                        id: self.id(Role::Layer(id)),
                        name: layer.name.clone(),
                        visible: layer.visible,
                        opacity: 1.0,
                        blend: *b"pass",
                        mask,
                        comps: Vec::new(),
                        kind: Kind::Group(children),
                    }
                }
            };
            out.push(node);
        }
        Ok(out)
    }

    /// The mask of layer or folder `id` over the covered texels, when it hides any of them.
    /// `tiles` are the mask tiles flattening gave (a layer); the texels they leave out, such as
    /// tiles where a paint layer is empty, are evaluated on their own.
    fn mask(&self, id: LayerId, tiles: &[texturecraft_compose::eval::MaskTile]) -> Result<Option<EncodedMask>> {
        let (w, h) = self.size;
        let index = |x: u32, y: u32| (x < w && y < h).then(|| y as usize * w as usize + x as usize);
        let mut data: Vec<Option<u16>> = vec![None; w as usize * h as usize];
        for ((tx, ty), values) in tiles {
            for (o, v) in values.iter().enumerate().take(TILE_AREA) {
                if let Some(d) = index(tx * TILE + o as u32 % TILE, ty * TILE + o as u32 / TILE).and_then(|i| data.get_mut(i)) {
                    *d = Some(*v);
                }
            }
        }
        let missing: Vec<(u32, u32)> =
            self.covered.iter().copied().filter(|&(x, y)| index(x, y).and_then(|i| data.get(i)).is_some_and(Option::is_none)).collect();
        if !missing.is_empty() {
            let values = self.s.with_inputs(self.ts, |inp| texturecraft_compose::mask_at(inp, id, &missing)).unwrap_or_default();
            for (&(x, y), v) in missing.iter().zip(values) {
                if let Some(d) = index(x, y).and_then(|i| data.get_mut(i)) {
                    *d = Some(texturecraft_texel::to_u16(v));
                }
            }
        }
        let hides = self.covered.iter().any(|&(x, y)| index(x, y).and_then(|i| data.get(i).copied().flatten()).is_some_and(|v| v < 65535));
        if !hides {
            return Ok(None);
        }
        let m = Mask16 { width: w, height: h, data: data.into_iter().map(|v| v.unwrap_or(0)).collect() };
        psd::encode_mask(&m).map(Some).map_err(|e| EngineError::Other(e.to_string()))
    }
}

impl Session {
    /// Layer `id` of texture set `ts` alone, its content and its mask apart (see
    /// [`texturecraft_compose::flatten_layer_masked_apart`]).
    pub(crate) fn flatten_apart(
        &self,
        ts: usize,
        id: LayerId,
    ) -> Option<(Vec<texturecraft_compose::eval::FlatTile>, Vec<texturecraft_compose::eval::MaskTile>)> {
        let tiles: Vec<(u32, u32)> = self.caches.get(ts)?.covered_tiles.iter().copied().collect();
        self.with_inputs(ts, |inp| texturecraft_compose::flatten_layer_masked_apart(inp, id, &tiles))
    }

    /// Texture set `ts` as one layered PSD of `channels` (see the module docs), plus what it
    /// couldn't keep and the record of its nodes.
    pub fn photocraft_psd(&mut self, ts: usize, channels: &[Channel]) -> Result<(Vec<u8>, Vec<String>, SentMap)> {
        self.update();
        let set = self.doc.project.texture_set(ts).map_err(|e| EngineError::Other(e.to_string()))?;
        let channels: Vec<Channel> = channels.iter().copied().filter(|c| set.channel_list().contains(c)).collect();
        let Some(&first) = channels.first() else {
            return Err(EngineError::Other(format!("texture set \"{}\" has none of the channels asked for", set.name)));
        };
        let stack = set.stack.clone();
        let cache = self.caches.get(ts).ok_or_else(|| EngineError::Other(format!("no texture set {ts}")))?;
        let size = cache.size;
        let covered: Vec<(u32, u32)> = cache
            .gbuf
            .coverage_mask()
            .iter()
            .enumerate()
            .filter(|(_, c)| **c)
            .map(|(i, _)| ((i % size.0 as usize) as u32, (i / size.0 as usize) as u32))
            .collect();
        let mut tree = Tree { s: self, ts, size, channels: channels.clone(), covered, warnings: Vec::new(), roles: BTreeMap::new() };
        let layers = tree.nodes(&stack, 1)?;
        let roles = tree.roles;
        let mut warnings = tree.warnings;
        if channels.iter().any(|c| c.kind() == ChannelKind::NormalMap) {
            warnings.push("normal channels: PhotoCraft blends normal layers as colours, not as normals".into());
        }
        let doc = psd::Document { layers, merged: merged_image(self, ts, first)?, comps: channels.iter().map(|c| c.label().to_string()).collect() };
        let bytes = psd::write_psd(&doc).map_err(|e| EngineError::Other(e.to_string()))?;
        // Recorded from the file itself, so a save that changes nothing compares equal.
        let back = psd::read_psd(&bytes).map_err(|e| EngineError::Other(e.to_string()))?;
        let mut sent = SentMap::new();
        record_all(&back.layers, &roles, &mut sent, 0);
        Ok((bytes, warnings, sent))
    }
}

fn record_all(nodes: &[ReadLayer], roles: &BTreeMap<u32, Role>, out: &mut SentMap, depth: usize) {
    if depth > psd::MAX_DEPTH {
        return;
    }
    for node in nodes {
        if let Some((id, role)) = node.id.and_then(|id| roles.get(&id).map(|r| (id, *r))) {
            out.insert(id, record(node, role));
        }
        if let ReadKind::Group(children) = &node.kind {
            record_all(children, roles, out, depth + 1);
        }
    }
}

/// A channel value from a saved pixel (straight, 0..65535).
pub(crate) fn decode(c: Channel, px: &[u16]) -> [f32; 3] {
    let v = |k: usize| px.get(k).map_or(0.0, |v| from_u16(*v));
    match c.kind() {
        ChannelKind::Color => [0, 1, 2].map(|k| srgb_to_linear(v(k))),
        ChannelKind::NormalMap => [v(0), v(1), v(2)],
        ChannelKind::Scalar => [v(0); 3],
        ChannelKind::Signed => [v(0) * 2.0 - 1.0; 3],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execute;
    use serde_json::Value;
    use texturecraft_geom::primitives;

    fn tree(s: &Session) -> Tree<'_> {
        let size = s.caches[0].size;
        let covered = s.caches[0]
            .gbuf
            .coverage_mask()
            .iter()
            .enumerate()
            .filter(|(_, c)| **c)
            .map(|(i, _)| ((i % size.0 as usize) as u32, (i / size.0 as usize) as u32))
            .collect();
        Tree { s, ts: 0, size, channels: vec![Channel::BaseColor], covered, warnings: Vec::new(), roles: BTreeMap::new() }
    }

    #[test]
    fn texels_flattening_leaves_out_are_masked_as_the_layer_is() {
        let mut s = Session::new_project("t", primitives::plane(2.0, 2), "builtin:plane", None, 128).unwrap();
        s.update();
        let paint = s.selected_layer().unwrap();
        // No mask tiles at all (an empty paint layer flattens to nothing): no mask either.
        assert!(tree(&s).mask(paint, &[]).unwrap().is_none());
        // A black mask hides every texel, tiles or not.
        execute(&mut s, "mask.add.black", &Value::Null).unwrap();
        s.update();
        assert!(tree(&s).mask(paint, &[]).unwrap().is_some());
    }
}
