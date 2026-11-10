//! The PhotoCraft hand-off as one layered PSD per texture set.
//!
//! The stack keeps its shape: every folder is a group and every layer a pass-through group
//! named after it, carrying the layer's mask as an editable group mask; inside it, one pixel
//! layer per channel (`Base color`, `Roughness`, ...) holds the layer's content for that
//! channel with the channel's blend mode and opacity. One layer comp per channel shows only
//! that channel's layers, so the whole material is one document and a mask edited once
//! shapes every channel.
//!
//! When PhotoCraft saves the file, [`composite`] rebuilds each channel from the saved layers:
//! a pixel layer counts for the channel it is named after, through every group mask, opacity
//! and visibility above it, blended with TextureCraft's own blend math.

use texturecraft_core::{BlendMode, Channel, ChannelKind, LayerId, LayerKind, blend1, blend3, mix3};
use texturecraft_formats::ora::Rgba16;
use texturecraft_formats::psd::{self, EncodedMask, Kind, Mask16, ReadDocument, ReadKind, ReadLayer};
use texturecraft_texel::{TILE, TILE_AREA, from_u16, srgb_to_linear};

use crate::photocraft_cmds::{encode_value, flat_image, merged_image};
use crate::{EngineError, Result, Session};

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
fn mode_of(key: &[u8; 4]) -> BlendMode {
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
}

impl Tree<'_> {
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
                    // A folder has one blend mode and opacity in the PSD; TextureCraft has one per channel.
                    let first = enabled.first().map(|(_, b)| *b).unwrap_or_default();
                    let uniform = enabled.iter().all(|(_, b)| b.mode == first.mode && (b.opacity - first.opacity).abs() < 1e-4);
                    let (blend, opacity) = if uniform {
                        let (key, note) = psd_key(first.mode);
                        if let Some(note) = note {
                            self.warn(&layer.name, note);
                        }
                        (key, first.opacity)
                    } else {
                        self.warn(&layer.name, "the folder blends its channels differently; it is sent as a pass-through group at 100%");
                        (*b"pass", 1.0)
                    };
                    let children = self.nodes(children, depth + 1)?;
                    psd::Layer {
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
    /// couldn't keep.
    pub fn photocraft_psd(&mut self, ts: usize, channels: &[Channel]) -> Result<(Vec<u8>, Vec<String>)> {
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
        let mut tree = Tree { s: self, ts, size, channels: channels.clone(), covered, warnings: Vec::new() };
        let layers = tree.nodes(&stack, 1)?;
        let mut warnings = tree.warnings;
        if channels.iter().any(|c| c.kind() == ChannelKind::NormalMap) {
            warnings.push("normal channels: PhotoCraft blends normal layers as colours, not as normals".into());
        }
        let doc = psd::Document { layers, merged: merged_image(self, ts, first)?, comps: channels.iter().map(|c| c.label().to_string()).collect() };
        let bytes = psd::write_psd(&doc).map_err(|e| EngineError::Other(e.to_string()))?;
        Ok((bytes, warnings))
    }
}

/// A channel value from a saved pixel (straight, 0..65535).
fn decode(c: Channel, px: &[u16]) -> [f32; 3] {
    let v = |k: usize| px.get(k).map_or(0.0, |v| from_u16(*v));
    match c.kind() {
        ChannelKind::Color => [0, 1, 2].map(|k| srgb_to_linear(v(k))),
        ChannelKind::NormalMap => [v(0), v(1), v(2)],
        ChannelKind::Scalar => [v(0); 3],
        ChannelKind::Signed => [v(0) * 2.0 - 1.0; 3],
    }
}

/// Channel `c` of a saved document, `w`×`h`: the channel's default value with every layer named
/// after `c` blended over it (see the module docs). Encoded as the hand-off writes channels;
/// opaque where `coverage` is set.
pub(crate) fn composite(doc: &ReadDocument, c: Channel, coverage: &[bool], notes: &mut Vec<String>) -> Rgba16 {
    let (w, h) = (doc.width as usize, doc.height as usize);
    let mut acc = vec![c.default_value(); w * h];
    walk(&doc.layers, c, &mut acc, (w, h), None, 0, notes);
    let mut img = Rgba16::transparent(doc.width, doc.height);
    for (i, v) in acc.iter().enumerate() {
        if coverage.get(i).copied().unwrap_or(false)
            && let Some(dst) = img.data.get_mut(i * 4..i * 4 + 4)
        {
            dst.copy_from_slice(&encode_value(c, *v));
        }
    }
    img
}

fn is_channel_name(name: &str) -> bool {
    Channel::ALL.iter().any(|c| c.label() == name.trim())
}

/// Blend `nodes` (top-first) into `acc`, each weighted by `weight` (the masks, opacities and
/// visibility of the groups above; None: 1).
fn walk(nodes: &[ReadLayer], c: Channel, acc: &mut [[f32; 3]], (w, h): (usize, usize), weight: Option<&[f32]>, depth: usize, notes: &mut Vec<String>) {
    if depth > psd::MAX_DEPTH {
        return;
    }
    for node in nodes.iter().rev() {
        match &node.kind {
            ReadKind::Group(children) => {
                if !node.visible || node.opacity <= 0.0 {
                    continue;
                }
                let mut wgt: Vec<f32> = weight.map_or_else(|| vec![1.0; w * h], <[f32]>::to_vec);
                for (i, v) in wgt.iter_mut().enumerate() {
                    let m = node.mask.as_ref().map_or(1.0, |m| from_u16(m.at((i % w) as i32, (i / w) as i32)));
                    *v *= m * node.opacity;
                }
                walk(children, c, acc, (w, h), Some(&wgt), depth + 1, notes);
            }
            ReadKind::Pixels { x, y, image } => {
                // Channel layers are shown one comp at a time, so their own visibility is ignored.
                if node.name.trim() != c.label() {
                    if node.visible && !is_channel_name(&node.name) {
                        let note = format!("\"{}\" is not named after a channel (such as \"{}\"), so it isn't brought back", node.name, c.label());
                        if !notes.contains(&note) {
                            notes.push(note);
                        }
                    }
                    continue;
                }
                let mode = mode_of(&node.blend);
                let (iw, ih) = (image.width as usize, image.height as usize);
                for ly in 0..ih {
                    let cy = i64::from(*y) + ly as i64;
                    if cy < 0 || cy >= h as i64 {
                        continue;
                    }
                    for lx in 0..iw {
                        let cx = i64::from(*x) + lx as i64;
                        if cx < 0 || cx >= w as i64 {
                            continue;
                        }
                        let i = cy as usize * w + cx as usize;
                        let Some(px) = image.data.get((ly * iw + lx) * 4..(ly * iw + lx) * 4 + 4) else { continue };
                        let mask = node.mask.as_ref().map_or(1.0, |m| from_u16(m.at(cx as i32, cy as i32)));
                        let a = from_u16(px[3]) * node.opacity * mask * weight.and_then(|wg| wg.get(i)).copied().unwrap_or(1.0);
                        if a <= 0.0 {
                            continue;
                        }
                        let Some(dst) = acc.get_mut(i) else { continue };
                        let v = decode(c, px);
                        let blended = if c.components() == 3 { blend3(mode, *dst, v) } else { [blend1(mode, dst[0], v[0]); 3] };
                        *dst = mix3(*dst, blended, a.min(1.0));
                    }
                }
            }
        }
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
        Tree { s, ts: 0, size, channels: vec![Channel::BaseColor], covered, warnings: Vec::new() }
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
