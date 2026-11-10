//! A PSD saved in PhotoCraft mirrored back into the stack it was sent from.
//!
//! The saved layers are compared, by layer id, with what the send recorded
//! ([`crate::photocraft_psd::SentMap`]):
//!
//! - A group that was a folder stays that folder, and a group that was a layer stays that layer;
//!   their names, visibility, masks, blending and order follow the document.
//! - A channel's pixel layer whose pixels changed becomes that channel's paint (a fill or other
//!   procedural layer turns into a paint layer that keeps its other channels as they rendered).
//! - A new group becomes a folder. A new pixel layer becomes a paint layer of the same name in
//!   the channel its name says (`Base Color`, `rough`, `Albedo`, ...); when the name says none,
//!   it comes in with no channel on, and `layer.assignChannel` gives it one.
//! - Layers removed in PhotoCraft are removed; layers the send left out stay where they were.
//!
//! The whole update is one undo step.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Value, json};
use texturecraft_core::{BlendMode, Channel, ChannelBlend, ChannelKind, Effect, EffectKind, Layer, LayerId, LayerKind, Mask, MaskBackground};
use texturecraft_formats::psd::{self, ReadDocument, ReadKind, ReadLayer, ReadMask};
use texturecraft_texel::{TileImage, from_u16, from_u16_signed, to_u16, to_u16_signed};

use crate::commands::{CommandSpec, always, bad};
use crate::photocraft_psd::{Role, Sent, SentMap, decode, mode_of, record};
use crate::{EngineError, Result, Session, cmd};

pub fn specs() -> Vec<CommandSpec> {
    vec![cmd!(
        "layer.assignChannel",
        "Assign layer to channel",
        [],
        None,
        "{layer?, channel} — a paint layer's painting moves to `channel` (converted when it was another channel's) and that becomes its only channel; for layers brought in with no channel on",
        always,
        assign_channel
    )]
}

/// The channel of `set` a layer's name stands for: an id or label (any case, spacing, a
/// trailing number or "copy"), or a common name for it in another word of the name. None when
/// no channel or more than one matches.
pub fn channel_from_name(name: &str, set: &[Channel]) -> Option<Channel> {
    let in_set = |c: Channel| set.contains(&c).then_some(c);
    if let Some(c) = Channel::parse(name.trim()).and_then(in_set) {
        return Some(c);
    }
    let words: Vec<String> = name
        .split(|ch: char| !ch.is_alphanumeric())
        .map(str::to_lowercase)
        .filter(|w| !w.is_empty() && w != "copy" && !w.chars().all(|ch| ch.is_ascii_digit()))
        .collect();
    if let Some(c) = Channel::parse(&words.concat()).and_then(in_set) {
        return Some(c);
    }
    let alias = |w: &str| -> Option<Channel> {
        Some(match w {
            "basecolor" | "basecolour" | "base" | "albedo" | "diffuse" | "color" | "colour" | "col" => Channel::BaseColor,
            "metal" | "metallic" | "metalness" | "metallicity" => Channel::Metallic,
            "rough" | "roughness" | "rgh" => Channel::Roughness,
            "normal" | "normals" | "nrm" | "nor" => Channel::Normal,
            "height" | "bump" | "displacement" | "disp" | "depth" => Channel::Height,
            "emissive" | "emission" | "emit" | "glow" => Channel::Emissive,
            "opacity" | "alpha" | "transparency" => Channel::Opacity,
            "ao" | "occlusion" | "ambientocclusion" => Channel::AmbientOcclusion,
            "gloss" | "glossiness" | "smoothness" => Channel::Glossiness,
            "specular" | "spec" => Channel::SpecularLevel,
            _ => return None,
        })
    };
    let mut found: BTreeSet<Channel> = BTreeSet::new();
    for (i, w) in words.iter().enumerate() {
        let pair = words.get(i + 1).map(|n| format!("{w}{n}"));
        if let Some(c) = pair.as_deref().and_then(|p| Channel::parse(p).or_else(|| alias(p))).and_then(in_set) {
            found.insert(c);
        } else if let Some(c) = Channel::parse(w).and_then(in_set).or_else(|| alias(w).and_then(in_set)) {
            found.insert(c);
        }
    }
    if found.len() == 1 { found.into_iter().next() } else { None }
}

/// Where a layer with no channel keeps its painting until it gets one: the set's first colour
/// channel, else its first channel.
fn holding_channel(set: &[Channel]) -> Option<Channel> {
    set.iter().copied().find(|c| c.kind() == ChannelKind::Color).or_else(|| set.first().copied())
}

/// Mirror `doc`, saved from a document sent as `sent`, into texture set `ts`. Returns the new
/// record (for the next save) and notes on what changed in a way the user may not expect.
pub(crate) fn apply(s: &mut Session, ts: usize, doc: &ReadDocument, sent: &SentMap) -> Result<(SentMap, Vec<String>)> {
    let size = s.caches.get(ts).map(|c| c.size).ok_or_else(|| EngineError::Other(format!("no texture set {ts}")))?;
    if (doc.width, doc.height) != size {
        return Err(EngineError::Other(format!(
            "the document is {}×{} but the texture set is {}×{}; resize it back to bring it in",
            doc.width, doc.height, size.0, size.1
        )));
    }
    let channels = s.doc.project.texture_set(ts).map_err(|e| EngineError::Other(e.to_string()))?.channel_list();
    let sent_layers: BTreeSet<LayerId> = sent.values().filter(|r| !matches!(r.role, Role::Channel(..))).map(|r| r.role.layer()).collect();
    let mut a = Apply { s, ts, size, channels, sent, sent_layers, next: SentMap::new(), lists: Vec::new(), placed: BTreeSet::new(), notes: Vec::new() };
    let top = a.list(&doc.layers, None, 1)?;
    a.lists.push((None, top));
    a.finish()?;
    Ok((a.next, a.notes))
}

struct Apply<'a> {
    s: &'a mut Session,
    ts: usize,
    size: (u32, u32),
    channels: Vec<Channel>,
    sent: &'a SentMap,
    /// The layers (not channels) the send stands for.
    sent_layers: BTreeSet<LayerId>,
    next: SentMap,
    /// Each sibling list the document gives (None: the stack), bottom → top.
    lists: Vec<(Option<LayerId>, Vec<LayerId>)>,
    placed: BTreeSet<LayerId>,
    notes: Vec<String>,
}

fn other(e: impl std::fmt::Display) -> EngineError {
    EngineError::Other(e.to_string())
}

impl Apply<'_> {
    fn note(&mut self, msg: String) {
        if !self.notes.contains(&msg) {
            self.notes.push(msg);
        }
    }

    fn sent_of(&self, node: &ReadLayer) -> Option<&Sent> {
        node.id.and_then(|id| self.sent.get(&id))
    }

    /// The layer `node` was sent as, when it still is one of this set of the right kind.
    fn role_of(&self, node: &ReadLayer) -> Option<Role> {
        let role = self.sent_of(node)?.role;
        let layer = self.s.doc.project.layer(role.layer()).ok()?;
        let here = self.s.doc.project.locate(role.layer()).is_some_and(|(t, _, _)| t == self.ts);
        let fits = match (role, &node.kind) {
            (Role::Folder(_), ReadKind::Group(_)) => layer.is_group(),
            (Role::Layer(_), ReadKind::Group(_)) | (Role::Pixels(_), ReadKind::Pixels { .. }) => !layer.is_group(),
            _ => false,
        };
        (here && fits && !self.placed.contains(&role.layer())).then_some(role)
    }

    fn keep(&mut self, node: &ReadLayer, role: Role) {
        if let Some(id) = node.id {
            self.next.insert(id, record(node, role));
        }
    }

    fn layer_mut(&mut self, id: LayerId) -> Result<&mut Layer> {
        self.s.doc.project.layer_mut(id).map_err(other)
    }

    /// `nodes` (top-first) as layers of one sibling list, bottom → top. `inherit` is a mask the
    /// new layers among them take on (a layer group's mask, for what was added inside it).
    fn list(&mut self, nodes: &[ReadLayer], inherit: Option<&[u16]>, depth: usize) -> Result<Vec<LayerId>> {
        if depth > psd::MAX_DEPTH {
            return Err(EngineError::Other(format!("groups are nested more than {} deep", psd::MAX_DEPTH)));
        }
        let mut out = Vec::new();
        for node in nodes.iter().rev() {
            match (&node.kind, self.role_of(node)) {
                (ReadKind::Group(children), Some(Role::Folder(f))) => {
                    self.placed.insert(f);
                    self.common(f, node)?;
                    self.folder_blend(f, node)?;
                    self.mask(f, node)?;
                    let kids = self.list(children, None, depth + 1)?;
                    self.lists.push((Some(f), kids));
                    self.keep(node, Role::Folder(f));
                    out.push(f);
                }
                (ReadKind::Group(children), Some(Role::Layer(l))) => out.extend(self.layer_group(l, node, children, depth)?),
                (ReadKind::Group(children), _) => {
                    let id = self.s.doc.project.alloc_id();
                    let mut layer = Layer::new(id, node.name.clone(), LayerKind::Group { children: Vec::new(), collapsed: false }, &self.channels);
                    layer.visible = node.visible;
                    for b in layer.channels.values_mut() {
                        *b = ChannelBlend { mode: BlendMode::Passthrough, opacity: node.opacity };
                    }
                    self.isolated_note(node);
                    self.s.doc.project.layers.insert(id, layer);
                    self.placed.insert(id);
                    let grid = self.grid(node.mask.as_ref(), inherit);
                    self.set_mask(id, grid)?;
                    let kids = self.list(children, None, depth + 1)?;
                    self.lists.push((Some(id), kids));
                    self.keep(node, Role::Folder(id));
                    out.push(id);
                }
                (ReadKind::Pixels { .. }, Some(Role::Pixels(l))) => {
                    self.placed.insert(l);
                    self.pixels_layer(l, node, inherit)?;
                    out.push(l);
                }
                (ReadKind::Pixels { .. }, _) => out.push(self.new_pixels(node, inherit)?),
            }
        }
        Ok(out)
    }

    /// Name and visibility, when PhotoCraft changed them.
    fn common(&mut self, id: LayerId, node: &ReadLayer) -> Result<()> {
        let Some(rec) = self.sent_of(node).cloned() else { return Ok(()) };
        let layer = self.layer_mut(id)?;
        if node.name != rec.name && !node.name.trim().is_empty() {
            layer.name = node.name.clone();
        }
        if node.visible != rec.visible {
            layer.visible = node.visible;
        }
        Ok(())
    }

    fn changed_blend(&self, node: &ReadLayer) -> (bool, bool) {
        let rec = self.sent_of(node);
        let opacity = (node.opacity.clamp(0.0, 1.0) * 255.0).round() as u8;
        (rec.is_some_and(|r| r.blend != node.blend), rec.is_some_and(|r| r.opacity != opacity))
    }

    /// A folder's opacity, when PhotoCraft changed it. Folders always pass through in
    /// TextureCraft, so another blend mode on the group only gets a note.
    fn folder_blend(&mut self, f: LayerId, node: &ReadLayer) -> Result<()> {
        let (blend, opacity) = self.changed_blend(node);
        if blend {
            self.isolated_note(node);
        }
        if opacity {
            for b in self.layer_mut(f)?.channels.values_mut().filter(|b| b.mode != BlendMode::Disable) {
                b.opacity = node.opacity;
            }
        }
        Ok(())
    }

    fn isolated_note(&mut self, node: &ReadLayer) {
        if &node.blend != b"pass" {
            self.note(format!("\"{}\": folders in TextureCraft pass through, so the group's blend mode is not kept", node.name));
        }
    }

    /// The mask of `node`, when PhotoCraft changed it.
    fn mask(&mut self, id: LayerId, node: &ReadLayer) -> Result<()> {
        if self.sent_of(node).is_some_and(|r| r.mask == record(node, Role::Folder(id)).mask) {
            return Ok(());
        }
        let had_effects =
            self.s.doc.project.layer(id).ok().and_then(|l| l.mask.as_ref()).is_some_and(|m| m.effects.iter().any(|e| !matches!(e.kind, EffectKind::Paint)));
        if had_effects {
            let name = self.s.doc.project.layer(id).map(|l| l.name.clone()).unwrap_or_default();
            self.note(format!("\"{name}\": its mask was edited in PhotoCraft, so it is now the painted result (its generators and filters are gone)"));
        }
        let grid = self.grid(node.mask.as_ref(), None);
        self.set_mask(id, grid)
    }

    /// `m` times `inherit` over the canvas (0..65535), or None when neither hides anything.
    fn grid(&self, m: Option<&ReadMask>, inherit: Option<&[u16]>) -> Option<Vec<u16>> {
        if m.is_none() && inherit.is_none() {
            return None;
        }
        let (w, h) = self.size;
        let mut out = Vec::with_capacity(w as usize * h as usize);
        for i in 0..w as usize * h as usize {
            let v = m.map_or(1.0, |m| from_u16(m.at((i % w as usize) as i32, (i / w as usize) as i32)));
            let k = inherit.and_then(|g| g.get(i)).map_or(1.0, |g| from_u16(*g));
            out.push(to_u16(v * k));
        }
        Some(out)
    }

    /// Replace `id`'s mask by the painted `grid` (None: no mask).
    fn set_mask(&mut self, id: LayerId, grid: Option<Vec<u16>>) -> Result<()> {
        let Some(grid) = grid else {
            self.layer_mut(id)?.mask = None;
            return Ok(());
        };
        let effect = self.s.doc.project.alloc_id();
        let (w, _) = self.size;
        let img = self.s.doc.pixels.mask_mut(effect, self.size);
        for (i, v) in grid.iter().enumerate() {
            if *v > 0 {
                img.set((i % w as usize) as u32, (i / w as usize) as u32, &[*v, 65535]);
            }
        }
        let mut mask = Mask::new(MaskBackground::Black);
        mask.effects.push(Effect::new(effect, EffectKind::Paint));
        self.layer_mut(id)?.mask = Some(mask);
        Ok(())
    }

    /// A layer sent as a group of channel layers, and whatever was added inside that group.
    fn layer_group(&mut self, l: LayerId, node: &ReadLayer, children: &[ReadLayer], depth: usize) -> Result<Vec<LayerId>> {
        self.placed.insert(l);
        self.common(l, node)?;
        // The group itself is pass-through at 100%: a change to its opacity scales the channels.
        let (_, opacity) = self.changed_blend(node);
        if opacity {
            let was = self.sent_of(node).map_or(255, |r| r.opacity).max(1);
            let k = node.opacity * 255.0 / f32::from(was);
            for b in self.layer_mut(l)?.channels.values_mut() {
                b.opacity = (b.opacity * k).clamp(0.0, 1.0);
            }
        }
        self.mask(l, node)?;
        let mut seen = BTreeSet::new();
        let mut foreign = Vec::new();
        for child in children {
            match (self.sent_of(child).map(|r| r.role), &child.kind) {
                (Some(Role::Channel(owner, c)), ReadKind::Pixels { .. }) if owner == l && !seen.contains(&c) => {
                    seen.insert(c);
                    self.channel(l, c, child)?;
                    self.keep(child, Role::Channel(l, c));
                }
                _ => foreign.push(child.clone()),
            }
        }
        let removed: Vec<Channel> = self
            .sent
            .values()
            .filter_map(|r| match r.role {
                Role::Channel(owner, c) if owner == l && !seen.contains(&c) => Some(c),
                _ => None,
            })
            .collect();
        for c in removed {
            if let Some(b) = self.layer_mut(l)?.channels.get_mut(&c) {
                b.mode = BlendMode::Disable;
            }
        }
        self.keep(node, Role::Layer(l));
        let mut out = vec![l];
        if !foreign.is_empty() {
            // A TextureCraft layer holds no layers: what was added inside its group goes above it,
            // shaped by the group's mask.
            let name = self.s.doc.project.layer(l).map(|x| x.name.clone()).unwrap_or_default();
            self.note(format!("layers added inside \"{name}\" are placed above it, with its mask"));
            let grid = self.grid(node.mask.as_ref(), None);
            out.extend(self.list(&foreign, grid.as_deref(), depth + 1)?);
        }
        Ok(out)
    }

    /// Channel `c` of layer `l` from its pixel layer `node`.
    fn channel(&mut self, l: LayerId, c: Channel, node: &ReadLayer) -> Result<()> {
        let Some(rec) = self.sent_of(node).cloned() else { return Ok(()) };
        if rec.pixels != record(node, rec.role).pixels {
            self.make_paint(l)?;
            self.write(l, c, node, None)?;
        }
        let (blend, opacity) = self.changed_blend(node);
        if let Some(b) = self.layer_mut(l)?.channels.get_mut(&c) {
            if blend {
                b.mode = mode_of(&node.blend);
            }
            if opacity {
                b.opacity = node.opacity;
            }
        }
        Ok(())
    }

    /// Turn layer `l` into a paint layer holding its content as it renders now (masks stay).
    fn make_paint(&mut self, l: LayerId) -> Result<()> {
        if matches!(self.s.doc.project.layer(l).map_err(other)?.kind, LayerKind::Paint) {
            return Ok(());
        }
        let (content, _) = self.s.flatten_apart(self.ts, l).unwrap_or_default();
        for c in self.channels.clone() {
            self.s.doc.pixels.layers.remove(&(l, c));
        }
        for (c, (tx, ty), data) in content {
            self.s.doc.pixels.layer_mut(l, c, self.size).set_tile(tx, ty, Some(std::sync::Arc::new(data)));
        }
        let layer = self.layer_mut(l)?;
        layer.kind = LayerKind::Paint;
        layer.content_effects.clear();
        let name = layer.name.clone();
        self.note(format!("\"{name}\" was painted on in PhotoCraft, so it is now a paint layer"));
        Ok(())
    }

    /// Replace channel `c` of paint layer `l` by `node`'s pixels, times its mask and `inherit`.
    fn write(&mut self, l: LayerId, c: Channel, node: &ReadLayer, inherit: Option<&[u16]>) -> Result<()> {
        let ReadKind::Pixels { x, y, image } = &node.kind else { return Ok(()) };
        let (w, h) = self.size;
        let comps = c.components();
        let mut img = TileImage::new(w, h, comps + 1);
        let iw = image.width as usize;
        for (i, px) in image.data.as_chunks::<4>().0.iter().enumerate() {
            let (cx, cy) = (i64::from(*x) + (i % iw.max(1)) as i64, i64::from(*y) + (i / iw.max(1)) as i64);
            if px[3] == 0 || cx < 0 || cy < 0 || cx >= i64::from(w) || cy >= i64::from(h) {
                continue;
            }
            let (cx, cy) = (cx as u32, cy as u32);
            let mask = node.mask.as_ref().map_or(1.0, |m| from_u16(m.at(cx as i32, cy as i32)));
            let k = inherit.and_then(|g| g.get(cy as usize * w as usize + cx as usize)).map_or(1.0, |g| from_u16(*g));
            let a = (from_u16(px[3]) * mask * k).clamp(0.0, 1.0);
            if a <= 0.0 {
                continue;
            }
            let v = decode(c, px);
            let mut texel = [0u16; 4];
            for (slot, v) in texel.iter_mut().zip(v.iter()).take(comps) {
                *slot = if c.kind() == ChannelKind::Signed { to_u16_signed(v * a) } else { to_u16(v * a) };
            }
            if let Some(slot) = texel.get_mut(comps) {
                *slot = to_u16(a);
            }
            img.set(cx, cy, texel.get(..=comps).unwrap_or(&[]));
        }
        self.s.doc.pixels.layers.insert((l, c), img);
        Ok(())
    }

    /// A new pixel layer as a paint layer in the channel its name says.
    fn new_pixels(&mut self, node: &ReadLayer, inherit: Option<&[u16]>) -> Result<LayerId> {
        let named = channel_from_name(&node.name, &self.channels);
        let Some(c) = named.or_else(|| holding_channel(&self.channels)) else {
            return Err(EngineError::Other("the texture set has no channels".into()));
        };
        let id = self.s.doc.project.alloc_id();
        let mut layer = Layer::new(id, node.name.clone(), LayerKind::Paint, &[c]);
        layer.visible = node.visible;
        let mode = if named.is_some() { mode_of(&node.blend) } else { BlendMode::Disable };
        layer.channels.insert(c, ChannelBlend { mode, opacity: node.opacity });
        self.s.doc.project.layers.insert(id, layer);
        self.placed.insert(id);
        self.write(id, c, node, inherit)?;
        if named.is_none() {
            self.note(format!("\"{}\": no channel goes by that name, so it came in with no channel on; choose one on the layer", node.name));
        }
        self.keep(node, Role::Pixels(id));
        Ok(id)
    }

    /// A layer brought in earlier from a pixel layer, from that layer as saved now.
    fn pixels_layer(&mut self, l: LayerId, node: &ReadLayer, inherit: Option<&[u16]>) -> Result<()> {
        let Some(rec) = self.sent_of(node).cloned() else { return Ok(()) };
        self.common(l, node)?;
        let layer = self.s.doc.project.layer(l).map_err(other)?;
        let current = layer.channels.iter().find(|(_, b)| b.mode != BlendMode::Disable).or_else(|| layer.channels.iter().next()).map(|(c, b)| (*c, *b));
        let renamed = node.name != rec.name;
        let named = renamed.then(|| channel_from_name(&node.name, &self.channels)).flatten();
        let target = named.or(current.map(|(c, _)| c)).or_else(|| holding_channel(&self.channels));
        let Some(target) = target else { return Ok(()) };
        let (blend, opacity) = self.changed_blend(node);
        let moved = current.map(|(c, _)| c) != Some(target);
        if moved || rec.pixels != record(node, rec.role).pixels {
            for c in self.channels.clone() {
                self.s.doc.pixels.layers.remove(&(l, c));
            }
            self.write(l, target, node, inherit)?;
        }
        let mut b = current.map_or_else(ChannelBlend::default, |(_, b)| b);
        if named.is_some() && b.mode == BlendMode::Disable {
            b.mode = mode_of(&node.blend);
        }
        if blend && b.mode != BlendMode::Disable {
            b.mode = mode_of(&node.blend);
        }
        if opacity {
            b.opacity = node.opacity;
        }
        let layer = self.layer_mut(l)?;
        layer.channels.clear();
        layer.channels.insert(target, b);
        self.keep(node, Role::Pixels(l));
        Ok(())
    }

    /// Put every sibling list in place, keep the layers the send left out next to the layer they
    /// were above, and remove what PhotoCraft removed.
    fn finish(&mut self) -> Result<()> {
        // First what PhotoCraft removed, while it is still in the stack; what moved out of a
        // removed folder is kept.
        let gone: Vec<LayerId> = self.sent_layers.iter().copied().filter(|id| !self.placed.contains(id)).collect();
        for &id in &gone {
            let placed = self.placed.clone();
            if let Ok(Layer { kind: LayerKind::Group { children, .. }, .. }) = self.layer_mut(id) {
                children.retain(|c| !placed.contains(c));
            }
        }
        for id in gone {
            if self.s.doc.project.locate(id).is_some_and(|(t, _, _)| t == self.ts) {
                let name = self.s.doc.project.layer(id).map(|l| l.name.clone()).unwrap_or_default();
                self.s.doc.project.remove_layer(id).map_err(other)?;
                self.note(format!("\"{name}\" was removed in PhotoCraft"));
            }
        }
        let lists = std::mem::take(&mut self.lists);
        let visited: BTreeSet<Option<LayerId>> = lists.iter().map(|(p, _)| *p).collect();
        // Groups the document doesn't list give up what it placed elsewhere.
        let groups: Vec<LayerId> = self.s.doc.project.walk(self.ts).into_iter().map(|(id, _)| id).filter(|id| !visited.contains(&Some(*id))).collect();
        for g in groups {
            let placed = self.placed.clone();
            if let Ok(Layer { kind: LayerKind::Group { children, .. }, .. }) = self.layer_mut(g) {
                children.retain(|c| !placed.contains(c));
            }
        }
        for (parent, desired) in &lists {
            let old: Vec<LayerId> = match parent {
                None => self.s.doc.project.texture_set(self.ts).map_err(other)?.stack.clone(),
                Some(f) => self.s.doc.project.layer(*f).map(|l| l.children().to_vec()).unwrap_or_default(),
            };
            let merged = self.merge(&old, desired);
            match parent {
                None => self.s.doc.project.texture_set_mut(self.ts).map_err(other)?.stack = merged,
                Some(f) => {
                    if let LayerKind::Group { children, .. } = &mut self.layer_mut(*f)?.kind {
                        *children = merged;
                    }
                }
            }
        }
        crate::instance_cmds::prune_pixels(&mut self.s.doc);
        Ok(())
    }

    /// `desired`, with the layers of `old` the send didn't stand for kept above the layer they
    /// followed.
    fn merge(&self, old: &[LayerId], desired: &[LayerId]) -> Vec<LayerId> {
        let mut after: BTreeMap<Option<LayerId>, Vec<LayerId>> = BTreeMap::new();
        let mut anchor = None;
        for id in old {
            if desired.contains(id) {
                anchor = Some(*id);
            } else if !self.sent_layers.contains(id) && !self.placed.contains(id) {
                after.entry(anchor).or_default().push(*id);
            }
        }
        let mut out = after.remove(&None).unwrap_or_default();
        for id in desired {
            out.push(*id);
            out.extend(after.remove(&Some(*id)).unwrap_or_default());
        }
        // Anchors that moved to another list: keep their followers at the top.
        for rest in after.into_values() {
            out.extend(rest);
        }
        out
    }
}

/// A stored paint texel of channel `from` (premultiplied, coverage last) as channel `to`.
fn convert(from: Channel, to: Channel, px: &[u16]) -> Option<Vec<u16>> {
    let (fc, tc) = (from.components(), to.components());
    let cov = from_u16(*px.get(fc)?);
    if cov <= 0.0 {
        return None;
    }
    let stored = |k: usize| {
        px.get(k.min(fc.saturating_sub(1))).map_or(0.0, |v| if from.kind() == ChannelKind::Signed { from_u16_signed(*v) } else { from_u16(*v) } / cov)
    };
    let v = [stored(0), stored(1), stored(2)];
    // Through the hand-off encoding, so a gray colour reads as the same gray value.
    let hand = crate::photocraft_cmds::encode_value(from, v);
    let out = decode(to, &hand);
    let mut texel: Vec<u16> = out.iter().take(tc).map(|v| if to.kind() == ChannelKind::Signed { to_u16_signed(v * cov) } else { to_u16(v * cov) }).collect();
    texel.push(to_u16(cov));
    Some(texel)
}

fn assign_channel(s: &mut Session, p: &Value) -> Result<Value> {
    const ID: &str = "layer.assignChannel";
    let id = match p.get("layer") {
        None | Some(Value::Null) => s.selected_layer().ok_or_else(|| bad(ID, "no layer is selected"))?,
        Some(v) => LayerId(v.as_u64().ok_or_else(|| bad(ID, "`layer` must be a layer id"))?),
    };
    let to = p.get("channel").and_then(Value::as_str).and_then(Channel::parse).ok_or_else(|| bad(ID, "`channel` must be a channel id"))?;
    let (ts, _, _) = s.doc.project.locate(id).ok_or_else(|| bad(ID, format!("no layer {}", id.0)))?;
    if !s.doc.project.texture_set(ts).map_err(other)?.channel_list().contains(&to) {
        return Err(bad(ID, format!("the texture set has no {} channel", to.label())));
    }
    let layer = s.doc.project.layer(id).map_err(other)?;
    if !matches!(layer.kind, LayerKind::Paint) {
        return Err(bad(ID, format!("\"{}\" is not a paint layer", layer.name)));
    }
    let has = |c: &Channel| s.doc.pixels.layer(id, *c).is_some_and(|img| img.allocated_count() > 0);
    let from = layer
        .channels
        .iter()
        .filter(|(c, b)| b.mode != BlendMode::Disable && has(c))
        .map(|(c, _)| *c)
        .next()
        .or_else(|| layer.channels.keys().copied().find(|c| has(c)))
        .or_else(|| Channel::ALL.into_iter().find(|c| has(c)));
    let blend = layer.channels.values().find(|b| b.mode != BlendMode::Disable).copied().unwrap_or_default();
    s.edit(ID, |s| {
        if let Some(from) = from.filter(|f| *f != to)
            && let Some(src) = s.doc.pixels.layer(id, from).cloned()
        {
            let mut img = TileImage::new(src.width(), src.height(), to.components() + 1);
            let tiles: Vec<(u32, u32)> = src.allocated().collect();
            let stride = from.components() + 1;
            for (tx, ty) in tiles {
                let Some(data) = src.tile(tx, ty) else { continue };
                for (o, px) in data.chunks(stride).enumerate() {
                    let (x, y) = (
                        tx * texturecraft_texel::TILE + o as u32 % texturecraft_texel::TILE,
                        ty * texturecraft_texel::TILE + o as u32 / texturecraft_texel::TILE,
                    );
                    if x < img.width()
                        && y < img.height()
                        && let Some(texel) = convert(from, to, px)
                    {
                        img.set(x, y, &texel);
                    }
                }
            }
            s.doc.pixels.layers.insert((id, to), img);
        }
        for c in Channel::ALL.into_iter().filter(|c| *c != to) {
            s.doc.pixels.layers.remove(&(id, c));
        }
        let layer = s.doc.project.layer_mut(id).map_err(other)?;
        layer.channels.clear();
        layer.channels.insert(to, blend);
        Ok(())
    })?;
    Ok(json!({ "layer": id.0, "channel": to.id() }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_say_their_channel() {
        let set = [Channel::BaseColor, Channel::Metallic, Channel::Roughness, Channel::Normal, Channel::Height];
        for (name, want) in [
            ("Base color", Some(Channel::BaseColor)),
            ("Base Color", Some(Channel::BaseColor)),
            ("BASECOLOR", Some(Channel::BaseColor)),
            ("base_color 2", Some(Channel::BaseColor)),
            ("Albedo", Some(Channel::BaseColor)),
            ("Rust roughness", Some(Channel::Roughness)),
            ("rough copy", Some(Channel::Roughness)),
            ("Height", Some(Channel::Height)),
            ("Bump", Some(Channel::Height)),
            ("metalness", Some(Channel::Metallic)),
            ("Normal map", Some(Channel::Normal)),
            ("Lorem Ipsum", None),
            ("Layer 2", None),
            ("Roughness and height", None),
            ("Emissive", None),
        ] {
            assert_eq!(channel_from_name(name, &set), want, "{name}");
        }
    }
}
