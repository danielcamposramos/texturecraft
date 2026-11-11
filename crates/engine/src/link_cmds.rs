//! MaterialCraft materials as fill layers. A `.mcmat` portable material becomes a fill whose
//! channel sources are its cooked maps (image resources) and whose [`MaterialLink`] keeps the
//! graph's exposed parameters. Changing one re-cooks the maps through `materialcraft-cli` at the
//! texture set's resolution, with the set's baked mesh maps feeding graph inputs of the same name.
//! The apps share the file format and the command line, not code.

use std::sync::Arc;

#[cfg(not(target_arch = "wasm32"))]
mod cook_process;

use serde_json::{Map, Value, json};
use texturecraft_core::{BakedMap, Channel, Fill, Layer, LayerId, LayerKind, LinkParam, MaterialLink, Source};
#[cfg(not(target_arch = "wasm32"))]
use texturecraft_formats::mcmat::MCMAT_EXT;
use texturecraft_formats::mcmat::{MCRAFT_EXT, PortableMaterial, is_mcraft, read_mcmat};

use crate::commands::{CommandSpec, bad, boolp, has_layer, layer_param, strp};
use crate::materials::Place;
use crate::{EngineError, Result, Session, cmd};

/// The undo label of parameter edits and re-cooks (a re-cook merges into the edit before it).
const EDIT: &str = "Edit MaterialCraft material";

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!(
            "material.link.set",
            "Set material parameter",
            [],
            None,
            "{layer?, id, value (number, [..], bool, \"#rrggbb\" or {t, v}), cook?: true} — sets an exposed input of a MaterialCraft material layer and re-cooks its maps (cook: false only stores the value)",
            has_layer,
            set
        ),
        cmd!(
            "material.link.preset",
            "Apply material preset",
            [],
            None,
            "{layer?, name, cook?: true} — applies a preset of a MaterialCraft material layer",
            has_layer,
            preset
        ),
        cmd!(
            "material.link.cook",
            "Re-cook MaterialCraft material",
            [],
            None,
            "{layer?} — re-cooks a MaterialCraft material layer at its texture set's resolution with the set's baked mesh maps",
            has_layer,
            |s, p| {
                let id = layer_param(s, "material.link.cook", p)?;
                cook(s, id, Map::new())
            }
        ),
        cmd!(
            "material.link.saveSource",
            "Save MaterialCraft package",
            [],
            None,
            "{layer?, path} — writes the MaterialCraft package (.mcraft) behind a material layer, to open in MaterialCraft",
            has_layer,
            save_source
        ),
    ]
}

/// The link of fill layer `id`.
fn link_of(s: &Session, cmd: &str, id: LayerId) -> Result<MaterialLink> {
    match &s.doc.project.layer(id)?.kind {
        LayerKind::Fill(Fill { link: Some(link), .. }) => Ok(link.as_ref().clone()),
        _ => Err(bad(cmd, format!("layer {} is not a MaterialCraft material", id.0))),
    }
}

/// Value types other apps may set (MaterialCraft's portable inputs), with their component count.
fn components(t: &str) -> Option<usize> {
    Some(match t {
        "float" | "int" | "enum" | "bool" => 1,
        "float2" | "int2" => 2,
        "float3" => 3,
        "float4" | "color" => 4,
        _ => return None,
    })
}

fn type_of(v: &Value) -> &str {
    v.get("t").and_then(Value::as_str).unwrap_or("")
}

/// `v` (plain or `{t, v}`) as a typed value for `param`, with finite numbers.
pub fn typed(param: &LinkParam, v: &Value) -> Option<Value> {
    let t = type_of(&param.default);
    let n = components(t)?;
    let payload = if v.get("t").is_some() { v.get("v")? } else { v };
    let num = |x: &Value| x.as_f64().filter(|f| f.is_finite());
    let payload = match (t, payload) {
        ("bool", Value::Bool(b)) => json!(b),
        ("bool", x) => json!(num(x)? != 0.0),
        ("color", Value::String(hex)) => {
            let h = hex.trim_start_matches('#');
            let c = |i: usize| Some(f64::from(u8::from_str_radix(h.get(i..i + 2)?, 16).ok()?) / 255.0);
            // Hex colors are sRGB; values are linear.
            let lin = |x: f64| if x <= 0.04045 { x / 12.92 } else { ((x + 0.055) / 1.055).powf(2.4) };
            json!([lin(c(0)?), lin(c(2)?), lin(c(4)?), 1.0])
        }
        (_, x) if n == 1 => {
            let f = num(x)?;
            if matches!(t, "int" | "enum") { json!(f.round() as i64) } else { json!(f) }
        }
        (_, Value::Array(a)) if a.len() >= n.min(3) => {
            let mut out: Vec<f64> = a.iter().take(n).map(num).collect::<Option<_>>()?;
            out.resize(n, 1.0);
            if t == "int2" { json!(out.iter().map(|f| f.round() as i64).collect::<Vec<_>>()) } else { json!(out) }
        }
        _ => return None,
    };
    Some(json!({"t": t, "v": payload}))
}

/// `overrides` checked against the link's parameters.
fn checked(cmd: &str, link: &MaterialLink, overrides: &Map<String, Value>) -> Result<Map<String, Value>> {
    overrides
        .iter()
        .map(|(id, v)| {
            let p = link.params.iter().find(|p| p.id == *id).ok_or_else(|| bad(cmd, format!("`{}` has no parameter `{id}`", link.name)))?;
            Ok((id.clone(), typed(p, v).ok_or_else(|| bad(cmd, format!("`{id}` takes a {}", type_of(&p.default))))?))
        })
        .collect()
}

fn set(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "material.link.set";
    let id = layer_param(s, CMD, p)?;
    let pid = strp(p, "id").ok_or_else(|| bad(CMD, "`id` is required"))?;
    let value = p.get("value").ok_or_else(|| bad(CMD, "`value` is required"))?;
    update(s, CMD, id, Map::from_iter([(pid.to_string(), value.clone())]), boolp(p, "cook").unwrap_or(true))
}

fn preset(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "material.link.preset";
    let id = layer_param(s, CMD, p)?;
    let name = strp(p, "name").ok_or_else(|| bad(CMD, "`name` is required"))?;
    let link = link_of(s, CMD, id)?;
    let values = link.presets.iter().find(|pr| pr.name == name).ok_or_else(|| bad(CMD, format!("`{}` has no preset `{name}`", link.name)))?.values.clone();
    // Presets may name inputs the painter can't set; those stay as cooked.
    let values = values.into_iter().filter(|(k, _)| link.params.iter().any(|p| p.id == *k));
    update(s, CMD, id, values, boolp(p, "cook").unwrap_or(true))
}

/// Set parameter values of linked layer `id`; with `cook`, re-cook now (one undo step).
fn update(s: &mut Session, cmd: &str, id: LayerId, overrides: impl IntoIterator<Item = (String, Value)>, cook_now: bool) -> Result<Value> {
    let link = link_of(s, cmd, id)?;
    let values = checked(cmd, &link, &overrides.into_iter().collect::<Map<_, _>>())?;
    if cook_now {
        return cook(s, id, values);
    }
    s.edit(EDIT, |s| {
        if let LayerKind::Fill(Fill { link: Some(link), .. }) = &mut s.doc.project.layer_mut(id)?.kind {
            for p in &mut link.params {
                if let Some(v) = values.get(&p.id) {
                    p.value = v.clone();
                }
            }
        }
        Ok(())
    })?;
    Ok(json!({"layer": id.0, "cooked": false}))
}

fn cook(s: &mut Session, id: LayerId, overrides: Map<String, Value>) -> Result<Value> {
    let job = s.link_job(id, &overrides)?;
    let bytes = job.run()?;
    s.apply_link_cook(id, &bytes, false)?;
    Ok(json!({"layer": id.0, "cooked": true, "size": job.size}))
}

#[cfg(not(target_arch = "wasm32"))]
fn save_source(s: &mut Session, p: &Value) -> Result<Value> {
    const CMD: &str = "material.link.saveSource";
    let id = layer_param(s, CMD, p)?;
    let path = strp(p, "path").ok_or_else(|| bad(CMD, "`path` is required"))?;
    let path = if path.to_ascii_lowercase().ends_with(&format!(".{MCRAFT_EXT}")) { path.to_string() } else { format!("{path}.{MCRAFT_EXT}") };
    let bytes = s.link_package(id)?;
    texturecraft_formats::write_atomic(std::path::Path::new(&path), &bytes).map_err(|e| bad(CMD, e.to_string()))?;
    Ok(json!({"path": path}))
}

#[cfg(target_arch = "wasm32")]
fn save_source(_: &mut Session, _: &Value) -> Result<Value> {
    Err(EngineError::Other("file paths are not available in the browser".into()))
}

/// A re-cook of a linked layer, to run anywhere (a worker thread) and hand back to
/// [`Session::apply_link_cook`].
#[derive(Clone, Debug)]
pub struct LinkJob {
    pub layer: LayerId,
    pub package: Arc<Vec<u8>>,
    /// Pixels across (the texture set's resolution).
    pub size: u32,
    /// Every parameter's value, by id, in MaterialCraft's typed form.
    pub values: Map<String, Value>,
    /// Graph inputs fed from baked mesh maps: identifier → 16-bit PNG.
    pub inputs: Vec<(String, Vec<u8>)>,
}

/// A MaterialCraft executable: `$<env>`, else next to this program, else `name` on `PATH`.
#[cfg(not(target_arch = "wasm32"))]
pub fn materialcraft_tool(name: &str, env: &str) -> std::path::PathBuf {
    if let Some(p) = std::env::var_os(env).filter(|p| !p.is_empty()) {
        return p.into();
    }
    let exe = format!("{name}{}", std::env::consts::EXE_SUFFIX);
    std::env::current_exe().ok().and_then(|e| Some(e.parent()?.join(&exe))).filter(|p| p.is_file()).unwrap_or_else(|| exe.into())
}

/// A command for a MaterialCraft executable that opens no console window.
#[cfg(not(target_arch = "wasm32"))]
fn quiet(program: &std::path::Path) -> std::process::Command {
    #[allow(unused_mut)]
    let mut c = std::process::Command::new(program);
    #[cfg(windows)]
    std::os::windows::process::CommandExt::creation_flags(&mut c, 0x0800_0000); // CREATE_NO_WINDOW
    c
}

/// Open `package` (a `.mcraft` file) in the MaterialCraft app (`$MATERIALCRAFT`, next to this
/// program, or on `PATH`).
#[cfg(not(target_arch = "wasm32"))]
pub fn open_in_materialcraft(package: &std::path::Path) -> Result<()> {
    let app = materialcraft_tool("materialcraft", "MATERIALCRAFT");
    quiet(&app).arg(package).spawn().map(drop).map_err(|e| {
        EngineError::Other(format!("MaterialCraft ({}) did not start: {e}. Install MaterialCraft next to TextureCraft or set MATERIALCRAFT", app.display()))
    })
}

/// Longest a cook may run (it takes about a second at 2K).
#[cfg(not(target_arch = "wasm32"))]
const COOK_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(300);
/// Largest cooked `.mcmat` read back.
#[cfg(not(target_arch = "wasm32"))]
const MAX_COOKED_BYTES: u64 = 2 << 30;

/// A new, empty, owner-only folder in the temp dir: exclusive creation (fails if the name
/// exists, even as a symlink) under a name generated from operating-system randomness.
#[cfg(not(target_arch = "wasm32"))]
fn private_temp_dir() -> std::io::Result<std::path::PathBuf> {
    let mut last = None;
    for _ in 0..16 {
        let mut random = [0; 16];
        getrandom::fill(&mut random).map_err(|error| std::io::Error::other(error.to_string()))?;
        let directory = std::env::temp_dir().join(format!("texturecraft-link-{:032x}", u128::from_le_bytes(random)));
        let mut builder = std::fs::DirBuilder::new();
        #[cfg(unix)]
        std::os::unix::fs::DirBuilderExt::mode(&mut builder, 0o700);
        match builder.create(&directory) {
            Ok(()) => return Ok(directory),
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => last = Some(error),
            Err(error) => return Err(error),
        }
    }

    Err(last.unwrap_or_else(|| std::io::ErrorKind::AlreadyExists.into()))
}

impl LinkJob {
    /// Cook through `materialcraft-cli` (`$MATERIALCRAFT_CLI`, next to this program, or on
    /// `PATH`); returns `.mcmat` bytes. Blocks for the cook (about a second at 2K).
    #[cfg(not(target_arch = "wasm32"))]
    pub fn run(&self) -> Result<Vec<u8>> {
        let other = |error: String| EngineError::Other(error);
        let directory = private_temp_dir().map_err(|error| other(format!("temporary folder: {error}")))?;
        let result = (|| {
            let utf8 = |path: &std::path::Path| path.to_str().map(str::to_string).ok_or_else(|| other(format!("{} is not a UTF-8 path", path.display())));
            let write = |name: &str, bytes: &[u8]| {
                let path = directory.join(name);
                std::fs::write(&path, bytes).map_err(|error| other(format!("{name}: {error}")))?;
                utf8(&path)
            };

            let source = write("source.mcraft", &self.package)?;
            let mut inputs = Map::new();
            for (index, (id, png)) in self.inputs.iter().enumerate() {
                inputs.insert(id.clone(), json!(write(&format!("input{index}.png"), png)?));
            }

            let output = directory.join(format!("out.{MCMAT_EXT}"));
            let params = json!({"path": utf8(&output)?, "size": self.size, "values": self.values, "inputs": inputs});
            let cli = materialcraft_tool("materialcraft-cli", "MATERIALCRAFT_CLI");
            let mut command = quiet(&cli);
            command.arg("run").arg(&source).arg("file.exportMaterial").arg(params.to_string());
            cook_process::run(command, COOK_TIMEOUT)?;

            texturecraft_formats::read_capped(&output, MAX_COOKED_BYTES).map_err(|error| other(format!("{}: {error}", output.display())))
        })();
        let _ = std::fs::remove_dir_all(&directory);
        result
    }

    #[cfg(target_arch = "wasm32")]
    pub fn run(&self) -> Result<Vec<u8>> {
        Err(EngineError::Other("re-cooking a MaterialCraft material needs its command line, which the browser can't run".into()))
    }
}

/// Decode every map of `m` (validating them) with the channel it feeds.
fn decoded_maps(m: &PortableMaterial) -> Result<Vec<(Channel, Vec<u8>, texturecraft_formats::DecodedImage)>> {
    m.maps
        .iter()
        .filter_map(|(usage, bytes)| Some((Channel::ALL.into_iter().find(|c| c.id() == usage)?, bytes)))
        .map(|(c, bytes)| {
            let img = texturecraft_formats::decode_image(bytes).map_err(|e| EngineError::Other(format!("{} map: {e}", c.id())))?;
            Ok((c, bytes.clone(), img))
        })
        .collect()
}

impl Session {
    /// Add a MaterialCraft material (`.mcmat` bytes, or a `.mcraft` package cooked first) as a
    /// fill layer at `place` (one undo step). Returns the new layer.
    pub fn add_linked_material_bytes_at(&mut self, bytes: &[u8], place: Place) -> Result<LayerId> {
        let cooked;
        let bytes = if is_mcraft(bytes) {
            let ts = self.ts();
            let size = self.doc.project.texture_set(ts)?.resolution;
            let job = LinkJob { layer: LayerId(0), package: Arc::new(bytes.to_vec()), size, values: Map::new(), inputs: Vec::new() };
            cooked = job.run()?;
            cooked.as_slice()
        } else {
            bytes
        };
        let m = read_mcmat(bytes).map_err(|e| EngineError::Other(e.to_string()))?;
        let maps = decoded_maps(&m)?;
        if maps.is_empty() {
            return Err(EngineError::Other(format!("{} has no maps TextureCraft can paint with", m.manifest.name)));
        }
        let label = format!("Add {}", m.manifest.name);
        let (id, names) = self.edit(&label, |s| {
            let ts = s.ts();
            let stem = texturecraft_formats::sanitize_resource_name(&m.manifest.name);
            let package = unique(s, &format!("{stem}.{MCRAFT_EXT}"));
            s.doc.resources.insert(package.clone(), Arc::new(m.source.clone()));
            let link = MaterialLink {
                name: m.manifest.name.clone(),
                package,
                graph: m.manifest.graph.clone(),
                params: m.manifest.parameters.iter().filter(|p| components(type_of(&p.default)).is_some()).cloned().collect(),
                presets: m.manifest.presets.clone(),
                inputs: m.manifest.inputs.iter().map(|i| i.identifier.clone()).collect(),
                size: m.manifest.size[0],
            };
            let mut fill = Fill { link: Some(Box::new(link)), ..Fill::default() };
            let names = put_maps(s, &stem, &maps, &mut fill);
            for c in fill.sources.keys() {
                crate::set_cmds::add_channel(s, ts, *c)?;
            }
            let channels: Vec<Channel> = fill.sources.keys().copied().collect();
            let lid = s.doc.project.alloc_id();
            let name = s.doc.project.next_layer_name(ts, &m.manifest.name);
            let lid = crate::layer_cmds::insert_selected(s, Layer::new(lid, name, LayerKind::Fill(fill), &channels))?;
            place.apply(s, lid)?;
            s.selected.insert(ts, lid);
            s.target = crate::session::StackTarget::Content;
            Ok((lid, names))
        })?;
        self.cache_decoded(names, maps);
        Ok(id)
    }

    /// The `.mcraft` package behind linked layer `id`.
    pub fn link_package(&self, id: LayerId) -> Result<Arc<Vec<u8>>> {
        let link = link_of(self, "material.link", id)?;
        self.doc.resources.get(&link.package).cloned().ok_or_else(|| EngineError::Other(format!("the package of {} is missing", link.name)))
    }

    /// The parameter values of linked layer `id`, by id.
    pub fn link_values(&self, id: LayerId) -> Result<Map<String, Value>> {
        Ok(link_of(self, "material.link", id)?.params.iter().map(|p| (p.id.clone(), p.value.clone())).collect())
    }

    /// Prepare a re-cook with current parameter values and matching baked mesh maps
    pub fn link_job(&self, id: LayerId, overrides: &Map<String, Value>) -> Result<LinkJob> {
        let link = link_of(self, "material.link", id)?;
        let (texture_set, _, _) = self.doc.project.locate(id).ok_or_else(|| EngineError::Other(format!("layer {} is not in a stack", id.0)))?;
        let mut values = self.link_values(id)?;
        values.extend(overrides.clone());

        let mut inputs = Vec::new();
        if let Some(baked) = self.doc.baked.get(&texture_set) {
            for identifier in &link.inputs {
                let Some((map, data)) = BakedMap::parse(identifier).and_then(|map| Some((map, baked.maps.get(&map)?))) else { continue };

                let png = texturecraft_formats::mcmat::mesh_map_png(baked.resolution, map.components(), data)
                    .map_err(|error| EngineError::Other(error.to_string()))?;
                inputs.push((identifier.clone(), png));
            }
        }

        // UDIM tiles share one cook at the texture set's base resolution
        let size = self.doc.project.texture_set(texture_set)?.resolution;
        Ok(LinkJob { layer: id, package: self.link_package(id)?, size, values, inputs })
    }

    /// Swap in maps a [`LinkJob`] cooked (`.mcmat` bytes) and the values they were cooked with.
    /// With `merge`, this folds into the previous undo step when that was an edit of the same kind
    /// (a parameter set while the cook ran).
    pub fn apply_link_cook(&mut self, id: LayerId, bytes: &[u8], merge: bool) -> Result<()> {
        let m = read_mcmat(bytes).map_err(|e| EngineError::Other(e.to_string()))?;
        let maps = decoded_maps(&m)?;
        let names = self.edit_merging(EDIT, merge, |s| {
            let LayerKind::Fill(mut fill) = s.doc.project.layer(id)?.kind.clone() else {
                return Err(EngineError::Other(format!("layer {} is not a fill", id.0)));
            };
            let link = fill.link.as_mut().ok_or_else(|| EngineError::Other(format!("layer {} is not a MaterialCraft material", id.0)))?;
            for p in &mut link.params {
                if let Some(cooked) = m.manifest.parameters.iter().find(|c| c.id == p.id) {
                    p.value = cooked.value.clone();
                }
            }
            link.size = m.manifest.size[0];
            let stem = texturecraft_formats::sanitize_resource_name(&link.name);
            let old: Vec<String> =
                fill.sources.values().filter_map(|src| if let Source::Image { name, .. } = src { Some(name.clone()) } else { None }).collect();
            let names = put_maps(s, &stem, &maps, &mut fill);
            s.doc.project.layer_mut(id)?.kind = LayerKind::Fill(fill);
            // Old maps go unless another layer (a duplicate) still shows them.
            let mut used = Vec::new();
            for set in 0..s.doc.project.texture_sets.len() {
                for root in s.doc.project.texture_sets.get(set).map(|t| t.stack.clone()).unwrap_or_default() {
                    crate::material_lib::images_used(&s.doc.project, root, &mut used, 0);
                }
            }
            for name in old.iter().filter(|n| !used.contains(n)) {
                s.doc.resources.remove(name);
            }
            Ok(names)
        })?;
        self.sync_images();
        self.cache_decoded(names, maps);
        Ok(())
    }

    /// Keep the images decoded while validating maps instead of decoding them again.
    fn cache_decoded(&mut self, names: Vec<String>, maps: Vec<(Channel, Vec<u8>, texturecraft_formats::DecodedImage)>) {
        for (name, (_, _, img)) in names.into_iter().zip(maps) {
            self.images.insert(name, crate::resource_cmds::decoded(img));
        }
    }
}

/// A resource name not in use: `name`, else `2_name`, `3_name`…
fn unique(s: &Session, name: &str) -> String {
    (1..).map(|n| if n == 1 { name.to_string() } else { format!("{n}_{name}") }).find(|n| !s.doc.resources.contains_key(n)).unwrap_or_else(|| name.to_string())
}

/// Store `maps` as new resources (names carry a fresh id, so the decoded-image cache never
/// confuses two cooks) and point `fill`'s channels at them. Returns the names, in map order.
fn put_maps(s: &mut Session, stem: &str, maps: &[(Channel, Vec<u8>, texturecraft_formats::DecodedImage)], fill: &mut Fill) -> Vec<String> {
    let cook = s.doc.project.alloc_id().0;
    let mut names = Vec::new();
    for (c, bytes, _) in maps {
        let name = unique(s, &format!("{stem}_{}_{cook}.png", c.id()));
        s.doc.resources.insert(name.clone(), Arc::new(bytes.clone()));
        // A normal map the user marked DirectX stays marked through re-cooks.
        let normal_format = match fill.sources.get(c) {
            Some(Source::Image { normal_format, .. }) => *normal_format,
            _ => Default::default(),
        };
        fill.sources.insert(*c, Source::Image { name: name.clone(), normal_format });
        names.push(name);
    }
    names
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod temp_dir_tests {
    #[test]
    fn private_temp_dirs_are_fresh_and_owner_only() {
        let first = super::private_temp_dir().unwrap();
        let second = super::private_temp_dir().unwrap();

        assert_ne!(first, second);
        #[cfg(unix)]
        assert_eq!(std::os::unix::fs::PermissionsExt::mode(&std::fs::metadata(&first).unwrap().permissions()) & 0o077, 0);

        std::fs::remove_dir(first).unwrap();
        std::fs::remove_dir(second).unwrap();
    }
}
