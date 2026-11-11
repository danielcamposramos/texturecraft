//! The `.tcraft` project file: a deflate zip holding
//!
//! | entry | content |
//! |---|---|
//! | `project.json` | serde JSON of [`Project`] |
//! | `mesh.<ext>` | the original imported mesh file (absent for built-in meshes, `Project.mesh = "builtin:<id>"`) |
//! | `highpoly/<name>` | a high-definition mesh file for baking, in list order |
//! | `cage/<name>` | the baking cage mesh file |
//! | `bake_paint/<ts>_<target>.bin` | cage or skew paint, stored as a one-component baked map |
//! | `pixels/layer_<id>_<channelId>.bin` | painted pixels of a paint layer (or content paint effect) channel |
//! | `pixels/mask_<id>.bin` | painted pixels of a mask paint effect |
//! | `baked/<ts>_<mapId>.bin` | a baked mesh map of texture set `<ts>` |
//! | `thumbnail.png` | optional preview supplied by the caller |
//!
//! Pixel `.bin` files are sparse and lossless: a header (magic `TCTX`, version, width, height,
//! components, tile count; u32 LE each) followed, for every allocated 64×64 tile, by `tx`, `ty`
//! (u32 LE) and the tile's raw `u16` LE components. Baked maps use the same header with a tile
//! count of 0, followed by `width × height × components` f32 LE. Pixels of a UV-tiled texture
//! set cover its whole atlas (`resolution` per tile, see `TileGrid`).

use std::cell::Cell;
use std::collections::{BTreeMap, HashSet};
use std::io::{Cursor, Read, Seek, Write};
use std::sync::Arc;

use texturecraft_compose::{BakedMaps, PixelStore};
use texturecraft_core::{BakePaintTarget, Channel, FORMAT_VERSION, LayerId, Project};
use texturecraft_texel::{TILE_AREA, TileImage};
use zip::write::SimpleFileOptions;
use zip::{CompressionMethod, ZipArchive, ZipWriter};

use self::entries::Entry;
use self::load_budget::LoadBudget;
use self::metadata_reader::MetadataReader;
use crate::{FormatError, MAX_IMAGE_SIZE, Result, malformed};

mod entries;
mod load_budget;
mod metadata_reader;
mod project_open_limit;
#[cfg(not(target_arch = "wasm32"))]
pub use entries::{DEFAULT_RAM_PERCENT, DEFAULT_RESERVED_CORES, available_save_resources, save_ram_budget, set_save_resources};
pub use project_open_limit::{DEFAULT_PROJECT_OPEN_LIMIT_GIB, project_open_limit_gib, set_project_open_limit_gib};

const KIND: &str = "project file";
const MAGIC: &[u8; 4] = b"TCTX";
/// Version of the `.bin` layout.
const BIN_VERSION: u32 = 1;
const BIN_HEADER_BYTES: usize = 24;
const MAX_JSON: u64 = 256 << 20;
const MAX_THUMBNAIL: u64 = 64 << 20;
const MAX_RESOURCE: u64 = 512 << 20;
const MAX_ENTRIES: usize = 100_000;
/// Bound ZIP index parsing before the archive exposes its entry count.
const MAX_ZIP_METADATA_BYTES: u64 = 32 << 20;
/// Largest image (texels) of a texture set or baked map: UV-tiled sets hold several tiles.
const MAX_TEXELS: u64 = 1 << 28;

mod bake_paint_data;
mod recovery;
pub use bake_paint_data::BakePaintData;

/// Everything a project file holds.
#[derive(Clone, Debug)]
pub struct ProjectFile {
    pub project: Project,
    pub pixels: PixelStore,
    /// Baked mesh maps per texture set index.
    pub baked: BTreeMap<usize, BakedMaps>,
    /// Original mesh file: (extension, bytes).
    pub mesh: Option<(String, Vec<u8>)>,
    /// High-definition mesh files for baking: (file name, bytes). Names are
    /// [`sanitize_resource_name`]d on save.
    pub high_poly: Vec<(String, Vec<u8>)>,
    /// Baking cage mesh file: (file name, bytes).
    pub cage: Option<(String, Vec<u8>)>,
    pub bake_paint: BTreeMap<(usize, BakePaintTarget), BakePaintData>,
    pub thumbnail: Option<Vec<u8>>,
    /// Imported resources (images used by fills, alphas, stencils), by name: original bytes.
    pub resources: BTreeMap<String, Vec<u8>>,
    /// Import warnings for the caller's log; never saved in the archive
    pub warnings: Vec<String>,
}

/// A resource name safe as a zip entry and a file name: `[A-Za-z0-9_.-]`, at most 120 chars.
pub fn sanitize_resource_name(name: &str) -> String {
    let s: String = name.chars().map(|c| if c.is_ascii_alphanumeric() || matches!(c, '_' | '.' | '-') { c } else { '_' }).take(120).collect();
    let s = s.trim_start_matches('.').to_string();
    if s.is_empty() { "resource".into() } else { s }
}

// ---------------------------------------------------------------------------------------------
// Save.

pub fn save_project_bytes(pf: &ProjectFile) -> Result<Vec<u8>> {
    let mut out = Cursor::new(Vec::new());
    write_project(&mut out, pf)?;
    Ok(out.into_inner())
}

/// Save atomically: `<path>.tmp` first, then rename over `path`.
#[cfg(not(target_arch = "wasm32"))]
pub fn save_project(path: &std::path::Path, pf: &ProjectFile) -> Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".tmp");
    let tmp = std::path::PathBuf::from(tmp);
    let res = (|| -> Result<()> {
        let mut file = std::io::BufWriter::new(std::fs::File::create(&tmp)?);
        write_project(&mut file, pf)?;
        file.into_inner().map_err(|e| e.into_error())?.sync_all()?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    })();
    if res.is_err() {
        let _ = std::fs::remove_file(&tmp);
    }
    res
}

fn zip_err(e: zip::result::ZipError) -> FormatError {
    match e {
        zip::result::ZipError::Io(e) => FormatError::Io(e),
        e => malformed(KIND, e),
    }
}

fn header(w: u32, h: u32, comps: usize, count: usize) -> Vec<u8> {
    let mut b = MAGIC.to_vec();
    for v in [BIN_VERSION, w, h, comps as u32, count as u32] {
        b.extend_from_slice(&v.to_le_bytes());
    }
    b
}

fn write_tiles(w: &mut impl Write, img: &TileImage) -> std::io::Result<()> {
    w.write_all(&header(img.width(), img.height(), img.comps(), img.allocated_count()))?;
    let mut buf = Vec::with_capacity(TILE_AREA * img.comps() * 2 + 8);
    for (tx, ty) in img.allocated() {
        buf.clear();
        buf.extend_from_slice(&tx.to_le_bytes());
        buf.extend_from_slice(&ty.to_le_bytes());
        for v in img.tile(tx, ty).unwrap_or_default() {
            buf.extend_from_slice(&v.to_le_bytes());
        }
        w.write_all(&buf)?;
    }
    Ok(())
}

/// Keep `[a-z0-9]`, at most 8 characters (the mesh entry's extension).
fn sanitize_ext(ext: &str) -> String {
    let e: String = ext.trim_start_matches('.').chars().filter(char::is_ascii_alphanumeric).take(8).collect::<String>().to_ascii_lowercase();
    if e.is_empty() { "bin".into() } else { e }
}

/// The sanitized `name`, with a numeric suffix before the extension when `used` already holds it.
fn unique_entry_name(dir: &str, name: &str, used: &mut HashSet<String>) -> String {
    let base = sanitize_resource_name(name);
    let mut entry = format!("{dir}/{base}");
    let mut n = 2u32;
    while !used.insert(entry.clone()) {
        let (stem, ext) = match base.rfind('.') {
            Some(i) if i > 0 => base.split_at(i),
            _ => (base.as_str(), ""),
        };
        entry = format!("{dir}/{stem}_{n}{ext}");
        n += 1;
    }
    entry
}

/// Floats converted to bytes at a time when a baked map streams into the compressor.
const BAKED_CHUNK: usize = 1 << 16;

fn write_baked(w: &mut dyn Write, res: u32, h: u32, comps: usize, data: &[f32]) -> std::io::Result<()> {
    w.write_all(&header(res, h, comps, 0))?;
    let mut buf = Vec::with_capacity(BAKED_CHUNK * 4);
    for chunk in data.chunks(BAKED_CHUNK) {
        buf.clear();
        buf.extend(chunk.iter().flat_map(|v| v.to_le_bytes()));
        w.write_all(&buf)?;
    }
    Ok(())
}

/// The zip entries of a project file, in file order. Bodies run later, possibly on other threads.
fn entries(pf: &ProjectFile) -> Result<Vec<Entry<'_>>> {
    let deflate = SimpleFileOptions::default().compression_method(CompressionMethod::Deflated).large_file(true);
    let stored = SimpleFileOptions::default().compression_method(CompressionMethod::Stored);
    let json = serde_json::to_vec_pretty(&pf.project).map_err(|e| FormatError::Invalid(e.to_string()))?;
    let mut out = vec![Entry { name: "project.json".into(), options: deflate, weight: json.len() as u64, body: Box::new(move |w| w.write_all(&json)) }];
    if let Some((ext, bytes)) = &pf.mesh {
        out.push(Entry {
            name: format!("mesh.{}", sanitize_ext(ext)),
            options: deflate,
            weight: bytes.len() as u64,
            body: Box::new(move |w| w.write_all(bytes)),
        });
    }
    let mut used = HashSet::new();
    for (dir, (name, bytes)) in pf.high_poly.iter().map(|m| ("highpoly", m)).chain(pf.cage.iter().map(|m| ("cage", m))) {
        out.push(Entry {
            name: unique_entry_name(dir, name, &mut used),
            options: deflate,
            weight: bytes.len() as u64,
            body: Box::new(move |w| w.write_all(bytes)),
        });
    }
    for ((texture_set, target), map) in &pf.bake_paint {
        if map.width == 0 || map.height == 0 || map.values.len() as u64 != u64::from(map.width) * u64::from(map.height) {
            return Err(FormatError::Invalid("bake paint map dimensions do not match its values".into()));
        }
        let target_name = match target {
            BakePaintTarget::Cage => "cage",
            BakePaintTarget::Skew => "skew",
        };
        out.push(Entry {
            name: format!("bake_paint/{texture_set}_{target_name}.bin"),
            options: deflate,
            weight: map.values.len() as u64 * 4,
            body: Box::new(move |w| write_baked(w, map.width, map.height, 1, &map.values)),
        });
    }
    let mut layers: Vec<_> = pf.pixels.layers.iter().collect();
    layers.sort_by_key(|(k, _)| **k);
    for ((id, c), img) in layers {
        let weight = (img.allocated_count() * (TILE_AREA * img.comps() * 2 + 8)) as u64;
        out.push(Entry {
            name: format!("pixels/layer_{}_{}.bin", id.0, c.id()),
            options: deflate,
            weight,
            body: Box::new(move |mut w| write_tiles(&mut w, img)),
        });
    }
    let mut masks: Vec<_> = pf.pixels.masks.iter().collect();
    masks.sort_by_key(|(k, _)| **k);
    for (id, img) in masks {
        let weight = (img.allocated_count() * (TILE_AREA * img.comps() * 2 + 8)) as u64;
        out.push(Entry { name: format!("pixels/mask_{}.bin", id.0), options: deflate, weight, body: Box::new(move |mut w| write_tiles(&mut w, img)) });
    }
    for (ts, maps) in &pf.baked {
        let r = maps.resolution;
        for (m, data) in &maps.maps {
            // A map whose size disagrees with its width cannot be loaded; it can be re-baked.
            let h = maps.height(*m);
            if h == 0 || data.len() != (r as usize) * (h as usize) * m.components() {
                continue;
            }
            let comps = m.components();
            out.push(Entry {
                name: format!("baked/{ts}_{}.bin", m.id()),
                options: deflate,
                weight: data.len() as u64 * 4,
                body: Box::new(move |w| write_baked(w, r, h, comps, data)),
            });
        }
    }
    if let Some(png) = &pf.thumbnail {
        out.push(Entry { name: "thumbnail.png".into(), options: stored, weight: png.len() as u64, body: Box::new(move |w| w.write_all(png)) });
    }
    used.clear();
    for (name, bytes) in &pf.resources {
        out.push(Entry {
            name: unique_entry_name("resources", name, &mut used),
            options: stored,
            weight: bytes.len() as u64,
            body: Box::new(move |w| w.write_all(bytes)),
        });
    }
    Ok(out)
}

fn write_project<W: Write + Seek>(out: W, pf: &ProjectFile) -> Result<()> {
    let mut zip = ZipWriter::new(out);
    entries::write_entries(&mut zip, entries(pf)?)?;
    zip.finish().map_err(zip_err)?;
    Ok(())
}

// ---------------------------------------------------------------------------------------------
// Load.

pub fn load_project_bytes(bytes: &[u8]) -> Result<ProjectFile> {
    crate::guard(KIND, || load(Cursor::new(bytes)))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn load_project(path: &std::path::Path) -> Result<ProjectFile> {
    let file = std::fs::File::open(path)?;
    crate::guard(KIND, || load(std::io::BufReader::new(file)))
}

fn read_all(reader: &mut impl Read, cap: u64, budget: &mut LoadBudget, what: &str) -> Result<Vec<u8>> {
    let mut out = Vec::new();

    // Read at most one byte beyond either limit to detect overflow without a full entry allocation
    let read_limit = cap.min(budget.remaining()).saturating_add(1);
    reader.take(read_limit).read_to_end(&mut out).map_err(|e| malformed(KIND, format!("{what}: {e}")))?;
    if out.len() as u64 > cap {
        return Err(FormatError::TooLarge(format!("{what} is larger than {cap} bytes")));
    }

    budget.take(out.len() as u64).map_err(FormatError::TooLarge)?;
    Ok(out)
}

/// Decode bounded chunks so a false header cannot allocate an entire missing map.
fn read_f32s(reader: &mut impl Read, count: usize, what: &str) -> Result<Vec<f32>> {
    let mut values = Vec::new();
    let mut bytes = vec![0u8; BAKED_CHUNK.min(count) * 4];
    let mut remaining = count;
    while remaining > 0 {
        let chunk_values = BAKED_CHUNK.min(remaining);
        let chunk = bytes.get_mut(..chunk_values * 4).ok_or_else(|| malformed(KIND, format!("{what}: invalid float chunk")))?;
        read_exact(reader, chunk, what)?;
        values.try_reserve(chunk_values).map_err(|_| FormatError::TooLarge(format!("{what}: not enough memory for {count} values")))?;
        values.extend(chunk.as_chunks::<4>().0.iter().map(|bytes| f32::from_le_bytes(*bytes)));
        remaining -= chunk_values;
    }

    Ok(values)
}

fn read_exact(r: &mut impl Read, buf: &mut [u8], what: &str) -> Result<()> {
    r.read_exact(buf).map_err(|e| malformed(KIND, format!("{what}: {e}")))
}

fn u32_at(b: &[u8], i: usize) -> u32 {
    b.get(i * 4..i * 4 + 4).map(|s| u32::from_le_bytes([s[0], s[1], s[2], s[3]])).unwrap_or(0)
}

/// Read and check a `.bin` header: returns (width, height, comps, count).
fn read_header(r: &mut impl Read, what: &str) -> Result<(u32, u32, usize, usize)> {
    let mut h = [0u8; BIN_HEADER_BYTES];
    read_exact(r, &mut h, what)?;
    if h.get(..4) != Some(MAGIC.as_slice()) {
        return Err(malformed(KIND, format!("{what}: bad magic")));
    }
    let v = u32_at(&h, 1);
    if v > BIN_VERSION {
        return Err(FormatError::NewerVersion { found: v, supported: BIN_VERSION });
    }
    Ok((u32_at(&h, 2), u32_at(&h, 3), u32_at(&h, 4) as usize, u32_at(&h, 5) as usize))
}

/// Error unless the entry has no bytes left (also makes the zip reader verify the CRC).
fn expect_end(r: &mut impl Read, what: &str) -> Result<()> {
    let mut b = [0u8; 1];
    match r.read(&mut b) {
        Ok(0) => Ok(()),
        Ok(_) => Err(malformed(KIND, format!("{what}: trailing data"))),
        Err(e) => Err(malformed(KIND, format!("{what}: {e}"))),
    }
}

fn read_tiles(reader: &mut impl Read, size: (u32, u32), components: usize, budget: &mut LoadBudget, what: &str) -> Result<TileImage> {
    budget.take(BIN_HEADER_BYTES as u64).map_err(FormatError::TooLarge)?;
    let (width, height, stored_components, count) = read_header(reader, what)?;
    if (width, height) != size || stored_components != components {
        return Err(malformed(
            KIND,
            format!("{what}: {width}×{height}×{stored_components} does not match the texture set ({}×{}×{components})", size.0, size.1),
        ));
    }

    let mut image = TileImage::new(width, height, components);
    if count > (image.tiles_x() as usize) * (image.tiles_y() as usize) {
        return Err(malformed(KIND, format!("{what}: {count} tiles is more than the image holds")));
    }

    let mut bytes = vec![0u8; TILE_AREA * components * 2];
    let mut position_bytes = [0u8; 8];
    for _ in 0..count {
        budget.take((position_bytes.len() + bytes.len()) as u64).map_err(FormatError::TooLarge)?;
        read_exact(reader, &mut position_bytes, what)?;
        let (tx, ty) = (u32_at(&position_bytes, 0), u32_at(&position_bytes, 1));
        if tx >= image.tiles_x() || ty >= image.tiles_y() || image.is_allocated(tx, ty) {
            return Err(malformed(KIND, format!("{what}: bad or repeated tile ({tx}, {ty})")));
        }

        read_exact(reader, &mut bytes, what)?;
        let data: Vec<u16> = bytes.as_chunks::<2>().0.iter().map(|b| u16::from_le_bytes(*b)).collect();
        image.set_tile(tx, ty, Some(Arc::new(data)));
    }

    expect_end(reader, what)?;

    Ok(image)
}

/// A baked map: (width, height, data).
fn read_baked(reader: &mut impl Read, declared_bytes: u64, components: usize, budget: &mut LoadBudget, what: &str) -> Result<(u32, u32, Vec<f32>)> {
    budget.take(BIN_HEADER_BYTES as u64).map_err(FormatError::TooLarge)?;
    let (width, height, stored_components, _) = read_header(reader, what)?;
    if width == 0 || height == 0 || u64::from(width) * u64::from(height) > MAX_TEXELS || stored_components != components {
        return Err(malformed(KIND, format!("{what}: bad size {width}×{height}×{stored_components}")));
    }

    let value_count = u64::from(width) * u64::from(height) * components as u64;
    let payload_bytes = value_count * 4;
    if declared_bytes != BIN_HEADER_BYTES as u64 + payload_bytes {
        return Err(malformed(KIND, format!("{what}: {declared_bytes} bytes for a {width}×{height}×{components} map")));
    }

    budget.take(payload_bytes).map_err(FormatError::TooLarge)?;
    let value_count = usize::try_from(value_count).map_err(|_| FormatError::TooLarge(format!("{what}: too many values for this platform")))?;
    let data = read_f32s(reader, value_count, what)?;
    expect_end(reader, what)?;

    Ok((width, height, data))
}

/// Texture set holding a paint layer, a content effect or a mask effect id.
fn texture_set_of(project: &Project, id: LayerId) -> Option<usize> {
    let owner = if project.layers.contains_key(&id) { id } else { project.effect(id)?.0 };
    project.locate(owner).map(|(ts, _, _)| ts)
}

fn load<R: Read + Seek>(reader: R) -> Result<ProjectFile> {
    let limit_bytes = u64::from(project_open_limit_gib()) << 30;
    load_with_budget(reader, LoadBudget::new(limit_bytes))
}

fn check_archive<R: Read + Seek>(zip: &mut ZipArchive<R>, limit_bytes: u64) -> Result<()> {
    if zip.len() > MAX_ENTRIES {
        return Err(FormatError::TooLarge(format!("project file has more than {MAX_ENTRIES} entries")));
    }

    // Count every declared payload, including ignored entries, before any decompression
    let mut declared_budget = LoadBudget::new(limit_bytes);
    for index in 0..zip.len() {
        let entry = zip.by_index_raw(index).map_err(zip_err)?;
        declared_budget.take(entry.size()).map_err(FormatError::TooLarge)?;
    }

    // Shared compressed ranges can amplify one payload into many apparent files
    if zip.has_overlapping_entries().map_err(zip_err)? {
        return Err(malformed(KIND, "overlapping ZIP entries are not supported"));
    }

    Ok(())
}

fn open_archive<'a, R: Read + Seek>(reader: R, metadata_only: &'a Cell<bool>, metadata_bytes: u64) -> Result<ZipArchive<MetadataReader<'a, R>>> {
    let reader = MetadataReader::new(reader, metadata_only, metadata_bytes)?;
    // Project archives start at byte zero; avoid scanning payloads to guess an embedded ZIP offset
    let config = zip::read::Config { archive_offset: zip::read::ArchiveOffset::Known(0) };
    let archive = ZipArchive::with_config(config, reader).map_err(zip_err)?;
    metadata_only.set(false);
    Ok(archive)
}

fn load_with_budget<R: Read + Seek>(reader: R, mut budget: LoadBudget) -> Result<ProjectFile> {
    let metadata_only = Cell::new(true);
    let mut zip = open_archive(reader, &metadata_only, MAX_ZIP_METADATA_BYTES)?;
    check_archive(&mut zip, budget.remaining())?;

    let json = read_all(&mut zip.by_name("project.json").map_err(zip_err)?, MAX_JSON, &mut budget, "project.json")?;
    #[derive(serde::Deserialize)]
    struct Version {
        format_version: u32,
    }
    let v: Version = serde_json::from_slice(&json).map_err(|e| malformed(KIND, format!("project.json: {e}")))?;
    if v.format_version > FORMAT_VERSION {
        return Err(FormatError::NewerVersion { found: v.format_version, supported: FORMAT_VERSION });
    }

    let (project, mut recovered) = recovery::read_project(&json).map_err(|e| malformed(KIND, format!("project.json: {e}")))?;
    project.validate()?;
    let too_large = |t: &texturecraft_core::TextureSet| u64::from(t.size().0) * u64::from(t.size().1) > MAX_TEXELS;
    if let Some(ts) = project.texture_sets.iter().find(|t| t.resolution == 0 || t.resolution > MAX_IMAGE_SIZE || too_large(t)) {
        return Err(malformed(KIND, format!("texture set \"{}\" has resolution {} over {} UV tiles", ts.name, ts.resolution, ts.uv_tiles.len().max(1))));
    }

    let mut pf = ProjectFile {
        pixels: PixelStore::default(),
        baked: BTreeMap::new(),
        mesh: None,
        high_poly: Vec::new(),
        cage: None,
        bake_paint: BTreeMap::new(),
        thumbnail: None,
        resources: BTreeMap::new(),
        warnings: Vec::new(),
        project,
    };

    for i in 0..zip.len() {
        let mut f = zip.by_index(i).map_err(zip_err)?;
        let declared_bytes = f.size();
        let name = f.name().map_err(zip_err)?.into_owned();
        let size_of = |id: LayerId| texture_set_of(&pf.project, id).and_then(|ts| pf.project.texture_sets.get(ts)).map(|t| t.size());
        if name == "thumbnail.png" {
            pf.thumbnail = Some(read_all(&mut f, MAX_THUMBNAIL, &mut budget, &name)?);
        } else if let Some(res_name) = name.strip_prefix("resources/") {
            if res_name.is_empty() || sanitize_resource_name(res_name) != res_name {
                return Err(malformed(KIND, format!("bad resource name {name}")));
            }

            let bytes = read_all(&mut f, MAX_RESOURCE, &mut budget, &name)?;
            pf.resources.insert(res_name.to_string(), bytes);
        } else if let Some((dir, file)) = name.split_once('/').filter(|(d, _)| *d == "highpoly" || *d == "cage") {
            if file.is_empty() || sanitize_resource_name(file) != file {
                return Err(malformed(KIND, format!("bad mesh name {name}")));
            }

            let entry = (file.to_string(), read_all(&mut f, crate::mesh_import::MAX_MESH_FILE_BYTES, &mut budget, &name)?);
            match dir {
                "cage" => pf.cage = Some(entry),
                _ => pf.high_poly.push(entry),
            }
        } else if let Some(rest) = name.strip_prefix("bake_paint/").and_then(|name| name.strip_suffix(".bin")) {
            let (texture_set, target) = rest.split_once('_').ok_or_else(|| malformed(KIND, format!("bad entry {name}")))?;
            let texture_set: usize = texture_set.parse().map_err(|_| malformed(KIND, format!("bad entry {name}")))?;
            if texture_set >= pf.project.texture_sets.len() {
                return Err(malformed(KIND, format!("unknown texture set in {name}")));
            }

            let target = match target {
                "cage" => BakePaintTarget::Cage,
                "skew" => BakePaintTarget::Skew,
                _ => return Err(malformed(KIND, format!("unknown bake paint target in {name}"))),
            };
            let (width, height, values) = read_baked(&mut f, declared_bytes, 1, &mut budget, &name)?;
            pf.bake_paint.insert((texture_set, target), BakePaintData { width, height, values });
        } else if let Some(ext) = name.strip_prefix("mesh.").filter(|e| !e.contains('/')) {
            let bytes = read_all(&mut f, crate::mesh_import::MAX_MESH_FILE_BYTES, &mut budget, &name)?;
            pf.mesh = Some((ext.to_string(), bytes));
        } else if let Some(rest) = name.strip_prefix("pixels/layer_").and_then(|r| r.strip_suffix(".bin")) {
            let (id, chan) = rest.split_once('_').ok_or_else(|| malformed(KIND, format!("bad entry name {name}")))?;
            let id = LayerId(id.parse().map_err(|_| malformed(KIND, format!("bad entry name {name}")))?);
            let c = Channel::ALL.into_iter().find(|c| c.id() == chan).ok_or_else(|| malformed(KIND, format!("unknown channel in {name}")))?;
            // Drop pixels of layers that are no longer in any stack
            let Some(size) = size_of(id) else { continue };

            let img = read_tiles(&mut f, size, c.components() + 1, &mut budget, &name)?;
            pf.pixels.layers.insert((id, c), img);
        } else if let Some(rest) = name.strip_prefix("pixels/mask_").and_then(|r| r.strip_suffix(".bin")) {
            let id = LayerId(rest.parse().map_err(|_| malformed(KIND, format!("bad entry name {name}")))?);
            let Some(size) = size_of(id) else { continue };

            let img = read_tiles(&mut f, size, 2, &mut budget, &name)?;
            pf.pixels.masks.insert(id, img);
        } else if let Some(rest) = name.strip_prefix("baked/").and_then(|r| r.strip_suffix(".bin")) {
            let (ts, map) = rest.split_once('_').ok_or_else(|| malformed(KIND, format!("bad entry name {name}")))?;
            let ts: usize = ts.parse().map_err(|_| malformed(KIND, format!("bad entry name {name}")))?;
            let Some(m) = recovery::baked_map(map) else {
                recovered = true;
                continue;
            };

            if ts >= pf.project.texture_sets.len() {
                continue;
            }

            let (res, height, data) = read_baked(&mut f, declared_bytes, m.components(), &mut budget, &name)?;
            let maps = pf.baked.entry(ts).or_default();
            if maps.maps.is_empty() {
                maps.resolution = res;
            } else if maps.resolution != res || maps.maps.keys().next().is_some_and(|k| maps.height(*k) != height) {
                return Err(malformed(KIND, format!("{name}: baked maps of one texture set differ in size")));
            }
            maps.maps.insert(m, Arc::new(data));
        }
        // Ignore other entries from newer minor revisions
    }

    pf.project.format_version = FORMAT_VERSION;

    if recovered {
        pf.warnings.push(
            "Warning: Some project settings or mesh maps could not be imported. TextureCraft recovered what it could, using defaults where needed.".into(),
        );
    }

    Ok(pf)
}

#[cfg(test)]
mod tests;
