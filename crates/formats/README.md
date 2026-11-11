# texturecraft-formats

File formats for TextureCraft: mesh import and export, texture export with presets, and the
`.tcraft` project file. Every reader takes untrusted bytes and returns `Result`; nothing panics on bad
input. Byte-based functions work on every target (the web build loads and saves through bytes);
path-based functions are native only (`#[cfg(not(target_arch = "wasm32"))]`).

## Public API

### Mesh import (`mesh_import`)

```rust
pub fn import_mesh_bytes(file_name: &str, bytes: &[u8]) -> Result<ImportedMesh>;
pub fn import_mesh(path: &Path) -> Result<ImportedMesh>;            // native, = read_mesh_file + import_mesh_bytes
pub fn self_contained_mesh(name: &str, bytes: Vec<u8>, read: &mut dyn FnMut(&str) -> Option<Vec<u8>>) -> Result<MeshFile>;
pub fn read_mesh_file(path: &Path) -> Result<MeshFile>;             // native
pub struct MeshFile { pub name: String, pub bytes: Vec<u8>, pub warnings: Vec<String> }
pub struct ImportedMesh { pub mesh: Mesh, pub report: MeshReport }
pub struct MeshReport {
    pub triangles: usize, pub vertices: usize, pub materials: usize,
    pub uvs_generated: bool, pub uv_out_of_range: usize, pub overlap_ratio: f32,
    pub zero_area_uv_triangles: usize, pub warnings: Vec<String>,
}
```

| Format | Notes |
|---|---|
| `.glb` / `.gltf` | Default scene's node tree flattened to world space; normals by inverse transpose; winding flipped under mirroring transforms; one submesh per material (`Default` for none, `Material_<i>` when unnamed); missing normals are flat (glTF rule). Bytes: GLB or data-URI buffers and images; `self_contained_mesh` folds a `.gltf`'s `.bin` files and images from its folder in as data URIs (files outside the folder are never read). Triangle primitives only; sparse accessors are refused. Cameras (perspective `yfov`, orthographic `ymag`) become `ImportedMesh::cameras`. |
| `.fbx` | FBX 6.x and 7.x, binary (32- and 64-bit node records, zlib arrays) and ASCII. Every `Mesh` geometry connected to a Model, with the Model hierarchy's transforms (`Lcl Translation/Rotation/Scaling`, `RotationOrder`, `Pre/PostRotation`, pivots and offsets, geometric transform); normals and UV set 0 from the layer elements (`ByPolygonVertex`, `ByControlPoint`, `ByPolygon`, `AllSame`; `Direct` / `IndexToDirect`); per-polygon materials named by the Materials connected to the Model. Units (`UnitScaleFactor`, cm by default) become meters and the axis system becomes Y up. Camera Models become cameras (looking down +X; lens from the Camera NodeAttribute or, in FBX 6, the Model: `FocalLength` over `FilmHeight`, else `FieldOfViewY` / `FieldOfView`; `CameraProjectionType` 1 orthographic); FBX 6's Producer views are skipped. Materials bring `DiffuseColor` × `DiffuseFactor` and the diffuse texture, embedded or a file: `self_contained_mesh` packages an FBX with the texture files it names (`RelativeFilename`, else the file name of `FileName`) found in its folder, a stored zip that still reads as `.fbx`. FBX 5 and older are refused; decoded data is capped at 2 GiB. |
| `.dae` | COLLADA: `triangles`, `polylist`, `polygons` (holes ignored) with `VERTEX` / `NORMAL` / `TEXCOORD` inputs at any offset (lowest `set`); materials by `bind_material` symbol; the visual scene's node tree (`matrix`, `translate`, `rotate`, `scale`, `lookat`, `instance_node`, skinned `instance_controller` with its bind shape); `instance_camera` (perspective `yfov` or `xfov` + `aspect_ratio`, orthographic `ymag` or `xmag` + `aspect_ratio`) as cameras; `unit` to meters, `Z_UP` / `X_UP` to Y up. Lines and strips are skipped with a warning. |
| `.usda` / `.usdc` / `.usd` / `.usdz` | USD, text or Crate binary (0.4.0 and newer: LZ4 sections, compressed integer and float arrays, inlined values; `.usd` by magic) or a usdz package (its first layer). Every defined, active, non-guide, visible `Mesh` prim: points, faceVertexCounts/Indices (holes skipped, n-gons fanned), normals (`primvars:normals` or `normals`) and UVs (`primvars:st`, `st0`, `UVMap`, `uv` or the first `texCoord2f[]` primvar), indexed or not, any interpolation; `orientation = leftHanded`. Xform ops (`translate`, `scale`, `rotateX/Y/Z`, the six `rotateXYZ` orders, `orient`, `transform`, `!invert!`, `!resetXformStack!`) by `xformOpOrder`. Material names from `material:binding` (inherited) and `materialBind` GeomSubsets. Internal references compose (`class` prims are only drawn through them); time samples use the first sample. Stage `metersPerUnit` (fallback 0.01, as in USD) and `upAxis` become meters, Y up. `Camera` prims (`focalLength` over `verticalAperture`, `projection`) become cameras. Skipped with a warning: external references and payloads, variant sets, PointInstancers. Decoded data is capped at 2 GiB. `self_contained_mesh` packages a loose layer whose UsdPreviewSurface base color textures sit in its folder into a `.usdz` with them, so high polys keep their textures. |
| `.obj` | Triangulated, one vertex per unique `v/vt/vn`, **v flipped** (OBJ is bottom-left). Material names come from `usemtl` (MTL files are never read; they only carry shading data). |
| `.ply` | ascii, binary little/big endian; `x y z`, `nx ny nz`, `s t` / `u v` / `texture_u texture_v`; n-gons fan-triangulated. |
| `.stl` | ascii and binary (size-checked, so binary files whose header says `solid` work); flat normals; no UVs. |

FBX and COLLADA polygons are fan-triangulated with one vertex per unique corner. Missing
normals (OBJ/PLY/FBX/COLLADA) are smooth over welded positions. Missing UVs run
`texturecraft_geom::unwrap::auto_unwrap` (charts within 60° of their seed normal, principal-axis
rotation, shelf packing per material with one texel density). Tangents are MikkTSpace. Limits:
2 GiB files, `MAX_TRIANGLES` triangles, 3 × that many vertices; meshes whose triangles are all
degenerate are refused (MikkTSpace crashes on them).

### Mesh export (`mesh_export`)

```rust
pub fn obj_bytes(mesh: &Mesh, name: &str) -> Vec<u8>;
pub fn glb_bytes(mesh: &Mesh, name: &str, textures: &[GlbTextures]) -> Result<Vec<u8>>;
pub fn glb_export_options() -> ExportOptions;                     // render GlbTextures with these
pub struct GlbTextures { pub base_color, pub orm, pub normal, pub emissive: Option<Vec<u8>> }  // PNG
pub fn usd_bytes(mesh: &Mesh, name: &str, textures: &[Vec<ExportedFile>], format: UsdFormat) -> Result<UsdExport>;
pub fn usd_export_options() -> ExportOptions;                     // render each material's textures with these
pub enum UsdFormat { Usda, Usdc, Usdz }
pub struct UsdExport { pub bytes: Vec<u8>, pub textures: Vec<(String, Vec<u8>)> }  // textures: usda/usdc only
```

USD: one `Mesh` prim (vertex normals and `primvars:st`, `subdivisionScheme = "none"`), one
`materialBind` GeomSubset per material when there are several, and a UsdPreviewSurface material
per material with a UsdUVTexture per map (base color and emissive sRGB; roughness, metallic,
OpenGL normal with scale 2 / bias -1, opacity, clear coat and clear coat roughness raw; black
emissive maps dropped, and both clear coat maps when the coat is black). Meters, Y up.
`.usdc` is a Crate 0.8.0 file; `.usdz` is that layer first plus the PNGs, stored uncompressed at
64-byte aligned offsets; `.usda` / `.usdc` reference `<name>_<material>_<slot>.png` next to the
layer.

GLB: one primitive per submesh, one metallic-roughness material per mesh material with its
textures embedded in the binary chunk (base color sRGB, `alphaMode: BLEND` when it has alpha;
ORM as occlusion + metallic-roughness; OpenGL normal; emissive only when not black).

### Texture export (`export`, `presets`)

```rust
pub fn list_outputs(project: &Project, ts: usize, mesh_name: &str, opts: &ExportOptions) -> Result<Vec<PlannedFile>>;
pub fn render_maps(input: &ExportInput, opts: &ExportOptions) -> Result<Vec<ExportedFile>>;
pub fn write_files(files: &[ExportedFile], dir: &Path) -> Result<Vec<PathBuf>>;   // native, atomic
pub fn builtin_presets() -> Vec<ExportPreset>;
pub fn builtin_preset(name: &str) -> Option<ExportPreset>;
pub fn preset_from_json(json: &str) -> Result<ExportPreset>;
pub fn preset_to_json(preset: &ExportPreset) -> Result<String>;
```

- `ExportPreset { name, maps: Vec<MapSpec> }`, `MapSpec { file, format, bit_depth, channels: Vec<(Dest, SourceRef)> }`
  (serde JSON, so presets can be user files).
- `Dest`: `R G B A` or `L` (gray; `L` + `A` = gray + alpha).
- `SourceRef`: `Channel { channel, component }` (`Normal` = the **final** normal in the export
  normal format), `Virtual { map, component }` (`NormalOpenGl`, `NormalDirectX`, `Glossiness` =
  1 − roughness, `AoMixed` = baked AO × AO channel, `F0` = 0.08 × specular level),
  `Baked { map, component }`, `Constant(f32)`.
- `ExportOptions { preset, format_override, bit_depth_override, padding, size, normal_format }`;
  `Padding::{None, Dilation(n), DilationDefault(n), Infinite}`; `size` must be a power of two
  (box filter down, bilinear up).
- File templates: `$project`, `$mesh`, `$textureSet`, `$channel`, `$colorSpace`; names are
  sanitized to `[A-Za-z0-9_.-]` (reserved device names get a `_` prefix); case-insensitive
  collisions get `_2`, `_3`, … plus a warning.
- A map is written only when one of its sources has data in the texture set (channel present,
  map baked, …); alpha from a missing channel is dropped.
- Formats: PNG 8/16, JPEG 8 (q 95), TGA 8, TIFF 8/16, BMP 8, WebP 8 (lossless), EXR 16f/32f
  (`Y`, `YA`, `RGB`, `RGBA`, ZIP), HDR 32f. Color maps are sRGB in 8/16-bit and linear in float
  formats; data maps are linear; height is stored as `h · 0.5 + 0.5`.
- Final normal: Sobel normal from the height channel (strength = `height_strength`; with 1, a
  height change of 1 over 1/16 of the texture width tilts 45°; `+Y` up the image), combined with
  the normal channel, then with the baked normal map, by Reoriented Normal Mapping. DirectX
  flips green.

Built-in presets:

| Preset | Files (`$mesh_$textureSet_…`) |
|---|---|
| Document channels + Normal + AO | one per document channel (`$channel`), plus `normal` and `ambientOcclusion` (mixed) |
| glTF PBR Metal Roughness | `BaseColor` (sRGB, A = opacity), `OcclusionRoughnessMetallic`, `Normal` (OpenGL), `Emissive` (if present) |
| Unreal Engine (Packed) | `BaseColor`, `OcclusionRoughnessMetallic`, `Normal` (DirectX) |
| Unity HDRP (Mask Map) | `BaseMap`, `MaskMap` (R metal, G AO, B detail = 0, A smoothness), `Normal` (OpenGL) |
| Unity Standard (Metallic Smoothness) | `Albedo`, `MetallicSmoothness` (R metal, A smoothness), `Normal` (OpenGL), `Occlusion` |
| Blender (Principled BSDF) | `BaseColor`, `Metallic`, `Roughness`, `Normal` (OpenGL), `Height` (16-bit), `Emission`, `Specular` |
| Godot 4 (ORM) | `albedo` (A = opacity), `orm` (R AO, G roughness, B metal), `normal` (OpenGL), `emission` |
| OpenPBR (separate maps) | one map per OpenPBR Surface input the set has: `base_color`, `base_metalness`, `specular_roughness`, `specular_roughness_anisotropy`, `coat_weight`, `coat_roughness`, `fuzz_weight`, `fuzz_color`, `fuzz_roughness`, `transmission_weight`, `subsurface_weight`, `emission_color`, `geometry_opacity`, `geometry_normal` (OpenGL), `height` (16-bit) |
| Separate maps | one file per channel, gray where scalar, height 16-bit |
| Mesh maps | every baked map (`$channel` = map id), 16-bit |

### Project file (`project_file`)

```rust
pub struct ProjectFile {
    pub project: Project, pub pixels: PixelStore, pub baked: BTreeMap<usize, BakedMaps>,
    pub mesh: Option<(String /* ext */, Vec<u8>)>, pub thumbnail: Option<Vec<u8>>,
    pub warnings: Vec<String>, // Import warnings, never saved
}
pub fn save_project_bytes(pf: &ProjectFile) -> Result<Vec<u8>>;
pub fn load_project_bytes(bytes: &[u8]) -> Result<ProjectFile>;
pub fn save_project(path: &Path, pf: &ProjectFile) -> Result<()>;   // native, atomic
pub fn load_project(path: &Path) -> Result<ProjectFile>;            // native
pub fn set_project_open_limit_gib(limit_gib: u32);                 // subsequent loads, minimum 1 GiB
pub fn project_open_limit_gib() -> u32;
```

`.tcraft` layout (zip, deflate):

| Entry | Content |
|---|---|
| `project.json` | serde JSON of `Project` |
| `mesh.<ext>` | the original mesh file bytes (absent for `builtin:<id>` meshes) |
| `pixels/layer_<id>_<channelId>.bin` | painted layer (or content paint effect) channel |
| `pixels/mask_<id>.bin` | painted mask effect |
| `baked/<ts>_<mapId>.bin` | baked mesh map of texture set `<ts>` |
| `thumbnail.png` | optional, stored |

`.bin`: header `TCTX`, version, width, height, components, tile count (u32 LE each), then per
allocated 64×64 tile `tx`, `ty` (u32 LE) and the raw `u16` LE components (sparse, lossless).
Baked maps: the same header with 0 tiles, then `width × height × components` f32 LE.

Project format version 2 stores baked maps in uniform UV tile blocks, independent of working
tile-resolution overrides. Older projects load their stored bake pixels as-is, without converting
the old layout; their appearance may change until rebaked. Painted layers and cage/skew correction
maps retain their working atlas layout.

Loading runs `Project::validate`, refuses a newer `format_version` (`FormatError::NewerVersion`),
checks every tile against its texture set's resolution and component count, rejects repeated or
out-of-range tiles and trailing data, and drops pixels of layers that are in no stack. Total
decompressed project data is limited to 8 GiB by default. Settings › Project › Max project size on
open changes this limit, in GiB, independently of the save CPU and RAM preferences. Each load keeps
the limit it started with; individual resource, mesh, image and entry-count limits still apply.
The limit counts file data, including binary headers and tile coordinates, rather than peak process
memory. Saves and autosaves are not refused by this open limit; raise it to reopen a larger project.

Opening checks the sum of all declared ZIP payload sizes before decompression, including unknown
entries, and checks actual reads against the same limit independently. Overlapping ZIP entries are
rejected. ZIP indexing is restricted to the final 32 MiB with a 64 MiB read allowance before the
100,000-entry limit can be checked. Baked-map dimensions must agree with the entry size; float data
is read in bounded chunks with fallible allocation. Nested archives are not recursively extracted
by the project reader. Subsequent image and mesh decoding has separate format-specific limits;
the project-open limit is not a combined budget for those decoded assets or the application's heap.

Native saves compress entries in parallel within the configured CPU allowance and an estimated
in-flight byte budget. Stored resources and entries larger than that budget stream directly into
the destination without a whole-entry temporary buffer. The save RAM percentage is an estimate,
excluding compressor, allocator and project-snapshot overhead; it is not a peak process memory cap.

Unreadable bake settings fall back to defaults field by field while readable settings and project
content are retained. Legacy baked-map filenames remain readable; unsupported baked-map types are
skipped. Recovery adds a warning to `ProjectFile::warnings`, displayed in the session log when the
project opens. Structural errors and corrupt pixel data still fail validation.
