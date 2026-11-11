//! The headless backend: a [`Session`] plus file I/O and windowless renders. Synchronous and
//! UI-free; the MCP server, the RPC server and the CLI all drive it.

use std::path::Path;

use serde_json::{Value, json};
use texturecraft_engine::core::PROJECT_EXT;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, commands, file_cmds, inspect};
use texturecraft_render::ViewKind;

use crate::budgets::{MAX_LOCAL_RENDER_SIDE, MAX_PNG_SIDE, check_png, encode_png};
use crate::render::Gpu;
use crate::workspace::{MAX_READ_BYTES, authorize_command};
use crate::{AuthorizedWorkspace, AutomationError, bad};

/// Where file paths given to [`Headless`] may point.
pub enum Filesystem {
    /// No file access (an MCP server started without roots).
    Denied,
    /// Host paths as given: the local CLI user typed them.
    TrustedLocal,
    /// Relative paths under the authorized roots (remote or agent callers).
    Workspace(AuthorizedWorkspace),
}

/// What [`Headless::render_png`] draws.
#[derive(Clone, Debug, Default)]
pub struct RenderOptions {
    /// `3d` (default) or `2d`/`uv` (the UV view of the active texture set).
    pub view: Option<String>,
    /// Pixels; default the session viewport (the space `tool.stroke` points live in).
    pub width: Option<u32>,
    pub height: Option<u32>,
    /// Camera overrides in degrees (the session camera is not changed).
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    /// Path trace with this many samples per pixel (Rendering mode) instead of the GPU view.
    pub samples: Option<u32>,
    /// Light bounces for path tracing (default 6).
    pub bounces: Option<u32>,
    /// Path trace on the CPU even when a GPU with compute shaders exists.
    pub cpu: bool,
}

/// A PNG render and what drew it (the GPU adapter, or the CPU path tracer).
pub struct Rendered {
    pub width: u32,
    pub height: u32,
    pub png: Vec<u8>,
    pub by: String,
}

/// Most samples per pixel a path-traced render may take.
pub const MAX_TRACE_SAMPLES: u32 = 4096;

pub struct Headless {
    pub session: Session,
    fs: Filesystem,
    /// Created on the first render; `Err` once no adapter was found.
    gpu: Option<Result<Gpu, String>>,
}

fn ext_of(path: &str) -> String {
    Path::new(path).extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase()
}

fn join(dir: &str, name: &str) -> String {
    if dir == "." { name.to_string() } else { format!("{}/{name}", dir.trim_end_matches('/')) }
}

impl Headless {
    pub fn new(session: Session, fs: Filesystem) -> Headless {
        // Bakes run here (CLI, MCP) cast their rays on the GPU when there is one.
        texturecraft_engine::bake::install_gpu_tracer(std::sync::Arc::new(texturecraft_render::gpu_bake::GpuBaker::new()));
        Headless { session, fs, gpu: None }
    }

    /// The session a fresh server starts with: the material ball at 1024.
    pub fn default_session() -> Result<Session, AutomationError> {
        Ok(Session::new_project("Untitled", primitives::material_ball(), "builtin:material_ball", None, 1024)?)
    }

    fn trusted(&self) -> bool {
        matches!(self.fs, Filesystem::TrustedLocal)
    }

    fn read(&self, path: &str) -> Result<Vec<u8>, AutomationError> {
        match &self.fs {
            Filesystem::Denied => Err(bad("file access is not granted: start the server with --read-root <dir>")),
            Filesystem::TrustedLocal => {
                let len = std::fs::metadata(path).map_err(|e| AutomationError::Io(format!("cannot read {path}: {e}")))?.len();
                if len > MAX_READ_BYTES {
                    return Err(bad(format!("{path} is {len} bytes (limit {MAX_READ_BYTES})")));
                }
                std::fs::read(path).map_err(|e| AutomationError::Io(format!("cannot read {path}: {e}")))
            }
            Filesystem::Workspace(w) => w.read(path),
        }
    }

    fn write(&self, path: &str, bytes: &[u8]) -> Result<(), AutomationError> {
        match &self.fs {
            Filesystem::Denied => Err(bad("file access is not granted: start the server with --write-root <dir>")),
            Filesystem::TrustedLocal => {
                // Atomic: a crash or full disk never leaves a half-written project.
                let tmp = format!("{path}.tmp");
                let res = std::fs::write(&tmp, bytes).and_then(|()| std::fs::rename(&tmp, path));
                if res.is_err() {
                    let _ = std::fs::remove_file(&tmp);
                }
                res.map_err(|e| AutomationError::Io(format!("cannot write {path}: {e}")))
            }
            Filesystem::Workspace(w) => w.write(path, bytes),
        }
    }

    fn create_dir_all(&self, dir: &str) -> Result<(), AutomationError> {
        match &self.fs {
            _ if dir == "." => Ok(()),
            Filesystem::Denied => Err(bad("file access is not granted: start the server with --write-root <dir>")),
            Filesystem::TrustedLocal => std::fs::create_dir_all(dir).map_err(|e| AutomationError::Io(format!("cannot create {dir}: {e}"))),
            Filesystem::Workspace(w) => w.create_dir_all(dir),
        }
    }

    /// Replace the session, keeping the viewport size scripted strokes use.
    fn replace(&mut self, mut next: Session) -> Value {
        next.viewport = self.session.viewport;
        self.session = next;
        self.inspect()
    }

    pub fn inspect(&self) -> Value {
        inspect::session(&self.session)
    }

    pub fn command_list(&self, filter: Option<&str>) -> Value {
        commands::list(&self.session, filter)
    }

    /// Run an engine command. Untrusted callers can't reach host paths through it.
    pub fn command_run(&mut self, id: &str, params: Value) -> Result<Value, AutomationError> {
        if !self.trusted() {
            authorize_command(id, &params)?;
        }
        Ok(texturecraft_engine::execute(&mut self.session, id, &params)?)
    }

    /// Run `[{id, params?}]` in order, stopping at the first error (reported with its index).
    pub fn batch(&mut self, steps: &[Value]) -> Value {
        let mut results = Vec::new();
        for (i, step) in steps.iter().enumerate() {
            let id = step.get("id").and_then(Value::as_str).unwrap_or_default();
            match self.command_run(id, step.get("params").cloned().unwrap_or(Value::Null)) {
                Ok(v) => results.push(v),
                Err(e) => return batch_reply(results, Some((i, id, e.to_string()))),
            }
        }
        batch_reply(results, None)
    }

    /// Open a `.tcraft` project or a mesh file (a new project at 2048).
    pub fn open(&mut self, path: &str) -> Result<Value, AutomationError> {
        if ext_of(path) != PROJECT_EXT {
            return self.import_mesh(path, None, None);
        }
        let mut next = Session::from_project_bytes(&self.read(path)?)?;
        next.path = Some(path.to_string());
        next.modified = false;
        Ok(self.replace(next))
    }

    /// A new project from a mesh file: one texture set per material.
    pub fn import_mesh(&mut self, path: &str, resolution: Option<u32>, name: Option<&str>) -> Result<Value, AutomationError> {
        let f = self.read_mesh(path)?;
        let stem = Path::new(path).file_stem().and_then(|s| s.to_str()).unwrap_or("Untitled");
        let mut next = Session::from_mesh_bytes(&f.name, &f.bytes, resolution.unwrap_or(2048), name.unwrap_or(stem))?;
        next.log.extend(f.warnings);
        Ok(self.replace(next))
    }

    /// Mesh file `path` with the files it references next to it (a `.gltf`'s buffers and images, a
    /// loose USD layer's textures), read under the same access rules.
    fn read_mesh(&self, path: &str) -> Result<texturecraft_formats::MeshFile, AutomationError> {
        let bytes = self.read(path)?;
        let dir = Path::new(path).parent().and_then(|d| d.to_str()).filter(|d| !d.is_empty()).unwrap_or(".");
        // The references stay in the mesh's folder (a link out of it is not followed) and are read up to the cap the
        // importer keeps (a texture's, not the whole-file limit), whichever access the session has.
        let cap = texturecraft_formats::mesh_import::reference_cap(path);
        let mut read = |rel: &str| match &self.fs {
            Filesystem::Workspace(w) => w.read_beneath(dir, rel, cap).ok(),
            Filesystem::TrustedLocal => texturecraft_formats::mesh_import::read_mesh_reference(Path::new(dir), rel, cap),
            Filesystem::Denied => None,
        };
        texturecraft_formats::self_contained_mesh(path, bytes, &mut read).map_err(|e| bad(format!("{path}: {e}")))
    }

    /// Add a high-definition mesh for baking from `path` (or, with `cage`, set the bake cage).
    pub fn add_bake_mesh(&mut self, path: &str, cage: bool) -> Result<Value, AutomationError> {
        let f = self.read_mesh(path)?;
        Ok(self.session.add_bake_mesh(f, cage)?)
    }

    /// Reimport the mesh from `path`, or from the project's own copy, keeping the project.
    pub fn reimport_mesh(&mut self, path: Option<&str>, preserve_strokes: bool) -> Result<Value, AutomationError> {
        let file = path.map(|p| self.read_mesh(p)).transpose()?;
        Ok(self.session.reimport_mesh(file, preserve_strokes)?)
    }

    /// Export the working mesh to `path` (obj, glb, usda, usdc or usdz, from `format` or the
    /// extension); obj writes its .mtl and textures, usda and usdc their textures, next to it.
    pub fn export_mesh(&mut self, path: &str, format: Option<&str>) -> Result<Value, AutomationError> {
        let out = self.session.mesh_export(path, format)?;
        let dir = Path::new(&out.path).parent().and_then(|d| d.to_str()).filter(|d| !d.is_empty()).unwrap_or(".");
        self.create_dir_all(dir)?;
        let mut textures = Vec::new();
        for (name, bytes) in &out.textures {
            let file = join(dir, name);
            self.write(&file, bytes)?;
            textures.push(file);
        }
        self.write(&out.path, &out.bytes)?;
        Ok(out.reply(textures))
    }

    /// Import an image file as a project resource (for `layer.add.image`, `tool.stencil`).
    pub fn import_resource(&mut self, path: &str) -> Result<Value, AutomationError> {
        let bytes = self.read(path)?;
        Ok(json!({ "name": self.session.import_resource_bytes(path, bytes)? }))
    }

    /// Save a layer (default: the selected one) as a smart material file (`.tcmat`).
    pub fn material_save(&mut self, p: &Value) -> Result<Value, AutomationError> {
        let path = p.get("path").and_then(Value::as_str).ok_or_else(|| bad("`path` is required"))?;
        let s = &self.session;
        let id = match p.get("layer").and_then(Value::as_u64) {
            Some(i) => texturecraft_engine::core::LayerId(i),
            None => s.selected_layer().ok_or_else(|| bad("no layer selected and no `layer` given"))?,
        };
        let name = match p.get("name").and_then(Value::as_str) {
            Some(n) => n.to_string(),
            None => s.doc.project.layer(id).map(|l| l.name.clone()).map_err(|_| bad(format!("no layer {}", id.0)))?,
        };
        let bytes = s.smart_material_bytes(id, &name)?;
        let ext = texturecraft_engine::material_lib::SMART_MATERIAL_EXT;
        let path = if ext_of(path) == ext { path.to_string() } else { format!("{path}.{ext}") };
        self.write(&path, &bytes)?;
        Ok(json!({ "path": path }))
    }

    /// Add a smart material from a `.tcmat` file above the selection.
    pub fn material_load(&mut self, path: &str) -> Result<Value, AutomationError> {
        let bytes = self.read(path)?;
        Ok(json!({ "layer": self.session.add_smart_material_bytes(&bytes)?.0 }))
    }

    /// Save as `.tcraft` to `path`, or to the project's own file.
    pub fn save(&mut self, path: Option<&str>) -> Result<Value, AutomationError> {
        let target = match path.or(self.session.path.as_deref()) {
            Some(p) if ext_of(p) == PROJECT_EXT => p.to_string(),
            Some(p) => format!("{p}.{PROJECT_EXT}"),
            None => return Err(bad("the project has no file yet: give `path`")),
        };
        self.write(&target, &self.session.to_project_bytes()?)?;
        self.session.path = Some(target.clone());
        self.session.modified = false;
        Ok(json!({ "path": target }))
    }

    /// Export textures to `dir` (`file.exportTextures` params: preset, ts, format, padding, size…).
    pub fn export(&mut self, p: &Value) -> Result<Value, AutomationError> {
        let dir = p.get("dir").and_then(Value::as_str).filter(|d| !d.is_empty()).ok_or_else(|| bad("`dir` is required"))?;
        let opts = file_cmds::export_options(&self.session, p, "textures.export")?;
        let n = self.session.set_count();
        let sets: Vec<usize> = match p.get("ts") {
            None | Some(Value::Null) => (0..n).collect(),
            Some(Value::String(a)) if a == "all" => (0..n).collect(),
            Some(v) => vec![v.as_u64().and_then(|t| usize::try_from(t).ok()).filter(|t| *t < n).ok_or_else(|| bad(format!("invalid `ts` {v}")))?],
        };
        let files = self.session.export_files(&sets, &opts)?;
        self.create_dir_all(dir)?;
        let mut written = Vec::new();
        for f in &files {
            let path = join(dir, &f.name);
            self.write(&path, &f.bytes)?;
            written.push(path);
        }
        let warnings: Vec<&String> = files.iter().flat_map(|f| &f.warnings).collect();
        Ok(json!({ "files": written, "warnings": warnings }))
    }

    /// Render the session to PNG. The viewport needs a GPU adapter; path tracing (`samples`)
    /// runs on the GPU when it has compute shaders and falls back to the CPU.
    pub fn render_png(&mut self, o: &RenderOptions) -> Result<Rendered, AutomationError> {
        let s = &self.session;
        let kind = match o.view.as_deref().unwrap_or("3d") {
            "3d" => ViewKind::Scene,
            "2d" | "uv" => ViewKind::Uv { ts: s.ts(), zoom: 1.0, pan: [0.0, 0.0] },
            other => return Err(bad(format!("unknown view `{other}` (3d, 2d or uv)"))),
        };
        let max = if self.trusted() { MAX_LOCAL_RENDER_SIDE } else { MAX_PNG_SIDE };
        let side = |v: Option<u32>, d: f32| {
            let v = v.unwrap_or(d.round() as u32);
            if (16..=max).contains(&v) { Ok(v) } else { Err(bad(format!("render size {v} must be 16 to {max} pixels"))) }
        };
        let size = [side(o.width, s.viewport.x)?, side(o.height, s.viewport.y)?];
        let mut camera = s.camera;
        for (v, slot, lim) in [(o.yaw, &mut camera.yaw, 1e6), (o.pitch, &mut camera.pitch, 89.0)] {
            if let Some(v) = v {
                if !v.is_finite() {
                    return Err(bad("camera angles must be finite"));
                }
                *slot = v.clamp(-lim, lim).to_radians();
            }
        }
        if let Some(samples) = o.samples {
            if !(1..=MAX_TRACE_SAMPLES).contains(&samples) || o.view.as_deref().is_some_and(|v| v != "3d") {
                return Err(bad(format!("path tracing renders the 3D view with 1 to {MAX_TRACE_SAMPLES} samples")));
            }
            let settings = texturecraft_render::trace::TraceSettings { bounces: o.bounces.unwrap_or(6).clamp(1, 32) };
            let mut traced = None;
            let mut by = "CPU path tracer".to_string();
            if !o.cpu
                && let Ok(gpu) = self.gpu.get_or_insert_with(Gpu::new)
            {
                // Last-resort guard, as for viewport renders below.
                let session = &mut self.session;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.path_trace(session, size, camera, samples, &settings))) {
                    Ok(Ok(px)) => {
                        by = format!("{} (GPU path tracer)", gpu.adapter());
                        traced = Some(px);
                    }
                    Ok(Err(e)) => by = format!("CPU path tracer ({e})"),
                    Err(_) => {
                        by = "CPU path tracer (the GPU path tracer failed)".into();
                        self.gpu = None;
                    }
                }
            }
            let (w, h, rgba) = match traced {
                Some(px) => px,
                None => crate::render::path_trace(&mut self.session, size, camera, samples, &settings),
            };
            let png = encode_png(w, h, rgba)?;
            if !self.trusted() {
                check_png(png.len())?;
            }
            return Ok(Rendered { width: w, height: h, png, by });
        }
        let gpu = match self.gpu.get_or_insert_with(Gpu::new) {
            Ok(g) => g,
            Err(e) => return Err(AutomationError::Other(format!("cannot render: {e}"))),
        };
        // Last-resort guard: wgpu panics on device errors; drop the device and report instead.
        let session = &mut self.session;
        let out = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| gpu.render(session, kind, size, camera)));
        let (w, h, rgba) = match out {
            Ok(Some(px)) => px,
            Ok(None) => return Err(AutomationError::Other("the GPU returned no image".into())),
            Err(_) => {
                self.gpu = None;
                return Err(AutomationError::Other("the GPU renderer failed".into()));
            }
        };
        let png = encode_png(w, h, rgba)?;
        if !self.trusted() {
            check_png(png.len())?;
        }
        Ok(Rendered { width: w, height: h, png, by: self.adapter().unwrap_or("GPU").to_string() })
    }

    /// The GPU adapter used for renders, if one was created.
    pub fn adapter(&self) -> Option<&str> {
        self.gpu.as_ref().and_then(|g| g.as_ref().ok()).map(Gpu::adapter)
    }
}

/// `{completed, results, failed: {index, id, error} | null}` for `command_batch`.
pub fn batch_reply(results: Vec<Value>, failed: Option<(usize, &str, String)>) -> Value {
    json!({
        "completed": results.len(),
        "results": results,
        "failed": failed.map(|(index, id, error)| json!({ "index": index, "id": id, "error": error })),
    })
}
