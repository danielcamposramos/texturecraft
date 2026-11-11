//! Capability-based file access for untrusted automation paths (MCP, `serve`).
//!
//! Roots are trusted launch-time configuration (`--read-root`, `--write-root`); request paths
//! are untrusted, forward-slash relative paths resolved by `cap-std` beneath the held directory
//! handle, so `..`, absolute paths, drive prefixes and symlink or junction escapes all fail.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use cap_std::ambient_authority;
use cap_std::fs::{Dir, OpenOptions};
use serde_json::Value;

use crate::{AutomationError, bad};

/// Open for reading without blocking on a FIFO with no writer; the callers then reject non-regular files.
fn open_nonblocking(dir: &Dir, rel: &Path) -> std::io::Result<cap_std::fs::File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    cap_std::fs::OpenOptionsExt::custom_flags(&mut options, rustix::fs::OFlags::NONBLOCK.bits() as i32);
    dir.open_with(rel, &options)
}

/// Largest file read through automation (the mesh importer's own limit).
pub const MAX_READ_BYTES: u64 = texturecraft_engine::formats::mesh_import::MAX_MESH_FILE_BYTES;

/// Separate read and write directory capabilities. A missing authority fails closed.
#[derive(Clone, Default)]
pub struct AuthorizedWorkspace {
    read: Option<Arc<Dir>>,
    write: Option<Arc<Dir>>,
}

impl AuthorizedWorkspace {
    pub fn new(read_root: Option<&Path>, write_root: Option<&Path>) -> Result<Self, AutomationError> {
        Ok(Self { read: open_root(read_root, "read")?, write: open_root(write_root, "write")? })
    }

    /// Read one regular file below the read root.
    pub fn read(&self, path: &str) -> Result<Vec<u8>, AutomationError> {
        let rel = relative_path(path)?;
        let root = self.read.as_ref().ok_or_else(|| denied("read"))?;
        let file = open_nonblocking(root, &rel).map_err(|e| file_error("read", path, e))?;
        let meta = file.metadata().map_err(|e| file_error("read", path, e))?;
        if !meta.is_file() {
            return Err(bad(format!("not a regular file: `{path}`")));
        }
        if meta.len() > MAX_READ_BYTES {
            return Err(bad(format!("`{path}` is {} bytes (limit {MAX_READ_BYTES})", meta.len())));
        }
        let mut bytes = Vec::new();
        file.take(MAX_READ_BYTES).read_to_end(&mut bytes).map_err(|e| file_error("read", path, e))?;
        Ok(bytes)
    }

    /// Read one regular file of at most `cap` bytes from `relative`, a path below the folder `dir` of the read root
    /// (`.` for the root itself): the folder is opened first and the file is resolved beneath it, so a link that leaves
    /// the folder fails (a mesh's references must stay in the mesh's folder), and no more than `cap` bytes are read.
    pub fn read_beneath(&self, dir: &str, relative: &str, cap: u64) -> Result<Vec<u8>, AutomationError> {
        let rel = relative_path(relative)?;
        let root = self.read.as_ref().ok_or_else(|| denied("read"))?;
        let folder = if dir == "." { root.try_clone() } else { root.open_dir(relative_path(dir)?) }.map_err(|e| file_error("read", dir, e))?;
        let file = open_nonblocking(&folder, &rel).map_err(|e| file_error("read", relative, e))?;
        let meta = file.metadata().map_err(|e| file_error("read", relative, e))?;
        if !meta.is_file() {
            return Err(bad(format!("not a regular file: `{relative}`")));
        }
        if meta.len() > cap {
            return Err(bad(format!("`{relative}` is {} bytes (limit {cap})", meta.len())));
        }
        let mut bytes = Vec::new();
        file.take(cap.saturating_add(1)).read_to_end(&mut bytes).map_err(|e| file_error("read", relative, e))?;
        if bytes.len() as u64 > cap {
            return Err(bad(format!("`{relative}` grew past the limit {cap}")));
        }
        Ok(bytes)
    }

    /// Create or replace one file below the write root: a temporary file beside the target,
    /// synced and renamed over it. The parent directory must exist.
    pub fn write(&self, path: &str, bytes: &[u8]) -> Result<(), AutomationError> {
        let rel = relative_path(path)?;
        let root = self.write.as_ref().ok_or_else(|| denied("write"))?;
        let leaf = rel.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let tmp = rel.with_file_name(format!(".{leaf}.{}.tmp", std::process::id()));
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        let res = root
            .open_with(&tmp, &options)
            .and_then(|mut f| {
                f.write_all(bytes)?;
                f.sync_all()
            })
            .and_then(|()| root.rename(&tmp, root, &rel));
        if let Err(e) = res {
            let _ = root.remove_file(&tmp);
            return Err(file_error("write", path, e));
        }
        Ok(())
    }

    /// Create a directory (and its parents) below the write root.
    pub fn create_dir_all(&self, path: &str) -> Result<(), AutomationError> {
        let rel = relative_path(path)?;
        let root = self.write.as_ref().ok_or_else(|| denied("write"))?;
        root.create_dir_all(&rel).map_err(|e| file_error("create", path, e))
    }
}

fn open_root(path: Option<&Path>, authority: &str) -> Result<Option<Arc<Dir>>, AutomationError> {
    let Some(path) = path else { return Ok(None) };
    if path.as_os_str().is_empty() {
        return Err(bad(format!("the automation {authority} root is empty")));
    }
    let dir =
        Dir::open_ambient_dir(path, ambient_authority()).map_err(|e| AutomationError::Io(format!("cannot open {authority} root `{}`: {e}", path.display())))?;
    Ok(Some(Arc::new(dir)))
}

/// Validate an untrusted path before any I/O (cap-std then confines resolution to the root).
pub fn relative_path(raw: &str) -> Result<PathBuf, AutomationError> {
    let reject = |why: &str| bad(format!("path rejected ({why}): `{raw}`; use a forward-slash path relative to the workspace root"));
    if raw.is_empty() {
        return Err(reject("empty"));
    }
    if raw.contains('\\') {
        return Err(reject("backslash"));
    }
    if raw.starts_with('/') {
        return Err(reject("absolute"));
    }
    if raw.contains(':') {
        return Err(reject("drive, scheme or stream prefix"));
    }
    for c in raw.split('/') {
        if c.is_empty() || c == "." || c == ".." {
            return Err(reject("empty, `.` or `..` component"));
        }
        if c.ends_with(['.', ' ']) || c.chars().any(char::is_control) {
            return Err(reject("component ends in a dot or space, or has control characters"));
        }
        let stem = c.split('.').next().unwrap_or(c).to_ascii_uppercase();
        let device = matches!(stem.as_str(), "CON" | "PRN" | "AUX" | "NUL" | "CONIN$" | "CONOUT$")
            || (stem.len() == 4 && (stem.starts_with("COM") || stem.starts_with("LPT")) && stem.ends_with(|ch: char| ch.is_ascii_digit()));
        if device {
            return Err(reject("reserved device name"));
        }
    }
    Ok(PathBuf::from(raw))
}

fn denied(authority: &str) -> AutomationError {
    bad(format!("file access is not granted: no {authority} root (start the server with --{authority}-root <dir>)"))
}

fn file_error(op: &str, path: &str, e: std::io::Error) -> AutomationError {
    AutomationError::Io(format!("{op} `{path}` failed: {e}"))
}

/// Host keys a command reads or writes a file through, at the top level or in each `files` entry
const HOST_PATH_KEYS: [&str; 2] = ["path", "dir"];

/// Refuses engine commands that would reach host paths around the workspace roots, so clients use the workspace tools
pub fn authorize_command(id: &str, params: &Value) -> Result<(), AutomationError> {
    let has_path = names_host_path(params) || params.get("files").and_then(Value::as_array).is_some_and(|files| files.iter().any(names_host_path));

    // `file.save` is refused even without a path: a headless session's path is workspace-relative, not a host path
    let allowed = match id {
        "file.close" | "file.removeUnused" => true,
        "file.reimportMesh" => !has_path,
        _ => !id.starts_with("file.") && !has_path,
    };
    if allowed {
        Ok(())
    } else {
        Err(bad(format!(
            "`{id}` with host paths is disabled for automation; use project_open, project_save, mesh_import, resource_import, material_save, material_load or textures_export"
        )))
    }
}

/// Whether `params` names a host path or folder at its own top level
fn names_host_path(params: &Value) -> bool {
    HOST_PATH_KEYS.iter().any(|key| params.get(key).is_some_and(|value| !value.is_null()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn roots(name: &str) -> (PathBuf, PathBuf, AuthorizedWorkspace) {
        let base = std::env::temp_dir().join(format!("tc-ws-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&base);
        let (inside, outside) = (base.join("inside"), base.join("outside"));
        std::fs::create_dir_all(&inside).unwrap();
        std::fs::create_dir_all(&outside).unwrap();
        let ws = AuthorizedWorkspace::new(Some(&inside), Some(&inside)).unwrap();
        (inside, outside, ws)
    }

    #[cfg(unix)]
    #[test]
    fn a_reference_read_stays_in_its_folder_and_under_its_cap() {
        let (inside, _, ws) = roots("beneath");
        std::fs::create_dir_all(inside.join("mesh")).unwrap();
        std::fs::create_dir_all(inside.join("other")).unwrap();
        std::fs::write(inside.join("mesh/tex.png"), b"pixels").unwrap();
        std::fs::write(inside.join("other/secret.png"), b"secret").unwrap();
        // a link to a sibling folder of the same workspace leaves the mesh's folder
        std::os::unix::fs::symlink("../other/secret.png", inside.join("mesh/linked.png")).unwrap();
        assert_eq!(ws.read_beneath("mesh", "tex.png", 6).unwrap(), b"pixels");
        assert!(ws.read_beneath("mesh", "tex.png", 5).is_err(), "over the cap");
        assert!(ws.read_beneath("mesh", "linked.png", 16).is_err(), "a link out of the folder");
        assert!(ws.read_beneath("mesh", "../other/secret.png", 16).is_err());
        std::fs::write(inside.join("top.png"), b"top").unwrap();
        assert_eq!(ws.read_beneath(".", "top.png", 16).unwrap(), b"top");
    }

    #[cfg(unix)]
    #[test]
    fn a_fifo_is_refused_without_blocking() {
        use rustix::fs::{CWD, FileType, Mode, mknodat};
        let (inside, _, ws) = roots("fifo");
        std::fs::create_dir_all(inside.join("mesh")).unwrap();
        for name in ["pipe", "mesh/pipe"] {
            mknodat(CWD, inside.join(name), FileType::Fifo, Mode::from_raw_mode(0o600), 0).unwrap();
        }
        let (tx, rx) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            let _ = tx.send((ws.read("pipe").is_err(), ws.read_beneath("mesh", "pipe", 16).is_err()));
        });
        assert_eq!(rx.recv_timeout(std::time::Duration::from_secs(10)).expect("a FIFO read blocked"), (true, true));
    }

    #[test]
    fn reads_and_writes_stay_in_the_root() {
        let (inside, _, ws) = roots("valid");
        std::fs::write(inside.join("a.txt"), b"canary").unwrap();
        assert_eq!(ws.read("a.txt").unwrap(), b"canary");
        ws.create_dir_all("out/deep").unwrap();
        ws.write("out/deep/b.txt", b"one").unwrap();
        ws.write("out/deep/b.txt", b"two").unwrap();
        assert_eq!(std::fs::read(inside.join("out/deep/b.txt")).unwrap(), b"two");
        let leftovers = std::fs::read_dir(inside.join("out/deep")).unwrap().count();
        assert_eq!(leftovers, 1, "no temporary files left behind");
        assert!(ws.write("missing/c.txt", b"x").is_err(), "parent must exist");
    }

    #[test]
    fn escapes_and_malformed_paths_are_rejected() {
        let (_, outside, ws) = roots("reject");
        let abs = outside.join("x.txt").display().to_string();
        for p in ["", "/abs", "../x", "a/../../x", "a\\b", "C:/x", "a//b", "./a", "a:b", "NUL", "con.txt", "x/COM1.log", "dot.", abs.as_str()] {
            assert!(ws.read(p).is_err(), "read accepted {p:?}");
            assert!(ws.write(p, b"x").is_err(), "write accepted {p:?}");
            assert!(ws.create_dir_all(p).is_err(), "mkdir accepted {p:?}");
        }
    }

    #[test]
    fn authorities_are_separate_and_fail_closed() {
        let (inside, _, _) = roots("separate");
        std::fs::write(inside.join("r.txt"), b"r").unwrap();
        let ro = AuthorizedWorkspace::new(Some(&inside), None).unwrap();
        assert!(ro.read("r.txt").is_ok());
        assert!(ro.write("w.txt", b"x").is_err());
        let none = AuthorizedWorkspace::default();
        assert!(none.read("r.txt").is_err());
    }

    #[cfg(unix)]
    #[test]
    fn symlink_escape_is_rejected() {
        let (inside, outside, ws) = roots("symlink");
        std::fs::write(outside.join("secret.txt"), b"s").unwrap();
        std::os::unix::fs::symlink(&outside, inside.join("link")).unwrap();
        assert!(ws.read("link/secret.txt").is_err());
        assert!(ws.write("link/new.txt", b"x").is_err());
        assert!(!outside.join("new.txt").exists());
    }

    #[cfg(windows)]
    #[test]
    fn junction_escape_is_rejected() {
        let (inside, outside, ws) = roots("junction");
        std::fs::write(outside.join("secret.txt"), b"s").unwrap();
        let link = inside.join("junction");
        let made = std::process::Command::new("cmd.exe").args(["/D", "/C", "mklink", "/J"]).arg(&link).arg(&outside).output();
        if !made.is_ok_and(|o| o.status.success()) {
            eprintln!("skipping: cannot create a junction");
            return;
        }
        assert!(ws.read("junction/secret.txt").is_err());
        assert!(ws.write("junction/new.txt", b"x").is_err());
        assert!(!outside.join("new.txt").exists());
        let _ = std::fs::remove_dir(link);
    }

    #[test]
    fn path_commands_are_denied() {
        for id in ["file.open", "file.new", "file.save", "file.saveAs", "file.saveCopy", "file.exportTextures"] {
            assert!(authorize_command(id, &json!({})).is_err(), "{id}");
        }
        for id in ["resource.import", "material.save", "material.load", "anything.new"] {
            assert!(authorize_command(id, &json!({"path": "/etc/x"})).is_err(), "{id}");
            assert!(authorize_command(id, &json!({"dir": "/etc"})).is_err(), "{id}");
        }
        assert!(authorize_command("file.reimportMesh", &json!({"path": "/etc/x.obj"})).is_err());
        assert!(authorize_command("file.reimportMesh", &json!({})).is_ok());
        assert!(authorize_command("file.close", &json!({})).is_ok());
        assert!(authorize_command("layer.add.fill", &json!({"color": "#ff0000", "path": null})).is_ok());
    }

    #[test]
    fn nested_file_paths_are_denied() {
        let outside_file = json!({"files": [{"path": "/etc/x_ao.png", "map": "ao"}]});
        assert!(authorize_command("textureSet.meshMap.import", &outside_file).is_err());

        let one_of_many = json!({"files": [{"map": "ao"}, {"path": "/etc/x_ao.png"}]});
        assert!(authorize_command("textureSet.meshMap.import", &one_of_many).is_err());

        let no_paths = json!({"files": [{"map": "ao"}, {"path": null}]});
        assert!(authorize_command("textureSet.meshMap.import", &no_paths).is_ok());
    }

    /// Every registry command documenting a host path must be classified (fails when a new
    /// path-taking command is added without an automation policy).
    #[test]
    fn registry_path_params_are_classified() {
        let probe = json!({"path": "/outside/probe", "dir": "/outside", "mesh": "/outside/m.obj"});
        let open: Vec<&str> = texturecraft_engine::command_specs()
            .iter()
            .filter(|c| c.params.contains("path") || c.params.contains("{dir"))
            .filter(|c| authorize_command(c.id, &probe).is_ok())
            .map(|c| c.id)
            .collect();
        assert!(open.is_empty(), "path-taking commands need an automation policy: {open:?}");
    }
}
