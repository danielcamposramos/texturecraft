//! Dependency layering rules, checked by `cargo xtask layers`.
//!
//! - A crate may depend (normal, build or dev) only on workspace crates in strictly lower layers.
//! - Nothing below `ui-egui` (L5) may depend on a UI toolkit: egui, eframe, winit or rfd.
//! - Only `render` and `ui-egui` (and the apps) may depend on wgpu.
//! - The apps and xtask are exempt; any other workspace package must be registered in [`TABLE`].
//!
//! The rule engine works on a small model so it can be unit-tested; `from_metadata` builds that
//! model from `cargo metadata`.

use serde_json::Value;

/// Library crates by directory under `crates/` (package `texturecraft-<dir>`) and their layer.
pub const TABLE: &[(&str, u8)] = &[
    ("core", 0),
    ("texel", 0),
    ("geom", 0),
    ("compose", 1),
    ("bake", 1),
    ("paint", 2),
    ("formats", 2),
    ("engine", 3),
    ("render", 3),
    ("automation", 4),
    ("ui-egui", 5),
];

/// Binaries and build tooling on top of every layer: exempt from the rules.
pub const EXEMPT: &[&str] = &["texturecraft", "texturecraft-cli", "texturecraft-web", "xtask"];

/// External UI toolkit / windowing crates (a trailing `*` is a prefix), allowed from L5 up.
const UI_CRATES: &[&str] = &["egui", "egui_*", "egui-*", "eframe", "winit", "rfd"];
const UI_MIN_LAYER: u8 = 5;

/// External GPU crates, allowed only in [`GPU_ALLOWED`] (and the apps).
const GPU_CRATES: &[&str] = &["wgpu", "wgpu-*"];
const GPU_ALLOWED: &[&str] = &["render", "ui-egui"];

pub fn package(dir: &str) -> String {
    format!("texturecraft-{dir}")
}

/// The `crates/` directory of a layered package.
fn dir_of(pkg: &str) -> Option<&str> {
    pkg.strip_prefix("texturecraft-").filter(|d| TABLE.iter().any(|(t, _)| t == d))
}

fn layer(pkg: &str) -> Option<u8> {
    let dir = dir_of(pkg)?;
    TABLE.iter().find(|(t, _)| *t == dir).map(|(_, l)| *l)
}

fn matches(name: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| match p.strip_suffix('*') {
        Some(prefix) => name.starts_with(prefix),
        None => name == *p,
    })
}

pub fn describe(pkg: &str) -> String {
    match layer(pkg) {
        Some(l) => format!("L{l}"),
        None if EXEMPT.contains(&pkg) => "exempt".into(),
        None => "UNKNOWN".into(),
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DepKind {
    Normal,
    Dev,
    Build,
}

#[derive(Debug, Clone)]
pub struct Dep {
    pub name: String,
    pub kind: DepKind,
    /// `true` if the dependency is a workspace member (a path dependency).
    pub workspace: bool,
}

#[derive(Debug, Clone)]
pub struct Crate {
    pub name: String,
    pub deps: Vec<Dep>,
}

/// Every rule violation, as a sorted, de-duplicated list of messages.
pub fn check(crates: &[Crate]) -> Vec<String> {
    let mut out = Vec::new();
    for c in crates {
        if EXEMPT.contains(&c.name.as_str()) {
            continue;
        }
        let Some(from) = layer(&c.name) else {
            out.push(format!("{}: unknown workspace crate; register it in xtask/src/layers.rs TABLE", c.name));
            continue;
        };
        for d in c.deps.iter().filter(|d| d.name != c.name) {
            let kind = format!("{:?}", d.kind).to_lowercase();
            if d.workspace {
                match layer(&d.name) {
                    Some(to) if to >= from => {
                        out.push(format!("{} (L{from}) -> {} (L{to}) [{kind}]: may only depend on strictly lower layers", c.name, d.name));
                    }
                    Some(_) => {}
                    None => out.push(format!("{} (L{from}) -> {} [{kind}]: libraries may not depend on apps or tooling", c.name, d.name)),
                }
            } else if from < UI_MIN_LAYER && matches(&d.name, UI_CRATES) {
                out.push(format!("{} (L{from}) -> `{}` [{kind}]: UI toolkits are only allowed in ui-egui and the apps", c.name, d.name));
            } else if matches(&d.name, GPU_CRATES) && !dir_of(&c.name).is_some_and(|dir| GPU_ALLOWED.contains(&dir)) {
                out.push(format!("{} (L{from}) -> `{}` [{kind}]: wgpu is only allowed in render, ui-egui and the apps", c.name, d.name));
            }
        }
    }
    out.sort();
    out.dedup();
    out
}

/// Build the model from `cargo metadata --format-version 1 --no-deps`.
pub fn from_metadata(meta: &Value) -> Result<Vec<Crate>, String> {
    let pkgs = meta["packages"].as_array().ok_or("cargo metadata: no packages array")?;
    let members: Vec<&str> = pkgs.iter().filter_map(|p| p["name"].as_str()).collect();
    let mut out = Vec::new();
    for p in pkgs {
        let name = p["name"].as_str().ok_or("cargo metadata: package without a name")?.to_owned();
        let deps = p["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .map(|d| {
                let name = d["name"].as_str().unwrap_or_default().to_owned();
                let kind = match d["kind"].as_str() {
                    Some("dev") => DepKind::Dev,
                    Some("build") => DepKind::Build,
                    _ => DepKind::Normal,
                };
                let workspace = members.contains(&name.as_str());
                Dep { name, kind, workspace }
            })
            .collect();
        out.push(Crate { name, deps });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use DepKind::*;

    fn c(name: &str, deps: &[(&str, DepKind, bool)]) -> Crate {
        Crate { name: name.into(), deps: deps.iter().map(|(n, k, w)| Dep { name: (*n).into(), kind: *k, workspace: *w }).collect() }
    }

    #[test]
    fn downward_graph_passes() {
        let g = [
            c("texturecraft-core", &[("serde", Normal, false)]),
            c("texturecraft-compose", &[("texturecraft-core", Normal, true), ("texturecraft-texel", Normal, true)]),
            c("texturecraft-engine", &[("texturecraft-paint", Normal, true), ("texturecraft-geom", Dev, true)]),
            c("texturecraft-render", &[("texturecraft-compose", Normal, true), ("wgpu", Normal, false)]),
            c("texturecraft-ui-egui", &[("texturecraft-automation", Normal, true), ("egui", Normal, false), ("wgpu", Normal, false)]),
            c("texturecraft", &[("eframe", Normal, false), ("texturecraft-ui-egui", Normal, true), ("winresource", Build, false)]),
        ];
        assert!(check(&g).is_empty(), "{:?}", check(&g));
    }

    #[test]
    fn same_or_higher_layer_flagged() {
        assert_eq!(check(&[c("texturecraft-core", &[("texturecraft-geom", Normal, true)])]).len(), 1);
        assert_eq!(check(&[c("texturecraft-compose", &[("texturecraft-paint", Dev, true)])]).len(), 1);
        assert_eq!(check(&[c("texturecraft-paint", &[("texturecraft-formats", Build, true)])]).len(), 1);
        assert_eq!(check(&[c("texturecraft-engine", &[("texturecraft", Normal, true)])]).len(), 1);
    }

    #[test]
    fn ui_only_from_ui_egui() {
        for dep in ["egui", "egui_extras", "egui-wgpu", "eframe", "winit", "rfd"] {
            assert_eq!(check(&[c("texturecraft-automation", &[(dep, Normal, false)])]).len(), 1, "{dep}");
        }
        assert!(check(&[c("texturecraft-automation", &[("eguilike", Normal, false)])]).is_empty());
    }

    #[test]
    fn wgpu_only_in_render_and_ui() {
        assert_eq!(check(&[c("texturecraft-engine", &[("wgpu", Normal, false)])]).len(), 1);
        assert_eq!(check(&[c("texturecraft-bake", &[("wgpu-types", Normal, false)])]).len(), 1);
        assert!(check(&[c("texturecraft-render", &[("wgpu", Normal, false)])]).is_empty());
    }

    #[test]
    fn unknown_crate_flagged_and_exempt_skipped() {
        assert_eq!(check(&[c("texturecraft-mystery", &[])]).len(), 1);
        assert!(check(&[c("xtask", &[("texturecraft-ui-egui", Normal, true)])]).is_empty());
        assert!(check(&[c("texturecraft-core", &[("texturecraft-core", Dev, true)])]).is_empty());
    }

    #[test]
    fn metadata_parsing() {
        let meta: Value = serde_json::from_str(
            r#"{"packages":[
                {"name":"texturecraft-compose","dependencies":[
                    {"name":"texturecraft-core","kind":null,"path":"/x/crates/core"},
                    {"name":"rayon","kind":null},
                    {"name":"proptest","kind":"dev"}]},
                {"name":"texturecraft-core","dependencies":[]}
            ]}"#,
        )
        .unwrap();
        let g = from_metadata(&meta).unwrap();
        let compose = g.iter().find(|c| c.name == "texturecraft-compose").unwrap();
        assert!(compose.deps[0].workspace && !compose.deps[1].workspace);
        assert_eq!(compose.deps[2].kind, Dev);
        assert!(check(&g).is_empty());
    }

    #[test]
    fn external_path_libraries_are_not_workspace_members() {
        let meta = serde_json::json!({"packages": [
            {"name":"texturecraft-core", "dependencies":[]},
            {"name":"texturecraft-render", "dependencies":[
                {"name":"texturecraft-core", "kind":null, "path":"/fixture/app/crates/core"},
                {"name":"craft-stereo-camera", "kind":null, "path":"/fixture/shared/crates/stereo-camera"}
            ]}
        ]});
        let graph = from_metadata(&meta).unwrap();
        assert!(check(&graph).is_empty(), "a sibling shared library is external, not an unknown app crate");
        let renderer = graph.iter().find(|c| c.name == "texturecraft-render").unwrap();
        assert!(renderer.deps[0].workspace);
        assert!(!renderer.deps[1].workspace);
        let mut forbidden = meta;
        forbidden["packages"][1]["dependencies"][1]["name"] = serde_json::json!("egui");
        assert!(!check(&from_metadata(&forbidden).unwrap()).is_empty(), "an external path must still obey the UI-toolkit ban");
    }
}
