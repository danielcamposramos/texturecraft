//! UI state (serde, so the control channel can read and drive it).

use serde::{Deserialize, Serialize};
use texturecraft_engine::core::{BakedMap, Channel, NormalFormat};

use crate::docking::Dock;
use crate::keymap::Keymap;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Mode {
    #[default]
    Painting,
    Baking,
    /// Path-traced view (F10).
    Rendering,
}

/// The Alpha tab's text: what to draw and in which font (empty: the default font).
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TextUi {
    pub text: String,
    pub font: String,
    pub style: String,
    /// Installed and built-in families with their styles (`text.fonts`), read when first shown.
    #[serde(skip)]
    pub fonts: Vec<(String, Vec<String>)>,
}

/// Rendering mode settings.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct RenderUi {
    pub max_samples: u32,
    pub bounces: u32,
    pub denoise: bool,
    /// Path trace on the CPU even where the GPU tracer runs.
    pub cpu: bool,
}

impl Default for RenderUi {
    fn default() -> Self {
        // The web traces on one thread: fewer bounces keep it interactive.
        RenderUi { max_samples: 256, bounces: if cfg!(target_arch = "wasm32") { 3 } else { 6 }, denoise: true, cpu: false }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ViewLayout {
    /// 3D and 2D side by side.
    #[default]
    Split,
    Only3D,
    Only2D,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PropsTab {
    #[default]
    Brush,
    Alpha,
    Material,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Windows {
    pub export: bool,
    pub new_project: bool,
    pub about: bool,
    pub shortcuts: bool,
    #[serde(default)]
    pub settings: bool,
    /// Browser File › Export Mesh: the download format choice.
    #[serde(default)]
    pub export_mesh: bool,
    /// The start window: new, open, recent projects (the app opens it at launch).
    #[serde(skip)]
    pub start: bool,
    /// What waits on the unsaved-changes dialog: save, don't save or cancel.
    #[serde(skip)]
    pub unsaved: Option<Unsaved>,
}

/// An action that replaces or ends the project, held while the project has unsaved changes.
#[derive(Clone, Debug, PartialEq)]
pub enum Unsaved {
    Quit,
    /// A menu command (New project, Open, Close, Open sample project).
    Command(String),
    /// Open this project or mesh file.
    #[cfg(not(target_arch = "wasm32"))]
    Open(String),
    /// Open a project or mesh file's bytes (picked or dropped in the browser): its name and bytes.
    Load(String, std::sync::Arc<Vec<u8>>),
}

/// Edit › Settings.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Prefs {
    /// Interface zoom (1 = 100%).
    pub ui_scale: f32,
    /// Minutes between autosaves of a modified project (0: off; desktop).
    pub autosave_minutes: u32,
    /// UI language code (`i18n::LANGUAGES`).
    pub language: String,
    /// Viewport texture resolution: 0 full, 1 half, 2 quarter of each texture set's size (lighter
    /// on GPU memory and uploads for 4K and 8K sets; painting and export keep full resolution).
    pub viewport_texture: u8,
    /// Interface brightness (the views keep their studio slate).
    pub brightness: crate::theme::Brightness,
    /// The colour of what is on, chosen or focused.
    pub accent: crate::theme::Accent,
    /// The system draws the window's title bar instead of TextureCraft (desktop; read at startup).
    pub system_title_bar: bool,
    /// Show the start window at launch.
    pub start_window: bool,
    /// Where autosaves, smart materials, presets, templates and layouts go (desktop; read at
    /// startup). Empty or relative: `Documents/TextureCraft`.
    pub data_folder: String,
    /// Cores a project save leaves to the UI; Settings bounds this to the available logical CPUs
    /// (desktop). At least one save worker always runs.
    pub reserved_cores: u32,
    /// Maximum decompressed project size accepted on open, in GiB (at least one).
    pub project_open_limit_gib: u32,
    /// Percent of the free RAM a project save may hold at once (desktop; 1 to 100).
    pub save_ram_percent: u8,
}

impl Default for Prefs {
    fn default() -> Self {
        Prefs {
            ui_scale: 1.0,
            autosave_minutes: 10,
            language: "en".into(),
            viewport_texture: 0,
            brightness: crate::theme::Brightness::default(),
            accent: crate::theme::Accent::default(),
            system_title_bar: false,
            start_window: true,
            data_folder: String::new(),
            reserved_cores: 1,
            project_open_limit_gib: texturecraft_engine::formats::DEFAULT_PROJECT_OPEN_LIMIT_GIB,
            save_ram_percent: 50,
        }
    }
}

impl Prefs {
    /// Apply resource preferences before commands or file workers can start.
    pub fn apply_resource_limits(&self) {
        texturecraft_engine::formats::set_project_open_limit_gib(self.project_open_limit_gib);
        #[cfg(not(target_arch = "wasm32"))]
        texturecraft_engine::formats::set_save_resources(self.reserved_cores as usize, self.save_ram_percent);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NewProject {
    pub mesh: String,
    pub resolution: u32,
    pub name: String,
    /// A mesh file chosen from disk (desktop only).
    pub file: Option<String>,
    /// UV tile (UDIM) workflow: on, off, or None for when the mesh's UVs leave tile 1001.
    pub uv_tiles: Option<bool>,
    /// Project template name (built-in or the user's).
    #[serde(default = "default_template")]
    pub template: String,
    /// Name typed to save the current project's settings as a template.
    #[serde(skip)]
    pub template_name: String,
    /// Normal map format of the project and its imported normal maps; None keeps the template's
    #[serde(default)]
    pub normal_format: Option<NormalFormat>,
    /// Baked map files to import at creation, matched to sets and mesh maps by name (desktop only)
    #[serde(skip)]
    pub baked_maps: Vec<String>,
}

fn default_template() -> String {
    texturecraft_engine::template_cmds::DEFAULT_TEMPLATE.into()
}

impl Default for NewProject {
    fn default() -> Self {
        NewProject {
            mesh: "material_ball".into(),
            resolution: 2048,
            name: "Untitled".into(),
            file: None,
            uv_tiles: None,
            template: default_template(),
            template_name: String::new(),
            normal_format: None,
            baked_maps: Vec::new(),
        }
    }
}

/// The export dialog's settings (remembered between sessions).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExportUi {
    pub dir: String,
    pub preset: String,
    /// File extension override ("" = the preset's formats).
    pub format: String,
    /// Bit depth override (0 = the preset's).
    pub bit_depth: u8,
    /// none | dilation | dilationDefault | infinite
    pub padding: String,
    pub dilation: u32,
    /// Output size override (0 = each texture set's size).
    pub size: u32,
    /// "" (project), openGl or directX.
    pub normal_format: String,
    /// "" (by file format: sRGB, or linear for float files), or a colour space name.
    #[serde(default)]
    pub color_space: String,
    /// Texture sets excluded from export.
    pub skip: Vec<usize>,
    pub tab: u8,
    /// Output templates tab: the selected user preset being edited (saved whenever it is valid).
    #[serde(skip)]
    pub draft: Option<texturecraft_engine::formats::ExportPreset>,
    /// Output templates tab: the new name while renaming the selected preset.
    #[serde(skip)]
    pub renaming: Option<String>,
}

impl Default for ExportUi {
    fn default() -> Self {
        ExportUi {
            dir: default_export_dir(),
            preset: "Document channels + Normal + AO".into(),
            format: String::new(),
            bit_depth: 0,
            padding: "infinite".into(),
            dilation: 16,
            size: 0,
            normal_format: String::new(),
            color_space: String::new(),
            skip: Vec::new(),
            tab: 0,
            draft: None,
            renaming: None,
        }
    }
}

fn default_export_dir() -> String {
    #[cfg(not(target_arch = "wasm32"))]
    {
        crate::user_dir("export").display().to_string()
    }
    #[cfg(target_arch = "wasm32")]
    {
        String::new()
    }
}

/// The split layout starts with the 3D and 2D views at equal width
fn default_split_ratio() -> f32 {
    0.5
}
/// The bake layout's version: saved bake docks from an older one load as `Dock::for_baking`
const BAKE_LAYOUT_VERSION: u32 = 1;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UiState {
    pub mode: Mode,
    pub layout: ViewLayout,
    /// The 3D view's share of the viewport width in the split layout
    #[serde(default = "default_split_ratio")]
    pub split_ratio: f32,
    /// The docked, tabbed and floating panels around the viewport.
    #[serde(default)]
    pub dock: Dock,
    /// The panel open against the icon rail, if any; not saved, so the app starts with the rail closed
    #[serde(skip)]
    pub rail_panel: Option<crate::docking::Panel>,
    #[serde(default = "Dock::for_baking")]
    pub(crate) inactive_mode_dock: Dock,
    #[serde(default)]
    pub(crate) dock_is_baking: bool,
    /// The bake layout version the bake dock was saved with; 0 before versions existed.
    #[serde(default)]
    pub(crate) bake_layout_version: u32,
    /// The mesh map whose settings Mesh Map Settings shows; `None` follows the first enabled map.
    #[serde(default)]
    pub(crate) selected_baked_map: Option<BakedMap>,
    /// Window ▸ Hide UI (Tab): only the viewport.
    pub hide_ui: bool,
    pub windows: Windows,
    pub uv_zoom: f32,
    pub uv_pan: [f32; 2],
    /// Channel whose blending the layer rows show.
    pub layer_channel: Channel,
    pub shelf_filter: String,
    /// The asset shelf's open tab.
    #[serde(default)]
    pub shelf_tab: usize,
    pub props_tab: PropsTab,
    #[serde(default)]
    pub text: TextUi,
    /// Layer being renamed and the edit buffer.
    pub rename: Option<(u64, String)>,
    pub new_project: NewProject,
    /// Show the UV wireframe in the 2D view.
    pub uv_wireframe: bool,
    /// Show the mesh wireframe over the 3D view.
    #[serde(default)]
    pub wireframe_3d: bool,
    #[serde(default)]
    pub render: RenderUi,
    #[serde(default)]
    pub prefs: Prefs,
    #[serde(default)]
    pub export: ExportUi,
    /// Recently opened or saved projects (desktop).
    #[serde(default)]
    pub recent: Vec<String>,
    /// Custom keyboard shortcuts (item id → "Ctrl+Shift+E", "" = none) over the defaults.
    #[serde(default)]
    pub keymap: Keymap,
    /// The brush that paints the skew correction map in the 2D view of bake mode.
    #[serde(default)]
    pub bake_paint_brush: crate::bake_mode::BakePaintBrush,
    /// How the 3D view shows symmetry
    #[serde(default)]
    pub symmetry: crate::symmetry::SymmetryDisplay,
    /// What bake mode's 3D view draws besides the low poly (the Baking Visualization panel)
    #[serde(default)]
    pub bake_visualization: crate::bake_mode::BakeVisualizationToggles,
}

impl UiState {
    /// Replace an outdated or unreadable saved bake dock by the bake layout, then give the mode its dock
    pub(crate) fn sync_mode_dock(&mut self) {
        self.reset_outdated_bake_dock();

        let baking = self.mode == Mode::Baking;
        if baking != self.dock_is_baking {
            std::mem::swap(&mut self.dock, &mut self.inactive_mode_dock);
            self.dock_is_baking = baking;
        }
    }

    /// Replace a bake dock saved with an older bake layout, or one that could not be read, by the bake default
    fn reset_outdated_bake_dock(&mut self) {
        let bake_dock = if self.dock_is_baking { &mut self.dock } else { &mut self.inactive_mode_dock };
        if self.bake_layout_version >= BAKE_LAYOUT_VERSION && !bake_dock.replaced_unreadable() {
            return;
        }

        *bake_dock = Dock::for_baking();
        self.bake_layout_version = BAKE_LAYOUT_VERSION;
    }

    /// Take `layout`'s panels, windows and view settings, keeping this state's dialogs, export
    /// settings and recent files.
    pub fn adopt_layout(&mut self, layout: UiState) {
        let keep = std::mem::replace(self, layout);
        self.new_project = keep.new_project;
        self.export = keep.export;
        self.recent = keep.recent;
        self.shelf_filter = keep.shelf_filter;
        self.prefs = keep.prefs;
        self.keymap = keep.keymap;
        self.rename = None;
    }
}

impl Default for UiState {
    fn default() -> Self {
        UiState {
            mode: Mode::Painting,
            layout: ViewLayout::Split,
            split_ratio: default_split_ratio(),
            dock: Dock::default(),
            rail_panel: None,
            inactive_mode_dock: Dock::for_baking(),
            dock_is_baking: false,
            bake_layout_version: BAKE_LAYOUT_VERSION,
            selected_baked_map: None,
            hide_ui: false,
            windows: Windows::default(),
            uv_zoom: 1.0,
            uv_pan: [0.0, 0.0],
            layer_channel: Channel::BaseColor,
            shelf_filter: String::new(),
            shelf_tab: 0,
            props_tab: PropsTab::Brush,
            text: TextUi::default(),
            rename: None,
            new_project: NewProject::default(),
            uv_wireframe: true,
            wireframe_3d: false,
            render: RenderUi::default(),
            prefs: Prefs::default(),
            export: ExportUi::default(),
            recent: Vec::new(),
            keymap: Keymap::new(),
            bake_paint_brush: crate::bake_mode::BakePaintBrush::default(),
            symmetry: crate::symmetry::SymmetryDisplay::default(),
            bake_visualization: crate::bake_mode::BakeVisualizationToggles::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adopting_a_layout_keeps_recent_files_export_settings_and_shortcuts() {
        let mut ui = UiState { recent: vec!["a.tcraft".into()], ..UiState::default() };
        ui.export.preset = "Mine".into();
        ui.keymap.insert("file.save".into(), "F5".into());
        let mut layout = UiState::default();
        layout.dock.close(crate::docking::Panel::Assets);
        ui.adopt_layout(layout);
        assert!(!ui.dock.is_open(crate::docking::Panel::Assets));
        assert_eq!(ui.recent, vec!["a.tcraft".to_string()]);
        assert_eq!(ui.export.preset, "Mine");
        assert_eq!(ui.keymap.get("file.save").map(String::as_str), Some("F5"));
    }

    #[test]
    fn the_data_folder_is_an_absolute_choice_or_documents_texturecraft() {
        let default = crate::data_folder("");
        assert!(default.ends_with(std::path::Path::new("Documents").join("TextureCraft")));
        assert_eq!(crate::data_folder("relative/dir"), default, "a relative path is not a choice");
        let chosen = std::env::temp_dir().join("tc-data");
        assert_eq!(crate::data_folder(&format!(" {} ", chosen.display())), chosen);
        assert_eq!(Prefs::default().data_folder, "", "the default is no choice");
    }

    #[test]
    fn a_bake_dock_saved_before_the_bake_layout_version_loads_the_current_bake_layout() {
        let old_bake_dock = serde_json::json!({"version": 2, "layout": {"root": {"Split": {"axis": "Horizontal", "size": {"FixedFirst": 320.0},
            "first": {"Tabs": {"panels": ["TextureSetList", "BakeSettings", "BakeMaps"], "active": 0}},
            "second": {"Tabs": {"panels": ["Viewport"], "active": 0}}}}, "floating": []}});
        for dock_is_baking in [false, true] {
            let mut painting_dock = Dock::default();
            painting_dock.close(crate::docking::Panel::Assets);
            let (bake_field, painting_field) = if dock_is_baking { ("dock", "inactive_mode_dock") } else { ("inactive_mode_dock", "dock") };
            let mut saved = serde_json::to_value(UiState::default()).unwrap();
            let fields = saved.as_object_mut().unwrap();
            fields.remove("bake_layout_version");
            fields.insert("dock_is_baking".into(), serde_json::json!(dock_is_baking));
            fields.insert("mode".into(), serde_json::to_value(if dock_is_baking { Mode::Baking } else { Mode::Painting }).unwrap());
            fields.insert(bake_field.into(), old_bake_dock.clone());
            fields.insert(painting_field.into(), serde_json::to_value(&painting_dock).unwrap());

            let mut ui: UiState = serde_json::from_value(saved).unwrap();
            let loaded_bake_dock = if dock_is_baking { &ui.dock } else { &ui.inactive_mode_dock };
            assert!(!loaded_bake_dock.is_open(crate::docking::Panel::MeshMapBakers), "the old layout loads without the bakers list");

            ui.sync_mode_dock();
            let (bake_dock, painting) = if dock_is_baking { (&ui.dock, &ui.inactive_mode_dock) } else { (&ui.inactive_mode_dock, &ui.dock) };
            assert_eq!(*bake_dock, Dock::for_baking());
            assert_eq!(*painting, painting_dock, "the painting layout is kept");

            let mut reloaded: UiState = serde_json::from_value(serde_json::to_value(&ui).unwrap()).unwrap();
            reloaded.sync_mode_dock();
            assert_eq!(reloaded, ui, "a current layout is not reset again");
        }
    }

    #[test]
    fn the_bake_layout_survives_a_restart_whichever_mode_was_open() {
        for mode in [Mode::Painting, Mode::Baking] {
            let mut ui = UiState { mode, ..UiState::default() };
            ui.sync_mode_dock();

            let mut restarted = crate::ui_state_store::tests::restart(&ui);
            restarted.sync_mode_dock();
            let bake_dock = if mode == Mode::Baking { &restarted.dock } else { &restarted.inactive_mode_dock };
            assert_eq!(*bake_dock, Dock::for_baking(), "{mode:?}");
            assert!(bake_dock.is_open(crate::docking::Panel::MeshMapBakers), "{mode:?}");
        }
    }

    #[test]
    fn older_saved_state_without_a_dock_loads_the_default_layout() {
        let mut v = serde_json::to_value(UiState::default()).unwrap();
        let obj = v.as_object_mut().unwrap();
        obj.remove("dock");
        obj.remove("keymap");
        obj.insert("show_shelf".into(), serde_json::json!(false));
        let ui: UiState = serde_json::from_value(v).unwrap();
        assert_eq!(ui.dock, Dock::default());
    }
}
