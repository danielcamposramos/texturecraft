//! Layers, masks, effects and fill sources.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

use crate::{BlendMode, Channel, ChannelKind, lum};

mod curvature_detail;
mod curvature_mode;

pub use curvature_detail::CurvatureDetail;
pub use curvature_mode::CurvatureMode;

/// Id of a layer or effect (one id space per project).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LayerId(pub u64);

/// Blend mode and opacity of one channel of a layer. A channel missing from
/// [`Layer::channels`] is disabled for that layer.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ChannelBlend {
    pub mode: BlendMode,
    pub opacity: f32,
}

impl Default for ChannelBlend {
    fn default() -> Self {
        ChannelBlend { mode: BlendMode::Normal, opacity: 1.0 }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub kind: LayerKind,
    pub channels: BTreeMap<Channel, ChannelBlend>,
    #[serde(default)]
    pub mask: Option<Mask>,
    /// Effects applied to the layer's content, bottom → top.
    #[serde(default)]
    pub content_effects: Vec<Effect>,
    /// Color tag (0..8), shown as a stripe on the layer row.
    #[serde(default)]
    pub color_tag: Option<u8>,
    /// Combine the mask with the layer's height ("mask with height").
    #[serde(default)]
    pub height_mask: HeightMask,
    /// Geometry mask: the mesh parts (submesh names) the layer shows on; None shows it on all.
    #[serde(default)]
    pub geometry_mask: Option<Vec<String>>,
}

/// Mask with height: partial mask values follow the layer's own height, so a transition
/// reveals the high parts first. With mask `m` and the layer's height `h` (−1..1):
/// `m' = saturate(m + contrast · 4m(1 − m) · (h + offset))`. Fully hidden (0) and fully shown
/// (1) texels are unchanged, as is everything where the layer has no height.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct HeightMask {
    pub enabled: bool,
    /// How strongly the height pushes the mask (0 = plain mask), 0..64.
    pub contrast: f32,
    /// Added to the height before it pushes the mask, −1..1.
    pub offset: f32,
}

impl Default for HeightMask {
    fn default() -> Self {
        HeightMask { enabled: false, contrast: 1.0, offset: 0.0 }
    }
}

impl HeightMask {
    /// The mask `m` combined with height `h` (see [`HeightMask`]).
    pub fn apply(&self, m: f32, h: f32) -> f32 {
        if !self.enabled || !h.is_finite() {
            return m;
        }
        (m + self.contrast * 4.0 * m * (1.0 - m) * (h + self.offset)).clamp(0.0, 1.0)
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerKind {
    /// Painted pixels, stored per channel by the engine under the layer id.
    Paint,
    /// A procedural or uniform material applied everywhere (shaped by the mask).
    Fill(Fill),
    /// A folder of layers, bottom → top.
    Group { children: Vec<LayerId>, collapsed: bool },
    /// A linked instance of `source` (possibly in another texture set): renders the source's
    /// content, content effects and channel blending, and its mask unless the instance has its
    /// own. Edits to the source show in every instance.
    Instance { source: LayerId },
}

impl LayerKind {
    pub fn name(&self) -> &'static str {
        match self {
            LayerKind::Paint => "paint",
            LayerKind::Fill(_) => "fill",
            LayerKind::Group { .. } => "group",
            LayerKind::Instance { .. } => "instance",
        }
    }
}

impl Layer {
    /// A layer with every channel of `channels` enabled at Normal/100%.
    pub fn new(id: LayerId, name: impl Into<String>, kind: LayerKind, channels: &[Channel]) -> Layer {
        Layer {
            id,
            name: name.into(),
            visible: true,
            kind,
            channels: channels.iter().map(|c| (*c, ChannelBlend::default())).collect(),
            mask: None,
            content_effects: Vec::new(),
            color_tag: None,
            height_mask: HeightMask::default(),
            geometry_mask: None,
        }
    }

    pub fn is_group(&self) -> bool {
        matches!(self.kind, LayerKind::Group { .. })
    }

    pub fn children(&self) -> &[LayerId] {
        match &self.kind {
            LayerKind::Group { children, .. } => children,
            _ => &[],
        }
    }

    pub fn blend(&self, c: Channel) -> Option<ChannelBlend> {
        self.channels.get(&c).copied()
    }

    /// The layer this one instances, if it is an instance.
    pub fn instance_source(&self) -> Option<LayerId> {
        match self.kind {
            LayerKind::Instance { source } => Some(source),
            _ => None,
        }
    }
}

/// Where a texture or procedural is sampled from on the surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Projection {
    /// Texture coordinates of the mesh.
    Uv { scale: [f32; 2], rotation: f32, offset: [f32; 2] },
    /// World-space box projection blended by the normal.
    Triplanar { scale: f32, sharpness: f32, offset: [f32; 3] },
    /// World-space planar projection along an axis (0 = X, 1 = Y, 2 = Z).
    Planar { scale: f32, axis: u8, offset: [f32; 3] },
    /// Spherical projection around the mesh center.
    Spherical { scale: f32 },
    /// Cylindrical projection around an axis.
    Cylindrical { scale: f32, axis: u8 },
    /// A geometry decal: the image projected from a frame placed on the surface (outside the
    /// frame, behind it or beyond its depth the fill does not cover).
    Decal(Decal),
}

/// The frame a decal projects from (world space).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Decal {
    /// The surface point the decal was placed on.
    pub center: [f32; 3],
    /// The surface normal there: the decal projects along it.
    pub normal: [f32; 3],
    /// Up direction of the unrotated frame.
    pub up: [f32; 3],
    /// Width and height (world units).
    pub size: [f32; 2],
    /// Rotation around the normal (degrees).
    pub rotation: f32,
    /// Shift of the frame from `center` along the unrotated right and up directions (world units).
    pub offset: [f32; 2],
    /// How far in front of and behind the frame the decal reaches (world units).
    pub depth: f32,
}

impl Default for Projection {
    fn default() -> Self {
        Projection::Uv { scale: [1.0, 1.0], rotation: 0.0, offset: [0.0, 0.0] }
    }
}

impl Projection {
    pub fn label(&self) -> &'static str {
        match self {
            Projection::Uv { .. } => "UV projection",
            Projection::Triplanar { .. } => "Tri-planar projection",
            Projection::Planar { .. } => "Planar projection",
            Projection::Spherical { .. } => "Spherical projection",
            Projection::Cylindrical { .. } => "Cylindrical projection",
            Projection::Decal(_) => "Decal projection",
        }
    }
}

/// Procedural patterns available to fills, generators and masks.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProceduralKind {
    /// Smooth fractal value noise.
    Clouds,
    /// Gradient (Perlin-style) noise.
    Perlin,
    /// Fine high-frequency noise.
    Noise,
    /// Worley / cellular noise.
    Cells,
    /// Blotchy grunge: thresholded multi-scale noise.
    Grunge,
    /// Thin random scratches.
    Scratches,
    /// Checkerboard.
    Checker,
    /// Stripes.
    Stripes,
    /// Linear gradient along U (or the projection axis).
    Gradient,
    /// Wood grain: growth rings with fibers along X.
    Wood,
    /// Running-bond bricks with mortar joints (0 = mortar).
    Bricks,
    /// Square tiles with grout joints (0 = grout).
    Tiles,
    /// Plain fabric weave.
    Weave,
    /// 2/2 twill: diagonal ribs, the warp brighter than the weft (denim, carbon fiber).
    Twill,
    /// Leather grain: pebbled cells with creases.
    Leather,
    /// Hammered metal dimples.
    Hammered,
    /// Fine brushed-metal lines along X.
    Brushed,
    /// Marble veins.
    Marble,
    /// Flat random tone per Voronoi cell (crystals, spangle, stones).
    Voronoi,
    /// Vertical streaks and leaks (along the projection's up axis).
    Streaks,
    /// Fingerprints and smudges.
    Smudges,
    /// Small round pores and pits (0 = pore).
    Pores,
    /// Sharp-edged paint chips.
    Chips,
}

impl ProceduralKind {
    pub const ALL: [ProceduralKind; 23] = [
        ProceduralKind::Clouds,
        ProceduralKind::Perlin,
        ProceduralKind::Noise,
        ProceduralKind::Cells,
        ProceduralKind::Grunge,
        ProceduralKind::Scratches,
        ProceduralKind::Checker,
        ProceduralKind::Stripes,
        ProceduralKind::Gradient,
        ProceduralKind::Wood,
        ProceduralKind::Bricks,
        ProceduralKind::Tiles,
        ProceduralKind::Weave,
        ProceduralKind::Twill,
        ProceduralKind::Leather,
        ProceduralKind::Hammered,
        ProceduralKind::Brushed,
        ProceduralKind::Marble,
        ProceduralKind::Voronoi,
        ProceduralKind::Streaks,
        ProceduralKind::Smudges,
        ProceduralKind::Pores,
        ProceduralKind::Chips,
    ];
    pub fn label(self) -> &'static str {
        match self {
            ProceduralKind::Clouds => "Clouds",
            ProceduralKind::Perlin => "Perlin noise",
            ProceduralKind::Noise => "Fine noise",
            ProceduralKind::Cells => "Cells",
            ProceduralKind::Grunge => "Grunge",
            ProceduralKind::Scratches => "Scratches",
            ProceduralKind::Checker => "Checker",
            ProceduralKind::Stripes => "Stripes",
            ProceduralKind::Gradient => "Gradient",
            ProceduralKind::Wood => "Wood grain",
            ProceduralKind::Bricks => "Bricks",
            ProceduralKind::Tiles => "Tiles",
            ProceduralKind::Weave => "Fabric weave",
            ProceduralKind::Twill => "Twill weave",
            ProceduralKind::Leather => "Leather grain",
            ProceduralKind::Hammered => "Hammered",
            ProceduralKind::Brushed => "Brushed lines",
            ProceduralKind::Marble => "Marble veins",
            ProceduralKind::Voronoi => "Voronoi",
            ProceduralKind::Streaks => "Streaks",
            ProceduralKind::Smudges => "Fingerprints",
            ProceduralKind::Pores => "Pores",
            ProceduralKind::Chips => "Paint chips",
        }
    }
    pub fn parse(s: &str) -> Option<ProceduralKind> {
        let norm = |x: &str| x.to_ascii_lowercase().replace([' ', '_', '-'], "");
        ProceduralKind::ALL.into_iter().find(|k| norm(k.label()) == norm(s) || norm(&format!("{k:?}")) == norm(s))
    }
}

/// A procedural grayscale pattern. Its output maps 0..1 between `low` and `high` values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Procedural {
    pub kind: ProceduralKind,
    /// Pattern frequency (repeats per projection unit).
    pub scale: f32,
    pub seed: u32,
    /// Contrast around 0.5 (1 = unchanged).
    pub contrast: f32,
    /// Shifts the pattern brighter (+) or darker (−).
    pub balance: f32,
    pub invert: bool,
}

impl Procedural {
    pub fn new(kind: ProceduralKind) -> Procedural {
        Procedural { kind, scale: 4.0, seed: 1, contrast: 1.0, balance: 0.0, invert: false }
    }
}

/// What a fill writes into one channel.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Source {
    /// A constant value (linear RGB for colors; `[v, v, v]` for scalars).
    Uniform([f32; 3]),
    /// A procedural pattern mixing two values.
    Procedural { pattern: Procedural, low: [f32; 3], high: [f32; 3] },
    /// An imported image resource, by name. `normal_format` is the convention of a normal map
    /// image (DirectX images have their green flipped where a normal channel samples them).
    Image {
        name: String,
        #[serde(default)]
        normal_format: crate::NormalFormat,
    },
    /// The stack value recorded by an anchor point effect (`anchor` is the effect id) for
    /// `channel`; colors read as gray where a scalar is needed.
    Anchor { anchor: LayerId, channel: Channel },
}

impl Source {
    pub fn uniform_of(c: Channel) -> Source {
        Source::Uniform(c.default_value())
    }

    /// Image resource `name` (an OpenGL normal map where a normal channel samples it).
    pub fn image(name: impl Into<String>) -> Source {
        Source::Image { name: name.into(), normal_format: crate::NormalFormat::OpenGl }
    }

    /// The anchor point this source reads, if any.
    pub fn anchor(&self) -> Option<LayerId> {
        match self {
            Source::Anchor { anchor, .. } => Some(*anchor),
            _ => None,
        }
    }
}

/// A fill layer's material: one source per channel, plus how it is projected.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, Default)]
pub struct Fill {
    pub sources: BTreeMap<Channel, Source>,
    pub projection: Projection,
    /// Set when the sources are maps cooked from a MaterialCraft material.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub link: Option<Box<MaterialLink>>,
}

/// A MaterialCraft material behind a fill (from a `.mcmat` portable material): its channel
/// sources are image resources cooked from the graph, and changing a parameter re-cooks them
/// through `materialcraft-cli`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialLink {
    pub name: String,
    /// The image resource holding the material's package (`.mcraft`).
    pub package: String,
    /// The package's graph the maps come from.
    pub graph: String,
    pub params: Vec<LinkParam>,
    #[serde(default)]
    pub presets: Vec<LinkPreset>,
    /// Image inputs of the graph (identifiers such as `curvature`), fed from baked mesh maps.
    #[serde(default)]
    pub inputs: Vec<String>,
    /// Size the maps were cooked at (pixels across).
    pub size: u32,
}

/// An exposed input of a linked material. Values keep MaterialCraft's typed form,
/// `{"t": "float" | "float2" | "float3" | "float4" | "int" | "int2" | "bool" | "color" | "enum", "v": …}`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkParam {
    pub id: String,
    pub label: String,
    #[serde(default)]
    pub group: String,
    pub value: serde_json::Value,
    pub default: serde_json::Value,
    #[serde(default)]
    pub min: f32,
    #[serde(default = "one")]
    pub max: f32,
    /// Labels of an `enum` value.
    #[serde(default)]
    pub options: Vec<String>,
}

fn one() -> f32 {
    1.0
}

/// A named set of linked-material values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkPreset {
    pub name: String,
    pub values: BTreeMap<String, serde_json::Value>,
}

impl Fill {
    /// Value used for `c` when the fill has no source for it: the channel default.
    pub fn source(&self, c: Channel) -> Source {
        self.sources.get(&c).cloned().unwrap_or_else(|| Source::uniform_of(c))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum MaskBackground {
    #[default]
    White,
    Black,
}

/// A layer mask: a background color with a stack of effects on top.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Mask {
    pub enabled: bool,
    pub background: MaskBackground,
    /// Effects bottom → top. Paint effects hold bitmap data keyed by their id.
    pub effects: Vec<Effect>,
}

impl Mask {
    pub fn new(background: MaskBackground) -> Mask {
        Mask { enabled: true, background, effects: Vec::new() }
    }
}

/// Levels remap: input range, gamma, output range.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Levels {
    pub in_low: f32,
    pub in_high: f32,
    pub gamma: f32,
    pub out_low: f32,
    pub out_high: f32,
}

impl Default for Levels {
    fn default() -> Self {
        Levels { in_low: 0.0, in_high: 1.0, gamma: 1.0, out_low: 0.0, out_high: 1.0 }
    }
}

impl Levels {
    pub fn apply(&self, x: f32) -> f32 {
        let t = ((x - self.in_low) / (self.in_high - self.in_low).max(1e-5)).clamp(0.0, 1.0);
        let t = t.powf(1.0 / self.gamma.max(1e-3));
        self.out_low + (self.out_high - self.out_low) * t
    }
}

/// Mesh-map driven mask generators (the "smart mask" building blocks).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GeneratorKind {
    /// Convex edges from the curvature map.
    EdgeWear,
    /// Crevices from ambient occlusion.
    Dirt,
    /// Thin parts from the thickness map.
    ThinAreas,
    /// Raw curvature.
    Curvature,
    /// Raw ambient occlusion.
    AmbientOcclusion,
    /// Gradient along a world axis (bottom → top by default).
    Position,
    /// Surfaces facing a direction (e.g. up for dust).
    Direction,
}

impl GeneratorKind {
    pub const ALL: [GeneratorKind; 7] = [
        GeneratorKind::EdgeWear,
        GeneratorKind::Dirt,
        GeneratorKind::ThinAreas,
        GeneratorKind::Curvature,
        GeneratorKind::AmbientOcclusion,
        GeneratorKind::Position,
        GeneratorKind::Direction,
    ];
    pub fn label(self) -> &'static str {
        match self {
            GeneratorKind::EdgeWear => "Mask Editor: Edge wear",
            GeneratorKind::Dirt => "Dirt",
            GeneratorKind::ThinAreas => "Thin areas",
            GeneratorKind::Curvature => "Curvature",
            GeneratorKind::AmbientOcclusion => "Ambient occlusion",
            GeneratorKind::Position => "Position gradient",
            GeneratorKind::Direction => "Light / direction",
        }
    }
    pub fn parse(s: &str) -> Option<GeneratorKind> {
        let norm = |x: &str| x.to_ascii_lowercase().replace([' ', '_', '-', ':'], "");
        GeneratorKind::ALL.into_iter().find(|k| norm(&format!("{k:?}")) == norm(s) || norm(k.label()) == norm(s))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Generator {
    pub kind: GeneratorKind,
    /// Where the mask starts (0..1, generator specific threshold).
    pub level: f32,
    /// Edge softness (0 = hard).
    pub softness: f32,
    /// Breaks the result up with world-space noise (0 = none).
    pub noise_amount: f32,
    pub noise_scale: f32,
    pub seed: u32,
    /// Direction for `Position` / `Direction` generators (world space).
    pub direction: [f32; 3],
    pub invert: bool,
    /// Mode and detail sizes of `Curvature` / `EdgeWear` generators (None: the curvature map as baked, as before detail sizes existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub detail: Option<CurvatureDetail>,
    /// Painted height read from an anchor point below, added as micro detail (None: baked maps only).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub micro: Option<MicroDetail>,
}

/// A generator's micro detail: the height an anchor point records, so the generated mask
/// follows painted bumps and dents (raised height reads as an edge, recessed as a cavity).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MicroDetail {
    /// The anchor point effect id.
    pub anchor: LayerId,
    /// How strongly the height moves the signal (0..1).
    pub amount: f32,
}

impl MicroDetail {
    pub const DEFAULT_AMOUNT: f32 = 0.5;
}

impl Generator {
    pub fn new(kind: GeneratorKind) -> Generator {
        let (level, softness, noise_amount) = match kind {
            GeneratorKind::EdgeWear => (0.5, 0.25, 0.3),
            GeneratorKind::Curvature => (0.5, 0.5, 0.0),
            GeneratorKind::Dirt => (0.55, 0.25, 0.35),
            GeneratorKind::ThinAreas => (0.3, 0.2, 0.0),
            _ => (0.5, 0.25, 0.0),
        };
        let detail = match kind {
            GeneratorKind::Curvature => Some(CurvatureDetail::default()),
            GeneratorKind::EdgeWear => Some(CurvatureDetail { mode: CurvatureMode::Edges, weights: CurvatureDetail::EDGE_WEIGHTS }),
            _ => None,
        };
        Generator { kind, level, softness, noise_amount, noise_scale: 8.0, seed: 7, direction: [0.0, 1.0, 0.0], invert: false, detail, micro: None }
    }
}

/// Filters usable on masks and content.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Filter {
    Blur {
        radius: f32,
    },
    Sharpen {
        amount: f32,
    },
    Invert,
    Threshold {
        level: f32,
    },
    /// Hue/saturation/lightness shift (color channels; lightness only on grays and masks).
    /// `hue` is in turns, `saturation` and `lightness` in −1..1; all zero = unchanged.
    Hsl {
        hue: f32,
        saturation: f32,
        lightness: f32,
    },
    /// Contrast around mid gray and a luminosity shift, both −1..1 (0 = unchanged).
    ContrastLuminosity {
        contrast: f32,
        luminosity: f32,
    },
    /// Color offsets (−1..1 per component) weighted towards the shadows, midtones and highlights.
    ColorBalance {
        shadows: [f32; 3],
        midtones: [f32; 3],
        highlights: [f32; 3],
    },
    /// Luminance mapped through evenly spaced colors (linear RGB; scalars read their gray).
    GradientMap {
        colors: Vec<[f32; 3]>,
    },
    /// Quantize to `levels` steps (2..64).
    Posterize {
        levels: f32,
    },
    /// Keep values within `min..max`.
    Clamp {
        min: f32,
        max: f32,
    },
    /// A ramp slid through the values: those above `position` go towards white, those below
    /// towards black, over a width `contrast` (0..1) narrows to a hard step.
    HistogramScan {
        position: f32,
        contrast: f32,
    },
    /// The values squeezed towards `position`: `range` 1 keeps them, 0 sets them all to it.
    HistogramRange {
        range: f32,
        position: f32,
    },
    /// The values offset by `offset` (−1..1), wrapping around past white and black.
    HistogramShift {
        offset: f32,
    },
    /// Shift each texel's lookup along a noise vector field: `intensity` in texels at 2K,
    /// `scale` noise features per UV unit.
    Warp {
        intensity: f32,
        scale: f32,
        seed: u32,
    },
    /// Blur along one direction: `radius` in texels at 2K, `angle` in degrees.
    DirectionalBlur {
        radius: f32,
        angle: f32,
    },
    /// Smear values downhill along a noise height field ("slope blur"): `radius` in texels at
    /// 2K, `scale` noise features per UV unit.
    SlopeBlur {
        radius: f32,
        scale: f32,
        seed: u32,
    },
    /// The detail finer than `radius` (texels at 2K) around mid gray: the value minus its blur.
    HighPass {
        radius: f32,
    },
    /// Gradient magnitude across `width` texels at 2K: edges light, flat areas black.
    EdgeDetect {
        width: f32,
    },
    /// Like [`Filter::Warp`], but every texel moves along one direction (`angle` in degrees).
    DirectionalWarp {
        intensity: f32,
        angle: f32,
        scale: f32,
        seed: u32,
    },
    /// Colors to their luminance (scalars and masks unchanged).
    Grayscale,
    /// Flip the green component of normal maps (OpenGL ↔ DirectX); other channels unchanged.
    NormalInvert,
    /// Shapes rounded off at their edges: values ramp up from 0 at a shape's edge to their own
    /// level over about `width` texels at 2K (a mask's or height's bevel).
    Bevel {
        width: f32,
    },
    /// Square cells `size` texels at 2K across, each the value at its center.
    Mosaic {
        size: f32,
    },
    /// Adds a blur of the values (`radius` texels at 2K) times `amount`: light bleeds around
    /// bright areas.
    Glow {
        radius: f32,
        amount: f32,
    },
    /// Lighting painted into the base color: `ambient` (times the baked ambient occlusion by
    /// `occlusion`) plus a directional light of strength `light` from `azimuth` around +Y
    /// (degrees from +Z) and `elevation` above the horizon (degrees), on the mesh's normals.
    BakedLighting {
        azimuth: f32,
        elevation: f32,
        light: f32,
        ambient: f32,
        occlusion: f32,
    },
    /// Colors near `from` (within `tolerance`, fading out over `softness`, as RGB distance
    /// 0..1) shifted by `to − from`, so their shading and texture stay (linear RGB).
    ReplaceColor {
        from: [f32; 3],
        to: [f32; 3],
        tolerance: f32,
        softness: f32,
    },
}

/// Push `x` (0..1) towards 1 (`amount` > 0) or 0 (`amount` < 0); `amount` in −1..1.
fn shift_unit(x: f32, amount: f32) -> f32 {
    let a = amount.clamp(-1.0, 1.0);
    if a >= 0.0 { x + (1.0 - x) * a } else { x * (1.0 + a) }
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = t.rem_euclid(1.0);
    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 0.5 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}

/// HSL adjustment of a 0..1 color (see [`Filter::Hsl`]). A gray stays gray and only its
/// lightness changes.
pub fn hsl_adjust(rgb: [f32; 3], hue: f32, saturation: f32, lightness: f32) -> [f32; 3] {
    let [r, g, b] = rgb.map(|v| if v.is_finite() { v.clamp(0.0, 1.0) } else { 0.0 });
    let (mx, mn) = (r.max(g).max(b), r.min(g).min(b));
    let (d, l) = (mx - mn, (mx + mn) * 0.5);
    let (h, s) = if d <= 1e-6 {
        (0.0, 0.0)
    } else {
        let s = if l > 0.5 { d / (2.0 - mx - mn) } else { d / (mx + mn) };
        let h = if mx == r {
            ((g - b) / d).rem_euclid(6.0)
        } else if mx == g {
            (b - r) / d + 2.0
        } else {
            (r - g) / d + 4.0
        };
        (h / 6.0, s)
    };
    let h = h + if hue.is_finite() { hue } else { 0.0 };
    let (s, l) = (if d <= 1e-6 { 0.0 } else { shift_unit(s, saturation) }, shift_unit(l, lightness));
    if s <= 0.0 {
        return [l; 3];
    }
    let q = if l < 0.5 { l * (1.0 + s) } else { l + s - l * s };
    let p = 2.0 * l - q;
    [hue_to_rgb(p, q, h + 1.0 / 3.0), hue_to_rgb(p, q, h), hue_to_rgb(p, q, h - 1.0 / 3.0)]
}

/// Apply a 0..1 function to each component of a channel value: signed values map through
/// 0..1, normals are left alone.
pub fn per_unit(kind: ChannelKind, v: [f32; 3], f: impl Fn(f32) -> f32) -> [f32; 3] {
    match kind {
        ChannelKind::NormalMap => v,
        // One-component channels hold their value in the first slot (read as gray).
        ChannelKind::Signed => [f(v[0] * 0.5 + 0.5) * 2.0 - 1.0; 3],
        ChannelKind::Scalar => [f(v[0]); 3],
        ChannelKind::Color => v.map(f),
    }
}

/// The color at `t` (0..1) of a gradient through evenly spaced `colors` (None when empty).
pub fn gradient(colors: &[[f32; 3]], t: f32) -> Option<[f32; 3]> {
    let last = colors.len().checked_sub(1)?;
    let pos = if t.is_finite() { t.clamp(0.0, 1.0) * last as f32 } else { 0.0 };
    let i = (pos.floor() as usize).min(last);
    let (a, b) = (colors.get(i)?, colors.get((i + 1).min(last))?);
    let f = pos - i as f32;
    Some(std::array::from_fn(|k| a[k] + (b[k] - a[k]) * f))
}

impl Filter {
    /// Whether the filter reads neighbouring texels (the compositor runs it over a region).
    pub fn is_neighbourhood(&self) -> bool {
        matches!(
            self,
            Filter::Blur { .. }
                | Filter::Sharpen { .. }
                | Filter::Warp { .. }
                | Filter::DirectionalBlur { .. }
                | Filter::SlopeBlur { .. }
                | Filter::HighPass { .. }
                | Filter::EdgeDetect { .. }
                | Filter::DirectionalWarp { .. }
                | Filter::Bevel { .. }
                | Filter::Mosaic { .. }
                | Filter::Glow { .. }
        )
    }

    /// The filter applied to one value of a channel of `kind` (masks are scalars). None for
    /// neighbourhood filters and where it changes nothing (an empty gradient).
    pub fn point(&self, kind: ChannelKind, v: [f32; 3]) -> Option<[f32; 3]> {
        let unit = |x: f32| if x.is_finite() { x.clamp(0.0, 1.0) } else { 0.0 };
        Some(match self {
            Filter::Invert => per_unit(kind, v, |x| 1.0 - x),
            Filter::Threshold { level } => per_unit(kind, v, |x| if x >= *level { 1.0 } else { 0.0 }),
            Filter::Hsl { hue, saturation, lightness } => match kind {
                ChannelKind::Color => hsl_adjust(v, *hue, *saturation, *lightness),
                _ => per_unit(kind, v, |x| hsl_adjust([x; 3], 0.0, 0.0, *lightness)[0]),
            },
            Filter::ContrastLuminosity { contrast, luminosity } => {
                let k = 1.0 + contrast.clamp(-1.0, 1.0);
                per_unit(kind, v, |x| unit((x - 0.5) * k + 0.5 + luminosity))
            }
            Filter::ColorBalance { shadows, midtones, highlights } => {
                // Smooth weights summing to 1: shadows fade out and highlights in with lightness.
                let weights = |l: f32| {
                    let l = unit(l);
                    ((1.0 - l) * (1.0 - l), 2.0 * l * (1.0 - l), l * l)
                };
                let shift = |k: usize, l: f32| {
                    let (s, m, h) = weights(l);
                    shadows.get(k).copied().unwrap_or(0.0) * s + midtones.get(k).copied().unwrap_or(0.0) * m + highlights.get(k).copied().unwrap_or(0.0) * h
                };
                match kind {
                    ChannelKind::Color => {
                        let l = lum(v);
                        std::array::from_fn(|k| unit(v[k] + shift(k, l)))
                    }
                    _ => per_unit(kind, v, |x| unit(x + (shift(0, x) + shift(1, x) + shift(2, x)) / 3.0)),
                }
            }
            Filter::GradientMap { colors } => match kind {
                ChannelKind::Color => gradient(colors, lum(v))?,
                _ => per_unit(kind, v, |x| gradient(colors, x).map_or(x, lum)),
            },
            Filter::Posterize { levels } => {
                let steps = (levels.clamp(2.0, 64.0).round() - 1.0).max(1.0);
                per_unit(kind, v, |x| (unit(x) * steps).round() / steps)
            }
            Filter::Clamp { min, max } => {
                let (lo, hi) = (min.min(*max), min.max(*max));
                per_unit(kind, v, |x| x.max(lo).min(hi))
            }
            // Full contrast is a threshold: the position itself goes to white.
            Filter::HistogramScan { position, contrast } if *contrast >= 1.0 => per_unit(kind, v, |x| if x >= *position { 1.0 } else { 0.0 }),
            Filter::HistogramScan { position, contrast } => {
                let width = (1.0 - contrast.clamp(0.0, 1.0)).max(1e-3);
                per_unit(kind, v, |x| unit((x - position) / width + 0.5))
            }
            Filter::HistogramRange { range, position } => per_unit(kind, v, |x| unit(position + (x - position) * range)),
            Filter::HistogramShift { offset } => per_unit(kind, v, |x| {
                // Wrapped by hand: `rem_euclid` would also send white (1 + 0) to black.
                let y = unit(x) + offset.clamp(-1.0, 1.0);
                unit(if y > 1.0 {
                    y - 1.0
                } else if y < 0.0 {
                    y + 1.0
                } else {
                    y
                })
            }),
            Filter::Grayscale => match kind {
                ChannelKind::Color => [lum(v); 3],
                _ => v,
            },
            Filter::NormalInvert => match kind {
                ChannelKind::NormalMap => [v[0], 1.0 - v[1], v[2]],
                _ => v,
            },
            Filter::ReplaceColor { from, to, tolerance, softness } => match kind {
                ChannelKind::Color => {
                    let d = ((v[0] - from[0]).powi(2) + (v[1] - from[1]).powi(2) + (v[2] - from[2]).powi(2)).sqrt() / 3f32.sqrt();
                    let (t, s) = (tolerance.clamp(0.0, 1.0), softness.clamp(0.0, 1.0));
                    let w = if d <= t {
                        1.0
                    } else if d >= t + s {
                        0.0
                    } else {
                        let x = 1.0 - (d - t) / s.max(1e-6);
                        x * x * (3.0 - 2.0 * x)
                    };
                    std::array::from_fn(|k| unit(v[k] + (to[k] - from[k]) * w))
                }
                _ => v,
            },
            Filter::Blur { .. }
            | Filter::Sharpen { .. }
            | Filter::Warp { .. }
            | Filter::DirectionalBlur { .. }
            | Filter::SlopeBlur { .. }
            | Filter::HighPass { .. }
            | Filter::EdgeDetect { .. }
            | Filter::DirectionalWarp { .. }
            | Filter::Bevel { .. }
            | Filter::Mosaic { .. }
            | Filter::Glow { .. }
            | Filter::BakedLighting { .. } => return None,
        })
    }

    pub fn label(&self) -> &'static str {
        match self {
            Filter::Blur { .. } => "Blur",
            Filter::Sharpen { .. } => "Sharpen",
            Filter::Invert => "Invert",
            Filter::Threshold { .. } => "Threshold",
            Filter::Hsl { .. } => "HSL Perceptive",
            Filter::ContrastLuminosity { .. } => "Contrast/Luminosity",
            Filter::ColorBalance { .. } => "Color Balance",
            Filter::GradientMap { .. } => "Gradient Map",
            Filter::Posterize { .. } => "Posterize",
            Filter::Clamp { .. } => "Clamp",
            Filter::HistogramScan { .. } => "Histogram Scan",
            Filter::HistogramRange { .. } => "Histogram Range",
            Filter::HistogramShift { .. } => "Histogram Shift",
            Filter::Warp { .. } => "Warp",
            Filter::DirectionalBlur { .. } => "Blur Directional",
            Filter::SlopeBlur { .. } => "Slope Blur",
            Filter::HighPass { .. } => "High Pass",
            Filter::EdgeDetect { .. } => "Edge Detect",
            Filter::DirectionalWarp { .. } => "Warp Directional",
            Filter::Grayscale => "Grayscale",
            Filter::NormalInvert => "Normal Invert",
            Filter::Bevel { .. } => "Bevel",
            Filter::Mosaic { .. } => "Mosaic",
            Filter::Glow { .. } => "Glow",
            Filter::BakedLighting { .. } => "Baked Lighting",
            Filter::ReplaceColor { .. } => "Replace Color",
        }
    }

    /// Baked lighting's factor on the base color of a surface facing `nrm` (world space) with
    /// ambient occlusion `ao` (None for other filters).
    pub fn baked_light(&self, nrm: [f32; 3], ao: f32) -> Option<f32> {
        let Filter::BakedLighting { azimuth, elevation, light, ambient, occlusion } = self else { return None };
        let (sa, ca) = azimuth.to_radians().sin_cos();
        let (se, ce) = elevation.to_radians().sin_cos();
        let facing = (nrm[0] * ce * sa + nrm[1] * se + nrm[2] * ce * ca).max(0.0);
        let ao = if ao.is_finite() { ao.clamp(0.0, 1.0) } else { 1.0 };
        let k = ambient * (1.0 + (ao - 1.0) * occlusion.clamp(0.0, 1.0)) + light * facing;
        Some(if k.is_finite() { k.max(0.0) } else { 1.0 })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EffectKind {
    /// Painted grayscale (mask) or material (content) pixels keyed by the effect id.
    Paint,
    /// A path sublayer: pixels keyed by the effect id like [`EffectKind::Paint`], but painted
    /// only from its stored paths and repainted whenever one is added, edited or removed.
    Path(Vec<PathShape>),
    /// A uniform or procedural grayscale fill.
    Fill {
        source: Source,
        projection: Projection,
    },
    Generator(Generator),
    Levels(Levels),
    Filter(Filter),
    /// Select ID-map colors within a tolerance.
    ColorSelection {
        colors: Vec<[f32; 3]>,
        tolerance: f32,
    },
    /// Compare the mask with a constant.
    Compare {
        op: CompareOp,
        value: f32,
        tolerance: f32,
    },
    /// Content only: a material fill (one source per channel; channels without a source are
    /// left as they are).
    Material(Fill),
    /// Records the stack at its place so masks, fills and generators above can read it
    /// ([`Source::Anchor`]). In a content stack it records the composited stack right after its
    /// layer (that layer and everything below); in a mask, the mask value at that point (every
    /// channel reads that gray).
    Anchor,
}

/// One path of a path sublayer: its control points and how it was painted.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PathShape {
    /// Surface positions (3D view) or texture coordinates with z = 0 (2D view).
    pub points: Vec<[f32; 3]>,
    pub uv: bool,
    pub closed: bool,
    /// A closed path fills its interior instead of stroking the curve.
    pub fill: bool,
    /// Viewport size (pixels) the path was painted in.
    pub viewport: [f32; 2],
    /// Brush, material and camera the engine paints the path with (owned by the engine).
    pub paint: serde_json::Value,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompareOp {
    LesserThan,
    WithinTolerance,
    GreaterThan,
}

impl EffectKind {
    pub fn name(&self) -> &'static str {
        match self {
            EffectKind::Paint => "Paint",
            EffectKind::Path(_) => "Path",
            EffectKind::Fill { .. } => "Fill",
            EffectKind::Generator(_) => "Generator",
            EffectKind::Levels(_) => "Levels",
            EffectKind::Filter(f) => f.label(),
            EffectKind::ColorSelection { .. } => "Color selection",
            EffectKind::Compare { .. } => "Compare mask",
            EffectKind::Material(_) => "Fill",
            EffectKind::Anchor => "Anchor point",
        }
    }

    /// Whether the effect belongs in a layer's content stack (`true`), its mask (`false`), or
    /// both (`None`).
    pub fn content_only(&self) -> Option<bool> {
        match self {
            EffectKind::Material(_) => Some(true),
            EffectKind::Fill { .. } | EffectKind::Generator(_) | EffectKind::ColorSelection { .. } | EffectKind::Compare { .. } => Some(false),
            EffectKind::Paint | EffectKind::Path(_) | EffectKind::Levels(_) | EffectKind::Filter(_) | EffectKind::Anchor => None,
        }
    }

    /// Holds painted pixels keyed by the effect id (paint and path sublayers).
    pub fn painted(&self) -> bool {
        matches!(self, EffectKind::Paint | EffectKind::Path(_))
    }

    /// Anchor points read by this effect's sources.
    fn reads_anchor(&self) -> bool {
        match self {
            EffectKind::Fill { source, .. } => source.anchor().is_some(),
            EffectKind::Material(f) => f.sources.values().any(|s| s.anchor().is_some()),
            EffectKind::Generator(g) => g.micro.is_some(),
            _ => false,
        }
    }
}

impl Layer {
    /// Whether any source of the layer (fill, mask or content effects) reads an anchor point.
    pub fn reads_anchor(&self) -> bool {
        let fill = match &self.kind {
            LayerKind::Fill(f) => f.sources.values().any(|s| s.anchor().is_some()),
            _ => false,
        };
        fill || self.mask.iter().flat_map(|m| m.effects.iter()).chain(self.content_effects.iter()).any(|e| e.kind.reads_anchor())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Effect {
    pub id: LayerId,
    pub name: String,
    pub visible: bool,
    pub kind: EffectKind,
    pub blend: BlendMode,
    pub opacity: f32,
}

impl Effect {
    pub fn new(id: LayerId, kind: EffectKind) -> Effect {
        // Generators and fills that build a mask usually add on top of the background.
        Effect { id, name: kind.name().to_string(), visible: true, kind, blend: BlendMode::Normal, opacity: 1.0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn curvature_generators_saved_before_detail_sizes_load_without_them() {
        let mut saved = serde_json::to_value(Generator::new(GeneratorKind::Curvature)).unwrap();
        assert!(saved.get("detail").is_some(), "new curvature generators have detail sizes");

        saved.as_object_mut().unwrap().remove("detail");
        let loaded: Generator = serde_json::from_value(saved).unwrap();
        assert_eq!(loaded.detail, None);
        assert_eq!(Generator::new(GeneratorKind::Dirt).detail, None, "only curvature and edge wear generators have detail sizes");
    }

    #[test]
    fn levels_map_range_and_gamma() {
        let l = Levels { in_low: 0.2, in_high: 0.6, gamma: 1.0, out_low: 0.0, out_high: 1.0 };
        assert!((l.apply(0.4) - 0.5).abs() < 1e-6);
        assert_eq!(l.apply(0.0), 0.0);
        assert_eq!(l.apply(1.0), 1.0);
        let g = Levels { gamma: 2.0, ..Levels::default() };
        assert!((g.apply(0.25) - 0.5).abs() < 1e-6);
        // Degenerate ranges never divide by zero.
        let d = Levels { in_low: 0.5, in_high: 0.5, ..Levels::default() };
        assert!(d.apply(0.7).is_finite());
    }

    #[test]
    fn hsl_adjust_shifts_hue_saturation_and_lightness() {
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4);
        let c = [0.8, 0.3, 0.1];
        assert!(close(hsl_adjust(c, 0.0, 0.0, 0.0), c), "identity");
        assert!(close(hsl_adjust([1.0, 0.0, 0.0], 1.0 / 3.0, 0.0, 0.0), [0.0, 1.0, 0.0]), "red → green");
        assert!(close(hsl_adjust([1.0, 0.0, 0.0], -1.0 / 3.0, 0.0, 0.0), [0.0, 0.0, 1.0]), "red → blue");
        let gray = hsl_adjust(c, 0.0, -1.0, 0.0);
        assert!((gray[0] - gray[1]).abs() < 1e-6 && (gray[1] - gray[2]).abs() < 1e-6);
        // Grays only change lightness, whatever the hue and saturation.
        assert!(close(hsl_adjust([0.4; 3], 0.3, 1.0, 0.5), [0.7; 3]));
        assert!(close(hsl_adjust([0.4; 3], 0.0, 0.0, -0.5), [0.2; 3]));
        assert!(hsl_adjust([f32::NAN, 2.0, -1.0], f32::NAN, 5.0, -5.0).iter().all(|v| v.is_finite()));
    }

    #[test]
    fn point_filters_map_values_and_skip_neighbourhood_filters() {
        let c = ChannelKind::Color;
        let close = |a: [f32; 3], b: [f32; 3]| (0..3).all(|i| (a[i] - b[i]).abs() < 1e-4);
        let cl = Filter::ContrastLuminosity { contrast: 1.0, luminosity: 0.0 };
        assert!(close(cl.point(c, [0.25, 0.5, 0.75]).unwrap(), [0.0, 0.5, 1.0]));
        assert!(close(Filter::ContrastLuminosity { contrast: 0.0, luminosity: 0.2 }.point(ChannelKind::Scalar, [0.5; 3]).unwrap(), [0.7; 3]));
        let gm = Filter::GradientMap { colors: vec![[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]] };
        assert!(close(gm.point(c, [0.0; 3]).unwrap(), [1.0, 0.0, 0.0]));
        assert!(close(gm.point(c, [1.0; 3]).unwrap(), [0.0, 0.0, 1.0]));
        assert_eq!(Filter::GradientMap { colors: vec![] }.point(c, [0.3; 3]), None, "an empty gradient changes nothing");
        assert!(close(Filter::Posterize { levels: 2.0 }.point(c, [0.2, 0.6, 0.9]).unwrap(), [0.0, 1.0, 1.0]));
        assert!(close(Filter::Clamp { min: 0.8, max: 0.3 }.point(c, [0.1, 0.5, 0.9]).unwrap(), [0.3, 0.5, 0.8]), "swapped bounds never panic");
        let scan = |position, contrast| Filter::HistogramScan { position, contrast }.point(c, [0.2, 0.5, 0.9]).unwrap();
        assert!(close(scan(0.5, 0.0), [0.2, 0.5, 0.9]), "a full-width ramp at the middle keeps the values");
        assert!(close(scan(0.4, 1.0), [0.0, 1.0, 1.0]), "full contrast is a step at the position");
        assert!(close(scan(0.5, 1.0), [0.0, 1.0, 1.0]), "the position itself goes to white");
        assert!(close(scan(0.7, 0.5), [0.0, 0.1, 0.9]));
        let range = |range, position| Filter::HistogramRange { range, position }.point(c, [0.2, 0.5, 0.9]).unwrap();
        assert!(close(range(1.0, 0.3), [0.2, 0.5, 0.9]) && close(range(0.0, 0.3), [0.3; 3]));
        assert!(close(range(0.5, 0.5), [0.35, 0.5, 0.7]));
        let shift = |offset| Filter::HistogramShift { offset }.point(c, [0.0, 0.5, 1.0]).unwrap();
        assert!(close(shift(0.0), [0.0, 0.5, 1.0]), "no offset keeps white white");
        assert!(close(shift(0.25), [0.25, 0.75, 0.25]) && close(shift(-0.25), [0.75, 0.25, 0.75]));
        let cb = Filter::ColorBalance { shadows: [0.2, 0.0, 0.0], midtones: [0.0; 3], highlights: [0.0, 0.0, 0.2] };
        assert!(cb.point(c, [0.05; 3]).unwrap()[0] > 0.2 && cb.point(c, [0.95; 3]).unwrap()[2] > 0.99);
        assert!(close(Filter::Invert.point(ChannelKind::Signed, [0.5; 3]).unwrap(), [-0.5; 3]));
        assert_eq!(Filter::Invert.point(ChannelKind::NormalMap, [0.5, 0.5, 1.0]), Some([0.5, 0.5, 1.0]));
        for f in [Filter::Blur { radius: 1.0 }, Filter::Warp { intensity: 1.0, scale: 1.0, seed: 0 }, Filter::DirectionalBlur { radius: 1.0, angle: 0.0 }] {
            assert!(f.is_neighbourhood() && f.point(c, [0.5; 3]).is_none());
        }
        let nan = Filter::ContrastLuminosity { contrast: f32::NAN, luminosity: f32::NAN };
        assert!(nan.point(c, [0.5; 3]).unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn effect_targets_and_anchor_reads() {
        assert_eq!(EffectKind::Anchor.content_only(), None, "anchors go on content and masks");
        assert_eq!(EffectKind::Generator(Generator::new(GeneratorKind::Dirt)).content_only(), Some(false));
        assert_eq!(EffectKind::Filter(Filter::Invert).content_only(), None);
        let mut l = Layer::new(LayerId(1), "l", LayerKind::Paint, &[Channel::BaseColor]);
        assert!(!l.reads_anchor());
        let source = Source::Anchor { anchor: LayerId(9), channel: Channel::Height };
        l.mask = Some(Mask {
            effects: vec![Effect::new(LayerId(2), EffectKind::Fill { source, projection: Projection::default() })],
            ..Mask::new(MaskBackground::Black)
        });
        assert!(l.reads_anchor());

        // A generator with micro detail reads its anchor too.
        let mut g = Layer::new(LayerId(3), "g", LayerKind::Paint, &[Channel::BaseColor]);
        let mut generator = Generator::new(GeneratorKind::Dirt);
        generator.micro = Some(MicroDetail { anchor: LayerId(9), amount: 0.5 });
        g.mask = Some(Mask { effects: vec![Effect::new(LayerId(4), EffectKind::Generator(generator))], ..Mask::new(MaskBackground::Black) });
        assert!(g.reads_anchor());
    }

    #[test]
    fn height_mask_keeps_the_ends_and_follows_height() {
        let off = HeightMask::default();
        assert_eq!(off.apply(0.3, 1.0), 0.3, "disabled");
        let hm = HeightMask { enabled: true, contrast: 1.0, offset: 0.0 };
        for h in [-1.0, 0.0, 1.0] {
            assert_eq!(hm.apply(0.0, h), 0.0);
            assert_eq!(hm.apply(1.0, h), 1.0);
        }
        assert_eq!(hm.apply(0.5, 0.0), 0.5, "no height, no change");
        assert_eq!(hm.apply(0.5, 0.5), 1.0);
        assert_eq!(hm.apply(0.5, -0.5), 0.0);
        assert!(hm.apply(0.5, 0.1) > 0.5 && hm.apply(0.5, -0.1) < 0.5);
        assert!(HeightMask { offset: 0.5, ..hm }.apply(0.5, 0.0) > 0.5);
        assert_eq!(hm.apply(0.5, f32::NAN), 0.5);
    }

    #[test]
    fn generator_kinds_parse() {
        for k in GeneratorKind::ALL {
            assert_eq!(GeneratorKind::parse(&format!("{k:?}")), Some(k));
        }
        for k in ProceduralKind::ALL {
            assert_eq!(ProceduralKind::parse(k.label()), Some(k));
        }
    }
}
