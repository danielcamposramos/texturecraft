//! Text: an image resource drawn from text in an installed font (or a built-in one), to stamp
//! with the brush as its alpha or to place as a decal.

use serde_json::{Value, json};
use texturecraft_paint::text::render_text;

use crate::commands::{CommandSpec, always, bad, color_param, f32p, strp};
use crate::{Result, Session, cmd};

pub fn specs() -> Vec<CommandSpec> {
    vec![
        cmd!("text.fonts", "List fonts", [], None, "{} — the installed font families with their styles, and the built-in ones", always, |_, _| Ok(fonts())),
        cmd!(
            "resource.text",
            "Add text",
            [],
            None,
            "{text (lines split at \\n), font?: family (text.fonts; default a common sans-serif), style?: e.g. Bold (default Regular), size?: 8..1024 texels per em (default 256), color?: [r,g,b] linear or \"#rrggbb\" (default white), name?} — draws the text as an image resource with an alpha (white for tool.brush {alpha: {image}}, colored for layer.add.decal)",
            always,
            add_text
        ),
    ]
}

/// Fonts bundled with the app (egui's), used when nothing is installed or in the browser.
const BUILT_IN: [(&str, &[u8]); 2] = [("Ubuntu Light", epaint_default_fonts::UBUNTU_LIGHT), ("Hack", epaint_default_fonts::HACK_REGULAR)];
/// Families tried, in order, when no font is asked for.
#[cfg(not(target_arch = "wasm32"))]
const DEFAULT_FAMILIES: [&str; 6] = ["Arial", "Segoe UI", "Helvetica", "Liberation Sans", "DejaVu Sans", "Noto Sans"];
/// Largest font file read.
#[cfg(not(target_arch = "wasm32"))]
const MAX_FONT_BYTES: u64 = 128 << 20;

#[cfg(not(target_arch = "wasm32"))]
fn installed() -> &'static [texturecraft_formats::fonts::FontFace] {
    static FONTS: std::sync::OnceLock<Vec<texturecraft_formats::fonts::FontFace>> = std::sync::OnceLock::new();
    FONTS.get_or_init(texturecraft_formats::fonts::system_fonts)
}

fn fonts() -> Value {
    #[cfg(not(target_arch = "wasm32"))]
    let families = {
        let mut out: Vec<(String, Vec<String>)> = Vec::new();
        for f in installed() {
            match out.last_mut() {
                Some((family, styles)) if *family == f.family => styles.push(f.style.clone()),
                _ => out.push((f.family.clone(), vec![f.style.clone()])),
            }
        }
        out
    };
    #[cfg(target_arch = "wasm32")]
    let families: Vec<(String, Vec<String>)> = Vec::new();
    json!({
        "installed": families.into_iter().map(|(family, styles)| json!({"family": family, "styles": styles})).collect::<Vec<_>>(),
        "builtIn": BUILT_IN.map(|(name, _)| name),
    })
}

/// The bytes and face index of `family` in `style` (a regular face when the style is missing),
/// or the first default family installed, else the built-in font.
fn font_bytes(family: Option<&str>, style: Option<&str>) -> Result<(std::borrow::Cow<'static, [u8]>, u32)> {
    let is = |a: &str, b: &str| a.eq_ignore_ascii_case(b);
    if let Some((_, bytes)) = BUILT_IN.iter().find(|(name, _)| family.is_some_and(|f| is(f, name))) {
        return Ok((std::borrow::Cow::Borrowed(bytes), 0));
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let faces = installed();
        let pick = |family: &str| {
            let of: Vec<_> = faces.iter().filter(|f| is(&f.family, family)).collect();
            let regular = ["Regular", "Normal", "Book", "Roman"];
            let found = style.and_then(|s| of.iter().find(|f| is(&f.style, s)));
            found.or_else(|| of.iter().find(|f| regular.iter().any(|r| is(&f.style, r)))).or(of.first()).copied()
        };
        let face = match family {
            Some(f) => Some(pick(f).ok_or_else(|| bad("resource.text", format!("no font family `{f}` (see text.fonts)")))?),
            None => DEFAULT_FAMILIES.iter().find_map(|f| pick(f)),
        };
        if let Some(face) = face {
            let size = std::fs::metadata(&face.path).map(|m| m.len()).unwrap_or(u64::MAX);
            if size > MAX_FONT_BYTES {
                return Err(bad("resource.text", format!("{} is too large", face.path.display())));
            }
            let bytes = std::fs::read(&face.path).map_err(|e| bad("resource.text", format!("cannot read {}: {e}", face.path.display())))?;
            return Ok((std::borrow::Cow::Owned(bytes), face.index));
        }
    }
    #[cfg(target_arch = "wasm32")]
    let _ = style;
    match family {
        Some(f) => Err(bad("resource.text", format!("no font family `{f}` (see text.fonts)"))),
        None => Ok((std::borrow::Cow::Borrowed(BUILT_IN[0].1), 0)),
    }
}

fn add_text(s: &mut Session, p: &Value) -> Result<Value> {
    let cmd = "resource.text";
    let text = strp(p, "text").ok_or_else(|| bad(cmd, "`text` is required"))?;
    let (font, index) = font_bytes(strp(p, "font"), strp(p, "style"))?;
    let size = f32p(p, "size").unwrap_or(256.0).clamp(8.0, 1024.0);
    let img = render_text(&font, index, text, size).map_err(|e| bad(cmd, e))?;
    let rgb = color_param(p, "color").unwrap_or([1.0; 3]).map(|c| (texturecraft_texel::linear_to_srgb(c.clamp(0.0, 1.0)) * 255.0).round() as u8);
    let png = texturecraft_formats::coverage_png(img.width, img.height, &img.coverage, rgb).map_err(|e| bad(cmd, e.to_string()))?;
    // The resource is named after the text (its first line, shortened) unless a name is given.
    let label: String = strp(p, "name").unwrap_or_else(|| text.lines().next().unwrap_or_default()).chars().take(32).collect();
    let name = s.import_resource_bytes(&format!("Text {}.png", label.trim()), png)?;
    Ok(json!({ "name": name, "width": img.width, "height": img.height }))
}
