//! CJK and Arabic fallback fonts for the UI. egui's default fonts have no Japanese, Chinese,
//! Korean or Arabic glyphs, so translated UIs (and such layer or file names) need more fonts.
//! They come from:
//!
//! 1. craft-fonts, embedded at build time when `CRAFT_FONTS_DIR` points at a checkout (optional;
//!    see `build.rs` and storytold/craft-fonts' `docs/integration.md`);
//! 2. on desktop, the operating system's fonts (Windows, macOS, Linux and BSD paths).
//!
//! Fonts load lazily: one script when the UI language needs it, all of them while the language
//! picker shows every language's own name. They are appended after the app's fonts
//! (`theme::font_definitions`), so Latin text and row heights keep the interface font; the current language's script goes first, so
//! shared Han characters take that language's glyph forms. egui shapes Arabic and orders
//! right-to-left runs itself. In the browser only craft-fonts can supply them (without it, the
//! language shows its English name in the picker).

use std::sync::Arc;

use egui::{FontData, FontFamily};

/// A font from the optional craft-fonts build input (empty unless built with `CRAFT_FONTS_DIR`).
pub struct CraftFont {
    pub family: &'static str,
    pub style: &'static str,
    /// ISO 15924 scripts the font is for, e.g. `"Jpan"`.
    pub scripts: &'static [&'static str],
    pub bytes: &'static [u8],
}

include!(concat!(env!("OUT_DIR"), "/craft_fonts.rs"));

/// Skip absurdly large files (a corrupt or non-font path must not eat memory).
#[cfg(not(target_arch = "wasm32"))]
const MAX_FONT_BYTES: u64 = 128 << 20;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Script {
    Japanese,
    Chinese,
    Korean,
    Arabic,
}

impl Script {
    pub const ALL: [Script; 4] = [Script::Japanese, Script::Chinese, Script::Korean, Script::Arabic];

    /// The script a UI language needs beyond Latin.
    pub fn of_language(code: &str) -> Option<Script> {
        match code {
            "ja" => Some(Script::Japanese),
            "zh-Hans" | "zh-Hant" => Some(Script::Chinese),
            "ko" => Some(Script::Korean),
            "ar" => Some(Script::Arabic),
            _ => None,
        }
    }

    /// ISO 15924 code (craft-fonts' manifest).
    fn iso(self) -> &'static str {
        match self {
            Script::Japanese => "Jpan",
            Script::Chinese => "Hans",
            Script::Korean => "Kore",
            Script::Arabic => "Arab",
        }
    }

    /// Candidate system font files: (path, face index in a collection).
    #[cfg(not(target_arch = "wasm32"))]
    fn system_files(self) -> Vec<(std::path::PathBuf, u32)> {
        use std::path::PathBuf;
        let mut v: Vec<(PathBuf, u32)> = Vec::new();
        if cfg!(windows) {
            let fonts = PathBuf::from(std::env::var("WINDIR").unwrap_or_else(|_| "C:\\Windows".into())).join("Fonts");
            let files: &[&str] = match self {
                Script::Japanese => &["YuGothR.ttc", "meiryo.ttc", "msgothic.ttc"],
                Script::Chinese => &["msyh.ttc", "simsun.ttc", "simhei.ttf"],
                Script::Korean => &["malgun.ttf", "gulim.ttc"],
                Script::Arabic => &["segoeui.ttf", "tahoma.ttf", "arial.ttf"],
            };
            v.extend(files.iter().map(|f| (fonts.join(f), 0)));
        } else if cfg!(target_os = "macos") {
            let files: &[&str] = match self {
                Script::Japanese => &["/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc", "/System/Library/Fonts/Hiragino Sans GB.ttc"],
                Script::Chinese => {
                    &["/System/Library/Fonts/Hiragino Sans GB.ttc", "/System/Library/Fonts/STHeiti Light.ttc", "/System/Library/Fonts/PingFang.ttc"]
                }
                Script::Korean => &["/System/Library/Fonts/AppleSDGothicNeo.ttc"],
                Script::Arabic => &["/System/Library/Fonts/SFArabic.ttf", "/System/Library/Fonts/Supplemental/GeezaPro.ttc"],
            };
            v.extend(files.iter().map(|f| (PathBuf::from(f), 0)));
            v.push((PathBuf::from("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"), 0));
        } else if self == Script::Arabic {
            for f in [
                "/usr/share/fonts/truetype/noto/NotoSansArabic-Regular.ttf",
                "/usr/share/fonts/noto/NotoSansArabic-Regular.ttf",
                "/usr/share/fonts/google-noto/NotoSansArabic-Regular.ttf",
                "/usr/local/share/fonts/noto/NotoSansArabic-Regular.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/dejavu/DejaVuSans.ttf",
                "/usr/local/share/fonts/dejavu/DejaVuSans.ttf",
            ] {
                v.push((PathBuf::from(f), 0));
            }
        } else {
            // Noto Sans CJK collections hold JP, KR, SC, TC and HK faces, in that order.
            let face = match self {
                Script::Japanese | Script::Arabic => 0,
                Script::Korean => 1,
                Script::Chinese => 2,
            };
            for dir in ["/usr/share/fonts/opentype/noto", "/usr/share/fonts/noto-cjk", "/usr/share/fonts/google-noto-cjk", "/usr/local/share/fonts/noto"] {
                v.push((PathBuf::from(dir).join("NotoSansCJK-Regular.ttc"), face));
            }
            if self == Script::Korean {
                v.push((PathBuf::from("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"), 0));
            }
            for f in [
                "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
                "/usr/share/fonts/wenquanyi/wqy-microhei/wqy-microhei.ttc",
                "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
            ] {
                v.push((PathBuf::from(f), 0));
            }
        }
        v
    }
}

/// The fallback fonts installed in the egui context, and the ones loaded so far.
#[derive(Default)]
pub struct UiFonts {
    /// Scripts installed, in priority order.
    installed: Vec<Script>,
    /// Font data per script (`None`: nothing found), loaded once.
    loaded: Vec<(Script, Option<Arc<FontData>>)>,
}

impl UiFonts {
    /// Install the fonts `language` needs, or every script's when `all` (the language picker is
    /// showing). Fonts stay once loaded; the language's script moves to the front.
    pub fn update(&mut self, ctx: &egui::Context, language: &str, all: bool) {
        let first = Script::of_language(language);
        if first.is_none_or(|f| self.installed.first() == Some(&f)) && (!all || self.installed.len() == Script::ALL.len()) {
            return;
        }
        let mut want: Vec<Script> = first.into_iter().collect();
        for s in Script::ALL {
            if (all || self.installed.contains(&s)) && !want.contains(&s) {
                want.push(s);
            }
        }
        if want == self.installed {
            return;
        }
        let mut defs = crate::theme::font_definitions();
        for s in &want {
            let Some(data) = self.data(*s) else { continue };
            let name = format!("script-{}", s.iso());
            defs.font_data.insert(name.clone(), data);
            for family in [FontFamily::Proportional, FontFamily::Monospace, FontFamily::Name(crate::theme::SEMIBOLD.into())] {
                defs.families.entry(family).or_default().push(name.clone());
            }
        }
        ctx.set_fonts(defs);
        self.installed = want;
    }

    /// Whether some installed font draws `script` (the picker falls back to English names).
    pub fn covers(&self, script: Script) -> bool {
        self.installed.contains(&script) && self.loaded.iter().any(|(s, d)| *s == script && d.is_some())
    }

    fn data(&mut self, script: Script) -> Option<Arc<FontData>> {
        if let Some((_, d)) = self.loaded.iter().find(|(s, _)| *s == script) {
            return d.clone();
        }
        let d = load(script).map(Arc::new);
        match &d {
            Some(_) => log::info!("UI font fallback for {}", script.iso()),
            None => log::warn!("no font found for {} text; it shows as boxes", script.iso()),
        }
        self.loaded.push((script, d.clone()));
        d
    }
}

/// The first available font for `script`: craft-fonts' (embedded), then the system's.
fn load(script: Script) -> Option<FontData> {
    if let Some(f) = CRAFT_FONTS.iter().find(|f| f.scripts.contains(&script.iso()) && f.style == "Regular" && !f.bytes.is_empty()) {
        return Some(aligned(FontData::from_static(f.bytes)));
    }
    #[cfg(not(target_arch = "wasm32"))]
    for (path, index) in script.system_files() {
        let Ok(meta) = std::fs::metadata(&path) else { continue };
        if !meta.is_file() || meta.len() == 0 || meta.len() > MAX_FONT_BYTES {
            continue;
        }
        if let Ok(bytes) = std::fs::read(&path) {
            let mut data = FontData::from_owned(bytes);
            data.index = index;
            return Some(aligned(data));
        }
    }
    None
}

/// `data` shifted so its baseline meets the UI font's. egui centers a fallback face's line box
/// in the primary font's, so faces with taller boxes (most CJK fonts) would sit high.
fn aligned(mut data: FontData) -> FontData {
    let defs = crate::theme::font_definitions();
    let primary = defs.families.get(&FontFamily::Proportional).and_then(|f| f.first()).and_then(|n| defs.font_data.get(n));
    if let (Some((pa, pd, pg)), Some((ca, cd, cg))) =
        (primary.and_then(|p| line_metrics(p.font.as_ref().as_ref(), p.index)), line_metrics(data.font.as_ref().as_ref(), data.index))
    {
        // Baseline of a fallback glyph: its ascent plus half the difference of the line boxes.
        let drift = ca + 0.5 * ((pa - pd + pg) - (ca - cd + cg)) - pa;
        if drift.is_finite() && drift.abs() < 1.0 {
            data.tweak.y_offset_factor = -drift;
        }
    }
    data
}

/// (ascent, descent, line gap) of face `index` in em units, chosen as egui's font reader does:
/// OS/2 typographic metrics when the font asks for them, else `hhea` (then OS/2 when `hhea` is
/// zero). `descent` is negative.
fn line_metrics(font: &[u8], index: u32) -> Option<(f32, f32, f32)> {
    let u16_at = |o: usize| font.get(o..o.checked_add(2)?).map(|b| u16::from_be_bytes([b[0], b[1]]));
    let u32_at = |o: usize| font.get(o..o.checked_add(4)?).map(|b| u32::from_be_bytes([b[0], b[1], b[2], b[3]]));
    let i16_at = |o: usize| u16_at(o).map(|v| v as i16 as f32);
    let dir = if font.get(..4)? == b"ttcf" { u32_at(12usize.checked_add(4usize.checked_mul(index as usize)?)?)? as usize } else { 0 };
    let tables = u16_at(dir.checked_add(4)?)? as usize;
    let table = |tag: &[u8; 4]| {
        (0..tables).find_map(|i| {
            let rec = dir.checked_add(12)?.checked_add(i.checked_mul(16)?)?;
            (font.get(rec..rec.checked_add(4)?)? == tag).then(|| u32_at(rec + 8).map(|o| o as usize)).flatten()
        })
    };
    let upem = f32::from(u16_at(table(b"head")?.checked_add(18)?)?.max(1));
    let os2 = table(b"OS/2");
    let typo = |o: usize| Some((i16_at(o + 68)?, i16_at(o + 70)?, i16_at(o + 72)?));
    let use_typo = os2.and_then(|o| u16_at(o + 62)).is_some_and(|sel| sel & (1 << 7) != 0);
    let (a, d, g) = match (use_typo, os2) {
        (true, Some(o)) => typo(o)?,
        _ => {
            let h = table(b"hhea")?;
            let m = (i16_at(h + 4)?, i16_at(h + 6)?, i16_at(h + 8)?);
            if m.0 == 0.0 && m.1 == 0.0 { os2.and_then(typo).unwrap_or(m) } else { m }
        }
    };
    Some((a / upem, d / upem, g / upem))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn line_metrics_read_the_default_font_and_reject_garbage() {
        let defs = egui::FontDefinitions::default();
        let ubuntu = defs.font_data.get("Ubuntu-Light").unwrap();
        let (a, d, g) = line_metrics(ubuntu.font.as_ref().as_ref(), 0).unwrap();
        assert!(a > 0.5 && a < 1.5 && d < 0.0 && d > -0.6 && g >= 0.0, "{a} {d} {g}");
        // The primary font (Inter with craft-fonts, else egui's) needs no shift against itself.
        let ui = crate::theme::font_definitions();
        let primary = ui.families.get(&FontFamily::Proportional).and_then(|f| f.first()).and_then(|n| ui.font_data.get(n)).unwrap();
        assert_eq!(aligned(FontData::from_owned(primary.font.as_ref().as_ref().to_vec())).tweak.y_offset_factor, 0.0);
        for junk in [&b""[..], b"ttcf", b"\0\x01\0\0\0\x09", &[0xff; 64]] {
            assert!(line_metrics(junk, 0).is_none());
            assert!(line_metrics(junk, 7).is_none());
        }
    }

    #[test]
    fn languages_map_to_scripts() {
        assert_eq!(Script::of_language("ja"), Some(Script::Japanese));
        assert_eq!(Script::of_language("zh-Hans"), Some(Script::Chinese));
        assert_eq!(Script::of_language("ko"), Some(Script::Korean));
        assert_eq!(Script::of_language("ar"), Some(Script::Arabic));
        assert_eq!(Script::of_language("de"), None);
        for l in &crate::i18n::LANGUAGES {
            // Past the Latin blocks and their diacritics.
            let other_script = l.name.chars().any(|c| c > '\u{036F}');
            assert_eq!(other_script, Script::of_language(l.code).is_some(), "{}", l.code);
        }
    }

    #[test]
    fn fonts_install_once_and_keep_the_language_script_first() {
        let ctx = egui::Context::default();
        let mut f = UiFonts::default();
        f.update(&ctx, "en", false);
        assert!(f.installed.is_empty() && f.loaded.is_empty(), "English loads nothing");
        f.update(&ctx, "ko", false);
        assert_eq!(f.installed, [Script::Korean]);
        f.update(&ctx, "ja", true);
        assert_eq!(f.installed, Script::ALL);
        // Closing the picker keeps what is loaded; each script was read at most once.
        f.update(&ctx, "ja", false);
        assert_eq!(f.installed.len(), Script::ALL.len());
        assert_eq!(f.loaded.len(), Script::ALL.len());
    }
}
