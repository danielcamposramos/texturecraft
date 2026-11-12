//! UI text lookup over the optional shared craft-i18n catalogs.
//! Missing translations and builds without catalogs use the English source text.

use std::fmt::Display;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, OnceLock};

include!(concat!(env!("OUT_DIR"), "/craft_i18n.rs"));

static I18N: LazyLock<craft_i18n::I18n> = LazyLock::new(|| {
    let offered: Vec<_> = LANGUAGES.iter().map(|l| l.code).collect();
    let (i18n, problems) = craft_i18n::I18n::new(CRAFT_I18N, &offered);
    for problem in problems {
        log::warn!("translation catalog: {problem}");
    }
    i18n
});

static SHARED_LANGUAGES: OnceLock<[craft_i18n::Lang; COUNT]> = OnceLock::new();

/// Resolve the fixed picker codes once; hot text lookups only load the cached language handle.
fn shared_languages() -> &'static [craft_i18n::Lang; COUNT] {
    SHARED_LANGUAGES.get_or_init(|| LANGUAGES.each_ref().map(|l| I18N.from_code(l.code).unwrap_or(craft_i18n::Lang::EN)))
}

/// Whether this build embeds a catalog for a supported language.
pub fn available(code: &str) -> bool {
    I18N.from_code(code).is_some()
}

pub struct Language {
    /// BCP 47 code ("en", "pt-BR", "zh-Hans").
    pub code: &'static str,
    /// The language's own name, for the picker.
    pub name: &'static str,
    /// Its English name (the picker's fallback when no font can draw `name`).
    pub english: &'static str,
}

const COUNT: usize = 12;

pub static LANGUAGES: [Language; COUNT] = [
    Language { code: "en", name: "English", english: "English" },
    Language { code: "pl", name: "Polski", english: "Polish" },
    Language { code: "zh-Hant", name: "繁體中文", english: "Chinese (Traditional)" },
    Language { code: "fr", name: "Français", english: "French" },
    Language { code: "de", name: "Deutsch", english: "German" },
    Language { code: "es", name: "Español", english: "Spanish" },
    Language { code: "pt-BR", name: "Português (Brasil)", english: "Portuguese (Brazil)" },
    Language { code: "it", name: "Italiano", english: "Italian" },
    Language { code: "ja", name: "日本語", english: "Japanese" },
    Language { code: "zh-Hans", name: "简体中文", english: "Chinese (Simplified)" },
    Language { code: "ko", name: "한국어", english: "Korean" },
    Language { code: "ar", name: "العربية", english: "Arabic" },
];

static CURRENT: AtomicUsize = AtomicUsize::new(0);

/// Display labels belong to the UI; the model exposes only map identities
pub(crate) fn baked_map_label(map: texturecraft_engine::core::BakedMap) -> &'static str {
    use texturecraft_engine::core::BakedMap;

    match map {
        BakedMap::Normal => n_("Normal"),
        BakedMap::WorldNormal => n_("World Space Normal"),
        BakedMap::Id => n_("ID"),
        BakedMap::AmbientOcclusion => n_("Ambient Occlusion"),
        BakedMap::Curvature => n_("Curvature"),
        BakedMap::Position => n_("Position"),
        BakedMap::Thickness => n_("Thickness"),
        BakedMap::Height => n_("Height"),
        BakedMap::BentNormal => n_("Bent Normals"),
        BakedMap::Opacity => n_("Opacity"),
    }
}

/// Switch the UI language (unknown codes fall back to English).
pub fn set_language(code: &str) {
    CURRENT.store(LANGUAGES.iter().position(|l| l.code.eq_ignore_ascii_case(code) && available(l.code)).unwrap_or(0), Ordering::Relaxed);
}

/// The current UI language.
pub fn current() -> &'static Language {
    LANGUAGES.get(CURRENT.load(Ordering::Relaxed)).unwrap_or(&LANGUAGES[0])
}

/// `text` in the current language (unchanged when it has no translation).
pub fn tr(text: &str) -> &str {
    let lang = shared_languages().get(CURRENT.load(Ordering::Relaxed)).copied().unwrap_or(craft_i18n::Lang::EN);
    I18N.t(lang, text)
}

/// Marks a static string for translation where it is defined (gettext's `N_`); it is translated
/// with [`tr`] where it is shown. The key test collects both.
pub const fn n_(text: &'static str) -> &'static str {
    text
}

/// `template` translated, with each `{name}` replaced by its value in `args`.
pub fn trf(template: &str, args: &[(&str, &dyn Display)]) -> String {
    let values: Vec<_> = args.iter().map(|(name, value)| (*name, value.to_string())).collect();
    let args: Vec<_> = values.iter().map(|(name, value)| (*name, value.as_str())).collect();
    craft_i18n::fmt(tr(template), &args)
}

/// The supported language for a locale such as "de-AT", "pt_BR.UTF-8" or "zh-Hant-TW" ("en" when
/// none matches). Chinese script and territory distinguish Traditional Chinese; Portuguese uses Brazilian Portuguese.
pub fn language_for_locale(locale: &str) -> &'static str {
    let normalized = locale.to_ascii_lowercase().replace('_', "-");
    let parts: Vec<_> = normalized.split(['-', '.', '@']).collect();
    if parts.first() == Some(&"zh") && parts.iter().any(|p| matches!(*p, "hant" | "tw" | "hk" | "mo")) {
        return "zh-Hant";
    }
    let lang = locale.split(['-', '_', '.', '@']).next().unwrap_or("").to_ascii_lowercase();
    match lang.as_str() {
        "pt" => "pt-BR",
        "zh" => "zh-Hans",
        l => LANGUAGES.iter().find(|x| x.code == l).map_or("en", |x| x.code),
    }
}

/// The operating system's (or browser's) UI language, for the first run.
pub fn os_language() -> &'static str {
    #[cfg(not(target_arch = "wasm32"))]
    let locale = sys_locale::get_locale();
    #[cfg(target_arch = "wasm32")]
    let locale = web_sys::window().and_then(|w| w.navigator().language());
    locale.map_or("en", |l| language_for_locale(&l))
}

/// Tests that switch the global UI language, or draw the UI under it, run one at a time.
#[cfg(test)]
pub(crate) static TEST_LANGUAGE: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(test)]
mod tests {
    use super::TEST_LANGUAGE as LANGUAGE;
    use super::*;
    use std::collections::{BTreeSet, HashMap};

    /// Every key the UI looks up: `tr("…")` / `trf("…"` literals in the sources, plus the
    /// static catalogs whose labels are translated where they are shown.
    fn keys() -> Vec<String> {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
        let mut keys = Vec::new();
        let mut directories = vec![dir];
        while let Some(directory) = directories.pop() {
            for entry in std::fs::read_dir(directory).unwrap() {
                let path = entry.unwrap().path();
                if path.is_dir() && path.file_name().is_some_and(|name| name != "i18n") {
                    directories.push(path);
                } else if path.extension().is_some_and(|extension| extension == "rs") {
                    literals(&std::fs::read_to_string(path).unwrap(), &mut keys);
                }
            }
        }
        keys.extend(catalog_keys().into_iter().map(str::to_owned));
        keys.sort();
        keys.dedup();
        keys
    }

    /// The labels of the menu catalog, panels, shortcuts, shelf tabs and engine enums that are
    /// translated where they are shown.
    fn catalog_keys() -> Vec<&'static str> {
        use texturecraft_engine::core::{BlendMode, Channel, EffectKind, GeneratorKind, ProceduralKind};
        use texturecraft_engine::filter_lib::{FILTERS, build, scalar_params};
        let mut k: Vec<&'static str> = Vec::new();
        k.extend(crate::menus::TOP);
        for i in crate::menus::CATALOG.iter().filter(|i| i.label != "-") {
            k.push(i.label);
            k.extend(i.sub);
        }
        k.extend(crate::docking::VIEWS.iter().map(|(_, p)| p.name()));
        for b in crate::keymap::bindings() {
            k.push(b.label);
            k.push(b.group);
        }
        k.extend(crate::shelf::TABS.iter().map(|t| t.2));
        k.extend(Channel::ALL.iter().flat_map(|c| [c.label(), c.short()]));
        k.extend(BlendMode::all().map(|m| m.label()));
        k.extend(texturecraft_engine::formats::export::VirtualMap::ALL.iter().map(|m| m.label()));
        k.extend(texturecraft_engine::Tool::ALL.iter().map(|t| t.label()));
        k.extend(ProceduralKind::ALL.iter().map(|p| p.label()));
        k.extend(GeneratorKind::ALL.iter().map(|g| g.label()));
        k.extend(texturecraft_engine::paint::BrushAlpha::ALL.iter().map(|a| a.label()));
        k.extend(texturecraft_engine::geom::primitives::BUILTINS.iter().map(|(_, l)| *l));
        for f in FILTERS {
            k.extend([f.label, f.group]);
            if let Ok(EffectKind::Filter(x)) = build(f.id, &serde_json::Value::Null) {
                k.push(x.label());
                k.extend(scalar_params(&x).into_iter().map(|p| p.1));
            }
        }
        // Effect, view and camera names the engine gives.
        k.extend([
            "Paint",
            "Path",
            "Fill",
            "Generator",
            "Levels",
            "Color selection",
            "Compare mask",
            "Anchor point",
            "Layer mask",
            texturecraft_engine::camera_cmds::DEFAULT_CAMERA,
            crate::dock::SOLO_HINT,
        ]);
        k.retain(|s| !s.is_empty() && !UNTRANSLATED.contains(s));
        k
    }

    /// String literals passed straight to `tr(` or `trf(` in `src`.
    fn literals(src: &str, out: &mut Vec<String>) {
        let src = src
            .split(
                "#[cfg(test)]
mod tests",
            )
            .next()
            .unwrap_or(src);
        for pat in ["tr(", "trf(", "n_("] {
            let mut rest = src;
            while let Some(i) = rest.find(pat) {
                let before = rest[..i].chars().next_back();
                rest = rest[i + pat.len()..].trim_start();
                if before.is_some_and(|c| c.is_alphanumeric() || c == '_') {
                    continue;
                }
                let Some(after) = rest.strip_prefix('"') else { continue };
                rest = after;
                let mut s = String::new();
                let mut chars = rest.chars();
                while let Some(c) = chars.next() {
                    match c {
                        '"' => break,
                        '\\' => match chars.next() {
                            Some('n') => s.push('\n'),
                            Some(c) => s.push(c),
                            None => break,
                        },
                        c => s.push(c),
                    }
                }
                out.push(s);
            }
        }
    }

    #[test]
    fn embedded_catalogs_are_valid() {
        assert!(I18N.check_all().is_empty(), "{:?}", I18N.check_all());
    }

    /// Translation work never blocks a feature: report missing/stale keys without failing.
    #[test]
    fn translation_coverage_report() {
        let keys = keys();
        assert!(keys.len() > 300, "found only {} keys", keys.len());
        let used: BTreeSet<_> = keys.iter().map(String::as_str).collect();
        for &(locale, common, app) in CRAFT_I18N {
            let Some(lang) = I18N.from_code(locale) else { continue };
            let missing: Vec<_> = keys.iter().filter(|key| I18N.get(lang, "", key).is_none()).collect();
            // Decode TSV escapes for comparison with the literal source strings.
            let mut rows = HashMap::new();
            for text in [common, app] {
                for line in text.lines().filter(|l| !l.starts_with('#')) {
                    let columns: Vec<_> = line.split('\t').collect();
                    if let ["", source, _] = columns.as_slice() {
                        let mut decoded = String::new();
                        let mut chars = source.chars();
                        while let Some(c) = chars.next() {
                            decoded.push(if c == '\\' {
                                match chars.next() {
                                    Some('n') => '\n',
                                    Some('t') => '\t',
                                    Some(c) => c,
                                    None => '\\',
                                }
                            } else {
                                c
                            });
                        }
                        rows.insert(decoded, ());
                    }
                }
            }
            let stale: Vec<_> = rows.keys().filter(|key| !used.contains(key.as_str())).collect();
            eprintln!("{locale}: missing {} {missing:?}; unused/shared {} {stale:?}", missing.len(), stale.len());
        }
    }

    const UNTRANSLATED: &[&str] = &["RGB", "RGBA"];

    #[test]
    fn lookup_formatting_and_fallbacks() {
        let _lock = LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        set_language("xx");
        assert_eq!(current().code, "en");
        assert_eq!(tr("Layers"), "Layers");
        assert_eq!(trf("Saved {path}", &[("path", &"a.tcraft")]), "Saved a.tcraft");
        assert_eq!(trf("{a} of {b} {c}", &[("a", &1), ("b", &2)]), "1 of 2 {c}");
        assert_eq!(trf("{unclosed", &[]), "{unclosed");
        for (locale, code) in
            [("de-AT", "de"), ("pt_PT.UTF-8", "pt-BR"), ("zh-Hant-TW", "zh-Hant"), ("ja", "ja"), ("ko_KR", "ko"), ("ar-EG", "ar"), ("nl-NL", "en"), ("", "en")]
        {
            assert_eq!(language_for_locale(locale), code, "{locale}");
        }
        set_language("de");
        assert_eq!(current().code, if available("de") { "de" } else { "en" });
        assert_eq!(tr("not a key"), "not a key");
        set_language("en");
    }

    #[test]
    fn shared_lookup_overrides_and_missing_translations() {
        static CATALOGS: &craft_i18n::Embedded = &[("es", "\tSave\tGuardar\n\tOpen\tAbrir\n", "\tSave\tGuardar proyecto\n")];
        let (runtime, problems) = craft_i18n::I18n::new(CATALOGS, &["es"]);
        assert!(problems.is_empty());
        assert!(runtime.check_all().is_empty());
        let lang = runtime.from_code("es").unwrap();
        assert_eq!(runtime.t(lang, "Save"), "Guardar proyecto");
        assert_eq!(runtime.t(lang, "Open"), "Abrir");
        assert_eq!(runtime.t(lang, "New feature"), "New feature");
        assert_eq!(runtime.t(craft_i18n::Lang::EN, "Save"), "Save");
    }

    #[test]
    fn saved_language_codes_and_spanish_lookup() {
        let _lock = LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        for language in &LANGUAGES {
            set_language(language.code);
            assert_eq!(current().code, if available(language.code) { language.code } else { "en" });
        }
        set_language("es");
        assert_eq!(tr("Layers"), if available("es") { "Capas" } else { "Layers" });
        set_language("en");
    }

    #[test]
    fn lookup_is_fast() {
        let _lock = LANGUAGE.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
        let keys = keys();
        set_language("ja");
        let t = std::time::Instant::now();
        let mut n = 0usize;
        for _ in 0..100 {
            for k in &keys {
                n += tr(k).len();
            }
        }
        let per = t.elapsed().as_secs_f64() * 1e9 / (100 * keys.len()) as f64;
        set_language("en");
        assert!(n > 0);
        // A frame looks up a few hundred strings: well under a microsecond each.
        assert!(per < 1000.0, "{per:.0} ns per lookup");
    }
}
