#![cfg(all(unix, not(target_os = "macos"), not(target_arch = "wasm32")))]
use egui::{Rect, pos2};
use texturecraft_ui_egui::stereo_view::{pack, render_size};

#[test]
fn source_grid_uses_rounded_pixel_edges_and_keeps_an_explicit_choice() {
    let rect = Rect::from_min_max(pos2(0.3, 0.3), pos2(100.6, 60.6));
    assert_eq!(render_size(None, rect, [1280.0, 800.0], 1.5).unwrap(), [151, 91]);
    assert_eq!(render_size(Some([640, 360]), rect, [1280.0, 800.0], 1.5).unwrap(), [640, 360]);
    assert_eq!(render_size(None, Rect::NOTHING, [1280.0, 800.0], 1.5).unwrap(), [1920, 1200]);
    for size in [[0, 360], [640, 0], [8193, 1]] {
        assert!(render_size(Some(size), rect, [1280.0, 800.0], 1.5).is_err());
    }
    assert!(render_size(None, rect, [1280.0, 800.0], f32::NAN).is_err());
}

#[test]
fn pair_serialization_preserves_complete_rows_without_resampling() {
    let left = [10, 20, 30, 255, 40, 50, 60, 255];
    let right = [70, 80, 90, 255, 100, 110, 120, 255];
    let picture = pack([1, 2], &left, &right).unwrap();
    assert_eq!((picture.width, picture.height, picture.stereo), (2, 2, true));
    assert_eq!(&*picture.rgba, &[10, 20, 30, 255, 70, 80, 90, 255, 40, 50, 60, 255, 100, 110, 120, 255]);
    assert!(pack([0, 2], &left, &right).is_err());
    assert!(pack([1, 2], &left[..4], &right).is_err());
    assert!(pack([1, 2], &left, &right[..4]).is_err());
}

#[test]
fn stereo_choices_and_diagnostics_resolve_in_the_required_languages() {
    use texturecraft_ui_egui::i18n::{available, language_for_locale, set_language, tr};
    assert_eq!(language_for_locale("zh_TW.UTF-8"), "zh-Hant");
    assert_eq!(language_for_locale("zh-Hant-HK"), "zh-Hant");
    assert_eq!(language_for_locale("zh_CN"), "zh-Hans");
    assert_eq!(language_for_locale("es-AR"), "es");
    for language in ["pl", "pt-BR", "fr", "de", "es", "ja", "ko", "zh-Hans", "zh-Hant"] {
        assert!(available(language), "the edition needs its partial stereo catalog: {language}");
        set_language(language);
        for key in [
            "Enable 3D",
            "Render resolution per eye",
            "Desktop Display Settings",
            "Stereo View",
            "The model viewport size preserves its current sampling; the desktop scales once.",
            "Use model viewport size",
            "Width",
            "Height",
            "No GPU renderer available",
            "Could not show the stereo view: {error}",
            "Could not open Display Settings: {error}",
            "Could not read the stereo views from the GPU.",
            "The stereo views have different native dimensions.",
            "The stereo window closed before the pair was shown.",
            "The stereo view has invalid native dimensions.",
            "Stereo depth is unavailable on this desktop; showing the flat model.",
        ] {
            assert_ne!(tr(key), key, "{language}: {key}");
        }
    }
    set_language("en");
}
