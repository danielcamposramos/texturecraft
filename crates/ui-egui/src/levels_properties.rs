//! Properties of a levels effect: the levels editor, then the channel picker (content), Reset, Auto and Invert

use std::sync::Arc;

use egui::Ui;
use serde_json::json;
use texturecraft_engine::core::{Channel, ChannelKind, Layer, LayerId, Levels};

use crate::TextureCraftApp;
use crate::i18n::tr;
use crate::levels_editor::levels_editor;
use crate::theme::caption;

/// Buckets of the histogram, over 0..1
const HISTOGRAM_BUCKETS: usize = 64;

/// Share of the counted texels a bucket needs to hold for Auto to keep it in the input range
const AUTO_MIN_SHARE: f64 = 0.001;

/// The properties of levels effect `effect_id` of `layer`; every edit goes through `effect.set`
pub(crate) fn levels_properties(app: &mut TextureCraftApp, ui: &mut Ui, layer: &Layer, effect_id: LayerId, levels: &Levels) {
    ui.label(caption(tr("Levels")));

    let is_content = layer.content_effects.iter().any(|effect| effect.id == effect_id);
    let channels: Vec<Channel> =
        if is_content { layer.channels.keys().copied().filter(|channel| channel.kind() != ChannelKind::NormalMap).collect() } else { Vec::new() };
    let picked = egui::Id::unique(("levels_channel", effect_id.0));
    let channel = ui
        .data(|data| data.get_temp(picked))
        .filter(|channel| channels.contains(channel))
        .or(channels.first().copied())
        .or(layer.channels.keys().next().copied())
        .unwrap_or(Channel::BaseColor);
    let counts = histogram_counts(app, ui, effect_id, channel);

    let mut edited = *levels;
    let mut edit = levels_editor(ui, effect_id, counts.as_ref().map(|counts| counts.as_slice()), &mut edited);
    ui.horizontal(|ui| {
        if !channels.is_empty() {
            egui::ComboBox::from_id_salt(picked)
                .selected_text(tr(channel.label()))
                .show_ui(ui, |ui| {
                    for option in channels {
                        if ui.selectable_label(option == channel, tr(option.label())).clicked() {
                            ui.data_mut(|data| data.insert_temp(picked, option));
                        }
                    }
                })
                .response
                .on_hover_text(tr("Channel"));
        }

        // Right to left: the last button comes first
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            if ui.button(tr("Invert")).clicked() {
                std::mem::swap(&mut edited.out_low, &mut edited.out_high);
                edit = Some(false);
            }

            let auto_range = counts.as_ref().and_then(|counts| auto_input_range(counts));
            let auto = ui.add_enabled(auto_range.is_some(), egui::Button::new(tr("Auto"))).on_hover_text(tr("Fit the input range to the histogram"));
            if auto.clicked()
                && let Some((low, high)) = auto_range
            {
                (edited.in_low, edited.in_high) = (low, high);
                edit = Some(false);
            }

            if ui.add_enabled(edited != Levels::default(), egui::Button::new(tr("Reset"))).clicked() {
                edited = Levels::default();
                edit = Some(false);
            }
        });
    });

    // Auto or Invert with nothing to change would add an empty undo step
    if let Some(merge) = edit
        && edited != *levels
    {
        let Levels { in_low, in_high, gamma, out_low, out_high } = edited;
        app.run(
            "effect.set",
            json!({"effect": effect_id.0, "inLow": in_low, "inHigh": in_high, "gamma": gamma, "outLow": out_low, "outHigh": out_high, "merge": merge}),
        );
    }
}

/// The histogram levels effect `effect_id` remaps, recomputed after a change once no pointer button is held
fn histogram_counts(app: &TextureCraftApp, ui: &Ui, effect_id: LayerId, channel: Channel) -> Option<Arc<Vec<u32>>> {
    let key = egui::Id::unique(("levels_histogram", effect_id.0, channel));
    let owner = ui.make_persistent_id("levels_histogram_owner");

    // Selection changes the revision, so old effects' histograms cannot be reused
    ui.data_mut(|data| {
        let previous = std::mem::replace(data.get_temp_mut_or(owner, key), key);
        if previous != key {
            data.remove::<(u64, Arc<Vec<u32>>)>(previous);
        }
    });

    let revision = app.session.revision;
    match ui.data(|data| data.get_temp::<(u64, Arc<Vec<u32>>)>(key)) {
        Some((cached, counts)) if cached == revision || ui.input(|input| input.pointer.any_down()) => Some(counts),
        _ => app.session.effect_histogram(effect_id, channel, HISTOGRAM_BUCKETS).map(|counts| {
            let counts = Arc::new(counts);
            ui.data_mut(|data| data.insert_temp(key, (revision, counts.clone())));
            counts
        }),
    }
}

/// The input range Auto picks: the first to the last bucket holding more than [`AUTO_MIN_SHARE`] of the texels
fn auto_input_range(counts: &[u32]) -> Option<(f32, f32)> {
    let total: u64 = counts.iter().map(|&count| u64::from(count)).sum();
    let holds_enough = |count: &u32| f64::from(*count) > total as f64 * AUTO_MIN_SHARE;
    let first = counts.iter().position(holds_enough)?;
    let last = counts.iter().rposition(holds_enough)?;
    let bucket_count = counts.len() as f32;
    Some((first as f32 / bucket_count, (last + 1) as f32 / bucket_count))
}

#[cfg(test)]
mod tests {
    use super::{HISTOGRAM_BUCKETS, auto_input_range};

    #[test]
    fn auto_skips_buckets_under_the_share() {
        let mut counts = vec![0u32; HISTOGRAM_BUCKETS];
        counts[3] = 1;
        counts[10] = 5_000;
        counts[40] = 5_000;
        counts[60] = 1;

        assert_eq!(auto_input_range(&counts), Some((10.0 / HISTOGRAM_BUCKETS as f32, 41.0 / HISTOGRAM_BUCKETS as f32)));
    }

    #[test]
    fn auto_has_nothing_to_fit_in_an_empty_histogram() {
        assert_eq!(auto_input_range(&[0; HISTOGRAM_BUCKETS]), None);
        assert_eq!(auto_input_range(&[]), None);
    }
}
