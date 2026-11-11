//! Levels handles over a histogram and output gradient, with labelled fields for exact values

use egui::{Color32, CornerRadius, CursorIcon, Id, Pos2, Rect, Response, Sense, Stroke, Ui, WidgetInfo, pos2, vec2};
use texturecraft_engine::core::Levels;

use crate::i18n::tr;
use crate::levels_handle::LevelsHandle;
use crate::theme::Tokens;

const HISTOGRAM_HEIGHT: f32 = 56.0;
const HANDLE_HEIGHT: f32 = 11.0;
const HANDLE_HALF_WIDTH: f32 = 5.0;

/// Height of a handle's straight sides; the rest is its tip pointing at the track
const HANDLE_SIDE_HEIGHT: f32 = 6.0;

/// A larger target than the painted pointer makes handles easier to grab
const HANDLE_TARGET_SIZE: f32 = 24.0;

/// Fraction of the track moved by one arrow key press
const KEYBOARD_STEP: f32 = 0.01;

const GRADIENT_HEIGHT: f32 = 8.0;
const GAP_BELOW_HISTOGRAM: f32 = 3.0;
const VALUE_ROW_HEIGHT: f32 = 16.0;
const VALUE_FONT_SIZE: f32 = 11.5;

/// Padding of the histogram well around its bars
const WELL_PADDING: f32 = 3.0;

/// Gray of the gamma handle, between the black input-low and white input-high handles
const GAMMA_HANDLE_GRAY: u8 = 128;

/// Grid lines across the histogram, at these fractions of its width
const GRID_FRACTIONS: [f32; 3] = [0.25, 0.5, 0.75];

/// Decimals printed beside a handle; exact entry keeps the stored precision
const VALUE_DECIMALS: usize = 3;

/// Space kept around a printed value; a value closer to one already printed is left out
const VALUE_SPACING: f32 = 2.0;

/// Opacity of the shade over the histogram outside the input range
const OUTSIDE_RANGE_SHADE: f32 = 0.7;

/// Opacity of the line at gamma, fainter than the input range's accent lines
const GAMMA_LINE_OPACITY: f32 = 0.6;

/// Edits `levels` over histogram `counts`; `Some(merge)` per change, `merge` true for a drag's later frames
pub fn levels_editor(ui: &mut Ui, id_salt: impl std::hash::Hash + std::fmt::Debug, counts: Option<&[u32]>, levels: &mut Levels) -> Option<bool> {
    let height = VALUE_ROW_HEIGHT * 2.0 + HANDLE_HEIGHT * 2.0 + HISTOGRAM_HEIGHT + GAP_BELOW_HISTOGRAM + GRADIENT_HEIGHT;
    let (rect, _) = ui.allocate_exact_size(vec2(ui.available_width(), height), Sense::hover());
    let histogram_top = rect.top() + VALUE_ROW_HEIGHT + HANDLE_HEIGHT;
    let gradient_top = output_handles_top(rect) - GRADIENT_HEIGHT;
    let track = track(rect);
    let id = ui.make_persistent_id(("levels_editor", id_salt));

    // Gamma goes last so it stays grabbable between input handles that close in on it
    let handles = [LevelsHandle::InputLow, LevelsHandle::InputHigh, LevelsHandle::Gamma, LevelsHandle::OutputLow, LevelsHandle::OutputHigh];
    forget_previous_editor(ui, id, &handles);

    let mut edit = None;
    let mut responses = Vec::with_capacity(handles.len());
    let mut focus_value = None;
    for handle in handles {
        let x = track_x(&track, handle.position(levels));
        let top = handle_top(rect, handle);
        let hit = Rect::from_center_size(pos2(x, top + HANDLE_HEIGHT / 2.0), vec2(HANDLE_TARGET_SIZE, HANDLE_TARGET_SIZE));
        let response = ui
            .interact(hit, id.with(handle), Sense::click_and_drag())
            .on_hover_cursor(CursorIcon::ResizeHorizontal)
            .on_hover_text(tr("Double-click or press Enter to edit"));
        if response.drag_started() {
            ui.data_mut(|data| data.remove_temp::<bool>(response.id));
        }

        if response.dragged()
            && let Some(pointer) = response.interact_pointer_pos()
        {
            let mut next = *levels;
            handle.set_position(&mut next, track_position(&track, pointer.x));
            if next != *levels {
                *levels = next;
                edit = Some(continues_drag(ui, response.id));
            }
        }

        edit = edit.or(handle_keys(ui, &response, handle, levels));
        if response.double_clicked() || (response.has_focus() && ui.input_mut(|input| input.consume_key(egui::Modifiers::NONE, egui::Key::Enter))) {
            focus_value = Some(handle);
        }

        let value = handle.value(levels);
        response.widget_info(|| WidgetInfo::slider(true, f64::from(value), handle.label()));
        let range = handle.value_range(levels);
        ui.ctx().accesskit_node_builder(response.id, |node| {
            use egui::accesskit::Action;

            node.set_min_numeric_value(f64::from(*range.start()));
            node.set_max_numeric_value(f64::from(*range.end()));
            node.add_action(Action::SetValue);
            if value < *range.end() {
                node.add_action(Action::Increment);
            }
            if value > *range.start() {
                node.add_action(Action::Decrement);
            }
        });
        responses.push((handle, response));
    }

    let well = Rect::from_x_y_ranges(rect.x_range().shrink(HANDLE_HALF_WIDTH), histogram_top..=histogram_top + HISTOGRAM_HEIGHT);
    paint_histogram(ui, well, counts.unwrap_or_default(), levels);
    paint_output_gradient(ui, Rect::from_x_y_ranges(track, gradient_top..=gradient_top + GRADIENT_HEIGHT));
    paint_handles_and_values(ui, rect, &responses, levels);

    let cancel_typing = responses.iter().any(|(_, response)| response.is_pointer_button_down_on());
    let fields = [LevelsHandle::InputLow, LevelsHandle::Gamma, LevelsHandle::InputHigh, LevelsHandle::OutputLow, LevelsHandle::OutputHigh];
    let mut labels = Vec::with_capacity(fields.len());

    // A shared label row keeps the fields aligned when translated names wrap
    ui.columns(fields.len(), |columns| {
        for (ui, handle) in columns.iter_mut().zip(fields) {
            let label =
                ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| ui.label(egui::RichText::new(handle.label()).size(VALUE_FONT_SIZE))).inner;
            labels.push(label.id);
        }
    });

    let mut invalid = false;
    ui.columns(fields.len(), |columns| {
        for ((ui, handle), label) in columns.iter_mut().zip(fields).zip(labels) {
            let (typed, field_invalid) = number_field(ui, id.with((handle, "value")), label, handle, levels, focus_value == Some(handle), cancel_typing);
            edit = edit.or(typed);
            invalid |= field_invalid;
        }
    });
    if invalid {
        ui.colored_label(Tokens::get(ui.ctx()).warning, tr("Enter a finite number"));
    }

    edit
}

/// Properties shows one effect at a time; its old cursor, draft and drag state must not accumulate
fn forget_previous_editor(ui: &Ui, id: Id, handles: &[LevelsHandle]) {
    let owner = ui.make_persistent_id("levels_editor_owner");
    ui.data_mut(|data| {
        let previous = data.get_temp::<Id>(owner);
        if previous == Some(id) {
            return;
        }

        if let Some(previous) = previous {
            for handle in handles {
                let field = previous.with((handle, "value"));
                data.remove::<egui::text_edit::TextEditState>(field);
                data.remove::<(String, bool)>(field);
                data.remove::<bool>(previous.with(handle));
            }
        }
        data.insert_temp(owner, id);
    });
}

fn handle_top(rect: Rect, handle: LevelsHandle) -> f32 {
    if LevelsHandle::INPUT.contains(&handle) { rect.top() + VALUE_ROW_HEIGHT } else { output_handles_top(rect) }
}

fn output_handles_top(rect: Rect) -> f32 {
    rect.bottom() - VALUE_ROW_HEIGHT - HANDLE_HEIGHT
}

/// The x range the handles move along in the editor's `rect`, lined up with the histogram bars
fn track(rect: Rect) -> egui::Rangef {
    rect.x_range().shrink(HANDLE_HALF_WIDTH + WELL_PADDING)
}

fn track_x(track: &egui::Rangef, position: f32) -> f32 {
    track.min + track.span() * position.clamp(0.0, 1.0)
}

fn track_position(track: &egui::Rangef, x: f32) -> f32 {
    ((x - track.min) / track.span().max(1.0)).clamp(0.0, 1.0)
}

/// True once the drag of `id` has already changed a value, so its later changes share that undo step
fn continues_drag(ui: &Ui, id: Id) -> bool {
    ui.data_mut(|data| std::mem::replace(data.get_temp_mut_or_default::<bool>(id), true))
}

/// `value` with [`VALUE_DECIMALS`] decimals, trailing zeros dropped ("1", "0.25", "0.634")
fn shown_value(value: f32) -> String {
    let fixed = format!("{value:.VALUE_DECIMALS$}");
    fixed.trim_end_matches('0').trim_end_matches('.').to_string()
}

/// Where `handle`'s value of `size` goes: over an input handle, under an output one, inside `rect`
fn value_rect(rect: Rect, handle: LevelsHandle, levels: &Levels, size: egui::Vec2) -> Rect {
    let x = track_x(&track(rect), handle.position(levels));
    let top = handle_top(rect, handle);
    let placed = if LevelsHandle::INPUT.contains(&handle) {
        egui::Align2::CENTER_BOTTOM.anchor_size(pos2(x, top), size)
    } else {
        egui::Align2::CENTER_TOP.anchor_size(pos2(x, top + HANDLE_HEIGHT), size)
    };
    let shift = (rect.left() - placed.left()).max(0.0) + (rect.right() - placed.right()).min(0.0);
    placed.translate(vec2(shift, 0.0))
}

/// The histogram in its well: quarter lines, the outside of the input range dimmed, lines at the handles
fn paint_histogram(ui: &Ui, well: Rect, counts: &[u32], levels: &Levels) {
    let tokens = Tokens::get(ui.ctx());
    let painter = ui.painter_at(well);
    painter.rect(well, CornerRadius::same(tokens.radius), tokens.field, Stroke::new(1.0, tokens.field_border), egui::StrokeKind::Inside);

    let inner = well.shrink(WELL_PADDING);
    let vertical_line = |x: f32, color: Color32| painter.line_segment([pos2(x, inner.top()), pos2(x, inner.bottom())], Stroke::new(1.0, color));
    for fraction in GRID_FRACTIONS {
        vertical_line(track_x(&inner.x_range(), fraction), tokens.field_border);
    }

    let fullest = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let bar_width = inner.width() / counts.len().max(1) as f32;
    for (index, count) in counts.iter().enumerate() {
        let bar_height = inner.height() * *count as f32 / fullest;
        let x = inner.left() + index as f32 * bar_width;
        painter.rect_filled(Rect::from_min_max(pos2(x, inner.bottom() - bar_height), pos2(x + bar_width, inner.bottom())), 0.0, tokens.text_dim);
    }

    let x_at = |handle: LevelsHandle| track_x(&inner.x_range(), handle.position(levels));
    let (low, high) = (x_at(LevelsHandle::InputLow), x_at(LevelsHandle::InputHigh));
    let shade = tokens.field.gamma_multiply(OUTSIDE_RANGE_SHADE);
    painter.rect_filled(Rect::from_x_y_ranges(inner.left()..=low, inner.y_range()), 0.0, shade);
    painter.rect_filled(Rect::from_x_y_ranges(high..=inner.right(), inner.y_range()), 0.0, shade);
    vertical_line(low, tokens.accent);
    vertical_line(high, tokens.accent);
    vertical_line(x_at(LevelsHandle::Gamma), tokens.text_dim.gamma_multiply(GAMMA_LINE_OPACITY));
}

/// A black-to-white strip: the output values the output handles pick from
fn paint_output_gradient(ui: &Ui, strip: Rect) {
    let mut mesh = egui::Mesh::default();
    for (corner, color) in
        [(strip.left_top(), Color32::BLACK), (strip.right_top(), Color32::WHITE), (strip.right_bottom(), Color32::WHITE), (strip.left_bottom(), Color32::BLACK)]
    {
        mesh.colored_vertex(corner, color);
    }
    mesh.add_triangle(0, 1, 2);
    mesh.add_triangle(0, 2, 3);

    let painter = ui.painter();
    painter.add(mesh);
    painter.rect_stroke(strip, 0.0, Stroke::new(1.0, Tokens::get(ui.ctx()).field_border), egui::StrokeKind::Outside);
}

/// Each handle filled with its gray, the active one on top, and its value beside it, overlapping values left out
fn paint_handles_and_values(ui: &Ui, rect: Rect, responses: &[(LevelsHandle, Response)], levels: &Levels) {
    let track = track(rect);
    let tokens = Tokens::get(ui.ctx());
    let painter = ui.painter();
    let priority = |response: &Response| (response.dragged(), response.hovered(), response.has_focus());

    // Active handles last, so the one on top is the one the pointer grabs
    let mut by_paint_order: Vec<&(LevelsHandle, Response)> = responses.iter().collect();
    by_paint_order.sort_by_key(|(_, response)| priority(response));
    for &(handle, ref response) in by_paint_order {
        let x = track_x(&track, handle.position(levels));
        let fill = match handle {
            LevelsHandle::InputLow => Color32::BLACK,
            LevelsHandle::Gamma => Color32::from_gray(GAMMA_HANDLE_GRAY),
            LevelsHandle::InputHigh => Color32::WHITE,
            LevelsHandle::OutputLow | LevelsHandle::OutputHigh => Color32::from_gray((handle.value(levels).clamp(0.0, 1.0) * 255.0).round() as u8),
        };
        let outline = if response.dragged() || response.has_focus() {
            tokens.accent
        } else if response.hovered() {
            tokens.text_strong
        } else {
            tokens.text
        };
        let points_down = LevelsHandle::INPUT.contains(&handle);
        painter.add(egui::Shape::convex_polygon(handle_outline(x, handle_top(rect, handle), points_down), fill, Stroke::new(1.5, outline)));
    }

    // Active values first, so a neighbour's overlapping value gives way to them
    let font = egui::FontId::proportional(VALUE_FONT_SIZE);
    let mut printed: Vec<Rect> = Vec::new();
    let mut by_priority: Vec<&(LevelsHandle, Response)> = responses.iter().collect();
    by_priority.sort_by_key(|(_, response)| std::cmp::Reverse(priority(response)));
    for &(handle, ref response) in by_priority {
        let color = if response.hovered() || response.has_focus() || response.dragged() { tokens.text_strong } else { tokens.text };
        let galley = painter.layout_no_wrap(shown_value(handle.value(levels)), font.clone(), color);
        let text_rect = value_rect(rect, handle, levels, galley.size());
        if printed.iter().any(|other| other.expand(VALUE_SPACING).intersects(text_rect)) {
            continue;
        }

        painter.galley(text_rect.min, galley, color);
        printed.push(text_rect);
    }
}

/// A pentagon pointer centred on `x`: flat side away from the track, tip touching it
fn handle_outline(x: f32, top: f32, points_down: bool) -> Vec<Pos2> {
    let bottom = top + HANDLE_HEIGHT;
    let (flat, side_end, tip) = if points_down { (top, top + HANDLE_SIDE_HEIGHT, bottom) } else { (bottom, bottom - HANDLE_SIDE_HEIGHT, top) };
    vec![
        pos2(x - HANDLE_HALF_WIDTH, flat),
        pos2(x + HANDLE_HALF_WIDTH, flat),
        pos2(x + HANDLE_HALF_WIDTH, side_end),
        pos2(x, tip),
        pos2(x - HANDLE_HALF_WIDTH, side_end),
    ]
}

/// Arrow keys move along the track; accessibility increment/decrement change the numeric value
fn handle_keys(ui: &Ui, response: &Response, handle: LevelsHandle, levels: &mut Levels) -> Option<bool> {
    use egui::accesskit::{Action, ActionData};

    let mut next = *levels;
    if response.has_focus() {
        ui.memory_mut(|memory| memory.set_focus_lock_filter(response.id, egui::EventFilter { horizontal_arrows: true, ..Default::default() }));
        let steps = ui.input(|input| input.num_presses(egui::Key::ArrowRight) as f32 - input.num_presses(egui::Key::ArrowLeft) as f32);
        if steps != 0.0 {
            handle.set_position(&mut next, handle.position(levels) + steps * KEYBOARD_STEP);
        }
    }

    ui.input(|input| {
        let steps = input.num_accesskit_action_requests(response.id, Action::Increment) as f32
            - input.num_accesskit_action_requests(response.id, Action::Decrement) as f32;
        if steps != 0.0 {
            handle.set_value(&mut next, handle.value(levels) + steps * KEYBOARD_STEP);
        }
        for request in input.accesskit_action_requests(response.id, Action::SetValue) {
            if let Some(ActionData::NumericValue(value)) = request.data {
                handle.set_value(&mut next, value as f32);
            }
        }
    });
    if next == *levels {
        return None;
    }

    *levels = next;
    Some(false)
}

/// Exact entry commits on Enter or focus loss; invalid input stays editable and Escape cancels
fn number_field(ui: &mut Ui, id: Id, label: Id, handle: LevelsHandle, levels: &mut Levels, focus: bool, cancel: bool) -> (Option<bool>, bool) {
    let shown = handle.value(levels).to_string();
    let (mut text, mut invalid) = ui.data(|data| data.get_temp::<(String, bool)>(id)).unwrap_or_else(|| (shown.clone(), false));
    if focus || cancel {
        text = shown.clone();
        invalid = false;
    }

    let response =
        ui.add(egui::TextEdit::singleline(&mut text).id(id).desired_width(ui.available_width()).align(egui::Align2::CENTER_CENTER)).labelled_by(label);
    if response.changed() {
        invalid = false;
    }
    if focus {
        let mut state = egui::text_edit::TextEditState::load(ui.ctx(), id).unwrap_or_default();
        state.cursor.set_char_range(Some(egui::text::CCursorRange::select_all_str(&text)));
        state.store(ui.ctx(), id);
        response.request_focus();
        ui.data_mut(|data| data.insert_temp(id, (text, false)));
        return (None, false);
    }

    // The app can consume Escape before the field sees it, so read the raw event
    let escape = ui.input(|input| input.raw.events.iter().any(|event| matches!(event, egui::Event::Key { key: egui::Key::Escape, pressed: true, .. })));
    if cancel || escape {
        ui.data_mut(|data| data.remove::<(String, bool)>(id));
        ui.memory_mut(|memory| memory.surrender_focus(id));
        return (None, false);
    }

    if response.has_focus() {
        ui.data_mut(|data| data.insert_temp(id, (text, invalid)));
        return (None, invalid);
    }

    // Discard abandoned input when a hidden editor reappears
    ui.data_mut(|data| data.remove::<(String, bool)>(id));
    if !response.lost_focus() || text == shown {
        return (None, false);
    }

    let Some(value) = text.trim().parse::<f32>().ok().filter(|value| value.is_finite()) else {
        response.request_focus();
        ui.data_mut(|data| data.insert_temp(id, (text, true)));
        return (None, true);
    };

    let before = *levels;
    handle.set_value(levels, value);
    ((before != *levels).then_some(false), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn switching_effects_releases_old_field_and_drag_state() {
        let context = egui::Context::default();
        let mut levels = Levels::default();
        for effect in 0..100 {
            context
                .run_ui(egui::RawInput::default(), |ui| {
                    levels_editor(ui, effect, None, &mut levels);
                    let id = ui.make_persistent_id(("levels_editor", effect));
                    ui.data_mut(|data| {
                        data.insert_temp(id.with((LevelsHandle::Gamma, "value")), ("invalid draft".to_string(), true));
                        data.insert_temp(id.with(LevelsHandle::Gamma), true);
                    });
                })
                .drop_without_applying_deltas();
        }

        context.data(|data| {
            assert_eq!(data.count::<egui::text_edit::TextEditState>(), 5, "only the current effect's fields remain");
            assert_eq!(data.count::<(String, bool)>(), 1, "only the current draft remains");
            assert_eq!(data.count::<bool>(), 1, "only the current drag state remains");
        });
    }
}
