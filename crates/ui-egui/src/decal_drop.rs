//! A shelf image dropped on the 3D view waits for a menu pick at the drop point: a mask, or a channel of the texture set under it

use egui::{Id, Pos2, Ui, Vec2};
use serde_json::json;
use texturecraft_engine::geom::OrbitCamera;

use crate::TextureCraftApp;
use crate::i18n::tr;

/// What the menu offers besides the texture set's channels: a new fill layer masked by the image
const MASK_CHOICE: &str = "mask";

/// The most of the window's height the channel list takes before it scrolls
const MENU_MAX_HEIGHT_SHARE: f32 = 0.6;

/// An image dropped on the 3D view, placed as a decal once the menu at the drop point is picked
#[derive(Clone)]
pub struct DecalDrop {
    /// The image resource
    pub image: String,
    /// The texture set under the drop point
    pub texture_set: usize,
    /// The drop point in view pixels
    pub view_position: Vec2,
    /// The 3D view's size in pixels
    pub view_size: Vec2,
    /// The camera used to pick the drop point
    pub camera: OrbitCamera,
    /// The drop point on screen, where the menu opens
    pub screen_position: Pos2,
}

impl DecalDrop {
    /// Where the waiting drop and its menu's state are kept
    fn menu_id() -> Id {
        Id::unique("decal_drop_menu")
    }

    /// Open the menu for this drop (replacing any other waiting drop): it stays open while the drop waits
    pub fn open(self, ui: &Ui) {
        let frame = ui.ctx().cumulative_frame_nr();
        ui.ctx().data_mut(|data| data.insert_temp(Self::menu_id(), (self, frame)));
    }

    /// Keep input out of both views while the menu waits and for the rest of its closing frame
    pub fn blocks_input(ctx: &egui::Context) -> bool {
        Self::current(ctx).is_some() || ctx.data(|data| data.get_temp::<u64>(Self::menu_id().with("closed_frame"))) == Some(ctx.cumulative_frame_nr())
    }

    /// The waiting drop; one whose menu went undrawn for a frame (the 3D view was hidden) is dropped instead of blocking the views
    fn current(ctx: &egui::Context) -> Option<DecalDrop> {
        let frame = ctx.cumulative_frame_nr();
        let (drop, menu_frame) = ctx.data(|data| data.get_temp::<(DecalDrop, u64)>(Self::menu_id()))?;
        if frame > menu_frame + 1 {
            ctx.data_mut(|data| data.remove::<(DecalDrop, u64)>(Self::menu_id()));
            return None;
        }

        Some(drop)
    }

    /// Close the menu and consume the rest of this frame's input
    fn close(ctx: &egui::Context) {
        let id = Self::menu_id();
        let frame = ctx.cumulative_frame_nr();

        // The 2D view reads input after this menu closes in the 3D view
        ctx.data_mut(|data| {
            data.remove::<(DecalDrop, u64)>(id);
            data.insert_temp(id.with("closed_frame"), frame);
        });
    }

    /// Show the waiting drop's menu over the 3D view `response`; a pick places the decal, a click elsewhere drops nothing
    pub fn show_menu(app: &mut TextureCraftApp, ui: &Ui, response: &egui::Response) {
        let id = Self::menu_id();
        let Some(drop) = Self::current(ui.ctx()) else { return };

        // Saved view pixels must not be picked through a different camera
        if drop.camera != app.session.camera {
            Self::close(ui.ctx());
            return;
        }

        let frame = ui.ctx().cumulative_frame_nr();
        ui.ctx().data_mut(|data| data.insert_temp(id, (drop.clone(), frame)));

        let channels = app.session.doc.project.texture_set(drop.texture_set).map(|set| set.channel_list()).unwrap_or_default();
        let mut picked = None;
        let shown = egui::Popup::new(id, ui.ctx().clone(), egui::PopupAnchor::Position(drop.screen_position), response.layer_id)
            .kind(egui::PopupKind::Menu)
            .layout(egui::Layout::top_down_justified(egui::Align::Min))
            .style(egui::containers::menu::menu_style)
            .open(true)
            .close_behavior(egui::PopupCloseBehavior::CloseOnClickOutside)
            .show(|ui| {
                if ui.button(tr("Mask")).clicked() {
                    picked = Some(MASK_CHOICE);
                }
                ui.separator();

                // Many channels scroll instead of running off the window
                let max_height = ui.ctx().content_rect().height() * MENU_MAX_HEIGHT_SHARE;
                egui::ScrollArea::vertical().max_height(max_height).show(ui, |ui| {
                    for channel in channels {
                        if ui.button(tr(channel.label())).clicked() {
                            picked = Some(channel.id());
                        }
                    }
                });
            });

        if picked.is_some() || shown.is_none_or(|menu| menu.response.should_close()) {
            Self::close(ui.ctx());
        }

        let Some(channel) = picked else { return };

        app.run(
            "layer.add.decal",
            json!({
                "image": drop.image,
                "channel": channel,
                "x": drop.view_position.x,
                "y": drop.view_position.y,
                "viewport": [drop.view_size.x, drop.view_size.y],
            }),
        );
    }
}
