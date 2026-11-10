//! Isolated folder evaluation with the original stack available for anchor inputs.

use super::{Channel, ChannelKind, Composite, Inputs, LayerId, Texel, all_tiles, neighbourhood_fields, plan, tile_texels, to_u16, to_u16_signed};

/// Composite one layer over channel defaults, keeping anchors from the stack below it.
pub(crate) fn composite_layer(inputs: &Inputs, layer: LayerId, channel: Channel) -> Option<Composite> {
    let set = inputs.project.texture_set(inputs.ts).ok()?;
    let channels = set.channel_list();
    let (width, height) = (inputs.gbuf.width, inputs.gbuf.height);
    let tiles = all_tiles(width, height);
    let fields = neighbourhood_fields(inputs, &channels, &tiles);
    let mut composite = Composite::with_size(&[channel], width, height);
    let output = composite.channels.get_mut(&channel)?;

    for (tile_x, tile_y) in tiles {
        let nodes = plan(inputs, &[layer], &channels, tile_x, tile_y, 0, Some(&fields));
        let full_stack = if set.stack.as_slice() == [layer] { Vec::new() } else { plan(inputs, &set.stack, &channels, tile_x, tile_y, 0, Some(&fields)) };
        for (x, y, offset) in tile_texels(inputs.gbuf, tile_x, tile_y) {
            let mut texel = Texel::new(inputs, x, y, offset, 0.0);
            texel.anchors_below(&full_stack, layer);
            let mut values = Channel::ALL.map(Channel::default_value);
            texel.eval(&nodes, &mut values);
            let value = values.get(channel.index()).copied().unwrap_or_else(|| channel.default_value());
            let encoded = value.map(|component| if channel.kind() == ChannelKind::Signed { to_u16_signed(component) } else { to_u16(component) });
            output.set(x, y, &encoded[..channel.components()]);
        }
    }

    Some(composite)
}
