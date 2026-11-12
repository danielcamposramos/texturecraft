//! Importing meshes with embedded material textures

use serde_json::json;
use texturecraft_engine::Session;
use texturecraft_engine::core::{Channel, LayerKind, Source};
use texturecraft_engine::geom::primitives;

fn png(rgba: [u8; 4]) -> Vec<u8> {
    let image = image::RgbaImage::from_pixel(4, 2, image::Rgba(rgba));
    let mut png = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    png
}

/// A GLB of the built-in plane whose material has an embedded base color and a packed
/// metallic (blue) / roughness (green) texture.
fn pbr_glb(base_color: &[u8], metallic_roughness: &[u8]) -> Vec<u8> {
    let mesh = primitives::builtin("plane").unwrap();
    let mut bin = Vec::new();
    let mut view = |bytes: &[u8]| {
        let offset = bin.len();
        bin.extend_from_slice(bytes);
        bin.resize(bin.len().next_multiple_of(4), 0);
        json!({ "buffer": 0, "byteOffset": offset, "byteLength": bytes.len() })
    };
    let floats = |v: Vec<f32>| v.iter().flat_map(|x| x.to_le_bytes()).collect::<Vec<u8>>();
    let views = [
        view(&floats(mesh.positions.iter().flat_map(|p| [p.x, p.y, p.z]).collect())),
        view(&floats(mesh.uvs.iter().flat_map(|uv| [uv.x, uv.y]).collect())),
        view(&mesh.indices.iter().flat_map(|i| i.to_le_bytes()).collect::<Vec<u8>>()),
        view(base_color),
        view(metallic_roughness),
    ];
    let (lo, hi) = mesh.bounds();
    let doc = json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "name": "Box", "mesh": 0 }],
        "meshes": [{ "primitives": [{ "attributes": { "POSITION": 0, "TEXCOORD_0": 1 }, "indices": 2, "material": 0 }] }],
        "materials": [{ "name": "Rust", "pbrMetallicRoughness": { "baseColorTexture": { "index": 0 }, "metallicRoughnessTexture": { "index": 1 } } }],
        "textures": [{ "source": 0 }, { "source": 1 }],
        "images": [{ "bufferView": 3, "mimeType": "image/png" }, { "bufferView": 4, "mimeType": "image/png" }],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": views,
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": mesh.positions.len(), "type": "VEC3", "min": [lo.x, lo.y, lo.z], "max": [hi.x, hi.y, hi.z] },
            { "bufferView": 1, "componentType": 5126, "count": mesh.positions.len(), "type": "VEC2" },
            { "bufferView": 2, "componentType": 5125, "count": mesh.indices.len(), "type": "SCALAR" }
        ]
    });
    let mut text = serde_json::to_vec(&doc).unwrap();
    text.resize(text.len().next_multiple_of(4), b' ');
    let mut glb = Vec::new();
    glb.extend_from_slice(b"glTF");
    glb.extend_from_slice(&2u32.to_le_bytes());
    glb.extend_from_slice(&((12 + 8 + text.len() + 8 + bin.len()) as u32).to_le_bytes());
    for (chunk, kind) in [(&text, b"JSON"), (&bin, b"BIN\0")] {
        glb.extend_from_slice(&(chunk.len() as u32).to_le_bytes());
        glb.extend_from_slice(kind);
        glb.extend_from_slice(chunk);
    }
    glb
}

/// The first texel of the image resource `channel` of `fill` reads.
fn texel(s: &Session, sources: &std::collections::BTreeMap<Channel, Source>, channel: Channel) -> [u8; 4] {
    let Some(Source::Image { name, .. }) = sources.get(&channel) else { panic!("{channel:?} is not an image: {sources:?}") };
    let bytes = s.doc.resources.get(name).unwrap_or_else(|| panic!("resource {name} is missing"));
    let img = image::load_from_memory(bytes).unwrap().into_rgba8();
    assert_eq!(img.dimensions(), (4, 2), "{channel:?}");
    img.get_pixel(0, 0).0
}

fn assert_material_fill(s: &Session) {
    let set = s.doc.project.texture_set(0).unwrap();
    let fill = set
        .stack
        .iter()
        .find_map(|id| match &s.doc.project.layer(*id).unwrap().kind {
            LayerKind::Fill(f) => Some(f.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no fill layer from the embedded textures: {:?}", set.stack));
    assert_eq!(texel(s, &fill.sources, Channel::BaseColor), [200, 90, 40, 255]);
    assert_eq!(texel(s, &fill.sources, Channel::Roughness)[0], 160);
    assert_eq!(texel(s, &fill.sources, Channel::Metallic)[0], 220);
}

#[test]
fn glb_embedded_pbr_textures_become_a_fill_layer() {
    let glb = pbr_glb(&png([200, 90, 40, 255]), &png([0, 160, 220, 255]));
    let s = Session::from_mesh_bytes("rost.glb", &glb, 256, "rost").unwrap();
    assert_material_fill(&s);
    let reopened = Session::from_project_bytes(&s.to_project_bytes().unwrap()).unwrap();
    assert_material_fill(&reopened);
}
