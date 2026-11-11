use texturecraft_geom::glam::{Vec2, Vec3, Vec4};

use super::*;

/// A GLB with one mesh of two primitives (materials "Red" and an unnamed one), placed under a
/// mirrored parent node (negative determinant) and a translated child.
fn glb(with_normals: bool, with_uvs: bool) -> Vec<u8> {
    let mut bin = Vec::new();
    for p in [[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
        bin.extend(p.iter().flat_map(|v| v.to_le_bytes()));
    }
    for _ in 0..3 {
        bin.extend([0.0f32, 0.0, 1.0].iter().flat_map(|v| v.to_le_bytes()));
    }
    for uv in [[0.0f32, 1.0], [1.0, 1.0], [0.0, 0.0]] {
        bin.extend(uv.iter().flat_map(|v| v.to_le_bytes()));
    }
    for i in [0u16, 1, 2, 0] {
        bin.extend(i.to_le_bytes());
    }
    let mut attrs = serde_json::json!({ "POSITION": 0 });
    if with_normals {
        attrs["NORMAL"] = 1.into();
    }
    if with_uvs {
        attrs["TEXCOORD_0"] = 2.into();
    }
    let json = serde_json::json!({
        "asset": { "version": "2.0" },
        "scene": 0,
        "scenes": [{ "nodes": [0] }],
        "nodes": [{ "scale": [-1.0, 1.0, 1.0], "children": [1] }, { "translation": [1.0, 0.0, 0.0], "mesh": 0 }],
        "meshes": [{ "primitives": [
            { "attributes": attrs, "indices": 3, "material": 0 },
            { "attributes": attrs, "indices": 3, "material": 1 }
        ] }],
        "materials": [{ "name": "Red" }, {}],
        "buffers": [{ "byteLength": bin.len() }],
        "bufferViews": [
            { "buffer": 0, "byteOffset": 0, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 36, "byteLength": 36 },
            { "buffer": 0, "byteOffset": 72, "byteLength": 24 },
            { "buffer": 0, "byteOffset": 96, "byteLength": 6 }
        ],
        "accessors": [
            { "bufferView": 0, "componentType": 5126, "count": 3, "type": "VEC3", "min": [0.0, 0.0, 0.0], "max": [1.0, 1.0, 0.0] },
            { "bufferView": 1, "componentType": 5126, "count": 3, "type": "VEC3" },
            { "bufferView": 2, "componentType": 5126, "count": 3, "type": "VEC2" },
            { "bufferView": 3, "componentType": 5123, "count": 3, "type": "SCALAR" }
        ]
    });
    let mut js = serde_json::to_vec(&json).unwrap();
    while !js.len().is_multiple_of(4) {
        js.push(b' ');
    }
    let total = 12 + 8 + js.len() + 8 + bin.len();
    let mut out = Vec::new();
    out.extend(b"glTF");
    out.extend(2u32.to_le_bytes());
    out.extend((total as u32).to_le_bytes());
    out.extend((js.len() as u32).to_le_bytes());
    out.extend(b"JSON");
    out.extend(&js);
    out.extend((bin.len() as u32).to_le_bytes());
    out.extend(b"BIN\0");
    out.extend(&bin);
    out
}

fn base64(data: &[u8]) -> String {
    const T: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut s = String::new();
    for c in data.chunks(3) {
        let b = [c[0], *c.get(1).unwrap_or(&0), *c.get(2).unwrap_or(&0)];
        let n = (b[0] as u32) << 16 | (b[1] as u32) << 8 | b[2] as u32;
        for k in 0..4 {
            s.push(if k <= c.len() { T[(n >> (18 - 6 * k) & 63) as usize] as char } else { '=' });
        }
    }
    s
}

/// The same scene as an embedded .gltf (data-URI buffer).
fn embedded_gltf() -> Vec<u8> {
    let g = glb(true, true);
    let jlen = u32::from_le_bytes(g[12..16].try_into().unwrap()) as usize;
    let mut json: serde_json::Value = serde_json::from_slice(&g[20..20 + jlen]).unwrap();
    let bin = &g[20 + jlen + 8..];
    json["buffers"][0]["uri"] = format!("data:application/octet-stream;base64,{}", base64(bin)).into();
    serde_json::to_vec(&json).unwrap()
}

fn assert_outward(m: &Mesh) {
    for t in 0..m.triangle_count() {
        let [a, b, c] = m.tri(t).unwrap();
        let face = (m.positions[b] - m.positions[a]).cross(m.positions[c] - m.positions[a]);
        assert!(face.dot(m.normals[a]) > 0.0, "triangle {t} winding disagrees with its normal");
    }
}

#[test]
fn glb_two_materials_mirrored_node() {
    let im = import_mesh_bytes("scene.glb", &glb(true, true)).unwrap();
    let m = &im.mesh;
    assert_eq!(m.materials, vec!["Red".to_string(), "Material_1".to_string()]);
    assert_eq!(m.submeshes.len(), 2);
    assert_eq!((m.submeshes[1].start, m.submeshes[1].count), (3, 3));
    assert_eq!(im.report.triangles, 2);
    assert!(!im.report.uvs_generated);
    // Translation, then mirror in x.
    assert!(m.positions.contains(&Vec3::new(-1.0, 0.0, 0.0)));
    assert!(m.positions.contains(&Vec3::new(-2.0, 0.0, 0.0)));
    for n in &m.normals {
        assert!((*n - Vec3::Z).length() < 1e-5, "{n}");
    }
    assert_outward(m);
    assert!(m.uvs.contains(&Vec2::new(0.0, 1.0)));
    m.validate().unwrap();
}

#[test]
fn gltf_embedded_buffers_and_missing_attributes() {
    let im = import_mesh_bytes("scene.gltf", &embedded_gltf()).unwrap();
    assert_eq!(im.mesh.triangle_count(), 2);
    // No normals: flat normals with consistent winding; no UVs: automatic unwrap.
    let im = import_mesh_bytes("x.glb", &glb(false, false)).unwrap();
    assert!(im.report.uvs_generated);
    assert_outward(&im.mesh);
    assert!(im.mesh.uvs.iter().all(|uv| (0.0..=1.0).contains(&uv.x) && (0.0..=1.0).contains(&uv.y)));
    assert_eq!(im.report.uv_out_of_range, 0);
}

#[test]
fn gltf_external_buffer_needs_a_path() {
    let mut json: serde_json::Value = serde_json::from_slice(&embedded_gltf()).unwrap();
    json["buffers"][0]["uri"] = "scene.bin".into();
    let err = import_mesh_bytes("scene.gltf", &serde_json::to_vec(&json).unwrap()).unwrap_err();
    assert!(matches!(err, FormatError::Unsupported(_)), "{err}");
}

#[cfg(not(target_arch = "wasm32"))]
#[test]
fn gltf_external_buffer_from_disk() {
    let g = glb(true, true);
    let jlen = u32::from_le_bytes(g[12..16].try_into().unwrap()) as usize;
    let dir = std::env::temp_dir().join(format!("tc-formats-gltf-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    std::fs::write(dir.join("my scene.bin"), &g[20 + jlen + 8..]).unwrap();
    let mut json: serde_json::Value = serde_json::from_slice(&g[20..20 + jlen]).unwrap();
    json["buffers"][0]["uri"] = "my%20scene.bin".into();
    std::fs::write(dir.join("scene.gltf"), serde_json::to_vec(&json).unwrap()).unwrap();
    let im = import_mesh(&dir.join("scene.gltf")).unwrap();
    assert_eq!(im.mesh.triangle_count(), 2);
    // A buffer that is a directory is refused before reading.
    json["buffers"][0]["uri"] = ".".into();
    std::fs::write(dir.join("bad.gltf"), serde_json::to_vec(&json).unwrap()).unwrap();
    assert!(import_mesh(&dir.join("bad.gltf")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

/// A .gltf whose buffer and base color image are files next to it becomes one self-contained
/// file that imports with its texture.
#[test]
fn gltf_external_files_are_folded_in() {
    let g = glb(true, true);
    let jlen = u32::from_le_bytes(g[12..16].try_into().unwrap()) as usize;
    let mut json: serde_json::Value = serde_json::from_slice(&g[20..20 + jlen]).unwrap();
    json["buffers"][0]["uri"] = "scene.bin".into();
    json["images"] = serde_json::json!([{ "uri": "tex/base%20color.png" }]);
    json["textures"] = serde_json::json!([{ "source": 0 }]);
    json["materials"][0]["pbrMetallicRoughness"] = serde_json::json!({ "baseColorTexture": { "index": 0 } });
    let gltf = serde_json::to_vec(&json).unwrap();
    let mut png = std::io::Cursor::new(Vec::new());
    image::write_buffer_with_format(&mut png, &[10u8, 200, 30].repeat(4), 2, 2, image::ColorType::Rgb8, image::ImageFormat::Png).unwrap();
    let files = [("scene.bin", g[20 + jlen + 8..].to_vec()), ("tex/base color.png", png.into_inner())];
    let mut read = |p: &str| files.iter().find(|(n, _)| *n == p).map(|(_, b)| b.clone());
    let f = self_contained_mesh("dir/scene.gltf", gltf.clone(), &mut read).unwrap();
    assert_eq!((f.name.as_str(), f.warnings.len()), ("dir/scene.gltf", 0));
    let hp = import_high_poly_bytes(&f.name, &f.bytes).unwrap();
    let red = hp.mesh.materials.iter().position(|m| m == "Red").unwrap();
    assert_eq!(hp.base_color[red].as_ref().map(|c| (c.width, c.texels[0])), Some((2, [10, 200, 30, 255])));
    // A missing image is named; the mesh still imports, untextured.
    let f = self_contained_mesh("scene.gltf", gltf, &mut |p: &str| (p == "scene.bin").then(|| files[0].1.clone())).unwrap();
    assert_eq!(f.warnings, vec!["scene.gltf: tex/base color.png was not found next to the mesh".to_string()]);
    assert!(import_high_poly_bytes(&f.name, &f.bytes).unwrap().base_color[red].is_none());
}

const OBJ: &str = "mtllib missing.mtl
o quad
v 0 0 0
v 1 0 0
v 1 1 0
v 0 1 0
v 2 0 0
vt 0 0
vt 1 0
vt 1 1
vt 0 1
vn 0 0 1
usemtl Wood
f 1/1/1 2/2/1 3/3/1
usemtl Metal
f 1/1/1 3/3/1 4/4/1
usemtl Wood
f 2/2/1 5/2/1 3/3/1 4/4/1
";

#[test]
fn obj_flips_v_and_keeps_usemtl_materials() {
    let im = import_mesh_bytes("a.obj", OBJ.as_bytes()).unwrap();
    let m = &im.mesh;
    assert_eq!(m.materials, vec!["Wood".to_string(), "Metal".to_string()]);
    // Wood: one triangle + a fan-triangulated quad; Metal: one triangle.
    assert_eq!(m.submeshes.iter().map(|s| s.count / 3).collect::<Vec<_>>(), vec![3, 1]);
    for (p, uv) in m.positions.iter().zip(&m.uvs) {
        if *p == Vec3::ZERO {
            assert_eq!(*uv, Vec2::new(0.0, 1.0), "OBJ v must be flipped");
        }
        if *p == Vec3::new(1.0, 1.0, 0.0) {
            assert_eq!(*uv, Vec2::new(1.0, 0.0));
        }
    }
    assert_outward(m);
    m.validate().unwrap();
}

#[test]
fn obj_without_uvs_or_normals_is_completed() {
    let im = import_mesh_bytes("b.OBJ", b"v 0 0 0\nv 1 0 0\nv 0 1 0\nv 0 0 1\nf 1 3 2\nf 1 2 4\nf 1 4 3\nf 2 3 4\n").unwrap();
    assert!(im.report.uvs_generated);
    assert_eq!(im.mesh.materials, vec!["Default".to_string()]);
    im.mesh.validate().unwrap();
    // Smooth normals over welded positions: every copy of a corner shares one normal.
    let n0: Vec<Vec3> = im.mesh.positions.iter().zip(&im.mesh.normals).filter(|(p, _)| **p == Vec3::ZERO).map(|(_, n)| *n).collect();
    assert!(n0.len() > 1 && n0.iter().all(|n| (*n - n0[0]).length() < 1e-5));
}

#[test]
fn obj_with_some_objects_having_uvs_preserves_uvs_without_full_unwrap() {
    let obj = "o Body
v 0 0 0
v 1 0 0
v 1 1 0
vt 0 0
vt 1 0
vt 1 1
f 1/1 2/2 3/3
o Helper
v 2 0 0
v 3 0 0
v 3 1 0
f 4 5 6
";
    let im = import_mesh_bytes("character.obj", obj.as_bytes()).unwrap();
    assert!(!im.report.uvs_generated);
    assert_eq!(im.mesh.uvs[0], Vec2::new(0.0, 1.0));
    assert_eq!(im.mesh.uvs[1], Vec2::new(1.0, 1.0));
    assert_eq!(im.mesh.uvs[2], Vec2::new(1.0, 0.0));
    assert_eq!(im.mesh.uvs[3], Vec2::ZERO);
    assert_eq!(im.mesh.uvs[4], Vec2::ZERO);
    assert_eq!(im.mesh.uvs[5], Vec2::ZERO);
    assert!(im.report.warnings.iter().any(|w| w.contains("some objects have no UVs")));
}

#[test]
fn polygons_keep_their_edges_for_the_wireframe() {
    // An OBJ quad and pentagon (with UVs, so no unwrap splits them), a triangle, and a PLY quad.
    let obj = "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 2 0 0\nv 3 0 0\nv 3 1 0\nv 2.5 2 0\nv 2 1 0\nvt 0 0\n\
               f 1/1 2/1 3/1 4/1\nf 5/1 6/1 7/1 8/1 9/1\nf 1/1 2/1 5/1\n";
    let m = import_mesh_bytes("p.obj", obj.as_bytes()).unwrap().mesh;
    assert_eq!(m.triangle_count(), 2 + 3 + 1);
    assert_eq!(m.wire_edges()[0].len(), 4 + 5 + 2, "the triangle adds edges 1-5 and 2-5");
    let ply = import_mesh_bytes("a.ply", ply_ascii().as_bytes()).unwrap().mesh;
    assert_eq!(ply.wire_edges()[0].len(), 4);
    // Triangles alone keep every edge.
    let tris = import_mesh_bytes("t.obj", b"v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvt 0 0\nf 1/1 2/1 3/1\nf 1/1 3/1 4/1\n").unwrap().mesh;
    assert!(tris.diagonals.is_empty());
    assert_eq!(tris.wire_edges()[0].len(), 5);
}

fn ply_ascii() -> String {
    "ply\nformat ascii 1.0\ncomment test\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nproperty float s\nproperty float t\n\
     element face 1\nproperty list uchar int vertex_indices\nend_header\n\
     0 0 0 0 0\n1 0 0 1 0\n1 1 0 1 1\n0 1 0 0 1\n4 0 1 2 3\n"
        .to_string()
}

fn ply_binary(big_endian: bool) -> Vec<u8> {
    let enc = if big_endian { "binary_big_endian" } else { "binary_little_endian" };
    let mut b = format!(
        "ply\r\nformat {enc} 1.0\r\nelement vertex 4\r\nproperty double x\r\nproperty double y\r\nproperty double z\r\nproperty float nx\r\nproperty float ny\r\nproperty float nz\r\n\
         property uchar red\r\nelement face 1\r\nproperty list uchar uint vertex_indices\r\nend_header\r\n"
    )
    .into_bytes();
    let f64b = |v: f64| if big_endian { v.to_be_bytes().to_vec() } else { v.to_le_bytes().to_vec() };
    let f32b = |v: f32| if big_endian { v.to_be_bytes().to_vec() } else { v.to_le_bytes().to_vec() };
    let u32b = |v: u32| if big_endian { v.to_be_bytes().to_vec() } else { v.to_le_bytes().to_vec() };
    for p in [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]] {
        for v in p {
            b.extend(f64b(v));
        }
        for v in [0.0, 0.0, 1.0] {
            b.extend(f32b(v));
        }
        b.push(200);
    }
    b.push(4);
    for i in [0, 1, 2, 3] {
        b.extend(u32b(i));
    }
    b
}

#[test]
fn ply_ascii_and_binary() {
    let a = import_mesh_bytes("a.ply", ply_ascii().as_bytes()).unwrap();
    assert_eq!(a.mesh.triangle_count(), 2);
    assert!(!a.report.uvs_generated);
    // PLY t is bottom-up like OBJ.
    assert_eq!(a.mesh.uvs[0], Vec2::new(0.0, 1.0));
    for be in [false, true] {
        let b = import_mesh_bytes("b.ply", &ply_binary(be)).unwrap();
        assert_eq!(b.mesh.triangle_count(), 2);
        assert!(b.report.uvs_generated);
        assert!(b.mesh.positions.contains(&Vec3::new(1.0, 1.0, 0.0)));
        assert_outward(&b.mesh);
    }
}

fn stl_binary() -> Vec<u8> {
    let mut b = vec![0u8; 80];
    b.extend(2u32.to_le_bytes());
    for tri in [[[0.0f32, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]], [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]] {
        b.extend([0.0f32; 3].iter().flat_map(|v| v.to_le_bytes()));
        for v in tri {
            b.extend(v.iter().flat_map(|x| x.to_le_bytes()));
        }
        b.extend([0u8, 0]);
    }
    b
}

const STL_ASCII: &str = "solid test
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 1 1 0
 endloop
endfacet
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 1 0
  vertex 0 1 0
 endloop
endfacet
endsolid test
";

#[test]
fn stl_ascii_and_binary() {
    for (name, bytes) in [("a.stl", STL_ASCII.as_bytes().to_vec()), ("b.STL", stl_binary())] {
        let im = import_mesh_bytes(name, &bytes).unwrap();
        assert_eq!(im.mesh.triangle_count(), 2, "{name}");
        assert!(im.report.uvs_generated);
        assert_outward(&im.mesh);
    }
    // A binary file whose header starts with "solid" is still binary.
    let mut b = stl_binary();
    b[..5].copy_from_slice(b"solid");
    assert_eq!(import_mesh_bytes("c.stl", &b).unwrap().mesh.triangle_count(), 2);
}

/// Every prefix and some corruptions of a valid file: errors, never panics (and never an
/// error from the panic guard, which would mean a parser panicked).
pub(crate) fn hammer(name: &str, good: &[u8]) {
    let check = |bytes: &[u8]| {
        if std::env::var("TC_TRACE").is_ok() {
            std::fs::write(std::env::temp_dir().join("tc_last_case.bin"), bytes).unwrap();
            eprintln!("{name} {}", bytes.len());
        }
        if let Err(e) = import_mesh_bytes(name, bytes) {
            assert!(!e.to_string().contains("parser failed"), "{name}: a parser panicked on {} bytes", bytes.len());
        }
    };
    for len in 0..good.len() {
        check(&good[..len]);
    }
    for i in 0..good.len() {
        let mut b = good.to_vec();
        b[i] ^= 0xA5;
        check(&b);
        b[i] = 0xFF;
        check(&b);
    }
    assert!(import_mesh_bytes(name, &[]).is_err());
    assert!(import_mesh_bytes(name, b"\x00\x01garbage\xff\xfe").is_err());
}

#[test]
fn malformed_meshes_are_errors() {
    hammer("x.glb", &glb(true, true));
    hammer("x.gltf", &embedded_gltf());
    hammer("x.obj", OBJ.as_bytes());
    hammer("x.ply", ply_ascii().as_bytes());
    hammer("x.ply", &ply_binary(false));
    hammer("x.ply", &ply_binary(true));
    hammer("x.stl", STL_ASCII.as_bytes());
    hammer("x.stl", &stl_binary());
    assert!(matches!(import_mesh_bytes("x.blend", b"whatever"), Err(FormatError::Unsupported(_))));
    // Hostile counts never allocate their claim.
    let huge = "ply\nformat binary_little_endian 1.0\nelement vertex 4000000000\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    assert!(import_mesh_bytes("h.ply", huge.as_bytes()).is_err());
    let mut stl = vec![0u8; 80];
    stl.extend(u32::MAX.to_le_bytes());
    assert!(import_mesh_bytes("h.stl", &stl).is_err());
    // Meshes of only degenerate triangles are refused instead of crashing tangent generation.
    assert!(import_mesh_bytes("p.obj", b"v 1 1 1\nv 1 1 1\nv 1 1 1\nf 1 2 3\n").is_err());
    assert!(import_mesh_bytes("q.obj", b"v 0 0 0\nv 0 0 0\nv 0 1 0\nf 1 2 3\nf 2 1 3\n").is_err());
}

#[test]
fn report_counts_uv_problems() {
    let obj = "v 0 0 0\nv 1 0 0\nv 0 1 0\nv 1 1 0\nvt 0 0\nvt 2 0\nvt 0 1\nvt 0 0\nf 1/1 2/2 3/3\nf 2/4 4/4 3/4\n";
    let r = import_mesh_bytes("r.obj", obj.as_bytes()).unwrap().report;
    assert_eq!(r.uv_out_of_range, 1);
    assert_eq!(r.zero_area_uv_triangles, 1);
    assert!(!r.warnings.is_empty());
}

#[test]
fn gltf_vertex_colors_and_node_names() {
    // The embedded scene plus a COLOR_0 (RGBA float) on the first primitive and a named node.
    let mut json: serde_json::Value = serde_json::from_slice(&embedded_gltf()).unwrap();
    let colors: Vec<u8> = [[1.0f32, 0.0, 0.0, 0.5]; 3].iter().flatten().flat_map(|v| v.to_le_bytes()).collect();
    json["buffers"]
        .as_array_mut()
        .unwrap()
        .push(serde_json::json!({ "byteLength": 48, "uri": format!("data:application/octet-stream;base64,{}", base64(&colors)) }));
    json["bufferViews"].as_array_mut().unwrap().push(serde_json::json!({ "buffer": 1, "byteOffset": 0, "byteLength": 48 }));
    json["accessors"].as_array_mut().unwrap().push(serde_json::json!({ "bufferView": 4, "componentType": 5126, "count": 3, "type": "VEC4" }));
    json["meshes"][0]["primitives"][0]["attributes"]["COLOR_0"] = 4.into();
    json["nodes"][1]["name"] = "Body_low".into();
    let m = import_mesh_bytes("scene.gltf", &serde_json::to_vec(&json).unwrap()).unwrap().mesh;
    assert_eq!(m.colors.len(), m.vertex_count());
    let red = Vec4::new(1.0, 0.0, 0.0, 0.5);
    // The colored primitive's vertices are red, the other primitive's white.
    let first = &m.indices[m.submeshes[0].start as usize..][..3];
    assert!(first.iter().all(|i| m.colors[*i as usize] == red), "{:?}", m.colors);
    assert!(m.colors.contains(&Vec4::ONE));
    assert!(m.submeshes.iter().all(|s| s.name == "Body_low"));
    // Unnamed parts take the file stem; a mesh without COLOR_0 has no colors.
    let plain = import_mesh_bytes("scene.glb", &glb(true, true)).unwrap().mesh;
    assert!(plain.submeshes.iter().all(|s| s.name == "scene") && plain.colors.is_empty());
}

#[test]
fn obj_and_ply_vertex_colors_and_objects() {
    let obj = "o Wheel_low\nv 0 0 0 1 0 0\nv 1 0 0 1 0 0\nv 0 1 0 1 0 0\nf 1 2 3\no Body_low\nv 0 0 1 0 1 0\nv 1 0 1 0 1 0\nv 0 1 1 0 1 0\nf 4 5 6\n";
    let m = import_mesh_bytes("car.obj", obj.as_bytes()).unwrap().mesh;
    assert_eq!(m.submeshes.iter().map(|s| s.name.as_str()).collect::<Vec<_>>(), ["Wheel_low", "Body_low"]);
    assert_eq!(m.materials.len(), 1, "one material, two parts");
    assert_eq!(m.colors.len(), m.vertex_count());
    for (p, c) in m.positions.iter().zip(&m.colors) {
        assert_eq!(*c, if p.z == 0.0 { Vec4::new(1.0, 0.0, 0.0, 1.0) } else { Vec4::new(0.0, 1.0, 0.0, 1.0) });
    }
    let ply = "ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\n\
               element face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0 255 0 51\n1 0 0 255 0 51\n0 1 0 255 0 51\n3 0 1 2\n";
    let m = import_mesh_bytes("rock_high.ply", ply.as_bytes()).unwrap().mesh;
    assert!(m.colors.iter().all(|c| (*c - Vec4::new(1.0, 0.0, 0.2, 1.0)).length() < 1e-6), "{:?}", m.colors);
    assert_eq!(m.submeshes[0].name, "rock_high");
}

#[test]
fn high_poly_import_skips_unwrap_and_rejects_garbage() {
    let m = import_high_poly_bytes("x.glb", &glb(false, false)).unwrap().mesh;
    m.validate().unwrap();
    assert!(m.uvs.iter().all(|uv| *uv == Vec2::ZERO), "no automatic unwrap");
    assert_eq!(m.tangents.len(), m.vertex_count());
    assert!(import_high_poly_bytes("x.glb", b"glTF\x02\0\0\0garbage").is_err());
    assert!(import_high_poly_bytes("x.obj", b"v 0 0 0\nv nan 1 0\nv 0 1 0\nf 1 2 3\n").is_err());
    assert!(import_high_poly_bytes("x.ply", b"ply\nformat ascii 1.0\nelement vertex 99999999\n").is_err());
    assert!(import_high_poly_bytes("x.txt", b"").is_err());
}

/// The embedded scene plus a perspective camera looking at the triangle along -Z and an unnamed
/// orthographic one looking straight down.
fn gltf_with_cameras() -> Vec<u8> {
    let mut json: serde_json::Value = serde_json::from_slice(&embedded_gltf()).unwrap();
    json["cameras"] = serde_json::json!([
        { "name": "Shot", "type": "perspective", "perspective": { "yfov": 0.8, "znear": 0.1 } },
        { "type": "orthographic", "orthographic": { "xmag": 1.0, "ymag": 0.5, "znear": 0.1, "zfar": 10.0 } }
    ]);
    let nodes = json["nodes"].as_array_mut().unwrap();
    nodes.push(serde_json::json!({ "camera": 0, "translation": [-1.5, 0.5, 4.0] }));
    nodes.push(serde_json::json!({ "camera": 1, "translation": [-1.5, 5.5, 0.0], "rotation": [-std::f32::consts::FRAC_1_SQRT_2, 0.0, 0.0, std::f32::consts::FRAC_1_SQRT_2] }));
    json["scenes"][0]["nodes"] = serde_json::json!([0, 2, 3]);
    serde_json::to_vec(&json).unwrap()
}

#[test]
fn gltf_cameras_become_orbit_cameras() {
    let im = import_mesh_bytes("scene.gltf", &gltf_with_cameras()).unwrap();
    let [shot, top] = im.cameras.as_slice() else { panic!("{:?}", im.cameras) };
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert_eq!(shot.name, "Shot");
    assert!(!shot.ortho && near(shot.fov, 0.8f32.to_degrees()) && near(shot.distance, 4.0), "{shot:?}");
    assert!(near(shot.yaw, 0.0) && near(shot.pitch, 0.0) && shot.target.iter().zip([-1.5, 0.5, 0.0]).all(|(a, b)| near(*a, b)), "{shot:?}");
    assert_eq!(top.name, "Camera 2");
    assert!(top.ortho && near(top.distance, 5.0) && near(top.fov, 2.0 * 0.1f32.atan().to_degrees()), "{top:?}");
    assert!(near(top.pitch, 89f32.to_radians()), "looking down, pitch clamped: {top:?}");
    assert!(import_mesh_bytes("scene.gltf", &embedded_gltf()).unwrap().cameras.is_empty());
}

#[test]
fn obj_without_object_name_takes_file_stem() {
    let name = |file: &str, src: &str| import_mesh_bytes(file, src.as_bytes()).unwrap().mesh.submeshes[0].name.clone();
    let body = "v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n";
    assert_eq!(name("rock.obj", body), "rock");
    assert_eq!(name("rock.obj", &format!("o Boulder\n{body}")), "Boulder");
    assert_eq!(name("rock.obj", &format!("g Pebble\n{body}")), "Pebble");
}

#[test]
fn ply_face_element_before_vertex_element() {
    let head = "ply\nformat ascii 1.0\nelement face 1\nproperty list uchar int vertex_indices\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nend_header\n";
    let ok = format!("{head}3 0 1 2\n0 0 0\n1 0 0\n0 1 0\n");
    let mesh = import_mesh_bytes("a.ply", ok.as_bytes()).unwrap().mesh;
    assert_eq!(mesh.positions.len(), 3);
    let bad = format!("{head}3 0 1 5\n0 0 0\n1 0 0\n0 1 0\n");
    assert!(import_mesh_bytes("a.ply", bad.as_bytes()).is_err());
}
