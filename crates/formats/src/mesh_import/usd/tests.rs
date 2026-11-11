//! USD import from hand-written usda, and usda / usdc / usdz round trips through the exporter.

use std::io::Read as _;

use texturecraft_geom::Mesh;
use texturecraft_geom::glam::{Vec2, Vec3};

use super::codec;
use crate::mesh_import::tests::hammer;
use crate::{ExportedFile, UsdFormat, import_high_poly_bytes, import_mesh_bytes, self_contained_mesh, usd_bytes};

/// Two meshes in a Z-up, centimeter stage: a quad + triangle (`Red`, with a `Blue` subset on
/// the triangle; faceVarying indexed st and normals) under a translated root, and a triangle
/// under a rotated, scaled child that inherits `Green`. Undefined, inactive and class prims
/// are not drawn.
const SCENE: &str = r#"#usda 1.0
(
    defaultPrim = "World"
    doc = """A test "scene"
    over two lines"""
    metersPerUnit = 0.01
    upAxis = "Z"
)

def Xform "World" (
    kind = "assembly"
)
{
    double3 xformOp:translate = (0, 0, 100)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Quad" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [4, 3]
        int[] faceVertexIndices = [0, 1, 2, 3, 1, 4, 2]
        point3f[] points = [(0, 0, 0), (100, 0, 0), (100, 100, 0), (0, 100, 0), (200, 0, 0)]
        normal3f[] normals = [(0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1)] (
            interpolation = "faceVarying"
        )
        texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
            interpolation = "faceVarying"
        )
        int[] primvars:st:indices = [0, 1, 2, 3, 0, 1, 2]
        uniform token subdivisionScheme = "none"
        rel material:binding = </World/Looks/Red>

        def GeomSubset "blue" (
            prepend apiSchemas = ["MaterialBindingAPI"]
        )
        {
            uniform token elementType = "face"
            uniform token familyName = "materialBind"
            int[] indices = [1]
            rel material:binding = </World/Looks/Blue>
        }
    }

    def Xform "Child"
    {
        rel material:binding = </World/Looks/Green>
        float3 xformOp:rotateXYZ = (0, 0, 90)
        float3 xformOp:scale = (2, 2, 2)
        uniform token[] xformOpOrder = ["xformOp:rotateXYZ", "xformOp:scale"]

        def Mesh "Tri"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (50, 0, 0), (0, 50, 0)]
            texCoord2f[] primvars:st = [(0, 0), (1, 0), (0, 1)] (
                interpolation = "vertex"
            )
        }
    }

    def Scope "Looks"
    {
        def Material "Red" {}
        def Material "Blue" {}
        def Material "Green"
        {
            token outputs:surface.connect = </World/Looks/Green/PBR.outputs:surface>

            def Shader "PBR"
            {
                uniform token info:id = "UsdPreviewSurface"
                color3f inputs:diffuseColor = (0.5, 0.25, 1)
                token outputs:surface
            }
        }
    }

    over "Ghost"
    {
        def Mesh "Hidden"
        {
            int[] faceVertexCounts = [3]
            int[] faceVertexIndices = [0, 1, 2]
            point3f[] points = [(0, 0, 0), (1, 0, 0), (0, 1, 0)]
        }
    }

    def Mesh "Off" (
        active = false
    )
    {
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 1, 2]
        point3f[] points = [(0, 0, 0), (9, 0, 0), (0, 9, 0)]
    }
}

class Mesh "Template"
{
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0, 1, 2]
    point3f[] points = [(0, 0, 0), (9, 0, 0), (0, 9, 0)]
}

def Camera "Shot"
{
    float focalLength = 35
    float verticalAperture = 20.955
    double3 xformOp:translate = (0, -500, 100)
    float3 xformOp:rotateXYZ = (90, 0, 0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
}

def Camera "Top"
{
    token projection = "orthographic"
    float verticalAperture = 40
    double3 xformOp:translate = (0, 0, 1000)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}
"#;

/// Internal references (with a hole and a binding inside the referenced class), a
/// time-sampled translate, a matrix op, an external reference and a variant set (skipped with
/// warnings), and a left-handed mesh turned by a quaternion under a reset xform stack.
const FEATURES: &str = r#"#usda 1.0
(
    metersPerUnit = 1
)
# comment
class Xform "Proto"
{
    def Mesh "Leaf" (
        prepend apiSchemas = ["MaterialBindingAPI"]
    )
    {
        int[] faceVertexCounts = [3, 3]
        int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
        int[] holeIndices = [1]
        point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
        rel material:binding = </Looks/Leaf>
    }
}

def Xform "A" (
    prepend references = </Proto>
    instanceable = true
)
{
    matrix4d xformOp:transform = ( (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (10, 0, 0, 1) )
    uniform token[] xformOpOrder = ["xformOp:transform"]
}

def Xform "B" (
    prepend references = [</Proto>, @./other.usda@</X>]
)
{
    float3 xformOp:translate.timeSamples = {
        0: (20, 0, 0),
        10: (30, 0, 0),
    }
    uniform token[] xformOpOrder = ["xformOp:translate"]
    variantSet "look" = {
        "a" {
            def Mesh "V" {}
        }
    }
}

def Xform "Parent"
{
    double3 xformOp:translate = (100, 100, 100)
    uniform token[] xformOpOrder = ["xformOp:translate"]

    def Mesh "Left"
    {
        uniform token orientation = "leftHanded"
        quatf xformOp:orient = (0.70710677, 0, 0, 0.70710677)
        uniform token[] xformOpOrder = ["!resetXformStack!", "xformOp:orient"]
        int[] faceVertexCounts = [3]
        int[] faceVertexIndices = [0, 2, 1]
        point3f[] points = [(0, 0, 5), (1, 0, 5), (0, 1, 5)]
    }
}
"#;

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

/// Bounds of the vertices material `name` uses.
fn material_bounds(m: &Mesh, name: &str) -> (Vec3, Vec3) {
    let k = m.materials.iter().position(|x| x == name).unwrap();
    let mut b = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for sm in m.submeshes.iter().filter(|s| s.material == k) {
        for &i in &m.indices[sm.start as usize..(sm.start + sm.count) as usize] {
            b = (b.0.min(m.positions[i as usize]), b.1.max(m.positions[i as usize]));
        }
    }
    b
}

fn face_normal(m: &Mesh, t: &[u32]) -> Vec3 {
    let p = |i: u32| m.positions[i as usize];
    (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]))
}

#[test]
fn usda_hierarchy_units_axes_and_subsets() {
    let imported = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap();
    let m = imported.mesh;
    assert_eq!(m.triangle_count(), 4);
    assert_eq!(m.materials, ["Red", "Blue", "Green"]);
    assert!(!imported.report.uvs_generated);
    // cm → m, Z up → Y up: (x, y, z) becomes (x, z, -y).
    let (lo, hi) = m.bounds();
    assert!(close(lo, Vec3::new(-1.0, 1.0, -1.0)) && close(hi, Vec3::new(2.0, 1.0, 0.0)), "bounds {lo} {hi}");
    let (r0, r1) = material_bounds(&m, "Red");
    assert!(close(r0, Vec3::new(0.0, 1.0, -1.0)) && close(r1, Vec3::new(1.0, 1.0, 0.0)));
    let (b0, b1) = material_bounds(&m, "Blue");
    assert!(close(b0, Vec3::new(1.0, 1.0, -1.0)) && close(b1, Vec3::new(2.0, 1.0, 0.0)));
    // Rotated 90° about Z and scaled 2×: (50, 0) → (0, 100), (0, 50) → (-100, 0).
    let (g0, g1) = material_bounds(&m, "Green");
    assert!(close(g0, Vec3::new(-1.0, 1.0, -1.0)) && close(g1, Vec3::new(0.0, 1.0, 0.0)), "green {g0} {g1}");
    let mut parts: Vec<&str> = m.submeshes.iter().map(|s| s.name.as_str()).collect();
    parts.dedup();
    assert_eq!(parts, ["Quad", "Tri"]);
    assert!(m.normals.iter().all(|n| close(*n, Vec3::Y)), "normals {:?}", m.normals);
    for t in m.indices.chunks(3) {
        assert!(face_normal(&m, t).dot(Vec3::Y) > 0.0, "winding stays front-facing");
    }
    let mut uvs = m.uvs.clone();
    uvs.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    uvs.dedup();
    assert_eq!(uvs, [Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)]);
    // Cameras: "Shot" stands 5 m back and 1 m up (Z up, cm), turned to look along +Y, which is
    // -Z once converted; "Top" looks straight down orthographically.
    let [shot, top] = imported.cameras.as_slice() else { panic!("{:?}", imported.cameras) };
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert_eq!((shot.name.as_str(), top.name.as_str()), ("Shot", "Top"));
    assert!(!shot.ortho && near(shot.fov, 2.0 * (20.955f32 / 70.0).atan().to_degrees()), "{shot:?}");
    assert!(near(shot.yaw, 0.0) && near(shot.pitch, 0.0) && near(shot.distance, 5.5), "{shot:?}");
    assert!(top.ortho && near(top.pitch, 89f32.to_radians()) && near(top.distance, 9.0), "{top:?}");
}

#[test]
fn usda_references_time_samples_holes_and_orientation() {
    let imported = import_mesh_bytes("features.usda", FEATURES.as_bytes()).unwrap();
    let m = &imported.mesh;
    assert_eq!(m.triangle_count(), 3, "the class itself is not drawn and the hole is skipped");
    assert_eq!(m.materials, ["Leaf", "Default"]);
    let (lo, hi) = material_bounds(m, "Leaf");
    assert!(close(lo, Vec3::new(10.0, 0.0, 0.0)) && close(hi, Vec3::new(21.0, 1.0, 0.0)), "leaf {lo} {hi}");
    // The orient turns +X to +Y; the parent's translate is reset.
    let (lo, hi) = material_bounds(m, "Default");
    assert!(close(lo, Vec3::new(-1.0, 0.0, 5.0)) && close(hi, Vec3::new(0.0, 1.0, 5.0)), "left {lo} {hi}");
    let sm = m.submeshes.iter().find(|s| s.name == "Left").unwrap();
    let t = &m.indices[sm.start as usize..sm.start as usize + 3];
    assert!(face_normal(m, t).z > 0.0, "left-handed winding is reversed");
    let warnings = imported.report.warnings.join("\n");
    assert!(warnings.contains("external references") && warnings.contains("variant sets"), "{warnings}");
}

fn png(rgb: [u8; 3]) -> Vec<u8> {
    let mut out = std::io::Cursor::new(Vec::new());
    image::write_buffer_with_format(&mut out, &rgb.repeat(4), 2, 2, image::ColorType::Rgb8, image::ImageFormat::Png).unwrap();
    out.into_inner()
}

fn file(name: &str, bytes: Vec<u8>) -> ExportedFile {
    ExportedFile { name: name.into(), bytes, width: 2, height: 2, warnings: vec![] }
}

/// Exported, then imported again: same triangles, materials, bounds, UVs and normals.
fn round_trip(m: &Mesh, format: UsdFormat, ext: &str, textures: &[Vec<ExportedFile>]) -> Vec<u8> {
    let out = usd_bytes(m, "Test scene", textures, format).unwrap();
    let back = import_mesh_bytes(&format!("t.{ext}"), &out.bytes).unwrap().mesh;
    assert_eq!(back.triangle_count(), m.triangle_count(), "{ext}");
    assert_eq!(back.materials, m.materials, "{ext}");
    let (a, b) = (m.bounds(), back.bounds());
    assert!(close(a.0, b.0) && close(a.1, b.1), "{ext}");
    let corners = |m: &Mesh| {
        let mut v: Vec<[i64; 8]> = m
            .indices
            .iter()
            .map(|&i| {
                let (p, n, t) = (m.positions[i as usize], m.normals[i as usize], m.uvs[i as usize]);
                [p.x, p.y, p.z, n.x, n.y, n.z, t.x, t.y].map(|x| (x * 1e4).round() as i64)
            })
            .collect();
        v.sort_unstable();
        v
    };
    assert_eq!(corners(&back), corners(m), "{ext}: positions, normals and UVs per corner");
    out.bytes
}

#[test]
fn usda_usdc_and_usdz_round_trip_with_textures() {
    let m = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap().mesh;
    let textures = vec![
        vec![
            file("BaseColor.png", png([200, 10, 10])),
            file("Roughness.png", png([128; 3])),
            file("Normal.png", png([128, 128, 255])),
            file("Emissive.png", png([0; 3])),
            file("Clearcoat.png", png([0; 3])),
            file("ClearcoatRoughness.png", png([30; 3])),
        ],
        vec![
            file("Metallic.png", png([255; 3])),
            file("Opacity.png", png([255; 3])),
            file("Clearcoat.png", png([255; 3])),
            file("ClearcoatRoughness.png", png([30; 3])),
        ],
        vec![],
    ];
    let usda = round_trip(&m, UsdFormat::Usda, "usda", &textures);
    let text = String::from_utf8(usda).unwrap();
    assert!(
        text.contains("UsdPreviewSurface") && text.contains("@./Test_scene_Red_BaseColor.png@") && !text.contains("Emissive.png"),
        "black emissive dropped"
    );
    assert!(
        text.contains("Materials/Blue/Clearcoat.outputs:r>")
            && text.contains("Materials/Blue/ClearcoatRoughness.outputs:r>")
            && !text.contains("Red_Clearcoat"),
        "a coat exports with its roughness; no coat (black) drops both"
    );
    // The primvar reader's varname is a string (usdchecker --arkit refuses a token), in both encodings.
    assert!(text.contains("string inputs:varname = \"st\""), "{text}");
    let usdc = round_trip(&m, UsdFormat::Usdc, "usdc", &textures);
    let layer = super::usdc::read(&usdc, &mut super::Budget::new()).unwrap();
    let varnames: Vec<&super::Prop> = layer.prims.iter().filter_map(|p| p.props.get("inputs:varname")).collect();
    assert!(!varnames.is_empty());
    for v in varnames {
        assert_eq!((v.type_name.as_str(), v.value.clone()), ("string", Some(super::Val::Tok(vec!["st".into()]))));
    }
    let usdz = round_trip(&m, UsdFormat::Usdz, "usdz", &textures);
    let mut zip = zip::ZipArchive::new(std::io::Cursor::new(&usdz)).unwrap();
    let names: Vec<String> = zip.file_names().map(|n| n.unwrap().into_owned()).collect();
    assert_eq!(names.first().map(String::as_str), Some("Test_scene.usdc"));
    assert_eq!(names.len(), 1 + 7, "the layer and every non-black texture");
    for i in 0..zip.len() {
        let mut f = zip.by_index(i).unwrap();
        assert_eq!(f.compression(), zip::CompressionMethod::Stored);
        assert_eq!(f.data_start().unwrap() % 64, 0, "usdz entries are 64-byte aligned");
        let mut data = Vec::new();
        f.read_to_end(&mut data).unwrap();
        if f.name().unwrap().ends_with(".png") {
            assert!(image::load_from_memory(&data).is_ok());
        }
    }
    // usda and usdc keep their textures for writing next to the layer.
    let out = usd_bytes(&m, "Test scene", &textures, UsdFormat::Usda).unwrap();
    assert_eq!(out.textures.len(), 7);
    assert!(usd_bytes(&Mesh::default(), "x", &[], UsdFormat::Usdc).is_err());
}

/// UsdPreviewSurface base colors reach high polys: a `diffuseColor` value (usda), and the
/// connected UsdUVTexture inside a usdz package (a usdc layer's texture files are not in the
/// mesh bytes: white).
#[test]
fn usd_base_colors_reach_high_polys() {
    let hp = import_high_poly_bytes("scene.usda", SCENE.as_bytes()).unwrap();
    assert_eq!(hp.mesh.materials, vec!["Red".to_string(), "Blue".to_string(), "Green".to_string()]);
    assert!(hp.base_color[0].is_none() && hp.base_color[1].is_none());
    assert_eq!(hp.base_color[2].as_ref().map(|c| (c.factor, c.width)), Some((texturecraft_geom::glam::Vec4::new(0.5, 0.25, 1.0, 1.0), 0)));
    let m = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap().mesh;
    let textures = vec![vec![file("BaseColor.png", png([200, 10, 10]))], vec![file("Roughness.png", png([128; 3]))], vec![]];
    for (format, ext, textured) in [(UsdFormat::Usdz, "usdz", true), (UsdFormat::Usdc, "usdc", false)] {
        let bytes = usd_bytes(&m, "Test scene", &textures, format).unwrap().bytes;
        let hp = import_high_poly_bytes(&format!("t.{ext}"), &bytes).unwrap();
        let red = hp.base_color[0].as_ref().map(|c| (c.width, c.height, c.texels.first().copied()));
        assert_eq!(red, textured.then_some((2, 2, Some([200, 10, 10, 255]))), "{ext}");
        assert!(hp.base_color[1].is_none(), "{ext}: no base color texture, white");
    }
}

#[test]
fn usd_extension_detects_the_encoding() {
    let m = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap().mesh;
    let binary = usd_bytes(&m, "s", &[], UsdFormat::Usdc).unwrap().bytes;
    assert_eq!(import_mesh_bytes("a.usd", &binary).unwrap().mesh.triangle_count(), 4);
    assert_eq!(import_mesh_bytes("a.usd", SCENE.as_bytes()).unwrap().mesh.triangle_count(), 4);
    assert!(import_mesh_bytes("a.usdc", b"PXR-USDC\x00\x03\x00").is_err(), "old or truncated crate files are errors");
    assert!(import_mesh_bytes("a.usdz", &binary).is_err());
}

#[test]
fn integer_coding_round_trips() {
    let values: Vec<i64> = vec![0, 1, 2, 3, 3, 3, -5, 1000, -70000, i64::from(i32::MAX), i64::from(i32::MIN), 7, 7, 7, 8];
    for width in [4, 8] {
        let enc = codec::encode_ints(&values, width);
        assert!(enc.len() <= codec::encoded_ints_len(values.len(), width));
        assert_eq!(codec::decode_ints(&enc, values.len(), width).unwrap(), values);
        assert!(codec::decode_ints(&enc[..enc.len() - 1], values.len(), width).is_err());
    }
    let wide = vec![i64::MAX, i64::MIN, 0, 1 << 40];
    assert_eq!(codec::decode_ints(&codec::encode_ints(&wide, 8), 4, 8).unwrap(), wide);
    let big: Vec<u8> = (0..5000u32).flat_map(|i| (i % 7).to_le_bytes()).collect();
    assert_eq!(codec::decompress(&codec::compress(&big), big.len()).unwrap(), big);
    assert!(codec::decompress(&[0, 0xF0], 10).is_err());
}

#[test]
fn malformed_usd_never_panics() {
    // A small two-material mesh keeps the byte loops fast.
    let src = r#"#usda 1.0
(
    upAxis = "Z"
)
def Mesh "M" {
    int[] faceVertexCounts = [3, 3]
    int[] faceVertexIndices = [0, 1, 2, 0, 2, 3]
    point3f[] points = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)]
    texCoord2f[] primvars:st = [(0, 0), (1, 0), (1, 1), (0, 1)] (
        interpolation = "vertex"
    )
    rel material:binding = </A>
    def GeomSubset "b" {
        uniform token familyName = "materialBind"
        int[] indices = [1]
        rel material:binding = </B>
    }
}
"#;
    hammer("m.usda", src.as_bytes());
    let m = import_mesh_bytes("m.usda", src.as_bytes()).unwrap().mesh;
    assert_eq!(m.materials, ["A", "B"]);
    let usdc = usd_bytes(&m, "m", &[], UsdFormat::Usdc).unwrap().bytes;
    hammer("m.usdc", &usdc);
    let usdz = usd_bytes(&m, "m", &[vec![file("BaseColor.png", png([1, 2, 3]))]], UsdFormat::Usdz).unwrap().bytes;
    hammer("m.usdz", &usdz);
    for bad in [
        "#usda 1.0\ndef Mesh \"M\" {",
        "#usda 1.0\ndef \"",
        "#usda 1.0\n(((((",
        "#usda 1.0\ndef X \"a\" { int[] b = [[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[[ }",
    ] {
        assert!(import_mesh_bytes("bad.usda", bad.as_bytes()).is_err());
    }
}

/// A loose usda or usdc whose texture files sit next to it becomes a usdz package holding them,
/// so its base color reaches high-poly bakes.
#[test]
fn loose_layer_textures_are_packaged() {
    let m = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap().mesh;
    let textures = vec![vec![file("BaseColor.png", png([200, 10, 10]))], vec![], vec![]];
    for (format, ext) in [(UsdFormat::Usda, "usda"), (UsdFormat::Usdc, "usdc")] {
        let out = usd_bytes(&m, "Test scene", &textures, format).unwrap();
        let mut read = |p: &str| out.textures.iter().find(|(n, _)| n == p).map(|(_, b)| b.clone());
        let f = self_contained_mesh(&format!("dir/t.{ext}"), out.bytes.clone(), &mut read).unwrap();
        assert_eq!((f.name.as_str(), f.warnings.len()), ("dir/t.usdz", 0), "{ext}");
        let hp = import_high_poly_bytes(&f.name, &f.bytes).unwrap();
        assert_eq!(hp.base_color[0].as_ref().map(|c| c.texels[0]), Some([200, 10, 10, 255]), "{ext}");
        // Without the texture file the layer stays as it was, and the file is named.
        let f = self_contained_mesh(&format!("t.{ext}"), out.bytes.clone(), &mut |_: &str| None).unwrap();
        assert_eq!((f.name, f.bytes == out.bytes, f.warnings.len()), (format!("t.{ext}"), true, 1), "{ext}");
    }
}

#[test]
fn package_stops_reading_textures_after_exceeding_the_budget() {
    let mesh = import_mesh_bytes("scene.usda", SCENE.as_bytes()).unwrap().mesh;
    let textures: Vec<_> = (0..3).map(|index| vec![file("BaseColor.png", png([index * 40, 10, 10]))]).collect();
    let exported = usd_bytes(&mesh, "Budget test", &textures, UsdFormat::Usda).unwrap();
    let mut reads = 0;
    let mut read = |name: &str| {
        reads += 1;
        let normalized = super::below(name)?;
        exported.textures.iter().find(|(stored, _)| *stored == normalized).map(|(_, bytes)| bytes.clone())
    };

    let result = super::package_with_limit("scene.usda", &exported.bytes, &mut read, exported.bytes.len() as u64);

    assert!(matches!(result, Err(crate::FormatError::TooLarge(_))));
    assert_eq!(reads, 1, "later materials must not allocate more texture buffers");
}
