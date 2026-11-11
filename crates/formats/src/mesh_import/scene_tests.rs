//! FBX and COLLADA import, built from tiny files written here (binary FBX through a minimal
//! node writer, ASCII FBX from the same tree, COLLADA as text).

use std::io::Write as _;

use texturecraft_geom::glam::{Vec2, Vec3};

use super::tests::hammer;
use super::*;

enum P {
    I(i32),
    L(i64),
    D(f64),
    S(&'static str),
    Doubles(Vec<f64>),
    Ints(Vec<i32>),
    /// Binary data (`R`); ASCII writes it as base64 after an empty first value.
    R(Vec<u8>),
}

struct N {
    name: &'static str,
    props: Vec<P>,
    children: Vec<N>,
}

fn n(name: &'static str, props: Vec<P>, children: Vec<N>) -> N {
    N { name, props, children }
}

/// `P: name, type, subtype, flags, values...` in a Properties70 block.
fn prop(name: &'static str, values: &[f64]) -> N {
    let mut props = vec![P::S(name), P::S(name), P::S(""), P::S("A")];
    props.extend(values.iter().map(|v| P::D(*v)));
    n("P", props, vec![])
}

fn props70(entries: Vec<N>) -> N {
    n("Properties70", vec![], entries)
}

fn layer(name: &'static str, mapping: &'static str, reference: &'static str, data: Vec<N>) -> N {
    let mut children = vec![n("MappingInformationType", vec![P::S(mapping)], vec![]), n("ReferenceInformationType", vec![P::S(reference)], vec![])];
    children.extend(data);
    n(name, vec![P::I(0)], children)
}

/// A quad (material slot 0) and a triangle (slot 1) under a parented Model, two materials, and
/// a camera 5 units out on +Z (Y up) turned to look down -Z, with its lens on a NodeAttribute.
fn scene(z_up: bool) -> Vec<N> {
    // The Z-up variant also turns the parent with PreRotation instead of Lcl Rotation.
    let rotation = if z_up { "PreRotation" } else { "Lcl Rotation" };
    // (UpAxis, FrontAxis, FrontAxisSign)
    let (up, front, front_sign) = if z_up { (2.0, 1.0, -1.0) } else { (1.0, 2.0, 1.0) };
    let verts = vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 0.0, 200.0, 0.0, 0.0, 300.0, 0.0, 0.0, 200.0, 100.0, 0.0];
    let geometry = n(
        "Geometry",
        vec![P::L(10), P::S("Quad\u{0}\u{1}Geometry"), P::S("Mesh")],
        vec![
            n("Vertices", vec![P::Doubles(verts)], vec![]),
            n("PolygonVertexIndex", vec![P::Ints(vec![0, 1, 2, -4, 4, 5, -7])], vec![]),
            layer("LayerElementNormal", "ByPolygonVertex", "Direct", vec![n("Normals", vec![P::Doubles([0.0, 0.0, 1.0].repeat(7))], vec![])]),
            layer(
                "LayerElementUV",
                "ByPolygonVertex",
                "IndexToDirect",
                vec![
                    n("UV", vec![P::Doubles(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])], vec![]),
                    n("UVIndex", vec![P::Ints(vec![0, 1, 2, 3, 0, 1, 3])], vec![]),
                ],
            ),
            layer("LayerElementMaterial", "ByPolygon", "IndexToDirect", vec![n("Materials", vec![P::Ints(vec![0, 1])], vec![])]),
        ],
    );
    let model = |id: i64, name: &'static str, entries: Vec<N>| n("Model", vec![P::L(id), P::S(name), P::S("Mesh")], vec![props70(entries)]);
    let material = |id: i64, name: &'static str| n("Material", vec![P::L(id), P::S(name), P::S("")], vec![]);
    let c = |child: i64, parent: i64| n("C", vec![P::S("OO"), P::L(child), P::L(parent)], vec![]);
    vec![
        n("FBXHeaderExtension", vec![], vec![n("FBXVersion", vec![P::I(7400)], vec![])]),
        n(
            "GlobalSettings",
            vec![],
            vec![props70(vec![
                prop("UpAxis", &[up]),
                prop("FrontAxis", &[front]),
                prop("FrontAxisSign", &[front_sign]),
                prop("CoordAxis", &[0.0]),
                prop("UnitScaleFactor", &[1.0]),
            ])],
        ),
        n(
            "Objects",
            vec![],
            vec![
                geometry,
                model(20, "Parent\u{0}\u{1}Model", vec![prop("Lcl Translation", &[0.0, 0.0, 100.0]), prop(rotation, &[0.0, 0.0, 90.0])]),
                model(21, "Child\u{0}\u{1}Model", vec![prop("Lcl Translation", &[100.0, 0.0, 0.0])]),
                material(30, "Red\u{0}\u{1}Material"),
                material(31, "Blue\u{0}\u{1}Material"),
                n(
                    "Model",
                    vec![P::L(40), P::S("Shot\u{0}\u{1}Model"), P::S("Camera")],
                    vec![props70(vec![prop("Lcl Translation", &[0.0, 0.0, 500.0]), prop("Lcl Rotation", &[0.0, 90.0, 0.0])])],
                ),
                n(
                    "NodeAttribute",
                    vec![P::L(41), P::S("\u{0}\u{1}NodeAttribute"), P::S("Camera")],
                    vec![props70(vec![prop("FocalLength", &[35.0]), prop("FilmHeight", &[0.612])])],
                ),
            ],
        ),
        n("Connections", vec![], vec![c(10, 21), c(21, 20), c(20, 0), c(30, 21), c(31, 21), c(40, 0), c(41, 40)]),
    ]
}

/// Minimal binary FBX writer: 64-bit node records from version 7500, optional zlib arrays.
struct FbxWriter {
    wide: bool,
    compress: bool,
}

impl FbxWriter {
    fn uint(&self, out: &mut Vec<u8>, v: usize) {
        if self.wide { out.extend((v as u64).to_le_bytes()) } else { out.extend((v as u32).to_le_bytes()) }
    }

    fn null_record(&self, out: &mut Vec<u8>) {
        out.extend(vec![0u8; if self.wide { 25 } else { 13 }]);
    }

    fn array(&self, out: &mut Vec<u8>, ty: u8, count: usize, raw: Vec<u8>) {
        out.push(ty);
        out.extend((count as u32).to_le_bytes());
        let data = if self.compress {
            let mut z = flate2::write::ZlibEncoder::new(Vec::new(), flate2::Compression::default());
            z.write_all(&raw).unwrap();
            z.finish().unwrap()
        } else {
            raw
        };
        out.extend((self.compress as u32).to_le_bytes());
        out.extend((data.len() as u32).to_le_bytes());
        out.extend(data);
    }

    fn node(&self, out: &mut Vec<u8>, node: &N) {
        let mut props = Vec::new();
        for p in &node.props {
            match p {
                P::I(v) => {
                    props.push(b'I');
                    props.extend(v.to_le_bytes());
                }
                P::L(v) => {
                    props.push(b'L');
                    props.extend(v.to_le_bytes());
                }
                P::D(v) => {
                    props.push(b'D');
                    props.extend(v.to_le_bytes());
                }
                P::S(s) => {
                    props.push(b'S');
                    props.extend((s.len() as u32).to_le_bytes());
                    props.extend(s.as_bytes());
                }
                P::Doubles(v) => self.array(&mut props, b'd', v.len(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
                P::Ints(v) => self.array(&mut props, b'i', v.len(), v.iter().flat_map(|x| x.to_le_bytes()).collect()),
                P::R(b) => {
                    props.push(b'R');
                    props.extend((b.len() as u32).to_le_bytes());
                    props.extend(b);
                }
            }
        }
        let start = out.len();
        self.uint(out, 0);
        self.uint(out, node.props.len());
        self.uint(out, props.len());
        out.push(node.name.len() as u8);
        out.extend(node.name.as_bytes());
        out.extend(props);
        if !node.children.is_empty() {
            for c in &node.children {
                self.node(out, c);
            }
            self.null_record(out);
        }
        let mut end = Vec::new();
        self.uint(&mut end, out.len());
        out[start..start + end.len()].copy_from_slice(&end);
    }
}

fn binary_fbx(nodes: &[N], version: u32, compress: bool) -> Vec<u8> {
    let w = FbxWriter { wide: version >= 7500, compress };
    let mut out = b"Kaydara FBX Binary  \0\x1a\0".to_vec();
    out.extend(version.to_le_bytes());
    for node in nodes {
        w.node(&mut out, node);
    }
    w.null_record(&mut out);
    out.extend(b"footer padding");
    out
}

fn ascii_fbx(nodes: &[N]) -> String {
    fn write(s: &mut String, node: &N, indent: usize) {
        let num = |v: f64| format!("{v}");
        let values: Vec<String> = node
            .props
            .iter()
            .map(|p| match p {
                P::I(v) => v.to_string(),
                P::L(v) => v.to_string(),
                P::D(v) => num(*v),
                P::S(t) => format!("\"{}\"", t.split_once("\u{0}\u{1}").map_or(t.to_string(), |(name, class)| format!("{class}::{name}"))),
                P::Doubles(v) => format!("*{} {{\n a: {}\n}}", v.len(), v.iter().map(|x| num(*x)).collect::<Vec<_>>().join(",")),
                P::Ints(v) => format!("*{} {{\n a: {}\n}}", v.len(), v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(",")),
                P::R(b) => {
                    use base64::Engine as _;
                    format!(", \"{}\"", base64::engine::general_purpose::STANDARD.encode(b))
                }
            })
            .collect();
        s.push_str(&format!("{}{}: {}", " ".repeat(indent), node.name, values.join(", ")));
        if !node.children.is_empty() {
            s.push_str(" {\n");
            for c in &node.children {
                write(s, c, indent + 1);
            }
            s.push_str(&format!("{}}}", " ".repeat(indent)));
        }
        s.push('\n');
    }
    let mut s = "; FBX 7.4.0 project file\n; ----------------------------------------------------\n\n".to_string();
    for node in nodes {
        write(&mut s, node, 0);
    }
    s
}

/// Bounds of the vertices material `name` uses.
fn material_bounds(m: &Mesh, name: &str) -> (Vec3, Vec3) {
    let sm = m.submeshes.iter().find(|s| m.materials[s.material] == name).unwrap();
    let ids = &m.indices[sm.start as usize..(sm.start + sm.count) as usize];
    ids.iter().fold((Vec3::splat(f32::MAX), Vec3::splat(f32::MIN)), |(lo, hi), i| (lo.min(m.positions[*i as usize]), hi.max(m.positions[*i as usize])))
}

fn close(a: Vec3, b: Vec3) -> bool {
    (a - b).length() < 1e-4
}

fn check_fbx_scene(m: &Mesh, z_up: bool) {
    assert_eq!(m.materials, vec!["Red".to_string(), "Blue".to_string()]);
    assert_eq!(m.triangle_count(), 3);
    assert_eq!(m.vertex_count(), 7);
    assert!(m.submeshes.iter().all(|s| s.name == "Child"), "parts are named after their model");
    // cm to m, child (1, 0, 0) under a parent rotated 90° about Z and lifted 1 m.
    let (lo, hi) = m.bounds();
    let (want_lo, want_hi) = if z_up { (Vec3::new(-1.0, 1.0, -4.0), Vec3::new(0.0, 1.0, -1.0)) } else { (Vec3::new(-1.0, 1.0, 1.0), Vec3::new(0.0, 4.0, 1.0)) };
    assert!(close(lo, want_lo) && close(hi, want_hi), "bounds {lo} {hi}");
    let up = if z_up { Vec3::Y } else { Vec3::Z };
    assert!(m.normals.iter().all(|nrm| close(*nrm, up)), "normals {:?}", m.normals);
    let (red_lo, red_hi) = material_bounds(m, "Red");
    assert!(close(red_hi - red_lo, if z_up { Vec3::new(1.0, 0.0, 1.0) } else { Vec3::new(1.0, 1.0, 0.0) }));
    // UVs through UVIndex, v flipped to glTF's convention: every corner of the quad is used.
    let mut uvs: Vec<Vec2> = m.uvs.clone();
    uvs.sort_by(|a, b| a.x.total_cmp(&b.x).then(a.y.total_cmp(&b.y)));
    uvs.dedup();
    assert_eq!(uvs, vec![Vec2::new(0.0, 0.0), Vec2::new(0.0, 1.0), Vec2::new(1.0, 0.0), Vec2::new(1.0, 1.0)]);
    // Winding stays outward (front faces toward the normals).
    for t in m.indices.chunks(3) {
        let p = |i: u32| m.positions[i as usize];
        let face = (p(t[1]) - p(t[0])).cross(p(t[2]) - p(t[0]));
        assert!(face.dot(m.normals[t[0] as usize]) > 0.0);
    }
}

/// The camera of [`scene`] (Y up): 35 mm over a 0.612" film, 5 m out, looking at the mesh.
fn check_fbx_camera(im: &ImportedMesh) {
    let [shot] = im.cameras.as_slice() else { panic!("{:?}", im.cameras) };
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert_eq!(shot.name, "Shot");
    assert!(!shot.ortho && near(shot.fov, 2.0 * (0.612f32 * 25.4 / 70.0).atan().to_degrees()), "{shot:?}");
    assert!(near(shot.yaw, 0.0) && near(shot.pitch, 0.0) && near(shot.distance, 4.0), "{shot:?}");
}

#[test]
fn fbx_binary_32_and_64_bit_compressed_or_not() {
    let nodes = scene(false);
    for (version, compress) in [(7400, false), (7400, true), (7500, false), (7500, true)] {
        let bytes = binary_fbx(&nodes, version, compress);
        let im = import_mesh_bytes("scene.fbx", &bytes).unwrap_or_else(|e| panic!("{version} {compress}: {e}"));
        check_fbx_scene(&im.mesh, false);
        check_fbx_camera(&im);
    }
}

/// [`scene`] with a diffuse color on Red and an embedded diffuse texture on Blue.
fn textured_scene(png: &[u8]) -> Vec<N> {
    let mut nodes = scene(false);
    let objects = nodes.iter_mut().find(|n| n.name == "Objects").unwrap();
    let red = objects.children.iter_mut().find(|n| matches!(n.props.first(), Some(P::L(30)))).unwrap();
    red.children.push(props70(vec![prop("DiffuseColor", &[0.5, 0.25, 1.0]), prop("DiffuseFactor", &[0.5])]));
    objects.children.push(n("Texture", vec![P::L(40), P::S("Tex\u{0}\u{1}Texture"), P::S("")], vec![]));
    objects.children.push(n("Video", vec![P::L(50), P::S("Tex\u{0}\u{1}Video"), P::S("Clip")], vec![n("Content", vec![P::R(png.to_vec())], vec![])]));
    let connections = nodes.iter_mut().find(|n| n.name == "Connections").unwrap();
    connections.children.push(n("C", vec![P::S("OP"), P::L(40), P::L(31), P::S("DiffuseColor")], vec![]));
    connections.children.push(n("C", vec![P::S("OO"), P::L(50), P::L(40)], vec![]));
    nodes
}

/// A 2 × 2 PNG: red in the left column, green in the top row, blue 7.
fn tiny_png() -> Vec<u8> {
    let mut png = Vec::new();
    let img = image::RgbaImage::from_fn(2, 2, |x, y| image::Rgba([if x == 0 { 255 } else { 0 }, if y == 0 { 255 } else { 0 }, 7, 255]));
    image::DynamicImage::ImageRgba8(img).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    png
}

#[test]
fn fbx_diffuse_colors_and_embedded_textures_reach_high_polys() {
    let png = tiny_png();
    let nodes = textured_scene(&png);
    for (label, bytes) in [("binary", binary_fbx(&nodes, 7500, true)), ("ascii", ascii_fbx(&nodes).into_bytes())] {
        let hp = import_high_poly_bytes("textured.fbx", &bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
        assert_eq!(hp.mesh.materials, vec!["Red".to_string(), "Blue".to_string()], "{label}");
        let red = hp.base_color[0].as_ref().unwrap();
        assert_eq!((red.factor, red.width), (texturecraft_geom::glam::Vec4::new(0.25, 0.125, 0.5, 1.0), 0), "{label}");
        let blue = hp.base_color[1].as_ref().unwrap();
        assert_eq!((blue.width, blue.height, blue.texels[0], blue.texels[3]), (2, 2, [255, 255, 7, 255], [0, 0, 7, 255]), "{label}");
        assert!(hp.warnings.is_empty(), "{label}: {:?}", hp.warnings);
        // The plain import reads the same file.
        check_fbx_scene(&import_mesh_bytes("textured.fbx", &bytes).unwrap().mesh, false);
    }
    hammer("x.fbx", &binary_fbx(&nodes, 7500, false));
}

#[test]
fn fbx_retains_white_materials_and_colors_beside_textures() {
    let mut nodes = textured_scene(&tiny_png());
    let objects = nodes.iter_mut().find(|node| node.name == "Objects").unwrap();
    let red = objects.children.iter_mut().find(|node| matches!(node.props.first(), Some(P::L(30)))).unwrap();
    red.children = vec![props70(vec![prop("DiffuseColor", &[1.0, 1.0, 1.0])])];
    let blue = objects.children.iter_mut().find(|node| matches!(node.props.first(), Some(P::L(31)))).unwrap();
    blue.children.push(props70(vec![prop("DiffuseColor", &[0.1, 0.2, 0.3])]));

    let imported = import_high_poly_bytes("colors.fbx", &binary_fbx(&nodes, 7500, true)).unwrap();
    let white = imported.base_color[0].as_ref().unwrap();
    let textured = imported.base_color[1].as_ref().unwrap();
    assert_eq!(white.factor, texturecraft_geom::glam::Vec4::ONE);
    assert_eq!(textured.factor, texturecraft_geom::glam::Vec4::new(0.1, 0.2, 0.3, 1.0));
    assert_eq!(textured.texels.len(), 4);
}

/// [`textured_scene`] with Blue's texture kept as a file: `tex/wood.png` relative to the FBX, or
/// only an absolute path elsewhere (found by its file name next to the FBX).
fn texture_file_scene(relative: &'static str) -> Vec<N> {
    let mut nodes = textured_scene(&[]);
    let objects = nodes.iter_mut().find(|n| n.name == "Objects").unwrap();
    objects.children.retain(|n| n.name != "Video");
    let tex = objects.children.iter_mut().find(|n| n.name == "Texture").unwrap();
    tex.children.push(n("FileName", vec![P::S("C:/Users/artist/maps/wood.png")], vec![]));
    tex.children.push(n("RelativeFilename", vec![P::S(relative)], vec![]));
    nodes
}

#[test]
fn fbx_package_stops_reading_textures_after_exceeding_the_budget() {
    let mut nodes = texture_file_scene("texture.png");
    let connections = nodes.iter_mut().find(|node| node.name == "Connections").unwrap();
    connections.children.push(n("C", vec![P::S("OP"), P::L(40), P::L(30), P::S("DiffuseColor")], vec![]));
    let bytes = ascii_fbx(&nodes).into_bytes();
    let texture = tiny_png();
    let mut reads = 0;
    let mut read = |_: &str| {
        reads += 1;
        Some(texture.clone())
    };

    let result = super::fbx::package_with_limit("mesh.fbx", &bytes, &mut read, bytes.len() as u64);

    assert!(matches!(result, Err(crate::FormatError::TooLarge(_))));
    assert_eq!(reads, 1, "later materials must not allocate more texture buffers");
}

#[test]
fn fbx_texture_files_are_packaged_with_the_mesh() {
    let png = tiny_png();
    let blue = |bytes: &[u8]| import_high_poly_bytes("wood.fbx", bytes).unwrap().base_color[1].as_ref().map(|b| (b.width, b.texels[0]));
    for (label, bytes) in
        [("binary", binary_fbx(&texture_file_scene("tex\\wood.png"), 7500, true)), ("ascii", ascii_fbx(&texture_file_scene("tex/wood.png")).into_bytes())]
    {
        assert_eq!(blue(&bytes), None, "{label}: the file alone has no texture");
        // Beside the FBX under its relative path: packaged, and the package reads like the FBX.
        let mut asked = Vec::new();
        let f = self_contained_mesh("scenes/wood.fbx", bytes.clone(), &mut |p| {
            asked.push(p.to_string());
            (p == "tex/wood.png").then(|| png.clone())
        })
        .unwrap();
        assert_eq!(asked, ["tex/wood.png"], "{label}");
        assert!(f.warnings.is_empty() && f.name == "scenes/wood.fbx" && f.bytes.starts_with(b"PK"), "{label}: {:?}", f.warnings);
        assert_eq!(blue(&f.bytes), Some((2, [255, 255, 7, 255])), "{label}");
        check_fbx_scene(&import_mesh_bytes("wood.fbx", &f.bytes).unwrap().mesh, false);
        assert_eq!(self_contained_mesh("wood.fbx", f.bytes.clone(), &mut |_| panic!("packaged already")).unwrap().bytes, f.bytes);
        // Only the absolute path's file name next to the FBX.
        let g = self_contained_mesh("wood.fbx", bytes.clone(), &mut |p| (p == "wood.png").then(|| png.clone())).unwrap();
        assert!(g.warnings.is_empty(), "{label}: {:?}", g.warnings);
        assert_eq!(blue(&g.bytes), Some((2, [255, 255, 7, 255])), "{label}");
        // Missing: unchanged, with a warning naming it.
        let h = self_contained_mesh("wood.fbx", bytes.clone(), &mut |_| None).unwrap();
        assert_eq!(h.bytes, bytes, "{label}");
        assert!(h.warnings.iter().any(|w| w.contains("wood.png") && w.contains("not found")), "{label}: {:?}", h.warnings);
    }
    let packaged = self_contained_mesh("wood.fbx", ascii_fbx(&texture_file_scene("tex/wood.png")).into_bytes(), &mut |_| Some(png.clone())).unwrap();
    hammer("x.fbx", &packaged.bytes[..packaged.bytes.len().min(4096)]);
    // A package without an FBX, or holding another package, is an error.
    let empty = crate::mesh_export::usd::usdz("a.png", &png, &[]).unwrap();
    assert!(import_mesh_bytes("x.fbx", &empty).is_err());
    let nested = crate::mesh_export::usd::usdz("inner.fbx", &packaged.bytes, &[]).unwrap();
    assert!(import_mesh_bytes("x.fbx", &nested).is_err());
}

#[test]
fn fbx_ascii_and_z_up() {
    let text = ascii_fbx(&scene(false));
    let im = import_mesh_bytes("scene.FBX", text.as_bytes()).unwrap();
    check_fbx_scene(&im.mesh, false);
    check_fbx_camera(&im);
    check_fbx_scene(&import_mesh_bytes("z.fbx", ascii_fbx(&scene(true)).as_bytes()).unwrap().mesh, true);
    check_fbx_scene(&import_mesh_bytes("z.fbx", &binary_fbx(&scene(true), 7700, true)).unwrap().mesh, true);
}

/// The scene of [`scene`] (Y up) as FBX 6.1 writes it: geometry inside the Model, objects named
/// `Class::Name`, `Properties60`, `Connect` records, GlobalSettings inside Objects.
const FBX61: &str = r#"; FBX 6.1.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 6100
}
Definitions:  {
	Version: 100
	Count: 4
}
Objects:  {
	Model: "Model::Producer Perspective", "Camera" {
		Version: 232
		Properties60:  {
			Property: "FocalLength", "Number", "A+",35
		}
	}
	Model: "Model::Parent", "Null" {
		Version: 232
		Properties60:  {
			Property: "QuaternionInterpolate", "bool", "",0
			Property: "Lcl Translation", "Lcl Translation", "A+",0,0,100
			Property: "Lcl Rotation", "Lcl Rotation", "A+",0,0,90
			Property: "Lcl Scaling", "Lcl Scaling", "A+",1,1,1
		}
		MultiLayer: 0
		Shading: Y
		Culling: "CullingOff"
		TypeFlags: "Null"
	}
	Model: "Model::Child", "Mesh" {
		Version: 232
		Properties60:  {
			Property: "Lcl Translation", "Lcl Translation", "A+",100,0,0
		}
		Vertices: 0,0,0,100,0,0,100,100,0,0,100,0,200,0,0,300,0,0,200,100,0
		PolygonVertexIndex: 0,1,2,-4,4,5,-7
		GeometryVersion: 124
		LayerElementNormal: 0 {
			Version: 101
			Name: ""
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "Direct"
			Normals: 0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
		}
		LayerElementUV: 0 {
			Version: 101
			Name: "UVChannel_1"
			MappingInformationType: "ByPolygonVertex"
			ReferenceInformationType: "IndexToDirect"
			UV: 0,0,1,0,1,1,0,1
			UVIndex: 0,1,2,3,0,1,3
		}
		LayerElementMaterial: 0 {
			Version: 101
			Name: ""
			MappingInformationType: "ByPolygon"
			ReferenceInformationType: "IndexToDirect"
			Materials: 0,1
		}
		Layer: 0 {
			Version: 100
			LayerElement:  {
				Type: "LayerElementNormal"
				TypedIndex: 0
			}
		}
	}
	Material: "Material::Red", "" {
		Version: 102
		ShadingModel: "lambert"
	}
	Material: "Material::Blue", "" {
		Version: 102
	}
	GlobalSettings:  {
		Version: 1000
		Properties60:  {
			Property: "UpAxis", "int", "",1
			Property: "UpAxisSign", "int", "",1
			Property: "FrontAxis", "int", "",2
			Property: "UnitScaleFactor", "double", "",1
		}
	}
}
Connections:  {
	Connect: "OO", "Model::Parent", "Model::Scene"
	Connect: "OO", "Model::Child", "Model::Parent"
	Connect: "OO", "Material::Red", "Model::Child"
	Connect: "OO", "Material::Blue", "Model::Child"
}
Takes:  {
	Current: ""
}
"#;

/// [`FBX61`] as a node tree for the binary encoding (FBX 6.1 binary names are `Name\0\1Class`).
fn scene61() -> Vec<N> {
    let p60 = |name: &'static str, values: &[f64]| {
        let mut props = vec![P::S(name), P::S(name), P::S("A+")];
        props.extend(values.iter().map(|v| P::D(*v)));
        n("Property", props, vec![])
    };
    let mut child = vec![
        n("Properties60", vec![], vec![p60("Lcl Translation", &[100.0, 0.0, 0.0])]),
        n(
            "Vertices",
            vec![P::Doubles(vec![0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 100.0, 100.0, 0.0, 0.0, 100.0, 0.0, 200.0, 0.0, 0.0, 300.0, 0.0, 0.0, 200.0, 100.0, 0.0])],
            vec![],
        ),
        n("PolygonVertexIndex", vec![P::Ints(vec![0, 1, 2, -4, 4, 5, -7])], vec![]),
    ];
    child.push(layer("LayerElementNormal", "ByPolygonVertex", "Direct", vec![n("Normals", vec![P::Doubles([0.0, 0.0, 1.0].repeat(7))], vec![])]));
    child.push(layer(
        "LayerElementUV",
        "ByPolygonVertex",
        "IndexToDirect",
        vec![n("UV", vec![P::Doubles(vec![0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0])], vec![]), n("UVIndex", vec![P::Ints(vec![0, 1, 2, 3, 0, 1, 3])], vec![])],
    ));
    child.push(layer("LayerElementMaterial", "ByPolygon", "IndexToDirect", vec![n("Materials", vec![P::Ints(vec![0, 1])], vec![])]));
    let connect = |a: &'static str, b: &'static str| n("Connect", vec![P::S("OO"), P::S(a), P::S(b)], vec![]);
    vec![
        n("FBXHeaderExtension", vec![], vec![n("FBXVersion", vec![P::I(6100)], vec![])]),
        n(
            "Objects",
            vec![],
            vec![
                n(
                    "Model",
                    vec![P::S("Parent\u{0}\u{1}Model"), P::S("Null")],
                    vec![n("Properties60", vec![], vec![p60("Lcl Translation", &[0.0, 0.0, 100.0]), p60("Lcl Rotation", &[0.0, 0.0, 90.0])])],
                ),
                n("Model", vec![P::S("Child\u{0}\u{1}Model"), P::S("Mesh")], child),
                n("Material", vec![P::S("Red\u{0}\u{1}Material"), P::S("")], vec![]),
                n("Material", vec![P::S("Blue\u{0}\u{1}Material"), P::S("")], vec![]),
            ],
        ),
        n(
            "Connections",
            vec![],
            vec![
                connect("Parent\u{0}\u{1}Model", "Scene\u{0}\u{1}Model"),
                connect("Child\u{0}\u{1}Model", "Parent\u{0}\u{1}Model"),
                connect("Red\u{0}\u{1}Material", "Child\u{0}\u{1}Model"),
                connect("Blue\u{0}\u{1}Material", "Child\u{0}\u{1}Model"),
            ],
        ),
    ]
}

#[test]
fn fbx_6_ascii_and_binary_read_like_7() {
    let six = import_mesh_bytes("six.fbx", FBX61.as_bytes()).unwrap();
    check_fbx_scene(&six.mesh, false);
    assert!(six.cameras.is_empty(), "the Producer views are not scene cameras");
    check_fbx_scene(&import_mesh_bytes("six.fbx", &binary_fbx(&scene61(), 6100, false)).unwrap().mesh, false);
    hammer("x.fbx", FBX61.as_bytes());
    hammer("x.fbx", &binary_fbx(&scene61(), 6100, true));
}

#[test]
fn fbx_old_versions_and_garbage_are_errors() {
    let mut old = binary_fbx(&scene(false), 7400, false);
    old[23..27].copy_from_slice(&5800u32.to_le_bytes());
    assert!(matches!(import_mesh_bytes("old.fbx", &old), Err(FormatError::Unsupported(_))));
    assert!(import_mesh_bytes("a.fbx", b"; FBX 6.1.0 project file\nObjects: {\n}\n").is_err());
    // An array claiming 2^32-1 doubles never allocates its claim.
    let mut huge = binary_fbx(&[n("Objects", vec![P::Doubles(vec![1.0])], vec![])], 7400, false);
    let at = huge.windows(5).position(|w| w == [b'd', 1, 0, 0, 0]).unwrap() + 1;
    huge[at..at + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(import_mesh_bytes("h.fbx", &huge).is_err());
}

#[test]
fn malformed_fbx_and_dae_never_panic() {
    hammer("x.fbx", &binary_fbx(&scene(false), 7400, false));
    hammer("x.fbx", &binary_fbx(&scene(true), 7500, true));
    hammer("x.fbx", ascii_fbx(&scene(false)).as_bytes());
    hammer("x.dae", DAE.as_bytes());
    // Random multi-byte corruptions (a fixed LCG keeps failures reproducible).
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (seed >> 33) as usize
    };
    for good in [binary_fbx(&scene(false), 7500, true), ascii_fbx(&scene(true)).into_bytes(), DAE.as_bytes().to_vec()] {
        let name = if good.starts_with(b"<?xml") { "r.dae" } else { "r.fbx" };
        for _ in 0..1500 {
            let mut b = good.clone();
            for _ in 0..1 + next() % 4 {
                let i = next() % b.len();
                b[i] = next() as u8;
            }
            if let Err(e) = import_mesh_bytes(name, &b) {
                assert!(!e.to_string().contains("parser failed"), "{name}: a parser panicked");
            }
        }
    }
}

const DAE: &str = r##"<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="centimeter" meter="0.5"/><up_axis>Z_UP</up_axis></asset>
  <library_materials>
    <material id="red-mat" name="Red"/>
    <material id="blue-mat" name="Blue"/>
  </library_materials>
  <library_geometries>
    <geometry id="quad" name="Quad">
      <mesh>
        <source id="quad-pos">
          <float_array id="quad-pos-array" count="12">0 0 0 2 0 0 2 2 0 0 2 0</float_array>
          <technique_common><accessor source="#quad-pos-array" count="4" stride="3"/></technique_common>
        </source>
        <source id="quad-nrm">
          <float_array id="quad-nrm-array" count="3">0 0 1</float_array>
          <technique_common><accessor source="#quad-nrm-array" count="1" stride="3"/></technique_common>
        </source>
        <source id="quad-uv">
          <float_array id="quad-uv-array" count="8">0 0 1 0 1 1 0 1</float_array>
          <technique_common><accessor source="#quad-uv-array" count="4" stride="2"/></technique_common>
        </source>
        <vertices id="quad-verts"><input semantic="POSITION" source="#quad-pos"/></vertices>
        <polylist material="mat0" count="1">
          <input semantic="VERTEX" source="#quad-verts" offset="0"/>
          <input semantic="NORMAL" source="#quad-nrm" offset="1"/>
          <input semantic="TEXCOORD" source="#quad-uv" offset="2" set="0"/>
          <vcount>4</vcount>
          <p>0 0 0 1 0 1 2 0 2 3 0 3</p>
        </polylist>
      </mesh>
    </geometry>
    <geometry id="tris" name="Tris">
      <mesh>
        <source id="tris-pos">
          <float_array id="tris-pos-array" count="21">4 0 0 6 0 0 4 2 0 8 0 0 10 0 0 10 2 0 8 2 0</float_array>
          <technique_common><accessor source="#tris-pos-array" count="7" stride="3"/></technique_common>
        </source>
        <source id="tris-uv">
          <float_array id="tris-uv-array" count="14">0 0 1 0 0 1 0 0 1 0 1 1 0 1</float_array>
          <technique_common><accessor source="#tris-uv-array" count="7" stride="2"/></technique_common>
        </source>
        <vertices id="tris-verts">
          <input semantic="POSITION" source="#tris-pos"/>
          <input semantic="TEXCOORD" source="#tris-uv"/>
        </vertices>
        <triangles material="mat1" count="1"><input semantic="VERTEX" source="#tris-verts" offset="0"/><p>0 1 2</p></triangles>
        <polygons material="mat1" count="1"><input semantic="VERTEX" source="#tris-verts" offset="0"/><p>3 4 5 6</p></polygons>
        <lines count="1"><input semantic="VERTEX" source="#tris-verts" offset="0"/><p>0 1</p></lines>
      </mesh>
    </geometry>
  </library_geometries>
  <library_cameras>
    <camera id="lens"><optics><technique_common><perspective><yfov>30</yfov></perspective></technique_common></optics></camera>
  </library_cameras>
  <library_visual_scenes>
    <visual_scene id="scene">
      <node id="parent">
        <translate>0 0 2</translate>
        <rotate>0 0 1 90</rotate>
        <node id="child">
          <scale>2 2 2</scale>
          <instance_geometry url="#quad">
            <bind_material><technique_common><instance_material symbol="mat0" target="#red-mat"/></technique_common></bind_material>
          </instance_geometry>
        </node>
      </node>
      <node id="cam" name="Overview">
        <lookat>0 -10 0 0 0 0 0 0 1</lookat>
        <instance_camera url="#lens"/>
      </node>
      <node id="other">
        <instance_geometry url="#tris">
          <bind_material><technique_common><instance_material symbol="mat1" target="#blue-mat"/></technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#scene"/></scene>
</COLLADA>
"##;

#[test]
fn dae_polylist_z_up_units_and_node_transforms() {
    let r = import_mesh_bytes("scene.dae", DAE.as_bytes()).unwrap();
    let m = &r.mesh;
    assert_eq!(m.materials, vec!["Red".to_string(), "Blue".to_string()]);
    assert_eq!(m.triangle_count(), 2 + 1 + 2);
    assert!(!r.report.uvs_generated);
    assert!(r.report.warnings.iter().any(|w| w.contains("lines")));
    // Quad: ×2 scale, 90° about Z, lifted 2 (Z up), half-meter units, then Z up → Y up.
    let (lo, hi) = material_bounds(m, "Red");
    assert!(close(lo, Vec3::new(-2.0, 1.0, -2.0)) && close(hi, Vec3::new(0.0, 1.0, 0.0)), "{lo} {hi}");
    let (lo, hi) = material_bounds(m, "Blue");
    assert!(close(lo, Vec3::new(2.0, 0.0, -1.0)) && close(hi, Vec3::new(5.0, 0.0, 0.0)), "{lo} {hi}");
    for t in m.indices.chunks(3) {
        for i in t {
            assert!(close(m.normals[*i as usize], Vec3::Y), "normal {}", m.normals[*i as usize]);
        }
    }
    assert!(m.uvs.iter().all(|uv| uv.x >= 0.0 && uv.x <= 1.0 && uv.y >= 0.0 && uv.y <= 1.0));
    // Parts are named after their nodes (ids when unnamed).
    let parts: Vec<&str> = m.submeshes.iter().map(|s| s.name.as_str()).collect();
    assert!(parts.contains(&"child") && parts.contains(&"other"), "{parts:?}");
    // The camera looks at the origin from -10 along Z-up Y: on +Z once converted, facing -Z.
    let [cam] = r.cameras.as_slice() else { panic!("{:?}", r.cameras) };
    let near = |a: f32, b: f32| (a - b).abs() < 1e-3;
    assert_eq!(cam.name, "Overview");
    assert!(!cam.ortho && near(cam.fov, 30.0) && near(cam.yaw, 0.0) && near(cam.pitch, 0.0), "{cam:?}");
}

#[test]
fn glb_export_round_trips_with_textures() {
    let mesh = texturecraft_geom::primitives::material_ball();
    let png = |color: image::ColorType| {
        let mut out = std::io::Cursor::new(Vec::new());
        let channels = color.channel_count() as usize;
        image::write_buffer_with_format(&mut out, &vec![200u8; 4 * channels], 2, 2, color, image::ImageFormat::Png).unwrap();
        out.into_inner()
    };
    let black = {
        let mut out = std::io::Cursor::new(Vec::new());
        image::write_buffer_with_format(&mut out, &[0u8; 12], 2, 2, image::ColorType::Rgb8, image::ImageFormat::Png).unwrap();
        out.into_inner()
    };
    let file = |name: &str, bytes: Vec<u8>| crate::ExportedFile { name: name.into(), bytes, width: 2, height: 2, warnings: vec![] };
    let files = vec![
        file("BaseColor.png", png(image::ColorType::Rgba8)),
        file("OcclusionRoughnessMetallic.png", png(image::ColorType::Rgb8)),
        file("Normal.png", png(image::ColorType::Rgb8)),
        file("Emissive.png", black),
    ];
    let first = crate::GlbTextures::from_files(files);
    assert!(first.emissive.is_none(), "black emissive maps are dropped");
    let bytes = crate::glb_bytes(&mesh, "Ball", &[first]).unwrap();
    let back = import_mesh_bytes("ball.glb", &bytes).unwrap().mesh;
    assert_eq!(back.triangle_count(), mesh.triangle_count());
    assert_eq!(back.materials, mesh.materials);
    let doc = ::gltf::Gltf::from_slice(&bytes).unwrap();
    assert_eq!(doc.images().count(), 3);
    let mat = doc.materials().next().unwrap();
    assert!(mat.pbr_metallic_roughness().base_color_texture().is_some() && mat.normal_texture().is_some() && mat.occlusion_texture().is_some());
    assert_eq!(mat.alpha_mode(), ::gltf::material::AlphaMode::Blend);
    // The second material has no textures and is not metallic.
    let plain = doc.materials().nth(1).unwrap();
    assert!(plain.pbr_metallic_roughness().base_color_texture().is_none());
    assert_eq!(plain.pbr_metallic_roughness().metallic_factor(), 0.0);
    for img in doc.images() {
        assert!(matches!(img.source(), ::gltf::image::Source::View { mime_type: "image/png", .. }));
    }
}
