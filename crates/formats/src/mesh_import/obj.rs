//! Wavefront OBJ (via `tobj`): polygons fan-triangulated, one vertex per unique v/vt/vn.

use texturecraft_geom::glam::{Vec2, Vec3, Vec4};

use super::{RawMesh, check_triangles, check_vertices};
use crate::{Result, malformed};

const KIND: &str = "OBJ";

pub(crate) fn read(bytes: &[u8]) -> Result<RawMesh> {
    crate::guard(KIND, || read_inner(bytes))
}

fn read_inner(bytes: &[u8]) -> Result<RawMesh> {
    let text = String::from_utf8_lossy(bytes);

    // A synthetic library lets tobj assign usemtl names without reading external MTL files
    let mut names: Vec<&str> = Vec::new();
    for line in text.lines().map(str::trim) {
        if line.split_whitespace().next() == Some("usemtl")
            && let Some((_, name)) = line.split_once(' ')
            && !name.trim().is_empty()
            && !names.contains(&name.trim())
        {
            names.push(name.trim());
        }
    }

    let library: String = names.iter().map(|name| format!("newmtl {name}\n")).collect();
    let source = format!("mtllib texturecraft-usemtl.mtl\n{text}");

    // Polygons stay whole so the wireframe can show them; `RawMesh::polygon` triangulates
    let options = tobj::LoadOptions { single_index: true, triangulate: false, ignore_points: true, ignore_lines: true };
    let (models, materials) =
        tobj::load_obj_buf(&mut source.as_bytes(), &options, |_| tobj::load_mtl_buf(&mut library.as_bytes())).map_err(|e| malformed(KIND, e))?;
    let materials = materials.unwrap_or_default();

    let mut raw = RawMesh::default();
    let any_uvs = models.iter().any(|m| !m.mesh.indices.is_empty() && m.mesh.texcoords.len() == (m.mesh.positions.len() / 3) * 2);
    let (mut normals, mut uvs, mut colors) = (Some(Vec::new()), any_uvs.then(Vec::new), Vec::new());
    let mut partial_uvs = false;
    for model in &models {
        let mesh = &model.mesh;
        if mesh.indices.is_empty() {
            continue;
        }
        let vertex_count = mesh.positions.len() / 3;
        check_vertices(raw.positions.len().saturating_add(vertex_count))?;

        // Corners per face; without arities every face is a triangle
        let arities: Vec<usize> =
            if mesh.face_arities.is_empty() { vec![3; mesh.indices.len() / 3] } else { mesh.face_arities.iter().map(|a| *a as usize).collect() };
        check_triangles(raw.triangle_count(), arities.iter().map(|a| a.saturating_sub(2)).sum())?;
        let base = raw.positions.len() as u32;
        raw.positions.extend(mesh.positions.as_chunks::<3>().0.iter().map(|p| Vec3::new(p[0], p[1], p[2])));
        if mesh.normals.len() == vertex_count * 3 {
            if let Some(v) = &mut normals {
                v.extend(mesh.normals.as_chunks::<3>().0.iter().map(|p| Vec3::new(p[0], p[1], p[2])));
            }
        } else {
            normals = None;
        }
        if let Some(v) = &mut uvs {
            if mesh.texcoords.len() == vertex_count * 2 {
                // OBJ UVs have their origin bottom-left
                v.extend(mesh.texcoords.as_chunks::<2>().0.iter().map(|t| Vec2::new(t[0], 1.0 - t[1])));
            } else {
                v.extend(std::iter::repeat_n(Vec2::ZERO, vertex_count));
                partial_uvs = true;
            }
        }

        // `v x y z r g b`: objects without colors are white
        match mesh.vertex_color.len() == vertex_count * 3 {
            true => colors.extend(mesh.vertex_color.as_chunks::<3>().0.iter().map(|c| Vec4::new(c[0], c[1], c[2], 1.0))),
            false => colors.extend(std::iter::repeat_n(Vec4::ONE, vertex_count)),
        }

        let name = match mesh.material_id {
            Some(id) => materials.get(id).map(|mat| mat.name.clone()).filter(|s| !s.is_empty()).unwrap_or_else(|| format!("Material_{id}")),
            None => "Default".to_string(),
        };
        if let Some(i) = mesh.indices.iter().find(|i| **i as usize >= vertex_count) {
            return Err(malformed(KIND, format!("face index {i} out of range")));
        }

        // tobj names models without `o`/`g` "unnamed_object"; an empty name takes the file stem
        let part = if model.name == "unnamed_object" { "" } else { model.name.as_str() };
        let (mut rest, mut corners) = (mesh.indices.as_slice(), Vec::new());
        for arity in arities {
            let Some((face, tail)) = rest.split_at_checked(arity) else { break };
            rest = tail;
            corners.clear();
            corners.extend(face.iter().map(|i| base + i));
            if arity >= 3 {
                raw.polygon(&name, part, &corners);
            }
        }
    }

    if partial_uvs {
        raw.warnings.push("some objects have no UVs; default UVs were assigned".into());
    }

    raw.normals = normals;
    raw.uvs = uvs;
    raw.colors = models.iter().any(|model| !model.mesh.vertex_color.is_empty()).then_some(colors);

    Ok(raw)
}
