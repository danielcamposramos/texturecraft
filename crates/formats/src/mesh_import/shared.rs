//! Mapping craft-mesh scenes to TextureCraft's world-space mesh and material data.

use std::collections::HashSet;

use craft_mesh::{Camera, Error as MeshError, Limits, Projection, Scene, no_external};
use texturecraft_geom::glam::{Mat3, Mat4, Vec2, Vec3, Vec4};

use super::{Corner, MaterialTextures, PolygonSink, RawBaseColor, RawCamera, RawMesh};
use crate::{FormatError, Result, malformed};

pub(crate) fn obj(bytes: &[u8]) -> Result<RawMesh> {
    let limits = limits();
    let scene = craft_mesh::obj::read(bytes, &limits, &mut no_external).map_err(|e| mesh_error("OBJ", e))?;
    adapt(scene, &limits, false)
}

pub(crate) fn gltf(bytes: &[u8]) -> Result<RawMesh> {
    let limits = limits();
    let scene = craft_mesh::gltf::read(bytes, &limits, &mut no_external).map_err(|e| mesh_error("glTF", e))?;
    adapt(scene, &limits, true)
}

fn limits() -> Limits {
    Limits { max_vertices: super::MAX_VERTICES, max_triangles: super::MAX_TRIANGLES, max_depth: 64, ..Limits::default() }
}

fn mesh_error(kind: &'static str, error: MeshError) -> FormatError {
    match error {
        MeshError::Malformed(message) => malformed(kind, message),
        MeshError::TooLarge(message) => FormatError::TooLarge(message),
        MeshError::Unsupported(message) | MeshError::Missing(message) => FormatError::Unsupported(message),
    }
}

fn adapt(scene: Scene, limits: &Limits, gltf: bool) -> Result<RawMesh> {
    let instances = placed_meshes(&scene, limits, gltf)?;
    let mut sink = PolygonSink::default();
    sink.raw.warnings.extend(scene.warnings.iter().cloned());
    if gltf {
        add_materials(&scene, &mut sink.raw);
        add_cameras(&scene, &mut sink.raw, limits)?;
    }

    for (mesh_index, transform, instance_name) in instances {
        let Some(mesh) = scene.meshes.get(mesh_index) else { continue };
        let world = matrix(transform);
        let normal_transform = Mat3::from_mat4(world).inverse().transpose();
        for part in &mesh.parts {
            let material = part
                .material
                .and_then(|i| scene.materials.get(i).map(|m| (i, m.name.as_str())))
                .map(|(i, name)| {
                    if name.is_empty() && !gltf {
                        "Default".to_string()
                    } else if name.is_empty() {
                        format!("Material_{i}")
                    } else {
                        name.to_string()
                    }
                })
                .unwrap_or_else(|| "Default".to_string());
            sink.part = if gltf {
                instance_name.clone()
            } else if part.name.is_empty() {
                mesh.name.clone()
            } else {
                part.name.clone()
            };

            for face in part.faces() {
                let mut corners = face.iter().map(|corner| mapped_corner(mesh, *corner, world, normal_transform)).collect::<Result<Vec<_>>>()?;
                if gltf && world.determinant() < 0.0 {
                    corners.reverse();
                }
                if gltf && corners.iter().any(|c| c.n.is_none()) && corners.len() == 3 {
                    let normal = (corners[1].p - corners[0].p).cross(corners[2].p - corners[0].p).try_normalize().unwrap_or(Vec3::Y);
                    for corner in &mut corners {
                        corner.n = Some(normal);
                    }
                }
                sink.polygon(&material, &corners)?;
            }
        }
    }
    Ok(if gltf { sink.finish() } else { sink.finish_obj() })
}

fn placed_meshes(scene: &Scene, limits: &Limits, gltf: bool) -> Result<Vec<(usize, craft_mesh::Mat4, String)>> {
    let mut out = Vec::new();
    let mut stack: Vec<(usize, craft_mesh::Mat4, usize)> = scene.roots.iter().rev().map(|&i| (i, craft_mesh::IDENTITY, 0)).collect();
    let mut visits = 0usize;
    while let Some((index, parent, depth)) = stack.pop() {
        visits += 1;
        if visits > limits.max_nodes || depth > limits.max_depth {
            return Err(FormatError::TooLarge(format!("the {} node hierarchy is too large", if gltf { "glTF" } else { "OBJ" })));
        }
        let Some(node) = scene.nodes.get(index) else { continue };
        let world = craft_mesh::mat_mul(&parent, &node.transform);
        if let Some(mesh_index) = node.mesh.filter(|i| *i < scene.meshes.len()) {
            let mesh_name = scene.meshes.get(mesh_index).map(|m| m.name.as_str()).unwrap_or_default();
            let name = if node.name.is_empty() { mesh_name.to_string() } else { node.name.clone() };
            out.push((mesh_index, world, name));
        }
        stack.extend(node.children.iter().rev().map(|&child| (child, world, depth + 1)));
    }
    Ok(out)
}

fn matrix(m: craft_mesh::Mat4) -> Mat4 {
    Mat4::from_cols_array(&m.map(|v| v as f32))
}

fn mapped_corner(mesh: &craft_mesh::Mesh, c: craft_mesh::Corner, world: Mat4, normal_transform: Mat3) -> Result<Corner> {
    let p = mesh.positions.get(c.position as usize).copied().ok_or_else(|| malformed("mesh", "corner position is out of range"))?;
    let n = mesh
        .normals
        .get(c.normal as usize)
        .copied()
        .map(|n| (normal_transform * Vec3::new(n[0] as f32, n[1] as f32, n[2] as f32)).try_normalize().unwrap_or(Vec3::Y));
    let uv = mesh.uv_sets.first().and_then(|set| set.get(c.uv as usize)).map(|uv| Vec2::new(uv[0], uv[1]));
    let color = mesh.colors.get(c.position as usize).copied().map(|c| Vec4::new(c[0], c[1], c[2], c[3]));
    Ok(Corner { p: world.transform_point3(Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32)), n, uv, color })
}

fn add_materials(scene: &Scene, raw: &mut RawMesh) {
    let mut added = HashSet::new();
    for (index, material) in scene.materials.iter().enumerate() {
        let material_name = if material.name.is_empty() { format!("Material_{index}") } else { material.name.clone() };
        if !added.insert(material_name.clone()) {
            continue;
        }
        let base_color = Vec4::from_array(material.base_color);
        let base_color_image = texture_bytes(scene, material.base_color_texture, &material_name, "base color", raw);
        let metallic_roughness_image = texture_bytes(scene, material.metallic_roughness_texture, &material_name, "metallic/roughness", raw);
        let normal_image = texture_bytes(scene, material.normal_texture, &material_name, "normal", raw);
        let occlusion_image = texture_bytes(scene, material.occlusion_texture, &material_name, "occlusion", raw);
        let emissive_image = texture_bytes(scene, material.emissive_texture, &material_name, "emissive", raw);
        if base_color_image.is_some() || base_color != Vec4::ONE {
            raw.base_colors.push(RawBaseColor { material: material_name.clone(), factor: base_color, image: base_color_image.clone() });
        }
        raw.materials.push(MaterialTextures {
            material: material_name,
            base_color: Some(base_color),
            base_color_image,
            metallic: Some(material.metallic),
            roughness: Some(material.roughness),
            metallic_roughness_image,
            normal_image,
            occlusion_image,
            emissive: Some(Vec3::from_array(material.emissive)),
            emissive_image,
        });
    }
}

fn texture_bytes(scene: &Scene, texture: Option<craft_mesh::TextureRef>, material: &str, slot: &str, raw: &mut RawMesh) -> Option<Vec<u8>> {
    let texture = texture?;
    if texture.uv_set != 0 {
        raw.warnings.push(format!("the {slot} texture of {material} uses UV set {}; only the first is read", texture.uv_set));
        return None;
    }
    scene.images.get(texture.image).and_then(|image| image.data.clone())
}

fn add_cameras(scene: &Scene, raw: &mut RawMesh, limits: &Limits) -> Result<()> {
    let mut stack: Vec<(usize, craft_mesh::Mat4, usize)> = scene.roots.iter().rev().map(|&i| (i, craft_mesh::IDENTITY, 0)).collect();
    let mut visits = 0usize;
    while let Some((index, parent, depth)) = stack.pop() {
        visits += 1;
        if visits > limits.max_nodes || depth > limits.max_depth {
            return Err(FormatError::TooLarge("the glTF node hierarchy is too large to read cameras".into()));
        }
        let Some(node) = scene.nodes.get(index) else { continue };
        let world = craft_mesh::mat_mul(&parent, &node.transform);
        if let Some(camera) = node.camera.and_then(|i| scene.cameras.get(i))
            && raw.cameras.len() < super::MAX_CAMERAS
        {
            let (fov, ortho) = camera_projection(camera);
            let name = if !camera.name.is_empty() {
                camera.name.clone()
            } else if !node.name.is_empty() {
                node.name.clone()
            } else {
                format!("Camera {}", raw.cameras.len() + 1)
            };
            raw.cameras.push(RawCamera::placed(name, matrix(world), Vec3::NEG_Z, fov, ortho));
        }
        stack.extend(node.children.iter().rev().map(|&child| (child, world, depth + 1)));
    }
    Ok(())
}

fn camera_projection(camera: &Camera) -> (f32, Option<f32>) {
    match camera.projection {
        Projection::Perspective { yfov, .. } => (yfov.to_degrees() as f32, None),
        Projection::Orthographic { ymag, .. } => (40.0, Some(ymag as f32)),
    }
}
