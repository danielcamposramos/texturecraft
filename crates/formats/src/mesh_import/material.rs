//! A material's PBR inputs as a mesh file embeds them, turned into the maps of a fill layer.

use texturecraft_core::Channel;
use texturecraft_geom::glam::{Vec3, Vec4};

use crate::{FormatError, Result};

/// A material's PBR inputs as the file gives them: linear factors (None: the file has none)
/// and encoded textures (glTF conventions: metallic in blue and roughness in green of one
/// texture, occlusion in red, OpenGL normal maps).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialTextures {
    pub material: String,
    pub base_color: Option<Vec4>,
    pub base_color_image: Option<Vec<u8>>,
    pub metallic: Option<f32>,
    pub roughness: Option<f32>,
    pub metallic_roughness_image: Option<Vec<u8>>,
    pub normal_image: Option<Vec<u8>>,
    pub occlusion_image: Option<Vec<u8>>,
    pub emissive: Option<Vec3>,
    pub emissive_image: Option<Vec<u8>>,
}

/// What a fill layer needs for one material: encoded images (with a file extension) and
/// constant values (linear) per channel, plus textures that could not be used.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialMaps {
    pub images: Vec<(Channel, Vec<u8>, &'static str)>,
    pub values: Vec<(Channel, [f32; 3])>,
    pub warnings: Vec<String>,
}

impl MaterialTextures {
    /// True when the material carries at least one texture.
    pub fn has_images(&self) -> bool {
        [&self.base_color_image, &self.metallic_roughness_image, &self.normal_image, &self.occlusion_image, &self.emissive_image].iter().any(|i| i.is_some())
    }

    /// The fill maps of the material: textures with their factors applied (decoded only when a
    /// factor changes them or channels must be split), factors without a texture as values.
    pub fn maps(&self) -> MaterialMaps {
        let mut out = MaterialMaps::default();
        let name = &self.material;
        let image = |out: &mut MaterialMaps, c: Channel, r: Result<Vec<u8>>| match r {
            Ok(bytes) => {
                let ext = image::guess_format(&bytes).ok().and_then(|f| f.extensions_str().first().copied()).unwrap_or("png");
                out.images.push((c, bytes, ext));
            }
            Err(e) => out.warnings.push(format!("the {} texture of {name}: {e}", c.id())),
        };
        let factor = self.base_color.unwrap_or(Vec4::ONE);
        match &self.base_color_image {
            Some(b) if factor == Vec4::ONE => image(&mut out, Channel::BaseColor, Ok(b.clone())),
            Some(b) => image(&mut out, Channel::BaseColor, scaled_color(b, factor)),
            None => out.values.push((Channel::BaseColor, factor.truncate().to_array())),
        }
        let (metallic, roughness) = (self.metallic.unwrap_or(1.0), self.roughness.unwrap_or(1.0));
        match &self.metallic_roughness_image {
            Some(b) => match decode(b) {
                Ok(img) => {
                    image(&mut out, Channel::Metallic, gray_png(&img, |p| p[2], metallic));
                    image(&mut out, Channel::Roughness, gray_png(&img, |p| p[1], roughness));
                }
                Err(e) => out.warnings.push(format!("the metallic/roughness texture of {name}: {e}")),
            },
            None => {
                out.values.extend(self.metallic.map(|v| (Channel::Metallic, [v; 3])));
                out.values.extend(self.roughness.map(|v| (Channel::Roughness, [v; 3])));
            }
        }
        if let Some(b) = &self.normal_image {
            image(&mut out, Channel::Normal, Ok(b.clone()));
        }
        if let Some(b) = &self.occlusion_image {
            image(&mut out, Channel::AmbientOcclusion, Ok(b.clone()));
        }
        let emissive = self.emissive.unwrap_or(Vec3::ZERO);
        match &self.emissive_image {
            Some(b) if emissive == Vec3::ONE => image(&mut out, Channel::Emissive, Ok(b.clone())),
            Some(b) => image(&mut out, Channel::Emissive, scaled_color(b, emissive.extend(1.0))),
            None if emissive != Vec3::ZERO => out.values.push((Channel::Emissive, emissive.to_array())),
            None => {}
        }
        out
    }
}

fn decode(bytes: &[u8]) -> Result<image::RgbaImage> {
    let bad = |e: image::ImageError| FormatError::Malformed { kind: "image", msg: e.to_string() };
    let mut reader = image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| bad(e.into()))?;
    let mut limits = image::Limits::default();
    limits.max_image_width = Some(crate::MAX_IMAGE_SIZE);
    limits.max_image_height = Some(crate::MAX_IMAGE_SIZE);
    reader.limits(limits);
    Ok(crate::guard("image", || reader.decode().map_err(bad))?.into_rgba8())
}

fn png(img: impl Into<image::DynamicImage>) -> Result<Vec<u8>> {
    let mut png = Vec::new();
    img.into().write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| FormatError::Invalid(e.to_string()))?;
    Ok(png)
}

/// An sRGB color texture times a linear `factor` (alpha times its alpha), as PNG.
fn scaled_color(bytes: &[u8], factor: Vec4) -> Result<Vec<u8>> {
    use texturecraft_texel::{linear_to_srgb, srgb_to_linear};
    let mut img = decode(bytes)?;
    let f = factor.to_array().map(|v| if v.is_finite() { v.max(0.0) } else { 1.0 });
    let byte = |v: f32| (v.clamp(0.0, 1.0) * 255.0).round() as u8;
    for p in img.pixels_mut() {
        for k in 0..3 {
            p.0[k] = byte(linear_to_srgb(srgb_to_linear(f32::from(p.0[k]) / 255.0) * f[k]));
        }
        p.0[3] = byte(f32::from(p.0[3]) / 255.0 * f[3]);
    }
    png(img)
}

/// One component of `img` times `factor`, as a grayscale PNG.
fn gray_png(img: &image::RgbaImage, component: impl Fn(&[u8; 4]) -> u8, factor: f32) -> Result<Vec<u8>> {
    let factor = if factor.is_finite() { factor.clamp(0.0, 1.0) } else { 1.0 };
    let gray =
        image::GrayImage::from_fn(img.width(), img.height(), |x, y| image::Luma([(f32::from(component(&img.get_pixel(x, y).0)) * factor).round() as u8]));
    png(gray)
}
