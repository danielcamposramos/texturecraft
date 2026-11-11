//! Decoding imported image resources (fills, alphas, stencils).

use crate::{FormatError, MAX_IMAGE_SIZE, Result};

/// A decoded image: RGBA as 0..1 floats in the file's own encoding (8/16-bit files stay
/// sRGB-encoded; `srgb` says so), row-major, top row first.
#[derive(Clone, Debug, PartialEq)]
pub struct DecodedImage {
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<[f32; 4]>,
    /// True for 8/16-bit images (sRGB-encoded); false for float images (linear).
    pub srgb: bool,
}

/// Texel cap of [`decode_image`]: the documented side limit squared (16384 x 16384), checked from
/// the header so no larger image is ever allocated.
pub const MAX_DECODE_TEXELS: u64 = MAX_IMAGE_SIZE as u64 * MAX_IMAGE_SIZE as u64;

/// Decode PNG, JPEG, TGA, TIFF, BMP, WEBP, EXR or HDR bytes (at most [`MAX_DECODE_TEXELS`]).
pub fn decode_image(bytes: &[u8]) -> Result<DecodedImage> {
    decode_image_within(bytes, MAX_DECODE_TEXELS)
}

/// [`decode_image`] for an image of at most `max_texels` (width × height), checked from the
/// header, before any pixel is decoded or allocated.
pub fn decode_image_within(bytes: &[u8], max_texels: u64) -> Result<DecodedImage> {
    let mut reader =
        image::ImageReader::new(std::io::Cursor::new(bytes)).with_guessed_format().map_err(|e| FormatError::Malformed { kind: "image", msg: e.to_string() })?;

    // TGA has no magic signature; its decoder validates the header when no format was recognized
    let format = reader.format().unwrap_or(image::ImageFormat::Tga);
    reader.set_format(format);

    let (width, height) = reader.into_dimensions().map_err(|e| FormatError::Malformed { kind: "image", msg: e.to_string() })?;
    if width == 0 || height == 0 || width > MAX_IMAGE_SIZE || height > MAX_IMAGE_SIZE {
        return Err(FormatError::TooLarge(format!("image is {width}x{height} (limit {MAX_IMAGE_SIZE})")));
    }
    if u64::from(width) * u64::from(height) > max_texels {
        return Err(FormatError::TooLarge(format!("image is {width}x{height}, more than {max_texels} texels")));
    }

    let image =
        crate::guard("image", || image::load_from_memory_with_format(bytes, format).map_err(|e| FormatError::Malformed { kind: "image", msg: e.to_string() }))?;
    let srgb = !matches!(image, image::DynamicImage::ImageRgb32F(_) | image::DynamicImage::ImageRgba32F(_));
    let float_image = image.into_rgba32f();
    let (width, height) = float_image.dimensions();
    let rgba = float_image.pixels().map(|pixel| pixel.0).collect();
    Ok(DecodedImage { width, height, rgba, srgb })
}

/// PNG bytes of an `rgb` (sRGB) image whose alpha is `coverage` (`width × height`, top row first):
/// drawn text as an image resource.
pub fn coverage_png(width: u32, height: u32, coverage: &[u8], rgb: [u8; 3]) -> Result<Vec<u8>> {
    let rgba: Vec<u8> = coverage.iter().flat_map(|a| [rgb[0], rgb[1], rgb[2], *a]).collect();
    let img = image::RgbaImage::from_raw(width, height, rgba)
        .ok_or_else(|| FormatError::Invalid(format!("{} coverage values for {width}x{height}", coverage.len())))?;
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).map_err(|e| FormatError::Invalid(e.to_string()))?;
    Ok(png)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn decodes_png_and_rejects_garbage() {
        let mut png = Vec::new();
        let img = image::RgbaImage::from_fn(4, 2, |x, _| image::Rgba([x as u8 * 60, 0, 0, 255]));
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
        let d = decode_image(&png).unwrap();
        assert_eq!((d.width, d.height), (4, 2));
        assert!(d.srgb);
        assert!((d.rgba[1][0] - 60.0 / 255.0).abs() < 1e-3);
        assert!(decode_image(b"not an image").is_err());
        assert!(decode_image(&png[..png.len() / 2]).is_err());
        let text = decode_image(&coverage_png(2, 1, &[0, 255], [255; 3]).unwrap()).unwrap();
        assert_eq!(text.rgba, vec![[1.0, 1.0, 1.0, 0.0], [1.0; 4]]);
        assert!(coverage_png(2, 2, &[0], [255; 3]).is_err());
    }

    #[test]
    fn tga_without_a_magic_signature_decodes_within_the_texel_budget() {
        let mut bytes = Vec::new();
        image::RgbaImage::from_pixel(4, 4, image::Rgba([128, 51, 255, 255])).write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Tga).unwrap();
        let image = decode_image_within(&bytes, 16).unwrap();
        assert_eq!((image.width, image.height), (4, 4));
        assert!((image.rgba[0][1] - 0.2).abs() < 1e-6);
        assert!(image.srgb);
        let header = &bytes[..18];
        assert!(matches!(decode_image_within(header, 15), Err(FormatError::TooLarge(_))));
        assert!(decode_image(header).is_err());
    }

    #[test]
    fn decoding_within_a_texel_limit_refuses_big_images_from_the_header() {
        let png = {
            let mut b = Vec::new();
            image::RgbImage::from_pixel(8, 8, image::Rgb([1, 2, 3])).write_to(&mut std::io::Cursor::new(&mut b), image::ImageFormat::Png).unwrap();
            b
        };
        assert_eq!(decode_image_within(&png, 64).unwrap().width, 8);
        assert!(matches!(decode_image_within(&png, 63), Err(FormatError::TooLarge(_))));
        assert!(decode_image(&png).is_ok());
    }

    #[test]
    fn decode_image_caps_texels_at_the_side_limit_squared() {
        let mut png = Vec::new();
        image::RgbImage::from_pixel(2, 2, image::Rgb([1, 2, 3])).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();

        assert_eq!(MAX_DECODE_TEXELS, 1 << 28);
        assert_eq!(decode_image(&png).unwrap(), decode_image_within(&png, MAX_DECODE_TEXELS).unwrap());
        assert!(matches!(decode_image_within(&png, 3), Err(FormatError::TooLarge(_))));
    }
}
