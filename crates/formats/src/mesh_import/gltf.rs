//! Map the shared glTF reader onto TextureCraft's world-space mesh and material model.

use super::RawMesh;
use crate::Result;

pub(crate) fn read(bytes: &[u8]) -> Result<RawMesh> {
    super::shared::gltf(bytes)
}
