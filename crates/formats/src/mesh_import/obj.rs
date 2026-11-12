//! Map the shared Wavefront reader onto TextureCraft's mesh model.

use super::RawMesh;
use crate::Result;

pub(crate) fn read(bytes: &[u8]) -> Result<RawMesh> {
    super::shared::obj(bytes)
}
