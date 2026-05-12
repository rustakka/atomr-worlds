//! Dense 16³ voxel "brick".
//!
//! 16³ = 4096 voxels × 2 bytes = 8 KiB per brick. Fits in L1, matches
//! VDB/GPU voxel-cone-tracing conventions, and trades off octree depth vs
//! per-leaf memory cost well.

use atomr_worlds_core::coord::IVec3;

use crate::voxel::Voxel;

pub const BRICK_EDGE: usize = 16;
pub const BRICK_LEN: usize = BRICK_EDGE * BRICK_EDGE * BRICK_EDGE; // 4096

#[derive(Clone)]
pub struct Brick {
    pub voxels: Box<[Voxel; BRICK_LEN]>,
    pub nonempty_count: u16,
}

impl std::fmt::Debug for Brick {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Brick").field("nonempty_count", &self.nonempty_count).finish()
    }
}

impl Default for Brick {
    fn default() -> Self {
        Self { voxels: Box::new([Voxel::EMPTY; BRICK_LEN]), nonempty_count: 0 }
    }
}

impl Brick {
    pub fn new() -> Self {
        Self::default()
    }

    /// Convert a 0..16 local coordinate to a flat index. Returns `None` if any
    /// component is out of range.
    #[inline]
    pub const fn local_index(local: IVec3) -> Option<usize> {
        if local.x < 0 || local.y < 0 || local.z < 0 {
            return None;
        }
        let (x, y, z) = (local.x as usize, local.y as usize, local.z as usize);
        if x >= BRICK_EDGE || y >= BRICK_EDGE || z >= BRICK_EDGE {
            return None;
        }
        Some((z * BRICK_EDGE + y) * BRICK_EDGE + x)
    }

    /// Get a voxel by local coordinate. Returns [`Voxel::EMPTY`] if out of range.
    #[inline]
    pub fn get(&self, local: IVec3) -> Voxel {
        match Self::local_index(local) {
            Some(i) => self.voxels[i],
            None => Voxel::EMPTY,
        }
    }

    /// Set every voxel where the predicate returns `true` to `v`. `mask` is
    /// called once per local coordinate in `[0, BRICK_EDGE)³`. Returns the
    /// number of voxels that changed value. Updates `nonempty_count` along
    /// the way. Used by the host's `WriteRegion` flow to apply a brush.
    pub fn set_region<F>(&mut self, mask: F, v: Voxel) -> u32
    where F: Fn(IVec3) -> bool
    {
        let mut changed = 0u32;
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    let p = IVec3::new(x, y, z);
                    if mask(p) && self.set(p, v) {
                        changed += 1;
                    }
                }
            }
        }
        changed
    }

    /// Set a voxel; updates `nonempty_count`. Returns `true` if the cell changed.
    pub fn set(&mut self, local: IVec3, v: Voxel) -> bool {
        let Some(i) = Self::local_index(local) else { return false };
        let prev = self.voxels[i];
        if prev == v {
            return false;
        }
        match (prev.is_empty(), v.is_empty()) {
            (true, false) => self.nonempty_count = self.nonempty_count.saturating_add(1),
            (false, true) => self.nonempty_count = self.nonempty_count.saturating_sub(1),
            _ => {}
        }
        self.voxels[i] = v;
        true
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nonempty_count == 0
    }

    /// Encode as `[count: u16-le, voxels: 4096 × u16-le]`.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(2 + BRICK_LEN * 2);
        out.extend_from_slice(&self.nonempty_count.to_le_bytes());
        out.extend_from_slice(bytemuck::cast_slice::<Voxel, u8>(self.voxels.as_ref()));
        out
    }

    /// Decode a brick from a byte slice produced by [`to_bytes`].
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BrickDecodeError> {
        if bytes.len() != 2 + BRICK_LEN * 2 {
            return Err(BrickDecodeError::Length(bytes.len()));
        }
        let count = u16::from_le_bytes([bytes[0], bytes[1]]);
        let voxel_bytes = &bytes[2..];
        let voxels: &[Voxel] = bytemuck::cast_slice(voxel_bytes);
        let mut arr: Box<[Voxel; BRICK_LEN]> = Box::new([Voxel::EMPTY; BRICK_LEN]);
        arr.copy_from_slice(voxels);
        Ok(Self { voxels: arr, nonempty_count: count })
    }
}

#[derive(Debug, thiserror::Error)]
pub enum BrickDecodeError {
    #[error("brick byte slice has length {0}, expected {}", 2 + BRICK_LEN * 2)]
    Length(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_brick_has_zero_count() {
        let b = Brick::new();
        assert!(b.is_empty());
        assert_eq!(b.nonempty_count, 0);
    }

    #[test]
    fn set_get_round_trip() {
        let mut b = Brick::new();
        let p = IVec3::new(3, 5, 7);
        assert!(b.set(p, Voxel::new(42)));
        assert_eq!(b.get(p), Voxel::new(42));
        assert_eq!(b.nonempty_count, 1);
    }

    #[test]
    fn set_to_same_value_is_noop() {
        let mut b = Brick::new();
        b.set(IVec3::new(0, 0, 0), Voxel::new(1));
        assert!(!b.set(IVec3::new(0, 0, 0), Voxel::new(1)));
        assert_eq!(b.nonempty_count, 1);
    }

    #[test]
    fn clearing_decrements_count() {
        let mut b = Brick::new();
        b.set(IVec3::new(1, 2, 3), Voxel::new(9));
        b.set(IVec3::new(1, 2, 3), Voxel::EMPTY);
        assert_eq!(b.nonempty_count, 0);
        assert!(b.is_empty());
    }

    #[test]
    fn out_of_range_returns_empty() {
        let b = Brick::new();
        assert_eq!(b.get(IVec3::new(16, 0, 0)), Voxel::EMPTY);
        assert_eq!(b.get(IVec3::new(-1, 0, 0)), Voxel::EMPTY);
    }

    #[test]
    fn byte_round_trip() {
        let mut b = Brick::new();
        b.set(IVec3::new(3, 5, 7), Voxel::new(42));
        b.set(IVec3::new(0, 0, 0), Voxel::new(1));
        let bytes = b.to_bytes();
        let back = Brick::from_bytes(&bytes).unwrap();
        assert_eq!(back.nonempty_count, 2);
        assert_eq!(back.get(IVec3::new(3, 5, 7)), Voxel::new(42));
        assert_eq!(back.get(IVec3::new(0, 0, 0)), Voxel::new(1));
    }

    #[test]
    fn bad_byte_length_errors() {
        assert!(Brick::from_bytes(&[]).is_err());
        assert!(Brick::from_bytes(&[0; 100]).is_err());
    }
}
