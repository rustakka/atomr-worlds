//! Optional 4-bit-per-voxel skylight overlay for a single brick.
//!
//! Two nibbles per byte halves memory vs a u8 grid; the overlay is derived
//! state that the renderer fills out lazily.

use crate::brick::{BRICK_EDGE, BRICK_LEN};

pub const LIGHT_OVERLAY_BYTES: usize = BRICK_LEN / 2;

#[derive(Clone)]
pub struct LightOverlay {
    data: Box<[u8; LIGHT_OVERLAY_BYTES]>,
}

impl LightOverlay {
    pub fn new_zero() -> Self {
        Self { data: Box::new([0u8; LIGHT_OVERLAY_BYTES]) }
    }

    pub fn new_full() -> Self {
        // 0xFF stores two nibbles of 15 each, so a full byte represents
        // two voxels at the max skylight level.
        Self { data: Box::new([0xFFu8; LIGHT_OVERLAY_BYTES]) }
    }

    #[inline]
    fn flat_index(x: u8, y: u8, z: u8) -> usize {
        let (x, y, z) = (x as usize, y as usize, z as usize);
        debug_assert!(x < BRICK_EDGE && y < BRICK_EDGE && z < BRICK_EDGE);
        (z * BRICK_EDGE + y) * BRICK_EDGE + x
    }

    pub fn get(&self, x: u8, y: u8, z: u8) -> u8 {
        let i = Self::flat_index(x, y, z);
        let byte = self.data[i >> 1];
        if (i & 1) == 0 { byte & 0x0F } else { byte >> 4 }
    }

    pub fn set(&mut self, x: u8, y: u8, z: u8, level: u8) {
        let level = level.min(15);
        let i = Self::flat_index(x, y, z);
        let slot = i >> 1;
        let byte = self.data[slot];
        let new = if (i & 1) == 0 { (byte & 0xF0) | level } else { (byte & 0x0F) | (level << 4) };
        self.data[slot] = new;
    }

    pub fn as_bytes(&self) -> &[u8] {
        self.data.as_ref()
    }
}

impl std::fmt::Debug for LightOverlay {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut sum: u64 = 0;
        let mut hist = [0u32; 16];
        for &b in self.data.iter() {
            let lo = (b & 0x0F) as usize;
            let hi = (b >> 4) as usize;
            hist[lo] += 1;
            hist[hi] += 1;
            sum += lo as u64 + hi as u64;
        }
        f.debug_struct("LightOverlay")
            .field("sum", &sum)
            .field("histogram", &hist)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_zero_is_all_zeros() {
        let lo = LightOverlay::new_zero();
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    assert_eq!(lo.get(x, y, z), 0);
                }
            }
        }
    }

    #[test]
    fn new_full_is_all_fifteens() {
        let lo = LightOverlay::new_full();
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    assert_eq!(lo.get(x, y, z), 15);
                }
            }
        }
    }

    #[test]
    fn get_set_round_trip() {
        let mut lo = LightOverlay::new_zero();
        lo.set(3, 5, 7, 9);
        lo.set(0, 0, 0, 1);
        lo.set(15, 15, 15, 15);
        assert_eq!(lo.get(3, 5, 7), 9);
        assert_eq!(lo.get(0, 0, 0), 1);
        assert_eq!(lo.get(15, 15, 15), 15);
        // Neighbours are independent: setting (0,0,0) and (1,0,0) share a byte.
        lo.set(1, 0, 0, 12);
        assert_eq!(lo.get(0, 0, 0), 1);
        assert_eq!(lo.get(1, 0, 0), 12);
    }

    #[test]
    fn level_is_clamped() {
        let mut lo = LightOverlay::new_zero();
        lo.set(2, 2, 2, 200);
        assert_eq!(lo.get(2, 2, 2), 15);
    }

    #[test]
    fn debug_does_not_dump_all_bytes() {
        let lo = LightOverlay::new_full();
        let s = format!("{lo:?}");
        assert!(s.contains("sum"));
        assert!(s.len() < 512);
    }
}
