//! Coordinate hashing used as the noise lattice's "random" function.

use atomr_worlds_core::seed::splitmix64;

#[inline]
fn mix3(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    let mut h = splitmix64(seed);
    h = splitmix64(h ^ (x as u64));
    h = splitmix64(h ^ (y as u64).rotate_left(21));
    splitmix64(h ^ (z as u64).rotate_left(42))
}

/// Hash a 3-D integer lattice point to a `u64`.
#[inline]
pub fn hash3_u64(seed: u64, x: i64, y: i64, z: i64) -> u64 {
    mix3(seed, x, y, z)
}

/// Hash a 3-D integer lattice point to an `f32` in `[0.0, 1.0)`.
#[inline]
pub fn hash3_f01(seed: u64, x: i64, y: i64, z: i64) -> f32 {
    // top 24 bits → mantissa-sized integer → scale into [0, 1).
    let top = (mix3(seed, x, y, z) >> 40) as u32;
    (top as f32) / ((1u32 << 24) as f32)
}

/// Hash a 3-D integer lattice point to a 3-component unit vector (Perlin gradients).
#[inline]
pub fn hash3_gradient(seed: u64, x: i64, y: i64, z: i64) -> [f32; 3] {
    // 12 canonical Perlin gradients ([±1,±1,0], [±1,0,±1], [0,±1,±1]).
    const GRADS: [[f32; 3]; 12] = [
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
        [1.0, -1.0, 0.0],
        [-1.0, -1.0, 0.0],
        [1.0, 0.0, 1.0],
        [-1.0, 0.0, 1.0],
        [1.0, 0.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, 1.0],
        [0.0, -1.0, 1.0],
        [0.0, 1.0, -1.0],
        [0.0, -1.0, -1.0],
    ];
    GRADS[(mix3(seed, x, y, z) % 12) as usize]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hash_is_deterministic() {
        assert_eq!(hash3_u64(7, 1, 2, 3), hash3_u64(7, 1, 2, 3));
        assert_ne!(hash3_u64(7, 1, 2, 3), hash3_u64(8, 1, 2, 3));
    }

    #[test]
    fn hash_f01_is_in_range() {
        for seed in 0..50u64 {
            for x in -10..10i64 {
                let v = hash3_f01(seed, x, x * 3, x * 7);
                assert!((0.0..1.0).contains(&v), "value {v} out of [0,1)");
            }
        }
    }
}
