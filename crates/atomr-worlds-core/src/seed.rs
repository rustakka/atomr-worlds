//! Deterministic, cross-platform seed derivation.
//!
//! Built on SplitMix64's finalizer — 12 lines, excellent avalanche, no
//! floating-point, no platform-dependent hashing, and `const fn`.
//!
//! The hierarchical principle is:
//!
//! ```text
//! child_seed = hash(parent_seed, dim_id, child_coord)
//! ```
//!
//! Walking [`crate::WorldAddr::seed_chain`] produces a deterministic
//! `[u64; 5]` of seeds for `[universe, galaxy, sector, system, world]`
//! from a single root seed.

use crate::coord::IVec3;

/// SplitMix64 finalizer. See <https://prng.di.unimi.it/splitmix64.c>.
#[inline]
pub const fn splitmix64(mut z: u64) -> u64 {
    z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Derive a child seed from a parent seed, dimension id, and child coordinate.
///
/// Pure, deterministic, endian-independent at the level of the `u64` result.
pub const fn child_seed(parent: u64, dim: u32, coord: IVec3) -> u64 {
    let mut h = splitmix64(parent);
    h = splitmix64(h ^ (dim as u64).wrapping_mul(0xD1B5_4A32_D192_ED03));
    h = splitmix64(h ^ (coord.x as u64));
    h = splitmix64(h ^ (coord.y as u64).rotate_left(21));
    h = splitmix64(h ^ (coord.z as u64).rotate_left(42));
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splitmix_is_deterministic() {
        assert_eq!(splitmix64(0), splitmix64(0));
        assert_ne!(splitmix64(0), splitmix64(1));
    }

    #[test]
    fn child_seed_is_pure() {
        let a = child_seed(42, 0, IVec3::new(1, 2, 3));
        let b = child_seed(42, 0, IVec3::new(1, 2, 3));
        assert_eq!(a, b);
    }

    #[test]
    fn child_seed_differentiates_dim() {
        assert_ne!(
            child_seed(42, 0, IVec3::new(1, 2, 3)),
            child_seed(42, 1, IVec3::new(1, 2, 3))
        );
    }

    #[test]
    fn child_seed_differentiates_coord() {
        assert_ne!(
            child_seed(42, 0, IVec3::new(1, 2, 3)),
            child_seed(42, 0, IVec3::new(1, 2, 4))
        );
    }

    #[test]
    fn child_seed_differentiates_parent() {
        assert_ne!(
            child_seed(42, 0, IVec3::new(1, 2, 3)),
            child_seed(43, 0, IVec3::new(1, 2, 3))
        );
    }
}
