//! Deterministic, cross-platform seed derivation.
//!
//! Built on SplitMix64's finalizer — 12 lines, excellent avalanche, no
//! floating-point, no platform-dependent hashing, and `const fn`.
//!
//! # Hierarchical hash invariant
//!
//! Every parent → child seed transition in the system MUST go through
//! [`derive_child`] (or its underlying [`child_seed`]). The rule is:
//!
//! ```text
//! child_seed = hash(parent_seed, identifier.dim(), identifier.coord())
//! ```
//!
//! where `identifier` implements [`HierarchicalIdentifier`]. Adding a new
//! addressable tier means:
//!
//! 1. Define an identifier type for the tier.
//! 2. Implement [`HierarchicalIdentifier`] on it (reduce to `dim: u32` and
//!    `coord: IVec3`; pack non-spatial identifiers into `IVec3`).
//! 3. Compose it after the parent's seed using [`derive_child`].
//!
//! No tier-specific hash function is ever added — the same primitive applies
//! uniformly at every level, including future tiers below `World` (vehicles,
//! entity slots, variable-depth wrappers).
//!
//! Walking [`crate::WorldAddr::seed_chain`] produces a deterministic
//! `[u64; 5]` of seeds for `[universe, galaxy, sector, system, world]`
//! from a single root seed by repeatedly applying this rule.

use crate::coord::IVec3;
use crate::dim::DimensionId;

/// Anything that addresses a single tier of the hierarchy reduces to a
/// `(dim, coord)` pair fed into [`child_seed`]. Implementations exist for
/// [`crate::LevelKey`] (the standard five tiers) and for sub-world tiers
/// (e.g. vehicle slots) that pack their identifier into `IVec3`.
///
/// See the module-level invariant statement.
pub trait HierarchicalIdentifier {
    fn dim(&self) -> DimensionId;
    fn coord(&self) -> IVec3;
}

/// Derive a child seed from a parent seed and any [`HierarchicalIdentifier`].
///
/// This is the *only* parent → child seed transition in the system. New tiers
/// implement [`HierarchicalIdentifier`] and call this; no tier-specific hash
/// function is ever introduced.
#[inline]
pub fn derive_child<I: HierarchicalIdentifier + ?Sized>(parent: u64, id: &I) -> u64 {
    child_seed(parent, id.dim(), id.coord())
}

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

    struct TestId(DimensionId, IVec3);
    impl HierarchicalIdentifier for TestId {
        fn dim(&self) -> DimensionId { self.0 }
        fn coord(&self) -> IVec3 { self.1 }
    }

    #[test]
    fn derive_child_matches_child_seed() {
        let id = TestId(7, IVec3::new(-3, 4, 9));
        assert_eq!(derive_child(0xDEAD_BEEF, &id), child_seed(0xDEAD_BEEF, 7, IVec3::new(-3, 4, 9)));
    }
}
