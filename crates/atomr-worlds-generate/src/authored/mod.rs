//! Authored-region overlay system — Phase 13d.
//!
//! Stipulation: a user can register a region (AABB) backed by literal
//! voxel data which overlays procedural generation. Procedural fill
//! produces the base brick; authored cells are then written on top
//! before the user-overlay (journalled writes) is applied.
//!
//! Two-phase composition per brick miss:
//! 1. Inner generator (per [`crate::registry::Resolved`]) produces the
//!    procedural baseline.
//! 2. [`AuthoredRegion::apply_to_brick`] overlays its data.
//! 3. Existing per-actor overlay (journalled writes) applies last.
//!
//! Determinism contract: registration is deterministic — same
//! `(world_seed, shape, registered region set)` produces byte-identical
//! brick output across runs.
//!
//! See:
//! - [`AuthoredRegion`] — the trait. Implementors are pure functions of
//!   their registered data.
//! - [`AuthoredRegionStore`] — the per-host registry, shared via `Arc`.
//! - [`LiteralRegion`] — in-memory voxel-data region (the Phase-13d
//!   default; file-backed loaders land in Phase 13e).

pub mod heightmap;
pub mod literal;
pub mod voxfile;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::Brick;

pub use heightmap::{heightmap_from_columns, HeightmapRegion};
pub use literal::LiteralRegion;
pub use voxfile::{VoxFileRegion, VoxelTransform};

/// Stable region identifier — FNV-1a 64-bit hash of the region name.
pub type RegionId = u64;

/// FNV-1a 64-bit hash of a region name, computable at compile time.
pub const fn region_id(name: &str) -> RegionId {
    let bytes = name.as_bytes();
    let mut hash: u64 = 0xCBF2_9CE4_8422_2325;
    let mut i = 0;
    while i < bytes.len() {
        hash ^= bytes[i] as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01B3);
        i += 1;
    }
    hash
}

/// Continuous-meter AABB used for region bounds. Distinct from the
/// integer-voxel-coord `proto::AABB`; this one lives in voxel-index space
/// (i64) since regions are addressed at the voxel-coord level.
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub struct RegionAabb {
    pub min: IVec3,
    pub max: IVec3,
}

impl RegionAabb {
    #[inline]
    pub const fn new(min: IVec3, max: IVec3) -> Self {
        Self { min, max }
    }

    #[inline]
    pub fn contains(&self, p: IVec3) -> bool {
        p.x >= self.min.x
            && p.x < self.max.x
            && p.y >= self.min.y
            && p.y < self.max.y
            && p.z >= self.min.z
            && p.z < self.max.z
    }

    /// True if any voxel in `brick_aabb` (16³ at `brick_coord`) overlaps
    /// this region.
    #[inline]
    pub fn brick_overlaps(&self, brick_coord: IVec3, brick_edge: i64) -> bool {
        let bmin = IVec3::new(
            brick_coord.x * brick_edge,
            brick_coord.y * brick_edge,
            brick_coord.z * brick_edge,
        );
        let bmax = IVec3::new(bmin.x + brick_edge, bmin.y + brick_edge, bmin.z + brick_edge);
        bmin.x < self.max.x
            && bmax.x > self.min.x
            && bmin.y < self.max.y
            && bmax.y > self.min.y
            && bmin.z < self.max.z
            && bmax.z > self.min.z
    }
}

/// An authored region of voxel data that overlays procedural generation.
///
/// Implementors are pure: same brick coord + same region state → same
/// voxels every call. File-backed loaders (Phase 13e) hold their state
/// in memory after loading so the trait stays pure.
pub trait AuthoredRegion: Send + Sync + Debug {
    fn id(&self) -> RegionId;
    fn bounds(&self) -> RegionAabb;
    #[inline]
    fn contains_brick(&self, brick_coord: IVec3, brick_edge: i64) -> bool {
        self.bounds().brick_overlaps(brick_coord, brick_edge)
    }
    /// Apply authored voxels to a brick. The brick is pre-filled with
    /// procedural content; this method writes the authored cells in
    /// brick-local coordinates. Returns the number of voxels written.
    fn apply_to_brick(&self, brick_coord: IVec3, brick: &mut Brick) -> usize;
}

/// Per-host registry of authored regions. Shared via `Arc`; deterministic
/// iteration via a sorted vector of ids for `apply_all`.
#[derive(Default)]
pub struct AuthoredRegionStore {
    by_id: HashMap<RegionId, Arc<dyn AuthoredRegion>>,
}

impl Debug for AuthoredRegionStore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthoredRegionStore")
            .field("regions", &self.by_id.keys().collect::<Vec<_>>())
            .finish()
    }
}

impl AuthoredRegionStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn register(&mut self, region: Arc<dyn AuthoredRegion>) {
        self.by_id.insert(region.id(), region);
    }

    pub fn get(&self, id: RegionId) -> Option<Arc<dyn AuthoredRegion>> {
        self.by_id.get(&id).cloned()
    }

    pub fn contains(&self, id: RegionId) -> bool {
        self.by_id.contains_key(&id)
    }

    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// All registered ids in sorted order (deterministic iteration).
    pub fn ids_sorted(&self) -> Vec<RegionId> {
        let mut v: Vec<_> = self.by_id.keys().copied().collect();
        v.sort();
        v
    }

    /// Apply every region whose bounds overlap the given brick, in
    /// sorted-id order (deterministic). Returns the total voxels written.
    pub fn apply_all(
        &self,
        brick_coord: IVec3,
        brick_edge: i64,
        brick: &mut Brick,
    ) -> usize {
        let mut total = 0;
        for id in self.ids_sorted() {
            if let Some(r) = self.by_id.get(&id) {
                if r.contains_brick(brick_coord, brick_edge) {
                    total += r.apply_to_brick(brick_coord, brick);
                }
            }
        }
        total
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_id_stable_across_runs() {
        assert_eq!(region_id("foo"), region_id("foo"));
        assert_ne!(region_id("foo"), region_id("bar"));
    }

    #[test]
    fn aabb_brick_overlap() {
        let r = RegionAabb::new(IVec3::new(0, 0, 0), IVec3::new(32, 32, 32));
        // Brick at (0,0,0) covers voxels (0..16) — overlaps.
        assert!(r.brick_overlaps(IVec3::new(0, 0, 0), 16));
        // Brick at (3,0,0) covers voxels (48..64) — outside.
        assert!(!r.brick_overlaps(IVec3::new(3, 0, 0), 16));
        // Brick at (1,1,1) covers (16..32) — overlaps the boundary.
        assert!(r.brick_overlaps(IVec3::new(1, 1, 1), 16));
    }
}
