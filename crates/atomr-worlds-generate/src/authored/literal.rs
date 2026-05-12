//! In-memory authored region — voxels supplied as a `HashMap<IVec3, Voxel>`.
//!
//! The minimum-viable stipulation source. File loaders (Phase 13e) will
//! either reuse this internally (load file → fill HashMap → wrap in
//! `LiteralRegion`) or implement [`AuthoredRegion`] directly when
//! streaming the file makes sense.

use std::collections::HashMap;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};

use super::{region_id, AuthoredRegion, RegionAabb, RegionId};

#[derive(Debug, Clone)]
pub struct LiteralRegion {
    id: RegionId,
    name: String,
    bounds: RegionAabb,
    voxels: HashMap<IVec3, Voxel>,
}

impl LiteralRegion {
    /// Construct a region from a name + bounds + voxel map. Bounds are
    /// the *outer* extents of the authored data (inclusive min, exclusive
    /// max). Voxels outside the bounds are silently ignored during
    /// application.
    pub fn new(name: impl Into<String>, bounds: RegionAabb, voxels: HashMap<IVec3, Voxel>) -> Self {
        let name = name.into();
        let id = region_id(&name);
        Self { id, name, bounds, voxels }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn voxel_count(&self) -> usize {
        self.voxels.len()
    }
}

impl AuthoredRegion for LiteralRegion {
    fn id(&self) -> RegionId {
        self.id
    }

    fn bounds(&self) -> RegionAabb {
        self.bounds
    }

    fn apply_to_brick(&self, brick_coord: IVec3, brick: &mut Brick) -> usize {
        let edge = BRICK_EDGE as i64;
        let origin = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
        let mut count = 0;
        // Iterate the brick's voxel range; only check the HashMap on each
        // covered cell. This is O(brick_edge³) — fine for sparse authored
        // data (most cells will miss) and the bound is < 4096.
        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let p = IVec3::new(origin.x + lx, origin.y + ly, origin.z + lz);
                    if !self.bounds.contains(p) {
                        continue;
                    }
                    if let Some(v) = self.voxels.get(&p) {
                        brick.set(IVec3::new(lx, ly, lz), *v);
                        count += 1;
                    }
                }
            }
        }
        count
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lr(name: &str, min: IVec3, max: IVec3, voxels: Vec<(IVec3, u16)>) -> LiteralRegion {
        let mut m = HashMap::new();
        for (p, v) in voxels {
            m.insert(p, Voxel::new(v));
        }
        LiteralRegion::new(name, RegionAabb::new(min, max), m)
    }

    #[test]
    fn apply_writes_inside_bounds() {
        let r = lr("test", IVec3::new(0, 0, 0), IVec3::new(16, 16, 16),
                   vec![(IVec3::new(2, 2, 2), 42)]);
        let mut brick = Brick::new();
        let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut brick);
        assert_eq!(written, 1);
        assert_eq!(brick.get(IVec3::new(2, 2, 2)), Voxel::new(42));
    }

    #[test]
    fn apply_skips_voxels_outside_bounds() {
        // Voxel coord (100, 0, 0) is outside bounds (0..16, 0..16, 0..16).
        let r = lr("test", IVec3::new(0, 0, 0), IVec3::new(16, 16, 16),
                   vec![(IVec3::new(100, 0, 0), 42)]);
        let mut brick = Brick::new();
        let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut brick);
        assert_eq!(written, 0);
    }

    #[test]
    fn apply_targets_correct_brick() {
        // Region voxel at (18, 0, 0) → brick (1, 0, 0), local (2, 0, 0).
        let r = lr("test", IVec3::new(16, 0, 0), IVec3::new(32, 16, 16),
                   vec![(IVec3::new(18, 0, 0), 7)]);
        let mut brick = Brick::new();
        let written = r.apply_to_brick(IVec3::new(1, 0, 0), &mut brick);
        assert_eq!(written, 1);
        assert_eq!(brick.get(IVec3::new(2, 0, 0)), Voxel::new(7));

        // Same voxel applied to brick (0, 0, 0) writes nothing.
        let mut brick0 = Brick::new();
        let w0 = r.apply_to_brick(IVec3::new(0, 0, 0), &mut brick0);
        assert_eq!(w0, 0);
    }

    #[test]
    fn id_is_stable_for_same_name() {
        let a = LiteralRegion::new("name", RegionAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1)),
                                   HashMap::new());
        let b = LiteralRegion::new("name", RegionAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1)),
                                   HashMap::new());
        assert_eq!(a.id(), b.id());
    }
}
