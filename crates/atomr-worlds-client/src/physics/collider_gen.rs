//! Voxel → rapier collider generation.
//!
//! The heavy, deterministic part (coalescing solid voxels into boxes) lives in
//! the engine-agnostic [`atomr_worlds_physics::box_merge`]; this is the thin
//! rapier adapter that maps boxes into a [`Collider::compound`]. Splitting it
//! this way keeps the testable geometry out of the Bevy/rapier layer.

use atomr_worlds_core::coord::IVec3 as VoxCoord;
// Aliased to avoid clashing with Bevy's `Cuboid` mesh primitive (in the prelude).
use atomr_worlds_physics::box_merge::{greedy_boxes, Cuboid as VoxBox};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};
use bevy::prelude::*;
use bevy_rapier3d::prelude::Collider;

/// Greedy box-merge of a brick's solid voxels (the collision analogue of greedy
/// meshing — a fully-solid brick collapses to a single box).
pub fn brick_boxes(brick: &Brick) -> Vec<VoxBox> {
    let edge = BRICK_EDGE as i32;
    greedy_boxes([edge, edge, edge], |x, y, z| {
        !brick.get(VoxCoord::new(x as i64, y as i64, z as i64)).is_empty()
    })
}

/// One unit box per solid voxel — the un-merged form. Useful as an A/B
/// alternative and as a correctness oracle for the greedy merge (same voxel
/// set, more shapes).
pub fn per_voxel_boxes(brick: &Brick) -> Vec<VoxBox> {
    let edge = BRICK_EDGE as i32;
    let mut out = Vec::new();
    for x in 0..edge {
        for y in 0..edge {
            for z in 0..edge {
                if !brick.get(VoxCoord::new(x as i64, y as i64, z as i64)).is_empty() {
                    out.push(VoxBox {
                        min: [x, y, z],
                        max: [x + 1, y + 1, z + 1],
                    });
                }
            }
        }
    }
    out
}

/// Map a set of half-open voxel boxes into a single brick-local compound
/// collider, scaling voxel units to meters. Returns `None` for an empty set.
pub fn compound_from_boxes(boxes: &[VoxBox], voxel_size_m: f32) -> Option<Collider> {
    if boxes.is_empty() {
        return None;
    }
    let parts: Vec<(Vec3, Quat, Collider)> = boxes
        .iter()
        .map(|b| {
            let size = b.size();
            // Cuboid half-extents (rapier takes half-extents) in meters.
            let half = Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32)
                * 0.5
                * voxel_size_m;
            // Box center, in brick-local meters: min corner + half the extent.
            let center = (Vec3::new(b.min[0] as f32, b.min[1] as f32, b.min[2] as f32)
                + Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32) * 0.5)
                * voxel_size_m;
            (center, Quat::IDENTITY, Collider::cuboid(half.x, half.y, half.z))
        })
        .collect();
    Some(Collider::compound(parts))
}

/// Greedy-box compound collider for a brick (the default strategy's worker).
pub fn brick_to_collider(brick: &Brick, voxel_size_m: f32) -> Option<Collider> {
    compound_from_boxes(&brick_boxes(brick), voxel_size_m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::voxel::Voxel;

    fn brick_with(solids: &[[i32; 3]]) -> Brick {
        let mut b = Brick::new();
        for &[x, y, z] in solids {
            b.set(VoxCoord::new(x as i64, y as i64, z as i64), Voxel::new(1));
        }
        b
    }

    #[test]
    fn empty_brick_has_no_collider() {
        assert!(brick_to_collider(&Brick::new(), 1.0).is_none());
        assert!(compound_from_boxes(&[], 1.0).is_none());
    }

    #[test]
    fn non_empty_brick_yields_collider() {
        let b = brick_with(&[[0, 0, 0], [1, 0, 0], [2, 0, 0]]);
        assert!(brick_to_collider(&b, 1.0).is_some());
    }

    #[test]
    fn greedy_and_per_voxel_cover_same_voxels() {
        // The two strategies must describe the identical solid set: same total
        // covered volume, with greedy using no more boxes than per-voxel.
        let b = brick_with(&[[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [5, 5, 5]]);
        let greedy = brick_boxes(&b);
        let per_voxel = per_voxel_boxes(&b);
        let vol = |boxes: &[VoxBox]| boxes.iter().map(|c| c.volume()).sum::<i64>();
        assert_eq!(vol(&greedy), vol(&per_voxel));
        assert_eq!(vol(&per_voxel), 5, "5 solid voxels → 5 covered cells");
        assert!(greedy.len() <= per_voxel.len());
    }

    #[test]
    fn full_brick_is_single_box() {
        let mut b = Brick::new();
        for x in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for z in 0..BRICK_EDGE as i64 {
                    b.set(VoxCoord::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let boxes = brick_boxes(&b);
        assert_eq!(boxes.len(), 1);
        assert!(brick_to_collider(&b, 1.0).is_some());
    }
}
