//! Greedy 3D box-merge: coalesce a region's solid voxels into a small set of
//! axis-aligned boxes.
//!
//! A voxel collider built one-cuboid-per-voxel is correct but heavy — a solid
//! 16³ brick would be 4096 boxes, bloating the broad-phase and memory. Terrain
//! is mostly large solid slabs, so merging runs of solid voxels into maximal
//! boxes collapses that to a handful (a fully-solid brick → a *single* box).
//! This is the collision analogue of greedy meshing, and it feeds the client's
//! rapier compound-collider builder (`brick_to_collider`).
//!
//! Like [`crate::flood_fill::connected_components`], the merge is a pure,
//! deterministic function of a `dims + is_solid` closure: voxels are consumed in
//! `(x, y, z)`-ascending order and each seed grows maximally `+x`, then `+y`,
//! then `+z`, so the same region yields the identical box list on every machine.

use crate::flood_fill::Dims;

/// A half-open axis-aligned voxel box `[min, max)` in voxel coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cuboid {
    /// Inclusive minimum corner.
    pub min: [i32; 3],
    /// Exclusive maximum corner.
    pub max: [i32; 3],
}

impl Cuboid {
    /// Extent in voxels along each axis (`max - min`).
    #[inline]
    pub fn size(&self) -> [i32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    /// Number of voxels the box covers.
    #[inline]
    pub fn volume(&self) -> i64 {
        let s = self.size();
        s[0] as i64 * s[1] as i64 * s[2] as i64
    }
}

/// Coalesce the solid cells of a `dims`-sized region into a set of disjoint
/// boxes whose union is exactly the solid set.
///
/// `is_solid(x, y, z)` is queried only for in-bounds coordinates. The returned
/// boxes are pairwise disjoint and together cover every solid voxel exactly
/// once. A region with a non-positive dimension yields an empty `Vec`.
pub fn greedy_boxes(dims: Dims, is_solid: impl Fn(i32, i32, i32) -> bool) -> Vec<Cuboid> {
    let [nx, ny, nz] = dims;
    if nx <= 0 || ny <= 0 || nz <= 0 {
        return Vec::new();
    }
    let lin = |x: i32, y: i32, z: i32| (x * ny * nz + y * nz + z) as usize;
    let mut consumed = vec![false; (nx * ny * nz) as usize];
    let mut out = Vec::new();

    // A cell can join the box being grown if it's solid and not yet consumed.
    // `consumed` is passed in (not captured) so the seed loop can mutate it
    // after each box is emitted without aliasing the closure's borrow.
    let avail = |x: i32, y: i32, z: i32, consumed: &[bool]| {
        is_solid(x, y, z) && !consumed[lin(x, y, z)]
    };

    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                if !avail(x, y, z, &consumed) {
                    continue;
                }
                // 1) Grow a run along +x.
                let mut xe = x + 1;
                while xe < nx && avail(xe, y, z, &consumed) {
                    xe += 1;
                }
                // 2) Grow along +y while the whole [x, xe) run stays available.
                let mut ye = y + 1;
                'grow_y: while ye < ny {
                    for xi in x..xe {
                        if !avail(xi, ye, z, &consumed) {
                            break 'grow_y;
                        }
                    }
                    ye += 1;
                }
                // 3) Grow along +z while the whole [x, xe) × [y, ye) slab stays
                //    available.
                let mut ze = z + 1;
                'grow_z: while ze < nz {
                    for xi in x..xe {
                        for yi in y..ye {
                            if !avail(xi, yi, ze, &consumed) {
                                break 'grow_z;
                            }
                        }
                    }
                    ze += 1;
                }
                // Consume the box's cells so later seeds skip them, then emit.
                for xi in x..xe {
                    for yi in y..ye {
                        for zi in z..ze {
                            consumed[lin(xi, yi, zi)] = true;
                        }
                    }
                }
                out.push(Cuboid {
                    min: [x, y, z],
                    max: [xe, ye, ze],
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(solid: &[[i32; 3]]) -> impl Fn(i32, i32, i32) -> bool + '_ {
        move |x, y, z| solid.iter().any(|&[a, b, c]| a == x && b == y && c == z)
    }

    /// Assert the boxes are disjoint and cover exactly the solid set: every
    /// solid voxel is in exactly one box, every empty voxel in none.
    fn assert_exact_cover(dims: Dims, is_solid: impl Fn(i32, i32, i32) -> bool, boxes: &[Cuboid]) {
        let [nx, ny, nz] = dims;
        for x in 0..nx {
            for y in 0..ny {
                for z in 0..nz {
                    let hits = boxes
                        .iter()
                        .filter(|b| {
                            (b.min[0]..b.max[0]).contains(&x)
                                && (b.min[1]..b.max[1]).contains(&y)
                                && (b.min[2]..b.max[2]).contains(&z)
                        })
                        .count();
                    let want = if is_solid(x, y, z) { 1 } else { 0 };
                    assert_eq!(
                        hits, want,
                        "voxel ({x},{y},{z}) covered {hits} times, expected {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn empty_region_has_no_boxes() {
        let boxes = greedy_boxes([4, 4, 4], |_, _, _| false);
        assert!(boxes.is_empty());
    }

    #[test]
    fn non_positive_dims_are_empty() {
        assert!(greedy_boxes([0, 4, 4], |_, _, _| true).is_empty());
        assert!(greedy_boxes([4, -1, 4], |_, _, _| true).is_empty());
    }

    #[test]
    fn full_brick_collapses_to_one_box() {
        let dims = [16, 16, 16];
        let boxes = greedy_boxes(dims, |_, _, _| true);
        assert_eq!(boxes.len(), 1, "a solid brick must merge to a single box");
        assert_eq!(
            boxes[0],
            Cuboid {
                min: [0, 0, 0],
                max: [16, 16, 16]
            }
        );
        assert_eq!(boxes[0].volume(), 16 * 16 * 16);
        assert_exact_cover(dims, |_, _, _| true, &boxes);
    }

    #[test]
    fn single_voxel_is_one_unit_box() {
        let dims = [3, 3, 3];
        let solid = [[1, 2, 0]];
        let boxes = greedy_boxes(dims, grid(&solid));
        assert_eq!(
            boxes,
            vec![Cuboid {
                min: [1, 2, 0],
                max: [2, 3, 1]
            }]
        );
        assert_exact_cover(dims, grid(&solid), &boxes);
    }

    #[test]
    fn l_shape_covered_exactly_and_disjoint() {
        // An L in the z=0 plane of a 3×3×1 grid.
        let dims = [3, 3, 1];
        let solid = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0], [0, 2, 0]];
        let boxes = greedy_boxes(dims, grid(&solid));
        assert_exact_cover(dims, grid(&solid), &boxes);
        // The two arms can't be one box, so it must take more than one.
        assert!(boxes.len() >= 2);
    }

    #[test]
    fn checkerboard_is_all_unit_boxes() {
        // No two solids are face-adjacent, so nothing merges.
        let dims = [4, 4, 4];
        let is_solid = |x: i32, y: i32, z: i32| (x + y + z) % 2 == 0;
        let boxes = greedy_boxes(dims, is_solid);
        let solid_count = (0..4)
            .flat_map(|x| (0..4).flat_map(move |y| (0..4).map(move |z| (x, y, z))))
            .filter(|&(x, y, z)| is_solid(x, y, z))
            .count();
        assert_eq!(boxes.len(), solid_count);
        assert!(boxes.iter().all(|b| b.volume() == 1));
        assert_exact_cover(dims, is_solid, &boxes);
    }

    #[test]
    fn slab_merges_along_each_axis() {
        // A 4×1×3 solid slab in a larger grid → exactly one box.
        let dims = [6, 5, 4];
        let is_solid = |x: i32, y: i32, z: i32| (1..5).contains(&x) && y == 2 && (0..3).contains(&z);
        let boxes = greedy_boxes(dims, is_solid);
        assert_eq!(boxes.len(), 1);
        assert_eq!(
            boxes[0],
            Cuboid {
                min: [1, 2, 0],
                max: [5, 3, 3]
            }
        );
        assert_exact_cover(dims, is_solid, &boxes);
    }

    #[test]
    fn merge_is_deterministic() {
        let dims = [7, 6, 5];
        // A reproducible pseudo-pattern (no RNG — determinism is the point).
        let is_solid = |x: i32, y: i32, z: i32| (x * 13 + y * 7 + z * 5) % 3 != 0;
        let a = greedy_boxes(dims, is_solid);
        let b = greedy_boxes(dims, is_solid);
        assert_eq!(a, b);
        assert_exact_cover(dims, is_solid, &a);
    }
}
