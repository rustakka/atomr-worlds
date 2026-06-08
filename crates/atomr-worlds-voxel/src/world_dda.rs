//! World-space voxel picker — an Amanatides–Woo DDA over the *unbounded*
//! integer voxel grid (1 m per voxel in render/index space).
//!
//! ## Explicitly NOT a WGSL mirror
//!
//! This is a sibling of [`ray_dda_first_hit`](crate::ray_dda_first_hit), but
//! unlike that function it carries **no determinism-gate obligation** and is
//! free to evolve independently of the fragment raymarcher. Differences are
//! deliberate:
//!
//! - **f64**, not f32 — world coordinates are [`DVec3`](atomr_worlds_core::coord::DVec3);
//!   there is no GPU-precision constraint.
//! - **Unbounded** — it marches through any number of bricks via the caller's
//!   `sample(cell)` closure rather than a single fixed `[0, edge]³` box.
//! - **CPU gameplay queries** — voxel editing / picking now; NPC line-of-sight
//!   and server-side ray queries later.
//!
//! Keep it Bevy-free and unit-testable. Do **not** add it to
//! [`raymarch`](crate::raymarch)'s "keep in lock-step with the WGSL" contract.

use atomr_worlds_core::coord::IVec3;

use crate::voxel::Voxel;

/// The first solid voxel a world-space ray encounters, plus the empty neighbour
/// to place against.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorldRayHit {
    /// Integer world-voxel cell of the first solid voxel hit (the *remove*
    /// target).
    pub cell: IVec3,
    /// Outward face normal at the entry face: `-step` on the axis the ray
    /// crossed to enter `cell`. [`IVec3::ZERO`] when the ray *originated inside*
    /// a solid voxel — there is no exposed face, so the caller should treat the
    /// hit as remove-only.
    pub normal: IVec3,
    /// The empty neighbour cell to place into: `cell + normal`. Equals `cell`
    /// when `normal == ZERO` (origin inside solid).
    pub place_cell: IVec3,
    /// Material id at `cell`.
    pub material: u16,
    /// Ray parameter `t` at which the ray enters `cell`. Because `dir` is
    /// normalized internally and the grid is 1 m per voxel, `t` is in meters.
    pub t_entry: f64,
}

/// March `origin + t·dir` through the unbounded integer voxel grid and return
/// the first solid voxel within `max_reach_m` meters, or `None` on a miss.
///
/// `sample(cell)` returns the voxel at an integer world-voxel coordinate;
/// [`Voxel::EMPTY`] means air (and is what callers should return for any cell
/// whose brick is not resident). `dir` need not be normalized — it is
/// normalized internally so `t_entry`/`max_reach_m` are in meters. A zero-length
/// `dir` or non-positive `max_reach_m` yields `None`.
///
/// When `origin` is already inside a solid voxel the hit is that voxel with
/// `normal == ZERO`, `place_cell == cell`, and `t_entry == 0.0`.
pub fn world_ray_first_solid<F>(
    origin: [f64; 3],
    dir: [f64; 3],
    max_reach_m: f64,
    mut sample: F,
) -> Option<WorldRayHit>
where
    F: FnMut(IVec3) -> Voxel,
{
    let len = (dir[0] * dir[0] + dir[1] * dir[1] + dir[2] * dir[2]).sqrt();
    if !(len > 0.0) || !(max_reach_m > 0.0) {
        return None;
    }
    let d = [dir[0] / len, dir[1] / len, dir[2] / len];

    let ivc = |c: [i64; 3]| IVec3::new(c[0], c[1], c[2]);

    // The voxel grid is 1 m per cell, so the integer cell is the floor of the
    // world position. `floor` (not truncation) gives the correct cell for
    // negative coordinates.
    let mut cell = [origin[0].floor() as i64, origin[1].floor() as i64, origin[2].floor() as i64];

    // Origin already inside a solid voxel → remove-only hit, no exposed face.
    let v0 = sample(ivc(cell));
    if !v0.is_empty() {
        return Some(WorldRayHit {
            cell: ivc(cell),
            normal: IVec3::ZERO,
            place_cell: ivc(cell),
            material: v0.0,
            t_entry: 0.0,
        });
    }

    // Per-axis DDA setup. An axis with `d[a] == 0` never advances (step 0,
    // t_max ∞), so it is inert in the nearest-boundary selection below.
    let mut step = [0i64; 3];
    let mut t_max = [f64::INFINITY; 3];
    let mut t_delta = [f64::INFINITY; 3];
    for a in 0..3 {
        if d[a] > 0.0 {
            step[a] = 1;
            t_max[a] = ((cell[a] + 1) as f64 - origin[a]) / d[a];
            t_delta[a] = 1.0 / d[a];
        } else if d[a] < 0.0 {
            step[a] = -1;
            t_max[a] = (cell[a] as f64 - origin[a]) / d[a];
            t_delta[a] = -1.0 / d[a];
        }
    }

    // Backstop on iteration count; the `t_max[a] > max_reach_m` early-out below
    // is the real terminator. Boundary crossings within reach number at most
    // ~reach·√3, so 3·reach + 8 is comfortably sufficient.
    let max_steps = ((max_reach_m.ceil() as i64).saturating_mul(3) + 8).max(8) as u64;
    let mut t_entry;
    for _ in 0..max_steps {
        // Cross the nearest axis boundary.
        let a = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        // The boundary we are about to cross is the entry-t of the next cell;
        // if that is already beyond reach, every remaining cell is too.
        if t_max[a] > max_reach_m {
            return None;
        }
        cell[a] += step[a];
        t_entry = t_max[a];
        t_max[a] += t_delta[a];

        let v = sample(ivc(cell));
        if !v.is_empty() {
            let mut normal = [0i64; 3];
            normal[a] = -step[a];
            let place = [cell[0] + normal[0], cell[1] + normal[1], cell[2] + normal[2]];
            return Some(WorldRayHit {
                cell: ivc(cell),
                normal: ivc(normal),
                place_cell: ivc(place),
                material: v.0,
                t_entry,
            });
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a sampler over an explicit solid set `(x, y, z, material)`.
    fn solids(set: &'static [(i64, i64, i64, u16)]) -> impl FnMut(IVec3) -> Voxel {
        move |c: IVec3| {
            for &(x, y, z, m) in set {
                if c == IVec3::new(x, y, z) {
                    return Voxel::new(m);
                }
            }
            Voxel::EMPTY
        }
    }

    #[test]
    fn axis_aligned_remove() {
        let hit = world_ray_first_solid(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            10.0,
            solids(&[(5, 0, 0, 7)]),
        )
        .expect("ray should hit the solid voxel");
        assert_eq!(hit.cell, IVec3::new(5, 0, 0));
        assert_eq!(hit.normal, IVec3::new(-1, 0, 0), "entered through the -x face");
        assert_eq!(hit.place_cell, IVec3::new(4, 0, 0));
        assert_eq!(hit.material, 7);
    }

    #[test]
    fn ground_slab_place_cell_is_empty_neighbour() {
        // Infinite ground plane at y == 0; look straight down from above.
        let hit = world_ray_first_solid([0.5, 5.5, 0.5], [0.0, -1.0, 0.0], 16.0, |c: IVec3| {
            if c.y == 0 {
                Voxel::new(3)
            } else {
                Voxel::EMPTY
            }
        })
        .expect("downward ray should hit the ground");
        assert_eq!(hit.cell, IVec3::new(0, 0, 0));
        assert_eq!(hit.normal, IVec3::new(0, 1, 0), "entered through the +y (top) face");
        assert_eq!(hit.place_cell, IVec3::new(0, 1, 0), "place into the empty cell above");
        assert_eq!(hit.material, 3);
    }

    #[test]
    fn deterministic_across_runs() {
        let q = || {
            world_ray_first_solid(
                [-2.3, 4.1, 0.7],
                [1.0, -1.0, 0.3],
                32.0,
                solids(&[(3, 0, 1, 9), (7, -3, 2, 4)]),
            )
        };
        assert_eq!(q(), q());
    }

    #[test]
    fn origin_inside_solid_is_remove_only() {
        let hit = world_ray_first_solid(
            [5.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            10.0,
            solids(&[(5, 0, 0, 2)]),
        )
        .expect("origin sits inside a solid voxel");
        assert_eq!(hit.cell, IVec3::new(5, 0, 0));
        assert_eq!(hit.normal, IVec3::ZERO, "no exposed face when starting inside");
        assert_eq!(hit.place_cell, hit.cell);
        assert_eq!(hit.t_entry, 0.0);
        assert_eq!(hit.material, 2);
    }

    #[test]
    fn pure_miss_returns_none() {
        assert!(world_ray_first_solid([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 10.0, solids(&[])).is_none());
        // Grazes past a lone voxel without entering its cell.
        let r = world_ray_first_solid(
            [0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            10.0,
            solids(&[(5, 9, 9, 1)]),
        );
        assert!(r.is_none());
    }

    #[test]
    fn just_beyond_reach_is_none_but_at_reach_hits() {
        // Solid at x=8; from x=0.5 the entry boundary is t = 7.5 m.
        let s: &'static [(i64, i64, i64, u16)] = &[(8, 0, 0, 1)];
        assert!(
            world_ray_first_solid([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 5.0, solids(s)).is_none(),
            "7.5 m hit is beyond a 5 m reach"
        );
        let hit = world_ray_first_solid([0.5, 0.5, 0.5], [1.0, 0.0, 0.0], 8.0, solids(s))
            .expect("7.5 m hit is within an 8 m reach");
        assert_eq!(hit.cell, IVec3::new(8, 0, 0));
    }

    #[test]
    fn negative_coords_use_floor() {
        // Origin floors to cell (-1, 0, 0); march +x to a solid at (2, 0, 0).
        let hit = world_ray_first_solid(
            [-0.5, 0.5, 0.5],
            [1.0, 0.0, 0.0],
            10.0,
            solids(&[(2, 0, 0, 6)]),
        )
        .expect("ray from a negative cell should hit");
        assert_eq!(hit.cell, IVec3::new(2, 0, 0));
        assert_eq!(hit.place_cell, IVec3::new(1, 0, 0));
        assert_eq!(hit.material, 6);
    }

    #[test]
    fn degenerate_inputs_return_none() {
        assert!(world_ray_first_solid([0.0; 3], [0.0; 3], 10.0, solids(&[(0, 0, 0, 1)])).is_none());
        assert!(world_ray_first_solid([0.0; 3], [1.0, 0.0, 0.0], 0.0, solids(&[(0, 0, 0, 1)])).is_none());
    }
}
