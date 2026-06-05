//! CPU ray-DDA reference over a flattened [`DagGpu`] — the deterministic mirror
//! of the WGSL fragment raymarcher (`atomr-worlds-client/.../voxel_raymarch.wgsl`).
//!
//! [`gpu_get`](crate::gpu_get) mirrors the shader's *point* lookup; this mirrors
//! its *ray* traversal. Together they are the determinism gate for the GPU render
//! path: the GPU's float output is itself hash-exempt (driver-divergent, same
//! precedent as the CUDA tests), but this CPU DDA is fully deterministic, so it
//! is what the view-crate raymarch golden renders and what the client cross-checks
//! the WGSL against.
//!
//! ## Keep in lock-step with the WGSL `@fragment` DDA
//!
//! The stepping is a line-for-line port of `voxel_raymarch.wgsl`'s `@fragment`:
//! slab-intersect against `[0, edge]³`; `sign(dir)` step (0 on an axis-parallel
//! ray); a `1e-12` small-axis guard with a `1e30` boundary sentinel so a
//! zero-direction axis is inert; a 64-step cap; the entry-face normal is
//! `-step[enter_axis]`, or `-dir` when the ray began inside the hit voxel. Any
//! change here must be mirrored in the WGSL and vice versa.

use crate::brick::BRICK_EDGE;
use crate::dag::{gpu_get, DagGpu};
use crate::voxel::Voxel;

/// The first solid voxel a ray encounters while DDA-marching a brick's [`DagGpu`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit {
    /// Brick-local cell `[0, 16)` of the first solid voxel.
    pub cell: [i32; 3],
    /// Material id at `cell` (the `gpu_get(cell)` value).
    pub material: u16,
    /// Ray parameter `t` at which the ray enters `cell` — the value the GPU path
    /// transforms to a reversed-Z depth.
    pub t_entry: f32,
    /// Axis (0/1/2) the ray crossed to enter `cell`, or `-1` if it began inside
    /// the brick already in this cell (or entered through the bounding box face).
    pub enter_axis: i32,
    /// Outward face normal at the hit: `-step` on `enter_axis`, or `-dir` when the
    /// ray began inside the hit voxel (`enter_axis == -1`).
    pub normal: [f32; 3],
}

const EDGE: i32 = BRICK_EDGE as i32;

/// `sign(x)` with WGSL semantics: `0.0` maps to `0.0` (not `+1`), so an
/// axis-parallel ray contributes a zero step on that axis.
#[inline]
fn sign1(x: f32) -> f32 {
    if x > 0.0 {
        1.0
    } else if x < 0.0 {
        -1.0
    } else {
        0.0
    }
}

#[inline]
fn solid(gpu: &DagGpu, cell: [i32; 3]) -> Option<u16> {
    if cell.iter().any(|&c| !(0..EDGE).contains(&c)) {
        return None;
    }
    let v = gpu_get(gpu, cell[0] as u8, cell[1] as u8, cell[2] as u8);
    (v != Voxel::EMPTY).then_some(v.0)
}

/// March `origin + t·dir` (both in brick-local voxel space) through the brick and
/// return the first solid voxel, or `None` on a miss. Exact CPU mirror of the
/// WGSL `@fragment` DDA — see the module docs.
pub fn ray_dda_first_hit(gpu: &DagGpu, origin: [f32; 3], dir: [f32; 3]) -> Option<RayHit> {
    let edge = EDGE as f32;

    // Slab-intersect against the [0, edge]^3 box (matches the shader's inv_dir /
    // min/max form, including +/-inf on axis-parallel rays).
    let mut t_enter = f32::NEG_INFINITY;
    let mut t_exit = f32::INFINITY;
    for a in 0..3 {
        let inv = 1.0 / dir[a];
        let t0 = (0.0 - origin[a]) * inv;
        let t1 = (edge - origin[a]) * inv;
        t_enter = t_enter.max(t0.min(t1));
        t_exit = t_exit.min(t0.max(t1));
    }
    if t_enter > t_exit || t_exit < 0.0 {
        return None;
    }

    let start = t_enter.max(0.0);
    let p = [
        origin[0] + dir[0] * start,
        origin[1] + dir[1] * start,
        origin[2] + dir[2] * start,
    ];

    let mut cell = [
        (p[0].floor() as i32).clamp(0, EDGE - 1),
        (p[1].floor() as i32).clamp(0, EDGE - 1),
        (p[2].floor() as i32).clamp(0, EDGE - 1),
    ];
    let stepf = [sign1(dir[0]), sign1(dir[1]), sign1(dir[2])];
    let step = [stepf[0] as i32, stepf[1] as i32, stepf[2] as i32];
    let small = [
        dir[0].abs() < 1e-12,
        dir[1].abs() < 1e-12,
        dir[2].abs() < 1e-12,
    ];

    let mut t_max = [0.0_f32; 3];
    let mut t_delta = [0.0_f32; 3];
    for a in 0..3 {
        let inv = 1.0 / if small[a] { 1.0 } else { dir[a] };
        let next_boundary = cell[a] as f32 + stepf[a].max(0.0);
        t_max[a] = start + if small[a] { 1e30 } else { (next_boundary - p[a]) * inv };
        t_delta[a] = if small[a] { 1e30 } else { inv.abs() };
    }

    let mut t_entry = start;
    let mut enter_axis: i32 = -1;
    for _ in 0..64u32 {
        if cell.iter().any(|&c| !(0..EDGE).contains(&c)) {
            break;
        }
        if let Some(material) = solid(gpu, cell) {
            let normal = match enter_axis {
                0 => [-stepf[0], 0.0, 0.0],
                1 => [0.0, -stepf[1], 0.0],
                2 => [0.0, 0.0, -stepf[2]],
                _ => [-dir[0], -dir[1], -dir[2]],
            };
            return Some(RayHit { cell, material, t_entry, enter_axis, normal });
        }
        // Advance to the next cell across the nearest boundary.
        let axis = if t_max[0] <= t_max[1] && t_max[0] <= t_max[2] {
            0
        } else if t_max[1] <= t_max[2] {
            1
        } else {
            2
        };
        cell[axis] += step[axis];
        t_entry = t_max[axis];
        t_max[axis] += t_delta[axis];
        enter_axis = axis as i32;
    }
    None
}

#[cfg(test)]
mod tests {
    //! The voxel crate self-verifies its mirror: every hit the DDA reports must
    //! be solid per `gpu_get`, and over the canonical fixtures the DDA never
    //! skips a solid cell. The client keeps a thinner "matches what the WGSL
    //! should do" set built on this same function.

    use super::*;
    use crate::brick::Brick;
    use crate::dag::DagBrick;
    use atomr_worlds_core::coord::IVec3;

    fn norm(v: [f32; 3]) -> [f32; 3] {
        let m = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
        [v[0] / m, v[1] / m, v[2] / m]
    }

    fn uniform_brick() -> Brick {
        let mut b = Brick::new();
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    b.set(IVec3::new(x as i64, y as i64, z as i64), Voxel::new(1));
                }
            }
        }
        b
    }

    fn half_brick() -> Brick {
        let mut b = Brick::new();
        for z in 0..EDGE {
            for y in 0..(EDGE / 2) {
                for x in 0..EDGE {
                    b.set(IVec3::new(x as i64, y as i64, z as i64), Voxel::new(2));
                }
            }
        }
        b
    }

    fn sparse_brick() -> Brick {
        let mut b = Brick::new();
        b.set(IVec3::new(1, 1, 1), Voxel::new(3));
        b.set(IVec3::new(8, 9, 10), Voxel::new(4));
        b.set(IVec3::new(15, 15, 15), Voxel::new(5));
        b
    }

    /// A reported hit must be solid, and every cell strictly before it (re-walked
    /// here) must be empty — the property the WGSL relies on.
    fn assert_consistent(gpu: &DagGpu, origin: [f32; 3], dir: [f32; 3]) {
        let dir = norm(dir);
        if let Some(hit) = ray_dda_first_hit(gpu, origin, dir) {
            assert!(solid(gpu, hit.cell).is_some(), "hit {hit:?} is empty per gpu_get");
            // Re-march to just before the hit; nothing earlier may be solid.
            let mut t = hit.t_entry - 1.0;
            let mut steps = 0;
            while t < hit.t_entry && steps < 256 {
                let p = [origin[0] + dir[0] * t, origin[1] + dir[1] * t, origin[2] + dir[2] * t];
                let c = [p[0].floor() as i32, p[1].floor() as i32, p[2].floor() as i32];
                if c != hit.cell {
                    assert!(solid(gpu, c).is_none(), "solid cell {c:?} skipped before {hit:?}");
                }
                t += 0.05;
                steps += 1;
            }
        }
    }

    #[test]
    fn uniform_hits_entry_plane() {
        let gpu = DagBrick::from_brick(&uniform_brick()).to_gpu();
        let hit = ray_dda_first_hit(&gpu, [-5.0, 8.5, 8.5], norm([1.0, 0.0, 0.0]))
            .expect("ray should enter the solid brick");
        assert_eq!(hit.cell[0], 0, "first solid cell is the entry plane x=0");
        assert_eq!(hit.material, 1);
        assert_consistent(&gpu, [-4.0, -4.0, -4.0], [1.0, 1.0, 1.0]);
    }

    #[test]
    fn half_hits_top_of_block() {
        let gpu = DagBrick::from_brick(&half_brick()).to_gpu();
        let hit = ray_dda_first_hit(&gpu, [8.5, 25.0, 8.5], norm([0.0, -1.0, 0.0]))
            .expect("downward ray should hit the lower half");
        assert_eq!(hit.cell[1], (EDGE / 2) - 1, "first solid cell is the top of the lower half");
        assert_eq!(hit.material, 2);
        assert_consistent(&gpu, [-3.0, 20.0, 8.0], [1.0, -1.0, 0.2]);
        assert_consistent(&gpu, [8.0, 8.0, -3.0], [0.1, -0.3, 1.0]);
    }

    #[test]
    fn sparse_matches_oracle() {
        let b = sparse_brick();
        let gpu = DagBrick::from_brick(&b).to_gpu();
        // gpu_get must agree with the source brick for every cell (guards the
        // encoding the DDA reads).
        for z in 0..EDGE {
            for y in 0..EDGE {
                for x in 0..EDGE {
                    let expect = b.get(IVec3::new(x as i64, y as i64, z as i64));
                    let got = gpu_get(&gpu, x as u8, y as u8, z as u8);
                    assert_eq!(got, expect, "gpu_get mismatch at ({x},{y},{z})");
                }
            }
        }
        assert_consistent(&gpu, [-2.0, -2.0, -2.0], [1.0, 1.0, 1.0]);
        assert_consistent(&gpu, [20.0, 20.0, 20.0], [-1.0, -1.0, -1.0]);
    }

    #[test]
    fn miss_returns_none() {
        let gpu = DagBrick::from_brick(&sparse_brick()).to_gpu();
        // A ray that grazes past the brick entirely.
        assert!(ray_dda_first_hit(&gpu, [-5.0, -5.0, 30.0], norm([1.0, 0.0, 0.0])).is_none());
    }
}
