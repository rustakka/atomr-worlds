//! Isosurface-intersection caves: Cheese ∪ Spaghetti ∪ Noodle.
//!
//! - **Cheese** carves wherever `|simplex(p)| < ε_y`.
//! - **Spaghetti** is the intersection of two distinct simplex fields,
//!   each thresholded — narrow corridors where both fields agree.
//! - **Noodle** is a thin-threshold Spaghetti — tunnels rather than rooms.
//! `ε_y` is modulated by `(y² − 1)` (in `y_falloff_m`-normalized units) so
//! caves close near the surface and near the world bottom.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_noise::simplex_noise_3d;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::pipeline::strategies::CaveStrategy;
use crate::pipeline::workspace::BrickWorkspace;

const CHEESE_SALT: u64 = 0x00C4_EE5E_1A05_A17F;
const SPAG_SALT_A: u64 = 0x5949_9117_A1A0_A001;
const SPAG_SALT_B: u64 = 0x5949_9117_B1B0_B002;
const NOODLE_SALT_A: u64 = 0x4007_DEE0_A1A0_C001;
const NOODLE_SALT_B: u64 = 0x4007_DEE0_B1B0_C002;

#[derive(Clone, Debug)]
pub struct IsosurfaceIntersection {
    pub frequency: f32,
    pub cheese_epsilon: f32,
    pub spaghetti_epsilon: f32,
    pub noodle_epsilon: f32,
    /// `y` value at which the parabolic mask reaches zero — caves close at
    /// `|y| >= y_falloff_m`.
    pub y_falloff_m: f32,
}

impl Default for IsosurfaceIntersection {
    fn default() -> Self {
        Self {
            frequency: 1.0 / 32.0,
            cheese_epsilon: 0.06,
            spaghetti_epsilon: 0.10,
            noodle_epsilon: 0.04,
            y_falloff_m: 96.0,
        }
    }
}

impl IsosurfaceIntersection {
    /// `(y² − 1)` parabola normalized so `y = 0` carves at full strength and
    /// `|y| >= y_falloff_m` cuts off entirely. Negated so the returned scale
    /// is in `[0, 1]`.
    #[inline]
    fn parabolic_scale(&self, world_y_m: f32) -> f32 {
        let y = world_y_m / self.y_falloff_m;
        // 1 − y² clipped to [0, 1].
        (1.0 - y * y).max(0.0)
    }

    /// True if the world-meter point lies in any of Cheese / Spaghetti /
    /// Noodle.
    pub fn is_cave(&self, seed: u64, x: f32, y: f32, z: f32) -> bool {
        let s = self.parabolic_scale(y);
        if s <= 0.0 {
            return false;
        }
        let nx = x * self.frequency;
        let ny = y * self.frequency;
        let nz = z * self.frequency;

        let cheese = simplex_noise_3d(seed ^ CHEESE_SALT, nx, ny, nz).abs();
        if cheese < self.cheese_epsilon * s {
            return true;
        }
        let sa = simplex_noise_3d(seed ^ SPAG_SALT_A, nx, ny, nz).abs();
        let sb = simplex_noise_3d(seed ^ SPAG_SALT_B, nx, ny, nz).abs();
        if sa < self.spaghetti_epsilon * s && sb < self.spaghetti_epsilon * s {
            return true;
        }
        let na = simplex_noise_3d(seed ^ NOODLE_SALT_A, nx, ny, nz).abs();
        let nb = simplex_noise_3d(seed ^ NOODLE_SALT_B, nx, ny, nz).abs();
        if na < self.noodle_epsilon * s && nb < self.noodle_epsilon * s {
            return true;
        }
        false
    }
}

impl CaveStrategy for IsosurfaceIntersection {
    fn id(&self) -> &'static str {
        "IsosurfaceIntersection"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i64;
        let ox = ws.ctx.brick_coord.x * edge;
        let oy = ws.ctx.brick_coord.y * edge;
        let oz = ws.ctx.brick_coord.z * edge;
        let voxel_m = (1u64 << ws.ctx.lod.depth as u32) as f32;
        let seed = ws.ctx.world_seed;
        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let wx = (ox + lx) as f32 * voxel_m;
                    let wy = (oy + ly) as f32 * voxel_m;
                    let wz = (oz + lz) as f32 * voxel_m;
                    if self.is_cave(seed, wx, wy, wz) {
                        ws.brick.set(IVec3::new(lx, ly, lz), Voxel::EMPTY);
                        ws.set_material(lx as i32, ly as i32, lz as i32, Voxel::EMPTY);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;

    #[test]
    fn parabola_closes_at_falloff() {
        let iso = IsosurfaceIntersection::default();
        assert_eq!(iso.parabolic_scale(iso.y_falloff_m), 0.0);
        assert_eq!(iso.parabolic_scale(-iso.y_falloff_m), 0.0);
        assert!(iso.parabolic_scale(0.0) > 0.99);
        assert_eq!(iso.parabolic_scale(iso.y_falloff_m * 2.0), 0.0);
    }

    #[test]
    fn cave_field_is_deterministic() {
        let iso = IsosurfaceIntersection::default();
        assert_eq!(iso.is_cave(7, 1.5, 2.5, 3.5), iso.is_cave(7, 1.5, 2.5, 3.5));
    }

    #[test]
    fn deterministic_carve_brick() {
        let make = || {
            let mut w = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
            for z in 0..BRICK_EDGE as i64 {
                for y in 0..BRICK_EDGE as i64 {
                    for x in 0..BRICK_EDGE as i64 {
                        w.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                    }
                }
            }
            IsosurfaceIntersection::default().run(&mut w);
            w.brick.nonempty_count
        };
        assert_eq!(make(), make());
    }
}
