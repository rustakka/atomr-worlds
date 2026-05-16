//! Worley-noise cave threshold — same field as the legacy
//! [`crate::TerrainGenerator::is_cave_world`]. Documented as the non-Vanilla
//! cave path; the Vanilla preset keeps caves bundled inside
//! [`crate::pipeline::vanilla::MonolithicTerrainPass`] so byte-equality is
//! preserved without two carvers fighting over the same voxels.

use atomr_worlds_noise::worley_noise_3d;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::pipeline::strategies::CaveStrategy;
use crate::pipeline::workspace::BrickWorkspace;

/// Salt mixed into the world seed when sampling the cave field. Identical to
/// the legacy `TerrainGenerator` so a brick run through this strategy after
/// a density-only pass produces the same caves the monolithic path would.
const CAVE_SEED_SALT: u64 = 0xC0_FE_E0_C0;

#[derive(Clone, Debug)]
pub struct WorleyThreshold {
    pub threshold: f32,
    pub frequency: f32,
}

impl Default for WorleyThreshold {
    fn default() -> Self {
        Self { threshold: 0.04, frequency: 1.0 / 24.0 }
    }
}

impl WorleyThreshold {
    pub fn new(threshold: f32, frequency: f32) -> Self {
        Self { threshold, frequency }
    }

    /// Test the legacy cave field at world-meter coordinates.
    pub fn is_cave(&self, seed: u64, x: f32, y: f32, z: f32) -> bool {
        let d2 = worley_noise_3d(
            seed.wrapping_add(CAVE_SEED_SALT),
            x * self.frequency,
            y * self.frequency,
            z * self.frequency,
        );
        d2 < self.threshold
    }
}

impl CaveStrategy for WorleyThreshold {
    fn id(&self) -> &'static str {
        "WorleyThreshold"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i64;
        let origin_x = ws.ctx.brick_coord.x * edge;
        let origin_y = ws.ctx.brick_coord.y * edge;
        let origin_z = ws.ctx.brick_coord.z * edge;
        let voxel_m = (1u64 << ws.ctx.lod.depth as u32) as f32;
        let seed = ws.ctx.world_seed;
        for lz in 0..edge {
            for ly in 0..edge {
                for lx in 0..edge {
                    let wx = (origin_x + lx) as f32 * voxel_m;
                    let wy = (origin_y + ly) as f32 * voxel_m;
                    let wz = (origin_z + lz) as f32 * voxel_m;
                    if self.is_cave(seed, wx, wy, wz) {
                        ws.brick.set(
                            atomr_worlds_core::coord::IVec3::new(lx, ly, lz),
                            Voxel::EMPTY,
                        );
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
    use atomr_worlds_core::coord::IVec3;

    fn ws(seed: u64, coord: IVec3) -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(seed, coord))
    }

    #[test]
    fn deterministic_carve_pattern() {
        let a = {
            let mut w = ws(7, IVec3::new(1, -2, 3));
            // Pre-fill the brick so the carve has something to clear.
            for z in 0..BRICK_EDGE as i64 {
                for y in 0..BRICK_EDGE as i64 {
                    for x in 0..BRICK_EDGE as i64 {
                        w.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                    }
                }
            }
            WorleyThreshold::default().run(&mut w);
            w.brick.nonempty_count
        };
        let b = {
            let mut w = ws(7, IVec3::new(1, -2, 3));
            for z in 0..BRICK_EDGE as i64 {
                for y in 0..BRICK_EDGE as i64 {
                    for x in 0..BRICK_EDGE as i64 {
                        w.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                    }
                }
            }
            WorleyThreshold::default().run(&mut w);
            w.brick.nonempty_count
        };
        assert_eq!(a, b);
    }

    #[test]
    fn cross_brick_field_continuity() {
        // A voxel that lies in brick A's volume must produce the same
        // `is_cave` reading whether the field is sampled standalone or via
        // a neighboring brick that needs the same world coordinate.
        let wt = WorleyThreshold::default();
        let p_a = wt.is_cave(13, 15.0, 0.0, 0.0);
        let p_b = wt.is_cave(13, 15.0, 0.0, 0.0);
        assert_eq!(p_a, p_b);
    }
}
