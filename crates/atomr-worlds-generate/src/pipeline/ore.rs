//! Ore-vein strategies.
//!
//! Two CPU reference impls of [`OreVeinStrategy`]:
//!
//! * [`ThresholdNoise`] — converts any stone voxel whose fBm-gradient
//!   sample exceeds a configured threshold into an ore voxel.
//! * [`BiasedRandomWalk`] — per [`super::anchor::FeatureKind::OreVein`]
//!   anchor: walks a small per-anchor LCG-seeded sequence of voxel steps
//!   biased toward the bedding plane, converting visited stone voxels.
//!
//! Both impls are pure CPU and allocation-light. `BiasedRandomWalk` is
//! strictly non-allocating in its inner walk.

use atomr_worlds_noise::{fbm_gradient, FbmConfig};
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::terrain::{MATERIAL_GLOW_ROCK, MATERIAL_STONE};

use super::anchor::FeatureKind;
use super::strategies::OreVeinStrategy;
use super::workspace::BrickWorkspace;

/// Coarse "dim" tag mixed into LCG state so two anchors at the same column
/// don't collide with structure / worm seeders sharing the same seed root.
const ORE_DIM: u64 = 0x4F52_4500_5645_494E; // "ORE.VEIN"

/// Default ore material when not configured — reuses the existing
/// `MATERIAL_GLOW_ROCK` so renderers light up without palette work.
pub const DEFAULT_ORE_MATERIAL: u16 = MATERIAL_GLOW_ROCK;

/// Configuration for [`ThresholdNoise`].
#[derive(Copy, Clone, Debug)]
pub struct OreVeinConfig {
    pub ore_id: u16,
    pub fbm: FbmConfig,
    /// Threshold in `[-1, 1]`. fBm-gradient outputs above this become ore.
    pub threshold: f32,
}

impl Default for OreVeinConfig {
    fn default() -> Self {
        Self {
            ore_id: DEFAULT_ORE_MATERIAL,
            fbm: FbmConfig { octaves: 3, lacunarity: 2.0, gain: 0.5, frequency: 0.08 },
            threshold: 0.55,
        }
    }
}

/// Convert stone voxels to ore where `fbm_gradient > threshold`.
#[derive(Clone, Debug)]
pub struct ThresholdNoise {
    pub config: OreVeinConfig,
}

impl Default for ThresholdNoise {
    fn default() -> Self {
        Self { config: OreVeinConfig::default() }
    }
}

impl OreVeinStrategy for ThresholdNoise {
    fn id(&self) -> &'static str {
        "ThresholdNoise"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let seed = ws.ctx.world_seed ^ ORE_DIM;
        let base = ws.ctx.brick_coord;
        let edge = BRICK_EDGE as i32;
        let ore = Voxel::new(self.config.ore_id);
        for z in 0..edge {
            for y in 0..edge {
                for x in 0..edge {
                    if ws.material_at(x, y, z).0 != MATERIAL_STONE {
                        continue;
                    }
                    let wx = (base.x as i32 * edge + x) as f32;
                    let wy = (base.y as i32 * edge + y) as f32;
                    let wz = (base.z as i32 * edge + z) as f32;
                    let n = fbm_gradient(seed, wx, wy, wz, self.config.fbm);
                    if n > self.config.threshold {
                        ws.set_material(x, y, z, ore);
                    }
                }
            }
        }
    }
}

/// Bias used by the random walk: directional weights for horizontal vs
/// vertical motion. Defaults follow real-world ore-vein bedding bias.
#[derive(Copy, Clone, Debug)]
pub struct StrataBias {
    /// Probability of choosing a horizontal step (±X / ±Z). Vertical
    /// probability is `1 - horizontal`. Default 0.7.
    pub horizontal: f32,
}

impl Default for StrataBias {
    fn default() -> Self {
        Self { horizontal: 0.7 }
    }
}

/// Configuration for [`BiasedRandomWalk`].
#[derive(Copy, Clone, Debug)]
pub struct BiasedRandomWalkConfig {
    pub ore_id: u16,
    pub steps_per_anchor: u32,
    pub bias: StrataBias,
    /// Maximum displacement (in voxels) from anchor origin. The walker
    /// terminates early if it leaves this cube, keeping the vein local.
    pub walk_radius: i32,
}

impl Default for BiasedRandomWalkConfig {
    fn default() -> Self {
        Self {
            ore_id: DEFAULT_ORE_MATERIAL,
            steps_per_anchor: 48,
            bias: StrataBias::default(),
            walk_radius: 8,
        }
    }
}

/// Per-anchor walker state. Non-allocating: lives on the stack.
#[derive(Copy, Clone, Debug)]
struct Walker {
    pos: [i32; 3],
    origin: [i32; 3],
    lcg_state: u64,
    steps_remaining: u32,
}

/// Linear congruential generator step. Numerical Recipes constants.
#[inline]
fn lcg_next(state: u64) -> u64 {
    state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407)
}

#[inline]
fn lcg_f32(state: u64) -> f32 {
    // Top 24 bits → [0, 1).
    ((state >> 40) as f32) / ((1u32 << 24) as f32)
}

impl Walker {
    fn step(&mut self, bias: StrataBias) {
        self.lcg_state = lcg_next(self.lcg_state);
        let pick = lcg_f32(self.lcg_state);
        self.lcg_state = lcg_next(self.lcg_state);
        let sign = if (self.lcg_state >> 63) & 1 == 1 { 1 } else { -1 };
        if pick < bias.horizontal {
            self.lcg_state = lcg_next(self.lcg_state);
            // Within horizontal: split evenly between X and Z.
            if (self.lcg_state >> 62) & 1 == 1 {
                self.pos[0] += sign;
            } else {
                self.pos[2] += sign;
            }
        } else {
            self.pos[1] += sign;
        }
        self.steps_remaining = self.steps_remaining.saturating_sub(1);
    }
}

/// Anchor-driven biased random walk.
#[derive(Clone, Debug)]
pub struct BiasedRandomWalk {
    pub config: BiasedRandomWalkConfig,
}

impl Default for BiasedRandomWalk {
    fn default() -> Self {
        Self { config: BiasedRandomWalkConfig::default() }
    }
}

impl OreVeinStrategy for BiasedRandomWalk {
    fn id(&self) -> &'static str {
        "BiasedRandomWalk"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i32;
        let brick_origin = [
            ws.ctx.brick_coord.x as i32 * edge,
            ws.ctx.brick_coord.y as i32 * edge,
            ws.ctx.brick_coord.z as i32 * edge,
        ];
        let radius = self.config.walk_radius;
        let radius_sq = radius * radius;
        let ore = Voxel::new(self.config.ore_id);

        // Collect anchors first so we can borrow `ws` mutably in the inner
        // loop without holding an immutable borrow on `ws.anchors`.
        let anchors: Vec<_> = ws
            .anchors
            .iter()
            .filter(|a| a.kind == FeatureKind::OreVein)
            .copied()
            .collect();

        for anchor in anchors {
            // Anchor origin is in world meters; convert to world-voxels
            // (1 m per voxel at lod 0; the macro layer scales otherwise,
            // but ore veins use voxel space directly).
            let ox = anchor.origin_m[0].floor() as i32;
            let oy = anchor.origin_m[1].floor() as i32;
            let oz = anchor.origin_m[2].floor() as i32;
            let mut walker = Walker {
                pos: [ox, oy, oz],
                origin: [ox, oy, oz],
                lcg_state: anchor.seed ^ ORE_DIM,
                steps_remaining: self.config.steps_per_anchor,
            };

            while walker.steps_remaining > 0 {
                // Early termination if we leave the walk-radius cube.
                let dx = walker.pos[0] - walker.origin[0];
                let dy = walker.pos[1] - walker.origin[1];
                let dz = walker.pos[2] - walker.origin[2];
                if dx * dx + dy * dy + dz * dz > radius_sq {
                    break;
                }

                // Convert from world voxels to brick-local; skip stamps
                // that land outside this brick's body (the apron is read
                // by neighboring bricks via their own walker traces).
                let lx = walker.pos[0] - brick_origin[0];
                let ly = walker.pos[1] - brick_origin[1];
                let lz = walker.pos[2] - brick_origin[2];
                if (0..edge).contains(&lx) && (0..edge).contains(&ly) && (0..edge).contains(&lz)
                    && ws.material_at(lx, ly, lz).0 == MATERIAL_STONE
                {
                    ws.set_material(lx, ly, lz, ore);
                }

                walker.step(self.config.bias);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::FeatureAnchor;
    use atomr_worlds_core::coord::IVec3;

    fn make_ws_with_stone() -> BrickWorkspace {
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(0xA5A5, IVec3::new(0, 0, 0)));
        // Fill body with stone so ore conversions are visible.
        let stone = Voxel::new(MATERIAL_STONE);
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    ws.set_material(x, y, z, stone);
                }
            }
        }
        ws
    }

    #[test]
    fn threshold_noise_deterministic() {
        let s = ThresholdNoise::default();
        let mut a = make_ws_with_stone();
        let mut b = make_ws_with_stone();
        s.run(&mut a);
        s.run(&mut b);
        for i in 0..a.materials.len() {
            assert_eq!(a.materials[i], b.materials[i]);
        }
    }

    #[test]
    fn threshold_noise_converts_some_stone() {
        // Threshold near zero gives roughly half-coverage on the
        // gradient-noise field (output range is `[-1, 1]`).
        let s = ThresholdNoise { config: OreVeinConfig { threshold: 0.0, ..Default::default() } };
        let mut ws = make_ws_with_stone();
        s.run(&mut ws);
        let ore = Voxel::new(s.config.ore_id);
        let stone = Voxel::new(MATERIAL_STONE);
        let n_ore = ws.materials.iter().filter(|v| **v == ore).count();
        let n_stone = ws.materials.iter().filter(|v| **v == stone).count();
        assert!(n_ore > 0, "expected some ore");
        assert!(n_stone > 0, "expected some stone left");
    }

    #[test]
    fn biased_random_walk_deterministic() {
        let s = BiasedRandomWalk::default();
        let mut a = make_ws_with_stone();
        let mut b = make_ws_with_stone();
        let anchor = FeatureAnchor {
            kind: FeatureKind::OreVein,
            column: IVec3::new(0, 0, 0),
            origin_m: [8.0, 8.0, 8.0],
            seed: 0xC0FF_EE,
        };
        a.anchors.push(anchor);
        b.anchors.push(anchor);
        s.run(&mut a);
        s.run(&mut b);
        for i in 0..a.materials.len() {
            assert_eq!(a.materials[i], b.materials[i]);
        }
    }

    #[test]
    fn biased_random_walk_stays_within_radius_cube() {
        let cfg = BiasedRandomWalkConfig {
            walk_radius: 3,
            steps_per_anchor: 500,
            ..Default::default()
        };
        let s = BiasedRandomWalk { config: cfg };
        let mut ws = make_ws_with_stone();
        let anchor = FeatureAnchor {
            kind: FeatureKind::OreVein,
            column: IVec3::new(0, 0, 0),
            origin_m: [8.0, 8.0, 8.0],
            seed: 1,
        };
        ws.anchors.push(anchor);
        s.run(&mut ws);
        let ore_id = s.config.ore_id;
        let r = cfg.walk_radius;
        // Inspect every brick-local cell that ended up ore; ensure it's
        // within the radius-cube of the anchor origin (8,8,8 in this test).
        for z in 0..BRICK_EDGE as i32 {
            for y in 0..BRICK_EDGE as i32 {
                for x in 0..BRICK_EDGE as i32 {
                    if ws.material_at(x, y, z).0 != ore_id {
                        continue;
                    }
                    let dx = (x - 8).abs();
                    let dy = (y - 8).abs();
                    let dz = (z - 8).abs();
                    // Walker uses Euclidean distance for termination, but
                    // axis-aligned displacements bound that — check the
                    // looser 2*radius cube as the published invariant.
                    assert!(dx <= 2 * r && dy <= 2 * r && dz <= 2 * r);
                }
            }
        }
    }

    #[test]
    fn biased_random_walk_ignores_non_ore_anchors() {
        let s = BiasedRandomWalk::default();
        let mut ws = make_ws_with_stone();
        let baseline = ws.materials.clone();
        ws.anchors.push(FeatureAnchor {
            kind: FeatureKind::Worm,
            column: IVec3::new(0, 0, 0),
            origin_m: [8.0, 8.0, 8.0],
            seed: 1,
        });
        s.run(&mut ws);
        assert_eq!(ws.materials, baseline);
    }
}
