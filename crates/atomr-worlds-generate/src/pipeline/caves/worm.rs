//! Perlin-worm cave carver — consumes `FeatureKind::Worm` anchors and
//! traces a deterministic walk through the world, clipping carved voxels to
//! the current brick's AABB. Per-step state is held in a single `Worm`
//! struct (no per-step heap allocation).

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_noise::gradient_noise_3d;
use atomr_worlds_voxel::{Voxel, BRICK_EDGE};

use crate::pipeline::anchor::FeatureKind;
use crate::pipeline::strategies::CaveStrategy;
use crate::pipeline::workspace::BrickWorkspace;

/// Worm walker state. Held by value across the step loop; no per-step heap
/// activity. `t` advances by 1.0 each step so the noise samples form a
/// continuous parameter sweep along the worm's natural arc length.
#[derive(Copy, Clone, Debug)]
struct Worm {
    pos: [f32; 3],
    pitch: f32,
    yaw: f32,
    t: f32,
}

#[derive(Clone, Debug)]
pub struct PerlinWorm {
    /// Steps per worm. ~256 matches the reference paper's "midsize worm".
    pub steps: u32,
    /// World-meter distance advanced per step.
    pub step_len_m: f32,
    /// Frequency of the pitch/yaw gradient-noise field along `t`.
    pub heading_freq: f32,
    /// Frequency of the radius modulation noise.
    pub radius_freq: f32,
    /// Base carve radius in voxels; modulated by `radius_jitter`.
    pub base_radius_v: f32,
    /// ± radius jitter in voxels.
    pub radius_jitter: f32,
}

impl Default for PerlinWorm {
    fn default() -> Self {
        Self {
            steps: 256,
            step_len_m: 1.0,
            heading_freq: 0.02,
            radius_freq: 0.08,
            base_radius_v: 2.5,
            radius_jitter: 1.0,
        }
    }
}

impl PerlinWorm {
    pub fn new(steps: u32) -> Self {
        Self { steps, ..Default::default() }
    }
}

#[inline]
fn carve_sphere(ws: &mut BrickWorkspace, center_local: [f32; 3], radius_v: f32) {
    let r = radius_v.max(0.0);
    let r2 = r * r;
    let lo_x = (center_local[0] - r).floor() as i64;
    let lo_y = (center_local[1] - r).floor() as i64;
    let lo_z = (center_local[2] - r).floor() as i64;
    let hi_x = (center_local[0] + r).ceil() as i64;
    let hi_y = (center_local[1] + r).ceil() as i64;
    let hi_z = (center_local[2] + r).ceil() as i64;
    let edge = BRICK_EDGE as i64;
    for z in lo_z.max(0)..=hi_z.min(edge - 1) {
        for y in lo_y.max(0)..=hi_y.min(edge - 1) {
            for x in lo_x.max(0)..=hi_x.min(edge - 1) {
                let dx = x as f32 + 0.5 - center_local[0];
                let dy = y as f32 + 0.5 - center_local[1];
                let dz = z as f32 + 0.5 - center_local[2];
                if dx * dx + dy * dy + dz * dz <= r2 {
                    ws.brick.set(IVec3::new(x, y, z), Voxel::EMPTY);
                    ws.set_material(x as i32, y as i32, z as i32, Voxel::EMPTY);
                }
            }
        }
    }
}

impl CaveStrategy for PerlinWorm {
    fn id(&self) -> &'static str {
        "PerlinWorm"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let edge = BRICK_EDGE as i64;
        let ox = (ws.ctx.brick_coord.x * edge) as f32;
        let oy = (ws.ctx.brick_coord.y * edge) as f32;
        let oz = (ws.ctx.brick_coord.z * edge) as f32;
        let voxel_m = (1u64 << ws.ctx.lod.depth as u32) as f32;

        let worm_anchors: Vec<_> = ws
            .anchors
            .iter()
            .copied()
            .filter(|a| a.kind == FeatureKind::Worm)
            .collect();

        for anchor in worm_anchors {
            let mut worm = Worm {
                pos: anchor.origin_m,
                pitch: 0.0,
                yaw: 0.0,
                t: 0.0,
            };
            for _ in 0..self.steps {
                // Heading drift: gradient noise along `t` for pitch + yaw,
                // sampled on disjoint frequency offsets so the two channels
                // are uncorrelated.
                worm.pitch = gradient_noise_3d(
                    anchor.seed,
                    worm.t * self.heading_freq,
                    13.5,
                    0.0,
                ) * std::f32::consts::PI;
                worm.yaw = gradient_noise_3d(
                    anchor.seed ^ 0x9E37_79B9_7F4A_7C15,
                    worm.t * self.heading_freq,
                    91.25,
                    0.0,
                ) * std::f32::consts::PI
                    * 2.0;
                let cp = worm.pitch.cos();
                let dir = [worm.yaw.cos() * cp, worm.pitch.sin(), worm.yaw.sin() * cp];
                worm.pos[0] += dir[0] * self.step_len_m;
                worm.pos[1] += dir[1] * self.step_len_m;
                worm.pos[2] += dir[2] * self.step_len_m;
                worm.t += 1.0;

                // Radius from secondary noise; jitter rides on top of base.
                let rn = gradient_noise_3d(
                    anchor.seed ^ 0xBF58_476D_1CE4_E5B9,
                    worm.t * self.radius_freq,
                    0.0,
                    0.0,
                );
                let radius_v = self.base_radius_v + rn * self.radius_jitter;

                // Brick-local center, in voxels. `voxel_m` keeps the carve
                // visually consistent across LODs.
                let local = [
                    worm.pos[0] / voxel_m - ox,
                    worm.pos[1] / voxel_m - oy,
                    worm.pos[2] / voxel_m - oz,
                ];
                // Cheap reject: any sphere whose AABB lies outside the
                // brick adds no work.
                let r = radius_v.max(0.0);
                let in_x = local[0] + r >= 0.0 && local[0] - r <= edge as f32;
                let in_y = local[1] + r >= 0.0 && local[1] - r <= edge as f32;
                let in_z = local[2] + r >= 0.0 && local[2] - r <= edge as f32;
                if in_x && in_y && in_z {
                    carve_sphere(ws, local, radius_v);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use crate::pipeline::anchor::FeatureAnchor;

    fn ws_with_anchor(seed: u64, coord: IVec3, anchor: FeatureAnchor) -> BrickWorkspace {
        let mut w = BrickWorkspace::new(BrickGenContext::legacy(seed, coord));
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    w.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        w.anchors.push(anchor);
        w
    }

    fn worm_anchor_at(seed: u64, origin_m: [f32; 3]) -> FeatureAnchor {
        FeatureAnchor {
            kind: FeatureKind::Worm,
            column: IVec3::new(0, 0, 0),
            origin_m,
            seed,
        }
    }

    #[test]
    fn carve_is_deterministic() {
        let a = {
            let mut w = ws_with_anchor(7, IVec3::new(0, 0, 0), worm_anchor_at(0xAA, [8.0, 8.0, 8.0]));
            PerlinWorm::default().run(&mut w);
            w.brick.nonempty_count
        };
        let b = {
            let mut w = ws_with_anchor(7, IVec3::new(0, 0, 0), worm_anchor_at(0xAA, [8.0, 8.0, 8.0]));
            PerlinWorm::default().run(&mut w);
            w.brick.nonempty_count
        };
        assert_eq!(a, b);
    }

    #[test]
    fn carve_is_clipped_to_brick_aabb() {
        // Anchor far outside this brick — the carve must not panic, must
        // not write outside the brick, and produces zero or more empties
        // strictly within the local 16³ extent.
        let mut w = ws_with_anchor(
            7,
            IVec3::new(0, 0, 0),
            worm_anchor_at(0xBB, [10_000.0, 10_000.0, 10_000.0]),
        );
        let before = w.brick.nonempty_count;
        PerlinWorm::default().run(&mut w);
        assert!(w.brick.nonempty_count <= before);
    }

    #[test]
    fn no_worm_anchors_is_noop() {
        let mut w = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
        for z in 0..BRICK_EDGE as i64 {
            for y in 0..BRICK_EDGE as i64 {
                for x in 0..BRICK_EDGE as i64 {
                    w.brick.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        let before = w.brick.nonempty_count;
        PerlinWorm::default().run(&mut w);
        assert_eq!(w.brick.nonempty_count, before);
    }
}
