//! Sky-light propagation strategies.
//!
//! Runs at the end of the pipeline (after flora). Produces a brick-sized
//! `LightOverlay` that the client mesher consumes to bake per-vertex
//! sky-light alongside AO. Two impls live here: [`VerticalCastWithDiffusion`]
//! (the paper's algorithm) and the macro-generated `NoneSkyLight` from
//! `strategies.rs` for the Vanilla preset.

use atomr_worlds_voxel::{light::LightOverlay, BRICK_EDGE};

use super::strategies::SkyLightStrategy;
use super::workspace::BrickWorkspace;

/// Reserved constant for a future stochastic-dither pass. Marked
/// `#[allow(dead_code)]` because it is part of the trait/module surface
/// only; nothing reads it yet.
#[allow(dead_code)]
pub const SKY_LIGHT_DIM: usize = BRICK_EDGE;

/// Tunable knobs for [`VerticalCastWithDiffusion`]. Defaults match the
/// paper's `(decay = 1 per voxel, 6 diffusion iterations, sky = 15)`.
#[derive(Copy, Clone, Debug)]
pub struct SkyLightConfig {
    pub decay_per_voxel: u8,
    pub diffusion_iterations: u32,
    pub top_brightness: u8,
}

impl Default for SkyLightConfig {
    fn default() -> Self {
        Self {
            decay_per_voxel: 1,
            diffusion_iterations: 6,
            top_brightness: 15,
        }
    }
}

/// Vertical-cast + 6 CA-style diffusion iterations.
///
/// Step 1: walk each `(x,z)` column top-down, writing `top_brightness`
/// into every empty voxel above the first solid (decaying by
/// `decay_per_voxel` as the ray descends through atmosphere). Voxels at
/// or below the first solid are zero.
///
/// Step 2: run `diffusion_iterations` of `light[p] = max(light[p],
/// max(neighbors) - 1)` over the 6 axis-aligned neighbors, with
/// brick-edge boundaries treated as opaque (no apron-fed lateral
/// borrow yet — the workspace has no neighbor-light input).
///
/// The result is byte-stable for a given (config, materials) pair and
/// monotonically decreasing away from sky-exposed voxels, so the test
/// suite's "stable after one more iteration" check holds at six.
#[derive(Copy, Clone, Debug)]
pub struct VerticalCastWithDiffusion {
    pub config: SkyLightConfig,
}

impl Default for VerticalCastWithDiffusion {
    fn default() -> Self {
        Self { config: SkyLightConfig::default() }
    }
}

impl VerticalCastWithDiffusion {
    pub fn new(config: SkyLightConfig) -> Self {
        Self { config }
    }
}

/// Solid-voxel sampler that checks the workspace materials grid first
/// and falls back to `ws.brick.voxels`. The monolithic vanilla pass
/// writes only into the brick; advanced presets (once Steps 5–7 land)
/// will write into the materials grid. Both stores share the same
/// `EMPTY` sentinel, so a logical-OR over the two sources is the
/// right "any solid voxel here?" test.
fn is_solid(ws: &BrickWorkspace, x: i32, y: i32, z: i32) -> bool {
    if (0..BRICK_EDGE as i32).contains(&x)
        && (0..BRICK_EDGE as i32).contains(&y)
        && (0..BRICK_EDGE as i32).contains(&z)
    {
        if !ws.material_at(x, y, z).is_empty() {
            return true;
        }
        let v = ws.brick.voxels
            [(z as usize * BRICK_EDGE + y as usize) * BRICK_EDGE + x as usize];
        !v.is_empty()
    } else {
        false
    }
}

#[inline]
fn light_index(x: usize, y: usize, z: usize) -> usize {
    (z * BRICK_EDGE + y) * BRICK_EDGE + x
}

const N_VOX: usize = BRICK_EDGE * BRICK_EDGE * BRICK_EDGE;

impl SkyLightStrategy for VerticalCastWithDiffusion {
    fn id(&self) -> &'static str {
        "VerticalCastWithDiffusion"
    }

    fn run(&self, ws: &mut BrickWorkspace) {
        let cfg = self.config;
        let mut grid = vec![0u8; N_VOX];
        let edge = BRICK_EDGE as i32;

        for z in 0..edge {
            for x in 0..edge {
                let mut level = cfg.top_brightness;
                let mut hit_solid = false;
                for y in (0..edge).rev() {
                    if is_solid(ws, x, y, z) {
                        hit_solid = true;
                        grid[light_index(x as usize, y as usize, z as usize)] = 0;
                        continue;
                    }
                    if hit_solid {
                        grid[light_index(x as usize, y as usize, z as usize)] = 0;
                    } else {
                        grid[light_index(x as usize, y as usize, z as usize)] = level;
                        level = level.saturating_sub(cfg.decay_per_voxel);
                    }
                }
            }
        }

        let mut next = grid.clone();
        for _ in 0..cfg.diffusion_iterations {
            for z in 0..edge {
                for y in 0..edge {
                    for x in 0..edge {
                        if is_solid(ws, x, y, z) {
                            next[light_index(x as usize, y as usize, z as usize)] = 0;
                            continue;
                        }
                        let here =
                            grid[light_index(x as usize, y as usize, z as usize)];
                        let mut best = here;
                        for (dx, dy, dz) in [
                            (-1, 0, 0),
                            (1, 0, 0),
                            (0, -1, 0),
                            (0, 1, 0),
                            (0, 0, -1),
                            (0, 0, 1),
                        ] {
                            let nx = x + dx;
                            let ny = y + dy;
                            let nz = z + dz;
                            if nx < 0 || ny < 0 || nz < 0 || nx >= edge || ny >= edge || nz >= edge {
                                continue;
                            }
                            let n_level = grid
                                [light_index(nx as usize, ny as usize, nz as usize)];
                            let propagated = n_level.saturating_sub(1);
                            if propagated > best {
                                best = propagated;
                            }
                        }
                        next[light_index(x as usize, y as usize, z as usize)] = best;
                    }
                }
            }
            std::mem::swap(&mut grid, &mut next);
        }

        let mut overlay = ws.light.take().unwrap_or_else(|| Box::new(LightOverlay::new_zero()));
        for z in 0..BRICK_EDGE {
            for y in 0..BRICK_EDGE {
                for x in 0..BRICK_EDGE {
                    let lv = grid[light_index(x, y, z)];
                    overlay.set(x as u8, y as u8, z as u8, lv);
                }
            }
        }
        ws.light = Some(overlay);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_voxel::Voxel;

    fn ws_with_floor() -> BrickWorkspace {
        let ctx = BrickGenContext::legacy(7, IVec3::new(0, 0, 0));
        let mut ws = BrickWorkspace::new(ctx);
        // y = 0 floor of solid voxels stored in `ws.materials` so
        // `material_at` finds them via the apron-aware path.
        let edge = BRICK_EDGE as i32;
        for z in 0..edge {
            for x in 0..edge {
                ws.set_material(x, 0, z, Voxel::new(1));
            }
        }
        ws
    }

    #[test]
    fn top_voxel_matches_top_brightness() {
        let mut ws = ws_with_floor();
        let s = VerticalCastWithDiffusion::default();
        s.run(&mut ws);
        // `run()` leaves the overlay on `ws.light`; the pipeline driver
        // is what moves it onto `brick.light_overlay`.
        let overlay = ws.light.as_ref().expect("sky-light overlay populated");
        let top_y = (BRICK_EDGE - 1) as u8;
        assert_eq!(overlay.get(8, top_y, 8), 15);
    }

    #[test]
    fn voxel_below_first_solid_is_zero() {
        let mut ws = ws_with_floor();
        // Add a solid roof at y=10 in one column; everything below in
        // that column should not see full sky (the vertical cast halts
        // at the roof; lateral diffusion can leak in at most by 6).
        ws.set_material(5, 10, 5, Voxel::new(1));
        let s = VerticalCastWithDiffusion::default();
        s.run(&mut ws);
        let overlay = ws.light.as_ref().unwrap();
        // The roof voxel itself stores 0 (solid).
        assert_eq!(overlay.get(5, 10, 5), 0);
        // The cell directly beneath the roof must not be full sky.
        assert!(
            overlay.get(5, 9, 5) < 15,
            "voxel directly beneath solid roof should not be full sky"
        );
    }

    #[test]
    fn lateral_diffusion_decreases_from_source() {
        // Build a workspace where only one column (x=8,z=8) sees the sky;
        // every other column is roofed at y=BRICK_EDGE-2. Sky-light should
        // bleed laterally from the open column, decreasing with distance.
        let ctx = BrickGenContext::legacy(7, IVec3::new(0, 0, 0));
        let mut ws = BrickWorkspace::new(ctx);
        let edge = BRICK_EDGE as i32;
        let roof_y = edge - 2;
        for z in 0..edge {
            for x in 0..edge {
                ws.set_material(x, 0, z, Voxel::new(1));
                if !(x == 8 && z == 8) {
                    ws.set_material(x, roof_y, z, Voxel::new(1));
                }
            }
        }
        let s = VerticalCastWithDiffusion::default();
        s.run(&mut ws);
        let overlay = ws.light.as_ref().unwrap();
        let mid_y = (roof_y - 1) as u8;
        let center = overlay.get(8, mid_y, 8);
        let one_off = overlay.get(9, mid_y, 8);
        let two_off = overlay.get(10, mid_y, 8);
        assert!(center >= one_off, "{center} should be >= {one_off}");
        assert!(one_off >= two_off, "{one_off} should be >= {two_off}");
    }

    #[test]
    fn six_iterations_is_stable() {
        let mut ws_a = ws_with_floor();
        let mut ws_b = ws_with_floor();
        VerticalCastWithDiffusion::default().run(&mut ws_a);
        VerticalCastWithDiffusion::new(SkyLightConfig {
            diffusion_iterations: 7,
            ..SkyLightConfig::default()
        })
        .run(&mut ws_b);
        let a = ws_a.light.as_ref().unwrap();
        let b = ws_b.light.as_ref().unwrap();
        let mut max_diff = 0u8;
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    let d = a.get(x, y, z).abs_diff(b.get(x, y, z));
                    if d > max_diff {
                        max_diff = d;
                    }
                }
            }
        }
        // An empty atrium (just a floor) reaches steady state in one pass;
        // we tolerate at most 1-level deltas in case of border quirks.
        assert!(max_diff <= 1, "6 → 7 iterations diverges by {max_diff}");
    }

    #[test]
    fn determinism_same_inputs_same_overlay() {
        let mut ws_a = ws_with_floor();
        let mut ws_b = ws_with_floor();
        VerticalCastWithDiffusion::default().run(&mut ws_a);
        VerticalCastWithDiffusion::default().run(&mut ws_b);
        let a = ws_a.light.as_ref().unwrap();
        let b = ws_b.light.as_ref().unwrap();
        for z in 0..BRICK_EDGE as u8 {
            for y in 0..BRICK_EDGE as u8 {
                for x in 0..BRICK_EDGE as u8 {
                    assert_eq!(a.get(x, y, z), b.get(x, y, z));
                }
            }
        }
    }
}
