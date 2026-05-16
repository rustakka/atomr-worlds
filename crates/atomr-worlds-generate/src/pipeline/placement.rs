//! Surface-point placement strategies for the layered pipeline.
//!
//! Implements four `PlacementStrategy` variants:
//! - `WhiteNoise`: uniform random sampling.
//! - `UniformGrid`: regular lattice on the brick top face.
//! - `PoissonDiskBridson`: Bridson's O(N) Poisson-disk in 2D (X/Z).
//! - `MitchellBestCandidate`: greedy best-of-N candidate selection.
//!
//! All randomness is sourced from `child_seed(world_seed, PLACEMENT_DIM,
//! brick_coord)` and chained through `splitmix64`, so identical
//! `(world_seed, brick_coord)` always produce identical point sets.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::{child_seed, splitmix64};
use atomr_worlds_voxel::BRICK_EDGE;

use super::strategies::PlacementStrategy;
use super::workspace::BrickWorkspace;

/// Sentinel `dim` arg for `child_seed` so placement seeds never collide with
/// feature/anchor/noise seeds rooted at other dims. Hex spells PLAC-EBED.
pub const PLACEMENT_DIM: u32 = 0x1AC_EBED;

/// Side length of the placement surface in meters (one brick).
const SURFACE_EDGE_M: f32 = BRICK_EDGE as f32;

#[inline]
fn brick_column(ws: &BrickWorkspace) -> IVec3 {
    ws.ctx.brick_coord
}

#[inline]
fn next_unit(rng: &mut u64) -> f32 {
    *rng = splitmix64(*rng);
    // 53-bit float mantissa: standard fast u64 -> [0,1).
    ((*rng >> 11) as f64 / (1u64 << 53) as f64) as f32
}

/// Uniform random sampling on the brick's top face.
#[derive(Debug, Clone, Copy)]
pub struct WhiteNoiseConfig {
    /// Expected points per square meter on the surface.
    pub density: f32,
}

impl Default for WhiteNoiseConfig {
    fn default() -> Self {
        Self { density: 0.05 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct WhiteNoise {
    pub config: WhiteNoiseConfig,
}

impl WhiteNoise {
    pub fn new(config: WhiteNoiseConfig) -> Self {
        Self { config }
    }

    pub fn with_density(density: f32) -> Self {
        Self::new(WhiteNoiseConfig { density })
    }
}

impl PlacementStrategy for WhiteNoise {
    fn id(&self) -> &'static str {
        "WhiteNoise"
    }

    fn place(&self, ws: &BrickWorkspace) -> Vec<[f32; 3]> {
        let mut rng = child_seed(ws.ctx.world_seed, PLACEMENT_DIM, brick_column(ws));
        let area = SURFACE_EDGE_M * SURFACE_EDGE_M;
        let count = (self.config.density.max(0.0) * area).round() as usize;
        let mut out = Vec::with_capacity(count);
        for _ in 0..count {
            let x = next_unit(&mut rng) * SURFACE_EDGE_M;
            let z = next_unit(&mut rng) * SURFACE_EDGE_M;
            out.push([x, 0.0, z]);
        }
        out
    }
}

/// Regular grid lattice on the brick top face.
#[derive(Debug, Clone, Copy)]
pub struct UniformGridConfig {
    pub spacing_m: f32,
}

impl Default for UniformGridConfig {
    fn default() -> Self {
        Self { spacing_m: 4.0 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct UniformGrid {
    pub config: UniformGridConfig,
}

impl UniformGrid {
    pub fn new(config: UniformGridConfig) -> Self {
        Self { config }
    }

    pub fn with_spacing(spacing_m: f32) -> Self {
        Self::new(UniformGridConfig { spacing_m })
    }
}

impl PlacementStrategy for UniformGrid {
    fn id(&self) -> &'static str {
        "UniformGrid"
    }

    fn place(&self, _ws: &BrickWorkspace) -> Vec<[f32; 3]> {
        let s = self.config.spacing_m.max(0.0001);
        let n = (SURFACE_EDGE_M / s).floor() as i32;
        if n <= 0 {
            return Vec::new();
        }
        // Center the lattice inside the brick face.
        let used = n as f32 * s;
        let pad = (SURFACE_EDGE_M - used) * 0.5;
        let mut out = Vec::with_capacity(((n + 1) * (n + 1)) as usize);
        for iz in 0..=n {
            for ix in 0..=n {
                let x = pad + ix as f32 * s;
                let z = pad + iz as f32 * s;
                out.push([x, 0.0, z]);
            }
        }
        out
    }
}

/// Bridson's O(N) Poisson-disk sampling, 2D variant on the X/Z plane.
#[derive(Debug, Clone, Copy)]
pub struct PoissonDiskConfig {
    pub min_distance_m: f32,
    pub k_attempts: u32,
}

impl Default for PoissonDiskConfig {
    fn default() -> Self {
        Self { min_distance_m: 4.0, k_attempts: 30 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PoissonDiskBridson {
    pub config: PoissonDiskConfig,
}

impl PoissonDiskBridson {
    pub fn new(config: PoissonDiskConfig) -> Self {
        Self { config }
    }

    pub fn with_min_distance(min_distance_m: f32) -> Self {
        Self::new(PoissonDiskConfig { min_distance_m, ..PoissonDiskConfig::default() })
    }
}

impl PlacementStrategy for PoissonDiskBridson {
    fn id(&self) -> &'static str {
        "PoissonDiskBridson"
    }

    fn place(&self, ws: &BrickWorkspace) -> Vec<[f32; 3]> {
        let r = self.config.min_distance_m.max(0.0001);
        let k = self.config.k_attempts.max(1) as usize;
        let r2 = r * r;
        // Bridson cell size: r / sqrt(d). For d=2 we get r/√2.
        let cell = r / std::f32::consts::SQRT_2;
        let grid_n = (SURFACE_EDGE_M / cell).ceil() as usize + 1;
        let mut grid: Vec<i32> = vec![-1; grid_n * grid_n];
        let mut points: Vec<[f32; 2]> = Vec::new();
        let mut active: Vec<usize> = Vec::new();
        let mut rng = child_seed(ws.ctx.world_seed, PLACEMENT_DIM, brick_column(ws));

        let cell_of = |p: [f32; 2], gn: usize, c: f32| -> (usize, usize) {
            let cx = ((p[0] / c).floor() as i32).clamp(0, gn as i32 - 1) as usize;
            let cz = ((p[1] / c).floor() as i32).clamp(0, gn as i32 - 1) as usize;
            (cx, cz)
        };

        // Initial sample uniformly inside the face.
        let p0 = [
            next_unit(&mut rng) * SURFACE_EDGE_M,
            next_unit(&mut rng) * SURFACE_EDGE_M,
        ];
        points.push(p0);
        active.push(0);
        let (cx, cz) = cell_of(p0, grid_n, cell);
        grid[cz * grid_n + cx] = 0;

        while !active.is_empty() {
            // Pick a random active index.
            rng = splitmix64(rng);
            let idx = (rng as usize) % active.len();
            let center = points[active[idx]];
            let mut placed = false;
            for _ in 0..k {
                let u1 = next_unit(&mut rng);
                let u2 = next_unit(&mut rng);
                // Sample uniformly in the annulus [r, 2r].
                let radius = r * (1.0 + u1);
                let theta = u2 * std::f32::consts::TAU;
                let cand = [
                    center[0] + radius * theta.cos(),
                    center[1] + radius * theta.sin(),
                ];
                if cand[0] < 0.0
                    || cand[0] >= SURFACE_EDGE_M
                    || cand[1] < 0.0
                    || cand[1] >= SURFACE_EDGE_M
                {
                    continue;
                }
                let (gx, gz) = cell_of(cand, grid_n, cell);
                let gx0 = gx.saturating_sub(2);
                let gz0 = gz.saturating_sub(2);
                let gx1 = (gx + 2).min(grid_n - 1);
                let gz1 = (gz + 2).min(grid_n - 1);
                let mut ok = true;
                'outer: for zz in gz0..=gz1 {
                    for xx in gx0..=gx1 {
                        let pi = grid[zz * grid_n + xx];
                        if pi >= 0 {
                            let q = points[pi as usize];
                            let dx = q[0] - cand[0];
                            let dz = q[1] - cand[1];
                            if dx * dx + dz * dz < r2 {
                                ok = false;
                                break 'outer;
                            }
                        }
                    }
                }
                if ok {
                    let new_idx = points.len();
                    points.push(cand);
                    active.push(new_idx);
                    grid[gz * grid_n + gx] = new_idx as i32;
                    placed = true;
                    break;
                }
            }
            if !placed {
                active.swap_remove(idx);
            }
        }

        points.into_iter().map(|p| [p[0], 0.0, p[1]]).collect()
    }
}

/// Mitchell's best-candidate algorithm: pick the candidate (of N) that
/// maximizes the minimum distance to the existing point set.
#[derive(Debug, Clone, Copy)]
pub struct MitchellConfig {
    pub candidates_per_point: u32,
    pub target_count: u32,
}

impl Default for MitchellConfig {
    fn default() -> Self {
        Self { candidates_per_point: 16, target_count: 32 }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct MitchellBestCandidate {
    pub config: MitchellConfig,
}

impl MitchellBestCandidate {
    pub fn new(config: MitchellConfig) -> Self {
        Self { config }
    }
}

impl PlacementStrategy for MitchellBestCandidate {
    fn id(&self) -> &'static str {
        "MitchellBestCandidate"
    }

    fn place(&self, ws: &BrickWorkspace) -> Vec<[f32; 3]> {
        let n_candidates = self.config.candidates_per_point.max(1) as usize;
        let target = self.config.target_count as usize;
        let mut rng = child_seed(ws.ctx.world_seed, PLACEMENT_DIM, brick_column(ws));
        let mut points: Vec<[f32; 2]> = Vec::with_capacity(target);
        for _ in 0..target {
            let mut best = [0.0f32, 0.0];
            let mut best_min = f32::NEG_INFINITY;
            for _ in 0..n_candidates {
                let cand = [
                    next_unit(&mut rng) * SURFACE_EDGE_M,
                    next_unit(&mut rng) * SURFACE_EDGE_M,
                ];
                let mut min_d2 = f32::INFINITY;
                for q in &points {
                    let dx = cand[0] - q[0];
                    let dz = cand[1] - q[1];
                    let d2 = dx * dx + dz * dz;
                    if d2 < min_d2 {
                        min_d2 = d2;
                    }
                }
                if points.is_empty() {
                    best = cand;
                    break;
                }
                if min_d2 > best_min {
                    best_min = min_d2;
                    best = cand;
                }
            }
            points.push(best);
        }
        points.into_iter().map(|p| [p[0], 0.0, p[1]]).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;

    fn ws(seed: u64, coord: IVec3) -> BrickWorkspace {
        BrickWorkspace::new(BrickGenContext::legacy(seed, coord))
    }

    #[test]
    fn white_noise_is_deterministic() {
        let s = WhiteNoise::with_density(0.05);
        let a = s.place(&ws(42, IVec3::new(1, 2, 3)));
        let b = s.place(&ws(42, IVec3::new(1, 2, 3)));
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn white_noise_varies_with_brick_coord() {
        let s = WhiteNoise::with_density(0.05);
        let a = s.place(&ws(42, IVec3::new(0, 0, 0)));
        let b = s.place(&ws(42, IVec3::new(1, 0, 0)));
        assert_ne!(a, b);
    }

    #[test]
    fn uniform_grid_count_is_predictable() {
        let s = UniformGrid::with_spacing(4.0);
        let pts = s.place(&ws(42, IVec3::ZERO));
        // 16m / 4m = 4 cells -> 5x5 = 25 points.
        assert_eq!(pts.len(), 25);
    }

    #[test]
    fn poisson_min_distance_held() {
        let s = PoissonDiskBridson::with_min_distance(3.0);
        let pts = s.place(&ws(0xC0FFEE, IVec3::new(2, 0, 5)));
        assert!(!pts.is_empty(), "poisson produced no points");
        let r2 = 3.0 * 3.0;
        for i in 0..pts.len() {
            for j in (i + 1)..pts.len() {
                let dx = pts[i][0] - pts[j][0];
                let dz = pts[i][2] - pts[j][2];
                let d2 = dx * dx + dz * dz;
                assert!(
                    d2 >= r2 - 1e-3,
                    "poisson disk violation: i={i} j={j} d2={d2}",
                );
            }
        }
    }

    #[test]
    fn poisson_is_deterministic() {
        let s = PoissonDiskBridson::with_min_distance(3.0);
        let a = s.place(&ws(7, IVec3::new(1, 1, 1)));
        let b = s.place(&ws(7, IVec3::new(1, 1, 1)));
        assert_eq!(a, b);
    }

    #[test]
    fn mitchell_returns_target_count() {
        let s = MitchellBestCandidate::new(MitchellConfig {
            candidates_per_point: 8,
            target_count: 17,
        });
        let pts = s.place(&ws(42, IVec3::ZERO));
        assert_eq!(pts.len(), 17);
    }
}
