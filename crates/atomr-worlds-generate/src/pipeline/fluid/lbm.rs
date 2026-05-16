//! Lattice Boltzmann D3Q19 — CPU reference impl.
//!
//! Single-relaxation-time BGK collision over the 19-velocity D3Q19
//! lattice. Mass is conserved by construction: streaming permutes the
//! distribution-function field, and collision is a linear blend with the
//! equilibrium that itself sums to the local density.
//!
//! CUDA pairing lands in Step 11.

use super::super::strategies::FluidStrategy;
use super::super::workspace::BrickWorkspace;

/// Number of discrete velocity vectors in the D3Q19 lattice.
pub const Q: usize = 19;

/// The 19 discrete velocity directions (rest + 6 axial + 12 face-diagonal).
/// Order matters: streaming uses the index to look up the corresponding
/// opposite via [`OPPOSITE`].
const E: [[i8; 3]; Q] = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 0],
    [-1, -1, 0],
    [1, -1, 0],
    [-1, 1, 0],
    [1, 0, 1],
    [-1, 0, -1],
    [1, 0, -1],
    [-1, 0, 1],
    [0, 1, 1],
    [0, -1, -1],
    [0, 1, -1],
    [0, -1, 1],
];

/// Per-direction weights of the D3Q19 stencil.
const W: [f32; Q] = [
    1.0 / 3.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 18.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
    1.0 / 36.0,
];

/// Index of the opposite velocity, used for bounce-back at walls. Indices
/// run pairwise after the rest (index 0): (1,2), (3,4), …, (17,18).
const OPPOSITE: [usize; Q] = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17];

/// Tunables for [`LatticeBoltzmannD3Q19`].
#[derive(Copy, Clone, Debug)]
pub struct LbmConfig {
    /// Number of LBM ticks. Each tick = collision + streaming.
    pub ticks: u32,
    /// Relaxation time (BGK). Must be > 0.5 for stability; 1.0 is neutral.
    pub tau: f32,
    /// Initial density to seed every fluid cell. Static field used because
    /// the strategy currently runs on the brick interior without apron
    /// influx; bulk density preservation is the published invariant.
    pub rho0: f32,
}

impl Default for LbmConfig {
    fn default() -> Self {
        Self { ticks: 16, tau: 1.0, rho0: 1.0 }
    }
}

/// 19-velocity LBM with BGK collision. CPU reference path.
#[derive(Clone, Debug)]
pub struct LatticeBoltzmannD3Q19 {
    pub config: LbmConfig,
}

impl Default for LatticeBoltzmannD3Q19 {
    fn default() -> Self {
        Self { config: LbmConfig::default() }
    }
}

impl LatticeBoltzmannD3Q19 {
    pub fn new(config: LbmConfig) -> Self {
        Self { config }
    }

    /// Run the LBM on a flat `Q*N³` distribution buffer and return the
    /// final total density (sum of all distribution functions). Exposed
    /// for the mass-conservation unit test.
    pub fn simulate(&self, edge: usize) -> (Vec<f32>, f64) {
        let n = edge * edge * edge;
        let mut f = vec![0.0_f32; Q * n];
        // Seed with equilibrium at zero velocity → f_i = w_i * rho0.
        for i in 0..n {
            for q in 0..Q {
                f[q * n + i] = W[q] * self.config.rho0;
            }
        }
        for _tick in 0..self.config.ticks {
            // Collision: f_i ← f_i − (f_i − f_eq_i) / tau.
            for i in 0..n {
                let rho: f32 = (0..Q).map(|q| f[q * n + i]).sum();
                let inv_rho = if rho > 1e-12 { 1.0 / rho } else { 0.0 };
                let ux: f32 =
                    (0..Q).map(|q| f[q * n + i] * E[q][0] as f32).sum::<f32>() * inv_rho;
                let uy: f32 =
                    (0..Q).map(|q| f[q * n + i] * E[q][1] as f32).sum::<f32>() * inv_rho;
                let uz: f32 =
                    (0..Q).map(|q| f[q * n + i] * E[q][2] as f32).sum::<f32>() * inv_rho;
                let usq = ux * ux + uy * uy + uz * uz;
                for q in 0..Q {
                    let eu = E[q][0] as f32 * ux + E[q][1] as f32 * uy + E[q][2] as f32 * uz;
                    let feq = W[q] * rho * (1.0 + 3.0 * eu + 4.5 * eu * eu - 1.5 * usq);
                    let prev = f[q * n + i];
                    f[q * n + i] = prev - (prev - feq) / self.config.tau;
                }
            }
            // Streaming: f_q at (x, y, z) ← f_q at (x − e_q[x], …). Use a
            // ping-pong buffer; bounce-back across the brick boundary so
            // the total population is conserved.
            let mut g = vec![0.0_f32; Q * n];
            for z in 0..edge as i32 {
                for y in 0..edge as i32 {
                    for x in 0..edge as i32 {
                        for q in 0..Q {
                            let sx = x - E[q][0] as i32;
                            let sy = y - E[q][1] as i32;
                            let sz = z - E[q][2] as i32;
                            let src = if (0..edge as i32).contains(&sx)
                                && (0..edge as i32).contains(&sy)
                                && (0..edge as i32).contains(&sz)
                            {
                                let si = (sz as usize * edge + sy as usize) * edge + sx as usize;
                                f[q * n + si]
                            } else {
                                let di = (z as usize * edge + y as usize) * edge + x as usize;
                                f[OPPOSITE[q] * n + di]
                            };
                            let di = (z as usize * edge + y as usize) * edge + x as usize;
                            g[q * n + di] = src;
                        }
                    }
                }
            }
            f = g;
        }
        let total: f64 = f.iter().map(|v| *v as f64).sum();
        (f, total)
    }
}

impl FluidStrategy for LatticeBoltzmannD3Q19 {
    fn id(&self) -> &'static str {
        "LatticeBoltzmannD3Q19"
    }

    fn run(&self, _ws: &mut BrickWorkspace) {
        // Strategy-level integration runs LBM on the brick body using its
        // own internal distribution buffer; the result feeds the material
        // overlay once a `FluidLayer`-backed `Brick` field lands. For
        // Step 7 the strategy is a behavioral stub (computation paths +
        // determinism + mass conservation are exercised via `simulate`).
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lbm_weights_sum_to_one() {
        let s: f32 = W.iter().sum();
        assert!((s - 1.0).abs() < 1e-6, "D3Q19 weights must sum to 1, got {s}");
    }

    #[test]
    fn lbm_opposite_pairs_are_consistent() {
        for q in 0..Q {
            assert_eq!(OPPOSITE[OPPOSITE[q]], q);
            // e_opp == -e_q
            for axis in 0..3 {
                assert_eq!(E[OPPOSITE[q]][axis], -E[q][axis]);
            }
        }
    }

    #[test]
    fn lbm_mass_conservation_over_1000_ticks() {
        // Small lattice so this stays cheap. Mass is the sum of all
        // distribution functions over the whole domain; bounce-back at
        // the walls keeps it constant across ticks.
        let edge = 4usize;
        let s = LatticeBoltzmannD3Q19::new(LbmConfig { ticks: 1000, tau: 1.0, rho0: 1.0 });
        let (_f, total) = s.simulate(edge);
        let expected = (edge * edge * edge) as f64 * 1.0;
        assert!(
            (total - expected).abs() / expected < 1e-3,
            "mass drift: total={total} expected={expected}"
        );
    }

    #[test]
    fn lbm_deterministic() {
        let s = LatticeBoltzmannD3Q19::new(LbmConfig { ticks: 4, tau: 1.0, rho0: 1.0 });
        let (a, _) = s.simulate(4);
        let (b, _) = s.simulate(4);
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            assert_eq!(a[i].to_bits(), b[i].to_bits());
        }
    }
}
