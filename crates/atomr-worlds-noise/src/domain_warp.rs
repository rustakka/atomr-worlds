//! Domain-warping helpers: displace input coordinates by an fBm-warped offset.
//! Provides a single warp and the three-level iterated warp `F(p) = N(p + N(p + N(p)))`.

use atomr_worlds_core::seed::splitmix64;

use crate::fbm::{fbm_gradient, FbmConfig};

#[derive(Copy, Clone, Debug)]
pub struct WarpConfig {
    pub octaves: u8,
    pub amplitude: f32,
    pub frequency: f32,
    pub lacunarity: f32,
    pub gain: f32,
}

impl Default for WarpConfig {
    fn default() -> Self {
        Self { octaves: 3, amplitude: 0.5, frequency: 1.0, lacunarity: 2.0, gain: 0.5 }
    }
}

// Per-axis salts keep the three displacement components decorrelated.
const SALT_X: u64 = 0xA5A5_A5A5_A5A5_A5A5;
const SALT_Y: u64 = 0x5A5A_5A5A_5A5A_5A5A;
const SALT_Z: u64 = 0x3C3C_3C3C_3C3C_3C3C;

#[inline]
fn axis_seed(seed: u64, salt: u64) -> u64 {
    splitmix64(seed ^ salt)
}

#[inline]
fn warp_cfg(cfg: WarpConfig) -> FbmConfig {
    FbmConfig {
        octaves: cfg.octaves,
        lacunarity: cfg.lacunarity,
        gain: cfg.gain,
        frequency: cfg.frequency,
    }
}

/// Single domain warp: `p + amplitude * (N_x(p), N_y(p), N_z(p))`.
pub fn warp_point(seed: u64, p: [f32; 3], cfg: WarpConfig) -> [f32; 3] {
    let fcfg = warp_cfg(cfg);
    let sx = axis_seed(seed, SALT_X);
    let sy = axis_seed(seed, SALT_Y);
    let sz = axis_seed(seed, SALT_Z);
    let dx = fbm_gradient(sx, p[0], p[1], p[2], fcfg);
    let dy = fbm_gradient(sy, p[0], p[1], p[2], fcfg);
    let dz = fbm_gradient(sz, p[0], p[1], p[2], fcfg);
    [p[0] + cfg.amplitude * dx, p[1] + cfg.amplitude * dy, p[2] + cfg.amplitude * dz]
}

/// Three-level iterated warp: `F(p) = warp(warp(warp(p)))`, the additive
/// composition of three displacement layers that yields the
/// `N(p + N(p + N(p)))` shape described in the literature.
pub fn iterated_warp(seed: u64, p: [f32; 3], cfg: WarpConfig) -> [f32; 3] {
    let q1 = warp_point(seed, p, cfg);
    let q2 = warp_point(seed, q1, cfg);
    warp_point(seed, q2, cfg)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warp_point_deterministic() {
        let a = warp_point(1, [0.3, 0.5, 0.7], WarpConfig::default());
        let b = warp_point(1, [0.3, 0.5, 0.7], WarpConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn iterated_warp_deterministic() {
        let a = iterated_warp(1, [0.3, 0.5, 0.7], WarpConfig::default());
        let b = iterated_warp(1, [0.3, 0.5, 0.7], WarpConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn identity_when_amplitude_zero() {
        let cfg = WarpConfig { amplitude: 0.0, ..WarpConfig::default() };
        let p = [0.3, 0.5, 0.7];
        let out = iterated_warp(123, p, cfg);
        for i in 0..3 {
            assert!((out[i] - p[i]).abs() < 1e-6, "expected identity at amp=0, got {:?}", out);
        }
    }

    #[test]
    fn warp_displaces_point() {
        let p = [0.3, 0.5, 0.7];
        let out = warp_point(1, p, WarpConfig::default());
        let d = (out[0] - p[0]).abs() + (out[1] - p[1]).abs() + (out[2] - p[2]).abs();
        assert!(d > 1e-4, "warp produced no displacement: {:?}", out);
    }
}
