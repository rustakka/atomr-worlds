//! Fractal Brownian motion combinator: sum N octaves of a base noise,
//! halving amplitude and doubling frequency each step.

use crate::gradient::gradient_noise_3d;
use crate::value::value_noise_3d;

#[derive(Copy, Clone, Debug)]
pub struct FbmConfig {
    pub octaves: u8,
    pub lacunarity: f32, // frequency multiplier per octave (typical 2.0)
    pub gain: f32,       // amplitude multiplier per octave (typical 0.5)
    pub frequency: f32,  // initial frequency multiplier
}

impl Default for FbmConfig {
    fn default() -> Self {
        Self { octaves: 4, lacunarity: 2.0, gain: 0.5, frequency: 1.0 }
    }
}

/// FBM over value noise. Output range approximately `[0, sum_of_amplitudes]`,
/// already normalised to `[0, 1]` by the inverse total amplitude.
pub fn fbm_value(seed: u64, x: f32, y: f32, z: f32, cfg: FbmConfig) -> f32 {
    let mut amp = 1.0_f32;
    let mut freq = cfg.frequency;
    let mut sum = 0.0_f32;
    let mut total = 0.0_f32;
    for o in 0..cfg.octaves {
        let octave_seed = seed.wrapping_add(o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        sum += amp * value_noise_3d(octave_seed, x * freq, y * freq, z * freq);
        total += amp;
        amp *= cfg.gain;
        freq *= cfg.lacunarity;
    }
    if total > 0.0 {
        sum / total
    } else {
        0.0
    }
}

/// FBM over gradient noise. Output range approximately `[-1, 1]`.
pub fn fbm_gradient(seed: u64, x: f32, y: f32, z: f32, cfg: FbmConfig) -> f32 {
    let mut amp = 1.0_f32;
    let mut freq = cfg.frequency;
    let mut sum = 0.0_f32;
    let mut total = 0.0_f32;
    for o in 0..cfg.octaves {
        let octave_seed = seed.wrapping_add(o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        sum += amp * gradient_noise_3d(octave_seed, x * freq, y * freq, z * freq);
        total += amp;
        amp *= cfg.gain;
        freq *= cfg.lacunarity;
    }
    if total > 0.0 {
        sum / total
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fbm_value_deterministic() {
        let a = fbm_value(11, 0.3, 0.4, 0.5, FbmConfig::default());
        let b = fbm_value(11, 0.3, 0.4, 0.5, FbmConfig::default());
        assert_eq!(a, b);
    }

    #[test]
    fn fbm_value_in_unit_range() {
        let cfg = FbmConfig::default();
        for i in 0..500 {
            let x = (i as f32) * 0.21;
            let v = fbm_value(13, x, x * 0.9, x * 1.1, cfg);
            assert!((0.0..=1.0).contains(&v), "out of range: {v}");
        }
    }
}
