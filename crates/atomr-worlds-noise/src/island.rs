//! Floating-island density field: 3-D gradient noise modulated by a soft
//! radial falloff around an anchor, with a bottom-hemisphere stalactite perturbation.

use crate::gradient::gradient_noise_3d;

#[derive(Copy, Clone, Debug)]
pub struct FloatingIslandConfig {
    pub radius_m: f32,
    pub noise_frequency: f32,
    pub noise_octaves: u8,
    pub stalactite_strength: f32,
}

impl Default for FloatingIslandConfig {
    fn default() -> Self {
        Self {
            radius_m: 24.0,
            noise_frequency: 0.08,
            noise_octaves: 4,
            stalactite_strength: 0.3,
        }
    }
}

#[inline]
fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn fbm_octaves(seed: u64, x: f32, y: f32, z: f32, octaves: u8) -> f32 {
    let mut amp = 1.0_f32;
    let mut freq = 1.0_f32;
    let mut sum = 0.0_f32;
    let mut total = 0.0_f32;
    for o in 0..octaves {
        let s = seed.wrapping_add(o as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        sum += amp * gradient_noise_3d(s, x * freq, y * freq, z * freq);
        total += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    if total > 0.0 {
        sum / total
    } else {
        0.0
    }
}

/// Floating-island density at `p` relative to `anchor`. Positive values are
/// inside the island; callers threshold at zero to decide solid/empty.
pub fn island_density(
    seed: u64,
    p: [f32; 3],
    anchor: [f32; 3],
    cfg: FloatingIslandConfig,
) -> f32 {
    let dx = p[0] - anchor[0];
    let dy = p[1] - anchor[1];
    let dz = p[2] - anchor[2];
    let dist = (dx * dx + dy * dy + dz * dz).sqrt();

    let n = fbm_octaves(
        seed,
        p[0] * cfg.noise_frequency,
        p[1] * cfg.noise_frequency,
        p[2] * cfg.noise_frequency,
        cfg.noise_octaves.max(1),
    );

    let normalized = (dist / cfg.radius_m).clamp(0.0, 1.0);
    let falloff = smoothstep(1.0 - normalized);

    // Falloff is 1 at anchor and 0 at the radius edge; remap to [-1, +1] so
    // adding it to the [-1, 1] noise yields strictly positive density at the
    // anchor and strictly non-positive density beyond the radius.
    let mut density = n + (2.0 * falloff - 1.0);

    if p[1] < anchor[1] {
        let stalactite = gradient_noise_3d(seed ^ 0xDEAD, p[0] * 0.5, p[1] * 0.5, p[2] * 0.5);
        density += cfg.stalactite_strength * stalactite;
    }

    density
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let cfg = FloatingIslandConfig::default();
        let a = island_density(1, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], cfg);
        let b = island_density(1, [0.5, 0.5, 0.5], [0.0, 0.0, 0.0], cfg);
        assert_eq!(a, b);
    }

    #[test]
    fn solid_at_anchor() {
        let cfg = FloatingIslandConfig::default();
        let anchor = [0.0_f32, 0.0_f32, 0.0_f32];
        let d = island_density(7, anchor, anchor, cfg);
        assert!(d > 0.0, "expected positive density at anchor, got {d}");
    }

    #[test]
    fn empty_far_away() {
        let cfg = FloatingIslandConfig::default();
        let anchor = [0.0_f32, 0.0_f32, 0.0_f32];
        // Sample multiple far-away offsets in different directions so we don't
        // accidentally land on a single high noise sample.
        let far = 2.5 * cfg.radius_m;
        let offsets = [
            [far, 0.0, 0.0],
            [-far, 0.0, 0.0],
            [0.0, far, 0.0],
            [0.0, 0.0, far],
            [0.0, 0.0, -far],
        ];
        for o in offsets {
            let d = island_density(7, o, anchor, cfg);
            assert!(d <= 0.0, "expected non-positive density at {o:?}, got {d}");
        }
    }
}
