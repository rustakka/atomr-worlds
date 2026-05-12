//! Value noise (trilinear-interpolated random lattice).

use crate::hash::hash3_f01;

#[inline]
fn smoothstep(t: f32) -> f32 {
    // 6t^5 − 15t^4 + 10t^3 — smoother than 3t^2 − 2t^3, derivative continuous.
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

/// 3-D value noise. Returns a value in `[0, 1]`.
pub fn value_noise_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let zi = z.floor() as i64;
    let fx = x - xi as f32;
    let fy = y - yi as f32;
    let fz = z - zi as f32;

    let c000 = hash3_f01(seed, xi, yi, zi);
    let c100 = hash3_f01(seed, xi + 1, yi, zi);
    let c010 = hash3_f01(seed, xi, yi + 1, zi);
    let c110 = hash3_f01(seed, xi + 1, yi + 1, zi);
    let c001 = hash3_f01(seed, xi, yi, zi + 1);
    let c101 = hash3_f01(seed, xi + 1, yi, zi + 1);
    let c011 = hash3_f01(seed, xi, yi + 1, zi + 1);
    let c111 = hash3_f01(seed, xi + 1, yi + 1, zi + 1);

    let u = smoothstep(fx);
    let v = smoothstep(fy);
    let w = smoothstep(fz);

    let x00 = lerp(c000, c100, u);
    let x10 = lerp(c010, c110, u);
    let x01 = lerp(c001, c101, u);
    let x11 = lerp(c011, c111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = value_noise_3d(42, 1.25, 0.75, -0.5);
        let b = value_noise_3d(42, 1.25, 0.75, -0.5);
        assert_eq!(a, b);
    }

    #[test]
    fn within_unit_range() {
        for i in 0..1000 {
            let x = (i as f32) * 0.137;
            let v = value_noise_3d(1, x, x * 1.3, x * 0.7);
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn varies_with_seed() {
        let a = value_noise_3d(1, 0.5, 0.5, 0.5);
        let b = value_noise_3d(2, 0.5, 0.5, 0.5);
        assert!((a - b).abs() > 1e-6);
    }
}
