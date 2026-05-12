//! Gradient (Perlin-style) noise. Returns ~[-1, 1].

use crate::hash::hash3_gradient;

#[inline]
fn smoothstep(t: f32) -> f32 {
    t * t * t * (t * (t * 6.0 - 15.0) + 10.0)
}

#[inline]
fn lerp(a: f32, b: f32, t: f32) -> f32 {
    a + (b - a) * t
}

#[inline]
fn dot_grad(g: [f32; 3], dx: f32, dy: f32, dz: f32) -> f32 {
    g[0] * dx + g[1] * dy + g[2] * dz
}

/// 3-D gradient noise. Output range is approximately `[-1, 1]`.
pub fn gradient_noise_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let zi = z.floor() as i64;
    let fx = x - xi as f32;
    let fy = y - yi as f32;
    let fz = z - zi as f32;

    let g000 = hash3_gradient(seed, xi, yi, zi);
    let g100 = hash3_gradient(seed, xi + 1, yi, zi);
    let g010 = hash3_gradient(seed, xi, yi + 1, zi);
    let g110 = hash3_gradient(seed, xi + 1, yi + 1, zi);
    let g001 = hash3_gradient(seed, xi, yi, zi + 1);
    let g101 = hash3_gradient(seed, xi + 1, yi, zi + 1);
    let g011 = hash3_gradient(seed, xi, yi + 1, zi + 1);
    let g111 = hash3_gradient(seed, xi + 1, yi + 1, zi + 1);

    let n000 = dot_grad(g000, fx, fy, fz);
    let n100 = dot_grad(g100, fx - 1.0, fy, fz);
    let n010 = dot_grad(g010, fx, fy - 1.0, fz);
    let n110 = dot_grad(g110, fx - 1.0, fy - 1.0, fz);
    let n001 = dot_grad(g001, fx, fy, fz - 1.0);
    let n101 = dot_grad(g101, fx - 1.0, fy, fz - 1.0);
    let n011 = dot_grad(g011, fx, fy - 1.0, fz - 1.0);
    let n111 = dot_grad(g111, fx - 1.0, fy - 1.0, fz - 1.0);

    let u = smoothstep(fx);
    let v = smoothstep(fy);
    let w = smoothstep(fz);

    let x00 = lerp(n000, n100, u);
    let x10 = lerp(n010, n110, u);
    let x01 = lerp(n001, n101, u);
    let x11 = lerp(n011, n111, u);

    let y0 = lerp(x00, x10, v);
    let y1 = lerp(x01, x11, v);

    lerp(y0, y1, w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = gradient_noise_3d(1, 0.4, 0.6, 0.8);
        let b = gradient_noise_3d(1, 0.4, 0.6, 0.8);
        assert_eq!(a, b);
    }

    #[test]
    fn zero_at_lattice_points() {
        // Perlin noise is zero on integer lattice (gradient × zero offset).
        let v = gradient_noise_3d(7, 3.0, -2.0, 5.0);
        assert!(v.abs() < 1e-5, "expected ~0 at lattice point, got {v}");
    }

    #[test]
    fn varies_smoothly() {
        // Adjacent samples should differ; nothing pathological.
        let a = gradient_noise_3d(7, 0.1, 0.1, 0.1);
        let b = gradient_noise_3d(7, 0.2, 0.1, 0.1);
        assert_ne!(a, b);
        assert!((a - b).abs() < 0.5);
    }
}
