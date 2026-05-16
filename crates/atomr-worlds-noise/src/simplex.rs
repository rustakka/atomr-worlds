//! 3-D Simplex noise (Perlin 2001). Output range approximately `[-1, 1]`.

use crate::hash::hash3_u64;

const F3: f32 = 1.0 / 3.0;
const G3: f32 = 1.0 / 6.0;

// Same 12 canonical gradients used by Perlin's reference Simplex implementation.
const GRADS: [[f32; 3]; 12] = [
    [1.0, 1.0, 0.0],
    [-1.0, 1.0, 0.0],
    [1.0, -1.0, 0.0],
    [-1.0, -1.0, 0.0],
    [1.0, 0.0, 1.0],
    [-1.0, 0.0, 1.0],
    [1.0, 0.0, -1.0],
    [-1.0, 0.0, -1.0],
    [0.0, 1.0, 1.0],
    [0.0, -1.0, 1.0],
    [0.0, 1.0, -1.0],
    [0.0, -1.0, -1.0],
];

#[inline]
fn grad(seed: u64, x: i64, y: i64, z: i64) -> [f32; 3] {
    GRADS[(hash3_u64(seed, x, y, z) % 12) as usize]
}

#[inline]
fn dot(g: [f32; 3], x: f32, y: f32, z: f32) -> f32 {
    g[0] * x + g[1] * y + g[2] * z
}

/// 3-D Simplex noise. Output range is approximately `[-1, 1]`.
pub fn simplex_noise_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let s = (x + y + z) * F3;
    let i = (x + s).floor();
    let j = (y + s).floor();
    let k = (z + s).floor();

    let t = (i + j + k) * G3;
    let x0 = x - (i - t);
    let y0 = y - (j - t);
    let z0 = z - (k - t);

    // Determine which simplex (tetrahedron) of the cube contains (x0, y0, z0).
    let (i1, j1, k1, i2, j2, k2);
    if x0 >= y0 {
        if y0 >= z0 {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 1;
            k2 = 0;
        } else if x0 >= z0 {
            i1 = 1;
            j1 = 0;
            k1 = 0;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        } else {
            i1 = 0;
            j1 = 0;
            k1 = 1;
            i2 = 1;
            j2 = 0;
            k2 = 1;
        }
    } else if y0 < z0 {
        i1 = 0;
        j1 = 0;
        k1 = 1;
        i2 = 0;
        j2 = 1;
        k2 = 1;
    } else if x0 < z0 {
        i1 = 0;
        j1 = 1;
        k1 = 0;
        i2 = 0;
        j2 = 1;
        k2 = 1;
    } else {
        i1 = 0;
        j1 = 1;
        k1 = 0;
        i2 = 1;
        j2 = 1;
        k2 = 0;
    }

    let x1 = x0 - i1 as f32 + G3;
    let y1 = y0 - j1 as f32 + G3;
    let z1 = z0 - k1 as f32 + G3;
    let x2 = x0 - i2 as f32 + 2.0 * G3;
    let y2 = y0 - j2 as f32 + 2.0 * G3;
    let z2 = z0 - k2 as f32 + 2.0 * G3;
    let x3 = x0 - 1.0 + 3.0 * G3;
    let y3 = y0 - 1.0 + 3.0 * G3;
    let z3 = z0 - 1.0 + 3.0 * G3;

    let ii = i as i64;
    let jj = j as i64;
    let kk = k as i64;

    let g0 = grad(seed, ii, jj, kk);
    let g1 = grad(seed, ii + i1, jj + j1, kk + k1);
    let g2 = grad(seed, ii + i2, jj + j2, kk + k2);
    let g3 = grad(seed, ii + 1, jj + 1, kk + 1);

    let mut n0 = 0.0;
    let t0 = 0.6 - x0 * x0 - y0 * y0 - z0 * z0;
    if t0 > 0.0 {
        let t0sq = t0 * t0;
        n0 = t0sq * t0sq * dot(g0, x0, y0, z0);
    }
    let mut n1 = 0.0;
    let t1 = 0.6 - x1 * x1 - y1 * y1 - z1 * z1;
    if t1 > 0.0 {
        let t1sq = t1 * t1;
        n1 = t1sq * t1sq * dot(g1, x1, y1, z1);
    }
    let mut n2 = 0.0;
    let t2 = 0.6 - x2 * x2 - y2 * y2 - z2 * z2;
    if t2 > 0.0 {
        let t2sq = t2 * t2;
        n2 = t2sq * t2sq * dot(g2, x2, y2, z2);
    }
    let mut n3 = 0.0;
    let t3 = 0.6 - x3 * x3 - y3 * y3 - z3 * z3;
    if t3 > 0.0 {
        let t3sq = t3 * t3;
        n3 = t3sq * t3sq * dot(g3, x3, y3, z3);
    }

    // Scale factor empirically chosen to keep the sum within roughly [-1, 1].
    32.0 * (n0 + n1 + n2 + n3)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = simplex_noise_3d(1, 0.4, 0.6, 0.8);
        let b = simplex_noise_3d(1, 0.4, 0.6, 0.8);
        assert_eq!(a, b);
    }

    #[test]
    fn varies_with_seed() {
        let a = simplex_noise_3d(1, 0.4, 0.6, 0.8);
        let b = simplex_noise_3d(2, 0.4, 0.6, 0.8);
        assert!((a - b).abs() > 1e-6);
    }

    #[test]
    fn approx_unit_range_and_nonconstant() {
        let mut min = f32::INFINITY;
        let mut max = f32::NEG_INFINITY;
        for i in 0..500 {
            let t = i as f32 * 0.137;
            let v = simplex_noise_3d(7, t, t * 0.83, t * 1.27);
            assert!(v.abs() <= 1.05, "out of approximate unit range: {v}");
            if v < min {
                min = v;
            }
            if v > max {
                max = v;
            }
        }
        assert!(max - min > 0.5, "output is too constant: range {min}..{max}");
    }
}
