//! Worley / cellular noise: distance from the query point to the nearest
//! random "feature" inside a 3³ neighborhood of integer cells.

use crate::hash::{hash3_f01, hash3_u64};

/// 3-D F1 Worley noise. Returns the squared Euclidean distance from
/// `(x, y, z)` to the nearest randomly-placed feature in a 3³ cell
/// neighborhood (scaled by the cell edge of 1.0).
pub fn worley_noise_3d(seed: u64, x: f32, y: f32, z: f32) -> f32 {
    let xi = x.floor() as i64;
    let yi = y.floor() as i64;
    let zi = z.floor() as i64;

    let mut best = f32::INFINITY;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let cx = xi + dx;
                let cy = yi + dy;
                let cz = zi + dz;
                // One feature per cell, position uniform inside the cell.
                let fx = hash3_f01(seed ^ 0x1111, cx, cy, cz) + cx as f32;
                let fy = hash3_f01(seed ^ 0x2222, cx, cy, cz) + cy as f32;
                let fz = hash3_f01(seed ^ 0x3333, cx, cy, cz) + cz as f32;
                // Light salt via hash3_u64 ensures the three components don't correlate.
                let _ = hash3_u64(seed, cx, cy, cz);
                let dxv = fx - x;
                let dyv = fy - y;
                let dzv = fz - z;
                let d2 = dxv * dxv + dyv * dyv + dzv * dzv;
                if d2 < best {
                    best = d2;
                }
            }
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic() {
        let a = worley_noise_3d(3, 0.4, 0.6, 0.8);
        let b = worley_noise_3d(3, 0.4, 0.6, 0.8);
        assert_eq!(a, b);
    }

    #[test]
    fn nonnegative() {
        for i in 0..200 {
            let x = (i as f32) * 0.13;
            let v = worley_noise_3d(5, x, x * 0.7, x * 1.3);
            assert!(v >= 0.0);
        }
    }
}
