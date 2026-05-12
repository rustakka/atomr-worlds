//! Phase 14b: `ChaseCamera`'s closed-form first-order smoothing converges
//! exactly to the steady-state for a fixed target anchor.
//!
//! Per-step we apply `s' = s + (target - s) * k`, `k = 1 - exp(-2π * f_c *
//! dt)`. After `N` identical steps with a stationary target:
//!
//! ```text
//!   s_N - target = (1 - k)^N * (s_0 - target)
//!                = exp(-2π * f_c * dt * N) * (s_0 - target)
//! ```
//!
//! So after 10 s at `f_c = 4 Hz` the residual scale is `exp(-80π) ≈
//! 1.4e-109` — well below f64's smallest positive normal. The smoothed
//! anchor must therefore equal the target to within 1 ULP.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::ChaseCamera;

#[test]
fn fixed_anchor_converges_to_target_within_1_ulp() {
    let start = DVec3::new(0.0, 0.0, 0.0);
    let target = DVec3::new(100.0, 50.0, -25.0);
    let mut chase = ChaseCamera::new(start, 1.0);
    // Default smoothing_hz is 4 Hz; lock it down explicitly.
    chase.smoothing_hz = 4.0;
    // Snap the "anchor" without smoothing (`tick` resets `anchor` to the
    // arg every call) — but we want `smoothed_anchor` to start at zero.
    chase.anchor = start;
    chase.smoothed_anchor = start;

    let dt = 1.0_f32 / 60.0;
    let steps = 600; // 10 s
    for _ in 0..steps {
        chase.tick(target, 0.0, 0.0, dt);
    }
    // After 10 s at 4 Hz cutoff the residual is exp(-80π) ≈ 1e-109. The
    // smoothed anchor must therefore equal `target` to within 1 ULP. We
    // compare via `to_bits` for the strict ULP test.
    let bits_eq = |a: f64, b: f64| (a.to_bits() as i64 - b.to_bits() as i64).abs() <= 1;
    assert!(
        bits_eq(chase.smoothed_anchor.x, target.x),
        "x diff: {} vs {} (bits {} vs {})",
        chase.smoothed_anchor.x,
        target.x,
        chase.smoothed_anchor.x.to_bits(),
        target.x.to_bits(),
    );
    assert!(
        bits_eq(chase.smoothed_anchor.y, target.y),
        "y diff: {} vs {}",
        chase.smoothed_anchor.y,
        target.y,
    );
    assert!(
        bits_eq(chase.smoothed_anchor.z, target.z),
        "z diff: {} vs {}",
        chase.smoothed_anchor.z,
        target.z,
    );
}

#[test]
fn smoothing_is_monotonic_for_moving_anchor() {
    // With a target moving along +Z and smoothing_hz finite, the smoothed
    // copy must trail behind — `smoothed.z < anchor.z` at all times after
    // step 0. Catches sign-flip / overshoot regressions.
    let mut chase = ChaseCamera::new(DVec3::ZERO, 1.0);
    chase.smoothing_hz = 4.0;
    let dt = 1.0_f32 / 60.0;
    for n in 1..=120 {
        let anchor = DVec3::new(0.0, 0.0, n as f64);
        chase.tick(anchor, 0.0, 0.0, dt);
        assert!(
            chase.smoothed_anchor.z < anchor.z,
            "step {n}: smoothed.z={} >= anchor.z={}",
            chase.smoothed_anchor.z,
            anchor.z,
        );
        assert!(chase.smoothed_anchor.z > 0.0, "step {n}: smoothed.z should have moved off zero",);
    }
}
