//! Phase 14e — `pick_pyramid_level` heuristic sanity.
//!
//! Small viewport at "full world" extent should pick the coarsest level
//! (0); a large viewport at extreme zoom should pick the finest baked
//! level. Anything in between is fine — the test pins the two extremes,
//! not the curve.

use atomr_worlds_view::{pick_pyramid_level, OverviewCamera, OverviewProjection};

fn cam(extent: f64) -> OverviewCamera {
    OverviewCamera {
        center: [0.0, 0.0],
        extent,
        projection: OverviewProjection::Equirectangular,
        aspect: 1.0,
    }
}

#[test]
fn small_viewport_small_extent_picks_coarse_level() {
    // Full world (extent = 1.0) on a tiny viewport: no benefit from a
    // fine pyramid level; expect level 0.
    let l = pick_pyramid_level(&cam(1.0), [32, 32], 5);
    assert_eq!(l, 0, "tiny viewport at full extent should select level 0, got {l}");
}

#[test]
fn large_viewport_extreme_zoom_picks_fine_level() {
    // Zoomed deep in (extent = 1/64), large viewport: expect the
    // finest baked level (here, levels - 1).
    let l = pick_pyramid_level(&cam(1.0 / 64.0), [2048, 2048], 5);
    assert_eq!(l, 4, "zoomed-in large viewport should select level 4, got {l}");
}

#[test]
fn level_never_exceeds_pyramid_depth() {
    // Even with an insane zoom and a small pyramid, we must not return
    // a level past the baked depth.
    let l = pick_pyramid_level(&cam(1e-9), [16384, 16384], 3);
    assert!(l < 3, "level {l} exceeds pyramid depth 3");
}

#[test]
fn one_level_pyramid_always_returns_zero() {
    let l_a = pick_pyramid_level(&cam(1.0), [4, 4], 1);
    let l_b = pick_pyramid_level(&cam(1e-9), [4096, 4096], 1);
    assert_eq!(l_a, 0);
    assert_eq!(l_b, 0);
}
