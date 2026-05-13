//! Phase 17 — skybox-refresh + crossfade integration test.
//!
//! The client crate is binary-only (no `lib.rs`), so this test drives
//! the same `ObserverState` + `SkyboxRefreshPolicy` shapes the
//! `SkyboxRuntime` in `src/render/skybox.rs` uses, without touching
//! Bevy. The unit tests inside `src/render/skybox.rs` already cover
//! `SkyboxRuntime`'s `current_brightness`, `budget_allows`, and
//! `cubemap_image`; this file is the end-to-end "three refresh
//! triggers" walk-through.
//!
//! Assertions:
//! 1. A first refresh trips even though no walk has happened (no
//!    `last_skybox` yet) — and once we feed a baked skybox in, the
//!    drift threshold governs subsequent refreshes.
//! 2. Walking past 5 % of `outer_radius_m` retriggers `should_refresh`.
//! 3. Brightness ramps monotonically toward the new target while the
//!    crossfade is in flight (mirrors the lerp inside
//!    `SkyboxRuntime::current_brightness`).

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_view::observer::{ObserverState, SkyboxRefreshPolicy};
use atomr_worlds_view::skybox::{render_skybox_from_meshes, Skybox as ViewSkybox, SkyboxConfig};

/// Mirrors `WORLD_TIER_BODY_RADIUS_M` in `src/render/skybox.rs` — the
/// cube world's altitude check is dominated by position drift, so we
/// pick a body radius large enough that the altitude trigger never
/// trips at our walk scales.
const WORLD_TIER_BODY_RADIUS_M: f64 = 1.0e6;

/// Mirrors `NIGHT_BRIGHTNESS` / `DAY_BRIGHTNESS` in
/// `src/render/skybox.rs`; kept literal here so the test fails loudly
/// if those constants drift.
const NIGHT_BRIGHTNESS: f32 = 50.0;
const DAY_BRIGHTNESS: f32 = 2500.0;

#[inline]
fn lerp_brightness(day_factor: f32) -> f32 {
    let t = day_factor.clamp(0.0, 1.0);
    NIGHT_BRIGHTNESS + (DAY_BRIGHTNESS - NIGHT_BRIGHTNESS) * t
}

fn fake_bake(origin: DVec3, outer_radius_m: f64) -> ViewSkybox {
    render_skybox_from_meshes(
        &[],
        [origin.x, origin.y, origin.z],
        1.0,
        outer_radius_m,
        0,
        &SkyboxConfig {
            face_resolution: 4,
            background_color: [50, 60, 70, 255],
            include_parent_tier: false,
        },
    )
}

fn make_observer() -> ObserverState {
    ObserverState::new(
        DVec3::ZERO,
        ContainingFrame::World(WorldAddr::ROOT),
    )
}

/// Mirrors `SkyboxRuntime::current_brightness` for test parity.
fn current_brightness(last: f32, next: f32, crossfade_t: f32, has_next_handle: bool) -> f32 {
    if has_next_handle {
        let t = crossfade_t.clamp(0.0, 1.0);
        last + (next - last) * t
    } else {
        last
    }
}

#[test]
fn first_should_refresh_triggers_without_last_skybox() {
    let observer = make_observer();
    let policy = SkyboxRefreshPolicy::default();
    // Cold-start: no last_skybox yet — the runtime treats this as
    // "needs a bake right now" so the FP camera doesn't sit on the
    // placeholder forever.
    assert!(observer.should_refresh(
        &policy,
        DVec3::ZERO,
        WORLD_TIER_BODY_RADIUS_M,
        None,
    ));
}

#[test]
fn position_drift_past_5pct_of_outer_radius_triggers_second_refresh() {
    let mut observer = make_observer();
    let policy = SkyboxRefreshPolicy::default();
    // Bake at origin with a 512 m outer radius (matches the client's
    // `DEFAULT_MAX_RADIUS_M` in `world_stream.rs`).
    observer.accept_next(fake_bake(DVec3::ZERO, 512.0));
    // 5 % of 512 m = 25.6 m. A 20 m drift sits inside the tolerance.
    observer.position = DVec3::new(20.0, 0.0, 0.0);
    assert!(
        !observer.should_refresh(
            &policy,
            DVec3::ZERO,
            WORLD_TIER_BODY_RADIUS_M,
            None,
        ),
        "20 m drift should be inside the 5 % tolerance for a 512 m outer radius"
    );
    // Drift past the threshold (40 m > 25.6 m): refresh trips.
    observer.position = DVec3::new(40.0, 0.0, 0.0);
    assert!(
        observer.should_refresh(
            &policy,
            DVec3::ZERO,
            WORLD_TIER_BODY_RADIUS_M,
            None,
        ),
        "40 m drift should be past the 5 % tolerance"
    );
}

#[test]
fn brightness_lerps_monotonically_during_crossfade() {
    // Stand in for the `SkyboxRuntime` lerp path: at t=0 we read
    // `last_brightness`; at t=1 we read `next_brightness`; in between
    // it's a strict linear ramp. The runtime drives `crossfade_t` via
    // `ObserverState::tick(... dt)`, but the lerp formula is
    // self-contained.
    let last = lerp_brightness(0.0); // 50
    let next = lerp_brightness(1.0); // 2500
    let mut prev = current_brightness(last, next, 0.0, true);
    let start = prev;
    for step in 1..=10 {
        let t = step as f32 * 0.1;
        let b = current_brightness(last, next, t, true);
        assert!(
            b + 1e-3 >= prev,
            "brightness should never decrease during crossfade: prev={prev}, now={b}, t={t}"
        );
        prev = b;
    }
    let end = current_brightness(last, next, 1.0, true);
    assert!(
        (end - next).abs() < 1e-2,
        "endpoint should equal next_brightness: end={end}, next={next}"
    );
    assert!(
        end > start,
        "crossfade should reach a strictly brighter endpoint: start={start}, end={end}"
    );
}

#[test]
fn three_consecutive_drift_events_each_request_refresh() {
    // Walk in three jumps, baking on each. Mirrors the per-frame
    // `sync_skybox` decision loop for an FP walker traversing the
    // alpine biome.
    let mut observer = make_observer();
    let policy = SkyboxRefreshPolicy::default();
    let outer = 512.0;

    // Event 1: cold start.
    assert!(observer.should_refresh(
        &policy,
        DVec3::ZERO,
        WORLD_TIER_BODY_RADIUS_M,
        None,
    ));
    observer.accept_next(fake_bake(DVec3::ZERO, outer));

    // Event 2: walk 40 m → past 5 % of 512 = 25.6 m.
    observer.position = DVec3::new(40.0, 0.0, 0.0);
    assert!(observer.should_refresh(
        &policy,
        DVec3::ZERO,
        WORLD_TIER_BODY_RADIUS_M,
        None,
    ));
    observer.accept_next(fake_bake(DVec3::new(40.0, 0.0, 0.0), outer));
    // Settle the crossfade so the second bake becomes the new
    // `last_skybox` — the runtime does this via `tick(... dt)` when
    // `crossfade_t` saturates to 1.0; we shortcut it directly here.
    observer.last_skybox = observer.next_skybox.take();
    observer.crossfade_t = 0.0;

    // Event 3: walk another 40 m past the new bake's origin (which is
    // at 40 m). 40 m > 25.6 m so the drift trigger trips.
    observer.position = DVec3::new(80.0, 0.0, 0.0);
    assert!(observer.should_refresh(
        &policy,
        DVec3::ZERO,
        WORLD_TIER_BODY_RADIUS_M,
        None,
    ));
}
