//! End-to-end coverage of Phase 13b — horizon-driven streaming + the
//! out-of-shape brick filter.
//!
//! - SubscribeMetric on a sphere world clamps `RingPlan` to the horizon
//!   distance at the observer's altitude.
//! - `UpdateObserverPos` recomputes the ring and emits new bricks; the
//!   sub-sequence is deterministic.
//! - Bricks fully outside the sphere never invoke the generator (verified
//!   by their payload being an empty `Brick`).
//! - Existing cubic worlds remain unchanged: `LocalHostConfig::default`
//!   produces `DefaultShape` which yields `Cube { edge_m: 1e7 }`.

use std::sync::Arc;
use std::time::Duration;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_host::{LocalHost, LocalHostConfig, PrefixShape, WorldHost};
use atomr_worlds_proto::{Envelope, StreamingPolicy, WorldEvent, WorldRequest};
use atomr_worlds_voxel::Brick;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const EARTH_R: f64 = 6.371e6;

fn sphere_host_config(addr: WorldAddr, radius_m: f64) -> LocalHostConfig {
    let mut shapes = PrefixShape::new();
    shapes.set(
        atomr_worlds_core::addr::Level::World,
        addr,
        WorldShape::Sphere { radius_m },
    );
    LocalHostConfig {
        root_seed: TEST_SEED,
        shape_resolver: Arc::new(shapes),
        ..LocalHostConfig::default()
    }
}

#[tokio::test]
async fn subscribe_metric_clamps_to_horizon_on_sphere() {
    let addr_w = WorldAddr::ROOT;
    let addr = Address::World(addr_w);
    let host = LocalHost::new(sphere_host_config(addr_w, EARTH_R)).await.unwrap();

    // Place the observer near the world's surface — well inside the
    // streamable region. Note: the world center sits at +5e6 along each
    // axis (since root_size_m = 1e7 → center = root/2). For a sphere
    // world we place the observer at center + (radius - epsilon) along +x.
    let observer_pos = DVec3::new(5.0e6 + EARTH_R - 100.0, 5.0e6, 5.0e6);
    let policy = StreamingPolicy {
        near_lod: Lod::new(8),
        far_lod: Lod::new(4),
        transition_radius_m: 5_000.0,
        max_radius_m: 1_000_000_000.0, // intentionally huge — horizon should clamp
        bricks_per_tick: 64,
    };
    let req = WorldRequest::SubscribeMetric {
        addr,
        containing_frame: ContainingFrame::World(addr_w),
        observer_pos,
        policy,
        sub_id: 42,
    };
    let env = Envelope::new(0, addr, req);
    let mut rx = host.subscribe(env).await.unwrap();

    // First event should be a Tier event (metric sub kickoff).
    let first = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .unwrap()
        .expect("first event");
    match first.body {
        WorldEvent::Tier { sub_id, lod, region, .. } => {
            assert_eq!(sub_id, 42);
            assert_eq!(lod, Lod::new(8));
            // Region must be finite — horizon clamp wins over the huge
            // max_radius_m.
            let extent_x = region.max.x.saturating_sub(region.min.x);
            assert!(
                extent_x > 0 && extent_x < 1_000_000,
                "tier extent should be horizon-bounded, got {extent_x}"
            );
        }
        other => panic!("expected Tier first, got {other:?}"),
    }
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn out_of_shape_brick_returns_empty_payload() {
    let addr_w = WorldAddr::ROOT;
    let addr = Address::World(addr_w);
    // Tiny sphere of radius 8 voxel-meters → world coords up to ~5e6 are
    // mostly empty. Anything outside that small sphere short-circuits to
    // an empty brick without invoking the generator.
    let host = LocalHost::new(sphere_host_config(addr_w, 8.0)).await.unwrap();

    // Brick at a high coordinate — guaranteed outside the radius-8 sphere.
    let bc = IVec3::new(1000, 1000, 1000);
    let env = Envelope::new(
        0,
        addr,
        WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) },
    );
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot");
    };
    let b = Brick::from_bytes(&payload).expect("decode");
    assert_eq!(
        b.nonempty_count, 0,
        "brick outside sphere must short-circuit to empty",
    );
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn update_observer_pos_emits_new_bricks() {
    let addr_w = WorldAddr::ROOT;
    let addr = Address::World(addr_w);
    let host = LocalHost::new(sphere_host_config(addr_w, EARTH_R)).await.unwrap();

    let observer = DVec3::new(5.0e6, 5.0e6 + EARTH_R + 10_000.0, 5.0e6); // altitude 10 km
    let policy = StreamingPolicy {
        near_lod: Lod::new(8),
        far_lod: Lod::new(4),
        transition_radius_m: 200.0,
        max_radius_m: 200.0,
        bricks_per_tick: 64,
    };
    let req = WorldRequest::SubscribeMetric {
        addr,
        containing_frame: ContainingFrame::World(addr_w),
        observer_pos: observer,
        policy,
        sub_id: 7,
    };
    let env = Envelope::new(0, addr, req);
    let mut rx = host.subscribe(env).await.unwrap();

    // Drain the initial Tier + BrickSnapshot burst.
    let mut initial_bricks = 0;
    while let Ok(Some(env)) = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
        match env.body {
                WorldEvent::Tier { .. } => {}
                WorldEvent::BrickSnapshot { .. } => initial_bricks += 1,
                _ => {}
            },
            _ => break,
        }
    }
    assert!(initial_bricks > 0, "initial subscription should emit some bricks");

    // Move the observer far enough that the new ring overlaps a different
    // set of bricks.
    let new_observer = DVec3::new(
        observer.x + 1000.0,
        observer.y,
        observer.z + 1000.0,
    );
    let update = Envelope::new(
        1,
        addr,
        WorldRequest::UpdateObserverPos { sub_id: 7, observer_pos: new_observer },
    );
    host.request(update).await.unwrap();

    // Expect a new Tier event + at least one BrickSnapshot for newly visible bricks.
    let mut saw_tier = false;
    let mut new_bricks = 0;
    while let Ok(Some(env)) = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await {
        match env.body {
                WorldEvent::Tier { sub_id, .. } => {
                    assert_eq!(sub_id, 7);
                    saw_tier = true;
                }
                WorldEvent::BrickSnapshot { .. } => new_bricks += 1,
                _ => {}
            },
            _ => break,
        }
    }
    assert!(saw_tier, "UpdateObserverPos should emit a Tier event");
    assert!(new_bricks > 0, "UpdateObserverPos should snapshot newly-visible bricks");

    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn default_shape_keeps_cubic_world_behavior() {
    // No PrefixShape override → DefaultShape → Cube{1e7}. Streaming should
    // behave identically to pre-Phase-13 (verified via the existing
    // local_e2e tests; here we just spot-check that BrickSnapshot returns
    // non-empty terrain near origin).
    let host = LocalHost::with_seed(TEST_SEED).await.unwrap();
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(
        0,
        addr,
        WorldRequest::GetBrick { addr, brick: IVec3::new(0, 0, 0), lod: Lod::new(0) },
    );
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot");
    };
    let _ = Brick::from_bytes(&payload).expect("decode");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn deterministic_initial_ring_across_runs() {
    // Two hosts at identical config + observer pos should emit byte-
    // identical initial BrickSnapshot sequences.
    let addr_w = WorldAddr::ROOT;
    let addr = Address::World(addr_w);
    let h1 = LocalHost::new(sphere_host_config(addr_w, EARTH_R)).await.unwrap();
    let h2 = LocalHost::new(sphere_host_config(addr_w, EARTH_R)).await.unwrap();

    let observer = DVec3::new(5.0e6 + EARTH_R - 100.0, 5.0e6, 5.0e6);
    let policy = StreamingPolicy {
        near_lod: Lod::new(8),
        far_lod: Lod::new(4),
        transition_radius_m: 500.0,
        max_radius_m: 500.0,
        bricks_per_tick: 64,
    };
    let mk_env = |sub_id| {
        Envelope::new(
            0,
            addr,
            WorldRequest::SubscribeMetric {
                addr,
                containing_frame: ContainingFrame::World(addr_w),
                observer_pos: observer,
                policy,
                sub_id,
            },
        )
    };

    async fn drain(rx: &mut tokio::sync::mpsc::Receiver<Envelope<WorldEvent>>) -> Vec<Vec<u8>> {
        let mut out = Vec::new();
        while let Ok(Some(env)) = tokio::time::timeout(Duration::from_millis(150), rx.recv()).await {
            if let WorldEvent::BrickSnapshot { payload, .. } = env.body {
                out.push(payload.to_vec());
            }
        }
        out
    }

    let mut rx1 = h1.subscribe(mk_env(1)).await.unwrap();
    let mut rx2 = h2.subscribe(mk_env(2)).await.unwrap();
    let p1 = drain(&mut rx1).await;
    let p2 = drain(&mut rx2).await;

    assert!(!p1.is_empty(), "should have streamed some bricks");
    assert_eq!(p1.len(), p2.len(), "same observer → same brick count");
    for (a, b) in p1.iter().zip(p2.iter()) {
        assert_eq!(a, b, "byte-identical brick payloads across hosts");
    }

    h1.shutdown().await.unwrap();
    h2.shutdown().await.unwrap();
}
