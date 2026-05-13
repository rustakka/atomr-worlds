//! End-to-end tests for `LocalHost`: read, write, subscribe, deterministic
//! brick payloads across separate hosts.

use std::time::Duration;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_voxel::{Brick, Voxel};

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

async fn fresh_host() -> LocalHost {
    LocalHost::with_seed(TEST_SEED).await.expect("host")
}

#[tokio::test]
async fn brick_round_trip() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let req = WorldRequest::GetBrick { addr, brick: IVec3::new(0, 0, 0), lod: Lod::new(0) };
    let env = Envelope::new(1, addr, req);
    let resp = host.request(env).await.expect("request");
    let WorldEvent::BrickSnapshot { payload, brick, .. } = resp.body else {
        panic!("expected BrickSnapshot")
    };
    assert_eq!(brick, IVec3::new(0, 0, 0));
    let _b = Brick::from_bytes(&payload).expect("decode");
    host.shutdown().await.unwrap();
}

/// Regression for the FP-streamer "only +X+Z loads" bug: bricks at
/// negative brick coords must generate terrain, not get short-circuited
/// to an empty Brick by `brick_inside_shape`. Pre-fix, `brick_inside_shape`
/// offset by `root_size_m/2`, so any brick with a coord ≤ -2 fell outside
/// the cube and was returned empty. With a 10 M m world that meant ~half
/// the loaded ring around an origin-anchored observer was empty.
///
/// The check here: pull bricks at (-3, 0, -3), (-3, 0, 0), (0, 0, -3),
/// (3, 0, 3) — all sit on the natural terrain surface — and assert each
/// has at least one solid voxel. If any side returns an entirely-empty
/// brick, the asymmetry is back.
#[tokio::test]
async fn negative_coord_bricks_generate_terrain() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let cases = [
        IVec3::new(-3, 0, -3),
        IVec3::new(-3, 0, 0),
        IVec3::new(0, 0, -3),
        IVec3::new(3, 0, 3),
    ];
    for bc in cases {
        let env = Envelope::new(
            0,
            addr,
            WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) },
        );
        let resp = host.request(env).await.expect("request");
        let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
            panic!("expected BrickSnapshot for {bc:?}");
        };
        let b = Brick::from_bytes(&payload).expect("decode");
        assert!(
            b.nonempty_count > 0,
            "brick at {bc:?} returned empty — `brick_inside_shape` likely \
             rejected it because of a corner-offset coordinate convention"
        );
    }
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn two_hosts_same_seed_produce_identical_bricks() {
    let h1 = fresh_host().await;
    let h2 = fresh_host().await;

    let addr = Address::World(WorldAddr::ROOT);
    let bc = IVec3::new(0, 1, 0);
    let mk = || Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) });

    let r1 = h1.request(mk()).await.expect("r1");
    let r2 = h2.request(mk()).await.expect("r2");

    let (WorldEvent::BrickSnapshot { payload: p1, .. }, WorldEvent::BrickSnapshot { payload: p2, .. }) =
        (r1.body, r2.body) else { panic!("variant mismatch") };
    assert_eq!(p1.as_ref(), p2.as_ref(), "brick payload must be deterministic across hosts");

    h1.shutdown().await.unwrap();
    h2.shutdown().await.unwrap();
}

#[tokio::test]
async fn write_then_read_returns_written_voxel() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(2, 2, 2);

    // Write.
    let w_env = Envelope::new(1, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(7) });
    let ack = host.request(w_env).await.expect("write");
    assert!(matches!(ack.body, WorldEvent::Ack { .. }));

    // Read back.
    let r_env = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos });
    let resp = host.request(r_env).await.expect("read");
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::new(7));

    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn subscribe_receives_initial_snapshots() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(
        1,
        addr,
        WorldRequest::Subscribe {
            addr,
            region: AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16)),
            lod: Lod::new(0),
            sub_id: 100,
        },
    );
    let mut rx = host.subscribe(env).await.expect("subscribe");

    let snap = tokio::time::timeout(Duration::from_secs(2), rx.recv())
        .await
        .expect("timeout")
        .expect("snapshot");

    assert!(matches!(snap.body, WorldEvent::BrickSnapshot { .. }));
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn subscribe_receives_delta_on_write() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(
        1,
        addr,
        WorldRequest::Subscribe {
            addr,
            region: AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16)),
            lod: Lod::new(0),
            sub_id: 101,
        },
    );
    let mut rx = host.subscribe(env).await.expect("subscribe");

    // Drain initial snapshot.
    let _initial =
        tokio::time::timeout(Duration::from_secs(2), rx.recv()).await.expect("init").expect("init");

    // Write inside the subscribed region.
    let pos = IVec3::new(3, 4, 5);
    let w_env = Envelope::new(2, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(99) });
    host.request(w_env).await.expect("write");

    // Wait for the delta — there may be other queued envelopes; scan for VoxelDelta.
    let mut found = false;
    for _ in 0..50 {
        let next = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        match next {
            Ok(Some(env)) => {
                if let WorldEvent::VoxelDelta { pos: p, after, .. } = env.body {
                    assert_eq!(p, pos);
                    assert_eq!(after, Voxel::new(99));
                    found = true;
                    break;
                }
            }
            _ => continue,
        }
    }
    assert!(found, "expected VoxelDelta for the written voxel");

    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn write_outside_region_does_not_emit_delta() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(
        1,
        addr,
        WorldRequest::Subscribe {
            addr,
            region: AABB::new(IVec3::new(0, 0, 0), IVec3::new(4, 4, 4)),
            lod: Lod::new(0),
            sub_id: 102,
        },
    );
    let mut rx = host.subscribe(env).await.expect("subscribe");
    // Drain initial snapshot.
    while let Ok(Some(_)) =
        tokio::time::timeout(Duration::from_millis(50), rx.recv()).await
    {}

    // Write OUTSIDE the region.
    let pos = IVec3::new(50, 50, 50);
    let w_env = Envelope::new(2, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(8) });
    host.request(w_env).await.expect("write");

    // No further events expected.
    let result = tokio::time::timeout(Duration::from_millis(150), rx.recv()).await;
    assert!(result.is_err(), "expected no event but got {:?}", result);

    host.shutdown().await.unwrap();
}
