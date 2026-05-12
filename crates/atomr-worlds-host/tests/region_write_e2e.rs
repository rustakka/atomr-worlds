//! End-to-end: `WriteRegion` applies a brush, fans out a `RegionDelta`,
//! reads reflect the change.

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::interaction::InteractionUnit;
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest, AABB};
use atomr_worlds_voxel::Voxel;
use std::time::Duration;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

async fn fresh_host() -> LocalHost {
    LocalHost::with_seed(TEST_SEED).await.expect("host")
}

#[tokio::test]
async fn sphere_brush_writes_voxels_inside_radius() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);

    // Pick a brush radius that covers ~1 voxel at world scale, so we don't
    // overwhelm the test with a 5e6-m planet-spanning brush.
    let scale = MetricScale::DEFAULT_WORLD;
    let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
    let center = DVec3::new(mpv * 2.5, mpv * 2.5, mpv * 2.5);
    let unit = InteractionUnit::sphere(mpv * 1.0, Lod::new(scale.max_depth));

    let req = WorldRequest::WriteRegion { addr, center, unit, voxel: Voxel::new(123) };
    let env = Envelope::new(0, addr, req);
    host.request(env).await.unwrap();

    // The voxel at the center should read back as 123.
    let pos_vox = IVec3::new(2, 2, 2);
    let r = Envelope::new(1, addr, WorldRequest::GetVoxel { addr, pos: pos_vox });
    let resp = host.request(r).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::new(123));
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn region_delta_fires_for_overlapping_subscribers() {
    let host = fresh_host().await;
    let addr = Address::World(WorldAddr::ROOT);

    let scale = MetricScale::DEFAULT_WORLD;
    let mpv = scale.meters_per_voxel(Lod::new(scale.max_depth));
    let center = DVec3::new(mpv * 2.5, mpv * 2.5, mpv * 2.5);
    let unit = InteractionUnit::sphere(mpv * 1.0, Lod::new(scale.max_depth));

    let sub_env = Envelope::new(
        0,
        addr,
        WorldRequest::Subscribe {
            addr,
            region: AABB::new(IVec3::new(0, 0, 0), IVec3::new(16, 16, 16)),
            lod: Lod::new(0),
            sub_id: 200,
        },
    );
    let mut rx = host.subscribe(sub_env).await.expect("subscribe");
    // Drain initial snapshot.
    while let Ok(Some(_)) = tokio::time::timeout(Duration::from_millis(50), rx.recv()).await {}

    let w =
        Envelope::new(1, addr, WorldRequest::WriteRegion { addr, center, unit, voxel: Voxel::new(77) });
    host.request(w).await.unwrap();

    // Drain the channel scanning for a RegionDelta.
    let mut found = false;
    for _ in 0..40 {
        let next = tokio::time::timeout(Duration::from_millis(200), rx.recv()).await;
        if let Ok(Some(env)) = next {
            if let WorldEvent::RegionDelta { voxel, .. } = env.body {
                assert_eq!(voxel, Voxel::new(77));
                found = true;
                break;
            }
        } else {
            break;
        }
    }
    assert!(found, "expected RegionDelta after WriteRegion");
    host.shutdown().await.unwrap();
}
