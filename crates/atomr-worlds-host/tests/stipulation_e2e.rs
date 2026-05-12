//! Phase 13d end-to-end coverage of in-memory authored regions.
//!
//! - Register a `LiteralRegion` before subscription/fetch.
//! - Verify the authored voxels appear in the fetched brick alongside
//!   procedural fill outside the region's bounds.
//! - Persistence is orthogonal to authored content for Phase 13d (only
//!   user-write events journal; region registration is in-memory). A
//!   future phase will add region-manifest persistence.
//! - Determinism: byte-identical brick payloads across two hosts.

use std::collections::HashMap;
use std::sync::Arc;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LiteralRegion, LocalHost, LocalHostConfig, RegionAabb, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, Voxel};

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const AUTHORED_MATERIAL: u16 = 0xBEEF;

fn build_region(name: &str, brick_coord: IVec3, fill: u16) -> Arc<LiteralRegion> {
    // A 4³ block of authored voxels at the corner of the given brick.
    let edge = 16i64;
    let origin = IVec3::new(brick_coord.x * edge, brick_coord.y * edge, brick_coord.z * edge);
    let mut m = HashMap::new();
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                m.insert(
                    IVec3::new(origin.x + x, origin.y + y, origin.z + z),
                    Voxel::new(fill),
                );
            }
        }
    }
    Arc::new(LiteralRegion::new(
        name,
        RegionAabb::new(origin, IVec3::new(origin.x + 4, origin.y + 4, origin.z + 4)),
        m,
    ))
}

async fn fetch_brick(host: &LocalHost, addr: Address, bc: IVec3) -> Brick {
    let env = Envelope::new(
        0,
        addr,
        WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) },
    );
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot");
    };
    Brick::from_bytes(&payload).expect("decode")
}

#[tokio::test]
async fn authored_region_overlays_procedural_fill() {
    let host = LocalHost::with_seed(TEST_SEED).await.unwrap();
    let addr = Address::World(WorldAddr::ROOT);

    // Register a 4³ authored cube at brick (5, 5, 5).
    let bc = IVec3::new(5, 5, 5);
    host.register_authored_region(build_region("city", bc, AUTHORED_MATERIAL));

    let brick = fetch_brick(&host, addr, bc).await;
    // Authored voxels written.
    for z in 0..4 {
        for y in 0..4 {
            for x in 0..4 {
                assert_eq!(
                    brick.get(IVec3::new(x, y, z)),
                    Voxel::new(AUTHORED_MATERIAL),
                    "expected authored voxel at ({x},{y},{z})",
                );
            }
        }
    }
    // Outside the 4³ block within the brick, voxels are procedural —
    // could be empty or some material; we just check that they are NOT
    // the authored sentinel.
    for z in 4..16 {
        let v = brick.get(IVec3::new(0, 0, z));
        assert_ne!(v, Voxel::new(AUTHORED_MATERIAL));
    }
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn brick_outside_region_is_pure_procedural() {
    let host = LocalHost::with_seed(TEST_SEED).await.unwrap();
    let addr = Address::World(WorldAddr::ROOT);
    let region_bc = IVec3::new(5, 5, 5);
    host.register_authored_region(build_region("city", region_bc, AUTHORED_MATERIAL));

    // Brick at (0, 0, 0) — does not overlap the region at brick (5,5,5).
    let brick = fetch_brick(&host, addr, IVec3::new(0, 0, 0)).await;
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                assert_ne!(
                    brick.get(IVec3::new(x, y, z)),
                    Voxel::new(AUTHORED_MATERIAL),
                    "no authored voxels expected outside region's brick",
                );
            }
        }
    }
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn registered_regions_visible_via_store() {
    let host = LocalHost::with_seed(TEST_SEED).await.unwrap();
    let bc = IVec3::new(2, 2, 2);
    host.register_authored_region(build_region("alpha", bc, 1));
    host.register_authored_region(build_region("beta", bc, 2));
    let store = host.authored_region_store();
    let len = store.lock().unwrap().len();
    assert_eq!(len, 2);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn empty_world_with_authored_region() {
    // Demonstrates the canonical "storytelling stage" pattern: empty
    // procedural fill + authored content. Uses GenerationPolicy::Empty
    // applied at the world level so every brick starts empty; authored
    // region writes its voxels on top.
    use atomr_worlds_core::addr::Level;
    use atomr_worlds_generate::GenerationPolicy;
    use atomr_worlds_host::PrefixPolicy;

    let mut policy = PrefixPolicy::new();
    policy.set(Level::World, WorldAddr::ROOT, GenerationPolicy::Empty);
    let host = LocalHost::new(LocalHostConfig {
        root_seed: TEST_SEED,
        policy: Arc::new(policy),
        ..LocalHostConfig::default()
    })
    .await
    .unwrap();
    let addr = Address::World(WorldAddr::ROOT);

    let bc = IVec3::new(7, 7, 7);
    host.register_authored_region(build_region("hero_island", bc, AUTHORED_MATERIAL));

    let brick = fetch_brick(&host, addr, bc).await;
    // 4³ = 64 authored voxels, no procedural content.
    assert_eq!(brick.nonempty_count, 64);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn deterministic_authored_brick_across_hosts() {
    let bc = IVec3::new(5, 5, 5);
    let make_host = || async {
        let h = LocalHost::with_seed(TEST_SEED).await.unwrap();
        h.register_authored_region(build_region("city", bc, AUTHORED_MATERIAL));
        h
    };
    let h1 = make_host().await;
    let h2 = make_host().await;
    let addr = Address::World(WorldAddr::ROOT);
    let b1 = fetch_brick(&h1, addr, bc).await;
    let b2 = fetch_brick(&h2, addr, bc).await;
    assert_eq!(b1.to_bytes(), b2.to_bytes());
    h1.shutdown().await.unwrap();
    h2.shutdown().await.unwrap();
}
