//! End-to-end: the `ice_shell` planetary archetype, forced across a world via
//! [`ForcePolicy`] + [`GenerationPolicy::Custom`], is served through the normal
//! `GetBrick` path. This exercises the full policy → registry → generator wiring
//! the client's `--world-gen ice` flag drives.
//!
//! The ice-shell column is, top-down: a thin SNOW rim, an ICE shell, a buried
//! WATER ocean, then a STONE core. We don't know exactly where the FBM surface
//! lands, so we scan a vertical stack of bricks around the surface band and
//! assert the frozen shell (ICE) and buried ocean (WATER) both appear.

use std::sync::Arc;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_generate::{MATERIAL_ICE, MATERIAL_WATER};
use atomr_worlds_host::{
    ForcePolicy, GenerationPolicy, LocalHost, LocalHostConfig, WorldHost, ICE_SHELL,
};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

const TEST_SEED: u64 = 0xC0FF_EE15_900D;

/// A `LocalHost` whose every world is forced to the ice-shell archetype.
async fn ice_host() -> LocalHost {
    let cfg = LocalHostConfig {
        root_seed: TEST_SEED,
        policy: Arc::new(ForcePolicy(GenerationPolicy::Custom(ICE_SHELL))),
        ..LocalHostConfig::default()
    };
    LocalHost::new(cfg).await.expect("host")
}

async fn fetch_brick(host: &LocalHost, brick: IVec3, lod: Lod) -> Brick {
    let addr = Address::World(WorldAddr::ROOT);
    let env = Envelope::new(0, addr, WorldRequest::GetBrick { addr, brick, lod });
    let resp = host.request(env).await.expect("request");
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot")
    };
    Brick::from_bytes(&payload).expect("decode")
}

fn materials(brick: &Brick) -> Vec<u16> {
    let edge = BRICK_EDGE as i64;
    let mut out = Vec::new();
    for z in 0..edge {
        for y in 0..edge {
            for x in 0..edge {
                let m = brick.get(IVec3::new(x, y, z)).0;
                if m != 0 {
                    out.push(m);
                }
            }
        }
    }
    out
}

/// LOD 0: scanning the surface band yields both the ICE shell and the buried
/// WATER ocean, proving the forced archetype actually generated voxels.
#[tokio::test]
async fn ice_world_has_frozen_shell_and_buried_ocean() {
    let host = ice_host().await;
    // base_surface_m=40 ± amplitude_m=10 → surface ∈ [30, 50]. The ICE band
    // (depth 2..14) and WATER band (depth 14..62) below it together live within
    // world y ∈ [-32, 48], i.e. brick y-indices -2..3.
    let mut saw_ice = false;
    let mut saw_water = false;
    let mut any_solid = false;
    for by in -2..=3 {
        let brick = fetch_brick(&host, IVec3::new(0, by, 0), Lod::new(0)).await;
        for m in materials(&brick) {
            any_solid = true;
            if m == MATERIAL_ICE {
                saw_ice = true;
            }
            if m == MATERIAL_WATER {
                saw_water = true;
            }
        }
    }
    assert!(any_solid, "ice world generated no solid voxels");
    assert!(saw_ice, "expected an ICE shell in the surface band");
    assert!(saw_water, "expected a buried WATER ocean below the shell");
    host.shutdown().await.unwrap();
}

/// LOD 1 (2 m voxels): the archetype's per-LOD sampling captures the surface.
/// A single coarse brick may sit just below or above the surface depending on
/// where the FBM lands, so we scan the LOD-1 stack bracketing the [30, 50]
/// surface band (y-index 0 ≈ below, 1 ≈ straddling, 2 ≈ above) and assert it is
/// neither all-solid nor all-air — i.e. the surface streams at coarse LOD.
#[tokio::test]
async fn ice_world_streams_at_coarse_lod() {
    let host = ice_host().await;
    let cells = BRICK_EDGE.pow(3);
    let mut total = 0usize;
    for by in 0..=2 {
        total += fetch_brick(&host, IVec3::new(0, by, 0), Lod::new(1)).await.nonempty_count as usize;
    }
    assert!(total > 0, "LOD-1 ice stack should have solids below the surface");
    assert!(total < 3 * cells, "LOD-1 ice stack should have air above the surface");
    host.shutdown().await.unwrap();
}
