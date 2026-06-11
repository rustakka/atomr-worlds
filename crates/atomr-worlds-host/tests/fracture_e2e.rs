//! End-to-end tests for the Rec 4 host-authoritative fracture handler: a
//! `Fracture` request runs the integer connectivity decision over the
//! authoritative bricks, carves any detached island through the LWW overlay,
//! and returns a deterministic `FractureApplied` command sequence.

use std::sync::Arc;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_host::{
    InMemoryJournal, InMemorySnapshotStore, LocalHost, LocalHostConfig, WorldHost, WorldPersistence,
};
use atomr_worlds_proto::{Envelope, Force, FractureCommand, FractureRequest, WorldEvent, WorldRequest};
use atomr_worlds_voxel::Voxel;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
/// A high-altitude origin that is procedural air, so a placed blob is the only
/// solid in the fracture region and detaches cleanly.
const BASE: IVec3 = IVec3::new(0, 400, 0);

fn persistence() -> Arc<WorldPersistence> {
    Arc::new(WorldPersistence::new(InMemoryJournal::new(), InMemorySnapshotStore::new()))
}

async fn host(persistence: Option<Arc<WorldPersistence>>) -> LocalHost {
    LocalHost::new(LocalHostConfig {
        root_seed: TEST_SEED,
        persistence,
        ..Default::default()
    })
    .await
    .expect("host")
}

async fn get_voxel(host: &LocalHost, addr: Address, pos: IVec3) -> Voxel {
    let resp =
        host.request(Envelope::new(0, addr, WorldRequest::GetVoxel { addr, pos })).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    voxel
}

/// Place a 2×2×2 floating solid blob centred near `BASE` and return its world
/// voxel positions.
async fn place_blob(host: &LocalHost, addr: Address) -> Vec<IVec3> {
    // Precondition: the region must be procedural air.
    assert_eq!(
        get_voxel(host, addr, IVec3::new(BASE.x + 5, BASE.y, BASE.z + 5)).await,
        Voxel::EMPTY,
        "precondition: {BASE:?} region must be air"
    );
    let mut blob = Vec::new();
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let pos = IVec3::new(BASE.x + dx, BASE.y + dy, BASE.z + dz);
                host.request(Envelope::new(
                    1,
                    addr,
                    WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(3) },
                ))
                .await
                .unwrap();
                blob.push(pos);
            }
        }
    }
    blob
}

async fn fracture(host: &LocalHost, addr: Address, force: Force, material_id: u16) -> atomr_worlds_proto::FractureApplied {
    let req = FractureRequest { addr, impact_pos: BASE, force, material_id };
    let resp = host.request(Envelope::new(2, addr, WorldRequest::Fracture(req))).await.unwrap();
    let WorldEvent::FractureApplied(applied) = resp.body else { panic!("variant") };
    applied
}

#[tokio::test]
async fn fracture_carves_detached_island_and_emits_commands() {
    let p = persistence();
    let addr = Address::World(WorldAddr::ROOT);
    let host = host(Some(p)).await;
    let mut blob = place_blob(&host, addr).await;

    // Zero force ⇒ carve-triggered: always evaluate connectivity.
    let applied = fracture(&host, addr, Force::ZERO, 0).await;

    // Exactly one detached island carrying all 8 blob voxels.
    let spawns: Vec<&FractureCommand> = applied
        .commands
        .iter()
        .filter(|c| matches!(c, FractureCommand::SpawnDebris { .. }))
        .collect();
    assert_eq!(spawns.len(), 1, "one detached island");
    let FractureCommand::SpawnDebris { voxels, anchor, .. } = spawns[0] else { unreachable!() };
    let mut got = voxels.clone();
    got.sort_by_key(|p| (p.x, p.y, p.z));
    blob.sort_by_key(|p| (p.x, p.y, p.z));
    assert_eq!(got, blob);
    assert_eq!(*anchor, BASE, "anchor is the island AABB min");

    // A SetVoxel→EMPTY carve per blob voxel.
    let carves: Vec<IVec3> = applied
        .commands
        .iter()
        .filter_map(|c| match c {
            FractureCommand::SetVoxel { pos, after, .. } if *after == Voxel::EMPTY => Some(*pos),
            _ => None,
        })
        .collect();
    assert_eq!(carves.len(), 8);

    // The host carved the blob authoritatively.
    assert_eq!(get_voxel(&host, addr, BASE).await, Voxel::EMPTY);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn fracture_is_deterministic_across_hosts() {
    let addr = Address::World(WorldAddr::ROOT);

    let run = || async {
        let host = host(None).await;
        place_blob(&host, addr).await;
        let applied = fracture(&host, addr, Force::ZERO, 0).await;
        host.shutdown().await.unwrap();
        applied
    };
    let a = run().await;
    let b = run().await;
    assert_eq!(a, b, "same scene ⇒ byte-identical fracture commands");
}

#[tokio::test]
async fn weak_impact_below_material_yield_does_not_fracture() {
    let addr = Address::World(WorldAddr::ROOT);
    let host = host(None).await;
    place_blob(&host, addr).await;

    // 1 N into stone (yield 1.5e7 Pa) is far below threshold ⇒ no fracture.
    let applied = fracture(&host, addr, Force::from_milli_n(IVec3::new(0, 1000, 0)), 1).await;
    assert!(applied.commands.is_empty(), "below-yield impact must not fracture");
    // Blob is untouched.
    assert_eq!(get_voxel(&host, addr, BASE).await, Voxel::new(3));
    host.shutdown().await.unwrap();
}
