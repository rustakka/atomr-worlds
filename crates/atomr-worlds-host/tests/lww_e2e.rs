//! End-to-end tests for the Rec 4 HLC-LWW voxel overlay: client-stamped writes
//! resolve by `(HlcTimestamp, WriterId)`, the loser gets a `WriteRejected`
//! reply, carves survive a restart as tombstones, and a fixed write script
//! under a manual clock produces byte-identical bricks across hosts.

use std::sync::Arc;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::lww::WriterId;
use atomr_worlds_core::HlcTimestamp;
use atomr_worlds_host::{
    clock::Clock, InMemoryJournal, InMemorySnapshotStore, LocalHost, LocalHostConfig, WorldHost,
    WorldPersistence,
};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::Voxel;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

fn persistence() -> Arc<WorldPersistence> {
    Arc::new(WorldPersistence::new(InMemoryJournal::new(), InMemorySnapshotStore::new()))
}

async fn host(persistence: Option<Arc<WorldPersistence>>, clock: Clock) -> LocalHost {
    LocalHost::new(LocalHostConfig { root_seed: TEST_SEED, persistence, clock, ..Default::default() })
        .await
        .expect("host")
}

fn stamped(addr: Address, pos: IVec3, v: u16, wall: u64, writer: u64) -> WorldRequest {
    WorldRequest::WriteVoxelStamped {
        addr,
        pos,
        voxel: Voxel::new(v),
        ts: HlcTimestamp::new(wall, 0),
        writer: WriterId(writer),
    }
}

#[tokio::test]
async fn stamped_writes_resolve_by_stamp_and_reject_the_loser() {
    let host = host(None, Clock::Wall).await;
    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(3, 3, 3);

    // Two writes at the same HLC; the higher writer id wins.
    host.request(Envelope::new(1, addr, stamped(addr, pos, 11, 5, 1))).await.unwrap();
    host.request(Envelope::new(2, addr, stamped(addr, pos, 22, 5, 2))).await.unwrap();

    // A later request with a *lower* stamp must lose and report the winner.
    let resp = host.request(Envelope::new(3, addr, stamped(addr, pos, 33, 5, 1))).await.unwrap();
    match resp.body {
        WorldEvent::WriteRejected(r) => {
            assert_eq!(r.pos, pos);
            assert_eq!(r.current, Voxel::new(22));
        }
        other => panic!("expected WriteRejected, got {other:?}"),
    }

    // The resolved value is the writer-2 write.
    let resp = host.request(Envelope::new(4, addr, WorldRequest::GetVoxel { addr, pos })).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::new(22));
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn carve_over_procedural_solid_survives_restart_as_tombstone() {
    // (0,-50,0) is well underground for the Earth-class default world.
    let p = persistence();
    let addr = Address::World(WorldAddr::ROOT);
    let pos = IVec3::new(0, -50, 0);

    let solid_before = {
        let host = host(Some(p.clone()), Clock::Wall).await;
        let resp =
            host.request(Envelope::new(1, addr, WorldRequest::GetVoxel { addr, pos })).await.unwrap();
        let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
        // Carve it (host-authoritative WriteVoxel → EMPTY).
        host.request(Envelope::new(2, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::EMPTY }))
            .await
            .unwrap();
        host.shutdown().await.unwrap();
        voxel
    };
    assert_ne!(solid_before, Voxel::EMPTY, "precondition: {pos:?} must be procedural-solid");

    // Fresh host, same persistence: the carve must persist (the old
    // remove-on-empty would have let the procedural solid reappear).
    let host = host(Some(p.clone()), Clock::Wall).await;
    let resp =
        host.request(Envelope::new(3, addr, WorldRequest::GetVoxel { addr, pos })).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::EMPTY, "carve must survive recovery as a tombstone");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn manual_clock_writes_are_byte_identical_across_hosts() {
    let addr = Address::World(WorldAddr::ROOT);
    let bc = IVec3::new(0, 0, 0);
    let script = [(IVec3::new(1, 1, 1), 7u16), (IVec3::new(2, 1, 1), 8), (IVec3::new(1, 1, 1), 9)];

    async fn run(addr: Address, bc: IVec3, script: &[(IVec3, u16)]) -> bytes::Bytes {
        let host = host(None, Clock::manual(1)).await;
        for (i, (pos, v)) in script.iter().enumerate() {
            host.request(Envelope::new(
                i as u64,
                addr,
                WorldRequest::WriteVoxel { addr, pos: *pos, voxel: Voxel::new(*v) },
            ))
            .await
            .unwrap();
        }
        let resp = host
            .request(Envelope::new(99, addr, WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) }))
            .await
            .unwrap();
        let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
        host.shutdown().await.unwrap();
        payload
    }

    let a = run(addr, bc, &script).await;
    let b = run(addr, bc, &script).await;
    assert_eq!(a.as_ref(), b.as_ref(), "same script + manual clock ⇒ byte-identical brick");
}
