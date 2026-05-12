//! End-to-end tests for the Phase 3 persistence binding: write voxels through
//! one `LocalHost`, drop it, recover state through a fresh `LocalHost`, and
//! verify reads return the persisted values.

use std::sync::Arc;

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{
    persistence_id_for, InMemoryJournal, InMemorySnapshotStore, LocalHost, LocalHostConfig,
    WorldHost, WorldPersistence,
};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::Voxel;

const TEST_SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

fn shared_persistence() -> Arc<WorldPersistence> {
    let journal = InMemoryJournal::new();
    let snapshots = InMemorySnapshotStore::new();
    Arc::new(WorldPersistence::new(journal, snapshots).with_snapshot_every(3))
}

async fn host_with(persistence: Arc<WorldPersistence>) -> LocalHost {
    LocalHost::new(LocalHostConfig {
        root_seed: TEST_SEED,
        persistence: Some(persistence),
        ..Default::default()
    })
    .await
    .expect("host")
}

#[tokio::test]
async fn writes_survive_host_restart() {
    let persistence = shared_persistence();

    // First session: write a voxel and shut down.
    {
        let host = host_with(persistence.clone()).await;
        let addr = Address::World(WorldAddr::ROOT);
        let pos = IVec3::new(2, 2, 2);
        let env =
            Envelope::new(1, addr, WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(7) });
        host.request(env).await.expect("write");
        host.shutdown().await.unwrap();
    }

    // Second session: same persistence, fresh host. The voxel should still be there.
    {
        let host = host_with(persistence.clone()).await;
        let addr = Address::World(WorldAddr::ROOT);
        let pos = IVec3::new(2, 2, 2);
        let env = Envelope::new(2, addr, WorldRequest::GetVoxel { addr, pos });
        let resp = host.request(env).await.expect("read");
        let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
        assert_eq!(voxel, Voxel::new(7));
        host.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn snapshot_triggers_after_n_writes_and_replay_still_works() {
    // snapshot_every = 3 → after 3 writes a snapshot is saved; the 4th write
    // appears as a tail event.
    let persistence = shared_persistence();
    let addr = Address::World(WorldAddr::ROOT);

    {
        let host = host_with(persistence.clone()).await;
        for (i, voxel) in [(IVec3::new(0, 0, 0), 1), (IVec3::new(1, 0, 0), 2), (IVec3::new(2, 0, 0), 3), (IVec3::new(3, 0, 0), 4)] {
            let env = Envelope::new(
                1,
                addr,
                WorldRequest::WriteVoxel { addr, pos: i, voxel: Voxel::new(voxel) },
            );
            host.request(env).await.expect("write");
        }
        host.shutdown().await.unwrap();
    }

    // Confirm a snapshot was actually saved at seq 3 (after the 3rd write).
    let loaded = persistence.snapshot_store().load(&persistence_id_for(addr)).await;
    let (meta, _) = loaded.expect("snapshot exists after 3 writes");
    assert_eq!(meta.sequence_nr, 3);

    // Recover and read all four voxels.
    {
        let host = host_with(persistence.clone()).await;
        for (p, expected) in [
            (IVec3::new(0, 0, 0), 1u16),
            (IVec3::new(1, 0, 0), 2),
            (IVec3::new(2, 0, 0), 3),
            (IVec3::new(3, 0, 0), 4),
        ] {
            let env = Envelope::new(0, addr, WorldRequest::GetVoxel { addr, pos: p });
            let resp = host.request(env).await.expect("read");
            let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
            assert_eq!(voxel, Voxel::new(expected), "voxel at {:?}", p);
        }
        host.shutdown().await.unwrap();
    }
}

#[tokio::test]
async fn brick_snapshot_reflects_persisted_overlay() {
    let persistence = shared_persistence();
    let addr = Address::World(WorldAddr::ROOT);

    {
        let host = host_with(persistence.clone()).await;
        let pos = IVec3::new(5, 5, 5);
        let env = Envelope::new(
            1,
            addr,
            WorldRequest::WriteVoxel { addr, pos, voxel: Voxel::new(0xCAFE) },
        );
        host.request(env).await.expect("write");
        host.shutdown().await.unwrap();
    }

    // After restart, fetching the brick should show the overlay applied.
    let host = host_with(persistence.clone()).await;
    let bc = IVec3::new(0, 0, 0);
    let env = Envelope::new(2, addr, WorldRequest::GetBrick { addr, brick: bc, lod: Lod::new(0) });
    let resp = host.request(env).await.expect("read");
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
    let brick = atomr_worlds_voxel::Brick::from_bytes(&payload).expect("decode");
    assert_eq!(brick.get(IVec3::new(5, 5, 5)), Voxel::new(0xCAFE));
    host.shutdown().await.unwrap();
}
