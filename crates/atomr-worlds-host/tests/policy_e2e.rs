//! End-to-end: `PrefixPolicy::Empty` at sector level produces empty bricks
//! for every world inside, while user overlay writes still survive.

use std::sync::Arc;

use atomr_worlds_core::addr::{Address, Level, LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{
    GenerationPolicy, LocalHost, LocalHostConfig, PolicyResolver, PrefixPolicy, WorldHost,
};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, Voxel};

#[tokio::test]
async fn empty_policy_at_sector_makes_all_worlds_empty() {
    let mut policy = PrefixPolicy::new();
    let sector_addr = WorldAddr {
        universe: LevelKey::ROOT,
        galaxy: LevelKey::at(IVec3::new(1, 0, 0)),
        sector: LevelKey::at(IVec3::new(2, 0, 0)),
        ..WorldAddr::ROOT
    };
    policy.set(Level::Sector, sector_addr, GenerationPolicy::Empty);

    let host = LocalHost::new(LocalHostConfig {
        root_seed: 0xCAFE_BABE,
        policy: Arc::new(policy),
        ..Default::default()
    })
    .await
    .unwrap();

    // A world *inside* the configured sector → bricks should be empty.
    let inside = Address::World(WorldAddr {
        universe: LevelKey::ROOT,
        galaxy: LevelKey::at(IVec3::new(1, 0, 0)),
        sector: LevelKey::at(IVec3::new(2, 0, 0)),
        system: LevelKey::at(IVec3::new(7, 7, 7)),
        world: LevelKey::at(IVec3::new(3, 3, 3)),
    });
    let req = WorldRequest::GetBrick { addr: inside, brick: IVec3::new(0, 0, 0), lod: Lod::new(0) };
    let env = Envelope::new(0, inside, req);
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
    let brick = Brick::from_bytes(&payload).unwrap();
    assert_eq!(brick.nonempty_count, 0, "Empty policy must produce empty bricks");
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn empty_policy_still_allows_user_writes() {
    let mut policy = PrefixPolicy::new();
    let addr = WorldAddr::ROOT;
    policy.set(Level::World, addr, GenerationPolicy::Empty);
    let host = LocalHost::new(LocalHostConfig {
        root_seed: 0x1234,
        policy: Arc::new(policy),
        ..Default::default()
    })
    .await
    .unwrap();

    let a = Address::World(addr);
    let pos = IVec3::new(1, 1, 1);
    let w = Envelope::new(0, a, WorldRequest::WriteVoxel { addr: a, pos, voxel: Voxel::new(42) });
    host.request(w).await.unwrap();
    let r = Envelope::new(1, a, WorldRequest::GetVoxel { addr: a, pos });
    let resp = host.request(r).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::new(42));
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn vehicle_inherits_parent_world_empty_policy() {
    use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr, VehicleSlot};
    let mut policy = PrefixPolicy::new();
    let parent_world = WorldAddr {
        universe: LevelKey::ROOT,
        galaxy: LevelKey::at(IVec3::new(1, 0, 0)),
        sector: LevelKey::at(IVec3::new(0, 0, 0)),
        system: LevelKey::at(IVec3::new(0, 0, 0)),
        world: LevelKey::at(IVec3::new(9, 9, 9)),
    };
    policy.set(Level::World, parent_world, GenerationPolicy::Empty);
    let host = LocalHost::new(LocalHostConfig {
        root_seed: 0x5678,
        policy: Arc::new(policy),
        ..Default::default()
    })
    .await
    .unwrap();

    let va = VehicleAddr::new(ParentAddr::World(parent_world), VehicleSlot::new(1, 0));
    let a = Address::Vehicle(va);
    let req = WorldRequest::GetBrick { addr: a, brick: IVec3::new(0, 0, 0), lod: Lod::new(0) };
    let env = Envelope::new(0, a, req);
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { panic!("variant") };
    let brick = Brick::from_bytes(&payload).unwrap();
    assert_eq!(brick.nonempty_count, 0);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn vehicle_frame_round_trips() {
    use atomr_worlds_core::coord::{DVec3, Quat};
    use atomr_worlds_core::vehicle::{AffineFrame, ParentAddr, VehicleAddr, VehicleSlot};
    let host = LocalHost::with_seed(0xABCD).await.unwrap();
    let va = VehicleAddr::new(
        ParentAddr::World(WorldAddr::ROOT),
        VehicleSlot::new(0xDEAD_BEEF, 0),
    );
    let a = Address::Vehicle(va);
    let frame = AffineFrame {
        position: DVec3::new(100.0, 200.0, 300.0),
        orientation: Quat::IDENTITY,
        parent: ParentAddr::World(WorldAddr::ROOT),
    };
    let set_env = Envelope::new(0, a, WorldRequest::SetVehicleFrame { addr: va, frame });
    let resp = host.request(set_env).await.unwrap();
    assert!(matches!(resp.body, WorldEvent::Ack { .. }));

    let get_env = Envelope::new(1, a, WorldRequest::GetVehicleFrame { addr: va });
    let resp = host.request(get_env).await.unwrap();
    let WorldEvent::VehicleFrame { frame: got, .. } = resp.body else { panic!("variant") };
    assert_eq!(got.position, frame.position);
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn vehicle_voxel_space_is_independent_from_parent_world() {
    use atomr_worlds_core::vehicle::{ParentAddr, VehicleAddr, VehicleSlot};
    let host = LocalHost::with_seed(0x1111).await.unwrap();
    let world = Address::World(WorldAddr::ROOT);
    let va = VehicleAddr::new(ParentAddr::World(WorldAddr::ROOT), VehicleSlot::new(7, 0));
    let vehicle = Address::Vehicle(va);

    let pos = IVec3::new(5, 5, 5);
    let w1 = Envelope::new(0, vehicle, WorldRequest::WriteVoxel { addr: vehicle, pos, voxel: Voxel::new(33) });
    host.request(w1).await.unwrap();

    // Reading the same position in the parent world must NOT see the vehicle write.
    let r = Envelope::new(1, world, WorldRequest::GetVoxel { addr: world, pos });
    let resp = host.request(r).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_ne!(voxel, Voxel::new(33), "vehicle voxel must not leak into parent world");

    // Vehicle's own read should see it.
    let rv = Envelope::new(2, vehicle, WorldRequest::GetVoxel { addr: vehicle, pos });
    let resp = host.request(rv).await.unwrap();
    let WorldEvent::Voxel { voxel, .. } = resp.body else { panic!("variant") };
    assert_eq!(voxel, Voxel::new(33));
    host.shutdown().await.unwrap();
}

#[tokio::test]
async fn _resolver_type_check() {
    // Compile-time check that DefaultPolicy implements PolicyResolver.
    fn _check(_: &dyn PolicyResolver) {}
    _check(&atomr_worlds_host::DefaultPolicy);
}
