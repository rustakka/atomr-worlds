//! `MessageExtractor` stability — same `WorldAddr` → same shard id and entity id.

use atomr_worlds_core::addr::{LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;

// Pull the extractor through `atomr-worlds-host`; cluster sharding behavior is
// not exercised here, just the pure-formatting impl.
use atomr_worlds_host::extractor::WorldExtractor;

fn sample_addr() -> WorldAddr {
    WorldAddr {
        universe: LevelKey { coord: IVec3::ZERO, dim: 0 },
        galaxy: LevelKey { coord: IVec3::new(3, -2, 1), dim: 0 },
        sector: LevelKey { coord: IVec3::new(0, 1, 0), dim: 0 },
        system: LevelKey { coord: IVec3::new(5, 5, 5), dim: 0 },
        world: LevelKey { coord: IVec3::ZERO, dim: 1 },
    }
}

#[test]
fn shard_id_is_stable() {
    let a = WorldExtractor::shard_id_for(&sample_addr());
    let b = WorldExtractor::shard_id_for(&sample_addr());
    assert_eq!(a, b);
    assert!(!a.is_empty());
}

#[test]
fn entity_id_is_stable() {
    let a = WorldExtractor::entity_id_for(&sample_addr());
    let b = WorldExtractor::entity_id_for(&sample_addr());
    assert_eq!(a, b);
}

#[test]
fn distinct_addrs_have_distinct_entity_ids() {
    let a = sample_addr();
    let mut b = a;
    b.world.coord = IVec3::new(1, 1, 1);
    assert_ne!(WorldExtractor::entity_id_for(&a), WorldExtractor::entity_id_for(&b));
}

#[test]
fn shard_id_co_locates_sibling_systems() {
    // Two systems in the same sector should share a shard id.
    let a = sample_addr();
    let mut b = a;
    b.system.coord = IVec3::new(9, 9, 9);
    assert_eq!(WorldExtractor::shard_id_for(&a), WorldExtractor::shard_id_for(&b));
}
