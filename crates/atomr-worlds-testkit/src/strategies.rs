use atomr_worlds_core::addr::{LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE};
use proptest::prelude::*;

pub fn arb_ivec3() -> impl Strategy<Value = IVec3> {
    (-1_000_000i64..1_000_000, -1_000_000i64..1_000_000, -1_000_000i64..1_000_000)
        .prop_map(|(x, y, z)| IVec3::new(x, y, z))
}

pub fn arb_level_key() -> impl Strategy<Value = LevelKey> {
    (arb_ivec3(), any::<u32>()).prop_map(|(coord, dim)| LevelKey { coord, dim })
}

pub fn arb_world_addr() -> impl Strategy<Value = WorldAddr> {
    (
        arb_level_key(),
        arb_level_key(),
        arb_level_key(),
        arb_level_key(),
        arb_level_key(),
    )
        .prop_map(|(u, g, s, sy, w)| WorldAddr {
            universe: u,
            galaxy: g,
            sector: s,
            system: sy,
            world: w,
        })
}

pub fn arb_lod(max_depth: u8) -> impl Strategy<Value = Lod> {
    (0u8..=max_depth).prop_map(Lod::new)
}

pub fn arb_voxel() -> impl Strategy<Value = Voxel> {
    any::<u16>().prop_map(Voxel::new)
}

/// Sparse-ish random brick: a handful of cells written, the rest empty.
pub fn arb_brick() -> impl Strategy<Value = Brick> {
    let edge = BRICK_EDGE as i64;
    let cell = (0i64..edge, 0i64..edge, 0i64..edge, 0u16..1024)
        .prop_map(|(x, y, z, m)| (IVec3::new(x, y, z), Voxel::new(m + 1)));
    proptest::collection::vec(cell, 0..64).prop_map(|writes| {
        let mut b = Brick::new();
        for (p, v) in writes {
            b.set(p, v);
        }
        b
    })
}
