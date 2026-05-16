//! Vanilla preset byte-equality regression.
//!
//! Asserts that `LayeredGenerator(Vanilla)` produces bricks byte-equal to
//! the existing `default_terrain()` for a fixed set of `(seed, brick,
//! lod)` triples. Step 4 satisfies this trivially via the
//! `MonolithicTerrainPass` delegation; Steps 5–7 swap in per-stage
//! byte-equal impls and must keep this test green.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{
    pipeline::LayeredGenerator, strategies::terrain::default_terrain, BrickGenContext,
    BrickGenerator, WorldGenConfig, WorldGenPreset,
};
use atomr_worlds_voxel::BRICK_EDGE;

fn assert_bricks_equal(
    seed: u64,
    coord: IVec3,
    a: &atomr_worlds_voxel::Brick,
    b: &atomr_worlds_voxel::Brick,
) {
    assert_eq!(
        a.nonempty_count, b.nonempty_count,
        "nonempty_count diverges at seed={seed:x} coord={coord:?}",
    );
    for z in 0..BRICK_EDGE as i64 {
        for y in 0..BRICK_EDGE as i64 {
            for x in 0..BRICK_EDGE as i64 {
                let p = IVec3::new(x, y, z);
                assert_eq!(
                    a.get(p),
                    b.get(p),
                    "voxel diverges at seed={seed:x} coord={coord:?} p={p:?}",
                );
            }
        }
    }
}

#[test]
fn layered_vanilla_matches_terrain_generator_byte_for_byte() {
    let terrain = default_terrain();
    let layered = LayeredGenerator::new(WorldGenConfig::preset(WorldGenPreset::Vanilla));

    let seeds = [42u64, 7, 0xCAFE_BABE_DEAD_BEEF, 0xDEAD_C0DE_F00D_FACE];
    let coords = [
        IVec3::new(0, 0, 0),
        IVec3::new(0, -10, 0),
        IVec3::new(3, -4, 5),
        IVec3::new(-2, -8, 11),
        IVec3::new(7, -2, -3),
        IVec3::new(1, 4, -1),
        IVec3::new(8, -1, 8),
        IVec3::new(-4, -6, 2),
    ];
    for &seed in &seeds {
        for &coord in &coords {
            let ctx = BrickGenContext::legacy(seed, coord);
            let a = terrain.generate_brick(&ctx);
            let b = layered.generate_brick(&ctx);
            assert_bricks_equal(seed, coord, &a, &b);
        }
    }
}
