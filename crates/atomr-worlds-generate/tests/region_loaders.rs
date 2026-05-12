//! Phase 13e gate: heightmap and voxfile authored-region loaders.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{
    heightmap_from_columns, AuthoredRegion, AuthoredRegionStore, HeightmapRegion, LiteralRegion,
    RegionAabb, VoxFileRegion, VoxelTransform,
};
use atomr_worlds_voxel::{Brick, Voxel};
use std::collections::HashMap;
use std::sync::Arc;

#[test]
fn heightmap_columns_helper_builds_expected_field() {
    let mut cols = HashMap::new();
    cols.insert((0, 0), 4);
    cols.insert((1, 1), 7);
    let r = heightmap_from_columns("test", IVec3::ZERO, 8, 8, cols, 1);
    let mut b = Brick::new();
    let n = r.apply_to_brick(IVec3::ZERO, &mut b);
    assert_eq!(n, 4 + 7); // 4 voxels at (0,_,0) + 7 voxels at (1,_,1)
    assert_eq!(b.get(IVec3::new(0, 0, 0)), Voxel::new(1));
    assert_eq!(b.get(IVec3::new(0, 3, 0)), Voxel::new(1));
    assert_eq!(b.get(IVec3::new(0, 4, 0)), Voxel::EMPTY);
}

#[test]
fn store_with_mixed_region_types() {
    let mut store = AuthoredRegionStore::new();
    // A literal cube and a heightmap stripe coexist.
    let mut lit_map = HashMap::new();
    lit_map.insert(IVec3::new(0, 0, 0), Voxel::new(1));
    let lit = Arc::new(LiteralRegion::new(
        "lit",
        RegionAabb::new(IVec3::ZERO, IVec3::new(1, 1, 1)),
        lit_map,
    ));
    store.register(lit);

    let mut cols = HashMap::new();
    cols.insert((4, 4), 2);
    let hm = Arc::new(HeightmapRegion::new(
        "hm",
        IVec3::ZERO,
        8,
        8,
        {
            let mut v = vec![0u16; 64];
            v[4 * 8 + 4] = 2;
            v
        },
        9,
    ));
    let _ = cols;
    store.register(hm);

    let mut b = Brick::new();
    let written = store.apply_all(IVec3::ZERO, 16, &mut b);
    assert_eq!(written, 1 /* lit */ + 2 /* hm column of height 2 */);
    assert_eq!(b.get(IVec3::new(0, 0, 0)), Voxel::new(1));
    assert_eq!(b.get(IVec3::new(4, 0, 4)), Voxel::new(9));
    assert_eq!(b.get(IVec3::new(4, 1, 4)), Voxel::new(9));
}

#[test]
fn voxfile_via_transform_round_trips() {
    let voxels = vec![(IVec3::new(0, 0, 0), 10), (IVec3::new(1, 0, 0), 20)];
    let r = VoxFileRegion::new(
        "import",
        voxels,
        VoxelTransform::translation(IVec3::new(5, 0, 0)),
    );
    let mut b = Brick::new();
    let written = r.apply_to_brick(IVec3::new(0, 0, 0), &mut b);
    assert_eq!(written, 2);
    assert_eq!(b.get(IVec3::new(5, 0, 0)), Voxel::new(10));
    assert_eq!(b.get(IVec3::new(6, 0, 0)), Voxel::new(20));
}

#[test]
fn heightmap_deterministic_across_constructions() {
    let mk = || {
        HeightmapRegion::new(
            "d",
            IVec3::ZERO,
            4,
            4,
            vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            5,
        )
    };
    let a = mk();
    let b = mk();
    let mut ba = Brick::new();
    let mut bb = Brick::new();
    a.apply_to_brick(IVec3::ZERO, &mut ba);
    b.apply_to_brick(IVec3::ZERO, &mut bb);
    assert_eq!(ba.to_bytes(), bb.to_bytes());
}
