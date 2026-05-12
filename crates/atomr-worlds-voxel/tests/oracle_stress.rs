//! Stress: octree must agree with a HashMap oracle, and empty-space skipping
//! must not blow up arena probe counts on sparse trees.

use std::collections::HashMap;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::splitmix64;
use atomr_worlds_voxel::{Octree, Voxel};

struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(1))
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0xA0B1_C2D3_E4F5_0617);
        splitmix64(self.0)
    }
    fn ranged_i64(&mut self, half: i64) -> i64 {
        let v = self.next() as i64;
        v.rem_euclid(2 * half) - half
    }
}

#[test]
fn octree_matches_hashmap_oracle() {
    let max_depth: u8 = 4;
    let mut oct = Octree::new(1024.0, max_depth);
    let half_extent = (1i64 << max_depth) * 8; // 128 for depth 4
    let mut oracle: HashMap<IVec3, Voxel> = HashMap::new();

    let mut rng = Rng::new(0xC0FE_E000_F00D);
    for i in 0..5_000 {
        let p = IVec3::new(
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
        );
        let v = Voxel::new(((rng.next() & 0x7FFF) + 1) as u16);
        oct.set_voxel(p, v).unwrap();
        oracle.insert(p, v);

        // Spot-check a previously-written cell every 100 writes.
        if i % 100 == 99 && !oracle.is_empty() {
            let keys: Vec<_> = oracle.keys().copied().collect();
            let pick = keys[(rng.next() as usize) % keys.len()];
            assert_eq!(oct.get_voxel(pick).unwrap(), oracle[&pick]);
        }
    }

    // Full oracle sweep.
    for (&p, &v) in &oracle {
        assert_eq!(oct.get_voxel(p).unwrap(), v, "mismatch at {p:?}");
    }
}

#[test]
fn empty_space_skip_bounded_probes() {
    let max_depth: u8 = 6;
    let mut oct = Octree::new(4096.0, max_depth);
    let half_extent = (1i64 << max_depth) * 8;

    // Sparse population: 10 written cells.
    let mut rng = Rng::new(0xDEAD_BEEF_BEEE);
    for _ in 0..10 {
        let p = IVec3::new(
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
        );
        oct.set_voxel(p, Voxel::new(1)).unwrap();
    }

    // 1000 random reads at unwritten coords.
    oct.reset_probes();
    let mut empties = 0u64;
    for _ in 0..1000 {
        let p = IVec3::new(
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
            rng.ranged_i64(half_extent),
        );
        if oct.get_voxel(p).unwrap() == Voxel::EMPTY {
            empties += 1;
        }
    }

    // Sparse tree: nearly every read should be empty.
    assert!(empties >= 990, "expected ≥99% empty reads on a sparse tree, got {empties}/1000");
    // And every read should bottom out at no more than `max_depth + 1` probes.
    let per_read_cap = (max_depth as u64 + 1) * 2;
    let total_cap = 1000 * per_read_cap;
    assert!(
        oct.probes() <= total_cap,
        "probe count {} exceeded budget {} (cap {} per read)",
        oct.probes(),
        total_cap,
        per_read_cap
    );
}
