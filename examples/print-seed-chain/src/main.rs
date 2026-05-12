//! Smoke-test binary: print the derived seed chain for a sample `WorldAddr`.

use atomr_worlds_core::addr::{Level, LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::MetricScale;

fn main() {
    let addr = WorldAddr {
        universe: LevelKey::new(IVec3::ZERO, 0),
        galaxy: LevelKey::new(IVec3::new(3, -2, 1), 0),
        sector: LevelKey::new(IVec3::new(0, 1, 0), 0),
        system: LevelKey::new(IVec3::new(7, 7, 7), 0),
        world: LevelKey::new(IVec3::ZERO, 1), // alt-plane at world level
    };

    let root_seed: u64 = 0xDEAD_BEEF_CAFE_F00D;
    let chain = addr.seed_chain(root_seed);

    println!("root seed: 0x{:016X}", root_seed);
    println!("address:   {:#?}", addr);
    println!("seed chain:");
    for l in Level::ALL {
        println!("  {:>9?}: 0x{:016X}", l, chain[l.depth()]);
    }

    println!();
    println!("metric scales (root cube edge / leaf voxel edge):");
    for (label, s) in [
        ("universe", MetricScale::DEFAULT_UNIVERSE),
        ("galaxy", MetricScale::DEFAULT_GALAXY),
        ("sector", MetricScale::DEFAULT_SECTOR),
        ("system", MetricScale::DEFAULT_SYSTEM),
        ("world", MetricScale::DEFAULT_WORLD),
    ] {
        println!("  {:>9}: root {:>10.3e} m   leaf {:.3e} m", label, s.root_size_m, s.leaf_size_m());
    }
}
