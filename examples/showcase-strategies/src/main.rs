//! Phase 19 demo: rotates through `WorldGenPreset::{Vanilla, Advanced,
//! Showcase}` and prints a YZ slice from a surface-straddling brick for
//! each, plus the strategy `id()` of every slot. The point is to see the
//! pipeline swap implementations without code edits.
//!
//! Run:
//!
//! ```sh
//! cargo run -p showcase-strategies                        # all presets
//! cargo run -p showcase-strategies -- --preset vanilla    # one preset
//! cargo run -p showcase-strategies -- --preset advanced
//! cargo run -p showcase-strategies -- --preset showcase
//! ```

use std::env;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{
    BrickGenContext, BrickGenerator, LayeredGenerator, WorldGenConfig, WorldGenPreset,
};
use atomr_worlds_voxel::BRICK_EDGE;

fn parse_preset(name: &str) -> Option<WorldGenPreset> {
    match name.to_ascii_lowercase().as_str() {
        "vanilla" => Some(WorldGenPreset::Vanilla),
        "advanced" => Some(WorldGenPreset::Advanced),
        "showcase" => Some(WorldGenPreset::Showcase),
        _ => None,
    }
}

fn print_preset(preset: WorldGenPreset) {
    let cfg = WorldGenConfig::preset(preset);
    println!("\n=== {:?} ===", preset);
    println!("config: {:#?}", cfg);

    let g = LayeredGenerator::new(cfg);
    let seed = 0xDEAD_BEEF_CAFE_F00Du64;
    let brick_coord = IVec3::new(0, 1, 0);
    let ctx = BrickGenContext::legacy(seed, brick_coord);
    let brick = g.generate_brick(&ctx);

    println!("brick {:?}  nonempty: {}", brick_coord, brick.nonempty_count);
    println!("YZ slice at x=8 (`#` = filled, `.` = empty):");
    for y in (0..BRICK_EDGE as i64).rev() {
        let row: String = (0..BRICK_EDGE as i64)
            .map(|z| {
                if brick.get(IVec3::new(8, y, z)).is_empty() {
                    '.'
                } else {
                    '#'
                }
            })
            .collect();
        println!("  y={:>2}  {}", y, row);
    }
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let selected = args
        .iter()
        .position(|a| a == "--preset")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| parse_preset(s));

    match selected {
        Some(p) => print_preset(p),
        None => {
            for p in [
                WorldGenPreset::Vanilla,
                WorldGenPreset::Advanced,
                WorldGenPreset::Showcase,
            ] {
                print_preset(p);
            }
        }
    }
}
