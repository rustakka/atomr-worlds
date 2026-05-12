//! proptest strategies and determinism helpers shared by phase-0 tests.
#![forbid(unsafe_code)]

pub mod strategies;

pub use strategies::{arb_brick, arb_ivec3, arb_level_key, arb_lod, arb_voxel, arb_world_addr};
