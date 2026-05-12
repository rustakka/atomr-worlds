//! Cross-backend determinism gate for Phase 5 full.
//!
//! Requires a real CUDA device. Run with:
//!
//! ```sh
//! cargo test -p atomr-worlds-accel --features cuda --test cuda_determinism -- --ignored
//! ```
//!
//! The `#[ignore]` annotation keeps these tests out of the default suite so
//! `cargo test` on a CUDA-less host still passes.

#![cfg(feature = "cuda")]

use atomr_worlds_accel::cuda::CudaAccelerator;
use atomr_worlds_accel::{Accelerator, CpuAccelerator};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{TerrainConfig, TerrainGenerator};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

fn coords() -> Vec<IVec3> {
    vec![
        IVec3::new(0, 0, 0),
        IVec3::new(0, -1, 0),
        IVec3::new(1, 0, 1),
        IVec3::new(-2, 0, 3),
        IVec3::new(0, -3, 0),
    ]
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a CUDA device; run with --ignored"]
async fn cuda_matches_cpu_byte_for_byte() {
    let cfg = TerrainConfig::default();
    let cpu = CpuAccelerator::new(TerrainGenerator::new(cfg));
    let gpu = CudaAccelerator::new(0, cfg).await.expect("cuda init");

    let cs = coords();
    let cpu_bricks: Vec<_> = cs.iter().map(|c| cpu.fill_brick(SEED, *c)).collect();
    let gpu_bricks = gpu.fill_bricks_batch_async(SEED, &cs).await.expect("gpu fill");

    assert_eq!(cpu_bricks.len(), gpu_bricks.len());
    for (i, (a, b)) in cpu_bricks.iter().zip(gpu_bricks.iter()).enumerate() {
        assert_eq!(a.to_bytes(), b.to_bytes(), "brick {i} ({:?}) diverged", cs[i]);
    }
}

#[tokio::test(flavor = "multi_thread")]
#[ignore = "requires a CUDA device; run with --ignored"]
async fn cuda_is_idempotent_across_runs() {
    let cfg = TerrainConfig::default();
    let gpu = CudaAccelerator::new(0, cfg).await.expect("cuda init");
    let cs = coords();
    let a = gpu.fill_bricks_batch_async(SEED, &cs).await.expect("gpu a");
    let b = gpu.fill_bricks_batch_async(SEED, &cs).await.expect("gpu b");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert_eq!(x.to_bytes(), y.to_bytes(), "brick {i} non-deterministic across launches");
    }
}
