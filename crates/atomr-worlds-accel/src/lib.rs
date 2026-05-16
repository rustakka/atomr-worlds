//! Brick-generation acceleration surface.
//!
//! The trait is GPU-friendly: a kernel can be dispatched per-brick with
//! `(seed, brick_coord)` as input and an output buffer to fill. The CPU
//! implementation defers to [`atomr_worlds_generate::BrickGenerator`]; a
//! CUDA implementation backed by `atomr-accel-cuda`'s NVRTC dispatch lives
//! in [`cuda`] behind the `cuda` feature flag.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

#[cfg(feature = "cuda")]
pub mod cuda;

#[cfg(feature = "cuda")]
pub mod strategy_kernels;

#[cfg(feature = "cuda")]
pub use cuda::{CudaAccelerator, CudaError};

#[cfg(feature = "cuda")]
pub use strategy_kernels::{ParityCase, StrategyKernel, PARITY_CASES};

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::BrickGenerator;
use atomr_worlds_voxel::Brick;

/// Acceleration backend. Implementations may be CPU, GPU, distributed, etc.
pub trait Accelerator: Send + Sync {
    /// Backend identifier for logging / telemetry.
    fn backend(&self) -> &'static str;

    /// Synchronously fill a brick for the given `(world_seed, brick_coord)`.
    ///
    /// GPU implementations will batch via `fill_bricks_batch`; this trivial
    /// single-brick path remains for incremental requests.
    fn fill_brick(&self, world_seed: u64, brick_coord: IVec3) -> Brick;

    /// Fill many bricks at once. Default impl loops over `fill_brick`; GPU
    /// backends will override with a batched kernel dispatch.
    fn fill_bricks_batch(&self, world_seed: u64, brick_coords: &[IVec3]) -> Vec<Brick> {
        brick_coords.iter().map(|c| self.fill_brick(world_seed, *c)).collect()
    }
}

/// CPU backend: delegates to a `BrickGenerator`.
#[derive(Debug)]
pub struct CpuAccelerator<G> {
    pub generator: G,
}

impl<G> CpuAccelerator<G> {
    pub fn new(generator: G) -> Self {
        Self { generator }
    }
}

impl<G: BrickGenerator + Send + Sync> Accelerator for CpuAccelerator<G> {
    fn backend(&self) -> &'static str {
        "cpu"
    }
    fn fill_brick(&self, world_seed: u64, brick_coord: IVec3) -> Brick {
        // Legacy two-arg path — preserved for CUDA byte-equality across
        // CPU and GPU. Macro-state-aware generation only flows through
        // `WorldActor::ensure_brick`, never through `Accelerator`.
        self.generator.generate_brick_legacy(world_seed, brick_coord)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_generate::{TerrainConfig, TerrainGenerator};

    #[test]
    fn cpu_backend_matches_direct_generator() {
        let g = TerrainGenerator::new(TerrainConfig::default());
        let accel = CpuAccelerator::new(g.clone());
        let p = IVec3::new(0, -2, 0);
        let direct = g.generate_brick_legacy(42, p);
        let routed = accel.fill_brick(42, p);
        assert_eq!(direct.nonempty_count, routed.nonempty_count);
        for i in 0..16i64 {
            for j in 0..16i64 {
                for k in 0..16i64 {
                    assert_eq!(direct.get(IVec3::new(i, j, k)), routed.get(IVec3::new(i, j, k)));
                }
            }
        }
    }

    #[test]
    fn batch_default_impl_works() {
        let g = TerrainGenerator::new(TerrainConfig::default());
        let accel = CpuAccelerator::new(g);
        let coords = [IVec3::new(0, -1, 0), IVec3::new(0, 0, 0), IVec3::new(0, 1, 0)];
        let out = accel.fill_bricks_batch(42, &coords);
        assert_eq!(out.len(), 3);
    }
}
