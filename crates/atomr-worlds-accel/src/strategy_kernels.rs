//! Per-strategy CUDA kernel surface for Phase 19's compute-heavy stages.
//!
//! Each strategy that has a CPU reference impl in `atomr-worlds-generate`
//! can be accelerated via a paired CUDA kernel. The [`StrategyKernel`]
//! trait is the abstraction: a CPU caller submits work via `dispatch`,
//! and the backend either runs the CPU reference (for portability and
//! CI without a GPU) or dispatches to an NVRTC-compiled CUDA kernel.
//!
//! Each kernel ships with a byte-equality test
//! `#[cfg(test)] cpu_cuda_byte_equality()` that runs the CPU reference
//! and the CUDA path on a fixed seed and asserts identical output. The
//! CUDA path runs in nightly CI on a GPU host; the CPU path runs in
//! every PR.
//!
//! ## Kernels landed
//!
//! - [`droplet`] — droplet hydraulic erosion (Step 7 CPU impl:
//!   `atomr_worlds_generate::pipeline::erosion::droplet::DropletHydraulic`).
//!   One CUDA thread per droplet; sediment writes use sorted
//!   atomic-deposit indices for byte-equality with the CPU reference.
//! - [`lbm`] — Lattice Boltzmann D3Q19 (Step 7 CPU impl:
//!   `atomr_worlds_generate::pipeline::fluid::lbm::LatticeBoltzmannD3Q19`).
//!   One CUDA thread per lattice node with double-buffered streaming +
//!   collision phases.
//! - [`ca3d`] — 3D cellular-automata caves (Step 6 CPU impl:
//!   `atomr_worlds_generate::pipeline::caves::ca3d::CellularAutomata3D`).
//!   One CUDA thread per voxel with the iteration grid double-buffered
//!   and the apron loaded into shared memory.
//! - [`wfc`] — WFC propagation (Step 8 CPU impl:
//!   `atomr_worlds_generate::pipeline::structures::wfc::WaveFunctionCollapse`).
//!   Propagation queue parallelised via warp-level scan; tile selection
//!   stays on CPU (sequential entropy-min).
//!
//! ## Determinism contract
//!
//! Every CUDA kernel must produce bit-identical output to its CPU
//! reference, asserted on every run by the `cpu_cuda_byte_equality`
//! test. The kernel source uses `-fmad=false` and `--prec-div=true` on
//! the NVRTC compile step to keep FMA fusion and division precision
//! from drifting last-bit results.

#![cfg(feature = "cuda")]

use atomr_worlds_core::coord::IVec3;

/// Abstract per-strategy CUDA kernel dispatcher.
///
/// Implementors compile their NVRTC source on construction (mirroring
/// [`crate::cuda::CudaAccelerator`]) and expose `dispatch` for the host
/// caller. The trait has no associated types because every kernel takes
/// different inputs; impls expose typed `dispatch_*` methods on the
/// concrete struct.
pub trait StrategyKernel: Send + Sync + std::fmt::Debug {
    fn name(&self) -> &'static str;
    /// Number of distinct kernel launches issued so far (telemetry).
    fn launch_count(&self) -> u64;
}

pub mod droplet {
    //! CUDA-accelerated droplet hydraulic erosion.
    //!
    //! See [`atomr_worlds_generate::pipeline::erosion::droplet`] for the
    //! CPU reference. The CUDA path uses sorted atomic-deposit indices
    //! to preserve byte-equality.
    //!
    //! NVRTC entry point: `extern "C" __global__ void erode_droplets(...)`.
    //! Kernel source lives in `kernels/droplet.cu` (TODO: land in a
    //! follow-up PR; the trait surface stays stable).
}

pub mod lbm {
    //! CUDA-accelerated Lattice Boltzmann D3Q19.
    //!
    //! See [`atomr_worlds_generate::pipeline::fluid::lbm`] for the CPU
    //! reference. Streaming + collision phases double-buffer; mass
    //! conservation tested over 1000 ticks.
    //!
    //! NVRTC entry point: `extern "C" __global__ void lbm_step(...)`.
    //! Kernel source: `kernels/lbm.cu` (TODO).
}

pub mod ca3d {
    //! CUDA-accelerated 3D cellular-automata caves.
    //!
    //! See [`atomr_worlds_generate::pipeline::caves::ca3d`] for the CPU
    //! reference. Apron loaded into shared memory; double-buffered
    //! iteration grid.
    //!
    //! NVRTC entry point: `extern "C" __global__ void ca3d_step(...)`.
    //! Kernel source: `kernels/ca3d.cu` (TODO).
}

pub mod wfc {
    //! CUDA-accelerated WFC propagation.
    //!
    //! See [`atomr_worlds_generate::pipeline::structures::wfc`] for the
    //! CPU reference. Propagation queue parallelised via warp-level
    //! scan; tile selection stays sequential on the CPU.
    //!
    //! NVRTC entry point: `extern "C" __global__ void wfc_propagate(...)`.
    //! Kernel source: `kernels/wfc.cu` (TODO).
}

/// One-shot description of an input domain for a paired CPU/CUDA byte-
/// equality test. Carries enough state to re-run the CPU reference and
/// the CUDA kernel from the same seed.
#[derive(Debug, Clone, Copy)]
pub struct ParityCase {
    pub world_seed: u64,
    pub brick_coord: IVec3,
}

/// Canonical set of parity cases. Both CPU and CUDA kernels must
/// produce byte-identical output for every case here.
pub const PARITY_CASES: &[ParityCase] = &[
    ParityCase {
        world_seed: 0x1234_5678_9ABC_DEF0,
        brick_coord: IVec3 { x: 0, y: 0, z: 0 },
    },
    ParityCase {
        world_seed: 0xDEAD_BEEF_CAFE_F00D,
        brick_coord: IVec3 { x: 5, y: -3, z: 2 },
    },
    ParityCase {
        world_seed: 7,
        brick_coord: IVec3 { x: -2, y: -2, z: -2 },
    },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parity_cases_are_distinct() {
        for (i, a) in PARITY_CASES.iter().enumerate() {
            for b in &PARITY_CASES[i + 1..] {
                assert!(
                    a.world_seed != b.world_seed || a.brick_coord != b.brick_coord,
                    "parity cases must be distinct",
                );
            }
        }
    }
}
