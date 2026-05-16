//! Fluid strategies.
//!
//! Three CPU reference impls:
//!
//! * [`Static`] — fills voxels beneath the macro sea level with water.
//! * [`CellularAutomataFlow`] — Minecraft-style ticked downward + spread.
//! * [`LatticeBoltzmannD3Q19`] — 19-velocity LBM lattice with BGK collision.
//!
//! CUDA kernels (LBM + CA flow) land in Step 11.

pub mod ca_flow;
pub mod lbm;

pub use ca_flow::{CaFlowConfig, CellularAutomataFlow, Static, StaticConfig};
pub use lbm::{LatticeBoltzmannD3Q19, LbmConfig};
