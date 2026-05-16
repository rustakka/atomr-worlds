//! Erosion strategies.
//!
//! The Vanilla preset's river carving lives inside [`super::vanilla::MonolithicTerrainPass`]
//! today; once the layered density/strata path is decomposed in a later
//! step, [`MacroRiverOnly`] will become the strategy-shaped wrapper that
//! reproduces it byte-equal. Until then, [`MacroRiverOnly`] is a no-op
//! stub kept for API parity, and Vanilla wires it as a no-op while river
//! carving stays inside the monolith.
//!
//! [`DropletHydraulic`] is the CPU reference impl of particle-based
//! gradient-descent hydraulic erosion. CUDA kernels are deferred to Step 11.

pub mod droplet;
pub mod macro_river;

pub use droplet::{DropletConfig, DropletHydraulic, DROPLET_DIM};
pub use macro_river::MacroRiverOnly;
