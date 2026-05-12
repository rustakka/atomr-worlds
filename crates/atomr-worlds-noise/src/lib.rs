//! Deterministic seeded noise primitives.
//!
//! All functions take a `u64` seed plus floating-point coordinates and
//! return reproducible scalar outputs across runs and platforms (no
//! `Hasher` randomness, no float-precision-dependent reductions).
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod fbm;
pub mod gradient;
pub mod hash;
pub mod value;
pub mod worley;

pub use fbm::{fbm_gradient, fbm_value, FbmConfig};
pub use gradient::gradient_noise_3d;
pub use hash::hash3_f01;
pub use value::value_noise_3d;
pub use worley::worley_noise_3d;
