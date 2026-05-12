//! Per-strategy [`BrickGenerator`] implementations.
//!
//! Each strategy is keyed by a [`StrategyId`] constant defined in
//! [`crate::registry`]. Phase 7 ships the `terrain` body (re-export of the
//! existing `TerrainGenerator`) plus minimal `gas_giant`, `asteroid_belt`,
//! and `empty_planetoid` stubs — the API is the focus here; bodies are filled
//! in subsequent phases.
//!
//! [`StrategyId`]: crate::registry::StrategyId
//! [`BrickGenerator`]: crate::BrickGenerator

pub mod asteroid_belt;
pub mod empty_planetoid;
pub mod gas_giant;
pub mod terrain;
