//! Client-side rapier physics (Rec 2 of the *Advanced Voxel Architectures*
//! plan).
//!
//! This module is the Bevy/rapier integration layer on top of the
//! engine-agnostic [`atomr_worlds_physics`] core. It is the **only** place in
//! the workspace that depends on `bevy_rapier3d`, and the whole module is gated
//! behind the client crate's `physics` feature so the determinism-tested crates
//! never gain a rapier edge.
//!
//! # Determinism boundary
//!
//! Physics here is client-side, non-deterministic, and *ephemeral*. The
//! canonical voxel grid is only ever mutated through the host
//! (`WriteVoxel`/`WriteRegion`, journaled); colliders and debris are derived,
//! never authoritative, and never flow back into `GetBrick` or the journal.
//!
//! # What's wired
//!
//! - A [`PhysicsConfig`] strategy resource mirroring the render crate's
//!   `RenderConfig` spine (a pluggable [`ColliderStrategy`], gravity, a runtime
//!   `enabled` toggle).
//! - Static leaf-LOD terrain colliders attached to LOD-0 brick entities
//!   ([`plugin::attach_brick_colliders`]).
//! - Carve → flood-fill → falling debris ([`debris`], [`fracture`]).

mod collider_gen;
mod config;
mod debris;
mod defaults;
mod fracture;
mod plugin;
mod registry;
mod strategy;

pub use config::PhysicsConfig;
pub use plugin::PhysicsPlugin;
pub use registry::apply_strategy_by_name;
