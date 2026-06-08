//! Engine-agnostic voxel-physics primitives for atomr-worlds.
//!
//! This crate is the **dependency-free core** of the physics subsystem (Rec 2 of
//! the *Advanced Voxel Architectures* plan): pure, deterministic logic with no
//! Bevy, rapier, or async runtime. The Bevy/rapier integration (collider
//! generation, the solver tick, debris entity lifecycle) lives in the client
//! crate and builds on these pieces.
//!
//! What's here today (Phase 1 foundations):
//!
//! - [`flood_fill`] — deterministic 6-connected structural connectivity used to
//!   detect floating islands when a structure is damaged.
//! - [`box_merge`] — greedy 3D box-merge that coalesces a region's solid voxels
//!   into a small set of axis-aligned boxes (the collision analogue of greedy
//!   meshing), feeding the client's rapier compound-collider builder.
//! - [`inertia`] — center of mass + inertia tensor from per-voxel density.
//! - [`debris`] — a [`debris::DebrisBody`] extracted from a voxel island, with
//!   its mass properties and rigid-body state.
//! - [`math`] — the small `f64` linear algebra the inertia solver needs.
//!
//! # Determinism boundary
//!
//! Everything here is a pure function of its inputs and never mutates world
//! voxel state. Physics is a client-side, non-deterministic *integration*
//! concern; these foundations only ever *read* voxels and *derive* quantities.
//! Removing voxels for a detached island is the caller's job, done through a
//! journaled write on the world actor — so `GetBrick` output is unaffected and
//! the byte-determinism contract holds.
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod box_merge;
pub mod debris;
pub mod flood_fill;
pub mod inertia;
pub mod math;

pub use box_merge::{greedy_boxes, Cuboid};
pub use debris::DebrisBody;
pub use flood_fill::{connected_components, Components};
pub use inertia::{mass_properties, MassProperties};
pub use math::Mat3;
