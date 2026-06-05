//! Wire types for atomr-worlds.
//!
//! These are the messages a client (renderer, agent, CLI) exchanges with a
//! [`crate::WorldHost`][crate-host]. The format is `bincode` 2 — fast, compact,
//! and matches the serializer atomr's remote layer already uses, so a process
//! that bridges both stays on a single codec.
//!
//! [crate-host]: ../atomr_worlds_host/index.html
#![forbid(unsafe_code)]
#![warn(missing_debug_implementations)]

pub mod aabb;
pub mod envelope;
pub mod error;
pub mod fracture;
pub mod messages;
pub mod streaming;
pub mod wire;

pub use aabb::AABB;
pub use envelope::Envelope;
pub use error::ProtoError;
pub use fracture::{
    DebrisStateDelta, Force, FractureApplied, FractureCommand, FractureRequest, WriteRejected,
};
pub use messages::{Portal, WorldEvent, WorldRequest};
pub use streaming::{RingPlan, StreamingPolicy};
pub use wire::{decode, encode};
