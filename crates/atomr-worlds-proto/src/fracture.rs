//! Deterministic fracture-event protocol types.
//!
//! These are the wire types for destructible-world physics (Rec 2 local
//! destruction; Rec 4 multiplayer sync). The design goal is that a *fracture
//! decision* replays byte-identically on every client: the server runs the
//! connectivity flood-fill on its authoritative voxel state and emits an
//! ordered list of [`FractureCommand`]s, which every client re-applies to reach
//! identical geometry. The *debris physics* that follows is float-based and
//! diverges across machines, so it is synced separately as
//! [`DebrisStateDelta`] snapshots rather than replayed.
//!
//! # Determinism: forces are fixed-point
//!
//! The trigger force is carried as **integer milli-newtons** ([`Force`]), not
//! `f32`, so the fracture decision does not depend on platform-specific
//! floating-point rounding. Convert from `f32` newtons at the call site with
//! [`Force::from_newtons`] *before* building the request.
//!
//! # Wiring status (Phase 1 foundations)
//!
//! These types are defined and serde-tested here, but are **not yet** added as
//! variants of [`WorldRequest`]/[`WorldEvent`] — that wiring (and the actor-side
//! handling) lands with the Rec 2 / Rec 4 phases. Adding them as *appended*
//! enum variants later is bincode-safe; adding fields to existing persisted
//! structs is not (see the plan's schema-evolution note).
//!
//! [`WorldRequest`]: crate::WorldRequest
//! [`WorldEvent`]: crate::WorldEvent

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::Voxel;
use serde::{Deserialize, Serialize};

/// Fixed-point force vector in **milli-newtons** per axis. Integer-valued so
/// fracture decisions are reproducible across platforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Force {
    /// Force per axis, in milli-newtons (1 N = 1000 units).
    pub milli_n: IVec3,
}

impl Force {
    /// Milli-newtons per newton — the fixed-point scale.
    pub const SCALE: f64 = 1000.0;

    pub const ZERO: Self = Self { milli_n: IVec3::ZERO };

    #[inline]
    pub const fn from_milli_n(milli_n: IVec3) -> Self {
        Self { milli_n }
    }

    /// Convert from `f32` newtons using deterministic round-half-away-from-zero.
    /// Do this conversion at the call site, before constructing the request, so
    /// the integer value (not the float) is what crosses the wire.
    #[inline]
    pub fn from_newtons(newtons: [f32; 3]) -> Self {
        let q = |n: f32| (n as f64 * Self::SCALE).round() as i64;
        Self {
            milli_n: IVec3::new(q(newtons[0]), q(newtons[1]), q(newtons[2])),
        }
    }

    /// Recover the approximate `f32`-newton vector (lossy; for display / the
    /// non-authoritative physics push only).
    #[inline]
    pub fn to_newtons(self) -> [f32; 3] {
        let f = |v: i64| (v as f64 / Self::SCALE) as f32;
        [f(self.milli_n.x), f(self.milli_n.y), f(self.milli_n.z)]
    }
}

/// One step of a fracture's deterministic command sequence. Applied in order by
/// every client to reach identical geometry.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum FractureCommand {
    /// Set a single voxel (the workhorse: carving a voxel sets `after =
    /// Voxel::EMPTY`). `before` is recorded for journal-replay symmetry.
    SetVoxel { pos: IVec3, before: Voxel, after: Voxel },
    /// A connected island detached and becomes a dynamic body. `voxels` are the
    /// island's world voxel coordinates; `anchor` is a stable reference point
    /// (e.g. the island AABB min) for the spawned body's frame.
    SpawnDebris { id: u32, voxels: Vec<IVec3>, anchor: IVec3 },
    /// A previously-created constraint/joint between bodies is released.
    DisconnectJoint { id: u64 },
}

/// A client (or AI agent) asks the authoritative world actor to evaluate a
/// fracture at `impact_pos` with `force`. The actor validates material yield,
/// geographic bounds, and rate limits, then runs the connectivity check.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FractureRequest {
    pub addr: Address,
    pub impact_pos: IVec3,
    pub force: Force,
    /// Material id at the impact point (used for the yield check).
    pub material_id: u16,
}

/// The authoritative result of a fracture: the ordered command sequence plus
/// the inclusive journal sequence-number range the commands were written at, so
/// late joiners can replay deterministically from a known point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct FractureApplied {
    pub addr: Address,
    pub commands: Vec<FractureCommand>,
    /// Inclusive `(first_seq, last_seq)` journal range for these commands.
    pub seq_range: (u64, u64),
}

/// A periodic, lossy snapshot of one active debris body's rigid-body state,
/// broadcast on the unreliable channel for client-side interpolation. Floats
/// are fine here: this is *not* replayed, it's interpolated.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct DebrisStateDelta {
    pub id: u32,
    /// Authoritative server tick this snapshot was sampled at.
    pub tick: u64,
    pub pos: [f32; 3],
    pub vel: [f32; 3],
    /// Orientation quaternion `(x, y, z, w)`.
    pub orient: [f32; 4],
    pub ang_vel: [f32; 3],
    /// `true` once the body has come to rest (velocity fields may be stale).
    pub sleeping: bool,
}

/// Sent to a writer whose voxel edit lost a concurrent (CRDT) merge, so it can
/// roll its local optimistic preview back to `current`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct WriteRejected {
    pub addr: Address,
    pub pos: IVec3,
    pub current: Voxel,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wire::{decode, encode};

    #[test]
    fn force_fixed_point_is_deterministic_and_round_trips() {
        let f = Force::from_newtons([1.5, -2.25, 0.001]);
        assert_eq!(f.milli_n, IVec3::new(1500, -2250, 1));
        // Same input → same fixed-point value, every time.
        assert_eq!(f, Force::from_newtons([1.5, -2.25, 0.001]));
        // Approximate recovery.
        let n = f.to_newtons();
        assert!((n[0] - 1.5).abs() < 1e-3);
    }

    #[test]
    fn fracture_command_bincode_round_trip() {
        let cmd = FractureCommand::SetVoxel {
            pos: IVec3::new(3, -7, 12),
            before: Voxel::new(1),
            after: Voxel::EMPTY,
        };
        let bytes = encode(&cmd).unwrap();
        let back: FractureCommand = decode(&bytes).unwrap();
        assert_eq!(cmd, back);
    }

    #[test]
    fn fracture_applied_bincode_round_trip() {
        let applied = FractureApplied {
            addr: Address::default(),
            commands: vec![
                FractureCommand::SetVoxel {
                    pos: IVec3::new(0, 0, 0),
                    before: Voxel::new(2),
                    after: Voxel::EMPTY,
                },
                FractureCommand::SpawnDebris {
                    id: 7,
                    voxels: vec![IVec3::new(0, 0, 0), IVec3::new(1, 0, 0)],
                    anchor: IVec3::ZERO,
                },
            ],
            seq_range: (10, 11),
        };
        let bytes = encode(&applied).unwrap();
        let back: FractureApplied = decode(&bytes).unwrap();
        assert_eq!(applied, back);
    }

    #[test]
    fn debris_delta_bincode_round_trip() {
        let d = DebrisStateDelta {
            id: 1,
            tick: 42,
            pos: [1.0, 2.0, 3.0],
            vel: [0.0, -9.8, 0.0],
            orient: [0.0, 0.0, 0.0, 1.0],
            ang_vel: [0.1, 0.0, 0.0],
            sleeping: false,
        };
        let bytes = encode(&d).unwrap();
        let back: DebrisStateDelta = decode(&bytes).unwrap();
        assert_eq!(d, back);
    }
}
