//! Vehicle / entity-space addressing.
//!
//! A vehicle is an addressable, mobile container with its own local voxel
//! space (a small octree) and an [`AffineFrame`] giving its pose in a parent
//! frame — a world, a system, or interstellar (sector) space.
//!
//! The seed for a vehicle is derived through the same rule as any other tier:
//!
//! ```text
//! vehicle_seed = derive_child(parent_seed, &VehicleSlot { slot_id, dim })
//! ```
//!
//! where `VehicleSlot` packs its 64-bit `slot_id` into an [`IVec3`] (low/high
//! 32-bit halves into `x`/`y`; `z = 0`). This keeps the hierarchical-hash
//! invariant uniform — no new hash function appears.

use serde::{Deserialize, Serialize};

use crate::addr::{Level, WorldAddr};
use crate::coord::{DVec3, IVec3, Quat};
use crate::dim::{DimensionId, PRIMARY};
use crate::seed::HierarchicalIdentifier;

/// Stable per-author identifier of a vehicle inside its parent frame.
pub type VehicleSlotId = u64;

/// Identifier tier for a vehicle: a [`VehicleSlotId`] plus a dimension.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default, Serialize, Deserialize)]
pub struct VehicleSlot {
    pub slot_id: VehicleSlotId,
    pub dim: DimensionId,
}

impl VehicleSlot {
    pub const ROOT: Self = Self { slot_id: 0, dim: PRIMARY };

    #[inline]
    pub const fn new(slot_id: VehicleSlotId, dim: DimensionId) -> Self {
        Self { slot_id, dim }
    }
}

impl HierarchicalIdentifier for VehicleSlot {
    #[inline]
    fn dim(&self) -> DimensionId { self.dim }
    #[inline]
    fn coord(&self) -> IVec3 {
        // Pack u64 slot_id into IVec3 (low/high 32 bits → x/y, z = 0). The
        // encoding is part of the persistent contract: never change.
        let lo = (self.slot_id & 0xFFFF_FFFF) as i64;
        let hi = ((self.slot_id >> 32) & 0xFFFF_FFFF) as i64;
        IVec3::new(lo, hi, 0)
    }
}

/// Which tier anchors this vehicle. Determines the seed parent and the
/// frame's reference origin (the center of the parent body).
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum ParentAddr {
    /// Inside a planet/world (frame origin = world center).
    World(WorldAddr),
    /// Interplanetary, anchored to a system (frame origin = system barycenter).
    System(WorldAddr),
    /// Interstellar, anchored to a sector (frame origin = sector center).
    Sector(WorldAddr),
}

impl ParentAddr {
    /// Seed of the parent frame in the seed chain rooted at `root`.
    #[inline]
    pub const fn parent_seed(&self, root: u64) -> u64 {
        match self {
            ParentAddr::World(a) => a.seed_at(root, Level::World),
            ParentAddr::System(a) => a.seed_at(root, Level::System),
            ParentAddr::Sector(a) => a.seed_at(root, Level::Sector),
        }
    }

    /// Which hierarchy level this parent sits at.
    #[inline]
    pub const fn level(&self) -> Level {
        match self {
            ParentAddr::World(_) => Level::World,
            ParentAddr::System(_) => Level::System,
            ParentAddr::Sector(_) => Level::Sector,
        }
    }
}

impl Default for ParentAddr {
    fn default() -> Self { ParentAddr::World(WorldAddr::ROOT) }
}

/// Full address of a vehicle: its parent frame plus its slot inside it.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Default, Serialize, Deserialize)]
pub struct VehicleAddr {
    pub parent: ParentAddr,
    pub slot: VehicleSlot,
}

impl VehicleAddr {
    #[inline]
    pub const fn new(parent: ParentAddr, slot: VehicleSlot) -> Self {
        Self { parent, slot }
    }

    /// Derive the vehicle's seed from a root seed via the hierarchical-hash
    /// invariant: `derive_child(parent_seed, &slot)`.
    #[inline]
    pub fn seed(&self, root: u64) -> u64 {
        crate::seed::derive_child(self.parent.parent_seed(root), &self.slot)
    }
}

/// A mutable pose: position (meters in parent frame), orientation, and the
/// parent reference frame.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub struct AffineFrame {
    pub position: DVec3,
    pub orientation: Quat,
    pub parent: ParentAddr,
}

impl AffineFrame {
    #[inline]
    pub const fn at_origin(parent: ParentAddr) -> Self {
        Self { position: DVec3::ZERO, orientation: Quat::IDENTITY, parent }
    }
}

impl Default for AffineFrame {
    fn default() -> Self { Self::at_origin(ParentAddr::default()) }
}

/// What the observer is currently "inside" — used by atmosphere streaming
/// and rendering to choose which tier's voxel space to stream.
#[derive(Copy, Clone, PartialEq, Debug, Serialize, Deserialize)]
pub enum ContainingFrame {
    World(WorldAddr),
    Vehicle(VehicleAddr),
    /// Not inside any solid voxel space; observer is in inter-body free space
    /// anchored to a parent tier (sector/system).
    Free(ParentAddr),
}

impl Default for ContainingFrame {
    fn default() -> Self { ContainingFrame::World(WorldAddr::ROOT) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::addr::LevelKey;

    #[test]
    fn vehicle_slot_identifier_round_trips() {
        let s = VehicleSlot::new(0xDEAD_BEEF_CAFE_F00D, 3);
        assert_eq!(s.dim(), 3);
        // Round-trip the slot_id from the IVec3 packing.
        let c = s.coord();
        let lo = (c.x as u64) & 0xFFFF_FFFF;
        let hi = (c.y as u64) & 0xFFFF_FFFF;
        let reconstructed = (hi << 32) | lo;
        assert_eq!(reconstructed, 0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(c.z, 0);
    }

    #[test]
    fn vehicle_seed_is_deterministic_and_parent_sensitive() {
        let parent_a = ParentAddr::World(WorldAddr {
            universe: LevelKey::ROOT,
            galaxy: LevelKey::at(IVec3::new(1, 2, 3)),
            sector: LevelKey::at(IVec3::new(0, 0, 0)),
            system: LevelKey::at(IVec3::new(0, 0, 0)),
            world: LevelKey::at(IVec3::new(5, 5, 5)),
        });
        let parent_b = ParentAddr::World(WorldAddr {
            universe: LevelKey::ROOT,
            galaxy: LevelKey::at(IVec3::new(1, 2, 3)),
            sector: LevelKey::at(IVec3::new(0, 0, 0)),
            system: LevelKey::at(IVec3::new(0, 0, 0)),
            world: LevelKey::at(IVec3::new(5, 5, 6)), // different world
        });
        let slot = VehicleSlot::new(42, 0);
        let va = VehicleAddr::new(parent_a, slot);
        let vb = VehicleAddr::new(parent_b, slot);
        assert_eq!(va.seed(0xCAFE), va.seed(0xCAFE));
        assert_ne!(va.seed(0xCAFE), vb.seed(0xCAFE));
    }

    #[test]
    fn vehicle_slot_id_differentiates() {
        let parent = ParentAddr::Sector(WorldAddr::ROOT);
        let a = VehicleAddr::new(parent, VehicleSlot::new(1, 0));
        let b = VehicleAddr::new(parent, VehicleSlot::new(2, 0));
        assert_ne!(a.seed(0xABCD), b.seed(0xABCD));
    }
}
