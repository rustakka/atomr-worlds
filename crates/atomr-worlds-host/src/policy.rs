//! Per-address generation policy.
//!
//! A [`PolicyResolver`] maps an [`Address`] to one of three [`GenerationPolicy`]
//! variants:
//!
//! - `Seeded`  — let the [`GeneratorRegistry`] selector pick a strategy
//!               deterministically from the world seed (default).
//! - `Empty`   — short-circuit generation; reads return [`Voxel::EMPTY`]. User
//!               writes still go through the overlay/journal as normal.
//! - `Custom`  — force a specific registered strategy id.
//!
//! [`Voxel::EMPTY`]: atomr_worlds_voxel::Voxel::EMPTY
//! [`GeneratorRegistry`]: atomr_worlds_generate::registry::GeneratorRegistry
//!
//! [`PrefixPolicy`] resolves hierarchically — a policy set at sector level
//! applies to all systems and worlds inside that sector unless a more-specific
//! key (system → world) overrides it. Vehicles can be keyed independently.

use std::collections::HashMap;
use std::fmt::Debug;

use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::vehicle::VehicleAddr;

pub use atomr_worlds_generate::GenerationPolicy;

/// Resolves an [`Address`] to a [`GenerationPolicy`]. Implementors must be
/// pure: the same address must always resolve to the same policy.
pub trait PolicyResolver: Send + Sync + Debug {
    fn resolve(&self, addr: &Address) -> GenerationPolicy;
}

/// Returns [`GenerationPolicy::Seeded`] for every address — the canonical
/// default. Existing test/example call sites get this for free.
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultPolicy;

impl PolicyResolver for DefaultPolicy {
    fn resolve(&self, _addr: &Address) -> GenerationPolicy { GenerationPolicy::Seeded }
}

/// Hierarchical lookup: most-specific match wins, walking
/// world → system → sector → galaxy → universe. Vehicles fall back to their
/// parent world's policy unless an explicit vehicle key is set.
#[derive(Debug, Default, Clone)]
pub struct PrefixPolicy {
    by_world: HashMap<WorldAddr, GenerationPolicy>,
    by_system: HashMap<WorldAddr, GenerationPolicy>,
    by_sector: HashMap<WorldAddr, GenerationPolicy>,
    by_galaxy: HashMap<WorldAddr, GenerationPolicy>,
    by_universe: HashMap<WorldAddr, GenerationPolicy>,
    by_vehicle: HashMap<VehicleAddr, GenerationPolicy>,
}

impl PrefixPolicy {
    pub fn new() -> Self { Self::default() }

    /// Insert a policy at the given hierarchy level. The address is truncated
    /// to `level`'s ancestor, so callers can supply any descendant.
    pub fn set(&mut self, level: Level, addr: WorldAddr, p: GenerationPolicy) {
        let key = addr.ancestor(level);
        match level {
            Level::World => self.by_world.insert(key, p),
            Level::System => self.by_system.insert(key, p),
            Level::Sector => self.by_sector.insert(key, p),
            Level::Galaxy => self.by_galaxy.insert(key, p),
            Level::Universe => self.by_universe.insert(key, p),
        };
    }

    pub fn set_vehicle(&mut self, addr: VehicleAddr, p: GenerationPolicy) {
        self.by_vehicle.insert(addr, p);
    }

    fn resolve_world(&self, addr: WorldAddr) -> GenerationPolicy {
        // Most-specific to least-specific.
        for level in [Level::World, Level::System, Level::Sector, Level::Galaxy, Level::Universe] {
            let key = addr.ancestor(level);
            let bucket = match level {
                Level::World => &self.by_world,
                Level::System => &self.by_system,
                Level::Sector => &self.by_sector,
                Level::Galaxy => &self.by_galaxy,
                Level::Universe => &self.by_universe,
            };
            if let Some(p) = bucket.get(&key) {
                return *p;
            }
        }
        GenerationPolicy::Seeded
    }
}

impl PolicyResolver for PrefixPolicy {
    fn resolve(&self, addr: &Address) -> GenerationPolicy {
        match addr {
            Address::World(a) => self.resolve_world(*a),
            Address::Vehicle(v) => {
                if let Some(p) = self.by_vehicle.get(v) {
                    return *p;
                }
                // Fall back to the parent world's policy.
                match v.parent {
                    atomr_worlds_core::vehicle::ParentAddr::World(a)
                    | atomr_worlds_core::vehicle::ParentAddr::System(a)
                    | atomr_worlds_core::vehicle::ParentAddr::Sector(a) => self.resolve_world(a),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::LevelKey;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_core::vehicle::{ParentAddr, VehicleSlot};

    fn world(g: i64, s: i64, sy: i64, w: i64) -> WorldAddr {
        WorldAddr {
            universe: LevelKey::ROOT,
            galaxy: LevelKey::at(IVec3::new(g, 0, 0)),
            sector: LevelKey::at(IVec3::new(s, 0, 0)),
            system: LevelKey::at(IVec3::new(sy, 0, 0)),
            world: LevelKey::at(IVec3::new(w, 0, 0)),
        }
    }

    #[test]
    fn default_policy_is_seeded() {
        let p = DefaultPolicy;
        let a = Address::World(world(1, 1, 1, 1));
        assert_eq!(p.resolve(&a), GenerationPolicy::Seeded);
    }

    #[test]
    fn most_specific_wins() {
        let mut p = PrefixPolicy::new();
        let sector = world(1, 2, 0, 0);
        let target = world(1, 2, 5, 7);
        p.set(Level::Sector, sector, GenerationPolicy::Empty);
        assert_eq!(p.resolve(&Address::World(target)), GenerationPolicy::Empty);
        // Now override the specific world.
        p.set(Level::World, target, GenerationPolicy::Seeded);
        assert_eq!(p.resolve(&Address::World(target)), GenerationPolicy::Seeded);
        // A different world under the same sector still inherits Empty.
        let sibling = world(1, 2, 9, 9);
        assert_eq!(p.resolve(&Address::World(sibling)), GenerationPolicy::Empty);
    }

    #[test]
    fn vehicle_inherits_parent_world_policy() {
        let mut p = PrefixPolicy::new();
        let parent = world(0, 0, 0, 5);
        p.set(Level::World, parent, GenerationPolicy::Empty);
        let va = VehicleAddr::new(ParentAddr::World(parent), VehicleSlot::new(1, 0));
        assert_eq!(p.resolve(&Address::Vehicle(va)), GenerationPolicy::Empty);

        // Explicit vehicle override beats parent.
        p.set_vehicle(va, GenerationPolicy::Seeded);
        assert_eq!(p.resolve(&Address::Vehicle(va)), GenerationPolicy::Seeded);
    }
}
