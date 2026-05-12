//! Per-address world-shape resolver.
//!
//! Parallels [`PolicyResolver`](crate::policy::PolicyResolver): every
//! address resolves to a [`WorldShape`]. The default — used by
//! [`LocalHostConfig::default`](crate::local::LocalHostConfig) — returns
//! [`WorldShape::default_world`] (cubic, Earth-class) for every address,
//! preserving pre-Phase-13 behavior. Worlds configured as spheres opt in
//! via [`PrefixShape::set`].
//!
//! Determinism contract: same address must always resolve to the same
//! shape within a host's lifetime.

use std::collections::HashMap;
use std::fmt::Debug;

use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_core::vehicle::VehicleAddr;

/// Resolves an [`Address`] to a [`WorldShape`]. Pure: same address →
/// same shape.
pub trait ShapeResolver: Send + Sync + Debug {
    fn resolve(&self, addr: &Address) -> WorldShape;
}

/// Default resolver — every address is a cubic Earth-class world. Used
/// by [`LocalHostConfig::default`](crate::local::LocalHostConfig).
#[derive(Debug, Default, Clone, Copy)]
pub struct DefaultShape;

impl ShapeResolver for DefaultShape {
    fn resolve(&self, _addr: &Address) -> WorldShape { WorldShape::default_world() }
}

/// Hierarchical lookup mirroring [`PrefixPolicy`](crate::policy::PrefixPolicy):
/// most-specific match wins, walking world → system → sector → galaxy →
/// universe. Vehicles inherit their parent world's shape unless an
/// explicit vehicle key is set.
#[derive(Debug, Default, Clone)]
pub struct PrefixShape {
    by_world: HashMap<WorldAddr, WorldShape>,
    by_system: HashMap<WorldAddr, WorldShape>,
    by_sector: HashMap<WorldAddr, WorldShape>,
    by_galaxy: HashMap<WorldAddr, WorldShape>,
    by_universe: HashMap<WorldAddr, WorldShape>,
    by_vehicle: HashMap<VehicleAddr, WorldShape>,
}

impl PrefixShape {
    pub fn new() -> Self { Self::default() }

    /// Insert a shape at the given hierarchy level. The address is truncated
    /// to `level`'s ancestor.
    pub fn set(&mut self, level: Level, addr: WorldAddr, s: WorldShape) {
        let key = addr.ancestor(level);
        match level {
            Level::World => self.by_world.insert(key, s),
            Level::System => self.by_system.insert(key, s),
            Level::Sector => self.by_sector.insert(key, s),
            Level::Galaxy => self.by_galaxy.insert(key, s),
            Level::Universe => self.by_universe.insert(key, s),
        };
    }

    pub fn set_vehicle(&mut self, addr: VehicleAddr, s: WorldShape) {
        self.by_vehicle.insert(addr, s);
    }

    fn resolve_world(&self, addr: WorldAddr) -> WorldShape {
        for level in [Level::World, Level::System, Level::Sector, Level::Galaxy, Level::Universe] {
            let key = addr.ancestor(level);
            let bucket = match level {
                Level::World => &self.by_world,
                Level::System => &self.by_system,
                Level::Sector => &self.by_sector,
                Level::Galaxy => &self.by_galaxy,
                Level::Universe => &self.by_universe,
            };
            if let Some(s) = bucket.get(&key) {
                return *s;
            }
        }
        WorldShape::default_world()
    }
}

impl ShapeResolver for PrefixShape {
    fn resolve(&self, addr: &Address) -> WorldShape {
        match addr {
            Address::World(a) => self.resolve_world(*a),
            Address::Vehicle(v) => {
                if let Some(s) = self.by_vehicle.get(v) {
                    return *s;
                }
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
    fn default_resolver_is_cubic() {
        let r = DefaultShape;
        let a = Address::World(world(1, 1, 1, 1));
        assert_eq!(r.resolve(&a), WorldShape::default_world());
    }

    #[test]
    fn most_specific_wins() {
        let mut r = PrefixShape::new();
        let sector = world(1, 2, 0, 0);
        let target = world(1, 2, 5, 7);
        let earth = WorldShape::Sphere { radius_m: 6.371e6 };
        let moon = WorldShape::Sphere { radius_m: 1.737e6 };
        r.set(Level::Sector, sector, earth);
        assert_eq!(r.resolve(&Address::World(target)), earth);
        // Override the specific world to moon-class.
        r.set(Level::World, target, moon);
        assert_eq!(r.resolve(&Address::World(target)), moon);
        // Sibling still inherits earth from the sector.
        let sibling = world(1, 2, 9, 9);
        assert_eq!(r.resolve(&Address::World(sibling)), earth);
    }

    #[test]
    fn unknown_address_falls_back_to_default() {
        let r = PrefixShape::new();
        let a = Address::World(world(1, 1, 1, 1));
        assert_eq!(r.resolve(&a), WorldShape::default_world());
    }
}
