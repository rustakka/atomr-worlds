//! Hierarchical addressing: [`Universe`, `Galaxy`, `Sector`, `System`, `World`].

use serde::{Deserialize, Serialize};

use crate::coord::IVec3;
use crate::dim::{DimensionId, PRIMARY};
use crate::seed::{child_seed, HierarchicalIdentifier};

/// One tier of a [`WorldAddr`]: a coordinate plus its dimension selector.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
pub struct LevelKey {
    pub coord: IVec3,
    pub dim: DimensionId,
}

impl LevelKey {
    pub const ROOT: Self = Self { coord: IVec3::ZERO, dim: PRIMARY };

    #[inline]
    pub const fn new(coord: IVec3, dim: DimensionId) -> Self {
        Self { coord, dim }
    }

    #[inline]
    pub const fn at(coord: IVec3) -> Self {
        Self { coord, dim: PRIMARY }
    }
}

impl HierarchicalIdentifier for LevelKey {
    #[inline]
    fn dim(&self) -> DimensionId { self.dim }
    #[inline]
    fn coord(&self) -> IVec3 { self.coord }
}

/// Which tier of the hierarchy a value refers to. Ordered: deeper tiers have larger values.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
#[repr(u8)]
pub enum Level {
    Universe = 0,
    Galaxy = 1,
    Sector = 2,
    System = 3,
    World = 4,
}

impl Level {
    pub const ALL: [Level; 5] = [Level::Universe, Level::Galaxy, Level::Sector, Level::System, Level::World];

    #[inline]
    pub const fn depth(self) -> usize {
        self as usize
    }
}

/// Fixed five-tier address spanning Universe → World.
///
/// `Copy`, hashes cheaply, and is `Serialize`/`Deserialize`. Closed at this
/// phase; if we later need a variable-depth hierarchy we can wrap this in an
/// enum without breaking call sites that only address the closed five.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Default, Debug, Serialize, Deserialize)]
pub struct WorldAddr {
    pub universe: LevelKey,
    pub galaxy: LevelKey,
    pub sector: LevelKey,
    pub system: LevelKey,
    pub world: LevelKey,
}

impl WorldAddr {
    pub const ROOT: Self = Self {
        universe: LevelKey::ROOT,
        galaxy: LevelKey::ROOT,
        sector: LevelKey::ROOT,
        system: LevelKey::ROOT,
        world: LevelKey::ROOT,
    };

    #[inline]
    pub const fn level_key(&self, l: Level) -> LevelKey {
        match l {
            Level::Universe => self.universe,
            Level::Galaxy => self.galaxy,
            Level::Sector => self.sector,
            Level::System => self.system,
            Level::World => self.world,
        }
    }

    /// Return a new address truncated to the given level. Tiers below `l` are
    /// reset to [`LevelKey::ROOT`].
    pub const fn ancestor(&self, l: Level) -> WorldAddr {
        let mut out = *self;
        if (l as u8) < (Level::Galaxy as u8) {
            out.galaxy = LevelKey::ROOT;
        }
        if (l as u8) < (Level::Sector as u8) {
            out.sector = LevelKey::ROOT;
        }
        if (l as u8) < (Level::System as u8) {
            out.system = LevelKey::ROOT;
        }
        if (l as u8) < (Level::World as u8) {
            out.world = LevelKey::ROOT;
        }
        out
    }

    /// Derive the seed chain `[universe, galaxy, sector, system, world]` from
    /// the supplied root seed by repeatedly applying [`child_seed`].
    ///
    /// This is the `const fn` fast path. Each step `child_seed(parent, k.dim,
    /// k.coord)` is equivalent to `derive_child(parent, &k)` via the
    /// [`HierarchicalIdentifier`] impl on [`LevelKey`] — see the
    /// `seed_chain_matches_derive_child_walk` test. New addressable tiers
    /// follow the same rule via [`derive_child`].
    pub const fn seed_chain(&self, root: u64) -> [u64; 5] {
        let u = child_seed(root, self.universe.dim, self.universe.coord);
        let g = child_seed(u, self.galaxy.dim, self.galaxy.coord);
        let s = child_seed(g, self.sector.dim, self.sector.coord);
        let sy = child_seed(s, self.system.dim, self.system.coord);
        let w = child_seed(sy, self.world.dim, self.world.coord);
        [u, g, s, sy, w]
    }

    /// The seed at a specific level, derived from `root`.
    #[inline]
    pub const fn seed_at(&self, root: u64, l: Level) -> u64 {
        self.seed_chain(root)[l.depth()]
    }
}

use crate::vehicle::{ContainingFrame, VehicleAddr};

/// The canonical addressable thing in atomr-worlds. Every host/proto/persist
/// API takes `Address` so the substrate can host both static voxel worlds
/// and mobile vehicle voxel spaces uniformly.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug, Serialize, Deserialize)]
pub enum Address {
    World(WorldAddr),
    Vehicle(VehicleAddr),
}

impl Address {
    /// Derive the seed of this address under the supplied root seed.
    #[inline]
    pub fn seed(&self, root: u64) -> u64 {
        match self {
            Address::World(a) => a.seed_at(root, Level::World),
            Address::Vehicle(v) => v.seed(root),
        }
    }

    /// Return a [`ContainingFrame`] viewpoint for this address.
    #[inline]
    pub fn containing_frame(&self) -> ContainingFrame {
        match *self {
            Address::World(a) => ContainingFrame::World(a),
            Address::Vehicle(v) => ContainingFrame::Vehicle(v),
        }
    }
}

impl Default for Address {
    fn default() -> Self { Address::World(WorldAddr::ROOT) }
}

impl From<WorldAddr> for Address {
    #[inline]
    fn from(a: WorldAddr) -> Self { Address::World(a) }
}

impl From<VehicleAddr> for Address {
    #[inline]
    fn from(v: VehicleAddr) -> Self { Address::Vehicle(v) }
}

/// Open-ended addressing wrapper. The standard `Address` is the fixed
/// closed-five-tier shape; `AddrEither::Open` is the future-proof variant
/// that accepts an arbitrary-length [`LevelKey`] vector for variable-depth
/// hierarchies. Walking [`AddrEither::seed_chain`] applies the same
/// hierarchical-hash invariant via [`crate::seed::derive_child`] over each
/// `LevelKey` in order, length-prefixing the chain so different depths can
/// never seed-collide.
///
/// Phase 12 ships the type and its seed-chain method. Host/persist/proto
/// integration is deferred — callers continue using [`Address`] until the
/// open-ended path is exercised.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AddrEither {
    Closed(Address),
    Open(Vec<LevelKey>),
}

impl AddrEither {
    /// Compute the seed chain via repeated [`crate::seed::derive_child`]
    /// calls. The chain begins with `splitmix64(root ^ len)` so depths can't
    /// collide.
    pub fn seed_chain(&self, root: u64) -> Vec<u64> {
        match self {
            AddrEither::Closed(Address::World(a)) => a.seed_chain(root).to_vec(),
            AddrEither::Closed(Address::Vehicle(v)) => {
                // World seed-chain followed by vehicle seed as the trailing element.
                let parent_world = match v.parent {
                    crate::vehicle::ParentAddr::World(a)
                    | crate::vehicle::ParentAddr::System(a)
                    | crate::vehicle::ParentAddr::Sector(a) => a,
                };
                let mut chain = parent_world.seed_chain(root).to_vec();
                chain.push(v.seed(root));
                chain
            }
            AddrEither::Open(keys) => {
                let mut chain = Vec::with_capacity(keys.len());
                let mut parent = crate::seed::splitmix64(root ^ keys.len() as u64);
                for k in keys {
                    let s = crate::seed::derive_child(parent, k);
                    chain.push(s);
                    parent = s;
                }
                chain
            }
        }
    }
}

impl From<Address> for AddrEither {
    fn from(a: Address) -> Self { AddrEither::Closed(a) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ancestor_truncates_below_level() {
        let addr = WorldAddr {
            universe: LevelKey::at(IVec3::new(0, 0, 0)),
            galaxy: LevelKey::at(IVec3::new(1, 1, 1)),
            sector: LevelKey::at(IVec3::new(2, 2, 2)),
            system: LevelKey::at(IVec3::new(3, 3, 3)),
            world: LevelKey::at(IVec3::new(4, 4, 4)),
        };
        let a = addr.ancestor(Level::Sector);
        assert_eq!(a.galaxy.coord, IVec3::new(1, 1, 1));
        assert_eq!(a.sector.coord, IVec3::new(2, 2, 2));
        assert_eq!(a.system.coord, IVec3::ZERO);
        assert_eq!(a.world.coord, IVec3::ZERO);
    }

    #[test]
    fn seed_chain_is_deterministic() {
        let addr = WorldAddr {
            universe: LevelKey::new(IVec3::ZERO, 0),
            galaxy: LevelKey::new(IVec3::new(7, -3, 2), 0),
            sector: LevelKey::new(IVec3::new(0, 1, 0), 0),
            system: LevelKey::new(IVec3::new(-2, -2, -2), 0),
            world: LevelKey::new(IVec3::new(0, 0, 0), 1),
        };
        let a = addr.seed_chain(0xDEAD_BEEF_CAFE_F00D);
        let b = addr.seed_chain(0xDEAD_BEEF_CAFE_F00D);
        assert_eq!(a, b);
        for i in 0..a.len() {
            for j in (i + 1)..a.len() {
                assert_ne!(a[i], a[j], "level seeds must be distinct");
            }
        }
    }

    #[test]
    fn dim_affects_seed_at_each_level() {
        let mut a = WorldAddr::ROOT;
        a.galaxy = LevelKey::new(IVec3::new(1, 2, 3), 0);
        let mut b = a;
        b.galaxy.dim = 1;
        assert_ne!(a.seed_chain(42), b.seed_chain(42));
    }

    #[test]
    fn seed_chain_matches_derive_child_walk() {
        // The const-fn `seed_chain` must produce bytes identical to walking
        // each `LevelKey` through `derive_child` (the trait-based extensible
        // path). This is the load-bearing invariant — if it ever drifts,
        // every future tier using `derive_child` would seed-collide with
        // the const path.
        use crate::seed::derive_child;
        let addr = WorldAddr {
            universe: LevelKey::new(IVec3::new(0, -1, 2), 3),
            galaxy: LevelKey::new(IVec3::new(7, -3, 2), 0),
            sector: LevelKey::new(IVec3::new(0, 1, 0), 1),
            system: LevelKey::new(IVec3::new(-2, -2, -2), 0),
            world: LevelKey::new(IVec3::new(4, 5, 6), 2),
        };
        let root = 0xCAFE_BABE_F00D_DEAD;
        let chain = addr.seed_chain(root);
        let u = derive_child(root, &addr.universe);
        let g = derive_child(u, &addr.galaxy);
        let s = derive_child(g, &addr.sector);
        let sy = derive_child(s, &addr.system);
        let w = derive_child(sy, &addr.world);
        assert_eq!(chain, [u, g, s, sy, w]);
    }
}
