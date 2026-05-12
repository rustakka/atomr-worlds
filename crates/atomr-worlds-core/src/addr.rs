//! Hierarchical addressing: [`Universe`, `Galaxy`, `Sector`, `System`, `World`].

use serde::{Deserialize, Serialize};

use crate::coord::IVec3;
use crate::dim::{DimensionId, PRIMARY};
use crate::seed::child_seed;

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
}
