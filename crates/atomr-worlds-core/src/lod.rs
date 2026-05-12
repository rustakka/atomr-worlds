//! Metric level-of-detail.
//!
//! Each spatial object has a [`MetricScale`] giving its root cube edge in
//! meters and the maximum octree depth. A [`Lod`] selects a depth in the
//! pyramid; `meters_per_voxel(lod) = root_size_m / 2^lod.depth`.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::addr::{Address, Level, LevelKey, WorldAddr};

#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug, Serialize, Deserialize)]
pub struct Lod {
    pub depth: u8,
}

impl Lod {
    pub const ROOT: Self = Self { depth: 0 };

    #[inline]
    pub const fn new(depth: u8) -> Self {
        Self { depth }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MetricScale {
    pub root_size_m: f64,
    pub max_depth: u8,
}

impl MetricScale {
    /// Observable-universe scale (≈8.8 × 10^26 m diameter; root cube edge 1e27).
    pub const DEFAULT_UNIVERSE: Self = Self { root_size_m: 1.0e27, max_depth: 64 };
    /// Milky-Way-class galaxy (~100 kly diameter; root cube edge 1e21 m).
    pub const DEFAULT_GALAXY: Self = Self { root_size_m: 1.0e21, max_depth: 56 };
    /// Generic sector, configurable; ~30 ly (root cube edge 1e18 m).
    pub const DEFAULT_SECTOR: Self = Self { root_size_m: 1.0e18, max_depth: 48 };
    /// Stellar-system scale (~100 AU; root cube edge 1e13 m).
    pub const DEFAULT_SYSTEM: Self = Self { root_size_m: 1.0e13, max_depth: 40 };
    /// Earth-class world (root cube edge 1e7 m; ~10 000 km).
    pub const DEFAULT_WORLD: Self = Self { root_size_m: 1.0e7, max_depth: 24 };

    #[inline]
    pub fn meters_per_voxel(&self, lod: Lod) -> f64 {
        // f64 exp2 is overflow-free across the full u8 depth range (unlike
        // `1u64 << depth`, which UB's at depth == 64).
        self.root_size_m / (lod.depth as f64).exp2()
    }

    /// Edge of a voxel at the leaf depth.
    #[inline]
    pub fn leaf_size_m(&self) -> f64 {
        self.meters_per_voxel(Lod::new(self.max_depth))
    }

    /// Pick the coarsest [`Lod`] whose voxel projects to at most
    /// `target_px_per_voxel` pixels at the given camera distance.
    ///
    /// `focal_px` is the camera's focal length in pixels (image-plane).
    pub fn lod_for_screen(&self, distance_m: f64, focal_px: f64, target_px_per_voxel: f64) -> Lod {
        debug_assert!(distance_m > 0.0 && focal_px > 0.0 && target_px_per_voxel > 0.0);
        let target_mpv = (target_px_per_voxel * distance_m) / focal_px;
        if target_mpv <= 0.0 || !target_mpv.is_finite() {
            return Lod::new(self.max_depth);
        }
        let ratio = (self.root_size_m / target_mpv).max(1.0);
        let depth = ratio.log2().ceil().clamp(0.0, self.max_depth as f64) as u8;
        Lod { depth }
    }

    /// As [`Self::lod_for_screen`] but clamps the effective distance to
    /// `max_distance_m` (typically the horizon distance for a spherical
    /// world). Beyond the horizon there's nothing to draw, so we pick the
    /// LOD as if the geometry sat at the horizon. `f64::INFINITY` produces
    /// the unclamped result.
    pub fn lod_for_screen_curved(
        &self,
        distance_m: f64,
        focal_px: f64,
        target_px_per_voxel: f64,
        max_distance_m: f64,
    ) -> Lod {
        let d = distance_m.min(max_distance_m);
        self.lod_for_screen(d, focal_px, target_px_per_voxel)
    }

    /// Pick the coarsest tier whose root cube fully contains `distance_m`.
    /// Used to demote streaming to a parent tier when the observer leaves a
    /// body's atmosphere.
    pub fn tier_for_distance(distance_m: f64) -> Level {
        // Walk from World up to Universe; the first tier whose root_size_m
        // is >= distance_m is the one that contains the observer.
        if distance_m <= MetricScale::DEFAULT_WORLD.root_size_m { Level::World }
        else if distance_m <= MetricScale::DEFAULT_SYSTEM.root_size_m { Level::System }
        else if distance_m <= MetricScale::DEFAULT_SECTOR.root_size_m { Level::Sector }
        else if distance_m <= MetricScale::DEFAULT_GALAXY.root_size_m { Level::Galaxy }
        else { Level::Universe }
    }
}

/// Per-address override of the default per-tier [`MetricScale`].
///
/// `LocalHostConfig` holds an `Arc<MetricScaleRegistry>`. Generation
/// strategies and per-body overrides feed into this registry before actor
/// spawn so the actor sees a stable scale for its lifetime.
#[derive(Clone, Default, Debug)]
pub struct MetricScaleRegistry {
    overrides: HashMap<(Level, LevelKey), MetricScale>,
}

impl MetricScaleRegistry {
    /// Empty registry — every address falls back to the per-tier defaults.
    pub fn new() -> Self { Self::default() }

    /// Identity registry — explicit "use standard defaults" constructor used
    /// by `LocalHostConfig::default()`.
    pub fn standard() -> Self { Self::default() }

    pub fn override_scale(&mut self, level: Level, key: LevelKey, scale: MetricScale) {
        self.overrides.insert((level, key), scale);
    }

    /// Resolve the [`MetricScale`] for a particular tier of an address.
    pub fn scale_for(&self, addr: &Address, level: Level) -> MetricScale {
        let key = match addr {
            Address::World(w) => w.level_key(level),
            Address::Vehicle(v) => {
                // Vehicles inherit their parent's tier scale.
                let parent_world = match v.parent {
                    crate::vehicle::ParentAddr::World(a)
                    | crate::vehicle::ParentAddr::System(a)
                    | crate::vehicle::ParentAddr::Sector(a) => a,
                };
                parent_world.level_key(level)
            }
        };
        if let Some(s) = self.overrides.get(&(level, key)) {
            return *s;
        }
        match level {
            Level::Universe => MetricScale::DEFAULT_UNIVERSE,
            Level::Galaxy => MetricScale::DEFAULT_GALAXY,
            Level::Sector => MetricScale::DEFAULT_SECTOR,
            Level::System => MetricScale::DEFAULT_SYSTEM,
            Level::World => MetricScale::DEFAULT_WORLD,
        }
    }

    /// Resolve the scale of the deepest tier present in the address — for
    /// worlds this is World; for vehicles this is the parent's deepest tier.
    pub fn scale_of(&self, addr: &Address) -> MetricScale {
        match addr {
            Address::World(_) => self.scale_for(addr, Level::World),
            Address::Vehicle(v) => self.scale_for(addr, v.parent.level()),
        }
    }
}

// `WorldAddr` is re-imported above for the registry; suppress an unused-import
// lint that surfaces when the registry isn't exercised in this crate's tests.
#[allow(dead_code)]
fn _unused_addr_imports(_: WorldAddr) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meters_per_voxel_at_root_matches_root_size() {
        let s = MetricScale { root_size_m: 1024.0, max_depth: 10 };
        assert_eq!(s.meters_per_voxel(Lod::new(0)), 1024.0);
        assert_eq!(s.meters_per_voxel(Lod::new(10)), 1.0);
    }

    #[test]
    fn lod_for_screen_monotonic_in_distance() {
        let s = MetricScale { root_size_m: 1024.0, max_depth: 10 };
        let near = s.lod_for_screen(10.0, 1000.0, 1.0);
        let far = s.lod_for_screen(10_000.0, 1000.0, 1.0);
        assert!(far.depth <= near.depth, "farther distance should pick coarser (smaller-depth) LOD");
    }

    #[test]
    fn lod_for_screen_respects_max_depth() {
        let s = MetricScale { root_size_m: 1024.0, max_depth: 10 };
        let l = s.lod_for_screen(1e-6, 1000.0, 1.0);
        assert!(l.depth <= 10);
    }
}
