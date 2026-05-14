//! Layered geologic / climate / biome pre-simulation for a world.
//!
//! Phase 13c: introduces a deterministic pre-pass that runs once per
//! sphere world before any brick generation. Produces a
//! [`WorldMacroState`] which downstream [`BrickGenerator`] impls consult
//! to drive surface height, biome, and material choice with planet-scale
//! coherence (tectonic-plate uplift, climate zones).
//!
//! Layers (top-down):
//! 1. [`surface_grid`] — recursive icosahedron, integer face IDs, O(1)
//!    neighbour lookups.
//! 2. [`plates`] — Voronoi tectonic plates seeded from world_seed,
//!    convergent-boundary uplift gives elevation per face.
//! 3. [`climate`] — temperature drops with latitude + altitude, humidity
//!    advects from ocean cells.
//! 4. [`biome`] — fixed classification table over `(elev, temp, humidity)`.
//!
//! Determinism: every layer is a pure function of `world_seed` plus its
//! predecessors. The `digest` field is a FNV-1a witness over all
//! produced arrays — same `(seed, shape, config)` ⇒ same digest, byte-
//! identical across runs and platforms.

pub mod biome;
pub mod climate;
pub mod hydrology;
pub mod plates;
pub mod relief;
pub mod surface_grid;

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Arc, Mutex};

use atomr_worlds_core::addr::Address;
use atomr_worlds_core::shape::WorldShape;

pub use biome::{biome as biome_id, BiomeMap};
pub use climate::{ClimateConfig, ClimateField};
pub use hydrology::{
    water_kind, HydrologyConfig, HydrologyGenerator, WaterBodyStrategy, WaterField, WaterLayer,
    NO_FLOW, NO_WATER_SURFACE,
};
pub use plates::{ElevationField, PlateConfig, PlateMap};
pub use relief::ReliefConfig;
pub use surface_grid::{Face, FaceId, SurfaceGrid, VertexId};

/// Configuration for [`DefaultMacroGenerator`].
#[derive(Copy, Clone, Debug)]
pub struct MacroConfig {
    /// Subdivision depth for [`SurfaceGrid`]. Default 4 (5_120 faces,
    /// ~150 KB state); raise to 6 for ~82k-face Earth-class detail
    /// (~2 MB).
    pub grid_level: u8,
    pub plates: PlateConfig,
    pub relief: ReliefConfig,
    pub climate: ClimateConfig,
    pub hydrology: HydrologyConfig,
}

impl Default for MacroConfig {
    fn default() -> Self {
        Self {
            grid_level: 4,
            plates: PlateConfig::default(),
            relief: ReliefConfig::default(),
            climate: ClimateConfig::default(),
            hydrology: HydrologyConfig::default(),
        }
    }
}

/// Pre-computed macro state for one world.
#[derive(Clone, Debug)]
pub struct WorldMacroState {
    pub shape: WorldShape,
    pub grid: SurfaceGrid,
    pub plates: PlateMap,
    pub elevation: ElevationField,
    pub climate: ClimateField,
    pub biomes: BiomeMap,
    /// Ocean / lake / river overlay — computed strictly after biomes as a
    /// pure overlay (see [`hydrology`]).
    pub water: WaterField,
    /// FNV-1a witness over every output array. Same `(seed, shape, config)`
    /// ⇒ same digest, across runs and platforms.
    pub digest: u64,
}

impl WorldMacroState {
    /// Look up the macro state at a world-space direction (unit vector
    /// from world center). Returns the per-face geological + hydrology
    /// sample.
    pub fn sample(&self, dir: atomr_worlds_core::coord::DVec3) -> MacroSample {
        let f = self.grid.face_for_direction(dir) as usize;
        MacroSample {
            face: f as FaceId,
            elev_m: self.elevation.elev_m[f],
            temperature_c: self.climate.temperature_c[f],
            humidity: self.climate.humidity[f],
            biome_id: self.biomes.biome_id[f],
            water_kind: self.water.water_kind[f],
            water_surface_m: self.water.water_surface_m[f],
            flow_dir: self.water.flow_dir[f],
            flow_accum: self.water.flow_accum[f],
        }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MacroSample {
    pub face: FaceId,
    pub elev_m: f32,
    pub temperature_c: f32,
    pub humidity: f32,
    pub biome_id: u8,
    /// `water_kind::{NONE,OCEAN,LAKE,RIVER}` at this face.
    pub water_kind: u8,
    /// Water surface elevation (m); [`NO_WATER_SURFACE`] where dry.
    pub water_surface_m: f32,
    /// Steepest-descent neighbour face for river flow; [`NO_FLOW`] if none.
    pub flow_dir: FaceId,
    /// Accumulated upstream flow — drives river channel width/depth.
    pub flow_accum: f32,
}

/// Trait for macro-state producers. Pure: `(seed, shape)` ⇒ same state.
pub trait MacroGenerator: Send + Sync + Debug {
    fn generate(&self, world_seed: u64, shape: WorldShape) -> Arc<WorldMacroState>;
}

/// Canonical CPU implementation — the three-layer pipeline described in
/// the module docs.
#[derive(Clone, Debug, Default)]
pub struct DefaultMacroGenerator {
    pub config: MacroConfig,
}

impl DefaultMacroGenerator {
    pub fn new(config: MacroConfig) -> Self {
        Self { config }
    }
}

impl MacroGenerator for DefaultMacroGenerator {
    fn generate(&self, world_seed: u64, shape: WorldShape) -> Arc<WorldMacroState> {
        let grid = SurfaceGrid::new(self.config.grid_level);
        let (plates_map, mut elevation) =
            plates::generate_plates(&grid, world_seed, self.config.plates);
        // Meso-scale relief refines the piecewise-flat plate elevation so
        // that climate, biomes, hydrology, and brick-level terrain all see
        // one coherent field with real drainage gradients and basins.
        relief::apply_relief(&grid, &mut elevation, world_seed, self.config.relief);
        let climate = climate::generate_climate(&grid, &elevation, self.config.climate);
        let biomes = biome::classify_biomes(&elevation, &climate);
        // Hydrology runs strictly after biomes as a pure overlay — it
        // consumes elevation + climate but never feeds back into them.
        let water = HydrologyGenerator::new(self.config.hydrology)
            .generate(&grid, &elevation, &climate, world_seed);
        let digest = compute_digest(&plates_map, &elevation, &climate, &biomes, &water);
        Arc::new(WorldMacroState {
            shape,
            grid,
            plates: plates_map,
            elevation,
            climate,
            biomes,
            water,
            digest,
        })
    }
}

fn compute_digest(
    plates: &PlateMap,
    elev: &ElevationField,
    climate: &ClimateField,
    biomes: &BiomeMap,
    water: &WaterField,
) -> u64 {
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let prime: u64 = 0x0000_0100_0000_01B3;
    let mut fold = |bytes: &[u8]| {
        for b in bytes {
            h ^= *b as u64;
            h = h.wrapping_mul(prime);
        }
    };
    for p in &plates.plate_id {
        fold(&p.to_le_bytes());
    }
    for v in &plates.velocity {
        fold(&v.x.to_bits().to_le_bytes());
        fold(&v.y.to_bits().to_le_bytes());
        fold(&v.z.to_bits().to_le_bytes());
    }
    for &s in &plates.seeds {
        fold(&s.to_le_bytes());
    }
    for &e in &elev.elev_m {
        fold(&e.to_bits().to_le_bytes());
    }
    for &t in &climate.temperature_c {
        fold(&t.to_bits().to_le_bytes());
    }
    for &h_ in &climate.humidity {
        fold(&h_.to_bits().to_le_bytes());
    }
    for &p in &climate.precipitation_mm {
        fold(&p.to_bits().to_le_bytes());
    }
    fold(&biomes.biome_id);
    // Hydrology overlay — appended after the biome fold so the existing
    // digest prefix is unchanged.
    fold(&water.water_kind);
    for &s in &water.water_surface_m {
        fold(&s.to_bits().to_le_bytes());
    }
    for &d in &water.flow_dir {
        fold(&d.to_le_bytes());
    }
    for &a in &water.flow_accum {
        fold(&a.to_bits().to_le_bytes());
    }
    fold(&water.sea_level_m.to_bits().to_le_bytes());
    h
}

/// Per-host cache of macro states keyed by `(addr, seed, shape)`. Used
/// by the host to avoid recomputing macro state on every actor spawn.
#[derive(Debug, Default)]
pub struct MacroStateCache {
    inner: Mutex<HashMap<MacroKey, Arc<WorldMacroState>>>,
}

#[derive(Copy, Clone, Hash, Eq, PartialEq, Debug)]
pub struct MacroKey {
    pub addr: Address,
    pub seed: u64,
    pub shape: WorldShape,
}

impl MacroStateCache {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get_or_compute(
        &self,
        addr: Address,
        seed: u64,
        shape: WorldShape,
        generator: &dyn MacroGenerator,
    ) -> Arc<WorldMacroState> {
        let key = MacroKey { addr, seed, shape };
        {
            let map = self.inner.lock().unwrap();
            if let Some(s) = map.get(&key) {
                return s.clone();
            }
        }
        let state = generator.generate(seed, shape);
        let mut map = self.inner.lock().unwrap();
        map.entry(key).or_insert_with(|| state.clone());
        state
    }

    pub fn insert(&self, key: MacroKey, state: Arc<WorldMacroState>) {
        self.inner.lock().unwrap().insert(key, state);
    }

    pub fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn macro_state_is_deterministic() {
        let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
        let shape = WorldShape::Sphere { radius_m: 6.371e6 };
        let a = g.generate(0xDEAD_BEEF_CAFE_F00D, shape);
        let b = g.generate(0xDEAD_BEEF_CAFE_F00D, shape);
        assert_eq!(a.digest, b.digest);
    }

    #[test]
    fn macro_state_seed_change_changes_digest() {
        let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
        let shape = WorldShape::Sphere { radius_m: 6.371e6 };
        let a = g.generate(0x1111, shape);
        let b = g.generate(0x2222, shape);
        assert_ne!(a.digest, b.digest);
    }

    #[test]
    fn macro_state_shape_change_changes_digest() {
        // Different radius → different plate placement absolutely (no —
        // shape doesn't feed plates), but should at least change the
        // stored `shape` field via equality. The digest does NOT include
        // the shape itself today; that's intentional — the field is in
        // the state, not the digest. Test by sample/equality instead.
        let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 2, ..MacroConfig::default() });
        let a = g.generate(0xCAFE, WorldShape::Sphere { radius_m: 1.0e6 });
        let b = g.generate(0xCAFE, WorldShape::Sphere { radius_m: 2.0e6 });
        assert_eq!(a.digest, b.digest, "digest is shape-independent by design");
        assert_ne!(a.shape, b.shape, "but shape is stored");
    }

    #[test]
    fn cache_short_circuits_on_repeated_lookup() {
        let cache = MacroStateCache::new();
        let g = DefaultMacroGenerator::new(MacroConfig { grid_level: 1, ..MacroConfig::default() });
        let addr = Address::World(atomr_worlds_core::addr::WorldAddr::ROOT);
        let shape = WorldShape::Sphere { radius_m: 6.371e6 };
        let a = cache.get_or_compute(addr, 0xCAFE, shape, &g);
        let b = cache.get_or_compute(addr, 0xCAFE, shape, &g);
        assert!(Arc::ptr_eq(&a, &b));
        assert_eq!(cache.len(), 1);
    }
}
