//! Hydrology overlay — ocean, lake, and river water bodies layered on top
//! of the geological macro pre-simulation.
//!
//! Phase 18: runs as the final macro layer, strictly *after* biome
//! classification, as a pure overlay — it consumes elevation + climate
//! but never feeds back into them. Three [`WaterBodyStrategy`] impls each
//! compute a whole-grid [`WaterLayer`]; [`HydrologyGenerator`] runs them in
//! dependency order (ocean → lake → river) and aggregates the layers into
//! a [`WaterField`] stored on the `WorldMacroState`.
//!
//! The brick generator consults the per-face water data (via
//! `MacroSample`) to place real water columns and carve river channels —
//! the global geological context here, blended with per-brick local-seed
//! FBM detail at generation time.
//!
//! Determinism: every strategy is a pure function of its inputs. Float
//! ordering uses `f32::total_cmp` (never `to_bits()` ordering — elevations
//! go negative and `to_bits` is only monotonic for non-negative floats)
//! with a `FaceId` tie-break, and no `HashMap` iteration ever influences
//! output. The `WaterField` arrays fold into the macro-state digest
//! exactly like the upstream layers.

pub mod lake;
pub mod ocean;
pub mod river;

use std::fmt::Debug;

use super::climate::ClimateField;
use super::plates::ElevationField;
use super::surface_grid::{FaceId, SurfaceGrid};

pub use lake::LakeStrategy;
pub use ocean::OceanStrategy;
pub use river::RiverStrategy;

/// Per-face water classification ids. Single byte, stable across builds —
/// mirrors the `biome` id module.
#[allow(clippy::module_inception)]
pub mod water_kind {
    pub const NONE: u8 = 0;
    pub const OCEAN: u8 = 1;
    pub const LAKE: u8 = 2;
    pub const RIVER: u8 = 3;
}

/// `flow_dir` sentinel: the face has no downhill neighbour (a drainage pit,
/// or a sink — ocean / lake face). Reuses the same `FaceId::MAX` sentinel
/// the surface grid uses for "no neighbour".
pub const NO_FLOW: FaceId = FaceId::MAX;

/// `water_surface_m` sentinel for faces carrying no standing water.
/// `NEG_INFINITY` so `voxel_y < water_surface` is always false for dry
/// faces, and the bit pattern is fixed → folds deterministically.
pub const NO_WATER_SURFACE: f32 = f32::NEG_INFINITY;

/// Aggregated per-world hydrology state. Struct-of-arrays (one entry per
/// surface-grid face), mirroring [`ElevationField`] / `ClimateField` /
/// `BiomeMap` for cache-friendly access and trivial digest folding.
#[derive(Clone, Debug)]
pub struct WaterField {
    /// Final classification per face: `water_kind::{NONE,OCEAN,LAKE,RIVER}`.
    pub water_kind: Vec<u8>,
    /// Water surface elevation (m, sea-level-relative) per face.
    /// [`NO_WATER_SURFACE`] where `water_kind == NONE`.
    pub water_surface_m: Vec<f32>,
    /// Steepest-descent neighbour `FaceId` per face; [`NO_FLOW`] for pits
    /// and sink faces. Retained for *every* face regardless of
    /// `water_kind` — the brick generator carves channels using this even
    /// on ocean-adjacent land.
    pub flow_dir: Vec<FaceId>,
    /// Accumulated upstream flow (precipitation units) per face. `0.0` on
    /// sink faces (ocean / lake).
    pub flow_accum: Vec<f32>,
    /// Sea level, copied from [`HydrologyConfig`] so consumers need only
    /// the `WaterField`, not the config.
    pub sea_level_m: f32,
}

/// Tunables for the hydrology layer.
#[derive(Copy, Clone, Debug)]
pub struct HydrologyConfig {
    /// Elevation (m) of the global ocean surface. Faces below this are
    /// ocean. Default `0.0` — matches the `biome.rs` ocean test.
    pub sea_level_m: f32,
    /// A priority-flood basin counts as a lake only where
    /// `flood_level - elev_m > min_lake_depth_m`. Default `8.0`.
    pub min_lake_depth_m: f32,
    /// Climate gate: a basin face becomes a lake only if its local
    /// humidity is at least this. Arid basins stay dry salt flats.
    /// Default `0.25`.
    pub lake_aridity_threshold: f32,
    /// `flow_accum` above which a land face is a river corridor.
    /// Default `60.0`.
    pub river_threshold: f32,
    /// Per-face base flow contribution, so even arid headwaters seed some
    /// flow. Default `1.0`.
    pub base_flow_per_face: f32,
    /// Scales `precipitation_mm` into flow units before accumulation.
    /// Default `0.01` (600 mm precip → 6.0 flow units).
    pub precip_to_flow_scale: f32,
}

impl Default for HydrologyConfig {
    fn default() -> Self {
        Self {
            sea_level_m: 0.0,
            min_lake_depth_m: 8.0,
            lake_aridity_threshold: 0.25,
            river_threshold: 60.0,
            base_flow_per_face: 1.0,
            precip_to_flow_scale: 0.01,
        }
    }
}

/// Read-only inputs visible to every [`WaterBodyStrategy`]. Borrows the
/// upstream macro layers plus the seed/config, and — crucially — the
/// `WaterLayer`s already produced this run via `prior`, so later
/// strategies can use earlier ones as drainage sinks (lake needs ocean;
/// river needs ocean + lake).
#[derive(Debug)]
pub struct HydrologyInput<'a> {
    pub grid: &'a SurfaceGrid,
    pub elevation: &'a ElevationField,
    pub climate: &'a ClimateField,
    pub world_seed: u64,
    pub cfg: HydrologyConfig,
    /// Layers produced by strategies that ran earlier in dependency order.
    /// Indexed by run sequence: `[]`, then `[ocean]`, then `[ocean, lake]`.
    pub prior: &'a [WaterLayer],
}

/// One strategy's whole-grid contribution. Every `Vec` has
/// `len == grid.face_count()`.
#[derive(Clone, Debug)]
pub struct WaterLayer {
    /// What this strategy claims at each face (`water_kind::NONE` where it
    /// abstains).
    pub kind: Vec<u8>,
    /// Surface elevation this strategy assigns; [`NO_WATER_SURFACE`] where
    /// `kind == NONE`.
    pub surface_m: Vec<f32>,
    /// Flow routing — only [`RiverStrategy`] populates these; ocean and
    /// lake leave them as `(NO_FLOW, 0.0)`.
    pub flow_dir: Vec<FaceId>,
    pub flow_accum: Vec<f32>,
}

impl WaterLayer {
    /// An all-`NONE` layer of the right length — strategies start here.
    pub fn empty(face_count: usize) -> Self {
        Self {
            kind: vec![water_kind::NONE; face_count],
            surface_m: vec![NO_WATER_SURFACE; face_count],
            flow_dir: vec![NO_FLOW; face_count],
            flow_accum: vec![0.0; face_count],
        }
    }
}

/// A whole-field hydrology pass. Pure: deterministic from `input`.
///
/// The method is a whole-grid `compute` rather than a per-face `classify`
/// because lake fill is a global priority-flood and river accumulation is
/// a global high→low sweep — neither is expressible per-face.
pub trait WaterBodyStrategy: Debug + Send + Sync {
    /// Stable name, used for the reordering `debug_assert` and debugging.
    fn name(&self) -> &'static str;
    /// Compute this strategy's contribution over the whole grid.
    fn compute(&self, input: &HydrologyInput) -> WaterLayer;
}

/// Runs the ocean / lake / river strategies in dependency order and
/// aggregates their layers into a [`WaterField`].
#[derive(Debug)]
pub struct HydrologyGenerator {
    /// Strategies in **fixed dependency order**: `[ocean, lake, river]`.
    /// `LakeStrategy` hard-indexes `prior[0]` (ocean); `RiverStrategy`
    /// indexes `prior[0]`/`prior[1]` (ocean, lake). A `debug_assert` in
    /// `generate` catches accidental reordering.
    strategies: Vec<Box<dyn WaterBodyStrategy>>,
    cfg: HydrologyConfig,
}

impl HydrologyGenerator {
    pub fn new(cfg: HydrologyConfig) -> Self {
        Self {
            strategies: vec![
                Box::new(OceanStrategy),
                Box::new(LakeStrategy),
                Box::new(RiverStrategy),
            ],
            cfg,
        }
    }

    /// Compute the world's [`WaterField`]. Pure function of
    /// `(grid, elevation, climate, world_seed, cfg)`.
    pub fn generate(
        &self,
        grid: &SurfaceGrid,
        elevation: &ElevationField,
        climate: &ClimateField,
        world_seed: u64,
    ) -> WaterField {
        let n = grid.face_count();

        debug_assert_eq!(self.strategies.len(), 3, "expect [ocean, lake, river]");
        debug_assert_eq!(self.strategies[0].name(), "Ocean");
        debug_assert_eq!(self.strategies[1].name(), "Lake");
        debug_assert_eq!(self.strategies[2].name(), "River");

        let mut layers: Vec<WaterLayer> = Vec::with_capacity(self.strategies.len());
        for strat in &self.strategies {
            let input = HydrologyInput {
                grid,
                elevation,
                climate,
                world_seed,
                cfg: self.cfg,
                prior: &layers,
            };
            let layer = strat.compute(&input);
            debug_assert_eq!(layer.kind.len(), n);
            layers.push(layer);
        }

        let ocean = &layers[0];
        let lake = &layers[1];
        let river = &layers[2];

        // Aggregate. Priority: ocean > lake > river for kind / surface.
        let mut water_kind_v = vec![water_kind::NONE; n];
        let mut water_surface_m = vec![NO_WATER_SURFACE; n];
        for f in 0..n {
            if ocean.kind[f] != water_kind::NONE {
                water_kind_v[f] = water_kind::OCEAN;
                water_surface_m[f] = ocean.surface_m[f];
            } else if lake.kind[f] != water_kind::NONE {
                water_kind_v[f] = water_kind::LAKE;
                water_surface_m[f] = lake.surface_m[f];
            } else if river.kind[f] != water_kind::NONE {
                water_kind_v[f] = water_kind::RIVER;
                water_surface_m[f] = river.surface_m[f];
            }
        }

        WaterField {
            water_kind: water_kind_v,
            water_surface_m,
            // flow_dir / flow_accum come unconditionally from the river
            // layer — retained for every face so the brick generator can
            // carve channels regardless of the aggregated water_kind.
            flow_dir: river.flow_dir.clone(),
            flow_accum: river.flow_accum.clone(),
            sea_level_m: self.cfg.sea_level_m,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::climate::{generate_climate, ClimateConfig};
    use crate::macro_state::plates::{generate_plates, PlateConfig};

    /// Build a real upstream pipeline (plates → climate) for a small grid.
    fn pipeline(level: u8, seed: u64) -> (SurfaceGrid, ElevationField, ClimateField) {
        let g = SurfaceGrid::new(level);
        let (_, elev) = generate_plates(&g, seed, PlateConfig::default());
        let cl = generate_climate(&g, &elev, ClimateConfig::default());
        (g, elev, cl)
    }

    #[test]
    fn generate_is_deterministic() {
        let (g, elev, cl) = pipeline(3, 0xCAFE_F00D);
        let gen = HydrologyGenerator::new(HydrologyConfig::default());
        let a = gen.generate(&g, &elev, &cl, 0xCAFE_F00D);
        let b = gen.generate(&g, &elev, &cl, 0xCAFE_F00D);
        assert_eq!(a.water_kind, b.water_kind);
        assert_eq!(a.flow_dir, b.flow_dir);
        for i in 0..a.water_surface_m.len() {
            assert_eq!(a.water_surface_m[i].to_bits(), b.water_surface_m[i].to_bits());
            assert_eq!(a.flow_accum[i].to_bits(), b.flow_accum[i].to_bits());
        }
    }

    #[test]
    fn water_field_arrays_are_one_per_face() {
        let (g, elev, cl) = pipeline(3, 0x1234);
        let gen = HydrologyGenerator::new(HydrologyConfig::default());
        let w = gen.generate(&g, &elev, &cl, 0x1234);
        let n = g.face_count();
        assert_eq!(w.water_kind.len(), n);
        assert_eq!(w.water_surface_m.len(), n);
        assert_eq!(w.flow_dir.len(), n);
        assert_eq!(w.flow_accum.len(), n);
    }

    #[test]
    fn classification_and_surface_are_consistent() {
        let (g, elev, cl) = pipeline(4, 0xDEAD_BEEF);
        let gen = HydrologyGenerator::new(HydrologyConfig::default());
        let w = gen.generate(&g, &elev, &cl, 0xDEAD_BEEF);
        for f in 0..g.face_count() {
            match w.water_kind[f] {
                water_kind::NONE => {
                    assert_eq!(w.water_surface_m[f], NO_WATER_SURFACE);
                }
                water_kind::OCEAN => {
                    assert_eq!(w.water_surface_m[f], w.sea_level_m);
                }
                water_kind::LAKE => {
                    // A lake sits above its own ground.
                    assert!(w.water_surface_m[f] > elev.elev_m[f]);
                }
                water_kind::RIVER => {
                    assert!(w.water_surface_m[f].is_finite());
                }
                other => panic!("unexpected water_kind {other}"),
            }
            assert!(w.flow_accum[f] >= 0.0);
        }
    }

    #[test]
    fn aggregation_prefers_ocean_over_lake_over_river() {
        // Synthesised: every face below sea level (all ocean). Lake and
        // river must abstain everywhere, so the aggregate is all-ocean.
        let g = SurfaceGrid::new(2);
        let n = g.face_count();
        let elevation = ElevationField { elev_m: vec![-500.0; n] };
        let climate = ClimateField {
            temperature_c: vec![15.0; n],
            humidity: vec![0.9; n],
            precipitation_mm: vec![400.0; n],
        };
        let gen = HydrologyGenerator::new(HydrologyConfig::default());
        let w = gen.generate(&g, &elevation, &climate, 0x1);
        assert!(w.water_kind.iter().all(|&k| k == water_kind::OCEAN));
    }
}
