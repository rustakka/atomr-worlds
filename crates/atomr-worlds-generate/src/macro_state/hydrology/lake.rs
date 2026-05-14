//! Lake strategy — Barnes-style priority-flood basin fill, climate-gated.
//!
//! The surface grid is a closed sphere with no boundary, so a classic
//! flood-from-the-border has nothing to seed from. Instead the flood is
//! seeded from the **ocean faces** produced by [`OceanStrategy`] — ocean
//! is the global drainage base level. Water propagates inland; each face
//! takes `max(its own elevation, the level it was reached at)`. A
//! non-ocean face whose flood level sits more than `min_lake_depth_m`
//! above its own ground is a closed basin, and becomes a lake only if its
//! local humidity clears `lake_aridity_threshold` (arid basins stay dry
//! salt flats).
//!
//! The flood also records, per face, the neighbour it was flooded *from*
//! (`parent`). That parent chain is a spanning forest rooted at the ocean
//! — a complete drainage network that routes correctly *through* filled
//! basins out their spill point. It is published as the layer's
//! `flow_dir` so [`RiverStrategy`](super::river::RiverStrategy) can
//! accumulate flow along it without recomputing drainage.
//!
//! Determinism: the priority queue is keyed by `(level, face)` ordered via
//! `f32::total_cmp` then `FaceId` — never `f32::to_bits` (elevations go
//! negative; `to_bits` is only monotonic for non-negative floats). The
//! `FaceId` tie-break makes pops a strict total order. No `HashMap`.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::macro_state::surface_grid::FaceId;

use super::{water_kind, HydrologyInput, WaterBodyStrategy, WaterLayer, NO_FLOW};

/// Fills closed basins to their spill elevation via priority-flood.
#[derive(Debug, Default, Clone, Copy)]
pub struct LakeStrategy;

/// Deterministic min-heap key. Ordered by flood `level` (via `total_cmp`,
/// so negative elevations sort correctly), tie-broken by ascending
/// `FaceId` so pops are a strict total order even when two faces share a
/// level.
#[derive(Copy, Clone, PartialEq)]
struct HeapKey {
    level: f32,
    face: FaceId,
}

impl Eq for HeapKey {}

impl Ord for HeapKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.level
            .total_cmp(&other.level)
            .then_with(|| self.face.cmp(&other.face))
    }
}

impl PartialOrd for HeapKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl WaterBodyStrategy for LakeStrategy {
    fn name(&self) -> &'static str {
        "Lake"
    }

    fn compute(&self, input: &HydrologyInput) -> WaterLayer {
        let grid = input.grid;
        let elev = &input.elevation.elev_m;
        let n = grid.face_count();
        let mut layer = WaterLayer::empty(n);

        // OceanStrategy ran first — prior[0].
        let ocean = &input.prior[0];

        // A world with no ocean has no base level to flood from; treat it
        // as having no lakes rather than drowning every basin.
        let has_ocean = ocean.kind.iter().any(|&k| k == water_kind::OCEAN);
        if !has_ocean {
            return layer;
        }

        let sea = input.cfg.sea_level_m;
        let mut water_level = vec![f32::INFINITY; n];
        let mut processed = vec![false; n];
        // Flood drainage tree: the neighbour each face was flooded from.
        // Ocean seed faces stay NO_FLOW (roots of the forest).
        let mut parent = vec![NO_FLOW; n];
        let mut heap: BinaryHeap<Reverse<HeapKey>> = BinaryHeap::new();

        // Seed: every ocean face at sea level. The heap re-sorts, so seed
        // insertion order is irrelevant to the result.
        for f in 0..n {
            if ocean.kind[f] == water_kind::OCEAN {
                water_level[f] = sea;
                processed[f] = true;
                heap.push(Reverse(HeapKey { level: sea, face: f as FaceId }));
            }
        }

        // Priority flood. Each face is pushed exactly once (when first
        // reached, which — because the heap pops in increasing level
        // order — is via its lowest spill path) and popped exactly once.
        while let Some(Reverse(HeapKey { level, face })) = heap.pop() {
            for nb in grid.neighbours_of(face) {
                if nb == FaceId::MAX {
                    continue; // never happens on a closed sphere; defensive
                }
                let nb_i = nb as usize;
                if processed[nb_i] {
                    continue;
                }
                // The lowest water can sit at `nb` without a lower escape
                // is the higher of its own ground and the level we
                // arrived with.
                let new_level = elev[nb_i].max(level);
                water_level[nb_i] = new_level;
                processed[nb_i] = true;
                parent[nb_i] = face; // drainage points back toward the ocean
                heap.push(Reverse(HeapKey { level: new_level, face: nb }));
            }
        }

        // Classify lakes — climate-gated.
        let min_depth = input.cfg.min_lake_depth_m;
        let aridity = input.cfg.lake_aridity_threshold;
        for f in 0..n {
            if ocean.kind[f] == water_kind::OCEAN {
                continue; // ocean owns these faces
            }
            let depth = water_level[f] - elev[f];
            if depth > min_depth && input.climate.humidity[f] >= aridity {
                layer.kind[f] = water_kind::LAKE;
                layer.surface_m[f] = water_level[f];
            }
        }

        // Publish the flood drainage tree for RiverStrategy.
        layer.flow_dir = parent;
        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::climate::{generate_climate, ClimateConfig, ClimateField};
    use crate::macro_state::hydrology::ocean::OceanStrategy;
    use crate::macro_state::hydrology::HydrologyConfig;
    use crate::macro_state::plates::{generate_plates, ElevationField, PlateConfig};
    use crate::macro_state::surface_grid::SurfaceGrid;

    fn ocean_layer(
        grid: &SurfaceGrid,
        elevation: &ElevationField,
        climate: &ClimateField,
        cfg: HydrologyConfig,
    ) -> WaterLayer {
        OceanStrategy.compute(&HydrologyInput {
            grid,
            elevation,
            climate,
            world_seed: 0xABCD,
            cfg,
            prior: &[],
        })
    }

    #[test]
    fn is_deterministic() {
        let g = SurfaceGrid::new(3);
        let (_, elev) = generate_plates(&g, 0xCAFE_F00D, PlateConfig::default());
        let cl = generate_climate(&g, &elev, ClimateConfig::default());
        let cfg = HydrologyConfig::default();
        let ocean = ocean_layer(&g, &elev, &cl, cfg);
        let prior = vec![ocean];
        let input = HydrologyInput {
            grid: &g,
            elevation: &elev,
            climate: &cl,
            world_seed: 0xCAFE_F00D,
            cfg,
            prior: &prior,
        };
        let a = LakeStrategy.compute(&input);
        let b = LakeStrategy.compute(&input);
        assert_eq!(a.kind, b.kind);
        for i in 0..a.surface_m.len() {
            assert_eq!(a.surface_m[i].to_bits(), b.surface_m[i].to_bits());
        }
    }

    #[test]
    fn floods_closed_basin_and_climate_gates_it() {
        // Synthesise: face 0 is ocean, one interior face is a deep basin,
        // every other face is a high plateau. The basin can only be
        // reached across the plateau, so it floods to plateau height.
        let g = SurfaceGrid::new(2);
        let n = g.face_count();
        let mut elev_m = vec![500.0_f32; n];
        elev_m[0] = -100.0;
        // A basin face that is NOT adjacent to the ocean face — the
        // neighbour relation is symmetric, so this guarantees the flood
        // must cross the plateau to reach it.
        let nbrs0 = g.neighbours_of(0);
        let basin = (1..n as u32)
            .find(|f| !nbrs0.contains(f))
            .expect("a non-ocean-adjacent face exists") as usize;
        elev_m[basin] = 10.0;
        let elevation = ElevationField { elev_m };

        let cfg = HydrologyConfig::default();

        // Wet climate → the basin becomes a lake.
        let wet = ClimateField {
            temperature_c: vec![15.0; n],
            humidity: vec![0.9; n],
            precipitation_mm: vec![400.0; n],
        };
        let ocean = ocean_layer(&g, &elevation, &wet, cfg);
        let prior = vec![ocean];
        let lake = LakeStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elevation,
            climate: &wet,
            world_seed: 0x1,
            cfg,
            prior: &prior,
        });
        assert_eq!(lake.kind[basin], water_kind::LAKE);
        assert!(lake.surface_m[basin] > elevation.elev_m[basin]);
        assert_eq!(lake.kind[0], water_kind::NONE, "ocean face is not a lake");

        // Dry climate → the same basin is gated out as a salt flat.
        let dry = ClimateField {
            temperature_c: vec![15.0; n],
            humidity: vec![0.0; n],
            precipitation_mm: vec![0.0; n],
        };
        let ocean_dry = ocean_layer(&g, &elevation, &dry, cfg);
        let prior_dry = vec![ocean_dry];
        let lake_dry = LakeStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elevation,
            climate: &dry,
            world_seed: 0x1,
            cfg,
            prior: &prior_dry,
        });
        assert_eq!(lake_dry.kind[basin], water_kind::NONE);
    }

    #[test]
    fn no_ocean_world_has_no_lakes() {
        let g = SurfaceGrid::new(2);
        let n = g.face_count();
        // Every face above sea level → OceanStrategy produces all-NONE.
        let elevation = ElevationField { elev_m: vec![300.0; n] };
        let climate = ClimateField {
            temperature_c: vec![15.0; n],
            humidity: vec![1.0; n],
            precipitation_mm: vec![800.0; n],
        };
        let cfg = HydrologyConfig::default();
        let ocean = ocean_layer(&g, &elevation, &climate, cfg);
        assert!(ocean.kind.iter().all(|&k| k == water_kind::NONE));
        let prior = vec![ocean];
        let lake = LakeStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elevation,
            climate: &climate,
            world_seed: 0x1,
            cfg,
            prior: &prior,
        });
        assert!(lake.kind.iter().all(|&k| k == water_kind::NONE));
    }
}
