//! River strategy — flow accumulation over the flood drainage tree.
//!
//! [`LakeStrategy`](super::lake::LakeStrategy) already produced a complete
//! drainage network as a side effect of its priority-flood: each face's
//! `flow_dir` is the neighbour it was flooded from, so the parent chains
//! form a spanning forest rooted at the ocean. Crucially this network
//! routes *through* filled lake basins and out their spill point, so a
//! river can chain headwater → stream → lake → stream → sea.
//!
//! This strategy takes that tree as given, gives every land face a local
//! flow contribution (base flow plus precipitation), and accumulates flow
//! downstream in topological order — every face is summed before its
//! downstream target. Land faces (non-ocean, non-lake) whose accumulated
//! flow clears `river_threshold` are river corridors. Lake faces still
//! pass flow downstream but read as lakes, not rivers.
//!
//! Determinism: the topological order uses a `FaceId` min-heap (Kahn's
//! algorithm with a deterministic ready-set order). No `HashMap`.

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::macro_state::surface_grid::FaceId;

use super::{water_kind, HydrologyInput, WaterBodyStrategy, WaterLayer, NO_FLOW};

/// Accumulates flow over the flood drainage tree and classifies rivers.
#[derive(Debug, Default, Clone, Copy)]
pub struct RiverStrategy;

impl WaterBodyStrategy for RiverStrategy {
    fn name(&self) -> &'static str {
        "River"
    }

    fn compute(&self, input: &HydrologyInput) -> WaterLayer {
        let n = input.grid.face_count();
        let mut layer = WaterLayer::empty(n);

        // prior = [ocean, lake].
        let ocean = &input.prior[0];
        let lake = &input.prior[1];

        // The lake layer's priority-flood produced a spanning forest
        // rooted at the ocean: flow_dir[f] is the neighbour f was flooded
        // from — the downstream direction, routing through filled basins.
        layer.flow_dir = lake.flow_dir.clone();

        // 1. Local flow contribution per face (ocean faces contribute 0).
        for f in 0..n {
            layer.flow_accum[f] = if ocean.kind[f] != water_kind::NONE {
                0.0
            } else {
                input.cfg.base_flow_per_face
                    + input.climate.precipitation_mm[f] * input.cfg.precip_to_flow_scale
            };
        }

        // 2. Accumulate downstream in topological order (Kahn's): a face
        //    is summed only once every face draining into it is done.
        let mut in_degree = vec![0u32; n];
        for f in 0..n {
            let d = layer.flow_dir[f];
            if d != NO_FLOW {
                in_degree[d as usize] += 1;
            }
        }
        // Deterministic ready-set: a FaceId min-heap.
        let mut ready: BinaryHeap<Reverse<FaceId>> = (0..n as FaceId)
            .filter(|&f| in_degree[f as usize] == 0)
            .map(Reverse)
            .collect();
        while let Some(Reverse(f)) = ready.pop() {
            let fi = f as usize;
            let d = layer.flow_dir[fi];
            if d == NO_FLOW {
                continue; // a forest root (ocean / unreached face)
            }
            let di = d as usize;
            // Flow draining into the ocean disappears — ocean stays at 0.
            if ocean.kind[di] == water_kind::NONE {
                layer.flow_accum[di] += layer.flow_accum[fi];
            }
            in_degree[di] -= 1;
            if in_degree[di] == 0 {
                ready.push(Reverse(d));
            }
        }

        // 3. Classify river corridors — land faces (non-ocean, non-lake)
        //    whose through-flow clears the threshold. Lakes still carry
        //    flow downstream but read as lakes, not rivers.
        let threshold = input.cfg.river_threshold;
        for f in 0..n {
            if ocean.kind[f] != water_kind::NONE || lake.kind[f] != water_kind::NONE {
                continue;
            }
            if layer.flow_accum[f] > threshold {
                layer.kind[f] = water_kind::RIVER;
                // Rivers sit ≈ ground level; the brick generator insets
                // the actual water surface inside the carved channel.
                layer.surface_m[f] = input.elevation.elev_m[f];
            }
        }

        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::climate::{generate_climate, ClimateConfig, ClimateField};
    use crate::macro_state::hydrology::lake::LakeStrategy;
    use crate::macro_state::hydrology::ocean::OceanStrategy;
    use crate::macro_state::hydrology::HydrologyConfig;
    use crate::macro_state::plates::{generate_plates, ElevationField, PlateConfig};
    use crate::macro_state::relief::{apply_relief, ReliefConfig};
    use crate::macro_state::surface_grid::SurfaceGrid;

    /// Build the [ocean, lake] prior layers for a real small pipeline.
    fn prior_layers(
        grid: &SurfaceGrid,
        elevation: &ElevationField,
        climate: &ClimateField,
        cfg: HydrologyConfig,
        seed: u64,
    ) -> Vec<WaterLayer> {
        let ocean = OceanStrategy.compute(&HydrologyInput {
            grid,
            elevation,
            climate,
            world_seed: seed,
            cfg,
            prior: &[],
        });
        let lake = {
            let prior = vec![ocean.clone()];
            LakeStrategy.compute(&HydrologyInput {
                grid,
                elevation,
                climate,
                world_seed: seed,
                cfg,
                prior: &prior,
            })
        };
        vec![ocean, lake]
    }

    /// A real plates → relief → climate pipeline for `level`/`seed`.
    fn pipeline(level: u8, seed: u64) -> (SurfaceGrid, ElevationField, ClimateField) {
        let g = SurfaceGrid::new(level);
        let (_, mut elev) = generate_plates(&g, seed, PlateConfig::default());
        apply_relief(&g, &mut elev, seed, ReliefConfig::default());
        let cl = generate_climate(&g, &elev, ClimateConfig::default());
        (g, elev, cl)
    }

    #[test]
    fn is_deterministic() {
        let (g, elev, cl) = pipeline(3, 0xCAFE_F00D);
        let cfg = HydrologyConfig::default();
        let prior = prior_layers(&g, &elev, &cl, cfg, 0xCAFE_F00D);
        let input = HydrologyInput {
            grid: &g,
            elevation: &elev,
            climate: &cl,
            world_seed: 0xCAFE_F00D,
            cfg,
            prior: &prior,
        };
        let a = RiverStrategy.compute(&input);
        let b = RiverStrategy.compute(&input);
        assert_eq!(a.kind, b.kind);
        assert_eq!(a.flow_dir, b.flow_dir);
        for i in 0..a.flow_accum.len() {
            assert_eq!(a.flow_accum[i].to_bits(), b.flow_accum[i].to_bits());
        }
    }

    #[test]
    fn flow_dir_is_a_valid_neighbour_or_sentinel() {
        let (g, elev, cl) = pipeline(4, 0xBEEF);
        let cfg = HydrologyConfig::default();
        let prior = prior_layers(&g, &elev, &cl, cfg, 0xBEEF);
        let layer = RiverStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elev,
            climate: &cl,
            world_seed: 0xBEEF,
            cfg,
            prior: &prior,
        });
        for f in 0..g.face_count() {
            let dir = layer.flow_dir[f];
            if dir != NO_FLOW {
                assert!(
                    g.neighbours_of(f as FaceId).contains(&dir),
                    "flow_dir must be an edge-adjacent neighbour",
                );
            }
        }
    }

    #[test]
    fn flow_accum_grows_downstream() {
        let (g, elev, cl) = pipeline(4, 0x1234_5678);
        let cfg = HydrologyConfig::default();
        let prior = prior_layers(&g, &elev, &cl, cfg, 0x1234_5678);
        let ocean = &prior[0];
        let layer = RiverStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elev,
            climate: &cl,
            world_seed: 0x1234_5678,
            cfg,
            prior: &prior,
        });
        // A face's full flow_accum is transferred to its downstream
        // target — which keeps its own positive base flow — so a
        // non-ocean downstream face must have at least as much flow.
        for f in 0..g.face_count() {
            let dir = layer.flow_dir[f];
            if dir != NO_FLOW && ocean.kind[dir as usize] == water_kind::NONE {
                assert!(
                    layer.flow_accum[dir as usize] >= layer.flow_accum[f],
                    "downstream flow_accum must not shrink",
                );
            }
            assert!(layer.flow_accum[f] >= 0.0);
        }
    }

    #[test]
    fn ocean_faces_carry_no_flow() {
        let (g, elev, cl) = pipeline(3, 0xAAAA);
        let cfg = HydrologyConfig::default();
        let prior = prior_layers(&g, &elev, &cl, cfg, 0xAAAA);
        let ocean = &prior[0];
        let layer = RiverStrategy.compute(&HydrologyInput {
            grid: &g,
            elevation: &elev,
            climate: &cl,
            world_seed: 0xAAAA,
            cfg,
            prior: &prior,
        });
        // Ocean is the only true terminal sink: flow drains into it and
        // disappears. (Lakes, by contrast, pass flow downstream.)
        for f in 0..g.face_count() {
            if ocean.kind[f] != water_kind::NONE {
                assert_eq!(layer.flow_accum[f], 0.0);
                assert_eq!(layer.flow_dir[f], NO_FLOW);
                assert_eq!(layer.kind[f], water_kind::NONE);
            }
        }
    }
}
