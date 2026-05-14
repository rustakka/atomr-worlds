//! Ocean strategy — faces whose geological elevation is below sea level.
//!
//! The simplest of the three water-body strategies: a pure per-face
//! threshold against `sea_level_m`, with the water surface pinned to sea
//! level. Runs first; [`LakeStrategy`](super::lake::LakeStrategy) seeds
//! its priority-flood from the ocean faces this produces.

use super::{water_kind, HydrologyInput, WaterBodyStrategy, WaterLayer};

/// Classifies every face below `sea_level_m` as ocean.
#[derive(Debug, Default, Clone, Copy)]
pub struct OceanStrategy;

impl WaterBodyStrategy for OceanStrategy {
    fn name(&self) -> &'static str {
        "Ocean"
    }

    fn compute(&self, input: &HydrologyInput) -> WaterLayer {
        let n = input.grid.face_count();
        let mut layer = WaterLayer::empty(n);
        let sea = input.cfg.sea_level_m;
        // Strict `<`: a face exactly at sea level is land, not ocean —
        // consistent with the `biome.rs` ocean test.
        for f in 0..n {
            if input.elevation.elev_m[f] < sea {
                layer.kind[f] = water_kind::OCEAN;
                layer.surface_m[f] = sea;
            }
        }
        layer
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::climate::ClimateField;
    use crate::macro_state::hydrology::HydrologyConfig;
    use crate::macro_state::plates::ElevationField;
    use crate::macro_state::surface_grid::SurfaceGrid;

    fn input_with<'a>(
        grid: &'a SurfaceGrid,
        elevation: &'a ElevationField,
        climate: &'a ClimateField,
        cfg: HydrologyConfig,
    ) -> HydrologyInput<'a> {
        HydrologyInput {
            grid,
            elevation,
            climate,
            world_seed: 0xABCD,
            cfg,
            prior: &[],
        }
    }

    #[test]
    fn classifies_below_sea_level_as_ocean() {
        let g = SurfaceGrid::new(2);
        let n = g.face_count();
        let mut elev_m = vec![100.0_f32; n];
        elev_m[0] = -1.0;
        elev_m[1] = -3500.0;
        elev_m[2] = 0.0; // exactly sea level → land
        let elevation = ElevationField { elev_m };
        let climate = ClimateField {
            temperature_c: vec![10.0; n],
            humidity: vec![0.5; n],
            precipitation_mm: vec![100.0; n],
        };
        let layer = OceanStrategy
            .compute(&input_with(&g, &elevation, &climate, HydrologyConfig::default()));
        assert_eq!(layer.kind[0], water_kind::OCEAN);
        assert_eq!(layer.kind[1], water_kind::OCEAN);
        assert_eq!(layer.kind[2], water_kind::NONE);
        assert_eq!(layer.kind[3], water_kind::NONE);
        assert_eq!(layer.surface_m[0], 0.0);
        assert_eq!(layer.surface_m[1], 0.0);
    }

    #[test]
    fn is_deterministic() {
        let g = SurfaceGrid::new(3);
        let n = g.face_count();
        let elevation = ElevationField {
            elev_m: (0..n).map(|i| (i as f32) - (n as f32) * 0.5).collect(),
        };
        let climate = ClimateField {
            temperature_c: vec![10.0; n],
            humidity: vec![0.5; n],
            precipitation_mm: vec![100.0; n],
        };
        let cfg = HydrologyConfig::default();
        let a = OceanStrategy.compute(&input_with(&g, &elevation, &climate, cfg));
        let b = OceanStrategy.compute(&input_with(&g, &elevation, &climate, cfg));
        assert_eq!(a.kind, b.kind);
        for i in 0..n {
            assert_eq!(a.surface_m[i].to_bits(), b.surface_m[i].to_bits());
        }
    }
}
