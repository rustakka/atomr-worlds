//! Meso-scale elevation relief.
//!
//! Tectonic plates ([`plates`](super::plates)) produce a *piecewise-flat*
//! elevation field: every face of a plate sits at exactly the plate base
//! elevation, with uplift only along convergent boundaries. That is far
//! too flat for hydrology — on a perfectly flat plate every interior face
//! is a drainage pit, so no rivers form and there are no closed basins
//! for lakes.
//!
//! This layer adds a smooth, deterministic multi-octave FBM relief on top
//! of the plate elevation, giving continents rolling large-scale relief.
//! It runs immediately after plate generation, so climate, biomes, the
//! hydrology overlay, *and* brick-level terrain all consume one coherent
//! elevation field. Land gets the full relief amplitude; the ocean floor
//! gets a gentler amount so it still reads as clearly sub-sea-level.
//!
//! Determinism: a pure function of `(grid, world_seed, cfg)` — `fbm_value`
//! sampled at face centroids. The modified `ElevationField` folds into the
//! macro-state digest exactly as before.

use atomr_worlds_noise::{fbm_value, FbmConfig};

use super::plates::ElevationField;
use super::surface_grid::{FaceId, SurfaceGrid};

/// Seed salt for the relief FBM — keeps it from aliasing the drainage
/// jitter, heightfield, or cave fields.
const RELIEF_SALT: u64 = 0x3E51_A07C_9D2F_46B8;

#[derive(Copy, Clone, Debug)]
pub struct ReliefConfig {
    /// Peak relief amplitude over land, in meters.
    pub land_relief_m: f32,
    /// Peak relief amplitude over the ocean floor, in meters.
    pub ocean_relief_m: f32,
    /// Spatial frequency over the unit sphere — low, so drainage basins
    /// span many faces.
    pub freq: f32,
    /// FBM octaves — kept low so the relief stays smooth.
    pub octaves: u8,
}

impl Default for ReliefConfig {
    fn default() -> Self {
        Self {
            land_relief_m: 850.0,
            ocean_relief_m: 400.0,
            freq: 1.7,
            octaves: 3,
        }
    }
}

/// Add smooth meso-scale relief to `elev` in place. Deterministic from
/// `(grid, world_seed, cfg)`.
pub fn apply_relief(
    grid: &SurfaceGrid,
    elev: &mut ElevationField,
    world_seed: u64,
    cfg: ReliefConfig,
) {
    let fbm_cfg = FbmConfig {
        octaves: cfg.octaves,
        lacunarity: 2.0,
        gain: 0.5,
        frequency: 1.0,
    };
    for f in 0..grid.face_count() {
        let c = grid.face_centroid(f as FaceId);
        let n = fbm_value(
            world_seed ^ RELIEF_SALT,
            c.x as f32 * cfg.freq,
            c.y as f32 * cfg.freq,
            c.z as f32 * cfg.freq,
            fbm_cfg,
        );
        // Map FBM [0, 1] → signed [-1, 1].
        let signed = n * 2.0 - 1.0;
        // Land takes the full relief; the ocean floor takes a gentler
        // amount so it stays clearly below sea level.
        let amp = if elev.elev_m[f] >= 0.0 {
            cfg.land_relief_m
        } else {
            cfg.ocean_relief_m
        };
        elev.elev_m[f] += signed * amp;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::plates::{generate_plates, PlateConfig};

    #[test]
    fn is_deterministic() {
        let g = SurfaceGrid::new(3);
        let (_, base) = generate_plates(&g, 0xCAFE_F00D, PlateConfig::default());
        let mut a = base.clone();
        let mut b = base.clone();
        apply_relief(&g, &mut a, 0xCAFE_F00D, ReliefConfig::default());
        apply_relief(&g, &mut b, 0xCAFE_F00D, ReliefConfig::default());
        for i in 0..a.elev_m.len() {
            assert_eq!(a.elev_m[i].to_bits(), b.elev_m[i].to_bits());
        }
    }

    #[test]
    fn breaks_up_piecewise_flat_plates() {
        // Plate elevation is piecewise-flat; after relief, neighbouring
        // faces should differ — otherwise rivers can never form.
        let g = SurfaceGrid::new(4);
        let (_, base) = generate_plates(&g, 0xBEEF, PlateConfig::default());
        let mut relieved = base.clone();
        apply_relief(&g, &mut relieved, 0xBEEF, ReliefConfig::default());
        let distinct: usize = {
            let mut v: Vec<u32> = relieved.elev_m.iter().map(|e| e.to_bits()).collect();
            v.sort_unstable();
            v.dedup();
            v.len()
        };
        // The raw plate field has only a handful of distinct values; the
        // relieved field should have nearly one per face.
        assert!(
            distinct > g.face_count() / 2,
            "relief should make most faces distinct ({distinct} of {})",
            g.face_count(),
        );
    }
}
