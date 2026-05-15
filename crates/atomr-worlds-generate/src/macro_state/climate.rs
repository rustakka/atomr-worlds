//! Climate field over the surface grid: temperature, humidity, precipitation.
//!
//! Simplified model — deterministic from `(world_seed, surface_grid,
//! plates, elevation)`:
//! - Latitude ≈ `|face_centroid.y|` (poles at ±1, equator at 0).
//! - Base temperature drops linearly with latitude (equator → pole).
//! - Lapse rate: temperature decreases with altitude.
//! - Humidity: oceanic faces start at 100%, then advect downwind one
//!   neighbour-step per iteration with attenuation. Without a real wind
//!   model we use a deterministic isotropic-diffusion proxy.
//! - Precipitation = humidity attenuation per face (the amount lost in
//!   the advection step).

use super::hydrology::water_kind;
use super::plates::ElevationField;
use super::surface_grid::SurfaceGrid;

#[derive(Clone, Debug)]
pub struct ClimateField {
    pub temperature_c: Vec<f32>,
    pub humidity: Vec<f32>,        // 0..1
    pub precipitation_mm: Vec<f32>, // arbitrary units; per-year-ish
}

#[derive(Copy, Clone, Debug)]
pub struct ClimateConfig {
    pub equator_temp_c: f32,
    pub pole_temp_c: f32,
    pub lapse_rate_c_per_m: f32,
    pub humidity_iters: u8,
    pub humidity_decay: f32,    // multiplier per advection step (0..1)
    pub precip_scale: f32,
    /// Phase 18 follow-up: how many extra humidity-diffusion iterations to
    /// run *after* hydrology has identified lake / river faces. Each
    /// iteration uses the same neighbour-max diffusion as the initial
    /// pass, so freshwater bodies act as humidity sources alongside the
    /// ocean. `0` disables the feedback (preserves the pre-feedback
    /// digest); the default `2` is enough for one or two faces of bleed
    /// without flooding inland deserts.
    pub hydrology_feedback_iters: u8,
    /// Humidity value seeded at lake / river faces during the feedback
    /// pass. Defaults to `1.0` so freshwater behaves identically to the
    /// ocean seed.
    pub hydrology_feedback_seed: f32,
}

impl Default for ClimateConfig {
    fn default() -> Self {
        Self {
            equator_temp_c: 30.0,
            pole_temp_c: -25.0,
            lapse_rate_c_per_m: 0.0065,
            humidity_iters: 4,
            humidity_decay: 0.85,
            precip_scale: 600.0,
            hydrology_feedback_iters: 2,
            hydrology_feedback_seed: 1.0,
        }
    }
}

pub fn generate_climate(
    grid: &SurfaceGrid,
    elev: &ElevationField,
    cfg: ClimateConfig,
) -> ClimateField {
    let n_faces = grid.face_count();
    let temperature_c: Vec<f32> = (0..n_faces)
        .map(|f| {
            let c = grid.face_centroid(f as super::surface_grid::FaceId);
            let lat = c.y.abs() as f32; // 0..1
            let base = cfg.equator_temp_c + (cfg.pole_temp_c - cfg.equator_temp_c) * lat;
            let alt = elev.elev_m[f].max(0.0);
            base - cfg.lapse_rate_c_per_m * alt
        })
        .collect();

    // Humidity: seed from ocean (negative elevation) at 1.0, else 0.
    let mut humidity: Vec<f32> = elev
        .elev_m
        .iter()
        .map(|&e| if e < 0.0 { 1.0 } else { 0.0 })
        .collect();

    // Diffusion iterations — each face takes the max of its neighbours
    // times `humidity_decay`. Iteration order is fixed (face index), so
    // results are deterministic. Precipitation accumulates the difference
    // before vs after each step.
    let mut precipitation_mm = vec![0.0_f32; n_faces];
    for _ in 0..cfg.humidity_iters {
        let prev = humidity.clone();
        for f in 0..n_faces {
            let mut best = prev[f];
            for n in grid.neighbours_of(f as super::surface_grid::FaceId) {
                if n == super::surface_grid::FaceId::MAX {
                    continue;
                }
                let v = prev[n as usize] * cfg.humidity_decay;
                if v > best {
                    best = v;
                }
            }
            // Precipitation = (incoming - outgoing) clamped to positive.
            // We approximate by recording how much humidity arrived at this
            // step.
            let delta = (best - prev[f]).max(0.0);
            precipitation_mm[f] += delta * cfg.precip_scale;
            humidity[f] = best;
        }
    }

    ClimateField { temperature_c, humidity, precipitation_mm }
}

/// Phase 18 follow-up: bleed humidity outward from lake / river faces.
///
/// Mutates the supplied [`ClimateField`] in place. Lake and river faces
/// are stamped to `cfg.hydrology_feedback_seed`, then `cfg.hydrology_feedback_iters`
/// extra neighbour-max diffusion steps run with the same `humidity_decay`
/// as the initial advection pass. Precipitation receives the same delta
/// accounting so downstream "is it wet here?" consumers see the bleed.
///
/// `kinds` must be `water_kind::*` per face — the function reads it and
/// expects `OCEAN` faces to remain at humidity `1.0` from the initial
/// pass (no second seed required).
///
/// `iters == 0` is a no-op so callers can preserve the pre-Phase-18-feedback
/// digest by setting [`ClimateConfig::hydrology_feedback_iters`] to `0`.
pub fn apply_hydrology_humidity_feedback(
    grid: &SurfaceGrid,
    climate: &mut ClimateField,
    kinds: &[u8],
    cfg: &ClimateConfig,
) {
    if cfg.hydrology_feedback_iters == 0 {
        return;
    }
    let n_faces = grid.face_count();
    debug_assert_eq!(climate.humidity.len(), n_faces);
    debug_assert_eq!(kinds.len(), n_faces);

    // Seed lake / river faces. Ocean already sits at 1.0 from the initial
    // pass; freshwater bodies didn't exist when that pass ran.
    let seed = cfg.hydrology_feedback_seed;
    for f in 0..n_faces {
        let k = kinds[f];
        if k == water_kind::LAKE || k == water_kind::RIVER {
            if climate.humidity[f] < seed {
                climate.humidity[f] = seed;
            }
        }
    }

    // Re-run the diffusion `iters` times, accumulating precipitation the
    // same way `generate_climate` does so the field stays consistent with
    // its documented semantics.
    for _ in 0..cfg.hydrology_feedback_iters {
        let prev = climate.humidity.clone();
        for f in 0..n_faces {
            let mut best = prev[f];
            for n in grid.neighbours_of(f as super::surface_grid::FaceId) {
                if n == super::surface_grid::FaceId::MAX {
                    continue;
                }
                let v = prev[n as usize] * cfg.humidity_decay;
                if v > best {
                    best = v;
                }
            }
            let delta = (best - prev[f]).max(0.0);
            climate.precipitation_mm[f] += delta * cfg.precip_scale;
            climate.humidity[f] = best;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::plates::{generate_plates, PlateConfig};
    use super::super::surface_grid::SurfaceGrid;
    use super::*;

    #[test]
    fn temperature_drops_with_latitude() {
        let g = SurfaceGrid::new(2);
        let (_, elev) = generate_plates(&g, 0xCAFE, PlateConfig::default());
        let c = generate_climate(&g, &elev, ClimateConfig::default());
        // Sample two faces: one near equator, one near pole.
        let mut equatorial = (0usize, f64::INFINITY);
        let mut polar = (0usize, 0.0);
        for i in 0..g.face_count() {
            let lat = g.face_centroid(i as super::super::surface_grid::FaceId).y.abs();
            if lat < equatorial.1 {
                equatorial = (i, lat);
            }
            if lat > polar.1 {
                polar = (i, lat);
            }
        }
        assert!(
            c.temperature_c[equatorial.0] > c.temperature_c[polar.0],
            "equator should be warmer than pole",
        );
    }

    #[test]
    fn deterministic_from_inputs() {
        let g = SurfaceGrid::new(2);
        let (_, elev) = generate_plates(&g, 0xABCD, PlateConfig::default());
        let a = generate_climate(&g, &elev, ClimateConfig::default());
        let b = generate_climate(&g, &elev, ClimateConfig::default());
        for i in 0..a.temperature_c.len() {
            assert_eq!(a.temperature_c[i].to_bits(), b.temperature_c[i].to_bits());
            assert_eq!(a.humidity[i].to_bits(), b.humidity[i].to_bits());
            assert_eq!(a.precipitation_mm[i].to_bits(), b.precipitation_mm[i].to_bits());
        }
    }
}
