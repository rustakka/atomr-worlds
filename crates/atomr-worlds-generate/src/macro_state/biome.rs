//! Biome classification from temperature + humidity + altitude.
//!
//! A small fixed-bin table. Values are intentionally coarse so the table
//! stays readable and the classification is stable under tiny climate
//! changes. Each biome id is a single byte for compact per-face storage.

use super::climate::ClimateField;
use super::plates::ElevationField;

#[derive(Clone, Debug)]
pub struct BiomeMap {
    pub biome_id: Vec<u8>,
}

/// Biome identifier — assigned per face by [`classify_biomes`]. Names
/// and ids are stable across compilations.
#[allow(clippy::module_inception)]
pub mod biome {
    pub const OCEAN: u8 = 0;
    pub const ICE: u8 = 1;
    pub const TUNDRA: u8 = 2;
    pub const TAIGA: u8 = 3;
    pub const TEMPERATE_FOREST: u8 = 4;
    pub const GRASSLAND: u8 = 5;
    pub const DESERT: u8 = 6;
    pub const SAVANNA: u8 = 7;
    pub const RAINFOREST: u8 = 8;
    pub const MOUNTAIN: u8 = 9;
}

pub fn classify_biomes(elev: &ElevationField, climate: &ClimateField) -> BiomeMap {
    let biome_id: Vec<u8> = elev
        .elev_m
        .iter()
        .zip(climate.temperature_c.iter())
        .zip(climate.humidity.iter())
        .map(|((e, t), h)| classify_one(*e, *t, *h))
        .collect();
    BiomeMap { biome_id }
}

fn classify_one(elev_m: f32, temp_c: f32, humidity: f32) -> u8 {
    if elev_m < 0.0 {
        return biome::OCEAN;
    }
    if elev_m > 3000.0 {
        return biome::MOUNTAIN;
    }
    if temp_c < -10.0 {
        return biome::ICE;
    }
    if temp_c < 0.0 {
        return biome::TUNDRA;
    }
    if temp_c < 5.0 {
        return biome::TAIGA;
    }
    // Above 5 °C: discriminate by humidity.
    if humidity < 0.15 {
        if temp_c > 20.0 { biome::DESERT } else { biome::GRASSLAND }
    } else if humidity < 0.5 {
        if temp_c > 20.0 { biome::SAVANNA } else { biome::TEMPERATE_FOREST }
    } else if temp_c > 22.0 {
        biome::RAINFOREST
    } else {
        biome::TEMPERATE_FOREST
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macro_state::climate::ClimateField;
    use crate::macro_state::plates::ElevationField;

    #[test]
    fn ocean_for_negative_elevation() {
        let elev = ElevationField { elev_m: vec![-100.0] };
        let cl = ClimateField {
            temperature_c: vec![25.0],
            humidity: vec![1.0],
            precipitation_mm: vec![0.0],
        };
        let b = classify_biomes(&elev, &cl);
        assert_eq!(b.biome_id, vec![biome::OCEAN]);
    }

    #[test]
    fn mountain_above_3000m() {
        let elev = ElevationField { elev_m: vec![3500.0] };
        let cl = ClimateField {
            temperature_c: vec![10.0],
            humidity: vec![0.5],
            precipitation_mm: vec![0.0],
        };
        let b = classify_biomes(&elev, &cl);
        assert_eq!(b.biome_id, vec![biome::MOUNTAIN]);
    }

    #[test]
    fn deterministic_classification() {
        let elev = ElevationField { elev_m: vec![100.0, 500.0, -200.0, 3500.0] };
        let cl = ClimateField {
            temperature_c: vec![25.0, -5.0, 5.0, 0.0],
            humidity: vec![0.8, 0.1, 1.0, 0.3],
            precipitation_mm: vec![0.0; 4],
        };
        let a = classify_biomes(&elev, &cl);
        let b = classify_biomes(&elev, &cl);
        assert_eq!(a.biome_id, b.biome_id);
    }
}
