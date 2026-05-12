//! Generator impls for each tier of the hierarchy.
//!
//! Phase 1: tiers above World pass their seed through into the next tier's
//! struct; World ships a `TerrainGenerator` via the `WorldGen::brick_gen`
//! accessor. Higher-tier content (galaxy density fields, system bodies)
//! lands in a later phase.

use atomr_worlds_core::addr::{Level, WorldAddr};
use atomr_worlds_core::dim::DimensionId;
use atomr_worlds_core::hierarchy::{Galaxy, Generator, Sector, System, Universe, World};
use atomr_worlds_core::lod::MetricScale;

use crate::error::GenerateError;
use crate::terrain::{TerrainConfig, TerrainGenerator};

#[derive(Debug, Clone)]
pub struct UniverseGen {
    pub scale: MetricScale,
}

impl Default for UniverseGen {
    fn default() -> Self {
        Self { scale: MetricScale::DEFAULT_UNIVERSE }
    }
}

impl Generator for UniverseGen {
    type Output = Universe;
    type Err = GenerateError;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<Universe, GenerateError> {
        Ok(Universe { seed, dim: addr.universe.dim as DimensionId, scale: self.scale })
    }
}

#[derive(Debug, Clone)]
pub struct GalaxyGen {
    pub scale: MetricScale,
}

impl Default for GalaxyGen {
    fn default() -> Self {
        Self { scale: MetricScale::DEFAULT_GALAXY }
    }
}

impl Generator for GalaxyGen {
    type Output = Galaxy;
    type Err = GenerateError;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<Galaxy, GenerateError> {
        Ok(Galaxy { addr: addr.ancestor(Level::Galaxy), seed, scale: self.scale })
    }
}

#[derive(Debug, Clone)]
pub struct SectorGen {
    pub scale: MetricScale,
}

impl Default for SectorGen {
    fn default() -> Self {
        Self { scale: MetricScale::DEFAULT_SECTOR }
    }
}

impl Generator for SectorGen {
    type Output = Sector;
    type Err = GenerateError;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<Sector, GenerateError> {
        Ok(Sector { addr: addr.ancestor(Level::Sector), seed, scale: self.scale })
    }
}

#[derive(Debug, Clone)]
pub struct SystemGen {
    pub scale: MetricScale,
}

impl Default for SystemGen {
    fn default() -> Self {
        Self { scale: MetricScale::DEFAULT_SYSTEM }
    }
}

impl Generator for SystemGen {
    type Output = System;
    type Err = GenerateError;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<System, GenerateError> {
        Ok(System { addr: addr.ancestor(Level::System), seed, scale: self.scale })
    }
}

#[derive(Debug, Clone)]
pub struct WorldGen {
    pub scale: MetricScale,
    pub terrain: TerrainConfig,
}

impl Default for WorldGen {
    fn default() -> Self {
        Self { scale: MetricScale::DEFAULT_WORLD, terrain: TerrainConfig::default() }
    }
}

impl WorldGen {
    pub fn brick_gen(&self) -> TerrainGenerator {
        TerrainGenerator::new(self.terrain)
    }
}

impl Generator for WorldGen {
    type Output = World;
    type Err = GenerateError;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<World, GenerateError> {
        Ok(World { addr, seed, scale: self.scale })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn each_tier_passes_seed_through() {
        let addr = WorldAddr::ROOT;
        assert_eq!(UniverseGen::default().generate(1, addr).unwrap().seed, 1);
        assert_eq!(GalaxyGen::default().generate(2, addr).unwrap().seed, 2);
        assert_eq!(SectorGen::default().generate(3, addr).unwrap().seed, 3);
        assert_eq!(SystemGen::default().generate(4, addr).unwrap().seed, 4);
        assert_eq!(WorldGen::default().generate(5, addr).unwrap().seed, 5);
    }
}
