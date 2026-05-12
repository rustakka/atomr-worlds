//! Hierarchy primitives: data-only Universe / Galaxy / Sector / System / World
//! plus a [`Generator`] trait stub for downstream crates to implement.

use crate::addr::WorldAddr;
use crate::dim::DimensionId;
use crate::lod::MetricScale;
use crate::shape::WorldShape;

#[derive(Copy, Clone, Debug)]
pub struct Universe {
    pub seed: u64,
    pub dim: DimensionId,
    pub scale: MetricScale,
}

#[derive(Copy, Clone, Debug)]
pub struct Galaxy {
    pub addr: WorldAddr,
    pub seed: u64,
    pub scale: MetricScale,
}

#[derive(Copy, Clone, Debug)]
pub struct Sector {
    pub addr: WorldAddr,
    pub seed: u64,
    pub scale: MetricScale,
}

#[derive(Copy, Clone, Debug)]
pub struct System {
    pub addr: WorldAddr,
    pub seed: u64,
    pub scale: MetricScale,
}

#[derive(Copy, Clone, Debug)]
pub struct World {
    pub addr: WorldAddr,
    pub seed: u64,
    pub scale: MetricScale,
    pub shape: WorldShape,
}

impl World {
    /// Construct a world with the default cubic shape — preserves the
    /// pre-Phase-13 behavior. Use `World::with_shape` to opt into sphere
    /// or cylinder worlds.
    #[inline]
    pub fn new(addr: WorldAddr, seed: u64, scale: MetricScale) -> Self {
        Self { addr, seed, scale, shape: WorldShape::default_world() }
    }

    #[inline]
    pub fn with_shape(addr: WorldAddr, seed: u64, scale: MetricScale, shape: WorldShape) -> Self {
        Self { addr, seed, scale, shape }
    }
}

/// A pure function from `(seed, addr)` to a generated value.
///
/// Implementations live downstream; this is just the shape.
pub trait Generator {
    type Output;
    type Err;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<Self::Output, Self::Err>;
}
