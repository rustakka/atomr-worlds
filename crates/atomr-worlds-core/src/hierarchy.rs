//! Hierarchy primitives: data-only Universe / Galaxy / Sector / System / World
//! plus a [`Generator`] trait stub for downstream crates to implement.

use crate::addr::WorldAddr;
use crate::dim::DimensionId;
use crate::lod::MetricScale;

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
}

/// A pure function from `(seed, addr)` to a generated value.
///
/// Implementations live downstream; this is just the shape.
pub trait Generator {
    type Output;
    type Err;
    fn generate(&self, seed: u64, addr: WorldAddr) -> Result<Self::Output, Self::Err>;
}
