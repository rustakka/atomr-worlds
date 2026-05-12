//! Atmosphere boundary for entity-space rendering.
//!
//! Each body (world / system / vehicle) has an [`AtmosphereRadius`] — the
//! distance from the body's center within which the host streams *that
//! body's* voxel grid to subscribers. When the observer crosses outside the
//! atmosphere, the host demotes streaming to the body's parent tier.
//!
//! Defaults are derived from the [`MetricScale`] of the body: the body
//! "occupies" half its root cube edge, and the atmosphere extends
//! [`AtmosphereRadius::DEFAULT_MULTIPLIER`] (1.25×) that distance from center.

use serde::{Deserialize, Serialize};

use crate::coord::DVec3;
use crate::lod::MetricScale;

/// Radius (meters) of the boundary that delimits "what is being rendered" for
/// an observer near a body.
#[derive(Copy, Clone, PartialEq, PartialOrd, Debug, Serialize, Deserialize)]
pub struct AtmosphereRadius(pub f64);

impl AtmosphereRadius {
    /// 1.25× body radius (where body radius = scale.root_size_m / 2). Sensible
    /// default for terrestrial planets; configurable per body.
    pub const DEFAULT_MULTIPLIER: f64 = 1.25;

    /// Construct an atmosphere radius from a metric scale and a multiplier
    /// over the body's radius (half the root cube edge).
    #[inline]
    pub fn from_scale(scale: MetricScale, multiplier: f64) -> Self {
        Self(multiplier * scale.root_size_m * 0.5)
    }

    /// Default for a body of the given scale: `DEFAULT_MULTIPLIER × radius`.
    #[inline]
    pub fn default_for(scale: MetricScale) -> Self {
        Self::from_scale(scale, Self::DEFAULT_MULTIPLIER)
    }

    /// True if the observer is outside the atmosphere given the body's center.
    #[inline]
    pub fn outside(self, observer: DVec3, center: DVec3) -> bool {
        observer.distance(center) > self.0
    }

    /// True if the observer is inside (or exactly on) the boundary.
    #[inline]
    pub fn inside(self, observer: DVec3, center: DVec3) -> bool {
        observer.distance(center) <= self.0
    }
}

impl Default for AtmosphereRadius {
    fn default() -> Self {
        Self::default_for(MetricScale::DEFAULT_WORLD)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_world_atmosphere_is_above_surface() {
        // World root_size_m = 1e7 → radius 5e6 → atmosphere 1.25 × 5e6 = 6.25e6.
        let r = AtmosphereRadius::default_for(MetricScale::DEFAULT_WORLD);
        assert!((r.0 - 6.25e6).abs() < 1.0);
    }

    #[test]
    fn outside_and_inside_are_complementary() {
        let r = AtmosphereRadius(100.0);
        let c = DVec3::ZERO;
        assert!(r.inside(DVec3::new(50.0, 0.0, 0.0), c));
        assert!(!r.outside(DVec3::new(50.0, 0.0, 0.0), c));
        assert!(r.outside(DVec3::new(200.0, 0.0, 0.0), c));
        assert!(!r.inside(DVec3::new(200.0, 0.0, 0.0), c));
    }
}
