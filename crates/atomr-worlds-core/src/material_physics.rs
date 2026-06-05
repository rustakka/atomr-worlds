//! Per-material **physical** properties — the substrate for the voxel physics
//! subsystem (rigid-body mass, friction/restitution response, and fracture
//! yield).
//!
//! This is the physics analogue of the render-side material palette
//! (`atomr-worlds-view::MaterialEntry` / the client's `HardcodedPalette`): both
//! are indexed by the same [`Voxel`] material id (a `u16`), so material id `1`
//! is "stone" for *both* its appearance and its mass. Keeping the physics table
//! here in the pure-data core means it carries **no** dependency on Bevy,
//! rapier, or the render crates, and it is shared by both the inertia solver
//! (mass from density) and the fracture system (yield strength).
//!
//! [`Voxel`]: https://docs.rs/atomr-worlds-voxel
//!
//! # Determinism
//!
//! These values are static, seeded-at-generation read-only data. They never
//! flow into `GetBrick` output and therefore do not participate in the
//! byte-determinism contract — but [`default_palette`] is itself a pure
//! function (identical output every call), so any physics quantity *derived*
//! from it on one machine is reproducible on another given the same voxel set.

use serde::{Deserialize, Serialize};

/// Canonical material ids, mirroring the 11-entry render palette
/// (`0` = air sentinel … `10` = ice). Kept in lock-step with the render
/// `HardcodedPalette` so a single `u16` keys both appearance and physics.
pub mod material_id {
    pub const AIR: u16 = 0;
    pub const STONE: u16 = 1;
    pub const DIRT: u16 = 2;
    pub const SAND: u16 = 3;
    pub const SNOW: u16 = 4;
    pub const WATER: u16 = 5;
    pub const GRASS: u16 = 6;
    pub const WOOD: u16 = 7;
    pub const LEAVES: u16 = 8;
    pub const GLOW_ROCK: u16 = 9;
    pub const ICE: u16 = 10;
}

/// Physical properties of a single material.
///
/// - `density_kg_m3` drives rigid-body mass and the inertia tensor
///   (`mass = density × voxel_volume`). `0.0` marks a massless material
///   (air / fluids) that contributes no mass to a body.
/// - `friction` is the Coulomb friction coefficient at a contact (combined
///   with the other body's coefficient by the solver).
/// - `restitution` is the bounciness / coefficient of restitution in `[0, 1]`.
/// - `yield_strength_pa` is the stress (pascals) above which a structural link
///   fails; consumed by the fracture system, *not* by the contact solver.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialPhysicsProps {
    pub density_kg_m3: f32,
    pub friction: f32,
    pub restitution: f32,
    pub yield_strength_pa: f32,
}

impl MaterialPhysicsProps {
    /// The air / empty sentinel: massless, frictionless, no yield. Used as the
    /// fallback for unknown material ids.
    pub const AIR: Self = Self {
        density_kg_m3: 0.0,
        friction: 0.0,
        restitution: 0.0,
        yield_strength_pa: 0.0,
    };

    #[inline]
    pub const fn new(
        density_kg_m3: f32,
        friction: f32,
        restitution: f32,
        yield_strength_pa: f32,
    ) -> Self {
        Self { density_kg_m3, friction, restitution, yield_strength_pa }
    }

    /// A material with no mass (air, or a fluid handled outside the rigid-body
    /// path) contributes nothing to a body's mass or inertia.
    #[inline]
    pub const fn is_massless(self) -> bool {
        self.density_kg_m3 <= 0.0
    }
}

impl Default for MaterialPhysicsProps {
    #[inline]
    fn default() -> Self {
        Self::AIR
    }
}

/// A palette of [`MaterialPhysicsProps`] indexed by material id (the `u16` in
/// `Voxel`). Lookups for ids past the end of the table fall back to
/// [`MaterialPhysicsProps::AIR`], so an out-of-range id is treated as empty
/// rather than panicking.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MaterialPhysicsPalette {
    entries: Vec<MaterialPhysicsProps>,
}

impl MaterialPhysicsPalette {
    /// Build a palette from an explicit table. Entry `i` is the props for
    /// material id `i`.
    #[inline]
    pub fn new(entries: Vec<MaterialPhysicsProps>) -> Self {
        Self { entries }
    }

    /// Look up a material's physics props. Unknown / out-of-range ids return
    /// [`MaterialPhysicsProps::AIR`].
    #[inline]
    pub fn get(&self, id: u16) -> MaterialPhysicsProps {
        self.entries.get(id as usize).copied().unwrap_or_default()
    }

    /// Number of populated material slots.
    #[inline]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for MaterialPhysicsPalette {
    #[inline]
    fn default() -> Self {
        default_palette()
    }
}

/// The built-in physics palette for the 11 stock materials, with plausible
/// real-world-ish values (densities in kg/m³, yields in pascals). It is a pure
/// function — identical output on every call and every machine.
///
/// Ordering invariants the unit tests pin (so callers can rely on them): stone
/// is denser than wood, wood denser than snow; ice is low-friction; water is
/// massless-for-rigid-bodies (fluids are handled outside the rigid-body path).
pub fn default_palette() -> MaterialPhysicsPalette {
    use MaterialPhysicsProps as P;
    // id:                density  friction  restitution  yield (Pa)
    MaterialPhysicsPalette::new(vec![
        P::AIR,                                  // 0  air (sentinel)
        P::new(2600.0, 0.70, 0.10, 1.5e7),       // 1  stone
        P::new(1500.0, 0.60, 0.05, 1.0e5),       // 2  dirt
        P::new(1600.0, 0.55, 0.05, 5.0e4),       // 3  sand (loose)
        P::new(300.0, 0.30, 0.10, 1.0e5),        // 4  snow
        P::new(1000.0, 0.02, 0.00, 0.0),         // 5  water (fluid: no rigid yield)
        P::new(450.0, 0.60, 0.10, 5.0e4),        // 6  grass / turf
        P::new(700.0, 0.50, 0.30, 4.0e7),        // 7  wood
        P::new(120.0, 0.40, 0.20, 1.0e4),        // 8  leaves
        P::new(2600.0, 0.70, 0.10, 1.5e7),       // 9  glow_rock (rock-like)
        P::new(917.0, 0.05, 0.10, 1.0e6),        // 10 ice (low friction)
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_returns_expected_materials() {
        let p = default_palette();
        assert_eq!(p.get(material_id::STONE).density_kg_m3, 2600.0);
        assert_eq!(p.get(material_id::WOOD).density_kg_m3, 700.0);
        assert_eq!(p.get(material_id::ICE).friction, 0.05);
    }

    #[test]
    fn air_is_massless_and_is_the_default() {
        let p = default_palette();
        assert!(p.get(material_id::AIR).is_massless());
        assert_eq!(MaterialPhysicsProps::default(), MaterialPhysicsProps::AIR);
    }

    #[test]
    fn unknown_id_falls_back_to_air() {
        let p = default_palette();
        assert_eq!(p.get(9999), MaterialPhysicsProps::AIR);
        // One past the last real entry.
        assert_eq!(p.get(p.len() as u16), MaterialPhysicsProps::AIR);
    }

    #[test]
    fn density_ordering_is_physically_sane() {
        let p = default_palette();
        let d = |id| p.get(id).density_kg_m3;
        assert!(d(material_id::STONE) > d(material_id::WOOD));
        assert!(d(material_id::WOOD) > d(material_id::SNOW));
        assert!(d(material_id::SNOW) > d(material_id::AIR));
        // Water is denser than ice (ice floats).
        assert!(d(material_id::WATER) > d(material_id::ICE));
    }

    #[test]
    fn default_palette_is_deterministic() {
        assert_eq!(default_palette(), default_palette());
        assert_eq!(default_palette().len(), 11);
    }

    #[test]
    fn serde_round_trip() {
        let p = default_palette();
        let json = serde_json::to_string(&p).unwrap();
        let back: MaterialPhysicsPalette = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }
}
