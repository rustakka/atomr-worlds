//! A dynamic debris body extracted from the voxel grid.
//!
//! When the fracture system detaches a floating island (see [`crate::flood_fill`]),
//! it copies that island's voxels into a small body-local dense grid and hands
//! it here. [`DebrisBody`] owns that local grid plus the rigid-body state the
//! solver integrates (pose + linear/angular velocity) and the
//! [`MassProperties`] derived from the per-voxel material densities.
//!
//! This is engine-agnostic: it carries no Bevy / rapier types. The client maps
//! the pose onto a render entity and feeds the mass properties into whatever
//! solver backend is in use. The *canonical* world voxels are removed
//! separately via a journaled write through the actor — this local grid is
//! ephemeral physics state and never flows back into `GetBrick`.

use atomr_worlds_core::{DVec3, IVec3, MaterialPhysicsPalette, Quat};
use serde::{Deserialize, Serialize};

use crate::inertia::{mass_properties, MassProperties};

/// A dynamic body made of voxels. The local grid is dense (`material[i] == 0`
/// means empty) and indexed `(x*ny*nz + y*nz + z)`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DebrisBody {
    /// World voxel coordinate of the local grid's `(0,0,0)` corner at spawn.
    pub origin: IVec3,
    /// Local grid extent in voxels `(nx, ny, nz)`.
    pub dims: [u32; 3],
    /// Material id per local cell (`0` = empty), linear order.
    pub material: Vec<u16>,
    /// Edge length of one voxel in meters (leaf-LOD scale).
    pub voxel_size_m: f64,
    /// Mass / center-of-mass / inertia, derived from `material` + palette.
    pub mass: MassProperties,
    /// World position of the center of mass.
    pub position: DVec3,
    /// Orientation (body → world).
    pub orientation: Quat,
    pub linear_velocity: DVec3,
    pub angular_velocity: DVec3,
}

impl DebrisBody {
    /// Build a debris body from a local material grid, computing mass
    /// properties from the physics palette. `world_origin_m` is the world
    /// position of the local `(0,0,0)` corner; the body's `position` is set to
    /// the resulting world center of mass so the local grid is centered on the
    /// solver's frame.
    pub fn from_voxels(
        origin: IVec3,
        dims: [u32; 3],
        material: Vec<u16>,
        voxel_size_m: f64,
        world_origin_m: DVec3,
        palette: &MaterialPhysicsPalette,
    ) -> Self {
        let [nx, ny, nz] = dims;
        debug_assert_eq!(material.len(), (nx * ny * nz) as usize);
        let voxel_volume = voxel_size_m * voxel_size_m * voxel_size_m;
        // Smallest principal moment we allow before inversion — scaled to the
        // body so a healthy body is untouched but a flat slab stays invertible.
        let min_principal = voxel_volume * voxel_size_m * voxel_size_m;

        let samples = local_samples(dims, &material, voxel_size_m, voxel_volume, palette);
        let mass = mass_properties(samples, min_principal);
        let position = world_origin_m + mass.com;

        Self {
            origin,
            dims,
            material,
            voxel_size_m,
            mass,
            position,
            orientation: Quat::IDENTITY,
            linear_velocity: DVec3::ZERO,
            angular_velocity: DVec3::ZERO,
        }
    }

    /// Number of solid (non-empty) voxels.
    pub fn solid_count(&self) -> usize {
        self.material.iter().filter(|&&m| m != 0).count()
    }
}

/// Yield `(local_center_m, mass_kg)` per solid voxel for the inertia solver.
fn local_samples(
    dims: [u32; 3],
    material: &[u16],
    voxel_size_m: f64,
    voxel_volume: f64,
    palette: &MaterialPhysicsPalette,
) -> Vec<(DVec3, f64)> {
    let [nx, ny, nz] = dims;
    let mut out = Vec::with_capacity(material.len());
    for x in 0..nx {
        for y in 0..ny {
            for z in 0..nz {
                let i = (x * ny * nz + y * nz + z) as usize;
                let mat = material[i];
                if mat == 0 {
                    continue;
                }
                let props = palette.get(mat);
                let m = props.density_kg_m3 as f64 * voxel_volume;
                if m <= 0.0 {
                    continue;
                }
                // Voxel center in local meters.
                let center = DVec3::new(
                    (x as f64 + 0.5) * voxel_size_m,
                    (y as f64 + 0.5) * voxel_size_m,
                    (z as f64 + 0.5) * voxel_size_m,
                );
                out.push((center, m));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::material_physics::material_id;
    use atomr_worlds_core::default_physics_palette;

    #[test]
    fn single_stone_voxel_mass_matches_density() {
        let palette = default_physics_palette();
        let body = DebrisBody::from_voxels(
            IVec3::ZERO,
            [1, 1, 1],
            vec![material_id::STONE],
            1.0,
            DVec3::ZERO,
            &palette,
        );
        // 1 m³ of stone at 2600 kg/m³.
        assert!((body.mass.mass_kg - 2600.0).abs() < 1e-6);
        assert_eq!(body.solid_count(), 1);
        // COM at the voxel center (0.5, 0.5, 0.5).
        assert!((body.position.x - 0.5).abs() < 1e-9);
    }

    #[test]
    fn empty_grid_is_massless() {
        let palette = default_physics_palette();
        let body = DebrisBody::from_voxels(
            IVec3::ZERO,
            [2, 2, 2],
            vec![0; 8],
            0.5,
            DVec3::ZERO,
            &palette,
        );
        assert_eq!(body.mass.mass_kg, 0.0);
        assert_eq!(body.solid_count(), 0);
    }

    #[test]
    fn mixed_materials_sum_masses() {
        let palette = default_physics_palette();
        // One stone + one wood voxel, 1 m³ each.
        let body = DebrisBody::from_voxels(
            IVec3::new(10, 0, 0),
            [2, 1, 1],
            vec![material_id::STONE, material_id::WOOD],
            1.0,
            DVec3::ZERO,
            &palette,
        );
        let expected = 2600.0 + 700.0;
        assert!((body.mass.mass_kg - expected).abs() < 1e-6);
        // COM pulled toward the heavier stone voxel (x < midpoint 1.0).
        assert!(body.position.x < 1.0);
    }

    #[test]
    fn round_trips_through_serde() {
        let palette = default_physics_palette();
        let body = DebrisBody::from_voxels(
            IVec3::new(1, 2, 3),
            [1, 1, 1],
            vec![material_id::ICE],
            2.0,
            DVec3::new(4.0, 5.0, 6.0),
            &palette,
        );
        let json = serde_json::to_string(&body).unwrap();
        let back: DebrisBody = serde_json::from_str(&json).unwrap();
        assert_eq!(back.mass, body.mass);
        assert_eq!(back.dims, body.dims);
    }
}
