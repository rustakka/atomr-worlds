//! Pure fracture analysis: region → unanchored islands, fully baked.
//!
//! This is the off-thread-able core of the carve → debris pipeline. Given a
//! region (sized `dims`, with its `(0,0,0)` corner at world voxel `region_min`)
//! and three local-coordinate closures that read the *snapshot* of the voxels
//! around a carve, [`analyze_region`] runs the deterministic structural
//! flood-fill, then for every floating island precomputes everything the
//! main-thread spawn step needs: the dense material grid, the greedy box
//! decomposition (collider + render mesh), and the rigid-body mass properties.
//!
//! Why "fully baked": the client used to run all of this *inline on the render
//! thread* (`process_fracture_checks`), so a large carve stalled the frame. By
//! moving the flood-fill, box-merge, and mass solve here — all pure CPU over a
//! cheap `Arc<Brick>` snapshot — the client can run it on a worker thread and
//! the main thread only spawns entities from the result.
//!
//! Like the rest of this crate it is engine-agnostic and **deterministic**:
//! identical inputs yield a byte-identical [`FractureAnalysis`]
//! (see [`crate::flood_fill`] and [`crate::box_merge`] for the ordering
//! guarantees it inherits).

use atomr_worlds_core::{DVec3, IVec3, MaterialPhysicsPalette};

use crate::box_merge::{greedy_boxes, Cuboid};
use crate::debris::DebrisBody;
use crate::flood_fill::connected_components;
use crate::inertia::MassProperties;

/// One floating island, with every derived quantity the spawn step needs
/// precomputed so the caller does no pure-CPU work.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalyzedIsland {
    /// World voxel coordinate of the island's local `(0,0,0)` corner.
    pub origin: IVec3,
    /// Local grid extent in voxels `(nx, ny, nz)`.
    pub dims: [u32; 3],
    /// Material id per local cell (`0` = empty), `(x*ny*nz + y*nz + z)` order.
    pub material: Vec<u16>,
    /// Greedy box decomposition of the solid cells (collider + render mesh).
    pub boxes: Vec<Cuboid>,
    /// Mass / center-of-mass / inertia from the per-material densities.
    pub mass: MassProperties,
    /// First solid material id — drives friction / restitution at spawn.
    pub dominant_material: u16,
}

/// Result of [`analyze_region`]: the floating islands plus the flattened list of
/// world voxels to remove (journaled by the caller through the world actor).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct FractureAnalysis {
    pub islands: Vec<AnalyzedIsland>,
    /// World voxel coordinates of every island cell, to be carved out.
    pub cells_to_remove: Vec<IVec3>,
}

impl FractureAnalysis {
    /// Whether any floating island was found.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.islands.is_empty()
    }
}

/// Analyze a carved region for floating islands and bake each one.
///
/// `dims` is the region extent; `region_min` is the world voxel coordinate of
/// its local `(0,0,0)`. The three closures are queried in **local** coordinates
/// `0..dims`:
/// - `is_solid(x,y,z)` — is the cell part of a body?
/// - `is_anchor(x,y,z)` — is a solid cell fixed to the surrounding world?
/// - `material_at(x,y,z)` — the cell's material id (only called on island cells).
///
/// The caller wires these to read its voxel snapshot at `region_min + local`.
/// Pure and deterministic: no Bevy, rapier, or async types appear here.
pub fn analyze_region(
    dims: [i32; 3],
    region_min: IVec3,
    voxel_size_m: f64,
    palette: &MaterialPhysicsPalette,
    is_solid: impl Fn(i32, i32, i32) -> bool,
    is_anchor: impl Fn(i32, i32, i32) -> bool,
    material_at: impl Fn(i32, i32, i32) -> u16,
) -> FractureAnalysis {
    let comps = connected_components(dims, &is_solid, &is_anchor);

    let mut islands: Vec<AnalyzedIsland> = Vec::new();
    let mut cells_to_remove: Vec<IVec3> = Vec::new();

    for island_cells in comps.unanchored_islands() {
        // Local-coordinate bounding box of the island.
        let mut lo = [i32::MAX; 3];
        let mut hi = [i32::MIN; 3];
        for c in &island_cells {
            for a in 0..3 {
                lo[a] = lo[a].min(c[a]);
                hi[a] = hi[a].max(c[a]);
            }
        }
        let idims = [
            (hi[0] - lo[0] + 1) as u32,
            (hi[1] - lo[1] + 1) as u32,
            (hi[2] - lo[2] + 1) as u32,
        ];
        let (iny, inz) = (idims[1], idims[2]);
        let mut material = vec![0u16; (idims[0] * idims[1] * idims[2]) as usize];
        for c in &island_cells {
            let (lx, ly, lz) = (
                (c[0] - lo[0]) as u32,
                (c[1] - lo[1]) as u32,
                (c[2] - lo[2]) as u32,
            );
            material[(lx * iny * inz + ly * inz + lz) as usize] = material_at(c[0], c[1], c[2]);
            cells_to_remove.push(IVec3::new(
                region_min.x + c[0] as i64,
                region_min.y + c[1] as i64,
                region_min.z + c[2] as i64,
            ));
        }

        let origin = IVec3::new(
            region_min.x + lo[0] as i64,
            region_min.y + lo[1] as i64,
            region_min.z + lo[2] as i64,
        );

        // Greedy boxes over the island's dense grid (same linear order as
        // `material`, so `box.min` indexes back into it for per-box color).
        let idims_i = [idims[0] as i32, idims[1] as i32, idims[2] as i32];
        let lin = |x: i32, y: i32, z: i32| {
            (x * idims_i[1] * idims_i[2] + y * idims_i[2] + z) as usize
        };
        let boxes = greedy_boxes(idims_i, |x, y, z| material[lin(x, y, z)] != 0);

        // Mass from per-material densities. `world_origin_m` only sets the
        // body's world position (unused here); the mass tensor is what we keep.
        let world_origin_m = DVec3::new(
            origin.x as f64 * voxel_size_m,
            origin.y as f64 * voxel_size_m,
            origin.z as f64 * voxel_size_m,
        );
        let body = DebrisBody::from_voxels(
            origin,
            idims,
            material.clone(),
            voxel_size_m,
            world_origin_m,
            palette,
        );
        let dominant_material = material.iter().copied().find(|&m| m != 0).unwrap_or(1);

        islands.push(AnalyzedIsland {
            origin,
            dims: idims,
            material,
            boxes,
            mass: body.mass,
            dominant_material,
        });
    }

    FractureAnalysis {
        islands,
        cells_to_remove,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::{default_physics_palette, material_physics::material_id};

    /// Closure over a sparse solid set with a uniform material, mirroring the
    /// `flood_fill` test fixtures.
    fn solids(cells: &[[i32; 3]]) -> impl Fn(i32, i32, i32) -> bool + '_ {
        move |x, y, z| cells.iter().any(|&[a, b, c]| a == x && b == y && c == z)
    }

    #[test]
    fn floating_blob_is_baked_into_one_island() {
        // y=0 row anchored; a 2-voxel blob floats at y=2 (mirrors flood_fill).
        let dims = [4, 4, 1];
        let cells = [[0, 0, 0], [1, 0, 0], [2, 2, 0], [3, 2, 0]];
        let is_solid = solids(&cells);
        let is_anchor = |x: i32, y: i32, z: i32| is_solid(x, y, z) && y == 0;
        let region_min = IVec3::new(100, 200, 300);
        let palette = default_physics_palette();

        let a = analyze_region(
            dims,
            region_min,
            1.0,
            &palette,
            &is_solid,
            &is_anchor,
            |_, _, _| material_id::STONE,
        );

        assert_eq!(a.islands.len(), 1);
        let island = &a.islands[0];
        // Blob spans local x∈[2,3], y=2 → origin = region_min + [2,2,0].
        assert_eq!(island.origin, IVec3::new(102, 202, 300));
        assert_eq!(island.dims, [2, 1, 1]);
        assert_eq!(island.material, vec![material_id::STONE; 2]);
        // 2×1×1 bar greedy-merges to one box.
        assert_eq!(island.boxes.len(), 1);
        assert!(island.mass.mass_kg > 0.0);
        assert_eq!(island.dominant_material, material_id::STONE);
        // Cells to remove are the two blob voxels in world space.
        assert_eq!(
            a.cells_to_remove,
            vec![IVec3::new(102, 202, 300), IVec3::new(103, 202, 300)]
        );
    }

    #[test]
    fn fully_anchored_region_has_no_islands() {
        let dims = [3, 3, 1];
        let cells = [[0, 0, 0], [1, 0, 0], [2, 0, 0], [1, 1, 0]];
        let is_solid = solids(&cells);
        let is_anchor = |x: i32, y: i32, z: i32| is_solid(x, y, z) && y == 0;
        let palette = default_physics_palette();

        let a = analyze_region(
            dims,
            IVec3::ZERO,
            1.0,
            &palette,
            &is_solid,
            &is_anchor,
            |_, _, _| material_id::STONE,
        );
        assert!(a.is_empty());
        assert!(a.cells_to_remove.is_empty());
    }

    #[test]
    fn analysis_is_deterministic() {
        let dims = [5, 5, 2];
        let cells = [[0, 4, 0], [1, 4, 0], [4, 4, 1], [3, 0, 0]];
        let is_solid = solids(&cells);
        // Anchor nothing → every component floats.
        let is_anchor = |_: i32, _: i32, _: i32| false;
        let palette = default_physics_palette();
        let run = || {
            analyze_region(
                dims,
                IVec3::new(-7, 3, 11),
                0.5,
                &palette,
                &is_solid,
                &is_anchor,
                |_, _, _| material_id::WOOD,
            )
        };
        assert_eq!(run(), run());
    }
}
