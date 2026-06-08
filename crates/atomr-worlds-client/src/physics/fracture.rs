//! Carve → flood-fill → debris orchestration.
//!
//! Listens for [`VoxelEditEvent`]s, and for *carves* runs the deterministic
//! structural flood-fill over the affected region. Any voxel island that no
//! longer reaches an anchor becomes a falling rigid body: the body is spawned
//! client-side, and the island's canonical voxels are removed through the host
//! (journaled) and the touched bricks are synchronously refreshed so the
//! terrain shows the hole.
//!
//! The flood-fill, mass, and box-merge are all in the engine-agnostic
//! [`atomr_worlds_physics`] core; this file is only the ECS glue.

use std::collections::HashSet;

use atomr_worlds_core::coord::IVec3 as VoxCoord;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_physics::connected_components;
use atomr_worlds_proto::{Envelope, WorldRequest};
use atomr_worlds_voxel::voxel::Voxel;
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::prelude::*;

use super::config::PhysicsConfig;
use super::debris::{spawn_island, Island};
use crate::brick_gen::fetch_and_build;
use crate::modes::edit::{sample_cell, EditSpawn, VoxelEditEvent};
use crate::modes::fp::{spawn_edited_brick, MaterialPool};
use crate::render::RenderConfig;
use crate::world_runtime::WorldRuntime;
use crate::world_stream::{ChunkStreamer, LoadedChunks};

/// Voxels of skirt added around the affected-brick AABB before flood-fill, so a
/// small island poking just outside the carved brick is still captured.
const REGION_SKIRT: i64 = 2;
/// Hard cap on the analyzed region volume (voxels). A brush spanning more than
/// this skips fracture analysis to keep the edit off the critical path; the
/// terrain still updates normally. (~ (3 bricks)³.)
const MAX_REGION_VOXELS: i64 = 48 * 48 * 48;

#[allow(clippy::too_many_arguments)]
pub fn process_fracture_checks(
    mut edits: MessageReader<VoxelEditEvent>,
    cfg: Res<PhysicsConfig>,
    runtime: Res<WorldRuntime>,
    render_cfg: Res<RenderConfig>,
    streamer: Res<ChunkStreamer>,
    material_pool: Res<MaterialPool>,
    mut loaded: ResMut<LoadedChunks>,
    mut spawn: EditSpawn,
    mut commands: Commands,
) {
    // Only carves can detach structure; placements can't.
    let jobs: Vec<VoxelEditEvent> = edits.read().filter(|e| e.removed).cloned().collect();
    if jobs.is_empty() {
        return;
    }

    let mut islands: Vec<Island> = Vec::new();
    let mut cells_to_remove: Vec<VoxCoord> = Vec::new();
    let mut removal_addr = None;

    for job in &jobs {
        let Some((region_min, dims)) = region_for_bricks(&job.bricks) else {
            continue;
        };
        let [nx, ny, nz] = [dims[0] as i32, dims[1] as i32, dims[2] as i32];
        if (nx as i64) * (ny as i64) * (nz as i64) > MAX_REGION_VOXELS {
            continue;
        }

        let loaded_ref: &LoadedChunks = &loaded;
        let world = |x: i32, y: i32, z: i32| {
            VoxCoord::new(
                region_min.x + x as i64,
                region_min.y + y as i64,
                region_min.z + z as i64,
            )
        };
        let is_solid = |x: i32, y: i32, z: i32| sample_cell(loaded_ref, world(x, y, z)).0 != 0;
        // Anchor: a solid cell on the region's outer shell *except* the top
        // face. Anything reaching the sides / bottom is treated as still
        // attached to the surrounding world; a piece only reachable upward (an
        // overhang whose support was carved) is unanchored and falls.
        let is_anchor = |x: i32, y: i32, z: i32| {
            sample_cell(loaded_ref, world(x, y, z)).0 != 0
                && (x == 0 || x == nx - 1 || z == 0 || z == nz - 1 || y == 0)
        };

        let comps = connected_components([nx, ny, nz], is_solid, is_anchor);
        for island_cells in comps.unanchored_islands() {
            // World-space bounding box of the island.
            let mut lo = [i64::MAX; 3];
            let mut hi = [i64::MIN; 3];
            for c in &island_cells {
                let w = world(c[0], c[1], c[2]);
                let wc = [w.x, w.y, w.z];
                for a in 0..3 {
                    lo[a] = lo[a].min(wc[a]);
                    hi[a] = hi[a].max(wc[a]);
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
                let w = world(c[0], c[1], c[2]);
                let (lx, ly, lz) = (
                    (w.x - lo[0]) as u32,
                    (w.y - lo[1]) as u32,
                    (w.z - lo[2]) as u32,
                );
                material[(lx * iny * inz + ly * inz + lz) as usize] = sample_cell(loaded_ref, w).0;
                cells_to_remove.push(w);
            }
            islands.push(Island {
                origin: VoxCoord::new(lo[0], lo[1], lo[2]),
                dims: idims,
                material,
            });
            removal_addr = Some(job.addr);
        }
    }

    if islands.is_empty() {
        return;
    }

    // 1) Spawn the falling rigid bodies (additive; no world mutation).
    for island in &islands {
        spawn_island(island, &cfg, &material_pool, &mut spawn.meshes, &mut commands);
    }

    // 2) Journal-remove the island voxels so terrain shows the hole, then
    //    synchronously refresh the touched bricks (reusing the editor's
    //    make-before-break swap — no flicker).
    let Some(addr) = removal_addr else { return };
    for w in &cells_to_remove {
        let env = Envelope::new(
            0,
            addr,
            WorldRequest::WriteVoxel {
                addr,
                pos: *w,
                voxel: Voxel::EMPTY,
            },
        );
        let _ = runtime.runtime.handle().block_on(runtime.host.request(env));
    }

    let bricks: HashSet<(VoxCoord, u8)> =
        cells_to_remove.iter().map(|w| (brick_of(*w), 0u8)).collect();
    let frame = streamer.frame;
    let shading_mode = render_cfg.shading.mode();
    let raymarch_tier = render_cfg.raymarch_tier;
    for key in bricks {
        // Only refresh resident, non-fading bricks; others self-heal on re-stream.
        if loaded.get(&key).map(|c| c.is_fading_out).unwrap_or(true) {
            continue;
        }
        let ready = runtime.runtime.handle().block_on(fetch_and_build(
            runtime.host.clone(),
            render_cfg.ao.clone(),
            addr,
            key.0,
            Lod::new(0),
        ));
        spawn_edited_brick(
            ready,
            frame,
            shading_mode,
            &spawn.pool,
            &spawn.voxel_pool,
            &spawn.res,
            raymarch_tier,
            &mut spawn.cache,
            &mut spawn.stats,
            &mut spawn.meshes,
            &mut spawn.materials,
            &mut spawn.storage_buffers,
            &mut commands,
            &mut loaded,
        );
    }
}

/// AABB (min corner + dims in voxels) covering the affected bricks plus a skirt.
fn region_for_bricks(bricks: &[VoxCoord]) -> Option<(VoxCoord, [u32; 3])> {
    if bricks.is_empty() {
        return None;
    }
    let e = BRICK_EDGE as i64;
    let mut lo = [i64::MAX; 3];
    let mut hi = [i64::MIN; 3];
    for b in bricks {
        let bmin = [b.x * e, b.y * e, b.z * e];
        for a in 0..3 {
            lo[a] = lo[a].min(bmin[a]);
            hi[a] = hi[a].max(bmin[a] + e); // exclusive upper bound
        }
    }
    let min = VoxCoord::new(lo[0] - REGION_SKIRT, lo[1] - REGION_SKIRT, lo[2] - REGION_SKIRT);
    let dims = [
        (hi[0] - lo[0] + 2 * REGION_SKIRT) as u32,
        (hi[1] - lo[1] + 2 * REGION_SKIRT) as u32,
        (hi[2] - lo[2] + 2 * REGION_SKIRT) as u32,
    ];
    Some((min, dims))
}

/// Brick coordinate containing world voxel `c` (`div_euclid` so negatives floor).
#[inline]
fn brick_of(c: VoxCoord) -> VoxCoord {
    let e = BRICK_EDGE as i64;
    VoxCoord::new(c.x.div_euclid(e), c.y.div_euclid(e), c.z.div_euclid(e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_covers_one_brick_plus_skirt() {
        let e = BRICK_EDGE as i64;
        let (min, dims) = region_for_bricks(&[VoxCoord::new(0, 0, 0)]).unwrap();
        // One brick [0, 16) expanded by the skirt on both sides.
        assert_eq!(min, VoxCoord::new(-REGION_SKIRT, -REGION_SKIRT, -REGION_SKIRT));
        assert_eq!(dims, [(e + 2 * REGION_SKIRT) as u32; 3]);
    }

    #[test]
    fn region_spans_adjacent_bricks() {
        let e = BRICK_EDGE as i64;
        let (min, dims) =
            region_for_bricks(&[VoxCoord::new(0, 0, 0), VoxCoord::new(1, 0, 0)]).unwrap();
        assert_eq!(min.x, -REGION_SKIRT);
        // Two bricks along x → 2*edge span, plus skirt both ends.
        assert_eq!(dims[0], (2 * e + 2 * REGION_SKIRT) as u32);
        assert_eq!(dims[1], (e + 2 * REGION_SKIRT) as u32);
    }

    #[test]
    fn empty_brick_set_has_no_region() {
        assert!(region_for_bricks(&[]).is_none());
    }

    #[test]
    fn brick_of_floors_negative_coords() {
        assert_eq!(brick_of(VoxCoord::new(0, 0, 0)), VoxCoord::new(0, 0, 0));
        assert_eq!(brick_of(VoxCoord::new(15, 16, -1)), VoxCoord::new(0, 1, -1));
    }
}
