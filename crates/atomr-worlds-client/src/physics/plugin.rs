//! [`PhysicsPlugin`] — wires rapier + the static-collider and debris systems.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::config::PhysicsConfig;
use crate::modes::fp::{BrickFadeOut, BrickLod};
use crate::world_stream::LoadedChunks;

pub struct PhysicsPlugin;

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<PhysicsConfig>() {
            app.insert_resource(PhysicsConfig::default());
        }
        // When physics is disabled (harness / `--physics off`), add nothing —
        // no rapier plugin, no systems, zero cost.
        if !app.world().resource::<PhysicsConfig>().enabled {
            return;
        }
        app.add_plugins(RapierPhysicsPlugin::<NoUserData>::default());
        app.init_resource::<super::character::CharacterState>();
        app.init_resource::<super::character::CharacterIntent>();
        // Construct the off-thread fracture scheduler once the tokio runtime
        // exists (mirrors `init_brick_gen_workers`).
        app.add_systems(Startup, super::fracture::init_fracture_workers);
        app.add_systems(
            Update,
            (
                sync_rapier_gravity,
                attach_brick_colliders,
                detach_brick_colliders,
                // Carve fracture is split: dispatch the analysis off-thread
                // after this frame's edit emits its event, then apply finished
                // analyses (debris + carve + brick refresh) on the main thread.
                super::fracture::dispatch_fracture_checks
                    .after(crate::modes::edit::fp_edit_voxels),
                super::fracture::apply_fracture_results
                    .after(super::fracture::dispatch_fracture_checks),
                super::debris::settle_and_despawn_debris,
                super::character::spawn_player,
            ),
        );
        // Character controller: set desired movement before rapier steps, read
        // the resolved pose back after its writeback. `drive_character` runs
        // after the FP input system (so it reads this frame's intent) and
        // `writeback_character` before `fp_update_motion_state` (which the FP
        // chain runs before `fp_sync_camera`), so the camera + motion EWMAs see
        // the resolved position the same frame.
        app.add_systems(
            Update,
            (
                super::character::drive_character
                    .after(crate::modes::fp::world_walk_input)
                    .before(PhysicsSet::SyncBackend),
                super::character::writeback_character
                    .after(PhysicsSet::Writeback)
                    .before(crate::modes::fp::fp_update_motion_state),
            ),
        );
    }
}

/// Mirror the configured gravity onto the rapier context. In bevy_rapier 0.34
/// `RapierConfiguration` is a per-world Component spawned during the plugin's
/// setup, so we sync each frame (writing only on change) to apply it as soon as
/// the context exists and to honour any later config edit.
fn sync_rapier_gravity(cfg: Res<PhysicsConfig>, mut q: Query<&mut RapierConfiguration>) {
    for mut rc in &mut q {
        if rc.gravity != cfg.gravity {
            rc.gravity = cfg.gravity;
        }
    }
}

/// Attach a static collider to every LOD-0 brick entity that doesn't have one
/// yet, built from its resident decoded voxels. Coarse LODs and empty bricks
/// are skipped for free (no resident brick / `None` collider).
///
/// `ColliderScale::Absolute(ONE)` makes the collider ignore the parent's
/// fade-in scale tween, so it stays full-size from the moment it attaches.
fn attach_brick_colliders(
    cfg: Res<PhysicsConfig>,
    loaded: Res<LoadedChunks>,
    q: Query<(Entity, &BrickLod), Without<Collider>>,
    mut commands: Commands,
) {
    for (ent, lod) in &q {
        if lod.depth != 0 {
            continue;
        }
        let Some(chunk) = loaded.get(&(lod.coord, 0)) else {
            continue;
        };
        let Some(brick) = chunk.brick.as_ref() else {
            continue;
        };
        let Some(collider) = cfg.collider.build(brick, cfg.voxel_size_m) else {
            continue;
        };
        commands.entity(ent).insert((
            RigidBody::Fixed,
            collider,
            ColliderScale::Absolute(Vec3::ONE),
        ));
    }
}

/// Strip the collider as soon as a brick begins fading out, so a shrinking /
/// soon-despawned brick stops colliding immediately.
fn detach_brick_colliders(
    q: Query<Entity, (With<BrickFadeOut>, With<Collider>)>,
    mut commands: Commands,
) {
    for ent in &q {
        commands
            .entity(ent)
            .remove::<(RigidBody, Collider, ColliderScale)>();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_core::lod::Lod;
    use atomr_worlds_voxel::brick::Brick;
    use atomr_worlds_voxel::voxel::Voxel;
    use std::sync::Arc;

    use crate::world_stream::{LoadedChunk, LoadedChunks};

    fn solid_brick() -> Brick {
        let mut b = Brick::new();
        for x in 0..4 {
            for y in 0..4 {
                for z in 0..4 {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        b
    }

    fn loaded_with(coord: IVec3, depth: u8, entity: Entity, brick: Option<Arc<Brick>>) -> LoadedChunks {
        let mut loaded = LoadedChunks::default();
        loaded.insert(
            LoadedChunk::key(coord, Lod::new(depth)),
            LoadedChunk {
                coord,
                lod: Lod::new(depth),
                entity: Some(entity),
                last_seen_frame: 0,
                is_fading_out: false,
                dag_digest: None,
                dag_tier: None,
                brick,
            },
        );
        loaded
    }

    /// `attach_brick_colliders` gives a resident LOD-0 brick entity a fixed
    /// collider built from its voxels.
    #[test]
    fn attaches_fixed_collider_to_resident_lod0_brick() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(PhysicsConfig::default());

        let coord = IVec3::new(0, 0, 0);
        let ent = app.world_mut().spawn(BrickLod { coord, depth: 0 }).id();
        app.insert_resource(loaded_with(coord, 0, ent, Some(Arc::new(solid_brick()))));
        app.add_systems(Update, attach_brick_colliders);
        app.update();

        assert!(app.world().get::<Collider>(ent).is_some(), "LOD-0 brick gets a collider");
        assert!(matches!(app.world().get::<RigidBody>(ent), Some(RigidBody::Fixed)));
    }

    /// Coarse-LOD bricks are never given colliders (leaf-LOD-only).
    #[test]
    fn skips_coarse_lod_bricks() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(PhysicsConfig::default());

        let coord = IVec3::new(0, 0, 0);
        let ent = app.world_mut().spawn(BrickLod { coord, depth: 1 }).id();
        // Even with a resident brick, depth != 0 must be skipped.
        app.insert_resource(loaded_with(coord, 1, ent, Some(Arc::new(solid_brick()))));
        app.add_systems(Update, attach_brick_colliders);
        app.update();

        assert!(app.world().get::<Collider>(ent).is_none(), "coarse LOD stays collider-free");
    }
}
