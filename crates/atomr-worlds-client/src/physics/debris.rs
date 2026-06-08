//! Debris rigid bodies — the falling chunks a fracture produces.
//!
//! A debris body is an *ephemeral, client-side* rigid body. Its canonical
//! voxels are removed from the world through the host (see [`super::fracture`]);
//! this body never flows back into `GetBrick`. Mass comes from the
//! engine-agnostic [`DebrisBody`] (per-material densities); the collider and the
//! render mesh both reuse the greedy box decomposition.

use atomr_worlds_core::coord::{DVec3, IVec3 as VoxCoord};
use atomr_worlds_core::default_physics_palette;
use atomr_worlds_physics::box_merge::greedy_boxes;
use atomr_worlds_physics::DebrisBody;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collider_gen::compound_from_boxes;
use super::config::PhysicsConfig;
use crate::modes::fp::MaterialPool;

/// A floating island extracted from the voxel grid, as a dense local material
/// grid — the bridge between flood-fill output and a rapier body.
pub(crate) struct Island {
    /// World voxel coordinate of the local grid's `(0,0,0)` corner.
    pub origin: VoxCoord,
    /// Local grid extent in voxels `(nx, ny, nz)`.
    pub dims: [u32; 3],
    /// Material id per local cell (`0` = empty), `(x*ny*nz + y*nz + z)` order.
    pub material: Vec<u16>,
}

/// Marker + lifetime bookkeeping on a debris rigid body.
#[derive(Component)]
pub struct Debris {
    /// Seconds since spawn.
    pub age: f32,
}

/// Despawn debris once it has settled (rapier put it to sleep), fallen below a
/// kill plane, or outlived its lifetime cap — keeps ephemeral bodies bounded.
pub fn settle_and_despawn_debris(
    time: Res<Time>,
    mut q: Query<(Entity, &mut Debris, &Transform, Option<&Sleeping>)>,
    mut commands: Commands,
) {
    const MAX_AGE_S: f32 = 30.0;
    const KILL_Y: f32 = -512.0;
    const MIN_SETTLE_S: f32 = 1.0;
    for (e, mut d, tf, sleeping) in &mut q {
        d.age += time.delta_secs();
        let asleep = sleeping.map(|s| s.sleeping).unwrap_or(false);
        let settled = asleep && d.age > MIN_SETTLE_S;
        if settled || d.age > MAX_AGE_S || tf.translation.y < KILL_Y {
            commands.entity(e).despawn();
        }
    }
}

/// Spawn one debris rigid body from an [`Island`]. Returns the entity, or
/// `None` if the island has no solid voxels.
pub(crate) fn spawn_island(
    island: &Island,
    cfg: &PhysicsConfig,
    material_pool: &MaterialPool,
    meshes: &mut Assets<Mesh>,
    commands: &mut Commands,
) -> Option<Entity> {
    let [nx, ny, nz] = island.dims;
    let dims_i = [nx as i32, ny as i32, nz as i32];
    let lin = |x: i32, y: i32, z: i32| (x * ny as i32 * nz as i32 + y * nz as i32 + z) as usize;

    let boxes = greedy_boxes(dims_i, |x, y, z| island.material[lin(x, y, z)] != 0);
    let collider = compound_from_boxes(&boxes, cfg.voxel_size_m)?;

    // Mass from per-material densities (exercises the inertia/debris core).
    let palette = default_physics_palette();
    let world_origin_m = DVec3::new(
        island.origin.x as f64 * cfg.voxel_size_m as f64,
        island.origin.y as f64 * cfg.voxel_size_m as f64,
        island.origin.z as f64 * cfg.voxel_size_m as f64,
    );
    let body = DebrisBody::from_voxels(
        island.origin,
        island.dims,
        island.material.clone(),
        cfg.voxel_size_m as f64,
        world_origin_m,
        &palette,
    );
    // Keep a floor so a degenerate single-voxel body still simulates.
    let mass_kg = (body.mass.mass_kg as f32).max(1.0);

    // Friction / restitution from the dominant (first solid) material.
    let dominant = island.material.iter().copied().find(|&m| m != 0).unwrap_or(1);
    let props = palette.get(dominant);

    // Render: one cuboid mesh per greedy box, colored from the existing
    // per-material `StandardMaterial` pool. Pre-build child bundles so the
    // `Assets<Mesh>` borrow doesn't tangle with the `with_children` closure.
    let vs = cfg.voxel_size_m;
    let children: Vec<(Mesh3d, MeshMaterial3d<StandardMaterial>, Transform)> = boxes
        .iter()
        .map(|b| {
            let size = b.size();
            let mesh = meshes.add(Cuboid::new(
                size[0] as f32 * vs,
                size[1] as f32 * vs,
                size[2] as f32 * vs,
            ));
            let center = (Vec3::new(b.min[0] as f32, b.min[1] as f32, b.min[2] as f32)
                + Vec3::new(size[0] as f32, size[1] as f32, size[2] as f32) * 0.5)
                * vs;
            let mat_id = island.material[lin(b.min[0], b.min[1], b.min[2])];
            let mat = material_pool.handle_for(mat_id).cloned().unwrap_or_default();
            (
                Mesh3d(mesh),
                MeshMaterial3d(mat),
                Transform::from_translation(center),
            )
        })
        .collect();

    let world_origin = Vec3::new(
        island.origin.x as f32,
        island.origin.y as f32,
        island.origin.z as f32,
    ) * vs;

    let entity = commands
        .spawn((
            Transform::from_translation(world_origin),
            Visibility::Visible,
            RigidBody::Dynamic,
            collider,
            AdditionalMassProperties::Mass(mass_kg),
            Friction::coefficient(props.friction),
            Restitution::coefficient(props.restitution),
            Debris { age: 0.0 },
        ))
        .with_children(|p| {
            for c in children {
                p.spawn(c);
            }
        })
        .id();
    Some(entity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use bevy::ecs::world::CommandQueue;

    fn spawn(world: &World, island: &Island) -> (CommandQueue, Option<Entity>) {
        let cfg = PhysicsConfig::default();
        let pool = MaterialPool {
            handles: vec![Handle::default(), Handle::default()],
        };
        let mut meshes = Assets::<Mesh>::default();
        let mut queue = CommandQueue::default();
        let ent = {
            let mut commands = Commands::new(&mut queue, world);
            spawn_island(island, &cfg, &pool, &mut meshes, &mut commands)
        };
        (queue, ent)
    }

    #[test]
    fn empty_island_spawns_nothing() {
        let world = World::new();
        let island = Island {
            origin: VoxCoord::new(0, 0, 0),
            dims: [2, 2, 2],
            material: vec![0; 8],
        };
        let (_q, ent) = spawn(&world, &island);
        assert!(ent.is_none());
    }

    #[test]
    fn spawns_dynamic_body_with_merged_child() {
        let mut world = World::new();
        // A 2×1×1 stone bar → greedy-merges to one box → one mesh child.
        let island = Island {
            origin: VoxCoord::new(0, 10, 0),
            dims: [2, 1, 1],
            material: vec![1, 1],
        };
        let (mut queue, ent) = spawn(&world, &island);
        let ent = ent.expect("a solid island spawns a body");
        queue.apply(&mut world);

        assert!(world.entities().contains(ent));
        assert!(world.get::<Debris>(ent).is_some());
        assert!(matches!(world.get::<RigidBody>(ent), Some(RigidBody::Dynamic)));
        assert!(world.get::<Collider>(ent).is_some());
        let n_children = world.get::<Children>(ent).map(|c| c.iter().count()).unwrap_or(0);
        assert_eq!(n_children, 1, "a 2×1×1 bar greedy-merges to one render box");
    }
}
