//! Debris rigid bodies — the falling chunks a fracture produces.
//!
//! A debris body is an *ephemeral, client-side* rigid body. Its canonical
//! voxels are removed from the world through the host (see [`super::fracture`]);
//! this body never flows back into `GetBrick`. Mass comes from the
//! engine-agnostic [`DebrisBody`] (per-material densities); the collider and the
//! render mesh both reuse the greedy box decomposition.

use atomr_worlds_core::default_physics_palette;
use atomr_worlds_physics::AnalyzedIsland;
use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use super::collider_gen::compound_from_boxes;
use super::config::PhysicsConfig;
use crate::modes::fp::MaterialPool;

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

/// Spawn one debris rigid body from a baked [`AnalyzedIsland`]. Returns the
/// entity, or `None` if the island has no solid voxels.
///
/// The island arrives **fully analyzed** (its greedy boxes and mass properties
/// were precomputed off the main thread by
/// [`atomr_worlds_physics::analyze_region`]); this function only does the
/// unavoidable main-thread work — building the rapier collider, the render
/// meshes, and the entity.
pub(crate) fn spawn_island(
    island: &AnalyzedIsland,
    cfg: &PhysicsConfig,
    material_pool: &MaterialPool,
    meshes: &mut Assets<Mesh>,
    commands: &mut Commands,
) -> Option<Entity> {
    let [_nx, ny, nz] = island.dims;
    let lin = |x: i32, y: i32, z: i32| (x * ny as i32 * nz as i32 + y * nz as i32 + z) as usize;

    let collider = compound_from_boxes(&island.boxes, cfg.voxel_size_m)?;

    // Keep a floor so a degenerate single-voxel body still simulates.
    let mass_kg = (island.mass.mass_kg as f32).max(1.0);

    // Friction / restitution from the precomputed dominant material.
    let palette = default_physics_palette();
    let props = palette.get(island.dominant_material);
    let boxes = &island.boxes;

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
    use atomr_worlds_core::coord::{DVec3, IVec3 as VoxCoord};
    use atomr_worlds_physics::box_merge::greedy_boxes;
    use atomr_worlds_physics::DebrisBody;
    use bevy::ecs::world::CommandQueue;

    /// Bake an [`AnalyzedIsland`] from a raw material grid, mirroring what
    /// `analyze_region` produces (greedy boxes + mass + dominant material).
    fn analyzed(origin: VoxCoord, dims: [u32; 3], material: Vec<u16>) -> AnalyzedIsland {
        let [nx, ny, nz] = [dims[0] as i32, dims[1] as i32, dims[2] as i32];
        let lin = |x: i32, y: i32, z: i32| (x * ny * nz + y * nz + z) as usize;
        let boxes = greedy_boxes([nx, ny, nz], |x, y, z| material[lin(x, y, z)] != 0);
        let palette = atomr_worlds_core::default_physics_palette();
        let body = DebrisBody::from_voxels(origin, dims, material.clone(), 1.0, DVec3::ZERO, &palette);
        let dominant_material = material.iter().copied().find(|&m| m != 0).unwrap_or(1);
        AnalyzedIsland { origin, dims, material, boxes, mass: body.mass, dominant_material }
    }

    fn spawn(world: &World, island: &AnalyzedIsland) -> (CommandQueue, Option<Entity>) {
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
        let island = analyzed(VoxCoord::new(0, 0, 0), [2, 2, 2], vec![0; 8]);
        let (_q, ent) = spawn(&world, &island);
        assert!(ent.is_none());
    }

    #[test]
    fn spawns_dynamic_body_with_merged_child() {
        let mut world = World::new();
        // A 2×1×1 stone bar → greedy-merges to one box → one mesh child.
        let island = analyzed(VoxCoord::new(0, 10, 0), [2, 1, 1], vec![1, 1]);
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
