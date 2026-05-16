//! Horizon-imposter shell plugin (Phase 19.2 Step 8).
//!
//! Bevy-side wiring for the polar-annulus terrain shell baked by
//! [`atomr_worlds_view::bake_polar_annulus`]. The pure baker is in the
//! view crate; this module owns the Bevy entity, the in-flight rebuild
//! handle, the macro-state cache, and the strategy-driven cadence:
//!
//! - [`HorizonShellRuntime`] — current mesh / entity handles + a
//!   `Mutex<mpsc::Receiver<HorizonImposterMesh>>` for the off-thread
//!   bake. Mirrors the `RebuildHandle` template from
//!   [`crate::world_stream::DesiredChunksCache`].
//! - [`HorizonImposterActive`] — resource read by the speed-aware layer
//!   (Step 10) to decide whether the LOD ladder can shed its outer tier
//!   and whether the rebuild-threshold strategy can widen.
//! - [`MacroStateProvider`] — lazily-computed shared
//!   `Arc<WorldMacroState>` keyed on `(seed, shape)` so the baker can
//!   sample elevation + biomes without round-tripping to the host.
//!
//! Spawn ordering matches the sky-dome plugin: the entity is created on
//! first frame that finds a [`WorldCamera`], then visibility is toggled
//! per-frame from [`HorizonImposterStrategy::enabled`]. The entity is
//! NOT parented to the camera — vertices are observer-relative XZ
//! samples on the world tangent plane, so the shell should translate
//! with the camera but not yaw / pitch with it.

use std::sync::{mpsc, Arc, Mutex};

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_generate::macro_state::{DefaultMacroGenerator, MacroConfig, MacroGenerator};
use atomr_worlds_generate::WorldMacroState;
use bevy::pbr::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::view::NoFrustumCulling;

use super::config::RenderConfig;
use super::strategy::{HorizonImposterInputs, HorizonImposterMesh};
use crate::modes::fp::{FpState, WorldCamera};
use crate::world_runtime::ActiveWorld;
use crate::world_stream::ChunkStreamer;

/// Marker on the imposter shell entity.
#[derive(Component)]
pub struct HorizonShell;

/// Per-frame state for the horizon shell: current mesh + entity handles,
/// the pose / digest the live mesh was built for, and a slot for an
/// in-flight off-thread rebuild.
#[derive(Resource, Default)]
pub struct HorizonShellRuntime {
    pub current_mesh: Option<Handle<Mesh>>,
    pub current_entity: Option<Entity>,
    /// `(observer, source_digest)` for the currently-installed mesh.
    /// `None` ⇒ no mesh installed yet.
    pub built_for: Option<(DVec3, u64)>,
    rebuild: Option<RebuildHandle>,
}

struct RebuildHandle {
    rx: Mutex<mpsc::Receiver<HorizonImposterMesh>>,
    pose: DVec3,
}

/// Set true when the imposter is enabled and at least one mesh has been
/// installed. The speed-aware layer (Step 10) reads this to decide
/// whether the LOD ladder can shed its outer tier and whether the
/// rebuild-threshold strategy can widen.
#[derive(Resource, Default, Debug, Clone, Copy)]
pub struct HorizonImposterActive(pub bool);

/// Client-side lazy cache for `Arc<WorldMacroState>` keyed on
/// `(seed, shape)`. The host's `LocalHostQuery` doesn't expose the
/// macro-state cache directly, but the macro generator is deterministic
/// — computing it client-side from the same `(seed, shape)` produces
/// the same fields the host's brick generators read. One Arc per active
/// world is fine; the imposter is the only client-side macro consumer.
#[derive(Resource, Default)]
pub struct MacroStateProvider {
    cached: Option<(u64, WorldShape, Arc<WorldMacroState>)>,
}

impl MacroStateProvider {
    /// Get (and build, on first call) the macro state for the active
    /// world. Cheap on the cached path (Arc clone); blocking on the
    /// first call (~5-30 ms at `grid_level = 4`). Called only once per
    /// world from the bake dispatch path, on a background thread.
    pub fn get(&mut self, seed: u64, shape: WorldShape) -> Arc<WorldMacroState> {
        if let Some((s, sh, ref state)) = self.cached {
            if s == seed && sh == shape {
                return state.clone();
            }
        }
        let gen = DefaultMacroGenerator::new(MacroConfig {
            grid_level: 4,
            ..MacroConfig::default()
        });
        let state = gen.generate(seed, shape);
        self.cached = Some((seed, shape, state.clone()));
        state
    }
}

/// Bevy plugin: inserts [`HorizonShellRuntime`], [`HorizonImposterActive`],
/// [`MacroStateProvider`]; registers the spawn + sync systems.
pub struct HorizonShellPlugin;

impl Plugin for HorizonShellPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<HorizonShellRuntime>()
            .init_resource::<HorizonImposterActive>()
            .init_resource::<MacroStateProvider>()
            .add_systems(Update, (ensure_horizon_shell, sync_horizon_shell).chain());
    }
}

/// Spawn the imposter entity the first time we have a camera, with an
/// empty placeholder mesh. The real bake replaces the handle once
/// `sync_horizon_shell` polls it back from the background thread.
fn ensure_horizon_shell(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut runtime: ResMut<HorizonShellRuntime>,
    cam_q: Query<Entity, With<WorldCamera>>,
) {
    if runtime.current_entity.is_some() {
        return;
    }
    if cam_q.get_single().is_err() {
        return;
    }

    let mesh_handle = meshes.add(empty_shell_mesh());
    let material = materials.add(StandardMaterial {
        // Vertex colors carry the elevation + biome lookup. White base
        // lets them through 1:1 without further tinting.
        base_color: Color::WHITE,
        unlit: true,
        cull_mode: Some(bevy::render::render_resource::Face::Back),
        ..default()
    });

    let entity = commands
        .spawn((
            PbrBundle {
                mesh: mesh_handle.clone(),
                material,
                visibility: Visibility::Hidden,
                ..default()
            },
            HorizonShell,
            // NotShadowCaster: a 16 km shell would otherwise inflate
            // the cascade frustum past every reasonable bound.
            NotShadowCaster,
            NotShadowReceiver,
            // NoFrustumCulling: vertices are observer-relative meters,
            // not world-space; Bevy's AABB-based frustum culling would
            // happily cull the whole thing.
            NoFrustumCulling,
        ))
        .id();

    runtime.current_mesh = Some(mesh_handle);
    runtime.current_entity = Some(entity);
}

/// Each frame:
/// 1. Toggle visibility based on the strategy's `enabled()` flag.
/// 2. Slide the entity's transform to the observer's XZ so the
///    observer-relative vertex coords land at the right world position
///    (Y kept at 0 so the elevation field lookups are relative to the
///    shape's natural surface, not the observer's eye height).
/// 3. Poll any in-flight off-thread rebuild and install the result.
/// 4. If the strategy is enabled and we've drifted past
///    `rebuild_drift_m`, dispatch a fresh rebuild.
#[allow(clippy::too_many_arguments)]
fn sync_horizon_shell(
    cfg: Res<RenderConfig>,
    fp: Res<FpState>,
    active: Option<Res<ActiveWorld>>,
    streamer: Option<Res<ChunkStreamer>>,
    mut runtime: ResMut<HorizonShellRuntime>,
    mut active_flag: ResMut<HorizonImposterActive>,
    mut macro_cache: ResMut<MacroStateProvider>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut q: Query<(&mut Visibility, &mut Transform, &Handle<Mesh>), With<HorizonShell>>,
) {
    let Ok((mut visibility, mut transform, mesh_handle)) = q.get_single_mut() else {
        return;
    };
    let strategy = &*cfg.horizon_imposter;
    let enabled = strategy.enabled() && fp.ready;

    // Visibility tracks `enabled() && we have a baked mesh`.
    let have_mesh = runtime.built_for.is_some();
    *visibility = if enabled && have_mesh {
        Visibility::Visible
    } else {
        Visibility::Hidden
    };
    active_flag.0 = enabled && have_mesh;

    if !enabled {
        // Drop any in-flight rebuild so a future re-enable starts fresh.
        runtime.rebuild = None;
        return;
    }

    let observer = fp.walk.observer.position;
    // Slide entity transform to the observer's XZ. Y stays at 0 so the
    // vertex elevations read directly as world-Y (the elevation field
    // is sampled relative to the shape's natural surface).
    transform.translation = Vec3::new(observer.x as f32, 0.0, observer.z as f32);

    // Poll any in-flight rebuild.
    if let Some(handle) = runtime.rebuild.as_ref() {
        let result = {
            let rx = handle.rx.lock().expect("horizon-shell rebuild rx poisoned");
            rx.try_recv()
        };
        match result {
            Ok(baked) => {
                let pose = handle.pose;
                runtime.rebuild = None;
                let new_mesh = bevy_mesh_from_imposter(&baked);
                if let Some(slot) = meshes.get_mut(mesh_handle) {
                    *slot = new_mesh;
                } else {
                    // Slot disappeared (asset eviction); re-add.
                    let new_handle = meshes.add(new_mesh);
                    runtime.current_mesh = Some(new_handle);
                }
                runtime.built_for = Some((pose, baked.source_digest));
            }
            Err(mpsc::TryRecvError::Empty) => { /* still baking */ }
            Err(mpsc::TryRecvError::Disconnected) => {
                runtime.rebuild = None;
            }
        }
    }

    // Should we dispatch a new rebuild?
    let need_rebuild = match runtime.built_for {
        None => runtime.rebuild.is_none(),
        Some((last_obs, _)) => {
            let dx = observer.x - last_obs.x;
            let dy = observer.y - last_obs.y;
            let dz = observer.z - last_obs.z;
            let drift = (dx * dx + dy * dy + dz * dz).sqrt();
            runtime.rebuild.is_none() && drift > strategy.rebuild_drift_m()
        }
    };

    if !need_rebuild {
        return;
    }

    // Compose strategy-driven radii from the active streamer's outer
    // ring + the shape's geometric horizon. Overview mode (no streamer)
    // falls back to a fixed 1 km inner so a CLI debug invocation
    // doesn't have to know about FP state.
    let streamer_outer = streamer.as_ref().map(|s| s.outer_radius_m()).unwrap_or(1024.0);
    let Some(active_world) = active.as_ref() else { return };
    let shape = active_world.shape;
    let seed = active_world.seed;
    let inner_radius_m = strategy.inner_radius_m(streamer_outer);
    let outer_radius_m = strategy.outer_radius_m(shape, observer);
    if outer_radius_m <= inner_radius_m {
        return;
    }

    // Spawn the bake off-thread. Macro state is computed once (cached);
    // bake itself is the per-rebuild cost (~5-15 ms at 32 × 128).
    let macro_state = macro_cache.get(seed, shape);
    let strategy_arc = cfg.horizon_imposter.clone();
    let (tx, rx) = mpsc::channel();
    runtime.rebuild = Some(RebuildHandle {
        rx: Mutex::new(rx),
        pose: observer,
    });
    std::thread::spawn(move || {
        let inputs = HorizonImposterInputs {
            macro_state: &macro_state,
            shape,
            observer,
            inner_radius_m,
            outer_radius_m,
        };
        let baked = strategy_arc.bake(&inputs);
        let _ = tx.send(baked);
    });
}

/// Convert a baked imposter mesh into a Bevy `Mesh`. Empty meshes
/// produce a valid empty `Mesh` (drawing it is a no-op).
fn bevy_mesh_from_imposter(baked: &HorizonImposterMesh) -> Mesh {
    let mut mesh = BevyMesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, baked.vertices.clone());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, baked.colors.clone());
    // Flat-up normals — the shell is a polar annulus draped near the
    // tangent plane; per-vertex normals would smear the elevation
    // shading. Future work could derive proper normals from the
    // adjacent ring/sector triangles, but the vertex colors already
    // carry the elevation + biome signal that's the point of the
    // imposter.
    let normals = vec![[0.0, 1.0, 0.0]; baked.vertices.len()];
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_indices(Indices::U32(baked.indices.clone()));
    mesh
}

fn empty_shell_mesh() -> Mesh {
    let mut mesh = BevyMesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, Vec::<[f32; 3]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, Vec::<[f32; 4]>::new());
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, Vec::<[f32; 3]>::new());
    mesh.insert_indices(Indices::U32(Vec::new()));
    mesh
}
