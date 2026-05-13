//! Phase 14a — 1st-person walk, native Bevy 3D.
//!
//! - `WalkCamera` (from `atomr-worlds-view`) drives input → pose.
//! - Each frame we ensure a fixed-radius cube of bricks around the camera is
//!   loaded into Bevy as `PbrBundle`s; bricks that fall outside the cube are
//!   despawned. Greedy-meshing uses `atomr-worlds-view::mesh::greedy_mesh`.
//! - Vertex colors carry per-material RGB so we render with a single
//!   `StandardMaterial`.

use std::collections::HashMap;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_view::{
    greedy_mesh_by_material, WalkCamera, WalkInput, WorldQuery,
};
// (WorldQuery brings ground_height_m into scope.)
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::input::mouse::MouseMotion;
use bevy::pbr::FogSettings;
use bevy::prelude::*;
use bevy::render::camera::{Exposure, RenderTarget};
use bevy::render::mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use crate::render::{
    OffscreenTarget, PaletteEntryGpu, RenderConfig, ShadingMode, SkyboxRuntime, VoxelMaterial,
    VoxelMaterialExt, WorldSunMarker,
};
use crate::view_mode::ViewMode;
use crate::world_runtime::{ActiveWorld, WorldRuntime};
use crate::world_stream::{desired_chunks, ChunkStreamer, LoadedChunk, LoadedChunks};

pub struct FpPlugin;

impl Plugin for FpPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpState>()
            .init_resource::<MaterialPool>()
            .init_resource::<VoxelMaterialPool>()
            .add_systems(Startup, setup_fp_scene)
            .add_systems(
                Update,
                (
                    grab_cursor,
                    world_walk_input,
                    fp_input_look,
                    fp_sync_camera,
                    fp_stream_bricks,
                    fp_visibility_toggle,
                )
                    .chain(),
            );
    }
}

/// Marker for the 3D world camera.
#[derive(Component)]
pub struct WorldCamera;

/// Marker for the directional light.
#[derive(Component)]
struct WorldSun;

/// Marker for an entity carrying a brick mesh.
#[derive(Component)]
struct BrickMesh;

/// 1st-person walk state. Public so other view modes can read the
/// camera pose (slice/rts follow the player; tp orbits it).
#[derive(Resource)]
pub struct FpState {
    pub walk: WalkCamera,
    /// Cached starting addr; chosen at startup, never changes for now.
    pub addr: WorldAddr,
    /// Set true after `setup_fp_scene` so update systems know the resource
    /// is initialised before [`ActiveWorld`] is inserted.
    pub ready: bool,
}

impl Default for FpState {
    fn default() -> Self {
        Self {
            walk: WalkCamera::new(
                DVec3::new(8.0, 24.0, 8.0),
                ContainingFrame::World(WorldAddr::ROOT),
                16.0 / 9.0,
            ),
            addr: WorldAddr::ROOT,
            ready: false,
        }
    }
}

/// One `StandardMaterial` handle per voxel material id, populated from
/// the canonical palette in [`setup_fp_scene`]. Brick meshes are split
/// per material (see [`atomr_worlds_view::greedy_mesh_by_material`]) so
/// each material renders with its own PBR (roughness / metallic /
/// emissive / alpha) rather than fighting through a single shared
/// `base_color: WHITE` + vertex-color flatten.
#[derive(Resource, Default)]
pub struct MaterialPool {
    /// Indexed by material id. `handles[id as usize]` is the cached
    /// handle for that material. Material id 0 (air) is unused but
    /// reserved for indexing safety.
    pub handles: Vec<Handle<StandardMaterial>>,
}

impl MaterialPool {
    pub fn handle_for(&self, mat: u16) -> Option<&Handle<StandardMaterial>> {
        self.handles.get(mat as usize)
    }
}

/// One shared handle to the merged-palette `VoxelMaterial`. Populated
/// in [`setup_fp_scene`] with a storage buffer holding all palette
/// entries; the fragment shader indexes the buffer by per-vertex
/// material id (encoded in `uv.x`). Used by the
/// `PaletteVoxelMaterial` shading mode (step 8).
#[derive(Resource, Default)]
pub struct VoxelMaterialPool {
    pub handle: Option<Handle<VoxelMaterial>>,
}

#[allow(clippy::too_many_arguments)]
fn setup_fp_scene(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut voxel_materials: ResMut<Assets<VoxelMaterial>>,
    mut material_pool: ResMut<MaterialPool>,
    mut voxel_pool: ResMut<VoxelMaterialPool>,
    mut fp_state: ResMut<FpState>,
    active: Option<Res<ActiveWorld>>,
    runtime: Res<WorldRuntime>,
    offscreen: Option<Res<OffscreenTarget>>,
    render_cfg: Res<RenderConfig>,
    skybox_runtime: Res<SkyboxRuntime>,
) {
    // Probe the host for the surface height at the spawn (x, z) so we
    // land a few voxels above the ground instead of inside a hill or
    // floating in mid-air. Fall back to the legacy y=24 if the column
    // is empty / the host has nothing to say.
    let spawn_xz = (8.0_f64, 8.0_f64);
    let addr = active.as_deref().map(|a| a.addr).unwrap_or(WorldAddr::ROOT);
    // Spawn well above ground so we have room to look around and don't
    // immediately wall-of-voxels the view at pitch=-0.4. 10 voxels ≈ a
    // two-storey perch above terrain.
    let spawn_y = runtime
        .query
        .ground_height_m(&addr, [spawn_xz.0, spawn_xz.1])
        .map(|h| h as f64 + 10.0)
        .unwrap_or(34.0);

    fp_state.addr = addr;
    fp_state.walk = WalkCamera::new(
        DVec3::new(spawn_xz.0, spawn_y, spawn_xz.1),
        ContainingFrame::World(addr),
        16.0 / 9.0,
    );
    // Look slightly down so the ground is in frame.
    fp_state.walk.pitch = -0.4;
    fp_state.ready = true;

    // Bevy 0.13's AmbientLight.brightness is on a 0–100ish scale (the
    // default is 80.0) — values < 5 produce near-black back faces. The
    // earlier `1.2` value was tuned against a stale assumption. Step 4
    // replaces this with a time-of-day-driven curve.
    commands.insert_resource(AmbientLight {
        color: Color::rgb(0.85, 0.88, 1.0),
        brightness: 80.0,
    });

    // Build one StandardMaterial per palette entry. The strategy supplies
    // base_color / roughness / metallic / emissive / alpha; we map them
    // straight onto Bevy's `StandardMaterial`. Materials with alpha < 1
    // get `AlphaMode::Blend` so water and ice render with translucency.
    let palette = render_cfg.palette.palette();
    let max_id = palette.entries.iter().map(|e| e.id as usize).max().unwrap_or(0);
    let mut handles = vec![Handle::<StandardMaterial>::default(); max_id + 1];
    for entry in &palette.entries {
        let alpha_mode = if entry.alpha < 0.999 {
            AlphaMode::Blend
        } else {
            AlphaMode::Opaque
        };
        let emissive_intense = entry.emissive[0].max(entry.emissive[1]).max(entry.emissive[2]) > 0.0;
        let mat = materials.add(StandardMaterial {
            base_color: Color::rgba_linear(
                entry.base_color[0],
                entry.base_color[1],
                entry.base_color[2],
                entry.alpha,
            ),
            perceptual_roughness: entry.roughness,
            metallic: entry.metallic,
            // Emissive is in nits-ish HDR space; Bevy multiplies by a constant
            // exposure later. A factor of 2.0 on linear RGB gives a soft
            // self-lit look without blowing out at noon exposure.
            emissive: if emissive_intense {
                Color::rgb_linear(
                    entry.emissive[0] * 2.0,
                    entry.emissive[1] * 2.0,
                    entry.emissive[2] * 2.0,
                )
            } else {
                Color::BLACK
            },
            alpha_mode,
            ..default()
        });
        handles[entry.id as usize] = mat;
    }
    material_pool.handles = handles;

    // Build the merged-palette voxel material (Step 8). The storage
    // buffer has one entry per material id 0..=max_id; the shader looks
    // it up by per-vertex material id encoded in `uv.x`. Base material
    // is set to alpha-blend so translucent entries (water/ice) render
    // correctly through the same draw call.
    let mut entries: Vec<PaletteEntryGpu> = vec![PaletteEntryGpu::default(); max_id + 1];
    for e in &palette.entries {
        entries[e.id as usize] = PaletteEntryGpu {
            base_color: Vec4::new(e.base_color[0], e.base_color[1], e.base_color[2], e.alpha),
            pbr: Vec4::new(e.roughness, e.metallic, 0.0, 0.0),
            emissive: Vec4::new(e.emissive[0] * 2.0, e.emissive[1] * 2.0, e.emissive[2] * 2.0, 0.0),
        };
    }
    let voxel_mat = voxel_materials.add(VoxelMaterial {
        base: StandardMaterial {
            // Base color is white so palette[id].rgb passes through
            // unchanged; the shader sets all PBR fields per-fragment.
            base_color: Color::WHITE,
            alpha_mode: AlphaMode::Blend,
            ..default()
        },
        extension: VoxelMaterialExt { palette: entries },
    });
    voxel_pool.handle = Some(voxel_mat);

    // When the harness is active, render to the offscreen `Image` target
    // instead of the window — sidesteps the X11/hybrid-GPU presentation
    // path so PNG readback always sees the rendered pixels.
    let camera_target = offscreen
        .as_deref()
        .map(|t| RenderTarget::Image(t.image.clone()))
        .unwrap_or_default();

    let tonemap = render_cfg.tonemap.tonemapping();
    let exposure = render_cfg.tonemap.exposure();
    let mut camera_ent = commands.spawn((
        Camera3dBundle {
            camera: Camera {
                target: camera_target,
                hdr: true, // required for bloom + good tonemapping headroom
                ..default()
            },
            tonemapping: tonemap,
            exposure,
            transform: Transform::from_xyz(8.0, 26.0, 8.0).looking_to(Vec3::Z, Vec3::Y),
            ..default()
        },
        WorldCamera,
    ));
    if let Some(bloom) = render_cfg.tonemap.bloom() {
        camera_ent.insert(bloom);
    } else {
        // ensure no stale BloomSettings on hot-reload — default fields are fine.
        camera_ent.insert(BloomSettings { intensity: 0.0, ..default() });
    }
    // Cubemap skybox: starts with the 1×1×6 black placeholder; the
    // first real bake from `sync_skybox` will replace the handle once
    // the streamer's far ring is populated. Brightness starts at 0 so
    // the placeholder doesn't add visible light to the scene.
    camera_ent.insert(bevy::core_pipeline::Skybox {
        image: skybox_runtime.current_handle.clone(),
        brightness: 0.0,
    });
    // Initial fog — `sync_sky_and_fog` overrides each frame from the
    // sky strategy's current horizon color and the streamer's load
    // horizon. Insert anything non-default so the
    // `Query<&mut FogSettings>` finds the component on frame 0.
    let initial_sun = render_cfg.sun_curve.sun_state(12.0);
    let initial_horizon = render_cfg.sky.horizon_color(initial_sun);
    camera_ent.insert(render_cfg.fog.fog_settings(initial_sun, initial_horizon, None));
    let shadows_on = render_cfg.shadow.enabled();
    let cascades = render_cfg.shadow.cascade_config();
    let (shadow_depth_bias, shadow_normal_bias) = render_cfg.shadow.biases();
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                // Values are overwritten each frame by `sync_sun` based on
                // the current `WorldTime` + sun-curve strategy. Initial
                // values keep the first-frame render sensible.
                illuminance: 50_000.0,
                shadows_enabled: shadows_on,
                shadow_depth_bias,
                shadow_normal_bias,
                ..default()
            },
            transform: Transform::from_xyz(50.0, 80.0, 30.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            cascade_shadow_config: cascades,
            ..default()
        },
        WorldSun,
        WorldSunMarker,
    ));
}

fn grab_cursor(
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mode: Res<ViewMode>,
    harness: Option<Res<crate::harness::HarnessActive>>,
) {
    let Ok(mut window) = windows.get_single_mut() else { return };
    if harness.is_some() {
        // Keep cursor unlocked & visible in harness mode so synthetic
        // MouseMotion events from the harness aren't ignored by fp_input.
        if window.cursor.grab_mode != CursorGrabMode::None {
            window.cursor.grab_mode = CursorGrabMode::None;
            window.cursor.visible = true;
        }
        return;
    }
    // Only grab the cursor in fp/tp modes; release for 2D overlay modes.
    let want_grab = matches!(*mode, ViewMode::Fp | ViewMode::Tp);
    if keys.just_pressed(KeyCode::Escape) {
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
        return;
    }
    if want_grab && window.cursor.grab_mode == CursorGrabMode::None {
        // Grab on a left-click inside the window. We don't auto-grab on
        // keypress: previously holding WASD while in a menu re-locked
        // the cursor unexpectedly. Click-to-grab matches the convention
        // every other voxel game uses.
        if mouse_buttons.just_pressed(MouseButton::Left) {
            window.cursor.grab_mode = CursorGrabMode::Locked;
            window.cursor.visible = false;
        }
    } else if !want_grab && window.cursor.grab_mode != CursorGrabMode::None {
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
    }
}

/// WASD / Space / Ctrl / Shift — drives [`FpState::walk`] in any view
/// mode that uses the world walk position (FP, TP, Slice, RTS). TP
/// orbits this anchor; Slice / RTS center their raster on it. The
/// mouse-look + arrow-key look part stays in [`fp_input_look`] which is
/// FP-only (TP handles its own orbit, slice/RTS don't look).
pub fn world_walk_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: ResMut<FpState>,
) {
    if !state.ready {
        return;
    }
    // Slice and RTS pan their view by moving the walk position; TP
    // orbits it; FP walks with it. Overview has its own state.
    if !matches!(*mode, ViewMode::Fp | ViewMode::Tp | ViewMode::Slice | ViewMode::Rts) {
        return;
    }
    let dt = time.delta_seconds().min(0.05);
    let speed = if keys.pressed(KeyCode::ShiftLeft) || keys.pressed(KeyCode::ShiftRight) {
        12.0
    } else {
        4.0
    };
    let mut mv = [0.0f32, 0.0, 0.0];
    if keys.pressed(KeyCode::KeyW) {
        mv[2] += speed * dt;
    }
    if keys.pressed(KeyCode::KeyS) {
        mv[2] -= speed * dt;
    }
    // WalkCamera's `+x_local = right` rotates into world +X at yaw=0, but
    // the Bevy camera (looking +Z world) has its screen-right axis aligned
    // with world -X. So `D` (screen-right) feeds mv[0] -= and `A` feeds +=
    // to keep WASD intuitive on the visible image.
    if keys.pressed(KeyCode::KeyA) {
        mv[0] += speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        mv[0] -= speed * dt;
    }
    if keys.pressed(KeyCode::Space) {
        mv[1] += speed * dt;
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        mv[1] -= speed * dt;
    }
    let crouch = keys.pressed(KeyCode::KeyC);
    state.walk.tick(
        WalkInput { move_local: mv, yaw_delta: 0.0, pitch_delta: 0.0, crouch },
        dt,
    );
}

/// Mouse-look + arrow-key yaw/pitch. FP only; TP has its own orbit
/// (`tp_input`), slice/RTS don't rotate the walk camera.
fn fp_input_look(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: EventReader<MouseMotion>,
    time: Res<Time>,
    mut state: ResMut<FpState>,
    windows: Query<&Window, With<PrimaryWindow>>,
    harness: Option<Res<crate::harness::HarnessActive>>,
) {
    if *mode != ViewMode::Fp {
        motion.clear();
        return;
    }
    if !state.ready {
        return;
    }
    let dt = time.delta_seconds().min(0.05);

    let mut yaw_delta = 0.0f32;
    let mut pitch_delta = 0.0f32;
    let harness_active = harness.is_some();
    let cursor_locked = harness_active
        || windows
            .get_single()
            .map(|w| w.cursor.grab_mode != CursorGrabMode::None)
            .unwrap_or(false);
    if cursor_locked {
        for ev in motion.read() {
            yaw_delta -= ev.delta.x * 0.0025;
            pitch_delta -= ev.delta.y * 0.0025;
        }
    } else {
        motion.clear();
    }

    // Keyboard fallback for headless / no-grab.
    let look_speed = 1.5;
    if keys.pressed(KeyCode::ArrowLeft) {
        yaw_delta += look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowRight) {
        yaw_delta -= look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowUp) {
        pitch_delta += look_speed * dt;
    }
    if keys.pressed(KeyCode::ArrowDown) {
        pitch_delta -= look_speed * dt;
    }

    state.walk.tick(
        WalkInput { move_local: [0.0; 3], yaw_delta, pitch_delta, crouch: false },
        dt,
    );
}

fn fp_sync_camera(
    state: Res<FpState>,
    mode: Res<ViewMode>,
    mut q: Query<&mut Transform, With<WorldCamera>>,
) {
    if !state.ready {
        return;
    }
    if *mode != ViewMode::Fp {
        return;
    }
    let cam = state.walk.camera();
    let eye = Vec3::new(cam.eye[0], cam.eye[1], cam.eye[2]);
    let target = Vec3::new(cam.target[0], cam.target[1], cam.target[2]);
    if let Ok(mut t) = q.get_single_mut() {
        t.translation = eye;
        t.look_at(target, Vec3::Y);
    }
}

#[allow(clippy::too_many_arguments)]
fn fp_stream_bricks(
    state: Res<FpState>,
    runtime: Res<WorldRuntime>,
    pool: Res<MaterialPool>,
    voxel_pool: Res<VoxelMaterialPool>,
    render_cfg: Res<RenderConfig>,
    mut streamer: ResMut<ChunkStreamer>,
    mut loaded: ResMut<LoadedChunks>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut commands: Commands,
) {
    if !state.ready || pool.handles.is_empty() {
        return;
    }
    let shading_mode = render_cfg.shading.mode();

    // Bump the streamer's frame counter once per tick so hysteresis
    // measurement uses a monotonic clock.
    streamer.tick_frame();
    let frame = streamer.frame;

    // Observer position drives both ring planning and the LOD selection.
    // We use the *world* position (no eye-height offset) so brick
    // boundaries are stable when crouching.
    let observer = state.walk.observer.position;

    // Plan near + far rings. World tier defaults to an infinite horizon
    // (the cube world); spherical bodies will pass their surface horizon
    // here once the body-aware ContainingFrame plumbing lands (Phase 18).
    let horizon_m = f64::INFINITY;
    let mut desired = desired_chunks(&streamer, observer, horizon_m);

    // Mark every desired chunk as "seen this frame" so the hysteresis
    // window resets while we're inside the ring.
    for (coord, lod) in &desired {
        let key = LoadedChunk::key(*coord, *lod);
        if let Some(entry) = loaded.0.get_mut(&key) {
            entry.last_seen_frame = frame;
        }
    }

    // Despawn chunks that have been outside the desired set for longer
    // than the hysteresis window. `despawn_recursive` cascades to the
    // per-material child meshes spawned below.
    let hyst = streamer.hysteresis_ticks;
    let stale_keys: Vec<(IVec3, u8)> = loaded
        .0
        .iter()
        .filter_map(|(k, v)| {
            if frame.saturating_sub(v.last_seen_frame) >= hyst {
                Some(*k)
            } else {
                None
            }
        })
        .collect();
    for k in stale_keys {
        if let Some(chunk) = loaded.0.remove(&k) {
            if let Some(ent) = chunk.entity {
                commands.entity(ent).despawn_recursive();
            }
        }
    }

    // Load up to the streamer's per-tick budget. `desired_chunks` is
    // already sorted closest-first.
    let mut budget = streamer.policy.bricks_per_tick as usize;
    desired.retain(|(c, lod)| !loaded.0.contains_key(&LoadedChunk::key(*c, *lod)));
    for (bc, lod) in desired {
        if budget == 0 {
            break;
        }
        let key = LoadedChunk::key(bc, lod);

        // Per-LOD world-space scale: each voxel at depth L is 2^L meters
        // wide, so a brick covers `BRICK_EDGE * 2^L` meters per side and
        // the SpatialBundle inherits a uniform scale of 2^L.
        let lod_scale = (1u64 << lod.depth as u32) as f32;
        let edge_m = BRICK_EDGE as f32 * lod_scale;

        let Some(brick) = runtime.query.brick(&state.addr, bc, lod) else {
            // Empty / missing brick — record an entity-less placeholder so
            // we don't re-query every frame.
            loaded.0.insert(
                key,
                LoadedChunk { coord: bc, lod, entity: None, last_seen_frame: frame },
            );
            budget -= 1;
            continue;
        };
        let mut by_material = greedy_mesh_by_material(&brick);
        if by_material.is_empty() {
            loaded.0.insert(
                key,
                LoadedChunk { coord: bc, lod, entity: None, last_seen_frame: frame },
            );
            budget -= 1;
            continue;
        }
        // Apply the AO strategy after splitting so each per-material
        // submesh's vertices get their corner AO factors.
        for sub_mesh in by_material.values_mut() {
            render_cfg.ao.bake(sub_mesh, &brick);
        }
        let origin = Vec3::new(
            (bc.x as f32) * edge_m,
            (bc.y as f32) * edge_m,
            (bc.z as f32) * edge_m,
        );
        let parent = commands
            .spawn(SpatialBundle::from_transform(
                Transform::from_translation(origin)
                    .with_scale(Vec3::splat(lod_scale)),
            ))
            .id();
        match shading_mode {
            ShadingMode::SplitPerMaterial => {
                for (mat_id, sub_mesh) in by_material.iter() {
                    if sub_mesh.indices.is_empty() {
                        continue;
                    }
                    let Some(material) = pool.handle_for(*mat_id) else { continue };
                    let bevy_mesh = atomr_to_bevy_mesh(sub_mesh);
                    let mesh_handle = meshes.add(bevy_mesh);
                    commands.entity(parent).with_children(|p| {
                        p.spawn((
                            PbrBundle {
                                mesh: mesh_handle,
                                material: material.clone(),
                                ..default()
                            },
                            BrickMesh,
                        ));
                    });
                }
            }
            ShadingMode::PaletteVoxelMaterial => {
                let Some(voxel_handle) = voxel_pool.handle.as_ref() else {
                    loaded.0.insert(
                        key,
                        LoadedChunk { coord: bc, lod, entity: Some(parent), last_seen_frame: frame },
                    );
                    budget -= 1;
                    continue;
                };
                let merged = merge_by_material(&by_material);
                if !merged.indices().map(|i| i.is_empty()).unwrap_or(true) {
                    let mesh_handle = meshes.add(merged);
                    commands.entity(parent).with_children(|p| {
                        p.spawn((
                            MaterialMeshBundle::<VoxelMaterial> {
                                mesh: mesh_handle,
                                material: voxel_handle.clone(),
                                ..default()
                            },
                            BrickMesh,
                        ));
                    });
                }
            }
        }
        loaded.0.insert(
            key,
            LoadedChunk { coord: bc, lod, entity: Some(parent), last_seen_frame: frame },
        );
        budget -= 1;
    }

    // Diagnostic: per-LOD per-side counts AND centroid drift. If loading
    // is symmetric, the centroid of loaded brick centers should equal
    // the observer (centroid_dx,dy,dz ≈ 0). Nonzero values prove
    // directional bias. Gated on `ATOMR_STREAM_DIAG=1` so production
    // runs stay quiet; the symmetry guarantee is exercised by the
    // `desired_chunks_load_symmetrically_in_all_four_cardinal_directions`
    // unit test in `world_stream.rs`.
    if frame == 60 && std::env::var("ATOMR_STREAM_DIAG").is_ok() {
        // One-shot terrain height probe at 16 cardinal/diagonal positions
        // around the observer at radii 256, 512, 1024. If `ground_height_m`
        // is heavily biased to one quadrant, that's a noise/terrain
        // generator issue — NOT a streaming bug.
        let mut sum = [0f64; 4]; // +X+Z, +X-Z, -X+Z, -X-Z
        let mut cnt = [0u32; 4];
        let radii = [256.0_f64, 512.0, 1024.0];
        let dirs: [(f64, f64); 8] = [
            (1.0, 0.0), (0.7071, 0.7071), (0.0, 1.0), (-0.7071, 0.7071),
            (-1.0, 0.0), (-0.7071, -0.7071), (0.0, -1.0), (0.7071, -0.7071),
        ];
        for r in radii {
            for (dx, dz) in dirs {
                let x = observer.x + dx * r;
                let z = observer.z + dz * r;
                if let Some(h) = runtime.query.ground_height_m(&state.addr, [x, z]) {
                    let qi = match (x - observer.x >= 0.0, z - observer.z >= 0.0) {
                        (true, true) => 0,
                        (true, false) => 1,
                        (false, true) => 2,
                        (false, false) => 3,
                    };
                    sum[qi] += h as f64;
                    cnt[qi] += 1;
                    eprintln!("TERRAIN_PROBE r={} dir=({:.2},{:.2}) world=({:.1},{:.1}) h={:.2}", r, dx, dz, x, z, h);
                }
            }
        }
        for i in 0..4 {
            let avg = if cnt[i] > 0 { sum[i] / cnt[i] as f64 } else { f64::NAN };
            eprintln!("TERRAIN_QUAD q{}: avg_h={:.2} n={}", i, avg, cnt[i]);
        }
    }
    if frame % 60 == 0 && std::env::var("ATOMR_STREAM_DIAG").is_ok() {
        // Per-quadrant counts of MESH-bearing entities (entity: Some(_)),
        // i.e., the bricks that actually contribute geometry to the render.
        // Empty bricks (entity: None) are excluded so we measure what's
        // actually drawn.
        let mut q_with_mesh = [0u32; 4]; // 0:+X+Z 1:+X-Z 2:-X+Z 3:-X-Z
        let mut q_no_mesh = [0u32; 4];
        let mut sum_cx = 0.0f64;
        let mut sum_cz = 0.0f64;
        let mut count_mesh = 0u32;
        for ((coord, depth), chunk) in loaded.0.iter() {
            let edge_m = BRICK_EDGE as f64 * (1u64 << *depth as u32) as f64;
            let cx = (coord.x as f64 + 0.5) * edge_m - observer.x;
            let cz = (coord.z as f64 + 0.5) * edge_m - observer.z;
            let qi = match (cx >= 0.0, cz >= 0.0) {
                (true, true) => 0,
                (true, false) => 1,
                (false, true) => 2,
                (false, false) => 3,
            };
            if chunk.entity.is_some() {
                q_with_mesh[qi] += 1;
                sum_cx += cx;
                sum_cz += cz;
                count_mesh += 1;
            } else {
                q_no_mesh[qi] += 1;
            }
            let _ = depth;
        }
        let n = count_mesh.max(1) as f64;
        eprintln!(
            "DIAG frame={frame} obs=({:.1},{:.1},{:.1}) MESH q[+X+Z:{} +X-Z:{} -X+Z:{} -X-Z:{}] NONE q[+X+Z:{} +X-Z:{} -X+Z:{} -X-Z:{}] mesh_centroid=({:.1},{:.1})",
            observer.x, observer.y, observer.z,
            q_with_mesh[0], q_with_mesh[1], q_with_mesh[2], q_with_mesh[3],
            q_no_mesh[0], q_no_mesh[1], q_no_mesh[2], q_no_mesh[3],
            sum_cx / n, sum_cz / n,
        );
    }
}

/// Merge `by_material` (one [`atomr_worlds_view::Mesh`] per material id)
/// into a single Bevy mesh. Vertex positions / normals / AO are copied
/// straight through; the material id is stored in `ATTRIBUTE_UV_0.x`
/// (`.y` left zero) so the fragment shader can index the palette
/// storage buffer. Indices are concatenated with the appropriate
/// offsets.
fn merge_by_material(
    by_material: &std::collections::HashMap<u16, atomr_worlds_view::Mesh>,
) -> BevyMesh {
    let mut positions: Vec<[f32; 3]> = Vec::new();
    let mut normals: Vec<[f32; 3]> = Vec::new();
    let mut colors: Vec<[f32; 4]> = Vec::new();
    let mut uvs: Vec<[f32; 2]> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    // Sorted iteration for determinism (HashMap order isn't stable).
    let mut pairs: Vec<(u16, &atomr_worlds_view::Mesh)> =
        by_material.iter().map(|(k, v)| (*k, v)).collect();
    pairs.sort_by_key(|(k, _)| *k);
    for (mat_id, sub_mesh) in pairs {
        if sub_mesh.indices.is_empty() {
            continue;
        }
        let base = positions.len() as u32;
        let id_f = mat_id as f32;
        for v in &sub_mesh.vertices {
            positions.push(v.pos);
            normals.push(v.normal);
            let ao = v.ao.clamp(0.0, 1.0);
            colors.push([ao, ao, ao, 1.0]);
            uvs.push([id_f, 0.0]);
        }
        indices.extend(sub_mesh.indices.iter().map(|i| *i + base));
    }
    let mut mesh =
        BevyMesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_UV_0, uvs);
    mesh.insert_indices(Indices::U32(indices));
    mesh
}

type WorldEntityVisibilityQuery<'w, 's> =
    Query<'w, 's, &'static mut Visibility, Or<(With<WorldCamera>, With<BrickMesh>)>>;

fn fp_visibility_toggle(mode: Res<ViewMode>, mut q: WorldEntityVisibilityQuery) {
    let want = matches!(*mode, ViewMode::Fp | ViewMode::Tp);
    let vis = if want { Visibility::Inherited } else { Visibility::Hidden };
    for mut v in q.iter_mut() {
        if *v != vis {
            *v = vis;
        }
    }
}

fn atomr_to_bevy_mesh(m: &atomr_worlds_view::Mesh) -> BevyMesh {
    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(m.vertices.len());
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(m.vertices.len());
    // Per-vertex tint = AO factor on RGB; alpha stays 1.0. Bevy's
    // `StandardMaterial` multiplies `base_color * ATTRIBUTE_COLOR`, so
    // an AO value of 0.55 darkens the material's surface by ~45% at
    // that vertex. The alpha pathway is left untouched because
    // translucent materials (water/ice) get their alpha from the
    // material itself.
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(m.vertices.len());
    for v in &m.vertices {
        positions.push(v.pos);
        normals.push(v.normal);
        let ao = v.ao.clamp(0.0, 1.0);
        colors.push([ao, ao, ao, 1.0]);
    }
    let mut mesh =
        BevyMesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(m.indices.clone()));
    mesh
}
