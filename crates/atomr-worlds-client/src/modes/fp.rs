//! Phase 14a — 1st-person walk, native Bevy 3D.
//!
//! - `WalkCamera` (from `atomr-worlds-view`) drives input → pose.
//! - Each frame we reconcile a desired-set of `(brick_coord, lod)` keys
//!   from the [`crate::world_stream::ChunkStreamer`] against the entities
//!   currently loaded into Bevy. Greedy-meshing uses
//!   `atomr-worlds-view::mesh::greedy_mesh`. Per-material vertex colors
//!   carry RGB so we can render through a single `StandardMaterial`.
//!
//! # LOD-transition pipeline
//!
//! The streamer is parameterised by a
//! [`crate::render::LodCoveragePolicy`]. With the default
//! [`crate::render::defaults::NestedSummary`] every region of the world
//! is covered by the finest LOD *and* every coarser parent LOD
//! simultaneously — parents are pre-cached "summaries" that the renderer
//! can fall back to instantly when a finer brick unloads.
//!
//! Two systems collaborate to keep the screen showing the right LOD per
//! region without popping:
//!
//! - [`fp_update_lod_visibility`] runs each frame, walks every loaded
//!   brick, and hides any whose immediate finer children are all
//!   resident (and not currently fading out). Bricks transitioning
//!   from hidden → visible get a fresh [`BrickFadeIn`] so the reveal
//!   blooms instead of popping.
//! - [`fp_animate_fade_out`] handles the despawn side: when a brick
//!   exits the desired set past the hysteresis window, the streamer
//!   attaches [`BrickFadeOut`] instead of destroying the entity
//!   immediately. The fade-out lasts longer than the fade-in by
//!   design — the overlap is the crossfade that smooths the LOD
//!   handoff.
//!
//! See [`crate::world_stream`] for the streaming-side rationale and
//! `harness/scenes/lod_crossfade*.toml` for the A/B visual regression
//! scenarios that drive a camera across a tier boundary under each
//! policy.

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_view::{WalkCamera, WalkInput, WorldQuery};
// (WorldQuery brings ground_height_m into scope.)
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::camera::RenderTarget;
use bevy::render::mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use crate::brick_gen::{BrickGenWorkers, BrickReady, DEFAULT_SPAWN_BUDGET};
use crate::render::{
    OffscreenTarget, PaletteEntryGpu, RenderConfig, ShadingMode, SkyboxRuntime, VoxelMaterial,
    VoxelMaterialExt, WorldSunMarker,
};
use crate::view_mode::ViewMode;
use crate::world_runtime::{ActiveWorld, WorldRuntime};
use crate::world_stream::{
    desired_chunks, prioritize_view, ChunkStreamer, DesiredChunksCache, LoadedChunk, LoadedChunks,
};

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
                    fp_update_lod_visibility,
                    fp_animate_fade_in,
                    fp_animate_fade_out,
                    fp_visibility_toggle,
                )
                    .chain(),
            );
    }
}

/// Tags a freshly-spawned brick entity that is mid-fade-in. The
/// streaming system installs it with `age = 0` and a per-LOD scale;
/// [`fp_animate_fade_in`] tweens the `SpatialBundle` transform to
/// full size over [`FADE_IN_SECONDS`] before removing the marker.
#[derive(Component)]
pub struct BrickFadeIn {
    /// Seconds since spawn.
    pub age: f32,
    /// Final scale to land on (= the LOD's voxel-edge scale).
    pub final_scale: f32,
}

/// Tags a brick entity that is fading out before despawn. Mirror of
/// [`BrickFadeIn`]: the streaming system replaces immediate
/// `despawn_recursive` with a scale shrink so the LOD transition has
/// a soft tail rather than a frame-perfect pop. [`fp_animate_fade_out`]
/// walks the scale to 0, then despawns and clears the corresponding
/// [`LoadedChunks`] entry.
#[derive(Component)]
pub struct BrickFadeOut {
    /// Seconds since the fade-out started.
    pub age: f32,
    /// Scale this brick was at when the fade-out began (= the LOD's
    /// voxel-edge scale unless it was caught mid-fade-in).
    pub from_scale: f32,
    /// `(coord, lod_depth)` key for the matching [`LoadedChunks`]
    /// entry, so the fade-out completion can drop the entry and
    /// release the parent brick to the visibility system.
    pub key: (IVec3, u8),
}

/// `(coord, lod_depth)` of the brick rendered by this entity. Stored
/// on the parent spatial entity so the visibility system can match
/// parent/child relationships across LODs without re-deriving them
/// from `LoadedChunks`.
#[derive(Component, Clone, Copy, Debug)]
pub struct BrickLod {
    pub coord: IVec3,
    pub depth: u8,
}

/// Duration of the per-brick scale-up reveal. Short — just enough to
/// soften the pop-in. Combined with the existing exponential fog the
/// load process looks like a ring expanding from the observer.
pub const FADE_IN_SECONDS: f32 = 0.18;
/// Starting scale fraction. 0.75 ⇒ each new brick is briefly 75 % of
/// its final extent so it "blooms" into place rather than appearing
/// in one frame.
pub const FADE_IN_START_FRACTION: f32 = 0.75;

/// Duration of the LOD-transition fade-out. Slightly longer than
/// [`FADE_IN_SECONDS`] so the parent brick has time to scale up
/// underneath while the child shrinks away — the two tweens overlap
/// to produce a crossfade rather than a strict sequence.
pub const FADE_OUT_SECONDS: f32 = 0.25;

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
        // `IsDefaultUiCamera` keeps `bevy_ui`'s default-camera resolver
        // from panicking on frame 0, before `hud::route_hud_target` has
        // had a chance to attach an explicit `TargetCamera` to the HUD
        // root. WorldCamera is spawned at Startup and never despawned, so
        // the marker is always live regardless of view mode. Once the
        // router runs, UI follows whichever of WorldCamera / BlitCamera
        // is `is_active` for the current mode — so the HUD lands above
        // the 3D scene in FP/TP and above the blit sprite in raster
        // modes, without ever pairing a Camera2d with a Camera3d on the
        // same offscreen target (which Bevy 0.13 mishandles by dropping
        // the 3D output).
        bevy::ui::IsDefaultUiCamera,
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

/// WASD / Space / Ctrl / Shift — drives [`FpState::walk`] in the view
/// modes that anchor on the world walk position (FP, TP, RTS). TP orbits
/// this anchor; RTS centers its raster on it; FP walks with it. Slice
/// mode is deliberately excluded — it has its own yaw-independent pan
/// (see [`crate::modes::slice`]) so its WASD scrolling doesn't inherit
/// the FP camera's heading. The mouse-look + arrow-key look part stays
/// in [`fp_input_look`] which is FP-only.
pub fn world_walk_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    mut state: ResMut<FpState>,
) {
    if !state.ready {
        return;
    }
    // RTS pans its view by moving the walk position; TP orbits it; FP
    // walks with it. Slice has its own pan state; Overview has its own.
    if !matches!(*mode, ViewMode::Fp | ViewMode::Tp | ViewMode::Rts) {
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

/// Copy the FP [`WalkCamera`]'s eye + look-at pose onto the Bevy
/// camera entity each frame. No-op outside FP mode; the entity's
/// transform is owned by other view modes' systems then.
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

/// Per-frame streaming loop for the 3D world entities.
///
/// 1. Bump the streamer frame counter for hysteresis bookkeeping.
/// 2. Recompute the desired `(coord, lod)` set when the observer
///    drifts / turns past the cache-rebuild thresholds (see
///    [`crate::world_stream::DesiredChunksCache`]); the new policy
///    parameter from [`RenderConfig::coverage`] controls shell vs
///    nested-summary shape.
/// 3. Refresh `last_seen_frame` on every desired entry so the
///    hysteresis window resets while the brick is in the ring.
/// 4. Convert stale entries to [`BrickFadeOut`] (instead of immediate
///    despawn) so the LOD transition has a soft tail; the fade-out
///    system clears [`LoadedChunks`] when the shrink completes.
/// 5. Dispatch async fetches for desired-but-not-loaded bricks via
///    [`BrickGenWorkers`] (capped at `MAX_IN_FLIGHT`).
/// 6. Drain a per-frame budget of finished payloads into Bevy
///    entities via [`spawn_brick_entity`].
#[allow(clippy::too_many_arguments)]
fn fp_stream_bricks(
    state: Res<FpState>,
    active: Res<crate::world_runtime::ActiveWorld>,
    pool: Res<MaterialPool>,
    voxel_pool: Res<VoxelMaterialPool>,
    render_cfg: Res<RenderConfig>,
    mut streamer: ResMut<ChunkStreamer>,
    mut loaded: ResMut<LoadedChunks>,
    mut plan_cache: ResMut<DesiredChunksCache>,
    mut workers: ResMut<BrickGenWorkers>,
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
    // Camera forward (world space) for view-priority sorting. Use the
    // WalkCamera-derived target so it matches the rendered view.
    let cam = state.walk.camera();
    let target = DVec3::new(cam.target[0] as f64, cam.target[1] as f64, cam.target[2] as f64);
    let mut forward = target - observer;
    let mag = (forward.x * forward.x + forward.y * forward.y + forward.z * forward.z).sqrt();
    if mag > 1e-6 {
        forward.x /= mag;
        forward.y /= mag;
        forward.z /= mag;
    } else {
        forward = DVec3::new(0.0, 0.0, 1.0);
    }

    // Rebuild the desired-chunks plan only when the observer drifts or
    // the camera turns past the configured thresholds. The full sweep
    // costs ms on the default 4-tier ladder — caching it lets quiet
    // frames spend almost no time in the streamer.
    if plan_cache.should_rebuild(observer, forward) {
        // Body-aware horizon: sphere/cylinder worlds clamp the streamer's
        // outer radius to the geometric horizon at the observer's
        // altitude (`sqrt(2*R*h + h²)`). Cube worlds short-circuit to
        // `f64::INFINITY` and the streamer's full ladder reach is used
        // unchanged — preserves the pre-Phase-17.x behaviour for the
        // default cube world.
        let horizon_m = active.shape.horizon_at_m(observer);
        let mut plan =
            desired_chunks(&streamer, observer, horizon_m, render_cfg.coverage.as_ref());
        prioritize_view(&mut plan, observer, forward);
        plan_cache.set(observer, forward, plan);
    }

    // Mark every desired chunk as "seen this frame" so the hysteresis
    // window resets while we're inside the ring.
    for (coord, lod) in &plan_cache.plan {
        let key = LoadedChunk::key(*coord, *lod);
        if let Some(entry) = loaded.0.get_mut(&key) {
            entry.last_seen_frame = frame;
        }
    }

    // Stale chunks (outside the desired set past the hysteresis window)
    // start fading out instead of being immediately despawned. The
    // [`fp_animate_fade_out`] system handles the eventual despawn +
    // [`LoadedChunks`] removal once the scale reaches 0. We keep the
    // entry in `LoadedChunks` while the fade plays so the visibility
    // system still sees the brick as "loaded" and can keep the parent
    // hidden until the crossfade hands off.
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
        let Some(chunk) = loaded.0.get(&k) else { continue };
        let Some(ent) = chunk.entity else {
            // Empty-brick placeholder; nothing to fade. Drop it.
            loaded.0.remove(&k);
            continue;
        };
        let from_scale = (1u64 << chunk.lod.depth as u32) as f32;
        commands
            .entity(ent)
            .remove::<BrickFadeIn>()
            .insert(BrickFadeOut { age: 0.0, from_scale, key: k });
    }

    // Dispatch async fetches for desired-but-not-loaded bricks. The
    // workers are capped at `MAX_IN_FLIGHT` so we don't pile up tasks
    // during initial world fill; remaining work resumes next frame.
    // We walk the (already view-priority-sorted) plan front-to-back so
    // forward-facing bricks resolve first.
    for (bc, lod) in plan_cache.plan.iter() {
        if workers.is_saturated() {
            break;
        }
        let key = LoadedChunk::key(*bc, *lod);
        if loaded.0.contains_key(&key) {
            continue;
        }
        if workers.contains(&key) {
            continue;
        }
        workers.dispatch(state.addr, *bc, *lod);
    }

    // Drain completed brick fetches and convert them into Bevy entities.
    // Capped per frame so mesh-asset upload stays inside the frame
    // budget — running a 64-brick backlog through `meshes.add` on one
    // frame produces a visible hitch.
    let ready_batch = workers.drain(DEFAULT_SPAWN_BUDGET);
    for ready in ready_batch {
        spawn_brick_entity(
            ready,
            frame,
            shading_mode,
            &pool,
            &voxel_pool,
            &mut meshes,
            &mut commands,
            &mut loaded,
        );
    }

    // Diagnostic (mesh quadrant counts). The original frame==60 terrain
    // height probe was dropped when the streamer moved to async dispatch
    // — its job (catch directional asymmetry in `ground_height_m`) is
    // already covered by `desired_chunks_load_symmetrically_in_all_four_cardinal_directions`
    // in `world_stream.rs`.
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

/// Build the Bevy entity for a single async-streamed brick.
///
/// Called on the main thread from `fp_stream_bricks` once a
/// [`BrickReady`] payload arrives from the worker pool. Splits into
/// per-material child meshes (`SplitPerMaterial`) or builds the
/// merged mesh + `VoxelMaterial` draw (`PaletteVoxelMaterial`) per
/// the active [`ShadingMode`].
///
/// Empty / missing bricks still record a `LoadedChunk` placeholder so
/// the streamer doesn't re-dispatch the same key every frame.
#[allow(clippy::too_many_arguments)]
fn spawn_brick_entity(
    ready: BrickReady,
    frame: u64,
    shading_mode: ShadingMode,
    pool: &MaterialPool,
    voxel_pool: &VoxelMaterialPool,
    meshes: &mut Assets<BevyMesh>,
    commands: &mut Commands,
    loaded: &mut LoadedChunks,
) {
    let BrickReady { coord: bc, lod, brick: _, meshes: mut by_material } = ready;
    let key = LoadedChunk::key(bc, lod);
    let lod_scale = (1u64 << lod.depth as u32) as f32;
    let edge_m = BRICK_EDGE as f32 * lod_scale;
    if by_material.is_empty() {
        loaded.0.insert(
            key,
            LoadedChunk { coord: bc, lod, entity: None, last_seen_frame: frame },
        );
        return;
    }
    let origin = Vec3::new(
        (bc.x as f32) * edge_m,
        (bc.y as f32) * edge_m,
        (bc.z as f32) * edge_m,
    );
    // Bloom-in reveal: start slightly under-scale and tween up to the
    // LOD scale over `FADE_IN_SECONDS`. The transform's `scale` is what
    // becomes `lod_scale` once `fp_animate_fade_in` finishes.
    //
    // The brick spawns Hidden — `fp_update_lod_visibility` runs in the
    // same frame and decides whether this LOD is the finest available
    // for its region. If so, it's made visible (with the bloom-in
    // tween). Otherwise it sits invisible behind whatever finer LOD
    // currently owns the region, ready to crossfade in when the
    // finer tier unloads. The `BrickLod` tag carries the
    // `(coord, depth)` key so the visibility system can match
    // parent/child relationships.
    let start_scale = lod_scale * FADE_IN_START_FRACTION;
    let parent = commands
        .spawn((
            SpatialBundle {
                transform: Transform::from_translation(origin)
                    .with_scale(Vec3::splat(start_scale)),
                visibility: Visibility::Hidden,
                ..default()
            },
            BrickFadeIn { age: 0.0, final_scale: lod_scale },
            BrickLod { coord: bc, depth: lod.depth },
        ))
        .id();
    match shading_mode {
        ShadingMode::SplitPerMaterial => {
            for (mat_id, sub_mesh) in by_material.iter_mut() {
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
                return;
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
}

/// Smoothstep the per-brick scale from [`FADE_IN_START_FRACTION`] up to
/// 1× over [`FADE_IN_SECONDS`], then strip the [`BrickFadeIn`] marker
/// so the entity is no longer queried.
///
/// Hidden bricks (those waiting for a finer LOD to unload) keep their
/// age frozen so the tween plays from `age=0` when they're eventually
/// revealed — otherwise the animation would silently elapse while
/// invisible and the brick would pop in at full scale.
fn fp_animate_fade_in(
    time: Res<Time>,
    mut commands: Commands,
    mut q: Query<(Entity, &mut Transform, &mut BrickFadeIn, &Visibility)>,
) {
    let dt = time.delta_seconds();
    for (ent, mut tf, mut fade, vis) in q.iter_mut() {
        if *vis == Visibility::Hidden {
            // Brick is being suppressed by a finer LOD; freeze the
            // tween until it's revealed.
            continue;
        }
        fade.age += dt;
        let t = (fade.age / FADE_IN_SECONDS).clamp(0.0, 1.0);
        // Smoothstep — gives a softer end-of-tween than a linear ramp.
        let s = t * t * (3.0 - 2.0 * t);
        let scale = fade.final_scale * (FADE_IN_START_FRACTION + (1.0 - FADE_IN_START_FRACTION) * s);
        tf.scale = Vec3::splat(scale);
        if t >= 1.0 {
            tf.scale = Vec3::splat(fade.final_scale);
            commands.entity(ent).remove::<BrickFadeIn>();
        }
    }
}

/// Smoothstep the per-brick scale from its starting LOD scale down to
/// 0 over [`FADE_OUT_SECONDS`], then despawn the entity and remove
/// the matching [`LoadedChunks`] entry. The fade overlaps with the
/// parent LOD's fade-in (revealed by [`fp_update_lod_visibility`]) so
/// LOD transitions crossfade instead of popping.
fn fp_animate_fade_out(
    time: Res<Time>,
    mut commands: Commands,
    mut loaded: ResMut<LoadedChunks>,
    mut q: Query<(Entity, &mut Transform, &mut BrickFadeOut)>,
) {
    let dt = time.delta_seconds();
    for (ent, mut tf, mut fade) in q.iter_mut() {
        fade.age += dt;
        let t = (fade.age / FADE_OUT_SECONDS).clamp(0.0, 1.0);
        // Reverse smoothstep so the shrink starts slow and accelerates.
        let s = t * t * (3.0 - 2.0 * t);
        let scale = fade.from_scale * (1.0 - s);
        tf.scale = Vec3::splat(scale.max(0.0));
        if t >= 1.0 {
            loaded.0.remove(&fade.key);
            commands.entity(ent).despawn_recursive();
        }
    }
}

/// Per-frame visibility pass for the nested-LOD pipeline.
///
/// With [`crate::render::defaults::NestedSummary`] as the active
/// [`crate::render::LodCoveragePolicy`], every brick region is covered
/// by the finer LOD *and* every coarser parent simultaneously. This
/// system decides which of those concurrent LODs actually renders:
///
/// 1. Build a "covered" set from every loaded child brick — for each
///    `(coord, depth)` in [`LoadedChunks`], emit the parent
///    `(coord/2, depth+1)`. A parent is considered covered iff *all 8*
///    of its immediate children are present in [`LoadedChunks`] (we
///    accumulate child counts per parent key and check `== 8`).
///    Bricks mid-fade-out do not count toward coverage, so the parent
///    is "uncovered" the moment any child starts to disappear — that's
///    what kicks off the crossfade reveal.
/// 2. Walk every entity with [`BrickLod`]. If its key is in the
///    "covered" set, force `Visibility::Hidden`. Otherwise force
///    `Visibility::Inherited` *and* — if it was previously hidden and
///    isn't already mid-fade-in — attach a fresh [`BrickFadeIn`] so it
///    blooms in smoothly rather than appearing in one frame.
fn fp_update_lod_visibility(
    loaded: Res<LoadedChunks>,
    fading_out: Query<(), With<BrickFadeOut>>,
    mut commands: Commands,
    mut q: Query<(
        Entity,
        &BrickLod,
        &mut Visibility,
        Option<&BrickFadeIn>,
    )>,
) {
    // Build a `parent_key → child_count` table from the loaded set,
    // excluding children that are currently fading out (they're on
    // their way to despawn; the parent should already be uncovered).
    let mut child_counts: std::collections::HashMap<(IVec3, u8), u32> =
        std::collections::HashMap::new();
    for ((coord, depth), chunk) in loaded.0.iter() {
        // Children only — depth 0 has no finer children that could
        // cover it. Bricks with `entity: None` (empty placeholders)
        // still count: they "cover" their parent's region because the
        // region really is empty there.
        let _ = chunk;
        // Bricks mid-fade-out don't contribute to coverage so the
        // parent gets revealed before the child fully disappears.
        if let Some(ent) = chunk.entity {
            if fading_out.get(ent).is_ok() {
                continue;
            }
        }
        // Parent of `(c, d)` lives at `(c.div_euclid(2), d+1)`. Using
        // `div_euclid` (not `/2`) keeps the relationship correct for
        // negative coords — `-1 / 2 == 0` in Rust truncation but
        // `(-1).div_euclid(2) == -1`, which matches our voxel-grid
        // convention.
        let parent = (
            IVec3::new(
                coord.x.div_euclid(2),
                coord.y.div_euclid(2),
                coord.z.div_euclid(2),
            ),
            depth + 1,
        );
        *child_counts.entry(parent).or_insert(0) += 1;
    }

    for (ent, lod, mut vis, fade_in) in q.iter_mut() {
        let key = (lod.coord, lod.depth);
        let covered = child_counts.get(&key).map(|n| *n == 8).unwrap_or(false);
        if covered {
            if *vis != Visibility::Hidden {
                *vis = Visibility::Hidden;
            }
        } else if *vis == Visibility::Hidden {
            *vis = Visibility::Inherited;
            // Brick is being revealed — re-attach BrickFadeIn so it
            // blooms instead of popping. Skip if it already has one
            // (e.g. it was hidden mid-fade-in then re-revealed on the
            // very next frame).
            if fade_in.is_none() {
                let lod_scale = (1u64 << lod.depth as u32) as f32;
                commands
                    .entity(ent)
                    .insert(BrickFadeIn { age: 0.0, final_scale: lod_scale });
            }
        }
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

/// Hide / show the 3D world entities (`WorldCamera` + every
/// `BrickMesh`) wholesale based on the active [`ViewMode`]. FP and TP
/// share the same scene, so both leave the entities visible; the
/// raster modes (slice / RTS / overview) blit a 2D framebuffer and
/// would otherwise render the brick meshes underneath. Distinct from
/// [`fp_update_lod_visibility`], which decides per-brick visibility
/// inside the FP scene.
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
