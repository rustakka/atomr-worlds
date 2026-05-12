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
use atomr_worlds_view::{greedy_mesh, material_color, WalkCamera, WalkInput, WorldQuery};
use atomr_worlds_voxel::BRICK_EDGE;
use bevy::input::mouse::MouseMotion;
use bevy::prelude::*;
use bevy::render::mesh::{Indices, Mesh as BevyMesh, PrimitiveTopology};
use bevy::render::render_asset::RenderAssetUsages;
use bevy::window::{CursorGrabMode, PrimaryWindow};

use crate::view_mode::ViewMode;
use crate::world_runtime::{ActiveWorld, WorldRuntime};

/// Radius (in bricks) of the streaming cube around the camera.
const STREAM_RADIUS_BRICKS: i64 = 3;
/// How many bricks to fetch per frame, max — keeps frame time bounded.
const STREAM_BUDGET_PER_FRAME: usize = 4;

pub struct FpPlugin;

impl Plugin for FpPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<FpState>()
            .init_resource::<LoadedBricks>()
            .init_resource::<FpMaterial>()
            .add_systems(Startup, setup_fp_scene)
            .add_systems(
                Update,
                (
                    grab_cursor,
                    fp_input,
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

#[derive(Resource, Default)]
struct LoadedBricks(HashMap<IVec3, Entity>);

/// Shared `StandardMaterial` for every brick — vertex colors carry the
/// per-face palette so we don't need a separate material per material id.
#[derive(Resource, Default)]
struct FpMaterial(Option<Handle<StandardMaterial>>);

fn setup_fp_scene(
    mut commands: Commands,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut fp_mat: ResMut<FpMaterial>,
    mut fp_state: ResMut<FpState>,
    active: Option<Res<ActiveWorld>>,
) {
    // Pull start addr from ActiveWorld if it has been inserted; otherwise
    // keep the default.
    if let Some(active) = active.as_deref() {
        fp_state.addr = active.addr;
        fp_state.walk = WalkCamera::new(
            DVec3::new(8.0, 24.0, 8.0),
            ContainingFrame::World(active.addr),
            16.0 / 9.0,
        );
    }
    // Look slightly down so the ground is in frame.
    fp_state.walk.pitch = -0.4;
    fp_state.ready = true;

    let mat = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        perceptual_roughness: 0.8,
        metallic: 0.0,
        ..default()
    });
    fp_mat.0 = Some(mat);

    commands.spawn((
        Camera3dBundle {
            transform: Transform::from_xyz(8.0, 26.0, 8.0).looking_to(Vec3::Z, Vec3::Y),
            ..default()
        },
        WorldCamera,
    ));
    commands.spawn((
        DirectionalLightBundle {
            directional_light: DirectionalLight {
                illuminance: 12000.0,
                shadows_enabled: false,
                ..default()
            },
            transform: Transform::from_xyz(50.0, 80.0, 30.0)
                .looking_at(Vec3::new(0.0, 0.0, 0.0), Vec3::Y),
            ..default()
        },
        WorldSun,
    ));
}

fn grab_cursor(
    mut windows: Query<&mut Window, With<PrimaryWindow>>,
    keys: Res<ButtonInput<KeyCode>>,
    mode: Res<ViewMode>,
) {
    let Ok(mut window) = windows.get_single_mut() else { return };
    // Only grab the cursor in fp/tp modes; release for 2D overlay modes.
    let want_grab = matches!(*mode, ViewMode::Fp | ViewMode::Tp);
    if keys.just_pressed(KeyCode::Escape) {
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
        return;
    }
    if want_grab && window.cursor.grab_mode == CursorGrabMode::None {
        // Re-grab on any keyboard input (e.g. WASD) so users don't get
        // stuck after Escape. Exclude Escape itself — otherwise holding
        // Escape past one frame re-locks the cursor it just released.
        if keys.get_pressed().any(|k| *k != KeyCode::Escape) {
            window.cursor.grab_mode = CursorGrabMode::Locked;
            window.cursor.visible = false;
        }
    } else if !want_grab && window.cursor.grab_mode != CursorGrabMode::None {
        window.cursor.grab_mode = CursorGrabMode::None;
        window.cursor.visible = true;
    }
}

fn fp_input(
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    mut motion: EventReader<MouseMotion>,
    time: Res<Time>,
    mut state: ResMut<FpState>,
    windows: Query<&Window, With<PrimaryWindow>>,
) {
    if *mode != ViewMode::Fp {
        motion.clear();
        return;
    }
    if !state.ready {
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
    if keys.pressed(KeyCode::KeyA) {
        mv[0] -= speed * dt;
    }
    if keys.pressed(KeyCode::KeyD) {
        mv[0] += speed * dt;
    }
    if keys.pressed(KeyCode::Space) {
        mv[1] += speed * dt;
    }
    if keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight) {
        mv[1] -= speed * dt;
    }

    let mut yaw_delta = 0.0f32;
    let mut pitch_delta = 0.0f32;
    let cursor_locked = windows
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

    let crouch = keys.pressed(KeyCode::KeyC);
    state.walk.tick(
        WalkInput { move_local: mv, yaw_delta, pitch_delta, crouch },
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

fn fp_stream_bricks(
    state: Res<FpState>,
    runtime: Res<WorldRuntime>,
    fp_mat: Res<FpMaterial>,
    mut loaded: ResMut<LoadedBricks>,
    mut meshes: ResMut<Assets<BevyMesh>>,
    mut commands: Commands,
) {
    if !state.ready {
        return;
    }
    let Some(material) = fp_mat.0.clone() else { return };

    // Brick currently containing the camera, in brick coords.
    let cam = state.walk.camera();
    let edge = BRICK_EDGE as f32;
    let cbx = (cam.eye[0] / edge).floor() as i64;
    let cby = (cam.eye[1] / edge).floor() as i64;
    let cbz = (cam.eye[2] / edge).floor() as i64;

    // Build the desired set (cube around camera).
    let mut desired: Vec<IVec3> = Vec::new();
    for dx in -STREAM_RADIUS_BRICKS..=STREAM_RADIUS_BRICKS {
        for dy in -1..=1 {
            for dz in -STREAM_RADIUS_BRICKS..=STREAM_RADIUS_BRICKS {
                desired.push(IVec3::new(cbx + dx, cby + dy, cbz + dz));
            }
        }
    }

    // Despawn bricks outside the desired set.
    let desired_set: std::collections::HashSet<IVec3> =
        desired.iter().copied().collect();
    let to_remove: Vec<IVec3> = loaded
        .0
        .keys()
        .filter(|k| !desired_set.contains(k))
        .copied()
        .collect();
    for k in to_remove {
        if let Some(ent) = loaded.0.remove(&k) {
            commands.entity(ent).despawn();
        }
    }

    // Load up to STREAM_BUDGET_PER_FRAME missing bricks per frame.
    let mut budget = STREAM_BUDGET_PER_FRAME;
    let lod = Lod::new(0);
    for bc in desired {
        if budget == 0 {
            break;
        }
        if loaded.0.contains_key(&bc) {
            continue;
        }
        let Some(brick) = runtime.query.brick(&state.addr, bc, lod) else {
            // Server returned an empty / out-of-shape brick — still record
            // a sentinel so we don't keep refetching. Use a dummy entity
            // (a spatial bundle with no mesh).
            let ent = commands.spawn(SpatialBundle::default()).id();
            loaded.0.insert(bc, ent);
            budget -= 1;
            continue;
        };
        let mesh = greedy_mesh(&brick);
        if mesh.indices.is_empty() {
            let ent = commands.spawn(SpatialBundle::default()).id();
            loaded.0.insert(bc, ent);
            budget -= 1;
            continue;
        }
        let bevy_mesh = atomr_to_bevy_mesh(&mesh);
        let mesh_handle = meshes.add(bevy_mesh);
        let origin = Vec3::new(
            (bc.x as f32) * edge,
            (bc.y as f32) * edge,
            (bc.z as f32) * edge,
        );
        let ent = commands
            .spawn((
                PbrBundle {
                    mesh: mesh_handle,
                    material: material.clone(),
                    transform: Transform::from_translation(origin),
                    ..default()
                },
                BrickMesh,
            ))
            .id();
        loaded.0.insert(bc, ent);
        budget -= 1;
    }
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
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(m.vertices.len());
    for v in &m.vertices {
        positions.push(v.pos);
        normals.push(v.normal);
        let c = material_color(v.material);
        colors.push([
            c[0] as f32 / 255.0,
            c[1] as f32 / 255.0,
            c[2] as f32 / 255.0,
            1.0,
        ]);
    }
    let mut mesh =
        BevyMesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default());
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(BevyMesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(m.indices.clone()));
    mesh
}
