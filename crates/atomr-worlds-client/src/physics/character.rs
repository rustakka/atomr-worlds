//! Rec 2 Phase B — collidable first-person character controller.
//!
//! Replaces the free-fly camera with a capsule driven by rapier's
//! [`KinematicCharacterController`] — but **only** when physics is enabled and
//! the active view is [`ViewMode::Fp`]. In every other case (harness,
//! `--physics off`, non-FP modes, or the `physics` feature off) the free-fly
//! path in [`crate::modes::fp::world_walk_input`] is untouched and stays
//! byte-identical.
//!
//! # Ownership split
//!
//! When the controller is active, the rapier capsule owns *position* and
//! [`WalkCamera`](atomr_worlds_view::WalkCamera) owns only *orientation*
//! (yaw/pitch from the mouse). Each frame:
//!
//! 1. [`crate::modes::fp::world_walk_input`] feeds the WASD heading into
//!    [`CharacterIntent`] and returns *before* `walk.tick` (no double
//!    integration of position).
//! 2. [`drive_character`] (before `PhysicsSet::SyncBackend`) integrates gravity
//!    / jump into a vertical velocity and sets `controller.translation`.
//! 3. rapier resolves the move against the static terrain colliders.
//! 4. [`writeback_character`] (after `PhysicsSet::Writeback`) reads the resolved
//!    capsule [`Transform`] back into `walk.observer` via `observer.tick`, so
//!    `fp_update_motion_state`, `fp_sync_camera`, streaming, and skybox refresh
//!    all keep working unchanged.
//!
//! Everything here is on the 1 m / voxel render grid (`voxel_size_m == 1.0`),
//! the same grid the static colliders and debris use — never the host brush
//! space. It is client-side, ephemeral, and never flows into `GetBrick`.

use bevy::prelude::*;
use bevy_rapier3d::prelude::*;

use atomr_worlds_core::coord::DVec3;

use super::config::PhysicsConfig;
use crate::modes::fp::FpState;
use crate::view_mode::ViewMode;

/// Capsule radius (m). Total standing height is `CAPSULE_HEAD_Y + radius`.
const CAPSULE_RADIUS: f32 = 0.3;
/// Lower segment endpoint (bottom hemisphere center), feet-relative.
const CAPSULE_FOOT_Y: f32 = CAPSULE_RADIUS; // 0.3 → capsule bottom touches y=0
/// Upper segment endpoint (top hemisphere center), feet-relative. With the
/// radius this gives a ~1.8 m standing capsule whose origin is at the feet, so
/// `observer.position` (the eye-base) maps to the capsule translation with no
/// offset arithmetic on writeback.
const CAPSULE_HEAD_Y: f32 = 1.5;

const WALK_SPEED: f32 = 4.0;
const SPRINT_SPEED: f32 = 12.0;
/// Jump take-off speed (m/s). On the 1 m grid with g = -9.81 this clears a
/// ~1.27 m apex — enough to hop a 1 m ledge.
const JUMP_SPEED: f32 = 5.0;
/// Small downward bias kept while grounded so snap-to-ground stays engaged on
/// flats and ramps (the controller clamps it to the surface).
const GROUND_STICK: f32 = -1.0;
/// Snap-to-ground reach (m). Disabled while ascending so a jump isn't cancelled
/// by the snap pulling the capsule straight back down.
const SNAP_TO_GROUND_M: f32 = 0.3;
/// Distance below the spawn height the player may fall before being respawned
/// (mirrors the debris kill-plane).
const FALL_KILL_M: f32 = 64.0;

/// Marker on the single capsule entity representing the player.
#[derive(Component)]
pub struct Player;

/// Gravity/jump integrator + spawn bookkeeping. The vertical velocity lives
/// here (not in rapier) so the jump/land state machine is a pure, testable
/// function.
#[derive(Resource, Default)]
pub struct CharacterState {
    pub spawned: bool,
    pub entity: Option<Entity>,
    pub vertical_velocity: f32,
    pub grounded: bool,
    /// Y the player spawned at — the kill-plane respawn target.
    pub spawn_y: f32,
}

/// The input seam. [`crate::modes::fp::world_walk_input`] writes the WASD
/// heading (already rotated into world space, horizontal only) plus the
/// sprint/crouch flags here when the controller is active; [`drive_character`]
/// consumes it. Jump is read live in `drive_character` (avoids a stale flag).
#[derive(Resource, Default)]
pub struct CharacterIntent {
    /// World-space horizontal heading (un-normalized; magnitude ignored).
    pub move_world: Vec3,
    pub sprint: bool,
    pub crouch: bool,
}

/// True when the controller should own movement this frame. The physics
/// systems only run when `cfg.enabled` (the plugin is a no-op otherwise), but
/// the predicate is also called from `world_walk_input`, which is always
/// scheduled — hence the explicit `enabled` check.
pub fn character_active(cfg: &PhysicsConfig, mode: ViewMode, state: &CharacterState) -> bool {
    cfg.enabled && mode == ViewMode::Fp && state.spawned
}

fn character_controller() -> KinematicCharacterController {
    KinematicCharacterController {
        up: Vec3::Y,
        offset: CharacterLength::Absolute(0.05),
        autostep: Some(CharacterAutostep {
            max_height: CharacterLength::Absolute(0.5),
            min_width: CharacterLength::Absolute(0.2),
            include_dynamic_bodies: false,
        }),
        max_slope_climb_angle: 50.0_f32.to_radians(),
        min_slope_slide_angle: 35.0_f32.to_radians(),
        snap_to_ground: Some(CharacterLength::Absolute(SNAP_TO_GROUND_M)),
        ..default()
    }
}

/// Spawn the player capsule once, at the FP camera's start position. Spawns
/// ~10 m above ground (the FP scene's spawn perch), then falls onto the first
/// streamed-in LOD-0 collider — so it doesn't matter that colliders attach a
/// few frames after the player exists.
pub fn spawn_player(
    mut state: ResMut<CharacterState>,
    mode: Res<ViewMode>,
    fp: Res<FpState>,
    mut commands: Commands,
) {
    if state.spawned || *mode != ViewMode::Fp || !fp.ready {
        return;
    }
    let p = fp.walk.observer.position;
    let translation = Vec3::new(p.x as f32, p.y as f32, p.z as f32);
    let ent = commands
        .spawn((
            Player,
            RigidBody::KinematicPositionBased,
            Collider::capsule(
                Vec3::new(0.0, CAPSULE_FOOT_Y, 0.0),
                Vec3::new(0.0, CAPSULE_HEAD_Y, 0.0),
                CAPSULE_RADIUS,
            ),
            character_controller(),
            Transform::from_translation(translation),
        ))
        .id();
    state.spawned = true;
    state.entity = Some(ent);
    state.spawn_y = translation.y;
    state.vertical_velocity = 0.0;
    state.grounded = false;
}

/// Advance the vertical-velocity integrator one tick. Pure so it can be unit
/// tested without a live rapier world. `gravity_y` is negative (downward).
pub fn step_vertical(vel: f32, grounded: bool, jump: bool, dt: f32, gravity_y: f32) -> f32 {
    if jump && grounded {
        return JUMP_SPEED;
    }
    if grounded {
        return GROUND_STICK;
    }
    vel + gravity_y * dt
}

/// Build the per-frame desired translation: horizontal = normalized heading ×
/// speed × dt, vertical = velocity × dt. Pure / testable.
pub fn desired_translation(move_world: Vec3, vertical_velocity: f32, speed: f32, dt: f32) -> Vec3 {
    let horizontal = move_world.normalize_or_zero() * speed * dt;
    Vec3::new(horizontal.x, vertical_velocity * dt, horizontal.z)
}

/// Set `controller.translation` from gravity + the input intent.
/// Ordered `.before(PhysicsSet::SyncBackend)` so rapier consumes it this frame.
pub fn drive_character(
    cfg: Res<PhysicsConfig>,
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    intent: Res<CharacterIntent>,
    mut state: ResMut<CharacterState>,
    mut q: Query<&mut KinematicCharacterController, With<Player>>,
) {
    if !character_active(&cfg, *mode, &state) {
        return;
    }
    let dt = time.delta_secs().min(0.05);
    let jump = keys.just_pressed(KeyCode::Space);
    state.vertical_velocity =
        step_vertical(state.vertical_velocity, state.grounded, jump, dt, cfg.gravity.y);

    let speed = if intent.sprint { SPRINT_SPEED } else { WALK_SPEED };
    let translation = desired_translation(intent.move_world, state.vertical_velocity, speed, dt);

    if let Ok(mut controller) = q.single_mut() {
        // Disable snap while ascending so the jump isn't immediately cancelled.
        controller.snap_to_ground = if state.vertical_velocity > 0.0 {
            None
        } else {
            Some(CharacterLength::Absolute(SNAP_TO_GROUND_M))
        };
        controller.translation = Some(translation);
    }
}

/// Read the resolved capsule pose + ground state back into the walk camera.
/// Ordered `.after(PhysicsSet::Writeback)` (output + Transform are valid) and
/// before `fp_update_motion_state` (which the FP chain runs before
/// `fp_sync_camera`), so the camera/EWMAs see the resolved position the same
/// frame.
pub fn writeback_character(
    cfg: Res<PhysicsConfig>,
    mode: Res<ViewMode>,
    time: Res<Time>,
    intent: Res<CharacterIntent>,
    mut state: ResMut<CharacterState>,
    mut fp: ResMut<FpState>,
    q: Query<(&Transform, Option<&KinematicCharacterControllerOutput>), With<Player>>,
    mut commands: Commands,
) {
    if !character_active(&cfg, *mode, &state) {
        return;
    }
    let Some(entity) = state.entity else {
        return;
    };
    let Ok((transform, output)) = q.get(entity) else {
        return;
    };

    state.grounded = output.map(|o| o.grounded).unwrap_or(false);
    if state.grounded && state.vertical_velocity < 0.0 {
        // Zero downward accumulation on landing so we don't carry speed into
        // the next airborne phase.
        state.vertical_velocity = 0.0;
    }

    let mut pos = transform.translation;
    if pos.y < state.spawn_y - FALL_KILL_M {
        // Pathological fall-through: teleport back to the spawn height.
        pos.y = state.spawn_y;
        state.vertical_velocity = 0.0;
        commands
            .entity(entity)
            .insert(Transform::from_translation(pos));
    }

    let dt = time.delta_secs().min(0.05);
    fp.walk
        .observer
        .tick(DVec3::new(pos.x as f64, pos.y as f64, pos.z as f64), None, dt);
    // Lower the eye for the frame when crouching (capsule resize deferred to a
    // later phase — see module docs).
    fp.walk.set_crouch(intent.crouch);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_only_from_ground() {
        // Grounded + jump → take-off speed.
        assert_eq!(step_vertical(GROUND_STICK, true, true, 0.016, -9.81), JUMP_SPEED);
        // Airborne + jump → ignored (no double-jump), gravity keeps applying.
        let v = step_vertical(2.0, false, true, 0.1, -10.0);
        assert!((v - 1.0).abs() < 1e-6, "airborne jump ignored: {v}");
    }

    #[test]
    fn grounded_holds_ground_stick() {
        assert_eq!(step_vertical(123.0, true, false, 0.016, -9.81), GROUND_STICK);
    }

    #[test]
    fn airborne_accumulates_gravity() {
        let v0 = 0.0;
        let v1 = step_vertical(v0, false, false, 0.1, -10.0);
        assert!((v1 - (-1.0)).abs() < 1e-6);
        let v2 = step_vertical(v1, false, false, 0.1, -10.0);
        assert!((v2 - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn jump_apex_matches_kinematics() {
        // Integrate a jump under gravity until it starts descending; apex
        // height should be ~ v²/2g.
        let g = -9.81_f32;
        let dt = 1.0 / 240.0;
        let mut v = step_vertical(GROUND_STICK, true, true, dt, g); // = JUMP_SPEED
        let mut h = 0.0_f32;
        while v > 0.0 {
            h += v * dt;
            v = step_vertical(v, false, false, dt, g);
        }
        let analytic = JUMP_SPEED * JUMP_SPEED / (2.0 * -g);
        assert!((h - analytic).abs() < 0.05, "apex {h} vs analytic {analytic}");
    }

    #[test]
    fn desired_translation_scales_and_normalizes() {
        let dt = 0.5;
        // Pure-forward heading, walk speed.
        let t = desired_translation(Vec3::new(0.0, 0.0, 2.0), 0.0, WALK_SPEED, dt);
        assert!((t.z - WALK_SPEED * dt).abs() < 1e-5, "z={}", t.z);
        assert_eq!(t.x, 0.0);
        assert_eq!(t.y, 0.0);
        // Diagonal stays speed-capped (normalized), not √2 faster.
        let d = desired_translation(Vec3::new(1.0, 0.0, 1.0), 0.0, WALK_SPEED, dt);
        let planar = (d.x * d.x + d.z * d.z).sqrt();
        assert!((planar - WALK_SPEED * dt).abs() < 1e-5, "planar={planar}");
        // Sprint is faster than walk.
        let s = desired_translation(Vec3::new(0.0, 0.0, 1.0), 0.0, SPRINT_SPEED, dt);
        assert!(s.z > t.z);
        // Vertical = velocity × dt.
        let up = desired_translation(Vec3::ZERO, JUMP_SPEED, WALK_SPEED, dt);
        assert!((up.y - JUMP_SPEED * dt).abs() < 1e-5);
        // Zero heading → zero horizontal.
        let z = desired_translation(Vec3::ZERO, 0.0, WALK_SPEED, dt);
        assert_eq!(z.x, 0.0);
        assert_eq!(z.z, 0.0);
    }

    #[test]
    fn active_only_when_enabled_fp_and_spawned() {
        let cfg = PhysicsConfig::default(); // enabled = true
        let spawned = CharacterState { spawned: true, ..default() };
        let not_spawned = CharacterState { spawned: false, ..default() };
        assert!(character_active(&cfg, ViewMode::Fp, &spawned));
        // Each negative drops it.
        assert!(!character_active(&cfg, ViewMode::Fp, &not_spawned));
        assert!(!character_active(&cfg, ViewMode::Tp, &spawned));
        assert!(!character_active(&cfg, ViewMode::Rts, &spawned));
        let mut disabled = PhysicsConfig::default();
        disabled.enabled = false;
        assert!(!character_active(&disabled, ViewMode::Fp, &spawned));
    }

    /// `spawn_player` spawns exactly one capsule with the controller
    /// components, and is idempotent (a second run is a no-op).
    #[test]
    fn spawn_player_spawns_once() {
        let mut app = App::new();
        app.add_plugins(MinimalPlugins);
        app.insert_resource(ViewMode::Fp);
        app.init_resource::<CharacterState>();
        let mut fp = FpState::default();
        fp.ready = true;
        app.insert_resource(fp);
        app.add_systems(Update, spawn_player);

        app.update();
        app.update(); // second run must not spawn a second player

        let count = app
            .world_mut()
            .query_filtered::<Entity, With<Player>>()
            .iter(app.world())
            .count();
        assert_eq!(count, 1, "exactly one player capsule");

        let ent = app.world().resource::<CharacterState>().entity.unwrap();
        assert!(app.world().get::<Collider>(ent).is_some());
        assert!(matches!(
            app.world().get::<RigidBody>(ent),
            Some(RigidBody::KinematicPositionBased)
        ));
        assert!(app.world().get::<KinematicCharacterController>(ent).is_some());
    }

    /// No player is spawned when the view isn't FP or the scene isn't ready.
    #[test]
    fn spawn_player_is_gated() {
        // Not FP.
        {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins);
            app.insert_resource(ViewMode::Tp);
            app.init_resource::<CharacterState>();
            let mut fp = FpState::default();
            fp.ready = true;
            app.insert_resource(fp);
            app.add_systems(Update, spawn_player);
            app.update();
            assert!(!app.world().resource::<CharacterState>().spawned);
        }
        // FP but not ready.
        {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins);
            app.insert_resource(ViewMode::Fp);
            app.init_resource::<CharacterState>();
            app.insert_resource(FpState::default()); // ready = false
            app.add_systems(Update, spawn_player);
            app.update();
            assert!(!app.world().resource::<CharacterState>().spawned);
        }
    }
}
