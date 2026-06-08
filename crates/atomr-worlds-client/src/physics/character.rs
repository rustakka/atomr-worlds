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

/// Capsule radius (m). Total standing height is `CAPSULE_HEAD_Y_STAND + radius`.
const CAPSULE_RADIUS: f32 = 0.3;
/// Lower segment endpoint (bottom hemisphere center), feet-relative. The same in
/// both stances — crouch only lowers the *top*, so the feet (and thus
/// snap-to-ground / autostep, which key off the bottom hemisphere) are
/// unchanged.
const CAPSULE_FOOT_Y: f32 = CAPSULE_RADIUS; // 0.3 → capsule bottom touches y=0
/// Standing upper segment endpoint (top hemisphere center), feet-relative. With
/// the radius this gives a ~1.8 m standing capsule whose origin is at the feet,
/// so `observer.position` (the eye-base) maps to the capsule translation with no
/// offset arithmetic on writeback.
const CAPSULE_HEAD_Y_STAND: f32 = 1.5;
/// Crouched upper segment endpoint — feet stay put, the head drops so the total
/// capsule is ~0.9 m (`0.6 + radius 0.3`), low enough to fit under a 1 m ledge.
const CAPSULE_HEAD_Y_CROUCH: f32 = 0.6;
/// While crouched, walk/sprint speed is scaled by this (a slow shuffle).
const CROUCH_WALK_SCALE: f32 = 0.5;
/// Headroom-probe slack (m). To stand back up, a standing capsule shrunk by this
/// on its top and radius must clear of any collider — so "clear" means ≥ this
/// much room (matches the controller `offset`), preventing stand/crouch
/// oscillation in a corridor exactly standing-height tall.
const HEADROOM_SKIN_M: f32 = 0.05;

const WALK_SPEED: f32 = 4.0;
const SPRINT_SPEED: f32 = 12.0;
/// Jump take-off speed (m/s). With [`CHARACTER_GRAVITY_SCALE`] this clears a
/// ~1.25 m apex (enough to hop a 1 m ledge) in ~0.7 s of airtime.
const JUMP_SPEED: f32 = 7.0;
/// The character integrates vertical motion at this multiple of world gravity.
/// >1 keeps the jump a brisk, *symmetric* parabola instead of floaty —
/// WITHOUT changing world/debris gravity (`cfg.gravity` stays Earth 9.81 for
/// falling bodies). A single uniform scale (rather than a fall-only multiplier)
/// avoids the disjoint "float up, then slam down" feel of asymmetric gravity:
/// the rise and fall are equally quick. ~2× ≈ 19.6 m/s².
const CHARACTER_GRAVITY_SCALE: f32 = 2.0;
/// Small downward bias kept while grounded so snap-to-ground stays engaged on
/// flats and ramps (the controller clamps it to the surface).
const GROUND_STICK: f32 = -1.0;
/// Snap-to-ground reach (m). Disabled while ascending so a jump isn't cancelled
/// by the snap pulling the capsule straight back down.
const SNAP_TO_GROUND_M: f32 = 0.3;
/// Distance below the spawn height the player may fall before being respawned
/// (mirrors the debris kill-plane).
const FALL_KILL_M: f32 = 64.0;

// --- Creative flight (double-tap Space) -------------------------------------
/// Two Space presses within this window toggle fly mode (Minecraft-style).
const DOUBLE_TAP_WINDOW_S: f32 = 0.30;
/// Horizontal fly speed (m/s) and its sprint (Shift) multiple.
const FLY_SPEED: f32 = 8.0;
const FLY_SPRINT_SPEED: f32 = 20.0;
/// Vertical fly speed (m/s, Space = up / Left Ctrl = down) and its sprint multiple.
const FLY_VERTICAL_SPEED: f32 = 6.0;
const FLY_SPRINT_VERTICAL_SPEED: f32 = 16.0;

/// Feet-relative capsule segment endpoints `(foot_y, head_y, radius)` for the
/// given stance. Feet and radius are stance-invariant; only the head moves.
/// Pure / testable.
pub fn capsule_endpoints(crouched: bool) -> (f32, f32, f32) {
    let head = if crouched { CAPSULE_HEAD_Y_CROUCH } else { CAPSULE_HEAD_Y_STAND };
    (CAPSULE_FOOT_Y, head, CAPSULE_RADIUS)
}

/// Build the player capsule [`Collider`] for the given stance. Single source of
/// truth for both the initial spawn and the crouch resize, so the two can never
/// drift.
fn crouch_collider(crouched: bool) -> Collider {
    let (foot, head, radius) = capsule_endpoints(crouched);
    Collider::capsule(Vec3::new(0.0, foot, 0.0), Vec3::new(0.0, head, 0.0), radius)
}

/// Resolve the *effective* crouch state from intent and headroom. Holding the
/// crouch key always crouches; releasing it only stands when there is headroom —
/// so you can't pop up into a low ceiling. Pure / testable.
pub fn resolve_crouch_state(intent_crouch: bool, headroom_clear: bool) -> bool {
    intent_crouch || !headroom_clear
}

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
    /// Resolved (effective) crouch state — drives both the physics capsule size
    /// and the camera eye height. May lag intent: releasing crouch under a low
    /// ceiling keeps this `true` until there is headroom to stand.
    pub crouched: bool,
    /// Creative-flight toggle (double-tap Space). When set, gravity is replaced
    /// by direct up/down control, but rapier still resolves collisions — you
    /// can't fly through terrain.
    pub flying: bool,
    /// Time (`Time::elapsed_secs`) of the last Space press, for double-tap
    /// detection. `None` until the first press / after a pair is consumed.
    pub last_space_tap: Option<f32>,
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
            crouch_collider(false),
            character_controller(),
            Transform::from_translation(translation),
        ))
        .id();
    state.spawned = true;
    state.entity = Some(ent);
    state.spawn_y = translation.y;
    state.vertical_velocity = 0.0;
    state.grounded = false;
    state.flying = false;
    state.last_space_tap = None;
    state.crouched = false;
}

/// Advance the vertical-velocity integrator one tick. Pure so it can be unit
/// tested without a live rapier world. `gravity_y` is negative (downward).
pub fn step_vertical(vel: f32, grounded: bool, jump: bool, dt: f32, gravity_y: f32) -> f32 {
    if jump && grounded {
        return JUMP_SPEED;
    }
    // Only re-stick to the ground when at rest or descending. During the early
    // ascent of a jump the capsule has barely left the surface, so rapier still
    // reports `grounded` — clamping there would cancel the jump after one frame.
    // While `vel > 0` we keep integrating gravity so the arc plays out.
    if grounded && vel <= 0.0 {
        return GROUND_STICK;
    }
    // Brisk, symmetric arc: integrate at CHARACTER_GRAVITY_SCALE× world gravity
    // both rising and falling (and when walking off a ledge). World/debris
    // gravity is unchanged.
    vel + gravity_y * CHARACTER_GRAVITY_SCALE * dt
}

/// Build the per-frame desired translation: horizontal = normalized heading ×
/// speed × dt, vertical = velocity × dt. Pure / testable.
pub fn desired_translation(move_world: Vec3, vertical_velocity: f32, speed: f32, dt: f32) -> Vec3 {
    let horizontal = move_world.normalize_or_zero() * speed * dt;
    Vec3::new(horizontal.x, vertical_velocity * dt, horizontal.z)
}

/// Register a Space press for double-tap detection. Returns the new
/// `last_space_tap` bookkeeping value and whether this press completed a
/// double-tap (two presses within `window`). A completed pair is *consumed*
/// (returns `None`) so a third quick press starts a fresh pair rather than
/// toggling again. Pure / testable.
pub fn register_space_tap(last: Option<f32>, now: f32, window: f32) -> (Option<f32>, bool) {
    match last {
        Some(t) if now - t <= window => (None, true),
        _ => (Some(now), false),
    }
}

/// Fly-mode vertical velocity (m/s) from the up/down keys: `+` up, `-` down,
/// `0` if neither or both. Sprint flies faster. Pure / testable.
pub fn fly_vertical_velocity(up: bool, down: bool, sprint: bool) -> f32 {
    let dir = (up as i32 - down as i32) as f32;
    let speed = if sprint { FLY_SPRINT_VERTICAL_SPEED } else { FLY_VERTICAL_SPEED };
    dir * speed
}

/// True when a standing-height capsule at `feet` would clear all terrain (other
/// than the player itself) — i.e. there is headroom to stand up. The probe is
/// the standing capsule shrunk by [`HEADROOM_SKIN_M`] on its top and radius, so
/// "clear" means a small margin of slack and standing won't clip a ceiling. A
/// one-shot rapier shape-overlap against the existing static colliders; one
/// frame of staleness is harmless against static terrain. Returns `true` if the
/// physics context isn't up yet (no terrain to block standing).
fn standing_headroom_clear(rapier: &ReadRapierContext<'_, '_>, player: Entity, feet: Vec3) -> bool {
    let Ok(ctx) = rapier.single() else {
        return true;
    };
    let (foot, head, radius) = capsule_endpoints(false);
    let probe = Collider::capsule(
        Vec3::new(0.0, foot, 0.0),
        Vec3::new(0.0, head - HEADROOM_SKIN_M, 0.0),
        radius - HEADROOM_SKIN_M,
    );
    let filter = QueryFilter::default().exclude_collider(player);
    let mut blocked = false;
    ctx.intersect_shape(feet, Quat::IDENTITY, &*probe.raw, filter, |_e| {
        blocked = true;
        false // stop on first hit
    });
    !blocked
}

/// Set `controller.translation` from gravity (or creative flight) + the input
/// intent, and resize the capsule for crouch. Ordered
/// `.before(PhysicsSet::SyncBackend)` so rapier consumes both this frame. The
/// move always goes through the controller, so terrain collision is enforced —
/// fly mode just removes gravity and adds up/down control.
///
/// **Crouch (hold C):** shrinks the capsule to ~0.9 m so you fit under a 1 m
/// ledge and lowers the eye to match (via `writeback_character`). Releasing C
/// only stands back up when a standing capsule would clear the terrain, so the
/// head can't pop up into a low ceiling.
///
/// **Fly mode (double-tap Space):** while `state.flying`, Space = ascend,
/// Left Ctrl = descend, Shift = faster. A single Space tap still jumps when
/// walking; a quick second tap toggles flight.
pub fn drive_character(
    cfg: Res<PhysicsConfig>,
    mode: Res<ViewMode>,
    keys: Res<ButtonInput<KeyCode>>,
    time: Res<Time>,
    intent: Res<CharacterIntent>,
    rapier: ReadRapierContext,
    mut state: ResMut<CharacterState>,
    mut q: Query<(Entity, &Transform, &mut Collider, &mut KinematicCharacterController), With<Player>>,
) {
    if !character_active(&cfg, *mode, &state) {
        return;
    }
    let dt = time.delta_secs().min(0.05);

    // Double-tap Space toggles creative flight (before the jump/gravity read so
    // the toggling frame doesn't also jump — the fly branch overrides vertical).
    if keys.just_pressed(KeyCode::Space) {
        let (next, toggled) = register_space_tap(state.last_space_tap, time.elapsed_secs(), DOUBLE_TAP_WINDOW_S);
        state.last_space_tap = next;
        if toggled {
            state.flying = !state.flying;
            state.vertical_velocity = 0.0;
            tracing::info!(target = "character", flying = state.flying, "fly mode toggled");
        }
    }

    let Ok((entity, transform, mut collider, mut controller)) = q.single_mut() else {
        return;
    };

    // Resolve crouch and resize the capsule on a stance flip only (no per-frame
    // collider churn). Feet stay at the origin, so the resize is upward-only and
    // never penetrates the ground.
    let want = intent.crouch;
    let headroom_clear = if state.crouched && !want {
        standing_headroom_clear(&rapier, entity, transform.translation)
    } else {
        true
    };
    let effective = resolve_crouch_state(want, headroom_clear);
    if effective != state.crouched {
        *collider = crouch_collider(effective);
        state.crouched = effective;
    }

    let mut speed;
    let snap: Option<CharacterLength>;
    if state.flying {
        // No gravity: vertical comes straight from the up/down keys. Collisions
        // are still resolved by the controller, and snap-to-ground is off so the
        // capsule isn't yanked back down while hovering.
        let up = keys.pressed(KeyCode::Space);
        let down = keys.pressed(KeyCode::ControlLeft) || keys.pressed(KeyCode::ControlRight);
        state.vertical_velocity = fly_vertical_velocity(up, down, intent.sprint);
        speed = if intent.sprint { FLY_SPRINT_SPEED } else { FLY_SPEED };
        snap = None;
    } else {
        let jump = keys.just_pressed(KeyCode::Space);
        state.vertical_velocity =
            step_vertical(state.vertical_velocity, state.grounded, jump, dt, cfg.gravity.y);
        speed = if intent.sprint { SPRINT_SPEED } else { WALK_SPEED };
        // A crouched walk is a slow shuffle (fly speed is unaffected).
        if state.crouched {
            speed *= CROUCH_WALK_SCALE;
        }
        // Disable snap while ascending so the jump isn't immediately cancelled.
        snap = if state.vertical_velocity > 0.0 {
            None
        } else {
            Some(CharacterLength::Absolute(SNAP_TO_GROUND_M))
        };
    }

    let translation = desired_translation(intent.move_world, state.vertical_velocity, speed, dt);
    controller.snap_to_ground = snap;
    controller.translation = Some(translation);
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
    // Lower the eye to match the *resolved* capsule stance (`state.crouched`,
    // set by `drive_character`) — not raw intent — so the eye can't pop up while
    // the body is physically held crouched under a low ceiling.
    fp.walk.set_crouch(state.crouched);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn jump_only_from_ground() {
        // Grounded + jump → take-off speed.
        assert_eq!(step_vertical(GROUND_STICK, true, true, 0.016, -9.81), JUMP_SPEED);
        // Airborne + jump → ignored (no double-jump), gravity keeps applying.
        // v = 2.0 + (-10 * SCALE) * 0.1 = 2.0 - 2.0 = 0.0 at SCALE = 2.
        let v = step_vertical(2.0, false, true, 0.1, -10.0);
        let expect = 2.0 + (-10.0 * CHARACTER_GRAVITY_SCALE) * 0.1;
        assert!((v - expect).abs() < 1e-6, "airborne jump ignored: {v}");
    }

    #[test]
    fn grounded_at_rest_holds_ground_stick() {
        // At rest / descending on the ground → re-stick.
        assert_eq!(step_vertical(GROUND_STICK, true, false, 0.016, -9.81), GROUND_STICK);
        assert_eq!(step_vertical(-3.0, true, false, 0.016, -9.81), GROUND_STICK);
    }

    #[test]
    fn grounded_but_ascending_keeps_jump_alive() {
        // Right after takeoff the capsule barely cleared the surface, so rapier
        // still reports grounded. The rising velocity must NOT be clamped — it
        // should keep decaying under gravity, or the jump dies after one frame.
        let v = step_vertical(JUMP_SPEED, /*grounded=*/ true, /*jump=*/ false, 0.05, -9.81);
        assert!(v > 0.0 && v < JUMP_SPEED, "ascending velocity decays under gravity: {v}");
        assert!((v - (JUMP_SPEED - 9.81 * CHARACTER_GRAVITY_SCALE * 0.05)).abs() < 1e-5);
    }

    #[test]
    fn gravity_is_symmetric_and_scaled() {
        // Rising and falling integrate the SAME (scaled) gravity — no asymmetry,
        // so the arc is a clean parabola with no "float up / slam down" split.
        let g = -10.0;
        let dt = 0.1;
        let step = (g * CHARACTER_GRAVITY_SCALE) * dt; // = -2.0 at SCALE = 2
        let up = step_vertical(3.0, false, false, dt, g); // ascending
        assert!((up - (3.0 + step)).abs() < 1e-6, "ascent: {up}");
        let down = step_vertical(-3.0, false, false, dt, g); // descending
        assert!((down - (-3.0 + step)).abs() < 1e-6, "descent: {down}");
        // Same Δv magnitude up and down.
        assert!(((3.0 - up) - (-3.0 - down)).abs() < 1e-6, "symmetric Δv");
        // And it's heavier than world gravity (snappier than 1×).
        assert!(step < g * dt, "scaled gravity exceeds world gravity");
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
        // Apex = v² / (2·g_eff), where g_eff = world gravity × the character scale.
        let g_eff = -g * CHARACTER_GRAVITY_SCALE;
        let analytic = JUMP_SPEED * JUMP_SPEED / (2.0 * g_eff);
        assert!((h - analytic).abs() < 0.05, "apex {h} vs analytic {analytic}");
    }

    #[test]
    fn double_tap_toggles_within_window() {
        let w = DOUBLE_TAP_WINDOW_S;
        // First press: records the time, no toggle.
        let (last, toggled) = register_space_tap(None, 1.00, w);
        assert_eq!((last, toggled), (Some(1.00), false));
        // Second press inside the window: toggle + consume (so a third quick
        // press starts fresh instead of toggling again).
        let (last, toggled) = register_space_tap(last, 1.00 + w * 0.5, w);
        assert_eq!((last, toggled), (None, true));
        // Third quick press after a consumed pair: fresh tap, no toggle.
        let (last, toggled) = register_space_tap(last, 1.00 + w * 0.6, w);
        assert_eq!((last, toggled), (Some(1.00 + w * 0.6), false));
    }

    #[test]
    fn slow_double_press_does_not_toggle() {
        let w = DOUBLE_TAP_WINDOW_S;
        let (last, _) = register_space_tap(None, 5.0, w);
        // Second press *outside* the window: not a double-tap; it just becomes
        // the new "first" press.
        let (last, toggled) = register_space_tap(last, 5.0 + w + 0.01, w);
        assert!(!toggled, "presses too far apart must not toggle");
        assert_eq!(last, Some(5.0 + w + 0.01));
    }

    #[test]
    fn fly_vertical_velocity_maps_keys() {
        // Up only → +; down only → -; neither / both → 0.
        assert_eq!(fly_vertical_velocity(true, false, false), FLY_VERTICAL_SPEED);
        assert_eq!(fly_vertical_velocity(false, true, false), -FLY_VERTICAL_SPEED);
        assert_eq!(fly_vertical_velocity(false, false, false), 0.0);
        assert_eq!(fly_vertical_velocity(true, true, false), 0.0);
        // Sprint flies faster (and never under gravity — purely key-driven).
        assert_eq!(fly_vertical_velocity(true, false, true), FLY_SPRINT_VERTICAL_SPEED);
        assert!(FLY_SPRINT_VERTICAL_SPEED > FLY_VERTICAL_SPEED);
    }

    #[test]
    fn capsule_endpoints_have_grounded_feet_and_expected_heights() {
        let (sf, sh, sr) = capsule_endpoints(false);
        let (cf, ch, cr) = capsule_endpoints(true);
        // Feet and radius are stance-invariant — only the head moves.
        assert_eq!((sf, sr), (CAPSULE_FOOT_Y, CAPSULE_RADIUS));
        assert_eq!((cf, cr), (CAPSULE_FOOT_Y, CAPSULE_RADIUS));
        assert_eq!(sh, CAPSULE_HEAD_Y_STAND);
        assert_eq!(ch, CAPSULE_HEAD_Y_CROUCH);
        assert!(ch < sh, "crouched head is below standing head");
        // Total capsule height = (head - foot) + 2·radius. Standing 1.8 m, crouched 0.9 m.
        let total = |f: f32, h: f32, r: f32| (h - f) + 2.0 * r;
        assert!((total(sf, sh, sr) - 1.8).abs() < 1e-6, "standing ~1.8 m");
        assert!((total(cf, ch, cr) - 0.9).abs() < 1e-6, "crouched ~0.9 m");
    }

    #[test]
    fn resolve_crouch_truth_table() {
        // Holding crouch always crouches, regardless of headroom.
        assert!(resolve_crouch_state(true, true));
        assert!(resolve_crouch_state(true, false));
        // Releasing: stand only when headroom is clear; stay crouched if blocked.
        assert!(!resolve_crouch_state(false, true));
        assert!(resolve_crouch_state(false, false));
    }

    #[test]
    fn crouch_collider_builds_expected_capsule() {
        let standing_col = crouch_collider(false);
        let crouched_col = crouch_collider(true);
        let standing = standing_col.as_capsule().expect("capsule shape");
        let crouched = crouched_col.as_capsule().expect("capsule shape");
        assert!((standing.radius() - CAPSULE_RADIUS).abs() < 1e-6);
        assert!((crouched.radius() - CAPSULE_RADIUS).abs() < 1e-6);
        // half_height = (head - foot) / 2 for the feet-origin segment.
        assert!((standing.half_height() - (CAPSULE_HEAD_Y_STAND - CAPSULE_FOOT_Y) / 2.0).abs() < 1e-6);
        assert!((crouched.half_height() - (CAPSULE_HEAD_Y_CROUCH - CAPSULE_FOOT_Y) / 2.0).abs() < 1e-6);
        assert!(crouched.half_height() < standing.half_height(), "crouched capsule is shorter");
    }

    #[test]
    fn crouched_eye_sits_below_capsule_crown() {
        // The lowered eye (standing eye-height × the view crate's crouch ratio)
        // must sit below the crouched capsule crown (head + radius) so the
        // camera never pokes above the physical body under a ledge.
        let eye = 1.7 * atomr_worlds_view::CROUCH_EYE_RATIO; // WalkCamera default eye_height_m
        let crown = CAPSULE_HEAD_Y_CROUCH + CAPSULE_RADIUS;
        assert!(eye < crown, "crouched eye {eye} below crown {crown}");
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

    /// `drive_character` resizes the capsule when the crouch stance flips, and
    /// writes the resolved stance to `state.crouched`. With no rapier context in
    /// the test `App`, the headroom probe reads "clear", so releasing crouch
    /// stands back up — exercising the resize-on-flip + writeback. The actual
    /// headroom *blocking* (a real overlap) is covered by interactive
    /// verification, since the repo has no stepped-rapier headless test harness.
    #[test]
    fn drive_character_resizes_capsule_on_stance_flip() {
        fn setup(start_crouched: bool, intent_crouch: bool) -> (App, Entity) {
            let mut app = App::new();
            app.add_plugins(MinimalPlugins);
            app.init_resource::<ButtonInput<KeyCode>>();
            app.insert_resource(ViewMode::Fp);
            app.insert_resource(PhysicsConfig::default());
            app.insert_resource(CharacterIntent { crouch: intent_crouch, ..default() });
            let ent = app
                .world_mut()
                .spawn((
                    Player,
                    RigidBody::KinematicPositionBased,
                    crouch_collider(start_crouched),
                    character_controller(),
                    Transform::from_translation(Vec3::new(0.0, 50.0, 0.0)),
                ))
                .id();
            app.insert_resource(CharacterState {
                spawned: true,
                entity: Some(ent),
                crouched: start_crouched,
                ..default()
            });
            app.add_systems(Update, drive_character);
            (app, ent)
        }

        let half = |c: &Collider| c.as_capsule().unwrap().half_height();
        let stand_hh = (CAPSULE_HEAD_Y_STAND - CAPSULE_FOOT_Y) / 2.0;
        let crouch_hh = (CAPSULE_HEAD_Y_CROUCH - CAPSULE_FOOT_Y) / 2.0;

        // Standing + hold crouch → capsule shrinks, state goes crouched.
        {
            let (mut app, ent) = setup(false, true);
            app.update();
            assert!(app.world().resource::<CharacterState>().crouched);
            let c = app.world().get::<Collider>(ent).unwrap();
            assert!((half(c) - crouch_hh).abs() < 1e-6, "held crouch shrinks capsule");
        }
        // Crouched + release with clear headroom → stands back up.
        {
            let (mut app, ent) = setup(true, false);
            app.update();
            assert!(!app.world().resource::<CharacterState>().crouched);
            let c = app.world().get::<Collider>(ent).unwrap();
            assert!((half(c) - stand_hh).abs() < 1e-6, "released crouch restores standing capsule");
        }
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
