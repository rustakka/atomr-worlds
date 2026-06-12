//! Pure, deterministic rigid-body integration for host-authoritative debris.
//!
//! The host owns debris motion (Rec 4 Slice 2): when a fracture detaches a
//! floating island the host builds a [`DebrisBody`], steps it here each tick,
//! and broadcasts the resulting pose for clients to interpolate. Keeping the
//! integrator in this crate preserves the determinism boundary — it is a pure
//! function of its inputs with **no Bevy / rapier / async** types — and lets the
//! host stay rapier-free.
//!
//! Unlike the integer fracture *decision* (see [`crate::flood_fill`]), debris
//! floats do **not** need cross-machine byte-determinism: a single host is the
//! authority and broadcasts the result, so plain `f64` is fine. The integrator
//! is still deterministic *on one machine* (same inputs → same output), which is
//! what the host tests rely on.
//!
//! # Model (slice-1 scope)
//!
//! Semi-implicit (symplectic) Euler under gravity, with a "settle believably"
//! terrain collision rather than a contact-manifold solver: the body is treated
//! as its axis-aligned box and rested on the highest solid voxel surface it
//! overlaps, with restitution + tangential friction on contact and a sleep
//! threshold. Terrain solidity is queried through an injected
//! `is_solid(world_voxel) -> bool` closure, mirroring how
//! [`crate::flood_fill::connected_components`] and [`crate::analyze_region`]
//! take their predicates — so this module never depends on host/brick types.
//!
//! Angular motion **is** now integrated: orientation advances from the body's
//! `angular_velocity` each tick ([`Quat::integrate`]), with the spin seeded at
//! fracture time from the off-center impulse (see the host's `handle_fracture`).
//! There is no per-tick angular damping — the existing sleep machinery
//! (`angular_sleep_eps`) terminates spin when the body settles. *Contact-induced*
//! torque (a corner strike imparting tumble) is still deferred: `resolve_terrain`
//! treats the body as an axis-aligned box and adjusts linear velocity only.

use atomr_worlds_core::{DVec3, IVec3};

use crate::debris::DebrisBody;

/// Standard gravity magnitude (m/s²). Matches the client's world gravity so the
/// host authority and any client-side prediction agree. (The client currently
/// hardcodes the same value; a shared core constant is a future cleanup.)
pub const GRAVITY_MPS2: f64 = 9.81;

/// Tunables for one [`step_body`]. All SI; defaults chosen for "settles
/// believably," not physical fidelity.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SimParams {
    /// Gravitational acceleration (default `(0, -GRAVITY_MPS2, 0)`).
    pub gravity: DVec3,
    /// Bounce coefficient on the contact axis for impacts faster than
    /// [`Self::bounce_threshold`]. `0` = no bounce, `1` = elastic.
    pub restitution: f64,
    /// Tangential velocity damping on contact, in `0..=1` (`0` = frictionless).
    pub friction: f64,
    /// Incoming contact speed (m/s) below which the body rests instead of
    /// bouncing. Keeps slow gravity-accumulated contact from jittering forever.
    pub bounce_threshold: f64,
    /// Linear speed (m/s) below which the body counts as quiescent.
    pub linear_sleep_eps: f64,
    /// Angular speed (rad/s) below which the body counts as quiescent.
    pub angular_sleep_eps: f64,
    /// Consecutive quiescent ticks before the body is put to sleep.
    pub sleep_ticks: u32,
    /// Velocity clamp (m/s), bounding the per-tick step so a fast body can't
    /// tunnel through thin terrain.
    pub max_speed: f64,
}

impl Default for SimParams {
    fn default() -> Self {
        Self {
            gravity: DVec3::new(0.0, -GRAVITY_MPS2, 0.0),
            restitution: 0.15,
            friction: 0.4,
            bounce_threshold: 0.5,
            linear_sleep_eps: 0.05,
            angular_sleep_eps: 0.05,
            sleep_ticks: 30,
            max_speed: 40.0,
        }
    }
}

/// Per-body integrator bookkeeping kept *outside* [`DebrisBody`] so the
/// serialized body type is unchanged.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SimState {
    /// `true` once the body has come to rest; [`step_body`] then no-ops.
    pub sleeping: bool,
    /// Count of consecutive sub-threshold ticks toward sleeping.
    pub sub_threshold_ticks: u32,
}

#[inline]
fn scale(v: DVec3, s: f64) -> DVec3 {
    DVec3::new(v.x * s, v.y * s, v.z * s)
}

/// Advance one body by `dt` seconds under gravity + terrain collision. Returns
/// nothing; mutates `body` (pose/velocity) and `state` (sleep bookkeeping).
///
/// `is_solid(world_voxel)` reports whether a **world voxel coordinate** is solid
/// terrain. A sleeping body is skipped (cheap idle). Pure: identical inputs and
/// closure outputs yield identical mutations on one machine.
pub fn step_body(
    body: &mut DebrisBody,
    state: &mut SimState,
    params: &SimParams,
    dt: f64,
    is_solid: impl Fn(IVec3) -> bool,
) {
    if state.sleeping {
        return;
    }

    // Semi-implicit Euler: accelerate, clamp, then integrate position.
    body.linear_velocity = body.linear_velocity + scale(params.gravity, dt);
    let speed = body.linear_velocity.length();
    if speed > params.max_speed && speed > 0.0 {
        body.linear_velocity = scale(body.linear_velocity, params.max_speed / speed);
    }
    body.position = body.position + scale(body.linear_velocity, dt);
    // Angular: advance orientation from the (constant-between-contacts) world-
    // frame angular velocity. Seeded at fracture; no torque accumulates here.
    if body.angular_velocity != DVec3::ZERO {
        body.orientation = body.orientation.integrate(body.angular_velocity, dt);
    }

    resolve_terrain(body, params, &is_solid);

    // Sleep accounting: quiescent for `sleep_ticks` consecutive ticks → sleep.
    let lin = body.linear_velocity.length();
    let ang = body.angular_velocity.length();
    if lin < params.linear_sleep_eps && ang < params.angular_sleep_eps {
        state.sub_threshold_ticks = state.sub_threshold_ticks.saturating_add(1);
        if state.sub_threshold_ticks >= params.sleep_ticks {
            state.sleeping = true;
            body.linear_velocity = DVec3::ZERO;
            body.angular_velocity = DVec3::ZERO;
        }
    } else {
        state.sub_threshold_ticks = 0;
    }
}

/// Rest the body's axis-aligned box on the highest solid voxel surface it
/// overlaps. Identity orientation in slice 1 means the box is axis-aligned, so
/// the body-local grid maps directly to a world AABB.
fn resolve_terrain(body: &mut DebrisBody, params: &SimParams, is_solid: &impl Fn(IVec3) -> bool) {
    const EPS: f64 = 1e-9;
    let vs = body.voxel_size_m;
    if vs <= 0.0 {
        return;
    }
    let [nx, ny, nz] = body.dims;
    // World AABB: the local grid's `(0,0,0)` corner sits at `position - com`
    // (since `position = world_origin_corner + com`), and the grid spans
    // `dims * voxel_size_m`.
    let corner = body.position - body.mass.com;
    let aabb_min = corner;
    let aabb_max = corner + DVec3::new(nx as f64 * vs, ny as f64 * vs, nz as f64 * vs);

    let cell = |m: f64| (m / vs).floor() as i64;
    let (x0, x1) = (cell(aabb_min.x), cell(aabb_max.x - EPS));
    let (y0, y1) = (cell(aabb_min.y), cell(aabb_max.y - EPS));
    let (z0, z1) = (cell(aabb_min.z), cell(aabb_max.z - EPS));

    // Highest solid surface (top face = `(vy+1) * vs`) the body overlaps.
    let mut surface_y = f64::NEG_INFINITY;
    for vx in x0..=x1 {
        for vy in y0..=y1 {
            for vz in z0..=z1 {
                if is_solid(IVec3::new(vx, vy, vz)) {
                    let top = (vy + 1) as f64 * vs;
                    if top > surface_y {
                        surface_y = top;
                    }
                }
            }
        }
    }

    if surface_y > aabb_min.y {
        // Penetration → lift the body so its base rests flush on the surface.
        body.position.y += surface_y - aabb_min.y;
        if body.linear_velocity.y < 0.0 {
            if -body.linear_velocity.y < params.bounce_threshold {
                body.linear_velocity.y = 0.0; // slow contact → rest, no bounce
            } else {
                body.linear_velocity.y = -params.restitution * body.linear_velocity.y;
            }
            let damp = (1.0 - params.friction).clamp(0.0, 1.0);
            body.linear_velocity.x *= damp;
            body.linear_velocity.z *= damp;
        }
    }
}

/// Convenience: step a slice of `(body, state)` pairs in order. The host stores
/// its registry in a map and calls [`step_body`] per entry, but this keeps a
/// single-call path for tests and any batch caller.
pub fn step_all(
    bodies: &mut [(DebrisBody, SimState)],
    params: &SimParams,
    dt: f64,
    is_solid: impl Fn(IVec3) -> bool,
) {
    for (body, state) in bodies.iter_mut() {
        step_body(body, state, params, dt, &is_solid);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::default_physics_palette;
    use atomr_worlds_core::material_physics::material_id;

    const DT: f64 = 1.0 / 30.0;

    /// A 1 m³ stone body whose local `(0,0,0)` corner starts at world y =
    /// `corner_y`. COM is at the voxel center, so `position.y == corner_y + 0.5`.
    fn unit_stone(corner_y: f64) -> DebrisBody {
        let palette = default_physics_palette();
        DebrisBody::from_voxels(
            IVec3::new(0, corner_y as i64, 0),
            [1, 1, 1],
            vec![material_id::STONE],
            1.0,
            DVec3::new(0.0, corner_y, 0.0),
            &palette,
        )
    }

    /// Solid half-space: every voxel below y = 0 is terrain (top surface y = 0).
    fn floor(p: IVec3) -> bool {
        p.y < 0
    }

    fn energy(body: &DebrisBody, params: &SimParams) -> f64 {
        let v = body.linear_velocity.length();
        // PE measured against gravity direction (downward positive g).
        let g = params.gravity.length();
        0.5 * body.mass.mass_kg * v * v + body.mass.mass_kg * g * body.position.y
    }

    #[test]
    fn box_falls_and_rests_on_floor() {
        let params = SimParams::default();
        let mut body = unit_stone(5.0);
        let mut state = SimState::default();
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
        }
        // Corner rests on y = 0 → COM at y ≈ 0.5.
        assert!((body.position.y - 0.5).abs() < 0.05, "y={}", body.position.y);
        assert!(body.linear_velocity.length() < 1e-3);
    }

    #[test]
    fn comes_to_sleep_after_settling() {
        let params = SimParams::default();
        let mut body = unit_stone(3.0);
        let mut state = SimState::default();
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
        }
        assert!(state.sleeping);
    }

    #[test]
    fn dissipates_energy_on_landing() {
        let params = SimParams::default();
        let mut body = unit_stone(5.0);
        let mut state = SimState::default();
        let e0 = energy(&body, &params);
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
        }
        let e1 = energy(&body, &params);
        // Restitution < 1 + friction → settled energy is strictly below the
        // starting energy (no spurious energy gain from the integrator).
        assert!(e1 < e0, "e0={e0} e1={e1}");
    }

    #[test]
    fn no_floor_keeps_falling_and_never_sleeps() {
        let params = SimParams::default();
        let mut body = unit_stone(0.0);
        let mut state = SimState::default();
        let mut last_y = body.position.y;
        for _ in 0..120 {
            step_body(&mut body, &mut state, &params, DT, |_| false);
            assert!(body.position.y < last_y, "should keep descending");
            last_y = body.position.y;
        }
        assert!(!state.sleeping);
        // Speed clamped, not runaway.
        assert!(body.linear_velocity.length() <= params.max_speed + 1e-9);
    }

    #[test]
    fn deterministic_on_one_machine() {
        let params = SimParams::default();
        let run = || {
            let mut body = unit_stone(4.0);
            let mut state = SimState::default();
            for _ in 0..200 {
                step_body(&mut body, &mut state, &params, DT, floor);
            }
            (body.position, body.linear_velocity, state)
        };
        let a = run();
        let b = run();
        assert_eq!(a.0, b.0);
        assert_eq!(a.1, b.1);
        assert_eq!(a.2, b.2);
    }

    #[test]
    fn high_speed_does_not_tunnel_through_floor() {
        let params = SimParams::default();
        let mut body = unit_stone(2.0);
        body.linear_velocity = DVec3::new(0.0, -1000.0, 0.0); // clamps to max_speed
        let mut state = SimState::default();
        let mut min_corner_y = f64::INFINITY;
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
            min_corner_y = min_corner_y.min(body.position.y - 0.5);
        }
        // The clamp + per-tick resolve keep the body from passing through the
        // half-space; its base never sinks more than a fraction of a voxel.
        assert!(min_corner_y > -0.5, "min_corner_y={min_corner_y}");
        assert!((body.position.y - 0.5).abs() < 0.05);
    }

    #[test]
    fn step_all_advances_every_body() {
        let params = SimParams::default();
        let mut bodies = vec![
            (unit_stone(10.0), SimState::default()),
            (unit_stone(20.0), SimState::default()),
        ];
        let y0: Vec<f64> = bodies.iter().map(|(b, _)| b.position.y).collect();
        step_all(&mut bodies, &params, DT, |_| false);
        for (i, (b, _)) in bodies.iter().enumerate() {
            assert!(b.position.y < y0[i]);
        }
    }

    fn quat_norm(q: atomr_worlds_core::Quat) -> f64 {
        (q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w).sqrt()
    }

    #[test]
    fn orientation_advances_under_angular_velocity() {
        let params = SimParams::default();
        let mut body = unit_stone(0.0);
        body.angular_velocity = DVec3::new(0.0, 2.0, 0.0);
        let mut state = SimState::default();
        for _ in 0..30 {
            step_body(&mut body, &mut state, &params, DT, |_| false); // free fall
        }
        // Spun about Y → orientation has left identity (w dropped below 1).
        assert!(body.orientation.w.abs() < 0.999, "orient={:?}", body.orientation);
    }

    #[test]
    fn zero_angular_velocity_keeps_identity() {
        let params = SimParams::default();
        let mut body = unit_stone(0.0); // angular_velocity defaults to ZERO
        let mut state = SimState::default();
        for _ in 0..60 {
            step_body(&mut body, &mut state, &params, DT, |_| false);
        }
        assert_eq!(body.orientation, atomr_worlds_core::Quat::IDENTITY);
    }

    #[test]
    fn orientation_stays_unit_while_spinning() {
        let params = SimParams::default();
        let mut body = unit_stone(0.0);
        body.angular_velocity = DVec3::new(1.0, -2.0, 0.5);
        let mut state = SimState::default();
        for _ in 0..600 {
            step_body(&mut body, &mut state, &params, DT, |_| false);
        }
        assert!((quat_norm(body.orientation) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn fast_spin_keeps_a_resting_body_awake() {
        // Linear motion settles but angular speed stays above the sleep eps →
        // the body must not sleep (and orientation keeps advancing).
        let params = SimParams::default();
        let mut body = unit_stone(3.0);
        body.angular_velocity = DVec3::new(0.0, 5.0, 0.0); // ≫ angular_sleep_eps
        let mut state = SimState::default();
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
        }
        assert!((body.position.y - 0.5).abs() < 0.05, "should rest linearly");
        assert!(!state.sleeping, "spinning body must stay awake");
        assert!(body.orientation.w.abs() < 0.999, "should have rotated");
    }

    #[test]
    fn slow_spin_still_lets_a_body_sleep() {
        // Angular speed below `angular_sleep_eps` must not block sleep.
        let params = SimParams::default();
        let mut body = unit_stone(3.0);
        body.angular_velocity = DVec3::new(0.0, params.angular_sleep_eps * 0.5, 0.0);
        let mut state = SimState::default();
        for _ in 0..400 {
            step_body(&mut body, &mut state, &params, DT, floor);
        }
        assert!(state.sleeping, "sub-eps spin should still sleep");
        assert_eq!(body.angular_velocity, DVec3::ZERO, "sleep zeroes spin");
    }

    #[test]
    fn deterministic_orientation_on_one_machine() {
        let params = SimParams::default();
        let run = || {
            let mut body = unit_stone(4.0);
            body.angular_velocity = DVec3::new(0.3, 1.1, -0.7);
            let mut state = SimState::default();
            for _ in 0..200 {
                step_body(&mut body, &mut state, &params, DT, floor);
            }
            body.orientation
        };
        assert_eq!(run(), run());
    }
}
