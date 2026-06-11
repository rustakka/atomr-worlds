//! Host-authoritative debris streaming (Rec 4 Slice 2).
//!
//! The host owns debris motion now: it integrates each detached body and
//! broadcasts a `DebrisStateDelta` per tick. This module opens a subscription to
//! that stream, buffers the latest two samples per body, and interpolates each
//! kinematic debris entity's transform between them — replacing the old local
//! rapier simulation (`RigidBody::Dynamic` + `settle_and_despawn_debris`).
//!
//! Everything here is reached only with the `physics` feature +
//! `PhysicsConfig.enabled` (the plugin adds these systems only when physics is
//! on), so the harness / physics-off path is byte-identical. The pose floats are
//! interpolated, never replayed, and never flow into `GetBrick`.

use std::collections::HashMap;
use std::sync::mpsc::Receiver;
use std::sync::Mutex;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_proto::{DebrisStateDelta, WorldEvent, AABB};
use atomr_worlds_view::WorldQuery;
use bevy::prelude::*;

use super::debris::HostDebris;
use crate::perf::Phase;
use crate::world_runtime::{ActiveWorld, WorldRuntime};

/// Half-extent (world voxels/metres) of the fixed debris subscription region,
/// centred on the world origin. Sized to cover the near-spawn play/carve area
/// while keeping the host's one-time initial brick-snapshot burst small
/// (`(2*H/BRICK_EDGE)^3` bricks at edge 16 ⇒ 8³). Debris carved well outside
/// this region won't stream — a player-following or debris-only (no initial
/// snapshot) subscription is the documented follow-up.
const DEBRIS_SUB_HALF: i64 = 64;

/// Host tick period (s) the interpolation spreads each delta over. Matches the
/// host's ~30 Hz debris tick so motion is continuous between samples.
const DEBRIS_TICK_DT: f32 = 1.0 / 30.0;

/// Despawn a settled body this long after the host marks it sleeping.
const SETTLE_TIMEOUT_S: f64 = 2.0;
/// Despawn any body whose stream has been silent this long — the host retired it
/// (or it left the subscribed region). The loss-robust catch-all.
const STREAM_DROP_S: f64 = 5.0;
/// Belt-and-suspenders kill plane (the host also retires below its own floor).
const KILL_Y: f32 = -512.0;

/// One buffered host snapshot, in client render units (world metres). The host
/// streams the body's local `(0,0,0)` corner, which is exactly the frame
/// `spawn_island` places the entity transform in, so no conversion is needed.
#[derive(Clone, Copy)]
struct DebrisSample {
    tick: u64,
    pos: Vec3,
    orient: Quat,
    sleeping: bool,
}

/// Per-id interpolation state: `prev` + `curr` bracket render time; `last_update`
/// (seconds) drives both interpolation and retirement.
#[derive(Default)]
struct DebrisTrack {
    /// `None` until the matching `SpawnDebris` entity is created.
    entity: Option<Entity>,
    prev: Option<DebrisSample>,
    curr: Option<DebrisSample>,
    last_update: f64,
}

/// Interpolation buffer for all live host debris, keyed by host debris id.
#[derive(Resource)]
pub struct DebrisInterp {
    tracks: HashMap<u32, DebrisTrack>,
    tick_dt: f32,
}

impl Default for DebrisInterp {
    fn default() -> Self {
        Self { tracks: HashMap::new(), tick_dt: DEBRIS_TICK_DT }
    }
}

impl DebrisInterp {
    /// Ingest a host snapshot, shifting `curr → prev`. Stale / duplicate ticks
    /// (≤ the current sample) are dropped so out-of-order delivery can't rewind.
    fn ingest(&mut self, d: &DebrisStateDelta, now: f64) {
        let sample = DebrisSample {
            tick: d.tick,
            pos: Vec3::new(d.pos[0], d.pos[1], d.pos[2]),
            orient: Quat::from_xyzw(d.orient[0], d.orient[1], d.orient[2], d.orient[3]).normalize(),
            sleeping: d.sleeping,
        };
        let track = self.tracks.entry(d.id).or_default();
        if let Some(c) = track.curr {
            if sample.tick <= c.tick {
                return;
            }
        }
        track.prev = track.curr;
        track.curr = Some(sample);
        track.last_update = now;
    }

    /// Bind a spawned entity to its host id. Buffered samples are kept, so the
    /// entity snaps to the latest known host pose on its first interpolated
    /// frame (handles deltas that arrived before the spawn).
    pub fn attach_entity(&mut self, id: u32, entity: Entity, now: f64) {
        let track = self.tracks.entry(id).or_default();
        track.entity = Some(entity);
        if track.last_update == 0.0 {
            track.last_update = now;
        }
    }
}

/// Interpolated `(translation, rotation)` for a track, or `None` if it has no
/// sample yet.
fn track_pose(track: &DebrisTrack, now: f64, tick_dt: f32) -> Option<(Vec3, Quat)> {
    let curr = track.curr?;
    // Snap on the first sample; freeze exactly at the settled pose.
    let Some(prev) = track.prev else {
        return Some((curr.pos, curr.orient));
    };
    if curr.sleeping {
        return Some((curr.pos, curr.orient));
    }
    // Animate prev → curr over one tick of wall time since `curr` arrived
    // (renders ~one host tick behind, smoothing the 30 Hz stream); hold at
    // `curr` once we run past it (starvation, e.g. a body that just settled).
    let alpha = if tick_dt > 0.0 {
        ((now - track.last_update) as f32 / tick_dt).clamp(0.0, 1.0)
    } else {
        1.0
    };
    Some((prev.pos.lerp(curr.pos, alpha), prev.orient.slerp(curr.orient, alpha)))
}

/// Bevy resource owning the host debris subscription. The `Receiver` is behind a
/// `Mutex` so the resource is `Sync`; the pump drains it single-threaded. `pub`
/// only so it can appear in the `pub` system signatures (the module is private,
/// so it stays crate-internal) — mirrors `FractureWorkers`.
#[derive(Resource)]
pub struct DebrisSubscription {
    rx: Mutex<Receiver<WorldEvent>>,
}

/// Startup: open the debris subscription on the host stream. Reuses the existing
/// [`WorldQuery::subscribe_region`]; the initial brick snapshots it triggers are
/// drained and ignored by the pump (the FP client streams its own bricks).
pub fn init_debris_subscription(
    mut commands: Commands,
    runtime: Res<WorldRuntime>,
    active: Res<ActiveWorld>,
) {
    let region = AABB::new(
        IVec3::new(-DEBRIS_SUB_HALF, -DEBRIS_SUB_HALF, -DEBRIS_SUB_HALF),
        IVec3::new(DEBRIS_SUB_HALF, DEBRIS_SUB_HALF, DEBRIS_SUB_HALF),
    );
    let rx = runtime.query.subscribe_region(&active.addr, region, Lod::new(0));
    commands.insert_resource(DebrisSubscription { rx: Mutex::new(rx) });
}

/// Update: drain the host stream into the interpolation buffer (ignoring every
/// non-`DebrisStates` event — brick snapshots, voxel/region deltas, etc.).
pub fn pump_debris_states(
    sub: Res<DebrisSubscription>,
    time: Res<Time>,
    mut interp: ResMut<DebrisInterp>,
    perf: Res<crate::perf::Perf>,
) {
    let _scope = perf.scope(Phase::DebrisStream);
    let now = time.elapsed_secs_f64();
    let Ok(rx) = sub.rx.lock() else { return };
    while let Ok(ev) = rx.try_recv() {
        if let WorldEvent::DebrisStates { deltas, .. } = ev {
            for d in &deltas {
                interp.ingest(d, now);
            }
        }
    }
}

/// Update (before `PhysicsSet::SyncBackend`): drive each kinematic debris
/// entity's transform from the interpolated host pose, so rapier sees the new
/// pose the same frame and resolves player-vs-debris contacts.
pub fn apply_debris_interpolation(
    time: Res<Time>,
    interp: Res<DebrisInterp>,
    mut q: Query<&mut Transform, With<HostDebris>>,
    perf: Res<crate::perf::Perf>,
) {
    let _scope = perf.scope(Phase::DebrisStream);
    let now = time.elapsed_secs_f64();
    for track in interp.tracks.values() {
        let Some(ent) = track.entity else { continue };
        let Some((pos, orient)) = track_pose(track, now, interp.tick_dt) else { continue };
        if let Ok(mut tf) = q.get_mut(ent) {
            tf.translation = pos;
            tf.rotation = orient;
        }
    }
}

/// Update: reap debris whose host body settled or stopped streaming, despawning
/// the entity and dropping the track. Replaces `debris::settle_and_despawn_debris`
/// (which keyed off the now-absent local rapier `Sleeping`).
pub fn retire_host_debris(
    time: Res<Time>,
    mut interp: ResMut<DebrisInterp>,
    q: Query<(Entity, &HostDebris, &Transform)>,
    mut commands: Commands,
) {
    let now = time.elapsed_secs_f64();
    // Which host bodies are retiring (settled long enough, or stream silent)?
    let mut retire: Vec<u32> = Vec::new();
    for (id, track) in interp.tracks.iter() {
        let quiet = now - track.last_update;
        let sleeping = track.curr.map(|c| c.sleeping).unwrap_or(false);
        if (sleeping && quiet > SETTLE_TIMEOUT_S) || quiet > STREAM_DROP_S {
            retire.push(*id);
        }
    }
    // Despawn the live entity for each retiring id (keyed by `HostDebris.id`, so
    // the despawn targets the actual entity), plus any that fell below the kill
    // plane; drop the matching track.
    for (ent, hd, tf) in &q {
        if retire.contains(&hd.id) || tf.translation.y < KILL_Y {
            commands.entity(ent).despawn();
            interp.tracks.remove(&hd.id);
        }
    }
    // Retiring tracks that never got an entity (orphan pre-spawn deltas).
    for id in retire {
        interp.tracks.remove(&id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(tick: u64, pos: Vec3, sleeping: bool) -> DebrisSample {
        DebrisSample { tick, pos, orient: Quat::IDENTITY, sleeping }
    }

    fn delta(id: u32, tick: u64, y: f32) -> DebrisStateDelta {
        DebrisStateDelta {
            id,
            tick,
            pos: [0.0, y, 0.0],
            vel: [0.0; 3],
            orient: [0.0, 0.0, 0.0, 1.0],
            ang_vel: [0.0; 3],
            sleeping: false,
        }
    }

    #[test]
    fn ingest_shifts_curr_to_prev_and_drops_stale() {
        let mut interp = DebrisInterp::default();
        interp.ingest(&delta(1, 10, 5.0), 1.0);
        interp.ingest(&delta(1, 11, 4.0), 2.0);
        let t = &interp.tracks[&1];
        assert_eq!(t.curr.unwrap().tick, 11);
        assert_eq!(t.prev.unwrap().tick, 10);
        // A stale (older) tick is ignored — out-of-order delivery can't rewind.
        interp.ingest(&delta(1, 9, 9.0), 3.0);
        assert_eq!(interp.tracks[&1].curr.unwrap().tick, 11);
    }

    #[test]
    fn track_pose_interpolates_midpoint() {
        let track = DebrisTrack {
            entity: None,
            prev: Some(sample(10, Vec3::new(0.0, 0.0, 0.0), false)),
            curr: Some(sample(20, Vec3::new(2.0, 0.0, 0.0), false)),
            last_update: 0.0,
        };
        // now = half a tick after `curr` arrived → alpha 0.5.
        let (pos, _) = track_pose(&track, (DEBRIS_TICK_DT / 2.0) as f64, DEBRIS_TICK_DT).unwrap();
        assert!((pos.x - 1.0).abs() < 1e-4, "pos.x={}", pos.x);
    }

    #[test]
    fn track_pose_snaps_on_first_sample() {
        let track = DebrisTrack {
            entity: None,
            prev: None,
            curr: Some(sample(5, Vec3::new(7.0, 8.0, 9.0), false)),
            last_update: 0.0,
        };
        let (pos, _) = track_pose(&track, 100.0, DEBRIS_TICK_DT).unwrap();
        assert_eq!(pos, Vec3::new(7.0, 8.0, 9.0));
    }

    #[test]
    fn track_pose_freezes_exactly_when_sleeping() {
        let track = DebrisTrack {
            entity: None,
            prev: Some(sample(10, Vec3::new(0.0, 0.0, 0.0), false)),
            curr: Some(sample(20, Vec3::new(2.0, 0.0, 0.0), true)),
            last_update: 0.0,
        };
        // Mid-interval, a sleeping body sits exactly at `curr` (no interpolation).
        let (pos, _) = track_pose(&track, (DEBRIS_TICK_DT / 2.0) as f64, DEBRIS_TICK_DT).unwrap();
        assert_eq!(pos, Vec3::new(2.0, 0.0, 0.0));
    }

    #[test]
    fn track_pose_slerps_orientation() {
        let prev = sample(10, Vec3::ZERO, false);
        let mut curr = sample(20, Vec3::ZERO, false);
        curr.orient = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2); // 90°
        let track = DebrisTrack { entity: None, prev: Some(prev), curr: Some(curr), last_update: 0.0 };
        let (_, q) = track_pose(&track, (DEBRIS_TICK_DT / 2.0) as f64, DEBRIS_TICK_DT).unwrap();
        let angle = q.to_axis_angle().1; // half-way → 45°
        assert!((angle - std::f32::consts::FRAC_PI_4).abs() < 1e-3, "angle={angle}");
    }

    #[test]
    fn attach_entity_keeps_buffered_samples() {
        let mut interp = DebrisInterp::default();
        // A delta arrives before the spawn (the spawn-vs-delta race).
        interp.ingest(&delta(7, 1, 3.0), 1.0);
        interp.attach_entity(7, Entity::PLACEHOLDER, 2.0);
        let t = &interp.tracks[&7];
        assert_eq!(t.entity, Some(Entity::PLACEHOLDER));
        assert!(t.curr.is_some(), "the pre-spawn sample is retained");
    }
}
