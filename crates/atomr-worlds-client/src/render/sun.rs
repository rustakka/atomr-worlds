//! World clock + sun-sync system.

use bevy::pbr::DistanceFog;
use bevy::prelude::*;

use super::config::RenderConfig;
use crate::world_stream::ChunkStreamer;

/// Marker component on the directional sun light entity. Set by FpPlugin.
#[derive(Component)]
pub struct WorldSunMarker;

/// World hours-of-day in `[0, 24)`. Defaults to noon (12.0).
#[derive(Resource, Clone, Copy, Debug)]
pub struct WorldTime(pub f32);

impl Default for WorldTime {
    fn default() -> Self {
        Self(12.0)
    }
}

impl WorldTime {
    pub fn hours(self) -> f32 {
        self.0.rem_euclid(24.0)
    }

    pub fn set(&mut self, hours: f32) {
        self.0 = hours.rem_euclid(24.0);
    }
}

/// Advance `WorldTime` when [`RenderConfig::time_advances_automatically`]
/// is on. Harness scenarios usually drive the clock directly via
/// `set_time_of_day`, so the default is off.
pub fn advance_world_time(
    cfg: Res<RenderConfig>,
    time: Res<Time>,
    mut world_time: ResMut<WorldTime>,
) {
    if !cfg.time_advances_automatically {
        return;
    }
    let dt_hours = time.delta_secs() / cfg.seconds_per_hour.max(1e-3);
    world_time.0 = (world_time.0 + dt_hours).rem_euclid(24.0);
}

/// Read `WorldTime` + the sun-curve strategy each frame and write the
/// sun's transform / color / illuminance and the ambient light's color /
/// brightness.
#[allow(clippy::type_complexity)]
pub fn sync_sun(
    cfg: Res<RenderConfig>,
    world_time: Res<WorldTime>,
    mut sun_q: Query<(&mut Transform, &mut DirectionalLight), With<WorldSunMarker>>,
    mut ambient: Option<ResMut<AmbientLight>>,
) {
    let state = cfg.sun_curve.sun_state(world_time.0);
    for (mut tx, mut light) in sun_q.iter_mut() {
        // `look_to` orients a light so it shines along `forward = direction`.
        // Pick a safe up vector that isn't parallel to the sun direction.
        let up = if state.direction.y.abs() > 0.95 {
            Vec3::Z
        } else {
            Vec3::Y
        };
        tx.look_to(state.direction, up);
        light.color = state.color;
        light.illuminance = state.illuminance;
    }
    if let Some(amb) = ambient.as_deref_mut() {
        let (color, brightness) = cfg.sun_curve.ambient(world_time.0);
        amb.color = color;
        // The Bevy 0.13 AmbientLight.brightness scale needs ~10–100 to be
        // perceptible. The strategy returns a normalised [0, ~0.5] curve;
        // multiply by 200 to land in the right ballpark (noon ≈ 90).
        amb.brightness = brightness * 200.0;
    }
}

/// Drive `ClearColor` and per-camera `DistanceFog` from the sky + sun
/// strategies. Runs after [`sync_sun`] so it sees the current
/// [`super::SunState`]; both depend on the same `WorldTime`.
///
/// Reads the progressive chunk streamer's `fog_band_m()` and threads
/// it into the [`FogStrategy`](super::FogStrategy) so the fog ramp
/// ends at the load horizon — bricks streaming into the outermost
/// tier dissolve into mist instead of popping in.
pub fn sync_sky_and_fog(
    cfg: Res<RenderConfig>,
    world_time: Res<WorldTime>,
    streamer: Res<ChunkStreamer>,
    motion: Option<Res<crate::modes::fp::CameraMotionState>>,
    mut clear: ResMut<ClearColor>,
    mut fog_q: Query<&mut DistanceFog>,
) {
    let sun_state = cfg.sun_curve.sun_state(world_time.0);
    let horizon = cfg.sky.horizon_color(sun_state);
    clear.0 = horizon;
    let (start, end) = streamer.fog_band_m();
    let band = Some((start as f32, end as f32));
    // `motion` is `Option<Res<_>>` so non-FP modes (slice / RTS /
    // overview) — which don't initialise the resource — still run the
    // fog sync without panicking. Fog strategies treat `None` as
    // "static, no speed-aware tightening".
    let motion_ref = motion.as_ref().map(|m| m.as_ref());
    for mut fog in fog_q.iter_mut() {
        let next = cfg.fog.fog_settings(sun_state, horizon, band, motion_ref);
        fog.color = next.color;
        fog.falloff = next.falloff;
    }
}
