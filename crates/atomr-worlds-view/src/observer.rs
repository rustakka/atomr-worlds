//! Observer state for the transitive skybox (Phase 13i).
//!
//! As the observer translates through the world the static skybox
//! captured by [`render_skybox_from_meshes`] drifts: distant geometry
//! continues to project from the captured pose, not from the *current*
//! pose. Beyond a tolerance the cubemap must be regenerated.
//!
//! [`ObserverState`] tracks the observer pose, derived velocity, and a
//! pair of skybox captures (`last`, `next`) plus a crossfade factor.
//! [`ObserverState::should_refresh`] returns `true` when any of:
//!
//! - position has moved > `position_delta_frac * outer_radius_m` from
//!   the last capture's `origin`;
//! - altitude (signed distance from the body's surface) has changed by
//!   > `altitude_delta_frac * body_radius_m`;
//! - the capture is older than `max_age_ticks`;
//! - the containing-frame tier has changed (sphere → free space, etc.).
//!
//! The skybox refresh itself is the caller's job — `ObserverState` only
//! advises *when*. Once a fresh skybox arrives, the caller invokes
//! [`ObserverState::accept_next`] and the crossfade animates over
//! `crossfade_duration_s` seconds.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::vehicle::ContainingFrame;

use crate::skybox::Skybox;

#[derive(Copy, Clone, Debug)]
pub struct SkyboxRefreshPolicy {
    pub position_delta_frac: f64,
    pub altitude_delta_frac: f64,
    pub max_age_ticks: u64,
    pub refresh_on_tier_change: bool,
}

impl Default for SkyboxRefreshPolicy {
    fn default() -> Self {
        Self {
            position_delta_frac: 0.05,
            altitude_delta_frac: 0.10,
            max_age_ticks: 600,
            refresh_on_tier_change: true,
        }
    }
}

#[derive(Debug)]
pub struct ObserverState {
    pub position: DVec3,
    pub velocity_mps: DVec3,
    pub containing_frame: ContainingFrame,
    pub last_skybox: Option<Skybox>,
    pub next_skybox: Option<Skybox>,
    pub crossfade_t: f32,
    pub crossfade_duration_s: f32,
    pub since_last_capture_ticks: u64,
}

impl ObserverState {
    pub fn new(position: DVec3, containing_frame: ContainingFrame) -> Self {
        Self {
            position,
            velocity_mps: DVec3::ZERO,
            containing_frame,
            last_skybox: None,
            next_skybox: None,
            crossfade_t: 0.0,
            crossfade_duration_s: 0.5,
            since_last_capture_ticks: 0,
        }
    }

    /// Advance the observer by one tick. `dt_s` is used to compute
    /// velocity and advance the crossfade. The caller supplies the new
    /// position and (optional) containing frame.
    pub fn tick(&mut self, new_position: DVec3, new_frame: Option<ContainingFrame>, dt_s: f32) {
        if dt_s > 0.0 {
            let dx = new_position.x - self.position.x;
            let dy = new_position.y - self.position.y;
            let dz = new_position.z - self.position.z;
            self.velocity_mps = DVec3::new(dx / dt_s as f64, dy / dt_s as f64, dz / dt_s as f64);
        }
        self.position = new_position;
        if let Some(f) = new_frame {
            self.containing_frame = f;
        }
        self.since_last_capture_ticks = self.since_last_capture_ticks.saturating_add(1);
        if self.next_skybox.is_some() && self.crossfade_duration_s > 0.0 {
            let inc = dt_s / self.crossfade_duration_s;
            self.crossfade_t = (self.crossfade_t + inc).clamp(0.0, 1.0);
            if self.crossfade_t >= 1.0 {
                self.last_skybox = self.next_skybox.take();
                self.crossfade_t = 0.0;
            }
        }
    }

    /// Returns `true` if any of the refresh thresholds tripped.
    pub fn should_refresh(
        &self,
        policy: &SkyboxRefreshPolicy,
        body_center: DVec3,
        body_radius_m: f64,
        previous_frame: Option<ContainingFrame>,
    ) -> bool {
        let Some(last) = self.last_skybox.as_ref() else {
            return true;
        };
        // Position delta (relative to the last capture's origin).
        let last_origin = DVec3::new(last.origin[0], last.origin[1], last.origin[2]);
        let pos_delta = (self.position - last_origin).length();
        if pos_delta > policy.position_delta_frac * last.outer_radius_m {
            return true;
        }
        // Altitude delta.
        let rel = self.position - body_center;
        let altitude_now = rel.length() - body_radius_m;
        let last_rel = last_origin - body_center;
        let altitude_then = last_rel.length() - body_radius_m;
        if (altitude_now - altitude_then).abs() > policy.altitude_delta_frac * body_radius_m {
            return true;
        }
        if self.since_last_capture_ticks > policy.max_age_ticks {
            return true;
        }
        if policy.refresh_on_tier_change {
            if let Some(prev) = previous_frame {
                if prev != self.containing_frame {
                    return true;
                }
            }
        }
        false
    }

    /// Adopt a freshly-generated skybox. If there's no last capture yet,
    /// this becomes the last directly (no crossfade). Otherwise it
    /// becomes the next and the crossfade begins at t = 0.
    pub fn accept_next(&mut self, sky: Skybox) {
        if self.last_skybox.is_none() {
            self.last_skybox = Some(sky);
            self.crossfade_t = 0.0;
        } else {
            self.next_skybox = Some(sky);
            self.crossfade_t = 0.0;
        }
        self.since_last_capture_ticks = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::skybox::{render_skybox_from_meshes, SkyboxConfig};
    use atomr_worlds_core::addr::WorldAddr;

    fn dummy_sky(origin: DVec3, outer_radius: f64) -> Skybox {
        render_skybox_from_meshes(
            &[],
            [origin.x, origin.y, origin.z],
            1.0,
            outer_radius,
            0,
            &SkyboxConfig::default(),
        )
    }

    #[test]
    fn no_capture_means_refresh() {
        let s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        let policy = SkyboxRefreshPolicy::default();
        assert!(s.should_refresh(&policy, DVec3::ZERO, 6.371e6, None));
    }

    #[test]
    fn position_delta_triggers_refresh() {
        let mut s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        s.accept_next(dummy_sky(DVec3::ZERO, 100.0));
        // Within 5 % of 100 m = 5 m: should NOT refresh.
        s.position = DVec3::new(4.0, 0.0, 0.0);
        let p = SkyboxRefreshPolicy::default();
        assert!(!s.should_refresh(&p, DVec3::ZERO, 6.371e6, None));
        // Beyond 5 %: should refresh.
        s.position = DVec3::new(6.0, 0.0, 0.0);
        assert!(s.should_refresh(&p, DVec3::ZERO, 6.371e6, None));
    }

    #[test]
    fn tier_change_triggers_refresh() {
        let mut s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        s.accept_next(dummy_sky(DVec3::ZERO, 100.0));
        let p = SkyboxRefreshPolicy::default();
        // Previous frame differs from current → tier changed.
        let prev = ContainingFrame::default();
        let _ = prev;
        // Force a tier change by switching to Free.
        s.containing_frame =
            ContainingFrame::Free(atomr_worlds_core::vehicle::ParentAddr::World(WorldAddr::ROOT));
        let prev = ContainingFrame::World(WorldAddr::ROOT);
        assert!(s.should_refresh(&p, DVec3::ZERO, 6.371e6, Some(prev)));
    }

    #[test]
    fn age_threshold_triggers_refresh() {
        let mut s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        s.accept_next(dummy_sky(DVec3::ZERO, 100.0));
        s.since_last_capture_ticks = 601;
        let p = SkyboxRefreshPolicy::default();
        assert!(s.should_refresh(&p, DVec3::ZERO, 6.371e6, None));
    }

    #[test]
    fn velocity_tracked_across_ticks() {
        let mut s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        s.tick(DVec3::new(10.0, 0.0, 0.0), None, 1.0);
        assert!((s.velocity_mps.x - 10.0).abs() < 1e-9);
        s.tick(DVec3::new(15.0, 0.0, 0.0), None, 0.5);
        assert!((s.velocity_mps.x - 10.0).abs() < 1e-9);
    }

    #[test]
    fn crossfade_advances_with_dt() {
        let mut s = ObserverState::new(DVec3::ZERO, ContainingFrame::World(WorldAddr::ROOT));
        s.accept_next(dummy_sky(DVec3::ZERO, 100.0));
        s.accept_next(dummy_sky(DVec3::new(50.0, 0.0, 0.0), 100.0));
        s.crossfade_duration_s = 1.0;
        s.tick(DVec3::ZERO, None, 0.5);
        assert!((s.crossfade_t - 0.5).abs() < 1e-6);
        s.tick(DVec3::ZERO, None, 0.5);
        // At t = 1.0 the next is promoted to last.
        assert!(s.next_skybox.is_none());
        assert!(s.last_skybox.is_some());
    }
}
