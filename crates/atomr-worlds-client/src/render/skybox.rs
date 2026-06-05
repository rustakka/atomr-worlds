//! Phase 17 — cubemap skybox runtime.
//!
//! Wires the existing [`atomr_worlds_view::skybox::Skybox`] cubemap into
//! Bevy as a `core_pipeline::Skybox` component on the FP camera. The
//! skybox is re-baked when [`atomr_worlds_view::observer::ObserverState`]
//! reports the observer has drifted far enough (per
//! [`SkyboxRefreshPolicy`]). Two bakes are kept around while the
//! crossfade animates: `current_handle` is what the camera component is
//! pointing at, `next_handle` is the pending bake. When the crossfade
//! completes the next handle is swapped in.
//!
//! The bake itself pulls the far-ring bricks from
//! [`crate::world_stream::LoadedChunks`] and feeds them to
//! [`atomr_worlds_view::skybox::render_skybox_from_meshes`]. No new mesh
//! kernels — we reuse `greedy_mesh_by_material` (the same one FP uses
//! for its near-ring entities) and bake the brick origins into vertex
//! positions so the cubemap renderer sees a single observer-relative
//! mesh slab.

use std::sync::Arc;

use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_view::observer::{ObserverState, SkyboxRefreshPolicy};
use atomr_worlds_view::scene::{MaterialPalette, MeshNode};
use atomr_worlds_view::skybox::{render_skybox_from_meshes, Skybox as ViewSkybox, SkyboxConfig};
use atomr_worlds_view::{greedy_mesh_by_material, Mesh as ViewMesh, WorldQuery};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};
use bevy::core_pipeline::Skybox as BevySkybox;
use bevy::prelude::*;
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{
    Extent3d, TextureDimension, TextureFormat, TextureViewDescriptor, TextureViewDimension,
};

use crate::modes::fp::{FpState, WorldCamera};
use crate::render::{RenderConfig, WorldTime};
use crate::world_runtime::{ActiveWorld, WorldRuntime};
use crate::world_stream::{ChunkStreamer, LoadedChunks};

/// Cube faces are emitted by the view crate in this fixed order; the
/// Bevy cubemap texture-array layer count must match.
const CUBE_FACE_COUNT: u32 = 6;

/// Resolution of each cube face (pixels). 256² × 6 ≈ 1.5 MB per bake; we
/// hold at most two in flight (current + next).
pub const SKYBOX_FACE_RESOLUTION: u32 = 256;

/// World tier doesn't have a body — pick an effectively-infinite radius
/// so the altitude-delta refresh trigger never trips. The position-delta
/// trigger (against the cubemap's `outer_radius_m`) still does.
const WORLD_TIER_BODY_RADIUS_M: f64 = 1.0e6;

/// Default budget guard. At 60 fps this caps the bake cadence to 2 Hz,
/// well under what an FP walker can drift.
pub const DEFAULT_MIN_FRAMES_BETWEEN_BAKES: u64 = 30;

/// Brightness at full midnight (0% day).
pub const NIGHT_BRIGHTNESS: f32 = 50.0;
/// Brightness at solar noon (100% day).
pub const DAY_BRIGHTNESS: f32 = 2500.0;

/// Per-frame skybox state: observer pose, refresh policy, the two
/// in-flight cubemap handles, and budget bookkeeping.
#[derive(Resource)]
pub struct SkyboxRuntime {
    pub observer: ObserverState,
    pub policy: SkyboxRefreshPolicy,
    /// Cubemap currently bound to the camera's [`BevySkybox`] component.
    pub current_handle: Handle<Image>,
    /// Freshly-baked cubemap awaiting the crossfade. Once
    /// `observer.crossfade_t >= 1.0` it replaces `current_handle`.
    pub next_handle: Option<Handle<Image>>,
    /// Brightness the camera's `BevySkybox` last targeted. Lerps from
    /// `last_brightness` toward `next_brightness` during crossfade.
    pub last_brightness: f32,
    pub next_brightness: f32,
    /// Last frame `bake_skybox` ran. Combined with
    /// `min_frames_between_bakes` to throttle the bake cadence.
    pub last_refresh_frame: u64,
    pub min_frames_between_bakes: u64,
    /// Per-tick budget guard so the first frame doesn't trip a refresh
    /// before the streamer has had a chance to populate any chunks.
    pub frame_observed: u64,
    /// Last `ContainingFrame` we saw, fed back into `should_refresh` so
    /// the tier-change trigger fires only on actual transitions.
    pub last_frame: Option<ContainingFrame>,
}

impl SkyboxRuntime {
    /// Construct with a placeholder cubemap handle (the
    /// `setup_fp_scene` 1×1×6 black placeholder lives at the same
    /// handle until the first real bake replaces it).
    pub fn new(placeholder: Handle<Image>) -> Self {
        Self {
            observer: ObserverState::new(
                DVec3::ZERO,
                ContainingFrame::default(),
            ),
            policy: SkyboxRefreshPolicy::default(),
            current_handle: placeholder,
            next_handle: None,
            last_brightness: 0.0,
            next_brightness: 0.0,
            last_refresh_frame: 0,
            min_frames_between_bakes: DEFAULT_MIN_FRAMES_BETWEEN_BAKES,
            frame_observed: 0,
            last_frame: None,
        }
    }

    /// Whether the budget allows another bake this frame.
    #[inline]
    pub fn budget_allows(&self, frame: u64) -> bool {
        // First bake always allowed.
        self.last_refresh_frame == 0
            || frame.saturating_sub(self.last_refresh_frame) >= self.min_frames_between_bakes
    }

    /// Interpolated brightness at the current crossfade position.
    #[inline]
    pub fn current_brightness(&self) -> f32 {
        if self.next_handle.is_some() {
            let t = self.observer.crossfade_t.clamp(0.0, 1.0);
            self.last_brightness + (self.next_brightness - self.last_brightness) * t
        } else {
            self.last_brightness
        }
    }
}

/// Build a 1×1×6 black placeholder cubemap. Used as the initial
/// camera-side handle before the first real bake lands.
pub fn placeholder_cubemap_image() -> Image {
    // 1 texel × 4 bytes × 6 faces = 24 bytes; format is RGBA8 in sRGB.
    let bytes = vec![0u8; 4 * CUBE_FACE_COUNT as usize];
    let mut img = Image::new(
        Extent3d {
            width: 1,
            height: 1,
            depth_or_array_layers: CUBE_FACE_COUNT,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    img
}

/// Convert a baked [`ViewSkybox`] into a Bevy cubemap [`Image`]. Face
/// pixels are concatenated in the same order produced by the view crate
/// ([`atomr_worlds_view::skybox::CubeFace::ALL`]: PosX, NegX, PosY,
/// NegY, PosZ, NegZ — matches the Bevy texture-array layer convention).
pub fn cubemap_image(sky: &ViewSkybox) -> Image {
    let res = sky.face_resolution;
    let face_bytes = (res * res * 4) as usize;
    let mut bytes = Vec::with_capacity(face_bytes * 6);
    for face in &sky.faces {
        debug_assert_eq!(face.pixels.len(), face_bytes);
        bytes.extend_from_slice(&face.pixels);
    }
    let mut img = Image::new(
        Extent3d {
            width: res,
            height: res,
            depth_or_array_layers: CUBE_FACE_COUNT,
        },
        TextureDimension::D2,
        bytes,
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::RENDER_WORLD,
    );
    img.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    img
}

/// Build a [`ViewSkybox`] capture from the far-ring chunks the streamer
/// has loaded around `observer`. The bake itself is pure-CPU
/// (rasterized by the view crate's software renderer) and runs on the
/// main thread; cubemap size is small enough (256² × 6) that one bake
/// per ~30 frames is comfortable.
///
/// Far-ring chunks are identified by `lod == streamer_far_lod`. Their
/// per-brick translations are baked into the [`MeshNode::transform`] so
/// the renderer's combined-mesh path positions vertices relative to the
/// world origin (same frame the observer position lives in).
pub fn bake_skybox(
    loaded: &LoadedChunks,
    query: &dyn WorldQuery,
    addr: &atomr_worlds_core::addr::WorldAddr,
    observer: DVec3,
    far_lod: Lod,
    inner_radius_m: f64,
    outer_radius_m: f64,
    seed: u64,
    palette: Arc<MaterialPalette>,
    background_color: [u8; 4],
) -> ViewSkybox {
    // Pull every loaded far-ring chunk. The streamer keeps a hysteresis
    // window so a freshly-released chunk lingers a couple ticks — the
    // skybox is fine to capture that.
    let far_depth = far_lod.depth;
    let mut nodes: Vec<MeshNode> = Vec::new();
    let lod_scale = (1u64 << far_depth as u32) as f32;
    let edge_m = BRICK_EDGE as f32 * lod_scale;
    for ((_, depth), chunk) in loaded.iter() {
        if *depth != far_depth {
            continue;
        }
        let Some(brick) = query.brick(addr, chunk.coord, chunk.lod) else {
            continue;
        };
        nodes.extend(brick_to_mesh_nodes(
            &brick,
            chunk.coord,
            edge_m,
            lod_scale,
            palette.clone(),
            far_lod,
        ));
    }

    let cfg = SkyboxConfig {
        face_resolution: SKYBOX_FACE_RESOLUTION,
        background_color,
        include_parent_tier: true,
    };
    render_skybox_from_meshes(
        &nodes,
        [observer.x, observer.y, observer.z],
        inner_radius_m,
        outer_radius_m,
        seed,
        &cfg,
    )
}

/// Greedy-mesh a brick (split by material) and translate each submesh
/// into the world frame so the combined cubemap renderer sees a single
/// observer-relative mesh slab. Translation lives in
/// [`MeshNode::transform`] (column-major affine).
fn brick_to_mesh_nodes(
    brick: &Brick,
    brick_coord: IVec3,
    edge_m: f32,
    lod_scale: f32,
    palette: Arc<MaterialPalette>,
    lod: Lod,
) -> Vec<MeshNode> {
    let by_material = greedy_mesh_by_material(brick);
    let mut nodes = Vec::with_capacity(by_material.len());
    let tx = brick_coord.x as f32 * edge_m;
    let ty = brick_coord.y as f32 * edge_m;
    let tz = brick_coord.z as f32 * edge_m;
    // Column-major affine: scale on the diagonal, translation in column 3.
    // (Matches the convention `scene_from_bricks` uses for its node
    // transforms.) The LOD scale is needed because a far-ring brick's
    // voxels cover `BRICK_EDGE * 2^L` meters per side; the meshing
    // kernel emits unit-voxel positions which the renderer's affine
    // transform widens.
    let transform = [
        [lod_scale, 0.0, 0.0, 0.0],
        [0.0, lod_scale, 0.0, 0.0],
        [0.0, 0.0, lod_scale, 0.0],
        [tx, ty, tz, 1.0],
    ];
    // Sorted iteration for determinism (HashMap order isn't stable —
    // matters because the bake's `digest` includes the mesh order).
    let mut pairs: Vec<(u16, ViewMesh)> = by_material.into_iter().collect();
    pairs.sort_by_key(|(k, _)| *k);
    for (idx, (_mat_id, mesh)) in pairs.into_iter().enumerate() {
        if mesh.vertices.is_empty() {
            continue;
        }
        nodes.push(MeshNode {
            id: idx as u64,
            mesh: Arc::new(mesh),
            transform,
            material_palette: palette.clone(),
            lod_hint: Some(lod),
        });
    }
    nodes
}

/// Bevy system: tick observer, decide whether to refresh, advance the
/// crossfade, and push the result onto the FP camera's
/// [`BevySkybox`] component.
#[allow(clippy::too_many_arguments)]
pub fn sync_skybox(
    time: Res<Time>,
    fp_state: Res<FpState>,
    streamer: Res<ChunkStreamer>,
    loaded: Res<LoadedChunks>,
    runtime: Res<WorldRuntime>,
    active: Option<Res<ActiveWorld>>,
    render_cfg: Res<RenderConfig>,
    world_time: Res<WorldTime>,
    mut images: ResMut<Assets<Image>>,
    mut runtime_state: ResMut<SkyboxRuntime>,
    mut camera_q: Query<&mut BevySkybox, With<WorldCamera>>,
) {
    if !fp_state.ready {
        return;
    }
    let dt_s = time.delta_secs().min(0.1);
    let walk_pos = fp_state.walk.observer.position;
    let containing_frame = fp_state.walk.observer.containing_frame;
    let prev_frame = runtime_state.last_frame;

    runtime_state.observer.tick(walk_pos, Some(containing_frame), dt_s);
    runtime_state.last_frame = Some(containing_frame);

    let frame = streamer.frame;
    let max_radius_m = streamer.policy.max_radius_m;
    let inner_radius_m = streamer.policy.transition_radius_m;
    let far_lod = streamer.policy.far_lod;

    // Refresh decision. `body_center` and `body_radius_m` would matter
    // for spherical bodies; for the cube world tier they're irrelevant
    // because the altitude-delta check is dominated by the
    // position-delta check at small radii.
    let should = runtime_state.observer.should_refresh(
        &runtime_state.policy,
        DVec3::ZERO,
        WORLD_TIER_BODY_RADIUS_M,
        prev_frame,
    );
    if should && runtime_state.budget_allows(frame) {
        // Sun-curve drives the bake's background sky color so the
        // far-ring cubemap reads as "horizon at this hour".
        let sun = render_cfg.sun_curve.sun_state(world_time.0);
        let horizon = render_cfg.sky.horizon_color(sun);
        let h_lin = Vec4::from_array(horizon.to_linear().to_f32_array());
        let bg = [
            (h_lin.x.clamp(0.0, 1.0) * 255.0) as u8,
            (h_lin.y.clamp(0.0, 1.0) * 255.0) as u8,
            (h_lin.z.clamp(0.0, 1.0) * 255.0) as u8,
            255,
        ];
        let addr = active.as_deref().map(|a| a.addr).unwrap_or(fp_state.addr);
        let palette = Arc::new(render_cfg.palette.palette());
        let sky = bake_skybox(
            &loaded,
            runtime.query.as_ref(),
            &addr,
            walk_pos,
            far_lod,
            inner_radius_m,
            max_radius_m,
            active.as_deref().map(|a| a.seed).unwrap_or(0),
            palette,
            bg,
        );
        let image = cubemap_image(&sky);
        let handle = images.add(image);

        let new_brightness = lerp_brightness(sun.day_factor);
        if runtime_state.observer.last_skybox.is_none() {
            // First bake: install directly, no crossfade.
            runtime_state.observer.accept_next(sky);
            runtime_state.current_handle = handle.clone();
            runtime_state.next_handle = None;
            runtime_state.last_brightness = new_brightness;
            runtime_state.next_brightness = new_brightness;
        } else {
            runtime_state.observer.accept_next(sky);
            runtime_state.next_handle = Some(handle);
            // `last_brightness` keeps whatever the previous bake settled
            // on so the lerp ramps from there.
            runtime_state.next_brightness = new_brightness;
        }
        runtime_state.last_refresh_frame = frame;
    }

    // Crossfade swap: when the observer flushed `next_skybox` into
    // `last_skybox` (happens at `crossfade_t >= 1.0` inside `tick`), the
    // `next_handle` becomes the new current.
    if runtime_state.next_handle.is_some() && runtime_state.observer.next_skybox.is_none() {
        if let Some(h) = runtime_state.next_handle.take() {
            runtime_state.current_handle = h;
        }
        runtime_state.last_brightness = runtime_state.next_brightness;
    }

    // Push the current handle + interpolated brightness onto the camera
    // component each frame so the user sees the lerp.
    let brightness = runtime_state.current_brightness();
    let current_handle = runtime_state.current_handle.clone();
    for mut sb in camera_q.iter_mut() {
        if sb.image != current_handle {
            sb.image = current_handle.clone();
        }
        sb.brightness = brightness;
    }

    runtime_state.frame_observed = frame;
}

/// Brightness ramp from `NIGHT_BRIGHTNESS` to `DAY_BRIGHTNESS` driven
/// by the sun-curve's `day_factor` (0 at deep night, 1 at noon).
#[inline]
pub fn lerp_brightness(day_factor: f32) -> f32 {
    let t = day_factor.clamp(0.0, 1.0);
    NIGHT_BRIGHTNESS + (DAY_BRIGHTNESS - NIGHT_BRIGHTNESS) * t
}

/// Bevy plugin: inserts the [`SkyboxRuntime`] resource (seeded with a
/// 1×1×6 black placeholder) and registers [`sync_skybox`] in `Update`
/// after the FP brick streamer.
pub struct SkyboxPlugin;

impl Plugin for SkyboxPlugin {
    fn build(&self, app: &mut App) {
        // The placeholder Image is added to the asset server on plugin
        // build; the handle is stored in the runtime so `setup_fp_scene`
        // can read it back when it spawns the FP camera.
        let placeholder_handle = {
            let mut images = app
                .world_mut()
                .get_resource_mut::<Assets<Image>>()
                .expect("Assets<Image> must exist before SkyboxPlugin (DefaultPlugins not added?)");
            images.add(placeholder_cubemap_image())
        };
        app.insert_resource(SkyboxRuntime::new(placeholder_handle));
        app.add_systems(Update, sync_skybox);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::addr::WorldAddr;
    use atomr_worlds_view::skybox::{render_skybox_from_meshes, SkyboxConfig};

    fn dummy_view_sky(origin: DVec3, outer_radius_m: f64) -> ViewSkybox {
        render_skybox_from_meshes(
            &[],
            [origin.x, origin.y, origin.z],
            1.0,
            outer_radius_m,
            0,
            &SkyboxConfig {
                face_resolution: 4,
                background_color: [40, 40, 40, 255],
                include_parent_tier: false,
            },
        )
    }

    #[test]
    fn first_should_refresh_is_true_when_no_last_skybox() {
        let observer = ObserverState::new(
            DVec3::ZERO,
            ContainingFrame::World(WorldAddr::ROOT),
        );
        let policy = SkyboxRefreshPolicy::default();
        // No accept_next yet — should_refresh returns true so the first
        // bake can land.
        assert!(observer.should_refresh(
            &policy,
            DVec3::ZERO,
            WORLD_TIER_BODY_RADIUS_M,
            None,
        ));
    }

    #[test]
    fn position_drift_past_5pct_of_outer_radius_triggers_refresh() {
        let mut observer = ObserverState::new(
            DVec3::ZERO,
            ContainingFrame::World(WorldAddr::ROOT),
        );
        observer.accept_next(dummy_view_sky(DVec3::ZERO, 100.0));
        let policy = SkyboxRefreshPolicy::default();
        // 5 % of 100 m = 5 m. Within tolerance: no refresh.
        observer.position = DVec3::new(3.0, 0.0, 0.0);
        assert!(
            !observer.should_refresh(
                &policy,
                DVec3::ZERO,
                WORLD_TIER_BODY_RADIUS_M,
                None,
            ),
            "3 m drift should be inside the 5 % tolerance"
        );
        // Past 5 % of 100 m: refresh trips.
        observer.position = DVec3::new(7.0, 0.0, 0.0);
        assert!(
            observer.should_refresh(
                &policy,
                DVec3::ZERO,
                WORLD_TIER_BODY_RADIUS_M,
                None,
            ),
            "7 m drift should be past 5 % of a 100 m outer radius"
        );
    }

    #[test]
    fn brightness_lerps_monotonically_during_crossfade() {
        // Seed the runtime with two distinct (last, next) brightness
        // values and an explicit crossfade_t; current_brightness should
        // step monotonically from `last` toward `next`.
        let placeholder = Handle::<Image>::default();
        let mut rt = SkyboxRuntime::new(placeholder);
        rt.last_brightness = 100.0;
        rt.next_brightness = 1000.0;
        rt.observer.accept_next(dummy_view_sky(DVec3::ZERO, 100.0));
        rt.observer.accept_next(dummy_view_sky(DVec3::new(20.0, 0.0, 0.0), 100.0));
        rt.next_handle = Some(Handle::<Image>::default());

        let mut last = rt.current_brightness();
        for step in 1..=4 {
            rt.observer.crossfade_t = step as f32 * 0.2;
            let b = rt.current_brightness();
            assert!(
                b >= last - 1e-3,
                "brightness should be monotonically non-decreasing: prev={last}, now={b}, t={}",
                rt.observer.crossfade_t
            );
            last = b;
        }
        // At t=1.0 the value should equal next_brightness.
        rt.observer.crossfade_t = 1.0;
        let final_b = rt.current_brightness();
        assert!(
            (final_b - rt.next_brightness).abs() < 1e-3,
            "at t=1.0 brightness should equal next_brightness: {final_b}"
        );
    }

    #[test]
    fn budget_allows_first_bake_immediately() {
        let mut rt = SkyboxRuntime::new(Handle::<Image>::default());
        // last_refresh_frame == 0 means no bake has happened yet.
        assert!(rt.budget_allows(0));
        assert!(rt.budget_allows(1));
        rt.last_refresh_frame = 10;
        rt.min_frames_between_bakes = 30;
        assert!(!rt.budget_allows(15));
        assert!(!rt.budget_allows(39));
        assert!(rt.budget_allows(40));
    }

    #[test]
    fn cubemap_image_has_six_layers_and_cube_view() {
        let sky = dummy_view_sky(DVec3::ZERO, 100.0);
        let img = cubemap_image(&sky);
        let size = img.texture_descriptor.size;
        assert_eq!(size.depth_or_array_layers, 6);
        assert_eq!(size.width, sky.face_resolution);
        assert_eq!(size.height, sky.face_resolution);
        // 6 faces × res² × 4 bytes (RGBA8).
        let res = sky.face_resolution as usize;
        assert_eq!(img.data.as_ref().unwrap().len(), 6 * res * res * 4);
        let view = img.texture_view_descriptor.as_ref().expect("cube view desc");
        assert_eq!(view.dimension, Some(TextureViewDimension::Cube));
    }

    #[test]
    fn lerp_brightness_endpoints() {
        assert!((lerp_brightness(0.0) - NIGHT_BRIGHTNESS).abs() < 1e-3);
        assert!((lerp_brightness(1.0) - DAY_BRIGHTNESS).abs() < 1e-3);
        // Strictly monotone in between.
        let half = lerp_brightness(0.5);
        assert!(half > NIGHT_BRIGHTNESS);
        assert!(half < DAY_BRIGHTNESS);
    }

    #[test]
    fn placeholder_image_is_1x1x6_black() {
        let img = placeholder_cubemap_image();
        let size = img.texture_descriptor.size;
        assert_eq!(size.width, 1);
        assert_eq!(size.height, 1);
        assert_eq!(size.depth_or_array_layers, 6);
        // 6 layers × 1 texel × 4 bytes, all zero.
        assert_eq!(img.data.as_ref().unwrap().len(), 24);
        assert!(img.data.as_ref().unwrap().iter().all(|b| *b == 0));
    }
}
