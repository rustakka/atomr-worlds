//! Strategy trait definitions.
//!
//! Each trait carves out one decision point in the render pipeline. The
//! traits are intentionally small (one or two methods) so a new
//! implementation is trivial to write. All trait objects are
//! `Send + Sync + 'static` so they can live inside `Arc<dyn Trait>`
//! fields of the [`RenderConfig`](super::RenderConfig) resource.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_view::{Framebuffer, MaterialPalette, Mesh, SliceCamera, SliceConfig, SliceTable};
use atomr_worlds_voxel::Brick;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{CascadeShadowConfig, FogSettings};
use bevy::prelude::*;
use bevy::render::camera::Exposure;

use crate::modes::fp::CameraMotionState;
use crate::world_stream::LodLadder;

// ---------------------------------------------------------------------------
// Mesh strategy
// ---------------------------------------------------------------------------

/// Turn a `Brick` into a triangle mesh.
pub trait MeshStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn mesh(&self, brick: &Brick) -> Mesh;
}

// ---------------------------------------------------------------------------
// Palette strategy
// ---------------------------------------------------------------------------

/// Source of the canonical material palette (id → PBR entry).
pub trait PaletteStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn palette(&self) -> MaterialPalette;
}

// ---------------------------------------------------------------------------
// Ambient-occlusion strategy
// ---------------------------------------------------------------------------

/// Per-vertex AO factor source. v1 default is `NoAo`; step 6 lands the
/// Minecraft-style corner sampler.
pub trait AoStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// `false` means AO is uniformly `1.0` (no shading change).
    fn enabled(&self) -> bool {
        false
    }
    /// Bake per-vertex AO into `mesh` based on `brick`. Default no-op
    /// for strategies that don't modify the mesh (e.g. `NoAo`).
    fn bake(&self, _mesh: &mut Mesh, _brick: &Brick) {}
}

// ---------------------------------------------------------------------------
// Shading strategy
// ---------------------------------------------------------------------------

/// How a brick mesh becomes Bevy renderables.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShadingMode {
    /// One child `PbrBundle<StandardMaterial>` per material id present in
    /// the brick. Step 2 default; full PBR, N draw calls per brick.
    SplitPerMaterial,
    /// One merged mesh per brick rendered through
    /// `ExtendedMaterial<StandardMaterial, VoxelMaterialExt>`. Per-vertex
    /// material id + AO carry the palette through a single draw call.
    /// Step 8.
    PaletteVoxelMaterial,
}

/// Picks how a brick mesh becomes Bevy renderables.
pub trait ShadingStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn mode(&self) -> ShadingMode {
        ShadingMode::SplitPerMaterial
    }
}

// ---------------------------------------------------------------------------
// Sun curve + state
// ---------------------------------------------------------------------------

/// Output of a [`SunCurveStrategy`] at a given hour. `direction` points
/// FROM the sun INTO the scene (i.e. light travels along `direction`), so
/// at noon it's roughly `(0, -1, 0)`.
#[derive(Clone, Copy, Debug)]
pub struct SunState {
    pub direction: Vec3,
    pub color: Color,
    pub illuminance: f32,
    /// 0 at deep night, 1 at solar noon. Lets ambient/sky strategies
    /// crossfade without re-deriving the angle.
    pub day_factor: f32,
}

impl Default for SunState {
    fn default() -> Self {
        Self {
            direction: Vec3::new(-0.4, -0.8, -0.3).normalize(),
            color: Color::rgb(1.0, 0.97, 0.9),
            illuminance: 80_000.0,
            day_factor: 1.0,
        }
    }
}

/// Maps a [`WorldTime`](crate::render::WorldTime) hour-of-day to a
/// [`SunState`] plus an ambient `(color, brightness)`. Drives the
/// directional light, ambient light, sky/fog tint, and skybox
/// brightness through [`crate::render::sync_sun`] /
/// [`crate::render::sync_sky_and_fog`] each frame.
pub trait SunCurveStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn sun_state(&self, hours: f32) -> SunState;
    /// Ambient (color, brightness) for the same hour.
    fn ambient(&self, hours: f32) -> (Color, f32);
}

// ---------------------------------------------------------------------------
// Sky strategy
// ---------------------------------------------------------------------------

/// Source of the sky's horizon + zenith color (driven by the current
/// [`SunState`]) and the optional procedural dome toggle. The horizon
/// color also feeds the sky-tinted fog so atmospheric perspective
/// stays consistent edge-to-edge.
pub trait SkyStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn horizon_color(&self, sun: SunState) -> Color;
    fn zenith_color(&self, sun: SunState) -> Color;
    /// If `true`, the sky-dome system spawns / shows a procedural
    /// SkyMaterial dome parented to the camera. Default false — the
    /// `ClearColor`-driven sky remains the basic path.
    fn dome_active(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Shadow strategy
// ---------------------------------------------------------------------------

/// Cascade configuration + per-light bias for the sun's
/// directional-light shadow map. `NoShadows` returns an empty cascade
/// config and `enabled() == false`; `BasicCascades` wires up Bevy's
/// `CascadeShadowConfigBuilder` with bounds tuned to the FP streaming
/// radius.
pub trait ShadowStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn enabled(&self) -> bool;
    fn cascade_config(&self) -> CascadeShadowConfig;
    /// Per-light biases. (depth, normal). Tunable per strategy.
    fn biases(&self) -> (f32, f32) {
        (0.02, 0.6)
    }
}

// ---------------------------------------------------------------------------
// Fog strategy
// ---------------------------------------------------------------------------

pub trait FogStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Build the per-frame [`FogSettings`].
    ///
    /// `horizon_band_m = Some((start, end))` is supplied by the
    /// progressive chunk streamer when it has a finite load horizon —
    /// `start` is the meter distance where mist should begin obscuring,
    /// `end` is where it's fully opaque (the absolute load horizon).
    /// Strategies that key off the streamer (the default
    /// [`super::defaults::ExpSquaredSkyTintedFog`]) honor it so chunks
    /// streaming into the outer tier fade in from mist instead of
    /// popping.
    ///
    /// `None` means no streamer horizon is available (legacy callers /
    /// tests / spherical-body modes still in flight). Strategies must
    /// degrade gracefully — typically by falling back to their own
    /// density / extent.
    ///
    /// `motion` is the current [`CameraMotionState`] (smoothed velocity
    /// / yaw rate / sprint hold). Strategies may use it to tighten the
    /// fog band when the camera moves fast so the visible streaming
    /// horizon shrinks gracefully under sprint. `None` for non-FP
    /// callers (slice / RTS / overview / Skybox bake passes).
    fn fog_settings(
        &self,
        sun: SunState,
        sky_horizon: Color,
        horizon_band_m: Option<(f32, f32)>,
        motion: Option<&CameraMotionState>,
    ) -> FogSettings;
}

// ---------------------------------------------------------------------------
// Tonemap strategy
// ---------------------------------------------------------------------------

/// HDR tonemap + camera exposure + optional bloom post-process. Set
/// once on the FP/TP camera at scene setup; rerun on
/// `set_render_preset` / `set_strategy` swap. `AcesTonemap` (the
/// default) returns `BloomSettings` so the HDR path has bloom enabled.
pub trait TonemapStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn tonemapping(&self) -> Tonemapping;
    fn exposure(&self) -> Exposure;
    fn bloom(&self) -> Option<BloomSettings> {
        None
    }
}

// ---------------------------------------------------------------------------
// LOD coverage policy
// ---------------------------------------------------------------------------

/// Decides whether the progressive chunk streamer keeps a coarser LOD
/// brick loaded *underneath* a finer-LOD shell that already covers it.
///
/// Two impls ship today (see [`super::defaults`]):
/// - `MaskedShells` — historical behaviour: each tier loads only its
///   shell, no overlap. Cheaper memory, but the transition from one
///   LOD to the next is a hard pop because the coarser brick has to
///   be generated + meshed the moment the finer one becomes
///   ineligible.
/// - `NestedSummary` — every tier loads its full inner sphere up to
///   its outer radius. The parent LOD is always resident as a
///   "summary" backdrop, so when a finer brick fades out the parent
///   is already in memory and just toggles visible / fades in. This
///   is the default; it eliminates the per-transition generation
///   stall and lets the visibility system crossfade between tiers.
///
/// The trait is intentionally narrow: a single predicate that the
/// inner-band test in [`crate::world_stream::desired_chunks`] consults.
/// `MaskedShells` keeps the existing mask; `NestedSummary` disables it.
pub trait LodCoveragePolicy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Whether a brick whose volume is fully inside the inner-band
    /// sphere (i.e. covered by the finer tier) should be skipped.
    ///
    /// `true`  → behave like `MaskedShells` (skip — one tier per shell).
    /// `false` → behave like `NestedSummary` (keep — parent stays loaded
    /// as a fallback summary).
    fn mask_finer_covered(&self) -> bool;

    /// Additive LOD-depth bias to apply at tier index `tier_index`
    /// during the desired-set sweep. Positive values coarsen the tier
    /// (use a deeper LOD = larger brick); 0 keeps the configured LOD.
    /// Used by the motion-aware layer to drop ladder fidelity at sprint
    /// without recomputing the radii. Default is 0 (no bias).
    fn tier_lod_bias(&self, _tier_index: usize, _motion: Option<&CameraMotionState>) -> i8 {
        0
    }
}

// ---------------------------------------------------------------------------
// Horizon-imposter shell (Phase 19.2)
// ---------------------------------------------------------------------------

/// One baked horizon-imposter shell: a polar annulus of triangles whose
/// inner edge sits just inside the streamer's outer ring and whose outer
/// edge extends to (a clamped fraction of) the geometric horizon for the
/// current observer. Coordinates are *observer-relative meters* so the
/// shell follows the camera every frame without re-baking.
#[derive(Debug, Clone)]
pub struct HorizonImposterMesh {
    pub vertices: Vec<[f32; 3]>,
    pub colors: Vec<[f32; 4]>,
    pub indices: Vec<u32>,
    pub r_inner_m: f32,
    pub r_outer_m: f32,
    /// Stable hash of the inputs (macro digest + shape + observer
    /// bucket). Used to dedupe re-bakes; identical digests must produce
    /// identical meshes for determinism.
    pub source_digest: u64,
}

/// Inputs the imposter baker reads from the world. Held by reference so
/// the baker doesn't take ownership of the macro state.
pub struct HorizonImposterInputs<'a> {
    /// Pre-baked elevation + biome + water + surface fields keyed by
    /// face. The baker samples this in the annulus directions.
    pub macro_state: &'a atomr_worlds_generate::WorldMacroState,
    pub shape: WorldShape,
    pub observer: DVec3,
    /// Horizon range to fill (`outer_radius_m - inner_radius_m` ≈ ring
    /// thickness in meters). The strategy is free to clip to its own
    /// `max_range_m()`.
    pub inner_radius_m: f64,
    pub outer_radius_m: f64,
}

/// Bakes a polar-annulus mesh covering the band from the streamer's
/// outer ring out to the geometric horizon. The mesh is observer-
/// relative and re-baked off-thread whenever the camera drifts more
/// than `rebuild_drift_m()`. Default impl: `PolarAnnulusShell`. The
/// `Legacy` preset overrides to `NoHorizonImposter` whose `enabled()`
/// is false.
pub trait HorizonImposterStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Whether to spawn / refresh a horizon shell at all. When `false`,
    /// the [`super::HorizonShellPlugin`] keeps the entity hidden and
    /// nothing else in the pipeline reads the imposter.
    fn enabled(&self) -> bool {
        true
    }
    /// Inner radius — slightly inside the streamer's outer load radius
    /// so the shell overlaps the LOD ring under fog. Default is 95% of
    /// the streamer outer.
    fn inner_radius_m(&self, streamer_outer_m: f64) -> f64 {
        streamer_outer_m * 0.95
    }
    /// Outer radius — clamped to [`Self::max_range_m`] so a sphere with
    /// a 6371 km radius doesn't try to fill a 256 km annulus.
    fn outer_radius_m(&self, shape: WorldShape, observer: DVec3) -> f64 {
        shape.horizon_at_m(observer).min(self.max_range_m())
    }
    /// Hard cap on the shell's outer radius. Default 16 km — enough for
    /// "see all the way to the horizon" on the default world without
    /// inflating the imposter mesh past `MAX_SHELL_VERTS`.
    fn max_range_m(&self) -> f64 {
        16_000.0
    }
    /// How far the observer must drift (meters) before the shell needs
    /// to be re-baked. Loose because the shell is a low-fidelity
    /// approximation — 64 m is ≈ 0.4% of a 16 km outer radius.
    fn rebuild_drift_m(&self) -> f64 {
        64.0
    }
    /// Bake the shell from the supplied macro state + observer. Returns
    /// an empty mesh if `enabled() == false`. Pure / off-thread safe —
    /// no Bevy types in or out.
    fn bake(&self, inputs: &HorizonImposterInputs<'_>) -> HorizonImposterMesh;
}

// ---------------------------------------------------------------------------
// Speed-aware strategy layer (Phase 19.2)
// ---------------------------------------------------------------------------

/// Pick the LOD ladder ([`LodLadder`]) to apply to the streamer this
/// frame. Returning `None` means "keep whatever ladder is currently
/// configured" — used by the rest-state default to avoid churning the
/// ladder on every frame. Motion-aware impls return `Some(coarser)`
/// while sustained sprint is detected and `None` once a hysteresis
/// window has elapsed since the last swap.
pub trait LodLadderPolicy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn ladder(&self, motion: &CameraMotionState) -> Option<LodLadder>;
}

/// Per-frame budget for converting completed brick payloads into Bevy
/// entities ([`crate::brick_gen::DEFAULT_SPAWN_BUDGET`] replacement).
/// Counter-intuitively the motion-scaled default *lowers* the budget at
/// sprint so the GPU-upload spike from a fresh batch is spread across
/// more frames instead of stacking into one expensive frame.
pub trait SpawnBudgetStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn budget_this_frame(&self, motion: &CameraMotionState) -> usize;
}

/// Stride at which `fp_update_lod_visibility` runs. 1 = every frame, 2
/// = every other, etc. Visibility updates are cheap-ish now (Step 4
/// made them O(n_q)) but still scale with brick count, so striding
/// them under sprint trades crispness for headroom on the
/// streaming-heavy frames.
pub trait VisibilityCadenceStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    /// Run the visibility pass on frames where `frame % stride == 0`.
    /// Return 1 to never skip.
    fn stride(&self, motion: &CameraMotionState) -> u32;
}

/// Thresholds for the plan rebuild trigger ([`crate::world_stream::DesiredChunksCache::should_rebuild`]).
/// `drift_m` is the position-drift trigger; `fwd_cos` is the
/// rotation-trigger cosine threshold (rebuild when
/// `forward · last_forward < fwd_cos`). The motion-scaled default
/// widens both when the camera is moving fast, but only when the
/// horizon imposter is active — otherwise the loose threshold leaves
/// outer-rim streaming gaps that the imposter would normally hide.
pub trait RebuildThresholdStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn drift_m(&self, motion: &CameraMotionState) -> f64;
    fn fwd_cos(&self, motion: &CameraMotionState) -> f64;
}

// ---------------------------------------------------------------------------
// Slice-view render strategy
// ---------------------------------------------------------------------------

/// Everything the slice-view raster needs for one frame. The
/// [`SliceTable`] is built by the `slice_render` system (which owns the
/// world-host handle); the strategy only decides how to turn it into
/// pixels.
#[derive(Debug)]
pub struct SliceRenderInputs<'a> {
    pub table: &'a SliceTable,
    pub cam: &'a SliceCamera,
    pub palette: &'a MaterialPalette,
    /// Base config — the strategy overrides `shading` / `light_dir_xz_y`
    /// and leaves the rest (dimensions, tile size, roof alpha) intact.
    pub base_cfg: SliceConfig,
    /// Sun direction FROM the sun INTO the scene, world space — the same
    /// value the FP view's directional light uses, so the slice's relief
    /// shading stays consistent with the 3D scene.
    pub sun_dir: Vec3,
}

/// How the Dwarf-Fortress slice view turns a [`SliceTable`] into a
/// [`Framebuffer`]. Mirrors the other render strategies: a small trait
/// with swappable impls (`FlatSlice`, `HillshadeSlice`) selected through
/// [`RenderConfig`](super::RenderConfig).
pub trait SliceRenderStrategy: Send + Sync + 'static {
    fn name(&self) -> &'static str;
    fn render(&self, inputs: &SliceRenderInputs<'_>) -> Framebuffer;
}
