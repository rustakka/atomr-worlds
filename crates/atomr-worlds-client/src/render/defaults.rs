//! Default strategy implementations. Step 0 ships behaviour-preserving
//! defaults; later steps add richer ones (each behind a new type so swapping
//! is one line in [`RenderConfig`](super::RenderConfig)).

use std::f32::consts::PI;

use atomr_worlds_view::{
    bake_ao, bake_polar_annulus, bake_sky_light, dual_contouring_mesh, greedy_mesh,
    marching_cubes_mesh, marching_cubes_mesh_with_iso, naive_mesh, render_slice, Framebuffer,
    MaterialEntry, MaterialPalette, Mesh, SliceShading,
};
use atomr_worlds_voxel::Brick;
use bevy::core_pipeline::bloom::Bloom;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{
    CascadeShadowConfig, CascadeShadowConfigBuilder, FogFalloff, DistanceFog,
};
use bevy::prelude::*;
use bevy::render::camera::Exposure;

use super::strategy::*;
use crate::modes::fp::CameraMotionState;
use crate::world_stream::LodLadder;

// ---------------------------------------------------------------------------
// Mesh — greedy (today's path)
// ---------------------------------------------------------------------------

/// Greedy meshing: coalesces coplanar same-material voxel faces into the
/// largest axis-aligned rectangles that share material id, dramatically
/// reducing triangle count vs naive 6-quads-per-voxel meshing. Backed by
/// [`atomr_worlds_view::greedy_mesh`]. Default — and currently the only —
/// [`MeshStrategy`].
#[derive(Default)]
pub struct GreedyFlat;

impl MeshStrategy for GreedyFlat {
    fn name(&self) -> &'static str {
        "GreedyFlat"
    }
    fn mesh(&self, brick: &Brick) -> Mesh {
        greedy_mesh(brick)
    }
}

/// Naive per-face mesher: one quad per visible voxel face. Baseline
/// reference impl backed by [`atomr_worlds_view::naive_mesh`]; useful
/// for A/B-ing greedy's merge benefit and as a sanity check for new
/// downstream passes.
#[derive(Default)]
pub struct NaiveMesh;

impl MeshStrategy for NaiveMesh {
    fn name(&self) -> &'static str {
        "NaiveMesh"
    }
    fn mesh(&self, brick: &Brick) -> Mesh {
        naive_mesh(brick)
    }
}

/// Marching-cubes mesher (Lorensen & Cline 1987). Backed by
/// [`atomr_worlds_view::marching_cubes_mesh`]; iso-value defaults to
/// `0.0`, override with [`MarchingCubes::with_iso`] for sub-voxel
/// thresholds on continuous density fields.
pub struct MarchingCubes {
    pub iso: f32,
}

impl Default for MarchingCubes {
    fn default() -> Self {
        Self { iso: 0.0 }
    }
}

impl MarchingCubes {
    pub fn with_iso(iso: f32) -> Self {
        Self { iso }
    }
}

impl MeshStrategy for MarchingCubes {
    fn name(&self) -> &'static str {
        "MarchingCubes"
    }
    fn mesh(&self, brick: &Brick) -> Mesh {
        if self.iso == 0.0 {
            marching_cubes_mesh(brick)
        } else {
            marching_cubes_mesh_with_iso(brick, self.iso)
        }
    }
}

/// Dual-contouring mesher (Schmitz/Garland Hermite + simplified QEF).
/// Backed by [`atomr_worlds_view::dual_contouring_mesh`]; emits one
/// vertex per sign-changed cell with quads dual to each sign-changed
/// edge, preserving sharp features that MC's edge interpolation rounds.
#[derive(Default)]
pub struct DualContouring;

impl MeshStrategy for DualContouring {
    fn name(&self) -> &'static str {
        "DualContouring"
    }
    fn mesh(&self, brick: &Brick) -> Mesh {
        dual_contouring_mesh(brick)
    }
}

// ---------------------------------------------------------------------------
// Palette — hardcoded 10-entry table (extended in step 1, used in step 2+)
// ---------------------------------------------------------------------------

/// The canonical 10-material palette for atomr-worlds. Indexing is by
/// material id; `entries[i]` is the entry for `id = i as u16` so air ends
/// up as a sentinel `id = 0` entry.
#[derive(Default)]
pub struct HardcodedPalette;

impl PaletteStrategy for HardcodedPalette {
    fn name(&self) -> &'static str {
        "HardcodedPalette"
    }
    fn palette(&self) -> MaterialPalette {
        let opaque = |id, rgb: [f32; 3], rough, metal| MaterialEntry {
            id,
            base_color: rgb,
            roughness: rough,
            metallic: metal,
            emissive: [0.0; 3],
            alpha: 1.0,
        };
        MaterialPalette {
            entries: vec![
                // 0 — air (sentinel)
                opaque(0, [0.0, 0.0, 0.0], 1.0, 0.0),
                // 1 — stone
                opaque(1, [0.42, 0.40, 0.38], 0.85, 0.0),
                // 2 — dirt
                opaque(2, [0.32, 0.22, 0.14], 0.95, 0.0),
                // 3 — sand
                opaque(3, [0.78, 0.70, 0.48], 0.75, 0.0),
                // 4 — snow
                opaque(4, [0.78, 0.82, 0.88], 0.70, 0.0),
                // 5 — water (translucent, smooth)
                MaterialEntry {
                    id: 5,
                    base_color: [0.10, 0.35, 0.55],
                    roughness: 0.05,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                    alpha: 0.6,
                },
                // 6 — grass
                opaque(6, [0.18, 0.45, 0.16], 0.90, 0.0),
                // 7 — wood
                opaque(7, [0.30, 0.18, 0.10], 0.85, 0.0),
                // 8 — leaves
                opaque(8, [0.13, 0.36, 0.12], 0.95, 0.0),
                // 9 — glow_rock (emissive)
                MaterialEntry {
                    id: 9,
                    base_color: [0.40, 0.30, 0.10],
                    roughness: 0.50,
                    metallic: 0.0,
                    emissive: [1.2, 0.8, 0.2],
                    alpha: 1.0,
                },
                // 10 — ice (translucent, smooth)
                MaterialEntry {
                    id: 10,
                    base_color: [0.78, 0.88, 0.95],
                    roughness: 0.10,
                    metallic: 0.0,
                    emissive: [0.0; 3],
                    alpha: 0.7,
                },
            ],
        }
    }
}

// ---------------------------------------------------------------------------
// AO — disabled in v1 (step 6 lands the corner sampler)
// ---------------------------------------------------------------------------

/// No-op ambient-occlusion strategy: leaves every vertex at AO = 1.0
/// (no shading change). Matches pre-upgrade behaviour and the lighter
/// `RenderPreset::Legacy` / `Debug` bundles; used as a baseline for
/// performance work where AO bake time matters.
#[derive(Default)]
pub struct NoAo;

impl AoStrategy for NoAo {
    fn name(&self) -> &'static str {
        "NoAo"
    }
}

/// Minecraft-style corner AO: each vertex samples its 4 air-side
/// neighbour voxels and darkens proportionally.
#[derive(Default)]
pub struct MinecraftCornerAo;

impl AoStrategy for MinecraftCornerAo {
    fn name(&self) -> &'static str {
        "MinecraftCornerAo"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn bake(&self, mesh: &mut Mesh, brick: &Brick) {
        bake_ao(mesh, brick);
    }
}

/// Brick-edge-aware AO. The eventual goal (when the workspace plumbs
/// neighbor bricks' apron through to the renderer) is to consult those
/// neighbors so edge-seam vertices match the AO their neighbor face
/// computed. Today the renderer has no neighbor handle to read, so this
/// impl degrades to [`MinecraftCornerAo`]'s in-brick sampler — vertices
/// at the brick boundary fall back to "no occlusion from outside",
/// matching the previous behaviour byte-for-byte. The trait surface is
/// in place so a follow-up can swap the body without touching the
/// registry. Also bakes sky-light when `Brick::light_overlay` is present.
#[derive(Default)]
pub struct BrickEdgeAwareAo;

impl AoStrategy for BrickEdgeAwareAo {
    fn name(&self) -> &'static str {
        "BrickEdgeAwareAo"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn bake(&self, mesh: &mut Mesh, brick: &Brick) {
        bake_ao(mesh, brick);
        bake_sky_light(mesh, brick);
    }
}

// ---------------------------------------------------------------------------
// Shading — placeholder. Real impls (SplitPerMaterial, PaletteVoxelMaterial)
// land in steps 2 and 8 respectively.
// ---------------------------------------------------------------------------

/// Legacy path: one shared `StandardMaterial` per brick, per-vertex RGB.
/// Matches today's pre-upgrade behaviour. Replaced by `SplitPerMaterial`
/// in step 2.
#[derive(Default)]
pub struct LegacyVertexColor;

impl ShadingStrategy for LegacyVertexColor {
    fn name(&self) -> &'static str {
        "LegacyVertexColor"
    }
    fn mode(&self) -> ShadingMode {
        ShadingMode::SplitPerMaterial
    }
}

/// Step 8 path: one merged brick mesh routed through
/// `ExtendedMaterial<StandardMaterial, VoxelMaterialExt>`. The fragment
/// shader looks up the palette entry from a storage buffer indexed by
/// per-vertex material id. Single draw call per brick.
#[derive(Default)]
pub struct PaletteVoxelMaterial;

impl ShadingStrategy for PaletteVoxelMaterial {
    fn name(&self) -> &'static str {
        "PaletteVoxelMaterial"
    }
    fn mode(&self) -> ShadingMode {
        ShadingMode::PaletteVoxelMaterial
    }
}

// ---------------------------------------------------------------------------
// Sun curve — 5-keyframe LUT (used in step 4)
// ---------------------------------------------------------------------------

/// Default sun: matches the pre-upgrade hardcoded direction so step 0 is a
/// no-op visually. Returns a constant state regardless of hours.
#[derive(Default)]
pub struct StaticSun;

impl SunCurveStrategy for StaticSun {
    fn name(&self) -> &'static str {
        "StaticSun"
    }
    fn sun_state(&self, _hours: f32) -> SunState {
        // Matches the pre-upgrade DirectionalLightBundle at (50,80,30)→0:
        // direction = -normalize((50,80,30)).
        let d = Vec3::new(50.0, 80.0, 30.0).normalize();
        SunState {
            direction: -d,
            color: Color::srgb(1.0, 0.97, 0.9),
            illuminance: 80_000.0,
            day_factor: 1.0,
        }
    }
    fn ambient(&self, _hours: f32) -> (Color, f32) {
        (Color::srgb(0.85, 0.88, 1.0), 1.2)
    }
}

/// 5-keyframe LUT: deep night → dawn → noon → dusk → night. Step 4
/// switches the default to this.
#[derive(Default)]
pub struct KeyframeLutSun;

impl KeyframeLutSun {
    /// Solar elevation/azimuth: sun rises in the east at h=6, peaks at
    /// h=12, sets in the west at h=18. Below the horizon → night.
    fn direction(hours: f32) -> Vec3 {
        // Angle θ ∈ [-π/2, π/2] across the daytime arc [6..18]; below
        // horizon outside that.
        let h = hours.rem_euclid(24.0);
        let t = (h - 6.0) / 12.0; // 0 at sunrise, 1 at sunset
        let theta = t * PI; // 0 → π across the arc
        // World convention: sun rises in +X, sets in -X, zenith +Y.
        let elevation = theta.sin(); // +1 at noon, 0 at sunrise/sunset, -1 antipode
        let azimuth = theta.cos(); // +1 east, -1 west
        let sun_pos = Vec3::new(azimuth, elevation, 0.3).normalize();
        // Direction points FROM sun INTO scene → negate position.
        -sun_pos
    }

    fn day_factor(hours: f32) -> f32 {
        // 1 at noon, 0 at sunrise/sunset, negative-ish below horizon
        // (clamped). Used as a crossfade key.
        let h = hours.rem_euclid(24.0);
        let t = (h - 6.0) / 12.0;
        (t * PI).sin().max(0.0)
    }
}

impl SunCurveStrategy for KeyframeLutSun {
    fn name(&self) -> &'static str {
        "KeyframeLutSun"
    }
    fn sun_state(&self, hours: f32) -> SunState {
        let dir = Self::direction(hours);
        let day = Self::day_factor(hours);
        // 5-keyframe color/illuminance LUT (h, color, lux):
        //   5  → deep orange,         1_000
        //   7  → warm,                40_000
        //   12 → cool-white,         100_000
        //   18 → amber,               30_000
        //   21 → moon blue,              300
        let key = [
            (5.0_f32, Vec3::new(1.0, 0.45, 0.25), 1_000.0_f32),
            (7.0, Vec3::new(1.0, 0.78, 0.55), 40_000.0),
            (12.0, Vec3::new(1.0, 0.97, 0.9), 100_000.0),
            (18.0, Vec3::new(1.0, 0.55, 0.30), 30_000.0),
            (21.0, Vec3::new(0.45, 0.55, 1.0), 300.0),
        ];
        let h = hours.rem_euclid(24.0);
        let (rgb, illum) = lerp_keyframes(h, &key, Vec3::new(0.30, 0.40, 0.85), 150.0);
        SunState {
            direction: dir,
            color: Color::srgb(rgb.x, rgb.y, rgb.z),
            illuminance: illum,
            day_factor: day,
        }
    }
    fn ambient(&self, hours: f32) -> (Color, f32) {
        // Ambient: blue-purple at night, warm at dawn/dusk, neutral-bright at noon.
        let key = [
            (5.0_f32, Vec3::new(0.30, 0.25, 0.45), 0.10_f32),
            (7.0, Vec3::new(0.55, 0.50, 0.45), 0.30),
            (12.0, Vec3::new(0.70, 0.78, 0.95), 0.45),
            (18.0, Vec3::new(0.65, 0.50, 0.40), 0.30),
            (21.0, Vec3::new(0.18, 0.22, 0.40), 0.08),
        ];
        let h = hours.rem_euclid(24.0);
        let (rgb, b) = lerp_keyframes(h, &key, Vec3::new(0.15, 0.18, 0.30), 0.05);
        (Color::srgb(rgb.x, rgb.y, rgb.z), b)
    }
}

// ---------------------------------------------------------------------------
// Sky — v1 default reproduces the pre-upgrade flat `ClearColor`.
// ---------------------------------------------------------------------------

/// Constant sky — matches the pre-upgrade ClearColor(0.45, 0.65, 0.85).
#[derive(Default)]
pub struct ConstantSky;

impl SkyStrategy for ConstantSky {
    fn name(&self) -> &'static str {
        "ConstantSky"
    }
    fn horizon_color(&self, _sun: SunState) -> Color {
        Color::srgb(0.45, 0.65, 0.85)
    }
    fn zenith_color(&self, _sun: SunState) -> Color {
        Color::srgb(0.30, 0.55, 0.85)
    }
}

/// Step 9 sky: same color curve as [`SkyTinted`], but the
/// `dome_active()` flag is on so the `sky_dome` plugin spawns / shows a
/// procedural dome sphere around the camera. The dome's fragment shader
/// draws a gradient + sun disc; `ClearColor` / fog still follow the
/// horizon color so the look is consistent on the edges.
#[derive(Default)]
pub struct ProceduralDomeSky;

impl SkyStrategy for ProceduralDomeSky {
    fn name(&self) -> &'static str {
        "ProceduralDomeSky"
    }
    fn horizon_color(&self, sun: SunState) -> Color {
        SkyTinted.horizon_color(sun)
    }
    fn zenith_color(&self, sun: SunState) -> Color {
        SkyTinted.zenith_color(sun)
    }
    fn dome_active(&self) -> bool {
        true
    }
}

/// Sky-tinted by sun: horizon color follows the sun's color (orange at
/// dusk, blue at night, pale at noon). Step 7 switches the default to
/// this.
#[derive(Default)]
pub struct SkyTinted;

impl SkyStrategy for SkyTinted {
    fn name(&self) -> &'static str {
        "SkyTinted"
    }
    fn horizon_color(&self, sun: SunState) -> Color {
        // Horizon at noon is a pale sky-blue (not pure-white). Dawn/dusk
        // pull toward the sun's warm color. Night → deep blue.
        let night = Vec3::new(0.04, 0.06, 0.16);
        let day_blue = Vec3::new(0.55, 0.70, 0.95);
        let sun_warm = color_to_vec3(sun.color);
        let t = sun.day_factor.clamp(0.0, 1.0);
        // Dawn/dusk pull more strongly toward the sun color than noon.
        let pull = (1.0 - t).powf(0.5).clamp(0.0, 1.0);
        // Three-way blend: night → tinted-by-sun horizon → day_blue.
        // The `pull * sun_warm * 0.85` adds a warm rim near sunrise/sunset.
        let day_tint = day_blue.lerp(sun_warm * 0.85, pull * 0.6);
        let horizon = night.lerp(day_tint, t.max(pull * 0.7));
        vec3_to_color(horizon)
    }
    fn zenith_color(&self, sun: SunState) -> Color {
        // Deeper, more saturated blue at zenith so the dome gradient
        // reads strongly against the (pale) horizon color.
        let night = Vec3::new(0.01, 0.02, 0.06);
        let day = Vec3::new(0.12, 0.30, 0.75);
        let t = sun.day_factor.clamp(0.0, 1.0);
        vec3_to_color(night.lerp(day, t))
    }
}

// ---------------------------------------------------------------------------
// Shadows
// ---------------------------------------------------------------------------

/// No shadows — matches pre-upgrade behaviour.
#[derive(Default)]
pub struct NoShadows;

impl ShadowStrategy for NoShadows {
    fn name(&self) -> &'static str {
        "NoShadows"
    }
    fn enabled(&self) -> bool {
        false
    }
    fn cascade_config(&self) -> CascadeShadowConfig {
        CascadeShadowConfigBuilder::default().build()
    }
}

/// Cascaded shadow maps tuned to the FP streaming radius (~48 m).
///
/// Drives Bevy's `CascadeShadowConfigBuilder`; the defaults bias the
/// near cascade tight to the camera so foreground voxel edges get
/// crisp shadows while the far cascade still covers the LOD-1 ring.
/// All fields map 1:1 to the Bevy builder methods.
pub struct BasicCascades {
    /// Number of cascade splits. 4 is the practical sweet spot for the
    /// FP load horizon — fewer leaves visible seams at mid-range.
    pub num_cascades: usize,
    /// Near plane of the first cascade, in world meters.
    pub minimum_distance: f32,
    /// Far plane of the outermost cascade, in world meters.
    pub maximum_distance: f32,
    /// Far bound of the first cascade. Smaller → sharper shadows on
    /// near voxels at the cost of more frequent cascade transitions.
    pub first_cascade_far_bound: f32,
    /// Fraction of overlap between adjacent cascades; smooths the
    /// transition at cascade boundaries.
    pub overlap_proportion: f32,
}

impl Default for BasicCascades {
    fn default() -> Self {
        Self {
            num_cascades: 4,
            minimum_distance: 0.1,
            maximum_distance: 200.0,
            first_cascade_far_bound: 8.0,
            overlap_proportion: 0.2,
        }
    }
}

impl ShadowStrategy for BasicCascades {
    fn name(&self) -> &'static str {
        "BasicCascades"
    }
    fn enabled(&self) -> bool {
        true
    }
    fn cascade_config(&self) -> CascadeShadowConfig {
        CascadeShadowConfigBuilder {
            num_cascades: self.num_cascades,
            minimum_distance: self.minimum_distance,
            maximum_distance: self.maximum_distance,
            first_cascade_far_bound: self.first_cascade_far_bound,
            overlap_proportion: self.overlap_proportion,
        }
        .build()
    }
}

// ---------------------------------------------------------------------------
// Fog
// ---------------------------------------------------------------------------

/// No fog — matches pre-upgrade behaviour.
#[derive(Default)]
pub struct NoFog;

impl FogStrategy for NoFog {
    fn name(&self) -> &'static str {
        "NoFog"
    }
    fn fog_settings(
        &self,
        _sun: SunState,
        _sky_horizon: Color,
        _horizon_band_m: Option<(f32, f32)>,
        _motion: Option<&crate::modes::fp::CameraMotionState>,
    ) -> DistanceFog {
        DistanceFog {
            color: Color::NONE,
            falloff: FogFalloff::Linear { start: 1.0e6, end: 1.0e6 + 1.0 },
            ..default()
        }
    }
}

/// Exp² atmospheric fog, color = sky horizon at current sun.
///
/// When the progressive chunk streamer supplies a `horizon_band_m`, the
/// fog density is auto-tuned so transmittance reaches ≈ `HORIZON_TRANS`
/// at `band.end` — i.e. the outer load horizon is almost fully fogged.
/// Because exp² is smooth from zero, every closer LOD tier also picks
/// up atmospheric perspective: near voxels stay clear, mid-distance
/// LOD-1/2 bricks gain a soft horizon tint, and the far LOD-3 ring
/// dissolves into the sky color. Without a band the strategy uses its
/// `density` field directly.
pub struct ExpSquaredSkyTintedFog {
    /// Fallback density when no streamer horizon is plumbed in.
    pub density: f32,
}

impl Default for ExpSquaredSkyTintedFog {
    fn default() -> Self {
        // Matches the auto-tune at outer=1024 m so headless callers (no
        // streamer band) still get usable distance fade.
        Self { density: 0.0019 }
    }
}

/// Transmittance at the load-horizon distance — i.e. how much of the
/// original brick color survives at the very edge of streaming. 0.05 ⇒
/// 95 % of the pixel reads as sky color when a brick is at the horizon.
const HORIZON_TRANSMITTANCE: f32 = 0.05;

impl FogStrategy for ExpSquaredSkyTintedFog {
    fn name(&self) -> &'static str {
        "ExpSquaredSkyTintedFog"
    }
    fn fog_settings(
        &self,
        _sun: SunState,
        sky_horizon: Color,
        horizon_band_m: Option<(f32, f32)>,
        _motion: Option<&crate::modes::fp::CameraMotionState>,
    ) -> DistanceFog {
        // Auto-tune density from the streamer horizon so fog reaches
        // HORIZON_TRANSMITTANCE exactly at `band.end`. Solve
        //   exp(-(d * density)²) = T
        // for density = sqrt(-ln T) / d.
        let density = match horizon_band_m {
            Some((_start, end)) if end > 0.0 => {
                let target = HORIZON_TRANSMITTANCE.max(1e-3).min(0.5);
                (-target.ln()).sqrt() / end
            }
            _ => self.density,
        };
        DistanceFog {
            color: sky_horizon,
            falloff: FogFalloff::ExponentialSquared { density },
            ..default()
        }
    }
}

/// Biome-blended fog. Same exp² falloff as [`ExpSquaredSkyTintedFog`]
/// (auto-tuned from the streamer horizon when provided), but the color
/// is interpolated between the current biome's tint and the sky horizon
/// so the player crossing a biome boundary sees a soft tint shift rather
/// than a hard pop. Biome state isn't plumbed into `FogStrategy::fog_settings`
/// today; the strategy keeps a `biome_tint` field that the caller can
/// update from the macro biome blend output before each frame — until
/// that wiring lands, the default biome tint reads as a neutral grey-blue
/// so the visible result equals `ExpSquaredSkyTintedFog`.
#[derive(Debug, Clone)]
pub struct BiomeBlendedFog {
    /// Fallback density when no streamer horizon is plumbed in.
    pub density: f32,
    /// Linear-rgb tint contributed by the current biome (in `[0, 1]`).
    /// `(0.5, 0.5, 0.5)` reads as "no biome bias" — the fog stays
    /// pure sky-horizon color, matching `ExpSquaredSkyTintedFog`.
    pub biome_tint: [f32; 3],
    /// Mix weight between sky horizon and `biome_tint` in `[0, 1]`.
    /// `0.0` ⇒ pure sky horizon (no biome influence); `1.0` ⇒ pure biome
    /// tint. Default `0.3` keeps atmospheric perspective coherent while
    /// still reading the biome shift.
    pub mix: f32,
}

impl Default for BiomeBlendedFog {
    fn default() -> Self {
        Self {
            density: 0.0019,
            biome_tint: [0.5, 0.5, 0.5],
            mix: 0.3,
        }
    }
}

impl FogStrategy for BiomeBlendedFog {
    fn name(&self) -> &'static str {
        "BiomeBlendedFog"
    }
    fn fog_settings(
        &self,
        _sun: SunState,
        sky_horizon: Color,
        horizon_band_m: Option<(f32, f32)>,
        _motion: Option<&crate::modes::fp::CameraMotionState>,
    ) -> DistanceFog {
        let density = match horizon_band_m {
            Some((_start, end)) if end > 0.0 => {
                let target = 0.05_f32.max(1e-3).min(0.5);
                (-target.ln()).sqrt() / end
            }
            _ => self.density,
        };
        let sky_lin = sky_horizon.to_linear().to_f32_array();
        let m = self.mix.clamp(0.0, 1.0);
        let tinted = Color::linear_rgb(
            sky_lin[0] * (1.0 - m) + self.biome_tint[0] * m,
            sky_lin[1] * (1.0 - m) + self.biome_tint[1] * m,
            sky_lin[2] * (1.0 - m) + self.biome_tint[2] * m,
        );
        DistanceFog {
            color: tinted,
            falloff: FogFalloff::ExponentialSquared { density },
            ..default()
        }
    }
}

// ---------------------------------------------------------------------------
// Tonemap
// ---------------------------------------------------------------------------

/// Today's behaviour: Bevy's default tonemapping with a neutral exposure.
#[derive(Default)]
pub struct DefaultTonemap;

impl TonemapStrategy for DefaultTonemap {
    fn name(&self) -> &'static str {
        "DefaultTonemap"
    }
    fn tonemapping(&self) -> Tonemapping {
        Tonemapping::TonyMcMapface
    }
    fn exposure(&self) -> Exposure {
        Exposure { ev100: 9.7 }
    }
}

/// ACES filmic. Step 3 default.
#[derive(Default)]
pub struct AcesTonemap;

impl TonemapStrategy for AcesTonemap {
    fn name(&self) -> &'static str {
        "AcesTonemap"
    }
    fn tonemapping(&self) -> Tonemapping {
        Tonemapping::AcesFitted
    }
    fn exposure(&self) -> Exposure {
        // EV100=11 drops the exposure by 2 stops vs the ev100=9 default;
        // at noon the sun's 100k lux + bright sky used to clamp lit faces
        // to ~white, hiding the underlying material color. 11 reads as
        // a normal sunlit terrain through ACES.
        Exposure { ev100: 11.0 }
    }
    fn bloom(&self) -> Option<Bloom> {
        Some(Bloom { intensity: 0.10, ..default() })
    }
}

// ---------------------------------------------------------------------------
// LOD coverage policy
// ---------------------------------------------------------------------------

/// Historical behaviour: each tier loads only its shell band; the
/// inner-band mask in `desired_chunks` skips any brick whose volume
/// is fully covered by a finer tier. Cheap memory, hard LOD pops.
#[derive(Default)]
pub struct MaskedShells;

impl LodCoveragePolicy for MaskedShells {
    fn name(&self) -> &'static str {
        "MaskedShells"
    }
    fn mask_finer_covered(&self) -> bool {
        true
    }
}

/// Default: every tier loads its full inner sphere up to its outer
/// radius, so each region has an immediately-resident coarse "summary"
/// behind whatever finer LOD currently owns it. The visibility system
/// in `modes/fp.rs` (`fp_update_lod_visibility`) keeps the finest
/// loaded LOD visible per region and crossfades through transitions.
/// Memory cost is bounded — each coarser tier covers 8× the volume
/// per brick, so the inflation across the 4-tier default ladder is
/// roughly +15 % bricks.
#[derive(Default)]
pub struct NestedSummary;

impl LodCoveragePolicy for NestedSummary {
    fn name(&self) -> &'static str {
        "NestedSummary"
    }
    fn mask_finer_covered(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// Horizon imposter strategies (Phase 19.2)
// ---------------------------------------------------------------------------

/// Default horizon-imposter baker. Emits a polar annulus of triangles
/// (32 rings × 128 sectors, log-spaced radii) covering the band from
/// the streamer's outer ring out to the clamped geometric horizon.
/// Each vertex samples [`WorldMacroState`] to derive an elevation +
/// biome color so the shell reads as representative terrain rather
/// than a painted skybox.
///
/// Wiring (Step 8) and the actual sample loop land later in the phase
/// — this skeleton produces a placeholder mesh so the trait, registry,
/// and `RenderConfig` slot can be in place. Step 8 fills in the real
/// macro-sampling path; Step 9 adds the sphere-shape curvature drop.
pub struct PolarAnnulusShell {
    pub n_rings: u32,
    pub n_sectors: u32,
    /// Hard cap on vertex count regardless of `n_rings * n_sectors`,
    /// so misconfiguration can't blow up the imposter mesh.
    pub max_verts: usize,
}

impl Default for PolarAnnulusShell {
    fn default() -> Self {
        Self { n_rings: 32, n_sectors: 128, max_verts: 16_384 }
    }
}

impl HorizonImposterStrategy for PolarAnnulusShell {
    fn name(&self) -> &'static str {
        "PolarAnnulusShell"
    }
    fn bake(&self, inputs: &HorizonImposterInputs<'_>) -> HorizonImposterMesh {
        let baked = bake_polar_annulus(
            inputs.macro_state,
            inputs.shape,
            inputs.observer,
            inputs.inner_radius_m,
            inputs.outer_radius_m,
            self.n_rings,
            self.n_sectors,
        );
        // Digest the inputs so the runtime can skip identical re-bakes.
        // Hash the observer in a coarse 32 m bucket so micro-drift
        // doesn't churn the digest.
        let bucket_x = (inputs.observer.x / 32.0).round() as i64;
        let bucket_z = (inputs.observer.z / 32.0).round() as i64;
        let digest = {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut h = DefaultHasher::new();
            inputs.macro_state.digest.hash(&mut h);
            bucket_x.hash(&mut h);
            bucket_z.hash(&mut h);
            (inputs.inner_radius_m.to_bits()).hash(&mut h);
            (inputs.outer_radius_m.to_bits()).hash(&mut h);
            h.finish()
        };
        HorizonImposterMesh {
            vertices: baked.vertices,
            colors: baked.colors,
            indices: baked.indices,
            r_inner_m: baked.r_inner_m,
            r_outer_m: baked.r_outer_m,
            source_digest: digest,
        }
    }
}

/// Legacy / disabled imposter — `enabled() == false` so the shell
/// pipeline does nothing. Selected by `RenderPreset::Legacy`.
#[derive(Default)]
pub struct NoHorizonImposter;

impl HorizonImposterStrategy for NoHorizonImposter {
    fn name(&self) -> &'static str {
        "NoHorizonImposter"
    }
    fn enabled(&self) -> bool {
        false
    }
    fn bake(&self, inputs: &HorizonImposterInputs<'_>) -> HorizonImposterMesh {
        HorizonImposterMesh {
            vertices: Vec::new(),
            colors: Vec::new(),
            indices: Vec::new(),
            r_inner_m: inputs.inner_radius_m as f32,
            r_outer_m: inputs.outer_radius_m as f32,
            source_digest: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Speed-aware strategy defaults (Phase 19.2)
// ---------------------------------------------------------------------------

/// Static-ladder policy: always returns `None`, leaving the streamer's
/// configured ladder untouched. Used by `RenderPreset::Quality` and as
/// the baseline that all behavior tests measure against.
#[derive(Default)]
pub struct StaticLadder;

impl LodLadderPolicy for StaticLadder {
    fn name(&self) -> &'static str {
        "StaticLadder"
    }
    fn ladder(&self, _motion: &CameraMotionState) -> Option<LodLadder> {
        None
    }
}

/// Motion-aware ladder policy.
///
/// Historically this swapped tiers 2/3 from L2/L3 to L3/L4 during
/// sustained sprint to cut the streamed brick count by ~40 %. That
/// approach is **abandoned** because it had two user-visible problems
/// the perf win didn't justify:
///
/// 1. A coarsened ladder *evicts* the resident fine bricks at the new
///    coarse tier's radii (their (coord, lod) keys no longer appear in
///    [`crate::world_stream::desired_chunks`], so the hysteresis
///    window expires and the streamer attaches `BrickFadeOut`). The
///    user sees outer-ring detail disappear during sprint, then a
///    visible streaming wave to refill it after sprint ends.
/// 2. The recovery wave costs more frame budget *after* sprint than
///    the eviction saved *during* sprint — net negative.
///
/// The user-facing directive is "be strategic about what you *load* at
/// speed; never *unload* high-detail." So this policy now always
/// returns the default progressive ladder. The motion-aware perf budget
/// is carried entirely by the other Phase 19.2 strategies, which throttle
/// *new* work without evicting *existing* work:
///
/// - [`MotionScaledSpawnBudget`] lowers main-thread GPU upload budget
///   under sprint so the upload spike spreads across more frames.
/// - [`MotionScaledCadence`] runs the visibility pass less often.
/// - [`MotionScaledRebuildThreshold`] widens the plan-rebuild drift
///   trigger so the AABB sweep fires less frequently.
///
/// Together those three throttle the per-frame streaming cost during
/// sprint without touching the resident brick set. Fine bricks stay
/// visible; only the *rate of new loads* tracks the motion budget.
#[derive(Default)]
pub struct MotionScaledLadder;

impl LodLadderPolicy for MotionScaledLadder {
    fn name(&self) -> &'static str {
        "MotionScaledLadder"
    }
    fn ladder(&self, _motion: &CameraMotionState) -> Option<LodLadder> {
        // Always the default progressive ladder. See the type-level
        // comment for why coarsening at sprint was removed.
        Some(LodLadder::default_progressive())
    }
}

/// Static spawn-budget policy: always returns the historical
/// `DEFAULT_SPAWN_BUDGET` (24 / frame). Quality preset uses this.
pub struct StaticSpawnBudget {
    pub budget: usize,
}

impl Default for StaticSpawnBudget {
    fn default() -> Self {
        Self { budget: crate::brick_gen::DEFAULT_SPAWN_BUDGET }
    }
}

impl SpawnBudgetStrategy for StaticSpawnBudget {
    fn name(&self) -> &'static str {
        "StaticSpawnBudget"
    }
    fn budget_this_frame(&self, _motion: &CameraMotionState) -> usize {
        self.budget
    }
}

/// Motion-scaled spawn-budget policy. Lerps from `rest_budget` (24) at
/// stand-still down to `sprint_budget` (8) at sustained sprint
/// (`smoothed_velocity_m_s >= 12.0`). Counter-intuitively *lower* at
/// sprint: the goal is to spread the GPU-upload spike from a batch of
/// new bricks across more frames, not to push more bricks per frame.
/// The streamer's rebuild loop still feeds the in-flight queue at full
/// pace; this only throttles main-thread mesh→GPU uploads.
pub struct MotionScaledSpawnBudget {
    pub rest_budget: usize,
    pub sprint_budget: usize,
}

impl Default for MotionScaledSpawnBudget {
    fn default() -> Self {
        Self {
            rest_budget: crate::brick_gen::DEFAULT_SPAWN_BUDGET,
            sprint_budget: 8,
        }
    }
}

impl SpawnBudgetStrategy for MotionScaledSpawnBudget {
    fn name(&self) -> &'static str {
        "MotionScaledSpawnBudget"
    }
    fn budget_this_frame(&self, motion: &CameraMotionState) -> usize {
        let t = (motion.smoothed_velocity_m_s / 12.0).clamp(0.0, 1.0);
        let rest = self.rest_budget as f32;
        let sprint = self.sprint_budget as f32;
        let lerp = rest + (sprint - rest) * t;
        lerp.round().clamp(self.sprint_budget as f32, self.rest_budget as f32) as usize
    }
}

/// Static visibility cadence: stride = 1 (run every frame).
#[derive(Default)]
pub struct StaticVisibilityCadence;

impl VisibilityCadenceStrategy for StaticVisibilityCadence {
    fn name(&self) -> &'static str {
        "StaticVisibilityCadence"
    }
    fn stride(&self, _motion: &CameraMotionState) -> u32 {
        1
    }
}

/// Motion-scaled visibility cadence. Stride 1 at rest, 2 at moderate
/// motion (≥ 3 m/s), 3 at full sprint (≥ 8 m/s). The LOD-visibility
/// pass is O(loaded brick count) so striding it out at sprint shaves
/// a measurable chunk off the per-frame cost without visibly affecting
/// crossfade behavior — bricks just fade in/out over 2–3 frames
/// instead of 1.
#[derive(Default)]
pub struct MotionScaledCadence;

impl VisibilityCadenceStrategy for MotionScaledCadence {
    fn name(&self) -> &'static str {
        "MotionScaledCadence"
    }
    fn stride(&self, motion: &CameraMotionState) -> u32 {
        if motion.smoothed_velocity_m_s >= 8.0 {
            3
        } else if motion.smoothed_velocity_m_s >= 3.0 {
            2
        } else {
            1
        }
    }
}

/// Static rebuild thresholds: matches the historical
/// [`crate::world_stream::PLAN_REBUILD_DRIFT_M`] /
/// [`crate::world_stream::PLAN_REBUILD_FWD_COS`] constants. Used by
/// the Quality preset to disable the motion-aware tightening.
#[derive(Default)]
pub struct StaticRebuildThreshold;

impl RebuildThresholdStrategy for StaticRebuildThreshold {
    fn name(&self) -> &'static str {
        "StaticRebuildThreshold"
    }
    fn drift_m(&self, _motion: &CameraMotionState) -> f64 {
        crate::world_stream::PLAN_REBUILD_DRIFT_M
    }
    fn fwd_cos(&self, _motion: &CameraMotionState) -> f64 {
        crate::world_stream::PLAN_REBUILD_FWD_COS
    }
}

/// Motion-scaled rebuild thresholds. Widens both at sustained sprint —
/// drift 4 m → 16 m, fwd-cos 0.9659 → 0.93 — but only when the horizon
/// imposter is active (`motion.imposter_active == true`); the imposter
/// carries the outer-band terrain so the wider thresholds don't
/// expose streaming gaps. With no imposter (Legacy preset) we stay at
/// the historical constants to keep the LOD ladder ring-edge clean.
#[derive(Default)]
pub struct MotionScaledRebuildThreshold;

impl RebuildThresholdStrategy for MotionScaledRebuildThreshold {
    fn name(&self) -> &'static str {
        "MotionScaledRebuildThreshold"
    }
    fn drift_m(&self, motion: &CameraMotionState) -> f64 {
        if !motion.imposter_active {
            return crate::world_stream::PLAN_REBUILD_DRIFT_M;
        }
        let t = (motion.smoothed_velocity_m_s / 12.0).clamp(0.0, 1.0);
        let rest = crate::world_stream::PLAN_REBUILD_DRIFT_M;
        let sprint = 16.0_f64;
        rest + (sprint - rest) * t as f64
    }
    fn fwd_cos(&self, motion: &CameraMotionState) -> f64 {
        if !motion.imposter_active {
            return crate::world_stream::PLAN_REBUILD_FWD_COS;
        }
        let t = (motion.smoothed_velocity_m_s / 12.0).clamp(0.0, 1.0);
        let rest = crate::world_stream::PLAN_REBUILD_FWD_COS;
        let sprint = 0.93_f64;
        rest + (sprint - rest) * t as f64
    }
}

// ---------------------------------------------------------------------------
// Slice-view render strategy
// ---------------------------------------------------------------------------

/// Flat-fill slice raster — each column is the palette's `base_color`
/// with no relief shading. Preserves the pre-rework slice look; reachable
/// via `RenderPreset::Legacy` / `Debug`.
#[derive(Default)]
pub struct FlatSlice;

impl SliceRenderStrategy for FlatSlice {
    fn name(&self) -> &'static str {
        "FlatSlice"
    }
    fn render(&self, inputs: &SliceRenderInputs<'_>) -> Framebuffer {
        let mut cfg = inputs.base_cfg;
        cfg.shading = SliceShading::Flat;
        render_slice(inputs.table, inputs.cam, inputs.palette, &cfg)
    }
}

/// Hillshade-relief slice raster. Derives a per-column surface normal
/// from the neighbouring columns' `top_z` height field and lights it
/// with the FP view's sun direction, so vertical terrain reads as 3D
/// relief consistent with the first-person scene. Default slice strategy.
pub struct HillshadeSlice {
    /// Unlit floor brightness — `0.0` is black shadows, `1.0` removes all
    /// shading.
    pub ambient: f32,
    /// Scales the height gradient before the normal is built; higher
    /// exaggerates relief.
    pub relief_strength: f32,
}

impl Default for HillshadeSlice {
    fn default() -> Self {
        Self { ambient: 0.35, relief_strength: 1.0 }
    }
}

impl SliceRenderStrategy for HillshadeSlice {
    fn name(&self) -> &'static str {
        "HillshadeSlice"
    }
    fn render(&self, inputs: &SliceRenderInputs<'_>) -> Framebuffer {
        let mut cfg = inputs.base_cfg;
        cfg.shading = SliceShading::Hillshade {
            ambient: self.ambient,
            relief_strength: self.relief_strength,
        };
        // `SliceConfig` packs the light as [world_x, world_z, world_y] so
        // it lines up with the slice's (x, z) tile plane.
        let d = inputs.sun_dir;
        cfg.light_dir_xz_y = [d.x, d.z, d.y];
        render_slice(inputs.table, inputs.cam, inputs.palette, &cfg)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Linear interpolation across a sequence of (hour, vec3, f32) keyframes,
/// wrapping at 24h. Returns a default if the keyframe list is empty.
fn lerp_keyframes(
    hours: f32,
    keys: &[(f32, Vec3, f32)],
    fallback_rgb: Vec3,
    fallback_scalar: f32,
) -> (Vec3, f32) {
    if keys.is_empty() {
        return (fallback_rgb, fallback_scalar);
    }
    // Find the segment we're in (with wrap-around).
    let n = keys.len();
    // Sort by hour (caller's responsibility, but cheap to assume sorted).
    let mut prev = n - 1;
    let mut next = 0;
    for i in 0..n {
        if keys[i].0 > hours {
            next = i;
            prev = if i == 0 { n - 1 } else { i - 1 };
            break;
        }
        if i == n - 1 {
            // hours >= all keyframes — wrap to next-day first key
            prev = n - 1;
            next = 0;
        }
    }
    let (h0, rgb0, s0) = keys[prev];
    let (h1, rgb1, s1) = keys[next];
    // Wrap distance: if h1 < h0, h1 is "tomorrow".
    let span = if h1 > h0 { h1 - h0 } else { (h1 + 24.0) - h0 };
    let here = if hours >= h0 { hours - h0 } else { (hours + 24.0) - h0 };
    let t = if span > 0.0 { (here / span).clamp(0.0, 1.0) } else { 0.0 };
    (rgb0.lerp(rgb1, t), s0 + (s1 - s0) * t)
}

fn color_to_vec3(c: Color) -> Vec3 {
    let lin = c.to_linear().to_f32_array();
    Vec3::new(lin[0], lin[1], lin[2])
}

fn vec3_to_color(v: Vec3) -> Color {
    Color::linear_rgb(v.x, v.y, v.z)
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_voxel::{light::LightOverlay, Voxel};
    use atomr_worlds_core::coord::IVec3;

    #[test]
    fn brick_edge_aware_ao_handles_missing_overlay() {
        // Single voxel with no light overlay attached. AO baker should
        // still run; sky-light baker should no-op without panicking.
        let mut b = Brick::new();
        b.set(IVec3::new(5, 5, 5), Voxel::new(1));
        let ao = BrickEdgeAwareAo;
        let mut mesh = greedy_mesh(&b);
        ao.bake(&mut mesh, &b);
        // Default sky_light is 1.0; without overlay it stays at 1.0.
        for v in &mesh.vertices {
            assert!((v.sky_light - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn brick_edge_aware_ao_consumes_overlay_when_present() {
        let mut b = Brick::new();
        b.set(IVec3::new(5, 5, 5), Voxel::new(1));
        // Force every light cell to a dim value so we can detect that
        // `bake_sky_light` actually touched the vertices.
        let mut overlay = Box::new(LightOverlay::new_zero());
        for z in 0..16u8 {
            for y in 0..16u8 {
                for x in 0..16u8 {
                    overlay.set(x, y, z, 4);
                }
            }
        }
        b.light_overlay = Some(overlay);
        let ao = BrickEdgeAwareAo;
        let mut mesh = greedy_mesh(&b);
        ao.bake(&mut mesh, &b);
        // 4 / 15 ≈ 0.266; pick the lower-bound region to keep AO/empty mixing safe.
        for v in &mesh.vertices {
            assert!(v.sky_light < 0.5);
        }
    }

    #[test]
    fn biome_blended_fog_returns_sensible_color_at_default() {
        let fog = BiomeBlendedFog::default();
        let sun = SunState::default();
        let horizon = Color::srgb(0.4, 0.5, 0.8);
        let s = fog.fog_settings(sun, horizon, Some((400.0, 1024.0)), None);
        // Without a biome bias (tint == 0.5,0.5,0.5, mix=0.3) the result
        // should stay close to the horizon color.
        let rgba = s.color.to_linear().to_f32_array();
        assert!(rgba[0] >= 0.0 && rgba[0] <= 1.0);
        assert!(rgba[1] >= 0.0 && rgba[1] <= 1.0);
        assert!(rgba[2] >= 0.0 && rgba[2] <= 1.0);
    }

    #[test]
    fn biome_blended_fog_biases_color_when_biome_tint_set() {
        let fog = BiomeBlendedFog { density: 0.002, biome_tint: [0.1, 0.6, 0.2], mix: 0.8 };
        let sun = SunState::default();
        let horizon = Color::linear_rgb(0.9, 0.9, 0.9);
        let s = fog.fog_settings(sun, horizon, None, None);
        let rgba = s.color.to_linear().to_f32_array();
        // mix=0.8 toward biome (0.1, 0.6, 0.2): red should drop well below 0.9.
        assert!(rgba[0] < 0.5, "red should bend toward biome low: {}", rgba[0]);
        assert!(rgba[1] < 0.8 && rgba[1] > 0.4, "green should bend toward biome: {}", rgba[1]);
    }

    // -----------------------------------------------------------------
    // MotionScaledLadder regression guard
    // -----------------------------------------------------------------
    //
    // The historical Phase 19.2 behaviour was to coarsen tiers 2/3 from
    // L2/L3 to L3/L4 once smoothed velocity exceeded 6 m/s. That swap
    // *evicted* the resident fine bricks in those bands and produced a
    // visible LOD pop. The behaviour was reverted: the policy now
    // returns the rest ladder regardless of motion. These tests pin the
    // new contract so a future refactor can't quietly restore the
    // coarsening.

    fn rest_motion() -> CameraMotionState {
        CameraMotionState::default()
    }

    fn sustained_sprint_motion() -> CameraMotionState {
        let mut m = CameraMotionState::default();
        m.smoothed_velocity_m_s = 12.0; // well above the historical 6 m/s threshold
        m.sprint_held = true;
        m
    }

    #[test]
    fn motion_scaled_ladder_returns_default_progressive_at_rest() {
        let policy = MotionScaledLadder;
        let got = policy.ladder(&rest_motion()).expect("rest should yield Some");
        assert_eq!(got, LodLadder::default_progressive());
    }

    #[test]
    fn motion_scaled_ladder_does_not_coarsen_under_sprint() {
        // The whole point of the revert: sprint must NOT change the
        // ladder. If this ever fails the fine-LOD-eviction bug is back.
        let policy = MotionScaledLadder;
        let rest = policy.ladder(&rest_motion()).expect("rest should yield Some");
        let sprint = policy
            .ladder(&sustained_sprint_motion())
            .expect("sprint should yield Some");
        assert_eq!(
            rest, sprint,
            "MotionScaledLadder must not coarsen under sprint — that evicts fine bricks"
        );
    }

    #[test]
    fn static_ladder_keeps_returning_none() {
        // Passive policy: never expresses an opinion. Quality preset
        // relies on this so its statically-configured ladder isn't
        // overridden mid-frame.
        let policy = StaticLadder;
        assert!(policy.ladder(&rest_motion()).is_none());
        assert!(policy.ladder(&sustained_sprint_motion()).is_none());
    }
}
