//! Default strategy implementations. Step 0 ships behaviour-preserving
//! defaults; later steps add richer ones (each behind a new type so swapping
//! is one line in [`RenderConfig`](super::RenderConfig)).

use std::f32::consts::PI;

use atomr_worlds_view::{bake_ao, greedy_mesh, MaterialEntry, MaterialPalette, Mesh};
use atomr_worlds_voxel::Brick;
use bevy::core_pipeline::bloom::BloomSettings;
use bevy::core_pipeline::tonemapping::Tonemapping;
use bevy::pbr::{
    CascadeShadowConfig, CascadeShadowConfigBuilder, FogFalloff, FogSettings,
};
use bevy::prelude::*;
use bevy::render::camera::Exposure;

use super::strategy::*;

// ---------------------------------------------------------------------------
// Mesh — greedy (today's path)
// ---------------------------------------------------------------------------

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
            color: Color::rgb(1.0, 0.97, 0.9),
            illuminance: 80_000.0,
            day_factor: 1.0,
        }
    }
    fn ambient(&self, _hours: f32) -> (Color, f32) {
        (Color::rgb(0.85, 0.88, 1.0), 1.2)
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
            color: Color::rgb(rgb.x, rgb.y, rgb.z),
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
        (Color::rgb(rgb.x, rgb.y, rgb.z), b)
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
        Color::rgb(0.45, 0.65, 0.85)
    }
    fn zenith_color(&self, _sun: SunState) -> Color {
        Color::rgb(0.30, 0.55, 0.85)
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
pub struct BasicCascades {
    pub num_cascades: usize,
    pub minimum_distance: f32,
    pub maximum_distance: f32,
    pub first_cascade_far_bound: f32,
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
    ) -> FogSettings {
        FogSettings {
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
    ) -> FogSettings {
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
        FogSettings {
            color: sky_horizon,
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
    fn bloom(&self) -> Option<BloomSettings> {
        Some(BloomSettings { intensity: 0.10, ..default() })
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
    let lin = c.as_linear_rgba_f32();
    Vec3::new(lin[0], lin[1], lin[2])
}

fn vec3_to_color(v: Vec3) -> Color {
    Color::rgb_linear(v.x, v.y, v.z)
}
