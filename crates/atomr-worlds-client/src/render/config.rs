//! [`RenderConfig`] — strategy registry resource.
//!
//! Every render decision is a `Arc<dyn Trait>` field here. The default
//! pipeline (`RenderConfig::default()`) ships behaviour-preserving
//! strategies so the strategy spine can land without visual change. Each
//! later step swaps one or more defaults to a richer impl.
//!
//! Use [`RenderPreset`] for one-line swaps from the harness or CLI.

use std::sync::Arc;

use bevy::prelude::*;

use super::defaults::*;
use super::strategy::*;

#[derive(Resource, Clone)]
pub struct RenderConfig {
    pub mesher:    Arc<dyn MeshStrategy>,
    pub palette:   Arc<dyn PaletteStrategy>,
    pub ao:        Arc<dyn AoStrategy>,
    pub shading:   Arc<dyn ShadingStrategy>,
    pub sky:       Arc<dyn SkyStrategy>,
    pub sun_curve: Arc<dyn SunCurveStrategy>,
    pub shadow:    Arc<dyn ShadowStrategy>,
    pub fog:       Arc<dyn FogStrategy>,
    pub tonemap:   Arc<dyn TonemapStrategy>,
    /// Decides whether coarser-LOD bricks stay resident underneath
    /// finer-LOD shells (`NestedSummary`, the default — smoother LOD
    /// transitions, ~15 % more bricks) or are masked out
    /// (`MaskedShells` — historical, one tier per shell).
    pub coverage:  Arc<dyn LodCoveragePolicy>,
    /// How the Dwarf-Fortress slice view rasterizes a slice table —
    /// `HillshadeSlice` (the default — relief-shaded) or `FlatSlice`
    /// (historical flat fill).
    pub slice:     Arc<dyn SliceRenderStrategy>,
    /// Polar-annulus terrain shell drawn between the streamer's outer
    /// ring and the geometric horizon. `PolarAnnulusShell` (default)
    /// fills the band with elevation-shaded representative terrain;
    /// `NoHorizonImposter` (Legacy preset) disables the shell.
    pub horizon_imposter: Arc<dyn HorizonImposterStrategy>,
    /// LOD ladder policy — returns a coarser ladder under sustained
    /// motion or `None` to keep the configured one. `StaticLadder`
    /// (Quality preset) always returns `None`.
    pub lod_ladder:        Arc<dyn LodLadderPolicy>,
    /// Per-frame brick-spawn budget. `MotionScaledSpawnBudget` (default)
    /// ramps down at sprint to smooth GPU-upload spikes.
    pub spawn_budget:      Arc<dyn SpawnBudgetStrategy>,
    /// Stride for `fp_update_lod_visibility`. `StaticVisibilityCadence`
    /// (Quality preset) keeps it at 1; `MotionScaledCadence` lifts it
    /// to 2/3 under motion.
    pub visibility_cadence: Arc<dyn VisibilityCadenceStrategy>,
    /// Drift / cosine thresholds for the plan rebuild trigger.
    /// `StaticRebuildThreshold` matches the historical constants;
    /// `MotionScaledRebuildThreshold` widens them at sprint when the
    /// horizon imposter is active.
    pub rebuild_threshold: Arc<dyn RebuildThresholdStrategy>,
    /// If true, [`super::WorldTime`] advances each frame; if false (the
    /// default), it only moves when the harness or user code sets it.
    pub time_advances_automatically: bool,
    /// Wall seconds per in-game hour when auto-advancing.
    pub seconds_per_hour: f32,
}

impl Default for RenderConfig {
    /// Ships the upgraded look that's the focus of this build: greedy
    /// meshing, Minecraft-style corner AO, the keyframe-LUT sun curve,
    /// cascaded shadows, a procedural sky dome, exp²-fog tinted by the
    /// sky horizon, ACES tonemapping, and the nested-summary LOD
    /// coverage that crossfades through tier transitions. Legacy
    /// (pre-upgrade) behaviour is still reachable via
    /// `RenderPreset::Legacy`.
    fn default() -> Self {
        Self {
            mesher:    Arc::new(GreedyFlat),
            palette:   Arc::new(HardcodedPalette),
            ao:        Arc::new(MinecraftCornerAo),
            shading:   Arc::new(LegacyVertexColor),
            sky:       Arc::new(ProceduralDomeSky),
            sun_curve: Arc::new(KeyframeLutSun),
            shadow:    Arc::new(BasicCascades::default()),
            fog:       Arc::new(ExpSquaredSkyTintedFog::default()),
            tonemap:   Arc::new(AcesTonemap),
            coverage:  Arc::new(NestedSummary),
            slice:     Arc::new(HillshadeSlice::default()),
            horizon_imposter:   Arc::new(PolarAnnulusShell::default()),
            lod_ladder:         Arc::new(MotionScaledLadder),
            spawn_budget:       Arc::new(MotionScaledSpawnBudget::default()),
            visibility_cadence: Arc::new(MotionScaledCadence),
            rebuild_threshold:  Arc::new(MotionScaledRebuildThreshold),
            time_advances_automatically: false,
            seconds_per_hour: 60.0,
        }
    }
}

impl RenderConfig {
    /// Apply a [`PerfPreset`]. `Balanced` (default) leaves the
    /// motion-aware strategy slots untouched; `Quality` swaps all four
    /// to static no-op implementations so motion stops driving any
    /// behavior. Exists so the `--perf` CLI flag can hand off the
    /// preset choice without leaking the strategy types into `main.rs`.
    pub fn apply_perf_preset(&mut self, preset: PerfPreset) {
        match preset {
            PerfPreset::Balanced => {
                self.lod_ladder = Arc::new(MotionScaledLadder);
                self.spawn_budget = Arc::new(MotionScaledSpawnBudget::default());
                self.visibility_cadence = Arc::new(MotionScaledCadence);
                self.rebuild_threshold = Arc::new(MotionScaledRebuildThreshold);
            }
            PerfPreset::Quality => {
                self.lod_ladder = Arc::new(StaticLadder);
                self.spawn_budget = Arc::new(StaticSpawnBudget::default());
                self.visibility_cadence = Arc::new(StaticVisibilityCadence);
                self.rebuild_threshold = Arc::new(StaticRebuildThreshold);
            }
        }
    }

    /// Apply a named preset. Returns `false` if the name is unknown so
    /// callers (the harness) can surface a clear error.
    pub fn apply_preset(&mut self, preset: RenderPreset) {
        match preset {
            RenderPreset::Legacy => {
                // Pre-upgrade defaults: greedy mesh, no AO, static sun,
                // no shadows, flat sky, no fog, stock tonemap. Keeps the
                // historical one-tier-per-shell coverage so the preset
                // really is "what it was".
                self.mesher = Arc::new(GreedyFlat);
                self.ao = Arc::new(NoAo);
                self.shading = Arc::new(LegacyVertexColor);
                self.sky = Arc::new(ConstantSky);
                self.sun_curve = Arc::new(StaticSun);
                self.shadow = Arc::new(NoShadows);
                self.fog = Arc::new(NoFog);
                self.tonemap = Arc::new(DefaultTonemap);
                self.coverage = Arc::new(MaskedShells);
                self.slice = Arc::new(FlatSlice);
                // No imposter — legacy renders only the LOD ladder.
                self.horizon_imposter = Arc::new(NoHorizonImposter);
            }
            RenderPreset::Stylized => {
                self.ao = Arc::new(MinecraftCornerAo);
                self.sun_curve = Arc::new(KeyframeLutSun);
                self.shadow = Arc::new(BasicCascades::default());
                self.fog = Arc::new(ExpSquaredSkyTintedFog::default());
                self.sky = Arc::new(SkyTinted);
                self.tonemap = Arc::new(AcesTonemap);
                self.slice = Arc::new(HillshadeSlice::default());
            }
            RenderPreset::Debug => {
                // No fog, no shadows, static sun. Useful for inspecting
                // raw geometry/material output.
                self.ao = Arc::new(NoAo);
                self.sun_curve = Arc::new(StaticSun);
                self.shadow = Arc::new(NoShadows);
                self.fog = Arc::new(NoFog);
                self.sky = Arc::new(ConstantSky);
                self.tonemap = Arc::new(DefaultTonemap);
                self.slice = Arc::new(FlatSlice);
            }
        }
    }
}

/// Performance preset. `Balanced` (default) wires the motion-aware
/// strategy layer ([`MotionScaledLadder`], [`MotionScaledSpawnBudget`],
/// [`MotionScaledCadence`], [`MotionScaledRebuildThreshold`]) — these
/// coarsen LOD, throttle spawn budget, stride visibility, and widen
/// rebuild thresholds when the camera is moving fast. `Quality` swaps
/// all four to static no-ops, fixing visual fidelity at the rest-state
/// level whether the player is moving or not. Surfaced by
/// [`crate::cli::PerfPreset`] via `--perf`.
#[derive(Clone, Copy, Debug)]
pub enum PerfPreset {
    Balanced,
    Quality,
}

/// One-line strategy bundles. Steps 1–10 wire individual strategies; this
/// enum exists so harness scenarios can swap entire looks atomically with
/// `set_render_preset`.
#[derive(Clone, Copy, Debug)]
pub enum RenderPreset {
    /// Pre-upgrade behaviour (greedy mesh + flat sky + static sun, no shadows/fog).
    Legacy,
    /// The shipped look for the lighting+materials upgrade.
    Stylized,
    /// Static sun + flat sky, useful for material inspection.
    Debug,
}

impl RenderPreset {
    pub fn from_str(s: &str) -> Option<Self> {
        Some(match s {
            "legacy" => Self::Legacy,
            "stylized" => Self::Stylized,
            "debug" => Self::Debug,
            _ => return None,
        })
    }
}
