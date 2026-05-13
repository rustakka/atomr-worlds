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
    /// If true, [`super::WorldTime`] advances each frame; if false (the
    /// default), it only moves when the harness or user code sets it.
    pub time_advances_automatically: bool,
    /// Wall seconds per in-game hour when auto-advancing.
    pub seconds_per_hour: f32,
}

impl Default for RenderConfig {
    /// Strategy spine ships in pre-upgrade behaviour: greedy meshing, no
    /// AO, legacy vertex-color shading, the previous static sun and flat
    /// sky, no shadows, no fog, Bevy's stock tonemap. Each subsequent
    /// step (1–9) flips one or more of these to the upgraded strategy.
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
            time_advances_automatically: false,
            seconds_per_hour: 60.0,
        }
    }
}

impl RenderConfig {
    /// Apply a named preset. Returns `false` if the name is unknown so
    /// callers (the harness) can surface a clear error.
    pub fn apply_preset(&mut self, preset: RenderPreset) {
        match preset {
            RenderPreset::Legacy => {
                // Pre-upgrade defaults: greedy mesh, no AO, static sun,
                // no shadows, flat sky, no fog, stock tonemap.
                self.mesher = Arc::new(GreedyFlat);
                self.ao = Arc::new(NoAo);
                self.shading = Arc::new(LegacyVertexColor);
                self.sky = Arc::new(ConstantSky);
                self.sun_curve = Arc::new(StaticSun);
                self.shadow = Arc::new(NoShadows);
                self.fog = Arc::new(NoFog);
                self.tonemap = Arc::new(DefaultTonemap);
            }
            RenderPreset::Stylized => {
                self.ao = Arc::new(MinecraftCornerAo);
                self.sun_curve = Arc::new(KeyframeLutSun);
                self.shadow = Arc::new(BasicCascades::default());
                self.fog = Arc::new(ExpSquaredSkyTintedFog::default());
                self.sky = Arc::new(SkyTinted);
                self.tonemap = Arc::new(AcesTonemap);
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
            }
        }
    }
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
