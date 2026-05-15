//! Name → strategy registry, used by the harness `set_strategy` event so
//! scenarios can A/B compare strategies in TOML without code edits.
//!
//! Adding a new strategy impl: write a constructor closure here under the
//! correct slot. The slot names are the [`RenderConfig`] field names.

use std::sync::Arc;

use super::config::RenderConfig;
use super::defaults::*;

/// Apply a strategy by `(slot, name)`. Returns `true` on success, `false`
/// if either the slot or the name is unknown.
pub fn apply_strategy_by_name(cfg: &mut RenderConfig, slot: &str, name: &str) -> bool {
    match slot {
        "mesher" => match name {
            "GreedyFlat" => {
                cfg.mesher = Arc::new(GreedyFlat);
                true
            }
            _ => false,
        },
        "palette" => match name {
            "HardcodedPalette" => {
                cfg.palette = Arc::new(HardcodedPalette);
                true
            }
            _ => false,
        },
        "ao" => match name {
            "NoAo" => {
                cfg.ao = Arc::new(NoAo);
                true
            }
            _ => false,
        },
        "shading" => match name {
            "LegacyVertexColor" => {
                cfg.shading = Arc::new(LegacyVertexColor);
                true
            }
            "PaletteVoxelMaterial" => {
                cfg.shading = Arc::new(PaletteVoxelMaterial);
                true
            }
            _ => false,
        },
        "sky" => match name {
            "ConstantSky" => {
                cfg.sky = Arc::new(ConstantSky);
                true
            }
            "SkyTinted" => {
                cfg.sky = Arc::new(SkyTinted);
                true
            }
            "ProceduralDomeSky" => {
                cfg.sky = Arc::new(ProceduralDomeSky);
                true
            }
            _ => false,
        },
        "sun_curve" => match name {
            "StaticSun" => {
                cfg.sun_curve = Arc::new(StaticSun);
                true
            }
            "KeyframeLutSun" => {
                cfg.sun_curve = Arc::new(KeyframeLutSun);
                true
            }
            _ => false,
        },
        "shadow" => match name {
            "NoShadows" => {
                cfg.shadow = Arc::new(NoShadows);
                true
            }
            "BasicCascades" => {
                cfg.shadow = Arc::new(BasicCascades::default());
                true
            }
            _ => false,
        },
        "fog" => match name {
            "NoFog" => {
                cfg.fog = Arc::new(NoFog);
                true
            }
            "ExpSquaredSkyTintedFog" => {
                cfg.fog = Arc::new(ExpSquaredSkyTintedFog::default());
                true
            }
            _ => false,
        },
        "tonemap" => match name {
            "DefaultTonemap" => {
                cfg.tonemap = Arc::new(DefaultTonemap);
                true
            }
            "AcesTonemap" => {
                cfg.tonemap = Arc::new(AcesTonemap);
                true
            }
            _ => false,
        },
        "coverage" => match name {
            "MaskedShells" => {
                cfg.coverage = Arc::new(MaskedShells);
                true
            }
            "NestedSummary" => {
                cfg.coverage = Arc::new(NestedSummary);
                true
            }
            _ => false,
        },
        "slice" => match name {
            "FlatSlice" => {
                cfg.slice = Arc::new(FlatSlice);
                true
            }
            "HillshadeSlice" => {
                cfg.slice = Arc::new(HillshadeSlice::default());
                true
            }
            _ => false,
        },
        _ => false,
    }
}
