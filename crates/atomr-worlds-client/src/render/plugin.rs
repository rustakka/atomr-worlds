//! Glue plugin that inserts the strategy resources + registers the
//! sun-sync systems + the custom material plugins (voxel + sky dome).

use bevy::pbr::MaterialPlugin;
use bevy::prelude::*;

use super::materials::VoxelMaterial;
use super::sky_dome::SkyDomePlugin;
use super::skybox::SkyboxPlugin;
use super::sun::{advance_world_time, sync_sky_and_fog, sync_sun};
use super::{RenderConfig, WorldTime};

pub struct RenderPlugin;

impl Plugin for RenderPlugin {
    fn build(&self, app: &mut App) {
        if !app.world.contains_resource::<RenderConfig>() {
            app.insert_resource(RenderConfig::default());
        }
        if !app.world.contains_resource::<WorldTime>() {
            app.insert_resource(WorldTime::default());
        }
        // The voxel material is always registered (so a runtime
        // `set_strategy` to `PaletteVoxelMaterial` doesn't require a
        // restart); FP only allocates handles when the strategy mode
        // is `PaletteVoxelMaterial`.
        app.add_plugins(MaterialPlugin::<VoxelMaterial>::default());
        app.add_plugins(SkyDomePlugin);
        // Cubemap skybox: seeds a placeholder 1×1×6 black handle into
        // `SkyboxRuntime` so the FP camera can spawn with a valid
        // `core_pipeline::Skybox` component on the first frame; the real
        // bake replaces the handle once the streamer's far ring is
        // populated (see `sync_skybox`).
        app.add_plugins(SkyboxPlugin);
        // Sun systems run in Update so they pick up `WorldTime` changes
        // pushed by the harness's `set_time_of_day` event in PreUpdate.
        // Order: advance the clock → drive the sun → drive sky/fog from
        // the new sun state.
        app.add_systems(
            Update,
            (advance_world_time, sync_sun, sync_sky_and_fog).chain(),
        );
    }
}
