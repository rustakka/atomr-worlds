//! Spawn / show / hide a procedural sky-dome sphere parented to the
//! active camera, driven by the `SkyStrategy::dome_active` flag.
//!
//! Implementation notes
//! --------------------
//! - The dome is one inside-out sphere mesh. `cull_mode = Some(Face::Front)`
//!   on [`SkyDomeMaterial`] makes the back faces visible when the camera
//!   sits inside.
//! - We tag it with [`NotShadowCaster`] / [`NotShadowReceiver`] so it
//!   doesn't interact with the cascaded shadow path.
//! - The sphere is parented to the camera so it tracks the observer; no
//!   per-frame position sync needed.
//! - Visibility is toggled in `sync_sky_dome` instead of spawn/despawn
//!   so strategy swaps don't re-build the asset every frame.

use bevy::pbr::{NotShadowCaster, NotShadowReceiver};
use bevy::prelude::*;
use bevy::render::view::NoFrustumCulling;

use super::config::RenderConfig;
use super::materials::SkyDomeMaterial;
use super::sun::WorldTime;
use crate::modes::fp::WorldCamera;

/// Marker on the sky-dome sphere entity (parented to the camera).
#[derive(Component)]
pub struct SkyDome;

pub struct SkyDomePlugin;

impl Plugin for SkyDomePlugin {
    fn build(&self, app: &mut App) {
        app.add_plugins(MaterialPlugin::<SkyDomeMaterial>::default()).add_systems(
            Update,
            (ensure_sky_dome, sync_sky_dome).chain(),
        );
    }
}

/// Spawn the sky-dome sphere the first time we have a camera. Done
/// lazily because the camera entity is created by `FpPlugin::Startup`,
/// which runs alongside `RenderPlugin`/`SkyDomePlugin` startup — and we
/// don't want to assume ordering.
#[allow(clippy::too_many_arguments)]
fn ensure_sky_dome(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<SkyDomeMaterial>>,
    cam_q: Query<Entity, With<WorldCamera>>,
    dome_q: Query<(), With<SkyDome>>,
) {
    if !dome_q.is_empty() {
        return;
    }
    let Ok(camera) = cam_q.get_single() else { return };

    let sphere = Mesh::from(Sphere::new(800.0).mesh().uv(32, 16));
    let mesh = meshes.add(sphere);
    let material = materials.add(SkyDomeMaterial::default());

    let dome = commands
        .spawn((
            MaterialMeshBundle {
                mesh,
                material,
                // Initial visibility is hidden; `sync_sky_dome` flips it
                // on if the strategy says so.
                visibility: Visibility::Hidden,
                ..default()
            },
            SkyDome,
            NotShadowCaster,
            NotShadowReceiver,
            NoFrustumCulling,
        ))
        .id();
    commands.entity(camera).add_child(dome);
}

/// Each frame: toggle visibility based on `cfg.sky.dome_active()`, and
/// update the material's uniforms from the current sun + sky state.
#[allow(clippy::type_complexity)]
fn sync_sky_dome(
    cfg: Res<RenderConfig>,
    world_time: Res<WorldTime>,
    mut materials: ResMut<Assets<SkyDomeMaterial>>,
    mut q: Query<(&mut Visibility, &Handle<SkyDomeMaterial>), With<SkyDome>>,
) {
    let active = cfg.sky.dome_active();
    let sun_state = cfg.sun_curve.sun_state(world_time.0);
    let horizon = color_to_vec4(cfg.sky.horizon_color(sun_state));
    let zenith = color_to_vec4(cfg.sky.zenith_color(sun_state));
    let sun_color = color_to_vec4(sun_state.color);
    let sun_dir = sun_state.direction;
    for (mut vis, handle) in q.iter_mut() {
        *vis = if active { Visibility::Visible } else { Visibility::Hidden };
        if !active {
            continue;
        }
        if let Some(mat) = materials.get_mut(handle) {
            mat.horizon_color = horizon;
            mat.zenith_color = zenith;
            mat.sun_color = sun_color;
            mat.sun_direction = Vec4::new(sun_dir.x, sun_dir.y, sun_dir.z, 0.0);
        }
    }
}

fn color_to_vec4(c: Color) -> Vec4 {
    let lin = c.to_linear().to_f32_array();
    Vec4::new(lin[0], lin[1], lin[2], lin[3])
}
