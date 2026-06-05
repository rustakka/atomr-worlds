//! Shared 2D blit path used by slice / rts / overview modes.
//!
//! Each frame those modes call into `atomr-worlds-view` (CPU rasterizer)
//! to produce a [`Framebuffer`](atomr_worlds_view::Framebuffer), then this
//! plugin copies the RGBA bytes into a Bevy [`Image`] displayed
//! fullscreen by a 2D camera + sprite.

use atomr_worlds_view::Framebuffer;
use bevy::prelude::*;
use bevy::camera::{ClearColorConfig, RenderTarget};
use bevy::asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

use crate::modes::fp::WorldCamera;
use crate::render::OffscreenTarget;
use crate::view_mode::ViewMode;

/// Fixed render-target size for the CPU rasterizer. Bevy scales the sprite
/// to fill the window. Keep it modest so a single-thread rasterize is
/// well under one frame at 60Hz.
pub const RASTER_W: u32 = 256;
pub const RASTER_H: u32 = 256;

#[derive(Resource)]
pub struct RasterTarget {
    pub image: Handle<Image>,
}

#[derive(Component)]
pub struct BlitSprite;

#[derive(Component)]
pub struct BlitCamera;

pub struct BlitPlugin;

impl Plugin for BlitPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Startup, setup_blit).add_systems(
            Update,
            (toggle_blit_visibility, fit_sprite_to_window).chain(),
        );
    }
}

fn setup_blit(
    mut commands: Commands,
    mut images: ResMut<Assets<Image>>,
    offscreen: Option<Res<OffscreenTarget>>,
) {
    let mut image = Image::new_fill(
        Extent3d { width: RASTER_W, height: RASTER_H, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    // Bevy 0.16: `Image.data` is `Option<Vec<u8>>`.
    image.data = Some(vec![0u8; (RASTER_W * RASTER_H * 4) as usize]);
    let handle = images.add(image);

    // When the harness is active the FP camera renders to an offscreen
    // image (see `render::offscreen`). The blit overlay must target that
    // same image — otherwise the slice / rts / overview raster, drawn by
    // this Camera2d, would never show up in harness screenshots. The
    // `order: 1` keeps it compositing on top of the (cleared) 3D camera.
    let camera_target = offscreen
        .as_deref()
        .map(|t| RenderTarget::Image(t.image.clone().into()))
        .unwrap_or_default();

    // Bevy 0.15+: bundles removed — spawn the required components directly.
    commands.spawn((
        Camera2d,
        Camera {
            order: 1,
            is_active: false,
            target: camera_target,
            // Solid black clear so the letterbox bars around the
            // 1:1 raster sprite (256² scaled into a non-square
            // target) are deterministic instead of showing whatever
            // the WorldCamera last rendered. With WorldCamera also
            // toggled inactive in raster modes, this clear owns the
            // entire offscreen / window target before the sprite
            // and the routed HUD UI composite on top (see
            // `hud::route_hud_target` — UI follows the active camera,
            // so it lands above the sprite in this mode).
            clear_color: ClearColorConfig::Custom(Color::BLACK),
            ..default()
        },
        BlitCamera,
    ));
    // Bevy 0.15: the sprite's image moved into `Sprite.image`.
    commands.spawn((
        Sprite {
            image: handle.clone(),
            custom_size: Some(Vec2::new(RASTER_W as f32, RASTER_H as f32)),
            ..default()
        },
        Visibility::Hidden,
        BlitSprite,
    ));

    commands.insert_resource(RasterTarget { image: handle });
}

fn toggle_blit_visibility(
    mode: Res<ViewMode>,
    mut blit_cameras: Query<&mut Camera, (With<BlitCamera>, Without<WorldCamera>)>,
    mut world_cameras: Query<&mut Camera, (With<WorldCamera>, Without<BlitCamera>)>,
    mut sprites: Query<&mut Visibility, With<BlitSprite>>,
) {
    let active = raster_mode_active(*mode);
    if let Ok(mut cam) = blit_cameras.single_mut() {
        cam.is_active = active;
    }
    // Disable the FP/TP world camera in raster modes. `Visibility::Hidden`
    // on a Bevy 0.13 Camera entity hides only its rendered geometry
    // descendants, not the camera's clear+render pass — without this
    // toggle, the world camera (order 0) would still clear the offscreen
    // image to its sky-blue ClearColor every frame, and any portion of
    // the target the BlitCamera's letterboxed sprite doesn't cover would
    // show that sky color in harness PNGs (the original "overview shows
    // empty sky" symptom).
    if let Ok(mut cam) = world_cameras.single_mut() {
        cam.is_active = !active;
    }
    if let Ok(mut vis) = sprites.single_mut() {
        *vis = if active { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn fit_sprite_to_window(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mode: Res<ViewMode>,
    mut q: Query<&mut Sprite, With<BlitSprite>>,
) {
    if !raster_mode_active(*mode) {
        return;
    }
    let Ok(win) = windows.single() else { return };
    let Ok(mut sprite) = q.single_mut() else { return };
    let w = win.width();
    let h = win.height();
    let scale = (w / RASTER_W as f32).min(h / RASTER_H as f32);
    sprite.custom_size = Some(Vec2::new(RASTER_W as f32 * scale, RASTER_H as f32 * scale));
}

/// True when [`ViewMode`] should activate the BlitCamera and (correspondingly)
/// deactivate the WorldCamera. Single source of truth for the toggle.
pub(crate) fn raster_mode_active(mode: ViewMode) -> bool {
    matches!(mode, ViewMode::Slice | ViewMode::Rts | ViewMode::Overview)
}

/// Copy a [`Framebuffer`] into the shared [`RasterTarget`] image. Mode
/// plugins call this each frame they're active.
pub fn copy_framebuffer_to_image(images: &mut Assets<Image>, target: &RasterTarget, fb: &Framebuffer) {
    let Some(img) = images.get_mut(&target.image) else { return };
    debug_assert_eq!(fb.width, RASTER_W);
    debug_assert_eq!(fb.height, RASTER_H);
    if (fb.pixels.len() as u32) == RASTER_W * RASTER_H * 4 {
        // Bevy 0.16: `Image.data` is `Option<Vec<u8>>`.
        if let Some(data) = img.data.as_mut() {
            data.copy_from_slice(&fb.pixels);
        }
    } else {
        tracing::warn!(
            len = fb.pixels.len(),
            expected = RASTER_W * RASTER_H * 4,
            "framebuffer size mismatch — dropping frame"
        );
    }
}

#[cfg(test)]
mod tests {
    //! Camera-toggle correctness. The original "overview shows empty
    //! sky" harness bug was the WorldCamera staying active while the
    //! BlitCamera's letterboxed sprite covered only part of the offscreen
    //! target — the world's sky ClearColor bled through the bars. These
    //! tests pin the invariant that *exactly one* of the two cameras is
    //! active per `ViewMode`.
    use super::*;
    use crate::view_mode::ViewMode;
    use bevy::ecs::system::RunSystemOnce;
    use bevy::prelude::*;

    fn setup_world(mode: ViewMode) -> World {
        let mut world = World::new();
        // Spawn stand-ins for the two cameras the toggle queries. We
        // start the BlitCamera *active* and the WorldCamera *inactive*
        // so a stale "active" flag from before the toggle ran can't
        // accidentally satisfy the assertion.
        world.spawn((
            Camera { is_active: true, ..default() },
            BlitCamera,
        ));
        world.spawn((
            Camera { is_active: false, ..default() },
            WorldCamera,
        ));
        world.insert_resource(mode);
        world
    }

    fn camera_active<F: Component>(world: &mut World) -> bool {
        let mut q = world.query_filtered::<&Camera, With<F>>();
        // Bevy 0.15+: `QueryState::single` returns a `Result`.
        q.single(world).unwrap().is_active
    }

    #[test]
    fn fp_mode_activates_world_camera_and_deactivates_blit() {
        let mut world = setup_world(ViewMode::Fp);
        world.run_system_once(toggle_blit_visibility);
        assert!(camera_active::<WorldCamera>(&mut world), "FP mode keeps the world camera active");
        assert!(!camera_active::<BlitCamera>(&mut world), "FP mode disables the blit camera");
    }

    #[test]
    fn tp_mode_activates_world_camera_and_deactivates_blit() {
        let mut world = setup_world(ViewMode::Tp);
        world.run_system_once(toggle_blit_visibility);
        assert!(camera_active::<WorldCamera>(&mut world));
        assert!(!camera_active::<BlitCamera>(&mut world));
    }

    #[test]
    fn overview_mode_disables_world_camera_and_enables_blit() {
        let mut world = setup_world(ViewMode::Overview);
        world.run_system_once(toggle_blit_visibility);
        assert!(
            !camera_active::<WorldCamera>(&mut world),
            "overview mode must disable the world camera so its sky-blue clear \
             does not bleed through the BlitCamera's letterbox bars",
        );
        assert!(camera_active::<BlitCamera>(&mut world));
    }

    #[test]
    fn slice_mode_disables_world_camera_and_enables_blit() {
        let mut world = setup_world(ViewMode::Slice);
        world.run_system_once(toggle_blit_visibility);
        assert!(!camera_active::<WorldCamera>(&mut world));
        assert!(camera_active::<BlitCamera>(&mut world));
    }

    #[test]
    fn rts_mode_disables_world_camera_and_enables_blit() {
        let mut world = setup_world(ViewMode::Rts);
        world.run_system_once(toggle_blit_visibility);
        assert!(!camera_active::<WorldCamera>(&mut world));
        assert!(camera_active::<BlitCamera>(&mut world));
    }

    #[test]
    fn raster_mode_active_classifies_every_view_mode() {
        assert!(!raster_mode_active(ViewMode::Fp));
        assert!(!raster_mode_active(ViewMode::Tp));
        assert!(raster_mode_active(ViewMode::Slice));
        assert!(raster_mode_active(ViewMode::Rts));
        assert!(raster_mode_active(ViewMode::Overview));
    }
}
