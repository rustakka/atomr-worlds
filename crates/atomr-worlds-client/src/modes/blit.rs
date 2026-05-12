//! Shared 2D blit path used by slice / rts / overview modes.
//!
//! Each frame those modes call into `atomr-worlds-view` (CPU rasterizer)
//! to produce a [`Framebuffer`](atomr_worlds_view::Framebuffer), then this
//! plugin copies the RGBA bytes into a Bevy [`Image`] displayed
//! fullscreen by a 2D camera + sprite.

use atomr_worlds_view::Framebuffer;
use bevy::prelude::*;
use bevy::render::render_asset::RenderAssetUsages;
use bevy::render::render_resource::{Extent3d, TextureDimension, TextureFormat};

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

fn setup_blit(mut commands: Commands, mut images: ResMut<Assets<Image>>) {
    let mut image = Image::new_fill(
        Extent3d { width: RASTER_W, height: RASTER_H, depth_or_array_layers: 1 },
        TextureDimension::D2,
        &[0, 0, 0, 255],
        TextureFormat::Rgba8UnormSrgb,
        RenderAssetUsages::default(),
    );
    image.data = vec![0u8; (RASTER_W * RASTER_H * 4) as usize];
    let handle = images.add(image);

    commands.spawn((
        Camera2dBundle {
            camera: Camera { order: 1, is_active: false, ..default() },
            ..default()
        },
        BlitCamera,
    ));
    commands.spawn((
        SpriteBundle {
            texture: handle.clone(),
            sprite: Sprite {
                custom_size: Some(Vec2::new(RASTER_W as f32, RASTER_H as f32)),
                ..default()
            },
            visibility: Visibility::Hidden,
            ..default()
        },
        BlitSprite,
    ));

    commands.insert_resource(RasterTarget { image: handle });
}

fn toggle_blit_visibility(
    mode: Res<ViewMode>,
    mut cameras: Query<&mut Camera, With<BlitCamera>>,
    mut sprites: Query<&mut Visibility, With<BlitSprite>>,
) {
    let active = matches!(
        *mode,
        ViewMode::Slice | ViewMode::Rts | ViewMode::Overview
    );
    if let Ok(mut cam) = cameras.get_single_mut() {
        cam.is_active = active;
    }
    if let Ok(mut vis) = sprites.get_single_mut() {
        *vis = if active { Visibility::Visible } else { Visibility::Hidden };
    }
}

fn fit_sprite_to_window(
    windows: Query<&Window, With<bevy::window::PrimaryWindow>>,
    mode: Res<ViewMode>,
    mut q: Query<&mut Sprite, With<BlitSprite>>,
) {
    if !matches!(*mode, ViewMode::Slice | ViewMode::Rts | ViewMode::Overview) {
        return;
    }
    let Ok(win) = windows.get_single() else { return };
    let Ok(mut sprite) = q.get_single_mut() else { return };
    let w = win.width();
    let h = win.height();
    let scale = (w / RASTER_W as f32).min(h / RASTER_H as f32);
    sprite.custom_size = Some(Vec2::new(RASTER_W as f32 * scale, RASTER_H as f32 * scale));
}

/// Copy a [`Framebuffer`] into the shared [`RasterTarget`] image. Mode
/// plugins call this each frame they're active.
pub fn copy_framebuffer_to_image(images: &mut Assets<Image>, target: &RasterTarget, fb: &Framebuffer) {
    let Some(img) = images.get_mut(&target.image) else { return };
    debug_assert_eq!(fb.width, RASTER_W);
    debug_assert_eq!(fb.height, RASTER_H);
    if (fb.pixels.len() as u32) == RASTER_W * RASTER_H * 4 {
        img.data.copy_from_slice(&fb.pixels);
    } else {
        tracing::warn!(
            len = fb.pixels.len(),
            expected = RASTER_W * RASTER_H * 4,
            "framebuffer size mismatch — dropping frame"
        );
    }
}
