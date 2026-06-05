//! Offscreen-image capture for the harness.
//!
//! Sidesteps the X11/hybrid-GPU presentation issue: the FP/TP camera is
//! pointed at an [`Image`] render target instead of the window, and a
//! `RenderApp` system copies the resulting texture to a CPU-mapped buffer
//! each requested frame. The bytes land in a `Mutex`-guarded resource
//! that the harness reads back out and writes to PNG.
//!
//! Wired up via [`OffscreenCapturePlugin`]. Activated by the
//! `--harness` path; without it the camera renders to the window as
//! usual and the plugin is a no-op (the image asset still exists but
//! nothing writes to it and nothing maps it back).

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use bevy::prelude::*;
use bevy::render::extract_resource::{ExtractResource, ExtractResourcePlugin};
// Bevy 0.17: RenderAssetUsages moved to bevy_asset; RenderAssets stays in bevy_render.
use bevy::asset::RenderAssetUsages;
use bevy::render::render_asset::RenderAssets;
use bevy::render::render_resource::{
    BufferDescriptor, BufferUsages, CommandEncoderDescriptor, Extent3d, TexelCopyBufferInfo,
    TexelCopyTextureInfo, TexelCopyBufferLayout, MapMode, PollType, Origin3d, TextureAspect,
    TextureDimension, TextureFormat, TextureUsages,
};
use bevy::render::renderer::{RenderDevice, RenderQueue};
use bevy::render::texture::GpuImage;
use bevy::render::{Render, RenderApp, RenderSystems};

/// Pending capture request: a (frame_counter, png_path) pair. Pushed by
/// the harness, drained by the `RenderApp`.
#[derive(Default)]
pub struct CaptureQueue {
    pub pending: VecDeque<(u64, PathBuf)>,
    pub next_id: u64,
}

#[derive(Resource, Clone, Default)]
pub struct CaptureQueueHandle(pub Arc<Mutex<CaptureQueue>>);

#[derive(Resource, Clone)]
pub struct OffscreenTarget {
    pub image: Handle<Image>,
    pub width: u32,
    pub height: u32,
}

impl ExtractResource for OffscreenTarget {
    type Source = OffscreenTarget;
    fn extract_resource(source: &Self::Source) -> Self {
        source.clone()
    }
}

/// Outcome bus: the RenderApp pushes `(counter, success_message)` lines
/// here after each successful capture. The harness drains it in
/// `PostUpdate` and emits `HARNESS_SHOT` to stdout. Using a separate
/// outcome queue (rather than logging from the render thread) keeps the
/// `HARNESS_SHOT` ordering predictable.
#[derive(Resource, Clone, Default)]
pub struct CaptureOutcomes(pub Arc<Mutex<Vec<CaptureOutcome>>>);

#[derive(Debug, Clone)]
pub struct CaptureOutcome {
    pub id: u64,
    pub path: PathBuf,
    pub ok: bool,
    pub message: Option<String>,
}

pub struct OffscreenCapturePlugin {
    pub width: u32,
    pub height: u32,
}

impl Plugin for OffscreenCapturePlugin {
    fn build(&self, app: &mut App) {
        let width = self.width;
        let height = self.height;

        let queue = CaptureQueueHandle::default();
        let outcomes = CaptureOutcomes::default();

        // Build the offscreen image at startup so the camera can target
        // it before any rendering happens.
        let mut image = Image::new_fill(
            Extent3d { width, height, depth_or_array_layers: 1 },
            TextureDimension::D2,
            &[0, 0, 0, 0],
            TextureFormat::Bgra8UnormSrgb,
            RenderAssetUsages::MAIN_WORLD | RenderAssetUsages::RENDER_WORLD,
        );
        image.texture_descriptor.usage = TextureUsages::COPY_SRC
            | TextureUsages::TEXTURE_BINDING
            | TextureUsages::RENDER_ATTACHMENT;
        let mut images = app
            .world_mut()
            .get_resource_mut::<Assets<Image>>()
            .expect("Assets<Image> not initialised — RenderPlugin must run first");
        let handle = images.add(image);

        let target = OffscreenTarget { image: handle, width, height };

        app.insert_resource(target)
            .insert_resource(queue.clone())
            .insert_resource(outcomes.clone())
            .add_plugins(ExtractResourcePlugin::<OffscreenTarget>::default());

        let render_app = app.sub_app_mut(RenderApp);
        render_app
            .insert_resource(queue)
            .insert_resource(outcomes)
            .add_systems(Render, image_copy_system.in_set(RenderSystems::Cleanup));
    }
}

#[allow(clippy::too_many_arguments)]
fn image_copy_system(
    target: Option<Res<OffscreenTarget>>,
    queue_handle: Res<CaptureQueueHandle>,
    outcomes: Res<CaptureOutcomes>,
    gpu_images: Res<RenderAssets<GpuImage>>,
    device: Res<RenderDevice>,
    queue: Res<RenderQueue>,
) {
    let Some(target) = target else { return };

    // Pop one pending request per render frame. (More than one in a
    // single frame would conflate the resulting PNGs anyway, since the
    // render target only contains the latest frame's pixels.)
    let request = {
        let mut q = queue_handle.0.lock().unwrap();
        q.pending.pop_front()
    };
    let Some((id, path)) = request else { return };

    let push_outcome = |ok: bool, msg: Option<String>| {
        outcomes.0.lock().unwrap().push(CaptureOutcome {
            id,
            path: path.clone(),
            ok,
            message: msg,
        });
    };

    let Some(gpu_image) = gpu_images.get(target.image.id()) else {
        push_outcome(false, Some("GPU image not yet prepared".into()));
        return;
    };

    let width = target.width;
    let height = target.height;
    let bytes_per_pixel: u32 = 4;
    let unpadded = width * bytes_per_pixel;
    let align: u32 = 256; // wgpu COPY_BYTES_PER_ROW_ALIGNMENT
    let padded = unpadded.div_ceil(align) * align;
    let buffer_size = (padded as u64) * (height as u64);

    let buffer = device.create_buffer(&BufferDescriptor {
        label: Some("atomr_offscreen_capture_buffer"),
        size: buffer_size,
        usage: BufferUsages::MAP_READ | BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });

    let mut encoder = device.create_command_encoder(&CommandEncoderDescriptor {
        label: Some("atomr_offscreen_capture_encoder"),
    });
    encoder.copy_texture_to_buffer(
        TexelCopyTextureInfo {
            texture: &gpu_image.texture,
            mip_level: 0,
            origin: Origin3d::ZERO,
            aspect: TextureAspect::All,
        },
        TexelCopyBufferInfo {
            buffer: &buffer,
            layout: TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(padded),
                rows_per_image: Some(height),
            },
        },
        Extent3d { width, height, depth_or_array_layers: 1 },
    );
    queue.submit(std::iter::once(encoder.finish()));

    let slice = buffer.slice(..);
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    slice.map_async(MapMode::Read, move |result| {
        let _ = tx.send(result);
    });
    // Block until the map completes. wgpu drains queued callbacks here.
    let _ = device.poll(PollType::wait_indefinitely());
    let map_result = rx.recv();
    if let Err(e) = map_result {
        push_outcome(false, Some(format!("map channel closed: {e}")));
        return;
    }
    if let Ok(Err(e)) = map_result {
        push_outcome(false, Some(format!("map_async failed: {e}")));
        return;
    }

    let data = slice.get_mapped_range();
    // Strip per-row padding, swap BGRA → RGBA.
    let mut rgba = Vec::with_capacity((width * height * 4) as usize);
    for y in 0..height {
        let row_start = (y as usize) * (padded as usize);
        for x in 0..(width as usize) {
            let off = row_start + x * 4;
            let b = data[off];
            let g = data[off + 1];
            let r = data[off + 2];
            let a = data[off + 3];
            rgba.push(r);
            rgba.push(g);
            rgba.push(b);
            rgba.push(a);
        }
    }
    drop(data);
    buffer.unmap();

    // Save PNG. Done from the render thread to keep the bytes off the
    // shared queue (the outcome only carries success/failure).
    let img = image::RgbaImage::from_raw(width, height, rgba);
    let save_result = match img {
        Some(img) => img.save(&path).map_err(|e| e.to_string()),
        None => Err("RgbaImage::from_raw size mismatch".into()),
    };
    match save_result {
        Ok(()) => push_outcome(true, None),
        Err(e) => push_outcome(false, Some(e)),
    }

    drop(id); // silence unused-warning if compiler ever needs it.
}
