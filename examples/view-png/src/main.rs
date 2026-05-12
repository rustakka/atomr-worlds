//! Phase 2 scaffold: ask `LocalHost` for a vertical slab of bricks and write a
//! PNG showing the surface profile (top-down "depth-of-first-solid" view).
//!
//! Headless; no display server required. Useful as a CI smoke test.

use std::fs::File;
use std::io::BufWriter;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

const TILES_X: i64 = 8;
const TILES_Z: i64 = 8;
const Y_TILES_TOP: i64 = 6;
const Y_TILES_BOT: i64 = 0;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = LocalHost::with_seed(0xDEAD_BEEF_CAFE_F00D).await?;
    let addr = WorldAddr::ROOT;
    let edge = BRICK_EDGE as i64;

    let width = (TILES_X * edge) as u32;
    let height = (TILES_Z * edge) as u32;
    let mut pixels = vec![0u8; (width * height * 3) as usize];

    // Walk each column (x, z); find topmost non-empty voxel within the y-tile range.
    for tz in 0..TILES_Z {
        for tx in 0..TILES_X {
            // Pull bricks at (tx, ty, tz) for each ty in the vertical range.
            for ty in (Y_TILES_BOT..=Y_TILES_TOP).rev() {
                let req = WorldRequest::GetBrick {
                    addr,
                    brick: IVec3::new(tx, ty, tz),
                    lod: Lod::new(0),
                };
                let env = Envelope::new(0, addr, req);
                let resp = host.request(env).await?;
                let WorldEvent::BrickSnapshot { payload, .. } = resp.body else { continue };
                let brick = Brick::from_bytes(&payload)?;
                if brick.is_empty() {
                    continue;
                }
                // Project: for each (lx, lz), find the topmost non-empty ly.
                for lz in 0..edge {
                    for lx in 0..edge {
                        let mut h_local = None;
                        for ly in (0..edge).rev() {
                            if !brick.get(IVec3::new(lx, ly, lz)).is_empty() {
                                h_local = Some(ly);
                                break;
                            }
                        }
                        if let Some(_ly) = h_local {
                            let px = (tx * edge + lx) as u32;
                            let py = (tz * edge + lz) as u32;
                            let i = ((py * width + px) * 3) as usize;
                            if pixels[i] == 0 && pixels[i + 1] == 0 && pixels[i + 2] == 0 {
                                // Colour by height tile.
                                let intensity = ((ty as f32) / (Y_TILES_TOP as f32 + 1.0)).clamp(0.0, 1.0);
                                pixels[i] = (60.0 + 195.0 * intensity) as u8;
                                pixels[i + 1] = (90.0 + 130.0 * (1.0 - intensity)) as u8;
                                pixels[i + 2] = 70;
                            }
                        }
                    }
                }
            }
        }
    }

    // Write PNG.
    let out = File::create("view-png-output.png")?;
    let mut w = BufWriter::new(out);
    let mut enc = png::Encoder::new(&mut w, width, height);
    enc.set_color(png::ColorType::Rgb);
    enc.set_depth(png::BitDepth::Eight);
    let mut writer = enc.write_header()?;
    writer.write_image_data(&pixels)?;

    println!("wrote view-png-output.png  {}x{}", width, height);
    host.shutdown().await?;
    Ok(())
}
