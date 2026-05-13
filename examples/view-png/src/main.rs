//! Phase 2 example: pull a slab of bricks from `LocalHost`, mesh them with
//! greedy meshing, and render an isometric perspective view to PNG via the
//! `atomr-worlds-view` software rasterizer.
//!
//! Headless — no display server required.

use atomr_worlds_core::addr::{Address, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_view::mesh::{greedy_mesh, Mesh, Vertex};
use atomr_worlds_view::{render_mesh, Camera, Projection, RenderConfig};
use atomr_worlds_voxel::{Brick, BRICK_EDGE};

const TILES_X: i64 = 4;
const TILES_Z: i64 = 4;
const Y_TILES_TOP: i64 = 3;
const Y_TILES_BOT: i64 = -2;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let host = LocalHost::with_seed(0xDEAD_BEEF_CAFE_F00D).await?;
    let addr = Address::World(WorldAddr::ROOT);
    let edge = BRICK_EDGE as f32;

    // Pull bricks across the slab, greedily mesh each, and stitch into one
    // big mesh in world-local coordinates (brick (bx, by, bz) origin at
    // (bx*16, by*16, bz*16)).
    let mut combined = Mesh::default();
    for ty in Y_TILES_BOT..=Y_TILES_TOP {
        for tz in 0..TILES_Z {
            for tx in 0..TILES_X {
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
                let mesh = greedy_mesh(&brick);
                let origin = [tx as f32 * edge, ty as f32 * edge, tz as f32 * edge];
                let base = combined.vertices.len() as u32;
                for v in &mesh.vertices {
                    combined.vertices.push(Vertex {
                        pos: [v.pos[0] + origin[0], v.pos[1] + origin[1], v.pos[2] + origin[2]],
                        normal: v.normal,
                        material: v.material,
                        ao: v.ao,
                    });
                }
                combined.indices.extend(mesh.indices.iter().map(|i| i + base));
            }
        }
    }

    let half = (TILES_X as f32 * edge) * 0.5;
    let camera = Camera {
        eye: [TILES_X as f32 * edge * 1.5, Y_TILES_TOP as f32 * edge * 2.0 + 24.0, TILES_Z as f32 * edge * 1.5],
        target: [half, 0.0, half],
        up: [0.0, 1.0, 0.0],
        fov_y_rad: std::f32::consts::FRAC_PI_4,
        aspect: 1.0,
        near: 0.5,
        far: 1024.0,
        projection: Projection::Perspective { fov_y_rad: std::f32::consts::FRAC_PI_4 },
    };
    let cfg = RenderConfig { width: 512, height: 512, ..Default::default() };
    let fb = render_mesh(&combined, &camera, &cfg);
    fb.write_png("view-png-output.png")?;

    println!(
        "wrote view-png-output.png  {}x{}  ({} tris)",
        cfg.width,
        cfg.height,
        combined.triangle_count()
    );
    host.shutdown().await?;
    Ok(())
}
