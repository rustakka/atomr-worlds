//! `view-fp` — Phase 14a demo.
//!
//! Spins up a default-shape (cube) `LocalHost`, builds a `LocalHostQuery`
//! bridge, and walks a `WalkCamera` forward for five frames. Each frame
//! writes a PNG under `/tmp/view-fp-NN.png` and prints the framebuffer's
//! FNV-1a digest.

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_host::{LocalHost, LocalHostConfig, LocalHostQuery, WorldHost};
use atomr_worlds_view::{render_fp, RenderConfig, WalkCamera, WalkInput};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const FRAMES: u32 = 5;
const FRAME_W: u32 = 96;
const FRAME_H: u32 = 96;

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() {
    let host = Arc::new(
        LocalHost::new(LocalHostConfig { root_seed: SEED, ..LocalHostConfig::default() }).await.unwrap(),
    );
    let handle = tokio::runtime::Handle::current();
    let world: Arc<dyn atomr_worlds_view::WorldQuery> =
        Arc::new(LocalHostQuery::new(host.clone(), handle.clone()));
    let addr = WorldAddr::ROOT;

    // The render work calls `Handle::block_on` to fetch bricks — it must
    // run off the tokio worker thread so the runtime can drive the actor
    // request in parallel. `spawn_blocking` does exactly that.
    let result = tokio::task::spawn_blocking(move || {
        let start = DVec3::new(8.0, 20.0, 8.0);
        let mut cam = WalkCamera::new(start, ContainingFrame::World(addr), FRAME_W as f32 / FRAME_H as f32);
        cam.pitch = -0.5; // look slightly down so terrain bricks are in view
        let cfg = RenderConfig { width: FRAME_W, height: FRAME_H, ..Default::default() };
        let lod = Lod::new(0);
        let region_m = 48.0;
        println!("view-fp: rendering {FRAMES} frames");
        for frame in 0..FRAMES {
            cam.tick(
                WalkInput { move_local: [0.0, 0.0, 1.0], yaw_delta: 0.05, ..Default::default() },
                1.0 / 60.0,
            );
            let camera = cam.camera();
            let fb = render_fp(&*world, &addr, &camera, lod, region_m, &[], &cfg);
            let path = format!("/tmp/view-fp-{frame:02}.png");
            fb.write_png(&path).expect("write png");
            println!("  frame {frame:02}: digest={:#018x}", fb.pixels_fnv1a());
        }
    })
    .await;
    result.expect("render task");
    println!("view-fp: done");
    host.shutdown().await.unwrap();
}
