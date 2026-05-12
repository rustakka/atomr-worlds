//! `view-tp` — Phase 14b demo.
//!
//! Orbits a `ChaseCamera` around a moving anchor for five frames. Same host
//! plumbing as `view-fp` but the camera trails the anchor at a fixed
//! distance/height and writes its PNGs to `/tmp/view-tp-NN.png`.

use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::DVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_host::{LocalHost, LocalHostConfig, LocalHostQuery, WorldHost};
use atomr_worlds_view::{render_tp, ChaseCamera, RenderConfig};

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

    let result = tokio::task::spawn_blocking(move || {
        let mut chase = ChaseCamera::new(DVec3::new(8.0, 20.0, 8.0), FRAME_W as f32 / FRAME_H as f32);
        let cfg = RenderConfig { width: FRAME_W, height: FRAME_H, ..Default::default() };
        let lod = Lod::new(0);
        let region_m = 48.0;
        println!("view-tp: rendering {FRAMES} frames");
        for frame in 0..FRAMES {
            let anchor = DVec3::new(8.0, 20.0, 8.0 + frame as f64 * 2.0);
            chase.tick(anchor, 0.05, 0.0, 1.0 / 60.0);
            let camera = chase.camera();
            let fb = render_tp(&*world, &addr, &camera, lod, region_m, &[], &cfg);
            let path = format!("/tmp/view-tp-{frame:02}.png");
            fb.write_png(&path).expect("write png");
            println!("  frame {frame:02}: digest={:#018x}", fb.pixels_fnv1a());
        }
    })
    .await;
    result.expect("render task");
    println!("view-tp: done");
    host.shutdown().await.unwrap();
}
