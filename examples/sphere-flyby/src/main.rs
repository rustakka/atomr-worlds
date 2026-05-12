//! `sphere-flyby` — Phase 13 demo.
//!
//! Configures a sphere world (Earth-class), registers a literal authored
//! "city" region near the equator, and flies an observer from a surface
//! grazing approach up to a near-orbital altitude. At each tick we
//! capture an iso-mesh of the brick under the observer and render a
//! composite frame (skybox + far-fade + near). Outputs N PNG frames
//! under `/tmp/sphere-flyby-XX.png` plus a final summary printed to
//! stdout.
//!
//! Run with `cargo run -p sphere-flyby`. The default frame count and
//! resolution keep CI-friendly wall time (< 5 s on a single core).

use std::collections::HashMap;
use std::sync::Arc;

use atomr_worlds_core::addr::{Address, Level, WorldAddr};
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::shape::WorldShape;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_generate::region_id;
use atomr_worlds_host::{
    LiteralRegion, LocalHost, LocalHostConfig, PrefixShape, RegionAabb, WorldHost,
};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_view::{
    greedy_mesh, render_composite, render_skybox_from_meshes, scene::MaterialPalette, Camera,
    CompositeScene, MeshNode, ObserverState, RenderConfig, SkyboxConfig,
};
use atomr_worlds_voxel::{Brick, Voxel};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;
const EARTH_R: f64 = 6.371e6;
const FRAMES: u32 = 12;
const FRAME_W: u32 = 96;
const FRAME_H: u32 = 96;

fn city_region() -> Arc<LiteralRegion> {
    let mut voxels = HashMap::new();
    for z in 0..4 {
        for y in 0..6 {
            for x in 0..4 {
                voxels.insert(IVec3::new(x, y, z), Voxel::new(0xBEEF));
            }
        }
    }
    let bounds = RegionAabb::new(IVec3::new(0, 0, 0), IVec3::new(4, 6, 4));
    Arc::new(LiteralRegion::new("city", bounds, voxels))
}

async fn fetch_brick(host: &LocalHost, addr: Address, bc: IVec3) -> Brick {
    let env = Envelope::new(
        0,
        addr,
        WorldRequest::GetBrick { addr, brick: bc, lod: atomr_worlds_core::lod::Lod::new(0) },
    );
    let resp = host.request(env).await.unwrap();
    let WorldEvent::BrickSnapshot { payload, .. } = resp.body else {
        panic!("expected BrickSnapshot");
    };
    Brick::from_bytes(&payload).unwrap()
}

#[tokio::main]
async fn main() {
    // Configure a sphere world Earth-class via PrefixShape.
    let addr_w = WorldAddr::ROOT;
    let mut shapes = PrefixShape::new();
    shapes.set(Level::World, addr_w, WorldShape::Sphere { radius_m: EARTH_R });

    let cfg = LocalHostConfig {
        root_seed: SEED,
        shape_resolver: Arc::new(shapes),
        ..LocalHostConfig::default()
    };
    let host = LocalHost::new(cfg).await.unwrap();
    host.register_authored_region(city_region());
    let addr = Address::World(addr_w);

    let _ = region_id; // confirms public re-export

    // Build a coarse skybox once at the starting pose. The transitive
    // refresh policy is exercised below, but for this demo we only do a
    // single capture — `ObserverState` drives the *when*, not the *how*.
    let sky_cfg = SkyboxConfig {
        face_resolution: 32,
        background_color: [16, 24, 50, 255],
        ..Default::default()
    };
    let initial_observer = DVec3::new(5.0e6 + EARTH_R - 1000.0, 5.0e6, 5.0e6);
    let skybox = render_skybox_from_meshes(
        &[],
        [initial_observer.x, initial_observer.y, initial_observer.z],
        100.0,
        100_000.0,
        SEED,
        &sky_cfg,
    );

    let mut state =
        ObserverState::new(initial_observer, ContainingFrame::World(addr_w));
    state.accept_next(skybox);

    let cam = Camera::isometric_default(FRAME_W as f32 / FRAME_H as f32);
    let render_cfg = RenderConfig {
        width: FRAME_W,
        height: FRAME_H,
        ..Default::default()
    };

    println!("sphere-flyby: simulating {} frames", FRAMES);
    let mut last_digest: u64 = 0;
    for frame in 0..FRAMES {
        // Move observer outward + along z over the flight. Altitude
        // increases linearly so we cover surface → 1000 km altitude.
        let t = frame as f64 / FRAMES as f64;
        let alt = 1000.0 + t * 1_000_000.0;
        let observer = DVec3::new(5.0e6 + EARTH_R + alt, 5.0e6, 5.0e6 + t * 500.0);
        state.tick(observer, None, 1.0);

        // Fetch one brick at the city site for the near ring.
        let brick = fetch_brick(&host, addr, IVec3::new(0, 0, 0)).await;
        let mesh = greedy_mesh(&brick);
        let node = MeshNode {
            id: 1,
            mesh: Arc::new(mesh),
            transform: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            material_palette: Arc::new(MaterialPalette::default()),
            lod_hint: None,
        };
        let near = vec![node];
        let far: Vec<MeshNode> = vec![];
        let sky_ref = state.last_skybox.as_ref();
        let scene = CompositeScene::new(
            sky_ref,
            &far,
            &near,
            [observer.x as f32, observer.y as f32, observer.z as f32],
            100.0,
            100_000.0,
        );
        let fb = render_composite(&scene, &cam, &render_cfg);
        let path = format!("/tmp/sphere-flyby-{:02}.png", frame);
        fb.write_png(&path).expect("write png");
        let d = fb.pixels_fnv1a();
        println!("  frame {:02}: alt={:.0}m digest={:#018x} (delta from prev: {})",
            frame, alt, d, if d != last_digest { "yes" } else { "no" });
        last_digest = d;
    }
    println!("sphere-flyby: done — {} frames written to /tmp/sphere-flyby-*.png", FRAMES);
    host.shutdown().await.unwrap();
}
