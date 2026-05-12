//! Phase 14a determinism gate: a scripted walk sequence against an inline
//! stub `WorldQuery` produces byte-identical framebuffers between two
//! identical runs.

use std::collections::HashMap;
use std::sync::mpsc;
use std::sync::Arc;

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::{DVec3, IVec3};
use atomr_worlds_core::lod::Lod;
use atomr_worlds_core::vehicle::ContainingFrame;
use atomr_worlds_proto::{WorldEvent, AABB};
use atomr_worlds_view::{render_fp, RenderConfig, WalkCamera, WalkInput, WorldQuery};
use atomr_worlds_voxel::brick::Brick;
use atomr_worlds_voxel::voxel::Voxel;
use atomr_worlds_voxel::BRICK_EDGE;

/// Returns the same canned brick for every (`addr`, `coord`, `lod`) tuple
/// inside `(-1..=1, -1..=1, -1..=1)`. Other coords return `None` so we
/// exercise the "missing brick" skip path too.
struct StubWorld {
    brick: Arc<Brick>,
}

impl StubWorld {
    fn new() -> Self {
        // Half-fill the brick: y < BRICK_EDGE/2 is stone. Produces a
        // visible flat plane.
        let mut b = Brick::new();
        let edge = BRICK_EDGE as i64;
        for z in 0..edge {
            for y in 0..edge / 2 {
                for x in 0..edge {
                    b.set(IVec3::new(x, y, z), Voxel::new(1));
                }
            }
        }
        Self { brick: Arc::new(b) }
    }
}

impl WorldQuery for StubWorld {
    fn brick(&self, _addr: &WorldAddr, c: IVec3, _lod: Lod) -> Option<Arc<Brick>> {
        if c.x.abs() <= 1 && c.y.abs() <= 1 && c.z.abs() <= 1 {
            Some(self.brick.clone())
        } else {
            None
        }
    }

    fn ground_height_m(&self, _addr: &WorldAddr, _xz: [f64; 2]) -> Option<f32> {
        Some((BRICK_EDGE / 2) as f32)
    }

    fn subscribe_region(&self, _addr: &WorldAddr, _r: AABB, _lod: Lod) -> mpsc::Receiver<WorldEvent> {
        let (_tx, rx) = mpsc::channel();
        rx
    }
}

fn run_scripted_walk() -> Vec<u64> {
    let world = StubWorld::new();
    let addr = WorldAddr::ROOT;
    let start = DVec3::new(8.0, 12.0, 0.0);
    let mut cam = WalkCamera::new(start, ContainingFrame::World(addr), 1.0);
    let cfg = RenderConfig { width: 48, height: 48, ..Default::default() };
    let _ = HashMap::<u32, u32>::new(); // silence unused-import lint

    let inputs = [
        WalkInput { move_local: [0.0, 0.0, 1.0], yaw_delta: 0.0, pitch_delta: 0.0, crouch: false },
        WalkInput { move_local: [0.5, 0.0, 0.5], yaw_delta: 0.1, pitch_delta: 0.0, crouch: false },
        WalkInput { move_local: [0.0, 0.0, 1.0], yaw_delta: 0.0, pitch_delta: -0.1, crouch: false },
        WalkInput { move_local: [-0.3, 0.0, 0.7], yaw_delta: -0.2, pitch_delta: 0.0, crouch: true },
        WalkInput { move_local: [0.0, 0.0, 1.5], yaw_delta: 0.0, pitch_delta: 0.0, crouch: false },
    ];
    let mut digests = Vec::with_capacity(inputs.len());
    for input in inputs {
        cam.tick(input, 1.0 / 60.0);
        let camera = cam.camera();
        let fb = render_fp(&world, &addr, &camera, Lod::new(0), 32.0, &[], &cfg);
        digests.push(fb.pixels_fnv1a());
    }
    digests
}

#[test]
fn scripted_walk_is_deterministic() {
    let a = run_scripted_walk();
    let b = run_scripted_walk();
    assert_eq!(a, b, "two identical walk runs must produce identical framebuffers");
    // Not all-zeros — would mean the stub never produced visible geometry.
    assert!(a.iter().any(|&d| d != 0), "walk should render something visible");
}
