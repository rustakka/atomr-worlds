//! Determinism gate for the GPU DAG raymarcher.
//!
//! [`render_raymarch`] is the deterministic CPU twin of the client's WGSL
//! `@fragment` DAG raymarcher: it marches a brick's flattened [`DagGpu`] per
//! pixel via [`atomr_worlds_voxel::ray_dda_first_hit`] and shades the first
//! solid voxel. The GPU's own float output is driver-divergent and hash-exempt;
//! this CPU path is bit-exact, so it is what the golden pins.
//!
//! Pattern mirrors `deterministic_screenshot.rs` — we compare a 64-bit FNV-1a
//! hash of the RGBA pixel data rather than the PNG bytes (PNG headers /
//! compression drift). Bump the pinned hash only with a documented reason; an
//! unexpected drift is the signal this test is supposed to catch.
//!
//! The golden pins [`RaymarchTier::Unlit`] (flat `material_color`, no lighting
//! term) so the hash is robust to float drift in any shading math.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_view::{render_raymarch, Camera, RaymarchTier, RenderConfig};
use atomr_worlds_voxel::{Brick, DagBrick, Voxel, BRICK_EDGE};

/// Pinned FNV-1a hash for the known half-filled brick + camera + render config,
/// rendered with [`RaymarchTier::Unlit`].
///
/// Bump this and document the reason whenever the raymarch render or its inputs
/// intentionally change; an unexpected drift is what this test catches.
const PINNED_HASH: u64 = 0x05ef_6114_1cc2_e9b5;

/// Material id of the solid lower half of the fixture brick (`3` = sand).
const FILL_MATERIAL: u16 = 3;

/// Build a half-filled brick: the lower half (`y < BRICK_EDGE/2`) is solid
/// `FILL_MATERIAL`, the upper half is air. Inlined here — the repo inlines test
/// fixtures rather than sharing them.
fn half_filled_brick() -> Brick {
    let mut b = Brick::new();
    let edge = BRICK_EDGE as i64;
    for z in 0..edge {
        for y in 0..(edge / 2) {
            for x in 0..edge {
                b.set(IVec3::new(x, y, z), Voxel::new(FILL_MATERIAL));
            }
        }
    }
    b
}

fn render_golden() -> u64 {
    let brick = half_filled_brick();
    let dag = DagBrick::from_brick(&brick).to_gpu();
    let cam = Camera::isometric_default(1.0);
    let cfg = RenderConfig { width: 64, height: 64, ..Default::default() };
    let fb = render_raymarch(&dag, &cam, &cfg, RaymarchTier::Unlit);
    fb.pixels_fnv1a()
}

#[test]
fn raymarch_renders_deterministically_across_runs() {
    let h1 = render_golden();
    let h2 = render_golden();
    assert_eq!(h1, h2, "raymarch render must be deterministic across runs");
}

#[test]
fn raymarch_golden_pinned_hash_matches() {
    let h = render_golden();
    assert_eq!(
        h, PINNED_HASH,
        "raymarch render hash drifted: got {h:#018x}, expected {PINNED_HASH:#018x}"
    );
}

#[test]
fn raymarch_produces_nonbackground_pixels() {
    let brick = half_filled_brick();
    let dag = DagBrick::from_brick(&brick).to_gpu();
    let cam = Camera::isometric_default(1.0);
    let cfg = RenderConfig { width: 64, height: 64, ..Default::default() };
    let fb = render_raymarch(&dag, &cam, &cfg, RaymarchTier::Unlit);

    let bg = cfg.background;
    let non_bg = fb
        .pixels
        .chunks_exact(4)
        .filter(|p| !(p[0] == bg[0] && p[1] == bg[1] && p[2] == bg[2] && p[3] == bg[3]))
        .count();
    assert!(non_bg > 0, "the camera should see the brick: expected non-background pixels");
}
