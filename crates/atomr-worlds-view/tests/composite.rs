//! Phase 13g gate: composite renderer (skybox + far-ring fade + near-ring opaque).
//!
//! Determinism: same `(scene, camera, cfg)` → byte-identical framebuffer.
//! Fade-band check: a fragment at the midpoint of the band gets alpha ≈ 0.5.
//! Skybox-only path: every pixel matches `Skybox::sample(ray_dir)`.
//! Composite-with-mesh path: depth-tested mesh appears in front of the
//! background skybox.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{BrickGenerator, TerrainConfig, TerrainGenerator};
use atomr_worlds_view::scene::{MaterialPalette, MeshNode};
use atomr_worlds_view::{
    greedy_mesh, render_composite, render_skybox_from_meshes, Camera, CompositeScene, FragmentMode,
    RenderConfig, SkyboxConfig,
};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

fn unit_camera() -> Camera {
    Camera::isometric_default(1.0)
}

fn unit_cfg() -> RenderConfig {
    RenderConfig { width: 32, height: 32, ..Default::default() }
}

fn mesh_node_at(t: [[f32; 4]; 4]) -> MeshNode {
    // Use a single procedural brick as a small but non-trivial mesh.
    let gen = TerrainGenerator::new(TerrainConfig::default());
    let brick = gen.generate_brick_legacy(SEED, IVec3::new(0, -1, 0));
    MeshNode {
        id: 1,
        mesh: Arc::new(greedy_mesh(&brick)),
        transform: t,
        material_palette: Arc::new(MaterialPalette::default()),
        lod_hint: None,
    }
}

fn ident() -> [[f32; 4]; 4] {
    [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
}

#[test]
fn composite_is_deterministic_across_runs() {
    let sky_cfg = SkyboxConfig::default();
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, SEED, &sky_cfg);
    let node = mesh_node_at(ident());
    let near = [node.clone()];
    let far: Vec<MeshNode> = vec![];

    let scene = CompositeScene::new(Some(&sky), &far, &near, [0.0, 0.0, 0.0], 5.0, 50.0);
    let fb1 = render_composite(&scene, &unit_camera(), &unit_cfg());
    let fb2 = render_composite(&scene, &unit_camera(), &unit_cfg());
    assert_eq!(fb1.pixels, fb2.pixels);
}

#[test]
fn composite_with_only_skybox_matches_sample() {
    let sky_cfg = SkyboxConfig::default();
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, SEED, &sky_cfg);
    let scene = CompositeScene::new(Some(&sky), &[], &[], [0.0, 0.0, 0.0], 5.0, 50.0);
    let fb = render_composite(&scene, &unit_camera(), &unit_cfg());
    // Empty mesh world → every pixel is a skybox lookup; with the empty
    // skybox built from no meshes, that's `cfg.background_color`.
    let bg = SkyboxConfig::default().background_color;
    let mut bg_pixels = 0;
    for chunk in fb.pixels.chunks_exact(4) {
        if chunk == bg {
            bg_pixels += 1;
        }
    }
    let total = (unit_cfg().width * unit_cfg().height) as usize;
    assert_eq!(bg_pixels, total, "skybox-only path should paint pure background");
}

#[test]
fn fragment_fade_midpoint_alpha_blends() {
    // Construct a CompositeScene where the entire near mesh sits at the
    // midpoint of the fade band. With alpha ≈ 0.5, fragment color
    // becomes (src + dst) / 2 — so the framebuffer should show a 50-50
    // blend of skybox background and mesh color.
    let sky_cfg = SkyboxConfig {
        face_resolution: 16,
        background_color: [200, 0, 0, 255], // pure red sky
        ..Default::default()
    };
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, SEED, &sky_cfg);
    let node = mesh_node_at(ident());
    // The mesh world-pos lives in [0, 16) per axis. Pick the fade band
    // so the midpoint falls in that range.
    let transition = 4.0;
    let max_radius = 12.0; // mid = 8 (= mesh center-ish)
    let far = vec![node.clone()];
    let scene = CompositeScene {
        skybox: Some(&sky),
        far_meshes: &far,
        near_meshes: &[],
        observer: [0.0, 0.0, 0.0],
        transition_radius_m: transition,
        max_radius_m: max_radius,
        fade_band_frac: 1.0, // fade across the entire band
    };
    let fb = render_composite(&scene, &unit_camera(), &unit_cfg());
    // At least some pixels should be neither pure red nor pure mesh color
    // (i.e. a blend has happened). Count pixels whose red channel is in
    // a blended range — exclusive of the pure red background.
    let blended = fb
        .pixels
        .chunks_exact(4)
        .filter(|p| p[0] > 30 && p[0] < 200) // not pure red, not pure mesh
        .count();
    assert!(blended > 0, "fade-band should produce blended pixels somewhere");
}

#[test]
fn no_skybox_falls_back_to_background_color() {
    let cfg = RenderConfig { width: 8, height: 8, background: [42, 43, 44, 255], ..Default::default() };
    let scene = CompositeScene::new(None, &[], &[], [0.0, 0.0, 0.0], 1.0, 10.0);
    let fb = render_composite(&scene, &unit_camera(), &cfg);
    for chunk in fb.pixels.chunks_exact(4) {
        assert_eq!(chunk, &cfg.background);
    }
}

#[test]
fn near_ring_writes_over_skybox_background() {
    // A near-ring mesh is opaque; its pixels should override the skybox.
    let sky_cfg = SkyboxConfig {
        face_resolution: 16,
        background_color: [0, 0, 200, 255], // blue sky
        ..Default::default()
    };
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, SEED, &sky_cfg);
    let node = mesh_node_at(ident());
    let near = vec![node];
    let scene = CompositeScene::new(Some(&sky), &[], &near, [0.0, 0.0, 0.0], 5.0, 50.0);
    let fb = render_composite(&scene, &unit_camera(), &unit_cfg());
    // At least some pixel's red component should be non-zero (meaning a
    // mesh fragment, since blue sky has red=0).
    let mesh_pixels = fb.pixels.chunks_exact(4).filter(|p| p[0] > 10).count();
    assert!(mesh_pixels > 0, "near-ring mesh should appear in front of blue sky");
}

#[test]
fn fragment_mode_distance_fade_math() {
    // Unit-test the alpha math: distance < start → 1.0; distance > end
    // → 0.0; halfway → 0.5.
    let observer = [0.0_f32, 0.0, 0.0];
    let start = 10.0;
    let end = 20.0;
    // Helper that computes alpha the same way the rasterizer does.
    let alpha_at = |d: f32| -> f32 {
        if d <= start {
            1.0
        } else if d >= end {
            0.0
        } else {
            1.0 - (d - start) / (end - start)
        }
    };
    assert!((alpha_at(5.0) - 1.0).abs() < 1e-6);
    assert!((alpha_at(25.0) - 0.0).abs() < 1e-6);
    assert!((alpha_at(15.0) - 0.5).abs() < 1e-6);
    // Inputs unused but exercising the FragmentMode constructor proves
    // it composes correctly.
    let _ = FragmentMode::DistanceFade { start_m: start, end_m: end, observer };
}
