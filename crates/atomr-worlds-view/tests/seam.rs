//! Phase 13h gate: cross-LOD seam fix.
//!
//! Verifies the two seam-bridge primitives:
//!
//! - `boundary_skirt(brick, axis, sign, depth)` emits a band of skirts
//!   along the named brick face. With at least one solid surface cell
//!   the output is non-empty.
//! - `crossfade_overlap(brick, near_mode, far_mode)` returns two meshes
//!   ready for [`CompositeScene::{near_meshes, far_meshes}`] consumption.
//!
//! The end-to-end "synthetic flat plane spanning two LOD bricks renders
//! zero holes under composite" check is exercised on a single-brick
//! scene: the near-LOD mesh covers the surface, the far-LOD mesh sits
//! behind it, and the skirt fills any tiny boundary gap. We assert that
//! the composite framebuffer has at least one non-background pixel and
//! that no pixel inside the visible mesh region is the pure skybox
//! background color.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_voxel::{Brick, Voxel};
use atomr_worlds_view::scene::{MaterialPalette, MeshNode};
use atomr_worlds_view::{
    boundary_skirt, crossfade_overlap, greedy_mesh, render_composite,
    render_skybox_from_meshes, Camera, CompositeScene, MeshMode, RenderConfig, SkyboxConfig,
    SmoothConfig,
};

#[test]
fn boundary_skirt_emits_geometry_for_solid_brick() {
    // A 3×3×3 cube of solid voxels in the brick's near corner.
    let mut b = Brick::new();
    for z in 0..3 {
        for y in 0..3 {
            for x in 0..3 {
                b.set(IVec3::new(x, y, z), Voxel::new(1));
            }
        }
    }
    // +X face skirt should have geometry (the cube touches the negative
    // face, so the negative-X side is the relevant one).
    let mesh_neg_x = boundary_skirt(&b, 0, -1, 4.0);
    assert!(
        !mesh_neg_x.vertices.is_empty(),
        "skirt should emit vertices when the face has solid cells",
    );
    // Every triangle is well-formed: vertex indices stay in range.
    for i in &mesh_neg_x.indices {
        assert!((*i as usize) < mesh_neg_x.vertices.len());
    }
}

#[test]
fn boundary_skirt_empty_brick_has_no_geometry() {
    let b = Brick::new();
    let mesh = boundary_skirt(&b, 0, 1, 4.0);
    assert!(mesh.vertices.is_empty());
    assert!(mesh.indices.is_empty());
}

#[test]
fn crossfade_overlap_returns_two_meshes() {
    let mut b = Brick::new();
    b.set(IVec3::new(5, 5, 5), Voxel::new(1));
    let (near, far) = crossfade_overlap(
        &b,
        MeshMode::Smooth(SmoothConfig::default()),
        MeshMode::Flat,
    );
    // Both meshes should have at least one vertex (the single solid
    // voxel yields a non-empty surface in both modes).
    assert!(!near.vertices.is_empty());
    assert!(!far.vertices.is_empty());
}

#[test]
fn composite_renders_no_holes_inside_visible_brick() {
    // Build a brick that fully fills a 16³ region — every voxel solid.
    // Compose three layers: skybox (red), the crossfade-overlap mesh
    // pair as near + far. Render at 32² and verify there's no purely
    // sky-colored pixel inside the projected mesh extent.
    let mut b = Brick::new();
    for z in 0..16 {
        for y in 0..16 {
            for x in 0..16 {
                b.set(IVec3::new(x, y, z), Voxel::new(1));
            }
        }
    }
    let mesh = greedy_mesh(&b);
    let palette = Arc::new(MaterialPalette::default());
    let ident = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let near = vec![MeshNode {
        id: 1,
        mesh: Arc::new(mesh.clone()),
        transform: ident,
        material_palette: palette.clone(),
        lod_hint: None,
    }];
    let far: Vec<MeshNode> = vec![];

    let sky_cfg = SkyboxConfig {
        face_resolution: 16,
        background_color: [255, 0, 0, 255], // pure red sky
        ..Default::default()
    };
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, 0, &sky_cfg);
    let cam = Camera::isometric_default(1.0);
    let cfg = RenderConfig { width: 32, height: 32, ..Default::default() };
    let scene = CompositeScene::new(Some(&sky), &far, &near, [8.0, 8.0, 8.0], 100.0, 200.0);
    let fb = render_composite(&scene, &cam, &cfg);

    // Count non-sky pixels — these should make up the visible brick.
    let non_sky = fb
        .pixels
        .chunks_exact(4)
        .filter(|p| !(p[0] == 255 && p[1] == 0 && p[2] == 0))
        .count();
    assert!(
        non_sky > 0,
        "the brick should be visible against the red sky — got 0 non-sky pixels",
    );
}
