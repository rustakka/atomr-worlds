//! Phase 13f gate tests: cube-face basis is well-formed, sampling is
//! scale-invariant, skybox rendering is deterministic, and the reversed-z
//! projection actually maps near→1 / far→0.
//!
//! The tests are intentionally self-contained: they build their own meshes,
//! drive `render_skybox_from_meshes` directly, and never touch `LocalHost` or
//! any wire format. That keeps Phase 13f testable without dragging in
//! `WorldHost` (which 13g/13i will wire on top).

use std::sync::Arc;

use atomr_worlds_view::camera::{transform_point, Camera};
use atomr_worlds_view::mesh::Mesh;
use atomr_worlds_view::{
    render_skybox_from_meshes, CubeFace, MaterialPalette, MeshNode, SkyboxConfig,
};

fn dot3(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross3(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}

fn len3(v: [f32; 3]) -> f32 {
    dot3(v, v).sqrt()
}

#[test]
fn cube_face_basis_is_orthonormal_right_handed() {
    for face in CubeFace::ALL {
        let f = face.forward();
        let u = face.up();
        let r = face.right();

        // Unit length.
        assert!((len3(f) - 1.0).abs() < 1e-6, "forward({:?}) not unit", face);
        assert!((len3(u) - 1.0).abs() < 1e-6, "up({:?}) not unit", face);
        assert!((len3(r) - 1.0).abs() < 1e-6, "right({:?}) not unit", face);

        // Orthogonal.
        assert!(dot3(f, u).abs() < 1e-6, "forward·up != 0 for {:?}", face);
        assert!(dot3(f, r).abs() < 1e-6, "forward·right != 0 for {:?}", face);
        assert!(dot3(u, r).abs() < 1e-6, "up·right != 0 for {:?}", face);

        // Right-handed: cross(right, up) == forward.
        let c = cross3(r, u);
        for k in 0..3 {
            assert!(
                (c[k] - f[k]).abs() < 1e-6,
                "cross(right, up) != forward for {:?}: got {:?}, want {:?}",
                face,
                c,
                f
            );
        }
    }
}

#[test]
fn sample_unit_x_lands_on_pos_x_face() {
    let cfg = SkyboxConfig::default();
    // Build a skybox, then paint the PosX face a distinctive color so we can
    // detect that `sample([1, 0, 0])` actually reads from it (and not, say,
    // PosZ — which would be the bug we're guarding against).
    let mut sky =
        render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, 0xDEADBEEF, &cfg);
    for chunk in sky.faces[CubeFace::PosX.index()].pixels.chunks_exact_mut(4) {
        chunk.copy_from_slice(&[10, 200, 30, 255]);
    }
    assert_eq!(sky.sample([1.0, 0.0, 0.0]), [10, 200, 30, 255]);
    // The face it sampled should also be exactly the PosX face's center texel.
    let posx = &sky.faces[CubeFace::PosX.index()];
    let cx = posx.width / 2;
    let cy = posx.height / 2;
    assert_eq!(posx.texel(cx, cy), [10, 200, 30, 255]);
}

#[test]
fn sample_is_scale_invariant() {
    let cfg = SkyboxConfig::default();
    let mut sky =
        render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, 0, &cfg);
    // Distinguish each face with a unique color.
    let palette: [[u8; 4]; 6] = [
        [200, 10, 10, 255],
        [10, 200, 10, 255],
        [10, 10, 200, 255],
        [200, 200, 10, 255],
        [10, 200, 200, 255],
        [200, 10, 200, 255],
    ];
    for face in CubeFace::ALL {
        let i = face.index();
        for chunk in sky.faces[i].pixels.chunks_exact_mut(4) {
            chunk.copy_from_slice(&palette[i]);
        }
    }
    for dir in [
        [1.0, 0.2, -0.1f32],
        [-0.3, 1.0, 0.2],
        [0.1, 0.05, 1.0],
        [-0.4, -0.1, -1.0],
        [-1.0, 0.0, 0.0],
        [0.0, -1.0, 0.0],
    ] {
        let a = sky.sample(dir);
        let scaled = [2.0 * dir[0], 2.0 * dir[1], 2.0 * dir[2]];
        let b = sky.sample(scaled);
        assert_eq!(a, b, "sample not scale-invariant for {:?}", dir);
    }
}

#[test]
fn empty_meshes_produce_uniform_background() {
    let cfg = SkyboxConfig { background_color: [42, 99, 7, 255], ..Default::default() };
    let sky = render_skybox_from_meshes(&[], [0.0, 0.0, 0.0], 1.0, 100.0, 0, &cfg);
    for face in &sky.faces {
        assert_eq!(face.width, cfg.face_resolution);
        assert_eq!(face.height, cfg.face_resolution);
        for chunk in face.pixels.chunks_exact(4) {
            assert_eq!(chunk, &cfg.background_color, "non-background pixel in empty skybox");
        }
    }
}

/// Build a tiny mesh with a quad placed along +X (so it shows up in the PosX
/// face).
fn mesh_with_pos_x_quad() -> MeshNode {
    use atomr_worlds_view::Vertex;
    let mut mesh = Mesh::default();
    // Quad at x = 5, spanning y/z in [-2, 2], normal facing -X (toward the camera).
    let n = [-1.0, 0.0, 0.0];
    mesh.vertices.push(Vertex { pos: [5.0, -2.0, -2.0], normal: n, material: 1 });
    mesh.vertices.push(Vertex { pos: [5.0, 2.0, -2.0], normal: n, material: 1 });
    mesh.vertices.push(Vertex { pos: [5.0, 2.0, 2.0], normal: n, material: 1 });
    mesh.vertices.push(Vertex { pos: [5.0, -2.0, 2.0], normal: n, material: 1 });
    mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
    MeshNode {
        id: 0,
        mesh: Arc::new(mesh),
        transform: [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        material_palette: Arc::new(MaterialPalette::default()),
        lod_hint: None,
    }
}

#[test]
fn skybox_digest_is_deterministic() {
    let cfg = SkyboxConfig::default();
    let node = mesh_with_pos_x_quad();
    let a = render_skybox_from_meshes(
        std::slice::from_ref(&node),
        [0.0, 0.0, 0.0],
        1.0,
        100.0,
        0xCAFE_F00D,
        &cfg,
    );
    let b = render_skybox_from_meshes(
        std::slice::from_ref(&node),
        [0.0, 0.0, 0.0],
        1.0,
        100.0,
        0xCAFE_F00D,
        &cfg,
    );
    assert_eq!(a.digest, b.digest, "skybox digest drifted across runs");
    assert_eq!(a.compute_digest(), a.digest, "compute_digest disagrees with stored digest");
    // The PosX face should contain non-background pixels (the quad is visible).
    let bg = cfg.background_color;
    let pos_x = &a.faces[CubeFace::PosX.index()];
    let non_bg = pos_x.pixels.chunks_exact(4).filter(|p| {
        !(p[0] == bg[0] && p[1] == bg[1] && p[2] == bg[2] && p[3] == bg[3])
    }).count();
    assert!(non_bg > 0, "PosX face should show the quad");
}

#[test]
fn skybox_digest_differs_for_different_observer() {
    let cfg = SkyboxConfig::default();
    let node = mesh_with_pos_x_quad();
    let a = render_skybox_from_meshes(
        std::slice::from_ref(&node),
        [0.0, 0.0, 0.0],
        1.0,
        100.0,
        0,
        &cfg,
    );
    let b = render_skybox_from_meshes(
        std::slice::from_ref(&node),
        [1.0, 0.0, 0.0],
        1.0,
        100.0,
        0,
        &cfg,
    );
    assert_ne!(a.digest, b.digest, "moving observer should change at least one pixel");
}

#[test]
fn reversed_z_maps_near_to_one_and_far_to_zero() {
    // Camera at origin looking down -Z (so a point at z = -near is at the near
    // plane and z = -far is at the far plane).
    let cam = Camera {
        eye: [0.0, 0.0, 0.0],
        target: [0.0, 0.0, -1.0],
        up: [0.0, 1.0, 0.0],
        fov_y_rad: std::f32::consts::FRAC_PI_4,
        aspect: 1.0,
        near: 0.5,
        far: 50.0,
    };
    let mvp = cam.view_proj();

    let near_clip = transform_point(mvp, [0.0, 0.0, -cam.near]);
    let near_depth = near_clip[2] / near_clip[3];
    assert!(
        (near_depth - 1.0).abs() < 1e-4,
        "near depth (reversed-z) should be ~1.0, got {near_depth}"
    );

    let far_clip = transform_point(mvp, [0.0, 0.0, -cam.far]);
    let far_depth = far_clip[2] / far_clip[3];
    assert!(
        far_depth.abs() < 1e-4,
        "far depth (reversed-z) should be ~0.0, got {far_depth}"
    );
}
