//! Phase 14d gate: the decal pass projects a world-XZ anchor to a
//! known pixel range, writes the decal color into those pixels, and
//! leaves all other pixels untouched.

use atomr_worlds_view::camera::Projection;
use atomr_worlds_view::{render_decals, Camera, Decal, Framebuffer};

fn topdown_ortho(w: u32, h: u32) -> Camera {
    // +Y eye, looking straight down at the origin: world (X, Z) maps
    // directly to screen (x, y) with a fixed pixel-per-meter scale.
    Camera {
        eye: [0.0, 50.0, 0.0],
        target: [0.0, 0.0, 0.0],
        up: [0.0, 0.0, -1.0],
        fov_y_rad: std::f32::consts::FRAC_PI_4,
        aspect: w as f32 / h as f32,
        near: 0.1,
        far: 200.0,
        // half_height_m = w/2 so 1 m = 1 px in the resulting image
        projection: Projection::Orthographic { half_height_m: (h as f32) / 2.0 },
    }
}

fn fb(w: u32, h: u32) -> Framebuffer {
    Framebuffer {
        width: w,
        height: h,
        pixels: vec![0u8; (w * h * 4) as usize],
        depth: vec![0.0f32; (w * h) as usize],
    }
}

#[test]
fn decal_projects_to_known_pixel_range() {
    let (w, h) = (16u32, 16u32);
    let mut f = fb(w, h);
    let cam = topdown_ortho(w, h);
    // 4×4 decal at world (0, 0) — projects to screen center (8, 8).
    // Center is at pixel 8 because (ndc 0 → (0+0.5)*16 = 8); the 4×4
    // rectangle is centered, so x ∈ [6, 10), y ∈ [6, 10).
    let color = [200, 50, 100, 255];
    let decal = Decal { world_xz_m: [0.0, 0.0], size_px: [4, 4], color, sprite: None };
    render_decals(&mut f, &cam, &[decal]);

    let stride = w as usize * 4;
    for y in 0..h as usize {
        for x in 0..w as usize {
            let pi = y * stride + x * 4;
            let p = [f.pixels[pi], f.pixels[pi + 1], f.pixels[pi + 2], f.pixels[pi + 3]];
            let in_rect = (6..10).contains(&x) && (6..10).contains(&y);
            if in_rect {
                assert_eq!(p, color, "pixel ({x}, {y}) should be decal color");
            } else {
                assert_eq!(p, [0, 0, 0, 0], "pixel ({x}, {y}) should be untouched");
            }
        }
    }
}

#[test]
fn off_center_decal_projects_off_center() {
    let (w, h) = (16u32, 16u32);
    let mut f = fb(w, h);
    let cam = topdown_ortho(w, h);
    // Decal at world (+4, -4). With `up = -Z` and the camera looking
    // straight down, view-Y = -world-Z, so the decal projects to
    // pixel-y = (1 - ((-(-4)/8)*0.5 + 0.5)) * 16 = (1 - 0.75)*16 = 4
    // and pixel-x = ((4/8)*0.5 + 0.5)*16 = 12. So the 4×4 decal
    // centered at (12, 4) covers x ∈ [10, 14), y ∈ [2, 6).
    let color = [10, 220, 40, 255];
    let decal = Decal { world_xz_m: [4.0, -4.0], size_px: [4, 4], color, sprite: None };
    render_decals(&mut f, &cam, &[decal]);

    for y in 2..6 {
        for x in 10..14 {
            let pi = (y * w as usize + x) * 4;
            let p = [f.pixels[pi], f.pixels[pi + 1], f.pixels[pi + 2], f.pixels[pi + 3]];
            assert_eq!(p, color, "decal pixel ({x}, {y})");
        }
    }
    // Pixel well outside the projected rect is untouched.
    let pi = (0 * w as usize + 0) * 4;
    assert_eq!(&f.pixels[pi..pi + 4], &[0, 0, 0, 0]);
}
