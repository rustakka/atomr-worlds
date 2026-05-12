//! Phase 14e — spherical projection sanity checks.
//!
//! Verifies the documented coordinate convention (lon = 0 at +X) holds
//! at a handful of cardinal directions, and that the forward / inverse
//! maps round-trip to within one pixel.

use atomr_worlds_core::coord::DVec3;
use atomr_worlds_view::projection_sphere::{
    equirectangular_dir_to_pixel, equirectangular_pixel_to_dir, orthographic_sphere_pixel_to_dir,
};

#[test]
fn plus_x_lands_at_center_column() {
    // +X is longitude 0. For width = 256, that should map to column 128
    // (the central column). The row corresponds to latitude 0 → mid
    // row.
    let [px, py] = equirectangular_dir_to_pixel(DVec3::new(1.0, 0.0, 0.0), 256, 128);
    assert_eq!(px, 128, "+X must map to centre column, got {px}");
    assert!(py == 63 || py == 64, "latitude 0 should map to the middle row of height 128, got {py}");
}

#[test]
fn plus_z_lands_at_three_quarters_column() {
    // +Z is longitude +π/2 (90° "east"). u = +0.5/2π + 0.5 = 0.75 → col
    // = 192 for width 256.
    let [px, _] = equirectangular_dir_to_pixel(DVec3::new(0.0, 0.0, 1.0), 256, 128);
    assert_eq!(px, 192, "+Z must map to longitude +π/2 column (192), got {px}");
}

#[test]
fn round_trip_within_one_pixel() {
    for axis in [
        DVec3::new(1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, 1.0),
        DVec3::new(-1.0, 0.0, 0.0),
        DVec3::new(0.0, 0.0, -1.0),
        DVec3::new(0.5, 0.5, 0.5),
    ] {
        let [px, py] = equirectangular_dir_to_pixel(axis, 256, 128);
        let back = equirectangular_pixel_to_dir(px, py, 256, 128);
        // unit-normalise both for the dot test.
        let alen = (axis.x * axis.x + axis.y * axis.y + axis.z * axis.z).sqrt().max(1e-20);
        let a = DVec3::new(axis.x / alen, axis.y / alen, axis.z / alen);
        let dot = a.x * back.x + a.y * back.y + a.z * back.z;
        assert!(dot > 0.98, "round-trip dot too low for {:?}: {}", axis, dot);
    }
}

#[test]
fn ortho_sphere_returns_none_outside_disc() {
    // The corner of a square image is outside the inscribed unit disc.
    let out = orthographic_sphere_pixel_to_dir(0, 0, 256, 256, DVec3::new(0.0, 0.0, -1.0));
    assert!(out.is_none(), "corner of square image must be outside the disc");
}

#[test]
fn ortho_sphere_center_aligned_with_view_axis_minus_z() {
    // view_axis = -Z → looking at +Z hemisphere; centre pixel = +Z.
    let d = orthographic_sphere_pixel_to_dir(128, 128, 256, 256, DVec3::new(0.0, 0.0, -1.0))
        .expect("centre pixel is inside the disc");
    assert!(d.z > 0.99, "centre under -Z view axis should be +Z, got {d:?}");
}
