//! Spherical-projection helpers for Phase 14e (regional / world overview).
//!
//! Two projections live here. Both treat the world-space sphere as a
//! geocentric unit-direction sampler — a renderer needs `(pixel) → DVec3`
//! to ask `WorldMacroState::sample(dir)` what biome / elevation / climate
//! lives in that direction.
//!
//! # Coordinate conventions
//!
//! - Right-handed, +Y up — matches [`crate::camera`] and the world frame.
//! - **Longitude** `λ` is measured in the XZ plane, with `λ = 0` at +X and
//!   increasing toward +Z. Range `(-π, π]`.
//! - **Latitude** `φ` is measured from the XZ plane upward toward +Y.
//!   Range `[-π/2, π/2]`; `+π/2` is the +Y pole, `-π/2` is the -Y pole.
//! - The direction-from-angles map is therefore
//!   `dir = (cos φ cos λ, sin φ, cos φ sin λ)`.
//!
//! # Equirectangular (plate carrée) projection
//!
//! The image is a `width × height` rectangle covering the full sphere.
//! Column `px` ↔ longitude `λ = 2π * (px + 0.5) / width − π`. Row `py` ↔
//! latitude `φ = π/2 − π * (py + 0.5) / height`. The `+ 0.5` centres each
//! pixel inside its cell — without it, the leftmost column would map to
//! `λ = -π` exactly, which equals `λ = +π` on the wrap; pixel-centred
//! sampling makes the two image edges represent slightly different
//! longitudes and removes the duplicated meridian.
//!
//! The inverse `dir_to_pixel` uses `λ = atan2(z, x)`, `φ = asin(y / |dir|)`,
//! then clamps to `[0, width) × [0, height)` so callers don't need to
//! second-guess float boundary cases.
//!
//! # Orthographic-sphere projection
//!
//! "What the sphere looks like from infinity along `view_axis`" — the
//! visible hemisphere, no perspective. The image is `width × height`; the
//! sphere fits in a centred disk of radius `min(width, height) * 0.5`.
//!
//! Pixel `(px, py)` is converted to a view-space offset `(x_view, y_view)`
//! in unit-radius units (`-1..1` along the smaller of width/height). If
//! `x² + y² > 1` the pixel is *outside* the disk and the function returns
//! `None`. Otherwise `z_view = +sqrt(1 - x² - y²)` gives the visible
//! hemisphere's depth.
//!
//! The `(x_view, y_view, z_view)` triple is in a view-space basis where
//! `-Z_view` points toward the sphere centre (looking at the sphere). To
//! map back to world-space we build a right-handed orthonormal frame
//! `(right, up, fwd)` with `fwd = -view_axis` and `up` chosen so the +Y
//! world axis is "up on screen" when possible (degenerate when
//! `view_axis` is parallel to +Y — we fall back to +Z right).
//!
//! Then `dir_world = x_view * right + y_view * up + z_view * (-fwd)`.

use atomr_worlds_core::coord::DVec3;

/// Inverse map: image-space pixel → world-space unit direction under the
/// equirectangular (plate carrée) projection. See module-level docs for
/// the conventions and the +0.5 cell-centre offset.
///
/// The result is unit-length to within IEEE-754 precision; callers that
/// need it normalised for an outer dot-product should normalise
/// themselves.
#[inline]
pub fn equirectangular_pixel_to_dir(px: u32, py: u32, width: u32, height: u32) -> DVec3 {
    let w = width.max(1) as f64;
    let h = height.max(1) as f64;
    // Pixel-centred sampling: + 0.5 so the leftmost column doesn't
    // coincide with the wrap meridian.
    let u = (px as f64 + 0.5) / w;
    let v = (py as f64 + 0.5) / h;
    let lon = 2.0 * core::f64::consts::PI * (u - 0.5);
    let lat = core::f64::consts::PI * (0.5 - v);
    let cos_lat = lat.cos();
    DVec3::new(cos_lat * lon.cos(), lat.sin(), cos_lat * lon.sin())
}

/// Forward map: world-space direction → image-space pixel under the
/// equirectangular projection. Clamps to `[0, width) × [0, height)` so
/// callers can index a framebuffer directly.
///
/// Non-unit `dir` is permitted — we normalise internally for the
/// latitude term (so `asin` is well-defined). A zero vector returns
/// `[width / 2, height / 2]` (the centre pixel) as a deterministic
/// fallback.
#[inline]
pub fn equirectangular_dir_to_pixel(dir: DVec3, width: u32, height: u32) -> [u32; 2] {
    let len = (dir.x * dir.x + dir.y * dir.y + dir.z * dir.z).sqrt();
    if !(len > 0.0) {
        return [width / 2, height / 2];
    }
    let y_n = (dir.y / len).clamp(-1.0, 1.0);
    let lat = y_n.asin();
    let lon = dir.z.atan2(dir.x);
    let w = width.max(1) as f64;
    let h = height.max(1) as f64;
    let u = lon / (2.0 * core::f64::consts::PI) + 0.5;
    let v = 0.5 - lat / core::f64::consts::PI;
    let px = ((u * w).floor() as i64).clamp(0, width as i64 - 1) as u32;
    let py = ((v * h).floor() as i64).clamp(0, height as i64 - 1) as u32;
    [px, py]
}

/// Inverse map: image-space pixel → world-space unit direction under an
/// orthographic-from-infinity look along `view_axis`. Returns `None` for
/// pixels outside the unit disk (i.e. pixels that show the background
/// behind the sphere).
///
/// `view_axis` is the direction *from the observer toward the sphere
/// centre*. So `view_axis = -Z` (looking from +Z toward origin) shows the
/// +Z-facing hemisphere; `view_axis = +Y` (looking down from above) shows
/// the north pole. Internally we treat `view_axis` as unit-length and
/// normalise if it isn't.
///
/// Frame construction: with `fwd = normalise(view_axis)`,
/// - if `|fwd · +Y| < 1 − ε`: `right = normalise(fwd × +Y)`,
///   `up = right × fwd`. (Wikipedia "Look-at matrix" RH variant.)
/// - otherwise (looking along ±Y): pick `right = +X`, `up = ∓Z` so
///   `(right, up, -fwd)` stays right-handed.
///
/// The returned world direction is `x_view * right + y_view * up −
/// z_view * fwd`, which lies on the visible hemisphere
/// (`dir · (-fwd) > 0` always).
#[inline]
pub fn orthographic_sphere_pixel_to_dir(
    px: u32,
    py: u32,
    width: u32,
    height: u32,
    view_axis: DVec3,
) -> Option<DVec3> {
    let w = width.max(1) as f64;
    let h = height.max(1) as f64;
    // Disk inscribed in the smaller of width/height, centred at image
    // centre. Pixel-centred sampling matches the equirect variant.
    let cx = w * 0.5;
    let cy = h * 0.5;
    let r = w.min(h) * 0.5;
    let dx = (px as f64 + 0.5) - cx;
    // +y on screen is *down* in our top-left-origin framebuffer, so flip
    // the sign to get standard mathematical "up = positive y_view".
    let dy = cy - (py as f64 + 0.5);
    let x_view = dx / r;
    let y_view = dy / r;
    let r2 = x_view * x_view + y_view * y_view;
    if r2 > 1.0 {
        return None;
    }
    let z_view = (1.0 - r2).sqrt();

    // Build the (right, up, fwd) basis.
    let len = (view_axis.x * view_axis.x + view_axis.y * view_axis.y + view_axis.z * view_axis.z).sqrt();
    let fwd = if len > 0.0 {
        DVec3::new(view_axis.x / len, view_axis.y / len, view_axis.z / len)
    } else {
        DVec3::new(0.0, 0.0, -1.0)
    };
    let up_world = DVec3::new(0.0, 1.0, 0.0);
    let parallel = fwd.x * up_world.x + fwd.y * up_world.y + fwd.z * up_world.z;
    let (right, up) = if parallel.abs() < 1.0 - 1e-9 {
        // right = normalise(fwd × +Y)
        let rx = fwd.y * up_world.z - fwd.z * up_world.y;
        let ry = fwd.z * up_world.x - fwd.x * up_world.z;
        let rz = fwd.x * up_world.y - fwd.y * up_world.x;
        let rl = (rx * rx + ry * ry + rz * rz).sqrt().max(1e-20);
        let right = DVec3::new(rx / rl, ry / rl, rz / rl);
        // up = right × fwd
        let ux = right.y * fwd.z - right.z * fwd.y;
        let uy = right.z * fwd.x - right.x * fwd.z;
        let uz = right.x * fwd.y - right.y * fwd.x;
        (right, DVec3::new(ux, uy, uz))
    } else {
        // Looking ±Y: pick a deterministic right-handed fallback.
        let sign = parallel.signum();
        (DVec3::new(1.0, 0.0, 0.0), DVec3::new(0.0, 0.0, -sign))
    };

    // dir = x_view * right + y_view * up + z_view * (-fwd)
    let dir = DVec3::new(
        x_view * right.x + y_view * up.x - z_view * fwd.x,
        x_view * right.y + y_view * up.y - z_view * fwd.y,
        x_view * right.z + y_view * up.z - z_view * fwd.z,
    );
    Some(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn equirect_center_is_plus_x() {
        // The centre of the image (column ~width/2, middle row) maps to
        // longitude 0, latitude 0 → +X direction.
        let d = equirectangular_pixel_to_dir(128, 64, 256, 128);
        assert!(d.x > 0.99, "centre column should map near +X, got {:?}", d);
        assert!(d.y.abs() < 0.05);
        assert!(d.z.abs() < 0.05);
    }

    #[test]
    fn equirect_round_trip_axes() {
        for axis in [
            DVec3::new(1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, 1.0),
            DVec3::new(-1.0, 0.0, 0.0),
            DVec3::new(0.0, 0.0, -1.0),
        ] {
            let [px, py] = equirectangular_dir_to_pixel(axis, 256, 128);
            let back = equirectangular_pixel_to_dir(px, py, 256, 128);
            let dot = axis.x * back.x + axis.y * back.y + axis.z * back.z;
            assert!(dot > 0.99, "axis {:?} round-trip dot = {}", axis, dot);
        }
    }

    #[test]
    fn ortho_sphere_corner_is_outside() {
        // A corner of a square image is outside the unit disk (distance
        // sqrt(2)/2 ≈ 0.707 from centre in normalised units, but the disk
        // radius is exactly 0.5 of min(w,h), so corners exceed 1.0 in
        // x_view units only when the image is square and the pixel is at
        // a true corner — here we exercise pixel (0,0), which has
        // normalised coords ~(-1+1/W, +1-1/H), still inside if W large).
        // To force "outside", evaluate pixel exactly outside the disk:
        let out = orthographic_sphere_pixel_to_dir(0, 0, 4, 4, DVec3::new(0.0, 0.0, -1.0));
        // pixel (0,0) maps to x_view = -3/4 * 1/2 ... actually let me
        // just test that returning None happens at a guaranteed-outside
        // sample location: synthesise via a large image and far pixel.
        let _ = out; // may be Some or None depending on pixel; rely on the next assertion
                     // For a 2x2 image, pixel (0,0) maps to dx = -0.5, dy = +0.5,
                     // r2 = 1.0, exactly on the disk boundary → Some.
                     // For pixel just outside (negative is impossible with u32), use a
                     // larger image and probe the corner.
        let big = orthographic_sphere_pixel_to_dir(255, 255, 256, 256, DVec3::new(0.0, 0.0, -1.0));
        // Corner of a square image — outside the inscribed unit disk.
        assert!(big.is_none(), "corner of square image should fall outside the unit disk");
    }

    #[test]
    fn ortho_sphere_center_matches_view_axis() {
        // Looking from +Z toward origin (view_axis = -Z). The image
        // centre should look at the +Z hemisphere's apex, i.e. +Z.
        let view_axis = DVec3::new(0.0, 0.0, -1.0);
        let d = orthographic_sphere_pixel_to_dir(128, 128, 256, 256, view_axis).expect("centre is inside");
        assert!(d.z > 0.99, "centre under -Z view axis should be +Z, got {:?}", d);
    }
}
