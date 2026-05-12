//! Axis-aligned 2D rect blits into a [`Framebuffer`].
//!
//! This is a *pure 2D* tile-blitter — no projection, no depth interaction,
//! no triangle setup. It exists because phases 14c (Dwarf-Fortress slice),
//! 14d (RTS decals), and 14e (regional overview) push millions of unit
//! axis-aligned quads at fixed depth, and routing those through the 3D
//! triangle rasterizer in [`crate::render`] wastes roughly an order of
//! magnitude of work. The 3D pipeline stays untouched.
//!
//! All operations clip against the framebuffer bounds; negative `x`/`y`
//! and oversized `w`/`h` are valid inputs and are silently trimmed.
//!
//! Pixel byte layout matches [`Framebuffer::pixels`] — RGBA8, row-major,
//! top-left origin, byte index for `(x, y)` is
//! `((y as usize) * (width as usize) + (x as usize)) * 4`.

use crate::render::Framebuffer;

/// Stipple/dither pattern used by [`fill_rect_stipple`].
#[derive(Copy, Clone, Debug)]
pub enum StipplePattern {
    /// Alternating pixels — half the pixels written, in a 2×2 checker.
    Checker,
    /// Alternating rows — even rows (0, 2, 4, …) written, odd skipped.
    Horizontal,
    /// Alternating columns — even columns written, odd skipped.
    Vertical,
    /// 1 of 4 pixels (LSB rule: `(x + y) % 2 == 0 && (x * y) % 2 == 0`).
    Dense25,
    /// 3 of 4 pixels — the complement of [`StipplePattern::Dense25`].
    Dense75,
}

/// Clip a requested rect against `[0, fb.width) × [0, fb.height)`.
///
/// Returns `(x0, y0, x1, y1)` framebuffer-pixel bounds (half-open) such
/// that the rect to draw is `x0..x1` by `y0..y1`. Returns `None` if the
/// rect is empty or entirely outside the framebuffer.
#[inline]
fn clip_rect(fb: &Framebuffer, x: i32, y: i32, w: u32, h: u32) -> Option<(u32, u32, u32, u32)> {
    if w == 0 || h == 0 {
        return None;
    }
    let fb_w = fb.width as i64;
    let fb_h = fb.height as i64;
    let x0 = (x as i64).max(0);
    let y0 = (y as i64).max(0);
    let x1 = ((x as i64) + (w as i64)).min(fb_w);
    let y1 = ((y as i64) + (h as i64)).min(fb_h);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    Some((x0 as u32, y0 as u32, x1 as u32, y1 as u32))
}

/// Fill an axis-aligned rectangle in RGBA8. Clips to framebuffer bounds.
///
/// Negative origin and oversized extents are clipped; `w == 0` or `h == 0`
/// is a no-op. Each pixel becomes exactly `color` (alpha included) —
/// this is a *write*, not a blend; for blending use [`blend_rect`].
pub fn fill_rect(fb: &mut Framebuffer, x: i32, y: i32, w: u32, h: u32, color: [u8; 4]) {
    let Some((x0, y0, x1, y1)) = clip_rect(fb, x, y, w, h) else {
        return;
    };
    let stride = fb.width as usize;
    for yy in y0..y1 {
        let row = yy as usize * stride;
        for xx in x0..x1 {
            let pi = (row + xx as usize) * 4;
            fb.pixels[pi] = color[0];
            fb.pixels[pi + 1] = color[1];
            fb.pixels[pi + 2] = color[2];
            fb.pixels[pi + 3] = color[3];
        }
    }
}

/// Like [`fill_rect`] but only writes the pixels selected by `pattern`.
///
/// Used by 14c slice mode to indicate "thin features" (e.g. a column
/// thinner than the band) without dimming the colour. Pattern membership
/// is evaluated in *framebuffer* coordinates so adjacent rects align
/// against the same grid — this matters when two rects abut.
pub fn fill_rect_stipple(
    fb: &mut Framebuffer,
    x: i32,
    y: i32,
    w: u32,
    h: u32,
    color: [u8; 4],
    pattern: StipplePattern,
) {
    let Some((x0, y0, x1, y1)) = clip_rect(fb, x, y, w, h) else {
        return;
    };
    let stride = fb.width as usize;
    for yy in y0..y1 {
        let row = yy as usize * stride;
        for xx in x0..x1 {
            if !stipple_writes(pattern, xx, yy) {
                continue;
            }
            let pi = (row + xx as usize) * 4;
            fb.pixels[pi] = color[0];
            fb.pixels[pi + 1] = color[1];
            fb.pixels[pi + 2] = color[2];
            fb.pixels[pi + 3] = color[3];
        }
    }
}

#[inline]
fn stipple_writes(pattern: StipplePattern, x: u32, y: u32) -> bool {
    match pattern {
        StipplePattern::Checker => ((x ^ y) & 1) == 0,
        StipplePattern::Horizontal => (y & 1) == 0,
        StipplePattern::Vertical => (x & 1) == 0,
        StipplePattern::Dense25 => ((x + y) & 1) == 0 && (x.wrapping_mul(y) & 1) == 0,
        StipplePattern::Dense75 => !(((x + y) & 1) == 0 && (x.wrapping_mul(y) & 1) == 0),
    }
}

/// Alpha-blend a rectangle (`src over dst`) into the framebuffer.
///
/// `color.a` controls the blend: `0` is fully transparent (no-op per
/// pixel), `255` is fully opaque (equivalent to [`fill_rect`]). All
/// arithmetic is integer:
///
/// ```text
/// out.rgb = (src.rgb * a + dst.rgb * (255 - a) + 127) / 255
/// out.a   = max(src.a, dst.a)
/// ```
///
/// The `/255` step uses the standard `(x * 257 + 255) >> 16` reciprocal
/// — exact for the full `u8 × u8` input range and avoids a division.
pub fn blend_rect(fb: &mut Framebuffer, x: i32, y: i32, w: u32, h: u32, color: [u8; 4]) {
    let Some((x0, y0, x1, y1)) = clip_rect(fb, x, y, w, h) else {
        return;
    };
    let a = color[3] as u32;
    if a == 0 {
        return;
    }
    let inv_a = 255 - a;
    let stride = fb.width as usize;
    for yy in y0..y1 {
        let row = yy as usize * stride;
        for xx in x0..x1 {
            let pi = (row + xx as usize) * 4;
            let dr = fb.pixels[pi] as u32;
            let dg = fb.pixels[pi + 1] as u32;
            let db = fb.pixels[pi + 2] as u32;
            let da = fb.pixels[pi + 3] as u32;
            let sr = color[0] as u32;
            let sg = color[1] as u32;
            let sb = color[2] as u32;
            fb.pixels[pi] = div255(sr * a + dr * inv_a) as u8;
            fb.pixels[pi + 1] = div255(sg * a + dg * inv_a) as u8;
            fb.pixels[pi + 2] = div255(sb * a + db * inv_a) as u8;
            fb.pixels[pi + 3] = a.max(da) as u8;
        }
    }
}

/// Exact `x / 255` for `x` in `[0, 255 * 255]` (the only range we feed
/// it). The identity is `x / 255 == (x * 257 + 255) >> 16` — see e.g.
/// Hacker's Delight §10-14.
#[inline]
fn div255(x: u32) -> u32 {
    (x * 257 + 255) >> 16
}

/// Copy a source RGBA8 byte slice into the framebuffer at `(x, y)`,
/// clipping to framebuffer bounds.
///
/// `src` must be `src_w * src_h * 4` bytes long, top-left origin,
/// row-major (same layout as [`Framebuffer::pixels`]). Negative `x`/`y`
/// shift the copy origin into the source — i.e. clipped pixels are
/// silently skipped, not wrapped. This is a *write*, not a blend;
/// the source alpha channel is copied verbatim.
///
/// # Panics
///
/// Panics if `src.len() != src_w as usize * src_h as usize * 4`. A
/// malformed source slice is a programmer error, not a runtime
/// condition.
pub fn blit_rgba(fb: &mut Framebuffer, x: i32, y: i32, src: &[u8], src_w: u32, src_h: u32) {
    let expected = src_w as usize * src_h as usize * 4;
    assert_eq!(
        src.len(),
        expected,
        "blit_rgba: src.len()={} but src_w*src_h*4={} ({}x{})",
        src.len(),
        expected,
        src_w,
        src_h,
    );
    let Some((x0, y0, x1, y1)) = clip_rect(fb, x, y, src_w, src_h) else {
        return;
    };
    // How far into the source we start — when `x`/`y` were negative, the
    // first few source columns/rows are skipped.
    let src_x_off = (x0 as i64 - x as i64) as u32;
    let src_y_off = (y0 as i64 - y as i64) as u32;
    let stride_fb = fb.width as usize;
    let stride_src = src_w as usize;
    let copy_w = (x1 - x0) as usize;
    for yy in y0..y1 {
        let src_row = (yy - y0 + src_y_off) as usize;
        let src_base = (src_row * stride_src + src_x_off as usize) * 4;
        let dst_base = (yy as usize * stride_fb + x0 as usize) * 4;
        fb.pixels[dst_base..dst_base + copy_w * 4].copy_from_slice(&src[src_base..src_base + copy_w * 4]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fb(w: u32, h: u32) -> Framebuffer {
        Framebuffer {
            width: w,
            height: h,
            pixels: vec![0u8; (w * h * 4) as usize],
            depth: vec![0.0f32; (w * h) as usize],
        }
    }

    fn pixel(fb: &Framebuffer, x: u32, y: u32) -> [u8; 4] {
        let pi = ((y * fb.width + x) * 4) as usize;
        [fb.pixels[pi], fb.pixels[pi + 1], fb.pixels[pi + 2], fb.pixels[pi + 3]]
    }

    #[test]
    fn fill_rect_writes_exact_pixels() {
        let mut f = fb(4, 4);
        fill_rect(&mut f, 1, 1, 2, 2, [255, 0, 0, 255]);
        // Pixels (1,1), (2,1), (1,2), (2,2) → byte indices 20..24, 24..28,
        // 36..40, 40..44.
        for i in 20..28 {
            let expected = if i % 4 == 0 {
                255
            } else if i % 4 == 3 {
                255
            } else {
                0
            };
            assert_eq!(f.pixels[i], expected, "byte {i}");
        }
        for i in 36..44 {
            let expected = if i % 4 == 0 {
                255
            } else if i % 4 == 3 {
                255
            } else {
                0
            };
            assert_eq!(f.pixels[i], expected, "byte {i}");
        }
        // Everything else stays zero.
        for i in 0..f.pixels.len() {
            if (20..28).contains(&i) || (36..44).contains(&i) {
                continue;
            }
            assert_eq!(f.pixels[i], 0, "byte {i} should be untouched");
        }
    }

    #[test]
    fn fill_rect_clips_negative_origin() {
        let mut f = fb(2, 2);
        fill_rect(&mut f, -1, -1, 3, 3, [10, 20, 30, 255]);
        // The 3×3 rect at (-1,-1) intersects the framebuffer in (0..2, 0..2).
        assert_eq!(pixel(&f, 0, 0), [10, 20, 30, 255]);
        assert_eq!(pixel(&f, 1, 0), [10, 20, 30, 255]);
        assert_eq!(pixel(&f, 0, 1), [10, 20, 30, 255]);
        assert_eq!(pixel(&f, 1, 1), [10, 20, 30, 255]);
    }

    #[test]
    fn fill_rect_clips_overflow() {
        let mut f = fb(4, 4);
        fill_rect(&mut f, 1, 1, 10, 10, [9, 9, 9, 255]);
        for y in 0..4 {
            for x in 0..4 {
                let in_rect = (1..4).contains(&x) && (1..4).contains(&y);
                let expected = if in_rect { [9, 9, 9, 255] } else { [0, 0, 0, 0] };
                assert_eq!(pixel(&f, x, y), expected, "pixel ({x},{y})");
            }
        }
    }

    #[test]
    fn fill_rect_zero_size_noop() {
        let mut f = fb(4, 4);
        fill_rect(&mut f, 0, 0, 0, 5, [1, 2, 3, 255]);
        assert!(f.pixels.iter().all(|b| *b == 0));
        fill_rect(&mut f, 0, 0, 5, 0, [1, 2, 3, 255]);
        assert!(f.pixels.iter().all(|b| *b == 0));
    }

    #[test]
    fn stipple_checker_writes_half() {
        let mut f = fb(4, 4);
        fill_rect_stipple(&mut f, 0, 0, 4, 4, [255, 255, 255, 255], StipplePattern::Checker);
        let mut written = 0;
        for y in 0..4 {
            for x in 0..4 {
                if pixel(&f, x, y) == [255, 255, 255, 255] {
                    written += 1;
                }
            }
        }
        assert_eq!(written, 8, "checker should cover exactly half of 16 pixels");
    }

    #[test]
    fn stipple_horizontal_alternates_rows() {
        let mut f = fb(4, 4);
        fill_rect_stipple(&mut f, 0, 0, 4, 4, [200, 100, 50, 255], StipplePattern::Horizontal);
        for x in 0..4 {
            assert_eq!(pixel(&f, x, 0), [200, 100, 50, 255], "row 0 col {x} written");
            assert_eq!(pixel(&f, x, 1), [0, 0, 0, 0], "row 1 col {x} untouched");
            assert_eq!(pixel(&f, x, 2), [200, 100, 50, 255], "row 2 col {x} written");
            assert_eq!(pixel(&f, x, 3), [0, 0, 0, 0], "row 3 col {x} untouched");
        }
    }

    #[test]
    fn blend_rect_50pct() {
        let mut f = fb(2, 2);
        fill_rect(&mut f, 0, 0, 2, 2, [0, 0, 255, 255]); // blue dst
        blend_rect(&mut f, 0, 0, 2, 2, [255, 0, 0, 128]); // red @ 50%
        let p = pixel(&f, 0, 0);
        // 255 * 128 / 255 ≈ 128; dst.b: 255 * 127 / 255 ≈ 127.
        assert!((p[0] as i32 - 128).abs() <= 1, "r ≈ 128, got {}", p[0]);
        assert_eq!(p[1], 0);
        assert!((p[2] as i32 - 127).abs() <= 1, "b ≈ 127, got {}", p[2]);
        assert_eq!(p[3], 255);
    }

    #[test]
    fn blend_rect_fully_opaque() {
        let mut a = fb(3, 3);
        let mut b = fb(3, 3);
        fill_rect(&mut a, 0, 0, 3, 3, [42, 99, 200, 255]);
        blend_rect(&mut b, 0, 0, 3, 3, [42, 99, 200, 255]);
        assert_eq!(a.pixels, b.pixels);
    }

    #[test]
    fn blend_rect_fully_transparent() {
        let mut f = fb(2, 2);
        fill_rect(&mut f, 0, 0, 2, 2, [11, 22, 33, 200]);
        let before = f.pixels.clone();
        blend_rect(&mut f, 0, 0, 2, 2, [255, 255, 255, 0]);
        assert_eq!(f.pixels, before);
    }

    #[test]
    fn blit_rgba_round_trips() {
        let mut f = fb(2, 2);
        let src: [u8; 16] = [
            // (0,0) red, (1,0) green, (0,1) blue, (1,1) white.
            255, 0, 0, 255, //
            0, 255, 0, 255, //
            0, 0, 255, 255, //
            255, 255, 255, 255,
        ];
        blit_rgba(&mut f, 0, 0, &src, 2, 2);
        assert_eq!(f.pixels, src);
    }

    #[test]
    fn blit_rgba_clips_negative_origin() {
        // A 3×3 source blitted at (-1, -1) into a 2×2 fb should land
        // source (1..3, 1..3) → fb (0..2, 0..2).
        let mut f = fb(2, 2);
        let mut src = vec![0u8; 3 * 3 * 4];
        for yy in 0..3u32 {
            for xx in 0..3u32 {
                let pi = ((yy * 3 + xx) * 4) as usize;
                src[pi] = (xx * 100 + yy) as u8;
                src[pi + 1] = xx as u8;
                src[pi + 2] = yy as u8;
                src[pi + 3] = 255;
            }
        }
        blit_rgba(&mut f, -1, -1, &src, 3, 3);
        assert_eq!(pixel(&f, 0, 0), [101, 1, 1, 255]);
        assert_eq!(pixel(&f, 1, 0), [201, 2, 1, 255]);
        assert_eq!(pixel(&f, 0, 1), [102, 1, 2, 255]);
        assert_eq!(pixel(&f, 1, 1), [202, 2, 2, 255]);
    }

    #[test]
    #[should_panic(expected = "blit_rgba")]
    fn blit_rgba_panics_on_size_mismatch() {
        let mut f = fb(2, 2);
        let bad = vec![0u8; 7]; // 2*2*4 = 16 expected
        blit_rgba(&mut f, 0, 0, &bad, 2, 2);
    }
}
