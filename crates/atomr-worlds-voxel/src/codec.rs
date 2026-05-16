//! Pluggable per-brick byte codecs.
//!
//! Encoders are paper-driven additive strategies: `RawU16` is the byte-equal
//! Vanilla default, the others (`Rle`, `Zlib`, `PaletteRle`) trade CPU for
//! smaller payloads.

use std::io::{Read, Write};

use flate2::Compression;
use flate2::read::ZlibDecoder;
use flate2::write::ZlibEncoder;

use crate::brick::{Brick, BRICK_LEN};
use crate::voxel::Voxel;

#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    #[error("malformed brick payload: {0}")]
    Malformed(String),
    #[error("io: {0}")]
    Io(String),
}

pub trait BrickCodec: Send + Sync + std::fmt::Debug {
    fn id(&self) -> &'static str;
    fn encode(&self, brick: &Brick) -> Vec<u8>;
    fn decode(&self, bytes: &[u8]) -> Result<Brick, CodecError>;
}

fn brick_from_voxels(voxels: Box<[Voxel; BRICK_LEN]>) -> Brick {
    let mut count = 0u16;
    for v in voxels.iter() {
        if !v.is_empty() {
            count = count.saturating_add(1);
        }
    }
    Brick { voxels, nonempty_count: count, light_overlay: None }
}

fn boxed_empty_voxels() -> Box<[Voxel; BRICK_LEN]> {
    Box::new([Voxel::EMPTY; BRICK_LEN])
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RawU16;

impl BrickCodec for RawU16 {
    fn id(&self) -> &'static str {
        "raw-u16"
    }

    fn encode(&self, brick: &Brick) -> Vec<u8> {
        bytemuck::cast_slice::<Voxel, u8>(brick.voxels.as_ref()).to_vec()
    }

    fn decode(&self, bytes: &[u8]) -> Result<Brick, CodecError> {
        if bytes.len() != BRICK_LEN * 2 {
            return Err(CodecError::Malformed(format!(
                "raw-u16 expects {} bytes, got {}",
                BRICK_LEN * 2,
                bytes.len()
            )));
        }
        // Decode element-wise so callers don't have to guarantee u16
        // alignment on the input slice.
        let mut arr = boxed_empty_voxels();
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            arr[i] = Voxel(u16::from_le_bytes([chunk[0], chunk[1]]));
        }
        Ok(brick_from_voxels(arr))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Rle;

impl BrickCodec for Rle {
    fn id(&self) -> &'static str {
        "rle-u16"
    }

    fn encode(&self, brick: &Brick) -> Vec<u8> {
        // (value: u16-le, count: u16-le) pairs. Runs are chunked at u16::MAX
        // because each count is u16.
        let mut out = Vec::with_capacity(8);
        let voxels = brick.voxels.as_ref();
        let mut i = 0usize;
        while i < BRICK_LEN {
            let v = voxels[i].0;
            let mut j = i + 1;
            while j < BRICK_LEN && voxels[j].0 == v && (j - i) < u16::MAX as usize {
                j += 1;
            }
            out.extend_from_slice(&v.to_le_bytes());
            out.extend_from_slice(&((j - i) as u16).to_le_bytes());
            i = j;
        }
        out
    }

    fn decode(&self, bytes: &[u8]) -> Result<Brick, CodecError> {
        if bytes.len() % 4 != 0 {
            return Err(CodecError::Malformed(format!(
                "rle payload not a multiple of 4 bytes: {}",
                bytes.len()
            )));
        }
        let mut arr = boxed_empty_voxels();
        let mut written = 0usize;
        let mut cursor = 0usize;
        while cursor < bytes.len() {
            let v = u16::from_le_bytes([bytes[cursor], bytes[cursor + 1]]);
            let cnt = u16::from_le_bytes([bytes[cursor + 2], bytes[cursor + 3]]) as usize;
            cursor += 4;
            if written + cnt > BRICK_LEN {
                return Err(CodecError::Malformed(format!(
                    "rle overflow: {} + {} > {}",
                    written, cnt, BRICK_LEN
                )));
            }
            for slot in arr[written..written + cnt].iter_mut() {
                *slot = Voxel(v);
            }
            written += cnt;
        }
        if written != BRICK_LEN {
            return Err(CodecError::Malformed(format!(
                "rle underflow: wrote {} voxels, expected {}",
                written, BRICK_LEN
            )));
        }
        Ok(brick_from_voxels(arr))
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct Zlib;

impl BrickCodec for Zlib {
    fn id(&self) -> &'static str {
        "zlib-raw-u16"
    }

    fn encode(&self, brick: &Brick) -> Vec<u8> {
        let raw = bytemuck::cast_slice::<Voxel, u8>(brick.voxels.as_ref());
        let mut enc = ZlibEncoder::new(Vec::with_capacity(raw.len() / 4), Compression::default());
        enc.write_all(raw).expect("zlib encoder write to Vec<u8> cannot fail");
        enc.finish().expect("zlib encoder finish to Vec<u8> cannot fail")
    }

    fn decode(&self, bytes: &[u8]) -> Result<Brick, CodecError> {
        let mut dec = ZlibDecoder::new(bytes);
        let mut raw = Vec::with_capacity(BRICK_LEN * 2);
        dec.read_to_end(&mut raw).map_err(|e| CodecError::Io(e.to_string()))?;
        RawU16.decode(&raw)
    }
}

const PALETTE_FLAG_PALETTE: u8 = 0;
const PALETTE_FLAG_RAW: u8 = 1;
const PALETTE_MAX: usize = 256;

#[derive(Debug, Default, Clone, Copy)]
pub struct PaletteRle;

impl BrickCodec for PaletteRle {
    fn id(&self) -> &'static str {
        "palette-rle"
    }

    fn encode(&self, brick: &Brick) -> Vec<u8> {
        let voxels = brick.voxels.as_ref();
        let mut palette: Vec<u16> = Vec::with_capacity(PALETTE_MAX);
        let mut indices = [0u8; BRICK_LEN];
        let mut overflow = false;

        for (slot, voxel) in indices.iter_mut().zip(voxels.iter()) {
            let v = voxel.0;
            let idx = match palette.iter().position(|p| *p == v) {
                Some(i) => i,
                None => {
                    if palette.len() >= PALETTE_MAX {
                        overflow = true;
                        break;
                    }
                    palette.push(v);
                    palette.len() - 1
                }
            };
            *slot = idx as u8;
        }

        if overflow {
            // Sentinel + raw passthrough preserves byte-exact recovery.
            let mut out = Vec::with_capacity(1 + BRICK_LEN * 2);
            out.push(PALETTE_FLAG_RAW);
            out.extend_from_slice(bytemuck::cast_slice::<Voxel, u8>(voxels));
            return out;
        }

        let mut out = Vec::with_capacity(1 + 1 + palette.len() * 2 + BRICK_LEN);
        out.push(PALETTE_FLAG_PALETTE);
        // palette length encoded as len-1 (so 256 entries fit in one byte; 0
        // means a single-entry palette).
        out.push((palette.len() - 1) as u8);
        for v in &palette {
            out.extend_from_slice(&v.to_le_bytes());
        }

        let mut i = 0usize;
        while i < BRICK_LEN {
            let v = indices[i];
            let mut j = i + 1;
            while j < BRICK_LEN && indices[j] == v && (j - i) < u16::MAX as usize {
                j += 1;
            }
            out.push(v);
            out.extend_from_slice(&((j - i) as u16).to_le_bytes());
            i = j;
        }
        out
    }

    fn decode(&self, bytes: &[u8]) -> Result<Brick, CodecError> {
        let flag = *bytes
            .first()
            .ok_or_else(|| CodecError::Malformed("empty palette-rle payload".into()))?;
        match flag {
            PALETTE_FLAG_RAW => {
                if bytes.len() != 1 + BRICK_LEN * 2 {
                    return Err(CodecError::Malformed(format!(
                        "palette-rle raw expects {} bytes, got {}",
                        1 + BRICK_LEN * 2,
                        bytes.len()
                    )));
                }
                // The 1-byte sentinel may have left the raw payload misaligned
                // for u16 casts, so decode element-wise instead of via
                // `RawU16.decode`.
                let mut arr = boxed_empty_voxels();
                for (i, chunk) in bytes[1..].chunks_exact(2).enumerate() {
                    arr[i] = Voxel(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                Ok(brick_from_voxels(arr))
            }
            PALETTE_FLAG_PALETTE => {
                if bytes.len() < 2 {
                    return Err(CodecError::Malformed(
                        "palette-rle missing palette header".into(),
                    ));
                }
                let palette_len = bytes[1] as usize + 1;
                let palette_start = 2usize;
                let palette_end = palette_start + palette_len * 2;
                if bytes.len() < palette_end {
                    return Err(CodecError::Malformed(format!(
                        "palette-rle truncated palette (expected {} entries)",
                        palette_len
                    )));
                }
                let mut palette = Vec::with_capacity(palette_len);
                for chunk in bytes[palette_start..palette_end].chunks_exact(2) {
                    palette.push(u16::from_le_bytes([chunk[0], chunk[1]]));
                }
                let mut cursor = palette_end;
                let mut arr = boxed_empty_voxels();
                let mut written = 0usize;
                while cursor < bytes.len() {
                    if cursor + 3 > bytes.len() {
                        return Err(CodecError::Malformed(
                            "palette-rle truncated run".into(),
                        ));
                    }
                    let idx = bytes[cursor] as usize;
                    let cnt =
                        u16::from_le_bytes([bytes[cursor + 1], bytes[cursor + 2]]) as usize;
                    cursor += 3;
                    if idx >= palette.len() {
                        return Err(CodecError::Malformed(format!(
                            "palette-rle index {} >= palette len {}",
                            idx,
                            palette.len()
                        )));
                    }
                    if written + cnt > BRICK_LEN {
                        return Err(CodecError::Malformed(format!(
                            "palette-rle overflow: {} + {} > {}",
                            written, cnt, BRICK_LEN
                        )));
                    }
                    let value = Voxel(palette[idx]);
                    for slot in arr[written..written + cnt].iter_mut() {
                        *slot = value;
                    }
                    written += cnt;
                }
                if written != BRICK_LEN {
                    return Err(CodecError::Malformed(format!(
                        "palette-rle underflow: wrote {} voxels, expected {}",
                        written, BRICK_LEN
                    )));
                }
                Ok(brick_from_voxels(arr))
            }
            other => Err(CodecError::Malformed(format!("unknown palette-rle flag {other}"))),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use atomr_worlds_core::coord::IVec3;

    fn make_empty_brick() -> Brick {
        Brick::new()
    }

    fn make_single_material_brick() -> Brick {
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    b.set(IVec3::new(x, y, z), Voxel::new(7));
                }
            }
        }
        b
    }

    fn make_random_brick(seed: u64) -> Brick {
        // Splitmix-style PRNG; deterministic and zero-deps.
        let mut state = seed;
        let mut b = Brick::new();
        for z in 0..16 {
            for y in 0..16 {
                for x in 0..16 {
                    state = state.wrapping_add(0x9E3779B97F4A7C15);
                    let mut z2 = state;
                    z2 ^= z2 >> 30;
                    z2 = z2.wrapping_mul(0xBF58476D1CE4E5B9);
                    z2 ^= z2 >> 27;
                    z2 = z2.wrapping_mul(0x94D049BB133111EB);
                    z2 ^= z2 >> 31;
                    let v = (z2 & 0xFFFF) as u16;
                    b.set(IVec3::new(x, y, z), Voxel::new(v));
                }
            }
        }
        b
    }

    fn make_many_materials_brick() -> Brick {
        // Force >256 distinct materials to exercise the PaletteRle overflow
        // fallback path.
        let mut b = Brick::new();
        for i in 0..BRICK_LEN {
            let z = (i / 256) as i64;
            let y = ((i / 16) % 16) as i64;
            let x = (i % 16) as i64;
            b.set(IVec3::new(x, y, z), Voxel::new((i % 4096) as u16));
        }
        b
    }

    fn assert_round_trip(codec: &dyn BrickCodec, brick: &Brick) {
        let encoded = codec.encode(brick);
        let decoded = codec.decode(&encoded).expect("decode");
        assert_eq!(decoded.voxels.as_ref(), brick.voxels.as_ref(), "{}", codec.id());
        assert_eq!(decoded.nonempty_count, brick.nonempty_count, "{}", codec.id());
        // Determinism: encoding twice must be byte-identical.
        let encoded2 = codec.encode(brick);
        assert_eq!(encoded, encoded2, "{} determinism", codec.id());
    }

    #[test]
    fn raw_u16_round_trip() {
        let codec = RawU16;
        for brick in [
            make_empty_brick(),
            make_single_material_brick(),
            make_random_brick(0xDEADBEEF),
        ] {
            assert_round_trip(&codec, &brick);
        }
    }

    #[test]
    fn rle_round_trip() {
        let codec = Rle;
        for brick in [
            make_empty_brick(),
            make_single_material_brick(),
            make_random_brick(0xCAFEF00D),
        ] {
            assert_round_trip(&codec, &brick);
        }
    }

    #[test]
    fn zlib_round_trip() {
        let codec = Zlib;
        for brick in [
            make_empty_brick(),
            make_single_material_brick(),
            make_random_brick(0x1234_5678_9ABC_DEF0),
        ] {
            assert_round_trip(&codec, &brick);
        }
    }

    #[test]
    fn palette_rle_round_trip_small_palette() {
        let codec = PaletteRle;
        for brick in [make_empty_brick(), make_single_material_brick()] {
            assert_round_trip(&codec, &brick);
        }
    }

    #[test]
    fn palette_rle_falls_back_on_overflow() {
        let codec = PaletteRle;
        let brick = make_many_materials_brick();
        assert_round_trip(&codec, &brick);
        let encoded = codec.encode(&brick);
        assert_eq!(encoded[0], PALETTE_FLAG_RAW, "expected raw sentinel on overflow");
    }
}
