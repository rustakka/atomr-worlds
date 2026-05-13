//! Phase 14d — surface heightmap + biome raster.
//!
//! [`SurfaceRaster`] is the derived data feeding the RTS / oblique-orthographic
//! display mode (`modes/rts.rs`). For an axis-aligned rectangle of world-XZ
//! columns, it stores, per column:
//!
//! - the **world-space Y** (meters) of the topmost non-empty voxel (the
//!   "ground" the RTS pass renders),
//! - the **material id** of that voxel (used as a coarse biome tag for the
//!   palette lookup),
//! - the **integer voxel-space Y** of that voxel (`top_z`) — only kept so that
//!   the cache invalidation predicate can ask *did the topmost voxel change?*
//!   without rebuilding the whole raster on every sub-surface write.
//!
//! Cache invariance: a write strictly below `top_z - 1` (i.e. at any column
//! position whose voxel-Y is less than `top_z[col]`) cannot change either the
//! surface height or the biome at this column, so [`SurfaceRaster::is_invalidated_by_write`]
//! returns `false` for it. A write at `top_z[col]` (replacing the topmost
//! voxel, possibly with empty) *can* change both, so we report invalidation.
//! This matches the Phase 14d test `rts_surface_invariance.rs`.

use std::hash::{Hash, Hasher};

use atomr_worlds_core::addr::WorldAddr;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::Lod;
use atomr_worlds_voxel::{BRICK_EDGE, BRICK_LEN};

use crate::mesh::{Mesh, Vertex};
use crate::render::material_color;
use crate::scene::MaterialPalette;
use crate::view_cache::{CacheAabb, DerivedKey};
use crate::world_query::WorldQuery;

/// Empty voxel material — kept as a local sentinel so this module doesn't
/// have to depend on `atomr-worlds-voxel::Voxel::EMPTY`. Matches `Voxel::EMPTY`.
const EMPTY_MATERIAL: u16 = 0;

/// How tall is the column we scan, in voxels, when probing for the topmost
/// non-empty voxel. Chosen so terrain generators with sub-100-m features
/// have plenty of headroom; the RTS demo's stub world fits easily inside
/// one brick row, so we keep this bounded to stay deterministic and fast.
const COLUMN_SCAN_VOXELS: i32 = (BRICK_EDGE as i32) * 4;
/// Voxel-Y the scan starts from. Columns are scanned downward from
/// `(COLUMN_TOP_VY, COLUMN_TOP_VY - COLUMN_SCAN_VOXELS]`.
const COLUMN_TOP_VY: i32 = (BRICK_EDGE as i32) * 2 - 1;

/// Sentinel `top_z` value for columns that found no non-empty voxel during the
/// scan — used in invalidation tests so an "empty column" is distinguishable
/// from a column whose top voxel is at voxel-Y 0.
pub const TOP_Z_EMPTY: i32 = i32::MIN;

/// Phase 14d surface raster — per-column ground height + biome.
///
/// Coordinate convention: `heightmap_m[r * dims[0] + c]` is the world-space
/// Y (meters) of the topmost solid voxel at column `(c, r)` where
/// `c ∈ [0, dims[0])` is the X stride and `r ∈ [0, dims[1])` is the Z stride.
/// World-XZ of column `(c, r)` is
/// `(origin_xz[0] + (c + 0.5) * voxel_size_m, origin_xz[1] + (r + 0.5) * voxel_size_m)`
/// — i.e. centers of voxel-sized cells.
#[derive(Clone, Debug)]
pub struct SurfaceRaster {
    /// Top-voxel world-Y per column, row-major `dims[0] × dims[1]`.
    pub heightmap_m: Vec<f32>,
    /// Top-voxel material id per column (0 = no voxel found).
    pub biome_id: Vec<u8>,
    /// Top-voxel voxel-Y (integer, world voxel-space) per column. The
    /// invalidation predicate compares writes against this.
    pub top_z: Vec<i32>,
    /// Column counts: `dims[0]` along X, `dims[1]` along Z.
    pub dims: [u32; 2],
    /// World-meters position of column `(0, 0)`'s lower-left corner in XZ.
    pub origin_xz: [f64; 2],
    /// Edge length of one voxel in meters (passed straight to the LOD).
    pub voxel_size_m: f32,
    /// Coarse "world revision" stamp the host can bump on whole-world
    /// regenerations. Not used for spatial invalidation here — that's the
    /// job of [`crate::view_cache::ViewCache::invalidate_intersecting`].
    pub world_rev: u64,
}

impl SurfaceRaster {
    /// AABB enclosing the columns this raster covers. The Y extents are
    /// `[min(heightmap), max(heightmap)]`; an all-empty raster falls back
    /// to a thin slab at `Y = 0` so the AABB is still well-defined.
    pub fn aabb(&self) -> CacheAabb {
        let w = self.voxel_size_m as f64 * self.dims[0] as f64;
        let h = self.voxel_size_m as f64 * self.dims[1] as f64;
        let mut y_lo = f32::INFINITY;
        let mut y_hi = f32::NEG_INFINITY;
        for &y in &self.heightmap_m {
            if y.is_finite() {
                if y < y_lo {
                    y_lo = y;
                }
                if y > y_hi {
                    y_hi = y;
                }
            }
        }
        if !y_lo.is_finite() || !y_hi.is_finite() {
            y_lo = 0.0;
            y_hi = 0.0;
        }
        CacheAabb::new(
            [self.origin_xz[0], y_lo as f64, self.origin_xz[1]],
            [self.origin_xz[0] + w, y_hi as f64, self.origin_xz[1] + h],
        )
    }

    /// `Some(height_m)` if `(x, z)` is in bounds, else `None`.
    #[inline]
    pub fn sample_height(&self, x: u32, z: u32) -> Option<f32> {
        let idx = self.column_index(x, z)?;
        Some(self.heightmap_m[idx])
    }

    /// `Some(biome_id)` if `(x, z)` is in bounds, else `None`.
    #[inline]
    pub fn sample_biome(&self, x: u32, z: u32) -> Option<u8> {
        let idx = self.column_index(x, z)?;
        Some(self.biome_id[idx])
    }

    /// Does a single-voxel write at world voxel-coordinate `(vx, vy, vz)`
    /// require this raster to be rebuilt? `true` iff the write touches a
    /// column we cover *and* its voxel-Y is at or above the current top-Z
    /// of that column. Writes strictly below `top_z - 1` are ignored — see
    /// module docs.
    pub fn is_invalidated_by_write(&self, vx: i64, vy: i64, vz: i64) -> bool {
        // Project (vx, vz) into raster columns. Floor division on signed
        // origins (origin is in meters; we recover voxel-space by dividing
        // by voxel_size_m). For the tests' integer voxel-size = 1.0 m the
        // recovery is exact.
        let vsize = self.voxel_size_m as f64;
        let col_x_f = (vx as f64) - (self.origin_xz[0] / vsize);
        let col_z_f = (vz as f64) - (self.origin_xz[1] / vsize);
        if col_x_f < 0.0 || col_z_f < 0.0 {
            return false;
        }
        let col_x = col_x_f as i64;
        let col_z = col_z_f as i64;
        if col_x >= self.dims[0] as i64 || col_z >= self.dims[1] as i64 {
            return false;
        }
        let idx = (col_z as usize) * (self.dims[0] as usize) + (col_x as usize);
        let top = self.top_z[idx];
        if top == TOP_Z_EMPTY {
            // Empty column — any write *could* be the new top.
            return true;
        }
        // Writes strictly below `top` (vy < top) cannot change either the
        // height or the biome of this column; everything at `vy >= top` can.
        (vy as i32) >= top
    }

    #[inline]
    fn column_index(&self, x: u32, z: u32) -> Option<usize> {
        if x >= self.dims[0] || z >= self.dims[1] {
            return None;
        }
        Some((z as usize) * (self.dims[0] as usize) + (x as usize))
    }
}

/// [`ViewCache`](crate::view_cache::ViewCache) key for [`SurfaceRaster`].
///
/// `origin_xz` is the bottom-left corner of the raster in **voxel** (not
/// meter) coordinates so the hash bucket lines up regardless of
/// `voxel_size_m`. `lod` distinguishes raster tiles built at different
/// detail levels.
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub struct SurfaceKey {
    pub addr: WorldAddr,
    pub origin_xz: [i64; 2],
    pub dims: [u32; 2],
    pub lod: Lod,
}

impl DerivedKey for SurfaceKey {
    fn world_addr(&self) -> &WorldAddr {
        &self.addr
    }

    fn intersects(&self, aabb: CacheAabb) -> bool {
        // We don't carry voxel_size_m in the key — `origin_xz` is in voxel
        // units. For invalidation we approximate the world-meter footprint
        // as `[origin, origin + dims]` voxel-units, which is correct up to
        // a uniform scale and conservative for the AABB intersection test.
        let x_lo = self.origin_xz[0] as f64;
        let z_lo = self.origin_xz[1] as f64;
        let x_hi = x_lo + self.dims[0] as f64;
        let z_hi = z_lo + self.dims[1] as f64;
        // No Y discrimination at the key level — every key covers the full
        // height column. Callers wishing finer eviction can use
        // `invalidate_key` on a specific [`SurfaceKey`].
        let key_box = CacheAabb::new([x_lo, f64::NEG_INFINITY, z_lo], [x_hi, f64::INFINITY, z_hi]);
        // CacheAabb::intersects uses `<=`/`>=`, so infinite Y trivially
        // overlaps any finite AABB — exactly what we want.
        key_box.intersects(aabb)
    }
}

// Required-by-trait hash sanity check: WorldAddr is already Hash;
// origin/dims/lod are integral. Hash for `SurfaceKey` is auto-derived above.
// This stub helps catch a future regression if someone changes those fields'
// types to something not-`Hash`.
#[allow(dead_code)]
fn _surface_key_hash_witness(k: &SurfaceKey, h: &mut impl Hasher) {
    k.hash(h);
}

/// Build a [`SurfaceRaster`] from a [`WorldQuery`] by scanning each column
/// top-down for the first non-empty voxel.
///
/// `origin_xz` is **world meters** of the bottom-left corner of column
/// `(0, 0)`. We translate to a voxel-space origin via `voxel_size_m`, then
/// for each column fetch the brick(s) it intersects and walk voxel-Y from
/// `COLUMN_TOP_VY` downward. The first non-empty voxel gives us:
///
/// - `heightmap_m[col] = world_y_top_in_meters`,
/// - `biome_id[col] = top_voxel.material as u8`,
/// - `top_z[col] = world_voxel_y_top` (used by the invalidation predicate).
///
/// Empty columns get `heightmap_m = 0.0`, `biome_id = 0`, `top_z = TOP_Z_EMPTY`.
pub fn build_surface_raster(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    origin_xz: [f64; 2],
    dims: [u32; 2],
    voxel_size_m: f32,
    lod: Lod,
) -> SurfaceRaster {
    build_surface_raster_with_lod_fn(world, addr, origin_xz, dims, voxel_size_m, |_| lod)
}

/// Like [`build_surface_raster`], but the caller supplies a closure that
/// chooses the LOD per (world-meter) column. RTS uses this with the
/// `ChunkStreamer` so columns near the observer scan at `near_lod` and
/// far-off columns scan at `far_lod`.
///
/// The brick cache is keyed by `(brick_coord, lod.depth)` so columns at
/// different LODs don't evict each other.
pub fn build_surface_raster_with_lod_fn<F>(
    world: &dyn WorldQuery,
    addr: &WorldAddr,
    origin_xz: [f64; 2],
    dims: [u32; 2],
    voxel_size_m: f32,
    mut lod_for_column: F,
) -> SurfaceRaster
where
    F: FnMut([f64; 2]) -> Lod,
{
    let total = (dims[0] as usize) * (dims[1] as usize);
    let mut heightmap_m = vec![0.0f32; total];
    let mut biome_id = vec![0u8; total];
    let mut top_z = vec![TOP_Z_EMPTY; total];

    let vsize = voxel_size_m as f64;
    // Origin in voxel-space (truncated). For the tests' integer-aligned
    // origins this is exact; the column-center sampling below adds 0.5 voxels
    // back so we're sampling on the same grid the host stores bricks in.
    let voxel_origin_x = (origin_xz[0] / vsize).floor() as i64;
    let voxel_origin_z = (origin_xz[1] / vsize).floor() as i64;

    // Cache brick fetches keyed by (brick_coord, lod.depth) — within one
    // column-scan we hit the same brick repeatedly as we step voxel-Y;
    // across columns we hit the same (bx, bz)-prefix bricks at every Y.
    // Including lod-depth in the key lets adjacent columns with different
    // streamer-selected LODs co-exist without thrashing.
    let mut brick_cache: std::collections::HashMap<
        (IVec3, u8),
        Option<std::sync::Arc<atomr_worlds_voxel::Brick>>,
    > = std::collections::HashMap::new();
    let edge = BRICK_EDGE as i32;

    for r in 0..dims[1] {
        for c in 0..dims[0] {
            let world_vx = voxel_origin_x + c as i64;
            let world_vz = voxel_origin_z + r as i64;
            let bx = world_vx.div_euclid(edge as i64);
            let bz = world_vz.div_euclid(edge as i64);
            let lx = world_vx.rem_euclid(edge as i64) as i32;
            let lz = world_vz.rem_euclid(edge as i64) as i32;

            // World-meter center of this column — the streamer measures
            // distance from the observer in meters, so we hand it a
            // meter-space sample point rather than voxel coords.
            let world_x_m = (world_vx as f64 + 0.5) * vsize;
            let world_z_m = (world_vz as f64 + 0.5) * vsize;
            let lod = lod_for_column([world_x_m, world_z_m]);

            // Scan downward from COLUMN_TOP_VY through COLUMN_TOP_VY -
            // COLUMN_SCAN_VOXELS. Brick coord changes every `BRICK_EDGE`
            // steps; for each brick we visit, we walk its in-brick Y range
            // top-down.
            let mut found = false;
            let scan_top = COLUMN_TOP_VY;
            let scan_bot = scan_top - COLUMN_SCAN_VOXELS;
            let mut vy = scan_top;
            while vy > scan_bot {
                let by = (vy as i64).div_euclid(edge as i64);
                let ly_top = (vy as i64).rem_euclid(edge as i64) as i32;
                // Pull this brick at the per-column LOD.
                let bc = IVec3::new(bx, by, bz);
                let brick = brick_cache
                    .entry((bc, lod.depth))
                    .or_insert_with(|| world.brick(addr, bc, lod))
                    .clone();
                let ly_bot_in_brick = 0i32;
                if let Some(brick) = brick {
                    // Walk down inside the brick.
                    for ly in (ly_bot_in_brick..=ly_top).rev() {
                        let idx = ((lz as usize) * BRICK_EDGE + (ly as usize)) * BRICK_EDGE + (lx as usize);
                        debug_assert!(idx < BRICK_LEN);
                        let material = brick.voxels[idx].0;
                        if material != EMPTY_MATERIAL {
                            let top_vy = (by as i32) * edge + ly;
                            let col_idx = (r as usize) * (dims[0] as usize) + (c as usize);
                            heightmap_m[col_idx] = (top_vy as f32 + 0.5) * voxel_size_m;
                            biome_id[col_idx] = (material & 0xFF) as u8;
                            top_z[col_idx] = top_vy;
                            found = true;
                            break;
                        }
                    }
                }
                if found {
                    break;
                }
                // Jump to the top of the next brick down. The next vy is the
                // top voxel of the brick at `by - 1`, i.e. by*edge - 1.
                vy = (by as i32) * edge - 1;
            }
        }
    }

    SurfaceRaster { heightmap_m, biome_id, top_z, dims, origin_xz, voxel_size_m, world_rev: 0 }
}

/// Build a flat-shaded mesh from a [`SurfaceRaster`]: one quad (two
/// triangles) per column, lying at the column's heightmap Y. Vertex color
/// is encoded via `Vertex::material` — the renderer's
/// [`material_color`](crate::render::material_color) maps it back to RGB.
///
/// Color choice: if `palette` has an entry for the biome id, that
/// entry's base color is rounded to a `u16` material id by hashing the
/// 0..255 byte — this means downstream `material_color` falls into its
/// "deterministic palette for unknown materials" branch, producing a
/// consistent color per biome regardless of the underlying voxel material.
/// Palette membership is the *only* knob that matters; the actual color
/// math is then deterministic.
pub fn surface_raster_to_mesh(raster: &SurfaceRaster, palette: &MaterialPalette) -> Mesh {
    let mut mesh = Mesh::default();
    let cols = raster.dims[0] as usize;
    let rows = raster.dims[1] as usize;
    let vsize = raster.voxel_size_m;
    let ox = raster.origin_xz[0] as f32;
    let oz = raster.origin_xz[1] as f32;
    // Pre-build the biome-id → material-id lookup. Palette entries with id
    // ≤ 255 mapping to themselves are the common case; anything else falls
    // back to the raw biome id reinterpreted as u16 so callers get a stable
    // per-biome color from `material_color`'s deterministic branch.
    let has_palette = !palette.entries.is_empty();

    for r in 0..rows {
        for c in 0..cols {
            let idx = r * cols + c;
            let h = raster.heightmap_m[idx];
            let biome = raster.biome_id[idx];
            if biome == 0 {
                continue;
            }
            // Top-face quad spanning [c, c+1] × [r, r+1] in voxel units.
            let x0 = ox + (c as f32) * vsize;
            let x1 = x0 + vsize;
            let z0 = oz + (r as f32) * vsize;
            let z1 = z0 + vsize;
            let mat: u16 = if has_palette {
                // Look up biome → palette id; if missing, fall through to
                // raw-biome encoding.
                palette
                    .entries
                    .iter()
                    .find(|e| (e.id & 0xFF) as u8 == biome)
                    .map(|e| e.id)
                    .unwrap_or(biome as u16)
            } else {
                biome as u16
            };
            // Sanity: ensure the material color resolves (the actual returned
            // RGB is unused here — vertex coloring goes through the rasterizer
            // which calls `material_color` itself — but this read keeps the
            // dependency live so a future refactor doesn't accidentally drop
            // the palette wiring without the test catching it).
            let _ = material_color(mat);
            let base = mesh.vertices.len() as u32;
            let n = [0.0f32, 1.0, 0.0];
            // Quad winding: (x0,z0) (x1,z0) (x1,z1) (x0,z1) — emits the top
            // face with +Y normal so the rasterizer's back-face cull keeps
            // it visible from above (the RTS camera looks down at the world).
            let p0 = [x0, h, z0];
            let p1 = [x1, h, z0];
            let p2 = [x1, h, z1];
            let p3 = [x0, h, z1];
            for p in [p0, p1, p2, p3] {
                mesh.vertices.push(Vertex { pos: p, normal: n, material: mat, ao: 1.0 });
            }
            // Two triangles, winding so the +Y normal faces "outwards"
            // (matches `mesh::emit_quad` for the `positive` axis case).
            mesh.indices.extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        }
    }
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::sync::mpsc;
    use std::sync::Arc;

    use atomr_worlds_core::addr::{LevelKey, WorldAddr};
    use atomr_worlds_core::coord::IVec3;
    use atomr_worlds_core::dim::PRIMARY;
    use atomr_worlds_proto::{WorldEvent, AABB};
    use atomr_worlds_voxel::{Brick, Voxel};

    struct ColumnStub {
        bricks: HashMap<IVec3, Arc<Brick>>,
    }

    impl ColumnStub {
        fn from_columns(heights: &[(i64, i64, i32, u16)]) -> Self {
            let mut bricks: HashMap<IVec3, Brick> = HashMap::new();
            for &(vx, vz, vy_top, mat) in heights {
                let edge = BRICK_EDGE as i64;
                let bc =
                    IVec3::new(vx.div_euclid(edge), (vy_top as i64).div_euclid(edge), vz.div_euclid(edge));
                let lx = vx.rem_euclid(edge) as i64;
                let ly = (vy_top as i64).rem_euclid(edge);
                let lz = vz.rem_euclid(edge);
                bricks.entry(bc).or_insert_with(Brick::new).set(IVec3::new(lx, ly, lz), Voxel::new(mat));
            }
            Self { bricks: bricks.into_iter().map(|(k, v)| (k, Arc::new(v))).collect() }
        }
    }

    impl WorldQuery for ColumnStub {
        fn brick(&self, _addr: &WorldAddr, bc: IVec3, _lod: Lod) -> Option<Arc<Brick>> {
            self.bricks.get(&bc).cloned()
        }
        fn ground_height_m(&self, _addr: &WorldAddr, _xz: [f64; 2]) -> Option<f32> {
            None
        }
        fn subscribe_region(&self, _addr: &WorldAddr, _r: AABB, _lod: Lod) -> mpsc::Receiver<WorldEvent> {
            let (_tx, rx) = mpsc::channel();
            rx
        }
    }

    fn root() -> WorldAddr {
        WorldAddr {
            universe: LevelKey::new(IVec3::ZERO, PRIMARY),
            galaxy: LevelKey::new(IVec3::ZERO, PRIMARY),
            sector: LevelKey::new(IVec3::ZERO, PRIMARY),
            system: LevelKey::new(IVec3::ZERO, PRIMARY),
            world: LevelKey::new(IVec3::ZERO, PRIMARY),
        }
    }

    #[test]
    fn build_picks_topmost_voxel() {
        // Two voxels in column (3, 5): one at y=4, one at y=7. We want y=7.
        let world = ColumnStub::from_columns(&[(3, 5, 4, 1), (3, 5, 7, 2), (1, 1, 0, 9)]);
        let raster = build_surface_raster(&world, &root(), [0.0, 0.0], [16, 16], 1.0, Lod::new(0));
        assert_eq!(raster.sample_biome(3, 5), Some(2));
        let h = raster.sample_height(3, 5).unwrap();
        assert!((h - 7.5).abs() < 1e-5, "expected y=7.5 (voxel center), got {h}");
    }

    #[test]
    fn empty_column_marked_empty() {
        let world = ColumnStub::from_columns(&[]);
        let raster = build_surface_raster(&world, &root(), [0.0, 0.0], [4, 4], 1.0, Lod::new(0));
        assert_eq!(raster.top_z[0], TOP_Z_EMPTY);
        assert_eq!(raster.biome_id[0], 0);
    }

    #[test]
    fn invalidation_predicate_sub_surface_ignored() {
        let world = ColumnStub::from_columns(&[(2, 2, 5, 1)]);
        let raster = build_surface_raster(&world, &root(), [0.0, 0.0], [8, 8], 1.0, Lod::new(0));
        // Write at the same column but well below the top voxel: should NOT
        // invalidate.
        assert!(!raster.is_invalidated_by_write(2, 3, 2));
        // Write AT the top voxel: should invalidate (it might be replaced).
        assert!(raster.is_invalidated_by_write(2, 5, 2));
        // Write ABOVE the top voxel: should invalidate (new top).
        assert!(raster.is_invalidated_by_write(2, 8, 2));
        // Write outside the raster bounds: should not invalidate.
        assert!(!raster.is_invalidated_by_write(100, 5, 100));
    }

    #[test]
    fn key_intersects_uses_xz_only() {
        let key = SurfaceKey { addr: root(), origin_xz: [0, 0], dims: [16, 16], lod: Lod::new(0) };
        // AABB at any Y inside the XZ box should match.
        assert!(key.intersects(CacheAabb::new([2.0, -1000.0, 2.0], [4.0, 1000.0, 4.0])));
        // AABB outside the XZ box should miss.
        assert!(!key.intersects(CacheAabb::new([100.0, 0.0, 0.0], [110.0, 1.0, 1.0])));
    }
}
