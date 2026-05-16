//! [`ColumnAnchorSeeder`] — scans the 3×3×3 neighborhood of coarse columns
//! around a brick and emits [`FeatureAnchor`]s on a `column_size_m`
//! (default 64 m) grid. Per-column seeds chain off `world_seed` via
//! [`child_seed`] under [`FEATURE_DIM`]; per-kind seeds mix the parent
//! column seed with a kind discriminator via [`splitmix64`].
//!
//! Anchors are memoized in an [`Arc<FeatureAnchorCache>`] so the same
//! column visited from neighboring bricks materializes its anchor list
//! exactly once — anchors are never recursive, and the same worm seeded in
//! column C traces identically regardless of which brick triggered it.

use std::sync::Arc;

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::{child_seed, splitmix64};
use atomr_worlds_voxel::BRICK_EDGE;

use crate::pipeline::anchor::{FeatureAnchor, FeatureAnchorCache, FeatureKind};
use crate::pipeline::strategies::FeatureSeederStrategy;
use crate::pipeline::workspace::BrickWorkspace;

/// `dim` value passed to [`child_seed`] for column-grid feature anchors.
/// Distinct from every other `child_seed` dimension to keep anchor seeds
/// from colliding with macro / brick chains.
pub const FEATURE_DIM: u32 = 0xFEA7_D1A0;

const WORM_DISC: u64 = 0x5A5A_5A5A_0000_0001;
const ORE_DISC: u64 = 0x5A5A_5A5A_0000_0002;
const STRUCT_DISC: u64 = 0x5A5A_5A5A_0000_0003;
const FLORA_DISC: u64 = 0x5A5A_5A5A_0000_0004;
const ISLAND_DISC: u64 = 0x5A5A_5A5A_0000_0005;
const BUFFER_DISC: u64 = 0x5A5A_5A5A_0000_0006;

/// Tunable mix of anchor kinds emitted per column. Each density value is
/// the *expected* count of anchors of that kind per column (Poisson-like;
/// values < 1 emit at most one anchor per column).
#[derive(Clone, Debug)]
pub struct SeederConfig {
    pub column_size_m: f32,
    /// Cardinal radius (in columns) scanned around the brick's home column.
    /// `1` produces the standard 3×3×3 neighborhood.
    pub neighborhood_radius: i32,
    pub worm_density: f32,
    pub ore_density: f32,
    pub structure_density: f32,
    pub flora_tree_density: f32,
    pub floating_island_density: f32,
    pub buffer_terrain_density: f32,
}

impl Default for SeederConfig {
    fn default() -> Self {
        Self {
            column_size_m: 64.0,
            neighborhood_radius: 1,
            worm_density: 1.0,
            ore_density: 1.0,
            structure_density: 0.0,
            flora_tree_density: 0.0,
            floating_island_density: 0.0,
            buffer_terrain_density: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ColumnAnchorSeeder {
    pub config: SeederConfig,
    cache: Arc<FeatureAnchorCache>,
}

impl Default for ColumnAnchorSeeder {
    fn default() -> Self {
        Self::new(SeederConfig::default())
    }
}

impl ColumnAnchorSeeder {
    pub fn new(config: SeederConfig) -> Self {
        Self { config, cache: Arc::new(FeatureAnchorCache::new()) }
    }

    pub fn with_cache(config: SeederConfig, cache: Arc<FeatureAnchorCache>) -> Self {
        Self { config, cache }
    }

    pub fn cache(&self) -> Arc<FeatureAnchorCache> {
        Arc::clone(&self.cache)
    }

    /// Brick-home column. A brick's world-meter centroid is mapped to the
    /// containing column on the coarse grid.
    fn home_column(&self, brick_coord: IVec3) -> IVec3 {
        let edge_m = BRICK_EDGE as f64;
        let cs = self.config.column_size_m as f64;
        let cx = (brick_coord.x as f64 * edge_m) / cs;
        let cy = (brick_coord.y as f64 * edge_m) / cs;
        let cz = (brick_coord.z as f64 * edge_m) / cs;
        IVec3::new(cx.floor() as i64, cy.floor() as i64, cz.floor() as i64)
    }

    /// Materialize the anchor list for a single column. Pure function of
    /// `(world_seed, column, config)` — fed through [`FeatureAnchorCache`]
    /// so neighbor bricks share one list.
    fn seed_column(&self, world_seed: u64, column: IVec3) -> Vec<FeatureAnchor> {
        let col_seed = child_seed(world_seed, FEATURE_DIM, column);
        let cs = self.config.column_size_m;
        let origin = [
            column.x as f32 * cs,
            column.y as f32 * cs,
            column.z as f32 * cs,
        ];

        let mut out = Vec::new();
        let kinds = [
            (FeatureKind::Worm, WORM_DISC, self.config.worm_density),
            (FeatureKind::OreVein, ORE_DISC, self.config.ore_density),
            (FeatureKind::Structure, STRUCT_DISC, self.config.structure_density),
            (FeatureKind::FloraTree, FLORA_DISC, self.config.flora_tree_density),
            (FeatureKind::FloatingIsland, ISLAND_DISC, self.config.floating_island_density),
            (FeatureKind::BufferTerrain, BUFFER_DISC, self.config.buffer_terrain_density),
        ];
        for (kind, disc, density) in kinds {
            if density <= 0.0 {
                continue;
            }
            let kind_seed = splitmix64(col_seed ^ disc);
            // Each density-unit emits one anchor; the fractional part is a
            // Bernoulli draw on the seed bits.
            let whole = density.floor() as u32;
            let frac = density - whole as f32;
            let count = if frac > 0.0 {
                let draw = ((kind_seed >> 11) as f32) * (1.0 / (1u64 << 53) as f32);
                whole + if draw < frac { 1 } else { 0 }
            } else {
                whole
            };
            for i in 0..count {
                let anchor_seed = splitmix64(kind_seed ^ (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));
                // Three independent draws place the anchor within its
                // column; clamping at the column edges keeps the anchor
                // inside its own (world_seed, column) cache key.
                let dx = ((anchor_seed >> 11) as f32) * (1.0 / (1u64 << 53) as f32);
                let dy = ((splitmix64(anchor_seed) >> 11) as f32) * (1.0 / (1u64 << 53) as f32);
                let dz = ((splitmix64(splitmix64(anchor_seed)) >> 11) as f32)
                    * (1.0 / (1u64 << 53) as f32);
                let origin_m = [
                    origin[0] + dx * cs,
                    origin[1] + dy * cs,
                    origin[2] + dz * cs,
                ];
                out.push(FeatureAnchor { kind, column, origin_m, seed: anchor_seed });
            }
        }
        out
    }
}

impl FeatureSeederStrategy for ColumnAnchorSeeder {
    fn id(&self) -> &'static str {
        "ColumnAnchorSeeder"
    }

    fn seed(&self, ws: &mut BrickWorkspace) {
        let home = self.home_column(ws.ctx.brick_coord);
        let r = self.config.neighborhood_radius;
        for dz in -r..=r {
            for dy in -r..=r {
                for dx in -r..=r {
                    let col = IVec3::new(home.x + dx as i64, home.y + dy as i64, home.z + dz as i64);
                    let anchors = self
                        .cache
                        .get_or_seed(ws.ctx.world_seed, col, || self.seed_column(ws.ctx.world_seed, col));
                    ws.anchors.extend(anchors);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brick::BrickGenContext;

    fn cfg() -> SeederConfig {
        SeederConfig {
            worm_density: 2.0,
            ore_density: 1.0,
            structure_density: 0.5,
            flora_tree_density: 0.5,
            ..Default::default()
        }
    }

    #[test]
    fn deterministic_per_seed_and_column() {
        let a = ColumnAnchorSeeder::new(cfg()).seed_column(42, IVec3::new(1, 2, 3));
        let b = ColumnAnchorSeeder::new(cfg()).seed_column(42, IVec3::new(1, 2, 3));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.seed, y.seed);
            assert_eq!(x.column, y.column);
            assert_eq!(x.origin_m, y.origin_m);
        }
    }

    #[test]
    fn cache_memoizes_per_column() {
        let seeder = ColumnAnchorSeeder::new(cfg());
        let mut a = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
        seeder.seed(&mut a);
        let after_first = seeder.cache.len();
        // Re-seeding the same brick is a pure cache hit — no new entries.
        let mut b = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
        seeder.seed(&mut b);
        assert_eq!(seeder.cache.len(), after_first);
        assert_eq!(a.anchors.len(), b.anchors.len());
    }

    #[test]
    fn neighbor_brick_shares_anchors_for_overlapping_columns() {
        let seeder = ColumnAnchorSeeder::new(cfg());
        let mut a = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
        let mut b = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(1, 0, 0)));
        seeder.seed(&mut a);
        seeder.seed(&mut b);
        // Any anchor with a column visible to both bricks must appear in
        // both anchor lists with the same seed.
        let home_a = seeder.home_column(IVec3::new(0, 0, 0));
        let home_b = seeder.home_column(IVec3::new(1, 0, 0));
        for col_dx in -1..=1 {
            for col_dy in -1..=1 {
                for col_dz in -1..=1 {
                    let col_a = IVec3::new(home_a.x + col_dx, home_a.y + col_dy, home_a.z + col_dz);
                    let in_b = (col_a.x - home_b.x).abs() <= 1
                        && (col_a.y - home_b.y).abs() <= 1
                        && (col_a.z - home_b.z).abs() <= 1;
                    if !in_b {
                        continue;
                    }
                    let from_a: Vec<_> = a.anchors.iter().filter(|x| x.column == col_a).collect();
                    let from_b: Vec<_> = b.anchors.iter().filter(|x| x.column == col_a).collect();
                    assert_eq!(from_a.len(), from_b.len());
                    for (x, y) in from_a.iter().zip(from_b.iter()) {
                        assert_eq!(x.seed, y.seed);
                        assert_eq!(x.origin_m, y.origin_m);
                    }
                }
            }
        }
    }

    #[test]
    fn zero_density_emits_no_anchors_of_kind() {
        let seeder = ColumnAnchorSeeder::new(SeederConfig {
            worm_density: 1.0,
            ore_density: 0.0,
            structure_density: 0.0,
            flora_tree_density: 0.0,
            floating_island_density: 0.0,
            buffer_terrain_density: 0.0,
            ..Default::default()
        });
        let mut ws = BrickWorkspace::new(BrickGenContext::legacy(7, IVec3::new(0, 0, 0)));
        seeder.seed(&mut ws);
        assert!(ws.anchors.iter().any(|a| a.kind == FeatureKind::Worm));
        assert!(!ws.anchors.iter().any(|a| a.kind == FeatureKind::OreVein));
    }
}
