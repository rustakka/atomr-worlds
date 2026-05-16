//! Strategy trait surface for the layered brick pipeline.
//!
//! Each stage takes `&mut BrickWorkspace`; absent stages are wired to
//! "none" no-op impls so the pipeline can early-return at zero cost when
//! a preset doesn't need a given pass.

use std::fmt::Debug;

use super::anchor::FeatureAnchor;
use super::workspace::BrickWorkspace;

macro_rules! none_impl {
    ($name:ident, $trait:ident) => {
        #[derive(Debug, Default, Copy, Clone)]
        pub struct $name;
        impl $trait for $name {
            fn id(&self) -> &'static str {
                concat!("none::", stringify!($name))
            }
            fn run(&self, _ws: &mut BrickWorkspace) {}
        }
    };
}

/// Fills `workspace.density` (the padded 18³ scalar field). Sign
/// convention: positive = solid, negative = empty, zero = surface.
pub trait DensityFieldStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Translates the density field into per-voxel materials (topsoil bands,
/// geological strata, etc.). Reads `workspace.density`; writes
/// `workspace.materials`.
pub trait StrataStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Carves caves by clearing voxels in `workspace.materials` (and/or
/// driving `workspace.density` negative on the apron when subsequent
/// passes still need the field).
pub trait CaveStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Replaces stone voxels with ore voxels via thresholded noise or
/// anchor-driven random walks.
pub trait OreVeinStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Surface-level erosion. The Vanilla impl reproduces the existing river
/// carve byte-equal; richer impls (e.g. droplet hydraulic) simulate
/// gradient descent particles.
pub trait ErosionStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Fluid fill (sea level, river water, CA flow, LBM lattice).
pub trait FluidStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Macro structures: WFC dungeons, Jigsaw villages, QWFC stub. Reads
/// `workspace.anchors` with `kind == Structure`.
pub trait StructureStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Flora: L-system trees, grass tufts, etc. Reads `workspace.anchors`
/// with `kind == FloraTree`.
pub trait FloraStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Surface point placement (Poisson-disk, Mitchell best-candidate, white
/// noise, uniform grid). Consumed by flora and structure stages.
pub trait PlacementStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    /// Produce a list of brick-local positions on the surface that flora
    /// or structures can attach to.
    fn place(&self, ws: &BrickWorkspace) -> Vec<[f32; 3]>;
}

/// Biome assignment matrix: per-face Whittaker, Voronoi cells, direct 2D
/// temp/humidity. Pure read — emits into a per-workspace biome table that
/// strata / flora / blend stages consume.
pub trait BiomeMatrixStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Inter-biome smoothing: hard borders, normalized-sparse-convolution,
/// buffer-terrain injection.
pub trait BiomeBlendStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Sky-light propagation. Produces (or refines) `workspace.light` so the
/// mesher can bake per-vertex sky brightness.
pub trait SkyLightStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn run(&self, ws: &mut BrickWorkspace);
}

/// Cross-brick feature anchors. Runs *before* the per-brick stages and
/// fills `workspace.anchors` with the union of anchors visible from this
/// brick's 3×3×3 column neighborhood.
pub trait FeatureSeederStrategy: Send + Sync + Debug {
    fn id(&self) -> &'static str;
    fn seed(&self, ws: &mut BrickWorkspace);
}

none_impl!(NoneDensity, DensityFieldStrategy);
none_impl!(NoneStrata, StrataStrategy);
none_impl!(NoneCaves, CaveStrategy);
none_impl!(NoneOre, OreVeinStrategy);
none_impl!(NoneErosion, ErosionStrategy);
none_impl!(NoneFluid, FluidStrategy);
none_impl!(NoneStructures, StructureStrategy);
none_impl!(NoneFlora, FloraStrategy);
none_impl!(NoneBiomeMatrix, BiomeMatrixStrategy);
none_impl!(NoneBiomeBlend, BiomeBlendStrategy);
none_impl!(NoneSkyLight, SkyLightStrategy);

#[derive(Debug, Default, Copy, Clone)]
pub struct NonePlacement;
impl PlacementStrategy for NonePlacement {
    fn id(&self) -> &'static str {
        "none::NonePlacement"
    }
    fn place(&self, _ws: &BrickWorkspace) -> Vec<[f32; 3]> {
        Vec::new()
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct EmptySeeder;
impl FeatureSeederStrategy for EmptySeeder {
    fn id(&self) -> &'static str {
        "none::EmptySeeder"
    }
    fn seed(&self, _ws: &mut BrickWorkspace) {}
}

/// Bundled output of [`PlacementStrategy::place`], surfaced for downstream
/// stages that want it without re-running placement.
#[derive(Debug, Default, Clone)]
pub struct PlacementOutput {
    pub points: Vec<[f32; 3]>,
}

impl PlacementOutput {
    pub fn from_anchors(anchors: &[FeatureAnchor]) -> Self {
        Self {
            points: anchors.iter().map(|a| a.origin_m).collect(),
        }
    }
}
