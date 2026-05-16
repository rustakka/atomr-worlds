//! Cave-carving strategy impls for the layered pipeline.
//!
//! Each strategy implements [`super::strategies::CaveStrategy`] and mutates
//! `BrickWorkspace::materials` (clearing solid voxels to [`Voxel::EMPTY`]).
//! [`worley::WorleyThreshold`] mirrors the legacy `TerrainGenerator` cave
//! carve; [`ca3d::CellularAutomata3D`] runs Conway-style 3D birth/death on
//! the apron; [`worm::PerlinWorm`] consumes `FeatureKind::Worm` anchors;
//! [`isosurface::IsosurfaceIntersection`] composes Cheese / Spaghetti /
//! Noodle simplex carves with a `y² - 1` parabola.

pub mod ca3d;
pub mod isosurface;
pub mod worley;
pub mod worm;

pub use ca3d::CellularAutomata3D;
pub use isosurface::IsosurfaceIntersection;
pub use worley::WorleyThreshold;
pub use worm::PerlinWorm;
