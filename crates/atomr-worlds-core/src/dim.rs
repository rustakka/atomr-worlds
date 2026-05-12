//! Dimension identifiers.
//!
//! Every hierarchy level carries a [`DimensionId`] — an orthogonal-plane
//! selector mixed into the seed hash. The default `0` is the primary plane.
//! Non-zero values let you have alt-physics universes, Nether-style alternate
//! world planes, etc., all sharing the same coordinate grid as the primary.

pub type DimensionId = u32;

/// The primary plane at every level.
pub const PRIMARY: DimensionId = 0;
