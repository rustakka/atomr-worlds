"""Python interface for atomr-worlds.

Thin wrapper around the Rust extension module `atomrworlds_native`. Re-exports
the user-facing types and provides convenience helpers (e.g. NumPy-friendly
brick reshaping if NumPy is available; falls back gracefully if not).
"""
from .atomrworlds_native import (  # type: ignore[import-not-found]
    BRICK_EDGE,
    Brick,
    LevelKey,
    Lod,
    MetricScale,
    Voxel,
    WorldAddr,
    WorldClient,
    child_seed,
    splitmix64,
)

__all__ = [
    "BRICK_EDGE",
    "Brick",
    "LevelKey",
    "Lod",
    "MetricScale",
    "Voxel",
    "WorldAddr",
    "WorldClient",
    "child_seed",
    "splitmix64",
    "brick_to_numpy",
]


def brick_to_numpy(brick):
    """Reshape a `Brick`'s flat 4096-element material list into a (16, 16, 16)
    NumPy array (z, y, x order). Returns the plain list if NumPy is missing.
    """
    materials = brick.materials()
    try:
        import numpy as np  # type: ignore[import-not-found]
    except ImportError:
        return materials
    return np.array(materials, dtype=np.uint16).reshape((BRICK_EDGE, BRICK_EDGE, BRICK_EDGE))
