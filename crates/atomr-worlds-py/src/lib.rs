//! Python bindings for atomr-worlds.
//!
//! Exposes:
//! - Pure-data primitives (`WorldAddr`, `LevelKey`, `Lod`, `MetricScale`,
//!   `Voxel`, `Brick`) and seed helpers (`splitmix64`, `child_seed`,
//!   `WorldAddr.seed_chain`).
//! - A `WorldClient` backed by `LocalHost` for queries.
//!
//! Build with `maturin develop -m crates/atomr-worlds-py/Cargo.toml`.
#![allow(clippy::useless_conversion)]

use std::sync::Arc;

use atomr_worlds_core::addr::{Level, LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_core::seed as seed_core;
use atomr_worlds_host::{LocalHost, LocalHostConfig, WorldHost};
use atomr_worlds_proto::{Envelope, WorldEvent, WorldRequest};
use atomr_worlds_voxel::{Brick as RustBrick, Voxel as RustVoxel, BRICK_EDGE};
use pyo3::exceptions::{PyRuntimeError, PyValueError};
use pyo3::prelude::*;
use tokio::runtime::Runtime;

// ─────────────────────────────────────────────────────────────────────────────
// Free functions: seed helpers.
// ─────────────────────────────────────────────────────────────────────────────

#[pyfunction]
fn splitmix64(z: u64) -> u64 {
    seed_core::splitmix64(z)
}

#[pyfunction]
fn child_seed(parent: u64, dim: u32, x: i64, y: i64, z: i64) -> u64 {
    seed_core::child_seed(parent, dim, IVec3::new(x, y, z))
}

// ─────────────────────────────────────────────────────────────────────────────
// PyLevelKey
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "LevelKey", module = "atomrworlds")]
#[derive(Clone, Copy, Debug)]
struct PyLevelKey(LevelKey);

#[pymethods]
impl PyLevelKey {
    #[new]
    #[pyo3(signature = (x=0, y=0, z=0, dim=0))]
    fn new(x: i64, y: i64, z: i64, dim: u32) -> Self {
        Self(LevelKey { coord: IVec3::new(x, y, z), dim })
    }
    #[getter]
    fn x(&self) -> i64 { self.0.coord.x }
    #[getter]
    fn y(&self) -> i64 { self.0.coord.y }
    #[getter]
    fn z(&self) -> i64 { self.0.coord.z }
    #[getter]
    fn dim(&self) -> u32 { self.0.dim }
    fn __repr__(&self) -> String {
        format!("LevelKey(x={}, y={}, z={}, dim={})", self.0.coord.x, self.0.coord.y, self.0.coord.z, self.0.dim)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyWorldAddr
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "WorldAddr", module = "atomrworlds")]
#[derive(Clone, Copy, Debug)]
struct PyWorldAddr(WorldAddr);

#[pymethods]
impl PyWorldAddr {
    #[new]
    fn new() -> Self {
        Self(WorldAddr::ROOT)
    }

    #[staticmethod]
    fn root() -> Self {
        Self(WorldAddr::ROOT)
    }

    /// Build from explicit per-tier `LevelKey`s.
    #[staticmethod]
    fn build(
        universe: PyLevelKey,
        galaxy: PyLevelKey,
        sector: PyLevelKey,
        system: PyLevelKey,
        world: PyLevelKey,
    ) -> Self {
        Self(WorldAddr {
            universe: universe.0,
            galaxy: galaxy.0,
            sector: sector.0,
            system: system.0,
            world: world.0,
        })
    }

    #[getter]
    fn universe(&self) -> PyLevelKey { PyLevelKey(self.0.universe) }
    #[getter]
    fn galaxy(&self) -> PyLevelKey { PyLevelKey(self.0.galaxy) }
    #[getter]
    fn sector(&self) -> PyLevelKey { PyLevelKey(self.0.sector) }
    #[getter]
    fn system(&self) -> PyLevelKey { PyLevelKey(self.0.system) }
    #[getter]
    fn world(&self) -> PyLevelKey { PyLevelKey(self.0.world) }

    /// Returns `[universe_seed, galaxy_seed, sector_seed, system_seed, world_seed]`.
    fn seed_chain(&self, root: u64) -> [u64; 5] {
        self.0.seed_chain(root)
    }

    fn world_seed(&self, root: u64) -> u64 {
        self.0.seed_at(root, Level::World)
    }

    fn __repr__(&self) -> String {
        format!("WorldAddr(universe={:?}, galaxy={:?}, sector={:?}, system={:?}, world={:?})",
            self.universe().__repr__(), self.galaxy().__repr__(),
            self.sector().__repr__(), self.system().__repr__(), self.world().__repr__())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyLod / PyMetricScale
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "Lod", module = "atomrworlds")]
#[derive(Clone, Copy, Debug)]
struct PyLod(Lod);

#[pymethods]
impl PyLod {
    #[new]
    #[pyo3(signature = (depth=0))]
    fn new(depth: u8) -> Self {
        Self(Lod::new(depth))
    }
    #[getter]
    fn depth(&self) -> u8 { self.0.depth }
    fn __repr__(&self) -> String { format!("Lod(depth={})", self.0.depth) }
}

#[pyclass(name = "MetricScale", module = "atomrworlds")]
#[derive(Clone, Copy, Debug)]
struct PyMetricScale(MetricScale);

#[pymethods]
impl PyMetricScale {
    #[new]
    fn new(root_size_m: f64, max_depth: u8) -> Self {
        Self(MetricScale { root_size_m, max_depth })
    }
    #[staticmethod]
    fn default_universe() -> Self { Self(MetricScale::DEFAULT_UNIVERSE) }
    #[staticmethod]
    fn default_galaxy() -> Self { Self(MetricScale::DEFAULT_GALAXY) }
    #[staticmethod]
    fn default_sector() -> Self { Self(MetricScale::DEFAULT_SECTOR) }
    #[staticmethod]
    fn default_system() -> Self { Self(MetricScale::DEFAULT_SYSTEM) }
    #[staticmethod]
    fn default_world() -> Self { Self(MetricScale::DEFAULT_WORLD) }
    #[getter]
    fn root_size_m(&self) -> f64 { self.0.root_size_m }
    #[getter]
    fn max_depth(&self) -> u8 { self.0.max_depth }
    fn meters_per_voxel(&self, lod: PyLod) -> f64 { self.0.meters_per_voxel(lod.0) }
    fn leaf_size_m(&self) -> f64 { self.0.leaf_size_m() }
    fn __repr__(&self) -> String {
        format!("MetricScale(root_size_m={:.3e}, max_depth={})", self.0.root_size_m, self.0.max_depth)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyVoxel / PyBrick
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "Voxel", module = "atomrworlds")]
#[derive(Clone, Copy, Debug)]
struct PyVoxel(RustVoxel);

#[pymethods]
impl PyVoxel {
    #[new]
    #[pyo3(signature = (material=0))]
    fn new(material: u16) -> Self {
        Self(RustVoxel::new(material))
    }
    #[staticmethod]
    fn empty() -> Self {
        Self(RustVoxel::EMPTY)
    }
    #[getter]
    fn material(&self) -> u16 { self.0.0 }
    fn is_empty(&self) -> bool { self.0.is_empty() }
    fn __repr__(&self) -> String { format!("Voxel(material={})", self.0.0) }
}

#[pyclass(name = "Brick", module = "atomrworlds")]
#[derive(Clone, Debug)]
struct PyBrick {
    inner: RustBrick,
}

#[pymethods]
impl PyBrick {
    #[new]
    fn new() -> Self {
        Self { inner: RustBrick::new() }
    }
    /// Build from raw bytes (the on-wire payload of `WorldEvent::BrickSnapshot`).
    #[staticmethod]
    fn from_bytes(bytes: Vec<u8>) -> PyResult<Self> {
        RustBrick::from_bytes(&bytes)
            .map(|inner| Self { inner })
            .map_err(|e| PyValueError::new_err(format!("{e}")))
    }
    #[staticmethod]
    fn edge() -> usize {
        BRICK_EDGE
    }
    fn nonempty_count(&self) -> u16 { self.inner.nonempty_count }
    fn is_empty(&self) -> bool { self.inner.is_empty() }
    fn get(&self, x: i64, y: i64, z: i64) -> PyVoxel {
        PyVoxel(self.inner.get(IVec3::new(x, y, z)))
    }
    fn set(&mut self, x: i64, y: i64, z: i64, voxel: PyVoxel) -> bool {
        self.inner.set(IVec3::new(x, y, z), voxel.0)
    }
    fn to_bytes(&self) -> Vec<u8> { self.inner.to_bytes() }
    /// Return all material ids as a flat list of length 4096 (z, y, x major).
    fn materials(&self) -> Vec<u16> {
        self.inner.voxels.iter().map(|v| v.0).collect()
    }
    fn __repr__(&self) -> String {
        format!("Brick(nonempty={})", self.inner.nonempty_count)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PyWorldClient — LocalHost-backed
// ─────────────────────────────────────────────────────────────────────────────

#[pyclass(name = "WorldClient", module = "atomrworlds")]
struct PyWorldClient {
    rt: Arc<Runtime>,
    host: Arc<LocalHost>,
}

#[pymethods]
impl PyWorldClient {
    /// Create a single-player client with the given root seed.
    #[new]
    #[pyo3(signature = (root_seed=0xDEAD_BEEF_CAFE_F00D_u64))]
    fn new(root_seed: u64) -> PyResult<Self> {
        let rt = Runtime::new().map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        let cfg = LocalHostConfig { root_seed, ..LocalHostConfig::default() };
        let host = rt
            .block_on(LocalHost::new(cfg))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { rt: Arc::new(rt), host: Arc::new(host) })
    }

    fn get_voxel(&self, addr: PyWorldAddr, x: i64, y: i64, z: i64) -> PyResult<PyVoxel> {
        let req = WorldRequest::GetVoxel { addr: addr.0, pos: IVec3::new(x, y, z) };
        let env = Envelope::new(0, addr.0, req);
        let resp = self
            .rt
            .block_on(self.host.request(env))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        match resp.body {
            WorldEvent::Voxel { voxel, .. } => Ok(PyVoxel(voxel)),
            other => Err(PyRuntimeError::new_err(format!("unexpected response: {other:?}"))),
        }
    }

    #[pyo3(signature = (addr, bx, by, bz, lod_depth=0))]
    fn get_brick(&self, addr: PyWorldAddr, bx: i64, by: i64, bz: i64, lod_depth: u8) -> PyResult<PyBrick> {
        let req = WorldRequest::GetBrick {
            addr: addr.0,
            brick: IVec3::new(bx, by, bz),
            lod: Lod::new(lod_depth),
        };
        let env = Envelope::new(0, addr.0, req);
        let resp = self
            .rt
            .block_on(self.host.request(env))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        match resp.body {
            WorldEvent::BrickSnapshot { payload, .. } => {
                let brick = RustBrick::from_bytes(&payload)
                    .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
                Ok(PyBrick { inner: brick })
            }
            other => Err(PyRuntimeError::new_err(format!("unexpected response: {other:?}"))),
        }
    }

    fn write_voxel(&self, addr: PyWorldAddr, x: i64, y: i64, z: i64, voxel: PyVoxel) -> PyResult<()> {
        let req = WorldRequest::WriteVoxel { addr: addr.0, pos: IVec3::new(x, y, z), voxel: voxel.0 };
        let env = Envelope::new(0, addr.0, req);
        let _ = self
            .rt
            .block_on(self.host.request(env))
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
        Ok(())
    }

    fn shutdown(&self) -> PyResult<()> {
        self.rt
            .block_on(self.host.shutdown())
            .map_err(|e| PyRuntimeError::new_err(format!("{e}")))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Module entrypoint
// ─────────────────────────────────────────────────────────────────────────────

#[pymodule]
fn atomrworlds_native(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(splitmix64, m)?)?;
    m.add_function(wrap_pyfunction!(child_seed, m)?)?;
    m.add_class::<PyLevelKey>()?;
    m.add_class::<PyWorldAddr>()?;
    m.add_class::<PyLod>()?;
    m.add_class::<PyMetricScale>()?;
    m.add_class::<PyVoxel>()?;
    m.add_class::<PyBrick>()?;
    m.add_class::<PyWorldClient>()?;
    m.add("BRICK_EDGE", BRICK_EDGE)?;
    Ok(())
}
