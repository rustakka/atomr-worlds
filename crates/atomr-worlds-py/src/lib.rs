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

use atomr_worlds_core::addr::{Address, Level, LevelKey, WorldAddr};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::lod::{Lod, MetricScale};
use atomr_worlds_core::seed as seed_core;
use atomr_worlds_host::{LiteralRegion, LocalHost, LocalHostConfig, RegionAabb, WorldHost};
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
    /// Return the raw little-endian voxel bytes as a Python `bytes` object —
    /// suitable for `numpy.frombuffer(bytes, dtype=numpy.uint16).reshape(16,
    /// 16, 16)`. Single allocation, no per-voxel copy. The `__getbuffer__`
    /// path below gives true zero-copy on Python 3.11+ (limited API exposes
    /// the `Py_bf_getbuffer` slot from 3.11 onward); this method stays as a
    /// fallback when callers want a copying byte buffer.
    fn buffer_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, pyo3::types::PyBytes> {
        let bytes = bytemuck::cast_slice::<RustVoxel, u8>(self.inner.voxels.as_ref());
        pyo3::types::PyBytes::new_bound(py, bytes)
    }

    /// Phase 11 follow-up — true zero-copy buffer protocol.
    ///
    /// Exposes the brick's voxel data as a `(16, 16, 16)` `uint16` buffer;
    /// `numpy.asarray(brick)` allocates no copy. Requires Python 3.11+
    /// because the `Py_bf_getbuffer` slot only entered the stable ABI
    /// (limited API) at 3.11; older Python wheels keep the
    /// `buffer_bytes()` helper as the zero-allocation alternative.
    ///
    /// # Safety
    /// `view` is filled in the same shape as
    /// `pyo3/tests/test_buffer_protocol.rs::fill_view_from_readonly_data`.
    /// `view.obj` borrows the `PyBrick` itself, so the underlying voxel
    /// slice stays alive as long as the buffer is held.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __getbuffer__(
        slf: pyo3::Bound<'_, Self>,
        view: *mut pyo3::ffi::Py_buffer,
        flags: std::os::raw::c_int,
    ) -> PyResult<()> {
        use pyo3::exceptions::PyBufferError;
        use std::ffi::CString;
        use std::os::raw::c_void;
        use std::ptr;

        if view.is_null() {
            return Err(PyBufferError::new_err("View is null"));
        }
        if (flags & pyo3::ffi::PyBUF_WRITABLE) == pyo3::ffi::PyBUF_WRITABLE {
            return Err(PyBufferError::new_err("Brick buffer is read-only"));
        }

        // Pull the voxel byte slice through `slf` so the data pointer
        // stays valid for as long as Python holds a reference to the
        // `PyBrick` (PyO3 increfs `view.obj` for us via `into_ptr`).
        let borrow = slf.borrow();
        let bytes = bytemuck::cast_slice::<RustVoxel, u8>(borrow.inner.voxels.as_ref());
        let len = bytes.len() as isize;
        let buf_ptr = bytes.as_ptr() as *mut c_void;
        // `borrow` keeps the `&PyBrick` alive only for the scope of this
        // function; the `view.obj` increment below makes the pointer's
        // lifetime tracked by Python instead.
        drop(borrow);

        (*view).obj = slf.into_ptr();
        (*view).buf = buf_ptr;
        (*view).len = len;
        (*view).readonly = 1;
        (*view).itemsize = std::mem::size_of::<RustVoxel>() as isize;

        // numpy reads the format string ("H" = uint16, native byte order)
        // when `PyBUF_FORMAT` is requested; otherwise leave it null so
        // simple consumers (memoryview) accept the default `B` layout.
        (*view).format = if (flags & pyo3::ffi::PyBUF_FORMAT) == pyo3::ffi::PyBUF_FORMAT {
            CString::new("H").unwrap().into_raw()
        } else {
            ptr::null_mut()
        };

        // Shape: (16, 16, 16). The Brick's flat layout is
        // `(z * 16 + y) * 16 + x`, so dim-0 is z, dim-1 is y, dim-2 is x.
        // We Box the shape array onto the heap and leak it through the
        // view; `__releasebuffer__` reclaims it.
        if (flags & pyo3::ffi::PyBUF_ND) == pyo3::ffi::PyBUF_ND {
            let edge = BRICK_EDGE as pyo3::ffi::Py_ssize_t;
            let shape = Box::leak(Box::new([edge, edge, edge]));
            (*view).ndim = 3;
            (*view).shape = shape.as_mut_ptr();

            if (flags & pyo3::ffi::PyBUF_STRIDES) == pyo3::ffi::PyBUF_STRIDES {
                let item = std::mem::size_of::<RustVoxel>() as pyo3::ffi::Py_ssize_t;
                let strides = Box::leak(Box::new([item * edge * edge, item * edge, item]));
                (*view).strides = strides.as_mut_ptr();
            } else {
                (*view).strides = ptr::null_mut();
            }
        } else {
            (*view).ndim = 1;
            (*view).shape = ptr::null_mut();
            (*view).strides = ptr::null_mut();
        }

        (*view).suboffsets = ptr::null_mut();
        (*view).internal = ptr::null_mut();

        Ok(())
    }

    /// Phase 11 follow-up: free the format string + shape/stride boxes
    /// allocated by `__getbuffer__`.
    #[cfg(any(not(Py_LIMITED_API), Py_3_11))]
    unsafe fn __releasebuffer__(&self, view: *mut pyo3::ffi::Py_buffer) {
        use std::ffi::CString;
        if view.is_null() {
            return;
        }
        if !(*view).format.is_null() {
            drop(CString::from_raw((*view).format));
            (*view).format = std::ptr::null_mut();
        }
        if !(*view).shape.is_null() {
            // Reconstruct the [Py_ssize_t; 3] box so it drops cleanly.
            let _ = Box::from_raw((*view).shape as *mut [pyo3::ffi::Py_ssize_t; 3]);
            (*view).shape = std::ptr::null_mut();
        }
        if !(*view).strides.is_null() {
            let _ = Box::from_raw((*view).strides as *mut [pyo3::ffi::Py_ssize_t; 3]);
            (*view).strides = std::ptr::null_mut();
        }
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
        let a = Address::World(addr.0);
        let req = WorldRequest::GetVoxel { addr: a, pos: IVec3::new(x, y, z) };
        let env = Envelope::new(0, a, req);
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
        let a = Address::World(addr.0);
        let req = WorldRequest::GetBrick {
            addr: a,
            brick: IVec3::new(bx, by, bz),
            lod: Lod::new(lod_depth),
        };
        let env = Envelope::new(0, a, req);
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
        let a = Address::World(addr.0);
        let req = WorldRequest::WriteVoxel { addr: a, pos: IVec3::new(x, y, z), voxel: voxel.0 };
        let env = Envelope::new(0, a, req);
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

    /// Phase 11 follow-up — async subscribe.
    ///
    /// Returns a coroutine that resolves to a [`PySubscriptionHandle`].
    /// The handle is an async iterator: `async for ev in handle: ...`.
    /// Each yielded `ev` is a dict shaped like one of:
    ///
    /// ```text
    /// {"kind": "snapshot", "brick": (bx, by, bz), "lod": int, "payload": bytes}
    /// {"kind": "delta",    "pos": (x, y, z),     "before": int, "after": int}
    /// {"kind": "stream_end", "sub_id": int}
    /// ```
    ///
    /// `region_min` and `region_max` are inclusive-min / exclusive-max
    /// world voxel coords. `lod_depth` selects the LOD the snapshot
    /// payloads are delivered at.
    #[pyo3(signature = (addr, region_min, region_max, lod_depth=0, sub_id=1))]
    fn subscribe_async<'py>(
        &self,
        py: Python<'py>,
        addr: PyWorldAddr,
        region_min: (i64, i64, i64),
        region_max: (i64, i64, i64),
        lod_depth: u8,
        sub_id: u64,
    ) -> PyResult<Bound<'py, PyAny>> {
        let host = self.host.clone();
        let a = Address::World(addr.0);
        let region = atomr_worlds_proto::AABB::new(
            IVec3::new(region_min.0, region_min.1, region_min.2),
            IVec3::new(region_max.0, region_max.1, region_max.2),
        );
        let env = Envelope::new(
            sub_id,
            a,
            WorldRequest::Subscribe { addr: a, region, lod: Lod::new(lod_depth), sub_id },
        );
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let receiver = host
                .subscribe(env)
                .await
                .map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;
            Ok(PySubscriptionHandle {
                receiver: Arc::new(tokio::sync::Mutex::new(Some(receiver))),
                sub_id,
            })
        })
    }

    /// Register an authored region of literal voxel data.
    ///
    /// `bounds_min` and `bounds_max` are inclusive-min, exclusive-max
    /// voxel coordinates. `voxels` maps `(x, y, z) -> material_id`. Any
    /// entries outside the bounds are silently ignored at application.
    /// Useful for storytelling worlds and (Phase 13e) DEM imports.
    #[pyo3(signature = (name, bounds_min, bounds_max, voxels))]
    fn register_literal_region(
        &self,
        name: &str,
        bounds_min: (i64, i64, i64),
        bounds_max: (i64, i64, i64),
        voxels: std::collections::HashMap<(i64, i64, i64), u16>,
    ) -> PyResult<()> {
        let bounds = RegionAabb::new(
            IVec3::new(bounds_min.0, bounds_min.1, bounds_min.2),
            IVec3::new(bounds_max.0, bounds_max.1, bounds_max.2),
        );
        let map: std::collections::HashMap<IVec3, RustVoxel> = voxels
            .into_iter()
            .map(|((x, y, z), m)| (IVec3::new(x, y, z), RustVoxel::new(m)))
            .collect();
        let region = std::sync::Arc::new(LiteralRegion::new(name, bounds, map));
        self.host.register_authored_region(region);
        Ok(())
    }

    /// Number of currently-registered authored regions.
    fn authored_region_count(&self) -> usize {
        self.host.authored_region_store().lock().unwrap().len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// PySubscriptionHandle — Phase 11 follow-up
// ─────────────────────────────────────────────────────────────────────────────

/// Async-iterable handle returned by [`PyWorldClient::subscribe_async`].
///
/// `__aiter__` returns `self`; `__anext__` resolves to the next
/// `WorldEvent` rendered as a tagged dict (`"kind"` + variant fields).
/// `StopAsyncIteration` is raised when the underlying mpsc channel
/// closes or yields `StreamEnd`. The handle keeps the receiver under a
/// `tokio::sync::Mutex<Option<…>>` so `__anext__` is safe to call from
/// multiple Python tasks even though the underlying `Receiver` is
/// `!Sync`.
#[pyclass(name = "SubscriptionHandle", module = "atomrworlds")]
struct PySubscriptionHandle {
    receiver: Arc<tokio::sync::Mutex<Option<tokio::sync::mpsc::Receiver<Envelope<WorldEvent>>>>>,
    sub_id: u64,
}

#[pymethods]
impl PySubscriptionHandle {
    /// `async for ev in handle` returns `self`.
    fn __aiter__(slf: pyo3::Py<Self>) -> pyo3::Py<Self> {
        slf
    }

    /// Resolve the next event or raise `StopAsyncIteration`.
    fn __anext__<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let recv = self.receiver.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            let mut guard = recv.lock().await;
            let next = match guard.as_mut() {
                Some(rx) => rx.recv().await,
                None => None,
            };
            match next {
                None => {
                    *guard = None;
                    Err(pyo3::exceptions::PyStopAsyncIteration::new_err(
                        "subscription stream ended",
                    ))
                }
                Some(env) => Python::with_gil(|py| Ok(world_event_to_py(py, env.body)?.unbind())),
            }
        })
    }

    /// `await handle.next()` returns the next event without `async for`
    /// semantics — useful for one-shot reads.
    fn next<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        self.__anext__(py)
    }

    /// Subscription id, mirrored from the originating Subscribe envelope.
    #[getter]
    fn sub_id(&self) -> u64 {
        self.sub_id
    }
}

fn world_event_to_py(py: Python<'_>, ev: WorldEvent) -> PyResult<Bound<'_, pyo3::types::PyDict>> {
    use pyo3::types::PyDict;
    let dict = PyDict::new_bound(py);
    match ev {
        WorldEvent::BrickSnapshot { addr: _, brick, lod, payload } => {
            dict.set_item("kind", "snapshot")?;
            dict.set_item("brick", (brick.x, brick.y, brick.z))?;
            dict.set_item("lod", lod.depth)?;
            dict.set_item(
                "payload",
                pyo3::types::PyBytes::new_bound(py, payload.as_ref()),
            )?;
        }
        WorldEvent::VoxelDelta { addr: _, pos, before, after } => {
            dict.set_item("kind", "delta")?;
            dict.set_item("pos", (pos.x, pos.y, pos.z))?;
            dict.set_item("before", before.0)?;
            dict.set_item("after", after.0)?;
        }
        WorldEvent::StreamEnd { sub_id } => {
            dict.set_item("kind", "stream_end")?;
            dict.set_item("sub_id", sub_id)?;
        }
        WorldEvent::Voxel { addr: _, pos, voxel } => {
            // Shouldn't appear on a subscribe stream but fold it for safety.
            dict.set_item("kind", "voxel")?;
            dict.set_item("pos", (pos.x, pos.y, pos.z))?;
            dict.set_item("material", voxel.0)?;
        }
        WorldEvent::Ack { addr: _ } => {
            dict.set_item("kind", "ack")?;
        }
        other => {
            // Catch-all for variants the Python API doesn't model yet
            // (Tier, RegionDelta, VehicleFrame, …). Surface as opaque
            // debug repr so callers don't silently miss messages.
            dict.set_item("kind", "other")?;
            dict.set_item("debug", format!("{other:?}"))?;
        }
    }
    Ok(dict)
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
    m.add_class::<PySubscriptionHandle>()?;
    m.add("BRICK_EDGE", BRICK_EDGE)?;
    Ok(())
}
