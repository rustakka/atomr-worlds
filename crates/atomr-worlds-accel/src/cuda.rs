//! CUDA-backed [`Accelerator`] for brick generation.
//!
//! Compiles `cuda_kernel.cu` via `atomr-accel-cuda`'s NVRTC actor at startup,
//! then dispatches batched brick fills via a single launch per call. The
//! kernel mirrors the CPU `TerrainGenerator` math line-for-line; the host
//! compiles with `--fmad=false` so FMA fusion does not drift last-bit results
//! and the GPU brick output is byte-identical to the CPU backend.
//!
//! ## Runtime contract
//!
//! [`CudaAccelerator::fill_brick`] is a CPU short-circuit — a single brick
//! is cheaper to generate locally than to round-trip through PCIe. The GPU
//! path is exercised by [`CudaAccelerator::fill_bricks_batch`] and the
//! async siblings [`CudaAccelerator::fill_brick_async`] /
//! [`CudaAccelerator::fill_bricks_batch_async`]. The blocking trait methods
//! call `Handle::block_on` on a stored runtime handle; call them from sync
//! code, or wrap in `tokio::task::spawn_blocking` from an async context.

use std::time::Duration;

use atomr::prelude::ActorSystem;
use atomr_accel_cuda::device::{
    DeviceActor, DeviceConfig, DeviceMsg, EnabledLibraries, HostBuf, KernelChildren,
};
use atomr_accel_cuda::dtype::CudaDtype;
use atomr_accel_cuda::error::GpuError;
use atomr_accel_cuda::gpu_ref::GpuRef;
use atomr_accel_cuda::kernel::{KernelArg, KernelHandle, NvrtcMsg, NvrtcOpts};
use atomr_config::Config;
use atomr_core::actor::ActorRef;
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{BrickGenerator, TerrainConfig, TerrainGenerator};
use atomr_worlds_voxel::{Brick, Voxel, BRICK_EDGE, BRICK_LEN};
use cudarc::driver::LaunchConfig;
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use crate::Accelerator;

const KERNEL_SRC: &str = include_str!("cuda_kernel.cu");
const KERNEL_NAME: &str = "fill_bricks";
const ASK_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("atomr actor system error: {0}")]
    Sys(String),
    #[error("CUDA device not ready after {0:?}")]
    DeviceNotReady(Duration),
    #[error("NVRTC actor not available (enable EnabledLibraries::NVRTC)")]
    NvrtcDisabled,
    #[error("ask error: {0}")]
    Ask(String),
    #[error("GPU error: {0}")]
    Gpu(String),
    #[error("kernel returned non-u16 voxel material: {0}")]
    BadMaterial(u32),
    #[error("no current Tokio runtime; construct CudaAccelerator from inside a runtime")]
    NoRuntime,
}

pub struct CudaAccelerator {
    _sys: ActorSystem,
    device: ActorRef<DeviceMsg>,
    nvrtc: ActorRef<NvrtcMsg>,
    kernel: KernelHandle,
    config: TerrainConfig,
    runtime: Handle,
    cpu_ref: TerrainGenerator,
}

impl std::fmt::Debug for CudaAccelerator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAccelerator")
            .field("config", &self.config)
            .field("kernel_name", &KERNEL_NAME)
            .finish_non_exhaustive()
    }
}

impl CudaAccelerator {
    /// Spin up a CUDA device actor on a private `ActorSystem`, compile the
    /// brick kernel, and return the accelerator. Must be called from inside
    /// a Tokio runtime.
    pub async fn new(device_id: u32, config: TerrainConfig) -> Result<Self, CudaError> {
        let sys = ActorSystem::create("atomr-worlds-cuda", Config::empty())
            .await
            .map_err(|e| CudaError::Sys(format!("{e}")))?;
        Self::new_with_system(sys, device_id, config).await
    }

    /// Same as [`Self::new`] but the caller owns the `ActorSystem`.
    pub async fn new_with_system(
        sys: ActorSystem,
        device_id: u32,
        config: TerrainConfig,
    ) -> Result<Self, CudaError> {
        let dev_cfg = DeviceConfig::new(device_id)
            .with_libraries(EnabledLibraries::NVRTC | EnabledLibraries::BLAS);
        let device = sys
            .actor_of(DeviceActor::props(dev_cfg), &format!("atomr-worlds-cuda-d{device_id}"))
            .map_err(|e| CudaError::Sys(format!("{e}")))?;

        let children = wait_for_children(&device, Duration::from_secs(10)).await?;
        let nvrtc = children.nvrtc.clone().ok_or(CudaError::NvrtcDisabled)?;

        let mut opts = NvrtcOpts::default();
        opts.extra_options.push("--fmad=false".to_string());
        let kernel = compile_kernel(&nvrtc, opts).await?;

        let runtime = Handle::try_current().map_err(|_| CudaError::NoRuntime)?;
        let cpu_ref = TerrainGenerator::new(config);
        Ok(Self { _sys: sys, device, nvrtc, kernel, config, runtime, cpu_ref })
    }

    pub async fn fill_brick_async(
        &self,
        world_seed: u64,
        brick_coord: IVec3,
    ) -> Result<Brick, CudaError> {
        let mut bricks = self.fill_bricks_batch_async(world_seed, &[brick_coord]).await?;
        Ok(bricks.pop().expect("one brick"))
    }

    pub async fn fill_bricks_batch_async(
        &self,
        world_seed: u64,
        brick_coords: &[IVec3],
    ) -> Result<Vec<Brick>, CudaError> {
        if brick_coords.is_empty() {
            return Ok(Vec::new());
        }
        let n = brick_coords.len();

        let mut flat: Vec<i64> = Vec::with_capacity(n * 3);
        for c in brick_coords {
            flat.push(c.x);
            flat.push(c.y);
            flat.push(c.z);
        }
        let coords_dev = self.alloc::<i64>(flat.len()).await?;
        let _ = self.copy_from_host::<i64>(HostBuf::Owned(flat), coords_dev.clone()).await?;

        let out_len = n * BRICK_LEN;
        let out_dev = self.alloc::<u32>(out_len).await?;

        let cfg = LaunchConfig {
            grid_dim: (n as u32, BRICK_EDGE as u32, 1),
            block_dim: (BRICK_EDGE as u32, BRICK_EDGE as u32, 1),
            shared_mem_bytes: 0,
        };
        let p = self.config;
        let args: Vec<KernelArg> = vec![
            KernelArg::Scalar(Box::new(n as u32)),
            KernelArg::Scalar(Box::new(world_seed)),
            KernelArg::DevSlice(Box::new(coords_dev)),
            KernelArg::Scalar(Box::new(p.base_height)),
            KernelArg::Scalar(Box::new(p.amplitude)),
            KernelArg::Scalar(Box::new(p.frequency)),
            KernelArg::Scalar(Box::new(p.octaves as u32)),
            KernelArg::Scalar(Box::new(p.cave_threshold)),
            KernelArg::Scalar(Box::new(p.cave_frequency)),
            KernelArg::Scalar(Box::new(p.dirt_layer as u32)),
            KernelArg::DevSlice(Box::new(out_dev.clone())),
        ];
        launch(&self.nvrtc, self.kernel.clone(), args, cfg).await?;

        let host = self
            .copy_to_host::<u32>(out_dev, HostBuf::Owned(vec![0u32; out_len]))
            .await?;
        let slice: Vec<u32> = match host {
            HostBuf::Owned(v) => v,
            HostBuf::Pinned(_) => unreachable!("Owned back"),
        };

        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * BRICK_LEN;
            let mut b = Brick::new();
            for z in 0..(BRICK_EDGE as i64) {
                for y in 0..(BRICK_EDGE as i64) {
                    for x in 0..(BRICK_EDGE as i64) {
                        let idx = base
                            + ((z as usize * BRICK_EDGE) + y as usize) * BRICK_EDGE
                            + x as usize;
                        let raw = slice[idx];
                        if raw > u16::MAX as u32 {
                            return Err(CudaError::BadMaterial(raw));
                        }
                        if raw != 0 {
                            b.set(IVec3::new(x, y, z), Voxel::new(raw as u16));
                        }
                    }
                }
            }
            out.push(b);
        }
        Ok(out)
    }

    async fn alloc<T: CudaDtype>(&self, len: usize) -> Result<GpuRef<T>, CudaError> {
        let (tx, rx) = oneshot::channel();
        self.device.tell(DeviceMsg::alloc::<T>(len, tx));
        rx.await
            .map_err(|e| CudaError::Ask(e.to_string()))?
            .map_err(|e| CudaError::Gpu(format!("{e}")))
    }

    async fn copy_from_host<T: CudaDtype>(
        &self,
        src: HostBuf<T>,
        dst: GpuRef<T>,
    ) -> Result<HostBuf<T>, CudaError> {
        let (tx, rx) = oneshot::channel();
        self.device.tell(DeviceMsg::copy_from_host::<T>(src, dst, tx));
        rx.await
            .map_err(|e| CudaError::Ask(e.to_string()))?
            .map_err(|e| CudaError::Gpu(format!("{e}")))
    }

    async fn copy_to_host<T: CudaDtype>(
        &self,
        src: GpuRef<T>,
        dst: HostBuf<T>,
    ) -> Result<HostBuf<T>, CudaError> {
        let (tx, rx) = oneshot::channel();
        self.device.tell(DeviceMsg::copy_to_host::<T>(src, dst, tx));
        rx.await
            .map_err(|e| CudaError::Ask(e.to_string()))?
            .map_err(|e| CudaError::Gpu(format!("{e}")))
    }
}

impl Accelerator for CudaAccelerator {
    fn backend(&self) -> &'static str {
        "cuda"
    }
    fn fill_brick(&self, world_seed: u64, brick_coord: IVec3) -> Brick {
        // Single brick: PCIe latency dominates. Match GPU's output by using
        // the same algorithm on the CPU reference.
        self.cpu_ref.generate_brick_legacy(world_seed, brick_coord)
    }
    fn fill_bricks_batch(&self, world_seed: u64, brick_coords: &[IVec3]) -> Vec<Brick> {
        self.runtime
            .block_on(self.fill_bricks_batch_async(world_seed, brick_coords))
            .expect("cuda batch fill")
    }
}

async fn wait_for_children(
    device: &ActorRef<DeviceMsg>,
    timeout: Duration,
) -> Result<KernelChildren, CudaError> {
    let start = std::time::Instant::now();
    loop {
        let (tx, rx) = oneshot::channel();
        device.tell(DeviceMsg::SnapshotChildren { reply: tx });
        let res = rx.await.map_err(|e| CudaError::Ask(e.to_string()))?;
        if let Some(c) = res {
            return Ok(c);
        }
        if start.elapsed() > timeout {
            return Err(CudaError::DeviceNotReady(timeout));
        }
        tokio::time::sleep(Duration::from_millis(50)).await;
    }
}

async fn compile_kernel(
    nvrtc: &ActorRef<NvrtcMsg>,
    opts: NvrtcOpts,
) -> Result<KernelHandle, CudaError> {
    nvrtc
        .ask_with(
            |reply| NvrtcMsg::Compile {
                src: KERNEL_SRC.to_string(),
                kernel_name: KERNEL_NAME.to_string(),
                opts,
                reply,
            },
            ASK_TIMEOUT,
        )
        .await
        .map_err(|e| CudaError::Ask(e.to_string()))?
        .map_err(|e: GpuError| CudaError::Gpu(format!("{e}")))
}

async fn launch(
    nvrtc: &ActorRef<NvrtcMsg>,
    kernel: KernelHandle,
    args: Vec<KernelArg>,
    cfg: LaunchConfig,
) -> Result<(), CudaError> {
    nvrtc
        .ask_with(
            |reply| NvrtcMsg::Launch { kernel, args, cfg, reply },
            ASK_TIMEOUT,
        )
        .await
        .map_err(|e| CudaError::Ask(e.to_string()))?
        .map_err(|e: GpuError| CudaError::Gpu(format!("{e}")))
}
