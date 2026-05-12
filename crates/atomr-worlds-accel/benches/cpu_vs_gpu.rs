//! Criterion bench: CPU vs CUDA on a representative mix of worlds.
//!
//! Requires `--features cuda` and a CUDA-capable host. Run with:
//!
//! ```sh
//! cargo bench -p atomr-worlds-accel --features cuda --bench cpu_vs_gpu
//! ```

#![cfg(feature = "cuda")]

use atomr_worlds_accel::{Accelerator, CpuAccelerator, CudaAccelerator};
use atomr_worlds_core::coord::IVec3;
use atomr_worlds_generate::{TerrainConfig, TerrainGenerator};
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};

const SEED: u64 = 0xDEAD_BEEF_CAFE_F00D;

fn coords(n: usize) -> Vec<IVec3> {
    (0..n)
        .map(|i| {
            let s = i as i64;
            IVec3::new(s % 8, -(s / 8) % 4, (s * 3) % 8)
        })
        .collect()
}

fn bench_backends(c: &mut Criterion) {
    let cfg = TerrainConfig::default();
    let cpu = CpuAccelerator::new(TerrainGenerator::new(cfg));

    let rt = tokio::runtime::Builder::new_multi_thread().enable_all().build().unwrap();
    let gpu = rt
        .block_on(CudaAccelerator::new(0, cfg))
        .expect("cuda init (set CUDA_VISIBLE_DEVICES if needed)");

    for &n in &[1usize, 8, 64, 256] {
        let cs = coords(n);
        let mut group = c.benchmark_group("fill_bricks_batch");
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("cpu", n), &cs, |b, cs| {
            b.iter(|| cpu.fill_bricks_batch(SEED, cs))
        });
        group.bench_with_input(BenchmarkId::new("cuda", n), &cs, |b, cs| {
            b.iter(|| gpu.fill_bricks_batch(SEED, cs))
        });
        group.finish();
    }
}

criterion_group!(benches, bench_backends);
criterion_main!(benches);
