//! Hash-quality checks for `child_seed`: avalanche and low-byte distribution.
//!
//! These are deterministic (seeded splitmix64 PRNG) and run in <2s.

use atomr_worlds_core::coord::IVec3;
use atomr_worlds_core::seed::{child_seed, splitmix64};

/// Tiny deterministic PRNG built on splitmix64.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed.wrapping_add(0x12345))
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0xA0B1_C2D3_E4F5_0617);
        splitmix64(self.0)
    }
    fn next_i64(&mut self) -> i64 {
        self.next_u64() as i64
    }
    fn next_u32(&mut self) -> u32 {
        self.next_u64() as u32
    }
}

fn random_inputs(rng: &mut Rng) -> (u64, u32, IVec3) {
    (rng.next_u64(), rng.next_u32(), IVec3::new(rng.next_i64(), rng.next_i64(), rng.next_i64()))
}

#[test]
fn avalanche_meets_threshold() {
    // For each of 5 perturbation sites (parent bit, dim bit, x bit, y bit, z bit),
    // flip a random bit, recompute, count Hamming distance, and accumulate.
    const SAMPLES_PER_SITE: usize = 2_000;

    let mut rng = Rng::new(0xC0DE_F00D_DEAD_BEEF);
    let mut total_flipped = 0u64;
    let mut total_bits = 0u64;

    for site in 0..5 {
        for _ in 0..SAMPLES_PER_SITE {
            let (p, d, c) = random_inputs(&mut rng);
            let a = child_seed(p, d, c);
            let (p2, d2, c2) = match site {
                0 => (p ^ (1u64 << (rng.next_u32() % 64)), d, c),
                1 => (p, d ^ (1u32 << (rng.next_u32() % 32)), c),
                2 => (p, d, IVec3::new(c.x ^ (1i64 << (rng.next_u32() % 64) as i64), c.y, c.z)),
                3 => (p, d, IVec3::new(c.x, c.y ^ (1i64 << (rng.next_u32() % 64) as i64), c.z)),
                _ => (p, d, IVec3::new(c.x, c.y, c.z ^ (1i64 << (rng.next_u32() % 64) as i64))),
            };
            let b = child_seed(p2, d2, c2);
            total_flipped += (a ^ b).count_ones() as u64;
            total_bits += 64;
        }
    }

    let ratio = total_flipped as f64 / total_bits as f64;
    // SplitMix64 typically achieves ~0.5 (ideal); allow slack for the lower
    // bound. 0.40 is a comfortable floor.
    assert!(
        ratio >= 0.40,
        "avalanche ratio {ratio:.4} below threshold 0.40 (flipped {total_flipped} / {total_bits})"
    );
}

#[test]
fn low_byte_distribution_is_uniform() {
    const SAMPLES: usize = 500_000;
    const BUCKETS: usize = 256;
    let expected = (SAMPLES / BUCKETS) as f64; // 1953.125

    let mut rng = Rng::new(0xFACE_BEEF_F00D_CAFE);
    let mut buckets = [0u32; BUCKETS];

    for _ in 0..SAMPLES {
        let (p, d, c) = random_inputs(&mut rng);
        let h = child_seed(p, d, c);
        buckets[(h & 0xFF) as usize] += 1;
    }

    // Natural standard deviation at 500k samples / 256 buckets is √(N·p·(1−p))
    // ≈ 44, so ~5σ ≈ 220 (about 11% of `expected`). 12% is a comfortable
    // sanity bound — we're catching gross bias, not measuring chi-square.
    let tolerance = expected * 0.12;
    let lo = expected - tolerance;
    let hi = expected + tolerance;
    let mut worst = 0i64;
    for (i, &count) in buckets.iter().enumerate() {
        let c = count as f64;
        let drift = (c - expected).abs() as i64;
        worst = worst.max(drift);
        assert!(c >= lo && c <= hi, "bucket {i} count {count} outside [{lo:.0}, {hi:.0}]");
    }
    eprintln!("worst absolute drift from uniform: {worst} (tolerance {tolerance:.0})");
}
