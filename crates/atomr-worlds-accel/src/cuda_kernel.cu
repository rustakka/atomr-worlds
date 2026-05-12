// NVRTC-compiled brick generator. Mirrors the CPU implementation in
// `atomr-worlds-generate::TerrainGenerator` and the noise primitives in
// `atomr-worlds-noise`. The byte-identical determinism contract requires:
//
//   * Same integer arithmetic (splitmix64, rotations) — exact at the bit level
//     because we only use 64-bit unsigned multiplies, shifts, and xors.
//   * Same f32 op ordering. NVCC's FMA fusion can drift last-bit results, so
//     the host compiles with `--fmad=false` (set via `NvrtcOpts::extra_options`).

extern "C" __device__ __forceinline__
unsigned long long splitmix64(unsigned long long z) {
    z = z + 0x9E3779B97F4A7C15ULL;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

extern "C" __device__ __forceinline__
unsigned long long rotl_u64(unsigned long long x, unsigned shift) {
    return (x << shift) | (x >> (64 - shift));
}

extern "C" __device__ __forceinline__
unsigned long long mix3(unsigned long long seed, long long x, long long y, long long z) {
    unsigned long long h = splitmix64(seed);
    h = splitmix64(h ^ (unsigned long long)x);
    h = splitmix64(h ^ rotl_u64((unsigned long long)y, 21));
    return splitmix64(h ^ rotl_u64((unsigned long long)z, 42));
}

extern "C" __device__ __forceinline__
float hash3_f01(unsigned long long seed, long long x, long long y, long long z) {
    unsigned int top = (unsigned int)(mix3(seed, x, y, z) >> 40);
    return ((float)top) / ((float)(1u << 24));
}

extern "C" __device__ __forceinline__
float smoothstep_(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
}

extern "C" __device__ __forceinline__
float lerp_(float a, float b, float t) {
    return a + (b - a) * t;
}

extern "C" __device__ __forceinline__
long long floor_i64(float f) {
    return (long long)floorf(f);
}

extern "C" __device__
float value_noise_3d(unsigned long long seed, float x, float y, float z) {
    long long xi = floor_i64(x);
    long long yi = floor_i64(y);
    long long zi = floor_i64(z);
    float fx = x - (float)xi;
    float fy = y - (float)yi;
    float fz = z - (float)zi;

    float c000 = hash3_f01(seed, xi,     yi,     zi);
    float c100 = hash3_f01(seed, xi + 1, yi,     zi);
    float c010 = hash3_f01(seed, xi,     yi + 1, zi);
    float c110 = hash3_f01(seed, xi + 1, yi + 1, zi);
    float c001 = hash3_f01(seed, xi,     yi,     zi + 1);
    float c101 = hash3_f01(seed, xi + 1, yi,     zi + 1);
    float c011 = hash3_f01(seed, xi,     yi + 1, zi + 1);
    float c111 = hash3_f01(seed, xi + 1, yi + 1, zi + 1);

    float u = smoothstep_(fx);
    float v = smoothstep_(fy);
    float w = smoothstep_(fz);

    float x00 = lerp_(c000, c100, u);
    float x10 = lerp_(c010, c110, u);
    float x01 = lerp_(c001, c101, u);
    float x11 = lerp_(c011, c111, u);

    float y0 = lerp_(x00, x10, v);
    float y1 = lerp_(x01, x11, v);

    return lerp_(y0, y1, w);
}

extern "C" __device__
float worley_noise_3d(unsigned long long seed, float x, float y, float z) {
    long long xi = floor_i64(x);
    long long yi = floor_i64(y);
    long long zi = floor_i64(z);
    float best = 1e30f;
    for (int dz = -1; dz <= 1; ++dz) {
        for (int dy = -1; dy <= 1; ++dy) {
            for (int dx = -1; dx <= 1; ++dx) {
                long long cx = xi + dx;
                long long cy = yi + dy;
                long long cz = zi + dz;
                float fx = hash3_f01(seed ^ 0x1111ULL, cx, cy, cz) + (float)cx;
                float fy = hash3_f01(seed ^ 0x2222ULL, cx, cy, cz) + (float)cy;
                float fz = hash3_f01(seed ^ 0x3333ULL, cx, cy, cz) + (float)cz;
                float dxv = fx - x;
                float dyv = fy - y;
                float dzv = fz - z;
                float d2 = dxv * dxv + dyv * dyv + dzv * dzv;
                if (d2 < best) best = d2;
            }
        }
    }
    return best;
}

extern "C" __device__
float fbm_value(unsigned long long seed, float x, float y, float z,
                unsigned octaves, float lacunarity, float gain, float frequency) {
    float amp = 1.0f, freq = frequency, sum = 0.0f, total = 0.0f;
    for (unsigned o = 0; o < octaves; ++o) {
        unsigned long long octave_seed = (seed + (unsigned long long)o) * 0x9E3779B97F4A7C15ULL;
        sum += amp * value_noise_3d(octave_seed, x * freq, y * freq, z * freq);
        total += amp;
        amp *= gain;
        freq *= lacunarity;
    }
    return total > 0.0f ? (sum / total) : 0.0f;
}

extern "C" __device__
unsigned short material_at(
    unsigned long long seed,
    long long x, long long y, long long z,
    float base_height, float amplitude, float frequency, unsigned octaves,
    float cave_threshold, float cave_frequency, unsigned dirt_layer
) {
    float n = fbm_value(seed,
                        (float)x * frequency, 0.0f, (float)z * frequency,
                        octaves, 2.0f, 0.5f, 1.0f);
    float surface = base_height + amplitude * (n * 2.0f - 1.0f);
    float fy = (float)y;
    if (fy >= surface) return 0;
    float d2 = worley_noise_3d(seed + 0xC0FEE0C0ULL,
                                (float)x * cave_frequency,
                                (float)y * cave_frequency,
                                (float)z * cave_frequency);
    if (d2 < cave_threshold) return 0;
    if (fy >= surface - (float)dirt_layer) return 2;
    return 1;
}

// grid = (num_bricks, 16, 1), block = (16, 16, 1)
// brick layout in `output`: idx = brick_id * 4096 + (z * 16 + y) * 16 + x
extern "C" __global__
void fill_bricks(
    unsigned int num_bricks,
    unsigned long long world_seed,
    const long long* __restrict__ brick_coords,
    float base_height, float amplitude, float frequency, unsigned int octaves,
    float cave_threshold, float cave_frequency, unsigned int dirt_layer,
    unsigned int* __restrict__ output
) {
    unsigned int brick_id = blockIdx.x;
    if (brick_id >= num_bricks) return;
    unsigned int x = threadIdx.x;
    unsigned int y = threadIdx.y;
    unsigned int z = blockIdx.y;
    if (x >= 16u || y >= 16u || z >= 16u) return;

    long long bcx = brick_coords[brick_id * 3 + 0];
    long long bcy = brick_coords[brick_id * 3 + 1];
    long long bcz = brick_coords[brick_id * 3 + 2];

    long long wx = bcx * 16 + (long long)x;
    long long wy = bcy * 16 + (long long)y;
    long long wz = bcz * 16 + (long long)z;

    unsigned short mat = material_at(
        world_seed, wx, wy, wz,
        base_height, amplitude, frequency, octaves,
        cave_threshold, cave_frequency, dirt_layer
    );

    unsigned int local_idx = (z * 16u + y) * 16u + x;
    output[brick_id * 4096u + local_idx] = (unsigned int)mat;
}
