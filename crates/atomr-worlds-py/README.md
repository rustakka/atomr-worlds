# atomrworlds (Python bindings)

Python interface for [`atomr-worlds`](../../README.md) via PyO3 + maturin.

Exposes the determinism-critical primitives (`WorldAddr`, `LevelKey`, `Lod`,
`MetricScale`, `Voxel`, `Brick`, `splitmix64`, `child_seed`) and a
`LocalHost`-backed `WorldClient` with `get_voxel`, `get_brick` (LOD-aware),
`write_voxel`, and `shutdown`. The client owns its own Tokio runtime and
in-process `LocalHost`; writes go through the host's persistence overlay when
the Rust side was constructed with one.

## Build

```sh
python3 -m venv .venv
source .venv/bin/activate
pip install maturin
maturin develop -m crates/atomr-worlds-py/Cargo.toml
```

`maturin develop` compiles the Rust extension and installs it as an editable
package named `atomrworlds`.

## Usage

```python
import atomrworlds as aw

# Seed math
chain = aw.WorldAddr.root().seed_chain(0xDEAD_BEEF)
print(chain)  # [u_seed, g_seed, s_seed, sy_seed, w_seed]

# Metric scales (per-tier root size + default depth)
world_scale = aw.MetricScale.default_world()
print(world_scale.leaf_size_m())  # ~0.6 m per voxel at max depth

# Query a generated world
client = aw.WorldClient(root_seed=0xDEAD_BEEF_CAFE_F00D)
brick = client.get_brick(aw.WorldAddr.root(), 0, -2, 0)  # lod_depth=0 by default
print(f"brick has {brick.nonempty_count()} non-air voxels")

# Reshape into a NumPy array (16x16x16 uint16)
arr = aw.brick_to_numpy(brick)

# Writes are propagated to subscribers (Rust side); reads see them
client.write_voxel(aw.WorldAddr.root(), 1, 2, 3, aw.Voxel(material=99))
assert client.get_voxel(aw.WorldAddr.root(), 1, 2, 3).material == 99

client.shutdown()
```

## Testing

```sh
python -m pytest python/tests/test_smoke.py
# or directly:
python python/tests/test_smoke.py
```

## Determinism

The seed chain, hashes, and noise functions produce identical bytes regardless
of caller language. See [`../../docs/PHASES.md`](../../docs/PHASES.md) for the
cross-language determinism contract.
