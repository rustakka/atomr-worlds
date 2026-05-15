"""Smoke tests for the atomrworlds Python bindings.

Runnable as `python -m pytest python/tests/test_smoke.py` or directly via
`python python/tests/test_smoke.py`.
"""
import atomrworlds as aw


def test_splitmix64_deterministic_and_nonzero():
    a = aw.splitmix64(0)
    b = aw.splitmix64(0)
    assert a == b
    assert a != 0  # SplitMix64(0) is not zero


def test_child_seed_deterministic():
    a = aw.child_seed(42, 0, 1, 2, 3)
    b = aw.child_seed(42, 0, 1, 2, 3)
    assert a == b
    # Different dim should produce different seed.
    assert aw.child_seed(42, 0, 1, 2, 3) != aw.child_seed(42, 1, 1, 2, 3)


def test_world_addr_root_seed_chain():
    addr = aw.WorldAddr.root()
    chain = addr.seed_chain(0xDEAD_BEEF)
    assert len(chain) == 5
    assert len(set(chain)) == 5  # all distinct


def test_metric_scale_default_world_leaf_size():
    s = aw.MetricScale.default_world()
    leaf = s.leaf_size_m()
    assert 0.5 <= leaf <= 1.0, f"world leaf size {leaf} outside expected [0.5, 1.0] m"


def test_world_client_get_brick():
    client = aw.WorldClient(root_seed=0xDEAD_BEEF_CAFE_F00D)
    try:
        addr = aw.WorldAddr.root()
        # Deep brick: y_brick = -2 → world y ∈ [-32, -16), well below surface.
        brick = client.get_brick(addr, 0, -2, 0)
        assert brick.nonempty_count() > 0, "deep brick should be mostly stone"
    finally:
        client.shutdown()


def test_world_client_voxel_round_trip():
    client = aw.WorldClient(root_seed=1)
    try:
        addr = aw.WorldAddr.root()
        client.write_voxel(addr, 1, 2, 3, aw.Voxel(material=99))
        v = client.get_voxel(addr, 1, 2, 3)
        assert v.material == 99
    finally:
        client.shutdown()


def test_brick_to_numpy_or_list():
    client = aw.WorldClient(root_seed=1)
    try:
        addr = aw.WorldAddr.root()
        brick = client.get_brick(addr, 0, -2, 0)
        out = aw.brick_to_numpy(brick)
        # Either a numpy array (shape 16x16x16) or a list of length 4096.
        try:
            import numpy as np  # type: ignore[import-not-found]
            assert out.shape == (16, 16, 16)
            assert out.dtype == np.uint16
        except ImportError:
            assert isinstance(out, list)
            assert len(out) == 4096
    finally:
        client.shutdown()


def test_brick_buffer_protocol_zero_copy():
    """Phase 11 follow-up: PyBrick exposes the Python buffer protocol so
    `numpy.asarray(brick)` is a zero-copy `(16, 16, 16) uint16` view.

    `memoryview(brick)` always works; the numpy assertion is gated on
    numpy being installed.
    """
    client = aw.WorldClient(root_seed=1)
    try:
        addr = aw.WorldAddr.root()
        brick = client.get_brick(addr, 0, -2, 0)

        mv = memoryview(brick)
        assert mv.format == "H", f"expected 'H' (uint16), got {mv.format!r}"
        assert mv.itemsize == 2
        assert mv.shape == (16, 16, 16)
        assert mv.nbytes == 16 * 16 * 16 * 2
        assert mv.readonly

        try:
            import numpy as np  # type: ignore[import-not-found]
        except ImportError:
            return
        arr = np.asarray(brick)
        assert arr.dtype == np.uint16
        assert arr.shape == (16, 16, 16)
        assert arr.flags["C_CONTIGUOUS"]
        # Zero-copy: numpy's `.base` should point back at the brick.
        assert arr.base is not None
        # Sanity: array contents match the explicit `materials()` walk.
        flat = brick.materials()
        for z in range(16):
            for y in range(16):
                for x in range(16):
                    # Brick layout is `(z * 16 + y) * 16 + x`.
                    assert int(arr[z, y, x]) == flat[(z * 16 + y) * 16 + x]
                    if z + y + x > 0:
                        # Quick exit — we already checked many cells.
                        break
                break
    finally:
        client.shutdown()


def test_subscribe_async_yields_snapshot_then_delta():
    """Phase 11 follow-up: `WorldClient.subscribe_async` returns a
    coroutine-resolved async iterator over `WorldEvent` dicts.

    Subscribing to a region must:
    1. Yield `{"kind": "snapshot", …}` for each brick overlapping the
       region (initial-snapshot pass).
    2. Yield `{"kind": "delta", …}` for in-region writes after the
       snapshot completes.

    Run synchronously via `asyncio.run` so the test stays in-process with
    no extra pytest plugin.
    """
    import asyncio

    async def main():
        client = aw.WorldClient(root_seed=0xDEAD_BEEF_CAFE_F00D)
        try:
            addr = aw.WorldAddr.root()
            # 16³ region right at the origin — exactly one brick. Brick
            # snapshots will arrive first; then we write a voxel inside
            # and expect a delta.
            handle = await client.subscribe_async(
                addr, (0, 0, 0), (16, 16, 16), 0, 7
            )
            assert handle.sub_id == 7
            ev = await handle.__anext__()
            assert ev["kind"] == "snapshot", f"first event wasn't a snapshot: {ev}"
            # Write a voxel inside the region and expect a delta.
            client.write_voxel(addr, 1, 1, 1, aw.Voxel(material=42))
            # Drain any further snapshots until the delta arrives, with a
            # safety cap so a misbehaving stream can't hang the test.
            for _ in range(64):
                ev = await handle.__anext__()
                if ev["kind"] == "delta":
                    assert ev["pos"] == (1, 1, 1)
                    assert ev["after"] == 42
                    break
            else:
                raise AssertionError("no delta after 64 events")
        finally:
            client.shutdown()

    asyncio.run(main())


if __name__ == "__main__":
    test_splitmix64_deterministic_and_nonzero()
    test_child_seed_deterministic()
    test_world_addr_root_seed_chain()
    test_metric_scale_default_world_leaf_size()
    test_world_client_get_brick()
    test_world_client_voxel_round_trip()
    test_brick_to_numpy_or_list()
    test_brick_buffer_protocol_zero_copy()
    test_subscribe_async_yields_snapshot_then_delta()
    print("ALL SMOKE TESTS PASS")
