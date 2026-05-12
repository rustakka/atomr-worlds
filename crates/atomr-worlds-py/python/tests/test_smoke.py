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


if __name__ == "__main__":
    test_splitmix64_deterministic_and_nonzero()
    test_child_seed_deterministic()
    test_world_addr_root_seed_chain()
    test_metric_scale_default_world_leaf_size()
    test_world_client_get_brick()
    test_world_client_voxel_round_trip()
    test_brick_to_numpy_or_list()
    print("ALL SMOKE TESTS PASS")
