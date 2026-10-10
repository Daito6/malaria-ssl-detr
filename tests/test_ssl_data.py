"""SSL two-view data tests (synthetic cache; no real data needed)."""
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from malaria_ssl.ssl.data import RandomImageSampler, TwoViewDataset, box_iou, sample_windows
from malaria_ssl.utils.config import load_config, resolve_path

VCFG = load_config(resolve_path("configs/ssl/data.yaml"))["views"]


@pytest.fixture(scope="module")
def cache(tmp_path_factory):
    path = tmp_path_factory.mktemp("ssl_cache") / "train_960.npy"
    arr = np.random.default_rng(0).integers(0, 256, (4, 720, 960, 3), dtype=np.uint8)
    np.save(path, arr)
    return path


def test_views_shape_dtype_and_normalization(cache):
    ds = TwoViewDataset(cache, VCFG)
    v1, v2, win = ds[(1, 123)]
    assert v1.shape == v2.shape == (3, 224, 224) and v1.dtype == torch.float32
    assert win.shape == (2, 4)
    assert not torch.equal(v1, v2)
    assert -3 < v1.mean() < 3


def test_crop_windows_inside_anchor_and_image():
    rng = np.random.default_rng(0)
    for _ in range(500):
        L, anchor, crops = sample_windows(rng, 720, 960, 960, VCFG)
        a = VCFG["anchor"] * 960 / L
        assert np.isclose(anchor[2] - anchor[0], a) and anchor[0] >= 0 and anchor[3] <= 720
        for c in crops:
            assert c[0] >= anchor[0] - 1e-9 and c[1] >= anchor[1] - 1e-9
            assert c[2] <= anchor[2] + 1e-9 and c[3] <= anchor[3] + 1e-9
            frac = (c[2] - c[0]) * (c[3] - c[1]) / a**2
            assert VCFG["crop_scale"][0] - 1e-9 <= frac <= 1 + 1e-9


def test_two_views_overlap():
    """Worst case for area >= 0.5 and ratio in [3/4, 4/3]: two 0.61 x 0.82 crops in opposite
    corners of the anchor -> IoU = 0.224 * 0.632 / (1 - 0.142) = 0.165."""
    rng = np.random.default_rng(1)
    ious = [box_iou(*sample_windows(rng, 720, 960, 960, VCFG)[2]) for _ in range(2000)]
    assert min(ious) >= 0.16, min(ious)
    assert np.median(ious) > 0.4


def test_same_seed_same_first_batch_any_workers(cache):
    def first_batch(workers):
        ds = TwoViewDataset(cache, VCFG)
        sampler = RandomImageSampler(len(ds), samples_per_epoch=8, seed=42)
        sampler.set_epoch(2)
        loader = DataLoader(ds, batch_size=4, sampler=sampler, num_workers=workers)
        return next(iter(loader))

    a, b, c = first_batch(0), first_batch(0), first_batch(2)
    for x, y, z in zip(a, b, c):
        assert torch.equal(x, y) and torch.equal(x, z)


def test_sampler_depends_on_epoch_only():
    s = RandomImageSampler(10, 20, seed=1)
    e0 = list(s)
    s.set_epoch(1)
    assert list(s) != e0
    s.set_epoch(0)
    assert list(s) == e0 and len(e0) == 20 and all(0 <= i < 10 for i, _ in e0)
