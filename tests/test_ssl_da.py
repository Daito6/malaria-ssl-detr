"""Phase 5 (self-supervised domain adaptation): folds, target cache, mixed sampling, encoder init (CPU, synthetic)."""
import builtins
import copy
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
import yaml

import malaria_ssl.ssl.data as ssl_data
from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.ssl.data import (MixedImageSampler, MixedTwoViewDataset, TwoViewDataset, build_target_cache,
                                  load_fold_images, make_folds)
from malaria_ssl.ssl.encoders import encoder_input_norm, init_encoder, is_imagenet_norm, load_encoder
from malaria_ssl.ssl.models import build_encoder
from malaria_ssl.ssl.probe import subset_images
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

VCFG = load_config(resolve_path("configs/ssl/data.yaml"))["views"]
HALF = ([0.5] * 3, [0.5] * 3)


def test_make_folds_deterministic_disjoint_balanced():
    ids = list(range(100, 221))                      # 121 images (odd)
    rng = np.random.default_rng(0)
    n_inf = {i: int(rng.poisson(3)) for i in ids}
    a, b = make_folds(ids, n_inf, 42), make_folds(ids, n_inf, 42)
    assert a == b and set(a) == set(ids) and set(a.values()) == {1, 2}
    sizes = [sum(f == k for f in a.values()) for k in (1, 2)]
    assert abs(sizes[0] - sizes[1]) <= 1
    inf = [sum(n_inf[i] for i in ids if a[i] == k) for k in (1, 2)]
    assert abs(inf[0] - inf[1]) <= max(n_inf.values()) * 2
    assert make_folds(ids, n_inf, 7) != a


@pytest.fixture
def raw_and_folds(tmp_path):
    raw = tmp_path / "raw" / "images"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(0)
    names = [f"img{i}.jpg" for i in range(4)]
    for n in names:
        cv2.imwrite(str(raw / n), rng.integers(0, 256, (138, 194, 3), dtype=np.uint8))
    (tmp_path / "coco").mkdir()
    (tmp_path / "coco" / "test.json").write_text("{}", encoding="utf-8")   # must never be opened
    folds = {"seed": 42, "folds": {"1": [{"id": 3, "file_name": names[3]}, {"id": 0, "file_name": names[0]}],
                                   "2": [{"id": 1, "file_name": names[1]}, {"id": 2, "file_name": names[2]}]}}
    path = tmp_path / "test_folds.json"
    path.write_text(json.dumps(folds), encoding="utf-8")
    return raw, path, names


def test_target_cache_reads_only_fold_images_and_no_annotations(tmp_path, raw_and_folds, monkeypatch):
    raw, folds_path, names = raw_and_folds
    opened = []
    real_open, real_fromfile, real_read_text = builtins.open, np.fromfile, Path.read_text
    monkeypatch.setattr(builtins, "open", lambda f, *a, **k: (opened.append(Path(str(f)).name), real_open(f, *a, **k))[1])
    monkeypatch.setattr(np, "fromfile", lambda f, *a, **k: (opened.append(Path(str(f)).name), real_fromfile(f, *a, **k))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (opened.append(self.name), real_read_text(self, *a, **k))[1])

    def no_coco(*a, **k):
        raise AssertionError("the SSL target loader must not read annotations")
    monkeypatch.setattr(ssl_data, "load_split_coco", no_coco)

    npy, idx = build_target_cache(folds_path, 1, raw, {"dir": str(tmp_path / "cache"), "long_side": 96}, Path)
    images_read = {n for n in opened if n.endswith(".jpg")}
    assert images_read == {names[0], names[3]}                       # fold 1 only
    assert not any(n.startswith("test") and n.endswith(".json") and n != "test_folds.json" for n in opened)
    arr = np.load(npy)
    assert arr.shape == (2, 68, 96, 3)
    index = json.loads(idx.read_text(encoding="utf-8"))
    assert [im["id"] for im in index["images"]] == [0, 3]            # sorted by id, no label fields
    assert all(set(im) == {"id", "file_name"} for im in index["images"])
    opened.clear()
    build_target_cache(folds_path, 1, raw, {"dir": str(tmp_path / "cache"), "long_side": 96}, Path)
    assert {n for n in opened if n.endswith(".jpg")} <= {names[0], names[3]}   # idempotent rerun, still fold 1 only


def test_fold_images_and_subset(raw_and_folds):
    _, folds_path, _ = raw_and_folds
    assert [im["id"] for im in load_fold_images(folds_path, 2)] == [1, 2]
    crops = {"image_id": np.array([0, 1, 1, 2, 3]), "label": np.arange(5), "box": np.zeros((5, 4))}
    sub = subset_images(crops, [1, 3])
    assert sub["label"].tolist() == [1, 2, 4] and sub["box"].shape == (3, 4)


def test_mixed_sampler_fraction_range_and_determinism():
    s = MixedImageSampler(100, 10, 0.5, 4000, seed=1)
    keys = list(s)
    idx = np.array([i for i, _ in keys])
    assert len(keys) == 4000 and idx.min() >= 0 and idx.max() < 110
    assert 0.45 < np.mean(idx >= 100) < 0.55
    assert list(s) == keys
    s.set_epoch(1)
    assert list(s) != keys
    assert all(i < 100 for i, _ in MixedImageSampler(100, 10, 0.0, 500, seed=1))


def test_mixed_dataset_routes_to_sources(tmp_path):
    a, b = tmp_path / "a.npy", tmp_path / "b.npy"
    np.save(a, np.zeros((3, 72, 96, 3), np.uint8))
    np.save(b, np.full((2, 68, 96, 3), 255, np.uint8))
    vcfg = {**VCFG, "anchor": 40, "out_size": 32, "train_long_side": [96, 96, 32]}
    ds = MixedTwoViewDataset([TwoViewDataset(a, vcfg, augment=False), TwoViewDataset(b, vcfg, augment=False)])
    assert len(ds) == 5
    assert ds[(2, 0)][0].mean() < 0 < ds[(3, 0)][0].mean()


def test_view_normalization_follows_input_norm(tmp_path):
    p = tmp_path / "c.npy"
    np.save(p, np.random.default_rng(0).integers(0, 256, (2, 72, 96, 3), dtype=np.uint8))
    vcfg = {**VCFG, "anchor": 40, "out_size": 32, "train_long_side": [96, 96, 32]}
    v_in = TwoViewDataset(p, vcfg, augment=False)[(0, 5)][0]
    v_half = TwoViewDataset(p, vcfg, augment=False, input_norm=HALF)[(0, 5)][0]
    x = v_in * torch.as_tensor(IMAGENET_STD).view(3, 1, 1) + torch.as_tensor(IMAGENET_MEAN).view(3, 1, 1)
    assert torch.allclose(v_half, (x - 0.5) / 0.5, atol=1e-5)


def _fake_run(tmp_path, norm):
    run = tmp_path / "ssl_fake"
    (run / "checkpoints").mkdir(parents=True)
    torch.manual_seed(3)
    enc = build_encoder("vit_small_patch16_224")
    torch.save(enc.state_dict(), run / "checkpoints" / "encoder_final.pt")
    cfg = {"model": {"encoder": "vit_small_patch16_224"}}
    if norm is not None:
        cfg["model"]["input_norm"] = {"mean": norm[0], "std": norm[1]}
    (run / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return run, enc


def test_init_from_state_dict_keeps_weights_and_run_norm(tmp_path):
    run, enc = _fake_run(tmp_path, HALF)
    path = run / "checkpoints" / "encoder_final.pt"
    new, norm = init_encoder({"encoder": "vit_small_patch16_224", "init": str(path), "drop_path_rate": 0.0},
                             "unused", Path)
    for k, v in enc.state_dict().items():
        assert torch.equal(new.state_dict()[k], v), k
    assert norm == {"mean": HALF[0], "std": HALF[1]}
    _, _, source = load_encoder({"probe": {"references": {}}}, run=str(run))
    assert source["input_mean"] == HALF[0] and source["input_std"] == HALF[1]


def test_runs_without_recorded_norm_use_imagenet(tmp_path):
    run, _ = _fake_run(tmp_path, None)
    mean, std = encoder_input_norm(run / "checkpoints" / "encoder_final.pt")
    assert is_imagenet_norm(mean, std) and not is_imagenet_norm(*HALF)
    _, _, source = load_encoder({"probe": {"references": {}}}, run=str(run))
    assert "input_mean" not in source
    _, norm = init_encoder({"encoder": "vit_small_patch16_224", "init": "random", "drop_path_rate": 0.0}, "unused", Path)
    assert is_imagenet_norm(norm["mean"], norm["std"])


def test_vit_detr_uses_the_ssl_run_norm(tmp_path):
    from malaria_ssl.detr.build import build_vit_backbone

    run, _ = _fake_run(tmp_path, HALF)
    bb = build_vit_backbone({"init": str(run / "checkpoints" / "encoder_final.pt"), "arch": "vit_small_patch16_224",
                             "stride": 16})
    assert torch.allclose(bb.scale.flatten(), torch.as_tensor(IMAGENET_STD) / 0.5)


def test_trainer_mixes_target_fold_and_saves_target_recon(tmp_path, monkeypatch):
    from malaria_ssl.ssl import trainer as tr_mod

    d = tmp_path / "cache"
    d.mkdir()
    rng = np.random.default_rng(0)
    caches = {}
    for split, n, h in (("train", 3, 720), ("val", 2, 720), ("target", 2, 683)):
        caches[split] = d / f"{split}_960.npy"
        np.save(caches[split], rng.integers(0, 256, (n, h, 960, 3), dtype=np.uint8))
    cfg = copy.deepcopy(load_experiment("configs/ssl/joint_da_f1.yaml"))
    cfg["model"]["init"] = "random"           # no download / no real run needed on CPU
    cfg["train"].update({"batch": 4, "samples_per_epoch": 8, "epochs": 1, "num_workers": 0, "amp_dtype": "none"})
    cfg["monitor"].update({"val_pairs": 6, "recon_every": 1, "recon_crops": 2, "probe_every": 0, "save_every": 0})
    t = tr_mod.SSLTrainer(cfg, tmp_path / "run", torch.device("cpu"), caches)
    t.setup(resume=False)
    assert isinstance(t.train_ds, MixedTwoViewDataset) and len(t.train_ds) == 5
    assert isinstance(t.sampler, MixedImageSampler) and t.sampler.fraction == 0.5 and t.sampler.n_source == 3
    t.fit()
    assert (tmp_path / "run" / "recon" / "target_epoch_001.png").exists()
    saved = yaml.safe_load((tmp_path / "run" / "config.yaml").read_text(encoding="utf-8"))
    assert saved["target"]["fold"] == 1 and is_imagenet_norm(saved["model"]["input_norm"]["mean"],
                                                             saved["model"]["input_norm"]["std"])


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def test_phase5_configs_differ_only_in_declared_keys():
    load = lambda n: _flat(load_experiment(f"configs/ssl/{n}.yaml"))   # noqa: E731
    base, ic, jc = load("joint"), load("imagenet_cont"), load("joint_cont")
    diff = lambda a, b: {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}   # noqa: E731
    assert diff(base, ic) == {"model.init", "optim.blr", "train.epochs"}
    assert ic["model.init"] == "imagenet" and ic["train.epochs"] == 30
    assert diff(ic, jc) == {"model.init"} and jc["model.init"] == "runs/ssl_joint/checkpoints/encoder_final.pt"
    for enc, cont in (("imagenet", ic), ("joint", jc)):
        for f in (1, 2):
            da = load(f"{enc}_da_f{f}")
            assert diff(cont, da) == {"target.folds_file", "target.fold", "target.fraction"}
            assert da["target.fold"] == f and da["target.fraction"] == 0.5


def test_arm_configs_differ_only_in_declared_keys():
    load = lambda n: _flat(load_experiment(f"configs/ssl/{n}.yaml"))   # noqa: E731
    diff = lambda a, b: {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}   # noqa: E731
    for f in (1, 2):
        da = load(f"imagenet_da_f{f}")
        kang = load(f"imagenet_da_kang_f{f}")
        assert diff(da, kang) == {"views.color_jitter.hue", "views.grayscale.p"}   # augmentation keys only
        assert kang["views.color_jitter.hue"] == 0.1 and kang["views.grayscale.p"] == 0.2
        assert da["views.grayscale.p"] == 0.0 and da["views.vflip_p"] == kang["views.vflip_p"] == 0.5
        assert kang["views.color_jitter.brightness"] == kang["views.color_jitter.contrast"] == 0.2
        assert kang["views.color_jitter.saturation"] == 0.2 and kang["views.color_jitter.p"] == 0.8
        seed2 = load(f"imagenet_da_seed2_f{f}")
        assert diff(da, seed2) == {"seed"} and da["seed"] == 42 and seed2["seed"] == 43


def test_grayscale_off_keeps_stream_and_on_drops_colour(tmp_path):
    p = tmp_path / "c.npy"
    np.save(p, np.random.default_rng(0).integers(0, 256, (2, 72, 96, 3), dtype=np.uint8))
    base = {**VCFG, "anchor": 40, "out_size": 32, "train_long_side": [96, 96, 32]}
    no_key = {k: v for k, v in base.items() if k != "grayscale"}
    off = {**base, "grayscale": {"p": 0.0}}
    on = {**base, "grayscale": {"p": 1.0}}
    a, b = TwoViewDataset(p, no_key)[(0, 7)], TwoViewDataset(p, off)[(0, 7)]
    assert all(torch.equal(x, y) for x, y in zip(a, b))
    v = TwoViewDataset(p, on, input_norm=HALF)[(0, 7)][0]   # mean = std = 0.5 per channel -> gray = equal channels
    assert torch.allclose(v[0], v[1], atol=1e-6) and torch.allclose(v[1], v[2], atol=1e-6)


def test_dino_reference_uses_timm_pretrained_norm(monkeypatch):
    import timm

    from malaria_ssl.ssl import encoders
    from malaria_ssl.ssl.probe import extract_features
    from malaria_ssl.ssl.data import normalize

    cfg = load_experiment("configs/ssl/base.yaml")
    name = cfg["probe"]["references"]["dino"]
    assert name == "vit_small_patch16_224.dino"
    monkeypatch.setattr(encoders, "build_encoder", lambda n, *a, pretrained=False, **k: build_encoder(n))  # no download
    enc, tag, source = load_encoder(cfg, ref="dino")
    pc = timm.models.get_pretrained_cfg(name)
    assert tag == "ref_dino" and source["timm"] == name
    assert source["input_mean"] == list(pc.mean) and source["input_std"] == list(pc.std)

    seen = []
    real = enc.forward_features
    enc.forward_features = lambda x: (seen.append(x.detach().clone()), real(x))[1]
    img = np.random.default_rng(1).integers(0, 256, (1, 600, 800, 3), dtype=np.uint8)
    crops = {"box": np.array([[300.0, 200.0, 340.0, 240.0]], np.float32), "row": np.array([0]),
             "image_id": np.array([5]), "label": np.array([0])}
    pcfg = {**cfg["probe"], "long_side": 800, "batch_size": 4}
    extract_features(enc, img, crops, pcfg, torch.device("cpu"), None,
                     input_norm=(source["input_mean"], source["input_std"]))
    x0, y0 = 320 - 112, 220 - 112
    expected = normalize(img[0, y0:y0 + 224, x0:x0 + 224], pc.mean, pc.std)
    assert torch.allclose(seen[0][0], expected, atol=1e-5)
