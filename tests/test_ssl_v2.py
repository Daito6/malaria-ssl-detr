"""Phase 7 (SSL v2): DINO self-distillation, stain consistency, cell-centred views, configs (CPU, synthetic)."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

import malaria_ssl.ssl.data as ssl_data
from malaria_ssl.ssl.data import TwoViewDataset, load_cell_boxes, sample_windows
from malaria_ssl.ssl.encoders import init_encoder, is_imagenet_norm
from malaria_ssl.ssl.models import SSLModel, build_encoder, dino_loss, param_groups, stain_consistency_loss
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

VCFG = load_config(resolve_path("configs/ssl/data.yaml"))["views"]
SMALL = {**VCFG, "anchor": 40, "out_size": 32, "train_long_side": [96, 96, 32]}
FULL = load_experiment("configs/ssl/v2_full_f1.yaml")


def _flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        out.update(_flat(v, f"{prefix}{k}.") if isinstance(v, dict) else {f"{prefix}{k}": v})
    return out


def _diff(a, b):
    return {k for k in a.keys() | b.keys() if a.get(k) != b.get(k)}


# --------------------------------------------------------------------------- configs

def test_v2_configs_differ_only_in_their_blocks():
    load = lambda n: _flat(load_experiment(f"configs/ssl/{n}.yaml"))   # noqa: E731
    imagenet_cont, cont = load("imagenet_cont"), load("v2_cont")
    assert _diff(imagenet_cont, cont) == {"model.init", "train.grad_checkpointing"} and cont["model.init"] == "dino"
    assert cont["train.grad_checkpointing"] is True
    assert cont["dino"] is None and cont["stain_consistency"] is None and cont["cells"] is None
    assert cont["target"] is None and cont["train.epochs"] == 30 and cont["optim.blr"] == 1.5e-4
    block = lambda keys: {k.split(".")[0] for k in keys}   # noqa: E731
    a, b = load("v2_a"), load("v2_b")
    assert block(_diff(cont, a)) == {"dino"} and block(_diff(cont, b)) == {"stain_consistency"}
    for f in (1, 2):
        c, full = load(f"v2_c_f{f}"), load(f"v2_full_f{f}")
        assert block(_diff(cont, c)) == {"cells", "target"} and c["target.fold"] == f
        assert c["target.fraction"] == 0.5 and c["cells.candidates_run"] == "runs/a_coco_detr_binary"
        assert block(_diff(cont, full)) == {"dino", "stain_consistency", "cells", "target"}
        for blk, src in (("dino", a), ("stain_consistency", b), ("cells", c), ("target", c)):
            assert {k: v for k, v in full.items() if k.startswith(blk + ".")} == \
                   {k: v for k, v in src.items() if k.startswith(blk + ".")}, blk
    # the shared screening/probe settings are the same for every variant
    assert all(load(n)["seed"] == 42 and load(n)["train.batch"] == 128 for n in ("v2_a", "v2_b", "v2_full_f2"))
    for n in ("v2_b", "v2_cont"):   # second SSL seed: only the seed differs
        assert _diff(load(n), load(f"{n}_seed43")) == {"seed"} and load(f"{n}_seed43")["seed"] == 43


# --------------------------------------------------------------------------- losses / model

def test_dino_loss_matches_manual_cross_entropy():
    torch.manual_seed(0)
    s = [torch.randn(3, 5) for _ in range(4)]
    q = [torch.softmax(torch.randn(3, 5), -1) for _ in range(2)]
    manual = []
    for i in range(2):
        for v in range(4):
            if v != i:
                manual.append(-(q[i] * torch.log_softmax(s[v] / 0.1, -1)).sum(-1).mean())
    assert torch.allclose(dino_loss(s, q, 0.1), torch.stack(manual).mean())


def test_stain_loss_zero_for_equal_tokens_and_no_grad_to_clean():
    clean = torch.randn(2, 4, 8, requires_grad=True)
    twin = clean.detach().clone().requires_grad_(True)
    assert stain_consistency_loss(twin, clean).abs() < 1e-6
    stain_consistency_loss(twin + torch.randn_like(twin), clean).backward()
    assert clean.grad is None and twin.grad is not None


def _model(cfg=FULL, **lams):
    cfg = copy.deepcopy(cfg)
    cfg["loss"].update({k: v for k, v in lams.items() if k.startswith("lambda_m") or k == "lambda_con"})
    if "lambda_dino" in lams:
        cfg["dino"]["lambda"] = lams["lambda_dino"]
    torch.manual_seed(0)
    return SSLModel(build_encoder(cfg["model"]["encoder"]), cfg)


def _inputs(b=2):
    g = torch.Generator().manual_seed(1)
    v1, v2 = torch.randn(b, 3, 64, 64, generator=g), torch.randn(b, 3, 64, 64, generator=g)
    return v1, v2, {"local": torch.randn(b, 2, 3, 32, 32, generator=g), "twin": torch.randn(b, 3, 64, 64, generator=g)}


def test_full_forward_teacher_ema_center_and_export():
    m = _model().train()
    assert not any(p.requires_grad for p in m.teacher_encoder.parameters())
    trainable = {id(p) for g in param_groups(m, 0.05) for p in g["params"]}
    assert not trainable & {id(p) for p in (*m.teacher_encoder.parameters(), *m.teacher_head.parameters())}
    assert id(m.dino_head.prototypes) in {id(p) for p in param_groups(m, 0.05)[1]["params"]}   # no weight decay
    v1, v2, ex = _inputs()
    out = m(v1, v2, **ex)
    for k in ("loss_mae", "loss_con", "loss_dino", "loss_stain"):
        assert out[k] is not None and torch.isfinite(out[k]), k
    assert torch.allclose(out["loss"], out["loss_mae"] + 0.2 * out["loss_con"] + out["loss_dino"] + out["loss_stain"])
    assert m.center.abs().sum() > 0                       # centre updated in train mode
    c = m.center.clone()
    m.eval()(v1, v2, **ex)
    assert torch.equal(m.center, c)                       # not in eval mode
    m.train()
    out["loss"].backward()
    with torch.no_grad():
        for p in m.encoder.parameters():
            p.add_(1.0)
    before = [p.clone() for p in m.teacher_encoder.parameters()]
    m.update_teacher(0.9)
    for p0, pt, ps in zip(before, m.teacher_encoder.parameters(), m.encoder.parameters()):
        assert torch.allclose(pt, 0.9 * p0 + 0.1 * ps, atol=1e-6)
    sd = m.encoder_state_dict()
    assert all(torch.equal(sd[k], v) for k, v in m.teacher_encoder.state_dict().items())   # export = teacher
    fresh = build_encoder("vit_small_patch16_224")
    fresh.load_state_dict(sd, strict=True)
    fresh.load_state_dict(m.encoder_state_dict(student=True), strict=True)


def test_grad_checkpointing_changes_memory_not_results():
    v1, v2, ex = _inputs()
    grads = []
    for ckpt in (False, True):
        m = _model().train()
        m.encoder.set_grad_checkpointing(ckpt)
        torch.manual_seed(5)                                # same MAE mask noise
        out = m(v1, v2, **ex)
        out["loss"].backward()
        grads.append((out["loss"].detach(), m.encoder.blocks[3].mlp.fc1.weight.grad.clone()))
    assert torch.allclose(grads[0][0], grads[1][0], atol=1e-5)
    assert torch.allclose(grads[0][1], grads[1][1], atol=1e-5)


def test_head_warmup_keeps_dino_gradient_off_the_encoder():
    m = _model(lambda_mae=0.0, lambda_con=0.0).train()
    m.stain_cfg, m.lambda_stain = None, 0.0               # DINO only
    v1, v2, ex = _inputs()
    m.dino_backbone_grad = False
    m(v1, v2, local=ex["local"])["loss"].backward()
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in m.encoder.parameters())
    assert m.dino_head.prototypes.grad is not None
    m.zero_grad(set_to_none=True)
    m.dino_backbone_grad = True
    m(v1, v2, local=ex["local"])["loss"].backward()
    assert m.encoder.blocks[0].attn.qkv.weight.grad.abs().sum() > 0


def test_old_configs_build_unchanged_model():
    m = _model(load_experiment("configs/ssl/joint.yaml"))
    assert not hasattr(m, "teacher_encoder") and not m.use_stain
    assert set(m.weights()) == {"w_mae", "w_con"}


def test_dino_init_uses_timm_dino_weights_and_norm(monkeypatch):
    import timm

    from malaria_ssl.ssl import encoders
    seen = []
    monkeypatch.setattr(encoders, "build_encoder",
                        lambda n, *a, pretrained=False, **k: (seen.append((n, pretrained)), build_encoder("vit_small_patch16_224"))[1])
    cfg = load_experiment("configs/ssl/v2_cont.yaml")
    _, norm = init_encoder(cfg["model"], cfg["probe"]["references"]["imagenet"], Path,
                           dino_name=cfg["probe"]["references"]["dino"])
    pc = timm.models.get_pretrained_cfg("vit_small_patch16_224.dino")
    assert seen == [("vit_small_patch16_224.dino", True)]
    assert norm == {"mean": list(pc.mean), "std": list(pc.std)} and is_imagenet_norm(norm["mean"], norm["std"])


# --------------------------------------------------------------------------- views

@pytest.fixture
def cache(tmp_path):
    p = tmp_path / "train_960.npy"
    np.save(p, np.random.default_rng(0).integers(0, 256, (3, 72, 96, 3), dtype=np.uint8))
    return p


LOCAL = {"n": 3, "scale": [0.05, 0.4], "ratio": [0.75, 1.3333], "out_size": 16}
TWIN = FULL["stain_consistency"]


def test_extras_do_not_change_the_two_views(cache):
    plain = TwoViewDataset(cache, SMALL)[(1, 11)]
    extra = TwoViewDataset(cache, SMALL, local_cfg=LOCAL, twin_cfg=TWIN)[(1, 11)]
    assert len(plain) == 3 and len(extra) == 4
    assert all(torch.equal(x, y) for x, y in zip(plain, extra[:3]))
    assert extra[3]["local"].shape == (3, 3, 16, 16) and extra[3]["twin"].shape == (3, 32, 32)
    again = TwoViewDataset(cache, SMALL, local_cfg=LOCAL, twin_cfg=TWIN)[(1, 11)]
    assert torch.equal(again[3]["local"], extra[3]["local"]) and torch.equal(again[3]["twin"], extra[3]["twin"])
    no_local = TwoViewDataset(cache, SMALL, local_cfg={**LOCAL, "n": 0})[(1, 11)]
    assert len(no_local) == 3


def test_twin_keeps_the_geometry_of_view_one(cache):
    mild = {**TWIN, "hed_sigma": 0.0, "color_jitter": {"brightness": 0.0, "contrast": 0.0, "saturation": 0.0, "hue": 0.0},
            "blur_sigma": [0.1, 0.1], "jpeg_quality": [100, 100], "resolution": [1.0, 1.0]}
    v1, v2, _, ex = TwoViewDataset(cache, SMALL, twin_cfg=mild)[(0, 3)]
    assert (ex["twin"] - v1).abs().mean() < 0.1 < (ex["twin"] - v2).abs().mean()
    strong = TwoViewDataset(cache, SMALL, twin_cfg=TWIN)[(0, 3)][3]["twin"]
    assert (strong - v1).abs().mean() > (ex["twin"] - v1).abs().mean()


def test_cell_centred_anchor_contains_the_cell():
    rng = np.random.default_rng(0)
    for _ in range(50):
        center = rng.uniform([0, 0], [96, 72])
        _, anchor, crops = sample_windows(np.random.default_rng(rng.integers(1 << 30)), 72, 96, 96, SMALL, center)
        assert anchor[0] >= 0 and anchor[1] >= 0 and anchor[2] <= 96 + 1e-6 and anchor[3] <= 72 + 1e-6
        inside = (anchor[0] <= center[0] <= anchor[2]) and (anchor[1] <= center[1] <= anchor[3])
        if inside and 20 <= center[0] <= 76 and 20 <= center[1] <= 52:   # anchor not clamped: centred, crops hit it
            assert np.allclose((anchor[:2] + anchor[2:]) / 2, center)
            assert all(c[0] <= center[0] <= c[2] and c[1] <= center[1] <= c[3] for c in crops)


def test_cell_fraction_one_centres_every_sample_on_a_box(cache):
    boxes = [np.array([[60.0, 30.0, 70.0, 40.0]], np.float32), np.zeros((0, 4), np.float32),
             np.array([[10.0, 20.0, 14.0, 24.0]], np.float32)]
    ds = TwoViewDataset(cache, SMALL, cell_boxes=boxes, cell_fraction=1.0)
    for seed in range(20):
        crops = ds[(0, seed)][2].numpy()
        assert all(c[0] <= 65 <= c[2] and c[1] <= 35 <= c[3] for c in crops)
    assert torch.equal(ds[(1, 5)][0], TwoViewDataset(cache, SMALL)[(1, 5)][0])   # no boxes: uniform as before


def test_load_cell_boxes_scales_filters_and_drops_other_images(tmp_path):
    cands = {"nms_iou": 0.5, "candidates": [
        {"image_id": 7, "bbox": [100, 50, 20, 10], "objectness": 0.9},
        {"image_id": 7, "bbox": [0, 0, 5, 5], "objectness": 0.1},
        {"image_id": 9, "bbox": [10, 10, 10, 10], "objectness": 0.99}]}   # image 9 is not in this cache (other fold)
    p = tmp_path / "candidates_test.json"
    p.write_text(json.dumps(cands), encoding="utf-8")
    index = {"scale": 0.5, "images": [{"id": 3, "file_name": "a.jpg"}, {"id": 7, "file_name": "b.jpg"}]}
    boxes = load_cell_boxes(p, index, 0.5)
    assert boxes[0].shape == (0, 4) and np.allclose(boxes[1], [[50, 25, 60, 30]])


# --------------------------------------------------------------------------- trainer (CPU)

def test_trainer_full_variant_cpu(tmp_path, monkeypatch):
    from malaria_ssl.ssl import trainer as tr_mod

    d = tmp_path / "cache"
    d.mkdir()
    rng = np.random.default_rng(0)
    caches = {}
    for split, n, h in (("train", 3, 720), ("val", 2, 720), ("target", 2, 683)):
        caches[split] = d / f"{split}_960.npy"
        np.save(caches[split], rng.integers(0, 256, (n, h, 960, 3), dtype=np.uint8))
    (d / "train_index.json").write_text(json.dumps(
        {"scale": 0.6, "images": [{"id": i, "file_name": f"{i}.png"} for i in (1, 2, 3)]}), encoding="utf-8")
    (d / "target_index.json").write_text(json.dumps(
        {"scale": 960 / 1944, "images": [{"id": i, "file_name": f"{i}.jpg"} for i in (11, 12)]}), encoding="utf-8")
    det = tmp_path / "det"
    det.mkdir()
    box = lambda i: {"image_id": i, "bbox": [400.0, 300.0, 90.0, 90.0], "objectness": 0.95}   # noqa: E731
    (det / "candidates_train.json").write_text(json.dumps({"candidates": [box(1), box(3)]}), encoding="utf-8")
    (det / "candidates_test.json").write_text(json.dumps({"candidates": [box(11), box(99)]}), encoding="utf-8")

    def no_coco(*a, **k):
        raise AssertionError("the SSL v2 loader must not read annotations")
    monkeypatch.setattr(ssl_data, "load_split_coco", no_coco)

    cfg = copy.deepcopy(FULL)
    cfg["model"]["init"] = "random"           # no download on CPU
    cfg["cells"]["candidates_run"] = str(det)
    cfg["dino"].update(n_prototypes=64, head_hidden=32, head_bottleneck=16)
    cfg["dino"]["local_crops"]["n"] = 2
    cfg["train"].update({"batch": 2, "samples_per_epoch": 4, "epochs": 2, "num_workers": 0, "amp_dtype": "none"})
    cfg["monitor"].update({"val_pairs": 4, "recon_every": 1, "recon_crops": 2, "probe_every": 0, "save_every": 0})
    t = tr_mod.SSLTrainer(cfg, tmp_path / "run", torch.device("cpu"), caches)
    t.setup(resume=False)
    assert t.train_ds.sources[0].cell_boxes[1].shape == (0, 4) and len(t.train_ds.sources[0].cell_boxes[0]) == 1
    assert [len(b) for b in t.target_ds.cell_boxes] == [1, 0]           # image 99 (other fold) dropped
    t.fit()
    assert t.model.dino_backbone_grad                                   # warm-up (1 epoch) over
    run = tmp_path / "run"
    import pandas as pd
    m = pd.read_csv(run / "metrics.csv")
    for col in ("loss_dino", "loss_stain", "loss_mae", "loss_con", "teacher_rep/erank", "rep/erank",
                "dino_teacher_entropy"):
        assert col in m and np.isfinite(m[col]).all(), col
    ck = run / "checkpoints"
    teacher = torch.load(ck / "encoder_final.pt", weights_only=True)
    student = torch.load(ck / "encoder_student_final.pt", weights_only=True)
    assert all(torch.equal(teacher[k], v.cpu()) for k, v in t.model.teacher_encoder.state_dict().items())
    assert any(not torch.equal(teacher[k], student[k]) for k in teacher)
    saved = yaml.safe_load((run / "config.yaml").read_text(encoding="utf-8"))
    assert saved["dino"]["lambda"] == 1.0 and saved["cells"]["fraction"] == 0.5

    t2 = tr_mod.SSLTrainer(copy.deepcopy(saved), run, torch.device("cpu"), caches)   # resume restores teacher + centre
    t2.setup(resume=True)
    assert t2.epoch == 2 and torch.equal(t2.model.center, t.model.center)
    assert all(torch.equal(a, b) for a, b in zip(t2.model.teacher_encoder.state_dict().values(),
                                                  t.model.teacher_encoder.state_dict().values()))
