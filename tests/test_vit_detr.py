"""ViT backbone in DETR (`vit_detr`): shapes, padding mask, token order, normalization, weight loading,
parameter groups. CPU only; the COCO DETR and the timm ImageNet ViT come from the local HF cache."""
import copy

import pytest
import torch
import torch.nn.functional as F

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.detr.build import ViTBackbone, build_model, build_vit_backbone, param_groups, vit_layer_id
from malaria_ssl.ssl.models import build_encoder
from malaria_ssl.utils.config import load_experiment

CFG = load_experiment("configs/detr/base.yaml")
CLASSES = ["uninfected", "infected"]


def _encoder(seed=0):
    torch.manual_seed(seed)
    return build_encoder(CFG["model"]["vit"]["arch"]).eval()


def _model_cfg(**vit):
    cfg = copy.deepcopy(CFG["model"])
    cfg["name"] = "vit_detr"
    cfg["vit"].update(vit)
    return cfg


def test_backbone_pads_to_patch_multiple_and_masks_padding():
    bb = ViTBackbone(_encoder(), IMAGENET_MEAN, IMAGENET_STD, 16)
    x = torch.randn(2, 3, 200, 264)
    mask = torch.ones(2, 200, 264, dtype=torch.long)
    mask[1, 100:], mask[1, :, 132:] = 0, 0          # image 1 is 100 x 132, batch-padded
    with torch.no_grad():
        (fmap, m), = bb(x, mask)
    assert fmap.shape == (2, 384, 13, 17) and m.shape == (2, 13, 17) and m.dtype == torch.bool  # 208 x 272
    assert m[0].all()                                # last row/col partly valid (nearest, as DetrConvEncoder)
    assert m[1, :7, :9].all() and not m[1, 7:].any() and not m[1, :, 9:].any()


def test_feature_map_is_row_major_patch_tokens():
    enc = _encoder()
    bb = ViTBackbone(enc, IMAGENET_MEAN, IMAGENET_STD, 16)   # identity normalization
    x = torch.randn(1, 3, 192, 256)
    with torch.no_grad():
        (fmap, _), = bb(x, torch.ones(1, 192, 256, dtype=torch.long))
        tokens = enc.forward_features(x)[:, enc.num_prefix_tokens:]
    gh, gw = 12, 16
    for i, j in [(0, 0), (0, 15), (5, 7), (11, 15)]:
        assert torch.allclose(fmap[0, :, i, j], tokens[0, i * gw + j], atol=1e-5)


def test_stride32_is_space_to_depth_of_stride16():
    enc = _encoder()
    bb16, bb32 = (ViTBackbone(enc, IMAGENET_MEAN, IMAGENET_STD, s) for s in (16, 32))
    x, mask = torch.randn(1, 3, 200, 264), torch.ones(1, 200, 264, dtype=torch.long)
    with torch.no_grad():
        (f32, m32), = bb32(x, mask)
        xp, mp = F.pad(x, (0, 24, 0, 24)), F.pad(mask, (0, 24, 0, 24))   # the stride-32 padding (224 x 288)
        (f16, _), = bb16(xp, mp)
    assert f32.shape == (1, 1536, 7, 9) and m32.shape == (1, 7, 9) and bb32.num_channels == 1536
    assert torch.allclose(f32, F.pixel_unshuffle(f16, 2), atol=1e-5)
    with pytest.raises(ValueError):
        ViTBackbone(enc, IMAGENET_MEAN, IMAGENET_STD, 8)


def test_input_is_renormalized_to_encoder_statistics():
    enc = _encoder()
    bb = ViTBackbone(enc, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 16)
    seen = []
    enc.patch_embed.register_forward_pre_hook(lambda mod, args: seen.append(args[0]))
    raw = torch.rand(1, 3, 64, 64)                                     # pixels in [0, 1]
    x = (raw - torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)) / torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    with torch.no_grad():
        bb(x, torch.ones(1, 64, 64, dtype=torch.long))
    assert torch.allclose(seen[0], (raw - 0.5) / 0.5, atol=1e-5)


def test_init_from_state_dict_is_strict(tmp_path):
    src = _encoder(seed=1)
    path = tmp_path / "encoder_final.pt"
    torch.save(src.state_dict(), path)
    bb = build_vit_backbone(_model_cfg(init=str(path))["vit"])
    for k, v in src.state_dict().items():
        assert torch.equal(bb.encoder.state_dict()[k], v), k
    assert torch.allclose(bb.scale, torch.ones(1, 3, 1, 1)) and torch.allclose(bb.shift, torch.zeros(1, 3, 1, 1))
    bad = dict(src.state_dict())
    bad.pop("blocks.0.attn.qkv.weight")
    torch.save(bad, path)
    with pytest.raises(RuntimeError):
        build_vit_backbone(_model_cfg(init=str(path))["vit"])


def test_imagenet_init_uses_its_own_normalization():
    bb = build_vit_backbone(_model_cfg(init="imagenet")["vit"])
    std = torch.as_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    mean = torch.as_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    assert torch.allclose(bb.scale, std / 0.5) and torch.allclose(bb.shift, (mean - 0.5) / 0.5)


@pytest.fixture(scope="module")
def vit_model():
    return build_model(_model_cfg(init="random"), CLASSES, seed=42)


def test_vit_detr_forward_and_structure(vit_model):
    hf = vit_model.hf
    assert isinstance(hf.model.backbone, ViTBackbone)
    assert hf.model.input_projection.in_channels == 384 and hf.model.input_projection.out_channels == 256
    assert hf.model.query_position_embeddings.weight.shape == (300, 256)
    assert all(p.requires_grad for p in vit_model.parameters())        # nothing frozen in the ViT
    vit_model.eval()
    with torch.no_grad():
        out = vit_model(torch.randn(1, 3, 128, 160), torch.ones(1, 128, 160, dtype=torch.long))
    assert out.logits.shape == (1, 300, 3) and out.pred_boxes.shape == (1, 300, 4)
    assert len(out.auxiliary_outputs) == 5 and torch.isfinite(out.logits).all()


def test_vit_detr_param_groups(vit_model):
    groups = param_groups(vit_model, {**CFG["optim"], "lr_projection": 3e-4})
    assert [g["name"] for g in groups] == ["transformer", "backbone", "projection"]
    assert [g["lr"] for g in groups] == [CFG["optim"]["lr"], CFG["optim"]["lr_backbone"], 3e-4]
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in vit_model.parameters() if p.requires_grad)
    by = {g["name"]: {id(p) for p in g["params"]} for g in groups}
    assert by["projection"] == {id(p) for p in vit_model.hf.model.input_projection.parameters()}
    assert by["backbone"] == {id(p) for p in vit_model.hf.model.backbone.parameters()}
    assert param_groups(vit_model, CFG["optim"])[2]["lr"] == CFG["optim"]["lr"]   # lr_projection null -> lr


def test_vit_detr_is_deterministic_and_keeps_coco_transformer(vit_model):
    again = build_model(_model_cfg(init="random"), CLASSES, seed=42)
    a, b = vit_model.state_dict(), again.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)
    coco = build_model({**copy.deepcopy(CFG["model"]), "name": "coco_detr_r50"}, CLASSES, seed=42)
    c = coco.state_dict()
    shared = [k for k in c if ".encoder." in k or ".decoder." in k or "query_position" in k or "bbox_predictor" in k]
    assert shared and all(torch.equal(a[k], c[k]) for k in shared)
    # R50 models keep two optimizer groups (input projection with the transformer) -> old checkpoints resume
    assert [g["name"] for g in param_groups(coco, CFG["optim"])] == ["transformer", "backbone"]


# --------------------------------------------------------------------------- step 2: LoRA, projection warmup

@pytest.fixture(scope="module")
def lora_model():
    return build_model(_model_cfg(init="random") | {"adapt": "lora"}, CLASSES, seed=42)


def test_lora_is_identity_at_init(vit_model, lora_model):
    x, m = torch.randn(1, 3, 128, 160), torch.ones(1, 128, 160, dtype=torch.long)
    vit_model.eval(), lora_model.eval()
    with torch.no_grad():
        a, b = vit_model(x, m), lora_model(x, m)
    assert torch.allclose(a.logits, b.logits, atol=1e-5) and torch.allclose(a.pred_boxes, b.pred_boxes, atol=1e-5)


def test_lora_targets_and_trainable_policy(lora_model):
    named = dict(lora_model.hf.named_parameters())
    lora = [n for n in named if "lora_" in n]
    adapted = {n.split(".lora_")[0] for n in lora}
    # 6 encoder self-attn + 6 decoder (self + cross) attn, 4 projections each
    assert len(adapted) == (6 + 6 * 2) * 4
    assert all(n.startswith(("model.encoder.layers.", "model.decoder.layers.")) for n in adapted)
    assert all(n.endswith(("q_proj", "k_proj", "v_proj", "o_proj")) for n in adapted)
    r = CFG["model"]["lora"]["r"]
    assert sum(named[n].numel() for n in lora) == len(adapted) * 2 * r * 256
    for n, p in named.items():
        in_transformer = n.startswith(("model.encoder.", "model.decoder."))
        assert p.requires_grad == ("lora_" in n or not in_transformer), n
    assert all(p.requires_grad for p in lora_model.hf.model.backbone.parameters())
    assert all(p.requires_grad for p in lora_model.hf.model.input_projection.parameters())
    assert lora_model.hf.model.query_position_embeddings.weight.requires_grad
    assert lora_model.hf.class_labels_classifier.weight.requires_grad and \
        all(p.requires_grad for p in lora_model.hf.bbox_predictor.parameters())
    groups = {g["name"]: {id(p) for p in g["params"]} for g in param_groups(lora_model, CFG["optim"])}
    assert {id(named[n]) for n in lora} <= groups["transformer"]
    assert not any(id(p) in s for s in groups.values() for p in lora_model.parameters() if not p.requires_grad)


def test_lora_state_dict_round_trip():
    trained = build_model(_model_cfg(init="random") | {"adapt": "lora"}, CLASSES, seed=42)
    with torch.no_grad():
        for n, p in trained.named_parameters():
            if "lora_B" in n:
                p.normal_()                                   # make the adapters non-trivial
    other = build_model(_model_cfg(init="random") | {"adapt": "lora"}, CLASSES, seed=0)
    other.load_state_dict(trained.state_dict())              # what evaluate.py does with a checkpoint
    a, b = trained.state_dict(), other.state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_unknown_adapt_raises():
    with pytest.raises(ValueError):
        build_model(_model_cfg(init="random") | {"adapt": "frozen"}, CLASSES, seed=0)


def test_projection_has_its_own_warmup():
    from malaria_ssl.detr.trainer import group_lr_lambdas

    params = [torch.nn.Parameter(torch.zeros(1)) for _ in range(3)]
    opt = torch.optim.AdamW([{"params": [p], "lr": 1e-4, "name": n}
                             for p, n in zip(params, ("transformer", "backbone", "projection"))])
    o = {**CFG["optim"], "warmup_steps": 200, "warmup_steps_projection": 50, "min_lr_ratio": 0.01}
    shared, b, proj = group_lr_lambdas(opt, o, total=1000)
    assert shared is b
    assert proj(0) == pytest.approx(1 / 50) and shared(0) == pytest.approx(1 / 200)
    assert proj(49) == pytest.approx(1.0) and shared(49) == pytest.approx(50 / 200)
    assert proj(1000) == pytest.approx(0.01) and shared(1000) == pytest.approx(0.01)
    same = group_lr_lambdas(opt, {**o, "warmup_steps_projection": None}, total=1000)
    assert same[0] is same[1] is same[2]


# --------------------------------------------------------------------------- step 4: layer-wise lr decay

def test_vit_layer_id():
    enc = _encoder()
    ids = {n: vit_layer_id(n, len(enc.blocks)) for n, _ in enc.named_parameters()}
    assert ids["cls_token"] == ids["pos_embed"] == ids["patch_embed.proj.weight"] == 0
    assert ids["blocks.0.attn.qkv.weight"] == 1 and ids["blocks.11.mlp.fc2.bias"] == 12
    assert ids["norm.weight"] == ids["norm.bias"] == 13
    assert set(ids.values()) == set(range(14))


def test_layer_decay_param_groups(vit_model):
    o = {**CFG["optim"], "layer_decay": 0.75, "lr_backbone": 1e-5}
    groups = param_groups(vit_model, o)
    names = [g["name"] for g in groups]
    # indices 0/1/2 keep their meaning for the trainer's logging and the projection warmup
    assert names[:3] == ["transformer", "backbone", "projection"]
    assert names[3:] == [f"backbone_layer{i}" for i in range(13)]
    lr = {g["name"]: g["lr"] for g in groups}
    assert lr["backbone"] == 1e-5                                         # final norm, depth 13
    assert lr["backbone_layer12"] == pytest.approx(1e-5 * 0.75)          # last block
    assert lr["backbone_layer0"] == pytest.approx(1e-5 * 0.75 ** 13)     # embeddings
    enc = vit_model.hf.model.backbone.encoder
    by_id = {id(p): g["name"] for g in groups for p in g["params"]}
    assert by_id[id(enc.blocks[11].attn.qkv.weight)] == "backbone_layer12"
    assert by_id[id(enc.patch_embed.proj.weight)] == "backbone_layer0"
    assert by_id[id(enc.norm.weight)] == "backbone"
    ids = [id(p) for g in groups for p in g["params"]]
    assert len(ids) == len(set(ids)) == sum(1 for p in vit_model.parameters() if p.requires_grad)
    assert {id(p) for p in enc.parameters()} == {i for i, n in by_id.items() if n.startswith("backbone")}
    # off by default: the three groups of step 1
    assert [g["name"] for g in param_groups(vit_model, CFG["optim"])] == ["transformer", "backbone", "projection"]


def test_layer_decay_groups_share_the_schedule(vit_model):
    from malaria_ssl.detr.trainer import group_lr_lambdas

    o = {**CFG["optim"], "layer_decay": 0.75, "warmup_steps_projection": 50}
    opt = torch.optim.AdamW(param_groups(vit_model, o))
    lams = group_lr_lambdas(opt, o, total=1000)
    names = [g["name"] for g in opt.param_groups]
    shared = lams[0]
    assert all(lam is shared for lam, n in zip(lams, names) if n != "projection")
    assert lams[names.index("projection")] is not shared


# --------------------------------------------------------------------------- frozen ViT (model.vit.freeze)

@pytest.fixture(scope="module")
def frozen_model():
    return build_model(_model_cfg(init="random", freeze=True, drop_path_rate=0.1), CLASSES, seed=42)


def test_frozen_vit_has_no_trainable_params_and_the_rest_trains(frozen_model):
    hf = frozen_model.hf
    assert not any(p.requires_grad for p in hf.model.backbone.parameters())
    rest = [p for n, p in frozen_model.named_parameters() if ".backbone." not in f".{n}"]
    assert rest and all(p.requires_grad for p in rest)
    groups = {g["name"]: g for g in param_groups(frozen_model, CFG["optim"])}
    assert groups["backbone"]["params"] == []                         # kept (trainer logs group 1), empty
    assert sum(p.numel() for g in groups.values() for p in g["params"]) == sum(p.numel() for p in rest)
    torch.optim.AdamW(list(groups.values())).step()                   # an empty group is accepted


def test_frozen_vit_stays_in_eval_mode_and_does_not_change(frozen_model):
    frozen_model.train()
    enc = frozen_model.hf.model.backbone.encoder
    assert frozen_model.training and frozen_model.hf.model.input_projection.training
    assert not any(m.training for m in enc.modules())                 # drop-path 0.1 is inactive
    x, m = torch.randn(1, 3, 128, 160), torch.ones(1, 128, 160, dtype=torch.long)
    with torch.no_grad():
        (f1, _), = frozen_model.hf.model.backbone(x, m)
        (f2, _), = frozen_model.hf.model.backbone(x, m)
    assert torch.equal(f1, f2)
    before = {k: v.clone() for k, v in enc.state_dict().items()}
    groups = param_groups(frozen_model, CFG["optim"])
    opt = torch.optim.AdamW(groups)
    out = frozen_model(x, m)
    (out.logits.sum() + out.pred_boxes.sum()).backward()
    assert all(p.grad is None for p in enc.parameters())
    assert frozen_model.hf.model.input_projection.weight.grad is not None
    opt.step()
    assert all(torch.equal(before[k], v) for k, v in enc.state_dict().items())


def test_freeze_survives_lora():
    model = build_model(_model_cfg(init="random", freeze=True) | {"adapt": "lora"}, CLASSES, seed=0)
    assert not any(p.requires_grad for p in model.hf.model.backbone.parameters())
    assert model.hf.model.input_projection.weight.requires_grad


def test_phase4_freeze_configs_differ_only_as_declared():
    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else out.__setitem__(f"{p}{k}", v)
        return out

    for enc in ("joint", "imagenet"):
        ref = flat(load_experiment(f"configs/experiments/b_vit_{enc}.yaml"))
        for variant, expected in (("frozen", {"model.vit.freeze": (False, True)}),
                                  ("lowlr", {"optim.lr_backbone": (1.0e-5, 1.0e-7)})):
            c = flat(load_experiment(f"configs/experiments/b_vit_{enc}_{variant}.yaml"))
            diff = {k: (ref.get(k), c.get(k)) for k in set(c) | set(ref) if ref.get(k) != c.get(k)}
            assert diff == expected, (enc, variant, diff)


def test_phase4_aug_configs_differ_only_as_declared():
    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else out.__setitem__(f"{p}{k}", v)
        return out

    expected = {"augment.stain_aug.p": (0.0, 0.5), "augment.train_long_side": ([640, 960, 32], [576, 1056, 32]),
                "train.select_on": ("val", "val_stain"), "eval.val_stain": (False, True)}
    for enc in ("joint", "imagenet"):
        ref = flat(load_experiment(f"configs/experiments/b_vit_{enc}.yaml"))
        c = flat(load_experiment(f"configs/experiments/b_vit_{enc}_aug.yaml"))
        diff = {k: (ref.get(k), c.get(k)) for k in set(c) | set(ref) if ref.get(k) != c.get(k)}
        assert diff == expected, (enc, diff)


def test_dino_init_matches_the_probe_reference():
    from malaria_ssl.ssl.encoders import load_encoder

    ssl_cfg = load_experiment("configs/ssl/base.yaml")
    assert CFG["model"]["vit"]["dino"] == ssl_cfg["probe"]["references"]["dino"]
    ref, _, source = load_encoder(ssl_cfg, ref="dino")
    bb = build_vit_backbone(_model_cfg(init="dino")["vit"])
    ref_sd, sd = ref.state_dict(), bb.encoder.state_dict()
    assert ref_sd.keys() == sd.keys()
    for k, v in ref_sd.items():
        assert torch.equal(sd[k], v), k                                  # bit-identical weights
    assert bb.input_norm == {"mean": source["input_mean"], "std": source["input_std"]}
    # effective normalization: x_enc = (x * in_std + in_mean - mean) / std, x ImageNet-normalized
    std = torch.as_tensor(IMAGENET_STD) / bb.scale.flatten()
    mean = torch.as_tensor(IMAGENET_MEAN) - bb.shift.flatten() * std
    assert torch.allclose(mean, torch.as_tensor(source["input_mean"]), atol=1e-6)
    assert torch.allclose(std, torch.as_tensor(source["input_std"]), atol=1e-6)


def test_recorded_input_norm_is_checked_on_rebuild():
    vit = _model_cfg(init="random")["vit"]
    norm = {"mean": [float(v) for v in IMAGENET_MEAN], "std": [float(v) for v in IMAGENET_STD]}
    assert build_vit_backbone(vit | {"input_norm": norm}).input_norm == norm
    with pytest.raises(ValueError):
        build_vit_backbone(vit | {"input_norm": {"mean": [0.5] * 3, "std": [0.5] * 3}})


def test_vit_input_norm_of_a_model(vit_model):
    from malaria_ssl.detr.build import vit_input_norm

    assert vit_input_norm(vit_model) == {"mean": [float(v) for v in IMAGENET_MEAN],
                                         "std": [float(v) for v in IMAGENET_STD]}
    assert vit_input_norm(torch.nn.Linear(2, 2)) is None


def test_phase6_dino_configs_differ_only_in_init():
    def flat(d, p=""):
        out = {}
        for k, v in d.items():
            out.update(flat(v, f"{p}{k}.")) if isinstance(v, dict) else out.__setitem__(f"{p}{k}", v)
        return out

    for dino, imagenet in (("b_vit_dino_full", "b_vit_imagenet"), ("b_vit_dino_frozen", "b_vit_imagenet_frozen")):
        ref = flat(load_experiment(f"configs/experiments/{imagenet}.yaml"))
        c = flat(load_experiment(f"configs/experiments/{dino}.yaml"))
        diff = {k: (ref.get(k), c.get(k)) for k in set(c) | set(ref) if ref.get(k) != c.get(k)}
        assert diff == {"model.vit.init": ("imagenet", "dino")}, (dino, diff)
