"""ViT backbone in DETR (`vit_detr`): shapes, padding mask, token order, normalization, weight loading,
parameter groups. CPU only; the COCO DETR and the timm ImageNet ViT come from the local HF cache."""
import copy

import pytest
import torch
import torch.nn.functional as F

from malaria_ssl.data.detection import IMAGENET_MEAN, IMAGENET_STD
from malaria_ssl.detr.build import ViTBackbone, build_model, build_vit_backbone, param_groups
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
