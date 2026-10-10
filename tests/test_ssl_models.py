"""SSL model tests (CPU, random-init ViT-S/16)."""
import copy

import pytest
import torch

from malaria_ssl.ssl.models import SSLModel, build_encoder, info_nce, param_groups
from malaria_ssl.utils.config import load_experiment

CFG = load_experiment("configs/ssl/base.yaml")


def _model(lambda_mae=1.0, lambda_con=0.2, weighting="fixed"):
    cfg = copy.deepcopy(CFG)
    cfg["loss"].update(lambda_mae=lambda_mae, lambda_con=lambda_con, loss_weighting=weighting)
    torch.manual_seed(0)
    return SSLModel(build_encoder(cfg["model"]["encoder"]), cfg).eval()


@pytest.fixture(scope="module")
def model():
    return _model()


def test_mask_ratio_and_shapes(model):
    x = torch.randn(3, 3, 224, 224)
    latent, mask, ids_restore = model.encode_masked(x, 0.75)
    n = 14 * 14
    assert mask.shape == (3, n) and ids_restore.shape == (3, n)
    assert torch.all(mask.sum(1) == n - 49)                    # exactly 75 % masked
    assert latent.shape == (3, 1 + 49, model.encoder.embed_dim)  # cls + visible only
    assert not torch.equal(mask[0], mask[1])                   # per-sample masks
    out = model.mae_forward(x)
    assert out["pred"].shape == (3, n, 16 * 16 * 3)


def test_patchify_roundtrip(model):
    x = torch.randn(2, 3, 224, 224)
    assert torch.equal(model.unpatchify(model.patchify(x), 224, 224), x)


def test_mae_loss_only_on_masked_patches(model):
    torch.manual_seed(1)
    target = torch.randn(2, 196, 768)
    pred = torch.randn(2, 196, 768)
    mask = (torch.rand(2, 196) < 0.75).float()
    base = model.mae_loss(pred, target, mask)
    perturbed = pred + (1 - mask).unsqueeze(-1) * 100.0       # change visible patches only
    assert torch.allclose(model.mae_loss(perturbed, target, mask), base)
    perfect = torch.where(mask.bool().unsqueeze(-1), target, pred)  # masked patches exact
    assert model.mae_loss(perfect, target, mask) < 1e-12


def test_info_nce_positives_are_the_diagonal_pairs():
    torch.manual_seed(2)
    z = torch.nn.functional.normalize(torch.randn(16, 32), dim=1) * 10
    aligned = info_nce(z, z.clone(), temperature=0.1)
    shuffled = info_nce(z, z[torch.randperm(16)], temperature=0.1)
    rand = info_nce(torch.randn(16, 32), torch.randn(16, 32), temperature=0.1)
    assert torch.isfinite(aligned) and torch.isfinite(rand)
    assert aligned < 0.05 < rand and aligned < shuffled


def test_joint_forward_finite_and_branches_skipped():
    x1, x2 = torch.randn(4, 3, 224, 224), torch.randn(4, 3, 224, 224)
    joint = _model().train()
    out = joint(x1, x2)
    assert torch.isfinite(out["loss"]) and out["loss_mae"] is not None and out["loss_con"] is not None
    assert torch.allclose(out["loss"], out["loss_mae"] + 0.2 * out["loss_con"])
    mae = _model(lambda_con=0.0).train()
    assert not hasattr(mae, "projector") and mae(x1, x2)["loss_con"] is None
    con = _model(lambda_mae=0.0).train()
    assert not hasattr(con, "decoder_blocks") and con(x1, x2)["loss_mae"] is None


def test_uncertainty_weighting():
    m = _model(weighting="uncertainty").train()
    assert m.weights() == {"w_mae": 1.0, "w_con": 1.0}
    out = m(torch.randn(4, 3, 224, 224), torch.randn(4, 3, 224, 224))
    out["loss"].backward()
    assert m.log_vars.grad is not None and torch.all(m.log_vars.grad != 0)
    no_decay = {id(p) for p in param_groups(m, 0.05)[1]["params"]}
    assert id(m.log_vars) in no_decay


def test_encoder_export_loads_strict_into_fresh_timm(model, tmp_path):
    path = tmp_path / "encoder.pt"
    torch.save(model.encoder_state_dict(), path)
    fresh = build_encoder(CFG["model"]["encoder"])
    fresh.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    fresh.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        assert torch.allclose(fresh.forward_features(x), model.encoder.forward_features(x), atol=1e-6)
        # dynamic_img_size: a DETR-sized input interpolates the position embeddings
        assert fresh.forward_features(torch.randn(1, 3, 320, 480)).shape == (1, 1 + 20 * 30, 384)
