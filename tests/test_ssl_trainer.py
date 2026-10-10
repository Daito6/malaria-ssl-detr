"""SSL trainer tests on CPU with a synthetic cache (no real data, no GPU)."""
import copy

import numpy as np
import pytest
import torch

from malaria_ssl.ssl.models import build_encoder
from malaria_ssl.ssl.trainer import SSLTrainer, collapse_stats
from malaria_ssl.utils.config import load_experiment

CFG = load_experiment("configs/ssl/base.yaml")


@pytest.fixture(scope="module")
def caches(tmp_path_factory):
    d = tmp_path_factory.mktemp("ssl_cache")
    rng = np.random.default_rng(0)
    out = {}
    for split, n in (("train", 3), ("val", 2)):
        np.save(d / f"{split}_960.npy", rng.integers(0, 256, (n, 720, 960, 3), dtype=np.uint8))
        out[split] = d / f"{split}_960.npy"
    return out


def _cfg(epochs=2):
    cfg = copy.deepcopy(CFG)
    cfg["loss"]["loss_weighting"] = "uncertainty"
    cfg["train"].update({"batch": 4, "samples_per_epoch": 8, "epochs": epochs, "num_workers": 0,
                         "amp_dtype": "none", "max_steps": None})
    cfg["monitor"].update({"val_pairs": 6, "recon_every": 1, "recon_crops": 2, "probe_every": 0, "save_every": 1})
    cfg["optim"]["warmup_fraction"] = 0.25
    return cfg


def test_resume_restores_state_and_continues(tmp_path, caches):
    cpu = torch.device("cpu")
    t1 = SSLTrainer(_cfg(), tmp_path, cpu, caches)
    t1.setup(resume=False)
    t1.cfg["train"]["epochs"] = 1          # stop after the first epoch
    t1.fit()
    for p in ("config.yaml", "run_info.json", "metrics.csv", "recon/init.png", "recon/epoch_001.png",
              "checkpoints/last.pt", "checkpoints/encoder_ep001.pt"):
        assert (tmp_path / p).exists(), p
    ref_model = {k: v.detach().clone() for k, v in t1.model.state_dict().items()}
    ref_opt = t1.optimizer.state_dict()
    assert not torch.equal(t1.model.log_vars.detach(), torch.zeros(2))   # log-variances were trained

    t2 = SSLTrainer(_cfg(), tmp_path, cpu, caches)
    t2.setup(resume=True)
    assert t2.epoch == 1 and t2.global_step == t1.global_step == 2
    assert t2.scheduler.state_dict()["last_epoch"] == t1.scheduler.state_dict()["last_epoch"]
    for k, v in t2.model.state_dict().items():
        assert torch.equal(v, ref_model[k]), k
    opt = t2.optimizer.state_dict()
    assert opt["state"].keys() == ref_opt["state"].keys() and len(opt["state"]) > 0
    for k, st in opt["state"].items():
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(st[name], ref_opt["state"][k][name]), (k, name)
    t2.fit()                               # runs only the remaining epoch
    assert t2.epoch == 2 and t2.global_step == 4
    enc = build_encoder(CFG["model"]["encoder"])
    enc.load_state_dict(torch.load(tmp_path / "checkpoints" / "encoder_final.pt", weights_only=True), strict=True)
    import pandas as pd
    m = pd.read_csv(tmp_path / "metrics.csv")
    assert m["epoch"].tolist() == [0, 1]
    for col in ("loss", "loss_mae", "loss_con", "w_mae", "w_con", "lr", "grad_norm", "pairs_per_s",
                "val_loss_mae", "val_loss_con", "rep/erank", "proj/uniformity"):
        assert col in m.columns and np.isfinite(m[col]).all(), col


def test_collapse_stats_detects_collapse():
    torch.manual_seed(0)
    spread = torch.randn(256, 64)
    collapsed = torch.ones(256, 64) + 1e-4 * torch.randn(256, 64)       # complete collapse -> std
    low_rank = torch.randn(256, 2) @ torch.randn(2, 64)                 # dimensional collapse -> erank
    s, c, d = (collapse_stats(e, e) for e in (spread, collapsed, low_rank))
    assert s["std"] > 1000 * c["std"]
    assert s["erank"] > 40 and d["erank"] < 2.5
    assert s["uniformity"] < c["uniformity"] and abs(s["alignment"]) < 1e-9
