"""Trainer resume test (needs CUDA, the processed data and the cached HF model)."""
import copy

import pytest
import torch

from malaria_ssl.detr.trainer import Trainer
from malaria_ssl.utils.config import load_config, load_experiment, resolve_path

CFG = load_experiment("configs/detr/base.yaml")
HAVE_DATA = (resolve_path(load_config(resolve_path(CFG["data_config"]))["paths"]["coco_dir"]) / "train.json").exists()

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and HAVE_DATA), reason="needs CUDA and data")


def _tiny_cfg():
    cfg = copy.deepcopy(CFG)
    cfg["train"].update({"overfit_images": 2, "micro_batch": 1, "effective_batch": 2, "epochs": 2,
                         "eval_every": 1, "num_workers": 0, "max_steps": None})
    cfg["optim"]["warmup_steps"] = 1
    return cfg


def test_resume_restores_optimizer_scheduler_and_epoch(tmp_path):
    t1 = Trainer(_tiny_cfg(), tmp_path)
    t1.setup(resume=False)
    t1.cfg["train"]["epochs"] = 1  # stop after the first epoch
    t1.fit()
    assert (tmp_path / "checkpoints" / "last.pt").exists() and (tmp_path / "metrics.csv").exists()
    ref_opt = t1.optimizer.state_dict()
    ref_model = {k: v.detach().cpu().clone() for k, v in t1.model.state_dict().items()}

    t2 = Trainer(_tiny_cfg(), tmp_path)
    t2.setup(resume=True)
    assert t2.epoch == 1 and t2.global_step == t1.global_step == 1
    assert t2.scheduler.state_dict()["last_epoch"] == t1.scheduler.state_dict()["last_epoch"]
    opt = t2.optimizer.state_dict()
    assert opt["state"].keys() == ref_opt["state"].keys() and len(opt["state"]) > 0
    for k, st in opt["state"].items():
        for name in ("exp_avg", "exp_avg_sq", "step"):
            assert torch.equal(st[name].cpu(), ref_opt["state"][k][name].cpu()), (k, name)
    for k, v in t2.model.state_dict().items():
        assert torch.equal(v.cpu(), ref_model[k]), k
    # continuing trains the remaining epoch only
    t2.fit()
    assert t2.epoch == 2 and t2.global_step == 2
