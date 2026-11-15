"""scripts/adapt.py evaluate: per-pair saving / resume gives the same tables (synthetic data, CPU, no saved features)."""
import importlib.util
import json
import logging
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "adapt.py"
RUNS = ["box_a", "box_b"]
ENCODERS = ["enc_x", "enc_y"]
TABLES = ["runs.csv", "diff.csv", "info.csv", "bins.csv", "bins_diff.csv", "meta.json"]


def load_script(path=SCRIPT, name="adapt_script"):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _split(ids, seed, shift):
    """GT (binary, cat 2 = infected) and candidates: jittered GT boxes + false positives, 8-d features."""
    rng = np.random.default_rng(seed)
    images, anns, cand = [], [], {"image_id": [], "box": [], "obj": [], "x": []}
    for i in ids:
        images.append({"id": i, "width": 400, "height": 400, "file_name": f"{i}.png"})
        for k in range(int(rng.integers(6, 12))):
            x, y = (k % 4) * 90 + 10, (k // 4) * 90 + 10
            inf = bool(rng.random() < 0.3)
            anns.append({"id": len(anns) + 1, "image_id": i, "bbox": [x, y, 50, 50], "area": 2500.0, "iscrowd": 0,
                         "category_id": 2 if inf else 1})
            cand["image_id"].append(i)
            cand["box"].append([x + rng.normal(0, 2), y + rng.normal(0, 2), 50, 50])
            cand["obj"].append(rng.uniform(0.4, 1.0))
            cand["x"].append(rng.normal(0, 1, 8) + (1.5 if inf else 0.0) * np.eye(8)[0] + shift)
        for _ in range(2):   # false positives
            cand["image_id"].append(i)
            cand["box"].append([rng.uniform(0, 300), 350, 40, 40])
            cand["obj"].append(rng.uniform(0.0, 0.6))
            cand["x"].append(rng.normal(0, 1, 8) + shift)
    gt = {"images": images, "annotations": anns,
          "categories": [{"id": 1, "name": "uninfected"}, {"id": 2, "name": "infected"}]}
    c = {"image_id": np.array(cand["image_id"]), "box": np.array(cand["box"], float), "obj": np.array(cand["obj"]),
         "x": torch.tensor(np.array(cand["x"]), dtype=torch.float32)}
    return c, gt


def install_fakes(mod, monkeypatch, tmp: Path, target: str, n: int = 20, runs=None, encoders=None, methods=None):
    """Replace the data loaders of `mod` by synthetic ones; returns (args, cfg) for cmd_evaluate."""
    pool_runs, pool_enc = runs or RUNS, encoders or ENCODERS
    tmp.mkdir(parents=True, exist_ok=True)
    def seed_of(run, tag):
        return pool_runs.index(run) * 10 + pool_enc.index(tag)

    def of_run(c, run):   # the GT is shared; candidate boxes / objectness depend on the box run
        rng = np.random.default_rng(300 + pool_runs.index(run))
        return {**c, "box": c["box"] + rng.normal(0, 1.5, c["box"].shape),
                "obj": np.clip(c["obj"] + rng.normal(0, 0.05, len(c["obj"])), 0, 1)}

    def bbbc_split(split, run, tag, limit=None, stain=False):
        c, gt = _split(list(range(1, 9)), 100, 0.0)
        c = of_run(c, run)
        g = torch.Generator().manual_seed(seed_of(run, tag) + (50 if stain else 0))
        c["x"] = c["x"] + 0.2 * torch.randn(c["x"].shape, generator=g) + (0.7 if stain else 0.0)
        return c, gt

    def load_target(t, run, tag, cfg, limit=None):
        ids = list(range(101, 113))
        c, gt = _split(ids, 200, 0.5)
        c = of_run(c, run)
        g = torch.Generator().manual_seed(seed_of(run, tag) + 99)
        c["x"] = c["x"] + 0.2 * torch.randn(c["x"].shape, generator=g)
        ga = mod.gt_arrays(gt)
        true_img = {i: float(ga["infected"][ga["image_id"] == i].mean()) for i in ids}
        patient = {i: f"p{(i - 101) // 3}" if t == "nih" else str(i) for i in ids}
        true_pat = {p: float(np.mean([true_img[i] for i in ids if patient[i] == p])) for p in set(patient.values())}
        return c, gt, ids, patient, true_img, true_pat

    def classifier(tag, pcfg):
        w = torch.linspace(1.0, 0.2, 8) * (1.0 + 0.3 * pool_enc.index(tag))
        return lambda x: torch.sigmoid(x.float() @ w - 1.0).numpy().astype(np.float64)

    sel = {"coral": {"lam": 1.0}, "cbr": {"iters": 1, "mode": "hard"}, "irc": {"scale": False, "shrink": 16}}
    monkeypatch.setattr(mod, "bbbc_split", bbbc_split)
    monkeypatch.setattr(mod, "load_target", load_target)
    monkeypatch.setattr(mod, "classifier", classifier)
    monkeypatch.setattr(mod, "check_selected", lambda cfg: sel)
    monkeypatch.setattr(mod, "load_experiment", lambda p: json.loads(Path(p).read_text(encoding="utf-8")))
    probe = tmp / "probe.yaml"   # JSON is valid YAML
    probe.write_text(json.dumps({"probe": {"linear": {"l2": 1e-3, "max_iter": 500}}}), encoding="utf-8")
    cfg = {"probe_config": str(probe), "runs": runs or RUNS, "mean_name": "box_2s", "encoders": encoders or ENCODERS,
           "iou": 0.5, "max_dets": 300, "ece_bins": 15, "bootstrap": 1000, "seed": 42, "selected": sel,
           "methods": methods or ["none", "none_stainthr", "adabn", "coral", "cbr", "irc", "irc_patient", "none+em",
                                  "none+bbse", "irc+em"],
           "parasitemia_bins": [0.0, 0.01, 0.05, 0.20], "bins_methods": ["none", "irc", "irc+em"],
           "tables": str(tmp / "tables" / "adapt")}
    config = tmp / "adapt.yaml"
    config.write_text(json.dumps(cfg), encoding="utf-8")
    args = Namespace(config=str(config), limit=None, target=target, n=n, force=False, encoders=None, runs=None)
    return args, cfg


def read_tables(tmp: Path, target: str) -> dict[str, str]:
    base = tmp / "tables" / f"adapt_{target}"
    return {t: Path(f"{base}_{t}").read_text(encoding="utf-8") for t in TABLES if Path(f"{base}_{t}").exists()}


@pytest.mark.parametrize("target", ["nih", "bbbc_test"])
def test_resume_after_crash_gives_identical_tables(tmp_path, monkeypatch, target, caplog):
    mod = load_script()
    args, cfg = install_fakes(mod, monkeypatch, tmp_path, target)
    mod.cmd_evaluate(args, cfg)
    full = read_tables(tmp_path, target)
    assert {"runs.csv", "diff.csv", "info.csv", "meta.json"} <= set(full)
    assert ("bins.csv" in full) == (target == "nih")
    parts = tmp_path / "tables" / f"adapt_{target}_parts"
    assert sorted(p.name for p in parts.iterdir()) == sorted(f"{r}__{t}.pt" for t in ENCODERS for r in RUNS)

    # "power cut": only the first pair survived, plus a half-written .tmp of the second
    for p in parts.iterdir():
        if p.name != f"{RUNS[0]}__{ENCODERS[0]}.pt":
            p.unlink()
    (parts / f"{RUNS[1]}__{ENCODERS[0]}.tmp").write_bytes(b"\x00partial")
    for f in (tmp_path / "tables").glob(f"adapt_{target}_*.*"):
        f.unlink()
    computed = []
    real_pair = mod.evaluate_pair
    monkeypatch.setattr(mod, "evaluate_pair", lambda run, tag, *a: computed.append((run, tag)) or real_pair(run, tag, *a))
    mod.cmd_evaluate(args, cfg)
    assert (RUNS[0], ENCODERS[0]) not in computed and len(computed) == len(RUNS) * len(ENCODERS) - 1
    assert read_tables(tmp_path, target) == full

    # everything cached: nothing recomputed, same tables; --force recomputes all, same tables
    computed.clear()
    mod.cmd_evaluate(args, cfg)
    assert computed == [] and read_tables(tmp_path, target) == full
    mod.cmd_evaluate(Namespace(**{**vars(args), "force": True}), cfg)
    assert len(computed) == len(RUNS) * len(ENCODERS) and read_tables(tmp_path, target) == full

    # another config: warning and recomputed (not skipped)
    computed.clear()
    Path(args.config).write_text(json.dumps({**cfg, "comment": "edited"}), encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="adapt"):
        mod.cmd_evaluate(args, cfg)
    assert len(computed) == len(RUNS) * len(ENCODERS)
    assert any("different ['config_sha256']" in r.message for r in caplog.records)
    meta_new = json.loads(read_tables(tmp_path, target)["meta.json"])
    assert meta_new["config_sha256"] != json.loads(full["meta.json"])["config_sha256"]
    assert {k: v for k, v in read_tables(tmp_path, target).items() if k != "meta.json"} == \
        {k: v for k, v in full.items() if k != "meta.json"}


def test_key_covers_resamples_limit_and_probe_config(tmp_path, monkeypatch):
    mod = load_script()
    args, cfg = install_fakes(mod, monkeypatch, tmp_path, "bbbc_test", n=10)
    mod.cmd_evaluate(args, cfg)
    methods = [m for m in cfg["methods"] if m != "irc_patient"]   # as cmd_evaluate for bbbc_test
    pcfg = mod.load_experiment(cfg["probe_config"])["probe"]
    part = tmp_path / "tables" / "adapt_bbbc_test_parts" / f"{RUNS[0]}__{ENCODERS[0]}.pt"
    assert mod.load_part(part, mod.part_key(args, cfg, methods, pcfg)) is not None
    assert mod.load_part(part, mod.part_key(Namespace(**{**vars(args), "n": 11}), cfg, methods, pcfg)) is None
    assert mod.load_part(part, mod.part_key(Namespace(**{**vars(args), "limit": 5}), cfg, methods, pcfg)) is None
    # probe config edited (e.g. linear.max_iter): both the file hash and the resolved section change
    Path(cfg["probe_config"]).write_text(json.dumps({"probe": {"linear": {"l2": 1e-3, "max_iter": 501}}}), "utf-8")
    pcfg2 = mod.load_experiment(cfg["probe_config"])["probe"]
    k1, k2 = mod.part_key(args, cfg, methods, pcfg), mod.part_key(args, cfg, methods, pcfg2)
    assert k1["probe_resolved_sha256"] != k2["probe_resolved_sha256"]
    assert mod.load_part(part, k2) is None


def test_pair_order_and_filters():
    mod = load_script()
    enc = ["ref_dino", "ssl_v2_b", "ssl_v2_b_seed43", "ref_imagenet", "ssl_v2_cont", "ssl_v2_cont_seed43"]
    runs = ["r42", "r43", "r44"]
    order = mod.pair_order(enc, runs)
    assert [t for _, t in order[::3]] == ["ssl_v2_b", "ref_dino", "ssl_v2_b_seed43", "ref_imagenet", "ssl_v2_cont",
                                          "ssl_v2_cont_seed43"]
    assert sorted(order) == sorted((r, t) for t in enc for r in runs) and [r for r, _ in order[:3]] == runs
    assert mod.pick(None, enc, "encoder") == enc
    assert mod.pick(["ssl_v2_b", "ref_dino"], enc, "encoder") == ["ref_dino", "ssl_v2_b"]   # config order
    with pytest.raises(SystemExit, match=r"unknown encoder \['nope'\]; available: \['ref_dino'"):
        mod.pick(["ssl_v2_b", "nope"], enc, "encoder")


def test_order_does_not_change_tables(tmp_path, monkeypatch):
    mod = load_script()
    args, cfg = install_fakes(mod, monkeypatch, tmp_path / "a", "nih", n=15)
    mod.cmd_evaluate(args, cfg)
    ref = read_tables(tmp_path / "a", "nih")
    monkeypatch.setattr(mod, "PRIORITY_ENCODERS", tuple(reversed(ENCODERS)))
    args2, cfg2 = install_fakes(mod, monkeypatch, tmp_path / "b", "nih", n=15)
    mod.cmd_evaluate(args2, cfg2)
    got = read_tables(tmp_path / "b", "nih")
    assert {k: v for k, v in got.items() if k != "meta.json"} == {k: v for k, v in ref.items() if k != "meta.json"}


def test_subset_then_rest_equals_full(tmp_path, monkeypatch):
    mod = load_script()
    args, cfg = install_fakes(mod, monkeypatch, tmp_path / "full", "nih", n=15)
    mod.cmd_evaluate(args, cfg)
    full = read_tables(tmp_path / "full", "nih")

    args, cfg = install_fakes(mod, monkeypatch, tmp_path / "split", "nih", n=15)
    computed = []
    real_pair = mod.evaluate_pair
    monkeypatch.setattr(mod, "evaluate_pair", lambda run, tag, *a: computed.append((run, tag)) or real_pair(run, tag, *a))
    mod.cmd_evaluate(Namespace(**{**vars(args), "encoders": [ENCODERS[1]], "runs": [RUNS[0]]}), cfg)
    assert computed == [(RUNS[0], ENCODERS[1])]
    base = tmp_path / "split" / "tables"
    assert not (base / "adapt_nih_runs.csv").exists()   # the full tables are not written by a subset
    sub_meta = json.loads((base / "adapt_nih_subset_meta.json").read_text(encoding="utf-8"))
    assert sub_meta["subset"] and sub_meta["encoders"] == [ENCODERS[1]] and sub_meta["runs"] == [RUNS[0]]
    sub_runs = (base / "adapt_nih_subset_runs.csv").read_text(encoding="utf-8").splitlines()
    full_runs = full["runs.csv"].splitlines()
    assert sub_runs[1:] == [ln for ln in full_runs[1:] if f",{RUNS[0]},{ENCODERS[1]}," in ln]   # same numbers
    computed.clear()
    mod.cmd_evaluate(args, cfg)   # the rest later, without filters
    assert len(computed) == len(RUNS) * len(ENCODERS) - 1 and (RUNS[0], ENCODERS[1]) not in computed
    got = read_tables(tmp_path / "split", "nih")
    assert {k: v for k, v in got.items() if k != "meta.json"} == {k: v for k, v in full.items() if k != "meta.json"}
    with pytest.raises(SystemExit, match="unknown box run"):
        mod.cmd_evaluate(Namespace(**{**vars(args), "runs": ["a_coco_detr_binary_seed99"]}), cfg)
