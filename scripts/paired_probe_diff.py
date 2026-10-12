"""Paired bootstrap of probe-metric differences between encoders (val crops only, CPU, no encoder pass).

Usage:
    python scripts/paired_probe_diff.py --pairs ssl_joint:ssl_con_only ssl_joint:ssl_mae_only \
        ssl_joint:ref_imagenet ssl_con_only:ref_imagenet ssl_mae_only:ref_imagenet \
        --out reports/tables/phase3_paired_diff

Each pair A:B reports metric(A) - metric(B).

Reads the saved probe features runs/probe/<tag>/features_{train,val,val_stain}.pt, refits the same kNN /
linear classifiers as scripts/probe.py (hyperparameters from the probe config) and checks that the point
estimates reproduce reports/tables/probe_<tag>.json. Then, for each metric, scores both encoders on the
SAME bootstrap resamples of val images (same seed as the marginal CIs) and reports metric(a) - metric(b)
with its 95% percentile CI. The 10%-label metric is the mean over the low-label seeds (same subsets for
both encoders); per-seed differences are reported too.

Writes <out>.csv and <out>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from malaria_ssl.ssl.probe import (compute_metrics, knn_predict, linear_predict, paired_bootstrap_diff,
                                   stratified_subset)
from malaria_ssl.utils.config import load_experiment, resolve_path

# (row name, method, eval set, label fraction, metric key)
METRICS = [("auprc_clean", "linear", "val", 1.0, "auprc_infected"),
           ("auprc_stain", "linear", "val_stain", 1.0, "auprc_infected"),
           ("auprc_10pct", "linear", "val", 0.1, "auprc_infected"),
           ("macro_f1", "linear", "val", 1.0, "macro_f1"),
           ("knn_balanced_acc", "knn", "val", 1.0, "balanced_acc")]


def load_feats(tag: str) -> dict[str, dict]:
    d = resolve_path("runs/probe") / tag
    return {s: torch.load(d / f"features_{s}.pt", map_location="cpu", weights_only=False)
            for s in ("train", "val", "val_stain")}


def fit_all(feats: dict[str, dict], pcfg: dict, n_classes: int) -> dict:
    """{(method, eval set, fraction): [val probabilities per seed]} for every entry of METRICS."""
    xtr, ytr = feats["train"]["features"].float(), feats["train"]["label"].long()
    out = {}
    for _, method, eval_set, frac, _ in METRICS:
        if (method, eval_set, frac) in out:
            continue
        seeds = [None] if frac >= 1 else range(pcfg["low_label"]["seeds"])
        probs = []
        for s in seeds:
            idx = (np.arange(len(ytr)) if s is None
                   else stratified_subset(ytr.numpy(), frac, n_classes, seed=pcfg["seed"] + s))
            idx = torch.as_tensor(idx)
            xte = feats[eval_set]["features"].float()
            if method == "knn":
                p = knn_predict(xtr[idx], ytr[idx], xte, n_classes, pcfg["knn"]["k"], pcfg["knn"]["temperature"])
            else:
                p = linear_predict(xtr[idx], ytr[idx], xte, n_classes, pcfg["linear"]["l2"], pcfg["linear"]["max_iter"])
            probs.append(p.numpy())
        out[(method, eval_set, frac)] = probs
    return out


def reference_value(probe: dict, method: str, eval_set: str, frac: float, key: str) -> float:
    """The same point estimate as stored by scripts/probe.py."""
    if frac < 1:
        return probe["low_label"][str(frac)][method]["mean"][key]
    src = probe["methods"] if eval_set == "val" else probe["stain_shift"]["methods"]
    return src[method]["metrics"][key]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pairs", nargs="+", required=True, metavar="A:B")
    ap.add_argument("--config", default="configs/ssl/base.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pcfg = load_experiment(args.config)["probe"]
    pairs = [tuple(x.split(":")) for x in args.pairs]
    if any(len(pr) != 2 for pr in pairs):
        raise SystemExit("pairs must be A:B")
    tags = list(dict.fromkeys(t for pr in pairs for t in pr))
    probes = {t: json.loads((resolve_path("reports/tables") / f"probe_{t}.json").read_text(encoding="utf-8"))
              for t in tags}
    names = probes[tags[0]]["classes"]
    infected = np.array([n in probes[tags[0]]["infected"] for n in names])
    feats = {t: load_feats(t) for t in tags}
    ref = feats[tags[0]]
    for t in tags:   # identical crop sets (same order) are what makes the bootstrap paired
        assert probes[t]["classes"] == names
        for s in ("train", "val", "val_stain"):
            assert torch.equal(feats[t][s]["label"], ref[s]["label"]), (t, s)
            assert torch.equal(feats[t][s]["image_id"], ref[s]["image_id"]), (t, s)
    y, image_ids = ref["val"]["label"].numpy(), ref["val"]["image_id"].numpy()

    torch.manual_seed(0)
    probs = {t: fit_all(feats[t], pcfg, len(names)) for t in tags}

    check = []
    for t in tags:
        for name, method, eval_set, frac, key in METRICS:
            v = float(np.mean([compute_metrics(y, p, infected, names)[key] for p in probs[t][(method, eval_set, frac)]]))
            check.append({"encoder": t, "metric": name, "refit": v,
                          "probe_json": reference_value(probes[t], method, eval_set, frac, key)})
    check = pd.DataFrame(check)
    check["abs_dev"] = (check["refit"] - check["probe_json"]).abs()
    print("CPU refit vs probe json (max abs deviation per metric):")
    print(check.groupby("metric")["abs_dev"].max().to_string(float_format=lambda v: f"{v:.2e}"))

    rows, per_seed = [], {}
    for a, b in pairs:
        for name, method, eval_set, frac, key in METRICS:
            pa, pb = probs[a][(method, eval_set, frac)], probs[b][(method, eval_set, frac)]
            r = paired_bootstrap_diff(y, pa, pb, image_ids, infected, names, key, pcfg["bootstrap"], pcfg["seed"])
            rows.append({"a": a, "b": b, "metric": name, "value_a": r["a"], "value_b": r["b"],
                         "diff": r["diff"], "ci_low": r["ci95"]["low"], "ci_high": r["ci95"]["high"],
                         "frac_le_0": r["frac_le_0"], "excludes_0": bool(r["ci95"]["low"] > 0 or r["ci95"]["high"] < 0)})
            if len(pa) > 1:
                per_seed[f"{a} - {b}: {name}"] = [
                    compute_metrics(y, p, infected, names)[key] - compute_metrics(y, q, infected, names)[key]
                    for p, q in zip(pa, pb)]
    df = pd.DataFrame(rows)

    out = resolve_path(args.out)
    df.to_csv(out.with_suffix(".csv"), index=False, float_format="%.4f")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "per_seed_diff": per_seed, "refit_check": check.to_dict("records"),
         "n_resamples": pcfg["bootstrap"], "seed": pcfg["seed"], "n_val_images": int(len(np.unique(image_ids))),
         "n_val_crops": int(len(y))}, indent=2), encoding="utf-8")
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    for k, v in per_seed.items():
        print(k, "per seed:", [round(x, 4) for x in v])
    print(f"wrote {out.with_suffix('.csv')} and .json")


if __name__ == "__main__":
    main()
