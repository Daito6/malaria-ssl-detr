"""Phase 5: held-out-fold test-crop probe table and paired bootstrap of differences (CPU, saved features only).

An encoder adapted on test fold F is only ever evaluated on the OTHER fold. A logical encoder is therefore
given as NAME=TAG1,TAG2: TAG1 is the encoder tag evaluated on fold 1, TAG2 the one evaluated on fold 2
(NAME=TAG when the same encoder is evaluated on both folds). Features come from
`probe.py --eval-split test --eval-fold k` (runs/probe/test_fold<k>_<tag>/features_{train,test}.pt).

Usage:
    python scripts/paired_fold_diff.py \
        --enc imagenet=ref_imagenet imagenet_cont=ssl_imagenet_cont \
              imagenet_da=ssl_imagenet_da_f2,ssl_imagenet_da_f1 ... \
        --pairs imagenet_da:imagenet imagenet_da:imagenet_cont ... --out reports/tables/phase5_da

Metrics (100% labels, classifiers fitted on train crops as in probe.py): linear and kNN AUPRC infected.
For each fold the classifiers are refitted (checked against the probe json) and both encoders of a pair are
scored on the same 1000 bootstrap resamples of that fold's images (probe seed). `avg` = mean of the two
folds; its CI pairs resample r of fold 1 with resample r of fold 2.

--mean NAME=A+B adds a derived encoder whose metric is the mean of encoders A and B (e.g. two SSL seeds), on the
same resamples; it can be used in --pairs like any other encoder.

Writes <out>_runs.csv (value + 95% CI per encoder, fold 1 / fold 2 / avg), <out>_diff.csv and <out>.json.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from malaria_ssl.ssl.probe import average_precision, compute_metrics, fit_all, image_resamples
from malaria_ssl.utils.config import load_experiment, resolve_path

# (row name, method, eval set, label fraction, metric key)
METRICS = [("linear_auprc", "linear", "test", 1.0, "auprc_infected"),
           ("knn_auprc", "knn", "test", 1.0, "auprc_infected")]
FOLDS = (1, 2)


def parse_enc(items: list[str]) -> dict[str, dict[int, str]]:
    out = {}
    for item in items:
        name, _, tags = item.partition("=")
        t = tags.split(",")
        if not name or len(t) not in (1, 2):
            raise SystemExit(f"bad --enc {item!r}: NAME=TAG or NAME=TAG1,TAG2")
        out[name] = {1: t[0], 2: t[-1]}
    return out


def ci(v: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--enc", nargs="+", required=True, metavar="NAME=TAG1[,TAG2]")
    ap.add_argument("--pairs", nargs="+", required=True, metavar="A:B")
    ap.add_argument("--mean", nargs="*", default=[], metavar="NAME=A+B", help="derived encoder: mean metric of A and B")
    ap.add_argument("--config", default="configs/ssl/base.yaml")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    pcfg = load_experiment(args.config)["probe"]
    encs = parse_enc(args.enc)
    means = {}
    for item in args.mean:
        name, _, parts = item.partition("=")
        members = parts.split("+")
        if not name or name in encs or len(members) < 2 or not set(members) <= set(encs):
            raise SystemExit(f"bad --mean {item!r}: NAME=A+B with A, B given in --enc")
        means[name] = members
    names_all = [*encs, *means]
    pairs = [tuple(x.split(":")) for x in args.pairs]
    if any(len(p) != 2 or p[0] not in names_all or p[1] not in names_all for p in pairs):
        raise SystemExit("pairs must be A:B with A, B given in --enc or --mean")
    n_res = pcfg["bootstrap"]

    values, points, check, meta = {}, {}, [], {}
    for k in FOLDS:
        probes, feats = {}, {}
        for name, tags in encs.items():
            tag = f"test_fold{k}_{tags[k]}"
            probes[name] = json.loads((resolve_path("reports/tables") / f"probe_{tag}.json").read_text(encoding="utf-8"))
            if probes[name].get("eval_fold", {}).get("fold") != k:
                raise SystemExit(f"probe_{tag}.json is not a fold-{k} evaluation")
            d = resolve_path("runs/probe") / tag
            feats[name] = {s: torch.load(d / f"features_{s}.pt", map_location="cpu", weights_only=False)
                           for s in ("train", "test")}
        ref = next(iter(feats.values()))
        names = probes[next(iter(probes))]["classes"]
        infected = np.array([n in probes[next(iter(probes))]["infected"] for n in names])
        for name in encs:   # identical crop sets (same order) are what makes the bootstrap paired
            for s in ("train", "test"):
                assert torch.equal(feats[name][s]["label"], ref[s]["label"]), (name, k, s)
                assert torch.equal(feats[name][s]["image_id"], ref[s]["image_id"]), (name, k, s)
        y, img = ref["test"]["label"].numpy(), ref["test"]["image_id"].numpy()
        y_bin = infected[y]
        resamples = list(image_resamples(img, n_res, pcfg["seed"]))
        meta[k] = {"n_images": int(len(np.unique(img))), "n_crops": int(len(y)), "n_infected_crops": int(y_bin.sum())}
        torch.manual_seed(0)
        for name in encs:
            probs = fit_all(feats[name], pcfg, len(names), METRICS)
            for mname, method, ev, frac, key in METRICS:
                p = probs[(method, ev, frac)][0]
                score = p[:, infected].sum(1)
                points[(name, k, mname)] = average_precision(y_bin, score)
                values[(name, k, mname)] = np.array([average_precision(y_bin[i], score[i]) for i in resamples])
                ref_v = probes[name]["methods"][method]["metrics"][key]
                assert abs(compute_metrics(y, p, infected, names)[key] - points[(name, k, mname)]) < 1e-12
                check.append({"encoder": name, "fold": k, "metric": mname, "refit": points[(name, k, mname)],
                              "probe_json": ref_v, "abs_dev": abs(points[(name, k, mname)] - ref_v)})
    for name, members in means.items():   # same resamples for every member -> the mean is paired too
        for k in FOLDS:
            for mname, *_ in METRICS:
                points[(name, k, mname)] = float(np.mean([points[(m, k, mname)] for m in members]))
                values[(name, k, mname)] = np.mean([values[(m, k, mname)] for m in members], axis=0)
    check = pd.DataFrame(check)
    print("CPU refit vs probe json (max abs deviation per metric):")
    print(check.groupby("metric")["abs_dev"].max().to_string(float_format=lambda v: f"{v:.2e}"))

    def series(name: str, mname: str) -> dict[str, tuple[float, np.ndarray]]:
        p1, p2 = points[(name, 1, mname)], points[(name, 2, mname)]
        v1, v2 = values[(name, 1, mname)], values[(name, 2, mname)]
        return {"fold1": (p1, v1), "fold2": (p2, v2), "avg": ((p1 + p2) / 2, (v1 + v2) / 2)}

    runs = []
    labels = {**{n: (t[1], t[2]) for n, t in encs.items()},
              **{n: ("mean(" + "+".join(m) + ")",) * 2 for n, m in means.items()}}
    for name, tags in labels.items():
        for mname, *_ in METRICS:
            for ev, (p, v) in series(name, mname).items():
                lo, hi = ci(v)
                runs.append({"encoder": name, "tag_fold1": tags[0], "tag_fold2": tags[1], "eval": ev, "metric": mname,
                             "value": p, "ci_low": lo, "ci_high": hi})
    diffs = []
    for a, b in pairs:
        for mname, *_ in METRICS:
            sa, sb = series(a, mname), series(b, mname)
            for ev in sa:
                d, dv = sa[ev][0] - sb[ev][0], sa[ev][1] - sb[ev][1]
                lo, hi = ci(dv)
                diffs.append({"a": a, "b": b, "eval": ev, "metric": mname, "value_a": sa[ev][0], "value_b": sb[ev][0],
                              "diff": d, "ci_low": lo, "ci_high": hi, "frac_le_0": float(np.mean(dv <= 0)),
                              "excludes_0": bool(lo > 0 or hi < 0)})
    runs, diffs = pd.DataFrame(runs), pd.DataFrame(diffs)

    out = resolve_path(args.out)
    runs.to_csv(out.with_name(out.name + "_runs.csv"), index=False, float_format="%.4f")
    diffs.to_csv(out.with_name(out.name + "_diff.csv"), index=False, float_format="%.4f")
    out.with_suffix(".json").write_text(json.dumps(
        {"encoders": {n: {str(k): t for k, t in v.items()} for n, v in encs.items()}, "means": means, "folds": meta,
         "n_resamples": n_res, "seed": pcfg["seed"], "runs": runs.to_dict("records"), "diffs": diffs.to_dict("records"),
         "refit_check": check.to_dict("records")}, indent=2), encoding="utf-8")
    print(runs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(diffs.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"wrote {out}_runs.csv, {out}_diff.csv and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
