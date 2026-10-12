import json

import pandas as pd

from malaria_ssl.detr.compare import run_summary, select_by_simplicity, val_curve


def _row(run, ap, lo, hi):
    return {"run": run, "AP50_infected": ap, "AP50_infected_lo": lo, "AP50_infected_hi": hi}


def test_select_picks_simplest_tied_run():
    rows = [_row("full", 0.77, 0.72, 0.83), _row("s32", 0.70, 0.65, 0.75), _row("llrd", 0.80, 0.75, 0.85)]
    sel = select_by_simplicity(rows)
    assert sel["leader"] == "llrd"
    assert sel["tied"] == ["full", "llrd"]  # s32 0.70 < leader's lower bound 0.75
    assert sel["winner"] == "full"


def test_select_leader_wins_when_alone():
    rows = [_row("full", 0.60, 0.55, 0.65), _row("llrd", 0.80, 0.75, 0.85)]
    assert select_by_simplicity(rows)["winner"] == "llrd"


def test_run_summary_and_curve(tmp_path):
    metrics = {"checkpoint_epoch": 3,
               "coco": {"per_class": {"uninfected": {"AP50": 0.9}, "infected": {"AP50": 0.7}}},
               "bootstrap": {"ci95": {"AP50/uninfected": {"low": 0.85, "high": 0.95},
                                      "AP50/infected": {"low": 0.6, "high": 0.8}}}}
    (tmp_path / "metrics_val.json").write_text(json.dumps(metrics), encoding="utf-8")
    (tmp_path / "train.log").write_text("x INFO trainer: trainable parameters: 10 of 12; groups {}\n",
                                        encoding="utf-8")
    pd.DataFrame({"epoch": [0, 1, 2, 3], "epoch_time_s": [60.0] * 4,
                  "val_time_s": [None, 30.0, None, 30.0],
                  "val_AP50/infected": [None, 0.5, None, 0.7]}).to_csv(tmp_path / "metrics.csv", index=False)
    row = run_summary(tmp_path)
    assert row["AP50_infected"] == 0.7 and row["AP50_infected_lo"] == 0.6
    assert row["trainable_params"] == 10 and row["total_params"] == 12
    assert row["epochs"] == 4 and row["train_min"] == 5.0
    curve = val_curve(tmp_path)
    assert list(curve.index) == [2, 4] and list(curve) == [0.5, 0.7]
