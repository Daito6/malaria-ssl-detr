# CLAUDE.md

## Project goal
Master's thesis: "Classification and segmentation of medical images with Detection Transformer
(DETR)". Self-supervised pretraining of a ViT backbone on unlabeled blood-smear images
(MAE reconstruction + contrastive InfoNCE, jointly), then plugging the backbone into DETR
(transformer initialized from COCO, adapted with LoRA / lower LR) and comparing against
baselines on malaria detection.

Dataset: BBBC041v1 (P. vivax blood smears, 1364 images, ~80k cells). Classes: red blood cell,
leukocyte, gametocyte, ring, trophozoite, schizont, difficult; >95% of cells are RBC.

The old bachelor repo (github.com/Daito6/malaria-detection-detr) is reference only — do not copy
its code or structure.

## Rules
- Talk to the user in Ukrainian. Code, comments, docstrings, commit messages in English.
  README and reports in Ukrainian.
- Plan first, wait for approval. Small steps, one logical commit per step. Run the tests.
- Never invent numbers: every figure/statistic comes from real runs of the code.
- Never modify raw data (`data/raw/`). Config-driven (YAML), fixed seeds, reproducible.
- ALL logic lives in `src/` and `scripts/`. Notebooks only load saved artifacts, call
  functions from `src/`, and plot/print tables.
- The test split is used only for final evaluation, never for tuning.
- No over-engineering: no unrequested frameworks or speculative abstractions.
- Windows: pathlib everywhere; DataLoader code under `if __name__ == "__main__":`,
  sensible `num_workers`, `persistent_workers=True`.

## Environment
Windows 11, RTX 5060 Ti 16 GB (Blackwell, sm_120), 32 GB RAM, conda env `medthesis`
(Python 3.11). Run things with `conda run -n medthesis python ...`.

## Layout
```
configs/               data.yaml (later: pretrain/, detr/, experiments/)
src/malaria_ssl/       data/, ssl/, detr/, eval/, utils/   (pip install -e .)
scripts/               CLI entry points
notebooks/             01_eda.ipynb ...
tests/                 pytest
reports/figures/, reports/tables/
data/raw, data/processed, data/splits   (git-ignored)
runs/                                   (git-ignored)
```

## Data pipeline
```
python scripts/check_env.py
python scripts/prepare_data.py      # download -> convert -> split (idempotent)
python scripts/compute_stats.py     # reports/tables/
python scripts/draw_boxes.py        # reports/figures/box_check/
pytest -q
```
Facts worth remembering (details in `reports/phase1_summary.md`):
- The zip has 1328 images (not 1364), all annotated; no extra unlabeled images.
- Source bbox key is `bounding_box` with r = row = y, c = col = x (verified).
- train/val: PNG 1600x1200; test: JPG 1944x1383, cells ~1.27x larger in px -> domain shift.
- Max 223 boxes per image; 10.5% of images have >100 -> DETR needs >100 queries.
- Leukocyte has 0 boxes in test.
- If Git Bash `python.exe` hits "Application Control policy has blocked" DLL errors,
  use `conda run -n medthesis python ...` from PowerShell.
- Smart App Control blocks some pip DLLs: scikit-learn must stay uninstalled (else
  `import transformers` fails); grpcio is pinned to 1.76.0 (newer builds are blocked).

## DETR pipeline (phase 2)
```
python scripts/train_detr.py --config configs/experiments/<exp>.yaml [--resume] [--set key=value]
python scripts/evaluate.py runs/<name> --split val --bootstrap   # val first: picks F1 thresholds
python scripts/evaluate.py runs/<name> --split test --bootstrap  # reuses val thresholds
```
Non-obvious decisions (details in `reports/phase2_summary.md`):
- Model is called WITHOUT labels; loss is ours (`detr/loss.py`, HF matcher/ImageLoss reused)
  so class weights work. Aux outputs come from decoder hooks: HF `auxiliary_loss=True`
  layer-norms the running decoder state and breaks the COCO-pretrained model.
- ResNet stem + layer1 are frozen explicitly (HF v5 does not). scratch_r50 loads
  `resnet50.tv_in1k` explicitly and verifies it.
- COCOeval uses maxDets [1,10,300]; never read `ev.stats[0]` (hard-coded maxDets=100 -> -1);
  summaries come from `eval["precision"/"recall"]`. Only area range "all" is evaluated.
- Batches are deterministic via `SeededSampler` yielding (index, seed) pairs.
- Measured: micro-batch 8 fits (6.98 GiB probe peak), ~58 s/epoch, ~43 min per 40-epoch run.

## Status
- Phase 1 (scaffold + data pipeline + EDA): done.
- Phase 2 (detection pipeline, eval, baseline A tooling, dry runs): done, awaiting user review.
  Full baseline-A runs not launched yet (user runs them).
- Next: SSL pretraining (MAE + InfoNCE) — not started.
