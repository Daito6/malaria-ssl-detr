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
configs/               data.yaml, detr/, experiments/, ssl/
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

## Evaluation protocol (applies to every phase)
- Headline metric for comparing variants: **infected AP50 on test** (threshold-free).
- Also report F1 / precision / recall at the per-class thresholds selected on val (max F1).
- Test is NEVER used for tuning, model/lambda selection or threshold selection; SSL lambdas are
  chosen by probe metrics on VAL crops only.
- The val -> test domain shift is a central theme of the thesis. Baseline A binary: infected AP50
  val 0.778 -> test 0.551; at the val threshold recall 0.883 -> 0.376; 35% of test infected cells
  are found but called uninfected (`scripts/diagnose_shift.py`, `reports/phase3_summary.md`).

## SSL pipeline (phase 3)
```
python scripts/pretrain.py --config configs/ssl/<exp>.yaml [--resume] [--set key=value] [--name ...]
python scripts/probe.py --run runs/ssl_<exp>        # or --encoder <pt> | --ref random | --ref imagenet
python scripts/diagnose_shift.py runs/<binary DETR run>
python scripts/compare_probes.py --tags <probe tags...> --out reports/tables/<name>.csv   # reads probe json only
powershell -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/<queue>.txt [-DryRun]
```
Queue lines are `<run name> | <command>`; output goes to `runs/<run name>/queue.log`; failures do not
stop the queue; commands logged `QUEUE DONE` are skipped on rerun; pretrain/train_detr get `--resume`
when `runs/<run name>/checkpoints/last.pt` exists (so the run name must be the run directory).
Run it from a PowerShell with `conda activate medthesis` (commands call plain `python`).
Config chain: `configs/ssl/data.yaml` <- `probe.yaml` <- `base.yaml` <- experiment files.
Non-obvious decisions (details in `reports/phase3_summary.md`):
- SSL trains on train images only (labels unused); val only for monitoring; test never read.
  Images are cached once as uint8 memmaps at long side 960 in `data/processed/ssl_cache/`.
- Views: random DETR train scale L in [640, 960] -> 260 px anchor at scale L -> two crops
  (area 0.5-1, aspect 3/4-4/3) -> 224. Cells stay at DETR scale (x0.86-1.22).
- Encoder = timm `vit_small_patch16_224`, random init, `dynamic_img_size=True`; exported as a
  plain timm state dict (`encoder_final.pt`) that loads strict=True. A branch with lambda 0 is
  not computed. `uncertainty` weighting = sum exp(-s_i) L_i + s_i.
- Batch fixed at 128 pairs (256 pairs need 20.8 GiB); no gradient accumulation (InfoNCE negatives).
  Measured: joint 258 pairs/s, mae_only 542 (loader-bound), con_only 311; epoch = 25600 pairs.
- Probe: box-centred 224 windows at long side 800, mean of patch tokens inside the box; RBC
  subsampled 4000/2000, difficult excluded; kNN + L-BFGS logreg with fixed hyperparameters;
  bootstrap over val images. Random-init linear AUPRC infected is already 0.924 (ImageNet 0.981):
  the 100%-label binary metric has little headroom.
- Stain-shifted probe (`probe.stain_shift`): the same classifiers, fitted on CLEAN train crops, are
  evaluated on val crops with a fixed per-image shift (seed = (2024, image id); HED 0.08, jitter
  0.3/0.3/0.3/0.05, blur 0.5-1.0, JPEG 30-60, all p=1) applied before cropping. Never touches test.
  Linear AUPRC infected clean -> shifted: random 0.924 -> 0.525, ImageNet 0.981 -> 0.943.
- Lambda selection (`select_run`, val only): linear AUPRC infected clean -> stain-shifted linear
  AUPRC -> 10%-label linear AUPRC -> linear macro-F1; a run is tied if its value is inside the
  leader's interval (95% CI lower bound; mean - std for the 10%-label step).
- `runs/dry_a_*` prefix also matches phase-2 DETR dry runs; SSL overfit run is `dry_a_overfit_mae64`.
- Checkpoints (`utils/run.py`): .tmp + fsync + os.replace; last.pt keeps the previous state as
  last.prev.pt and `load_checkpoint` falls back to it (lc02 once died on a truncated last.pt).

## Status
- Phase 1 (scaffold + data pipeline + EDA): done.
- Phase 2 (detection pipeline, eval module, baseline A tooling): done. `a_coco_detr_binary` trained
  and evaluated on val/test. `a_coco_detr_multiclass` and `a_scratch_r50_binary` not run yet.
- Phase 3 (SSL code, probe incl. stain-shifted probe, dry runs, references, shift diagnostic, run
  queue): done. `phase3_sweep` queue finished (5 sweeps + probes, `a_coco_detr_binary_stain` + eval).
  `select_run` picks lc005 (decided at the 10%-label step), but the user's final rule (stain-shifted
  AUPRC: lr5e4 0.973 within lc02's CI [0.956, 0.987]) chose lambda_con 0.2 + blr 5e-4, now in
  `configs/ssl/base.yaml` (shared lr for joint / mae_only / con_only, 100 epochs). Sweep configs pin
  their own blr 1.5e-4. `phase3_final` queue finished (ssl_joint resumed once at epoch 66); comparison in
  `reports/tables/phase3_final_comparison.csv`. Val probe: joint beats mae_only beyond CIs only on
  stain-shifted AUPRC (0.984 [0.971, 0.992] vs 0.946 [0.922, 0.966]); joint does NOT beat con_only beyond
  any 95% CI (10%-label 0.972±0.005 vs 0.954±0.010 separates only by mean±std). No collapse in any run.
  `a_coco_detr_binary_stain`: test infected AP50 0.583 [0.500, 0.649] vs baseline 0.551 (CIs overlap).
  Next: phase 4 (SSL backbone in DETR) — not started.
- User decisions: extra lr run (blr 5e-4) added; tie-breaks as in the selection rule; mae_only
  embedding anisotropy (std x sqrt(d) ~ 0.09) is acceptable, keep monitoring.
