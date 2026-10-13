# Фаза 4 — SSL ViT-S/16 як backbone у DETR (у роботі)

Усі числа — з реальних запусків у `runs/` і тестів. Test-спліт у фазі 4 поки не використовувався.

## Кроки 1–2: код (CPU-тести)
- `vit_detr` (`src/malaria_ssl/detr/build.py`): трансформер, queries і box head з COCO-DETR, нова класова
  голова й нова `input_projection` (384→256 для stride 16; 1536→256 для stride 32 через 2×2 space-to-depth).
  `ViTBackbone` має інтерфейс HF `DetrConvEncoder`, перераховує вхід у нормалізацію енкодера (ImageNet ViT
  `augreg`: mean = std = 0.5; random init і SSL-енкодери — ImageNet-статистика) і маскує доповнення до
  кратного stride.
- `model.adapt: full | lora`: LoRA (peft, r 16, α 32) на q/k/v/o_proj 72 лінійних шарів уваги енкодера й
  декодера DETR (589 824 параметри), базові ваги трансформера заморожено.
- `input_projection` — окрема група оптимізатора (`optim.lr_projection`) із власним warmup
  (`optim.warmup_steps_projection`); лише для `vit_detr`, R50-чекпоінти відновлюються як і раніше.
- Тести: `tests/test_vit_detr.py` (14), разом **75 passed, 1 skipped** (`test_trainer.py`, потребує GPU).

## Крок 3: dry runs (GPU)
Конфіги `configs/experiments/b_vit_{random,imagenet,mae,con,joint}.yaml`; рецепт поки базовий
(adapt full, lr 1e-4, lr_backbone 1e-5, warmup 200, батч 8) — його обирає крок 4.

### Пам'ять і швидкість: stride 16 проти 32 (`b_vit_imagenet`, 1 епоха)

| | micro-batch | пік probe (полотно 960×960, 223 бокси) | пік на тренуванні | епоха 0* | val* |
|---|---|---|---|---|---|
| stride 16 | 8 | 8.32 GiB | 8.21 GiB | 128.2 с | 49.0 с |
| stride 32 | 8 | 6.72 GiB | 5.82 GiB | 108.5 с | 44.9 с |
| R50 (фаза 2, для порівняння) | 8 | 6.98 GiB | 6.32 GiB | 93.6 с | — |

\* Епоха 0 і перша валідація включають старт воркерів (Windows spawn). Обидва stride вміщаються з
батчем 8 без акумуляції. Усталену швидкість stride 32 окремо не міряв.

### Overfit-gate: 8 train-зображень, ImageNet init, stride 16
Налаштування як у контрольному overfit R50 фази 2 (`dry_a_overfit8_constlr`): 400 кроків, warmup 10,
постійний LR, валідація на тих самих 8 зображеннях кожні 50 кроків. Критерій (зафіксовано до запуску):
loss на кроці 400 ≤ 50% від стартового.

| крок | loss R50 | loss ViT | AP50 R50 / ViT | AP50 infected R50 / ViT |
|---|---|---|---|---|
| 0 | 25.18 | 32.27 | — | — |
| 100 | 5.29 | 7.15 | 0.347 / 0.153 | 0.000 / 0.000 |
| 200 | 3.91 | 4.92 | 0.679 / 0.536 | 0.489 / 0.343 |
| 399 | 3.03 | 3.47 | 0.886 / 0.774 | 0.810 / 0.588 |

**Gate пройдено** (32.27 → 3.47, −89%). Нова проєкція поверх COCO-трансформера вчиться, але повільніше
за R50: на кожному кроці loss вищий, AP50 на 8 зображеннях 0.774 проти 0.886. 9.4 хв.

### Smoke: 2 епохи для кожного джерела енкодера (stride 16)

| енкодер | епоха 1, с | img/s | пік GiB | loss еп. 0 → 1 | val AP50 (еп. 1) | AP50 uninf / inf | cardinality err |
|---|---|---|---|---|---|---|---|
| random | 92.4 | 11.17 | 8.21 | 14.34 → 10.18 | 0.125 | 0.251 / 0.000 | 194.7 |
| imagenet | 92.1 | 11.20 | 8.21 | 15.41 → 9.74 | 0.114 | 0.229 / 0.000 | 211.1 |
| mae | 91.9 | 11.23 | 8.21 | 16.07 → 9.19 | 0.148 | 0.295 / 0.000 | 172.4 |
| con | 92.2 | 11.19 | 8.21 | 16.77 → 9.83 | 0.120 | 0.240 / 0.000 | 208.8 |
| joint | 92.2 | 11.19 | 8.21 | 15.58 → 9.60 | 0.134 | 0.269 / 0.000 | 199.4 |

Усі п'ять запусків проходять без помилок (≈4.8–5.0 хв кожен). Усталена епоха ViT stride 16 — ≈92 с
проти 58.4 с у R50 (×1.58). Після 2 епох val AP50 0.11–0.15 (R50 у фазі 2 після 2 епох: 0.317), infected
ще 0 — це перевірка пайплайну, а не порівняння енкодерів.

## Крок 4: вибір рецепта (виконано — результати в `reports/phase4_recipe_summary.md`)
Layer-wise lr decay: `optim.layer_decay` (null = вимкнено). Параметр ViT на глибині i (ембединги 0,
`blocks.i` → i + 1, фінальна норма L + 1 = 13) тренується з lr_backbone · d^(13 − i): остання блок-група
×0.75, ембединги ×0.75¹³ ≈ 0.024. Тести: `tests/test_vit_detr.py`, разом **78 passed, 1 skipped**.

Черга `configs/queues/phase4_recipe.txt`: 5 запусків `b_vit_imagenet` по 40 епох; кожен кандидат змінює рівно
один фактор відносно батьківського конфігу (перевірено diff розв'язаних конфігів і фактичні lr груп оптимізатора).

| Запуск | Зміна | Батьківський |
|---|---|---|
| `b_vit_imagenet` | — (full, stride 16, без decay) | — |
| `b_vit_imagenet_s32` | `model.vit.stride: 32` (рецепт full — щоб ефект stride був ізольований) | `b_vit_imagenet` |
| `b_vit_imagenet_llrd` | `optim.layer_decay: 0.75` | `b_vit_imagenet` |
| `b_vit_imagenet_lora` | `model.adapt: lora` (LoRA при lr групи трансформера 1e-4) | `b_vit_imagenet` |
| `b_vit_imagenet_lora_lr5e4` | lr групи трансформера 5e-4 (LoRA-адаптери + голови + queries, 800 007 параметрів); проєкцію закріплено на 1e-4 | `b_vit_imagenet_lora` |

Після кожного — `evaluate.py --split val --bootstrap` (**лише val**; test у кроці 4 не оцінюється).

### Правило вибору (зафіксовано 2026-10-12, до будь-якого результату кроку 4)
1. Метрика: val infected AP50 з 95% bootstrap CI (`evaluate.py --bootstrap`, 200 ресемплів по val-зображеннях).
2. Лідер — найвища точкова оцінка. Нічия — кожен кандидат, чия точкова оцінка ≥ нижньої межі CI лідера
   (правило фази 3).
3. Серед нічийних — за простотою, у фіксованому порядку: full (`b_vit_imagenet`) → s32 → llrd → lora →
   lora_lr5e4. Val F1 **не** використовується (поріг обирається на тому ж val — оптимістична оцінка).

Оцінка: ≈1.15–1.35 год на запуск, ≈5.8–6.8 год разом. Після кроку 4 — перевірка, чи вийшла крива val
AP50 найкращого рецепта на плато до епохи 40; якщо ще явно росте — пропозиція подовжити всі ViT-запуски
кроку 5 однаково (з оцінкою додаткових годин), без самостійної зміни кількості епох.

Підсумок кроку 4: лідер llrd (val infected AP50 0.798 [0.751, 0.854]); full 0.769, s32 0.790, lora_lr5e4 0.797 —
нічия, lora 0.670 — ні → переможець **full** (дефолти `configs/detr/base.yaml`); приріст 30→40 епох +0.013 →
40 епох.

## Крок 5: основне порівняння енкодерів (seed 42)

Черга `configs/queues/phase4_main.txt` (виконано 2026-10-13, ≈66 хв тренування на запуск). Рецепт full, 40 епох,
seed 42; конфіги відрізняються лише `model.vit.init`. `b_vit_imagenet` — запуск кроку 4. Baseline A —
`a_coco_detr_binary` (R50 COCO-DETR, фаза 2). Відтворення (CPU, лише читання артефактів):

```
python scripts/paired_detr_diff.py --runs a_coco_detr_binary b_vit_random b_vit_imagenet b_vit_mae b_vit_con \
    b_vit_joint --pairs b_vit_joint:b_vit_imagenet b_vit_joint:b_vit_con b_vit_joint:b_vit_mae \
    b_vit_joint:b_vit_random b_vit_joint:a_coco_detr_binary b_vit_imagenet:a_coco_detr_binary \
    --out reports/tables/phase4_main
python scripts/compare_detr_runs.py --runs b_vit_random b_vit_imagenet b_vit_mae b_vit_con b_vit_joint \
    --split val --out reports/tables/phase4_main_val.csv --curves reports/tables/phase4_main_curves.csv
```
Таблиці: `reports/tables/phase4_main_runs.csv`, `phase4_main_diff.csv`, `phase4_main.json`, `phase4_main_curves.csv`;
рисунки `reports/figures/shift_b_vit_*.png`; ноутбук `notebooks/04_detr_ssl.ipynb`.

**Обмеження.** Test — 120 зображень (303 infected GT), тому CI широкі. Кожен запуск — **один seed** DETR-тренування.
SSL-претренування кожного енкодера — теж **один запуск**, тож seed-реплікації (крок 5б) охоплюють лише
варіацію тренування DETR, не SSL.

### Val і test: AP50 (CI — `evaluate.py --bootstrap`, 200 ресемплів)

| запуск | best епоха | val AP50 infected | test AP50 infected | val AP50 uninf. | test AP50 uninf. |
|---|---|---|---|---|---|
| baseline A (R50) | 40 | 0.778 [0.728, 0.832] | **0.551** [0.479, 0.621] | 0.944 | 0.970 [0.960, 0.980] |
| b_vit_random | 38 | 0.711 [0.654, 0.769] | 0.327 [0.267, 0.392] | 0.943 | **0.523** [0.488, 0.554] |
| b_vit_imagenet | 36 | 0.769 [0.720, 0.833] | **0.528** [0.469, 0.593] | 0.947 | 0.935 [0.920, 0.949] |
| b_vit_mae | 38 | 0.762 [0.712, 0.825] | 0.336 [0.266, 0.405] | 0.957 | 0.921 [0.907, 0.938] |
| b_vit_con | 38 | 0.756 [0.706, 0.808] | 0.251 [0.199, 0.300] | 0.943 | 0.899 [0.879, 0.919] |
| b_vit_joint | 32 | 0.774 [0.717, 0.829] | 0.329 [0.269, 0.399] | 0.948 | 0.921 [0.904, 0.938] |

### Test P / R / F1 infected при val-порозі (CI — 1000 спільних ресемплів test, seed 42)

| запуск | val-поріг | precision | recall | F1 |
|---|---|---|---|---|
| baseline A | 0.867 | 0.844 [0.787, 0.894] | 0.376 [0.311, 0.447] | 0.521 [0.455, 0.585] |
| b_vit_random | 0.821 | 0.502 [0.448, 0.563] | 0.432 [0.346, 0.527] | 0.465 [0.404, 0.523] |
| b_vit_imagenet | 0.876 | 0.838 [0.754, 0.910] | 0.221 [0.172, 0.280] | 0.350 [0.284, 0.422] |
| b_vit_mae | 0.794 | 0.972 [0.906, 1.000] | 0.116 [0.080, 0.162] | 0.207 [0.147, 0.278] |
| b_vit_con | 0.890 | 0.857 [0.636, 1.000] | 0.020 [0.007, 0.037] | 0.039 [0.013, 0.070] |
| b_vit_joint | 0.810 | 0.821 [0.738, 0.901] | 0.211 [0.163, 0.265] | 0.336 [0.271, 0.403] |

### Діагностика зсуву (`diagnose_shift.py`, infected GT-клітини test, n = 303)

| запуск | медіана infected-score val → test | test: нема infected-детекції | test: uninfected-score вищий |
|---|---|---|---|
| baseline A | 0.968 → 0.716 | 35.0% | 37.0% |
| b_vit_random | 0.950 → 0.540 | 45.2% | 38.6% |
| b_vit_imagenet | 0.973 → 0.235 | 20.8% | 62.4% |
| b_vit_mae | 0.952 → 0.000 | 57.1% | 69.0% |
| b_vit_con | 0.956 → 0.000 | 60.4% | 77.2% |
| b_vit_joint | 0.956 → 0.001 | 46.2% | 65.0% |

«Нема infected-детекції» — жодної infected-детекції з IoU ≥ 0.5 (score 0); «uninfected-score вищий» — на клітині
є uninfected-детекція з вищим score, ніж найкраща infected (ці дві частки перетинаються).

### Парний bootstrap різниць на test (1000 спільних ресемплів 120 зображень, seed 42)

| різниця | test AP50 infected [95% CI] | частка ≤ 0 | test recall при val-порозі [95% CI] | частка ≤ 0 |
|---|---|---|---|---|
| joint − imagenet | **−0.198 [−0.254, −0.142]** | 1.000 | −0.010 [−0.060, 0.035] | 0.700 |
| joint − con | **+0.078 [0.024, 0.134]** | 0.001 | **+0.191 [0.145, 0.240]** | 0.000 |
| joint − mae | −0.007 [−0.071, 0.055] | 0.561 | **+0.096 [0.051, 0.138]** | 0.001 |
| joint − random | +0.003 [−0.077, 0.078] | 0.493 | **−0.221 [−0.306, −0.142]** | 1.000 |
| joint − baseline A | **−0.222 [−0.285, −0.154]** | 1.000 | **−0.165 [−0.229, −0.109]** | 1.000 |
| imagenet − baseline A | −0.024 [−0.074, 0.032] | 0.805 | **−0.155 [−0.208, −0.104]** | 1.000 |

Точкові різниці збігаються з різницями значень у `metrics_test.json`; P/R/F1 на повному test відтворюють
`evaluate.py` точно (assert у скрипті). Precision і F1 для тих самих пар — у `phase4_main_diff.csv`.

### Що значуще, а що ні (лише в межах CI; один seed, 120 test-зображень)

Значущі (95% CI парної різниці не містить 0):
- **joint гірший за imagenet за test AP50 infected:** −0.198 [−0.254, −0.142]. За recall при val-порозі різниці
  не видно: −0.010 [−0.060, 0.035].
- **joint кращий за con:** AP50 +0.078 [0.024, 0.134], recall +0.191 [0.145, 0.240].
- **joint кращий за mae за recall:** +0.096 [0.051, 0.138]; за AP50 різниці не видно (−0.007 [−0.071, 0.055]).
- **joint гірший за random за recall:** −0.221 [−0.306, −0.142] (у random вищий recall ціною precision 0.502);
  за AP50 різниці не видно (+0.003 [−0.077, 0.078]).
- **joint гірший за baseline A:** AP50 −0.222 [−0.285, −0.154], recall −0.165 [−0.229, −0.109].
- **imagenet гірший за baseline A за recall:** −0.155 [−0.208, −0.104]; за AP50 різниці не видно
  (−0.024 [−0.074, 0.032]).

Незначущі: joint − imagenet (recall), joint − mae (AP50), joint − random (AP50), imagenet − baseline A (AP50).

Поправки на множинні порівняння немає (6 пар × 2 метрики = 12). Усі значущі різниці мають частку ≤ 0, що дорівнює
0.000–0.001 або 1.000. Тобто вони лишаються значущими і за Bonferroni-порогу для 12 порівнянь (≤ 0.002 / ≥ 0.998
на хвіст), але роздільна здатність 1000 ресемплів груба. На val CI усіх ViT-варіантів, крім random, перетинаються
(0.756–0.774); random нижчий за точковою оцінкою (0.711 [0.654, 0.769]). Парних різниць на val не рахували.

Зв'язок із фазою 3 (факт, не пояснення): на stain-shifted probe (val-кропи зі штучним зсувом) joint не
відрізнявся від ImageNet. На реальному test-зсуві в DETR joint суттєво гірший за ImageNet за AP50.

### Аномалії
- **b_vit_random на test втрачає uninfected-клас:** test AP50 uninfected 0.523 (val 0.943). При val-порозі
  5113 з 5614 uninfected GT лишаються без uninfected-детекції (confusion IoU 0.5), AR@300 uninfected 0.445.
  На val запуск збігся (0.711 infected), але стартував найповільніше: infected AP50 0.040 після 10 епох
  (в інших 0.55–0.66).
- **b_vit_con: test recall infected 0.020** (6 з 303) при val-порозі; 60.4% infected-клітин без жодної
  infected-детекції, медіана test-score 0.
- **b_vit_mae, b_vit_joint:** медіана test infected-score ≈ 0 (0.000 / 0.001). Тобто більш ніж у половини
  infected-клітин test score infected нульовий або майже нульовий.
- **Best епоха joint — 32** (в інших 36–40). Чекпоінт `best` обирається за val AP@[.5:.95], а val infected AP50 joint
  ще росте: 0.750 на 30-й → 0.787 на 40-й, +0.037 (правило плато кроку 4 стосувалося лише imagenet).
  Оцінений чекпоінт має 0.774. За 30→40 епох: mae +0.021, con +0.016, random +0.004.
- Втрати на останній епосі в усіх ViT-запусках близькі (loss 3.9–4.3, loss_ce 0.088–0.104): збою тренування
  немає; провал — саме на test.

## Діагностика провалу на test (лише оцінка, 2026-10-13)

Нічого не тренувалося і нічого не налаштовувалося на test. Пороги — з val. Відтворення:
```
python scripts/compare_detr_runs.py --runs a_coco_detr_binary b_vit_random b_vit_imagenet b_vit_mae b_vit_con \
    b_vit_joint --split val --out reports/tables/phase4_main_val.csv --shift-out reports/tables/phase4_shift.csv
python scripts/probe.py --ref random --eval-split test        # також --ref imagenet, --run runs/ssl_{mae_only,con_only,joint}
python scripts/compare_probes.py --val-test --tags ref_random_init ref_imagenet ssl_mae_only ssl_con_only ssl_joint \
    --out reports/tables/phase4_probe_val_test.csv
python scripts/paired_probe_diff.py --split test --pairs ssl_joint:ref_imagenet ssl_joint:ssl_con_only \
    ssl_joint:ssl_mae_only ssl_joint:ref_random_init ssl_con_only:ref_imagenet ssl_mae_only:ref_imagenet \
    --out reports/tables/phase4_probe_test_paired_diff
python scripts/evaluate.py runs/<run> --split val --long-side {640,840,1016} --bootstrap   # 3 запуски
python scripts/scale_diagnostic.py --runs a_coco_detr_binary b_vit_imagenet b_vit_joint --long-sides 640 840 1016 \
    --out reports/tables/phase4_scale
```

### 1. Що відбувається з infected GT-клітинами (`reports/tables/phase4_shift.csv`)

`diagnose_shift.py` доповнено **взаємовиключним** розкладом кожної infected GT-клітини при IoU ≥ 0.5. Варіанти:
не локалізована жодним класом / локалізована, але uninfected-score вищий / infected-score найвищий. Сума трьох
часток = 1. Стара частка «нема infected-детекції» включала й клітини, знайдені як uninfected.

| запуск | split | медіана infected-score | ≥ val-порогу | не локалізована | uninfected вищий | infected найвищий |
|---|---|---|---|---|---|---|
| baseline A | val | 0.968 | 0.883 | 0.017 | 0.040 | 0.943 |
| | test | 0.716 | 0.376 | 0.013 | 0.370 | 0.617 |
| b_vit_random | val | 0.950 | 0.822 | 0.017 | 0.081 | 0.903 |
| | test | 0.540 | 0.432 | **0.076** | 0.386 | 0.538 |
| b_vit_imagenet | val | 0.973 | 0.913 | 0.010 | 0.034 | 0.956 |
| | test | 0.235 | 0.221 | 0.017 | 0.624 | 0.360 |
| b_vit_mae | val | 0.952 | 0.873 | 0.023 | 0.040 | 0.936 |
| | test | 0.000 | 0.116 | 0.030 | 0.690 | 0.281 |
| b_vit_con | val | 0.956 | 0.829 | 0.037 | 0.057 | 0.906 |
| | test | 0.000 | 0.020 | 0.033 | 0.772 | 0.195 |
| b_vit_joint | val | 0.956 | 0.883 | 0.034 | 0.034 | 0.933 |
| | test | 0.001 | 0.211 | 0.017 | 0.650 | 0.333 |

Локалізація на test не ламається: не локалізовано 1.3–3.3% infected-клітин, як і на val (1.0–3.7%).
Виняток — random: 7.6% на test проти 1.7% на val. Провал — у **класифікації** вже локалізованих клітин:
uninfected-score вищий у 62–77% infected-клітин у ViT-DETR з будь-яким претренованим енкодером, у baseline A — у 37%.
На val таких клітин 3–8%.

### 2. Probe на test-кропах: чи несуть самі заморожені ознаки сигнал на test

Той самий probe, що у фазі 3: класифікатори (linear, kNN; фіксовані гіперпараметри) навчені на **train**-кропах,
оцінені на **test**-кропах. Test-кропів: 2000 RBC (підвибірка) + 303 infected (ring 169, trophozoite 111,
gametocyte 12, schizont 11); leukocyte у test немає. CI — bootstrap по зображеннях (1000). Val і test — різні
вибірки, тому CI val→test не парні. Таблиця: `reports/tables/phase4_probe_val_test.csv`.

| енкодер | linear AUPRC infected val → test | linear AUROC infected val → test | kNN AUPRC infected val → test |
|---|---|---|---|
| random | 0.924 → **0.513** [0.452, 0.581] | 0.986 → 0.770 [0.735, 0.806] | 0.730 → 0.153 [0.134, 0.184] |
| ImageNet | 0.989 → **0.876** [0.832, 0.914] | 0.997 → 0.968 [0.956, 0.980] | 0.976 → 0.755 [0.704, 0.807] |
| mae_only | 0.979 → 0.835 [0.788, 0.882] | 0.996 → 0.932 [0.903, 0.957] | 0.985 → 0.782 [0.732, 0.836] |
| con_only | 0.986 → 0.853 [0.811, 0.894] | 0.998 → 0.954 [0.935, 0.972] | 0.978 → 0.667 [0.606, 0.732] |
| joint | 0.988 → 0.808 [0.763, 0.851] | 0.996 → 0.948 [0.931, 0.964] | 0.984 → 0.760 [0.706, 0.817] |

Парні різниці на test-кропах (ті самі 1000 ресемплів test-зображень, CPU-рефіт зі збережених ознак; відхилення
від probe json ≤ 3.2e-4 для linear AUPRC, ≤ 2.6e-3 для kNN AUPRC; `phase4_probe_test_paired_diff.csv`):

| різниця | linear AUPRC | linear AUPRC, 10% міток | kNN AUPRC |
|---|---|---|---|
| joint − ImageNet | **−0.068 [−0.107, −0.030]** | **−0.087 [−0.124, −0.047]** | +0.005 [−0.047, 0.055] |
| joint − con_only | **−0.045 [−0.080, −0.006]** | **+0.056 [0.019, 0.091]** | **+0.090 [0.049, 0.133]** |
| joint − mae_only | −0.027 [−0.061, 0.008] | −0.019 [−0.058, 0.019] | −0.023 [−0.059, 0.015] |
| joint − random | **+0.295 [0.237, 0.357]** | **+0.355 [0.295, 0.422]** | **+0.607 [0.546, 0.660]** |
| con_only − ImageNet | −0.023 [−0.057, 0.015] | **−0.143 [−0.187, −0.097]** | **−0.085 [−0.147, −0.020]** |
| mae_only − ImageNet | **−0.040 [−0.076, −0.005]** | **−0.068 [−0.110, −0.024]** | +0.028 [−0.022, 0.076] |

Test macro-F1 низький для всіх енкодерів, зокрема й ImageNet: linear 0.111 (random) – 0.410 (ImageNet). Головна
причина — стадії ring/gametocyte/schizont на test класифікуються погано. Для бінарного питання infected/uninfected
показові AUPRC/AUROC.

### 3. Масштаб: val при зміненому вхідному розмірі (`reports/tables/phase4_scale.csv`)

Eval-трансформ ресайзить кожне зображення до фіксованого long side 800, тому масштабувати сам файл марно.
Масштаб клітин на вході моделі змінено через `evaluate.py --long-side`: 1016 (×1.27), 640 (×0.8) і додатково
840 (×1.05). ×1.05 — **фактичний** test/val-масштаб на вході моделі: медіанний бік бокса після resize до 800 —
59.26 px на test проти 56.5 на val. Сирі ×1.27 майже повністю компенсуються ресайзом, бо test-зображення
більші (1944 проти 1600 px). DETR тренувався з long side 640–960, тобто з масштабами ×0.8–1.2.
Пороги — з val при 800. Різниці парні (1000 спільних ресемплів val-зображень).

| запуск | ×0.8: Δ AP50 inf | ×1.05: Δ AP50 inf | ×1.27: Δ AP50 inf | ×1.27: Δ recall |
|---|---|---|---|---|
| baseline A (×1: 0.778) | **−0.056 [−0.091, −0.015]** | +0.009 [−0.017, 0.038] | −0.002 [−0.029, 0.030] | **−0.081 [−0.120, −0.045]** |
| b_vit_imagenet (×1: 0.769) | −0.002 [−0.033, 0.019] | +0.009 [−0.021, 0.029] | −0.003 [−0.025, 0.017] | −0.024 [−0.054, 0.007] |
| b_vit_joint (×1: 0.774) | **−0.035 [−0.059, −0.013]** | −0.028 [−0.052, 0.003] | −0.011 [−0.032, 0.016] | −0.007 [−0.030, 0.016] |

AP50 infected при ×1.27: baseline A 0.775 [0.728, 0.826], imagenet 0.766 [0.713, 0.827], joint 0.763 [0.708, 0.818].
Найбільший ефект масштабу на val — −0.056. Падіння val→test становить −0.23 (baseline A), −0.24 (imagenet) і
−0.44 (joint).

### Що це показує (у межах CI; один seed DETR, один SSL-запуск на енкодер)

- **Масштаб не пояснює провалу.** Фактична різниця масштабу test/val на вході моделі — ×1.05, і при ×1.05 різниці
  в AP50 не видно в жодної моделі (усі CI містять 0). Навіть ×0.8 і ×1.27 змінюють val AP50 infected щонайбільше на
  0.056, на порядок менше, ніж падіння на test (joint −0.44).
- **Локалізація не є проблемою.** На test 97–99% infected-клітин локалізовані хоч якимось класом (random — 92%).
- **Провал — у класифікації infected/uninfected на test.** Це стосується всіх ViT-DETR, зокрема й ImageNet:
  uninfected-score вищий у 62% клітин проти 37% у baseline A. У SSL-енкодерів таких клітин ще більше (65–77%).
- **Представлення (заморожені ознаки) теж гіршають на test, але сигнал зберігають.** Linear AUROC infected на
  test-кропах для SSL-енкодерів 0.93–0.95, для ImageNet 0.968, для random 0.770. Joint гірший за ImageNet за
  linear AUPRC: −0.068 [−0.107, −0.030]. За kNN AUPRC різниці не видно: +0.005 [−0.047, 0.055].
  Отже, заморожені SSL-ознаки на test помірно слабші за ImageNet, але далекі від колапсу (random — 0.513).
- **Більша частина розриву з'являється після DETR-fine-tuning.** У DETR різниця joint − imagenet за test AP50
  становить −0.198 [−0.254, −0.142]. Між замороженими ознаками за AUPRC на GT-кропах різниця −0.068 [−0.107, −0.030].
  Метрики різні (детекційний AP50 проти AUPRC класифікації кропів), тож прямо віднімати їх не можна. Але
  заморожені ознаки всіх SSL-енкодерів класифікують test-кропи значно краще, ніж DETR на їх основі класифікує
  test-клітини: у DETR медіанний infected-score ≈ 0. Probe енкодерів **після** DETR-fine-tuning ще не робили,
  тому розділити внесок енкодера і голови DETR поки не можна.

## Двоетапний пайплайн «DETR локалізує, кроп-класифікатор класифікує» (лише оцінка, 2026-10-13)

Без перетренування DETR. Варіант скору, пороги та все інше обрано на val; test використано лише для фінальних чисел.
```
python scripts/two_stage.py candidates --run runs/a_coco_detr_binary        # і runs/b_vit_imagenet (GPU, ~1 хв)
python scripts/two_stage.py features --runs a_coco_detr_binary b_vit_imagenet \
    --encoders ref:imagenet run:runs/ssl_con_only run:runs/ssl_mae_only run:runs/ssl_joint   # GPU, 8 хв
python scripts/two_stage.py evaluate --runs a_coco_detr_binary b_vit_imagenet \
    --tags ref_imagenet ssl_con_only ssl_mae_only ssl_joint --encoder-runs ref_imagenet:b_vit_imagenet \
    ssl_con_only:b_vit_con ssl_mae_only:b_vit_mae ssl_joint:b_vit_joint --out reports/tables/phase4_two_stage   # CPU
```
- **Кандидати:** усі 300 queries DETR зі скором objectness = 1 − p(no-object), class-agnostic NMS з IoU 0.5
  (фіксовано заздалегідь). Після NMS лишається 113–131 кандидат на зображення.
- **Ознаки:** точно як у probe — вікно 224 при long side 800, середнє patch-токенів усередині бокса. Чисті
  train-ознаки побітово збігаються з ознаками probe фази 3.
- **Класифікатор:** лінійний (гіперпараметри probe, class-balanced), бінарний infected / uninfected, навчений на
  train GT-кропах. Дві версії: `clean`, а також `stain` — clean разом зі stain-shifted копією кожного кропа
  (HED / jitter / blur / JPEG, як у stain-shifted probe, один фіксований зсув на train-зображення).
- **Скор:** objectness × p(infected) або лише p(infected); варіант обрано за val infected AP50. Для всіх 16
  конфігурацій обрано objectness × p: на val він дав 0.735–0.800 проти 0.549–0.717 для p.
- **Поріг:** max F1 на val, на test застосовано без змін.
- **CI:** bootstrap по зображеннях, 1000 спільних ресемплів.

### Результати (infected; `reports/tables/phase4_two_stage.csv`)

| бокси | класифікатор | val AP50 | test AP50 [95% CI] | test P | test R | test F1 |
|---|---|---|---|---|---|---|
| baseline A | **end-to-end DETR** | 0.778 | **0.551** [0.486, 0.621] | 0.844 | 0.376 | 0.520 |
| baseline A | ImageNet clean | 0.764 | 0.599 [0.533, 0.675] | 0.757 | 0.422 | 0.542 |
| baseline A | ImageNet stain | 0.768 | 0.598 [0.528, 0.676] | 0.777 | 0.356 | 0.489 |
| baseline A | con clean | 0.798 | 0.529 [0.461, 0.598] | 0.660 | 0.353 | 0.460 |
| baseline A | con stain | 0.799 | 0.540 [0.475, 0.611] | 0.686 | 0.346 | 0.460 |
| baseline A | mae clean | 0.790 | 0.581 [0.513, 0.659] | 0.780 | **0.129** | 0.221 |
| baseline A | mae stain | 0.788 | 0.589 [0.530, 0.662] | 0.734 | 0.346 | 0.471 |
| baseline A | joint clean | 0.800 | 0.576 [0.510, 0.650] | 0.766 | 0.346 | 0.477 |
| baseline A | joint stain | 0.797 | 0.593 [0.526, 0.665] | 0.708 | 0.449 | 0.550 |
| b_vit_imagenet | **end-to-end DETR** | 0.769 | **0.528** [0.469, 0.596] | 0.838 | 0.221 | 0.350 |
| b_vit_imagenet | ImageNet clean | 0.735 | 0.562 [0.506, 0.635] | 0.676 | 0.323 | 0.438 |
| b_vit_imagenet | ImageNet stain | 0.749 | 0.576 [0.507, 0.657] | 0.746 | 0.290 | 0.418 |
| b_vit_imagenet | con clean | 0.763 | 0.524 [0.463, 0.598] | 0.646 | 0.271 | 0.381 |
| b_vit_imagenet | con stain | 0.770 | 0.535 [0.477, 0.614] | 0.667 | 0.271 | 0.385 |
| b_vit_imagenet | mae clean | 0.753 | 0.593 [0.522, 0.669] | 0.818 | **0.089** | 0.161 |
| b_vit_imagenet | mae stain | 0.757 | 0.606 [0.540, 0.679] | 0.712 | 0.294 | 0.416 |
| b_vit_imagenet | joint clean | 0.764 | 0.582 [0.518, 0.659] | 0.765 | 0.290 | 0.421 |
| b_vit_imagenet | joint stain | 0.768 | 0.593 [0.530, 0.667] | 0.733 | 0.363 | 0.486 |
| (b_vit_joint) | end-to-end DETR | 0.774 | 0.329 [0.276, 0.397] | 0.820 | 0.211 | 0.336 |

CI для P/R/F1 наведено в `phase4_two_stage.csv`.

### Парні різниці на test (`reports/tables/phase4_two_stage_diff.csv`)

**(a) Двоетапний мінус end-to-end DETR, з якого взято бокси.** Значущих різниць за AP50 немає в жодній із 16
конфігурацій. Найближчі до межі:
- бокси b_vit_imagenet + mae stain: +0.078 [−0.000, 0.149];
- бокси b_vit_imagenet + joint stain: +0.066 [−0.001, 0.135];
- бокси baseline A + ImageNet clean: +0.048 [−0.007, 0.116].

За recall при val-порозі:
- значуще вищий recall на боксах b_vit_imagenet у ImageNet (clean +0.102 [0.050, 0.159], stain +0.069
  [0.014, 0.120]), joint (clean +0.069 [0.009, 0.133], stain +0.142 [0.077, 0.210]) і mae stain
  (+0.073 [0.003, 0.142]);
- значуще нижчий recall у mae clean: −0.248 [−0.317, −0.189] на боксах A і −0.132 [−0.191, −0.080] на боксах
  b_vit_imagenet. Це провал val-порогу на test, а не AP: stain-версія mae має recall 0.346 / 0.294.

**(b) Joint мінус ImageNet у двоетапному пайплайні.**
- За AP50 різниці не видно ніде: A clean −0.023 [−0.074, 0.024], A stain −0.005 [−0.057, 0.044],
  IN clean +0.020 [−0.030, 0.068], IN stain +0.017 [−0.038, 0.077].
- За recall: зі `stain`-класифікатором joint вищий (A +0.092 [0.027, 0.157], IN +0.073 [0.020, 0.129]);
  з `clean` на боксах A — нижчий (−0.076 [−0.145, −0.016]).

**(c) Двоетапний з енкодером X мінус end-to-end ViT-DETR, натренований з тим самим X.** Це додаткове порівняння;
бокси тут інші — з baseline A або b_vit_imagenet.
- joint: +0.246…+0.264 за AP50 (усі CI ≥ +0.179);
- con: +0.273…+0.289;
- mae: +0.245…+0.270;
- ImageNet: +0.071 [0.003, 0.140] і +0.070 [0.004, 0.133] на боксах A, на власних боксах b_vit_imagenet —
  +0.035 і +0.049 (CI містять 0).

Порівнянь багато (2 бокси × 2 класифікатори × 9 пар × 2 метрики), поправки немає. Різниці в (c) для SSL-енкодерів
мають частку ≤ 0 рівну 0.000, тож вони не на межі.

## Енкодери після DETR fine-tuning (probe `--detr-run`)

ViT витягнуто з `best.pt` b_vit_imagenet і b_vit_joint з тією нормалізацією, з якою його дотреновували
(перевірено: для imagenet — 0.5 / 0.5). Probe — як у фазі 3: train → val / test кропи.
Таблиці: `reports/tables/phase4_probe_finetuned.csv`, `phase4_probe_finetuned_{val,test}_diff.csv`.

| енкодер | val linear AUPRC | val stain-shifted AUPRC | test linear AUPRC [95% CI] | test kNN AUPRC |
|---|---|---|---|---|
| ImageNet (заморожений) | 0.989 | 0.985 | 0.876 [0.832, 0.914] | 0.755 |
| ImageNet після DETR | 0.995 | 0.988 | 0.897 [0.864, 0.928] | 0.861 |
| joint (заморожений) | 0.988 | 0.984 | 0.808 [0.763, 0.852] | 0.760 |
| joint після DETR | 0.991 | 0.970 | 0.786 [0.732, 0.841] | 0.705 |

Парні різниці (той самий енкодер після мінус до fine-tuning):
- **ImageNet:** val linear AUPRC +0.006 [0.003, 0.011]; test linear +0.021 [−0.004, 0.049] (не значуще),
  test kNN **+0.112 [0.072, 0.148]**.
- **joint:** val +0.003 [0.001, 0.006]; val stain-shifted **−0.014 [−0.023, −0.004]**; test linear −0.022
  [−0.058, 0.014] (не значуще); test 10% міток **−0.131 [−0.185, −0.071]**; test kNN **−0.055 [−0.097, −0.013]**.
- **Після fine-tuning, joint − ImageNet на test:** linear **−0.111 [−0.146, −0.075]**, kNN **−0.162
  [−0.218, −0.097]**. До fine-tuning було −0.068 [−0.107, −0.030] і +0.005 [−0.047, 0.055].

Застереження: у DETR ViT обробляє все зображення (глобальна увага по ~1900 токенах), у probe — вікно 224.
Тож це ознаки того самого енкодера, але не ідентичні тим, які бачить DETR-декодер.

### Що працює, а що ні (у межах CI; один seed DETR, один SSL-запуск)

- **Двоетапний пайплайн усуває провал SSL-енкодерів на test.** Із замороженими joint / mae / con ознаками і
  боксами baseline A або b_vit_imagenet test AP50 становить 0.52–0.61. End-to-end ViT-DETR з тими самими
  енкодерами дає 0.25–0.34: це +0.25…+0.29 [CI ≥ +0.16]. У двоетапному пайплайні joint і ImageNet за AP50
  не розрізняються (−0.023…+0.020, усі CI містять 0).
- **Двоетапний пайплайн не дає значущого виграшу за AP50 над end-to-end DETR, з якого взято бокси.**
  Точкові оцінки здебільшого вищі (до +0.078), але CI містять 0. Найкраща точкова оцінка — 0.606
  (бокси b_vit_imagenet + mae stain) проти 0.551 у baseline A.
- **Stain-augmented класифікатор переважно підвищує recall при val-порозі**, тобто стабілізує поріг, а не AP:
  mae 0.129 → 0.346, joint 0.346 → 0.449 на боксах A. AP при цьому змінюється на ≤ 0.017.
- **DETR fine-tuning не руйнує ознак**, але для joint погіршує їх перенесення на test. Після fine-tuning joint
  гірший на stain-shifted val, при 10% міток і за kNN на test. ImageNet-енкодер після fine-tuning на test,
  навпаки, кращий за kNN. Проте навіть дотренований joint-енкодер дає на test-кропах linear AUPRC 0.786.
- **Де втрачається якість.** Не в локалізації: див. попередній розділ, бокси в усіх моделях хороші. І не в
  самій SSL-репрезентації: заморожені SSL-ознаки на тих самих боксах класифікують test на рівні ImageNet. Втрата
  відбувається в end-to-end DETR, тобто в класифікації через декодер / class head і, частково, в зміні енкодера
  під час fine-tuning (−0.014…−0.131 за метриками вище). Розділити ці два внески точніше (наприклад, DETR-бокси +
  дотренований енкодер у двоетапному пайплайні) цей експеримент не дозволяє.

## Експеримент «fine-tuning шкодить енкодеру» (EXPLORATORY; виконано 2026-10-13)

**Гіпотезу сформульовано ПІСЛЯ перегляду test-результатів** (phase4_main і діагностики вище), тому це
дослідницький, а не підтверджувальний експеримент. Жоден варіант не обирається за test; звітуються всі чотири.

Варіанти оголошено до запуску; 2026-10-13, до будь-якого результату. Конфіги відрізняються від `b_vit_<enc>`
лише вказаним ключем — це перевіряє `tests/test_vit_detr.py`. Решта — рецепт кроку 5: seed 42, 40 епох.

| запуск | зміна |
|---|---|
| `b_vit_joint_frozen`, `b_vit_imagenet_frozen` | `model.vit.freeze: true`: ViT без градієнтів і в eval-режимі; input projection, трансформер, queries і голови тренуються |
| `b_vit_joint_lowlr`, `b_vit_imagenet_lowlr` | повний fine-tuning з `optim.lr_backbone` 1e-5 → 1e-7 (у 100 разів менше) |

Код: `ViTBackbone.freeze()`. `train()` лишає ViT в eval-режимі, тож dropout і drop-path неактивні; група оптимізатора
`backbone` порожня. CPU-тести перевіряють, що градієнтів у ViT немає, ваги не змінюються після кроку оптимізатора і
заморожування переживає LoRA. GPU-smoke `runs/dry_c_joint_frozen` (2 епохи) показав:
- ViT після тренування побітово дорівнює `encoder_final.pt` (150 тензорів);
- епоха триває 56 с проти 93 с у full;
- пік пам'яті 3.9 GiB.

Черга `configs/queues/phase4_freeze.txt` (≈3.7 год): train → val → test → `diagnose_shift`.
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase4_freeze.txt
```
**План аналізу (зафіксовано до запуску):**
- Таблиця поруч із baseline A, b_vit_imagenet і b_vit_joint. Колонки: val / test infected AP50 [95% CI],
  recall при val-порозі, частки `diagnose_shift` (не локалізована / uninfected вищий / infected найвищий,
  медіанний score).
- Парний bootstrap на test (1000 спільних ресемплів) кожного варіанта проти його full-fine-tune відповідника:
  `_frozen − b_vit_<enc>` і `_lowlr − b_vit_<enc>`. Метрики: test AP50 infected і recall при val-порозі.
- Без вибору переможця, без поправки на множинність (4 пари × 2 метрики) — про це буде сказано явно.

### Результати (черга виконана 2026-10-13, усі 16 кроків `QUEUE DONE`)

```
python scripts/paired_detr_diff.py --runs a_coco_detr_binary b_vit_imagenet b_vit_joint b_vit_imagenet_frozen \
    b_vit_joint_frozen b_vit_imagenet_lowlr b_vit_joint_lowlr \
    --pairs b_vit_joint_frozen:b_vit_joint b_vit_imagenet_frozen:b_vit_imagenet b_vit_joint_lowlr:b_vit_joint \
    b_vit_imagenet_lowlr:b_vit_imagenet b_vit_joint_frozen:b_vit_imagenet_frozen b_vit_joint_lowlr:b_vit_imagenet_lowlr \
    --out reports/tables/phase4_freeze
```
Файли: `reports/tables/phase4_freeze_runs.csv` (таблиця), `phase4_freeze_diff.csv` (парні різниці), `phase4_freeze.json`.
`paired_detr_diff.py` тепер також пише val recall при val-порозі (лише точкове значення) і взаємовиключні частки
`diagnose_shift` для val і test (`*_excl_*`).

**Застереження.** Аналіз EXPLORATORY: гіпотезу сформульовано після перегляду test-результатів. **Поправки на
множинність немає.** Порівнянь 6 пар × 4 метрики = 24 різниці, з них заздалегідь заплановано 4 пари × 2 метрики, а
дві пари «joint − imagenet» додано на запит. Жоден варіант не обрано за test, і конфіги не змінювалися. Seed один (42),
у test 120 зображень і 303 infected GT-клітини.

Перевірка заморожування: у `best.pt` і `last.pt` обох frozen-запусків усі 150 тензорів ViT побітово дорівнюють
початковим вагам (`encoder_final.pt` для joint; timm `augreg_in21k_ft_in1k` для ImageNet).

**Infected: AP50 [95% CI] і recall при val-порозі.** AP50 CI взято з `evaluate.py --bootstrap` (200 ресемплів). Test
recall CI — 1000 спільних ресемплів test. Best epoch — за val.

| запуск | best ep | val AP50 | test AP50 | val-поріг | val recall | test recall |
|---|---|---|---|---|---|---|
| a_coco_detr_binary | 40 | 0.778 [0.728, 0.832] | 0.551 [0.479, 0.621] | 0.867 | 0.883 | 0.376 [0.311, 0.447] |
| b_vit_imagenet | 36 | 0.769 [0.720, 0.833] | 0.528 [0.469, 0.593] | 0.876 | 0.913 | 0.221 [0.172, 0.280] |
| b_vit_joint | 32 | 0.774 [0.717, 0.829] | 0.329 [0.269, 0.399] | 0.810 | 0.883 | 0.211 [0.163, 0.265] |
| b_vit_imagenet_frozen | 40 | 0.754 [0.703, 0.816] | 0.306 [0.249, 0.379] | 0.790 | 0.886 | 0.106 [0.070, 0.147] |
| b_vit_joint_frozen | 40 | 0.757 [0.709, 0.814] | 0.276 [0.221, 0.334] | 0.845 | 0.862 | 0.135 [0.094, 0.185] |
| b_vit_imagenet_lowlr | 40 | 0.786 [0.740, 0.848] | 0.465 [0.400, 0.534] | 0.864 | 0.896 | 0.158 [0.119, 0.206] |
| b_vit_joint_lowlr | 36 | 0.760 [0.706, 0.816] | 0.248 [0.185, 0.311] | 0.829 | 0.862 | 0.099 [0.060, 0.147] |

**`diagnose_shift`: взаємовиключні частки infected GT-клітин** (val n = 298, test n = 303). Колонки: не
локалізована жодним класом / uninfected вищий / infected найвищий. Медіанний score — infected-score клітини.

| запуск | val: не лок. / uninf. вищий / inf. найвищий | test: не лок. / uninf. вищий / inf. найвищий | медіанний score val → test |
|---|---|---|---|
| a_coco_detr_binary | 0.017 / 0.040 / 0.943 | 0.013 / 0.370 / 0.617 | 0.968 → 0.716 |
| b_vit_imagenet | 0.010 / 0.034 / 0.956 | 0.017 / 0.624 / 0.360 | 0.973 → 0.235 |
| b_vit_joint | 0.034 / 0.034 / 0.933 | 0.017 / 0.650 / 0.333 | 0.956 → 0.001 |
| b_vit_imagenet_frozen | 0.017 / 0.064 / 0.919 | 0.069 / 0.663 / 0.267 | 0.953 → 0.000 |
| b_vit_joint_frozen | 0.030 / 0.047 / 0.923 | 0.030 / 0.729 / 0.241 | 0.962 → 0.000 |
| b_vit_imagenet_lowlr | 0.007 / 0.044 / 0.950 | 0.033 / 0.584 / 0.383 | 0.964 → 0.063 |
| b_vit_joint_lowlr | 0.024 / 0.040 / 0.936 | 0.023 / 0.766 / 0.211 | 0.961 → 0.000 |

**Парний bootstrap на test** (1000 спільних ресемплів 120 зображень, seed 42; `share_le_0` — частка ресемплів з
різницею ≤ 0).

| a − b | Δ test AP50 [95% CI] | Δ test recall [95% CI] |
|---|---|---|
| joint_frozen − joint | −0.053 [−0.100, −0.012] | −0.076 [−0.124, −0.034] |
| imagenet_frozen − imagenet | −0.222 [−0.269, −0.169] | −0.116 [−0.170, −0.067] |
| joint_lowlr − joint | −0.081 [−0.138, −0.034] | −0.112 [−0.161, −0.067] |
| imagenet_lowlr − imagenet | −0.063 [−0.113, −0.009] | −0.063 [−0.115, −0.010] |
| joint_frozen − imagenet_frozen | −0.030 [−0.098, 0.028] | +0.030 [−0.022, 0.079] |
| joint_lowlr − imagenet_lowlr | −0.217 [−0.291, −0.140] | −0.059 [−0.114, −0.007] |

Precision і F1 тих самих пар є в `phase4_freeze_diff.csv`.

### Що змінилося порівняно з full fine-tuning (лише числа, у межах CI, без поправки на множинність)
- **Val.** Infected AP50 усіх чотирьох варіантів лежить у 0.754–0.786, і їхні CI перекриваються з CI відповідних
  full-запусків (0.769 / 0.774). Val recall — 0.862–0.896.
- **Test AP50.** Він нижчий за full-відповідника в усіх чотирьох варіантах; жоден CI різниці не містить 0. Різниці:
  joint_frozen −0.053, imagenet_frozen −0.222, joint_lowlr −0.081, imagenet_lowlr −0.063.
- **Test recall при val-порозі.** Він нижчий у всіх чотирьох варіантах; жоден CI не містить 0. Діапазон від −0.063 до
  −0.116.
- **Joint − ImageNet.** Для frozen CI різниці містять 0 і за AP50, і за recall. Для lowlr AP50 становить −0.217
  [−0.291, −0.140], recall −0.059 [−0.114, −0.007].
- **`diagnose_shift` на test.** Частка «uninfected вищий» у варіантів становить 0.584–0.766, у full-запусків — 0.624 / 0.650.
  Частка «не локалізована» — 0.023–0.069 (у full-запусків 0.017). Медіанний infected-score на test — 0.000–0.063.
- **Best epoch.** У трьох варіантах best epoch = 40, тобто останній, а в joint_lowlr — 36.

## Експеримент «аугментація проти зсуву» (EXPLORATORY; підготовлено, не запущено)

**Гіпотезу сформульовано ПІСЛЯ перегляду test-результатів.** Вона звучить так: декодер і класова голова DETR
перенавчаються на вигляд train-домену, тому сильна аугментація фарбування і масштабу під час тренування DETR має
допомогти при зсуві домену. Обмеження:
- seed один (42);
- **поправки на множинність немає**;
- **силу аугментації зафіксовано в конфігах до запуску й не підбирали ні на test, ні деінде**;
- жоден варіант не обирається за test.

**Синтетичний stain-shifted val** (`evaluate.py --split val --stain-shift`, вихід `metrics_val_stain.json`):
- Кожне val-зображення після eval-resize до 800 отримує фіксований зсув stain-shifted probe. Параметри:
  `configs/ssl/probe.yaml` → `probe.stain_shift`; seed 2024 + image id; HED 0.08, jitter 0.3/0.3/0.3/0.05, blur 0.5–1.0,
  JPEG 30–60.
- Пороги беруться з чистого val.
- Для test цей режим заборонено.
- Копія не зберігається на диск: зсув детермінований для кожного зображення, і CPU-тест це перевіряє.

**Варіанти оголошено до запуску** (2026-10-13). Вони відрізняються від `b_vit_<enc>` (повний fine-tuning, seed 42,
40 епох) лише такими ключами; це перевіряє `tests/test_vit_detr.py`:

| ключ | b_vit_<enc> | b_vit_<enc>_aug |
|---|---|---|
| `augment.stain_aug.p` | 0 (вимкнено) | 0.5: стратегія і сила ті самі, що в stain-shifted probe, параметри перевибираються для кожного семпла |
| `augment.train_long_side` | [640, 960] (×0.8–1.2 від 800) | [576, 1056] (×0.72–1.32), ширший scale jitter |
| `train.select_on` + `eval.val_stain` | best.pt за чистим val AP | best.pt за AP на синтетичному stain-shifted val |

Scale jitter у пайплайні вже був: це випадкова довша сторона `train_long_side`. Тут ширшає лише його діапазон.
Stain-аугментація додається в кінець train-пайплайна, і тільки коли `p > 0`. Тому випадковий потік старих
конфігів не змінюється; це перевіряє CPU-тест.

**Застереження до протоколу:**
1. Синтетичний val побудовано з того самого сімейства перетворень, яким аугментується train. Тому виграш на ньому
   частково закладено конструкцією.
2. Best epoch для `_aug` обирається за stain-shifted val, а для `b_vit_<enc>` — за чистим val. Протоколи вибору
   різні, як і вимагалося: для `_aug` вибір лише за synthetic-shifted val.

GPU-smoke `runs/dry_c_joint_aug` (2 епохи):
- micro-batch 8 (probe 9.76 GiB);
- пік 11.1 GiB;
- епоха 125 с, перша 161 с; для порівняння, у `b_vit_joint` в середньому 93 с;
- val і stain-shifted val приблизно по 50 с на 2-й епосі;
- best.pt обрано за val_stain.

Черга `configs/queues/phase4_aug.txt` (≈3.1 год): train → val → val `--stain-shift` → test → `diagnose_shift`.
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase4_aug.txt
```

### Уже натреновані запуски на синтетичному stain-shifted val (до аугментації)

Таблиця `reports/tables/phase4_aug_runs.csv` (`paired_detr_diff.py`). Рядки `_aug` буде додано після черги. CI для
AP50 — 200 ресемплів `evaluate.py`; best.pt усіх трьох обрано за чистим val.

| запуск | val AP50 inf. | stain-val AP50 inf. [95% CI] | val recall | stain-val recall | test AP50 inf. | test recall |
|---|---|---|---|---|---|---|
| a_coco_detr_binary | 0.778 | 0.682 [0.626, 0.746] | 0.883 | 0.728 | 0.551 | 0.376 |
| b_vit_imagenet | 0.769 | 0.739 [0.681, 0.808] | 0.913 | 0.812 | 0.528 | 0.221 |
| b_vit_joint | 0.774 | 0.722 [0.666, 0.784] | 0.883 | 0.789 | 0.329 | 0.211 |

**План аналізу (зафіксовано до запуску):**
- Ця сама таблиця з рядками `b_vit_{joint,imagenet}_aug`; до неї додаються частки `diagnose_shift`.
- Парний bootstrap на test (1000 спільних ресемплів): `_aug − b_vit_<enc>` і `joint_aug − imagenet_aug`.
  Метрики: AP50 і recall при val-порозі.
- Без вибору переможця і без поправки на множинність.

## Крок 5б: seed-реплікації (підготовлено, не запущено)
`configs/queues/phase4_seeds.txt`: `b_vit_{joint,imagenet}_seed{43,44}`. Вони відрізняються від батьківських конфігів
лише `seed`: порядок і аугментації батчів, ініціалізація нових голів і проєкції, а також bootstrap-ресемпли
`evaluate.py`. Енкодер той самий (один SSL-запуск). Кожен запуск: train → val → test → `diagnose_shift`; ≈4.6 год.
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase4_seeds.txt
```
