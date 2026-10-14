# Фаза 6: ViT-DETR з ініціалізацією DINO — план (EXPLORATORY; зафіксовано 2026-10-14 до запуску)

## Статус і гіпотеза

**Експеримент EXPLORATORY: гіпотезу сформульовано ПІСЛЯ перегляду test-результатів фази 5.**

Факти, з яких виходить гіпотеза:
- У фазі 5 заморожений DINO ViT-S/16 (timm `vit_small_patch16_224.dino`) дав найкращі crop-ознаки на test:
  linear AUPRC infected 0.911, kNN 0.925 (`reports/phase5_summary.md`).
- У двоетапному пайплайні на боксах baseline A DINO дав найвищий test infected AP50: 0.690 (clean); наскрізний
  baseline A — 0.551.
- У наскрізному ViT-DETR фази 4 (full fine-tuning, seed 42) test infected AP50: ImageNet 0.528, joint 0.329,
  baseline A 0.551.

Питання: чи допомагає DINO-ініціалізація всередині наскрізного ViT-DETR?

Обмеження, які діють для всієї фази:
- один seed (42) на етапі 1;
- поправки на множинність немає;
- усі гіперпараметри, пари порівнянь і критерій переходу на етап 2 зафіксовано тут і в конфігах до запуску;
- на test нічого не підбирається: чекпойнт (`best.pt`) обирається за чистим val infected AP50, однаково для всіх
  запусків; пороги — max F1 на чистому val, на test застосовуються без змін;
- test — 120 зображень, CI широкі;
- усе, що буде додано після перегляду test фази 6, позначається як **post hoc**.

## Етап 1 (≈2.2 год GPU; запускається лише після підтвердження користувача)

### Нові запуски

| запуск | конфіг | відрізняється від | режим |
|---|---|---|---|
| `b_vit_dino_full` | `configs/experiments/b_vit_dino_full.yaml` | `b_vit_imagenet` лише `model.vit.init`: imagenet → dino | основний рецепт фази 4: full fine-tuning |
| `b_vit_dino_frozen` | `configs/experiments/b_vit_dino_frozen.yaml` | `b_vit_imagenet_frozen` лише `model.vit.init`: imagenet → dino | exploratory: `model.vit.freeze: true` |

Обидва — seed 42, 40 епох, решта рецепту як у фазі 4 (stride 16, `adapt: full`, lr 1e-4, lr_backbone 1e-5,
без layer decay). Те, що конфіги відрізняються лише `model.vit.init`, перевіряє
`tests/test_vit_detr.py::test_phase6_dino_configs_differ_only_in_init`.

**Ініціалізація DINO** (`src/malaria_ssl/detr/build.py`):
- гілки `init: imagenet | dino` спільні: timm-ім'я береться з `model.vit.<init>`
  (`model.vit.dino: vit_small_patch16_224.dino` у `configs/detr/base.yaml`, дорівнює `probe.references.dino`);
- mean / std беруться з `timm.models.get_pretrained_cfg`; для DINO це 0.485 / 0.456 / 0.406 і 0.229 / 0.224 / 0.225;
- `scripts/train_detr.py` записує фактичну нормалізацію в `model.vit.input_norm` у `runs/<run>/config.yaml` (для
  кожного vit_detr-запуску); при перебудові моделі (evaluate, two_stage, `probe.py --detr-run`) записане значення
  звіряється з виведеним, розбіжність — помилка;
- тести (`tests/test_vit_detr.py`): ваги ViT у DETR побітово збігаються з `load_encoder(ref="dino")` probe;
  ефективні mean / std у `ViTBackbone` збігаються з `input_mean / input_std` probe.

### Запуски для порівняння (seed 42, НЕ перетреновуються)

| позначення в задачі | запуск |
|---|---|
| imagenet_full | `b_vit_imagenet` |
| baseline A | `a_coco_detr_binary` |
| joint_full (довідка) | `b_vit_joint` |
| imagenet_frozen | `b_vit_imagenet_frozen` |

До них лише додаються відсутні вхідні файли оцінки (без тренування):
- stain-shifted val для `b_vit_imagenet_frozen`;
- two-stage кандидати для `b_vit_imagenet_frozen` і `b_vit_joint`.

### Метрики (клас infected)

Для кожного запуску:
- COCOeval AP50 на чистому val, stain-shifted val (синтетичний зсув лише на val, як у фазі 4) і test;
  CI per-run — з `evaluate.py --bootstrap` (200 ресемплів зображень, як у фазі 4);
- precision / recall / F1 на test при порозі, обраному на чистому val (max F1);
- частки `diagnose_shift` для infected GT (не локалізована / uninfected вищий / infected найвищий), val і test.

### Парні порівняння (test, 1000 спільних ресемплів зображень, `scripts/paired_detr_diff.py`, seed 42)

Метрики: infected AP50, precision, recall, F1 (кожен запуск — при своєму val-порозі).

| # | a − b | роль |
|---|---|---|
| 1 | `b_vit_dino_full` − `b_vit_imagenet` | основна |
| 2 | `b_vit_dino_full` − `a_coco_detr_binary` | основна |
| 3 | `b_vit_dino_full` − `b_vit_joint` | довідка |
| 4 | `b_vit_dino_frozen` − `b_vit_imagenet_frozen` | exploratory |

Вихід: `reports/tables/phase6_dino_detr_{runs,diff}.csv`, `phase6_dino_detr.json`.

### Двоетапний пайплайн з ознаками DINO (`scripts/two_stage.py`, лише оцінка)

Рецепт той самий, що у фазах 4–5:
- кандидати: усі 300 queries DETR, objectness = 1 − p(no-object), class-agnostic NMS 0.5;
- класифікатор: лінійний (гіперпараметри probe) на заморожених ознаках `ref_dino`, навчений на train GT-кропах,
  `clean` і `stain`;
- варіант скору (objectness × p або p) обирається за val infected AP50, поріг — max F1 на val.

Бокси: `b_vit_dino_full`, `b_vit_dino_frozen` і, для таблиці поруч, `a_coco_detr_binary` (перерахунок рядка
фази 5: DINO на боксах A, ті самі ознаки і ресемпли, очікується 0.690).

Пари (test, 1000 спільних ресемплів, AP50 / P / R / F1):
- `ref_dino` two-stage − end-to-end **на тих самих боксах**, для кожного з трьох наборів боксів, clean і stain;
- `ref_dino` two-stage (на кожних боксах) − `b_vit_dino_full` end-to-end (`--encoder-runs ref_dino:b_vit_dino_full`).
  Для боксів `b_vit_dino_full` ця пара збігається з попередньою.

Вихід: `reports/tables/phase6_two_stage_dino{,_diff}.csv`, `phase6_two_stage_dino.json`.

### Діагностика «бокси чи класифікація» (`scripts/box_vs_cls.py`, CPU)

Визначення (`src/malaria_ssl/eval/box_vs_cls.py`, тести `tests/test_box_vs_cls.py`):
- усі GT-клітини об'єднуються в один клас; детекції — two-stage кандидати запуску (objectness, NMS 0.5);
- один COCOeval при IoU 0.5 (жадібне зіставлення за objectness, одна детекція на GT, maxDet 300, area «all»);
- `agnostic_AP50` — class-agnostic AP50 за всіма клітинами;
- `agnostic_recall_all` / `agnostic_recall_infected` — частка всіх / infected GT, зіставлених із class-agnostic
  кандидатом (без порогу скору);
- `infected_AP50` — наскрізний infected AP50 того самого запуску (`predictions_<split>.json`);
- val і test; bootstrap по зображеннях, 1000 ресемплів, спільних для всіх запусків у межах split-а.

Запуски: `a_coco_detr_binary`, `b_vit_imagenet`, `b_vit_joint`, `b_vit_imagenet_frozen`, `b_vit_dino_full`,
`b_vit_dino_frozen`. Вихід: `reports/tables/phase6_box_vs_cls.{csv,json}`. Парних різниць у цій таблиці немає.

### Черга і час

`configs/queues/phase6_dino_detr.txt` (19 кроків):
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase6_dino_detr.txt
```
Для кожного нового запуску: `train_detr` → `evaluate val` → `evaluate val --stain-shift` → `evaluate test` →
`diagnose_shift` → `two_stage candidates`. Далі — відсутні входи старих запусків і CPU-таблиці.

GPU-smoke (2026-10-14, 2 епохи, `dry_d_dino_full` і `dry_d_dino_frozen`; вихідні файли видалено після перевірки):

| smoke | нормалізація в лозі | loss епоха 0 → 1 | с / епоху (епоха 1) | пік GiB | для порівняння (smoke фази 4) |
|---|---|---|---|---|---|
| full | mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | 14.46 → 8.92 | 93.1 | 8.21 | `dry_c_smoke_imagenet`: 92.1 с, 8.20 GiB |
| frozen | те саме; тренованих параметрів ViT 0 | 15.36 → 9.93 | 56.5 | 3.89 | `dry_c_joint_frozen`: 56.4 с, 3.89 GiB |

Для frozen-smoke `evaluate.py --split val` перебудував модель із записаним `input_norm` без помилок.

| частина | хв |
|---|---|
| `b_vit_dino_full`: тренування (як `b_vit_imagenet`: 66–67 хв) + 3 evaluate + diagnose + candidates | ≈71 |
| `b_vit_dino_frozen`: тренування (як `b_vit_imagenet_frozen`: 42 хв) + те саме | ≈46 |
| відсутні входи старих запусків | ≈3 |
| CPU-таблиці (paired, box_vs_cls, ознаки DINO для кандидатів, two-stage evaluate) | ≈12 |
| **разом** | **≈132 (≈2.2 год)** |

## Критерій переходу на етап 2 (зафіксовано до запуску)

Етап 2 виправданий, якщо **обидві** умови виконано за точковими оцінками test infected AP50 (`phase6_dino_detr_runs.csv`):
1. AP50(`b_vit_dino_full`) > AP50(`b_vit_imagenet`) = 0.5276;
2. AP50(`b_vit_dino_full`) > AP50(`a_coco_detr_binary`) − 0.03 = 0.5213, тобто dino_full нижчий за baseline A
   менш ніж на 0.03.

Застереження: критерій використовує test-оцінки. Він не обирає модель і не змінює жоден гіперпараметр, але рішення
про реплікацію залежить від test; результати етапу 2 будуть звітуватися з цією приміткою. Значення val (чистий і
stain-shifted) звітуються поруч, але в критерій не входять.

## Етап 2 (лише описано; НЕ запускається без окремого рішення після огляду етапу 1)

- Запуски на seed 43 і 44:
  - `b_vit_dino_full_seed{43,44}` (нові конфіги: лише `seed`);
  - `b_vit_imagenet_seed{43,44}` (конфіги вже є з фази 4, черга `phase4_seeds` не запускалась);
  - `a_coco_detr_binary_seed{43,44}` (нові конфіги: лише `seed`).
- Оцінка та сама, що на етапі 1. Пари dino_full − imagenet_full і dino_full − baseline A: на кожному зерні та для
  середнього трьох зерен на тих самих 1000 ресемплах. Для середнього треба додати в `paired_detr_diff.py` опцію
  `--mean` за зразком `paired_fold_diff.py --mean`.
- Оцінка часу: 4 ViT-запуски × ≈71 хв + 2 запуски baseline A × ≈46 хв (43 хв тренування + оцінка) + CPU ≈ 6.4 год.

## Що звітується (`reports/phase6_summary.md`, після черги)

Лише числа з `reports/tables/`, у форматі `phase5_summary.md`, без інтерпретації:
- таблиця запусків: val / stain-val / test AP50 з CI, test P / R / F1 з CI, best epoch, частки `diagnose_shift`;
- усі парні різниці з 95% CI; позначено, які CI не містять 0;
- двоетапна таблиця і різниці;
- таблиця `box_vs_cls`;
- перевірка критерію переходу на етап 2 (лише арифметика за правилом вище).
