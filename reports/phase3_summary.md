# Фаза 3 — SSL-претренування ViT-S/16 (MAE + InfoNCE) та probe без DETR

Усі числа нижче — з реальних запусків у `runs/` і тестів. Довгі SSL-запуски (sweep, основний joint,
mae_only, con_only) **ще не виконувались** — нижче лише dry runs, референси та пропозиція бюджету.
Test-спліт у фазі 3 не використовувався ні для чого, крім діагностики вже натренованого baseline A
(розділ «Зсув домену»), яка нічого не налаштовує.

## Що побудовано

| Компонент | Файли |
|---|---|
| Кеш декодованих зображень + дві views | `src/malaria_ssl/ssl/data.py`, `configs/ssl/data.yaml` |
| Модель MAE + InfoNCE навколо timm ViT-S/16 | `src/malaria_ssl/ssl/models.py` |
| Тренер + CLI | `src/malaria_ssl/ssl/trainer.py`, `scripts/pretrain.py`, `configs/ssl/base.yaml` |
| Probe на рівні клітин (kNN / linear) | `src/malaria_ssl/ssl/probe.py`, `scripts/probe.py`, `configs/ssl/probe.yaml` |
| Конфіги експериментів | `configs/ssl/{joint,mae_only,con_only}.yaml`, `configs/ssl/sweep_{lc005,lc02,lc10,uncertainty}.yaml` |
| Діагностика зсуву val → test | `src/malaria_ssl/eval/shift.py`, `scripts/diagnose_shift.py` |
| DETR зі stain jitter (не запускався) | `configs/experiments/a_coco_detr_binary_stain.yaml` |
| Аналіз | `notebooks/03_ssl_analysis.ipynb`, `src/malaria_ssl/ssl/viz.py` |
| Тести | `tests/test_ssl_data.py`, `test_ssl_models.py`, `test_ssl_trainer.py`, `test_probe.py`, `test_shift.py` — разом з фазами 1–2 **51 passed** |

Спільні хелпери тренування (`lr_lambda`, `run_info`, чекпоінти, `metrics.csv`, `--set`) перенесено з
`detr/trainer.py` у `utils/run.py` і `utils/config.py`, щоб SSL-код не імпортував DETR/transformers.
Перевірка після рефактору: `tests/test_trainer.py` (GPU) проходить; 1-епоховий запуск
`a_coco_detr_binary` дав епоху 0 з loss 12.2146 проти 12.2134 у `dry_b_2epochs` фази 2 (той самий
seed, обидва в warmup). Різниця 0.01% — недетермінізм CUDA-ядер (детермінізм не вмикався), а не
зміна логіки.

## Ключові рішення

1. **Дані.** SSL тренується лише на 1027 train-зображеннях (мітки не використовуються); val —
   тільки моніторинг; test не читається. Зображення один раз декодуються в `uint8` memmap з довшою
   стороною 960: train 2.13 GB, val 0.38 GB, збірка 33 с. Читання пари views — 8.1 мс в одному
   процесі.
2. **Масштаб views.** Довша сторона L ∈ {640, …, 960} (як у DETR train) → якір 260×260 px у
   масштабі L → два random-resized crops у якорі (площа 0.5–1.0, aspect 3/4–4/3) → 224×224.
   Сторона кропа 184–260 px, тобто масштаб ×0.86–1.22 (≈1.0 у середньому) — клітини бачаться в
   масштабі DETR. IoU двох views на 20 000 семплах: мінімум 0.269, медіана 0.709
   (теоретичний мінімум 0.165). Далі незалежно для кожної view: flips, rot90, ColorJitter,
   HED stain jitter, blur, JPEG.
3. **Модель.** timm `vit_small_patch16_224`, random init, `dynamic_img_size=True`. MAE на view 1
   (маска 0.75, енкодер бачить лише видимі токени, декодер 4 блоки / 256 / 8 голів, норм-піксельні
   таргети, loss лише на замаскованих патчах). Контрастивна гілка — повні проходи обох views,
   середнє патч-токенів, projector 384→2048→256, симетричний NT-Xent (τ = 0.2). Гілка з λ = 0 не
   рахується. Експортується лише енкодер (`encoder_final.pt`, `encoder_epXXX.pt`) — звичайний timm
   state dict, вантажиться `strict=True`, інтерполяція позицій працює для DETR-розмірів (тест).
4. **Оптимізація.** AdamW (0.9, 0.95), wd 0.05 (без wd на norm/bias/pos/cls/log-variances),
   lr = blr·B/256 з blr = 1.5e-4, warmup 5% + cosine до 1%, bf16, grad clip 3.0.
   **Без градієнтної акумуляції** — вона зменшила б кількість негативів InfoNCE.
5. **Probe.** Для кожного GT-боксу (difficult виключено, як у binary-мапінгу) — вікно 224 на
   зображенні з довшою стороною 800, центроване на боксі; ознака — середнє патч-токенів, центри яких
   у боксі (медіана 16 патчів на клітину, мінімум 4; найбільший бокс 127.5 px < 224). RBC
   підвибірка 4000 train / 2000 val, решта класів повністю: train 5940, val 2312 кропів.
   kNN (k = 20, косинус, ваги exp(sim/0.07)) і логістична регресія (стандартизація, class-balanced,
   L2 = 1e-3, L-BFGS) — однакові гіперпараметри для всіх енкодерів, без підбору. CI — bootstrap по
   val-**зображеннях** (1000 ресемплів). Метрики реалізовано в numpy (sklearn не використовується).

## Dry runs

### (b) Smoke усіх трьох режимів
Probe батча (пік пам'яті fwd+bwd, joint): 512 і 384 пар — OOM, 256 пар — 20.82 GiB, **128 пар
(= 256 кропів) — 10.75 GiB** → батч 128 пар, зафіксовано в `base.yaml`.

Чистий GPU-крок на синтетиці (батч 128, bf16, fused attention): joint 430 мс (298 пар/с),
mae_only 86 мс (1496 пар/с), con_only 350 мс (366 пар/с).

Реальний запуск (2 епохи × 25 кроків; епоха 0 включає старт воркерів ≈30 с):

| Режим | пар/с (епоха 1) | пік GPU, GiB | train L_mae (еп. 0 → 1) | train L_con (еп. 0 → 1) |
|---|---|---|---|---|
| joint (λ_con = 0.2) | 258 | 10.75 | 0.878 → 0.655 | 3.589 → 3.126 |
| mae_only | 542 (упирається в loader) | 2.91 | 0.866 → 0.649 | — |
| con_only | 311 | 8.62 | — | 3.556 → 3.056 |

Усі losses скінченні. Моніторинг на 512 фіксованих val-парах — 0.6 с на епоху; light probe —
≈15 с (перевірено в `dry_b_joint_lightprobe`). Масштаби: L_mae ≈ 0.65, L_con ≈ 3.1 (для довідки:
випадкові ембединги дають ln 255 ≈ 5.54 при 255 кандидатах).

### (a) Overfit MAE на 64 фіксованих train-кропах (без аугментацій, mae_only)
Батч 64, blr підвищено до 1.5e-3 (лише для цієї перевірки), 3000 кроків, 263 с тренування.
Loss на тих самих 64 кропах із фіксованими масками: 1.105 (епоха 0) → 0.483 (крок 100) →
0.188 (1000) → 0.049 (2000) → **0.035 (3000)**. Сітки `runs/dry_a_overfit_mae64/recon/`: на кроці 500
реконструкція розмита, на кроці 3000 — майже точна.

Перший короткий варіант (400 кроків) вийшов на плато ≈0.45 з пласкими патчами. Щоб відрізнити повільне
навчання від помилки позицій/таргетів, окремо перевірено запам'ятовування з **однією фіксованою
маскою** на 8 кропах: loss 1.329 → 0.0473 (500 кроків) → 0.0042 (1500). Тобто маскування, порядок
патчів, таргети й декодер узгоджені, а плато — звичайна рання фаза MAE.

### (c) Референси probe (val-кропи, 95% CI)

| Енкодер | Метод | AUROC infected | **AUPRC infected** | macro-F1 | balanced acc |
|---|---|---|---|---|---|
| random init | kNN | 0.928 [0.908, 0.944] | 0.730 [0.660, 0.787] | 0.465 [0.384, 0.536] | 0.437 [0.371, 0.508] |
| random init | linear | 0.986 [0.980, 0.991] | **0.924 [0.883, 0.954]** | 0.515 [0.460, 0.567] | 0.702 [0.630, 0.765] |
| ImageNet (augreg_in21k_ft_in1k) | kNN | 0.993 [0.986, 0.997] | 0.971 [0.952, 0.985] | 0.683 [0.602, 0.746] | 0.616 [0.546, 0.686] |
| ImageNet | linear | 0.997 [0.995, 0.998] | **0.981 [0.972, 0.989]** | 0.646 [0.589, 0.696] | 0.736 [0.676, 0.797] |

Low-label, linear AUPRC infected (середнє ± std по 5 seeds): random — 1%: 0.483 ± 0.100 (60 кропів),
10%: 0.845 ± 0.015 (593); ImageNet — 1%: 0.914 ± 0.013, 10%: 0.978 ± 0.004.
Час probe: ознаки ≈11.5 с на енкодер, усе разом 23–36 с.

**Спостереження.** Бінарна задача на рівні кропа проста навіть для random init (linear AUPRC
0.924) — інфіковані клітини сильно відрізняються кольором. Між random і ImageNet лише ≈0.06, а CI
SSL-варіантів, імовірно, перекриватимуться. Low-label 10% розділяє енкодери значно сильніше
(0.845 проти 0.978). Мультиклас (macro-F1) складний для всіх; у val лише 14 leukocyte,
20 gametocyte, 25 schizont — CI широкі.

## Зсув домену val → test (baseline A, `a_coco_detr_binary`, best = епоха 39)

| | AP50 infected (95% CI) | P / R / F1 infected при val-порозі 0.867 |
|---|---|---|
| val | 0.778 [0.728, 0.832] | 0.741 / 0.883 / 0.806 |
| test | 0.551 [0.479, 0.621] | 0.844 / 0.376 / 0.521 |

`scripts/diagnose_shift.py` (`reports/figures/shift_a_coco_detr_binary.png`,
`reports/tables/shift_a_coco_detr_binary.json`) — для кожної GT infected-клітини береться найвищий
infected-score серед детекцій з IoU ≥ 0.5:

| | n | медіана score | ≥ val-порогу | нема infected-детекції | uninfected-score вищий |
|---|---|---|---|---|---|
| val | 298 | 0.968 | 88.3% | 5.7% | 4.0% |
| test | 303 | 0.716 | 37.6% | **35.0%** | **37.0%** |

Частка test-клітин з score ≥ порогу (37.6%) збігається з recall infected у `metrics_test.json`
(114/303). Два механізми: (1) ≈35% test infected-клітин локалізовані, але класифіковані як uninfected
— PR-крива на test обривається на recall ≈0.65, і жоден поріг цього не виправить; (2) серед
виявлених score зсунутий униз (медіана 0.716 проти 0.968), тому val-поріг 0.867 відсікає більшість.
Перший механізм — питання репрезентації / стійкості до забарвлення, а не калібрування; це мотивує і
SSL-претренування, і stain-аугментацію (`a_coco_detr_binary_stain.yaml`).

## Пропозиція: сітка λ і бюджет

Розрахунок за виміряними 258 пар/с (joint), епоха = 25 600 пар (200 кроків, ≈99 с):

| Запуск | Конфіг | Бюджет | Оцінка часу |
|---|---|---|---|
| sweep λ_con = 0.05 | `sweep_lc005.yaml` | 32 еп. = 0.82 M пар | ≈55 хв |
| sweep λ_con = 0.2 | `sweep_lc02.yaml` | 0.82 M | ≈55 хв |
| sweep λ_con = 1.0 | `sweep_lc10.yaml` | 0.82 M | ≈55 хв |
| sweep uncertainty | `sweep_uncertainty.yaml` | 0.82 M | ≈55 хв |
| основний joint | `joint.yaml` (λ_con = переможець) | 100 еп. = 2.56 M | ≈2.8 год |
| mae_only | `mae_only.yaml` | 2.56 M | ≈1.3 год (542 пар/с) |
| con_only | `con_only.yaml` | 2.56 M | ≈2.3 год (311 пар/с) |

λ_mae = 1. За масштабами smoke λ_con·L_con ≈ 0.16 / 0.62 / 3.1 при L_mae ≈ 0.65, тобто сітка
покриває «MAE домінує / баланс / InfoNCE домінує». Вибір — за linear-probe AUPRC infected на
val-кропах (tie-break macro-F1), як узгоджено. Кожен sweep-запуск має власний cosine-розклад на
32 епохи; припущення — ранжування λ переноситься на 100 епох.

### Відкриті питання (рішення за вами)
1. **Learning rate.** Лінійне масштабування дає lr = 7.5e-5 при батчі 128. Overfit (a) потребував
   lr 3.75e-4, а на рівних 50 кроках smoke-loss MAE вищий, ніж в overfit. Можна залишити
   як є (правило з ТЗ) або додати до sweep один запуск `--set optim.blr=5.0e-4` (lr 2.5e-4) з
   λ_con = 0.2 (+≈55 хв). Без такого запуску я не маю даних, що краще.
2. **Чутливість головної метрики.** Через вузький діапазон (0.924 → 0.981) λ можуть не розрізнитися
   в межах CI. Пропоную, якщо CI переможця перекривається з іншими, як другий tie-break
   (перед macro-F1) взяти linear AUPRC при 10% міток — він розділяє референси значно сильніше.
3. **mae_only: малий розкид ембедингів** (std×√d ≈ 0.09 проти 0.40–0.45 у joint/con_only при erank ≈17).
   Для MAE-ознак це відома анізотропія, не колапс; стежитимемо в повному запуску.

## Команди (запускаєте ви; нічого з цього не запущено)

```powershell
conda activate medthesis
cd C:\Users\Alex\PycharmProjects\MedImage_Detection\malaria-ssl-detr

# 1) sweep (~55 хв кожен) + повний probe на val-кропах
python scripts/pretrain.py --config configs/ssl/sweep_lc005.yaml
python scripts/pretrain.py --config configs/ssl/sweep_lc02.yaml
python scripts/pretrain.py --config configs/ssl/sweep_lc10.yaml
python scripts/pretrain.py --config configs/ssl/sweep_uncertainty.yaml
python scripts/probe.py --run runs/ssl_sweep_lc005
python scripts/probe.py --run runs/ssl_sweep_lc02
python scripts/probe.py --run runs/ssl_sweep_lc10
python scripts/probe.py --run runs/ssl_sweep_uncertainty

# 2) після вибору: виставити lambda_con (або uncertainty) у configs/ssl/joint.yaml, потім
python scripts/pretrain.py --config configs/ssl/joint.yaml       # ~2.8 год
python scripts/pretrain.py --config configs/ssl/mae_only.yaml    # ~1.3 год
python scripts/pretrain.py --config configs/ssl/con_only.yaml    # ~2.3 год
python scripts/probe.py --run runs/ssl_joint
python scripts/probe.py --run runs/ssl_mae_only
python scripts/probe.py --run runs/ssl_con_only

# 3) DETR baseline A зі stain jitter (~43 хв) і його оцінка
python scripts/train_detr.py --config configs/experiments/a_coco_detr_binary_stain.yaml
python scripts/evaluate.py runs/a_coco_detr_binary_stain --split val --bootstrap
python scripts/evaluate.py runs/a_coco_detr_binary_stain --split test --bootstrap
python scripts/diagnose_shift.py runs/a_coco_detr_binary_stain

# перервані запуски: та сама команда з --resume; моніторинг: tensorboard --logdir runs
```
Після цього — перевиконати `notebooks/03_ssl_analysis.ipynb` (він автоматично перемкнеться з dry runs
на `runs/ssl_*`).
