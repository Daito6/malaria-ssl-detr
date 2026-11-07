# План: крива робастності на val (EXPLORATORY)

**Статус: EXPLORATORY, один seed.** План зафіксовано 2026-10-14, до будь-якого запуску `scripts/robustness_curve.py`.
Протокол нижче не змінюється після перегляду результатів. Якщо зміна все ж знадобиться, вона буде описана в
окремому розділі «Відхилення від плану» з датою і причиною.

## Мета

Описати, як infected AP50 і recall трьох моделей деградують під контрольованими спотвореннями знімка. Це
описова діагностика. За кривою не обираються моделі, пороги чи гіперпараметри, і вона не замінює оцінку
на test. Вона доповнює фази 4–5, де домен-зсув val → test є центральною темою.

## Дані

- Лише **val** (181 знімок, бінарна розмітка `configs/data.yaml`, клас `difficult` відкинуто). **Test не читається.**
- Сирі знімки з `data/raw` тільки читаються.

## Моделі

`best.pt` кожного запуску, без перенавчання:

| id | Опис |
|---|---|
| `a_coco_detr_binary` | Baseline A, DETR R50 (COCO) |
| `b_vit_imagenet` | ViT-DETR, ImageNet ViT-S/16, рецепт `full` |
| `b_vit_joint` | ViT-DETR, SSL joint ViT-S/16 (`runs/ssl_joint/checkpoints/encoder_final.pt`) |

## Спотворення і сітка

Спотворення (`app/corruptions.py`) застосовується **на вході моделі**, тобто після eval-ресайзу до довгої сторони.
Саме там `evaluate.py --stain-shift` застосовує свій зсув. Масштаб змінює довгу сторону входу, як `--long-side`
у фазі 4, бо ресайз усього знімка фіксований eval-ресайз скасував би. Сила s = 0 повертає вхід без змін.

| Тип | Сила s → параметри | s = 1 |
|---|---|---|
| `stain` | HEDJitter σ = 0.08·s; ColorJitter яскравість / контраст / насиченість = 0.3·s, відтінок = 0.05·s; p = 1; seed (2024, id знімка) через `apply_stain_shift` (той самий напрямок для всіх s) | кольорова частина stain-shift probe (`configs/ssl/probe.yaml`) |
| `blur` | гаусове, σ = 2.0·s px на вході моделі | σ = 2 px (stain-shift: 0.5–1.0) |
| `jpeg` | якість round(95 − 85·s): 74 / 52 / 31 / 10 | q = 10 (stain-shift: 30–60) |
| `brightness_contrast` | контраст ×(1 − 0.5·s) відносно середнього каналу, потім яскравість ×(1 − 0.3·s) (лише затемнення) | контраст ×0.5, яскравість ×0.7 |
| `scale_down` | довга сторона round(800·0.8^s): 757 / 716 / 677 / 640 | 640 (точка `--long-side 640` фази 4) |
| `scale_up` | довга сторона round(800·1.27^s): 849 / 902 / 957 / 1016 | 1016 (1.27 = відношення розмірів клітин у сирих test- і val-знімках) |
| `stain_full` (контроль) | лише s = 1: повний stain-shift probe (HED + jitter + розмиття 0.5–1 + JPEG 30–60) | те саме, що `evaluate.py --stain-shift` |

Сітка: s ∈ {0, 0.25, 0.5, 0.75, 1}. Точка s = 0 спільна для всіх типів і рахується один раз. Разом на модель
1 + 6 × 4 + 1 = **26 умов**, на всі три моделі **78 прогонів** по 181 знімку.

Контрольна точка `stain_full` перевіряє, що спотворення відтворено коректно: її значення порівнюються з
`runs/<model>/metrics_val_stain.json`. Тест `test_stain_full_is_the_evaluation_stain_shift` уже перевіряє, що вхід
моделі побітово збігається з `DetectionDataset(shift=val_stain_shift)`.

## Інференс

- `app.inference.DetrPredictor`. Його паритет із конвеєром `evaluate.py` перевіряється тестом запит за запитом.
- Залишаються всі запити, без NMS, як у протоколі оцінки.
- fp32 на CPU (`--device cpu` за замовчуванням).
- Прогнози кожної умови кешуються в `app/cache/robustness/<model>/` (git-ignored).

## Метрики

- **Головна:** infected AP50 (без порогу; COCOeval, maxDets 300, як у `malaria_ssl.eval`).
- **Додатково:** precision, recall, F1 класу infected при **порозі з чистого val** (`runs/<model>/thresholds.json`),
  незмінному для всіх умов (як в `evaluate.py --stain-shift`).
- **95% CI:** перцентильний bootstrap по зображеннях val, **1000 ресемплів, seed 42**. Ресемпли спільні для всіх
  моделей і умов, тому різниці парні (`malaria_ssl.eval.paired`).
- **Парні різниці:**
  1. кожна умова мінус чиста точка тієї ж моделі;
  2. `b_vit_imagenet − a_coco_detr_binary`, `b_vit_joint − a_coco_detr_binary`, `b_vit_joint − b_vit_imagenet`
     в одній умові.

  Поправки на множинні порівняння немає: 78 точок і понад 200 різниць, тому окремі CI, що не перетинають 0,
  читаються обережно.
- Значення при s = 0 рахуються заново на CPU у fp32. Еталон `metrics_val.json` (CUDA bf16) записано поряд у json
  як контекст і не замінюється.

## Вихід

- `reports/tables/robustness_val.csv`: рядок на (модель, спотворення, сила) з AP50, precision, recall, F1 і CI.
- `reports/tables/robustness_val_diff.csv`: парні різниці.
- `reports/tables/robustness_val.json`: протокол, еталони `evaluate.py`, список відсутніх умов.
- Вкладка «Крива» в SmearScan читає ці файли і нічого не вшиває.

## Порядок запуску (частинами, лише після підтвердження)

```powershell
conda run -n medthesis python scripts/robustness_curve.py predict --corruptions stain blur jpeg
conda run -n medthesis python scripts/robustness_curve.py aggregate      # проміжна таблиця, відсутні умови в json
conda run -n medthesis python scripts/robustness_curve.py predict --corruptions brightness_contrast scale_down scale_up stain_full
conda run -n medthesis python scripts/robustness_curve.py aggregate
```

## Оцінка часу (CPU, 6 потоків)

Заміри 2026-10-14 на 10 val-знімках:

| Модель | довга сторона 800 | довга сторона 1016 |
|---|---|---|
| Baseline A | 356 мс | 529 мс |
| ViT-DETR ImageNet | 645 мс | 1159 мс |
| ViT-DETR joint | 598 мс | 1153 мс |

| Частина | Розрахунок | Час |
|---|---|---|
| Частина 1 (clean + stain, blur, jpeg) | 13 умов × 181 × 1.60 с | ≈ 1.0 год |
| Частина 2: brightness_contrast + stain_full | 5 умов на 800 | ≈ 0.4 год |
| Частина 2: scale_up | інтерполяція між 800 і 1016 | ≈ 0.5 год |
| Частина 2: scale_down | верхня межа | ≤ 0.3 год |
| Агрегація (78 прогонів, 1000 ресемплів) | оцінка, не замір | ~10–20 хв |

Якщо паралельно йде тренування, час може бути більшим.

## Чого крива не дає

- Висновків про test або про реальний зсув домену: синтетичні спотворення лише моделюють окремі його аспекти.
- Вибору між моделями: 1 seed, моделі не перенавчаються.
