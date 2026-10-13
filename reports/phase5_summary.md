# Фаза 5: self-supervised domain adaptation (EXPLORATORY; підготовлено, не запущено; + ARM 1–3)

## Гіпотеза і статус

**Експеримент EXPLORATORY: гіпотезу сформульовано ПІСЛЯ перегляду test-результатів фази 4.**

Гіпотеза. Наші in-domain SSL-енкодери «вузькі». У crop-probe вони на рівні ImageNet, але всередині end-to-end DETR
під зсувом домену поводяться як random init, тоді як ImageNet-init дає приблизно на 0.2 вищий test AP50
(`reports/phase4_summary.md`). Продовження MAE + InfoNCE претренування **від ImageNet-ініту** на train-зображеннях
разом із **нерозміченими** зображеннями target-домену може додати знання про домен і не втратити широту ImageNet.
Один лише ImageNet-енкодер цього не вміє.

Обмеження, які діють для всього експерименту:
- seed один (42);
- поправки на множинність немає;
- усі гіперпараметри зафіксовано в конфігах до запуску; на test нічого не підбирається;
- **мітки test ніколи не використовуються для адаптації**, лише для фінальної оцінки (probe на GT-кропах);
- чекпойнти не обираються: використовується `encoder_final.pt` після 30 епох;
- переможця за test не обираємо.

## Фолди test

`scripts/make_test_folds.py` (seed 42) створює `data/splits/test_folds.json`; таблиця —
`reports/tables/phase5_test_folds.csv`. Зображення відсортовано за кількістю infected-клітин, і в кожній сусідній
парі одне зображення випадково йде у фолд 1, друге — у фолд 2.

| фолд | зображень | infected-клітин | зображень з infected | клітин (усі класи) |
|---|---|---|---|---|
| 1 | 60 | 150 | 57 | 2919 |
| 2 | 60 | 153 | 58 | 3003 |

Мітки test тут використано тільки для балансування фолдів, тобто для підрахунку infected на зображення. Сам файл
фолдів містить лише id і назви файлів. SSL-loader читає тільки цей файл і JPEG-и свого фолду
(`build_target_cache`); анотацій test він не відкриває. Це перевіряє `tests/test_ssl_da.py`: тест перехоплює
відкриття файлів і `load_split_coco`.

Протокол: енкодер, адаптований на фолді F (лише зображення, без боксів і міток), оцінюється тільки на кропах
**іншого** фолду. Результати звітуються окремо для кожного held-out фолду і як середнє двох фолдів.

## Варіанти (оголошено до запуску, 2026-10-13)

Усі варіанти — ViT-S/16 і той самий об'єкт навчання, що в `ssl_joint`:
- MAE (mask 0.75) + симетричний InfoNCE (τ 0.2), λ_con 0.2;
- ті самі view на DETR-масштабі та ті самі аугментації;
- batch 128, 25600 пар на епоху;
- свіжі MAE-декодер і projector.

| варіант | конфіг | init | дані | вхідна нормалізація |
|---|---|---|---|---|
| imagenet_cont (КОНТРОЛЬ) | `configs/ssl/imagenet_cont.yaml` | timm `vit_small_patch16_224.augreg_in21k_ft_in1k` | лише train | 0.5 / 0.5 (timm) |
| imagenet_da, фолд F | `configs/ssl/imagenet_da_f{1,2}.yaml` | те саме | train + нерозмічений test-фолд F | 0.5 / 0.5 |
| joint_cont (КОНТРОЛЬ) | `configs/ssl/joint_cont.yaml` | `runs/ssl_joint/checkpoints/encoder_final.pt` | лише train | ImageNet (як у ssl_joint) |
| joint_da, фолд F | `configs/ssl/joint_da_f{1,2}.yaml` | те саме | train + нерозмічений test-фолд F | ImageNet |

Зафіксовані гіперпараметри (не підбиралися):
- **30 епох** (0.77 M пар). Це 30% бюджету 100-епохового претренування з нуля.
- **blr 1.5e-4** (lr 7.5e-5 при batch 128). Це дефолт MAE, у 3.3 раза менший за 5e-4 з нуля, щоб не зруйнувати
  претреновані ознаки. Warmup 5%, далі косинус до 1% lr.
- **target.fraction 0.5**: збалансоване семплювання доменів. Кожна пара береться з 60 target-зображень з
  імовірністю 0.5 (`MixedImageSampler`). Отже, одне target-зображення дає ≈213 пар за епоху, а одне train-зображення
  ≈12.5; це ризик перенавчання на 60 зображеннях фолду.
- Контролі `_cont` мають той самий бюджет кроків, але без target-зображень. Вони відокремлюють «адаптацію до
  домену» від «просто довшого претренування».

Конфіги перевіряє `tests/test_ssl_da.py`:
- `imagenet_cont` відрізняється від `joint` лише ключами `model.init`, `optim.blr`, `train.epochs`;
- `joint_cont` відрізняється від `imagenet_cont` лише `model.init`;
- `_da_f*` відрізняються від свого `_cont` лише блоком `target`.

**Нормалізація.** Вхідна нормалізація енкодера записується в `model.input_norm` конфігу запуску. Probe
(`load_encoder`) і `vit_detr` (`init: <шлях>`) читають її звідти. Для старих запусків без цього ключа
використовуються статистики ImageNet, тому їхні результати не змінюються.

## Оцінка (план, зафіксований до запуску)

1. **Probe на held-out фолді** (`probe.py --run ... --eval-split test --eval-fold k`). Протокол той самий, що раніше:
   box-centred кропи GT, kNN і лінійний L-BFGS, навчені на train-кропах. Метрики — AUPRC infected (лінійний і kNN,
   100% міток). Тег — `test_fold<k>_<encoder>`. RBC-кропи test (2000) сабсемплюються до розбиття на фолди, тож
   на фолд припадає ≈1000. Інфіковані кропи: 150 (фолд 1) і 153 (фолд 2). CI широкі.
2. **Парний bootstrap** (`scripts/paired_fold_diff.py`):
   - 1000 ресемплів зображень на фолд (probe seed);
   - для середнього ресемпл r фолду 1 об'єднується з ресемплом r фолду 2;
   - класифікатори перенавчаються на збережених ознаках і звіряються з json probe.
   
   Пари різниць:
   - imagenet_da − ImageNet (заморожений референс);
   - imagenet_da − imagenet_cont;
   - joint_da − joint (`ssl_joint`);
   - joint_da − joint_cont;
   - joint_da − imagenet_da.
   
   «imagenet_da» на фолді 1 — це енкодер, адаптований на фолді 2, і навпаки.
3. **Val-probe** нових енкодерів (`probe.py --run`). Він рахує чистий val, stain-shifted val і low-label, тобто
   наявний stain-shifted протокол. Stain-shift визначено лише для val, на test він не застосовується.
4. **Перевірки.**
   - No-collapse: `rep/std_x_sqrt_d`, `rep/erank`, `proj/*` на фіксованих val-парах щоепохи, як у фазі 3.
   - Grids реконструкції `recon/target_epoch_XXX.png` на фіксованих кропах target-зображень фолду адаптації,
     кожні 5 епох.
   - Light-probe на val кожні 10 епох; він лише для моніторингу і ні на що не впливає.

DETR із цими енкодерами не входить у цю фазу. Рішення про нього — після результатів probe.

## Застереження

- Обидва фолди походять з одного test-набору (одна партія знімків), тож адаптація трансдуктивна в межах домену.
  Held-out фолд адаптований енкодер не бачив, але його зображення однорідні з фолдом адаптації.
- Мітки test використано для балансування фолдів (лише кількість infected на зображення).
- imagenet_da − ImageNet змішує два ефекти: довше претренування на мікроскопії і target-зображення. Розділяє їх
  контроль imagenet_da − imagenet_cont.
- joint_da − imagenet_da порівнює різні ініціалізації з різною нормалізацією входу.
- ImageNet-енкодер раніше на test-кропах (усі 120 зображень): лінійний AUPRC 0.876, joint 0.808 (фаза 4). Тут ті
  самі енкодери перераховано на кожному фолді окремо.

## GPU-smoke і оцінка часу

`runs/dry_p5_imagenet_da_f1`, 2 епохи, ImageNet-init + фолд 1:
- target-кеш фолду 1 (60 × 683 × 960) збудовано;
- batch 128, lr 7.5e-5;
- епоха 89.7 с (перша 120 с), 285 пар/с, пік 10.75 GiB;
- light-probe ≈20 с;
- колапсу немає: `rep/erank` 128.8, `rep/std_x_sqrt_d` 0.60;
- grid реконструкції на target-кропах збережено.

Також на цьому dry-енкодері перевірено ланцюжок `probe.py --eval-fold` → `paired_fold_diff.py`. Його вихідні
файли видалено; числа dry-енкодера ніде не використовуються.

Оцінка: ≈0.8 год на 30-епоховий запуск.

| варіант | запусків | год |
|---|---|---|
| imagenet_cont | 1 | 0.8 |
| imagenet_da (фолди 1, 2) | 2 | 1.6 |
| joint_cont | 1 | 0.8 |
| joint_da (фолди 1, 2) | 2 | 1.6 |
| probe (6 val + 16 fold) + bootstrap | — | 0.3 |
| **разом** | | **≈5.1** |

Черга `configs/queues/phase5_da.txt`:
- спершу ImageNet-варіанти (основна гіпотеза), потім joint-варіанти (другорядні, їх можна відкинути, якщо бракує
  часу);
- далі probe;
- таблиця `phase5_da_imagenet` (лише ImageNet-пари) пишеться перед joint-probe, тож вона з'явиться, навіть якщо
  joint-запуски відкинуто.
```
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_queue.ps1 -Queue configs/queues/phase5_da.txt
```

## Додаткові руки (EXPLORATORY; оголошено 2026-10-13 до будь-якого запуску фази 5)

Три руки дописано в **кінець** черги `configs/queues/phase5_da.txt`. Наявні кроки 1–27 і їхній порядок не змінено,
тож кожну руку можна відкинути, якщо бракує часу. Обмеження ті самі, що для всієї фази:
- EXPLORATORY: гіпотеза після перегляду test;
- один seed, крім ARM 2;
- поправки на множинність немає;
- на test нічого не підбирається;
- test — лише 120 зображень; held-out фолд — 60 зображень і 150–153 infected-кропи, тож CI широкі.

### ARM 1: imagenet_da_kang (фолди 1 і 2)

Конфіги `configs/ssl/imagenet_da_kang_f{1,2}.yaml` відрізняються від `imagenet_da_f{1,2}` **лише** ключами
аугментації; це перевіряє `tests/test_ssl_da.py`. Рецепт узято з Kang et al., CVPR 2023, «Benchmarking
Self-Supervised Learning on Diverse Pathology Datasets» (arXiv 2212.04690v2, Додаток B.3):
- weak colour jitter: p 0.8, brightness / contrast / saturation / hue = 0.2 / 0.2 / 0.2 / 0.1;
- random grayscale: p 0.2;
- vertical flip: p 0.5;
- solarization немає;
- RandStainNA-GMM: p 0.8.

**Наш рецепт уже має** jitter p 0.8 з 0.2 / 0.2 / 0.2 / **0.02**, vertical flip p 0.5 і не має solarization. Тож
джиттер Kang **не слабший** за наш: B/C/S однакові, hue сильніший. За рішенням користувача беремо значення
статті буквально. Задекларовані зміни:

| ключ | imagenet_da | imagenet_da_kang |
|---|---|---|
| `views.color_jitter.hue` | 0.02 | 0.1 |
| `views.grayscale.p` | 0 (вимкнено) | 0.2 |

Незмінні: brightness / contrast / saturation 0.2, jitter p 0.8, vflip 0.5, hflip 0.5, rot90 0.5, HED jitter
(σ 0.03, p 0.5), blur, JPEG, solarization немає.

RandStainNA-GMM не реалізовано. Його роль stain-аугментації виконує наш HED jitter, тому повна відповідність
статті неможлива.

Grayscale (`A.ToGray`, 3 канали) стоїть після colour jitter і HED jitter, щоб знебарвлений кроп лишався сірим.
Він потрапляє в пайплайн лише при p > 0, тому випадковий потік інших конфігів не змінюється; це перевіряє тест.

GPU-smoke `runs/dry_p5_imagenet_da_kang_f1` (2 епохи): 90.3 с на епоху (перша 121 с), пік 10.75 GiB.

Оцінка (`phase5_da_kang`), парний bootstrap на тих самих held-out фолдах:
- imagenet_da_kang − imagenet_da;
- imagenet_da_kang − ImageNet (заморожений референс).

### ARM 2: imagenet_da_seed2 (фолди 1 і 2)

Конфіги `configs/ssl/imagenet_da_seed2_f{1,2}.yaml` відрізняються від `imagenet_da_f{1,2}` **лише** `seed`: 42 → 43;
це перевіряє тест. Від seed залежать порядок семплів, view і аугментації, маски MAE та ініціалізація свіжих
декодера і projector. ImageNet-ініт і фолди ті самі.

Оцінка (`phase5_da_seeds`):
- imagenet_da (seed 42) і imagenet_da_seed2 (seed 43) поруч, а також їхнє середнє `imagenet_da_2seeds`
  (`paired_fold_diff.py --mean`: метрика усереднюється на тих самих ресемплах);
- окремо для кожного held-out фолду і як середнє фолдів.

Пари різниць, задекларовані заздалегідь:
- seed2 − seed1 (варіабельність за seed);
- 2seeds − ImageNet;
- 2seeds − imagenet_cont.

Навіть два seed-и — це не оцінка дисперсії; результат інформативний лише щодо знаку і порядку величини.

### ARM 3: DINO ViT-S/16, референс (без тренування)

timm `vit_small_patch16_224.dino` (`probe.references.dino`, `probe.py --ref dino`, тег `ref_dino`). Mean і std
читаються з `timm.models.get_pretrained_cfg` і записуються в `source` probe json; для цієї моделі це 0.485 / 0.456
/ 0.406 і 0.229 / 0.224 / 0.225. Тест перевіряє, що `load_encoder` бере саме ці значення і що вхід енкодера в probe
дорівнює `(crop / 255 − mean) / std` з цього конфігу.

Оцінка (нічого не тренується, DETR із DINO не тренуємо):
- **Crop-probe:** val (чистий, stain-shifted, 10% міток; kNN і лінійний L-BFGS; AUPRC infected, 95% CI) і
  test-кропи (`--eval-split test`, фінальна оцінка).
- **Парний bootstrap** (`paired_probe_diff.py`, val і test): joint − dino і dino − ImageNet-supervised.
- **Двоетапний пайплайн** (`two_stage.py`):
  - ті самі кандидати (`a_coco_detr_binary`, `b_vit_imagenet`) і той самий лінійний класифікатор (clean і
    stain-augmented);
  - варіант скору і поріг обираються на val;
  - пари `--pairs ssl_joint:ref_dino ref_dino:ref_imagenet` (нова опція; за замовчуванням лишається joint −
    ImageNet, як у фазі 4);
  - ImageNet і joint у таблиці `phase5_two_stage_dino` перераховано з тими самими фічами, що у фазі 4.

### Оцінка часу всієї черги

| частина | год |
|---|---|
| основна частина (кроки 1–27, див. вище) | ≈5.1 |
| ARM 1 (2 × 0.8 год + 4 probe) | ≈1.7 |
| ARM 2 (2 × 0.8 год + 4 probe) | ≈1.7 |
| ARM 3 (2 probe, 2 bootstrap, ознаки 1 енкодера, evaluate на CPU) | ≈0.25 |
| **разом** | **≈8.75** |
