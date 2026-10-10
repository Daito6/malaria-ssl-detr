# Фаза 2 — детекційний пайплайн, оцінювання, baseline A

Усі числа нижче — з реальних запусків у `runs/` (dry runs) та тестів. Повні 40-епохові запуски
ще **не** виконувались; test-спліт не використовувався (ні в dry runs, ні в тестах, крім
перевірки GT-статистики розміру боксів після ресайзу).

## Що побудовано

| Компонент | Файли |
|---|---|
| Детекційний датасет, аугментації, collate | `src/malaria_ssl/data/detection.py`, `configs/detr/base.yaml` |
| Моделі `coco_detr_r50`, `scratch_r50`, параметр-групи | `src/malaria_ssl/detr/build.py` |
| Loss з вагами класів | `src/malaria_ssl/detr/loss.py` |
| Тренер + CLI | `src/malaria_ssl/detr/trainer.py`, `scripts/train_detr.py` |
| Оцінювання + CLI | `src/malaria_ssl/eval/{inference,coco_metrics,detection_metrics,bootstrap,report}.py`, `scripts/evaluate.py` |
| Конфіги експериментів | `configs/experiments/a_coco_detr_{binary,multiclass}.yaml`, `a_scratch_r50_binary.yaml` |
| Аналіз | `notebooks/02_baseline_analysis.ipynb`, `src/malaria_ssl/eval/viz.py` |
| Тести | `tests/test_detection.py`, `test_build.py`, `test_eval.py`, `test_trainer.py` — разом з фазою 1 **30 passed** |

## Середовище (крок 0)

- torchvision 0.29.0+cu130 — `nms` на CUDA працює; `check_env.py` тепер ще й перевіряє імпорт
  нативних розширень.
- **Smart App Control** блокував DLL двох пакетів:
  - `scikit-learn` → через нього падав `import transformers`. Видалено (як узгоджено).
  - `grpcio` 1.84 → не стартував сервер TensorBoard. Встановлено `grpcio==1.76.0`, з нею працює.
  Обидва моменти зафіксовано в `requirements.txt`.
- Додано: peft 0.21.2, albumentations 2.0.8 (+ opencv-python-headless 5.0.0.93), tensorboard 2.21.0.

## Ключові проєктні рішення

1. **Мітки.** Мультиклас — 7 класів; бінарний режим будується з того самого мапінгу
   `configs/data.yaml` (`difficult` відкидається). Внутрішні мітки 0..K−1, зворотний мапінг на
   COCO category id зберігається для предикцій.
2. **Аугментації** (усі значення в `configs/detr/base.yaml`): eval — довша сторона 800; train —
   випадкова довша сторона з {640, 672, …, 960}, crop від країв (p=0.3, ≤10% з кожного боку,
   `min_visibility=0.6`), hflip/vflip/rot90, мʼякий ColorJitter, blur (p=0.1), JPEG (p=0.3,
   якість 60–100), HED stain jitter — власна реалізація, за замовчуванням вимкнено.
   Після ресайзу медіана довшої сторони боксу: train 56.5 px, val 56.5 px, test 59.3 px
   (різниця 4.9%, тест < 10% проходить).
3. **Детермінізм.** Sampler видає пари `(індекс, seed)`, де seed залежить лише від
   (seed, епоха, позиція); датасет пересіює albumentations цим seed. Батчі не залежать від
   кількості воркерів (тест: однаковий seed → однаковий перший батч).
4. **300 queries.** Рядки 0–99 — претреновані COCO-queries без змін; рядки 100–299 —
   `query[i mod 100] + N(0, (0.1·std)²)`. Копії переносять вивчені просторові пріори COCO-queries,
   шум розбиває симетрію, щоб matcher міг розвести дублікати.
5. **Backbone.** У transformers 5.17 stem і `layer1` ResNet **не заморожувались** (хоча в
   оригінальному DETR і у вашому бакалаврському запуску заморожені). Заморожую явно: 222 400
   параметрів заморожено, тренується layer2–4. Для `scratch_r50` ImageNet-ваги
   (`resnet50.tv_in1k` — ті самі torchvision-ваги, що й в оригінальному DETR) завантажуються
   явно й звіряються після завантаження: `DetrForObjectDetection(config)` їх не підтягував,
   і backbone лишався б випадковим.
6. **Auxiliary loss — знайдена розбіжність у transformers.** З `auxiliary_loss=True` декодер HF
   застосовує фінальний LayerNorm до прихованого стану після **кожного** шару, а не до копії, як в
   оригінальному DETR. На претренованій COCO-моделі це зсуває передбачені бокси в середньому на
   0.196 (максимум 0.929) у нормованих координатах, тобто ламає претреновану мережу. Тому в моделі
   `auxiliary_loss=False`, а проміжні виходи збираються forward-хуками й нормуються копії
   (перевірено: хук відтворює фінальний стан з різницею 0.0).
7. **Loss.** transformers не підтримує ваги класів, тому loss рахується окремо: перевикористано
   `HungarianMatcher`/`ImageLoss` з HF, коефіцієнти ті самі (CE 1, L1 5, GIoU 2, eos 0.1), матчинг
   у fp32 поза autocast. Ключі `loss.eos_coefficient` і `loss.class_weights` — у конфігу,
   за замовчуванням без змін.
8. **Оптимізація.** AdamW, lr 1e-4 (трансформер, input projection, голови, queries), 1e-5
   (backbone), wd 1e-4, grad clip 0.1, warmup 200 кроків + cosine до 1% від lr, bf16 autocast,
   effective batch 8.
9. **Оцінювання.**
   - pycocotools з `maxDets=[1, 10, 300]`. **Пастка pycocotools:** `COCOeval.summarize` для першого
     показника (AP@[.5:.95]) жорстко шукає maxDets=100 і з нашими налаштуваннями повертає −1.
     Тому всі показники рахуються напряму з масивів `precision`/`recall` при 300 детекціях.
     Тест звіряє їх із `ev.stats` там, де pycocotools коректний (стандартні maxDets).
     Тест «GT як предикції» дає AP=AR=1 в обох режимах, а з maxDets=100 той самий тест дає AR<1,
     тобто обрізання на щільних зображеннях реальне.
   - Пороги P/R/F1 на клас (max F1) обираються на val, зберігаються в `thresholds.json` і без змін
     застосовуються до test; `evaluate.py --split test` без val-порогів відмовляється працювати.
   - Bootstrap переагрегує пер-зображеннєвий матчинг COCOeval замість повного перерахунку
     (тест: без ресемплу збігається з AP50 COCOeval з точністю 1e-12).
   - COCOeval рахується лише для діапазону площі «all» (інші не звітуються): валідація
     20.8 → 5.9 с.

## Dry runs

### (a) Overfit 8 train-зображень (без аугментацій, валідація на тих самих 8)
У цих 8 зображеннях 431 uninfected і 9 infected боксів.

| Запуск | loss: старт → кінець | AP | AP50 | AP50 uninfected | AP50 infected | AR@300 |
|---|---|---|---|---|---|---|
| 200 кроків, warmup 10 + cosine | 25.18 → 4.50 | 0.360 | 0.445 | 0.873 | 0.018 | 0.462 |
| 400 кроків, постійний LR (контроль) | 25.18 → 3.03 | 0.795 | 0.886 | 0.961 | 0.810 | 0.909 |

Після 200 кроків з cosine-розкладом infected-клітини були **правильно локалізовані**
(IoU 0.84–0.97 для всіх 9), але класифікувались як uninfected. Контрольний запуск із постійним LR
показує, що пайплайн вивчає й рідкісний клас. Тобто причина — замало кроків при дисбалансі
1:48, а не помилка в мітках чи loss.

### (b) 2 епохи на повному train (binary, `a_coco_detr_binary` з `epochs=2`)
- Probe micro-batch: батч 8 на полотні 960×960 з 223 боксами на зображення → пік 6.98 GiB
  (бюджет 13.52 GiB = 85% від 15.9). **micro_batch = 8, акумуляція не потрібна.**
- Епоха 0: 93.6 с (разом зі стартом воркерів); епоха 1: **58.4 с** (17.7 img/s), 129 кроків на епоху.
- Пікова пам'ять GPU під час тренування: 6.32 GiB.
- Валідація на val (181 зображення): інференс 3.5 с + COCOeval 5.9 с; плюс одноразово ~32 с на
  старт воркерів eval-loader'а (Windows spawn).
- Чекпоінти: `last.pt` 475 МБ, `best.pt` 159 МБ на запуск.
- Val після 2 епох: AP 0.191, AP50 0.317 — це лише перевірка, що все працює, не результат.
- `evaluate.py --split val --bootstrap` (200 ресемплів) разом із завантаженням моделі: 56 с.

### Оцінка часу на повні запуски
За виміряною епохою 58.4 с: 40 × 58.4 с ≈ 39 хв тренування + 20 валідацій × ≈9.4 с ≈ 3 хв +
≈1 хв на старт воркерів → **≈ 43 хв на binary-запуск**. Multiclass і scratch мають ту саму
архітектуру й дані, тому очікую подібний час (окремо не міряв). Разом ≈ 2.2 год на три запуски.

## Команди для повних запусків (запускаєте ви)

```powershell
conda activate medthesis
cd C:\Users\Alex\PycharmProjects\MedImage_Detection\malaria-ssl-detr

python scripts/train_detr.py --config configs/experiments/a_coco_detr_binary.yaml
python scripts/train_detr.py --config configs/experiments/a_coco_detr_multiclass.yaml
python scripts/train_detr.py --config configs/experiments/a_scratch_r50_binary.yaml

# після кожного запуску (val обов'язково першим — там обираються пороги):
python scripts/evaluate.py runs/a_coco_detr_binary --split val --bootstrap
python scripts/evaluate.py runs/a_coco_detr_binary --split test --bootstrap
# ... те саме для a_coco_detr_multiclass і a_scratch_r50_binary

# продовжити перерваний запуск:
python scripts/train_detr.py --config configs/experiments/a_coco_detr_binary.yaml --resume
```
Логи: `tensorboard --logdir runs`. Ноутбук `02_baseline_analysis.ipynb` автоматично підхопить
повні запуски замість dry run.

## Що варто знати / ризики

1. **Розклад LR і рідкісні класи.** Overfit-тест показав, що infected вчиться помітно повільніше
   за uninfected. При 40 епохах кроків значно більше (5160), але ризик недонавчання рідкісного
   класу лишається — роботу з дисбалансом винесено в наступну фазу, як ви планували.
2. **cardinality_error** після 2 епох ≈ 64 (кількість «не-порожніх» передбачень на зображення
   відрізняється від GT на ~64). Для раннього DETR це очікувано; варто стежити за ним у
   TensorBoard.
3. **Бінарна матриця плутанини після 2 епох** показує, що 60% infected класифікуються як
   uninfected — та сама картина, що в overfit-тесті.
4. Грубі розміри: ноутбук 02 з прикладами важить 6.3 МБ (зображення в повній роздільності).
5. Повні запуски й метрики test ще не виконувались — числа для порівняння baseline'ів з'являться
   лише після них.
