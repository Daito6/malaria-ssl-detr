# malaria-ssl-detr

Магістерська робота про детекцію та класифікацію клітин крові за допомогою Detection Transformer (DETR)
на прикладі малярії.

Початкова ідея: self-supervised передтренування ViT-backbone на нерозмічених мазках крові
(MAE-реконструкція + контрастивний InfoNCE одночасно), далі цей backbone підключається до
DETR і порівнюється з базовими моделями на BBBC041v1 (*P. vivax*). Другий датасет — NIH-NLM
(*P. falciparum*), лише зовнішня оцінка без навчання.

## Основні результати

Infected AP50, середнє 3 seed; числа й обмеження — у `docs/thesis_log.md` і `reports/`.

| Модель | BBBC041 test | NIH zero-shot |
|---|---|---|
| Baseline A (COCO DETR-R50) | 0.550 | 0.072 |
| ViT-DETR, ImageNet | 0.567 | — |
| ViT-DETR, DINO | 0.620 | — |
| Двоетапна: бокси A + ознаки DINO | 0.680 | 0.423 |
| Двоетапна: бокси A + SSL v2 (b), 2 SSL-seed | 0.751 / 0.773 | 0.557 / 0.579 |

MAE + InfoNCE з нуля в DETR на test не допоміг (фаза 4: 0.329, 1 seed). DETR добре локалізує клітини
(class-agnostic AP50 ≈ 0.97), а під зсувом домену помиляється класифікація. Фази 5–7 є дослідницькими:
гіпотези сформульовано після перегляду test.

## Встановлення (Windows, conda)

```powershell
conda activate medthesis
pip install torch==2.14.0 torchvision==0.29.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt
pip install -e .
python scripts/check_env.py
```

## Дані

```powershell
python scripts/prepare_data.py --config configs/data.yaml
```

Завантажує BBBC041 у `data/raw/`, конвертує анотації в COCO (`data/processed/coco/`) та
створює спліти (`data/splits/`). Сирі дані ніколи не змінюються.

> Windows Smart App Control: scikit-learn не встановлювати (блокується його DLL і падає
> `import transformers`); `grpcio` закріплено на 1.76.0.

## Baseline A: тренування DETR

```powershell
conda activate medthesis
cd C:\Users\Alex\PycharmProjects\MedImage_Detection\malaria-ssl-detr

python scripts/train_detr.py --config configs/experiments/a_coco_detr_binary.yaml
python scripts/train_detr.py --config configs/experiments/a_coco_detr_multiclass.yaml
python scripts/train_detr.py --config configs/experiments/a_scratch_r50_binary.yaml
```

Продовжити перерваний запуск — та сама команда з `--resume`. Результати: `runs/<name>/`
(config, metrics.csv, TensorBoard, чекпоінти). Логи: `tensorboard --logdir runs`.

## Оцінювання

Спершу val (там обираються пороги F1 для кожного класу), потім test із тими самими порогами:

```powershell
python scripts/evaluate.py runs/a_coco_detr_binary --split val --bootstrap
python scripts/evaluate.py runs/a_coco_detr_binary --split test --bootstrap

python scripts/evaluate.py runs/a_coco_detr_multiclass --split val --bootstrap
python scripts/evaluate.py runs/a_coco_detr_multiclass --split test --bootstrap

python scripts/evaluate.py runs/a_scratch_r50_binary --split val --bootstrap
python scripts/evaluate.py runs/a_scratch_r50_binary --split test --bootstrap
```

`--bootstrap` без числа = 200 ресемплів (95% CI). Результати: `runs/<name>/metrics_<split>.json`,
`predictions_<split>.json`, `thresholds.json`. Далі перевиконати
`notebooks/02_baseline_analysis.ipynb`:

```powershell
jupyter nbconvert --to notebook --execute --inplace notebooks/02_baseline_analysis.ipynb
```

## Тести

```powershell
pytest -q
```

## Структура

```
configs/      YAML-конфіги
src/malaria_ssl/  код пакета (data/, ssl/, detr/, eval/, utils/)
scripts/      CLI-скрипти
notebooks/    ноутбуки (лише читають артефакти та будують графіки)
tests/        pytest
reports/      фігури, таблиці, звіти
docs/         журнал роботи (thesis_log.md)
app/          SmearScan: вебзастосунок для демонстрації моделей (див. app/README.md)
```
