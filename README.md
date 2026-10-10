# malaria-ssl-detr

Магістерська робота: **«Класифікація та сегментація медичних зображень за допомогою
Detection Transformer (DETR)»**.

Ідея: self-supervised передтренування ViT-backbone на нерозмічених мазках крові
(MAE-реконструкція + контрастивний InfoNCE одночасно), далі цей backbone підключається до
DETR (трансформер ініціалізований з COCO, адаптація через LoRA / менший LR) і порівнюється з
базовими моделями на детекції малярії (BBBC041v1, *P. vivax*).

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
```
