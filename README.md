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

## Структура

```
configs/      YAML-конфіги
src/malaria_ssl/  код пакета (data/, ssl/, detr/, eval/, utils/)
scripts/      CLI-скрипти
notebooks/    ноутбуки (лише читають артефакти та будують графіки)
tests/        pytest
reports/      фігури, таблиці, звіти
```
