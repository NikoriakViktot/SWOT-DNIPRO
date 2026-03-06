# swot_kakhovka — SWOT Satellite Data Downloader

Автоматичне завантаження SWOT даних (NetCDF) через NASA CMR/Earthdata з Bearer Token
для Каховського та Дніпровського водосховищ і нижнього Дніпра.

---

## Швидкий старт

```bash
# 1. Клонувати / перейти у директорію проєкту
cd C:\Users\victo\PycharmProjects\SWOT-DNIPRO   # Windows
# або
cd /mnt/c/Users/victo/PycharmProjects/SWOT-DNIPRO  # WSL

# 2. Створити venv і встановити пакет
python -m venv .venv

# Windows (PowerShell):
.venv\Scripts\Activate.ps1

# Linux / WSL:
source .venv/bin/activate

pip install -U pip
pip install -e .                        # встановлює swotdl CLI
pip install -e ".[notebook]"            # + залежності для Jupyter notebook
```

---

## Налаштування токену Earthdata

> **ПОПЕРЕДЖЕННЯ:** Ніколи не комітьте токен у Git. `.env` знаходиться у `.gitignore`.

1. Отримайте Bearer Token на [urs.earthdata.nasa.gov](https://urs.earthdata.nasa.gov/users/<your_username>/user_tokens)
2. Скопіюйте `.env.example` → `.env`:

```bash
cp .env.example .env
```

3. Відредагуйте `.env`:

```
EARTHDATA_TOKEN=eyJ0...ваш_реальний_токен...
```

Або встановіть змінну середовища напряму:

```bash
# Linux / WSL:
export EARTHDATA_TOKEN="eyJ0..."

# Windows PowerShell:
$env:EARTHDATA_TOKEN="eyJ0..."
```

---

## Використання CLI

### Формат команди

```bash
python -m swotdl download \
    --aoi <path/to/aoi.geojson> \
    --product <CMR_SHORT_NAME> \
    --start YYYY-MM-DD \
    --end   YYYY-MM-DD
```

### Приклади для трьох AOI

**1. Каховське водосховище — рівень поверхні озер (LakeSP)**
```bash
python -m swotdl download \
    --aoi config/aoi/kakhovka_reservoir.geojson \
    --product SWOT_L2_HR_LakeSP_2.0 \
    --start 2022-12-01 --end 2026-03-05
```

**2. Дніпровське водосховище — LakeSP**
```bash
python -m swotdl download \
    --aoi config/aoi/dniprovske_reservoir.geojson \
    --product SWOT_L2_HR_LakeSP_2.0 \
    --start 2022-12-01 --end 2026-03-05
```

**3. Нижній Дніпро до моря — RiverSP**
```bash
python -m swotdl download \
    --aoi config/aoi/lower_dnipro_to_sea.geojson \
    --product SWOT_L2_HR_RiverSP_2.0 \
    --start 2022-12-01 --end 2026-03-05
```

**Dry-run (пошук без скачування):**
```bash
python -m swotdl download \
    --aoi config/aoi/kakhovka_reservoir.geojson \
    --product SWOT_L2_HR_LakeSP_2.0 \
    --dry-run
```

**Запустити все одразу:**
```bash
bash scripts/run_download.sh          # Linux / WSL
.\scripts\run_download.ps1            # Windows PowerShell
```

---

## SWOT продукти — що коли використовувати

| Short Name | Опис | Коли використовувати |
|---|---|---|
| `SWOT_L2_HR_LakeSP_2.0` | Lake Single Pass — площа, рівень поверхні, об'єм водойм | **Водосховища** (Каховка, Дніпровське) — агреговані характеристики озера як цілого |
| `SWOT_L2_HR_RiverSP_2.0` | River Single Pass — рівень, ширина, ухил, витрата для річкових відрізків | **Річкові ділянки** (нижній Дніпро до моря), аналіз витрат, паводки |
| `SWOT_L2_HR_PIXC` | Pixel Cloud — сирі водні пікселі з WSE | Детальний просторовий аналіз, власна класифікація; більший об'єм даних |

> **Примітка:** SWOT запущений **грудень 2022**. Дані до цієї дати відсутні.
> Руйнування Каховської дамби — **6 червня 2023** — чітко видне у часових рядах WSE.

---

## Структура виводу

```
data/
├── raw/
│   ├── swot_l2_hr_lakesp/<YYYY>/<MM>/*.nc
│   ├── swot_l2_hr_riversp/<YYYY>/<MM>/*.nc
│   └── swot_l2_hr_pixc/<YYYY>/<MM>/*.nc
├── index/
│   ├── granules.parquet    ← повний індекс гранул (оновлюється після кожного запуску)
│   └── granules.csv        ← те ж саме, зручний для перегляду
└── logs/
    └── run.log
```

---

## Jupyter Notebook (аналіз і візуалізація)

```bash
pip install -e ".[notebook]"
jupyter notebook notebooks/swot_analysis.ipynb
```

Notebook містить:
- Огляд завантажених гранул
- Інспекція структури NetCDF (LakeSP / RiverSP)
- Часові ряди WSE для Каховки
- Завантаження та аналіз даних гідропостів (`гідропости_каховка.xlsx`, `гідропости_дніпровське.xlsx`)
- Порівняння SWOT WSE з гідропостами (scatter, RMSE, MAE, r)
- Інтерактивна карта Folium
- Аналіз руйнування дамби (до/після червня 2023)

---

## Тести

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Безпека

- Токен читається **виключно** з `EARTHDATA_TOKEN` env var або `.env` файлу
- Токен **ніколи** не логується
- `.env` знаходиться у `.gitignore` — переконайтесь, що не комітите його
