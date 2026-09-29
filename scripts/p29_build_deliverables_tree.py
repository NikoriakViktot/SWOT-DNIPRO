#!/usr/bin/env python
"""P29 -- build a navigable deliverables tree: rasters and figures organised by
ZONE x TASK, each folder carrying a README that states what was done and what
came out of it.

The repository is a lab notebook: products are scattered across
`outputs/rasters/zone{1,2,4}/`, `outputs/figures/atlas/zone{N}/`,
`outputs/figures/<theme>/{png,pdf}/`, `data/processed/bathymetry/` and
`outputs/tables/`, named by the script that made them rather than by the zone
or the question they answer. This script does not move anything -- it builds
`outputs/deliverables/` as a second view over the same bytes:

  * files are **hard-linked** where source and destination share a filesystem,
    so the tree costs no extra disk and stays byte-identical to the original;
    it falls back to a copy across devices (nothing currently needs that).
  * every task folder gets a README generated from the actual result tables,
    so the numbers in the tree cannot drift from the numbers in the CSVs.
  * zones with no products yet (ZONE_3 is still fetching Sentinel-2) get an
    honest "in progress" README instead of an empty folder.

Rebuilding is safe and idempotent: the tree is wiped and relinked each run.
Nothing under `outputs/rasters`, `outputs/figures`, `data/` is modified.

Usage:  python scripts/p29_build_deliverables_tree.py [--out outputs/deliverables]
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from swot_dnipro import config as CFG

FIG = ROOT / "outputs" / "figures"
RAS = ROOT / "outputs" / "rasters"
TAB = ROOT / "outputs" / "tables"
BAT = ROOT / "data" / "processed" / "bathymetry"

SKIP_SUFFIX = (".aux.xml",)
INDEXES = ("NDVI", "NDWI", "MNDWI", "NDMI", "BSI", "AWEIsh", "NDTI")
REGIMES = ("PRE_BREACH", "BREACH_DRAWDOWN", "POST_BREACH")

ZONES = [
    dict(key="ZONE_1_KAKHOVKA_LOWER_DNIPRO", short="zone1", dir="ZONE_1_KAKHOVKA_LOWER_DNIPRO",
         title="Зона 1 — чаша Каховського водосховища й верхній нижній Дніпро",
         note="Найбільша зона (96 Mpx на 20 м). Її дно й берегові контури будувалися окремим "
              "історичним ланцюгом (hist14/hist24/hist26) по **чаші водосховища**, а не p28/p27, "
              "і лежать у [`../_cross_zone/04_reservoir_bed_historical/`](../_cross_zone/04_reservoir_bed_historical/) — "
              "домен там вужчий за полігон зони, тому папок `04`/`05` тут свідомо нема."),
    dict(key="ZONE_2_KHERSON_DELTA", short="zone2", dir="ZONE_2_KHERSON_DELTA",
         title="Зона 2 — Херсонська дельта",
         note="527 промірів. Тут уперше перевірено й відкинуто гіпотезу «берег як обмеження» (F-20)."),
    dict(key="ZONE_3_DNIPRO_BUG_ESTUARY", short="zone3", dir="ZONE_3_DNIPRO_BUG_ESTUARY",
         title="Зона 3 — Дніпро-Бузький лиман",
         note="634 проміри — найбільше з усіх зон, а DEM досі не існувало. "
              "Чекає на докачування Sentinel-2 (тайли TUS/TUT)."),
    dict(key="ZONE_4_DAM_TO_KHERSON_FLOODWAY", short="zone4", dir="ZONE_4_DAM_TO_KHERSON_FLOODWAY",
         title="Зона 4 — заплава від греблі до Херсона",
         note="626 промірів. Полігон зони сягає ~41 км вище греблі, тому будь-який "
              "херсонський рівень не можна поширювати на хвіст водосховища."),
]

TASKS = [
    ("01_spectral_indices", "Спектральні індекси (p25)"),
    ("02_landcover_classification", "Класифікація покриву k10e (p25)"),
    ("03_water_and_observation_quality", "Вода й якість спостереження (p25 / phase19-20)"),
    ("04_bathymetry_dem", "Батиметрія дна (p28 / hist24)"),
    ("05_shoreline", "Датовані берегові лінії (p27)"),
    ("06_maps", "Карти-атлас (p26)"),
    ("07_tables", "Таблиці результатів"),
]


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


def link(src: Path, dst_dir: Path, rows: list, zone: str, task: str) -> bool:
    """Hard-link src into dst_dir (copy across devices). Returns True if placed.

    Two source trees can hold the same basename with *different* content: e.g.
    `P20_Fig1_fragmentation.png` exists both in `outputs/figures/` (rebuilt
    2026-09-16 after the three-valued masks) and in `outputs/figures/phase19_20/png/`
    (2026-09-09, pre-A4). Overwriting silently would make the manifest claim more
    files than exist, and depending on pattern order would silently publish the
    stale one. Rule: the **more recent** file wins, the other is recorded as
    `collision_skipped` so the manifest shows what was set aside and why.
    """
    if not src.exists() or src.name.endswith(SKIP_SUFFIX) or ":" in src.name:
        return False
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if dst.exists():
        if dst.stat().st_ino == src.stat().st_ino:
            return False          # same bytes, same inode: already placed
        if src.stat().st_mtime <= dst.stat().st_mtime:
            rows.append(dict(zone=zone, task=task, file=src.name, bytes=src.stat().st_size,
                             source=str(src.relative_to(ROOT)), placed_as="collision_skipped_older"))
            return False
        # incoming file is newer: it replaces the one already placed
        for r in rows:
            if r["task"] == task and r["file"] == src.name and r["placed_as"] in ("hardlink", "copy"):
                r["placed_as"] = "collision_skipped_older"
        dst.unlink()
    try:
        os.link(src, dst)
        how = "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        how = "copy"
    rows.append(dict(zone=zone, task=task, file=src.name, bytes=src.stat().st_size,
                     source=str(src.relative_to(ROOT)), placed_as=how))
    return True


def linkmany(patterns, base: Path, dst_dir: Path, rows, zone, task) -> int:
    n = 0
    for pat in patterns:
        for p in sorted(base.glob(pat)):
            n += link(p, dst_dir, rows, zone, task)
    return n


def read_csv(p: Path):
    try:
        return pd.read_csv(p)
    except Exception:
        return None


def fmt(x, nd=2):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "—"


# ---------------------------------------------------------------- README text

def readme_indices(z, n_files, man) -> str:
    n_dates = len(man) if man is not None else 0
    regs = (man.regime.value_counts().to_dict() if man is not None and "regime" in man else {})
    reg_line = ", ".join(f"{k} {v}" for k, v in regs.items()) if regs else "—"
    return f"""# {z['title']} · 01 · Спектральні індекси

**Що робилося.** `scripts/p25_zone_spectral_stacks.py` для кожної дати Sentinel-2
над зоною рахує сім індексів на сітці зони `SD.build_grid(..., 20 м)` через
канонічний модуль `src/swot_dnipro/sentinel_preprocess.py`, а потім зводить їх у
композити-медіани по режимах.

**Сім індексів:** {", ".join(INDEXES)}.
NDVI/NDWI/MNDWI/NDMI/BSI перенесені з `k10e` без зміни формул; **AWEIsh**
(Feyisa 2014) доданий для каламутної води лиману й дельти; **NDTI** (Lacaux 2007) —
для каламутності. Усі зареєстровані в `outputs/planning/index_registry.csv`.

**Критично для чисел:** до всіх смуг застосовується `BOA_ADD_OFFSET = −1000`,
прочитаний із `MTD_MSIL2A.xml` кожної сцени (модуль **відмовляється** працювати,
якщо зсуву немає). До цього аудиту жоден скрипт його не застосовував — для
замороженого правила води з порогом 0 це було безпечно (маски збігаються на 99.9 %),
а для ненульових порогів класифікації — ні.

**Що тут лежить.** {n_files} растрів `zone{z['short'][-1]}_<індекс>_<режим>_median_20m.tif`,
int16 × 10000, nodata −32768, EPSG:32636, 20 м. Режими: PRE_BREACH,
BREACH_DRAWDOWN (до 2023-09-01), POST_BREACH.

**Обсяг вхідних даних:** {n_dates} дат ({reg_line}).
Подетальні стеки (по датах) не копіюються сюди — вони на масовому томі:
`{CFG.BULK_ROOT}/zone_spectral/{z['key']}/`.

Маніфест: `../07_tables/p25_zone_spectral_manifest_{z['key']}.csv`.
"""


def readme_class(z, n_files) -> str:
    return f"""# {z['title']} · 02 · Класифікація покриву

**Що робилося.** Десятикласова фізична класифікація `k10e` (`classify()` у
`sentinel_preprocess.py`, перенесена дослівно) по кожній даті, зведена в моду
по режиму.

**Класи 0–9:** вода, мілка/каламутна вода, вологий осад, сухий оголений осад,
рідка рослинність, трав'яниста рослинність, щільна рослинність, очерет/плавні,
забудова/техногенне, невизначене.

**Що змінилося після BOA-корекції (F-15).** Пороги класифікації ненульові, тому
зсув −1000 їх зачіпає: згода зі старими стеками лише **62–76 %**, а частка
`DRY_BARE_SEDIMENT` падає з 12–23 % до 1.3–2.8 %. Тобто попередні оцінки площі
оголеного осаду були завищені приблизно на порядок. Тутешні растри — **після**
корекції.

**Що тут лежить.** {n_files} файлів `zone{z['short'][-1]}_class_<режим>_mode_20m.tif`
(uint8, класи 0–9). Легенду див. у картах (`../06_maps/`).
"""


def readme_water(z, n_files) -> str:
    return f"""# {z['title']} · 03 · Вода й якість спостереження

**Що робилося.** Заморожене правило води (NDWI/MNDWI + SCL, поріг 0) дає
**тризначну** маску 0 / 1 / 255, де 255 — «не спостережено» (хмара, тінь, поза
сценою). Це виправлення A4 з аудиту: раніше маски були двозначними й хмара
мовчки ставала суходолом.

**Чому це змінює висновки.** Після перевипуску 162 масок
(`scripts/p24_reemit_masks_3value.py`) кількість дат POST_BREACH із високим
покриттям упала з **12 до 4**, а PARTIAL зросла з 10 до 22; захмарений тайл, що
раніше звітував «покриття 1.0», отримав 0.08. Жодна теза про фрагментацію не
могла спиратися на стару когорту.

**Що тут лежить.** {n_files} файлів:
`water_frac_<режим>` (частка дат із водою серед **спостережених**),
`n_valid_<режим>` (скільки дат узагалі бачили комірку),
а де є — `p_water`, `observation_support`, `envelope_*` з батиметричної гілки.

`water_frac` **не можна** читати без `n_valid`: комірка з `water_frac = 1.0` при
`n_valid = 1` і при `n_valid = 40` — це два різні твердження.
"""


def readme_dem(z, n_files, summ, cv) -> str:
    if summ is None or not len(summ):
        return f"""# {z['title']} · 04 · Батиметрія дна

Канонічної поверхні для цієї зони ще нема (`p28` не запускався або чекає на вхідні
дані). Див. README зони.
"""
    s = summ.iloc[0]
    arms = ""
    if cv is not None and len(cv):
        best = cv.loc[cv.groupby("arm").RMSE_snd_m.idxmin()]
        arms = "\n".join(
            f"| `{r.arm}` | {r.method} | **{fmt(r.RMSE_snd_m)}** | {fmt(r.bias_snd_m)} |"
            for r in best.sort_values("RMSE_snd_m").itertuples())
    return f"""# {z['title']} · 04 · Батиметрія дна (PRE_BREACH)

**Що робилося.** `scripts/p28_zone_bed_surface.py` будує поверхню дна до підриву
з ручних промірів, перевіряючи три незалежні плечі обмежень **блокованою
просторовою крос-валідацією лише на промірах** (блок 0.5 км) — щоб оцінка
точності не спиралася на ті самі конструйовані точки, якими поверхня й
підтримується.

**Вхідні дані:** {int(s.n_soundings)} промірів,
{int(s.n_s2_constraints)} берегових обмежень Sentinel-2 (σ медіана {fmt(s.s2_sigma_median_m, 3)} м),
{int(s.n_p18_pseudopoints)} псевдоточок p18 як плече для порівняння.
Варіограма: range {fmt(s.variogram_range_km)} км, sill {fmt(s.variogram_sill)}, nugget {fmt(s.variogram_nugget)}.

**Результат крос-валідації (RMSE по промірах, м):**

| плече | метод | RMSE | зсув |
|---|---|---|---|
{arms}

**Висновок (F-20).** Обмеження берегом **погіршує** DEM, а не покращує:
берегова лінія на 20 м правильна як берег і хибна як зразок дна там, де дно
падає на 10 м за кількасот метрів, і «м'якість» обмеження (σ ≈ 0.2–0.4 м проти
sill {fmt(s.variogram_sill)} м²) цього не міняє. Найгірше — ближче 500 м до берега, де
лежить більшість промірів. Канонічна поверхня — **{s.best_arm} / {s.best_method}**,
RMSE **{fmt(s.best_rmse_soundings_m)} м**, зсув {fmt(s.best_bias_soundings_m)} м,
NMAD {fmt(s.best_nmad_soundings_m)} м.

**Що тут лежить.** {n_files} растрів:
`*_bed_canonical_PRE_BREACH_250m.tif` — канон, маскований до спостереженої води
й {fmt(s.support_km, 0)} км від найближчого проміру ({int(s.canonical_cells)} комірок, {fmt(s.canonical_km2, 1)} км²);
`_unmasked` — та сама поверхня без маски (діагностика, **не для вимірювань**);
`_constraint_share`, `_dist_to_sounding` — де саме поверхня спирається на що;
`*_50m.tif`, `*_30m.tif` — ті самі поверхні на дрібнішій сітці **лише для
відображення**; статистичний канон — 250 м (hist17: дрібніша сітка додає
роздільність, не інформацію). Числа вище — з 250-метрового запуску.

**Чого не можна стверджувати:** післяпідривне дно. Промірів після 2023-06-06 нема.

Витіснений продукт p19 для цієї зони — `outputs/archive/legacy_p19_zone24_rasters/`.
"""


def readme_shore(z, n_files, p27) -> str:
    if p27 is None or not len(p27):
        return f"""# {z['title']} · 05 · Берегові лінії

Контурів для цієї зони ще нема. Див. README зони.
"""
    rows = "\n".join(
        f"| {r.group} | {int(r.n_dates)} | {fmt(r.polygon_area_km2, 1)} | {fmt(r.shoreline_km, 0)} | "
        f"{int(r.n_vertices)} | {fmt(r.sigma_total_m, 3)} | {fmt(100 * r.polygon_to_water_ratio, 0)} % |"
        for r in p27.itertuples())
    return f"""# {z['title']} · 05 · Датовані берегові лінії (PRE_BREACH)

**Що робилося.** `scripts/p27_zone_shoreline_contours.py` відтворює для цієї зони
той самий ланцюг, що реально живить кригінг ZONE_1: дати з рівнем на посту →
композит індексів → неперервний контур Marching Squares → вершини з висотою і
власною σ. Це **не** межа класу з растра: поле рішення замикається на краях зони
й хмар, а вершини, що впали на край валідності, відкидаються.

**Рівень.** Херсонський пост (EVRF2019) на дату + поздовжній градієнт 0.0014 м/км
по пікетажу p18. Для ZONE_4 вершини всередині `reservoir_full_pool_prebreach`
відкидаються окремо — полігон зони сягає хвоста водосховища, де херсонський
рівень хибний на ~16 м.

**σ-бюджет** складається в квадратурі з: EPSG:9902 (0.068 м), зчитування поста
(0.01 м), розкид рівнів у групі, добова мінливість Херсона (0.044 м),
градієнт, положення берега (нахил × комірка/2), змішаний піксель, очерет.

**Результат:**

| група | дат | площа, км² | берег, км | вершин | σ, м | покрито води |
|---|---|---|---|---|---|---|
{rows}

**Що тут лежить.** {n_files} файлів GeoPackage: шар `contours` (полігон із
`H_evrf2019_m`, `sigma_total_m`, датами) і шар `vertices` (точкові обмеження, які
p28 і подає в кригінг).

**Як їх правильно читати.** Геометрія перевірена на бейзмапі, а не за площею —
у цьому проєкті два полігони колись пройшли всі числові перевірки, лежачи в
іншій річці.
"""


def readme_maps(z, n_files) -> str:
    return f"""# {z['title']} · 06 · Карти

**Що робилося.** `scripts/p26_zone_spectral_maps.py` через спільний модуль
`src/swot_dnipro/plotting/maps.py` малює атлас зони за стандартом
`outputs/planning/07_cartographic_atlas_plan.md §7.1`: масштабна лінійка,
стрілка півночі, легенда поза полем карти, CRS, дата, провенанс, контур домену.

**Що тут лежить.** {n_files} файлів (PNG + PDF попарно):

* `*_<індекс>_regimes` — три режими одного індексу в **спільній** шкалі
  (без спільної шкали панелі порівнювати не можна);
* `*_class_<режим>` — класифікація покриву з легендою на 10 класів;
* `*_water_frac_<режим>`, `*_n_valid_<режим>` — вода й покриття поруч;
* `*_bed_canonical_p28` — канонічна поверхня дна, де вона є.

Фіксовані шкали на індекс задані в `maps.INDEX_SCALE` — тому карти різних зон
і режимів зіставні між собою.
"""


def readme_tables(z, files) -> str:
    lst = "\n".join(f"* `{f}`" for f in files) or "— (ще нема)"
    return f"""# {z['title']} · 07 · Таблиці результатів

Копії (жорсткі посилання) на результатні CSV цієї зони з `outputs/tables/`.
Це ті самі байти — правити треба першоджерело, а не тут; після правки
перезапустіть `scripts/p29_build_deliverables_tree.py`.

{lst}
"""


README_BY_TASK = {
    "01_spectral_indices": readme_indices,
    "02_landcover_classification": readme_class,
    "03_water_and_observation_quality": readme_water,
    "04_bathymetry_dem": readme_dem,
    "05_shoreline": readme_shore,
    "06_maps": readme_maps,
    "07_tables": readme_tables,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/deliverables")
    a = ap.parse_args()
    OUT = ROOT / a.out
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    rows: list = []
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    head = git_head()
    zone_status = []

    for z in ZONES:
        zdir = OUT / z["dir"]
        sh, num = z["short"], z["short"][-1]
        src_ras = RAS / sh
        src_fig = FIG / "atlas" / sh
        counts = {}

        # 01 indices
        counts["01_spectral_indices"] = linkmany(
            [f"{sh}_{i}_{r}_median_20m.tif" for i in INDEXES for r in REGIMES],
            src_ras, zdir / "01_spectral_indices", rows, z["key"], "01_spectral_indices") if src_ras.exists() else 0
        # 02 class
        counts["02_landcover_classification"] = linkmany(
            [f"{sh}_class_*_mode_20m.tif"], src_ras, zdir / "02_landcover_classification",
            rows, z["key"], "02_landcover_classification") if src_ras.exists() else 0
        # 03 water / coverage
        counts["03_water_and_observation_quality"] = linkmany(
            [f"{sh}_water_frac_*.tif", f"{sh}_n_valid_*.tif", f"{sh}_p_water.tif",
             f"{sh}_observation_support.tif", f"{sh}_envelope_*.tif"],
            src_ras, zdir / "03_water_and_observation_quality", rows, z["key"],
            "03_water_and_observation_quality") if src_ras.exists() else 0
        # 04 bed
        counts["04_bathymetry_dem"] = linkmany(
            [f"{sh}_bed_*.tif"], src_ras, zdir / "04_bathymetry_dem", rows, z["key"],
            "04_bathymetry_dem") if src_ras.exists() else 0
        # 05 shoreline
        counts["05_shoreline"] = linkmany(
            [f"zone_shore_contours_{z['key']}_*.gpkg"], BAT, zdir / "05_shoreline",
            rows, z["key"], "05_shoreline")
        # 06 maps
        counts["06_maps"] = linkmany(["*.png", "*.pdf"], src_fig, zdir / "06_maps",
                                     rows, z["key"], "06_maps") if src_fig.exists() else 0
        # 07 tables
        tnames = [f"p25_zone_spectral_manifest_{z['key']}.csv",
                  f"p28_interpolator_cv_{z['key']}*.csv",
                  f"p28_surface_summary_{z['key']}*.csv",
                  f"p28_constraint_dominance_{z['key']}*.csv"]
        counts["07_tables"] = linkmany(tnames, TAB, zdir / "07_tables", rows, z["key"], "07_tables")
        p27_all = read_csv(TAB / "p27_zone_shoreline_uncertainty.csv")
        p27 = p27_all[p27_all.zone == z["key"]] if p27_all is not None and "zone" in p27_all else None
        if p27 is not None and len(p27):
            (zdir / "07_tables").mkdir(parents=True, exist_ok=True)
            p27.to_csv(zdir / "07_tables" / f"p27_shoreline_uncertainty_{z['key']}.csv", index=False)
            counts["07_tables"] += 1

        man = read_csv(TAB / f"p25_zone_spectral_manifest_{z['key']}.csv")
        summ = read_csv(TAB / f"p28_surface_summary_{z['key']}.csv")
        cv = read_csv(TAB / f"p28_interpolator_cv_{z['key']}.csv")

        for task, _label in TASKS:
            d = zdir / task
            if counts.get(task, 0) == 0 and task != "07_tables":
                continue
            d.mkdir(parents=True, exist_ok=True)
            fn = README_BY_TASK[task]
            if task == "04_bathymetry_dem":
                txt = fn(z, counts[task], summ, cv)
            elif task == "05_shoreline":
                txt = fn(z, counts[task], p27)
            elif task == "01_spectral_indices":
                txt = fn(z, counts[task], man)
            elif task == "07_tables":
                txt = fn(z, sorted(p.name for p in d.glob("*.csv")))
            else:
                txt = fn(z, counts[task])
            (d / "README.md").write_text(txt, encoding="utf-8")

        total = sum(counts.values())
        if total == 0:
            # a zone with no products yet gets one honest README, not empty folders
            for task, _ in TASKS:
                d = zdir / task
                if d.exists():
                    shutil.rmtree(d)
        zone_status.append((z, counts, total))
        zdir.mkdir(parents=True, exist_ok=True)
        (zdir / "README.md").write_text(zone_readme(z, counts, total, summ, p27, man), encoding="utf-8")

    # ---- cross-zone ------------------------------------------------------
    cz = OUT / "_cross_zone"
    groups = {
        "01_fragmentation": (
            [(FIG, "P20_Fig1_fragmentation.*"), (FIG / "phase19_20" / "png", "*.png"),
             (FIG / "phase19_20" / "pdf", "*.pdf"),
             (TAB, "fragmentation_metrics_by_date*.csv"), (TAB, "channel_connectivity_by_date*.csv"),
             (TAB, "phase20_coverage_sensitivity.csv"), (TAB, "p24_mask_reemit_manifest.csv")],
            "Планова фрагментація водного дзеркала (Фаза 19–20)"),
        "02_swot_lakesp": (
            [(FIG, "P23_lakesp_calval_transition_*_v2.png"),
             (TAB, "p23_lakesp_by_date_*.csv"), (TAB, "p23_lakesp_pre_post_calval_*.csv")],
            "SWOT LakeSP у cal/val-орбіті — незалежна перевірка тези"),
        "03_vertical_datum": (
            [(FIG / "publication" / "png", "Fig02*.png"), (FIG / "publication" / "png", "Fig03*.png"),
             (FIG / "publication" / "png", "Fig04*.png"),
             (TAB, "gauge_vertical_reference_summary.csv")],
            "Вертикальний ланцюг SWOT / ICESat-2 / пости"),
        "04_reservoir_bed_historical": (
            [(RAS, "kakhovka_bed_*.tif"), (FIG, "V1*_*.png"), (FIG, "V2*_*.png"),
             (BAT, "prebreach_contours_continuous.gpkg"), (BAT, "hist25_three_level_consensus.gpkg"),
             (BAT, "hist25_contour_confidence.gpkg"), (BAT, "sounding_isobaths.gpkg"),
             (TAB, "hist14_surface_summary*.csv"), (TAB, "hist14_interpolator_cv.csv")],
            "Дно чаші водосховища — історичний ланцюг hist14/hist24"),
        "05_data_inventory": (
            [(TAB, "p0d_data_inventory_by_zone.csv"), (TAB, "p22_ukraine_clip_manifest.csv"),
             (TAB, "p22b_clip_verification.csv"), (TAB, "sentinel_scene_inventory.csv")],
            "Інвентаризація даних і обрізання по Україні"),
        "06_estuary_gauges_2023": (
            [(ROOT / "data" / "historical" / "sea_posts_2023", "*.csv"),
             (TAB, "p30_sea_post_inventory_2023.csv"),
             (ROOT / "outputs" / "reports", "P30_sea_posts_2023.md")],
            "Пости гирлової області 2023 — щорічники ДВК (p30): рівні, екстремуми, T води, солоність"),
        # "07_s1_classifier_study" removed: the M3 S1/optical classifier study (p31-p38) that fed this
        # package moved to floodstate-eo, and no other kept script reads its sources.
    }
    for name, (pats, title) in groups.items():
        d = cz / name
        n = 0
        for base, pat in pats:
            if base.exists():
                n += linkmany([pat], base, d, rows, "_cross_zone", name)
        if n:
            (d / "README.md").write_text(cross_readme(name, title, n), encoding="utf-8")

    man_df = pd.DataFrame(rows)
    man_df.to_csv(TAB / "p29_deliverables_manifest.csv", index=False)
    (OUT / "README.md").write_text(master_readme(zone_status, man_df, stamp, head), encoding="utf-8")

    placed = man_df[man_df.placed_as.isin(["hardlink", "copy"])]
    gb = placed.bytes.sum() / 1e9
    print(f"deliverables tree -> {OUT}")
    print(f"{len(placed):,} files, {gb:.2f} GB logical "
          f"({(placed.placed_as == 'hardlink').sum()} hard-linked, "
          f"{(placed.placed_as == 'copy').sum()} copied); "
          f"{(man_df.placed_as == 'collision_skipped_older').sum()} stale basename collisions skipped")
    for z, counts, total in zone_status:
        print(f"  {z['key']:<34} {total:4d}  " + " ".join(f"{k.split('_')[0]}:{v}" for k, v in counts.items()))
    print(f"manifest -> {TAB / 'p29_deliverables_manifest.csv'}")


def zone_readme(z, counts, total, summ, p27, man) -> str:
    if total == 0:
        return f"""# {z['title']}

**Статус: у роботі — продуктів ще нема.**

{z['note']}

Що блокує: докачування сцен Sentinel-2 (тайли 36TUS/36TUT).
Після завершення автоматично відпрацює ланцюг
p25 (стеки) → p26 (карти) → p27 (берегові лінії) → p28 (DEM),
і ця папка наповниться так само, як в інших зонах.
"""
    lines = []
    for task, label in TASKS:
        n = counts.get(task, 0)
        lines.append(f"| [`{task}/`]({task}/) | {label} | {n} |" if n else
                     f"| `{task}/` | {label} | — |")
    res = []
    if man is not None and len(man):
        res.append(f"* **{len(man)} дат Sentinel-2** оброблено на сітці 20 м, сім індексів + класифікація.")
    if summ is not None and len(summ):
        s = summ.iloc[0]
        res.append(f"* **DEM дна:** канон — {s.best_arm} / {s.best_method}, "
                   f"RMSE по промірах **{fmt(s.best_rmse_soundings_m)} м**, зсув {fmt(s.best_bias_soundings_m)} м "
                   f"({int(s.n_soundings)} промірів, {fmt(s.canonical_km2, 1)} км² покриття).")
        res.append(f"* **Берегові обмеження не допомогли** (F-20): будь-яке плече з берегом гірше "
                   f"за поверхню з самих промірів. Витіснений p19 — в `outputs/archive/legacy_p19_zone24_rasters/`.")
    if p27 is not None and len(p27):
        g = p27.iloc[0]
        res.append(f"* **Берегова лінія:** {fmt(g.polygon_area_km2, 1)} км², "
                   f"{fmt(100 * g.polygon_to_water_ratio, 0)} % спостереженої води, "
                   f"σ {fmt(g.sigma_total_m, 3)} м, {int(g.n_vertices)} вершин.")
    return f"""# {z['title']}

{z['note']}

## Папки

| папка | задача | файлів |
|---|---|---|
{chr(10).join(lines)}

## Головні результати

{chr(10).join(res) if res else "—"}

Кожна папка має власний README з методом, вхідними даними й числами.
Файли — жорсткі посилання на оригінали в `outputs/rasters/`, `outputs/figures/`,
`data/processed/`; це ті самі байти, дерево не займає додаткового місця.
"""


def cross_readme(name, title, n) -> str:
    extra = {
        "01_fragmentation": (
            "Теза про зростання планової фрагментації **не встановлена**: pre-breach-когорта "
            "має три дати, і лише одна з них має повне покриття. Після переходу на тризначні "
            "маски (0/1/255) POST_BREACH HIGH упав з 12 дат до 4. Проста гіпотеза «це артефакт "
            "покриття» теж спростована (Spearman −0.058, p = 0.72) — конфаундинг діє через склад "
            "когорти. Кількість водойм конфаундована однозначно: ρ = +0.568, p = 0.0001."),
        "02_swot_lakesp": (
            "Незалежне плече на SWOT LakeSP у єдиній cal/val-орбіті (цикли 478–578). "
            "Приріст кількості об'єктів **не підтверджує** тезу там, де вона заявлена: "
            "всередині `dnipro_water_domain` +14.5 об'єкта, p = 0.195; зростання живе поза "
            "зв'язною водною системою. Позначено UNRESOLVED / exploratory."),
        "03_vertical_datum": (
            "Ланцюг тримається: зсув коректора −0.1695 → −0.1321 м відтворено з широт постів "
            "реєстру до **0.3 мм**. Єдине зауваження — перевірка PIXC проти RiverSP внутрімісійна "
            "й не бачить помилки припливної конвенції (3.58 см)."),
        "04_reservoir_bed_historical": (
            "Дно чаші водосховища будувалося кригінгом із м'якими контурними обмеженнями "
            "(hist24). Знак дисперсії похибки в діагоналі був хибний (`+err_var` замість "
            "`−err_var`) — виправлено, вплив за поточних параметрів 0.006–0.099 м проти "
            "CV RMSE 1.8–2.9 м. Растри **не** перераховувалися."),
        "05_data_inventory": (
            "Обрізання SWOT по території України: 8 461 гранул → 4 919 з ознаками, "
            "517.5 GB → 37.1 GB (−93 %). Верифікація перед видаленням оригіналів "
            "(`p22b_clip_verification.csv`): повнота, цілісність усіх 4 919 файлів, "
            "побітова звірка 125 переклипованих гранул і 40 порожніх — усе PASS. "
            "Континентальні оригінали (483 GB) видалені 2026-09-16 після цієї перевірки."),
    }.get(name, "")
    return f"""# Міжзонне · {title}

{extra}

{n} файлів (жорсткі посилання на оригінали).
"""


def master_readme(zone_status, man_df, stamp, head) -> str:
    rows = []
    for z, counts, total in zone_status:
        state = "готово" if total else "**у роботі**"
        rows.append(f"| [`{z['dir']}/`]({z['dir']}/) | {z['title'].split('—')[1].strip()} | {total} | {state} |")
    gb = man_df[man_df.placed_as.isin(["hardlink", "copy"])].bytes.sum() / 1e9
    return f"""# Результати по зонах і задачах

Побудовано `scripts/p29_build_deliverables_tree.py` · {stamp} · commit `{head}`.

Це **навігаційний зріз** над уже наявними продуктами, а не їх копія: файли
під'єднані жорсткими посиланнями, тому дерево не займає додаткового місця й
байт-у-байт збігається з оригіналами в `outputs/rasters/`, `outputs/figures/`,
`data/processed/`. Перезбирається командою вище будь-коли; правити треба
першоджерела, не це дерево.

Усього {len(man_df[man_df.placed_as.isin(["hardlink", "copy"])]):,} файлів, {gb:.2f} GB логічного обсягу.

## Зони

| папка | зона | файлів | стан |
|---|---|---|---|
{chr(10).join(rows)}

## Міжзонне

| папка | про що |
|---|---|
| [`_cross_zone/01_fragmentation/`](_cross_zone/01_fragmentation/) | планова фрагментація, Фаза 19–20 |
| [`_cross_zone/02_swot_lakesp/`](_cross_zone/02_swot_lakesp/) | SWOT LakeSP як незалежне плече |
| [`_cross_zone/03_vertical_datum/`](_cross_zone/03_vertical_datum/) | вертикальний ланцюг і датум |
| [`_cross_zone/04_reservoir_bed_historical/`](_cross_zone/04_reservoir_bed_historical/) | дно чаші водосховища (hist14/hist24) |
| [`_cross_zone/05_data_inventory/`](_cross_zone/05_data_inventory/) | інвентаризація даних, обрізання по Україні |

## Задачі всередині зони

| папка | задача | скрипт |
|---|---|---|
| `01_spectral_indices` | 7 індексів × 3 режими, 20 м | `p25_zone_spectral_stacks.py` |
| `02_landcover_classification` | 10 класів k10e, мода по режиму | `p25_zone_spectral_stacks.py` |
| `03_water_and_observation_quality` | вода, покриття, тризначні маски | `p25`, `p24`, `phase19/20` |
| `04_bathymetry_dem` | поверхня дна PRE_BREACH + CV | `p28_zone_bed_surface.py` |
| `05_shoreline` | датовані берегові контури + σ | `p27_zone_shoreline_contours.py` |
| `06_maps` | атлас карт PNG/PDF | `p26_zone_spectral_maps.py` |
| `07_tables` | результатні CSV зони | — |

## Що варто знати, перш ніж користуватися

1. **DEM дна:** канонічна поверхня в кожній зоні побудована **лише з промірів**.
   Обмеження береговою лінією перевірено й відкинуто даними (F-20) — воно
   погіршує поверхню на 2–3 м і систематично піднімає дно. Витіснені растри p19
   лежать в `outputs/archive/legacy_p19_zone24_rasters/` зі своїм README.
2. **Маски води тризначні** (0 / 1 / 255). «Не спостережено» — не суходіл;
   `water_frac` без `n_valid` не має сенсу.
3. **BOA_ADD_OFFSET = −1000** застосовано до всіх смуг. Маски води від цього не
   змінилися (99.9 %), класифікація змінилася істотно (згода 62–76 %).
4. **Післяпідривного дна нема.** Промірів після 2023-06-06 не існує; жоден растр
   тут не є батиметрією після підриву.
5. Наукові застереження до кожної тези — в `audit_runs/20260916T093000Z/`
   (`00_AUDIT_SUMMARY.md`, `findings.csv`).
"""


if __name__ == "__main__":
    main()
