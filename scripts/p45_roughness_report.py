#!/usr/bin/env python
"""P45 -- report for plan 15: indices coverage (p40), below-dam floodplain (p42), Manning states (p43), HEC-RAS package (p44).

Reads only tables written by those scripts; every number in the report traces to a CSV. Writes
outputs/reports/P43_manning_roughness_states.md (Ukrainian, identifiers original).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import pandas as pd
import yaml

from swot_dnipro import config as CFG

Y = yaml.safe_load((ROOT / "config/roughness_classes.yaml").read_text())
T = CFG.TABLES
ZONES = {"ZONE_1_KAKHOVKA_LOWER_DNIPRO": "ZONE_1", "ZONE_2_KHERSON_DELTA": "ZONE_2", "ZONE_3_DNIPRO_BUG_ESTUARY": "ZONE_3", "ZONE_4_DAM_TO_KHERSON_FLOODWAY": "ZONE_4"}


def md(df: pd.DataFrame, cols=None, r=3) -> str:
    d = df if cols is None else df[cols]
    d = d.round(r)
    return "| " + " | ".join(d.columns) + " |\n|" + "---|" * len(d.columns) + "\n" + "\n".join("| " + " | ".join(str(v) for v in row) + " |" for row in d.values.tolist())


def main() -> None:
    out = ["# P43 — Коефіцієнт Маннінга за станами, зона затоплення нижче греблі, індекси по всіх зонах і роках\n",
           f"План `outputs/planning/15_*`; журнал `15_PROGRESS.md`. Побудовано `scripts/p45_roughness_report.py` з таблиць p40/p42/p43/p44. Класи → n: `config/roughness_classes.yaml` (версія {Y['version']}); n — лише літературні prior'и через класи, ніколи з індексу.\n"]
    # --- p40 coverage
    out.append("## 1. Індекси/класи/SCL по всіх зонах і роках (p40)\n\nВікно leaf‑on (VI–IX); `obs≥1` — частка зони, спостережена хоча б раз; `≥2` — ≥ 2 датами; частки класів — медіани по датах від спостереженої площі.\n")
    rows = []
    for z, zn in ZONES.items():
        p = T / f"p40_window_summary_{z}.csv"
        if not p.exists():
            rows.append(dict(zone=zn, year="—", n_dates="p40 не запущено")); continue
        s = pd.read_csv(p); s = s[s.window == "leafon"]
        for r in s.itertuples():
            rows.append(dict(zone=zn, year=int(r.year), n_dates=int(r.n_dates), obs_ge1=r.inside_observed_share, obs_ge2=r.inside_ge2_share, water=r.share_water_1_2_median, sediment=r.share_sediment_3_4_median, veg=r.share_veg_5_6_7_median, scl=r.have_scl))
    out.append(md(pd.DataFrame(rows)) + "\n")
    # --- p42
    out.append("## 2. Зона затоплення нижче греблі (p42)\n")
    if (T / "p42_domain_provenance.csv").exists():
        out.append(md(pd.read_csv(T / "p42_domain_provenance.csv")) + "\n\nЧутливість h₀ (покриття стійких клітинок паводку ≥ 2 подій):\n\n" + md(pd.read_csv(T / "p42_hand_sensitivity.csv")) + "\n")
        out.append("Домен зареєстровано як `below_dam_floodplain` (RESOLVED). Паводок червня 2023 — растри `$BULK_ROOT/floodplain/<ZONE>/` (11 подій S1, `flood_n_events`, `flood_max_envelope`, `flood_first/last_seen_idx`, `hand_m`).\n")
    # --- p43
    for dom in ("pool", "below_dam_floodplain"):
        p = T / f"p43_class_areas_{dom}.csv"
        out.append(f"## 3.{1 if dom == 'pool' else 2}. Класи шорсткості і n_base — `{dom}` (p43)\n")
        if not p.exists():
            out.append("_ще не розраховано_\n"); continue
        a = pd.read_csv(p)
        keep = ["grid", "state", "valid_km2", "n_base_area_weighted"] + [c for c in a.columns if c.endswith("_km2") and c not in ("valid_km2",) and a[c].sum() > 0]
        out.append(md(a[keep], r=3) + "\n")
        if dom == "pool" and (T / "p43_bed_recession_zoning.csv").exists():
            out.append("Зонування дна за рецесією 2023 (BREACH_2023_bed; пороги 06‑30 / 08‑06 / 09‑08):\n\n" + md(pd.read_csv(T / "p43_bed_recession_zoning.csv")) + "\n")
    if (T / "p43_roughness_transition_summary.csv").exists():
        out.append("## 4. Переходи станів і Δn_base\n\n" + md(pd.read_csv(T / "p43_roughness_transition_summary.csv"), r=4) + "\n")
    # --- p44
    sc = ROOT / "outputs/hecras/scenarios.csv"
    if sc.exists():
        s = pd.read_csv(sc)
        out.append(f"## 5. Пакет HEC‑RAS (p44)\n\n{len(s)} сценаріїв у `outputs/hecras/scenarios.csv` (по станах × N_LOW/N_BASE/N_HIGH, dam‑break ×0.8/×1.2/×2 біля греблі, парні прогони); дизайн чутливості T1–T9 у `sensitivity_design.csv`; шари `outputs/hecras/<domain>/<grid>_<state>/landcover_<state>.tif` + таблиці n (+ winter). HEC‑RAS не запускався.\n")
    out.append("## 6. Межі\n\n- Класи з правила S2 + слабких міток DW/WorldCover, без розміченої валідації (аудит F‑15/F‑29; вибірка 320 точок чекає розмітки).\n- n — prior'и з літератури (`roughness_classes.yaml`, джерела SRC‑01…SRC‑16 аудиту), без гідравлічного калібрування.\n- Стани з малим покриттям (одна одно‑тайлова дата) позначені `valid_km2`; читати разом із `n_valid`.\n- `young_woody` поза чашею у 2023 частково — міжрічний шум мітки DW.\n")
    (ROOT / "outputs/reports/P43_manning_roughness_states.md").write_text("\n".join(out))
    print("-> outputs/reports/P43_manning_roughness_states.md")


if __name__ == "__main__":
    main()
