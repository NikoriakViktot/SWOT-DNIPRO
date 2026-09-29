# Companion app for paper 1: report

## What was added

A Streamlit companion app (`app/`) that reads **only reproducible outputs**: the ms7 validation tables, the
frozen zone manifest, a light tracked data layer and the manuscript. No number is typed into the app.
If a summary selection does not match exactly one row, `lib.data.stat()` raises instead of showing a
different number.

### Pages

| Page | Contents |
|---|---|
| **Overview** | Title and abstract, parsed from the manuscript; an evidence-architecture diagram (gauges → ATL13 V1/V2, gauges → SWOT V3/V6, ATL13 ↔ SWOT V4); headline V1–V4 metrics and the slope/EGG2015 check; link buttons |
| **Study area & maps** | Folium map with switchable layers: zones R/F/D/E, pre-breach water, June 2023 flood envelope (event layer), ATL13 transects by period, the SWOT calibration-orbit reach, the Rozumivka/Kherson closure supports, gauges and yearbook posts (Rozumivka and Kherson highlighted), and the V4 crossings coloured by SWOT − ICESat-2. Also the zone table (area and hash), observation geometry by zone, and the gauge-record timeline |
| **Validation paths V1–V6** | One tab per path: its question, headline metrics, an interactive chart and the raw summary rows. V3 is presented as two independent closures, then their comparison. V4 carries the warning that raw r rests on one high crossing in R, the anomaly correlation and the time-gap sensitivity. V5 LakeSP is kept separate. V6 Kherson covers ICESat-2, RiverSP and PIXC |
| **Figures & tables** | The manuscript figures in text order with their captions, each marked **recomputed** or **carried from v5 — pending final rebuild**; the four tables with download buttons |
| **Notebooks & reproducibility** | The ms5–ms7d pipeline with script docstrings and run order; the notebooks with GitHub and nbviewer links; output folders |
| **Data & methods** | Data sources; the matching rules and independent units; 24 h vs 1–3 d vs 3–10 d, with numbers from the tables; bilinear vs nearest-cell EGG2015; why the validation figures were recomputed |

## Files

**New**
- `app/streamlit_app.py`: entry point, navigation, sidebar links.
- `app/lib/config.py`: every path and link; the production URL and branch live here.
- `app/lib/data.py`, `app/lib/charts.py`, `app/lib/maps.py`: cached loaders, plotly charts and the folium map.
- `app/views/{home,maps,validation,figures,reproducibility,methods}.py`
- `app/prepare_app_data.py` → `outputs/paper/app_data/`: about 1 MB of tracked layers plus `manifest.json`.
- `app/requirements.txt`
- `tests/test_app_pages.py`: every page runs without an exception (AppTest).

**Changed**
- `README.md`: a link block at the top (app, manuscript, notebooks, validation outputs, figures, zones) and a "how to launch locally" section.

## How to run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

To refresh the data after the analysis changes, run the pipeline in the full research environment:
ms5 → ms6 → ms7 → ms7b → ms7c → ms7d (must report 0 failures). Then run `python app/prepare_app_data.py`.

## Links

- **Repository**: `https://github.com/NikoriakViktot/SWOT-DNIPRO`. Every file link is built as `blob|tree/<BRANCH>/<path>` by `config.gh()`.
- **Streamlit URL**: `https://swot-dnipro.streamlit.app` is a **placeholder** (README and `config.APP_URL`).
- **Branch**: `BRANCH = "main"`. The app, `outputs/paper/` and the ms scripts currently exist only on
  `perf/phase20-bbox` and are uncommitted; `origin/main` is the slim initial commit. Until they are merged,
  the GitHub links resolve to 404.

## Pending

1. **Commit and merge.** `app/`, `outputs/paper/` (zones, validation, app_data, figures, manuscript) and
   `scripts/ms5…ms7d` must be committed and merged into the branch the links name. Decide whether the
   `.docx` files belong in git (v3 5 MB, v5 4.7 MB, v6 9.9 MB); the app does not need them.
2. **Deploy** to Streamlit Community Cloud with main file `app/streamlit_app.py` and requirements
   `app/requirements.txt`. Then replace `APP_URL` in `app/lib/config.py` and in the README.
3. **Eight figures carried from v5** (F02, F04, F15, F17, F22, F23, F25, F26) belong to sections the
   validation rebuild did not touch. The app labels them "carried from v5 — pending final rebuild".
4. **Text numbers not yet re-derived by ms7** (visible in the manuscript, not in the app):
   - the daily-orbit breach-fortnight correlations at the posts below the dam;
   - the withheld-gauge RMSE (4.0 cm);
   - the daily RiverSP recession correlation and the lag analysis;
   - the "product EGM2008 realisations" row of Table S2.
5. **Reviewer wording in the abstract and conclusions**: "an order of magnitude below the observed
   change", the priority claim, "established within weeks". The app shows the abstract as written.
6. **No paper-specific notebooks exist yet.** The app lists notebooks 10–15 (vertical chain, Kherson pilot)
   and points to the ms scripts as the reproducible path.
