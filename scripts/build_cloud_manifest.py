#!/usr/bin/env python
"""Build data/catalog/cloud_archive_manifest.parquet — the Drive<->local ledger.

One row per granule we hold or intend to acquire. Cloud fields stay empty until an
upload is actually verified; nothing is marked uploaded on optimism.
"""
from __future__ import annotations
import hashlib, sys, warnings
from pathlib import Path
warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import pandas as pd
from swot_dnipro import config as CFG

CATALOG = CFG.ROOT / "data" / "catalog"
DRIVE = {  # verified folder ids created 2026-09-07
    "root": "1-PAzxhLvmRBKpJ5wQJi_BMGxgC-E--0Y",
    "raw": "1F27rt_JyMDrqwgAF732_jkLAAEdB5T_e",
    "SWOT_PIXC": "1RGDKHX3CsPSbRSPWzb3_3Lh9s0kVUbu0",
    "SWOT_RiverSP": "1_NQBf1gGdeWXzj75O8QZWyBx4JYC5mFl",
    "SWOT_LakeSP": "1i5puqOhC5Uuy55ZGeAs19tyGM13V080A",
    "SWOT_Raster": "1VsZWvgLALu9eVH1pIpN3WjIiku4kbJSI",
    "ICESat2_optional": "1uzPNH4Qgm7eHHj2dfng3baGYY3iuXaRz",
    "Sentinel": "1qP5K5pdaNWuYVw8ZtN0dlC6zl0izsoJH",
    "Meteo": "1PMwb4iKwnuq78RrNFPHWapkO3dw_J7oL",
    "Geodesy": "1tiXBlHz7NS-M_btZBObGNkQWaEf8U2Y7",
    "metadata": "1lSbshMBJlVmaRD6icZ8EftLBrY1HCWBY",
    "manifests": "1EDBe2RDh5voIdrKcMZFT3wu-_Rsj0dC_",
    "checksums": "1ggtUsiPUY9wEl_DkP95TyFYmZhDybuBQ",
    "CMR_catalogs": "1GGxM7qFmrhGFpillxV2RbUfuE7ooeF-D",
    "provenance": "1s_raqVYNOXnDN61KbvBNoqSf0HOREvnN",
}
PRODUCT_DIR = {"SWOT_L2_HR_PIXC_2.0": "SWOT_PIXC", "SWOT_L2_HR_RiverSP_2.0": "SWOT_RiverSP",
               "SWOT_L2_HR_LakeSP_2.0": "SWOT_LakeSP", "SWOT_L2_HR_Raster_2.0": "SWOT_Raster"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> None:
    disc = pd.read_parquet(CATALOG / "granules_discovered.parquet")
    local = {p.name: p for p in CFG.PIXC_DIR.glob("*.nc")}
    rows = []
    for r in disc.itertuples():
        p = local.get(r.granule_id)
        rows.append({
            "granule_id": r.granule_id, "product": r.short_name, "version": "2.0",
            "cycle": r.cycle, "pass": getattr(r, "_6", ""), "tile": r.tile,
            "date": r.date, "regime": r.regime, "aoi": r.aoi,
            "source_url": ("https://archive.swot.podaac.earthdata.nasa.gov/"
                           f"podaac-swot-ops-cumulus-protected/{r.short_name}/{r.granule_id}"),
            "size_bytes": p.stat().st_size if p else None,
            "est_size_bytes": int(r.typical_mb * 1024 * 1024),
            "sha256": sha256(p) if p else "",
            "google_drive_file_id": "", "google_drive_folder_id":
                DRIVE.get(PRODUCT_DIR.get(r.short_name, ""), ""),
            "google_drive_path": f"swot_data/raw/{PRODUCT_DIR.get(r.short_name,'')}/{r.granule_id}",
            "local_cache_path": str(p) if p else "",
            "download_status": "DOWNLOADED" if p else "PENDING",
            "upload_status": "NOT_UPLOADED",
            "processing_status": "USED_IN_ANALYSIS" if p else "NOT_PROCESSED",
        })
    df = pd.DataFrame(rows)
    CATALOG.mkdir(parents=True, exist_ok=True)
    df.to_parquet(CATALOG / "cloud_archive_manifest.parquet", index=False)
    df.to_csv(CATALOG / "cloud_archive_manifest.csv", index=False)
    print(f"manifest rows: {len(df):,}")
    print(df.groupby(["product", "download_status"]).size().to_string())
    print(f"\ndownloaded & hashed: {(df.download_status=='DOWNLOADED').sum()}")
    print(f"pending:             {(df.download_status=='PENDING').sum()}")
    print(f"uploaded to Drive:   {(df.upload_status!='NOT_UPLOADED').sum()}")
    tot = df.est_size_bytes.sum() / 1e9
    print(f"\nfull archive if acquired: {tot:.0f} GB")
    # compact summary for the Drive smoke test
    s = (df.groupby(["product", "regime"])
           .agg(granules=("granule_id", "size"),
                est_GB=("est_size_bytes", lambda x: round(x.sum() / 1e9, 1)),
                held=("download_status", lambda x: int((x == "DOWNLOADED").sum())))
           .reset_index())
    s.to_csv(CATALOG / "cloud_archive_summary.csv", index=False)
    print("\n" + s.to_string(index=False))


if __name__ == "__main__":
    main()
