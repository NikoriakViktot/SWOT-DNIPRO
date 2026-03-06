#!/usr/bin/env bash
# Run SWOT downloads for all three AOI areas.
# Usage: bash scripts/run_download.sh
# Requires EARTHDATA_TOKEN to be set in .env or environment.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

if [[ -f .env ]]; then
    export $(grep -v '^#' .env | xargs)
fi

if [[ -z "${EARTHDATA_TOKEN:-}" ]]; then
    echo "ERROR: EARTHDATA_TOKEN is not set. Copy .env.example → .env and fill in your token."
    exit 1
fi

echo "=== Kakhovka Reservoir: LakeSP ==="
python -m swotdl download \
    --aoi config/aoi/kakhovka_reservoir.geojson \
    --product SWOT_L2_HR_LakeSP_2.0 \
    --start 2022-12-01 --end 2026-03-05

echo "=== Kakhovka Reservoir: RiverSP ==="
python -m swotdl download \
    --aoi config/aoi/kakhovka_reservoir.geojson \
    --product SWOT_L2_HR_RiverSP_2.0 \
    --start 2022-12-01 --end 2026-03-05

echo "=== Dniprovske Reservoir: LakeSP ==="
python -m swotdl download \
    --aoi config/aoi/dniprovske_reservoir.geojson \
    --product SWOT_L2_HR_LakeSP_2.0 \
    --start 2022-12-01 --end 2026-03-05

echo "=== Lower Dnipro → Sea: RiverSP ==="
python -m swotdl download \
    --aoi config/aoi/lower_dnipro_to_sea.geojson \
    --product SWOT_L2_HR_RiverSP_2.0 \
    --start 2022-12-01 --end 2026-03-05

echo "=== All downloads complete. Index: data/index/granules.parquet ==="
