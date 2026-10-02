#!/usr/bin/env bash
# Phase 19 end-to-end. Assumes scripts/fetch_sentinel.py has completed.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/niko/repo/SWOT-DNIPRO/.venv/bin/python

echo "=== 1. raw-scene manifest (fetch_sentinel.py already ran to completion) ==="
# re-run scripts/fetch_sentinel.py first only if scenes are still missing/failed
$PY scripts/phase19_manifest.py

echo "=== 2. water masks (+ sensitivity) ==="
$PY scripts/phase19_watermasks.py

echo "=== 3. classify ATL13 water observations ==="
$PY scripts/phase19_classify.py

echo "=== 4. profiles, regime stats, residual water ==="
$PY scripts/phase19_profiles.py

echo "=== 5. figures ==="
$PY scripts/phase19_figures.py

echo "=== 6. report ==="
$PY scripts/phase19_report.py

echo "=== 7. feasibility census ==="
$PY scripts/phase19_feasibility.py

echo "=== 8. baseline vs expanded comparison ==="
$PY scripts/phase19_compare.py

echo "=== DONE ==="
