# Run SWOT downloads for all three AOI areas (Windows PowerShell).
# Usage: .\scripts\run_download.ps1
# Requires EARTHDATA_TOKEN set in .env or as env variable.

$ErrorActionPreference = "Stop"
$ROOT = Split-Path $PSScriptRoot -Parent
Set-Location $ROOT

# Load .env if present
if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        if ($_ -match "^\s*([^#][^=]+)=(.+)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
}

if (-not $env:EARTHDATA_TOKEN) {
    Write-Error "EARTHDATA_TOKEN is not set. Copy .env.example to .env and fill in your token."
    exit 1
}

Write-Host "=== Kakhovka Reservoir: LakeSP ===" -ForegroundColor Cyan
python -m swotdl download --aoi config/aoi/kakhovka_reservoir.geojson --product SWOT_L2_HR_LakeSP_2.0 --start 2022-12-01 --end 2026-03-05

Write-Host "=== Kakhovka Reservoir: RiverSP ===" -ForegroundColor Cyan
python -m swotdl download --aoi config/aoi/kakhovka_reservoir.geojson --product SWOT_L2_HR_RiverSP_2.0 --start 2022-12-01 --end 2026-03-05

Write-Host "=== Dniprovske Reservoir: LakeSP ===" -ForegroundColor Cyan
python -m swotdl download --aoi config/aoi/dniprovske_reservoir.geojson --product SWOT_L2_HR_LakeSP_2.0 --start 2022-12-01 --end 2026-03-05

Write-Host "=== Lower Dnipro to Sea: RiverSP ===" -ForegroundColor Cyan
python -m swotdl download --aoi config/aoi/lower_dnipro_to_sea.geojson --product SWOT_L2_HR_RiverSP_2.0 --start 2022-12-01 --end 2026-03-05

Write-Host "=== Done. Index: data/index/granules.parquet ===" -ForegroundColor Green
