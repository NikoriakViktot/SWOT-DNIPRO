"""CLI entry point: `swotdl download` or `python -m swotdl download`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env before anything reads env vars
load_dotenv()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="swotdl",
        description="SWOT satellite data downloader (NASA CMR + Earthdata Bearer Token).",
    )
    p.add_argument(
        "--config",
        default="config/pipeline.yaml",
        metavar="PATH",
        help="Path to pipeline.yaml (default: config/pipeline.yaml)",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # ── download ──────────────────────────────────────────────────────────────
    dl = sub.add_parser("download", help="Search CMR and download SWOT granules.")
    dl.add_argument(
        "--aoi",
        required=True,
        metavar="GEOJSON",
        help="Path to AOI GeoJSON (Polygon or MultiPolygon, EPSG:4326).",
    )
    dl.add_argument(
        "--aoi-name",
        default=None,
        metavar="NAME",
        help="Label for AOI in the index (default: stem of GeoJSON filename).",
    )
    dl.add_argument(
        "--product",
        required=True,
        metavar="SHORT_NAME",
        help="CMR short_name, e.g. SWOT_L2_HR_RiverSP",
    )
    dl.add_argument(
        "--start",
        default=None,
        metavar="YYYY-MM-DD",
        help="Start date (overrides config defaults.start).",
    )
    dl.add_argument(
        "--end",
        default=None,
        metavar="YYYY-MM-DD",
        help="End date (overrides config defaults.end).",
    )
    dl.add_argument(
        "--dry-run",
        action="store_true",
        help="Search CMR and build index without downloading files.",
    )
    dl.add_argument(
        "--no-aoi-filter",
        action="store_true",
        help="Disable precise AOI polygon intersection filter (bbox-only).",
    )
    dl.add_argument(
        "--granule-type",
        default=None,
        metavar="TYPE",
        help="Filter LakeSP/RiverSP sub-type: Obs | Prior | Unassigned (default: all).",
    )

    # ── list-products ─────────────────────────────────────────────────────────
    sub.add_parser("list-products", help="Print default products from config and exit.")

    return p


def cmd_download(args: argparse.Namespace) -> int:
    from swotdl.settings import load_config
    from swotdl.aoi import load_aoi, aoi_to_bbox, aoi_geometry
    from swotdl.cmr import cmr_search_all
    from swotdl.download import download_many
    from swotdl.index import write_index
    from swotdl.logging_config import setup_logging

    cfg = load_config(Path(args.config))
    log = setup_logging(cfg.paths.log_dir)

    aoi_path = Path(args.aoi)
    log.info("Loading AOI: %s", aoi_path)
    gdf = load_aoi(aoi_path)
    bbox = aoi_to_bbox(gdf)
    aoi_geom = None if args.no_aoi_filter else aoi_geometry(gdf)

    start = args.start or cfg.defaults.start
    end = args.end or cfg.defaults.end
    aoi_name = args.aoi_name or aoi_path.stem

    log.info(
        "Searching CMR: product=%s  bbox=%.3f,%.3f,%.3f,%.3f  %s → %s",
        args.product, *bbox, start, end,
    )
    granule_type = getattr(args, "granule_type", None)
    if granule_type:
        log.info("Granule type filter: %s", granule_type)

    granules = cmr_search_all(
        base_url=cfg.cmr.base_url,
        short_name=args.product,
        bbox=bbox,
        start=start,
        end=end,
        page_size=cfg.cmr.page_size,
        aoi_geom=aoi_geom,
        granule_type=granule_type,
    )
    log.info("Found %d granules", len(granules))

    downloads = []
    if not args.dry_run:
        downloads = download_many(cfg, granules, product=args.product)
    else:
        log.info("Dry-run: skipping downloads.")

    write_index(cfg, granules, product=args.product, aoi_name=aoi_name, bbox=bbox, downloads=downloads)
    return 0


def cmd_list_products(args: argparse.Namespace) -> int:
    from swotdl.settings import load_config

    cfg = load_config(Path(args.config))
    print("Default products in config:")
    for p in cfg.defaults.products:
        print(f"  {p}")
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.cmd == "download":
        return cmd_download(args)
    if args.cmd == "list-products":
        return cmd_list_products(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
