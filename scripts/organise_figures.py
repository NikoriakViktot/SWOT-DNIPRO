#!/usr/bin/env python
"""Sort outputs/figures/ into <campaign>/<format>/ and repair report links.

outputs/figures/ had accumulated 218 flat files from every campaign at once.
Figures are grouped by the campaign prefix already used throughout the repo
(see CLAUDE.md, "scripts/ naming convention"), then by file format.

Windows "Zone.Identifier" alternate-data-stream stubs left by the migration
are moved to _zone_identifier_junk/ rather than deleted, so the move stays
reversible.

Idempotent: running it twice is a no-op. Nothing reads outputs/figures/
programmatically (scripts read outputs/figure_data/), so this is safe; the
markdown reports that LINK to figures are rewritten to the new paths.
"""
from __future__ import annotations

import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
REPORTS = ROOT / "outputs" / "reports"

# (regex on the filename, destination campaign folder)
RULES = [
    (r"^S?Fig\d+", "publication"),
    (r"^Fig[A-G](?![a-z])", "publication"),
    (r"^(EN|UK)_", "presentation"),
    (r"^hist\d*", "historical_bathymetry"),
    (r"^HIST\d+", "historical_bathymetry"),
    (r"^H\d+", "historical_bathymetry"),
    (r"^V\d+", "validation_series"),
    (r"^K\d+", "icesat2_kriging"),
    (r"^P\d+", "phase19_20"),
    (r"^QA\d+", "audit"),
    (r"^M\d+", "manuscript"),
    # one-off names that nonetheless belong to an existing chain
    (r"^channel_LOO_RGT", "icesat2_kriging"),
    (r"^lag_sensitivity", "icesat2_kriging"),
    (r"^(Q_vs_slope|dS_dQ|dniprohes|DniproHES)", "hydrology"),
]


def campaign_for(name: str) -> str:
    for pattern, folder in RULES:
        if re.match(pattern, name):
            return folder
    return "misc"


def main() -> None:
    apply = "--apply" in sys.argv
    if not FIG.is_dir():
        raise SystemExit(f"not found: {FIG}")

    moves, junk = [], []
    for p in sorted(FIG.iterdir()):
        if p.is_dir():
            continue
        if p.name.endswith("Zone.Identifier"):
            junk.append(p)
            continue
        ext = p.suffix.lower().lstrip(".")
        if not ext:
            continue
        dest = FIG / campaign_for(p.name) / ext / p.name
        moves.append((p, dest))

    by_campaign: dict[str, int] = {}
    for _, d in moves:
        by_campaign[d.parent.parent.name] = by_campaign.get(d.parent.parent.name, 0) + 1
    print(f"{len(moves)} figures -> {len(by_campaign)} campaign folders"
          f"{'' if apply else '   (dry run; pass --apply)'}")
    for c in sorted(by_campaign, key=lambda k: -by_campaign[k]):
        print(f"  {c:<26}{by_campaign[c]:>4}")
    print(f"  {'_zone_identifier_junk':<26}{len(junk):>4}")

    if not apply:
        print("\nnothing written")
        return

    for src, dest in moves:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    if junk:
        q = FIG / "_zone_identifier_junk"
        q.mkdir(parents=True, exist_ok=True)
        for p in junk:
            shutil.move(str(p), str(q / p.name))

    # ---- repair markdown links -------------------------------------------
    index = {d.name: d.as_posix() for _, d in moves}
    changed = []
    for md in sorted(REPORTS.glob("*.md")):
        text = original = md.read_text()
        for name, newrel in index.items():
            text = text.replace(f"outputs/figures/{name}", newrel)
        # the {png,pdf} shorthand some reports use
        for name, newrel in index.items():
            if not name.endswith(".png"):
                continue
            stem = name[:-4]
            old = f"outputs/figures/{stem}.{{png,pdf}}"
            if old in text:
                base = Path(newrel).parent.parent.as_posix()
                text = text.replace(old, f"{base}/{{png,pdf}}/{stem}.{{png,pdf}}")
        if text != original:
            md.write_text(text)
            changed.append(md.name)
    print(f"\nmoved {len(moves)} figures, quarantined {len(junk)} junk stubs")
    print(f"rewrote links in {len(changed)} report(s): {', '.join(changed) or '-'}")


if __name__ == "__main__":
    main()
