#!/usr/bin/env python
"""MS11 -- the manuscript .docx, built from the audited Markdown.

The .docx of v6 was produced by pandoc outside the pipeline, so it silently fell behind the
Markdown that ms7d audits. This script makes the conversion a pipeline step: it refuses to run
unless the audit stamp (build_info.json) names the current Markdown by its hash, converts with
the pandoc shipped in pypandoc_binary, takes the page and paragraph styles from the previous
.docx (--reference-doc), resolves `figures/...` against outputs/paper, and writes the version
into the file name and the document's keywords property.

    python scripts/ms11_build_manuscript_docx.py            # -> outputs/paper/paper1_manuscript_en-<version>.docx

The .docx is a release asset (not tracked in git).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import tempfile
import zipfile

import pypandoc

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "outputs/paper"
MD = PAPER / "paper1_manuscript_en-v6.md"
BUILD = PAPER / "validation/build_info.json"
REFERENCE = PAPER / "paper1_manuscript_en-v6.docx"     # styles of the previous release


def clean_reference(tmpdir: Path) -> Path:
    """pandoc's default reference.docx with only styles.xml taken from the previous release:
    using the old .docx itself as the reference would also carry its 33 images into the new file."""
    default = tmpdir / "default.docx"
    default.write_bytes(_default_reference())
    out = tmpdir / "reference.docx"
    with zipfile.ZipFile(REFERENCE) as old:
        styles = old.read("word/styles.xml")
    with zipfile.ZipFile(default) as src, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as dst:
        for item in src.infolist():
            dst.writestr(item, styles if item.filename == "word/styles.xml" else src.read(item.filename))
    return out


def _default_reference() -> bytes:
    import subprocess
    return subprocess.check_output([pypandoc.get_pandoc_path(), "--print-default-data-file", "reference.docx"])


def main() -> None:
    info = json.loads(BUILD.read_text())
    sha = hashlib.sha256(MD.read_bytes()).hexdigest()[:16]
    if info.get("manuscript_sha256") != sha or info.get("audit") != "passed":
        sys.exit("the Markdown changed since the last audit: run scripts/ms7d_manuscript_audit.py first")
    version = info["manuscript_version"]                 # e.g. v6.2-rc2
    out = PAPER / f"paper1_manuscript_en-{version}.docx"
    # the Markdown carries its own title heading; the version goes into the file name and the
    # document's keywords property, not into a second title block
    args = ["--resource-path", str(PAPER), f"--metadata=keywords:manuscript {version}, md {sha}"]
    with tempfile.TemporaryDirectory() as td:
        if REFERENCE.exists():
            args.append(f"--reference-doc={clean_reference(Path(td))}")
        pypandoc.convert_file(str(MD), "docx", format="markdown+pipe_tables+tex_math_dollars",
                              outputfile=str(out), extra_args=args)
    print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1e6:.1f} MB  ({version}, md {sha})")


if __name__ == "__main__":
    main()
