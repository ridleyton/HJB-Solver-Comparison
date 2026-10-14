"""
Extract text from the PDE literature PDFs so they can be searched and compared.

Uses the Ghostscript that ships inside TeX Live (the machine has no poppler/pdftotext and no
Python PDF library, and `pip install` cannot reach the network here). Ghostscript's `txtwrite`
device is adequate for locating timing tables and venue statements, which is what this is for;
it is not a layout-preserving extractor.

Usage: python src/extract_lit.py [--src PDE] [--out literature/pde_text]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Ghostscript is NOT needed to reproduce any result in this paper -- it is used only to turn
# reference PDFs into text for a literature check, which is an author-side step. It is resolved at
# run time rather than hardcoded, because the literals that used to sit here published one machine's
# TeX installation layout and made the tool unusable anywhere else.
#
#   GS      the ghostscript executable: $GS_BIN, else gswin64c/gswin32c/gs on PATH.
#   GS_LIB  ghostscript's resource path: $GS_LIB, else inferred from the executable's location.
#           It is only needed for the bundled TeX Live build of ghostscript; a system ghostscript
#           finds its own resources and an empty value is correct for it.
def _find_gs() -> Path | None:
    explicit = os.environ.get("GS_BIN")
    if explicit:
        return Path(explicit)
    for name in ("gswin64c", "gswin32c", "gs"):
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def _gs_libs(gs: Path | None) -> str:
    explicit = os.environ.get("GS_LIB")
    if explicit:
        return explicit
    if gs is None:
        return ""
    # TeX Live ships ghostscript under <tl>/tlpkg/tlgs/, with resources in sibling directories.
    tlgs = gs.parent.parent
    cands = [tlgs / "Resource" / "Init", tlgs / "lib", tlgs / "kanji", tlgs / "Resource" / "Font"]
    present = [str(c) for c in cands if c.exists()]
    return os.pathsep.join(present)


GS = _find_gs()
GS_LIB = _gs_libs(GS)
if GS is None:
    print("note: ghostscript not found. Set GS_BIN, or install it, to run the literature "
          "extraction. No table or figure in this paper depends on this step.", file=sys.stderr)


def extract(pdf: Path, txt: Path) -> bool:
    txt.parent.mkdir(parents=True, exist_ok=True)
    env = {"GS_LIB": GS_LIB, "PATH": str(GS.parent)}
    try:
        subprocess.run(
            [str(GS), "-dNOPAUSE", "-dBATCH", "-dQUIET", "-sDEVICE=txtwrite",
             f"-sOutputFile={txt}", str(pdf)],
            capture_output=True, timeout=180, env=env, check=False,
        )
    except subprocess.TimeoutExpired:
        return False
    return txt.exists() and txt.stat().st_size > 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="PDE")
    ap.add_argument("--out", default="literature/pde_text")
    args = ap.parse_args()

    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(p for p in src.rglob("*.pdf"))
    print(f"found {len(pdfs)} PDFs under {src}/")

    ok = failed = 0
    manifest: list[str] = []
    for pdf in pdfs:
        rel = pdf.relative_to(src)
        # Flatten the path into a safe filename so the text is easy to grep across.
        safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(rel.with_suffix("")))
        safe = safe[:150]
        txt = out / f"{safe}.txt"
        if extract(pdf, txt):
            chars = txt.stat().st_size
            print(f"  ok    {chars:>7d} chars  {rel}")
            manifest.append(f"{safe}\t{rel}\t{chars}")
            ok += 1
        else:
            print(f"  FAIL              {rel}")
            manifest.append(f"{safe}\t{rel}\t0")
            failed += 1

    (out / "MANIFEST.tsv").write_text(
        "slug\tsource_relpath\tchars\n" + "\n".join(manifest) + "\n", encoding="utf-8")
    print(f"\nextracted {ok}, failed {failed}; manifest at {out}/MANIFEST.tsv")


if __name__ == "__main__":
    main()
