"""
Measure how full each COLUMN of each compiled page is, from rendered page images.

Why this exists
---------------
The page count alone does not say whether a manuscript is 8 pages because it has too much text or
because floats are leaving columns half-empty. A whole-page metric cannot tell the two apart,
because a page whose left column is full and right column is empty still reports "full height".

This splits each page down the middle and reports each column's ink extent. The tallest column
seen anywhere is taken as a full text block, so the summary line reports how many page-equivalents
of CONTENT the document actually holds -- the number to compare against a page limit. If that is
below the limit but the PDF is over it, the problem is float placement, and cutting prose will not
fix it.

Requires page images produced by Ghostscript, e.g.
  gswin64c -sDEVICE=png16m -r70 -sOutputFile=.pagedump/p%02d.png main.pdf

Usage: python src/page_fill.py .pagedump
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg  # noqa: E402
import numpy as np  # noqa: E402

INK_THRESHOLD = 0.85


def column_extents(path: Path) -> tuple[float, float, int, int]:
    """Return (left fill, right fill, image height, image width) as fractions of image height."""
    img = mpimg.imread(path)
    if img.ndim == 3:
        img = img[:, :, :3]
    grey = img.mean(axis=2)
    ink = grey < INK_THRESHOLD
    h, w = ink.shape
    mid = w // 2

    def extent(sub: np.ndarray) -> float:
        rows = np.flatnonzero(sub.any(axis=1))
        if rows.size == 0:
            return 0.0
        return float(rows[-1] - rows[0] + 1) / h

    return extent(ink[:, :mid]), extent(ink[:, mid:]), h, w


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else ".pagedump")
    pngs = sorted(d.glob("*.png"))
    if not pngs:
        print(f"no page images in {d}/")
        return 1

    rows = [(p.name, *column_extents(p)) for p in pngs]
    full = max(max(l, r) for _, l, r, _, _ in rows)

    print(f"{'page':<10}{'left col':>10}{'right col':>11}   note")
    print("-" * 62)
    total = 0.0
    for name, left, right, _, _ in rows:
        total += left + right
        short = []
        for label, v in (("left", left), ("right", right)):
            if v < 0.90 * full:
                short.append(f"{label} {v / full:.0%}")
        note = f"short: {', '.join(short)}" if short else "full"
        print(f"{name:<10}{left / full:>10.0%}{right / full:>11.0%}   {note}")
    print("-" * 62)
    # Two columns per page.
    print(f"full column height      : {full:.1%} of page image")
    print(f"content in page-equivalents: {total / (2 * full):.2f}")
    print(f"PDF pages                  : {len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
