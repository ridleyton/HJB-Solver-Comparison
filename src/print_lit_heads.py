"""Print the title-page head of selected extracted papers, to read their own venue banner.

The automated venue scan was unreliable because it matched conference names inside citation
lists. This prints the first lines of each file instead, where a paper states its own venue.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

WANT = [
    "transolver", "GINO", "gino", "RNO", "gpwno", "hamlet", "diffPDE", "diverse_no",
    "fno", "cno", "neural_operator", "unoi", "insr", "irregular-fno", "3d-geoca",
    "3D-egno", "3d-egno", "Hierarchical_Neural_Operator", "fnin", "aeroGTO",
    "NeurIPS-2023-convolutional", "multipole", "DeepONet", "pde-refiner", "teng", "tnngp",
    "PDENNEVAL", "Localized_Integral", "U_grid", "difffno", "SRS-Net",
]


def clean(s: str) -> str:
    for a, b in (("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb00", "ff"), ("\u2019", "'"),
                 ("\u201c", '"'), ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "--")):
        s = s.replace(a, b)
    return re.sub(r"[^\x09\x0a\x20-\x7e]", " ", s)


def main() -> None:
    text_dir = Path("literature/pde_text")
    files = sorted(text_dir.glob("*.txt"))
    printed = 0
    for f in files:
        if f.name == "MANIFEST.tsv":
            continue
        if not any(w.lower() in f.stem.lower() for w in WANT):
            continue
        raw = f.read_text(encoding="utf-8", errors="replace")
        lines = [ln.strip() for ln in raw.split("\n") if ln.strip()][:14]
        head = clean(" | ".join(lines))[:420]
        print(f"### {f.stem[:88]}")
        print(f"    {head}\n")
        printed += 1
    print(f"printed {printed} heads")


if __name__ == "__main__":
    main()
