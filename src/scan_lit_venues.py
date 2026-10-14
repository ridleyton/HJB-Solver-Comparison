"""
Identify the publication venue of each extracted PDE paper.

The extracted text of a paper almost always names its venue in the first page or two -- in a
copyright line, a "Proceedings of ..." banner, a running header, or the arXiv stamp. This scans
the head of each file for those markers so the set can be partitioned into peer-reviewed
conference papers (and which are CCF-A) versus preprints.

It is a heuristic and it is reported as one: a paper that only says "arXiv:2406.11045" is a
preprint, and this script will say so rather than promote it.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Venue markers, roughly in order of how definitive they are.
VENUES: list[tuple[str, str, bool]] = [
    # (regex, canonical name, is_ccf_a)
    (r"advances in neural information processing systems|\bNeurIPS\b|NIPS\b", "NeurIPS", True),
    (r"international conference on machine learning|\bICML\b", "ICML", True),
    (r"international conference on learning representations|\bICLR\b", "ICLR", True),
    (r"conference on computer vision and pattern recognition|\bCVPR\b", "CVPR", True),
    (r"international conference on computer vision|\bICCV\b", "ICCV", True),
    (r"european conference on computer vision|\bECCV\b", "ECCV", True),
    (r"\bAAAI\b|association for the advancement of artificial intelligence", "AAAI", True),
    (r"international joint conference on artificial intelligence|\bIJCAI\b", "IJCAI", True),
    (r"\bKDD\b|knowledge discovery and data mining", "KDD", True),
    (r"\bSIGGRAPH\b", "SIGGRAPH", True),
    (r"annual meeting of the association for computational linguistics|\bACL\b", "ACL", True),
    (r"\bICRA\b|international conference on robotics and automation", "ICRA", True),
    (r"journal of machine learning research|\bJMLR\b", "JMLR", False),
    (r"nature machine intelligence", "Nature Mach. Intell.", False),
    (r"journal of computational physics", "J. Comput. Phys.", False),
    (r"computer methods in applied mechanics", "CMAME", False),
    (r"\bICLR\b", "ICLR", True),
    (r"arXiv preprint|arXiv:\d{4}\.\d{4,5}", "arXiv preprint", False),
]


def head(text: str, chars: int = 9000) -> str:
    return text[:chars]


def find_venues(text: str) -> list[tuple[str, bool, str]]:
    found: list[tuple[str, bool, str]] = []
    h = head(text)
    for pat, name, ccf_a in VENUES:
        m = re.search(pat, h, re.I)
        if m:
            start = max(0, m.start() - 90)
            ctx = re.sub(r"\s+", " ", h[start:m.end() + 90]).strip()
            found.append((name, ccf_a, ctx))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="literature/pde_text")
    args = ap.parse_args()

    files = sorted(f for f in Path(args.text).glob("*.txt") if f.name != "MANIFEST.tsv")
    rows = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        v = find_venues(text)
        rows.append((f.stem, v))

    ccf_a, other, preprint, unknown = [], [], [], []
    for slug, v in rows:
        names = [n for n, _, _ in v]
        is_a = any(a for _, a, _ in v)
        peer = [n for n, a, _ in v if a] or [n for n, a, _ in v if n != "arXiv preprint"]
        if is_a:
            ccf_a.append((slug, peer, v))
        elif "arXiv preprint" in names:
            preprint.append((slug, v))
        elif peer:
            other.append((slug, peer, v))
        else:
            unknown.append((slug,))

    def show(title: str, items, with_v=True):
        print(f"\n{'=' * 100}\n{title}  ({len(items)})\n{'=' * 100}")
        for it in items:
            slug = it[0]
            if with_v and len(it) > 2:
                ev = "; ".join(f"{n}: ...{c[:110]}..." for n, a, c in it[2][:2])
                print(f"  {slug[:78]:78s} {', '.join(it[1])}")
                print(f"      {ev}")
            elif with_v:
                print(f"  {slug[:78]:78s} preprint/unclear")
            else:
                print(f"  {slug[:96]}")

    show("CCF-A VENUES", ccf_a)
    show("OTHER NAMED PEER-REVIEWED VENUES", other)
    show("arXiv PREPRINTS OR UNIDENTIFIED", preprint, with_v=False)
    if unknown:
        show("NO VENUE MARKER FOUND", unknown, with_v=False)

    print(f"\n\nSUMMARY: {len(ccf_a)} CCF-A, {len(other)} other named venues, "
          f"{len(preprint)} preprint/unclear, {len(unknown)} unknown  (of {len(rows)})")


if __name__ == "__main__":
    main()
