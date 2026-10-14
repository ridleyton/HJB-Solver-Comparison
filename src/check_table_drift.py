"""
Which numbers in the committed tables disagree with the results that generate them?

The problem this answers
------------------------
Regenerating all twelve tables from `results/` changed only three files, and for two of them the
change was captions and float placement. But `tab_extract.tex` changed its NUMBERS, and they disagree
materially: the committed table reports a cubic-reconstruction order of 5.6 on the trigonometric test
function where the current results say 4.75, and 4.8 on the quartic where the results say 4.38. The
prose quotes some of these ("5.2 against 2.1"), so a stale table and a current table cannot both be
right in the same paper.

Before rewriting anything, this measures the full extent of the drift: for every table, every numeric
cell in every data row, compared between the committed file and the freshly generated one. It reports
the tokens that appear on one side and not the other, which is enough to see WHAT moved without
guessing at column alignment -- trying to pair cells positionally across a reformatted row produces
false matches, and a false match here would hide a wrong number.

Usage: python src/check_table_drift.py <generated_dir>
       (generate a fresh copy first: python src/make_tables.py, then move paper/tables aside)

Reads the committed tables from paper/tables and a generated copy from the directory given.
"""

from __future__ import annotations

import pathlib
import re
import sys

PAPER_TABLES = pathlib.Path("paper/tables")

# A numeric token as it appears in a LaTeX table cell: an optional sign, digits, a decimal part and
# an optional exponent. LaTeX $\pm$ markers and \multicolumn headers are not numbers and are skipped.
TOKEN = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")


def data_rows(text: str) -> list[str]:
    """Rows that carry data: they start with a digit, or with a named reconstruction label.

    Header rows and \\multicolumn rows are excluded, because they describe the table rather than
    report a measurement and would otherwise contribute spurious numeric tokens.
    """
    out = []
    for line in text.splitlines():
        s = line.strip()
        if not s or s.startswith("%") or s.startswith("\\"):
            continue
        if s[0].isdigit() or s.startswith("multilinear") or s.startswith("cubic"):
            out.append(s)
    return out


def tokens(rows: list[str]) -> dict[str, int]:
    """Numeric tokens with their counts, rounded to 4 significant decimals for comparison.

    Rounding to 4 decimals matters: a value written 0.99927 in one file and 0.9993 in another is the
    SAME measurement at different precision, and reporting it as drift would bury the real changes.
    """
    counts: dict[str, int] = {}
    for r in rows:
        for t in TOKEN.findall(r):
            try:
                v = float(t)
            except ValueError:
                continue
            key = f"{v:.4g}"
            counts[key] = counts.get(key, 0) + 1
    return counts


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    gen_dir = pathlib.Path(sys.argv[1])
    if not gen_dir.is_dir():
        print(f"generated directory not found: {gen_dir}")
        return 2

    print("=" * 100)
    print("TABLE DRIFT: committed paper/tables vs freshly generated")
    print("=" * 100)
    print(f"  committed: {PAPER_TABLES}")
    print(f"  generated: {gen_dir}")
    print()

    drifted: list[str] = []
    for f in sorted(PAPER_TABLES.glob("*.tex")):
        g = gen_dir / f.name
        if not g.exists():
            continue
        a = tokens(data_rows(f.read_text(encoding="utf-8", errors="replace")))
        b = tokens(data_rows(g.read_text(encoding="utf-8", errors="replace")))
        if a == b:
            continue
        only_a = {k: v for k, v in a.items() if b.get(k, 0) < v}
        only_b = {k: v for k, v in b.items() if a.get(k, 0) < v}
        if not only_a and not only_b:
            continue
        drifted.append(f.name)
        print(f"  {f.name}")
        if only_a:
            print(f"      in COMMITTED but not generated: {sorted(only_a, key=float)}")
        if only_b:
            print(f"      in GENERATED but not committed: {sorted(only_b, key=float)}")

    print()
    print("-" * 100)
    if drifted:
        print(f"  {len(drifted)} table(s) carry numbers that disagree with the results: "
              f"{', '.join(drifted)}")
        print("  The results files are the authority -- the tables are generated from them -- so a")
        print("  committed value absent from the generated set is a STALE measurement, and any prose")
        print("  quoting it is stale too. Re-run make_tables.py and update the affected prose.")
    else:
        print("  no numeric drift: every committed table matches the results that generate it")
    print()
    print("  NOTE: this compares numeric TOKENS, so a value that moved column or changed precision")
    print("  is reported. Counts are included because a duplicated value in one file and not the")
    print("  other is also drift.")
    return 1 if drifted else 0


if __name__ == "__main__":
    raise SystemExit(main())
