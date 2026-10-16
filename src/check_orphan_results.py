"""Is every measured result reported somewhere, or merely stored?

WHY THIS CHECK EXISTS
---------------------
The verification suite answers "does every number in the prose match a result file?" That is a
consistency check, and it has a blind spot the size of an experiment: a result file that nothing reads
is invisible to it. Two experiments in this repository had been run, committed, and then never
reported -- the implicit splitting scheme and the PI-PINN iteration sweep -- and the second one was
worse than unreported, because the abstract claimed "two PINN variants" while no table showed the
second. The question that exposed them ("did you only compare two methods?") came from a reader, not
from any check.

So this check exists: every `results/*.json` must be either consumed by something in `src/` or
`paper/`, or listed below with a reason. It is a file-scan, so it costs milliseconds and can run on
every change.

Usage: python src/check_orphan_results.py
Exit code: 0 if every result file is accounted for, 1 otherwise.
"""
from __future__ import annotations

import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
SRC = ROOT / "src"
PAPER = ROOT / "paper"

# Files that legitimately nothing reads, each with the reason. Adding an entry here is a decision that
# the file does not need to be reported; it is not a way to silence a real finding.
ALLOWED_ORPHANS = {
    "smoke_exp.json": "smoke test output, superseded by the full runs",
    "exp_bsde_dims_smoke.json": "smoke test before the multi-hour deep-BSDE dimension sweep",
    "exp_grid_refinement_highp.json": "intermediate merged into exp_grid_refinement_merged.json, "
                                      "which is what the tables read",
}


def norm(text: str) -> str:
    """LaTeX escapes underscores, so an audit that does not normalise reports false orphans.

    The first version of this scan reported 50 unreferenced files for exactly that reason: every
    filename inside a \\texttt{} in a caption reads as `exp\\_lsmc.json`.
    """
    return text.replace("\\_", "_").replace("\\%", "%")


def main() -> int:
    consumers: dict[str, set[str]] = {}
    for path in list(SRC.glob("*.py")) + list(PAPER.rglob("*.tex")) + list(PAPER.rglob("*.md")):
        text = norm(path.read_text(errors="replace"))
        for m in re.finditer(r"\b([a-z0-9_]+\.json)\b", text):
            consumers.setdefault(m.group(1), set()).add(path.name)

    files = sorted(p.name for p in RESULTS.glob("*.json"))
    orphans = [f for f in files if f not in consumers and f not in ALLOWED_ORPHANS]

    print("=" * 100)
    print("ORPHAN RESULT AUDIT: is everything measured also reported?")
    print("=" * 100)
    print(f"  result files                  : {len(files)}")
    print(f"  consumed by src/ or paper/    : {len(files) - len(orphans) - len(ALLOWED_ORPHANS)}")
    print(f"  allowed to be unreported      : {len(ALLOWED_ORPHANS)}")
    for name, why in sorted(ALLOWED_ORPHANS.items()):
        mark = "present" if (RESULTS / name).exists() else "absent "
        print(f"      [{mark}] {name:34s} {why}")
    if orphans:
        print(f"\n  UNREPORTED ({len(orphans)}) -- a measurement nothing reads:")
        for f in orphans:
            kb = (RESULTS / f).stat().st_size // 1024
            kind = "?"
            try:
                d = json.loads((RESULTS / f).read_text())
                kind = ("rows" if isinstance(d, dict) and "rows" in d else
                        "runs" if isinstance(d, dict) and "runs" in d else
                        "dims" if isinstance(d, dict) and "dims" in d else "check")
            except Exception:
                pass
            print(f"      {f:44s} {kb:>5d} KB  [{kind}]")
        print("\n  Each one is either reported in a table (if it holds results) or added to "
              "ALLOWED_ORPHANS\n  with a reason (if it is an intermediate or a smoke test).")
    print()
    print("-" * 100)
    if orphans:
        print(f"  FAIL: {len(orphans)} measured result file(s) that nothing reports")
    else:
        print("  OK: every result file is either reported or explicitly allowed to be unreported")
    print("-" * 100)
    return 1 if orphans else 0


if __name__ == "__main__":
    raise SystemExit(main())
