"""
Scan the extracted PDE literature for how each paper compares its cost against a classical
numerical solver.

The question is narrow and factual: when these papers claim a speed advantage over finite
differences / FEM / spectral methods, are they timing *training*, *inference*, or both? This
script finds the relevant sentences with context so they can be read in place rather than
guessed at.

Usage: python src/scan_lit_timing.py [--text literature/pde_text]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# The Windows console here is GBK and chokes on PDF ligatures (U+FB01 etc.).
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def _clean(s: str) -> str:
    """Normalise PDF ligatures and odd whitespace so output is console-safe."""
    for a, b in (("\ufb00", "ff"), ("\ufb01", "fi"), ("\ufb02", "fl"), ("\ufb03", "ffi"),
                 ("\ufb04", "ffl"), ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'),
                 ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "--"), ("\u00ad", "")):
        s = s.replace(a, b)
    return re.sub(r"[^\x09\x0a\x20-\x7e\u00a0-\u024f]", " ", s)

# Phrases that indicate a cost comparison against a classical solver.
SOLVER = r"(finite[- ]difference|finite[- ]element|\bFEM\b|\bFDM\b|spectral method|" \
         r"numerical solver|classical (?:numerical )?(?:solver|method)|Runge[- ]Kutta|" \
         r"high[- ]order scheme|Crank[- ]Nicolson)"
INFER = r"(inference|test[- ]time|evaluation time|forward pass|online (?:phase|stage)|" \
        r"at (?:test|inference) time)"
TRAIN = r"(training time|training cost|offline (?:phase|stage|cost)|training phase|" \
        r"train(?:ing)? (?:the|a) (?:model|network) (?:takes|costs))"
SPEED = r"((?:orders? of magnitude )?(?:faster|speed-?up|speedup|cheaper)|" \
        r"times faster|reduce[ds]? (?:the )?(?:computational )?cost)"

PATTERNS = {
    "speed_claim": re.compile(SPEED, re.I),
    "inference_time": re.compile(rf"{INFER}.{{0,160}}{SPEED}|{SPEED}.{{0,160}}{INFER}", re.I),
    "training_time": re.compile(rf"{TRAIN}.{{0,160}}{SPEED}|{SPEED}.{{0,160}}{TRAIN}", re.I),
    "vs_solver": re.compile(rf"(?:.{{0,120}}{SOLVER}.{{0,200}}{SPEED}|{SPEED}.{{0,200}}{SOLVER})",
                            re.I),
    "amortiz": re.compile(r"(amorti[sz]|break[- ]even|paid once|one[- ]time (?:training|cost)|"
                          r"per (?:query|instance|solve))", re.I),
}


def sentences(text: str) -> list[str]:
    flat = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.;])\s+", flat)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default="literature/pde_text")
    ap.add_argument("--context", type=int, default=2)
    args = ap.parse_args()

    files = sorted(Path(args.text).glob("*.txt"))
    files = [f for f in files if f.name != "MANIFEST.tsv"]
    print(f"scanning {len(files)} extracted papers\n")

    hits: dict[str, dict[str, list[str]]] = {}
    for f in files:
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        found: dict[str, list[str]] = {}
        for name, pat in PATTERNS.items():
            if name == "speed_claim":
                continue
            sents = sentences(text)
            keep = []
            for i, s in enumerate(sents):
                if pat.search(s) and 40 < len(s) < 700:
                    keep.append(s.strip())
                if len(keep) >= 3:
                    break
            if keep:
                found[name] = keep
        if found:
            hits[f.stem] = found

    # ---- per-paper summary -----------------------------------------------------------
    for slug, found in sorted(hits.items()):
        tags = ", ".join(sorted(found))
        print(f"### {_clean(slug)[:110]}")
        print(f"    signals: {tags}")
        for name in ("training_time", "inference_time", "vs_solver", "amortiz"):
            for s in found.get(name, [])[:1]:
                print(f"      [{name}] {_clean(s)[:340]}")
        print()

    print("=" * 100)
    print(f"{len(hits)} of {len(files)} papers contain at least one cost-comparison signal")
    counts: dict[str, int] = {}
    for found in hits.values():
        for k in found:
            counts[k] = counts.get(k, 0) + 1
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v}")


if __name__ == "__main__":
    main()
