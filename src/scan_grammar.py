"""Extract punctuation patterns for a grammar review, with just enough context to judge each.

Two of the checklist items are mechanical and are better reviewed as a list than in prose. A
semicolon count of 53 in a 12-page paper is high -- semicolons between two independent clauses
often read better as two sentences -- and the paper has 30 non-restrictive ", which" clauses that
should each be checked for whether the clause is genuinely non-restrictive or is really
restrictive and wants "that" with no comma.

This prints each instance with a window of surrounding text, numbered, so a human can accept or
reject each one quickly. It makes no judgements itself.
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


def load_prose(path: Path) -> list[tuple[int, str]]:
    """Body lines with comments and table inputs dropped, keeping line numbers."""
    out = []
    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        if not s or s.startswith("%") or s.startswith("\\input"):
            continue
        out.append((i, s))
    return out


def window(text: str, start: int, end: int, before: int = 95, after: int = 95) -> str:
    lo = max(0, start - before)
    hi = min(len(text), end + after)
    s = re.sub(r"\s+", " ", text[lo:hi]).strip()
    return ("..." if lo > 0 else "") + s + ("..." if hi < len(text) else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="paper/main.tex")
    ap.add_argument("--what", default="semicolon,which",
                    help="comma-separated: semicolon, which, passive, colon")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    lines = load_prose(Path(args.file))
    body = " ".join(t for _, t in lines)

    want = [w.strip() for w in args.what.split(",")]

    if "semicolon" in want:
        print("=" * 100)
        print("SEMICOLONS -- is each one joining two independent clauses that would read better "
              "as two sentences?")
        print("=" * 100)
        for n, m in enumerate(re.finditer(r";", body), 1):
            print(f"\n[{n}] {window(body, m.start(), m.end())}")
            if n >= args.limit:
                break

    if "which" in want:
        print("\n" + "=" * 100)
        print("', WHICH' -- each should be genuinely NON-restrictive. If the clause identifies "
              "which one, it is restrictive and wants 'that' with no comma.")
        print("=" * 100)
        for n, m in enumerate(re.finditer(r",\s+which\b", body), 1):
            print(f"\n[{n}] {window(body, m.start(), m.end(), 70, 120)}")
            if n >= args.limit:
                break

    if "passive" in want:
        print("\n" + "=" * 100)
        print("PASSIVE CANDIDATES -- be + past participle. Judge each: agent unknown or "
              "irrelevant, or avoidably weak?")
        print("=" * 100)
        pat = re.compile(r"\b(is|are|was|were|be|been|being)\s+(\w+(?:ed|en))\b")
        for n, m in enumerate(pat.finditer(body), 1):
            print(f"\n[{n}] {window(body, m.start(), m.end(), 80, 40)}")
            if n >= args.limit:
                break

    if "colon" in want:
        print("\n" + "=" * 100)
        print("COLONS -- should introduce a list or an explanation, not a clause")
        print("=" * 100)
        for n, m in enumerate(re.finditer(r":", body), 1):
            print(f"\n[{n}] {window(body, m.start(), m.end(), 80, 100)}")
            if n >= args.limit:
                break


if __name__ == "__main__":
    main()
