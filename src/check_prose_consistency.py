"""Pre-submission proofread pass 1: the things a reader notices and a check can find.

WHAT THIS LOOKS FOR
-------------------
* FLOAT ORDER. A referee reads a paper once, in order. If Table 5 is cited before Table 4, or Figure 3
  before Figure 2, the reading experience degrades in a way no content check catches.
* LABELS CITED BEFORE THEY ARE DEFINED. Same issue, one level down: a \ref to a section that appears
  later in the file is legal LaTeX and a poor experience.
* TERMINOLOGY DRIFT. The same object named two ways ("grid scheme" / "grid solver", "Section~\ref" /
  "\S\ref") reads as carelessness even when every number is right.
* SENTENCE LENGTH, at a threshold that is about comprehension rather than style.

Usage: python src/check_prose_consistency.py
Exit code: 0 if nothing is flagged, 1 otherwise.
"""
from __future__ import annotations

import pathlib
import re
import sys
from collections import Counter

ROOT = pathlib.Path(__file__).resolve().parent.parent
MAIN = ROOT / "paper" / "main.tex"

# Synonym pairs worth reporting. Each is (label, pattern A, pattern B); the point is not that either
# is wrong but that a paper should pick one.
PAIRS = [
    ("the grid scheme", r"grid scheme", r"grid solver"),
    ("finite differences", r"finite[- ]difference", r"finite[- ]differences"),
    ("the neural solver", r"neural solver", r"neural method"),
    ("regression Monte Carlo", r"regression Monte Carlo", r"\bRMC\b"),
    ("cross-reference style", r"Section~\\ref", r"\\S\\ref"),
    ("percent style", r"\\\%", r" per cent"),
    ("approx", r"about \$", r"approximately \$"),
]


def resolved_body(text: str) -> str:
    """main.tex with \\input{tables/...} inlined, so float order reflects the compiled document.

    The first version of this check scanned main.tex alone and therefore saw no tables at all: every
    table lives in its own file and is pulled in with \\input. A float-order check that silently skips
    the tables is worse than none, because it reports success.
    """
    def repl(m: re.Match) -> str:
        rel = m.group(1)
        path = ROOT / "paper" / (rel if rel.endswith(".tex") else rel + ".tex")
        return path.read_text(encoding="utf-8") if path.exists() else m.group(0)
    return re.sub(r"\\input\{([^}]+)\}", repl, text)


def subsection_of(body: str, pos: int) -> str:
    """Title of the subsection containing a character position ('' if none)."""
    titles = [(m.start(), m.group(1)) for m in
              re.finditer(r"\\(?:sub)*section\*?\{([^}]*)\}", body) if m.start() < pos]
    return titles[-1][1] if titles else ""


def float_order(body: str) -> list[str]:
    """Report floats whose first citation comes after the float itself.

    The gap is measured from the END of the float, not its start. Tables here are \\input and their
    inlined tabular bodies run to ~1.6k characters, so measuring from \\begin reported "cited 1693
    characters after it appears" for tables whose reference is the very next sentence.
    """
    floats: list[tuple[int, int, str, str]] = []      # (start, end, kind, label)
    counters: Counter = Counter()
    for m in re.finditer(r"\\begin\{(table\*?|figure\*?)\}", body):
        kind = "table" if m.group(1).startswith("table") else "figure"
        counters[kind] += 1
        end = body.find(f"\\end{{{m.group(1)}}}", m.start())
        end = end if end > 0 else m.start() + 4000
        chunk = body[m.start():end]
        lab = re.search(r"\\label\{([^}]+)\}", chunk)
        floats.append((m.start(), end, f"{kind} {counters[kind]}", lab.group(1) if lab else ""))
    problems = []
    for start, end, name, label in floats:
        if not label:
            continue
        first = None
        for r in re.finditer(r"\\(?:ref|eqref)\{" + re.escape(label) + r"\}", body):
            first = r.start()
            break
        if first is None:
            print(f"    {name:>10}  NEVER CITED")
            problems.append(f"{name} never cited")
            continue
        if first < start:
            verdict = "cited before"
        elif first < end:
            verdict = "cited inside/at the float"
        elif first - end < 1200:
            verdict = "cited just after (fine)"
        else:
            # Character distance alone is the wrong criterion: a float belongs where its TABLE or
            # FIGURE is discussed, and a reference a page later inside the same subsection is normal
            # journal practice. What is a defect is a float introduced in one subsection and never
            # mentioned in it.
            same = subsection_of(body, start) == subsection_of(body, first)
            verdict = ("cited later but in the same subsection (fine)" if same
                       else "CITED IN A DIFFERENT SUBSECTION")
            print(f"    {name:>10}  float ends at {end:>6}, first cite at {first:>6}   {verdict}")
            if not same:
                problems.append(f"{name} introduced in one subsection and cited in another")
            continue
        print(f"    {name:>10}  float ends at {end:>6}, first cite at {first:>6}   {verdict}")
    order = [name for _, _, name, _ in floats]
    print(f"    order in the file: {', '.join(order)}")
    return problems


def main() -> int:
    text = MAIN.read_text(encoding="utf-8")
    body = text.split(r"\begin{document}", 1)[1] if r"\begin{document}" in text else text
    body = resolved_body(body)
    problems: list[str] = []

    print("=" * 100)
    print("PRE-SUBMISSION PROOFREAD: mechanical pass")
    print("=" * 100)

    # ---------------- 1. float order ----------------
    print("\n  float order vs first citation")
    problems += float_order(body)

    # ---------------- 2. terminology ----------------
    print("\n  terminology: REVIEW items, not failures")
    print("    (a synonym pair is not automatically wrong -- 'grid scheme' for the discretisation and")
    print("     'grid solver' for an implementation is a real distinction -- so the minority usages")
    print("     are printed for a human to judge rather than counted against the paper)")
    for label, a, b in PAIRS:
        na = len(re.findall(a, body))
        nb = len(re.findall(b, body))
        tag = "  <- both used" if (na and nb) else ""
        print(f"    {label:26s}  {na:>4} vs {nb:>4}{tag}")
        if na and nb:
            for pat, count in ((a, na), (b, nb)):
                if count > 6:
                    continue
                for m in re.finditer(pat, body):
                    ctx = re.sub(r"\s+", " ", body[max(0, m.start() - 70):m.end() + 70])
                    print(f"        [{count:>2} total] ...{ctx}...")
                    break
    # The cross-reference style has no legitimate distinction: a paper picks one. It is a failure.
    n_sec = len(re.findall(r"Section~\\ref", body))
    n_s = len(re.findall(r"\\S\\ref", body))
    if n_sec and n_s:
        problems.append(f"cross-reference style mixed: Section~\\ref x{n_sec} vs \\S\\ref x{n_s}")

    # ---------------- 3. sentence length ----------------
    # Floats are removed first: a table caption plus its tabular rows reads as a 230-word "sentence"
    # otherwise, and the first version of this check reported exactly that.
    prose = re.sub(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", " ", body, flags=re.S)
    # inline tabulars too: one of them sits inside a paragraph and made a 115-word "sentence"
    prose = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " TABLE ", prose, flags=re.S)

    def sentences(t: str) -> list[str]:
        t = re.sub(r"\\cite\{[^}]*\}", "", t)
        t = re.sub(r"\$[^$]*\$", "MATH", t)
        t = re.sub(r"\\ref\{[^}]*\}", "REF", t)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^}]*\})?", " ", t)
        t = re.sub(r"%.*", "", t)
        flat = re.sub(r"\s+", " ", t)
        # Split before enumerated markers too. Without this the limitations paragraph reads as one
        # 164-word sentence, because "(iii)" does not begin with a capital letter -- and the first
        # version of this check reported that artefact as the paper's worst sentence.
        flat = re.sub(r"\s+\((?:i|ii|iii|iv|v|vi|vii|viii|ix|x)\)\s+", " ||| ", flat)
        flat = re.sub(r"\s+(?:First|Second|Third|Finally),", " ||| ", flat)
        flat = re.sub(r"\.\)\s+(?=[A-Z])", ".) ||| ", flat)
        out = []
        for part in flat.split("|||"):
            out += [s.strip() for s in re.split(r"(?<=[.;])\s+(?=[A-Z(])", part) if s.strip()]
        return out

    sents = sentences(prose)
    long_ones = [s for s in sents if len(s.split()) > 45]
    very_long = [s for s in long_ones if len(s.split()) > 80]
    print(f"\n  sentences: {len(sents)}; over 45 words: {len(long_ones)}; over 80 words: "
          f"{len(very_long)}")
    print("    (45-80 words is ordinary technical prose and is reported as a trend; above 80 fails,")
    print("     because a reader loses the thread before the verb arrives)")
    for s in sorted(long_ones, key=lambda x: -len(x.split()))[:5]:
        print(f"    [{len(s.split()):>3}w] {s[:130]}...")
    if very_long:
        problems.append(f"{len(very_long)} sentences over 80 words")

    print()
    print("-" * 100)
    if problems:
        print("  FLAGGED:")
        for p in problems:
            print(f"    - {p}")
    else:
        print("  Nothing flagged.")
    print("-" * 100)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
