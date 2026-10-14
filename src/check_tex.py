"""
LaTeX structural check for main.tex, because no LaTeX toolchain is available in this environment.

Why this exists
---------------
Four edits were made to the manuscript for the journal version: a deep-BSDE table was added, three
figures were inserted, and the repository placeholder was replaced by a macro. None of them can be
compile-checked here -- `pdflatex` is not on PATH, so `src/build_paper.py` cannot run -- and an
unbalanced brace or a missing graphics file is exactly the kind of defect that survives until the
author opens the project in Overleaf.

This does not replace compiling. It checks the failure modes that are mechanically checkable and
that a compile would otherwise be the first to catch:

  1. brace balance, ignoring escaped braces (\\{ \\}) and those inside comments;
  2. begin/end environment matching, in order, ignoring comments;
  3. every \\input and \\includegraphics target exists on disk;
  4. every \\label defined once, and every \\ref/\\eqref/\\cite target defined somewhere.

What it deliberately does NOT check: float placement, page count, overfull boxes, or whether the
prose makes sense. Those need a real build and a reader, and the script says so rather than implying
a clean run means the paper is fine.

Usage: python src/check_tex.py [paper/main.tex]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"


def strip_comments(text: str) -> str:
    """Remove LaTeX comments, honouring escaped percent signs.

    A naive split on '%' truncates any line containing '\\%', which appears in the manuscript, and
    would silently drop the rest of that line from every subsequent check.
    """
    out = []
    for line in text.splitlines():
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == "%":
                line = line[:i]
                break
            i += 1
        out.append(line)
    return "\n".join(out)


def check(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8", errors="replace")
    body = strip_comments(raw)
    problems: list[str] = []
    notes: list[str] = []

    # --- 1. brace balance ---
    depth = 0
    min_depth = 0
    for i, ch in enumerate(body):
        if ch == "\\":
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            min_depth = min(min_depth, depth)
    # Escaped braces are consumed by the backslash-skip above only partially; recount properly.
    depth = 0
    min_depth = 0
    i = 0
    while i < len(body):
        ch = body[i]
        if ch == "\\":
            i += 2
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            min_depth = min(min_depth, depth)
        i += 1
    if depth != 0:
        problems.append(f"brace imbalance: {depth:+d} at end of file")
    if min_depth < 0:
        problems.append(f"closing brace with no opener ({-min_depth} occurrence(s))")

    # --- 2. environment matching ---
    begins = [(m.start(), m.group(1)) for m in re.finditer(r"\\begin\{([^}]+)\}", body)]
    ends = [(m.start(), m.group(1)) for m in re.finditer(r"\\end\{([^}]+)\}", body)]
    stack: list[str] = []
    events = sorted([(p, "b", n) for p, n in begins] + [(p, "e", n) for p, n in ends])
    for _, kind, name in events:
        if kind == "b":
            stack.append(name)
        else:
            if not stack:
                problems.append(f"\\end{{{name}}} with no matching \\begin")
            elif stack[-1] != name:
                problems.append(f"\\end{{{name}}} closes \\begin{{{stack[-1]}}}")
                stack.pop()
            else:
                stack.pop()
    for name in stack:
        problems.append(f"\\begin{{{name}}} never closed")

    # --- 3. input and graphics targets ---
    targets: list[tuple[str, str]] = []
    for m in re.finditer(r"\\input\{([^}]+)\}", body):
        targets.append(("input", m.group(1)))
    for m in re.finditer(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", body):
        targets.append(("graphics", m.group(1)))
    graphpath_match = re.search(r"\\graphicspath\{((?:\{[^}]*\})*)\}", body)
    graphdirs = re.findall(r"\{([^}]*)\}", graphpath_match.group(1)) if graphpath_match else [""]

    for kind, t in targets:
        stem = t if t.endswith(".tex") or "." in Path(t).name else t + ".tex"
        cands = [PAPER / stem, PAPER / stem.replace("../", "")]
        if kind == "graphics":
            cands = []
            for gd in graphdirs:
                cands.append((PAPER / gd / t).resolve())
                cands.append((PAPER / gd / t).with_suffix(".pdf"))
                cands.append((PAPER / gd / t).with_suffix(".png"))
            cands.append((PAPER / t).resolve())
        if not any(c.exists() for c in cands):
            problems.append(f"{kind} target not found: {t}  (tried {[str(c) for c in cands[:3]]})")
    notes.append(f"resolved {len(targets)} input/graphics target(s)")

    # --- 4. labels and references ---
    # Labels are collected across \input'd files too. The tables are \input rather than inlined, so
    # checking only this file reports every table reference as unresolved -- which is a false
    # positive, and a validator that cries wolf is worse than none because it trains the reader to
    # ignore it.
    all_body = body
    for kind, t in targets:
        if kind != "input":
            continue
        stem = t if t.endswith(".tex") else t + ".tex"
        for cand in (PAPER / stem, PAPER / stem.replace("../", "")):
            if cand.exists():
                all_body += "\n" + strip_comments(cand.read_text(encoding="utf-8", errors="replace"))
                break

    labels = re.findall(r"\\label\{([^}]+)\}", all_body)
    dupes = {x for x in labels if labels.count(x) > 1}
    for d in sorted(dupes):
        problems.append(f"duplicate \\label{{{d}}}")
    refs = re.findall(r"\\(?:ref|eqref|autoref)\{([^}]+)\}", all_body)
    for r in sorted(set(refs)):
        if r not in labels:
            problems.append(f"\\ref{{{r}}} has no matching \\label")
    cites = set()
    for m in re.finditer(r"\\cite\{([^}]+)\}", body):
        cites.update(x.strip() for x in m.group(1).split(","))
    bib = (PAPER / "refs.bib").read_text(encoding="utf-8", errors="replace")
    bibkeys = set(re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M))
    for c in sorted(cites):
        if c not in bibkeys:
            problems.append(f"\\cite{{{c}}} not in refs.bib")
    uncited = sorted(bibkeys - cites)
    if uncited:
        notes.append(f"{len(uncited)} bib entry(ies) never cited: {', '.join(uncited[:8])}"
                     + (" ..." if len(uncited) > 8 else ""))
    notes.append(f"{len(labels)} labels, {len(set(refs))} distinct refs, "
                 f"{len(cites)} cited keys of {len(bibkeys)} in refs.bib")

    # Recorded RELATIVE to the repository. An absolute path here would publish the author's disk
    # layout in a results file, and -- because check_release.py scans results/ -- it would also make
    # the release check report its own findings back to itself, growing by one entry per run.
    try:
        rel = path.resolve().relative_to(ROOT)
    except ValueError:
        rel = path.name
    return {"path": rel.as_posix(), "problems": problems, "notes": notes}


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else PAPER / "main.tex"
    print("=" * 100)
    print("LaTeX STRUCTURAL CHECK (brace/env/targets/labels) -- NOT a compile")
    print("=" * 100)
    r = check(target)
    print(f"  {r['path']}")
    for n in r["notes"]:
        print(f"    note: {n}")
    if r["problems"]:
        print(f"\n  {len(r['problems'])} PROBLEM(S):")
        for p in r["problems"]:
            print(f"    - {p}")
    else:
        print("\n  no structural problems found")
    print()
    print("  This does NOT verify float placement, page count or overfull boxes, and no LaTeX")
    print("  toolchain is available here, so the manuscript has NOT been compiled. Build it in")
    print("  Overleaf (or with TeX Live) before treating the layout as verified.")
    out = ROOT / "results" / "check_tex.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(r, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 1 if r["problems"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
