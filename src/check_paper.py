"""
Static sanity checks on the manuscript, since no LaTeX toolchain is available locally.

This does not replace compilation. It catches the mechanical failures that would otherwise
only surface on Overleaf: unbalanced environments, undefined or duplicate labels, references
to labels that do not exist, missing \\input files, missing figure files, and citations whose
keys are absent from the .bib.

Run: python src/check_paper.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PAPER = Path("paper")
MAIN = PAPER / "main.tex"
BIB = PAPER / "refs.bib"
FIGDIR = Path("figures")

FAILS: list[str] = []
WARNS: list[str] = []


def fail(msg: str) -> None:
    FAILS.append(msg)
    print(f"  [FAIL] {msg}")


def warn(msg: str) -> None:
    WARNS.append(msg)
    print(f"  [WARN] {msg}")


def ok(msg: str) -> None:
    print(f"  [ OK ] {msg}")


def strip_comments(tex: str) -> str:
    out = []
    for line in tex.splitlines():
        # Remove unescaped % comments.
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def main() -> int:
    if not MAIN.exists():
        fail(f"{MAIN} not found")
        return 1
    raw = MAIN.read_text(encoding="utf-8")
    tex = strip_comments(raw)

    print("environments")
    begins = re.findall(r"\\begin\{(\w+\*?)\}", tex)
    ends = re.findall(r"\\end\{(\w+\*?)\}", tex)
    from collections import Counter
    cb, ce = Counter(begins), Counter(ends)
    bal = True
    for env in sorted(set(cb) | set(ce)):
        if cb[env] != ce[env]:
            fail(f"environment '{env}' unbalanced: {cb[env]} begin vs {ce[env]} end")
            bal = False
    if bal:
        ok(f"{len(set(cb))} environment types balanced ({len(begins)} begin/end pairs)")

    print("\nlabels and references")
    labels = re.findall(r"\\label\{([^}]+)\}", tex)
    dup = [l for l, c in Counter(labels).items() if c > 1]
    if dup:
        fail(f"duplicate labels: {dup}")
    else:
        ok(f"{len(labels)} labels, all unique")

    refs = set(re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", tex))
    # Labels may live in \input-ed table files.
    inputs = re.findall(r"\\input\{([^}]+)\}", tex)
    input_files = []
    for inc in inputs:
        p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
        if not p.exists():
            fail(f"\\input file missing: {p}")
            continue
        input_files.append(p)
        t = strip_comments(p.read_text(encoding="utf-8"))
        labels += re.findall(r"\\label\{([^}]+)\}", t)
    labelset = set(labels)
    missing = sorted(r for r in refs if r not in labelset)
    if missing:
        fail(f"references to undefined labels: {missing}")
    else:
        ok(f"{len(refs)} distinct references, all defined")
    if input_files:
        ok(f"{len(input_files)} \\input file(s) present: "
           f"{[str(p.relative_to(PAPER)) for p in input_files]}")

    print("\ncitations")
    cites = set()
    for key in re.findall(r"\\cite\{([^}]+)\}", tex):
        cites.update(k.strip() for k in key.split(","))
    if not BIB.exists():
        fail(f"{BIB} not found")
    else:
        bibkeys = set(re.findall(r"@\w+\{([^,]+),", BIB.read_text(encoding="utf-8")))
        missing = sorted(cites - bibkeys)
        unused = sorted(bibkeys - cites)
        if missing:
            fail(f"cited but absent from refs.bib: {missing}")
        else:
            ok(f"{len(cites)} cited keys all present in refs.bib")
        if unused:
            warn(f"{len(unused)} bib entries never cited: {unused}")

    print("\nfigures")
    figs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex)
    if not figs:
        warn("no \\includegraphics found")
    for f in figs:
        base = Path(f).name
        found = any((FIGDIR / base.replace(ext, "")).with_suffix(ext).exists()
                    for ext in (".pdf", ".png"))
        if found:
            ok(f"{base} present in figures/")
        else:
            warn(f"{base} not yet generated in figures/")

    print("\nstructure")
    if re.search(r"\\documentclass", tex):
        ok("\\documentclass present")
    else:
        fail("no \\documentclass")
    for need in ("\\begin{document}", "\\end{document}", "\\maketitle",
                 "\\begin{abstract}", "\\bibliography"):
        if need in tex:
            ok(f"{need} present")
        else:
            fail(f"{need} missing")
    n_refs_decl = len(re.findall(r"\\bibliographystyle", tex))
    ok(f"{n_refs_decl} \\bibliographystyle declaration(s)")

    # Leftover placeholders
    todo = re.findall(r"\\TBD\{([^}]*)\}", tex)
    if todo:
        warn(f"{len(todo)} unresolved \\TBD placeholder(s): {todo}")
    else:
        ok("no unresolved \\TBD placeholders")
    for pat, what in ((r"XXX", "XXX"), (r"FIXME", "FIXME"), (r"\?\?\?", "???")):
        if re.search(pat, tex):
            warn(f"found '{what}' in the source")

    print("\nsection count")
    secs = re.findall(r"\\section\*?\{([^}]*)\}", tex)
    ok(f"{len(secs)} sections: {secs}")

    print("\n" + "=" * 70)
    if FAILS:
        print(f"{len(FAILS)} FAILURE(S)")
        return 1
    print(f"no static failures ({len(WARNS)} warning(s) -- warnings are expected while "
          f"results are still being generated)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
