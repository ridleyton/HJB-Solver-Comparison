"""
Diagnostic: split the page count into text and float contributions.

Compiles a variant of the staged manuscript with every table and figure environment removed,
so the text-only page count can be compared with the full document. Removes the floats by
*environment matching*, not by regex on the source text, to avoid the escaping pitfalls that
invalidated a first attempt at this.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

# Resolved at run time instead of hardcoded, so a reader on another machine gets a clear message
# rather than a build that only works on the author's. TEXLIVE_BIN may name the engine directory
# directly; otherwise the engine is taken from PATH.
_TEX_ENV = os.environ.get("TEXLIVE_BIN")
TEXLIVE_BIN = (Path(_TEX_ENV) if _TEX_ENV
               else (Path(shutil.which("pdflatex")).parent if shutil.which("pdflatex") else None))
BUILD = Path("paper_build")


def strip_floats(tex: str) -> str:
    """Remove table/figure environments and table \\input lines, by scanning lines."""
    out: list[str] = []
    i = 0
    lines = tex.split("\n")
    while i < len(lines):
        line = lines[i]
        m = re.match(r"\s*\\begin\{(table\*?|figure\*?)\}", line)
        if m:
            env = m.group(1)
            end = rf"\end{{{env}}}"
            while i < len(lines):
                if end in lines[i]:
                    break
                i += 1
            i += 1
            continue
        if re.match(r"\s*\\input\{tables/", line):
            i += 1
            continue
        out.append(line)
        i += 1
    return "\n".join(out)


def compile_count(name: str) -> int | None:
    if TEXLIVE_BIN is None:
        print("  pdflatex not found: set TEXLIVE_BIN or put it on PATH")
        return None
    # The engine may be named pdflatex or pdflatex.exe depending on the platform.
    exe = next((str(TEXLIVE_BIN / c) for c in ("pdflatex.exe", "pdflatex")
                if (TEXLIVE_BIN / c).exists()), "pdflatex")
    for _ in range(2):
        subprocess.run([exe, "-interaction=nonstopmode", "-file-line-error", name + ".tex"],
                       cwd=BUILD, capture_output=True, text=True, errors="replace")
    log = BUILD / f"{name}.log"
    if not log.exists():
        return None
    pages = None
    for m in re.finditer(r"Output written on .*?\((\d+) pages?", log.read_text(
            encoding="utf-8", errors="replace")):
        pages = int(m.group(1))
    return pages


def main() -> None:
    src = BUILD / "main.tex"
    if not src.exists():
        print("run src/build_paper.py first")
        return
    tex = src.read_text(encoding="utf-8")
    stripped = strip_floats(tex)

    # Sanity-check the strip before spending a compile on it.
    n_tab = len(re.findall(r"\\begin\{table", stripped))
    n_fig = len(re.findall(r"\\begin\{figure", stripped))
    n_inc = len(re.findall(r"\\input\{tables/", stripped))
    print(f"  floats remaining after strip: table={n_tab} figure={n_fig} input={n_inc}")
    if n_tab or n_fig or n_inc:
        print("  STRIP FAILED -- aborting")
        return

    (BUILD / "main_nofloats.tex").write_text(stripped, encoding="utf-8")
    full = compile_count("main")
    text_only = compile_count("main_nofloats")
    print(f"  full document     : {full} pages")
    print(f"  text only (no floats): {text_only} pages")
    if full is not None and text_only is not None:
        print(f"  -> floats account for approximately {full - text_only} page(s)")


if __name__ == "__main__":
    main()
