"""
Measure the manuscript: prose word count, float inventory, and printed length.

Two ways to get the printed length:

* If `paper_build/main.pdf` exists (i.e. `src/build_paper.py` has been run with a local TeX
  Live installation), the page count is read from the LaTeX log -- this is the authoritative
  number.
* Otherwise the length is estimated in column-inches. That estimate is only a fallback: it
  initially predicted 5 pages for a manuscript that actually compiles to 8, because the
  float-height model is crude and float placement leaves whitespace that the model does not
  capture. Treat the estimate as a lower bound, not a measurement.

The word counter keeps the argument text of formatting commands (so ``\\emph{word}`` counts as
one word) and drops the arguments of commands whose content is not prose (``\\cite``, ``\\ref``,
``\\includegraphics``). An earlier version stripped command and argument together, which
undercounted the manuscript by roughly 3x and led to a badly wrong page estimate.
"""

from __future__ import annotations

import re
from pathlib import Path

PAPER = Path("paper")
LOG = Path("paper_build/main.log")
COLUMN_INCHES_PER_PAGE = 18.0
WORDS_PER_PAGE = 1800.0

# Commands whose arguments are not prose and should be removed entirely.
DROP_WITH_ARG = (
    r"\\(?:cite|citep|citet|ref|eqref|label|includegraphics|bibliography|"
    r"bibliographystyle|documentclass|usepackage|input|IEEEauthorblockN|"
    r"IEEEauthorblockA|IEEEoverridecommandlockouts)\*?(?:\[[^\]]*\])?(?:\{[^}]*\})?"
)
# Commands whose arguments ARE prose and should be kept.
KEEP_ARG = r"\\(?:emph|textbf|textit|texttt|mathrm|mathbf|text|textsc|sc|mbox|hbox)\{([^{}]*)\}"


def count_words(tex: str, include_inputs: bool = True) -> tuple[int, int]:
    """Return (prose words, table words)."""
    tex = re.sub(r"(?<!\\)%.*", "", tex)
    body = tex.split(r"\begin{document}")[-1]

    table_words = 0
    if include_inputs:
        for inc in re.findall(r"\\input\{([^}]+)\}", body):
            p = PAPER / (inc if inc.endswith(".tex") else inc + ".tex")
            if not p.exists():
                continue
            t = p.read_text(encoding="utf-8")
            t = re.sub(r"(?<!\\)%.*", "", t)
            t = re.sub(r"\\[a-zA-Z]+\*?", " ", t)
            t = re.sub(r"[{}$&\\]", " ", t)
            table_words += len(re.findall(r"[A-Za-z][A-Za-z'-]+", t))

    body = re.sub(r"\\input\{[^}]+\}", " ", body)
    body = re.sub(DROP_WITH_ARG, " ", body)
    for _ in range(3):                       # nested formatting commands
        body = re.sub(KEEP_ARG, r"\1", body)
    body = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?", " ", body)
    body = re.sub(r"[{}$&\\]", " ", body)
    return len(re.findall(r"[A-Za-z][A-Za-z'-]+", body)), table_words


def compiled_pages() -> int | None:
    if not LOG.exists():
        return None
    m = None
    for m in re.finditer(r"Output written on .*?\((\d+) pages?", LOG.read_text(
            encoding="utf-8", errors="replace")):
        pass
    return int(m.group(1)) if m else None


def main() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    words, table_words = count_words(tex)
    floats = len(re.findall(r"\\begin\{(?:table|figure)\*?\}", tex)) + \
        len(re.findall(r"\\input\{tables/", tex))

    print(f"  prose words        : {words}")
    print(f"  table words        : {table_words}")
    print(f"  floats             : {floats} "
          f"({len(re.findall(r'input.tables/', tex))} tables, "
          f"{len(re.findall(chr(92) + chr(92) + 'begin{table', tex)) if False else len(re.findall(r'begin.figure', tex))} figures)")
    print()
    pages = compiled_pages()
    if pages is not None:
        print(f"  COMPILED PAGES     : {pages}   <- authoritative (from paper_build/main.log)")
        print(f"  target             : 6 pages for the camera-ready "
              f"(initial submission has no page limit)")
    else:
        text_in = words / WORDS_PER_PAGE * COLUMN_INCHES_PER_PAGE
        est = text_in / COLUMN_INCHES_PER_PAGE + 0.75 + floats * 0.25
        print(f"  estimated pages    : ~{est:.1f} (LOWER BOUND — no compiled PDF found; "
              f"run src/build_paper.py)")


if __name__ == "__main__":
    main()
