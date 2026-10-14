"""Flag prose sentences longer than a threshold, per the proofreader checklist (TASK 1).

Heuristic and deliberately conservative: LaTeX is stripped of comments, table inputs, labels and
display math, equations are collapsed to a single token so they do not inflate word counts, and
sentences are split on periods that are followed by a space and a capital. The aim is a short
list for a human to read, not a verdict -- a 45-word sentence that is clear is fine, and this
script cannot tell.
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


def prose_paragraphs(path: Path) -> list[tuple[int, str]]:
    """Return (first line number, paragraph text) pairs.

    Paragraphs are the right unit: joining the whole file let a sentence at the end of one
    section run into the first sentence of the next, which produced bogus 100-word sentences at
    the top of the ranking. Boundaries are blank lines and any structural command.
    """
    out: list[tuple[int, str]] = []
    start = None
    buf: list[str] = []

    def flush():
        if buf:
            text = " ".join(buf)
            if len(text.split()) >= 6:
                out.append((start, text))

    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        structural = (not s) or s.startswith((
            "%", "\\section", "\\subsection", "\\subsubsection", "\\item", "\\input",
            "\\begin{table", "\\end{table", "\\begin{figure", "\\end{figure}",
            "\\caption", "\\label", "\\footnote", "\\textbf{", "$$", "\\[", "\\]",
            "\\begin{equation", "\\end{equation", "\\begin{align", "\\end{align",
            "\\begin{abstract}", "\\end{abstract}", "\\begin{IEEEkeywords}",
            "\\title", "\\author", "\\maketitle", "\\end{IEEEkeywords}", "\\begin{itemize}",
            "\\end{itemize}", "\\begin{enumerate}", "\\end{enumerate}",
        ))
        if structural:
            flush()
            buf, start = [], None
            if not s:
                continue
        if start is None:
            start = i
        t = re.sub(r"\$[^$]*\$", "MATH", s)
        t = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", t)
        t = re.sub(r"[{}]", " ", t)
        t = re.sub(r"\s+", " ", t).strip()
        if len(t.split()) >= 4:
            buf.append(t)
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="paper/main.tex")
    ap.add_argument("--max-words", type=int, default=40)
    args = ap.parse_args()

    paras = prose_paragraphs(Path(args.file))

    long_ones: list[tuple[int, int, str]] = []
    for ln, para in paras:
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z\\])", para):
            n = len(re.findall(r"[A-Za-z][A-Za-z'\-]*", s))
            if n > args.max_words:
                long_ones.append((n, ln, s.strip()))

    long_ones.sort(reverse=True)
    print(f"sentences over {args.max_words} words: {len(long_ones)}  "
          f"(across {len(paras)} paragraphs)")
    print("=" * 100)
    for n, ln, s in long_ones:
        print(f"\n[{n} words, near L{ln}] {s[:500]}{'...' if len(s) > 500 else ''}")


if __name__ == "__main__":
    main()
