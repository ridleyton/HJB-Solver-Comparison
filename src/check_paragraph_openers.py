"""Pass 3 of the Conciseness Editing Guide: does each paragraph's first sentence do real work?

The guide's paragraph-level checks are (1) does the first sentence do real work, (2) is anything
repeated from the previous paragraph, (3) can two paragraphs be merged, (4) does the paragraph
earn its place. This extracts the opening sentence of every body paragraph so (1) can be judged
by eye, and flags openers that only announce what follows ("In this section we...", "Table X
shows...", "We now turn to...").
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

# Openers that announce rather than state. Used only to mark candidates for a human to judge.
ANNOUNCERS = re.compile(
    r"^(in this (section|paper|subsection)|this (section|paper|subsection)|we now|"
    r"we next|the following|what follows|before (we|turning)|let us|next,|finally,|"
    r"having (established|shown)|with (this|that) in (mind|place))", re.I)


def strip_latex(s: str) -> str:
    """Drop LaTeX markup but KEEP the text inside formatting commands.

    A naive `\\\\[a-zA-Z]+\\{...\\}` strip deletes the argument too, so `\\emph{derivative}` vanished
    and a sentence read "is a of the value function" -- which looks like lost text but is only a
    tool artefact. Formatting commands keep their content; structural ones do not.
    """
    s = re.sub(r"\$[^$]*\$", "[math]", s)
    # formatting commands: keep the braced argument
    s = re.sub(r"\\(emph|textbf|textit|texttt|textsc|mathrm|mathbf|text)\{([^{}]*)\}", r"\2", s)
    # cross-references and citations: replace with a placeholder so sentence shape survives
    s = re.sub(r"\\(ref|eqref|cite\w*|label)\{([^{}]*)\}", "[x]", s)
    # any remaining command, with or without arguments
    s = re.sub(r"\\[a-zA-Z]+\*?(\[[^\]]*\])?(\{[^{}]*\})?", " ", s)
    s = re.sub(r"[{}]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def paragraphs(path: Path) -> list[tuple[int, str]]:
    out: list[tuple[int, str]] = []
    start, buf = None, []

    def flush():
        if buf:
            txt = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if len(txt.split()) >= 20:
                out.append((start, txt))

    for i, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = raw.strip()
        skip = (not s) or s.startswith((
            "%", "\\input", "\\begin{table", "\\end{table", "\\begin{figure", "\\end{figure}",
            "\\caption", "\\label", "\\begin{equation", "\\end{equation", "\\begin{align",
            "\\end{align", "\\begin{itemize}", "\\end{itemize}", "\\item", "\\subsection",
            "\\title", "\\author", "\\setlength", "\\def", "\\graphicspath", "\\includegraphics"))
        if skip:
            flush()
            buf, start = [], None
            continue
        if s.startswith(("\\section", "\\subsubsection")):
            flush()
            buf, start = [], i
            buf.append(strip_latex(s))
            continue
        if start is None:
            start = i
        buf.append(strip_latex(s))
    flush()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="paper/main.tex")
    ap.add_argument("--show", type=int, default=60)
    args = ap.parse_args()

    paras = paragraphs(Path(args.file))
    print(f"{len(paras)} body paragraphs\n" + "=" * 100)
    flagged = 0
    for n, (ln, text) in enumerate(paras, 1):
        first = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)[0]
        first = first.strip()
        mark = ""
        if ANNOUNCERS.match(re.sub(r"^[^A-Za-z]+", "", first)):
            mark = "  <-- ANNOUNCES rather than states"
            flagged += 1
        print(f"\n[{n}] L{ln}  {first[:190]}{mark}")
        if n >= args.show:
            break
    print(f"\n{flagged} opener(s) flagged as announcing rather than stating")


if __name__ == "__main__":
    main()
