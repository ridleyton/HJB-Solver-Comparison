"""
Report how much of the manuscript each section, float and display equation costs.

Why this exists
---------------
Cutting a 14-page manuscript to the 6-page camera-ready limit cannot be done by feel: the
expensive items are not the ones that look long. A section that reads as dense prose may be
cheap, while a single multi-row table can cost half a column. This script attributes the page
budget to concrete objects so the cut list is decided from measurements.

Prose words are counted on PROSE ONLY: comment lines, display equations, tabular bodies and
figure environments are excluded, because they are accounted for separately (as equation and
float counts) and would otherwise be double-counted.

Usage: python src/section_budget.py [paper/main.tex]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

FLOAT_ENVS = ("table", "table*", "figure", "figure*")
BODY_ENVS = ("tabular", "tabular*", "tikzpicture")


def strip_comments(tex: str) -> str:
    """Drop `%` comments, honouring the escaped `\\%`."""
    out = []
    for line in tex.splitlines():
        # An unescaped % starts a comment. Count the backslashes before it.
        m = re.search(r"(?<!\\)%", line)
        out.append(line[: m.start()] if m else line)
    return "\n".join(out)


def block_spans(tex: str, env: str) -> list[tuple[int, int]]:
    """Character spans of every `\\begin{env}...\\end{env}`."""
    spans = []
    for m in re.finditer(rf"\\begin\{{{re.escape(env)}\}}(.*?)\\end\{{{re.escape(env)}\}}",
                         tex, flags=re.DOTALL):
        spans.append((m.start(), m.end()))
    return spans


def mask(text: str, spans: list[tuple[int, int]]) -> str:
    """Blank out the given spans, preserving length so offsets stay meaningful."""
    chars = list(text)
    for a, b in spans:
        for i in range(a, b):
            if chars[i] != "\n":
                chars[i] = " "
    return "".join(chars)


def prose_words(text: str) -> int:
    """Count words in running text only."""
    t = re.sub(r"\\[a-zA-Z@]+\*?(\[[^\]]*\])?", " ", text)      # commands
    t = re.sub(r"[{}$&~^_\\]", " ", t)                           # markup
    t = re.sub(r"\s+", " ", t)
    return len([w for w in t.split() if any(c.isalnum() for c in w)])


def main() -> int:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "paper/main.tex")
    raw = path.read_text(encoding="utf-8")
    tex = strip_comments(raw)

    # Floats and display math are reported per section but excluded from prose words.
    float_spans = []
    for env in FLOAT_ENVS:
        float_spans += block_spans(tex, env)
    display_spans = block_spans(tex, "equation") + block_spans(tex, "align") + \
        block_spans(tex, "equation*") + block_spans(tex, "align*")
    tabular_spans = []
    for env in BODY_ENVS:
        tabular_spans += block_spans(tex, env)

    # Section boundaries.
    heads = [(m.start(), m.group(1), m.group(2)) for m in
             re.finditer(r"\\(section|subsection)\{([^}]*)\}", tex)]
    heads.append((len(tex), "_END", ""))

    # Mask ONCE on the full text, then slice: float spans are absolute offsets, so masking a
    # section substring with them would address the wrong characters.
    masked = mask(tex, float_spans)

    print(f"{'section':<52}{'words':>7}{'eq':>5}{'tab':>5}{'fig':>5}")
    print("-" * 75)
    total_w = 0
    for i in range(len(heads) - 1):
        a, kind, title = heads[i]
        b = heads[i + 1][0]
        seg = tex[a:b]
        nw = prose_words(masked[a:b])
        neq = len(block_spans(seg, "equation")) + len(block_spans(seg, "align")) + \
            len(block_spans(seg, "equation*")) + len(block_spans(seg, "align*"))
        ntab = sum(1 for s in block_spans(seg, "table") ) + sum(1 for s in block_spans(seg, "table*"))
        nfig = sum(1 for s in block_spans(seg, "figure")) + sum(1 for s in block_spans(seg, "figure*"))
        prefix = "  " if kind == "subsection" else ""
        total_w += nw
        print(f"{prefix}{title[:50]:<52}{nw:>7}{neq:>5}{ntab:>5}{nfig:>5}")

    # Preamble/abstract/front matter: everything before the first \section.
    first = heads[0][0] if heads else len(tex)
    front = tex[:first]
    fw = prose_words(masked[:first])
    print("-" * 75)
    print(f"{'FRONT MATTER (abstract + index terms)':<52}{fw:>7}")
    print(f"{'SECTION PROSE TOTAL':<52}{total_w:>7}")
    print(f"{'GRAND PROSE TOTAL':<52}{total_w + fw:>7}")

    print()
    print("floats and display math")
    print("-" * 75)
    for env in FLOAT_ENVS:
        n = len(block_spans(tex, env))
        if n:
            print(f"  {env:<12} {n}")
    print(f"  {'display eq':<12} {len(display_spans)}")
    # float environments that pull in an external file
    inputs = re.findall(r"\\input\{([^}]*)\}", tex)
    input_label = "\\input"
    print(f"  {input_label:<12} {len(inputs)}: {', '.join(sorted(set(inputs)))}")
    n_cites = len(re.findall(r"\\cite[a-z]*\{", tex))
    print(f"  {'cite calls':<12} {n_cites}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
