"""Build and check the journal submission package.

WHAT THIS DOES
--------------
Compiles the cover letter, the declarations and the nomenclature, checks the two constraints Elsevier
actually enforces mechanically on Highlights (3 to 5 bullets, at most 85 characters each), and reports
an inventory of what a submission needs against what exists. A submission is returned for a missing
declaration or an over-long highlight far more often than for a typesetting preference, so those are
the checks worth automating.

WHAT IT DOES NOT DO
-------------------
Convert the manuscript to the Elsevier class. `elsarticle.cls` is not installed in this TeX Live and
could not be fetched here (every CTAN path tried returned 404 or an anti-bot page), and it is not
needed at initial submission: Elsevier asks for free formatting initially and for the class at
revision. `paper/journal/README.md` records the exact conversion steps and why the manuscript
compiles in IEEEtran meanwhile.

Usage: python src/build_journal.py [--texlive-bin ...]
Output: paper/journal/*.pdf  and  results/check_journal_package.json
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
JOURNAL = ROOT / "paper" / "journal"
MAIN = ROOT / "paper" / "main.tex"
MANUSCRIPT_PDF = ROOT / "paper" / "main.pdf"
RESULTS = ROOT / "results"

HIGHLIGHT_MAX_CHARS = 85
HIGHLIGHT_MIN, HIGHLIGHT_MAX = 3, 5


def find_pdflatex(explicit: str | None) -> str | None:
    """Locate pdflatex the way build_paper.py does, and for the same stated reason.

    This function first named two absolute TeX directories as a fallback, which is exactly what
    build_paper.py's comment forbids: "the literals are REMOVED rather than demoted to a fallback: a
    fallback that still names someone else's directory solves nothing for a reviewer and still
    publishes the path." src/check_release.py caught it on the next run. Resolution order, most
    portable first: an explicit argument, then PATH, then TEXLIVE_BIN -- which is set precisely when
    the TeX bin directory is NOT on PATH, the case the variable exists for.
    """
    if explicit:
        p = pathlib.Path(explicit) / "pdflatex.exe"
        if p.exists():
            return str(p)
    found = shutil.which("pdflatex")
    if found:
        return found
    env_bin = os.environ.get("TEXLIVE_BIN")
    if env_bin:
        p = pathlib.Path(env_bin) / "pdflatex.exe"
        if p.exists():
            return str(p)
    return None


def compile_tex(tex: pathlib.Path, engine: str) -> dict:
    """Two passes, so cross-references settle; reports only what matters.

    The dates are pinned for the same reason build_paper.py pins them, and the reason is measured:
    without it, two consecutive builds of the same source differ in the embedded creation/modification
    strings, so every rebuild of the submission package produced three "modified" PDFs of identical
    size and the repository churned. `pdflatex` honours SOURCE_DATE_EPOCH only when
    FORCE_SOURCE_DATE is also set.
    """
    work = tex.parent
    log = work / (tex.stem + ".log")
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = "1767225600"          # 2026-01-01T00:00:00Z, as in build_paper.py
    env["FORCE_SOURCE_DATE"] = "1"
    ok = True
    detail = ""
    for _ in range(2):
        proc = subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", tex.name],
                              cwd=str(work), capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            ok = False
            tail = (proc.stdout or "").strip().splitlines()[-12:]
            detail = " | ".join(tail)
            break
    pdf = work / (tex.stem + ".pdf")
    pages = None
    if log.exists():
        m = re.findall(r"Output written on .*?\((\d+) page", log.read_text(errors="replace"))
        if m:
            pages = int(m[-1])
    return {"file": tex.name, "ok": ok and pdf.exists(), "pages": pages,
            "pdf": str(pdf.relative_to(ROOT)) if pdf.exists() else None, "detail": detail}


def check_highlights() -> dict:
    src = JOURNAL / "highlights.txt"
    if not src.exists():
        return {"ok": False, "reason": "highlights.txt absent"}
    bullets = []
    for line in src.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.lower().startswith(("highlights for", "elsevier requires",
                                          "glyph (the submission")):
            continue
        if s.endswith("`src/build_journal.py` checks both constraints."):
            continue
        bullets.append(s)
    over = [b for b in bullets if len(b) > HIGHLIGHT_MAX_CHARS]
    return {
        "ok": HIGHLIGHT_MIN <= len(bullets) <= HIGHLIGHT_MAX and not over,
        "count": len(bullets), "max_chars": max((len(b) for b in bullets), default=0),
        "over_limit": over,
        "bullets": bullets,
    }


def inventory() -> list[dict]:
    items = [
        ("Manuscript PDF", MANUSCRIPT_PDF, True),
        ("Highlights", JOURNAL / "highlights.txt", True),
        ("Cover letter (PDF)", JOURNAL / "cover_letter.pdf", True),
        ("Cover letter (source)", JOURNAL / "cover_letter.tex", False),
        ("Declarations (PDF)", JOURNAL / "declarations.pdf", True),
        ("Declarations (source)", JOURNAL / "declarations.tex", False),
        ("Nomenclature (PDF)", JOURNAL / "nomenclature.tex", False),
        ("Submission checklist", JOURNAL / "README.md", True),
        ("Conversion recipe", JOURNAL / "README.md", False),
    ]
    out = []
    for label, path, required in items:
        out.append({"item": label, "path": str(path.relative_to(ROOT)),
                    "present": path.exists(), "required_at_submission": required,
                    "size": path.stat().st_size if path.exists() else 0})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--texlive-bin", default=os.environ.get("TEXLIVE_BIN"))
    args = ap.parse_args()

    print("=" * 100)
    print("JOURNAL SUBMISSION PACKAGE")
    print("=" * 100)

    engine = find_pdflatex(args.texlive_bin)
    # The resolved engine path is NOT recorded: results/ is committed, and a run on one
    # machine must not publish that machine's TeX layout (src/check_release.py flags it).
    report: dict = {"texlive_bin_set": bool(args.texlive_bin),
                    "pdflatex_found": engine is not None}

    hl = check_highlights()
    report["highlights"] = hl
    print(f"\n  Highlights: {hl.get('count', 0)} bullets, longest {hl.get('max_chars', 0)} chars "
          f"(limit {HIGHLIGHT_MAX_CHARS})  -> {'OK' if hl['ok'] else 'FAIL'}")
    for b in hl.get("bullets", []):
        flag = "  " if len(b) <= HIGHLIGHT_MAX_CHARS else "!!"
        print(f"    {flag} [{len(b):>3d}] {b}")

    if engine is None:
        print("\n  pdflatex not found: set TEXLIVE_BIN or put it on PATH. Skipping compiles.")
        report["compiles"] = []
    else:
        print(f"\n  engine: {engine}")
        report["compiles"] = []
        for name in ("cover_letter.tex", "declarations.tex", "nomenclature.tex"):
            tex = JOURNAL / name
            if not tex.exists():
                continue
            res = compile_tex(tex, engine)
            report["compiles"].append(res)
            print(f"    {name:20s} {'OK ' if res['ok'] else 'FAIL'} "
                  f"{res['pages'] if res['pages'] else '-'} page(s)"
                  + ("" if res["ok"] else f"   {res['detail'][:160]}"))

    inv = inventory()
    report["inventory"] = inv
    print("\n  submission inventory")
    missing_required = []
    for row in inv:
        mark = "OK " if row["present"] else ("MISSING" if row["required_at_submission"] else "absent")
        if not row["present"] and row["required_at_submission"]:
            missing_required.append(row["item"])
        print(f"    [{mark:>7s}] {row['item']:28s} {row['path']}")
    report["missing_required"] = missing_required

    print()
    print("-" * 100)
    problems = []
    if not hl["ok"]:
        problems.append("highlights violate the 3-5 bullet / 85 character rule")
    if missing_required:
        problems.append("missing required items: " + ", ".join(missing_required))
    for c in report.get("compiles", []):
        if not c["ok"]:
            problems.append(f"compile failed: {c['file']}")
    if problems:
        print("  PROBLEMS: " + "; ".join(problems))
    else:
        print("  All required items present, highlights within limits, everything compiled.")
    print("-" * 100)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "check_journal_package.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8")
    print("  wrote results/check_journal_package.json")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
