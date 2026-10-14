"""
Compile the manuscript with the local TeX Live installation.

Why this is a separate build directory
--------------------------------------
The submission sources must keep `\\bibliographystyle{IEEEtran}`, but `IEEEtran.bst` is not in
the local TeX Live tree (and `tlmgr install` cannot run here because it cannot write to HKCU).
Rather than edit the submission source to work around a local limitation, this script mirrors
the paper into `paper_build/` and substitutes TeX Live's `ieeetr.bst` -- a standard IEEE
Transactions style -- for the local check only. The compiled PDF is therefore representative
of layout, length and float placement, but the reference list formatting may differ slightly
from what the official CISP-BMEI template produces.

Verified against a local TeX Live 2026 installation (pdfTeX 1.40.29). `IEEEtran.cls` is not in the
TeX tree either, so a local copy is used; the script finds both through `TEXLIVE_BIN` and
`IEEE_CLS_SRC` in the environment, falling back to the search path, so no machine-specific location
is recorded here. See README.md for the two variables to set.

Usage: python src/build_paper.py [--engine pdflatex] [--keep]
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

SRC = Path("paper")
BUILD = Path("paper_build")


def _which_dir(exe: str) -> Path | None:
    found = shutil.which(exe)
    return Path(found).parent if found else None


# These three used to be absolute paths to one machine (D:\texlive\..., D:\TeX_02\...), which made
# the manuscript build work for exactly one person and leaked that person's disk layout into a public
# repository. The literals are REMOVED rather than demoted to a fallback: a fallback that still names
# someone else's directory solves nothing for a reviewer and still publishes the path.
#
# Resolution order, most portable first:
#   * an environment variable, for anyone with a non-standard layout;
#   * the standard location -- PATH for the engine, or the class file that ships inside paper/;
#   * failing both, None, and the caller reports what is missing instead of failing on a path that
#     belongs to another machine.
def _env_path(var: str) -> Path | None:
    v = os.environ.get(var)
    return Path(v) if v else None


def _find_exe(name: str) -> str | None:
    """Locate a TeX binary on PATH, or inside TEXLIVE_BIN.

    The second half matters and was missing at first. `TEXLIVE_BIN` is set PRECISELY when the TeX
    bin directory is NOT on PATH -- that is the situation it exists to solve -- so looking only at
    PATH meant the engine was found through the variable while `kpsewhich`, `bibtex` and everything
    else in the same directory were reported as unavailable. Measured: the build ran, found
    `pdflatex` via TEXLIVE_BIN, and then warned that `ieeetr.bst` could not be resolved, even though
    `kpsewhich ieeetr.bst` in that very directory answers correctly.
    """
    found = shutil.which(name)
    if found:
        return found
    if TEXLIVE_BIN is not None:
        for cand in (name, f"{name}.exe"):
            p = TEXLIVE_BIN / cand
            if p.exists():
                return str(p)
    return None


TEXLIVE_BIN = _env_path("TEXLIVE_BIN") or _which_dir("pdflatex")
_KPSEWHICH = _find_exe("kpsewhich")


def _class_file() -> Path | None:
    """The IEEEtran class to use: an explicit override, else the copy that ships with the paper."""
    explicit = _env_path("IEEE_CLS")
    if explicit:
        return explicit
    bundled = SRC / "IEEEtran.cls"
    return bundled if bundled.exists() else None


IEEE_CLS_SRC = _class_file()


def _bst_file() -> Path | None:
    explicit = _env_path("IEEE_BST")
    if explicit:
        return explicit
    if _KPSEWHICH:
        out = subprocess.run([_KPSEWHICH, "ieeetr.bst"], capture_output=True, text=True).stdout
        if out.strip():
            return Path(out.strip())
    return None


IEEE_BST_SUBSTITUTE = _bst_file()


# A fixed build timestamp, so that the PDF is byte-reproducible.
#
# WHY THIS EXISTS: pdfTeX embeds /CreationDate and /ModDate, so two builds of IDENTICAL sources
# differ in ~60 bytes -- and only there -- giving different SHA-256 hashes for the same document.
# Measured directly: same source, two consecutive builds, 284825 bytes each, 60 differing byte
# positions, every one of them inside the two date strings. That makes the PDF hash useless as an
# integrity check, which is precisely how it was used before this was noticed.
#
# pdflatex honours SOURCE_DATE_EPOCH only when FORCE_SOURCE_DATE is also set. The value is fixed
# rather than derived from the clock, which is the whole point.
BUILD_EPOCH = "1767225600"          # 2026-01-01T00:00:00Z


def _run(cmd: list[str], cwd: Path) -> tuple[int, str]:
    env = dict(os.environ)
    env["SOURCE_DATE_EPOCH"] = BUILD_EPOCH
    env["FORCE_SOURCE_DATE"] = "1"
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, errors="replace", env=env)
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def stage(substitute_bst: bool = True) -> None:
    if BUILD.exists():
        shutil.rmtree(BUILD)
    (BUILD / "tables").mkdir(parents=True)
    (BUILD / "figures").mkdir(parents=True)

    tex = (SRC / "main.tex").read_text(encoding="utf-8")
    # The submission source uses \graphicspath{{../figures/}} because paper/ and figures/ are
    # siblings. In the build directory the figures are in ./figures, so the path is rewritten
    # for the local compile only; the submission source is left untouched.
    if r"\graphicspath{{../figures/}}" in tex:
        tex = tex.replace(r"\graphicspath{{../figures/}}", r"\graphicspath{{./figures/}}")
    else:
        print("  WARNING: expected \\graphicspath not found; figures may not resolve")

    marker = r"\bibliographystyle{IEEEtran}"
    if substitute_bst:
        # Match only a REAL command line: leading whitespace, the command, then end of line.
        # A blind str.replace() also rewrites the marker where it merely *appears*, and this bit
        # us once: the preamble comment documenting the substitution was itself substituted, so
        # the replacement's uncommented \bibliographystyle{ieeetr} executed before \documentclass
        # and the build died with "Missing \begin{document}" at the line after the class.
        pattern = re.compile(r"^([ \t]*)" + re.escape(marker) + r"[ \t]*$", re.MULTILINE)
        if not pattern.search(tex):
            print(f"  WARNING: no standalone {marker} line found; "
                  f"not substituting the bibliography style")
        else:
            tex = pattern.sub(
                lambda m: (m.group(1) + "% [local build only] IEEEtran.bst is absent from the\n"
                           + m.group(1) + "% local TeX Live tree; ieeetr.bst is a standard IEEE "
                           "style\n"
                           + m.group(1) + "% used here as a stand-in. The submission source keeps\n"
                           + m.group(1) + "% \\bibliographystyle{IEEEtran}.\n"
                           + m.group(1) + "\\bibliographystyle{ieeetr}"),
                tex, count=1)
    (BUILD / "main.tex").write_text(tex, encoding="utf-8")

    shutil.copy(SRC / "refs.bib", BUILD / "refs.bib")
    for f in sorted((SRC / "tables").glob("*.tex")):
        shutil.copy(f, BUILD / "tables" / f.name)
    # Figures that are LaTeX source rather than PDF: the methodology flowchart is drawn with TikZ
    # so that its fonts match the paper and no binary asset has to be regenerated. Without this
    # the build fails with "File `fig_method.tex' not found".
    latex_figs = sorted(SRC.glob("fig_*.tex"))
    for f in latex_figs:
        shutil.copy(f, BUILD / f.name)
    if latex_figs:
        print(f"  staged {len(latex_figs)} LaTeX figure(s): "
              f"{', '.join(f.name for f in latex_figs)}")
    figs = sorted(Path("figures").glob("*.pdf"))
    if not figs:
        print("  WARNING: no figure PDFs found in figures/ -- run src/make_figures.py first")
    for f in figs:
        shutil.copy(f, BUILD / "figures" / f.name)
    print(f"  staged {len(figs)} figures")

    # Every one of these may be None now that the machine-specific fallbacks are gone, so each is
    # guarded. Without the guards a reviewer would get an AttributeError on `None.exists()` -- a
    # hard failure that says nothing about what to install, which is the exact outcome the removal
    # of the absolute paths was meant to avoid.
    if IEEE_CLS_SRC is not None and IEEE_CLS_SRC.exists():
        shutil.copy(IEEE_CLS_SRC, BUILD / "IEEEtran.cls")
    else:
        print("  WARNING: no IEEEtran.cls resolved. Set IEEE_CLS, or place IEEEtran.cls in paper/. "
              "The build continues if a system-wide copy is available to the engine.")
    if substitute_bst and IEEE_BST_SUBSTITUTE is not None and IEEE_BST_SUBSTITUTE.exists():
        shutil.copy(IEEE_BST_SUBSTITUTE, BUILD / "ieeetr.bst")
    elif substitute_bst:
        print("  WARNING: no ieeetr.bst resolved (set IEEE_BST, or install a TeX tree providing it). "
              "The reference list may then keep whatever style the submission source names.")


def summarise(log: str) -> dict:
    out: dict = {}
    m = re.search(r"Output written on .*?\((\d+) pages?", log)
    out["pages"] = int(m.group(1)) if m else None
    out["overfull"] = re.findall(r"Overfull \\[hv]box \(([\d.]+)pt too \w+\)[^\n]*", log)
    out["underfull"] = len(re.findall(r"Underfull \\[hv]box", log))
    out["errors"] = re.findall(r"^! (.+)$", log, flags=re.MULTILINE)
    out["undefined_refs"] = sorted(set(re.findall(r"Reference `([^']+)' on page", log)))
    out["undefined_cites"] = sorted(set(re.findall(r"Citation `([^']+)' undefined", log)))
    out["missing_files"] = sorted(set(re.findall(r"File `([^']+)' not found", log)))
    out["latex_warnings"] = re.findall(r"LaTeX Warning: ([^\n]{0,120})", log)
    return out


def main() -> int:
    global BUILD
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", default="pdflatex")
    ap.add_argument("--keep", action="store_true", help="keep the build directory")
    ap.add_argument("--no-bst-substitute", action="store_true")
    ap.add_argument("--build-dir", default="paper_build",
                    help=("staging directory. Override when the default output is held open "
                          "by a PDF viewer: Acrobat locks the file on Windows and the rebuild "
                          "then fails with PermissionError."))
    args = ap.parse_args()
    BUILD = Path(args.build_dir)

    if TEXLIVE_BIN is None or not TEXLIVE_BIN.exists():
        print("TeX Live not found.")
        print("  The engine is located on PATH, or via the TEXLIVE_BIN environment variable "
              "pointing at the")
        print("  directory holding pdflatex (for example /usr/local/texlive/2026/bin/x86_64-linux).")
        print("  This script no longer carries a machine-specific fallback path: a build that works")
        print("  on exactly one computer is not a reproducible build.")
        return 2

    print(f"staging {BUILD}/")
    try:
        stage(substitute_bst=not args.no_bst_substitute)
    except PermissionError as e:
        print(f"\n  Cannot write into {BUILD}/: {e}")
        print(f"  A PDF viewer probably has {BUILD}/main.pdf open. Close it, or re-run with "
              f"--build-dir {BUILD.name}2")
        return 3

    engine = str(TEXLIVE_BIN / f"{args.engine}.exe")
    bibtex = str(TEXLIVE_BIN / "bibtex.exe")
    base = ["-interaction=nonstopmode", "-file-line-error", "main.tex"]

    log = ""
    last_pass = ""
    # Pass 1: create .aux for bibtex
    rc, o = _run([engine, *base], BUILD); log += o; last_pass = o
    print(f"  pass 1 ({args.engine}): rc={rc}")
    # bibtex
    rc_b, o = _run([bibtex, "main"], BUILD); log += o
    print(f"  bibtex: rc={rc_b}")
    if rc_b != 0:
        print("  " + "\n  ".join(o.strip().splitlines()[-12:]))
    # Passes 2-3: resolve refs and citations
    for i in (2, 3):
        rc, o = _run([engine, *base], BUILD); log += o
        last_pass = o
        print(f"  pass {i} ({args.engine}): rc={rc}")

    (BUILD / "build.log").write_text(log, encoding="utf-8")
    # Diagnose the FINAL pass only: earlier passes legitimately report undefined references
    # before the .aux and .bbl exist, so parsing the concatenated log would report false
    # failures.
    s = summarise(last_pass)

    print("\n" + "=" * 72)
    pdf = BUILD / "main.pdf"
    print(f"  PDF            : {pdf if pdf.exists() else 'NOT PRODUCED'}")
    print(f"  pages          : {s['pages']}")
    print(f"  errors         : {len(s['errors'])}")
    for e in s["errors"][:10]:
        print(f"      ! {e}")
    print(f"  undefined refs : {s['undefined_refs'] or 'none'}")
    print(f"  undefined cites: {s['undefined_cites'] or 'none'}")
    print(f"  missing files  : {s['missing_files'] or 'none'}")
    print(f"  overfull boxes : {len(s['overfull'])}")
    for ov in sorted(s["overfull"], key=lambda x: -float(x))[:8]:
        print(f"      {ov} pt")
    print(f"  underfull boxes: {s['underfull']}")
    print("=" * 72)

    if not args.keep:
        for ext in (".aux", ".bbl", ".blg", ".out", ".synctex.gz", ".toc", ".lof", ".lot"):
            f = BUILD / f"main{ext}"
            if f.exists():
                f.unlink()
    return 0 if (BUILD / "main.pdf").exists() and not s["errors"] else 1


if __name__ == "__main__":
    sys.exit(main())
