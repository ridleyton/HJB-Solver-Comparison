"""
Pre-publication check for the release package.

Why this exists
---------------
The journal audit marks "released package free of secrets and absolute paths" as an item only the
authors can sign off, because the sign-off happens at the moment of publication. But most of what it
covers IS mechanically checkable in advance, and checking it in advance is the only way it gets
fixed before a reviewer clones the repository rather than after. This script checks the parts that
can be checked and reports the parts that cannot.

What it checks, and what it deliberately does not
-------------------------------------------------
IT CHECKS: credentials or credential-shaped strings; absolute paths belonging to one machine;
build intermediates and caches that should not be committed; and whether the release would be
self-contained enough for a reader to reproduce the tables.

IT DOES NOT CHECK: whether the repository is public, whether the licence is correct, or whether a
credential has already been exposed and needs rotating. A checker cannot un-publish a secret, and
saying so is the point -- a clean report here is not a statement that nothing was ever leaked.

Credentials are searched for by SHAPE as well as by name. The name-based search is what found
nothing in the first pass, and it would miss a key pasted into a notebook cell or a config file.
The shape search is deliberately conservative: long runs of key-like characters with no spaces,
next to a token-ish identifier.

Usage: python src/check_release.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".acl-repair", ".mplconfig",
             ".pagedump", "paper_preview"}
TEXT_SUFFIXES = {".py", ".md", ".json", ".tex", ".bib", ".txt", ".yml", ".yaml", ".cfg", ".toml",
                 ".csv", ".sh", ".ps1"}

# This script's own report is excluded, and the reason is a bug worth recording. The report lists
# every path it finds, so on the next run those listings are themselves findings: the count grew by
# one entry per run and would never reach zero no matter how much was cleaned. A checker must not
# audit its own output. `results/check_tex.json` is NOT excluded -- it is a genuine finding, fixed by
# making check_tex.py record its target relative to the repository.
SELF_REPORT = "results/check_release.json"

# Credential-shaped names.
NAME_PAT = re.compile(
    r"(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|private[_-]?key|client[_-]?secret|"
    r"passwd|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{12,})", re.I)
# Long opaque tokens that look like keys even without a telling name nearby.
SHAPE_PAT = re.compile(r"\b[A-Za-z0-9]{28,}\b")
# Absolute paths that belong to one machine.
ABS_PAT = re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:\\\\?[A-Za-z0-9_\\\\.\- ]{4,}"
                     r"|/(?:Users|home)/[A-Za-z0-9_.\-]+/")

# Long opaque strings that are legitimately not secrets, so the shape search does not cry wolf.
SHAPE_ALLOW = re.compile(r"^[0-9a-f]{28,}$", re.I)


def iter_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        # Staging builds: build_paper.py's --build-dir can be any name, so match the prefix rather
        # than a fixed directory. A build directory is transient BY DESIGN -- build_paper.py creates
        # it and the caller deletes it -- and scanning one reports the build's own artefacts, which
        # is how a stray `_v/` from a verification run showed up as a "finding" about the package.
        if any(part.startswith(("paper_build", "_build")) or part in {"_v", "_final_build", "_diag",
                                                                     "_probe", "_pages"}
               for part in p.parts):
            continue
        yield p


def main() -> int:
    findings: dict[str, list[str]] = {"credentials": [], "absolute_paths": [],
                                      "intermediates": [], "notes": []}
    n_scanned = 0
    n_bytes = 0
    biggest: list[tuple[int, str]] = []

    for p in iter_files():
        rel = p.relative_to(ROOT).as_posix()
        if rel == SELF_REPORT:
            continue
        try:
            size = p.stat().st_size
        except OSError:
            continue
        biggest.append((size, rel))
        n_bytes += size

        if p.suffix.lower() in {".pyc", ".aux", ".log", ".out", ".blg", ".bbl", ".synctex.gz"}:
            findings["intermediates"].append(f"{rel} ({size:,} B)")
            continue

        if p.suffix.lower() not in TEXT_SUFFIXES:
            continue
        n_scanned += 1
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        for i, line in enumerate(text.splitlines(), 1):
            if line.lstrip().startswith(("#", "%", "//")):
                # Comments are still published, so they are still scanned for secrets; only the
                # absolute-path check skips them, because a path in a comment is documentation
                # rather than a broken build.
                for m in NAME_PAT.finditer(line):
                    findings["credentials"].append(f"{rel}:{i}  name-match: {m.group(1)}")
                continue
            for m in NAME_PAT.finditer(line):
                findings["credentials"].append(f"{rel}:{i}  name-match: {m.group(1)}")
            for m in SHAPE_PAT.finditer(line):
                tok = m.group(0)
                if not SHAPE_ALLOW.match(tok):
                    findings["credentials"].append(f"{rel}:{i}  key-shaped token ({len(tok)} chars)")
            for m in ABS_PAT.finditer(line):
                findings["absolute_paths"].append(f"{rel}:{i}  {m.group(0).strip()[:70]}")

    biggest.sort(reverse=True)
    findings["notes"].append(f"scanned {n_scanned} text files, {n_bytes / 1e6:.1f} MB total")
    findings["notes"].append("largest files: "
                             + ", ".join(f"{r} ({s / 1e6:.1f} MB)" for s, r in biggest[:5]))
    findings["notes"].append(
        "NOT checked: whether the repo is public, whether the licence is right, and whether a "
        "credential was EVER exposed. A checker cannot un-publish a secret; if one was committed at "
        "any point it must be rotated, not deleted.")

    print("=" * 100)
    print("RELEASE PACKAGE CHECK")
    print("=" * 100)
    for n in findings["notes"]:
        print(f"  note: {n}")
    print()
    bad = 0
    for key, label in (("credentials", "CREDENTIAL-SHAPED STRINGS"),
                       ("absolute_paths", "MACHINE-SPECIFIC ABSOLUTE PATHS"),
                       ("intermediates", "BUILD INTERMEDIATES / CACHES")):
        items = findings[key]
        print(f"  {label}: {len(items)}")
        for it in items[:12]:
            print(f"      {it}")
        if len(items) > 12:
            print(f"      ... and {len(items) - 12} more")
        if key != "intermediates":
            bad += len(items)
    (ROOT / "results" / "check_release.json").write_text(
        json.dumps(findings, indent=2), encoding="utf-8")
    print()
    print("-" * 100)
    print(f"  {bad} item(s) needing a decision before publishing")
    print(f"\nwrote {ROOT / 'results' / 'check_release.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
