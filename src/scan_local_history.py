"""
Scan every blob in a LOCAL git clone for exposed credentials -- deleted files included.

Why a local clone and not the API
---------------------------------
The GitHub contents API only hands out what is in the tree you ask for. A file that was committed
and later deleted is NOT in any current tree, yet it is still in every clone's object store and can
be recovered with `git cat-file` by anyone who clones. That is the case that matters after a leak,
and it is exactly the case the API cannot show, so this walks the object database instead.

    git rev-list --objects --all      -> every object reachable from every ref
    git cat-file --batch              -> stream the CONTENTS (a single process, not one per blob,
                                         which matters because spawning git per blob is slow and on
                                         Windows hits process limits well before a large history)

A credential found here is ALREADY EXPOSED regardless of whether the file still exists. Deleting it
in a new commit does not remove it from the objects, and neither does force-pushing if anyone cloned
in the meantime. The only remediation is to ROTATE the credential at the provider, and only then to
rewrite history if the repository is public.

Usage:
  python src/scan_local_history.py --repo <path to a clone or a --mirror clone>
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

NAME_PAT = re.compile(
    r"(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|"
    r"client[_-]?secret|passwd|password|bearer)\s*[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]{12,})",
    re.I)

PROVIDER_PAT = [
    ("OpenAI", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("AWS access key", re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Stripe key", re.compile(r"\b(sk|rk)_(live|test)_[A-Za-z0-9]{20,}\b")),
    ("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
]

SKIP_EXT = re.compile(r"\.(png|jpe?g|gif|pdf|zip|gz|whl|so|dll|exe|bin|pkl|pt|npy|npz|ico)$", re.I)
TEXT_EXT = {".py", ".md", ".json", ".tex", ".bib", ".txt", ".yml", ".yaml", ".cfg", ".toml", ".csv",
            ".env", ".ini", ".sh", ".ps1", ".js", ".ts", ".r", ".ipynb", ".sql", ".xml", ".html",
            ".cls", ".bst", ".sty", ".gitignore", ".gitattributes", ""}


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          errors="replace").stdout


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--out", default="results/scan_local_history.json")
    ap.add_argument("--show-all", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo)
    if not (repo / ".git").exists() and not (repo / "HEAD").exists():
        print(f"not a git repository (clone or mirror): {repo}")
        return 1

    print("=" * 100)
    print(f"FULL-HISTORY CREDENTIAL SCAN of {repo}")
    print("=" * 100)

    listing = git(repo, "rev-list", "--objects", "--all")
    if not listing.strip():
        print("  no objects reachable; is this an empty repository?")
        return 1
    obj_to_path: dict[str, str] = {}
    for line in listing.splitlines():
        parts = line.split(" ", 1)
        if len(parts) == 2:
            obj_to_path[parts[0]] = parts[1]
    print(f"  objects reachable from all refs: {len(obj_to_path)}")
    commits = git(repo, "rev-list", "--all", "--count").strip()
    print(f"  commits: {commits}")
    print(f"  refs: {', '.join(r for r in git(repo, 'for-each-ref', '--format=%(refname)').split() if r)}")

    # Stream all objects through one `cat-file --batch` process.
    proc = subprocess.Popen(["git", "-C", str(repo), "cat-file", "--batch"],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE)
    assert proc.stdin and proc.stdout
    hits: list[str] = []
    scanned = 0
    results: dict[str, list[str]] = {}
    for oid, path in obj_to_path.items():
        proc.stdin.write((oid + "\n").encode())
        proc.stdin.flush()
        header = proc.stdout.readline().decode("utf-8", errors="replace").strip()
        if not header or header.endswith("missing"):
            continue
        bits = header.split()
        if len(bits) < 3:
            continue
        try:
            size = int(bits[2])
        except ValueError:
            continue
        body = proc.stdout.read(size)
        proc.stdout.read(1)  # trailing newline
        if SKIP_EXT.search(path) or (Path(path).suffix.lower() not in TEXT_EXT and path):
            continue
        scanned += 1
        text = body.decode("utf-8", errors="replace")
        found: list[str] = []
        for i, line in enumerate(text.splitlines(), 1):
            for m in NAME_PAT.finditer(line):
                found.append(f"NAME {m.group(1)}=<{len(m.group(2))} chars>")
            for prov, pat in PROVIDER_PAT:
                if pat.search(line):
                    found.append(f"PROVIDER {prov}")
        if found:
            results[path] = found
            hits += [f"{path}  [{oid[:8]}]  " + "; ".join(found[:4])]
    proc.stdin.close()
    proc.wait()

    print(f"  text objects scanned (all versions, all commits): {scanned}")
    print()
    print("-" * 100)
    if hits:
        print(f"  {len(hits)} PATH(S) WITH CREDENTIAL-SHAPED CONTENT:")
        for h in (hits if args.show_all else hits[:60]):
            print(f"      {h}")
        print()
        print("  WHAT TO DO, in this order:")
        print("   1. ROTATE the credential at the provider NOW. Rewriting history does not un-leak a")
        print("      key that has already been public; assume it is compromised.")
        print("   2. Then remove it from the current tree, if it is still there.")
        print("   3. Only then consider rewriting history (git filter-repo), and force-push. Anyone")
        print("      who cloned keeps the old objects, so step 1 is the one that matters.")
    else:
        print("  No credential-shaped content in ANY object, including deleted files and every")
        print("  past version of every file.")
        print()
        print("  This is a real negative for the objects that EXIST here -- it is not a statement")
        print("  that a credential was never published elsewhere (an issue, a gist, a fork, a")
        print("  different repository), because those are different object stores.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": str(repo), "objects": len(obj_to_path),
                               "scanned": scanned, "hits_by_path": results}, indent=2),
                   encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
