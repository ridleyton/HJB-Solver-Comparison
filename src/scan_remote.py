"""
Scan a PUBLIC GitHub repository for exposed credentials -- including in history.

Why this exists
---------------
A repository that has already been pushed cannot be un-pushed by editing files. If a credential was
ever committed, it lives in every clone's object store until the history is rewritten, and editing
the current tree does nothing about that. So this scans BOTH the current tree and every blob in the
commit history, because those are different sets and only the second one matters if the file was
later deleted.

What it can and cannot do. It can find credential-shaped strings and tell you WHERE they are, which
is what decides the response. It cannot tell you whether a key is still valid, and it cannot revoke
anything -- the only remediation for a leaked credential is rotation at the provider, followed by
history rewriting if the repository is public. A clean report is not proof that nothing leaked
earlier, only that nothing credential-shaped is in the objects it could reach.

Unauthenticated, so it works on public repositories without a token. A private repository will
return 404 rather than 403, because GitHub does not distinguish the two.

Usage:
  python src/scan_remote.py --repo ridleyton/HJB-Solver-Comparison
  python src/scan_remote.py --repo <owner>/<name> --history
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

API = "https://api.github.com"
UA = {"User-Agent": "release-credential-scan", "Accept": "application/vnd.github+json"}

# Credential names. Kept broad: the cost of a false positive is one line of reading, the cost of a
# miss is a live key in a public repository.
NAME_PAT = re.compile(
    r"(api[_-]?key|apikey|secret[_-]?key|access[_-]?token|auth[_-]?token|private[_-]?key|"
    r"client[_-]?secret|passwd|password|bearer)\s*[\"']?\s*[:=]\s*[\"']?([A-Za-z0-9_\-\.]{12,})",
    re.I)

# Provider-specific shapes. These are worth naming because a matched provider tells you exactly which
# console to open to revoke it.
PROVIDER_PAT = [
    ("OpenAI", re.compile(r"\bsk-[A-Za-z0-9]{20,}\b")),
    ("Anthropic", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{20,}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b")),
    ("GitHub fine-grained", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{50,}\b")),
    ("AWS access key", re.compile(r"\b(AKIA|ASIA)[A-Z0-9]{16}\b")),
    ("Google API key", re.compile(r"\bAIza[A-Za-z0-9_\-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{10,}\b")),
    ("Stripe key", re.compile(r"\b(sk|rk)_(live|test)_[A-Za-z0-9]{20,}\b")),
    ("Alpha Vantage", re.compile(r"\b[A-Z0-9]{16}\b(?=[^A-Za-z0-9]*alphavantage)", re.I)),
    ("Private key block", re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("JWT", re.compile(r"\beyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b")),
]

# Paths that legitimately contain key-shaped strings and are noise.
SKIP_NAME = re.compile(r"(\.png|\.jpg|\.jpeg|\.gif|\.pdf|\.zip|\.ipynb_checkpoints)$", re.I)

TEXT_EXT = {".py", ".md", ".json", ".tex", ".bib", ".txt", ".yml", ".yaml", ".cfg", ".toml", ".csv",
            ".env", ".ini", ".sh", ".ps1", ".js", ".ts", ".r", ".ipynb", ".sql", ".xml", ".html"}


def get(url: str) -> tuple[int, object]:
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        return e.code, None
    except Exception as e:                                              # noqa: BLE001
        return -1, str(e)


def scan_text(text: str, where: str) -> list[str]:
    hits: list[str] = []
    for i, line in enumerate(text.splitlines(), 1):
        for m in NAME_PAT.finditer(line):
            hits.append(f"{where}:{i}  NAME  {m.group(1)}=<{len(m.group(2))} chars>")
        for prov, pat in PROVIDER_PAT:
            if pat.search(line):
                hits.append(f"{where}:{i}  PROVIDER  {prov}")
    return hits


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="owner/name")
    ap.add_argument("--history", action="store_true",
                    help="also walk the commit tree history, not just the default branch tip")
    ap.add_argument("--out", default="results/scan_remote.json")
    args = ap.parse_args()

    print("=" * 100)
    print(f"CREDENTIAL SCAN of the PUBLIC repository {args.repo}")
    print("=" * 100)

    st, meta = get(f"{API}/repos/{args.repo}")
    if meta is None or not isinstance(meta, dict):
        print(f"  cannot read the repository: HTTP {st}")
        if st == 404:
            print("  GitHub returns 404 for a private repo as well as a missing one.")
        return 1
    print(f"  visibility   : {meta.get('visibility')}")
    print(f"  default ref  : {meta.get('default_branch')}")
    print(f"  last pushed  : {meta.get('pushed_at')}")
    print(f"  size on disk : {meta.get('size')} KB")
    if meta.get("visibility") != "public":
        print("\n  The repository is not public, so nothing here is exposed to the internet. The scan")
        print("  below still matters if it ever becomes public or is shared.")

    ref = meta.get("default_branch") or "main"
    st, tree = get(f"{API}/repos/{args.repo}/git/trees/{ref}?recursive=1")
    if tree is None:
        print(f"  cannot list the tree: HTTP {st}")
        return 1
    entries = [e for e in tree.get("tree", []) if e.get("type") == "blob"]
    print(f"  files in tree: {len(entries)}")

    hits: list[str] = []
    scanned = 0
    sizes: list[tuple[int, str]] = []
    for e in entries:
        path = e["path"]
        sizes.append((e.get("size", 0), path))
        if SKIP_NAME.search(path):
            continue
        suffix = Path(path).suffix.lower()
        if suffix and suffix not in TEXT_EXT:
            continue
        st, blob = get(f"{API}/repos/{args.repo}/contents/{path}?ref={ref}")
        if blob is None or not isinstance(blob, dict) or "content" not in blob:
            continue
        try:
            text = base64.b64decode(blob["content"]).decode("utf-8", errors="replace")
        except Exception:                                               # noqa: BLE001
            continue
        scanned += 1
        hits += scan_text(text, path)

    print(f"  text files scanned: {scanned}")
    sizes.sort(reverse=True)
    print("  largest files: " + ", ".join(f"{p} ({s / 1e6:.1f} MB)" for s, p in sizes[:5]))

    if args.history:
        st, commits = get(f"{API}/repos/{args.repo}/commits?per_page=100")
        if isinstance(commits, list):
            print(f"\n  commits on {ref}: {len(commits)}")
            seen: set[str] = set()
            for c in commits:
                sha = c["sha"]
                st, ct = get(f"{API}/repos/{args.repo}/git/trees/{sha}?recursive=1")
                if not isinstance(ct, dict):
                    continue
                for e in ct.get("tree", []):
                    if e.get("type") != "blob" or e["path"] in seen:
                        continue
                    seen.add(e["path"])
            print(f"  distinct paths ever present in history: {len(seen)}")
            print("  NOTE: the API path above lists tips, not blob CONTENTS of deleted files. To scan")
            print("  deleted-file contents, clone with full history and run the local scan on each")
            print("  blob -- this script reports what the API will hand out unauthenticated.")

    print()
    print("-" * 100)
    if hits:
        print(f"  {len(hits)} CREDENTIAL-SHAPED MATCH(ES):")
        for h in hits[:60]:
            print(f"      {h}")
        if len(hits) > 60:
            print(f"      ... and {len(hits) - 60} more")
    else:
        print("  No credential-shaped strings found in the readable files.")
        print("  This is NOT proof that nothing was ever exposed: it covers the default branch tip")
        print("  as the API reports it, not every blob in every past commit.")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"repo": args.repo, "visibility": meta.get("visibility"),
                               "files_in_tree": len(entries), "scanned": scanned,
                               "hits": hits, "largest": sizes[:20]}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
