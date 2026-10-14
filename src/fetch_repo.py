"""
Download and extract a GitHub repository using Python's TLS stack.

Why not `git clone`: on this machine git uses the Windows Schannel backend, which fails with
`schannel: AcquireCredentialsHandle failed: SEC_E_NO_CREDENTIALS` inside this process context.
TCP to port 443 connects fine and DNS resolves, so the network is reachable -- only git's TLS
provider is unusable. Python links its own OpenSSL, which works, so we fetch the codeload zip
directly.

Usage: python src/fetch_repo.py <owner/repo> [--dest DIR] [--branch BRANCH]
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CANDIDATE_BRANCHES = ["main", "master"]
MAX_TRIES = 6


def try_url(url: str, timeout: int = 120, tries: int = MAX_TRIES) -> bytes | None:
    """Fetch a URL, retrying. The TLS connection here is flaky: a given attempt may time out
    or die with SSL UNEXPECTED_EOF, but a retry often succeeds."""
    last = None
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fetch_repo)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                print(f"    HTTP 404 for {url}")
                return None          # a 404 is definitive; do not retry
            last = f"HTTP {e.code}"
        except Exception as e:
            last = f"{type(e).__name__}: {str(e)[:70]}"
        print(f"    attempt {attempt}/{tries}: {last}")
    print(f"    giving up on {url} ({last})")
    return None


def try_url_stream(url: str, out: Path, timeout: int = 300, tries: int = 4) -> bool:
    """Stream a large file to disk with retries, so a slow download can be resumed on retry."""
    for attempt in range(1, tries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (fetch_repo)"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                total = int(r.headers.get("Content-Length") or 0)
                got = 0
                with out.open("wb") as fh:
                    while True:
                        chunk = r.read(65536)
                        if not chunk:
                            break
                        fh.write(chunk)
                        got += len(chunk)
                        if total and got % (1024 * 1024) < 65536:
                            print(f"      {got/1e6:.1f}/{total/1e6:.1f} MB", flush=True)
                print(f"    downloaded {got:,} bytes")
                return True
        except Exception as e:
            print(f"    stream attempt {attempt}/{tries}: {type(e).__name__}: {str(e)[:70]}")
    return False


def list_repo_files(repo: str) -> list[str]:
    """Return the file paths in the repository via the git trees API."""
    for br in CANDIDATE_BRANCHES:
        raw = try_url(f"https://api.github.com/repos/{repo}/git/trees/{br}?recursive=1",
                      timeout=90, tries=4)
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        files = [t["path"] for t in data.get("tree", []) if t.get("type") == "blob"]
        if files:
            print(f"    branch '{br}': {len(files)} files")
            return files
    return []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("repo", help="owner/name, e.g. HeeJunYang09/hji-pi-pinn-repro")
    ap.add_argument("--dest", default=None)
    ap.add_argument("--branch", default=None)
    args = ap.parse_args()

    dest = Path(args.dest or args.repo.split("/")[-1])
    branches = [args.branch] if args.branch else CANDIDATE_BRANCHES
    dest.mkdir(parents=True, exist_ok=True)

    # Preferred route: stream the codeload zip, which gets everything in one connection.
    blob_file = dest / "_repo.zip"
    for br in branches:
        url = f"https://codeload.github.com/{args.repo}/zip/refs/heads/{br}"
        print(f"  trying zip {url}")
        if try_url_stream(url, blob_file, timeout=600, tries=4):
            print(f"  zip downloaded from branch '{br}'")
            with zipfile.ZipFile(blob_file) as z:
                names = z.namelist()
                z.extractall(dest)
            blob_file.unlink(missing_ok=True)
            print(f"  extracted {len(names)} entries into {dest}/")
            for n in sorted(names)[:50]:
                print(f"    {n}")
            if len(names) > 50:
                print(f"    ... and {len(names) - 50} more")
            return 0
        blob_file.unlink(missing_ok=True)

    # Fallback: enumerate via the API and fetch each file over raw.githubusercontent.com,
    # which is a different host and has been more reliable here than codeload.
    print("\n  zip route failed; falling back to per-file fetch")
    files = list_repo_files(args.repo)
    if not files:
        print("\nFAILED: no route to the repository worked.")
        return 1
    ok = 0
    for path in files:
        for br in branches:
            raw = try_url(
                f"https://raw.githubusercontent.com/{args.repo}/{br}/{path}",
                timeout=90, tries=3)
            if raw is not None:
                target = dest / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(raw)
                ok += 1
                print(f"    ok  {path} ({len(raw):,} B)")
                break
        else:
            print(f"    FAIL {path}")
    print(f"\n  fetched {ok}/{len(files)} files into {dest}/")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
