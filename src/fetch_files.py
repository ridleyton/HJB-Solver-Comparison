"""
Persistently fetch a small list of files from a GitHub repo over a slow, flaky link.

Context: this machine cannot use git (Schannel cannot acquire credentials) and the outbound TLS
link to GitHub runs at roughly 10 KB/s with frequent stalls. Python's own OpenSSL works, so this
fetches over raw.githubusercontent.com, one file at a time, retrying a lot and writing each file
as soon as it arrives so progress survives an interruption.

Usage: python src/fetch_files.py --repo owner/name --dest DIR --files a.py b.py ...
       python src/fetch_files.py --repo owner/name --dest DIR --from-list list.txt
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

BRANCHES = ["main", "master"]


def fetch_one(repo: str, path: str, dest: Path, tries: int = 8, timeout: int = 45) -> bool:
    target = dest / path
    if target.exists() and target.stat().st_size > 0:
        print(f"  have  {path} ({target.stat().st_size:,} B)", flush=True)
        return True
    for br in BRANCHES:
        url = f"https://raw.githubusercontent.com/{repo}/{br}/{path}"
        for attempt in range(1, tries + 1):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    blob = r.read()
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(blob)
                print(f"  ok    {path} ({len(blob):,} B)", flush=True)
                return True
            except Exception as e:
                msg = f"{type(e).__name__}"
                print(f"  retry {path} {attempt}/{tries} ({msg})", flush=True)
                time.sleep(1.5)
    print(f"  FAIL  {path}", flush=True)
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--files", nargs="*", default=[])
    ap.add_argument("--from-list", default=None)
    args = ap.parse_args()

    files = list(args.files)
    if args.from_list:
        files += [ln.strip() for ln in Path(args.from_list).read_text().splitlines()
                  if ln.strip() and not ln.startswith("#")]

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    print(f"fetching {len(files)} files into {dest}/", flush=True)
    ok = sum(fetch_one(args.repo, f, dest) for f in files)
    print(f"\n{ok}/{len(files)} files present", flush=True)
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    sys.exit(main())
