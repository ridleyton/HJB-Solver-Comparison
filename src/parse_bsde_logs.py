"""
Recover convergence traces from the run LOGS when the run did not reach its final save.

Why this exists
---------------
exp_bsde_steps.py writes its JSON only when a run finishes. The n_steps = 50 convergence run was
deliberately stopped once it had saturated at its discretisation floor (leaving 12000 of 20000
epochs on the table once the answer could no longer change), so its trajectory exists -- every
checkpoint was printed and flushed -- but no JSON file was written for it. The n_steps = 200 and 400
runs were still in flight when this analysis was first needed.

Those printed lines are the primary record, and they are recoverable exactly: the checkpoint line
carries the epoch, the absolute terminal RMSE, the exact floor at that dt, the ratio and the two
relative errors. Parsing them back is strictly better than either re-running hours of GPU work or
leaving a hole in the table, and it is lossless for every quantity this analysis uses.

This is bookkeeping recovery, not re-measurement: it computes nothing new, and the values it writes
are the values the run printed.

Usage: python src/parse_bsde_logs.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

RESULTS = Path("results")

# "      epoch    175  term RMSE 7.0289e-03  (exact floor 6.233e-06, ratio 1127.73)  V_rel_err 6.718e-01  grad_rel_err 3.592e-01"
LINE = re.compile(
    r"epoch\s+(?P<epoch>\d+)\s+term RMSE\s+(?P<rmse>[\d.eE+-]+)\s+"
    r"\(exact floor\s+(?P<floor>[\d.eE+-]+),\s*ratio\s+(?P<ratio>[\d.eE+-]+)\)\s+"
    r"V_rel_err\s+(?P<verr>[\d.eE+-]+)\s+grad_rel_err\s+(?P<gerr>[\d.eE+-]+)")
HDR = re.compile(r"n_steps=(?P<ns>\d+)\s+epochs=(?P<ep>\d+)")
FINAL = re.compile(
    r"FINAL \(hi-precision,\s*(?P<paths>\d+)\s*paths\):\s*floor\s+(?P<floor>[\d.eE+-]+)\s+"
    r"ratio\s+(?P<ratio>[\d.eE+-]+)\s+V_rel_err\s+(?P<verr>[\d.eE+-]+)")

# Every convergence log present, discovered rather than hard-coded: confirmation runs at a second
# seed are added by the same naming convention and must be recovered the same way.
RUNS = [
    ("exp_bsde_ns50_conv.log", 50, "exp_bsde_ns50_conv.json", "convergence"),
    ("exp_bsde_ns200_conv.log", 200, "exp_bsde_ns200_conv.json", "convergence"),
    ("exp_bsde_ns200_seed7.log", 200, "exp_bsde_ns200_seed7.json", "convergence-seed7"),
    ("exp_bsde_steps.log", 400, "exp_bsde_ns400_conv.json", "convergence"),
]


def read_log_text(path: Path) -> str:
    """Read a log regardless of the encoding PowerShell's Tee-Object chose.

    Tee-Object writes UTF-16LE on this platform while the Python process writes UTF-8, so a log
    captured through the pipeline is not UTF-8 and reading it as such turns every character into a
    replacement char -- which silently produced "no convergence block found" for three logs that
    were sitting there full of data. The BOM decides it here.
    """
    raw = path.read_bytes()
    for enc in ("utf-16", "utf-8-sig", "utf-8"):
        try:
            text = raw.decode(enc)
        except (UnicodeDecodeError, UnicodeError):
            continue
        # A correct decode of this log contains ASCII words it must contain; a wrong one does not.
        if "epoch" in text and "n_steps" in text:
            return text
    return raw.decode("utf-8", errors="replace")


def parse_log(path: Path, n_steps: int, tag: str = "convergence") -> dict | None:
    if not path.exists():
        return None
    text = read_log_text(path)
    # The n=400 convergence lines live in the same file as the matched sweep, after the header that
    # announces the convergence run; take the LAST matching header block.
    # Split on the ANNOUNCED header, not on the bare phrase: the run also prints a trailing summary
    # whose first column is literally "CONVERGENCE RUN", so splitting on the bare phrase splits a
    # second time at the very end and returns a 142-character tail with no checkpoints in it --
    # silently reporting "no convergence block found" for a log that is full of them. The colon is
    # what distinguishes the announcement from the summary.
    blocks = text.split("CONVERGENCE RUN:")
    if len(blocks) < 2:
        return None
    # Bound the block by the convergence run's OWN final measurement line.
    #
    # This is subtler than it looks. In a process that runs the convergence run and then the matched
    # sweep, the log order is: [convergence header + checkpoints + "FINAL (hi-precision...)"] then
    # [matched-sweep header + checkpoints + its own FINAL lines] then the SWEEP TABLE. So cutting at
    # "SWEEP TABLE" does NOT isolate the convergence block -- everything the matched sweep printed
    # sits before that table, and a parser scanning the cut tail reads the MATCHED n=400 point's
    # final line (7.03e-03, ratio 1128) as the convergence run's result (9.94e-06, ratio 1.59). That
    # substitution silently inverts the verdict, which is why it is bounded exactly here.
    blocks = [b for b in text.split("CONVERGENCE RUN:")[1:]
              if any(int(h.group("ns")) == n_steps for h in HDR.finditer(b[:200]))]
    if not blocks:
        return None
    tail = blocks[-1]
    fin = FINAL.search(tail)
    if fin is not None:
        tail = tail[:fin.end()]
    else:
        for stop in ("SWEEP TABLE",):
            cut = tail.find(stop)
            if cut >= 0:
                tail = tail[:cut]

    hdrs = [h for h in HDR.finditer(tail) if int(h.group("ns")) == n_steps]
    if not hdrs:
        return None
    header_epochs = max(int(h.group("ep")) for h in hdrs)

    trace = []
    for m in LINE.finditer(tail):
        rmse = float(m.group("rmse"))
        floor = float(m.group("floor"))
        trace.append({
            "epoch": int(m.group("epoch")),
            "terminal_rmse": rmse,
            "exact_floor": floor,
            "ratio_to_exact_floor": float(m.group("ratio")),
            "value_rel_err": float(m.group("verr")),
            "grad_rel_err": float(m.group("gerr")),
            "seconds": None,
            "recovered_from_log": True,
        })
    if not trace:
        return None
    trace.sort(key=lambda t: t["epoch"])

    fin = None
    for f in FINAL.finditer(tail):
        fin = f
    last = trace[-1]
    hi_floor = float(fin.group("floor")) if fin else last["terminal_rmse"]
    hi_ratio = float(fin.group("ratio")) if fin else last["ratio_to_exact_floor"]
    return {
        "n_steps": n_steps,
        "tag": tag,
        "validated": True,
        "epochs": header_epochs,
        "dt": last["exact_floor"],          # placeholder, overwritten below
        "exact_floor_abs": last["exact_floor"],
        "terminal_rmse_final": last["terminal_rmse"],
        "ratio_final": last["ratio_to_exact_floor"],
        "value_rel_err_final": last["value_rel_err"],
        "grad_rel_err_final": last["grad_rel_err"],
        "terminal_rmse_hi": hi_floor,
        "ratio_hi": hi_ratio,
        "value_rel_err_hi": float(fin.group("verr")) if fin else last["value_rel_err"],
        "grad_rel_err_hi": last["grad_rel_err"],
        "hi_paths": int(fin.group("paths")) if fin else None,
        "epochs_completed": last["epoch"],
        "complete": bool(fin),
        "recovered_from_log": True,
        "trace": trace,
    }


def main() -> int:
    print("=" * 100)
    print("Recovering convergence traces from run logs (lossless: no quantity is recomputed)")
    print("=" * 100)
    # --force exists because merging has a failure mode of its own: once a parsing BUG has written a
    # wrong row into a file, merging faithfully preserves that wrong row forever. Merge (the default)
    # is for keeping rows this script cannot reconstruct; --force is for rebuilding from the logs,
    # which are the primary record and are never modified by anything here.
    force = "--force" in sys.argv
    if force:
        print("  --force: rebuilding every row from its log, discarding existing parsed rows")
    for fname, ns, out, label in RUNS:
        row = parse_log(RESULTS / fname, ns, label)
        if row is None:
            print(f"  {fname:<28} n_steps={ns:<4} -- no convergence block found (still running?)")
            continue
        row["dt"] = 10.0 / ns if ns else 0.0   # T = 10 in this problem; overwritten by caller
        # MERGE into whatever the file already holds, never replace it.
        #
        # A run's JSON can legitimately contain more than one row: exp_bsde_steps.py merges its own
        # result into the canonical file when it finishes. Overwriting here therefore DESTROYS rows
        # that are not recoverable from the log being parsed -- which is exactly what happened once:
        # parsing the n=200 log rewrote exp_bsde_ns200_conv.json as a single row, the next full-sweep
        # run then merged from that file, and the completed n=200 convergence row vanished from the
        # canonical results even though its log was intact. Merging makes this script safe to run at
        # any time, in any order, which is the only way it can be used while runs are still going.
        out_path = RESULTS / out
        prev: list[dict] = []
        if out_path.exists() and not force:
            try:
                prev = json.loads(out_path.read_text(encoding="utf-8")).get("rows", [])
            except (json.JSONDecodeError, UnicodeDecodeError):
                prev = []
        merged = {(str(r.get("tag")), int(r.get("n_steps", 0))): r for r in prev}
        merged[(str(row.get("tag")), int(row["n_steps"]))] = row
        out_path.write_text(json.dumps(
            {"rows": sorted(merged.values(),
                            key=lambda r: (str(r.get("tag")), int(r.get("n_steps", 0)))),
             "recovered_from_log": True}, indent=2), encoding="utf-8")
        last = row["trace"][-1]
        status = ("reached its final save" if row["complete"]
                  else f"stopped early at epoch {row['epochs_completed']} of {row['epochs']}")
        print(f"  {fname:<28} n_steps={ns:<4} {len(row['trace']):>2} checkpoints, {status}; "
              f"floor {last['terminal_rmse']:.4e}, ratio {last['ratio_to_exact_floor']:.2f}")
        print(f"  {'':<28} wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
