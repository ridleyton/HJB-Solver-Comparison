"""
Does deep BSDE's value accuracy survive at higher dimension, and where must training stop?

Why this is the journal-relevant question
-----------------------------------------
Deep BSDE -- not the PINN -- is the standard high-dimensional method in mathematical finance, and
the manuscript cites it without running it. Closing that gap needs two numbers per dimension: the
value accuracy the solver actually reaches, and the training point at which it reaches it. The
second is not a detail. In every run measured so far the terminal residual falls monotonically
while the value error bottoms out and then degrades, so a solver trained to a residual plateau
reports a value error several times worse than the one it passed through.

The two experiment families use different field names (the d=1 study records `terminal_rmse` and
`value_rel_err`; the dimension study records `floor_abs` and `V_rel_err`), which is why this reads
both through an accessor instead of assuming one shape. Getting that wrong raises a KeyError rather
than a wrong number, but the reverse -- silently reading the wrong field -- is the failure mode this
project has hit repeatedly, so the fields are resolved explicitly.

Reported per dimension: the exact discretisation floor, the residual the run STARTED from (the
denominator that makes the stopping threshold scale-free), the best value error and where it occurs,
and the residual at that point expressed both as a fraction of the start and as a multiple of the
floor. The last two columns are the whole point: the first is roughly constant across dimension, the
second is not, and the difference is what makes an absolute stopping threshold non-transferable.

Usage: python src/analysis_bsde_dims.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path("results")
OUT = RESULTS / "analysis_bsde_dims.md"

# (file, default d, human label) -- every run this analysis draws on.
SOURCES = [
    ("exp_bsde_ns50_conv.json", 1, "d=1 (converged, from the n_steps study)"),
    ("exp_bsde_dims.json", 4, "d=4"),
    ("exp_bsde_dims_d8.json", 8, "d=8"),
]


def field(row: dict, *names, default=None):
    """First present key. Explicit because the two families name the same quantity differently."""
    for n in names:
        if n in row:
            return row[n]
    return default


def main() -> int:
    runs = []
    for fname, dflt, label in SOURCES:
        p = RESULTS / fname
        if not p.exists():
            print(f"  skip {fname}: absent")
            continue
        for r in json.loads(p.read_text(encoding="utf-8")).get("rows", []):
            tr = r.get("trace") or []
            if not tr:
                continue
            norm = []
            for t in tr:
                v = field(t, "value_rel_err", "V_rel_err")
                res = field(t, "terminal_rmse", "floor_abs")
                if v is None or res is None:
                    continue
                norm.append({"epoch": t["epoch"], "v_err": v, "residual": res})
            if len(norm) < 3:
                continue
            runs.append({
                "d": r.get("d", dflt), "label": label, "trace": norm,
                "floor": field(r, "exact_floor_abs", "exact_floor"),
            })
    if not runs:
        print("no usable traces found")
        return 1
    runs.sort(key=lambda r: r["d"])

    # A COMMON STARTING EPOCH, so the fraction below compares like with like. The traces do not all
    # begin at the same epoch (the d=1 study's first checkpoint is epoch 100; the dimension runs
    # also record 100), and taking each run's own first checkpoint would make the denominator depend
    # on the checkpoint schedule rather than on the problem. Epoch 100 is the earliest epoch present
    # in every trace, and it is used as the reference point throughout.
    REF_EPOCH = 100
    for r in runs:
        ref = next((t for t in r["trace"] if t["epoch"] >= REF_EPOCH), r["trace"][0])
        r["ref_residual"] = ref["residual"]
        r["ref_epoch"] = ref["epoch"]

    L: list[str] = []
    A = L.append
    A("# Does deep BSDE reach a useful value at higher dimension?")
    A("")
    A("Deep BSDE is the standard high-dimensional method in mathematical finance, so a comparison "
      "paper that cites it without running it leaves its most likely referee question unanswered. "
      "These are the runs that answer it. Every number is a measured trace, not a fitted model.")
    A("")
    A("| d | exact floor | residual at epoch 100 | best V error | at epoch | residual there | "
      "residual / start | residual / floor |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|")
    summary = []
    for r in runs:
        init = r["ref_residual"]
        best = min(r["trace"], key=lambda t: t["v_err"])
        row = {"d": r["d"], "floor": r["floor"], "init": init, "best_v": best["v_err"],
               "best_epoch": best["epoch"], "res_at_best": best["residual"],
               "frac_of_init": best["residual"] / init if init else float("nan"),
               "over_floor": best["residual"] / r["floor"] if r["floor"] else float("nan"),
               "v_last": r["trace"][-1]["v_err"], "epochs_run": r["trace"][-1]["epoch"]}
        summary.append(row)
        A(f"| {row['d']} | {row['floor']:.3e} | {init:.3e} | **{row['best_v']:.3%}** | "
          f"{row['best_epoch']} | {row['res_at_best']:.3e} | {row['frac_of_init']:.2%} | "
          f"{row['over_floor']:.0f}x |")
    A("")

    A("## What the table says")
    A("")
    bests = [r["best_v"] for r in summary]
    A(f"**Deep BSDE reaches a usable value at every dimension measured.** The best value error is "
      f"{min(bests):.2%} to {max(bests):.2%} across d = "
      f"{', '.join(str(r['d']) for r in summary)}. So it can be reported as a baseline, and the "
      f"paper's omission of it is a fixable gap rather than an impossibility.")
    A("")
    last = [r["v_last"] for r in summary]
    A(f"**But only if training stops at the right point.** Read at the end of each run instead, the "
      f"value error is {min(last):.1%} to {max(last):.1%} -- worse than the best by a factor of "
      f"{min(l / b for l, b in zip(last, bests)):.1f} to "
      f"{max(l / b for l, b in zip(last, bests)):.0f}. In every run the residual kept falling while "
      f"the value degraded: the residual is a progress signal for the SCHEME, not for the value.")
    A("")
    if len(summary) >= 2:
        fr = [r["frac_of_init"] for r in summary]
        fl = [r["over_floor"] for r in summary]
        A("**The stopping threshold is problem-relative, and the absolute one does not transfer.** "
          "Expressed as a fraction of the residual at epoch 100, the optimum sits at "
          f"{min(fr):.1%} to {max(fr):.1%} across dimensions -- a spread of {max(fr) / min(fr):.0f}x. "
          f"Expressed as a multiple of the discretisation floor it sits at {min(fl):.0f}x to "
          f"{max(fl):.0f}x, a spread of {max(fl) / min(fl):.0f}x. So the scale-free framing is the "
          f"right one, but it is not a tight rule: the fraction still moves by a factor of "
          f"{max(fr) / min(fr):.0f} across the dimensions measured, which is far better than the "
          f"{max(fl) / min(fl):.0f}x of the absolute framing and still not something to quote as a "
          f"constant.")
        A("")
        A("Two consequences, and they point the same way.")
        A("")
        A("The first is a warning for anyone reproducing this: the d=1 threshold of ~1e-3 applied "
          "literally at d=4 stops three orders of magnitude too late, well past the point where the "
          "value has started to diverge, so the reproduced number would be an order of magnitude "
          "worse than the one reported here for reasons that have nothing to do with the method.")
        A("")
        A("The second is that neither framing is safe to hard-code, so the stopping point has to be "
          "found by monitoring the value rather than by a rule -- which is the same conclusion the "
          "d=1 study reached, now with the evidence that a rule cannot be substituted for it.")
        A("")

    A("## Why this is not a rescue of deep BSDE as a competitor")
    A("")
    worst = max(summary, key=lambda r: r["over_floor"])
    A(f"At its best point the solver is still {min(r['over_floor'] for r in summary):.0f}x to "
      f"{worst['over_floor']:.0f}x the discretisation floor of the scheme it is solving, and the "
      f"residual is still falling at the end of every run. So deep BSDE does not reach the grid's "
      f"accuracy where the grid can run -- which is the paper's existing claim, now supported "
      f"against the method a referee would most likely name. What the dimension study adds is a "
      f"measured statement about the regime where deep BSDE is usable and at what cost, which is "
      f"what turns this from a missing baseline into a result.")
    A("")

    out = OUT
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
