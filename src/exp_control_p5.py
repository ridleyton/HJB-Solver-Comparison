"""
Part F/G -- p=5 with the snapshot cap raised, so its time resolution is held fixed.

The problem this fixes. In `exp_control_floor.py` the p=5 ratio sequence broke at N >= 17:
    2.7, 3.9, 8.8, 9.4, 1.9, 0.6
The cause was not the scheme, it was our solver silently halving the snapshot count as N^5 grew
(`while n_snap * N > 4e7: n_snap //= 2`), which coarsened the TIME resolution. That inflated the
value error -- at N=17 it went UP from 2.135e-4 to 9.207e-4 while the control error went DOWN --
and it made p=5 unquotable past N=15. This machine has 52 GB free, so 4e7 float64 elements
(320 MB) was needlessly tight. `snap_cap` is now a constructor argument.

Part F finds the snapshot count at which the value error stops improving at p=5, holding the
grid fixed and sweeping only the time resolution -- the same logic as Part D, but where the cap
was actually biting. Part G then reruns the p=5 refinement with that count and a raised cap, so
the only thing changing across the sweep is the spatial mesh.

Interpretation guard: the ratio is only meaningful once the time error is negligible. Below the
plateau snapshot count both errors are time-limited and the ratio underestimates the spatial
decoupling, which is exactly why Part F comes first.

Output: results/exp_control_p5.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_order import ProbedFDM, evaluate, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=5)
    ap.add_argument("--snaps", default="17,33,65,129")
    ap.add_argument("--plateau-N", type=int, default=13, help="grid used for the snapshot sweep")
    ap.add_argument("--grids", default="9,11,13,15,17,21")
    ap.add_argument("--n-snap-final", type=int, default=0,
                    help="0 = pick from Part F automatically")
    ap.add_argument("--snap-cap", type=float, default=6e8,
                    help="float64 elements; 6e8 is 4.8 GB, which this machine has to spare")
    ap.add_argument("--probe-scale", type=float, default=0.25)
    ap.add_argument("--out", default="results/exp_control_p5.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "p": args.p,
            "snap_cap": args.snap_cap,
            "problem": ("multi-asset optimal execution, LQ form, hetero=True "
                        "(non-separable), closed-form Riccati reference"),
            "probe": f"central difference with eps = dx x {args.probe_scale}",
            "why": ("the p=5 sweep in exp_control_floor.py broke at N>=17 because the solver's "
                    "snapshot memory cap silently coarsened the time resolution as N^5 grew; "
                    "this reruns it with the cap raised so only the spatial mesh changes"),
        },
        "part_f_snapshot_sweep": [],
        "part_g_refinement": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    p = args.p
    prob = ACProblem(p=p, hetero=True, het_seed=1)

    # ---------------- Part F: find where the value error stops improving in time ----------
    print("=" * 104)
    print(f"Part F -- p={p}, N={args.plateau_N} fixed, swept snapshot count (cap {args.snap_cap:g})")
    print("=" * 104)
    print(f"  {'n_snap':>7s} {'used':>5s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>9s} {'ord_v':>8s} {'ord_c':>8s}")
    prev = None
    best_ns = None
    for ns in [int(s) for s in args.snaps.split(",")]:
        try:
            s = ProbedFDM(prob, n_grid=args.plateau_N, eps_mode="h", eps_scale=args.probe_scale,
                          n_snap=ns, snap_cap=args.snap_cap)
            s.solve()
            m = evaluate(s, prob)
            ov = oc = float("nan")
            if prev is not None:
                ov = order(prev["value_err"], m["value_err"], 1.0 / prev["ns"], 1.0 / ns)
                oc = order(prev["control_err"], m["control_err"], 1.0 / prev["ns"], 1.0 / ns)
            row = {"n_snap": ns, "n_snap_used": s.n_snap_used, "N": args.plateau_N,
                   "dx": float(np.mean(s.dx)), **m, "order_value_vs_snap": ov,
                   "order_control_vs_snap": oc, "ratio": m["control_err"] / m["value_err"]}
            results["part_f_snapshot_sweep"].append(row)
            print(f"  {ns:>7d} {s.n_snap_used:>5d} {m['value_err']:>12.3e} "
                  f"{m['control_err']:>12.3e} {row['ratio']:>9.1f} {ov:>8.2f} {oc:>8.2f}",
                  flush=True)
            # The plateau is where the value error stops responding to more snapshots.
            if ov == ov and abs(ov) < 0.35 and best_ns is None:
                best_ns = prev["ns"] if prev else ns
            m["ns"] = ns
            prev = m
        except Exception as e:
            print(f"  n_snap={ns} FAILED: {type(e).__name__}: {e}")
        save()

    n_final = args.n_snap_final or best_ns or 33
    print(f"\n  -> plateau reached by n_snap = {best_ns}; using n_snap = {n_final} for Part G")
    results["meta"]["n_snap_used_in_part_g"] = n_final
    results["meta"]["plateau_detected_at"] = best_ns
    save()

    # ---------------- Part G: p=5 refinement with time resolution held fixed ------------
    print("\n" + "=" * 104)
    print(f"Part G -- p={p} refinement with n_snap={n_final} fixed (time resolution constant)")
    print("=" * 104)
    print(f"  {'N':>3s} {'h':>9s} {'used':>5s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>9s} {'ord_v':>7s} {'ord_c':>7s} {'t(s)':>9s}")
    prev = None
    for N in [int(g) for g in args.grids.split(",")]:
        try:
            s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=args.probe_scale,
                          n_snap=n_final, snap_cap=args.snap_cap)
            s.solve()
            m = evaluate(s, prob)
            h = float(np.mean(s.dx))
            ov = oc = float("nan")
            if prev:
                ov = order(prev["value_err"], m["value_err"], prev["h"], h)
                oc = order(prev["control_err"], m["control_err"], prev["h"], h)
            row = {"p": p, "N": N, "h": h, "n_snap": n_final, "n_snap_used": s.n_snap_used, **m,
                   "order_value": ov, "order_control": oc,
                   "ratio": m["control_err"] / m["value_err"],
                   "solve_time_s": s.solve_time, "n_time_steps": s.n_time_steps}
            results["part_g_refinement"].append(row)
            print(f"  {N:>3d} {h:>9.5f} {s.n_snap_used:>5d} {m['value_err']:>12.3e} "
                  f"{m['control_err']:>12.3e} {row['ratio']:>9.1f} {ov:>7.2f} {oc:>7.2f} "
                  f"{s.solve_time:>9.2f}", flush=True)
            m["h"] = h
            prev = m
        except Exception as e:
            print(f"  N={N} FAILED: {type(e).__name__}: {e}")
        save()

    # ---------------- verdict ---------------------------------------------------------
    print("\n" + "=" * 104)
    print("VERDICT")
    print("=" * 104)
    g = results["part_g_refinement"]
    if g:
        seq = ", ".join(f"{r['ratio']:.1f}" for r in g)
        oc = [r["order_control"] for r in g if r["order_control"] == r["order_control"]]
        ov = [r["order_value"] for r in g if r["order_value"] == r["order_value"]]
        print(f"  p={p} ratio over N={[r['N'] for r in g]} -> {seq}")
        if oc:
            print(f"    control error order: median {np.median(oc):.2f}")
        if ov:
            print(f"    value   error order: median {np.median(ov):.2f}")
        monotone = all(b >= a for a, b in zip(
            [r["ratio"] for r in g], [r["ratio"] for r in g][1:]))
        print(f"    ratio monotone increasing: {monotone}")
        used = {r["n_snap_used"] for r in g}
        print(f"    snapshot counts actually used: {sorted(used)}"
              + ("  (constant -- time resolution held fixed)" if len(used) == 1
                 else "  (NOT constant -- cap is still biting)"))
    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
