"""
Two follow-ups that decide how the control-accuracy result may be stated.

PART D -- is the value error floor-limited by time interpolation?
    The refinement study found value-error orders that are too irregular to quote (median 4.11,
    range -4.32 to 5.60) and a value error that is non-monotone at the finest grids. That is the
    signature of a floor. The suspected source is `_GridSolver.value`, which interpolates
    LINEARLY IN TIME between the stored snapshots; that error does not improve as the spatial
    grid is refined.

    This matters for the paper because the two possible outcomes say different things:
      * If raising the snapshot count lowers the value error while leaving the control error
        essentially unchanged, then the value error is already at the time-interpolation floor
        while the control error is still spatially dominated. That is the stronger claim: you
        cannot fix control accuracy by refining the value function, because the two errors live
        in different places.
      * If both fall together, they share the floor and the gap is purely spatial at every
        snapshot count -- a weaker but still valid statement.
    The distinguishing measurement is a fixed grid with a swept snapshot count, so nothing but
    the time resolution changes.

PART E -- does the control/value error ratio keep growing at p = 4 and 5?
    At p=2 with a probe scaled to the mesh the ratio ran 2.4, 5.0, 12.7, 25.0, 49.2, 83.3, 178.9
    as N went 11 to 61. If that growth is a property of the scheme rather than of p=2, it should
    persist at higher dimension -- which is the regime where the grid's N^p cost bites and where
    the argument actually matters. Grid cost scales roughly as N^(p+1) (points times CFL-limited
    steps), so the N ranges differ per dimension; p=5 beyond N=21 costs about 15 minutes per
    solve and is not attempted.

Output: results/exp_control_floor.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_order import ProbedFDM, evaluate, order


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snap-dims", default="2,3")
    ap.add_argument("--snap-grids", default="21,35,45")
    ap.add_argument("--snaps", default="5,9,17,33,65,129,257")
    ap.add_argument("--highp", default="4:9,11,13,15,17,21,25;5:9,11,13,15,17,21")
    ap.add_argument("--probe-scale", type=float, default=0.25,
                    help="probe step as a multiple of dx; 0.25 was best-supported in Part B/C")
    ap.add_argument("--out", default="results/exp_control_floor.json")
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem": ("multi-asset optimal execution, LQ form, hetero=True "
                        "(non-separable), closed-form Riccati reference"),
            "probe": f"central difference with eps = dx x {args.probe_scale}",
            "part_d_question": ("is the value error floor-limited by the linear-in-time snapshot "
                                "interpolation, so that refining space cannot improve the "
                                "control beyond that floor?"),
            "part_e_question": ("does the control/value error ratio keep growing with mesh "
                                "refinement at p = 4 and 5?"),
        },
        "part_d_snapshot_sweep": [],
        "part_e_high_dimension": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    snap_grids = [int(g) for g in args.snap_grids.split(",")]
    snaps = [int(s) for s in args.snaps.split(",")]

    # ---------------- Part D: snapshot sweep at fixed grid -----------------------------
    print("=" * 104)
    print("Part D -- fixed grid, swept snapshot count (isolates the TIME interpolation error)")
    print("=" * 104)
    print(f"  {'p':>2s} {'N':>4s} {'n_snap':>7s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>9s} {'ord_v':>7s} {'ord_c':>7s}")
    for p in [int(x) for x in args.snap_dims.split(",")]:
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        for N in snap_grids:
            prev = None
            for ns in snaps:
                try:
                    s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=args.probe_scale,
                                  n_snap=ns)
                    s.solve()
                    m = evaluate(s, prob)
                    # Orders here are in n_snap, not h: the grid is fixed by construction.
                    ov = oc = float("nan")
                    if prev is not None:
                        ov = order(prev["value_err"], m["value_err"], 1.0 / prev["ns"], 1.0 / ns)
                        oc = order(prev["control_err"], m["control_err"], 1.0 / prev["ns"],
                                   1.0 / ns)
                    row = {"p": p, "N": N, "n_snap": ns, "n_snap_used": int(len(s.t_snap)),
                           "dx": float(np.mean(s.dx)), **m,
                           "order_value_vs_snap": ov, "order_control_vs_snap": oc,
                           "ratio": m["control_err"] / m["value_err"]}
                    results["part_d_snapshot_sweep"].append(row)
                    print(f"  {p:>2d} {N:>4d} {ns:>7d} {m['value_err']:>12.3e} "
                          f"{m['control_err']:>12.3e} {row['ratio']:>9.1f} "
                          f"{ov:>7.2f} {oc:>7.2f}", flush=True)
                    m["ns"] = ns
                    prev = m
                except Exception as e:
                    print(f"  p={p} N={N} n_snap={ns} FAILED: {type(e).__name__}: {e}")
                save()
            print()

    # ---------------- Part E: p = 4 and 5 refinement ----------------------------------
    print("=" * 104)
    print("Part E -- refinement at p = 4 and 5, probe scaled to the mesh")
    print("=" * 104)
    print(f"  {'p':>2s} {'N':>4s} {'h':>9s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>9s} {'ord_v':>7s} {'ord_c':>7s} {'t(s)':>9s}")
    for spec in args.highp.split(";"):
        if not spec.strip():
            continue
        p_str, grids_str = spec.split(":")
        p = int(p_str)
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        prev = None
        for N in [int(g) for g in grids_str.split(",")]:
            try:
                s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=args.probe_scale)
                s.solve()
                m = evaluate(s, prob)
                h = float(np.mean(s.dx))
                ov = oc = float("nan")
                if prev:
                    ov = order(prev["value_err"], m["value_err"], prev["h"], h)
                    oc = order(prev["control_err"], m["control_err"], prev["h"], h)
                row = {"p": p, "N": N, "h": h, **m, "order_value": ov, "order_control": oc,
                       "ratio": m["control_err"] / m["value_err"],
                       "solve_time_s": s.solve_time,
                       "n_grid_points": int(s.N), "n_time_steps": s.n_time_steps}
                results["part_e_high_dimension"].append(row)
                print(f"  {p:>2d} {N:>4d} {h:>9.5f} {m['value_err']:>12.3e} "
                      f"{m['control_err']:>12.3e} {row['ratio']:>9.1f} {ov:>7.2f} {oc:>7.2f} "
                      f"{s.solve_time:>9.2f}", flush=True)
                m["h"] = h
                prev = m
            except Exception as e:
                print(f"  p={p} N={N} FAILED: {type(e).__name__}: {e}")
            save()
        print()

    # ---------------- verdict ---------------------------------------------------------
    print("=" * 104)
    print("VERDICT")
    print("=" * 104)
    d = results["part_d_snapshot_sweep"]
    print("  Part D -- does raising the time resolution (snapshot count) help?")
    for p in sorted({r["p"] for r in d}):
        for N in sorted({r["N"] for r in d if r["p"] == p}):
            rs = sorted([r for r in d if r["p"] == p and r["N"] == N], key=lambda r: r["n_snap"])
            if len(rs) < 2:
                continue
            v0, v1 = rs[0]["value_err"], rs[-1]["value_err"]
            c0, c1 = rs[0]["control_err"], rs[-1]["control_err"]
            r0, r1 = rs[0]["ratio"], rs[-1]["ratio"]
            print(f"    p={p} N={N}: n_snap {rs[0]['n_snap']} -> {rs[-1]['n_snap']}")
            print(f"        value err   {v0:.3e} -> {v1:.3e}   (x{v0/v1:.1f} better)")
            print(f"        control err {c0:.3e} -> {c1:.3e}   (x{c0/c1:.2f} better)")
            print(f"        ratio       {r0:.1f} -> {r1:.1f}")
    e = results["part_e_high_dimension"]
    print("\n  Part E -- ratio growth with refinement at high dimension")
    for p in sorted({r["p"] for r in e}):
        rs = sorted([r for r in e if r["p"] == p], key=lambda r: r["N"])
        if len(rs) < 2:
            continue
        seq = ", ".join(f"{r['ratio']:.1f}" for r in rs)
        oc = [r["order_control"] for r in rs if r["order_control"] == r["order_control"]]
        print(f"    p={p}: ratio over N={[r['N'] for r in rs]} -> {seq}")
        if oc:
            print(f"          control order median {np.median(oc):.2f}")
    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
