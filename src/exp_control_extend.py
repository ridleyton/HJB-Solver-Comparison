"""
Extend the control-accuracy sweeps at p=4 and p=5, and fit convergence orders robustly.

Two things this does that the per-run scripts do not.

1. **Robust orders.** `exp_control_order.py` and friends report the order between consecutive
   grid pairs, which is noisy when the value error wanders. The defensible estimate is the slope
   of a least-squares line through log(error) against log(h) over the whole sweep, with the fit
   quality reported so a reader can judge it. That is what `fit_orders` does here, and it is why
   the paper can state the control order as 1.95-2.00 rather than a wider median range.

2. **A longer p=4 and p=5 sweep.** p=4 is cheap to extend and adds points to a dimension that
   already looks clean; p=5 is expensive (the p=5, N=25 solve is ~20 minutes and 5 GB of
   snapshots) so it is optional.

Existing rows are preserved: the script appends to the stored JSONs, chaining the order
calculation from the last row already present, and it backs the files up first.

Usage:
    python src/exp_control_extend.py --p4-grids 31,35 --p5-grids 25 --snap-cap 6e8
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_order import ProbedFDM, evaluate, order

RESULTS = Path("results")


def fit_orders(rows: list[dict], key: str) -> tuple[float, float]:
    """Least-squares slope of log(error) vs log(h), with the R^2 of that fit."""
    h = np.array([r["h"] for r in rows], dtype=float)
    e = np.array([r[key] for r in rows], dtype=float)
    m = (e > 0) & (h > 0)
    if m.sum() < 3:
        return float("nan"), float("nan")
    A = np.vstack([np.log(h[m]), np.ones(int(m.sum()))]).T
    slope, inter = np.linalg.lstsq(A, np.log(e[m]), rcond=None)[0]
    pred = A @ np.array([slope, inter])
    ss_res = float(((np.log(e[m]) - pred) ** 2).sum())
    ss_tot = float(((np.log(e[m]) - np.log(e[m]).mean()) ** 2).sum())
    return float(slope), (1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan"))


def parse_grids(spec: str) -> list[int]:
    """Parse a comma-separated grid list, treating anything non-numeric as "none"."""
    out = []
    for tok in (spec or "").split(","):
        tok = tok.strip()
        if not tok:
            continue
        try:
            out.append(int(tok))
        except ValueError:
            print(f"  ignoring non-numeric grid token {tok!r} (use e.g. --p5-grids 25)")
    return out


def snap_cap_for(N: int, p: int, n_snap: int, headroom: float = 1.3) -> float:
    return float(N**p * n_snap * headroom)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4-grids", default="31,35")
    ap.add_argument("--p5-grids", default="25")
    ap.add_argument("--p5-n-snap", type=int, default=65)
    ap.add_argument("--probe-scale", type=float, default=0.25)
    ap.add_argument("--skip-run", action="store_true",
                    help="only refit the stored rows, do not solve anything")
    args = ap.parse_args()

    p5_path = RESULTS / "exp_control_p5.json"
    fl_path = RESULTS / "exp_control_floor.json"
    for f in (p5_path, fl_path):
        if f.exists():
            shutil.copy2(f, f.with_suffix(".json.bak"))
            print(f"  backed up {f} -> {f.name}.bak")

    p5 = json.loads(p5_path.read_text()) if p5_path.exists() else {"part_g_refinement": []}
    fl = json.loads(fl_path.read_text()) if fl_path.exists() else {"part_e_high_dimension": []}

    # ---------------- p=4 extension -----------------------------------------------------
    p4_rows = sorted([r for r in fl["part_e_high_dimension"] if r["p"] == 4], key=lambda r: r["N"])
    print(f"\np=4: have N = {[r['N'] for r in p4_rows]}")
    if not args.skip_run:
        prob = ACProblem(p=4, hetero=True, het_seed=1)
        prev = p4_rows[-1] if p4_rows else None
        for N in parse_grids(args.p4_grids):
            if any(r["N"] == N for r in p4_rows):
                print(f"  N={N} already present, skipping")
                continue
            try:
                s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=args.probe_scale)
                s.solve()
                m = evaluate(s, prob)
                h = float(np.mean(s.dx))
                row = {"p": 4, "N": N, "h": h, **m,
                       "order_value": order(prev["value_err"], m["value_err"], prev["h"], h),
                       "order_control": order(prev["control_err"], m["control_err"],
                                              prev["h"], h),
                       "ratio": m["control_err"] / m["value_err"],
                       "solve_time_s": s.solve_time, "n_time_steps": s.n_time_steps}
                fl["part_e_high_dimension"].append(row)
                print(f"  N={N:3d} h={h:.5f} value={m['value_err']:.3e} "
                      f"ctrl={m['control_err']:.3e} ratio={row['ratio']:.1f} "
                      f"t={s.solve_time:.1f}s", flush=True)
                # Chain the order from the previous row: it must carry 'h', which `m` does not.
                prev = row
            except Exception as e:
                print(f"  N={N} FAILED: {type(e).__name__}: {e}")
            fl_path.write_text(json.dumps(fl, indent=2))

    # ---------------- p=5 extension -----------------------------------------------------
    p5_rows = sorted(p5["part_g_refinement"], key=lambda r: r["N"])
    print(f"\np=5: have N = {[r['N'] for r in p5_rows]} (n_snap={args.p5_n_snap})")
    if not args.skip_run:
        prob = ACProblem(p=5, hetero=True, het_seed=1)
        prev = p5_rows[-1] if p5_rows else None
        for N in parse_grids(args.p5_grids):
            if any(r["N"] == N for r in p5_rows):
                print(f"  N={N} already present, skipping")
                continue
            cap = snap_cap_for(N, 5, args.p5_n_snap)
            need_gb = N**5 * args.p5_n_snap * 8 / 1e9
            print(f"  N={N}: snap_cap={cap:.3g}, ~{need_gb:.1f} GB of snapshots", flush=True)
            try:
                s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=args.probe_scale,
                              n_snap=args.p5_n_snap, snap_cap=cap)
                s.solve()
                m = evaluate(s, prob)
                h = float(np.mean(s.dx))
                row = {"N": N, "h": h, "n_snap": args.p5_n_snap,
                       "n_snap_used": s.n_snap_used, **m,
                       "order_value": order(prev["value_err"], m["value_err"], prev["h"], h),
                       "order_control": order(prev["control_err"], m["control_err"],
                                              prev["h"], h),
                       "ratio": m["control_err"] / m["value_err"],
                       "solve_time_s": s.solve_time, "n_time_steps": s.n_time_steps}
                p5["part_g_refinement"].append(row)
                print(f"  N={N:3d} h={h:.5f} used={s.n_snap_used} value={m['value_err']:.3e} "
                      f"ctrl={m['control_err']:.3e} ratio={row['ratio']:.1f} "
                      f"t={s.solve_time:.0f}s", flush=True)
                # As above: the next iteration needs 'h', so chain from the row, not from `m`.
                prev = row
            except Exception as e:
                print(f"  N={N} FAILED: {type(e).__name__}: {e}")
            p5_path.write_text(json.dumps(p5, indent=2))

    # ---------------- refit everything -------------------------------------------------
    print("\n" + "=" * 92)
    print("Least-squares convergence orders: slope of log(error) vs log(h)")
    print("=" * 92)
    print(f"  {'p':>2s} {'source':<30s} {'ctrl order':>11s} {'fit R2':>8s} "
          f"{'value order':>12s} {'fit R2':>8s}")
    summary = {}
    sources = []
    if fl["part_e_high_dimension"]:
        sources.append(("exp_control_floor part_e", fl["part_e_high_dimension"]))
    if p5["part_g_refinement"]:
        sources.append(("exp_control_p5 part_g", p5["part_g_refinement"]))
    co = RESULTS / "exp_control_order.json"
    if co.exists():
        d = json.loads(co.read_text())
        if d.get("part_c_orders_at_best_probe"):
            sources.append(("exp_control_order part_c", d["part_c_orders_at_best_probe"]))

    for lab, rows in sources:
        for p in sorted({r.get("p") for r in rows if r.get("p") is not None}):
            rs = sorted([r for r in rows if r.get("p") == p], key=lambda r: r["h"])
            if len(rs) < 3:
                continue
            oc, r2c = fit_orders(rs, "control_err")
            ov, r2v = fit_orders(rs, "value_err")
            print(f"  {p:>2d} {lab:<30s} {oc:>11.2f} {r2c:>8.3f} {ov:>12.2f} {r2v:>8.3f}")
            summary[f"{lab}|p={p}"] = {
                "control_order": oc, "control_fit_r2": r2c,
                "value_order": ov, "value_fit_r2": r2v,
                "N_range": [rs[0]["N"], rs[-1]["N"]], "n_points": len(rs)}

    (RESULTS / "exp_control_orders_fit.json").write_text(
        json.dumps({"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "method": "least-squares slope of log(error) vs log(h)",
                    "fits": summary}, indent=2))
    print(f"\nwrote {RESULTS / 'exp_control_orders_fit.json'}")

    # Headline: the control order is the one that should be quoted. Fits are only quotable if
    # the straight line actually describes the data, so a low R^2 is reported as a failure of
    # the sweep rather than averaged into the answer. The p=5 row from the floor study is exactly
    # that case: its snapshot cap silently coarsened the time resolution, and R^2 = 0.23 exposes
    # it. The cap-fixed rerun in exp_control_p5 is the quotable version.
    good = {k: v for k, v in summary.items() if v["control_fit_r2"] >= 0.95}
    bad = {k: v for k, v in summary.items() if v["control_fit_r2"] < 0.95}
    if good:
        co_ = [v["control_order"] for v in good.values()]
        cr_ = [v["control_fit_r2"] for v in good.values()]
        print(f"\n  QUOTABLE control error orders ({len(good)} sweeps, fit R^2 >= 0.95): "
              f"{min(co_):.2f} to {max(co_):.2f}")
        print(f"  fit quality:                                        "
              f"{min(cr_):.3f} to {max(cr_):.3f}")
    for k, v in bad.items():
        print(f"  NOT QUOTABLE (fit R^2 = {v['control_fit_r2']:.2f}): {k} "
              f"-- order {v['control_order']:.2f}; the sweep does not follow a power law")


if __name__ == "__main__":
    main()
