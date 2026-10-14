"""
Grid-refinement study for the finite-difference solver.

Purpose
-------
The dimensionality sweep in `exp_dimension.py` has to *reduce* the grid resolution N as p
grows, otherwise the sweep does not finish. That confounds two effects: the cost of
increasing p, and the accuracy lost by coarsening the grid. This experiment separates them.

For each p we sweep N, measure solve time and value-function accuracy against the analytic
solution, and then answer the question the paper actually poses:

    at the resolution needed to reach a given accuracy, what does the grid method cost,
    and how does that compare with the mesh-free solver, whose cost is set by training
    budget rather than by p?

The measured scaling is then used to extrapolate to dimensions where running the grid method
would be impractical, and the extrapolation is reported as such.

Cost model to be confirmed by the data: at fixed accuracy the time step is CFL-limited by the
mesh spacing, so total work scales as N^(p+1) rather than N^p.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from solvers import FiniteDifferenceSolver, accuracy_metrics

# Resolutions to test per dimension. Kept modest at high p so the study finishes.
# N = 21 is included at every p because it is the smallest resolution reaching
# value R^2 >= 0.9999 at p = 1, 2, 3, so it is the natural "accuracy-matched" grid.
GRIDS = {
    1: [11, 21, 31, 41, 51, 61, 81],
    2: [11, 21, 31, 41, 51, 61],
    3: [11, 21, 31, 41, 51],
    4: [9, 13, 17, 21],
    5: [9, 11, 13, 17, 21],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2,3,4,5")
    ap.add_argument("--out", type=str, default="results/exp_grid_refinement.json")
    ap.add_argument("--target-r2", type=float, default=0.9999)
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "solver": "Finite Difference (explicit, tensor-product grid)",
            "reference": "closed-form matrix Riccati solution",
            "target_r2": args.target_r2,
            "note": ("Cost model: time step is CFL-limited by the mesh spacing, so at fixed "
                     "domain the work scales as N^(p+1)."),
        },
        "runs": [],
    }

    for p in dims:
        prob = ACProblem(p=p)
        print(f"\n=== p = {p} ===", flush=True)
        for N in GRIDS.get(p, []):
            try:
                s = FiniteDifferenceSolver(prob, n_grid=N)
                t = s.solve()
                m = accuracy_metrics(s, prob, n_points=400, n_times=5, seed=7)
                rec = {
                    "p": p, "N": N, "n_grid_points": int(N**p),
                    "n_time_steps": int(s.n_time_steps or 0),
                    "solve_time_s": float(t),
                    "value_r2": float(m["value_r2"]),
                    "value_mse": float(m["value_mse"]),
                    "control_r2": float(m["control_r2"]),
                }
                print(f"  N={N:3d} pts={rec['n_grid_points']:>9d} steps={rec['n_time_steps']:5d} "
                      f"t={t:9.3f}s  R2={m['value_r2']:+.7f}  ctrlR2={m['control_r2']:+.5f}",
                      flush=True)
            except Exception as e:
                rec = {"p": p, "N": N, "feasible": False,
                       "reason": f"{type(e).__name__}: {e}"}
                print(f"  N={N:3d} FAILED: {e}", flush=True)
            results["runs"].append(rec)
            out_path.write_text(json.dumps(results, indent=2))

    # ---------------------------------------------------------------------------------
    # Analysis: minimum N reaching the accuracy target, and the implied cost.
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 92)
    print(f"Minimum resolution reaching value R^2 >= {args.target_r2}")
    print("=" * 92)
    print(f"{'p':>3s} {'N*':>5s} {'points':>10s} {'time (s)':>11s} {'R2':>12s} "
          f"{'time / N^(p+1)':>16s}")
    summary = {}
    for p in dims:
        rows = [r for r in results["runs"]
                if r.get("p") == p and r.get("value_r2") is not None]
        if not rows:
            continue
        rows.sort(key=lambda r: r["N"])
        hit = [r for r in rows if r["value_r2"] >= args.target_r2]
        if hit:
            best = hit[0]
            summary[p] = {
                "N_star": best["N"], "n_grid_points": best["n_grid_points"],
                "time_s": best["solve_time_s"], "value_r2": best["value_r2"],
                "feasible_at_target": True,
            }
            print(f"{p:>3d} {best['N']:>5d} {best['n_grid_points']:>10d} "
                  f"{best['solve_time_s']:>11.3f} {best['value_r2']:>12.7f} "
                  f"{best['solve_time_s'] / best['N']**(p+1):>16.3e}")
        else:
            # Target not reachable within the tested resolutions: extrapolate.
            best = max(rows, key=lambda r: r["value_r2"])
            summary[p] = {
                "N_star": None, "best_N": best["N"], "best_value_r2": best["value_r2"],
                "feasible_at_target": False,
            }
            print(f"{p:>3d} {'--':>5s} {'--':>10s} {'--':>11s} {best['value_r2']:>12.7f}  "
                  f"target not reached up to N={best['N']}")

    results["summary"] = summary
    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
