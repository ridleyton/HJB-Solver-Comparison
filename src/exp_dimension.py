"""
Experiment 1-2: accuracy against the analytic solution, and computational cost as a
function of state dimension p.

This is the honest version of the scalability claim in the original thesis. That work
asserted the curse of dimensionality but (i) quoted its only scalability evidence from
Xiao et al. (2024) rather than producing it (`main.tex:1256`), and (ii) never ran its own
solvers beyond four network inputs, let alone p > 10.

What is measured, for each method and dimension:
  * value-function accuracy against the closed-form Riccati solution (MSE, R^2)
  * control accuracy (R^2) and control-sign agreement with the analytic optimum
  * wall-clock solve time, grid points, and memory footprint
  * whether the method is feasible at all at that dimension

Method-specific notes
---------------------
* Grid methods (Finite Difference, Policy Iteration) need N^p points. Their time step is
  also CFL-limited by dx, so total work scales as N^(p+1) for a fixed domain. To keep the
  sweep tractable N is reduced as p grows; the accuracy penalty this causes is reported
  rather than hidden, because that penalty *is* the curse of dimensionality.
* The PINN is mesh-free: its cost is set by epochs and batch size, not by p. It is run
  with several seeds so that its reported accuracy is a mean +/- sd, not a single draw.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ac_problem import ACProblem
from solvers import (
    FiniteDifferenceSolver,
    PolicyIterationSolver,
    accuracy_metrics,
)

try:
    from pinn import PINNSolver
except Exception:  # pragma: no cover
    PINNSolver = None


# Grid resolution per dimension: keep the *work* roughly comparable so the sweep finishes,
# and report the resolution so the accuracy/feasibility trade-off is visible.
def grid_for(p: int) -> int | None:
    return {1: 41, 2: 41, 3: 31, 4: 21, 5: 13}.get(p, None)


def estimate_grid_memory_bytes(p: int, n_grid: int, n_snap: int = 41) -> int:
    return n_snap * (n_grid**p) * 8


def run_grid_method(cls, prob: ACProblem, n_grid: int, **kw):
    s = cls(prob, n_grid=n_grid, **kw)
    t0 = time.perf_counter()
    solve_time = s.solve()
    wall = time.perf_counter() - t0
    m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
    return {
        "solve_time_s": float(solve_time),
        "wall_time_s": float(wall),
        "n_grid": n_grid,
        "n_grid_points": int(n_grid**prob.p),
        "n_time_steps": int(s.n_time_steps or 0),
        "est_memory_bytes": int(estimate_grid_memory_bytes(prob.p, n_grid)),
        **m,
    }


def run_pinn(prob: ACProblem, seed: int, epochs: int, device: str, hidden: int = 64,
             layers: int = 4):
    if PINNSolver is None:
        return None
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    s = PINNSolver(prob, hidden=hidden, layers=layers, n_collocation=2048,
                   epochs=epochs, lr=2e-3, device=device, seed=seed)
    solve_time = s.solve()
    m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
    peak = (torch.cuda.max_memory_allocated() if device == "cuda" else 0)
    return {
        "solve_time_s": float(solve_time),
        "wall_time_s": float(solve_time),
        "n_grid": None,
        "n_grid_points": None,
        "n_time_steps": int(epochs),
        "est_memory_bytes": int(peak),
        "residual_scale": float(s.residual_scale or 0.0),
        "epochs": int(epochs),
        "seed": int(seed),
        **m,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2,3,4,5")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results/exp_dimension.json")
    ap.add_argument("--skip-grid", action="store_true")
    ap.add_argument("--skip-pinn", action="store_true")
    ap.add_argument("--pi-max-p", type=int, default=3,
                    help=("Largest dimension at which Howard policy iteration is run. PI "
                          "costs ~13-15 grid sweeps per solve, so beyond p=3 it dominates "
                          "the sweep's runtime; the grid-cost scaling is measured instead "
                          "and reported as an extrapolation for larger p."))
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "device": args.device,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "epochs": args.epochs,
            "seeds": seeds,
            "problem": "multi-asset Almgren-Chriss optimal execution, LQ form",
            "reference": "closed-form matrix Riccati solution",
        },
        "runs": [],
    }

    for p in dims:
        prob = ACProblem(p=p)
        print(f"\n=== p = {p} ===", flush=True)

        if not args.skip_grid:
            n_grid = grid_for(p)
            if n_grid is None:
                print("  grid methods: skipped (infeasible at this dimension for N>=13)", flush=True)
                results["runs"].append({
                    "method": "Finite Difference", "p": p, "feasible": False,
                    "reason": "grid size N^p infeasible",
                })
                results["runs"].append({
                    "method": "Policy Iteration", "p": p, "feasible": False,
                    "reason": "grid size N^p infeasible",
                })
            else:
                for cls, nm, kw in (
                    (FiniteDifferenceSolver, "Finite Difference", {}),
                    (PolicyIterationSolver, "Policy Iteration", {"max_policy_iter": 15}),
                ):
                    if nm == "Policy Iteration" and p > args.pi_max_p:
                        r = {
                            "method": nm, "p": p, "feasible": False,
                            "reason": (f"not run beyond p={args.pi_max_p}: policy iteration "
                                       f"costs ~13-15 grid sweeps per solve; grid cost is "
                                       f"reported as an extrapolation instead"),
                        }
                        print(f"  {nm:20s} skipped at p={p} (capped at "
                              f"p<={args.pi_max_p}); grid cost extrapolated", flush=True)
                        results["runs"].append(r)
                        out_path.write_text(json.dumps(results, indent=2))
                        continue
                    try:
                        r = run_grid_method(cls, prob, n_grid, **kw)
                        r.update({"method": nm, "p": p, "feasible": True})
                        print(f"  {nm:20s} N={n_grid:3d} pts={r['n_grid_points']:>9d} "
                              f"t={r['solve_time_s']:8.3f}s R2={r['value_r2']:+.6f} "
                              f"ctrlR2={r['control_r2']:+.5f} sign={r['control_sign_agreement']:.3f}",
                              flush=True)
                    except Exception as e:  # memory / time guard
                        r = {"method": nm, "p": p, "feasible": False,
                             "reason": f"{type(e).__name__}: {e}"}
                        print(f"  {nm:20s} FAILED: {e}", flush=True)
                    results["runs"].append(r)
                    out_path.write_text(json.dumps(results, indent=2))

        if not args.skip_pinn:
            for sd in seeds:
                try:
                    r = run_pinn(prob, sd, args.epochs, args.device)
                    if r is None:
                        break
                    r.update({"method": "PINN", "p": p, "feasible": True})
                    print(f"  {'PINN':20s} seed={sd} t={r['solve_time_s']:8.2f}s "
                          f"R2={r['value_r2']:+.6f} ctrlR2={r['control_r2']:+.5f} "
                          f"sign={r['control_sign_agreement']:.3f} "
                          f"mem={r['est_memory_bytes']/1e6:.1f}MB", flush=True)
                except Exception as e:
                    r = {"method": "PINN", "p": p, "seed": sd, "feasible": False,
                         "reason": f"{type(e).__name__}: {e}"}
                    print(f"  PINN seed={sd} FAILED: {e}", flush=True)
                results["runs"].append(r)
                out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
