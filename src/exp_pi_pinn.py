"""
Comparison: PI-PINN vs direct PINN vs grid policy iteration vs finite differences.

Motivation
----------
`HeeJunYang09/hji-pi-pinn-repro` implements policy-iteration PINNs for viscous HJ/HJI equations.
Policy iteration is the same idea as our grid `PolicyIterationSolver`: freeze the control, solve
the resulting *linear* PDE, then improve the control from the gradient. The question this
experiment answers is whether that structure helps a *network* the way it helps a *grid*.

Four methods on one problem, so the representation and the PDE-treatment axes are separated:

                 nonlinear PDE            linear PDE per policy iteration
    grid         FiniteDifferenceSolver     PolicyIterationSolver
    network      PINNSolver (direct)       PIPINN

The problem is the non-separable LQ optimal-execution HJB (per-asset heterogeneous coefficients
and random off-diagonal risk), which is a viscous HJ equation with a quadratic-in-gradient
Hamiltonian -- exactly the class their scheme targets.

Budget
------
Their default is num_iters=500 x num_epochs=5000 = 2.5e6 optimiser steps, ~3 orders of magnitude
more than our sweep's 6e3-8e3 and on the order of 10-40 h per run here. We therefore run at a
matched total step count and sweep how it is split across outer iterations. That isolates the
question "does the PI structure help?" from "was it given more steps?" -- and the first smoke
test already showed the split matters a great deal: at 20x300 the PI-PINN lost badly to the
direct PINN because 300 epochs is not enough to solve the linear sub-problem before the policy is
replaced.

Output: results/exp_pi_pinn.json
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
from pinn import PINNSolver
from pinn_pi import PIPINN
from solvers import (
    FiniteDifferenceSolver,
    PolicyIterationSolver,
    accuracy_metrics,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--total-steps", type=int, default=24000,
                    help="total optimiser steps, shared by PI-PINN and the direct PINN")
    ap.add_argument("--splits", default="2x12000,4x6000,8x3000,16x1500,32x750,48x500",
                    help="num_iters x num_epochs splits of the total budget")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grid-n", type=int, default=21)
    ap.add_argument("--out", default="results/exp_pi_pinn.json")
    args = ap.parse_args()

    prob = ACProblem(p=args.p, hetero=True, het_seed=1)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    splits = []
    for tok in args.splits.split(","):
        ni, ne = tok.strip().split("x")
        splits.append((int(ni), int(ne)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem": (f"LQ optimal execution, p={args.p}, hetero=True (non-separable), "
                        f"closed-form Riccati reference"),
            "total_steps": args.total_steps,
            "splits": [f"{a}x{b}" for a, b in splits],
            "seeds": seeds,
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "attribution": ("PI-PINN reimplements the scheme of HeeJunYang09/hji-pi-pinn-repro "
                            "(sine-activation MLP, Adam 1e-3, frozen-policy evaluation then "
                            "policy improvement, budget parity between PI and direct). Their "
                            "JAX code could not be run here; see src/pinn_pi.py for what was "
                            "taken from their source and where."),
        },
        "runs": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---- grid methods: fast, run once ------------------------------------------------
    print("grid methods")
    for cls, nm, kw in ((FiniteDifferenceSolver, "Finite Difference", {}),
                        (PolicyIterationSolver, "Policy Iteration", {"max_policy_iter": 25})):
        try:
            s = cls(prob, n_grid=args.grid_n, **kw)
            t0 = time.perf_counter()
            s.solve()
            wall = time.perf_counter() - t0
            m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
            r = {"method": nm, "kind": "grid", "feasible": True,
                 "solve_time_s": float(wall), "total_steps": None, **m}
            print(f"  {nm:24s} {wall:8.3f}s  value_R2={m['value_r2']:+.6f} "
                  f"ctrl_R2={m['control_r2']:+.5f} sign={m['control_sign_agreement']:.3f}")
        except Exception as e:
            r = {"method": nm, "kind": "grid", "feasible": False,
                 "reason": f"{type(e).__name__}: {e}"}
            print(f"  {nm} FAILED: {e}")
        results["runs"].append(r)
        save()

    # ---- direct PINN at the same total budget ---------------------------------------
    print(f"\ndirect PINN at {args.total_steps} steps")
    for sd in seeds:
        s = PINNSolver(prob, hidden=64, layers=4, n_collocation=2048,
                       epochs=args.total_steps, lr=2e-3, device=args.device, seed=sd)
        st = s.solve()
        m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
        r = {"method": "Direct PINN", "kind": "network", "feasible": True, "seed": sd,
             "solve_time_s": float(st), "total_steps": args.total_steps, **m}
        results["runs"].append(r)
        print(f"  seed={sd} {st:7.1f}s  value_R2={m['value_r2']:+.6f} "
              f"ctrl_R2={m['control_r2']:+.5f} sign={m['control_sign_agreement']:.3f}")
        save()

    # ---- PI-PINN, sweeping the outer/inner split ------------------------------------
    print(f"\nPI-PINN at the same {args.total_steps} steps, varying iters x epochs")
    for (ni, ne) in splits:
        if ni * ne != args.total_steps:
            print(f"  skip {ni}x{ne} (does not equal {args.total_steps})")
            continue
        for sd in seeds:
            s = PIPINN(prob, num_iters=ni, num_epochs=ne, n_points=2048,
                       refresh_every=500, lr=1e-3, device=args.device, seed=sd)
            st = s.solve(direct=False)
            m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
            r = {"method": "PI-PINN", "kind": "network", "feasible": True, "seed": sd,
                 "num_iters": ni, "num_epochs": ne, "policy_updates": s.n_policy_updates,
                 "solve_time_s": float(st), "total_steps": args.total_steps, **m}
            results["runs"].append(r)
            print(f"  {ni:3d}x{ne:<5d} seed={sd} {st:7.1f}s  value_R2={m['value_r2']:+.6f} "
                  f"ctrl_R2={m['control_r2']:+.5f} sign={m['control_sign_agreement']:.3f}")
            save()

    # ---- summary --------------------------------------------------------------------
    print("\n" + "=" * 98)
    print(f"value R^2 against the closed-form solution   (p={args.p}, hetero, "
          f"{args.total_steps} optimiser steps)")
    print("=" * 98)
    grouped: dict[str, list[float]] = {}
    for r in results["runs"]:
        if not r.get("feasible") or r.get("value_r2") is None:
            continue
        key = (r["method"] if r["method"] != "PI-PINN"
               else f"PI-PINN {r['num_iters']}x{r['num_epochs']}")
        grouped.setdefault(key, []).append(r["value_r2"])
    for k, v in sorted(grouped.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"  {k:26s} {np.mean(v):+.6f}"
              + (f"  (min {min(v):+.4f})" if len(v) > 1 else ""))
    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
