"""
Experiment: does the neural solver hold up as the state dimension grows?

Two things this fixes about the original p = 1..5 sweep.

1. **The original problem was separable.** With the uniform defaults (`Q = lam*I`, `R = eta*I`,
   `A = 0`) the value function is V(x,t) = sum_i v(q_i,t) for one scalar function v, so solving
   it at p = 20 is barely harder than at p = 1: the network applies one learned univariate
   function and sums. That is not what the curse-of-dimensionality argument is about. This
   experiment uses `hetero=True`, which gives each asset its own risk aversion and impact
   coefficient and puts random off-diagonal terms in Q, so V becomes a genuinely p-dimensional
   quadratic with a full P matrix. A control run with `hetero=False` at one dimension isolates
   how much of any difficulty is separability rather than dimension.

2. **The sweep stopped at p = 5.** Grid methods stop being runnable past that (N^p), which is
   itself the curse of dimensionality, but the *analytic* reference is available at any p (a
   matrix Riccati equation, p^2 ODEs), so accuracy can still be measured. This extends the range
   to p = 20 and reports grid feasibility alongside.

Output: results/exp_highdim.json
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
from exp_dimension import run_grid_method, run_pinn
from solvers import FiniteDifferenceSolver, accuracy_metrics

# Grid resolutions: kept small because the point here is to show where the grid stops being
# runnable, not to squeeze accuracy out of it. N=21 is the accuracy-matched resolution found for
# the separable problem; a coupled problem needs at least as much.
GRID_FOR_P = {1: 21, 2: 21, 3: 21, 4: 17, 5: 13}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,2,3,5,8,12,20")
    ap.add_argument("--epochs", type=int, default=8000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="results/exp_highdim.json")
    ap.add_argument("--control-p", type=int, default=8,
                    help="dimension at which to also run the separable (hetero=False) control")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "epochs": args.epochs,
            "problem": ("multi-asset Almgren-Chriss optimal execution, LQ form, with per-asset "
                        "heterogeneous risk/impact coefficients and random off-diagonal risk "
                        "(hetero=True) so the value function is not separable"),
            "control": f"p={args.control_p} also run with hetero=False (separable)",
            "reference": "closed-form matrix Riccati solution, exact at every p",
        },
        "runs": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---- PINN, heterogeneous, across dimensions --------------------------------------
    for p in dims:
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        seeds = [0, 1] if p <= 8 else [0]
        for sd in seeds:
            try:
                r = run_pinn(prob, sd, args.epochs, args.device)
                r.update({"method": "PINN", "p": p, "hetero": True, "feasible": True})
                print(f"  PINN hetero p={p:2d} seed={sd} t={r['solve_time_s']:8.1f}s "
                      f"R2={r['value_r2']:+.6f} ctrlR2={r['control_r2']:+.5f} "
                      f"sign={r['control_sign_agreement']:.3f}", flush=True)
            except Exception as e:
                r = {"method": "PINN", "p": p, "hetero": True, "seed": sd, "feasible": False,
                     "reason": f"{type(e).__name__}: {e}"}
                print(f"  PINN hetero p={p} seed={sd} FAILED: {e}", flush=True)
            results["runs"].append(r)
            save()

    # ---- separability control: same dimension, uniform parameters --------------------
    pc = args.control_p
    prob_c = ACProblem(p=pc, hetero=False)
    try:
        r = run_pinn(prob_c, 0, args.epochs, args.device)
        r.update({"method": "PINN", "p": pc, "hetero": False, "feasible": True})
        print(f"  PINN SEPARABLE p={pc} (control) t={r['solve_time_s']:8.1f}s "
              f"R2={r['value_r2']:+.6f}", flush=True)
    except Exception as e:
        r = {"method": "PINN", "p": pc, "hetero": False, "seed": 0, "feasible": False,
             "reason": f"{type(e).__name__}: {e}"}
    results["runs"].append(r)
    save()

    # ---- grid methods, heterogeneous, where they are runnable ------------------------
    for p in dims:
        if p not in GRID_FOR_P:
            results["runs"].append({
                "method": "Finite Difference", "p": p, "hetero": True, "feasible": False,
                "reason": (f"N^p at N={min(GRID_FOR_P.values())} is {min(GRID_FOR_P.values())**p:,} "
                           f"points before the CFL-limited time step is counted; not runnable"),
            })
            print(f"  grid p={p:2d}: not runnable (N^p)", flush=True)
            save()
            continue
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        n = GRID_FOR_P[p]
        try:
            r = run_grid_method(FiniteDifferenceSolver, prob, n)
            r.update({"method": "Finite Difference", "p": p, "hetero": True, "feasible": True})
            print(f"  grid hetero p={p:2d} N={n} t={r['solve_time_s']:9.3f}s "
                  f"R2={r['value_r2']:+.6f}", flush=True)
        except Exception as e:
            r = {"method": "Finite Difference", "p": p, "hetero": True, "feasible": False,
                 "reason": f"{type(e).__name__}: {e}"}
            print(f"  grid hetero p={p} FAILED: {e}", flush=True)
        results["runs"].append(r)
        save()

    # ---- summary ---------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("Accuracy against the closed-form solution, heterogeneous (non-separable) problem")
    print("=" * 96)
    print(f"  {'p':>3s} {'PINN R2':>22s} {'PINN time (s)':>14s} {'grid R2':>10s} "
          f"{'grid time (s)':>14s}")
    for p in dims:
        pin = [r for r in results["runs"]
               if r.get("method") == "PINN" and r.get("p") == p
               and r.get("hetero") is True and r.get("feasible")]
        grd = [r for r in results["runs"]
               if r.get("method") == "Finite Difference" and r.get("p") == p
               and r.get("feasible")]
        r2 = [r["value_r2"] for r in pin]
        t = [r["solve_time_s"] for r in pin]
        pstr = (f"{np.mean(r2):+.4f} (min {min(r2):+.3f})" if len(r2) > 1
                else (f"{r2[0]:+.4f}" if r2 else "--"))
        tstr = f"{np.mean(t):.0f}" if t else "--"
        gstr = f"{grd[0]['value_r2']:+.5f}" if grd else "n/a"
        gtstr = f"{grd[0]['solve_time_s']:.2f}" if grd else "--"
        print(f"  {p:>3d} {pstr:>22s} {tstr:>14s} {gstr:>10s} {gtstr:>14s}")

    ctrl = [r for r in results["runs"]
            if r.get("method") == "PINN" and r.get("hetero") is False and r.get("feasible")]
    if ctrl:
        het = [r for r in results["runs"] if r.get("method") == "PINN"
               and r.get("p") == pc and r.get("hetero") is True and r.get("feasible")]
        print(f"\n  separability control at p={pc}:")
        print(f"    separable (hetero=False): R2 = {ctrl[0]['value_r2']:+.6f}")
        if het:
            print(f"    heterogeneous           : R2 = {np.mean([r['value_r2'] for r in het]):+.6f}")

    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
