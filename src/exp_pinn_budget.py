"""
Fairness check: does a larger training budget rescue the PINN at higher dimension?

The dimensionality sweep (`exp_dimension.py`) shows the PINN's accuracy degrading with p and
failing outright at p = 4 under a fixed budget of 6000 epochs and 2048 collocation points.
A referee is entitled to ask whether that is a property of the method or of the budget. This
experiment answers it by scaling the budget along both axes -- optimisation steps and
collocation points -- and tracking accuracy against the analytic solution.

If accuracy recovers with budget, the sweep's conclusion must be stated as budget-conditional.
If it does not, the failure is structural for this problem at this dimension, which is the
claim the paper makes.

Reported for each configuration: budget (epochs x batch), wall-clock time, value R^2,
control R^2, and sign agreement, over one or more seeds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ac_problem import ACProblem
from exp_dimension import run_pinn
from solvers import accuracy_metrics
from pinn import PINNSolver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="3,4")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results/exp_pinn_budget.json")
    ap.add_argument("--quick", action="store_true",
                    help="smaller budget ladder, for a fast confirmation")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",") if d.strip()]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # (epochs, collocation points) ladder. Kept modest so the study finishes in a
    # reasonable time while still spanning ~8x in optimisation steps.
    if args.quick:
        ladder = [(3000, 2048), (6000, 2048)]
    else:
        ladder = [(6000, 2048), (20000, 2048), (6000, 8192)]

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "device": args.device,
            "gpu": (torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
            "ladder": ladder,
            "note": ("Tests whether the PINN's accuracy loss with dimension is a budget "
                     "artefact. Architecture (4 x 64 tanh) and learning rate are held fixed "
                     "across the ladder, matching exp_dimension.py."),
        },
        "runs": [],
    }

    for p in dims:
        prob = ACProblem(p=p)
        print(f"\n=== p = {p} ===", flush=True)
        for (epochs, ncol) in ladder:
            # Use both seeds for the cheap end of the ladder, one for the expensive end.
            seeds = [0, 1] if epochs <= 6000 else [0]
            for seed in seeds:
                try:
                    t0 = time.perf_counter()
                    s = PINNSolver(prob, hidden=64, layers=4, n_collocation=ncol,
                                   epochs=epochs, lr=2e-3, device=args.device, seed=seed)
                    st = s.solve()
                    m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
                    rec = {
                        "p": p, "epochs": epochs, "n_collocation": ncol, "seed": seed,
                        "budget_steps": int(epochs * ncol),
                        "solve_time_s": float(st),
                        "value_r2": float(m["value_r2"]),
                        "value_mse": float(m["value_mse"]),
                        "control_r2": float(m["control_r2"]),
                        "control_sign_agreement": float(m["control_sign_agreement"]),
                        "residual_scale": float(s.residual_scale or 0.0),
                    }
                    print(f"  epochs={epochs:6d} batch={ncol:5d} seed={seed} "
                          f"steps={rec['budget_steps']:>11d} t={st:8.1f}s "
                          f"R2={rec['value_r2']:+.6f} ctrlR2={rec['control_r2']:+.5f} "
                          f"sign={rec['control_sign_agreement']:.3f}", flush=True)
                except Exception as e:
                    rec = {"p": p, "epochs": epochs, "n_collocation": ncol, "seed": seed,
                           "feasible": False, "reason": f"{type(e).__name__}: {e}"}
                    print(f"  epochs={epochs} batch={ncol} seed={seed} FAILED: {e}",
                          flush=True)
                results["runs"].append(rec)
                out_path.write_text(json.dumps(results, indent=2))

    # ---- summary -------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("Accuracy vs training budget")
    print("=" * 92)
    print(f"{'p':>3s} {'epochs':>8s} {'batch':>7s} {'seeds':>6s} "
          f"{'mean R2':>12s} {'best R2':>12s} {'time (s)':>10s}")
    for p in dims:
        for (epochs, ncol) in ladder:
            sel = [r for r in results["runs"]
                   if r.get("p") == p and r.get("epochs") == epochs
                   and r.get("n_collocation") == ncol and "value_r2" in r]
            if not sel:
                continue
            r2 = [r["value_r2"] for r in sel]
            tt = float(np.mean([r["solve_time_s"] for r in sel]))
            print(f"{p:>3d} {epochs:>8d} {ncol:>7d} {len(sel):>6d} "
                  f"{np.mean(r2):>+12.6f} {max(r2):>+12.6f} {tt:>10.1f}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
