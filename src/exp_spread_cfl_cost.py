"""
The cost half of the paper's claim, tested against real market spreads.

What prompted this
------------------
While calibrating to real data, the accuracy-matched grid solve stopped finishing at p = 2. The
cause is not dimension. The explicit scheme's time step is CFL-limited by the advection bound
dt <= 0.25 dx / |u|_inf, and the optimal control is u* = -R^-1 B' P x with R = eta I. Real market
data gives a much smaller eta than the paper's synthetic 0.01 -- eta is half the Roll effective
spread, 1.28e-4 for SPY, 78x smaller -- so the optimal rate is ~78x larger, and the required number
of time steps grows with it:

    synthetic (eta = 0.01)      max|u| on the grid = 1.25e+01  ->  dt = 1.25e-03  ->    800 steps
    SPY       (eta = 1.28e-04)  max|u| on the grid = 9.76e+02  ->  dt = 1.60e-05  -> 62,450 steps

WHY THIS MATTERS FOR THE PAPER. The headline boundary claim is that where a grid scheme can run it is
BOTH faster and more accurate, by 5-50x in time at d = 4-5. The accuracy half of that survives real
coefficients (exp_real_calibration.py). The speed half may not: if a market-calibrated eta multiplies
the time-step count by 78, the grid's wall-clock advantage can invert, and the practical
recommendation would change from "use the grid where it runs" to "use the grid where it runs AND the
instrument is not too liquid".

That is a strong claim, so it is measured rather than asserted. This sweeps eta across the range real
instruments imply and past the synthetic value, holding everything else fixed, and reports the time
step, the step count, and the implied work. Work -- not seconds -- is the honest axis here, because
it is hardware-independent and the paper already counts work in FLOPs elsewhere.

WHAT IS AND IS NOT CLAIMED. This measures the EXPLICIT scheme's time-step requirement, which is what
the paper's own solver uses and what its CFL discussion is about. The paper also reports an implicit
(one-dimensional splitting) variant that removes the stability restriction at four times the explicit
step; that was measured at the synthetic eta only, and this script does NOT re-measure it at real eta.
So the correct reading is a constraint on the explicit scheme's cost, not a claim that no implicit
scheme can be fast at real spreads.

Usage: python src/exp_spread_cfl_cost.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                # noqa: E402
from solvers import _GridSolver                 # noqa: E402

OUT = SRC.parent / "results"

# Measured half-Roll spreads, from data_causal/preprocess_manifest.json.
SPREADS = {"SPY": 2.563e-4, "QQQ": 3.199e-4, "MSFT": 3.155e-4, "AMZN": 2.915e-4,
           "GOOGL": 3.946e-4, "OXY": 4.486e-4, "AAPL": 5.453e-4, "NVDA": 5.370e-4,
           "TSLA": 8.828e-4}


def steps_for(eta: float, sigma: float = 0.01, lam_sigma2: float = 0.01, n_grid: int = 41,
              p: int = 1) -> dict:
    """Time step, step count and per-step grid work for the explicit scheme at this eta."""
    prob = ACProblem(p=p, eta=eta, lam_sigma2=lam_sigma2, q_terminal=0.1, sigma=sigma)
    s = _GridSolver(prob, n_grid=n_grid)
    s._build_grid()
    V = s._terminal()
    _, v = s._hamiltonian_and_control(V)
    max_u = float(np.max(np.abs(v)))
    dt = s._stable_dt(max_u)
    n_steps = prob.T / dt
    # Per-step cost is one grid-wide stencil pass: the point count, independent of eta.
    n_points = n_grid ** p
    return {"eta": eta, "max_u": max_u, "dt": dt, "n_steps": n_steps,
            "n_points": n_points, "grid_work": n_steps * n_points,
            "k": math.sqrt(lam_sigma2 / eta)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=41)
    ap.add_argument("--out", type=str, default="exp_spread_cfl_cost.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 110)
    print("SPREAD -> CFL COST: how real bid-ask spreads change the grid scheme's time-step requirement")
    print("=" * 110)
    print("  eta = half the Roll effective spread; dt is the explicit advection CFL bound")
    print("  dt <= 0.25 dx / |u|_inf, so dt scales as eta while the step count scales as 1/eta.")
    print("  lam_sigma2 and sigma held at the paper's values, so only the instrument's liquidity "
          "varies.\n")

    grid_rows = []
    for eta in (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2.563e-4, 8.828e-4):
        r = steps_for(eta, n_grid=args.n_grid)
        grid_rows.append({**r, "label": "synthetic" if abs(eta - 1e-2) < 1e-12 else ""})
    grid_rows.sort(key=lambda r: -r["eta"])

    base = next(r for r in grid_rows if abs(r["eta"] - 1e-2) < 1e-12)
    print(f"  {'eta':>10} {'max|u|':>10} {'dt':>11} {'steps':>11} {'vs synthetic':>13} "
          f"{'grid work':>12}")
    for r in grid_rows:
        print(f"  {r['eta']:>10.3e} {r['max_u']:>10.2f} {r['dt']:>11.3e} {r['n_steps']:>11.0f} "
              f"{r['n_steps'] / base['n_steps']:>12.1f}x {r['grid_work']:>12.3e}")

    print()
    print("  By instrument, at each one's own measured half-Roll spread:")
    print(f"  {'asset':>7} {'eta':>10} {'max|u|':>10} {'steps':>11} {'vs synthetic':>13}")
    per_asset = []
    for name, spread in sorted(SPREADS.items(), key=lambda kv: kv[1]):
        r = steps_for(spread / 2.0, n_grid=args.n_grid)
        per_asset.append({**r, "asset": name})
        print(f"  {name:>7} {r['eta']:>10.3e} {r['max_u']:>10.2f} {r['n_steps']:>11.0f} "
              f"{r['n_steps'] / base['n_steps']:>12.1f}x")

    (OUT / args.out).write_text(json.dumps(
        {"eta_sweep": grid_rows, "per_asset": per_asset, "baseline": base,
         "n_grid": args.n_grid}, indent=2), encoding="utf-8")

    worst = max(per_asset, key=lambda r: r["n_steps"])
    best = min(per_asset, key=lambda r: r["n_steps"])
    print()
    print("-" * 110)
    print(f"  Across the nine instruments the explicit scheme needs {best['n_steps'] / base['n_steps']:.0f}x "
          f"to {worst['n_steps'] / base['n_steps']:.0f}x the synthetic run's time steps "
          f"({best['asset']} to {worst['asset']}).")
    print()
    print("  WHAT THIS DOES AND DOES NOT SAY. It constrains the EXPLICIT scheme, which is the one the")
    print("  paper's CFL discussion and cost tables use. The paper's implicit variant was measured at")
    print("  the synthetic eta only and is NOT re-measured here, so this is not evidence that no")
    print("  implicit scheme is fast at real spreads. It is evidence that the paper's 5-50x grid")
    print("  speed-up is a statement about the synthetic calibration, and that a market-calibrated")
    print("  cost comparison has to be redone before the speed claim transfers.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
