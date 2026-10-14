"""
Where does the benchmark stop being solvable? The regime boundary, not a single calibrated point.

What this was meant to be, and what it turned out to be
------------------------------------------------------
The plan was a straightforward robustness check: rerun the comparison with coefficients estimated
from real market data instead of synthetic ones, and confirm the findings do not move. exp_real_
calibration.py did that, and the answer was not the expected one. At the synthetic coefficients the
grid scheme scores value R^2 ~ 1.0000 and control R^2 ~ 0.999. At the coefficients SPY's own data
implies, the same solver scores value R^2 = -997 and control R^2 = -26760 -- it has not lost a little
accuracy, it has failed.

So the useful question is no longer "does the comparison survive calibration" but "where is the
boundary". This sweeps the liquidity parameter k = sqrt(Q/eta) through the range real data implies
and past the synthetic default, and reports for each k the solver's accuracy, the magnitude of the
value function, and the control's scale. The point is to say where the benchmark is informative and
where it is not, which is a stronger claim than either a single calibrated run or a single synthetic
one.

Why k is the right axis
-----------------------
k = sqrt(Q/eta) = sigma*sqrt(lambda)/sqrt(eta) is dimensionless and sets how fast the position is
liquidated: the deterministic optimal inventory decays like exp(-k t) over the horizon. The paper's
synthetic coefficients give k = 1. Real coefficients with a defensible lambda in [1e-6, 1e-2] give
k in [0.001, 0.1] -- an order of magnitude SLOWER liquidation, i.e. a much weaker inventory-risk
penalty than the benchmark assumes. That gap is the substantive finding: it is not that the synthetic
numbers are wrong, it is that they sit in one corner of the space real liquidity admits.

Usage: python src/exp_regime_boundary.py --ks 0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0
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

from ac_problem import ACProblem                                    # noqa: E402
from solvers import FiniteDifferenceSolver, accuracy_metrics        # noqa: E402

OUT = SRC.parent / "results"

# SPY's measured coefficients, used as the reference liquidity setting.
SIGMA_SPY = 0.0099
ETA_SPY = 1.281e-4
Q_TERMINAL = 0.1


def problem_for_k(k: float, sigma: float = SIGMA_SPY, eta: float = ETA_SPY) -> ACProblem:
    """The benchmark at a prescribed liquidation speed k, holding the measured sigma and eta.

    lambda is recovered from k rather than assumed: k = sigma*sqrt(lambda)/sqrt(eta) inverts to
    lambda = k^2 * eta / sigma^2. So every row of the sweep has the SAME measured market liquidity
    and differs only in how much inventory risk the trader is assumed to price.
    """
    lam = (k ** 2) * eta / (sigma ** 2)
    return ACProblem(p=1, eta=eta, lam_sigma2=lam * sigma ** 2, q_terminal=Q_TERMINAL, sigma=sigma)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ks", type=str, default="0.001,0.003,0.01,0.03,0.1,0.3,1.0,3.0")
    ap.add_argument("--n-grid", type=int, default=41)
    ap.add_argument("--out", type=str, default="exp_regime_boundary.json")
    args = ap.parse_args()
    ks = [float(s) for s in args.ks.split(",") if s.strip()]

    x0 = np.zeros(1)
    x0[0] = 1.0
    print("=" * 118)
    print("REGIME BOUNDARY: for which liquidity settings is the benchmark informative?")
    print("=" * 118)
    print(f"  measured SPY coefficients held fixed: sigma = {SIGMA_SPY}, eta = {ETA_SPY} "
          f"(half the Roll spread)")
    print("  k = sqrt(Q/eta) is swept; lambda is recovered from k, so only the risk preference "
          "changes.")
    print("  the paper's synthetic default is k = 1.0\n")
    print(f"  {'k':>7} {'lambda':>11} {'Q':>10} {'mean|V|':>10} {'mean|u|':>10} "
          f"{'value R2':>12} {'ctrl R2':>13} {'sign agr':>9} {'steps':>7} {'verdict':>22}")

    rows = []
    for k in ks:
        prob = problem_for_k(k)
        lam = (k ** 2) * ETA_SPY / (SIGMA_SPY ** 2)
        t_grid = np.linspace(0.0, prob.T, 401)
        ref = prob.reference(t_grid)
        rng = np.random.default_rng(1)
        lo, hi = prob.domain()
        X = rng.uniform(lo, hi, size=(400, prob.state_dim))
        times = np.linspace(0.0, prob.T, 3)
        Vt = np.concatenate([ref.value(X, t) for t in times])
        ut = np.concatenate([ref.control(X, t) for t in times])
        mean_V, mean_u = float(np.mean(np.abs(Vt))), float(np.mean(np.abs(ut)))

        try:
            s = FiniteDifferenceSolver(prob, n_grid=args.n_grid)
            s.solve()
            m = accuracy_metrics(s, prob, n_points=400, seed=1)
        except Exception as exc:                                   # noqa: BLE001
            print(f"  {k:>7.4f} {lam:>11.3e} {'--':>10} {mean_V:>10.3e} {mean_u:>10.3e} "
                  f"{'FAILED':>12} {'':>13} {'':>9} {'':>7} {type(exc).__name__:>22}")
            rows.append({"k": k, "lam": lam, "mean_V": mean_V, "mean_u": mean_u, "failed": True})
            continue

        vr2, cr2 = m["value_r2"], m["control_r2"]
        if vr2 is None or cr2 is None or not (math.isfinite(vr2) and math.isfinite(cr2)):
            verdict = "non-finite"
        elif vr2 > 0.99 and cr2 > 0.99:
            verdict = "usable"
        elif vr2 > 0.9:
            verdict = "value only"
        else:
            verdict = "BROKEN"
        print(f"  {k:>7.4f} {lam:>11.3e} {lam * SIGMA_SPY ** 2:>10.3e} {mean_V:>10.3e} "
              f"{mean_u:>10.3e} {vr2:>+12.5f} {cr2:>+13.4f} "
              f"{m['control_sign_agreement']:>9.4f} {str(m['n_time_steps']):>7} {verdict:>22}")
        rows.append({"k": k, "lam": lam, "Q": lam * SIGMA_SPY ** 2, "mean_V": mean_V,
                     "mean_u": mean_u, "value_r2": vr2, "control_r2": cr2, "verdict": verdict,
                     "control_sign_agreement": m["control_sign_agreement"],
                     "n_time_steps": m["n_time_steps"], "value_mse": m["value_mse"],
                     "control_mse": m["control_mse"]})

    (OUT / args.out).write_text(json.dumps({"rows": rows, "sigma": SIGMA_SPY, "eta": ETA_SPY,
                                            "n_grid": args.n_grid}, indent=2), encoding="utf-8")

    good = [r for r in rows if r.get("verdict") == "usable"]
    print()
    print("-" * 118)
    if good:
        print(f"  The solver is usable for k >= {min(r['k'] for r in good):.4f}. Real coefficients")
        print("  with lambda in [1e-6, 1e-2] put k in [0.001, 0.1], which is BELOW or at the edge of")
        print("  that range; the paper's synthetic default (k = 1) is comfortably inside it.")
    else:
        print("  No swept k is usable -- the boundary is above the whole sweep.")
    print()
    print("  WHAT THIS MEANS FOR THE PAPER, stated so it cannot be over-read: this is NOT evidence")
    print("  that the synthetic coefficients are wrong. They define a well-posed problem with an")
    print("  exact solution, which is what a solver benchmark needs. It is evidence that they sit at")
    print("  faster liquidation than measured liquidity plus realistic risk aversion implies, and it")
    print("  fixes the range over which the paper's claims have been demonstrated.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
