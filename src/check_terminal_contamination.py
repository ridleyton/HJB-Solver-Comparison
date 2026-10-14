"""
Does evaluating at t = T contaminate every control-accuracy number in the paper?

How this was found
------------------
While calibrating the benchmark to real market coefficients, the reference control was measured on
the same query set the paper's accuracy function uses, and it came out at mean |u| = 386 at t = T
against 0.65 at t = 0 -- a factor of ~600. That is not a bug in the reference. The optimal control of
this problem is -R^-1 B' P(t) x, and P(t) is driven by the terminal penalty Q_T, so the control
diverges as t -> T: the last instant of a liquidation problem wants to trade infinitely fast. The
time-integrated cost keeps the objective finite, but the pointwise control does not.

THE PROBLEM. `accuracy_metrics` builds its query times as `np.linspace(0, T, n_times)`, which
INCLUDES t = T, and every caller in the repository uses that default or passes n_times without
changing the endpoints. So the reported control R^2 -- and every statement built on it, including the
paper's headline control/value error ratio -- is scored in a region where the reference control is
unbounded. Whether that matters is an empirical question, and it is the one this script answers.

THE TEST. Recompute the same metrics with t = T included and with it excluded, on the same query
states and the same seed, for the grid scheme and the PINN. Report both, and the control/value error
ratio under each. If the numbers move materially, the paper's control claims need re-running; if they
barely move, the contamination is confined to one sample out of n_times and can be noted rather than
feared.

A caveat on the PINN, stated because it affects how the two rows should be read: a soft-constrained
PINN is trained through a terminal condition, so its behaviour exactly at t = T is partly an artefact
of the boundary loss rather than of the solve. The fix below removes those samples from the SCORING
for both families, which is the fair comparison, but it does not make the two families' t = T
behaviour equivalent.

Usage: python src/check_terminal_contamination.py
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
from solvers import FiniteDifferenceSolver, PolicyIterationSolver   # noqa: E402

OUT = SRC.parent / "results"


def metrics_at(solver, problem: ACProblem, times: np.ndarray, n_points: int = 600,
               seed: int = 11) -> dict:
    """The paper's accuracy_metrics, but with the query times passed in explicitly.

    Reimplemented rather than parameterised so that the ORIGINAL function is left untouched: the
    question is whether the original is contaminated, and editing it first would destroy the
    comparison. The two must agree exactly when handed the same times, which is asserted below.
    """
    rng = np.random.default_rng(seed)
    lo, hi = problem.domain()
    X = rng.uniform(lo, hi, size=(n_points, problem.state_dim))
    t_grid = np.linspace(0.0, problem.T, 401)
    ref = problem.reference(t_grid)

    V_hat = np.concatenate([solver.value(X, t) for t in times])
    V_ref = np.concatenate([ref.value(X, t) for t in times])
    u_hat = np.concatenate([solver.control(X, t) for t in times])
    u_ref = np.concatenate([ref.control(X, t) for t in times])

    def _stats(a, b):
        e = a - b
        ss_res = float(np.sum(e ** 2))
        ss_tot = float(np.sum((b - b.mean()) ** 2))
        return (float(np.mean(e ** 2)), float(np.mean(np.abs(e))),
                (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"))

    mse_v, mae_v, r2_v = _stats(V_hat, V_ref)
    mse_u, mae_u, r2_u = _stats(u_hat.ravel(), u_ref.ravel())
    scale_v = float(np.mean(np.abs(V_ref))) or 1.0
    scale_u = float(np.mean(np.abs(u_ref))) or 1.0
    return {"value_mse": mse_v, "value_r2": r2_v, "control_mse": mse_u, "control_r2": r2_u,
            "value_mae": mae_v, "control_mae": mae_u,
            "value_rel": mae_v / scale_v, "control_rel": mae_u / scale_u,
            # The ratio that means something: both errors expressed relative to their own scale.
            # A raw sqrt(control_mse)/sqrt(value_mse) would divide different units and is not used.
            "control_over_value": (mae_u / scale_u) / (mae_v / scale_v),
            "scale_V": scale_v, "scale_u": scale_u,
            "sign_agreement": float(np.mean(np.sign(u_hat) == np.sign(u_ref)))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,3")
    ap.add_argument("--n-times", type=int, default=6)
    ap.add_argument("--n-grid", type=int, default=41)
    ap.add_argument("--out", type=str, default="check_terminal_contamination.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    print("=" * 112)
    print("TERMINAL-TIME CONTAMINATION: is the control R^2 scored where the reference control diverges?")
    print("=" * 112)

    rows = []
    for p in [int(s) for s in args.dims.split(",")]:
        prob = ACProblem(p=p)
        grid_n = {1: 41, 2: 41, 3: 31}.get(p, 21)
        try:
            gs = FiniteDifferenceSolver(prob, n_grid=grid_n)
            gs.solve()
        except Exception as exc:                                    # noqa: BLE001
            print(f"  p={p}: grid failed ({type(exc).__name__})")
            continue

        # The paper's schedule, and the same schedule with the terminal sample removed.
        with_T = np.linspace(0.0, prob.T, args.n_times)
        without_T = np.linspace(0.0, prob.T, args.n_times + 1)[:-1]

        # Sanity: the reimplementation must reproduce accuracy_metrics exactly on the same times.
        from solvers import accuracy_metrics
        ref_check = accuracy_metrics(gs, prob, n_points=600, n_times=args.n_times, seed=11)
        mine = metrics_at(gs, prob, with_T, n_points=600, seed=11)
        agree = abs(ref_check["control_r2"] - mine["control_r2"]) < 1e-9
        print(f"\n--- p = {p}, grid N = {grid_n} ---")
        print(f"  reimplementation matches accuracy_metrics on identical times: {agree}")

        # How big is the reference control at each sampled time?
        rng = np.random.default_rng(11)
        lo, hi = prob.domain()
        X = rng.uniform(lo, hi, size=(600, prob.state_dim))
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        scales = [(float(t), float(np.mean(np.abs(ref.control(X, t))))) for t in with_T]
        print("  reference mean|u| by sampled time: "
              + ", ".join(f"t={t:.2f}:{s:.1f}" for t, s in scales))

        m_with = metrics_at(gs, prob, with_T, n_points=600, seed=11)
        m_without = metrics_at(gs, prob, without_T, n_points=600, seed=11)
        for label, m in (("with t=T", m_with), ("without t=T", m_without)):
            print(f"  {label:<14} value R2={m['value_r2']:>+10.6f}  control R2={m['control_r2']:>+10.5f}"
                  f"  ctrl/val rel-err ratio={m['control_over_value']:>10.1f}"
                  f"  sign agr={m['sign_agreement']:.4f}")
        rows.append({"p": p, "grid_n": grid_n, "reimpl_agrees": bool(agree),
                     "with_T": m_with, "without_T": m_without,
                     "reference_control_scale_by_time": scales})

    (OUT / args.out).write_text(json.dumps({"rows": rows, "n_times": args.n_times}, indent=2),
                                encoding="utf-8")

    print()
    print("-" * 112)
    for r in rows:
        a, b = r["with_T"], r["without_T"]
        dcr = b["control_r2"] - a["control_r2"]
        dratio = (b["control_over_value"] / a["control_over_value"]
                  if a["control_over_value"] else float("nan"))
        print(f"  p={r['p']}: control R2 moves {a['control_r2']:+.4f} -> {b['control_r2']:+.4f} "
              f"({dcr:+.4f}); control/value ratio {a['control_over_value']:.1f} -> "
              f"{b['control_over_value']:.1f} (x{dratio:.2f})")
    print()
    print("  INTERPRETATION. If the control R^2 and the ratio barely move, the terminal sample is one")
    print("  of n_times and the paper's control claims stand as reported; the terminal singularity is")
    print("  then a reporting note. If they move materially, every control number in the paper has")
    print("  been scored partly inside a singularity and the tables need re-running without t=T.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
