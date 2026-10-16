"""Where does the error actually live? Boundary cells versus the interior.

WHY THIS EXPERIMENT EXISTS
--------------------------
Accuracy is scored on points drawn uniformly from the state domain, and a point is "in a boundary
cell" whenever it sits in the outermost cell of any axis. That fraction is a geometric fact,

    P(boundary cell) = 1 - ((N - 2) / N)^d,

which at N = 21 is 10% in one dimension and 41% in five. The solver's stencils are one-sided there,
the interpolation is weakest there, and -- for an execution problem -- it is exactly where a risk
limit or an inventory constraint bites. So a single domain-wide error number mixes two different
regimes, and at high dimension the boundary regime is the larger share of the average.

This script separates them: same solve, same nodes, error reported per region.

Usage: python src/exp_boundary_split.py [--dims 1,2,3,4] [--grids 11,21,31]
Output: results/exp_boundary_split.json
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from metrics import relative_l2  # noqa: E402
from solvers import FiniteDifferenceSolver  # noqa: E402
from exp_control_extraction import extract_cell, extract_spline  # noqa: E402

RESULTS = SRC.parent / "results"


def boundary_mask(X: np.ndarray, axes: list[np.ndarray], lo: np.ndarray,
                  hi: np.ndarray) -> np.ndarray:
    """True where the point lies within one cell of the PROBLEM domain's edge on any axis.

    Not the grid's edge: every solver here pads the domain by 0.25 before building the grid, so the
    grid's outermost cells are never queried at all (measured: a 0.000 boundary share when the mask
    was written that way, which is a fact about the padding rather than about the error). The band
    that matters is the outermost cell of the domain the problem is posed on.
    """
    h = axes[0][1] - axes[0][0]
    m = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[1]):
        m |= (X[:, i] <= lo[i] + h) | (X[:, i] >= hi[i] - h)
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,2,3,4")
    ap.add_argument("--grids", default="11,21,31")
    ap.add_argument("--n-points", type=int, default=6000)
    ap.add_argument("--out", default="exp_boundary_split.json")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    grids = [int(x) for x in args.grids.split(",")]
    print("=" * 100)
    print("BOUNDARY VERSUS INTERIOR: one solve, two regimes")
    print("=" * 100)

    out = {"dims": {}}
    for p in dims:
        prob = ACProblem(p=p)
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        lo, hi = prob.domain()
        print(f"\n  d = {p}")
        print(f"    {'N':>4s} {'bndy share':>11s} {'val rms int':>12s} {'val rms bnd':>12s} "
              f"{'ctrl rms int':>13s} {'ctrl rms bnd':>13s} {'bnd/int':>8s} "
              f"{'share of err':>13s} {'gain int':>9s} {'gain bnd':>9s}")
        rows = []
        for N in grids:
            t0 = time.perf_counter()
            solver = FiniteDifferenceSolver(prob, n_grid=N, n_snap=3)
            solver.solve()
            rng = np.random.default_rng(4)
            X = rng.uniform(lo, hi, size=(args.n_points, prob.state_dim))
            bnd = boundary_mask(X, solver.axes, lo, hi)
            h = solver.axes[0][1] - solver.axes[0][0]
            frac_per_axis = 2.0 * h / (hi[0] - lo[0])
            V_ref = np.asarray(ref.value(X, 0.0)).ravel()
            u_ref = ref.control(X, 0.0)
            V_hat = np.asarray(solver.value(X, 0.0)).ravel()
            snap = solver.snapshots[0]
            g_ml = extract_cell(solver.axes, snap, X, 1e-4)
            g_cu = extract_spline(solver.axes, snap, X, 1e-4)
            u_ml = g_ml @ prob.B @ prob.R_inv.T
            u_cu = g_cu @ prob.B @ prob.R_inv.T

            def rel(a, b, mask):
                return relative_l2(np.asarray(a)[mask].ravel(), np.asarray(b)[mask].ravel())

            def density(a, b, mask):
                """RMS error per region, NOT normalised inside the region.

                Normalising inside the region flatters the boundary band, because |V| is largest
                there: the first version of this script reported the boundary as the MORE accurate
                region for exactly that reason. What matters is the error density and the share of
                the total squared error the region contributes.
                """
                d_ = np.asarray(a)[mask].ravel() - np.asarray(b)[mask].ravel()
                return float(np.sqrt(np.mean(d_**2)))

            v_int, v_bnd = density(V_hat, V_ref, ~bnd), density(V_hat, V_ref, bnd)
            c_int, c_bnd = density(u_ml, u_ref, ~bnd), density(u_ml, u_ref, bnd)
            n_int, n_bnd = int((~bnd).sum()), int(bnd.sum())
            share = (n_bnd * c_bnd**2) / (n_bnd * c_bnd**2 + n_int * c_int**2)

            row = {
                "N": N, "n_steps": int(solver.n_time_steps),
                "boundary_share_measured": float(np.mean(bnd)),
                "boundary_share_geometric": float(1.0 - max(1.0 - frac_per_axis, 0.0) ** p),
                "value_rms_interior": v_int, "value_rms_boundary": v_bnd,
                "control_rms_interior": c_int, "control_rms_boundary": c_bnd,
                "control_error_density_ratio": c_bnd / c_int,
                "control_error_share_from_boundary": float(share),
                "value_rel_l2": relative_l2(V_hat, V_ref),
                "control_rel_l2": relative_l2(u_ml, u_ref),
                "cubic_gain_interior": rel(u_ml, u_ref, ~bnd) / rel(u_cu, u_ref, ~bnd),
                "cubic_gain_boundary": rel(u_ml, u_ref, bnd) / rel(u_cu, u_ref, bnd),
                "seconds": time.perf_counter() - t0,
            }
            rows.append(row)
            print(f"    {N:>4d} {row['boundary_share_measured']:>11.3f} "
                  f"{v_int:>12.3e} {v_bnd:>12.3e} {c_int:>13.3e} {c_bnd:>13.3e} "
                  f"{row['control_error_density_ratio']:>12.2f} "
                  f"{row['control_error_share_from_boundary']:>14.1%} "
                  f"{row['cubic_gain_interior']:>8.1f} {row['cubic_gain_boundary']:>8.1f}")
        out["dims"][str(p)] = rows

    print()
    print("=" * 100)
    print("READING")
    print("=" * 100)
    for p, rows in out["dims"].items():
        r = rows[-1]
        print(f"  d={p}: at N={r['N']} the boundary band is {r['boundary_share_measured']:.0%} of "
              f"the domain, its control-error density is "
              f"{r['control_error_density_ratio']:.2f}x the interior's, and it contributes "
              f"{r['control_error_share_from_boundary']:.0%} of the total squared error")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
