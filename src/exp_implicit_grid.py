"""
P0b: is "the grid is not runnable beyond p ~ 5" a fact about GRIDS, or about EXPLICIT stepping?

Why this test exists
--------------------
The paper's cost claim rests on the explicit time step being CFL-limited: work grows as N^(d+1)
rather than N^d, and that extra factor is the whole reason the grid stops being runnable at about
five state dimensions. But that restriction comes from EXPLICIT time stepping. In the
inventory-only benchmark the diffusion is sigma^2 = 1e-4 and the restriction is mild; the moment a
signal is carried its axes diffuse at 2*kappa = O(1), four orders of magnitude more, and THAT single
term sets the step and hence the cost at every dimension.

So the claim to test is narrow and falsifiable: with the diffusion treated implicitly, does the grid
still need a CFL-sized step? `solvers._GridSolver` now offers `time_scheme="lod"`, which is
implicit diffusion via locally-one-dimensional splitting with the Hamiltonian left explicit.

What is reported
----------------
For each problem and mesh, the number of explicit CFL steps, the number of LOD steps actually used
(forced), the speedup, and the value/control R^2 against the exact reference in both cases. The
reference is exact, so "LOD is accurate" is checked rather than argued.

The honest outcome may be that LOD at a large step is NOT accurate -- first-order splitting error
does not vanish on its own -- in which case the correct conclusion is that the CFL restriction is
removable but only at a step chosen for ACCURACY rather than stability, and the real gain is the
difference between those two limits. That is a legitimate and useful result either way, and this
script reports both.

Usage: python src/exp_implicit_grid.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_order import evaluate
from solvers import FiniteDifferenceSolver

OUT = Path("results")


def cfl_steps(prob: ACProblem, n: int) -> int:
    """Step count the EXPLICIT scheme is forced to take, from the solver's own bound."""
    s = FiniteDifferenceSolver(prob, n_grid=n, n_snap=5)
    s._build_grid()
    u_T = (-prob.QT @ s.X.T).T @ prob.B @ prob.R_inv.T
    max_u = float(np.max(np.abs(u_T))) * 1.05
    dt = s._stable_dt(max_u)
    return max(int(np.ceil(prob.T / dt)), 2)


def one(prob: ACProblem, n: int, scheme: str, n_steps: int | None, label: str) -> dict:
    t0 = time.perf_counter()
    s = FiniteDifferenceSolver(prob, n_grid=n, n_snap=101, snap_cap=4e9,
                               time_scheme=scheme, n_steps=n_steps)
    s.solve()
    wall = time.perf_counter() - t0
    m = evaluate(s, prob, n_points=800, n_times=6, seed=7)
    row = {"case": label, "N": n, "scheme": scheme, "n_steps": s.n_time_steps,
           "seconds": wall, "value_err": m["value_err"], "control_err": m["control_err"],
           "value_r2": m["value_r2"], "control_r2": m["control_r2"],
           "stable": bool(np.all(np.isfinite(s.snapshots)))}
    print(f"    {scheme:<9} steps={s.n_time_steps:>7}  {wall:>6.1f}s  "
          f"value R2={m['value_r2']:+.7f}  ctrl R2={m['control_r2']:+.7f}  "
          f"{'' if row['stable'] else 'NON-FINITE'}")
    return row


def main() -> int:
    OUT.mkdir(exist_ok=True)
    report: dict = {"rows": []}
    print("=" * 100)
    print("IS THE CFL RESTRICTION A PROPERTY OF GRIDS, OR OF EXPLICIT STEPPING?")
    print("=" * 100)
    print("  LOD = implicit diffusion (locally one-dimensional splitting), Hamiltonian explicit.")

    for label, prob, grids in (
            ("inventory-only, d=2", ACProblem(p=2), [21, 31]),
            ("signal, d=2", ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005), [21, 31]),
            ("signal, d=4", ACProblem(p=2, alpha_kappa=1.0, alpha_sig=0.005), [15, 21]),
    ):
        print(f"\n  {label}")
        for n in grids:
            n_expl = cfl_steps(prob, n)
            print(f"    N={n}: the explicit scheme is forced to take {n_expl} steps "
                  f"(dt = {prob.T / n_expl:.3e})")
            report["rows"].append(one(prob, n, "explicit", None, label))
            for mult in (1, 4, 16, 64):
                ns = max(int(n_expl / mult), 4)
                r = one(prob, n, "lod", ns, label)
                r["speedup_vs_explicit"] = n_expl / r["n_steps"]
                report["rows"].append(r)

    print()
    print("  READING: the explicit column shows the step the CFL bound forces. The LOD rows use")
    print("  steps divided by 1, 4, 16 and 64 -- i.e. 64x beyond the explicit limit at the last")
    print("  one. Whatever R^2 the 64x row retains is accuracy bought WITHOUT any CFL restriction.")

    (OUT / "exp_implicit_grid.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'exp_implicit_grid.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
