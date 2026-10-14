"""
Diagnostic: does the control-order result survive the alpha extension?

Background -- why this diagnostic exists
----------------------------------------
A first attempt measured the control order on the alpha benchmark using `accuracy_metrics` and got
about 1.06. That number is NOT a finding: `accuracy_metrics` reports raw MSE and drives the solver
through `_GridSolver.control`, whose probe step is a FIXED eps = 1e-4. The paper's own Table II
came from the harness in `exp_control_order.py`, which normalises by the target's spread and, for
the quoted orders, scales the probe to the mesh. The two configurations differ by ~87x in error and
give different slopes (order ~1 for the fixed probe, ~2 for the mesh-scaled one), so comparing them
is meaningless -- the lesson is to reuse the validated harness rather than invent a metric.

This script therefore uses `exp_control_order.evaluate` and `ProbedFDM` unchanged, and asks the one
question that matters: on the ALPHA benchmark, is the control error still (a) larger than the value
error and (b) converging at a stable lower order than the value error?

Usage: python src/diag_alpha_order.py
"""

from __future__ import annotations

import numpy as np

from ac_problem import ACProblem
from exp_control_order import ProbedFDM, evaluate, order


def sweep(prob: ACProblem, grids: list[int], eps_mode: str, eps_scale: float, label: str):
    print(f"  {label}  (probe: {eps_mode}"
          + (f" x {eps_scale}" if eps_mode != "fixed" else " 1e-4") + ")")
    print(f"    {'N':>5}{'h':>10}{'value_err':>13}{'ctrl_err':>13}{'ratio':>8}"
          f"{'ord(val)':>10}{'ord(ctrl)':>11}")
    prev = None
    v_orders, c_orders = [], []
    for n in grids:
        s = ProbedFDM(prob, n_grid=n, n_snap=101, snap_cap=4e9,
                      eps_mode=eps_mode, eps_scale=eps_scale)
        s.solve()
        m = evaluate(s, prob, n_points=800, n_times=6, seed=7)
        # evaluate() returns normalised errors but no mesh size, so supply h = 1/N here.
        h = 1.0 / n
        ov = oc = float("nan")
        if prev is not None:
            ov = order(prev["value_err"], m["value_err"], prev["h"], h)
            oc = order(prev["control_err"], m["control_err"], prev["h"], h)
            v_orders.append(ov); c_orders.append(oc)
        print(f"    {n:>5}{h:>10.4f}{m['value_err']:>13.3e}{m['control_err']:>13.3e}"
              f"{m['control_err'] / m['value_err']:>8.1f}{ov:>10.2f}{oc:>11.2f}")
        # Re-fit the normalisation so the spread does not drift with N: errors in the table are
        # already divided by the target spread, which is what makes them comparable across N.
        prev = {"value_err": m["value_err"], "control_err": m["control_err"], "h": h}
    v = float(np.mean(v_orders)) if v_orders else float("nan")
    c = float(np.mean(c_orders)) if c_orders else float("nan")
    print(f"    mean observed order: value {v:.2f}   control {c:.2f}")
    return {"value_order": v, "control_order": c}


def main() -> int:
    grids = [11, 15, 21, 25, 31, 35]
    print("=" * 84)
    print("A. inventory-only at state dim 2, as a control for the harness change")
    print("=" * 84)
    sweep(ACProblem(p=2), grids, "h", 0.25, "inventory-only, mesh-scaled probe")

    print()
    print("=" * 84)
    print("B. alpha benchmark, state dim 2, same harness")
    print("=" * 84)
    alpha = ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005)
    sweep(alpha, grids, "h", 0.25, "alpha, mesh-scaled probe")
    print()
    sweep(alpha, grids, "fixed", 1.0, "alpha, fixed 1e-4 probe (the paper's baseline config)")

    print()
    print("=" * 84)
    print("C. does the alpha cross term survive at higher state dimension?")
    print("=" * 84)
    sweep(ACProblem(p=2, alpha_kappa=1.0, alpha_sig=0.005), [11, 13, 15, 17], "h", 0.25,
          "alpha, 2 assets, state dim 4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
