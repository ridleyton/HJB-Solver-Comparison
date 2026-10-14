"""
Does grid control accuracy really converge more slowly than grid value accuracy?

This is the linchpin check for the proposed paper's central claim, so it is deliberately set up
to *falsify* that claim as well as confirm it.

The claim to beat: on the same grid, the control error is far larger than the value error and
converges at a lower order, because the control is a derivative and has to be extracted from an
interpolant. If true, the resolution an HJB application needs is set by the control target, not
the value target, and the grid's N^p cost is driven by a requirement that value-accuracy
benchmarks never state.

The risk this script exists to check: our grid control is a central difference with a FIXED
step, `eps = 1e-4`, taken on a multilinear interpolant (`solvers.py`, `_GridSolver.control`).
A fixed probe is a crude way to differentiate an interpolant, and if the measured error is
limited by that choice then the "decoupling" is a property of our implementation, not of grid
methods -- and not publishable as a general claim. Three parts:

  A. Refinement on the NON-SEPARABLE problem (hetero=True). The earlier order measurement used
     the separable problem; this repeats it where the value function is a genuinely
     p-dimensional quadratic with a full P matrix.
  B. Probe sensitivity: at fixed N, vary eps over four decades plus rules tied to h. If the
     control error barely moves, the fixed probe is not the limiter. If it moves a lot, the
     headline claim needs restating.
  C. Orders recomputed with the best probe found in B, so the reported order is the method's,
     not the probe's.

Reference is the closed-form Riccati solution throughout, so both errors are measured against
truth rather than against another solver.

Output: results/exp_control_order.json
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from solvers import FiniteDifferenceSolver


class ProbedFDM(FiniteDifferenceSolver):
    """FDM whose control probe step can be set, including as a multiple of the grid spacing."""

    name = "Finite Difference (probed)"

    def __init__(self, problem, n_grid=21, snap_eps=1e-4, eps_mode="fixed", eps_scale=1.0,
                 **kw):
        super().__init__(problem, n_grid=n_grid, **kw)
        self.snap_eps = snap_eps
        self.eps_mode = eps_mode
        self.eps_scale = eps_scale

    def _eps(self) -> float:
        if self.eps_mode == "fixed":
            return self.snap_eps
        # "h": probe scaled to the grid spacing, which is what a careful implementation would do.
        return self.eps_scale * float(min(self.dx))

    def control(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.atleast_2d(X)
        eps = self._eps()
        g = np.zeros_like(X)
        # state_dim, not p: with the alpha signal the state is 2p while the control stays p.
        for i in range(self.problem.state_dim):
            Xp = X.copy(); Xp[:, i] += eps
            Xm = X.copy(); Xm[:, i] -= eps
            g[:, i] = (self.value(Xp, t) - self.value(Xm, t)) / (2 * eps)
        return g @ self.problem.B @ self.problem.R_inv.T


def evaluate(solver, prob, n_points=800, n_times=6, seed=7):
    """Return MSE (not R^2) for value and control, against the closed-form reference.

    MSE is what convergence orders should be fitted to; R^2 is for reporting. Mixing them is how
    a "higher order" appears out of a saturating error.
    """
    rng = np.random.default_rng(seed)
    lo, hi = prob.domain()
    ref = prob.reference(np.linspace(0.0, prob.T, 64))
    X = rng.uniform(lo, hi, size=(n_points, prob.state_dim))
    ts = rng.uniform(0.0, prob.T, size=n_times)

    Xs, ts_all = [], []
    for t in ts:
        Xs.append(X); ts_all.append(np.full(n_points, t))
    Xs = np.vstack(Xs); ts_all = np.concatenate(ts_all)

    V_hat = solver.value(Xs, 0.0) if False else None
    # value/control are evaluated time-by-time so each solver call is homogeneous in t.
    V_hat = np.concatenate([solver.value(X, float(t)) for X, t in zip(
        [X] * n_times, ts)])
    U_hat = np.concatenate([solver.control(X, float(t)) for X, t in zip(
        [X] * n_times, ts)])

    V_ref = np.concatenate([ref.value(X, float(t)) for t in ts])
    U_ref = np.concatenate([ref.control(X, float(t)) for t in ts])

    v_mse = float(np.mean((V_hat - V_ref) ** 2))
    u_mse = float(np.mean((U_hat - U_ref) ** 2))
    v_scale = float(np.mean((V_ref - V_ref.mean()) ** 2)) or 1.0
    u_scale = float(np.mean((U_ref - U_ref.mean()) ** 2)) or 1.0
    return {
        "value_mse": v_mse, "control_mse": u_mse,
        "value_r2": 1.0 - v_mse / v_scale, "control_r2": 1.0 - u_mse / u_scale,
        # 1 - R^2, i.e. error normalised by the target's spread; the ratio is the cleanest
        # statement of "how much worse is the control than the value on the same grid".
        "value_err": v_mse / v_scale, "control_err": u_mse / u_scale,
    }


def order(e_prev: float, e_now: float, h_prev: float, h_now: float) -> float:
    if e_prev <= 0 or e_now <= 0 or h_prev <= 0 or h_now <= 0:
        return float("nan")
    return math.log(e_prev / e_now) / math.log(h_prev / h_now)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,2,3")
    ap.add_argument("--grids", default="11,15,21,27,35,45,61")
    ap.add_argument("--out", default="results/exp_control_order.json")
    args = ap.parse_args()

    dims = [int(d) for d in args.dims.split(",")]
    grids = [int(g) for g in args.grids.split(",")]
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem": ("multi-asset optimal execution, LQ form, hetero=True -- per-asset "
                        "heterogeneous risk/impact and random off-diagonal risk, so V is a "
                        "genuinely p-dimensional quadratic with a full P matrix"),
            "reference": "closed-form matrix Riccati solution",
            "grids": grids,
            "dims": dims,
            "purpose": ("verify that the control error decouples from the value error on the "
                        "NON-SEPARABLE problem, and test whether the result is an artefact of "
                        "the fixed eps=1e-4 central-difference probe rather than a property of "
                        "grid methods"),
        },
        "part_a_refinement": [],
        "part_b_probe_sensitivity": [],
        "part_c_orders_at_best_probe": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---------------- Part A: refinement, fixed probe (current implementation) -----------
    print("=" * 100)
    print("Part A -- refinement on the NON-SEPARABLE problem, current probe (eps=1e-4 fixed)")
    print("=" * 100)
    print(f"  {'p':>2s} {'N':>4s} {'h':>9s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>8s} {'ord_v':>7s} {'ord_c':>7s}")
    for p in dims:
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        prev = None
        for N in grids:
            try:
                s = ProbedFDM(prob, n_grid=N, eps_mode="fixed", snap_eps=1e-4)
                s.solve()
                m = evaluate(s, prob)
                h = float(np.mean(s.dx))
                ov = oc = float("nan")
                if prev:
                    ov = order(prev["value_err"], m["value_err"], prev["h"], h)
                    oc = order(prev["control_err"], m["control_err"], prev["h"], h)
                row = {"p": p, "N": N, "h": h, "probe": "fixed 1e-4", **m,
                       "order_value": ov, "order_control": oc,
                       "ratio": m["control_err"] / m["value_err"],
                       "solve_time_s": s.solve_time}
                results["part_a_refinement"].append(row)
                print(f"  {p:>2d} {N:>4d} {h:>9.5f} {m['value_err']:>12.3e} "
                      f"{m['control_err']:>12.3e} {row['ratio']:>8.1f} "
                      f"{ov:>7.2f} {oc:>7.2f}", flush=True)
                m["h"] = h
                prev = m
            except Exception as e:
                print(f"  p={p} N={N} FAILED: {type(e).__name__}: {e}")
            save()
        print()

    # ---------------- Part B: is the fixed probe the limiter? ---------------------------
    print("=" * 100)
    print("Part B -- probe sensitivity at fixed N (does the probe choice drive the error?)")
    print("=" * 100)
    probe_cases = [
        ("fixed", 1e-6, 1.0), ("fixed", 1e-5, 1.0), ("fixed", 1e-4, 1.0), ("fixed", 1e-3, 1.0),
        ("h", None, 0.125), ("h", None, 0.25), ("h", None, 0.5), ("h", None, 1.0),
    ]
    for p in dims:
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        for N in (21, 45):
            print(f"  p={p} N={N}")
            for mode, eps, scale in probe_cases:
                try:
                    s = ProbedFDM(prob, n_grid=N, eps_mode=mode, snap_eps=eps or 1e-4,
                                  eps_scale=scale)
                    s.solve()
                    m = evaluate(s, prob)
                    used = s._eps()
                    label = f"fixed {eps:g}" if mode == "fixed" else f"h x {scale:g}"
                    results["part_b_probe_sensitivity"].append(
                        {"p": p, "N": N, "mode": mode, "eps_used": used, "label": label, **m})
                    print(f"      {label:>12s} (eps={used:.3e})  value_err={m['value_err']:.4e}"
                          f"  control_err={m['control_err']:.4e}", flush=True)
                except Exception as e:
                    print(f"      {mode} {eps} FAILED: {e}")
            save()
        print()

    # ---------------- Part C: orders with the probe scaled to h -------------------------
    print("=" * 100)
    print("Part C -- orders with probe scaled to the grid spacing (h x 0.25)")
    print("=" * 100)
    print(f"  {'p':>2s} {'N':>4s} {'h':>9s} {'value err':>12s} {'control err':>12s} "
          f"{'ratio':>8s} {'ord_v':>7s} {'ord_c':>7s}")
    for p in dims:
        prob = ACProblem(p=p, hetero=True, het_seed=1)
        prev = None
        for N in grids:
            try:
                s = ProbedFDM(prob, n_grid=N, eps_mode="h", eps_scale=0.25)
                s.solve()
                m = evaluate(s, prob)
                h = float(np.mean(s.dx))
                ov = oc = float("nan")
                if prev:
                    ov = order(prev["value_err"], m["value_err"], prev["h"], h)
                    oc = order(prev["control_err"], m["control_err"], prev["h"], h)
                results["part_c_orders_at_best_probe"].append(
                    {"p": p, "N": N, "h": h, "probe": "h x 0.25", **m,
                     "order_value": ov, "order_control": oc,
                     "ratio": m["control_err"] / m["value_err"]})
                print(f"  {p:>2d} {N:>4d} {h:>9.5f} {m['value_err']:>12.3e} "
                      f"{m['control_err']:>12.3e} {m['control_err']/m['value_err']:>8.1f} "
                      f"{ov:>7.2f} {oc:>7.2f}", flush=True)
                m["h"] = h
                prev = m
            except Exception as e:
                print(f"  p={p} N={N} FAILED: {type(e).__name__}: {e}")
            save()
        print()

    # ---------------- verdict ---------------------------------------------------------
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    for part, key in (("A (fixed probe)", "part_a_refinement"),
                      ("C (probe = h/4)", "part_c_orders_at_best_probe")):
        rows = results[key]
        oc = [r["order_control"] for r in rows if r["order_control"] == r["order_control"]]
        ov = [r["order_value"] for r in rows if r["order_value"] == r["order_value"]]
        rt = [r["ratio"] for r in rows if r["ratio"] == r["ratio"]]
        if oc:
            print(f"  Part {part}:")
            print(f"    control error order : median {np.median(oc):.2f}  "
                  f"(range {min(oc):.2f} to {max(oc):.2f})")
            print(f"    value   error order : median {np.median(ov):.2f}  "
                  f"(range {min(ov):.2f} to {max(ov):.2f})")
            print(f"    control/value error ratio: median {np.median(rt):.1f}  "
                  f"(range {min(rt):.1f} to {max(rt):.1f})")
    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
