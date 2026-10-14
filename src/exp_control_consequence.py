"""
P0a: does the control error actually matter financially? Trace it along the executed path.

Why this experiment exists
--------------------------
Section V-B of the paper asserts that the control error matters even though the value error is tiny,
on the grounds that (i) the envelope theorem makes a control error delta-u perturb the optimal VALUE
only at second order, while (ii) the inventory path is a time integral of the control and therefore
inherits the control error at FIRST order. That is the paper's justification for reporting control
accuracy at all -- and until now it was an argument, not a measurement. A referee is entitled to ask
what the consequence actually is.

This script closes the loop and measures it. For each mesh resolution N:

  1. Solve the HJB on the grid.
  2. Integrate the EXECUTED path  dx = (A x + B u_hat(x,t)) dt  from a liquidation start, using a
     high-order integrator, where u_hat is the control the solver would actually execute:
        "fd"     multilinear reconstruction, central probe scaled to the mesh (the careful baseline)
        "spline" cubic reconstruction, same probe (the fix the paper recommends)
  3. Compare against the exact closed-loop path from u* = -R^-1 B' P(t) x.

Reported quantities, each chosen to be well defined rather than invented:
  path_dev        max_t ||x_hat - x*|| / ||x_0||   the executed position error. First order in the
                  control error, because position is the integral of the control.
  term_dev        ||x_hat(T) - x*(T)||              the terminal position error. For a liquidation
                  problem this is the quantity a desk actually checks: you were asked to be flat and
                  you are not. Note the true optimum does NOT end at zero (Q_T is finite), so the
                  deviation, not the level, is the right thing to look at.
  sign_disagree   fraction of the path where sign(u_hat) != sign(u*). A control that says BUY where
                  the optimum says SELL is a qualitative failure a trader recognises immediately,
                  and it needs no metric to be invented.
  obj_loss        (J* - J_hat) / |J*|              the objective actually lost, normalised. This is
                  what a value-accuracy benchmark would certify, and by the envelope theorem it is
                  SECOND order in the control error.

The point of putting path_dev and obj_loss side by side is the ratio between them: that ratio IS the
decoupling the paper claims, measured instead of asserted.

Scope, stated honestly: the integration is deterministic (the noise realisation is held at zero), so
this is an analysis of the closed-loop MAP from control to position, not of a realised P&L. It says
nothing about expected profitability, and no such claim is made.

Design note -- a degenerate comparison that had to be replaced
--------------------------------------------------------------
The first version compared "inventory-only with 2 assets" against "1 asset plus a signal", both at
state dimension 2, starting flat of signal. Every deviation column came out IDENTICAL across the two
cases while the unnormalised control-error column differed -- by a factor of 0.7041/0.4979 = 1.414,
which is exactly sqrt(2). That is the fingerprint of a degeneracy, not a coincidence: the
inventory-only problem at p = 2 is SEPARABLE, so it is two identical copies of p = 1, and the
deviation columns are normalised by ||x_0|| = sqrt(2) while the control-error column is not. Worse,
the signal case started on the s = 0 slice, where the alpha problem reproduces the no-signal problem
almost exactly, so the two "cases" were the same problem twice. Both are now avoided: the cases are
an inventory-only liquidation and a signalled position that is deliberately NOT a liquidation.

Usage: python src/exp_control_consequence.py [--quick]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_extraction import EXTRACTORS
from scipy.integrate import simpson
from scipy.interpolate import RegularGridInterpolator
from solvers import FiniteDifferenceSolver

OUT = Path("results")


class CachedControl:
    """The control the solver would execute, with the interpolants built once.

    WHY THIS CLASS EXISTS. `EXTRACTORS` constructs a RegularGridInterpolator on every call, which
    is fine for a one-shot accuracy measurement but not inside an ODE integration: the first
    version of this experiment rebuilt a cubic interpolant several thousand times and timed out.

    The interpolants are therefore cached, one per stored snapshot, and a query at time t blends the
    gradients from the two neighbouring snapshots. That is EXACTLY equal to blending the snapshots
    and then differentiating, because both the interpolant construction and the probe difference are
    linear in the nodal values -- so this is a speedup, not an approximation.
    """

    def __init__(self, solver, name: str, scale: float):
        self.solver = solver
        self.name = name
        self.scale = scale
        self.axes = solver.axes
        self.dx = np.array([a[1] - a[0] for a in self.axes])
        self.ts = solver.t_snap
        if name == "cell":
            self.interps = None
        else:
            method = "linear" if name == "fd" else "cubic"
            self.interps = [
                RegularGridInterpolator(self.axes, s, method=method, bounds_error=False,
                                        fill_value=None)
                for s in solver.snapshots
            ]

    def _grad_at_snapshot(self, k: int, X: np.ndarray) -> np.ndarray:
        if self.name == "cell":
            return EXTRACTORS["cell"](self.axes, self.solver.snapshots[k], X, None)
        rgi = self.interps[k]
        eps = self.scale * self.dx
        g = np.zeros_like(X)
        for i in range(X.shape[1]):
            Xp = X.copy(); Xp[:, i] += eps[i]
            Xm = X.copy(); Xm[:, i] -= eps[i]
            g[:, i] = (rgi(Xp) - rgi(Xm)) / (2 * eps[i])
        return g

    def __call__(self, x: np.ndarray, t: float) -> np.ndarray:
        j = int(np.clip(np.searchsorted(self.ts, t), 1, len(self.ts) - 1))
        t0, t1 = self.ts[j - 1], self.ts[j]
        w = 0.0 if t1 == t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        X = np.atleast_2d(x)
        g = (1.0 - w) * self._grad_at_snapshot(j - 1, X) + w * self._grad_at_snapshot(j, X)
        prob = self.solver.problem
        return g @ prob.B @ prob.R_inv.T


def integrate_path(prob: ACProblem, ctrl, x0: np.ndarray,
                   t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RK4 on dx = A x + B u(x,t), returning (states, controls) on t_grid."""
    xs = np.empty((len(t_grid), x0.size))
    us = np.empty((len(t_grid), prob.control_dim))
    x = x0.astype(float).copy()
    xs[0] = x
    us[0] = ctrl(x, t_grid[0])[0]
    for k in range(len(t_grid) - 1):
        t, dt = t_grid[k], t_grid[k + 1] - t_grid[k]

        def f(tt, xx):
            return prob.A @ xx + prob.B @ ctrl(xx, tt)[0]

        k1 = f(t, x)
        k2 = f(t + dt / 2, x + dt / 2 * k1)
        k3 = f(t + dt / 2, x + dt / 2 * k2)
        k4 = f(t + dt, x + dt * k3)
        x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        xs[k + 1] = x
        us[k + 1] = ctrl(x, t + dt)[0]
    return xs, us


def objective(prob: ACProblem, xs: np.ndarray, us: np.ndarray, t_grid: np.ndarray) -> float:
    """J = -1/2 int (x'Qx + u'Ru) dt - 1/2 x_T' Q_T x_T.

    Simpson rather than the trapezoid rule: the losses being compared here go down to ~1e-6 of
    |J|, so an O(dt^2) quadrature error is not small enough to resolve them. With Simpson the
    quadrature error is ~1e-11, well under everything reported.
    """
    run = np.einsum("ni,ij,nj->n", xs, prob.Q, xs) + \
        np.einsum("ni,ij,nj->n", us, prob.R, us)
    integ = float(simpson(run, x=t_grid))
    term = float(xs[-1] @ prob.QT @ xs[-1])
    return -0.5 * integ - 0.5 * term


def run_case(prob: ACProblem, grids: list[int], scale: float, n_steps: int, label: str,
             x0: np.ndarray, n_snap: int = 401) -> dict:
    # THE INTEGRATION STEP IS ALIGNED TO THE SNAPSHOT SPACING, and that alignment is the whole
    # reason this experiment is trustworthy. The controller is stored as snapshots and blended
    # linearly in time, so it is continuous but has a KINK at every snapshot time; RK4 that
    # straddles a kink loses order. Measured, for the signalled case at N=21 with a cubic
    # reconstruction, comparing the path at n_steps against 4*n_steps:
    #     n_snap=401,  n_steps=200  -> drift 1.2e-05   (dt = 2 dt_snap: steps straddle kinks)
    #     n_snap=401,  n_steps=400  -> drift 6.4e-11   (dt = dt_snap: steps land ON the nodes)
    #     n_snap=1201, n_steps=400  -> drift 1.6e-09
    #     n_snap=2401, n_steps=400  -> drift 1.4e-05   (WORSE: 6 snapshot intervals per step)
    # So more snapshots do not help and can hurt; landing on them does. With dt = dt_snap the
    # controller is exactly linear on each step and the integration is essentially exact, which is
    # what makes an objective loss of 1e-5 measurable at all. Earlier versions let n_steps and
    # n_snap drift out of alignment and reported NEGATIVE objective losses -- an extracted control
    # scoring better than the exact optimum, which is impossible -- and that is the tell that
    # exposed this.
    n_steps = n_snap - 1
    t_grid = np.linspace(0.0, prob.T, n_snap)
    # The reference control is read off an interpolated P(t), and np.interp is only piecewise
    # LINEAR -- so too coarse a grid here makes the REFERENCE path itself the accuracy floor.
    # With 401 points the reference path drifted by 4e-6, the same size as the smallest objective
    # losses being measured. 20001 points puts it at ~1e-9.
    ref = prob.reference(np.linspace(0.0, prob.T, 20001))
    x0 = np.asarray(x0, dtype=float)
    assert x0.size == prob.state_dim, f"x0 has {x0.size} entries, state dim is {prob.state_dim}"

    def ctrl_ref(x, t):
        return ref.control(np.atleast_2d(x), float(t))

    xs_ref, us_ref = integrate_path(prob, ctrl_ref, x0, t_grid)
    J_ref = objective(prob, xs_ref, us_ref, t_grid)
    # Self-check: the reference path must be integration-converged, otherwise the measured
    # deviations below would mix the solver's control error with the integrator's own error.
    t_fine = np.linspace(0.0, prob.T, 4 * n_steps + 1)
    xs_fine, _ = integrate_path(prob, ctrl_ref, x0, t_fine)
    ref_drift = float(np.max(np.linalg.norm(xs_fine[::4] - xs_ref, axis=1)))
    # Exact deterministic optimum: with no noise the cost-to-go is -1/2 x0' P(0) x0 (c(0) = 0).
    P0 = ref.P(0.0)
    J_star = -0.5 * x0 @ P0 @ x0

    print(f"\n  {label}   state_dim={prob.state_dim}  "
          f"J*={J_star:+.6e}  path-integrated J*={J_ref:+.6e}")
    print(f"    integration self-check: ref path moves {ref_drift:.2e} when n_steps is "
          f"quadrupled (must be far below every deviation below)")
    print(f"    {'N':>4} {'recon':>7} {'ctrl_dev':>11} {'path_dev':>11} {'term_dev':>11} "
          f"{'sign_dis':>9} {'obj_loss':>11} {'ratio':>9}")

    rows = []
    for n in grids:
        # n_snap is held FIXED across N, and n_steps = n_snap - 1 so the integrator lands exactly
        # on the stored snapshots (see the alignment note at the top of this function). Letting the
        # time resolution scale with N would mix a time-refinement effect into a study of the
        # spatial reconstruction.
        s = FiniteDifferenceSolver(prob, n_grid=n, n_snap=n_snap, snap_cap=4e9)
        s.solve()
        for name in ("fd", "spline"):
            ctrl = CachedControl(s, name, scale)
            xs, us = integrate_path(prob, ctrl, x0, t_grid)
            # Control error along the SAME executed path, so every column below is traceable to
            # it: without this the table cannot distinguish "the path error is small" from "the
            # control error is small", and identical rows across different problems are
            # uninterpretable.
            ctrl_dev = float(np.max(np.linalg.norm(us - us_ref, axis=1)))
            ctrl_scale = float(np.max(np.linalg.norm(us_ref, axis=1)))
            path_dev = float(np.max(np.linalg.norm(xs - xs_ref, axis=1)) /
                             np.linalg.norm(x0))
            term_dev = float(np.linalg.norm(xs[-1] - xs_ref[-1]) / np.linalg.norm(x0))
            sg = np.sign(us) != np.sign(us_ref)
            sign_dis = float(np.mean(sg))
            J = objective(prob, xs, us, t_grid)
            loss = (J_star - J) / abs(J_star)
            ratio = (path_dev / loss) if loss > 0 else float("inf")
            rows.append({"N": n, "recon": name, "ctrl_dev": ctrl_dev,
                         "ctrl_dev_rel": ctrl_dev / ctrl_scale,
                         "path_dev": path_dev, "term_dev": term_dev,
                         "sign_disagree": sign_dis, "obj_loss": loss, "ratio": ratio,
                         "J": J, "J_star": J_star})
            print(f"    {n:>4} {name:>7} {ctrl_dev:>11.3e} {path_dev:>11.3e} {term_dev:>11.3e} "
                  f"{sign_dis:>9.3f} {loss:>11.3e} {ratio:>9.1f}")

    # Orders: path deviation first order in the reconstruction error, objective loss second order.
    for name in ("fd", "spline"):
        sub = [r for r in rows if r["recon"] == name]
        if len(sub) >= 3:
            h = np.array([1.0 / r["N"] for r in sub])
            for key in ("path_dev", "obj_loss"):
                e = np.array([r[key] for r in sub])
                ok = e > 0
                if ok.sum() >= 2:
                    sl = float(np.polyfit(np.log(h[ok]), np.log(e[ok]), 1)[0])
                    print(f"    {name:>7} {key:<10} order = {sl:5.2f}   "
                          f"({ok.sum()}/{len(e)} points usable; non-positive values dropped)")

    # Self-check bundling BOTH error sources of the executed path, at the finest mesh. This
    # compares two resolutions of the STORED POLICY (401 against 802 snapshots), each integrated
    # with steps aligned to its own snapshots, so it bounds the TIME-discretisation effect -- which
    # the reference-only check above cannot see. The deviations reported in the table are 1e-3 to
    # 1e-2, so this must be well under them for the table to be about the spatial reconstruction.
    print(f"    executed-path self-check at N={grids[-1]}: 401 vs 802 stored snapshots")
    for name in ("fd", "spline"):
        sa = FiniteDifferenceSolver(prob, n_grid=grids[-1], n_snap=n_snap, snap_cap=4e9)
        sa.solve()
        sb = FiniteDifferenceSolver(prob, n_grid=grids[-1], n_snap=2 * n_snap, snap_cap=4e9)
        sb.solve()
        xa, _ = integrate_path(prob, CachedControl(sa, name, scale), x0,
                               np.linspace(0.0, prob.T, n_snap))
        xb, _ = integrate_path(prob, CachedControl(sb, name, scale), x0,
                               np.linspace(0.0, prob.T, 2 * n_snap))
        drift = float(np.max(np.linalg.norm(xb[::2] - xa, axis=1)))
        print(f"      {name:>7}  path drift = {drift:.2e}")

    return {"case": label, "state_dim": prob.state_dim, "J_star": J_star,
            "J_star_integrated": J_ref, "ref_path_drift": ref_drift, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    grids = [11, 15, 21] if args.quick else [11, 15, 21, 25, 31]
    # RK4 on a smooth state feedback, so a few hundred steps is ample; the self-check below
    # confirms it rather than assuming it. (2000 steps was tried first and made the run take
    # minutes, because the cost is per-call interpolation overhead on one-point arrays.)
    n_steps = 200 if args.quick else 400

    print("=" * 100)
    print("DOES THE CONTROL ERROR MATTER FINANCIALLY? executed path vs the exact closed loop")
    print("=" * 100)
    print("  path_dev  = max position error / ||x0||          (first order in control error)")
    print("  term_dev  = terminal position error / ||x0||     (was the book actually flat?)")
    print("  sign_dis  = fraction of path where the control's SIGN differs from optimal")
    print("  obj_loss  = (J* - J)/|J*|, the objective actually lost (second order, envelope theorem)")
    print("  ratio     = path_dev / obj_loss: how much more visible the path error is")

    report = {"scale": 0.25, "n_steps": n_steps, "cases": []}
    # Case 1: a pure liquidation, no signal. The clean baseline.
    report["cases"].append(run_case(ACProblem(p=1), grids, 0.25, n_steps,
                                    "liquidation, no signal", [1.0]))
    # Case 2: the same inventory but a POSITIVE signal, so the optimum is to HOLD rather than
    # liquidate. Deliberately not a liquidation: this is the case where a control error changes
    # what the desk is doing, and it is genuinely a different trajectory rather than a rescaled
    # copy of case 1.
    report["cases"].append(run_case(ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005),
                                    grids, 0.25, n_steps,
                                    "inventory 1 with signal +1 (holds, not liquidates)",
                                    [1.0, 1.0]))

    (OUT / "exp_control_consequence.json").write_text(json.dumps(report, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote {OUT / 'exp_control_consequence.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
