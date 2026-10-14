"""
Does the reconstruction-order result survive a value function that is NOT quadratic?

The paper's benchmark is linear-quadratic, so its value function is a quadratic form, and a cubic
reconstruction reproduces quadratics exactly. That makes the exact-nodal-value result a best case
and leaves the general claim resting on synthetic test functions rather than on a value function
that actually solves an HJB. A referee is entitled to ask whether the conclusion is an artefact of
the LQ structure.

This closes that gap with the Merton portfolio problem, which has a closed-form solution that is a
POWER function:

    V(x, t) = (x^gamma / gamma) * exp(kappa (T - t)),
    kappa   = gamma * ( r + (mu - r)^2 / (2 sigma^2 (1 - gamma)) ),
    pi*(x)  = (mu - r) / (sigma^2 (1 - gamma)),

maximising E[U(X_T)] with U(x) = x^gamma / gamma over a risky asset with drift mu, volatility
sigma and riskless rate r. Two properties make it the right test:

  * V is non-polynomial, so a cubic reconstruction is NOT exact and the measured order is the
    genuine asymptotic order rather than a reproduction artefact;
  * the HJB is non-linear in the derivatives --- the maximised term is
    -(mu-r)^2 V_x^2 / (2 sigma^2 V_xx), a ratio of a squared first derivative to a second
    derivative --- so the numerical solution is not a Riccati integration.

Two parts:

  Part A -- analytic value function placed on the grid. Isolates the extraction operator on a
            non-polynomial V exactly as `exp_control_extraction` did for the quadratic one.
  Part B -- the Merton HJB solved numerically by explicit backward time stepping, validated
            against the closed form, then the same extraction comparison on the solved grid.

The quantity extracted is the gradient V_x, which is what any HJB control is built from (here
pi* is proportional to V_x / (x V_xx)). It is non-constant in x, so the test is not degenerate.

Output: results/exp_merton_extraction.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from exp_control_extraction import EXTRACTORS


# ---------------------------------------------------------------------------------------
# The Merton problem and its closed-form solution
# ---------------------------------------------------------------------------------------
class Merton:
    def __init__(self, gamma: float = 0.5, r: float = 0.03, mu: float = 0.08,
                 sigma: float = 0.2, T: float = 1.0,
                 x_lo: float = 0.5, x_hi: float = 2.5):
        assert 0.0 < gamma < 1.0, "need concavity for this scheme"
        self.gamma = gamma
        self.r = r
        self.mu = mu
        self.sigma = sigma
        self.T = T
        self.x_lo = x_lo
        self.x_hi = x_hi
        self.kappa = gamma * (r + (mu - r) ** 2 / (2 * sigma**2 * (1 - gamma)))
        self.pi_star = (mu - r) / (sigma**2 * (1 - gamma))

    # closed-form solution and its derivatives
    def V(self, x, t):
        return (np.asarray(x) ** self.gamma / self.gamma) * np.exp(self.kappa * (self.T - t))

    def Vx(self, x, t):
        return (np.asarray(x) ** (self.gamma - 1.0)) * np.exp(self.kappa * (self.T - t))

    def Vxx(self, x, t):
        return ((self.gamma - 1.0) * np.asarray(x) ** (self.gamma - 2.0)
                * np.exp(self.kappa * (self.T - t)))

    def control(self, x, t):
        """Optimal risky-asset fraction, constant here but recovered from two derivatives."""
        return np.full_like(np.asarray(x, dtype=float),
                            -(self.mu - self.r) * self.Vx(x, t) / (self.sigma**2 * x * self.Vxx(x, t)))


def solve_merton(prob: Merton, n: int, n_steps: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Explicit backward-time finite differences for the Merton HJB.

    -dV/dt = r x V_x + sup_pi [ pi (mu-r) x V_x + 0.5 pi^2 sigma^2 x^2 V_xx ]
    with the sup in closed form: -(mu-r)^2 V_x^2 / (2 sigma^2 V_xx), positive because V_xx < 0.
    Mirrors the update convention of the LQ solver: V <- V + dt * (generator terms).

    Boundary handling matters more here than in the LQ case. The maximised term is a RATIO of a
    squared first derivative to a second derivative, so a relative error in a boundary one-sided
    difference is amplified rather than damped: with `edge_order=2` derivatives the residual of
    the exact solution against the discrete generator was 1.7e-2 at the low-x edge and 7.6e-3 at
    the high-x edge, against 5e-4 in the interior, and that edge error propagated inward and held
    the solution error at ~3e-3 however fine the grid became. Imposing the exact closed form as
    Dirichlet data at BOTH ends removes the edge residual, which is legitimate here because this
    is a benchmark with a known solution and the question is about the extraction step rather than
    about boundary treatment.
    """
    x = np.linspace(prob.x_lo, prob.x_hi, n)
    dx = x[1] - x[0]
    V = prob.V(x, prob.T)
    dt = prob.T / n_steps
    for k in range(n_steps):
        Vx = np.gradient(V, dx, edge_order=2)
        Vxx = np.gradient(Vx, dx, edge_order=2)
        # Concavity is guaranteed by gamma < 1; clip only to keep the ratio finite at roundoff.
        denom = np.where(Vxx < -1e-12, Vxx, -1e-12)
        sup_term = -(prob.mu - prob.r) ** 2 * Vx**2 / (2 * prob.sigma**2 * denom)
        V = V + dt * (prob.r * x * Vx + sup_term)
        t_new = prob.T - (k + 1) * dt
        V[0] = prob.V(prob.x_lo, t_new)
        V[-1] = prob.V(prob.x_hi, t_new)
    return x, V, dt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grids", default="11,15,21,27,35,45,61")
    ap.add_argument("--methods", default="fd,cell,spline")
    ap.add_argument("--probe-scale", type=float, default=0.25)
    ap.add_argument("--t-ref", type=float, default=0.5)
    ap.add_argument("--n-points", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--solve-grids", default="21,41,81,161")
    ap.add_argument("--dt-factor", type=float, default=4.0,
                    help="n_steps = dt_factor * N^2; stability needs n_steps ~ N^2")
    ap.add_argument("--out", default="results/exp_merton_extraction.json")
    args = ap.parse_args()

    prob = Merton()
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    rng = np.random.default_rng(args.seed)
    # Query points strictly inside the domain, away from the boundary.
    X = rng.uniform(prob.x_lo + 0.05, prob.x_hi - 0.05, size=(args.n_points, 1))
    t = args.t_ref
    g_ref = prob.Vx(X[:, 0], t)
    g_scale = float(np.mean((g_ref - g_ref.mean()) ** 2)) or 1.0

    print(f"Merton: gamma={prob.gamma}, r={prob.r}, mu={prob.mu}, sigma={prob.sigma}, "
          f"T={prob.T}")
    print(f"  kappa={prob.kappa:.5f}  pi*={prob.pi_star:.4f}  "
          f"x in [{prob.x_lo}, {prob.x_hi}]")
    print(f"  V is a power function x^gamma, so a cubic reconstruction is NOT exact here.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "problem": "Merton portfolio problem, CRRA utility, closed-form power solution",
            "gamma": prob.gamma, "r": prob.r, "mu": prob.mu, "sigma": prob.sigma,
            "T": prob.T, "kappa": prob.kappa, "pi_star": prob.pi_star,
            "x_range": [prob.x_lo, prob.x_hi],
            "quantity": "V_x, the gradient any HJB control is built from",
            "purpose": ("test whether the reconstruction-order result survives a value function "
                        "that is NOT quadratic, since the LQ benchmark's V is a quadratic form "
                        "and a cubic reconstruction reproduces quadratics exactly"),
        },
        "exact_nodes": [],
        "solved_grid": [],
        "solver_validation": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---------------- Part A: exact nodal values of a non-polynomial V -------------------
    print("\n" + "=" * 96)
    print("Part A -- extraction from EXACT nodal values of the power-function V")
    print("=" * 96)
    print(f"  {'N':>4s} {'h':>9s} " + " ".join(f"{m:>14s}" for m in methods))
    tracks = {m: [] for m in methods}
    for N in [int(g) for g in args.grids.split(",")]:
        axes = [np.linspace(prob.x_lo, prob.x_hi, N)]
        h = float(axes[0][1] - axes[0][0])
        snap = prob.V(axes[0], t)
        row = {"N": N, "h": h, "errors": {}}
        line = f"  {N:>4d} {h:>9.5f} "
        for m in methods:
            try:
                g = EXTRACTORS[m](axes, snap, X, args.probe_scale)
                err = float(np.mean((g[:, 0] - g_ref) ** 2) / g_scale)
                row["errors"][m] = err
                tracks[m].append((h, err))
                line += f"{err:>14.3e} "
            except Exception as e:
                row["errors"][m] = None
                line += f"{'FAIL ' + type(e).__name__:>14s} "
        print(line, flush=True)
        results["exact_nodes"].append(row)
        save()

    print("\n  convergence orders (least squares in log-log):")
    for m in methods:
        pts = [(h, e) for h, e in tracks[m] if e and e > 1e-16]
        if len(pts) < 3:
            print(f"    {m:>8s}: too few usable points")
            continue
        hh = np.log([h for h, _ in pts]); ee = np.log([e for _, e in pts])
        A = np.vstack([hh, np.ones_like(hh)]).T
        slope, inter = np.linalg.lstsq(A, ee, rcond=None)[0]
        pred = A @ np.array([slope, inter])
        r2 = 1 - ((ee - pred) ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
        print(f"    {m:>8s}: order {slope:5.2f}  (fit R^2 {r2:.3f})   "
              f"{pts[0][1]:.2e} -> {pts[-1][1]:.2e}  (factor {pts[0][1]/pts[-1][1]:.0f})")
        results["meta"].setdefault("exact_node_orders", {})[m] = {
            "order": float(slope), "fit_r2": float(r2)}

    # ---------------- Part B: solve the Merton HJB, then extract ------------------------
    print("\n" + "=" * 96)
    print("Part B -- Merton HJB solved numerically, validated against the closed form")
    print("=" * 96)
    # The solved array is V at t=0 (the backward sweep runs the full horizon), so the gradient
    # must be compared at t=0 and NOT at the query time used in Part A. Getting this wrong
    # produced an error of 1.25e-2 that was identical at every N for both extractors -- the same
    # tell as in exp_control_extraction, where an interpolation error cannot be both large and
    # mesh-independent.
    g_ref_solved = prob.Vx(X[:, 0], 0.0)
    g_scale_solved = float(np.mean((g_ref_solved - g_ref_solved.mean()) ** 2)) or 1.0
    print(f"  {'N':>5s} {'n_steps':>8s} {'dt':>10s} {'rel L2(V) vs closed form':>24s} "
          f"{'fd':>12s} {'spline':>12s}")
    ptracks = {m: [] for m in methods}
    for N in [int(g) for g in args.solve_grids.split(",")]:
        # The explicit scheme's stability limit scales as dx^2, not dx. An earlier version used
        # n_steps proportional to N, so dt fell only as dx and every refinement made the scheme
        # LESS stable: the value error stuck at ~3e-3 regardless of N and the finest grid
        # overflowed to NaN. The maximised term contributes an effective diffusivity of about
        # 0.8 x^2, so dt <= 0.5 dx^2 / 0.8; n_steps = 4 N^2 keeps dt a factor of ten inside that.
        n_steps = int(args.dt_factor * N * N)
        try:
            xg, Vg, dt = solve_merton(prob, N, n_steps)
            Vref = prob.V(xg, 0.0)
            rel = float(np.linalg.norm(Vg - Vref) / np.linalg.norm(Vref))
            row = {"N": N, "n_steps": n_steps, "dt": dt, "rel_l2_value": rel, "errors": {}}
            line = f"  {N:>5d} {n_steps:>8d} {dt:>10.2e} {rel:>24.3e} "
            for m in ("fd", "spline"):
                try:
                    g = EXTRACTORS[m]([xg], Vg, X, args.probe_scale)
                    err = float(np.mean((g[:, 0] - g_ref_solved) ** 2) / g_scale_solved)
                    row["errors"][m] = err
                    ptracks[m].append((float(xg[1] - xg[0]), err))
                    line += f"{err:>12.3e} "
                except Exception:
                    line += f"{'FAIL':>12s} "
            print(line, flush=True)
            results["solved_grid"].append(row)
        except Exception as e:
            print(f"  N={N} FAILED: {type(e).__name__}: {e}")
        save()

    print("\n  solver validation and extraction orders on the solved grid:")
    for r in results["solved_grid"]:
        pass
    if results["solved_grid"]:
        rels = [r["rel_l2_value"] for r in results["solved_grid"]]
        print(f"    solver relative L2 error vs closed form: "
              f"{min(rels):.2e} to {max(rels):.2e}")
    for m in ("fd", "spline"):
        pts = [(h, e) for h, e in ptracks[m] if e and e > 0]
        if len(pts) < 3:
            continue
        hh = np.log([h for h, _ in pts]); ee = np.log([e for _, e in pts])
        A = np.vstack([hh, np.ones_like(hh)]).T
        slope, _ = np.linalg.lstsq(A, ee, rcond=None)[0]
        print(f"    {m:>8s}: order {slope:5.2f}   "
              f"{pts[0][1]:.2e} -> {pts[-1][1]:.2e}  (factor {pts[0][1]/pts[-1][1]:.0f})")
        results["meta"].setdefault("solved_orders", {})[m] = {"order": float(slope)}

    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
