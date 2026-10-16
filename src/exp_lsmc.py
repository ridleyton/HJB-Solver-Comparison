"""Regression Monte Carlo: the classical mesh-free baseline the comparison was missing.

WHY THIS EXPERIMENT EXISTS
--------------------------
The paper's boundary claim is that beyond p ~ 5 the grid stops being runnable and "the mesh-free
solver is the only method that runs". That is true of the methods measured, and it is not a claim
about methods in general: regression Monte Carlo -- the Longstaff-Schwartz family, applied to control
rather than to stopping -- also needs no grid, and the paper currently admits in Related Work that it
was not run. A referee will ask why the classical mesh-free method is absent from a paper whose thesis
is about the boundary between classical and mesh-free. This runs it.

METHOD (policy iteration with backward regression, the standard construction)
-----------------------------------------------------------------------------
Basis: the quadratic features 1, x_i, x_i x_j (i < j), (1/2) x_i^2 -- the natural basis for a
linear-quadratic value function, and NOT the answer handed over: the coefficients are estimated from
simulated paths.

  1. simulate M paths under the current policy u^(k) (Euler, step dt = T/N);
  2. backward pass: at each t_n regress the realised future reward
     Y_n = sum_{k>=n} -1/2(x_k'Qx_k + u_k'Ru_k) dt - 1/2 x_N'Q_T x_N  on the basis at x_n, giving
     V_n(x) = c + g'x + 1/2 x'Hx for that time slice;
  3. improve: u^(k+1)(x,t_n) = R^-1 B'(g_n + H_n x);
  4. repeat. The value of the improved policy is regressed by the same pass, so the reported accuracy
     is the accuracy of a policy that was actually simulated, not of an extrapolated one.

Cost is M*N vector operations plus one least-squares solve of an (M x m) system per time slice, with
m = 1 + d + d(d+1)/2 basis functions. That m grows quadratically in d is where the curse lives for
this family -- not in the state dimension as such -- and the sweep reports it.

Usage: python src/exp_lsmc.py [--dims 1,4,8,12,20] [--paths 40000] [--steps 100]
Output: results/exp_lsmc.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from metrics import relative_l2  # noqa: E402

RESULTS = SRC.parent / "results"


def basis(X: np.ndarray) -> np.ndarray:
    """Quadratic features: [1, x_i, x_i x_j (i<j), (1/2) x_i^2]."""
    M, d = X.shape
    cols = [np.ones(M)]
    cols.extend(X[:, i] for i in range(d))
    for i in range(d):
        for j in range(i + 1, d):
            cols.append(X[:, i] * X[:, j])
    for i in range(d):
        cols.append(0.5 * X[:, i] ** 2)
    return np.column_stack(cols)


def unpack(coef: np.ndarray, d: int) -> tuple[float, np.ndarray, np.ndarray]:
    """Coefficients -> (c, g, H) so that V(x) = c + g'x + 1/2 x'Hx."""
    c = float(coef[0])
    g = coef[1:1 + d].copy()
    H = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i + 1, d):
            H[i, j] = H[j, i] = coef[k]
            k += 1
    for i in range(d):
        H[i, i] = coef[k]
        k += 1
    return c, g, H


def simulate(prob: ACProblem, policy, M: int, N: int, rng: np.random.Generator,
             x0: np.ndarray, dt: float, sigma_step: np.ndarray, spread: bool = True):
    """Euler paths under a policy given as a list of (c, g, H) per time slice.

    The starting states are drawn ACROSS THE DOMAIN rather than all at x_0, and that is what makes
    the regression well-posed. With every path starting at x_0 = (1, ..., 1) the cloud at each time
    slice is a sphere of radius sigma*sqrt(T) = 0.01 around a corner of the box, and a quadratic
    fitted there says nothing about the rest of the domain -- the first version of this script did
    exactly that, produced a meaningless Hessian, extracted a control from it, and diverged by a
    factor of 1e33 within two iterations. Regression Monte Carlo estimates the value FUNCTION over a
    region, so the paths have to cover the region.
    """
    d = prob.state_dim
    lo, hi = prob.domain()
    X = np.empty((N + 1, M, d))
    U = np.empty((N, M, prob.n_assets))
    if spread:
        X[0] = rng.uniform(lo, hi, size=(M, d))
    else:
        X[0] = np.broadcast_to(x0, (M, d))
    for n in range(N):
        c, g, H = policy[n]
        # u = R^-1 B' grad V with grad V = g + Hx
        U[n] = (g + X[n] @ H.T) @ prob.B @ prob.R_inv.T
        drift = X[n] @ prob.A.T + U[n] @ prob.B.T
        X[n + 1] = X[n] + drift * dt + sigma_step * rng.standard_normal((M, d))
    return X, U


def backward(prob: ACProblem, X: np.ndarray, U: np.ndarray, dt: float,
             ridge: float = 1e-10) -> list:
    """Regress the realised future reward backwards, one least-squares solve per time slice."""
    N = U.shape[0]
    M = X.shape[1]
    run = np.einsum("...i,ij,...j->...", X[:-1], prob.Q, X[:-1]) \
        + np.einsum("...i,ij,...j->...", U, prob.R, U)
    term = np.einsum("...i,ij,...j->...", X[-1], prob.QT, X[-1])
    # Y[n] = future reward from t_n onward
    Y = np.empty((N + 1, M))
    Y[N] = -0.5 * term
    acc = -0.5 * run[N - 1] * dt
    Y[N - 1] = acc - 0.5 * term
    for n in range(N - 2, -1, -1):
        acc = acc - 0.5 * run[n] * dt
        Y[n] = acc - 0.5 * term
    # Regress at each time slice, reusing one basis matrix per slice. A small ridge term keeps the
    # normal equations well posed where the cloud is thin in some quadratic direction.
    policy = []
    for n in range(N + 1):
        B = basis(X[n])
        m = B.shape[1]
        A_ = np.vstack([B, math.sqrt(ridge * M) * np.eye(m)])
        b_ = np.concatenate([Y[n], np.zeros(m)])
        coef, *_ = np.linalg.lstsq(A_, b_, rcond=None)
        policy.append(unpack(coef, prob.state_dim))
    return policy


def damp(old: list, fitted: list, alpha: float) -> list:
    """Blend the greedy control toward the previous one.

    The greedy update replaces the policy with u = R^-1 B' grad V_fit at every time slice. With a
    noisily estimated V_fit that is not a safe move: the gain is 1/eta = 100 here, so a Hessian
    error of 1e-3 becomes a feedback gain error of 0.1, and the closed loop then grows like
    exp(0.1 T) or worse -- the first version of this script replaced the policy outright and diverged
    to 1e33 in two iterations. Damping is the standard actor-critic remedy and costs only iterations.
    """
    out = []
    for (c0, g0, H0), (c1, g1, H1) in zip(old, fitted):
        out.append(((1 - alpha) * c0 + alpha * c1,
                    (1 - alpha) * g0 + alpha * g1,
                    (1 - alpha) * H0 + alpha * H1))
    return out


def evaluate(prob: ACProblem, policy, ref, x0: np.ndarray, seed: int = 11, n_points: int = 4000):
    """Score the fitted value function and its control over the domain against the exact reference."""
    rng = np.random.default_rng(seed)
    lo, hi = prob.domain()
    X = rng.uniform(lo, hi, size=(n_points, prob.state_dim))
    c, g, H = policy[0]
    V_hat = c + X @ g + 0.5 * np.einsum("ij,jk,ik->i", X, H, X)
    V_ref = np.asarray(ref.value(X, 0.0)).ravel()
    # control from the fitted quadratic
    u_hat = (g + X @ H.T) @ prob.B @ prob.R_inv.T
    u_ref = ref.control(X, 0.0)
    x0 = np.atleast_2d(x0)
    c0, g0, H0 = policy[0]
    V0_hat = float(c0 + x0 @ g0 + 0.5 * x0 @ H0 @ x0.T)
    V0_ref = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])
    return {
        "value_r2": float(1.0 - np.sum((V_hat - V_ref) ** 2) / np.sum((V_ref - V_ref.mean()) ** 2)),
        "value_rel_l2": relative_l2(V_hat, V_ref),
        "control_r2": float(1.0 - np.sum((u_hat - u_ref) ** 2) / np.sum((u_ref - u_ref.mean()) ** 2)),
        "control_rel_l2": relative_l2(u_hat.ravel(), u_ref.ravel()),
        "value_rel_err_at_x0": abs(V0_hat - V0_ref) / abs(V0_ref),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,4,8,12,20")
    ap.add_argument("--paths", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--iters", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.5,
                    help="damping of the greedy policy update (1.0 = undamped, which diverges)")
    ap.add_argument("--out", default="exp_lsmc.json")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    print("=" * 104)
    print("REGRESSION MONTE CARLO (LSMC-style policy iteration) on the non-separable benchmark")
    print("=" * 104)
    print(f"  paths {args.paths}, time steps {args.steps}, policy iterations {args.iters}")

    out = {"paths": args.paths, "steps": args.steps, "iters": args.iters, "dims": {}}
    for d in dims:
        prob = ACProblem(p=d, hetero=True, het_seed=1)
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        dt = prob.T / args.steps
        # Sigma is diagonal here; the per-axis step scale is sqrt(D_i * dt).
        D = np.diag(prob.Sigma @ prob.Sigma.T)
        sigma_step = np.sqrt(D * dt)
        m = 1 + prob.state_dim + prob.state_dim * (prob.state_dim + 1) // 2
        print(f"\n  d = {d}   basis functions m = {m}")
        print(f"    {'iter':>4s} {'value R2':>10s} {'ctrl R2':>10s} {'value relL2':>12s} "
              f"{'ctrl relL2':>11s} {'rel err at x0':>14s} {'sec':>7s}")
        policy = [None] * (args.steps + 1)
        # Initial policy: no trading. Its value function is whatever the regression finds.
        zero = (0.0, np.zeros(prob.state_dim), np.zeros((prob.state_dim, prob.state_dim)))
        policy = [zero] * (args.steps + 1)
        rows = []
        for it in range(args.iters):
            t0 = time.perf_counter()
            rng = np.random.default_rng(1000 + it)
            X, U = simulate(prob, policy, args.paths, args.steps, rng, x0, dt, sigma_step)
            fitted = backward(prob, X, U, dt)
            # The critic's value is what is reported: it is the value of the policy that was
            # actually simulated, not of one extrapolated from it.
            sc = evaluate(prob, fitted, ref, x0)
            sc.update({"iter": it, "seconds": time.perf_counter() - t0,
                       "basis_functions": m, "paths": args.paths, "steps": args.steps,
                       "alpha": args.alpha})
            rows.append(sc)
            policy = damp(policy, fitted, args.alpha)
            print(f"    {it:>4d} {sc['value_r2']:>10.4f} {sc['control_r2']:>10.4f} "
                  f"{sc['value_rel_l2']:>12.3e} {sc['control_rel_l2']:>11.3e} "
                  f"{sc['value_rel_err_at_x0']:>14.3e} {sc['seconds']:>7.1f}")
        out["dims"][str(d)] = rows

    print()
    print("=" * 104)
    print("READING")
    print("=" * 104)
    for d, rows in out["dims"].items():
        r = rows[-1]
        print(f"  d={d}: value R2 {r['value_r2']:+.4f}   control R2 {r['control_r2']:+.4f}   "
              f"m={r['basis_functions']}   {r['seconds']:.1f}s for {r['paths']} paths x "
              f"{r['steps']} steps")
    out["note"] = ("Regression Monte Carlo needs no grid and runs at every dimension here; its basis "
                   "count grows as 1 + d + d(d+1)/2.")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
