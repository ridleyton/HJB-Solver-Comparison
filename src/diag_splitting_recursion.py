"""Where does deep splitting lose this benchmark: the scheme, or the network?

WHY THIS DIAGNOSTIC EXISTS
--------------------------
The trained recursion on this benchmark is inaccurate for the value (R^2 of order 0.1 or worse) while
its CONTROL is roughly right (R^2 ~ 0.9). Two very different explanations fit that pattern:

  (a) the regression is the limiter -- each step's network is fitted imperfectly and the biases
      accumulate; or
  (b) the RECURSION is the limiter -- the per-step map amplifies errors, and a better fit makes it
      worse, not better, because a smoother approximation damps the amplified modes.

The trained runs point at (b): increasing the per-step budget from 300 to 2000 epochs improved the
median step loss but made the value R^2 worse. This script removes the network from the question
entirely by fitting each step EXACTLY in the quadratic basis the true solution lives in -- a linear
least-squares solve, no optimiser, no noise floor. Whatever error remains is the recursion's own.

If the quadratic-basis recursion is accurate, the recursion is fine and the network is the limiter.
If it also drifts, the recursion is the limiter and no amount of network capacity will fix it.

Usage: python src/diag_splitting_recursion.py [--steps 50] [--inner 1]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from deep_splitting import DeepSplitting  # noqa: E402


def quad_features(X: np.ndarray) -> np.ndarray:
    """1, x_i, x_i x_j -- the basis the exact solution lives in."""
    n, d = X.shape
    cols = [np.ones((n, 1)), X]
    for i in range(d):
        for j in range(i, d):
            cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
    return np.hstack(cols)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--paths", type=int, default=4096)
    ap.add_argument("--spread", action="store_true", default=True)
    ap.add_argument("--dim", type=int, default=1)
    args = ap.parse_args()

    if not self_test():
        print("  SELF-TEST FAILED: the gradient helper disagrees with finite differences.")
        return 1

    prob = ACProblem(p=args.dim)
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    s = DeepSplitting(prob, n_steps=args.steps, n_paths=args.paths, epochs=1, seed=0,
                      spread=args.spread, device="cpu", inner=args.inner if hasattr(args, "inner")
                      else 1)
    s._simulate(torch.Generator(device="cpu").manual_seed(0))

    # Backward recursion with an exact quadratic fit at every step (no network involved).
    coefs: list[np.ndarray | None] = [None] * (args.steps + 1)
    for n in range(args.steps - 1, -1, -1):
        x_n = s.paths[n].numpy()
        x_next = s.paths[n + 1].numpy()
        if n == args.steps - 1:
            base = -0.5 * np.einsum("ni,ij,nj->n", x_next, prob.QT, x_next)
            z = x_n @ (-prob.QT) @ prob.Sigma
        else:
            base = quad_features(x_next) @ coefs[n + 1]
            Pn = ref.P((n + 1) * s.dt)
            g = quad_features(x_n) @ coefs[n + 1]
            # gradient of the fitted quadratic: d/dx of (c + g'x + 1/2 x'Hx)
            grad = fitted_gradient(coefs[n + 1], x_n, prob.state_dim)
            z = grad @ prob.Sigma
            del g, Pn
        target = base + s.dt * s._f(torch.tensor(x_n, dtype=torch.float32),
                                    torch.tensor(z, dtype=torch.float32)).numpy().ravel()
        A = quad_features(x_n)
        coefs[n], *_ = np.linalg.lstsq(A, target, rcond=None)

    print("=" * 92)
    print(f"QUADRATIC-BASIS SPLITTING RECURSION (no network): d={args.dim}, {args.steps} steps")
    print("=" * 92)
    print(f"  {'step':>5} {'t':>6} {'max |fit - exact|':>18} {'value scale':>12} {'relative':>10}")
    X = np.random.default_rng(5).uniform(-1, 1, size=(4000, args.dim))
    for n in (args.steps - 1, args.steps - 2, args.steps - 5, args.steps // 2, 5, 1, 0):
        if n < 0 or coefs[n] is None:
            continue
        t = n * s.dt
        pred = quad_features(X) @ coefs[n]
        exact = np.asarray(ref.value(X, t)).ravel()
        err = np.abs(pred - exact).max()
        print(f"  {n:>5} {t:>6.2f} {err:>18.3e} {np.abs(exact).max():>12.3e} "
              f"{err/np.abs(exact).max():>10.3e}")
    return 0


def fitted_gradient(coef: np.ndarray, X: np.ndarray, d: int) -> np.ndarray:
    """grad of the fitted quadratic c + g'x + sum_{i<=j} H_ij x_i x_j.

    THE FACTOR OF TWO, which is where this diagnostic first went wrong. `quad_features` emits the
    raw products x_i x_j, so the fitted model is c + g'x + x'Hx, and its gradient is g + 2Hx. The
    first version returned g + Hx -- half the true gradient. Because the nonlinearity enters the
    recursion as z'Mz, halving z quarters that feedback term, and the recursion then inflates P by
    2.6x (measured: P(0) = 0.0322 against the true 0.01249). That looked exactly like a scheme
    instability and very nearly became a reported negative result about deep splitting. It is the
    same convention trap as the quadratic-vs-half-quadratic form, and it is why the returned value
    is checked against a finite difference in `self_test`.
    """
    n = X.shape[0]
    g = coef[1:1 + d]
    H = np.zeros((d, d))
    k = 1 + d
    for i in range(d):
        for j in range(i, d):
            H[i, j] = coef[k]
            H[j, i] = coef[k]
            k += 1
    # H is symmetric and holds the coefficients of the RAW products x_i x_j. Two corrections are
    # needed, and they are different for diagonal and off-diagonal entries:
    #   d(x_k^2)/dx_k = 2 x_k      -> the diagonal gets an extra x_k H_kk on top of (X @ H)_k;
    #   d(x_i x_j)/dx_i = x_j      -> the off-diagonal term is already counted once by X @ H,
    #                                 so the factor of two must NOT be applied to it.
    # Applying 2*(X @ H) uniformly (the previous version) is right for d = 1, where every entry is
    # diagonal, and wrong for every d >= 2 -- which is exactly what the self-test reported: 1e-10 for
    # d = 1 against 0.65 for d = 3.
    return np.tile(g, (n, 1)) + X @ H + X * np.diag(H)[None, :]


def self_test() -> bool:
    """The gradient helper against a finite difference. Run before trusting the diagnostic."""
    rng = np.random.default_rng(0)
    for d in (1, 3):
        X = rng.uniform(-1, 1, size=(5, d))
        coef = rng.normal(size=1 + d + d * (d + 1) // 2)
        g = fitted_gradient(coef, X, d)
        eps = 1e-6
        num = np.zeros_like(X)
        for i in range(d):
            Xp, Xm = X.copy(), X.copy()
            Xp[:, i] += eps
            Xm[:, i] -= eps
            num[:, i] = (quad_features(Xp) @ coef - quad_features(Xm) @ coef) / (2 * eps)
        err = np.abs(g - num).max()
        print(f"  self-test d={d}: max |analytic - finite difference| = {err:.2e}")
        if err > 1e-5:
            return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
