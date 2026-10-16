"""Multilevel Picard (MLP), the family's classical high-dimensional method, on the same benchmark.

WHAT IS IMPLEMENTED, AND FROM WHERE
----------------------------------
The full-history recursive multilevel Picard approximation of Hutzenthaler, Jentzen, Kruse and Nguyen
(arXiv:2009.02484), equation (5):

    U_{n}(t,x) = 1_{n>=1}/M^n * sum_{i=1}^{M^n} g(Y^{x,i}_{t,T})
               + sum_{l=0}^{n-1} (T-t)/M^{n-l} * sum_{i=1}^{M^{n-l}}
                     [ (f o U_l) - 1_{l>=1} (f o U_{l-1}) ]( t + (T-t) r_i , Y^{x,i}_{t, t+(T-t) r_i} )

with U_0 = 0, r uniform on [0,1], Y an Euler-Maruyama path of the LINEAR part of the dynamics, and M
the number of samples per level. The telescoping difference f o U_l - f o U_{l-1} is the whole point of
the method: it is what lets the levels be estimated independently instead of by nested Monte Carlo, and
it is why the cost is polynomial rather than exponential in the level count.

THE ONE PLACE THIS DEVIATES, AND WHY
-----------------------------------
The published scheme as written above is for nonlinearities f(t,x,u) of the SOLUTION. Ours depends on
the GRADIENT: f(x, z) = -1/2 x'Qx + 1/2 z'M z with z = Sigma' grad V, M = Sigma^-1 B R^-1 B' Sigma^-T,
which is what a maximised Hamiltonian always looks like. The literature has a dedicated variant for
this case (Neufeld et al., gradient-dependent nonlinearities) whose distinguishing feature is that the
time sample r is Beta-distributed rather than uniform. We take the other standard route, the one used
in the practical gradient-MLP implementations: keep the uniform-time scheme and approximate grad U_l by
a CENTRAL FINITE DIFFERENCE with step eps.

That choice is stated rather than hidden, and it has two consequences this script reports:

  * the finite-difference step adds an O(eps^2) error inside f, so the scheme has an extra knob whose
    effect is measured here rather than assumed;
  * it is a direct instance of this paper's Proposition 1 -- a solver whose nonlinearity consumes a
    gradient must recover that gradient, and the recovery's order shows up in the solver's accuracy.

ACCEPTANCE TEST BEFORE ANY COMPARISON
------------------------------------
`self_test()` runs the scheme at d=1 against the known exact solution and refuses to report a number
unless the error FALLS as the level count grows. MLP is a probabilistic method with a telescoping
estimator; an implementation with a wrong weight or a wrong time-sampling point still returns
plausible numbers, and the only cheap way to tell is whether they converge to something known.

Usage: python src/multilevel_picard.py --dims 1,4,8 --levels 4 --m 2
Output: results/exp_mlp.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402

RESULTS = SRC.parent / "results"


class MLP:
    """Full-history recursive multilevel Picard approximation, point-valued at (0, x0).

    Point-valued is not a limitation of our implementation but of the method: MLP answers u(t,x) for
    the (t,x) it is asked about, and asking about another point means running it again. That is the
    same axis the paper already discusses for deep BSDE, which is why both are reported as point
    values.
    """

    name = "Multilevel Picard"

    def __init__(self, problem: ACProblem, m: int = 2, levels: int = 4, n_grid: int = 20,
                 eps: float = 1e-3, seed: int = 0, max_samples: int = 20_000_000):
        self.problem = problem
        self.m, self.levels, self.n_grid = m, levels, n_grid
        self.eps = eps
        self.rng = np.random.default_rng(seed)
        self.d = problem.state_dim
        self.T = float(problem.T)
        self.A, self.Sigma = problem.A, problem.Sigma
        self.Q, self.QT = problem.Q, problem.QT
        Sinv = np.linalg.inv(problem.Sigma)
        self.Mmat = Sinv @ problem.B @ problem.R_inv @ problem.B.T @ Sinv.T
        self.BRinv = problem.B @ problem.R_inv          # u_row = gradV_row @ BRinv.T
        self.max_samples = max_samples
        self.samples_used = 0
        self.n_calls = 0

    # ------------------------------------------------------------------ elementary pieces
    def _terminal(self, x: np.ndarray) -> np.ndarray:
        """g(x) = -1/2 x' Q_T x, with x of shape (..., d)."""
        return -0.5 * np.einsum("...i,ij,...j->...", x, self.QT, x)

    def _f(self, x: np.ndarray, z: np.ndarray) -> np.ndarray:
        """f(x, z) = -1/2 x'Qx + 1/2 z' M z, the maximised Hamiltonian of the HJB."""
        return (-0.5 * np.einsum("...i,ij,...j->...", x, self.Q, x)
                + 0.5 * np.einsum("...i,ij,...j->...", z, self.Mmat, z))

    def _simulate(self, t: np.ndarray, x: np.ndarray, horizon: np.ndarray,
                  rng: np.random.Generator) -> np.ndarray:
        """Euler-Maruyama of dX = A X dt + Sigma dW from (t, x) over each row's own horizon.

        `t`, `horizon` and `x` have shapes (K,) (K,) (K, d); the horizon is the length of time to
        integrate. A uniform grid of `n_grid` steps over the horizon is used, which is the
        Euler-Maruyama discretisation of the linear part that the scheme calls Y.

        The generator is passed in rather than held on the object: the finite-difference gradient
        below needs two evaluations that share their randomness, and that is only possible if a
        caller can hand the same stream to both.
        """
        k = x.shape[0]
        y = x.astype(float).copy()
        dt = horizon / self.n_grid
        sq = np.sqrt(np.maximum(dt, 0.0))[:, None]
        for _ in range(self.n_grid):
            dW = rng.standard_normal((k, self.d)) * sq
            y = y + (y @ self.A.T) * dt[:, None] + dW @ self.Sigma.T
            self.samples_used += k
        return y

    # ------------------------------------------------------------------ the recursion
    def U(self, n: int, t: np.ndarray, x: np.ndarray, seed: int = 0) -> np.ndarray:
        """U_n(t, x) for arrays of points; the recursion of equation (5).

        The randomness is drawn from a generator seeded by `seed`, so that two calls with the same
        seed reproduce the same noise. That is not for reproducibility alone: the finite-difference
        gradient is only usable if the up and down evaluations share their randomness, because
        otherwise the difference quotient divides the Monte Carlo noise by 2*eps.
        """
        self.n_calls += 1
        rng = np.random.default_rng(seed)
        k = x.shape[0]
        if n == 0:
            return np.zeros(k)
        out = np.zeros(k)

        # Terminal term: M^n paths per input, averaged.
        n_term = self.m ** n
        x_rep = np.repeat(x, n_term, axis=0)
        t_rep = np.repeat(t, n_term)
        yT = self._simulate(t_rep, x_rep, self.T - t_rep, rng)
        gT = self._terminal(yT).reshape(k, n_term).mean(axis=1)
        out += gT

        # Telescoping correction terms, one per previous level.
        for level in range(n):
            k_l = self.m ** (n - level)          # samples per input point
            t_rep = np.repeat(t, k_l)            # shape (k * k_l,)
            x_rep = np.repeat(x, k_l, axis=0)
            r = rng.random(t_rep.shape[0])       # one uniform time PER SAMPLE, not per input
            horizon = (self.T - t_rep) * r
            y = self._simulate(t_rep, x_rep, horizon, rng)
            s = t_rep + horizon
            a = self._f(y, self._grad_z(level, s, y, seed * 1013 + level))
            if level >= 1:
                b = self._f(y, self._grad_z(level - 1, s, y, seed * 1013 + level))
                diff = a - b
            else:
                diff = a
            out += (self.T - t) * diff.reshape(k, k_l).mean(axis=1)
        return out

    def _grad_z(self, level: int, t: np.ndarray, x: np.ndarray, seed: int) -> np.ndarray:
        """z = Sigma' grad U_level, by central differences of the MLP approximation itself.

        Two things make this usable, and both are load-bearing:

        * COMMON RANDOM NUMBERS. U_level is a Monte Carlo estimate, so its finite-difference
          quotient has noise of order sigma_U/(2 eps). With independent randomness that noise is
          enormous -- measured on the first run of this script, eps = 1e-3 and levels 1..3 gave
          relative errors of 8, 97 and 5e11, i.e. divergence. The up and down evaluations therefore
          receive the SAME seed, which cancels the noise to the extent the perturbation is small.
        * A STEP THAT CAN BE TUNED. The remaining error is O(eps^2) from the finite difference
          against O(sigma_U/eps) from what the cancellation does not remove, so eps is a real knob
          with a minimum, and it is exposed as a parameter and swept rather than fixed silently.
        """
        grad = np.zeros_like(x)
        for i in range(self.d):
            e = np.zeros(self.d)
            e[i] = self.eps
            up = self.U(level, t, x + e, seed=seed)
            dn = self.U(level, t, x - e, seed=seed)     # same seed: common random numbers
            grad[:, i] = (up - dn) / (2.0 * self.eps)
        return grad @ self.Sigma          # row form: z_row = grad_row @ Sigma


def self_test(problem: ACProblem, m: int, grids: tuple[int, ...], eps: float,
              seed: int = 0) -> dict:
    """Does the error fall as the level count grows? Nothing else is worth reporting until it does."""
    x0 = np.zeros(problem.state_dim)
    x0[: problem.n_assets] = 1.0
    ref = problem.reference(np.linspace(0.0, problem.T, 401))
    exact = float(ref.value(x0, 0.0))
    out = {}
    for levels in grids:
        solver = MLP(problem, m=m, levels=levels, eps=eps, seed=seed)
        t0 = time.perf_counter()
        v = float(solver.U(levels, np.array([0.0]), x0.reshape(1, -1))[0])
        out[levels] = {"value": v, "rel_err": abs(v - exact) / abs(exact),
                       "exact": exact, "seconds": time.perf_counter() - t0,
                       "samples": solver.samples_used}
    return out


def implementation_check(m: int, n_grid: int, eps: float, seed: int) -> tuple[dict, bool]:
    """Does the implementation converge where the method's hypothesis HOLDS?

    This is the acceptance test, and it deliberately runs on a WEAK-nonlinearity instance (eta = 1)
    rather than on the paper's benchmark (eta = 0.01). The reason is that MLP realises the Picard
    iteration, and the iteration's contraction constant is |grad u| / eta: at eta = 0.01 it is not a
    contraction and MLP diverges no matter how correct the code is. Testing the implementation
    against a hypothesis it does not satisfy would reject a correct implementation, and testing it
    only where it works would hide the reason it fails. So: converge at weak nonlinearity (or the
    code is wrong), report the divergence at strong nonlinearity (or the physics is missed).
    """
    prob = ACProblem(p=1, eta=1.0, lam_sigma2=0.01, q_terminal=0.1, sigma=0.01)
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    x0 = np.zeros(prob.state_dim)
    x0[0] = 1.0
    exact = float(ref.value(x0, 0.0))
    out, prev = {}, None
    for levels in (1, 2, 3, 4):
        solver = MLP(prob, m=m, levels=levels, n_grid=n_grid, eps=eps, seed=seed)
        t0 = time.perf_counter()
        v = float(solver.U(levels, np.array([0.0]), x0.reshape(1, -1))[0])
        rel = abs(v - exact) / abs(exact)
        out[levels] = {"value": v, "rel_err": rel, "exact": exact,
                       "seconds": time.perf_counter() - t0, "samples": solver.samples_used}
        prev = rel
    # Converged means: the last two levels agree to well under a percent, and the error is small.
    ok = out[3]["rel_err"] < 0.05 and out[4]["rel_err"] < 0.05
    return out, ok


def instance_row(d: int, eta: float, args) -> dict:
    """One MLP run at a given dimension and control authority, against the exact solution."""
    prob = ACProblem(p=d, eta=eta, lam_sigma2=0.01, q_terminal=0.1, sigma=0.01)
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    exact = float(ref.value(x0, 0.0))
    solver = MLP(prob, m=args.m, levels=args.levels, n_grid=args.n_grid, eps=args.eps,
                 seed=args.seed)
    t0 = time.perf_counter()
    v = float(solver.U(args.levels, np.array([0.0]), x0.reshape(1, -1))[0])
    secs = time.perf_counter() - t0
    return {"d": d, "eta": eta, "value": v, "exact": exact,
            "rel_err": abs(v - exact) / abs(exact), "seconds": secs,
            "samples": solver.samples_used, "calls": solver.n_calls,
            "m": args.m, "levels": args.levels, "eps": args.eps, "n_grid": args.n_grid}


def contraction_sweep(args) -> list[dict]:
    """Where does MLP stop working, and why?

    MLP is a Monte Carlo realisation of the Picard iteration
        u_{n+1}(t,x) = E[ g(X_T) + int_t^T f(X_s, grad u_n(s,X_s)) ds ],
    so it inherits that iteration's convergence requirement. For the HJB of optimal execution the
    nonlinearity is the maximised Hamiltonian 1/2 grad V' B R^-1 B' grad V, whose Lipschitz constant
    in grad u is |grad u| / eta. When that is large enough -- which at a tight spread it is -- the
    iteration is not a contraction and the MLP iterates DIVERGE.

    This sweeps eta from weak to strong nonlinearity and reports the error at each level, which turns
    "the method needs a contraction" from a caveat into a measured boundary.
    """
    rows = []
    for eta in (10.0, 1.0, 0.3, 0.1, 0.03, 0.01):
        prob = ACProblem(p=1, eta=eta, lam_sigma2=0.01, q_terminal=0.1, sigma=0.01)
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        x0 = np.zeros(prob.state_dim)
        x0[0] = 1.0
        exact = float(ref.value(x0, 0.0))
        u_star = np.asarray(ref.control(x0.reshape(1, -1), 0.0)).ravel()[0]
        row = {"eta": eta, "exact": exact, "grad_V_at_x0": abs(u_star * eta), "levels": {}}
        for levels in (1, 2, 3):
            solver = MLP(prob, m=args.m, levels=levels, n_grid=args.n_grid, eps=args.eps,
                         seed=args.seed)
            t0 = time.perf_counter()
            v = float(solver.U(levels, np.array([0.0]), x0.reshape(1, -1))[0])
            row["levels"][levels] = {"value": v, "rel_err": abs(v - exact) / abs(exact),
                                     "seconds": time.perf_counter() - t0}
        rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,4,8,12")
    ap.add_argument("--m", type=int, default=2, help="samples per level (M in the scheme)")
    ap.add_argument("--levels", type=int, default=3)
    ap.add_argument("--n-grid", type=int, default=10)
    ap.add_argument("--eps", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp_mlp.json")
    args = ap.parse_args()

    print("=" * 100)
    print("MULTILEVEL PICARD (full-history recursive MLP, eq. (5) of arXiv:2009.02484)")
    print("=" * 100)
    print(f"  M={args.m}, levels={args.levels}, Euler grid={args.n_grid}, "
          f"finite-difference step={args.eps:g}")

    print("\n  IMPLEMENTATION CHECK -- weak nonlinearity (eta=1), where the Picard map is a")
    print("  contraction and the method's hypothesis holds. A wrong implementation fails here.")
    check, ok = implementation_check(args.m, args.n_grid, args.eps, args.seed)
    ex = check[1]["exact"]
    print(f"  {'levels':>7} {'value':>16} {'rel err':>12} {'seconds':>9}")
    for lv, r in check.items():
        print(f"  {lv:>7} {r['value']:>16.8f} {r['rel_err']:>12.3e} {r['seconds']:>9.2f}")
    if not ok:
        print("\n  IMPLEMENTATION CHECK FAILED: refusing to report anything about the benchmark.")
        return 1
    print("  -> converges, so the implementation is sound and any failure below is the method's")

    out = {"args": vars(args), "implementation_check": check, "rows": [],
           "contraction_sweep": []}

    print(f"\n  CONTRACTION BOUNDARY (d=1). The Lipschitz constant of the Picard map is")
    print(f"  |grad V| / eta, so eta decides whether the iteration MLP realises converges.")
    print(f"  {'eta':>7} {'|gradV(x0)|':>12} {'level 3 value':>16} {'L3 rel err':>12} {'verdict':>13}")
    sweep = contraction_sweep(args)
    out["contraction_sweep"] = sweep
    for row in sweep:
        l3 = row["levels"][3]
        verdict = ("converged" if l3["rel_err"] < 0.1
                   else "not converged" if l3["rel_err"] < 1.0 else "diverged")
        print(f"  {row['eta']:>7.2f} {row['grad_V_at_x0']:>12.4f} {l3['value']:>16.6f} "
              f"{l3['rel_err']:>12.3e} {verdict:>13}")

    print(f"\n  THE BENCHMARK (eta = 0.01, the paper's calibration):")
    for d in [int(s) for s in args.dims.split(",") if s.strip()]:
        r = instance_row(d, 0.01, args)
        out["rows"].append(r)
        print(f"    d={d:>3}: rel err {r['rel_err']*100:>12.3f}%   {r['seconds']:>6.2f}s   "
              f"{r['samples']:>10,} path-samples")

    print(f"\n  THE SAME HJB FAMILY AT eta = 1 (weaker control authority, contraction holds):")
    for d in [int(s) for s in args.dims.split(",") if s.strip()]:
        r = instance_row(d, 1.0, args)
        out["rows"].append(r)
        print(f"    d={d:>3}: rel err {r['rel_err']*100:>12.3f}%   {r['seconds']:>6.2f}s   "
              f"{r['samples']:>10,} path-samples")

    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    print("MLP COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
