"""Verification of Proposition 2: the exact suboptimality identity for the LQ benchmark.

THE CLAIM
---------
Let u* be the optimal feedback of the linear-quadratic benchmark and u any admissible control, with
x, x* the states each produces from the same initial condition (and the same noise). Then

    J(u*) - J(u)  =  1/2 * E[ int_0^T (u - u*)(x(t), t)' R (u - u*)(x(t), t) dt ]

EXACTLY, with no remainder. This matters here because the paper previously attributed the observed
loss order to "the envelope theorem", which is both weaker and differently emphasised: the identity
says the loss IS the time integral of the squared control error along the path actually taken, so a
degree-k reconstruction (control error O(h^k)) loses O(h^{2k}) of objective -- a rate, not a bound.

DERIVATION (what the script checks)
-----------------------------------
With V(x,t) = -1/2 x'P(t)x + c(t) solving the HJB, define the verification discrepancy

    D(x,u) := V_t + grad V'(Ax + Bu) + 1/2 tr(Sigma Sigma' grad^2 V) - 1/2 (x'Qx + u'Ru).

The Riccati equation cancels every quadratic term in x and leaves

    D(x,u) = -1/2 (u - u*(x,t))' R (u - u*(x,t))        exactly,

so Dynkin's formula, J(u) = V(x_0,0) + E[int D dt], and D(x,u*) = 0 give the claim.

HOW THE CHECK IS MADE HONEST
----------------------------
The two sides are both O(1) objectives (here ~6e-3) whose difference is much smaller, so the
attainable RELATIVE agreement is limited by the absolute accuracy of each integral -- about 2.5e-11
with Simpson and RK4 at 4000 steps. A check that reported only the relative difference would look
like a failure at small deviations and a success at large ones for no reason connected to the claim.
So the script reports both, and demonstrates the floor:

  A.  a SMOOTH analytic perturbation of the optimal feedback, at three sizes. The absolute
      difference stays at the floor while the relative difference falls as the gap grows.
  A'. the same at four step counts: the absolute difference falls like a power of dt, which is what
      identifies it as quadrature error rather than a defect in the identity.
  B.  the paper's own setting: control taken from a multilinear and a cubic reconstruction of the
      exact value function on a grid, so the deviation is the real reconstruction error.

Usage: python src/verify_suboptimality_identity.py
Output: results/check_suboptimality_identity.json
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from exp_control_extraction import extract_cell, extract_spline  # noqa: E402

RESULTS = SRC.parent / "results"


def rollout(problem: ACProblem, control, x0: np.ndarray, n_steps: int = 4000):
    """Integrate dx = (Ax + Bu) dt with RK4; controls are evaluated at every stage."""
    T = problem.T
    dt = T / n_steps
    xs = np.zeros((n_steps + 1, x0.size))
    us = np.zeros((n_steps + 1, x0.size))
    xs[0] = x0
    t = 0.0
    for k in range(n_steps):
        def rhs(x, tt):
            u = control(x.reshape(1, -1), tt)[0]
            return problem.A @ x + problem.B @ u, u
        k1, u1 = rhs(xs[k], t)
        k2, _ = rhs(xs[k] + 0.5 * dt * k1, t + 0.5 * dt)
        k3, _ = rhs(xs[k] + 0.5 * dt * k2, t + 0.5 * dt)
        k4, _ = rhs(xs[k] + dt * k3, t + dt)
        us[k] = u1
        xs[k + 1] = xs[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    us[n_steps] = control(xs[n_steps].reshape(1, -1), T)[0]
    return xs, us, dt


def _simpson(values: np.ndarray, dt: float) -> float:
    n = values.size - 1
    if n % 2:
        return dt * (0.5 * values[0] + values[1:-1].sum() + 0.5 * values[-1])
    w = np.ones(n + 1)
    w[1:-1:2] = 4.0
    w[2:-1:2] = 2.0
    return dt / 3.0 * float(w @ values)


def objective(problem: ACProblem, xs: np.ndarray, us: np.ndarray, dt: float) -> float:
    """-1/2 int (x'Qx + u'Ru) dt - 1/2 x_T'QT x_T, Simpson in time (see the module docstring)."""
    q = np.einsum("ij,jk,ik->i", xs, problem.Q, xs)
    r = np.einsum("ij,jk,ik->i", us, problem.R, us)
    return float(-_simpson(0.5 * (q + r), dt) - 0.5 * xs[-1] @ problem.QT @ xs[-1])


def compare(problem: ACProblem, ref, x0, control, n_steps: int = 4000):
    """Both sides of the identity for one control, plus the position-error bound."""
    def control_star(X, t):
        return ref.control(np.atleast_2d(X), float(t))

    xs_hat, us_hat, dt = rollout(problem, control, x0, n_steps=n_steps)
    xs_star, us_star, _ = rollout(problem, control_star, x0, n_steps=n_steps)
    gap_direct = objective(problem, xs_star, us_star, dt) - objective(problem, xs_hat, us_hat, dt)

    times = np.linspace(0.0, problem.T, us_hat.shape[0])
    dev = np.empty_like(us_hat)
    for k, tk in enumerate(times):
        dev[k] = us_hat[k] - control_star(xs_hat[k].reshape(1, -1), tk)[0]
    gap_identity = 0.5 * _simpson(np.einsum("ij,jk,ik->i", dev, problem.R, dev), dt)

    pos_err = float(np.max(np.linalg.norm(xs_hat - xs_star, axis=1)))
    bound = problem.T * float(np.linalg.norm(problem.B)) * float(np.max(np.linalg.norm(dev, axis=1)))
    return {
        "gap_direct": float(gap_direct), "gap_identity": float(gap_identity),
        "abs_difference": float(abs(gap_direct - gap_identity)),
        "rel_difference": float(abs(gap_direct - gap_identity) / abs(gap_direct)),
        "position_error_max": pos_err, "position_bound": bound,
        "bound_satisfied": bool(pos_err <= bound * (1 + 1e-9)),
        "n_steps": n_steps,
    }


def main() -> int:
    print("=" * 100)
    print("PROPOSITION 2:  J(u*) - J(u)  =  1/2 int (u - u*)(x,t)' R (u - u*)(x,t) dt")
    print("=" * 100)

    out: dict = {"cases": [], "smooth": [], "floor_study": [], "reference_study": []}
    # The reference's own P(t) is a PIECEWISE-LINEAR interpolation of a 401-point Riccati solve
    # (`ACProblem.reference` -> np.interp). So the control this script calls "optimal" is the exact
    # maximiser plus an interpolation error, and the identity checked against it acquires a residual
    # that is FIRST order in the control deviation -- measured below, and shown to fall as the
    # reference grid is refined. That is a property of the verifier, not of the proposition.
    N_REF = 20001
    for p in (1, 2):
        problem = ACProblem(p=p)
        problem.Sigma = np.zeros_like(problem.Sigma)      # deterministic: the identity is pathwise
        ref = problem.reference(np.linspace(0.0, problem.T, N_REF))
        x0 = np.zeros(problem.state_dim)
        x0[: problem.n_assets] = 1.0

        def control_star(X, t, _ref=ref):
            return _ref.control(np.atleast_2d(X), float(t))

        # ---- A: a smooth analytic perturbation of the optimal feedback -------------------
        print(f"\n  --- A. smooth perturbation, p={p} (the floor test) ---")
        print(f"      {'eps':>6s} {'gap':>14s} {'identity':>14s} {'abs diff':>12s} {'rel diff':>12s}")
        for eps in (0.05, 0.1, 0.2):
            def control(X, t, _eps=eps, _cs=control_star, _T=problem.T):
                return _cs(X, t) + _eps * np.sin(np.pi * float(t) / _T)

            r = compare(problem, ref, x0, control)
            r.update({"kind": "smooth", "p": p, "eps": eps})
            out["smooth"].append(r)
            print(f"      {eps:>6.2f} {r['gap_direct']:>14.6e} {r['gap_identity']:>14.6e} "
                  f"{r['abs_difference']:>12.3e} {r['rel_difference']:>12.3e}")

        # ---- B: the paper's setting -- control from a reconstruction ----------------------
        lo, hi = problem.domain()
        N = 21
        axes = [np.linspace(lo[i], hi[i], N) for i in range(p)]
        mesh = np.meshgrid(*axes, indexing="ij")
        Xn = np.stack([m.ravel() for m in mesh], axis=1)
        snap = np.asarray(ref.value(Xn, 0.0)).ravel().reshape(tuple([N] * p))
        print(f"\n  --- B. reconstructed control, p={p}, N={N} ---")
        for name, extractor in (("multilinear", extract_cell), ("cubic", extract_spline)):
            def control(X, t, _ex=extractor, _ax=axes, _sn=snap, _pr=problem):
                g = _ex(_ax, _sn, np.atleast_2d(X), 0.25)
                return g @ _pr.B @ _pr.R_inv.T

            r = compare(problem, ref, x0, control)
            r.update({"kind": "reconstruction", "p": p, "control": name})
            out["cases"].append(r)
            print(f"      {name:12s} gap={r['gap_direct']:+.6e}  identity={r['gap_identity']:+.6e}  "
                  f"rel diff={r['rel_difference']:.2e}  "
                  f"pos err {r['position_error_max']:.3e} <= bound {r['position_bound']:.3e}: "
                  f"{r['bound_satisfied']}")

    # ---- A': is the residual quadrature error? it must fall with dt -----------------------
    print()
    print("=" * 100)
    print("  A'.  the same smooth case at four step counts: does the residual fall with dt?")
    print("=" * 100)
    problem = ACProblem(p=1)
    problem.Sigma = np.zeros_like(problem.Sigma)
    ref = problem.reference(np.linspace(0.0, problem.T, N_REF))   # dense: see the note above
    x0 = np.zeros(problem.state_dim)
    x0[: problem.n_assets] = 1.0

    def control_star(X, t, _ref=ref):
        return _ref.control(np.atleast_2d(X), float(t))

    print(f"      {'n_steps':>8s} {'dt':>10s} {'abs diff':>14s} {'rel diff':>12s}")
    for n_steps in (1000, 2000, 4000, 8000):
        def control(X, t, _cs=control_star, _T=problem.T):
            return _cs(X, t) + 0.2 * np.sin(np.pi * float(t) / _T)

        r = compare(problem, ref, x0, control, n_steps=n_steps)
        out["floor_study"].append(r)
        print(f"      {n_steps:>8d} {problem.T / n_steps:>10.2e} {r['abs_difference']:>14.3e} "
              f"{r['rel_difference']:>12.3e}")
    st = out["floor_study"]
    if st and min(r["abs_difference"] for r in st) < 1e-12:
        print("      the residual is at the floating-point floor of differencing two O(1) "
              "objectives; no rate is claimed from it")
    elif len(st) >= 2 and st[0]["abs_difference"] > 0 and st[-1]["abs_difference"] > 0:
        rate = (np.log(st[-1]["abs_difference"]) - np.log(st[0]["abs_difference"])) / \
               (np.log(st[-1]["n_steps"]) - np.log(st[0]["n_steps"]))
        out["floor_order_in_dt"] = float(rate)
        print(f"      residual falls as dt^{rate:.1f}")

    # ---- reference-density study: the residual must fall as the reference grid is refined ----
    print()
    print("=" * 100)
    print("  A''. how dense must the reference's P(t) be? (the residual is the verifier's)")
    print("=" * 100)
    problem = ACProblem(p=2)
    problem.Sigma = np.zeros_like(problem.Sigma)
    x0 = np.zeros(problem.state_dim)
    x0[: problem.n_assets] = 1.0
    print(f"      {'n_ref':>9s} {'dt_ref':>10s} {'abs diff':>14s} {'rel diff':>12s}")
    for n_ref in (401, 4001, 40001, 200001):
        ref_d = problem.reference(np.linspace(0.0, problem.T, n_ref))

        def control_star_d(X, t, _ref=ref_d):
            return _ref.control(np.atleast_2d(X), float(t))

        def control_d(X, t, _cs=control_star_d, _T=problem.T):
            return _cs(X, t) + 0.2 * np.sin(np.pi * float(t) / _T)

        r = compare(problem, ref_d, x0, control_d)
        r["n_ref"] = n_ref
        out["reference_study"].append(r)
        print(f"      {n_ref:>9d} {problem.T / (n_ref - 1):>10.2e} {r['abs_difference']:>14.3e} "
              f"{r['rel_difference']:>12.3e}")
    rs = out["reference_study"]
    if len(rs) >= 2:
        rate = (np.log(rs[-1]["abs_difference"]) - np.log(rs[0]["abs_difference"])) / \
               (np.log(rs[-1]["n_ref"]) - np.log(rs[0]["n_ref"]))
        out["reference_residual_order"] = float(-rate)
        print(f"      residual falls as n_ref^-{-rate:.2f} -- i.e. the verifier's interpolation "
              f"error, not the identity")

    # ---- verdict -------------------------------------------------------------------------
    smooth_big = max(out["smooth"], key=lambda r: r["eps"])
    recon = out["cases"]
    max_rel_recon = max(r["rel_difference"] for r in recon)
    passes = smooth_big["rel_difference"] < 1e-5 and max_rel_recon < 1e-3 and \
        all(r["bound_satisfied"] for r in recon)
    out["verdict"] = {
        "smooth_case_rel_difference": smooth_big["rel_difference"],
        "reconstruction_max_rel_difference": max_rel_recon,
        "position_bound_satisfied_everywhere": all(r["bound_satisfied"] for r in recon),
        "pass": bool(passes),
    }
    print()
    print("=" * 100)
    print(f"  VERDICT: {'PASS' if passes else 'FAIL'}")
    print(f"    smooth perturbation (eps=0.2): relative difference "
          f"{smooth_big['rel_difference']:.2e}  (floor-limited)")
    print(f"    reconstruction controls      : max relative difference {max_rel_recon:.2e}")
    print(f"    position-error bound T|B|sup|du| satisfied in every case: "
          f"{out['verdict']['position_bound_satisfied_everywhere']}")
    print("=" * 100)
    (RESULTS / "check_suboptimality_identity.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8")
    print("  wrote results/check_suboptimality_identity.json")
    return 0 if passes else 1


if __name__ == "__main__":
    raise SystemExit(main())
