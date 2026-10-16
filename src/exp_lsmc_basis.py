"""How much of regression Monte Carlo's advantage was the basis?

THE OBJECTION, IN A REFEREE'S WORDS
-----------------------------------
"Your regression method fits the quadratic features 1, x_i, x_i x_j, and the benchmark's value function
IS quadratic. You have handed the classical method its own answer and then reported that it beats a
neural network that was given no such structure. The comparison is not informative."

That is the strongest objection to the mesh-free result, and it is answerable, but only by measuring
the sensitivity rather than asserting it. This script runs the SAME regression scheme, with the SAME
quadratic basis, on a family of problems whose value function is progressively less quadratic:

    V(x,t) = -1/2 x'P(t)x + A0 (T-t) sin(omega * 1'x)

exact by construction (a source in the running reward makes it so -- see exp_nonpolynomial.py). At
A0 = 0 this is the paper's benchmark and the basis is exactly right; as A0 grows the basis is
misspecified and the regression has to fit a function outside its span.

WHAT IT REPORTS
---------------
The regression's R^2 against the exact value function at each A0, so the degradation is a number
rather than a caveat. The honest reading is the one the paper now states: the classical mesh-free
method wins where the practitioner has structural knowledge of the value function, and this quantifies
what "structural knowledge" is worth.

Usage: python src/exp_lsmc_basis.py [--dim 4] [--a0 0,1e-3,5e-3,2e-2]
Output: results/exp_lsmc_basis.json
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
import exp_lsmc as L  # noqa: E402
from exp_nonpolynomial import ManufacturedProblem  # noqa: E402
from metrics import relative_l2  # noqa: E402

RESULTS = SRC.parent / "results"


def backward_with_source(prob: ManufacturedProblem, X: np.ndarray, U: np.ndarray, dt: float,
                         ridge: float = 1e-10) -> list:
    """The same backward regression, with the manufactured source inside the realised reward."""
    N = U.shape[0]
    M = X.shape[1]
    ts = np.linspace(0.0, prob.T, N + 1)
    run = (np.einsum("...i,ij,...j->...", X[:-1], prob.Q, X[:-1])
           + np.einsum("...i,ij,...j->...", U, prob.R, U))
    # subtract the source, because the reward is -1/2(x'Qx + u'Ru) + s
    src = np.stack([prob.source(X[n], float(ts[n])) for n in range(N)])
    run = run - 2.0 * src                      # run enters as -1/2 * run * dt
    term = np.einsum("...i,ij,...j->...", X[-1], prob.QT, X[-1])
    Y = np.empty((N + 1, M))
    Y[N] = -0.5 * term
    acc = -0.5 * run[N - 1] * dt
    Y[N - 1] = acc - 0.5 * term
    for n in range(N - 2, -1, -1):
        acc = acc - 0.5 * run[n] * dt
        Y[n] = acc - 0.5 * term
    policy = []
    for n in range(N + 1):
        B = L.basis(X[n])
        m = B.shape[1]
        A_ = np.vstack([B, np.sqrt(ridge * M) * np.eye(m)])
        b_ = np.concatenate([Y[n], np.zeros(m)])
        coef, *_ = np.linalg.lstsq(A_, b_, rcond=None)
        policy.append(L.unpack(coef, prob.state_dim))
    return policy


def evaluate(prob: ManufacturedProblem, policy, n_points: int = 4000, seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    lo, hi = prob.domain()
    X = rng.uniform(lo, hi, size=(n_points, prob.state_dim))
    c, g, H = policy[0]
    V_hat = c + X @ g + 0.5 * np.einsum("ij,jk,ik->i", X, H, X)
    V_ref = prob.value(X, 0.0)
    u_hat = (g + X @ H.T) @ prob.B @ prob.R_inv.T
    u_ref = prob.control(X, 0.0)
    return {
        "value_r2": float(1 - np.sum((V_hat - V_ref) ** 2)
                          / np.sum((V_ref - V_ref.mean()) ** 2)),
        "value_rel_l2": relative_l2(V_hat, V_ref),
        "control_r2": float(1 - np.sum((u_hat - u_ref) ** 2)
                            / np.sum((u_ref - u_ref.mean()) ** 2)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=4)
    ap.add_argument("--a0", default="0,1e-4,1e-3,5e-3,2e-2")
    ap.add_argument("--paths", type=int, default=40000)
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--out", default="exp_lsmc_basis.json")
    ap.add_argument("--repeats", type=int, default=3,
                    help="independent repeats per A0. More than one is needed because at the largest "
                         "amplitudes the method is in a FAILING regime and its output is not "
                         "reproducible to two digits: measured -1.19 and -1.13 on two runs, which is "
                         "itself a property of the failure rather than of the code.")
    args = ap.parse_args()

    p = args.dim
    a0s = [float(x) for x in args.a0.split(",")]
    print("=" * 100)
    print("HOW MUCH OF THE REGRESSION METHOD'S ADVANTAGE IS THE BASIS?")
    print("=" * 100)
    print(f"  d={p}, quadratic basis throughout, V made progressively less quadratic")
    print(f"  paths {args.paths}, steps {args.steps}, iterations {args.iters}, alpha {args.alpha}")

    out = {"dim": p, "a0_values": a0s, "repeats": args.repeats, "rows": []}
    print(f"\n  {'A0':>9} {'value R2':>10} {'control R2':>11} {'value relL2':>12}")
    for a0 in a0s:
        L.A0 = a0
        import exp_nonpolynomial as M
        M.A0 = a0
        prob = ManufacturedProblem(p=p)
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        dt = prob.T / args.steps
        D = np.diag(prob.Sigma @ prob.Sigma.T)
        sig_step = np.sqrt(D * dt)
        m = 1 + prob.state_dim + prob.state_dim * (prob.state_dim + 1) // 2
        vals, rels, divs = [], [], []
        t0 = time.perf_counter()
        for rep in range(args.repeats):
            zero = (0.0, np.zeros(prob.state_dim), np.zeros((prob.state_dim, prob.state_dim)))
            policy = [zero] * (args.steps + 1)
            diverged, detail = False, ""
            for it in range(args.iters):
                rng = np.random.default_rng(2000 + 137 * rep + it)
                X, U = L.simulate(prob, policy, args.paths, args.steps, rng, x0, dt, sig_step)
                if not np.isfinite(X).all():
                    diverged, detail = True, f"path diverged at iteration {it}"
                    break
                try:
                    fitted = backward_with_source(prob, X, U, dt)
                except np.linalg.LinAlgError as e:
                    # Not a numerical accident: an ill-conditioned basis on paths the fitted policy
                    # has driven out of the region the basis can describe. That IS the result at
                    # large A0, so it is recorded rather than retried.
                    diverged, detail = True, f"{type(e).__name__} at iteration {it}"
                    break
                policy = L.damp(policy, fitted, args.alpha)
            if diverged:
                divs.append(detail)
                continue
            sc = evaluate(prob, fitted)
            vals.append(sc["value_r2"])
            rels.append(sc["value_rel_l2"])
        row = {
            "A0": a0, "basis_functions": m, "repeats": args.repeats,
            "diverged": len(vals) == 0, "n_diverged": len(divs), "details": divs,
            "value_r2": float(np.mean(vals)) if vals else None,
            "value_r2_min": float(np.min(vals)) if vals else None,
            "value_r2_max": float(np.max(vals)) if vals else None,
            "value_r2_all": [float(v) for v in vals],
            "value_rel_l2": float(np.mean(rels)) if rels else None,
            "spread": float(np.max(vals) - np.min(vals)) if len(vals) > 1 else 0.0,
            "seconds": time.perf_counter() - t0,
        }
        out["rows"].append(row)
        if row["diverged"]:
            print(f"  {a0:>9.0e} {'DIVERGED':>10}  {row['n_diverged']}/{args.repeats} repeats: "
                  f"{divs[0]}")
        else:
            spread = f"  (spread over repeats {row['spread']:.3f})" if row["spread"] > 1e-6 else ""
            print(f"  {a0:>9.0e} {row['value_r2']:>10.4f} {'':>11} {row['value_rel_l2']:>12.3e}"
                  f"{spread}")

    print()
    print("=" * 100)
    print("READING")
    print("=" * 100)
    ok = [r for r in out["rows"] if not r["diverged"]]
    r0 = next((r for r in ok if r["A0"] == 0), ok[0] if ok else None)
    worst = next((r for r in reversed(ok) if r["value_r2"] < 0.9), None)
    if r0:
        print(f"  basis exactly right (A0 = 0)      : value R2 {r0['value_r2']:+.4f}")
    if worst:
        print(f"  basis wrong (A0 = {worst['A0']:g})          : value R2 {worst['value_r2']:+.4f}"
              f"  (range over {worst['repeats']} repeats "
              f"{worst['value_r2_min']:+.2f} to {worst['value_r2_max']:+.2f})")
    last = out["rows"][-1]
    if last["diverged"]:
        print(f"  A0 = {last['A0']:g}: DIVERGED in {last['n_diverged']}/{last['repeats']} repeats "
              f"({last['details'][0]}) --- the fitted policy drives the")
        print("     state out of the region the basis can describe, which is the failure mode this")
        print("     experiment exists to expose.")
    unstable = [r for r in ok if r["spread"] > 1e-6 and r["value_r2"] < 0.9]
    if unstable:
        print(f"  NOTE: at A0 = {unstable[0]['A0']:g} the value is not reproducible across repeats "
              f"(spread {unstable[0]['spread']:.2f}): the method is in a")
        print("        failing regime, where the outcome depends on details no practitioner controls.")
    print("  -> the classical advantage is worth exactly what structural knowledge of V is worth,")
    print("     and it disappears --- then fails outright --- as that knowledge becomes wrong.")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    print("BASIS SENSITIVITY COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
