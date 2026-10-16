"""How much of the grid's cost was the algorithm, and how much was NumPy?

WHY THIS EXPERIMENT EXISTS
--------------------------
Every grid timing in this paper comes from a NumPy implementation whose per-step cost is many small
array operations, and the paper says so: its achieved rate grows with N rather than saturating, which
means part of what it measures is interpreter and allocation overhead rather than arithmetic. A
referee is entitled to ask what the comparison looks like when the same scheme is compiled. This runs
the same scheme as a fused, parallel kernel (numba; the environment has no JAX) and reports:

  * a faithfulness check -- same solve, same grid, same step count, NumPy versus kernel, differ by
    how much, and does the accuracy match;
  * the implementation factor at matched work (same N, same n_steps);
  * the achieved GFLOP/s of both, against the paper's own analytic FLOP model.

The kernel does not change the scheme: central differences in the interior, second-order one-sided
stencils at the edges (the same edge_order=2 rule), the analytic Hamiltonian maximiser, the same CFL
bound. Only the memory traffic and the interpreter disappear.

Usage: python src/exp_compiled_grid.py [--dims 1,2,3,4,5] [--n-grid 21]
Output: results/exp_compiled_grid.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys
import time

import numpy as np
from numba import njit, prange

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from solvers import FiniteDifferenceSolver  # noqa: E402
from exp_cost_model import grid_flops  # noqa: E402

RESULTS = SRC.parent / "results"


# ---------------------------------------------------------------------------------------
# The kernel. Two fused parallel passes per time step: one for the gradient, one for the
# update, so the per-step temporaries that dominate the NumPy version never exist.
# ---------------------------------------------------------------------------------------
@njit(parallel=True, fastmath=True, cache=False)
def _grad_pass(V, G, shape, strides, dx, d):
    n = V.size
    for idx in prange(n):
        for i in range(d):
            c = (idx // strides[i]) % shape[i]
            s = strides[i]
            ni = shape[i]
            h = dx[i]
            if c == 0:
                g = (-3.0 * V[idx] + 4.0 * V[idx + s] - V[idx + 2 * s]) / (2.0 * h)
            elif c == ni - 1:
                g = (3.0 * V[idx] - 4.0 * V[idx - s] + V[idx - 2 * s]) / (2.0 * h)
            else:
                g = (V[idx + s] - V[idx - s]) / (2.0 * h)
            G[idx, i] = g


@njit(parallel=True, fastmath=True, cache=False)
def _update_pass(V, G, X, Q, R, R_inv, B, A, sig2, dt, shape, strides, dx, d, m):
    n = V.size
    for idx in prange(n):
        # second pass of the same derivative rule applied to G -> the Laplacian
        lap = 0.0
        for i in range(d):
            c = (idx // strides[i]) % shape[i]
            s = strides[i]
            ni = shape[i]
            h2 = dx[i] * dx[i]
            if c == 0:
                dd = (2.0 * G[idx, i] - 5.0 * G[idx + s, i] + 4.0 * G[idx + 2 * s, i]
                      - G[idx + 3 * s, i]) / h2
            elif c == ni - 1:
                dd = (2.0 * G[idx, i] - 5.0 * G[idx - s, i] + 4.0 * G[idx - 2 * s, i]
                      - G[idx - 3 * s, i]) / h2
            else:
                dd = (G[idx + s, i] - 2.0 * G[idx, i] + G[idx - s, i]) / h2
            lap += sig2[i] * dd
        # u = G B R_inv^T
        xq = 0.0
        for a in range(d):
            for b in range(d):
                xq += X[idx, a] * Q[a, b] * X[idx, b]
        u = np.zeros(m)
        for j in range(m):
            acc = 0.0
            for i in range(d):
                for k in range(m):
                    acc += G[idx, i] * B[i, k] * R_inv[j, k]
            u[j] = acc
        ur = 0.0
        for a in range(m):
            for b in range(m):
                ur += u[a] * R[a, b] * u[b]
        bu = 0.0
        for i in range(d):
            acc = 0.0
            for j in range(m):
                acc += u[j] * B[i, j]
            bu += G[idx, i] * acc
        ax = 0.0
        for i in range(d):
            acc = 0.0
            for j in range(d):
                acc += A[i, j] * X[idx, j]
            ax += G[idx, i] * acc
        V[idx] = V[idx] + dt * (-0.5 * (xq + ur) + ax + bu + 0.5 * lap)


def solve_kernel(problem: ACProblem, n_grid: int, n_steps: int, domain_pad: float = 0.25):
    """The paper's scheme, compiled: same grid, same stencil, same step count."""
    lo, hi = problem.domain()
    lo = lo - domain_pad
    hi = hi + domain_pad
    d = problem.state_dim
    shape = tuple([n_grid] * d)
    axes = [np.linspace(lo[i], hi[i], n_grid) for i in range(d)]
    dx = np.array([axes[i][1] - axes[i][0] for i in range(d)])
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    V = -0.5 * np.einsum("...i,ij,...j->...", X, problem.QT, X)
    V = np.ascontiguousarray(V.ravel())
    strides = np.array([int(np.prod(shape[i + 1:])) for i in range(d)], dtype=np.int64)
    shape = np.array(shape, dtype=np.int64)
    G = np.empty((V.size, d))
    dt = problem.T / n_steps
    sig2 = np.diag(problem.Sigma @ problem.Sigma.T).copy()
    Q = np.ascontiguousarray(problem.Q)
    R = np.ascontiguousarray(problem.R)
    R_inv = np.ascontiguousarray(problem.R_inv)
    B = np.ascontiguousarray(problem.B)
    A = np.ascontiguousarray(problem.A)
    m = problem.n_assets
    t0 = time.perf_counter()
    for _ in range(n_steps):
        _grad_pass(V, G, shape, strides, dx, d)
        _update_pass(V, G, X, Q, R, R_inv, B, A, sig2, dt, shape, strides, dx, d, m)
    secs = time.perf_counter() - t0
    return {"V": V, "X": X, "shape": tuple(int(x) for x in shape), "axes": axes,
            "seconds": secs, "n_steps": n_steps, "n_grid": n_grid}


def _stable_dt(problem: ACProblem, dx_min: float, max_u: float) -> float:
    diff = float(np.max(np.diag(problem.Sigma @ problem.Sigma.T)))
    dt_diff = 0.25 * dx_min ** 2 / max(diff, 1e-300)
    dt_adv = np.inf if max_u <= 0 else 0.25 * dx_min / max_u
    return min(dt_diff, dt_adv)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,2,3,4,5")
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument("--out", default="exp_compiled_grid.json")
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    print("=" * 104)
    print("COMPILED GRID KERNEL: what was NumPy, and what was the algorithm")
    print("=" * 104)
    print(f"  same scheme, same grid (N={args.n_grid}), same step count; numba parallel kernel")

    out = {"n_grid": args.n_grid, "rows": []}
    # Warm the JIT before any timing. Without this the first measured case includes ~1.3 s of
    # compilation and reads as infinitely slow -- which is how the first run of this script reported
    # a 0.0x speedup in one dimension and 2.6x in the next.
    warm = ACProblem(p=1)
    solve_kernel(warm, 5, 2)
    for d in dims:
        prob = ACProblem(p=d)
        # ---- reference implementation: the paper's own solver, with the step count forced ----
        probe = FiniteDifferenceSolver(prob, n_grid=args.n_grid, n_snap=3)
        lo, hi = prob.domain()
        axes_ = [np.linspace(lo[i] - 0.25, hi[i] + 0.25, args.n_grid) for i in range(d)]
        dx_min = min(a[1] - a[0] for a in axes_)
        Xg = np.stack([m.ravel() for m in np.meshgrid(*axes_, indexing="ij")], axis=1)
        u_T = (-prob.QT @ Xg.T).T @ prob.B @ prob.R_inv.T
        max_u = float(np.max(np.abs(u_T))) * 1.05
        n_steps = max(int(math.ceil(prob.T / _stable_dt(prob, dx_min, max_u))), 2)

        ref = FiniteDifferenceSolver(prob, n_grid=args.n_grid, n_snap=3, n_steps=n_steps)
        ref.solve()
        numpy_secs = float(ref.solve_time)
        V_np = ref.snapshots[0].ravel()
        ref_obj = prob.reference(np.linspace(0.0, prob.T, 401))
        lo, hi = prob.domain()
        rng = np.random.default_rng(2)
        Xq = rng.uniform(lo, hi, size=(1500, prob.state_dim))
        V_ref = np.asarray(ref_obj.value(Xq, 0.0)).ravel()
        r2_np = 1 - np.sum((np.asarray(ref.value(Xq, 0.0)).ravel() - V_ref) ** 2) \
            / np.sum((V_ref - V_ref.mean()) ** 2)

        # ---- compiled kernel ------------------------------------------------------------
        ker = solve_kernel(prob, args.n_grid, n_steps)
        V_k = ker["V"]
        n = V_k.size
        idx = np.arange(0, n, max(n // 1500, 1))[:1500]
        V_k_ref = np.asarray(ref_obj.value(ker["X"][idx], 0.0)).ravel()
        r2_k = float(1 - np.sum((V_k[idx] - V_k_ref) ** 2)
                     / np.sum((V_k_ref - V_k_ref.mean()) ** 2))
        # faithfulness: the two implementations must agree on the solved field. The NumPy
        # snapshot is at t=0 on the same grid, so the arrays are directly comparable.
        max_diff = float(np.max(np.abs(V_k - V_np))) if V_k.size == V_np.size else float("nan")
        scale = float(np.max(np.abs(V_np))) or 1.0

        flops = grid_flops(prob.state_dim, prob.n_assets, args.n_grid, n_steps)
        row = {
            "d": d, "n_grid": args.n_grid, "n_steps": n_steps,
            "numpy_seconds": numpy_secs, "kernel_seconds": ker["seconds"],
            "speedup": numpy_secs / ker["seconds"] if ker["seconds"] > 0 else float("nan"),
            "numpy_gflops": flops / numpy_secs / 1e9,
            "kernel_gflops": flops / ker["seconds"] / 1e9,
            "grid_points": int(n), "flops": flops,
            # faithfulness of the kernel: same field, and same accuracy
            "max_abs_diff_vs_numpy": max_diff,
            "rel_diff_vs_numpy": max_diff / scale,
            "r2_numpy": float(r2_np), "r2_kernel": r2_k,
        }
        out["rows"].append(row)
        print(f"  d={d}: N={args.n_grid}, {n_steps} steps, {n} points")
        print(f"        numpy {numpy_secs:8.3f}s ({row['numpy_gflops']:6.3f} GFLOP/s)   "
              f"kernel {ker['seconds']:8.3f}s ({row['kernel_gflops']:6.3f} GFLOP/s)   "
              f"-> {row['speedup']:5.1f}x")
        print(f"        kernel vs numpy field: max|dV| {max_diff:.3e} "
              f"(relative {row['rel_diff_vs_numpy']:.2e})")

    if out["rows"]:
        sp = [r["speedup"] for r in out["rows"] if r["speedup"] == r["speedup"]]
        rt = [r["kernel_gflops"] / r["numpy_gflops"] for r in out["rows"]]
        out["speedup_min"] = float(min(sp))
        out["speedup_max"] = float(max(sp))
        out["rate_ratio_min"] = float(min(rt))
        out["rate_ratio_max"] = float(max(rt))
        print()
        print("=" * 104)
        print(f"  implementation factor: {out['speedup_min']:.1f}x to {out['speedup_max']:.1f}x "
              f"faster at identical work; achieved rate up {min(rt):.1f}x to {max(rt):.1f}x")
        print("=" * 104)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
