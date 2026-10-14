"""
Is the O(h^2) control error intrinsic to the grid solution, or an artefact of how the control is
extracted from it?

This is the experiment that decides how the paper's central claim may be stated, so it is built
to separate two things that the earlier sweeps confounded:

  * the error in the grid SOLUTION (the nodal values of V), and
  * the error introduced when a CONTROL is extracted from those values.

The clean way to separate them is to remove the first entirely. `to_exact_nodes` puts the
closed-form value function on the grid as exact nodal values, so the discrete solution is perfect
by construction and every remaining error belongs to the extraction step. Sweeping N then gives
the extraction operator's convergence order with no solver error mixed in.

Three extraction methods are compared:

  fd      -- central difference with step ~ h taken on a MULTILINEAR interpolant. This is what
             the benchmark currently does. A multilinear interpolant is piecewise linear, so its
             gradient is piecewise constant (a stair-step) and the difference straddles cells;
             the earlier probe study showed this choice alone costs a factor of 40-74.
  cell    -- the EXACT gradient of the same multilinear interpolant, i.e. the cell slope. No
             probe step, so no probe error: this is the best a multilinear reconstruction can do
             and is what a careful implementation would use.
  spline  -- central difference on a CUBIC interpolant, which is C^2, so the same probe step now
             measures a smooth gradient. This tests whether a higher-order reconstruction
             recovers control accuracy.

If `spline` converges at a higher order and with a much smaller constant than `fd`, then the
decoupling reported in the paper is largely a property of the extraction step rather than of grid
methods, and the recommendation changes from "grid control accuracy is expensive" to "extract the
gradient with a reconstruction of adequate order". If the three agree, the decoupling is
intrinsic.

Output: results/exp_control_extraction.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator

from ac_problem import ACProblem
from solvers import FiniteDifferenceSolver


# ---------------------------------------------------------------------------------------
# Extraction methods. Each takes the grid and returns the control at query points X.
# ---------------------------------------------------------------------------------------
def extract_fd(axes, snap, X, scale):
    """Current method: central difference with step = scale * dx on a multilinear interpolant."""
    dx = np.array([a[1] - a[0] for a in axes])
    rgi = RegularGridInterpolator(axes, snap, method="linear", bounds_error=False,
                                  fill_value=None)
    eps = scale * dx
    p = X.shape[1]
    g = np.zeros_like(X)
    for i in range(p):
        Xp = X.copy(); Xp[:, i] += eps[i]
        Xm = X.copy(); Xm[:, i] -= eps[i]
        g[:, i] = (rgi(Xp) - rgi(Xm)) / (2 * eps[i])
    return g


def extract_cell(axes, snap, X, scale=None):
    """Exact gradient of the multilinear interpolant: the slope of the containing cell.

    No probe step, so this is the best a multilinear reconstruction can achieve, and it is what
    `extract_fd` is approximating. Within a cell the multilinear interpolant is the plane fit to
    that cell's 2^p corners, so its directional slope in direction i is exactly the forward
    difference of the nodal values across that cell.
    """
    p = X.shape[1]
    n = [len(a) for a in axes]
    dx = np.array([a[1] - a[0] for a in axes])
    # Lower corner of the containing cell, per dimension, clipped to the valid interior range.
    idx = [np.clip(np.floor((X[:, i] - axes[i][0]) / dx[i]).astype(int), 0, n[i] - 2)
           for i in range(p)]
    g = np.zeros_like(X)
    for i in range(p):
        k_lo = tuple(idx)
        k_hi = tuple(idx[j] + (1 if j == i else 0) for j in range(p))
        g[:, i] = (snap[k_hi] - snap[k_lo]) / dx[i]
    return g


def extract_spline(axes, snap, X, scale):
    """Central difference on a cubic (C^2) interpolant, so the probe measures a smooth gradient."""
    dx = np.array([a[1] - a[0] for a in axes])
    rgi = RegularGridInterpolator(axes, snap, method="cubic", bounds_error=False,
                                  fill_value=None)
    eps = scale * dx
    p = X.shape[1]
    g = np.zeros_like(X)
    for i in range(p):
        Xp = X.copy(); Xp[:, i] += eps[i]
        Xm = X.copy(); Xm[:, i] -= eps[i]
        g[:, i] = (rgi(Xp) - rgi(Xm)) / (2 * eps[i])
    return g


def to_exact_nodes(prob, axes, ref, t: float) -> np.ndarray:
    """The closed-form value function sampled exactly at grid nodes, at time `t`.

    Using this instead of a solved array removes the solver error entirely, so any remaining
    control error is the extraction operator's alone.

    `t` must match the time at which the control is compared. An earlier version hard-coded
    t=0 while the queries were at t=T/2, which produced a large error that was IDENTICAL for
    every N and every extractor -- the giveaway, since an interpolation error cannot be both
    large and independent of the mesh.
    """
    mesh = np.meshgrid(*axes, indexing="ij")
    X = np.stack([m.ravel() for m in mesh], axis=1)
    V = ref.value(X, t)
    return V.reshape(tuple(len(a) for a in axes))


EXTRACTORS = {"fd": extract_fd, "cell": extract_cell, "spline": extract_spline}


# ---------------------------------------------------------------------------------------
# Part C: what order does each reconstruction achieve on a general smooth function?
#
# Part A/B use the benchmark's own value function, which is QUADRATIC, and a cubic spline
# reproduces quadratics exactly -- so Part A's machine-precision result is a best case and must
# not be presented as the general behaviour. To get the order that applies to a general HJB, the
# same three extractors are run on functions of known, non-polynomial form.
# ---------------------------------------------------------------------------------------
SYNTHETIC = {
    "quadratic": (lambda X: -0.5 * np.sum(X**2, axis=1),
                  lambda X: -X),
    "quartic": (lambda X: 0.25 * np.sum(X**4, axis=1),
                lambda X: X**3),
    "exponential": (lambda X: np.sum(np.exp(X), axis=1),
                    lambda X: np.exp(X)),
    "trigonometric": (lambda X: np.sum(np.sin(3.0 * X), axis=1),
                      lambda X: 3.0 * np.cos(3.0 * X)),
}


def run_synthetic(p: int, grids: list[int], methods: list[str], scale: float,
                  n_points: int, seed: int) -> list[dict]:
    """Gradient-extraction error and order for each extractor on each test function."""
    rng = np.random.default_rng(seed)
    lo, hi = -1.0, 1.0
    X = rng.uniform(lo, hi, size=(n_points, p))
    rows = []
    print("\n" + "=" * 100)
    print("Part C -- gradient extraction on functions of known form (the general case)")
    print("=" * 100)
    for fname, (f, df) in SYNTHETIC.items():
        exact = df(X)
        scale_f = float(np.mean((exact - exact.mean()) ** 2)) or 1.0
        print(f"  {fname}:")
        tracks = {m: [] for m in methods}
        for N in grids:
            axes = [np.linspace(lo, hi, N)] * p
            mesh = np.meshgrid(*axes, indexing="ij")
            Xn = np.stack([m.ravel() for m in mesh], axis=1)
            snap = f(Xn).reshape(tuple([N] * p))
            h = (hi - lo) / (N - 1)
            line = f"    N={N:>3d} h={h:.4f} "
            for m in methods:
                try:
                    g = EXTRACTORS[m](axes, snap, X, scale)
                    err = float(np.mean((g - exact) ** 2) / scale_f)
                    tracks[m].append((h, err))
                    line += f"{m}={err:.2e} "
                except Exception as e:
                    line += f"{m}=FAIL "
            print(line, flush=True)
        orders = {}
        for m in methods:
            pts = [(h, e) for h, e in tracks[m] if e and e > 1e-16]
            if len(pts) < 3:
                orders[m] = None
                continue
            hh = np.log([h for h, _ in pts]); ee = np.log([e for _, e in pts])
            A = np.vstack([hh, np.ones_like(hh)]).T
            slope, _ = np.linalg.lstsq(A, ee, rcond=None)[0]
            pred = A @ np.array([slope, _])
            r2 = 1 - ((ee - pred) ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
            orders[m] = {"order": float(slope), "fit_r2": float(r2)}
        print("    orders: " + ", ".join(
            f"{m}={orders[m]['order']:.2f}" if orders[m] else f"{m}=n/a"
            for m in methods))
        rows.append({"function": fname, "orders": orders,
                     "tracks": {m: [[float(a), float(b)] for a, b in tracks[m]]
                                for m in methods}})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=2)
    ap.add_argument("--grids", default="11,15,21,27,35,45,61")
    ap.add_argument("--methods", default="fd,cell,spline")
    ap.add_argument("--probe-scale", type=float, default=0.25)
    ap.add_argument("--t-ref", type=float, default=0.5)
    ap.add_argument("--n-points", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--also-pde", action="store_true",
                    help="also test extraction from a SOLVED grid, not just exact nodes")
    ap.add_argument("--synthetic", action="store_true",
                    help="also measure each extractor's order on functions of known form")
    ap.add_argument("--out", default="results/exp_control_extraction.json")
    args = ap.parse_args()

    prob = ACProblem(p=args.p, hetero=True, het_seed=1)
    ref = prob.reference(np.linspace(0.0, prob.T, 64))
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    rng = np.random.default_rng(args.seed)
    lo, hi = prob.domain()
    # Query points must lie INSIDE the interpolation domain. An earlier version of this script
    # sampled 10% beyond it, so both interpolators were extrapolating, the errors were ~6e-2 and
    # flat in N (order 0.00), and nothing could be concluded. The grid axes are padded by the
    # solver's domain_pad; the queries are drawn from the unpadded problem domain.
    X = rng.uniform(lo, hi, size=(args.n_points, prob.p))
    t = args.t_ref
    u_ref = ref.control(X, t)
    u_scale = float(np.mean((u_ref - u_ref.mean()) ** 2)) or 1.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "p": args.p,
            "probe_scale": args.probe_scale,
            "t_query": t,
            "n_query_points": args.n_points,
            "purpose": ("separate the control-EXTRACTION error from the grid-SOLUTION error by "
                        "putting the exact closed-form value function on the grid as nodal "
                        "values, so the discrete solution is error-free and everything left "
                        "belongs to the extraction step"),
            "methods": {
                "fd": "central difference, step ~ scale*h, on a multilinear interpolant (current)",
                "cell": "exact gradient of the multilinear interpolant (cell slope)",
                "spline": "central difference on a cubic (C^2) interpolant",
            },
        },
        "exact_nodes": [],
        "solved_grid": [],
    }

    def save():
        out_path.write_text(json.dumps(results, indent=2))

    # ---------------- Part A: extraction from EXACT nodal values ------------------------
    print("=" * 100)
    print("Part A -- extraction from EXACT closed-form nodal values (solver error removed)")
    print("=" * 100)
    print(f"  {'N':>4s} {'h':>9s} " + " ".join(f"{m:>14s}" for m in methods))
    tracks = {m: [] for m in methods}
    for N in [int(g) for g in args.grids.split(",")]:
        axes = [np.linspace(lo[i] - 0.25, hi[i] + 0.25, N) for i in range(prob.p)]
        h = float(np.mean([a[1] - a[0] for a in axes]))
        snap = to_exact_nodes(prob, axes, ref, t)
        row = {"N": N, "h": h, "source": "exact nodes", "errors": {}}
        line = f"  {N:>4d} {h:>9.5f} "
        for m in methods:
            try:
                g = EXTRACTORS[m](axes, snap, X, args.probe_scale)
                u = g @ prob.R_inv.T
                err = float(np.mean((u - u_ref) ** 2) / u_scale)
                row["errors"][m] = err
                tracks[m].append((h, err))
                line += f"{err:>14.3e} "
            except Exception as e:
                row["errors"][m] = None
                line += f"{'FAIL:' + type(e).__name__:>14s} "
        print(line, flush=True)
        results["exact_nodes"].append(row)
        save()

    print("\n  convergence orders (least squares in log-log):")
    for m in methods:
        pts = [(h, e) for h, e in tracks[m] if e and e > 0]
        if len(pts) < 3:
            print(f"    {m:>8s}: too few points")
            continue
        hh = np.log(np.array([h for h, _ in pts]))
        ee = np.log(np.array([e for _, e in pts]))
        A = np.vstack([hh, np.ones_like(hh)]).T
        slope, inter = np.linalg.lstsq(A, ee, rcond=None)[0]
        pred = A @ np.array([slope, inter])
        r2 = 1 - ((ee - pred) ** 2).sum() / ((ee - ee.mean()) ** 2).sum()
        first = pts[0][1]
        last = pts[-1][1]
        print(f"    {m:>8s}: order {slope:5.2f}  (fit R^2 {r2:.3f})   "
              f"error {first:.2e} -> {last:.2e}  (factor {first/last:.0f})")
        results["meta"].setdefault("exact_node_orders", {})[m] = {
            "order": float(slope), "fit_r2": float(r2),
            "error_first": first, "error_last": last}

    # ---------------- Part B: extraction from a SOLVED grid ----------------------------
    if args.also_pde:
        print("\n" + "=" * 100)
        print("Part B -- extraction from a SOLVED grid (solver error included)")
        print("=" * 100)
        print(f"  {'N':>4s} {'h':>9s} " + " ".join(f"{m:>14s}" for m in methods))
        ptracks = {m: [] for m in methods}
        for N in [int(g) for g in args.grids.split(",")]:
            s = FiniteDifferenceSolver(prob, n_grid=N)
            s.solve()
            h = float(np.mean(s.dx))
            snap = s.snapshots[int(np.argmin(np.abs(s.t_snap - t)))]
            row = {"N": N, "h": h, "source": "solved", "errors": {}, "solve_time_s": s.solve_time}
            line = f"  {N:>4d} {h:>9.5f} "
            for m in methods:
                try:
                    g = EXTRACTORS[m](s.axes, snap, X, args.probe_scale)
                    u = g @ prob.R_inv.T
                    err = float(np.mean((u - u_ref) ** 2) / u_scale)
                    row["errors"][m] = err
                    ptracks[m].append((h, err))
                    line += f"{err:>14.3e} "
                except Exception as e:
                    row["errors"][m] = None
                    line += f"{'FAIL':>14s} "
            print(line, flush=True)
            results["solved_grid"].append(row)
            save()
        print("\n  convergence orders (least squares in log-log):")
        for m in methods:
            pts = [(h, e) for h, e in ptracks[m] if e and e > 0]
            if len(pts) < 3:
                continue
            hh = np.log(np.array([h for h, _ in pts]))
            ee = np.log(np.array([e for _, e in pts]))
            A = np.vstack([hh, np.ones_like(hh)]).T
            slope, inter = np.linalg.lstsq(A, ee, rcond=None)[0]
            print(f"    {m:>8s}: order {slope:5.2f}   "
                  f"{pts[0][1]:.2e} -> {pts[-1][1]:.2e}  (factor {pts[0][1]/pts[-1][1]:.0f})")
            results["meta"].setdefault("solved_orders", {})[m] = {"order": float(slope)}

    # ---------------- Part C: general smooth functions ---------------------------------
    if args.synthetic:
        rows = run_synthetic(args.p, [int(g) for g in args.grids.split(",")], methods,
                             args.probe_scale, args.n_points, args.seed)
        results["synthetic"] = rows
        save()

    save()
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
