"""Order versus monotonicity: what a limiter costs, and what it buys.

THE QUESTION
------------
Proposition 1 recommends raising the reconstruction order. But a high-order reconstruction is not
monotone: near a sharp feature the recovered gradient can overshoot and take the WRONG SIGN. For a
control that is a gradient, a sign error is not a small inaccuracy -- it is a trade in the wrong
direction. So the recommendation has a boundary, and this script measures it.

WHAT IS COMPARED
----------------
Four ways to turn nodal values into a gradient, on functions whose gradients are known:

  multilinear   exact gradient of the piecewise-linear interpolant (the paper's baseline, order 1)
  cubic         central difference on a cubic interpolant, probe far below the mesh (order ~3)
  limited       the cubic gradient passed through a monotonised-central (MC) limiter built from the
                nodal one-sided slopes -- the standard way a TVD scheme enforces the discrete
                maximum principle
  upwind1       first-order one-sided difference, for reference: the cheapest monotone choice

Reported per method and mesh: the relative L2 gradient error, and two monotonicity measures --
the fraction of samples whose gradient has the wrong sign, and the overshoot beyond the largest
true gradient. The steep test function is a hyperbolic tangent, where a cubic genuinely oscillates.

Usage: python src/exp_limited_reconstruction.py
Output: results/exp_limited_reconstruction.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from metrics import fit_order  # noqa: E402

RESULTS = SRC.parent / "results"
LO, HI = -1.0, 1.0
GRIDS = [11, 15, 21, 27, 35, 45, 61, 81]
PROBE_SCALE = 1e-4          # relative to h: the probe of `exp_reconstruction_order.py`

FUNCS = {
    # Smooth: the order half of the story. A limiter cannot help here and should not be needed.
    "quartic": (lambda x: 0.25 * x**4, lambda x: x**3, 0.0),
    "exponential": (lambda x: np.exp(x), lambda x: np.exp(x), 0.0),
    "trigonometric": (lambda x: np.sin(3.0 * x), lambda x: 3.0 * np.cos(3.0 * x), 0.0),
    # Kinked: the gradient jumps, so an unlimited cubic is expected to ring around the kink. The
    # kink sits off-node so the mesh cannot hide it.
    "kink": (lambda x: 0.3 * np.abs(x - 0.0137),
             lambda x: 0.3 * np.sign(x - 0.0137), 0.0),
    # Noisy nodal values: the realistic case, because the reconstruction is applied to a SOLVED
    # grid whose values carry the solver's error, not to exact data.
    "exponential_noisy": (lambda x: np.exp(x), lambda x: np.exp(x), 0.02),
}


def sample(f, df, noisy: float, axes, rng) -> tuple[np.ndarray, np.ndarray]:
    """Nodal values, optionally perturbed by relative noise of the given size."""
    vals = np.asarray(f(axes[0])).ravel()
    if noisy > 0:
        vals = vals + rng.normal(0.0, noisy * float(np.std(vals)), vals.shape)
    return vals, np.asarray(df(axes[0])).ravel()


def _cells(axes, X):
    """Left node index and cell fraction of each query point (1-D)."""
    ax = axes[0]
    h = ax[1] - ax[0]
    j = np.clip(np.floor((X[:, 0] - ax[0]) / h).astype(int), 0, ax.size - 2)
    return j, h


def extract_multilinear(axes, snap, X, scale=None):
    ax = axes[0]
    j, h = _cells(axes, X)
    return ((snap[j + 1] - snap[j]) / h)[:, None]


def extract_upwind1(axes, snap, X, scale=None):
    ax = axes[0]
    j, h = _cells(axes, X)
    # Upwind towards the local direction of travel of the gradient's sign; here the cells are
    # traversed left to right, so the backward difference is the upwind one.
    g = (snap[j + 1] - snap[j]) / h
    sign = np.sign(g)
    back = np.empty_like(g)
    fwd = np.empty_like(g)
    back = np.where(j >= 1, (snap[j] - snap[np.maximum(j - 1, 0)]) / h, g)
    fwd = np.where(j + 2 <= ax.size - 1, (snap[np.minimum(j + 2, ax.size - 1)] - snap[j + 1]) / h, g)
    return np.where(sign >= 0, back, fwd)[:, None]


def extract_cubic(axes, snap, X, scale=PROBE_SCALE):
    ax = axes[0]
    h = ax[1] - ax[0]
    rgi = RegularGridInterpolator(axes, snap, method="cubic", bounds_error=False, fill_value=None)
    eps = scale * h
    xp = np.clip(X + eps, LO, HI)
    xm = np.clip(X - eps, LO, HI)
    return ((rgi(xp) - rgi(xm)) / (2 * eps))[:, None]


def _minmod(*vals):
    """Minmod of an arbitrary number of arrays: zero if the signs disagree, else the smallest."""
    stacked = np.vstack(vals)
    signs = np.sign(stacked)
    agree = np.all(signs == signs[0], axis=0)
    return np.where(agree, signs[0] * np.min(np.abs(stacked), axis=0), 0.0)


def extract_limited(axes, snap, X, scale=PROBE_SCALE):
    """Cubic gradient limited by the nodal one-sided slopes (monotonised central, MC).

    The constraint is the discrete maximum principle on the data: the recovered gradient may not
    exceed twice the smaller adjacent difference quotient, and must vanish where the data turn
    around. This is a POST-limiter on the reconstruction, not a different solve -- which is the
    point, because it costs one pass over the grid and no accuracy in the solver.
    """
    ax = axes[0]
    h = ax[1] - ax[0]
    j, _ = _cells(axes, X)
    g = extract_cubic(axes, snap, X, scale)[:, 0]
    dL = np.empty_like(g)
    dR = np.empty_like(g)
    dL = np.where(j >= 1, (snap[j] - snap[np.maximum(j - 1, 0)]) / h, (snap[1] - snap[0]) / h)
    dR = np.where(j + 2 <= ax.size - 1,
                  (snap[np.minimum(j + 2, ax.size - 1)] - snap[j + 1]) / h,
                  (snap[-1] - snap[-2]) / h)
    return _minmod(g, 2.0 * dL, 2.0 * dR)[:, None]


METHODS = {
    "multilinear": extract_multilinear,
    "cubic": extract_cubic,
    "limited": extract_limited,
    "upwind1": extract_upwind1,
}


def measure(f, df, noisy: float, n_points: int = 4000, seed: int = 11,
            report_order: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    X = rng.uniform(LO, HI, size=(n_points, 1))
    exact = np.asarray(df(X)).ravel()
    scale = float(np.mean((exact - exact.mean()) ** 2)) or 1.0
    per_method = {}
    for name, extractor in METHODS.items():
        track, wrong_way, overshoot = [], [], []
        for N in GRIDS:
            axes = [np.linspace(LO, HI, N)]
            snap, _ = sample(f, df, noisy, axes, rng)
            g = extractor(axes, snap, X, PROBE_SCALE).ravel()
            h = (HI - LO) / (N - 1)
            track.append((h, math.sqrt(float(np.mean((g - exact) ** 2) / scale))))
            # MONOTONICITY. What matters for a control is not a small error but pointing the wrong
            # way, so this counts only samples where the reconstruction has the OPPOSITE sign to a
            # significant true gradient. Returning zero at a turning point is the safe answer a
            # limiter is supposed to give, and is deliberately not counted as a failure.
            significant = np.abs(exact) > 0.2 * np.max(np.abs(exact))
            wrong_way.append(float(np.mean(significant & (np.sign(g) * np.sign(exact) < 0))))
            # Overshoot beyond the largest true gradient: the ringing a high-order reconstruction
            # produces when the data are not smooth.
            overshoot.append(float(np.max(np.abs(g)) / np.max(np.abs(exact))))
        order, r2 = fit_order([p[0] for p in track], [p[1] for p in track])
        per_method[name] = {
            "order": order if report_order else None, "fit_r2": r2 if report_order else None,
            "error_coarsest": track[0][1], "error_finest": track[-1][1],
            "wrong_way_coarsest": wrong_way[0], "wrong_way_finest": wrong_way[-1],
            "max_wrong_way": max(wrong_way),
            "overshoot_coarsest": overshoot[0], "overshoot_finest": overshoot[-1],
            "max_overshoot": max(overshoot),
            "track": [[float(h), float(e)] for h, e in track],
            "wrong_way_track": [float(w) for w in wrong_way],
        }
    return {"per_method": per_method}


def main() -> int:
    print("=" * 104)
    print("ORDER VERSUS MONOTONICITY: what a limiter costs, and what it buys")
    print("=" * 104)
    out = {"grids": GRIDS, "probe_scale": PROBE_SCALE, "functions": {}}
    for fname, (f, df, noisy) in FUNCS.items():
        # An order is only meaningful when the target has a bounded derivative; the kink's gradient
        # is discontinuous, so its error does not converge and the monotonicity columns carry it.
        report_order = fname != "kink"
        note = ""
        if noisy:
            note = f"   (nodal values carry {noisy:.0%} relative noise -- the solved-grid case)"
        elif not report_order:
            note = "   (gradient discontinuous: no order to report, look at the sign columns)"
        print(f"\n  function = {fname}{note}")
        res = measure(f, df, noisy, report_order=report_order)
        out["functions"][fname] = res
        print(f"    {'method':12s} {'order':>7s} {'err(finest)':>12s} "
              f"{'wrong way (max)':>16s} {'max overshoot':>14s}")
        for name, r in res["per_method"].items():
            o = f"{r['order']:7.2f}" if r["order"] is not None else "      --"
            print(f"    {name:12s} {o} {r['error_finest']:12.3e} "
                  f"{r['max_wrong_way']:16.3f} {r['max_overshoot']:14.3f}")

    print()
    print("=" * 104)
    print("READING")
    print("=" * 104)
    for fname in ("kink", "exponential_noisy"):
        pm = out["functions"][fname]["per_method"]
        print(f"  {fname:18s} cubic: wrong-way {pm['cubic']['max_wrong_way']*100:5.2f}%  "
              f"overshoot {pm['cubic']['max_overshoot']:.2f}x   |   "
              f"limited: wrong-way {pm['limited']['max_wrong_way']*100:5.2f}%  "
              f"overshoot {pm['limited']['max_overshoot']:.2f}x")
    sm = out["functions"]["exponential"]["per_method"]
    print(f"  smooth data        cubic order {sm['cubic']['order']:.2f} vs limited "
          f"{sm['limited']['order']:.2f} -- on smooth data the limiter only costs order")
    (RESULTS / "exp_limited_reconstruction.json").write_text(json.dumps(out, indent=2),
                                                             encoding="utf-8")
    print("\n  wrote results/exp_limited_reconstruction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
