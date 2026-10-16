"""Does the probe step cap the order a reconstruction can deliver, and at what scaling?

WHY THIS EXPERIMENT EXISTS
--------------------------
Two of this paper's experiments reported different orders for the SAME cubic reconstruction:
`exp_reconstruction_order.py` measured ~3.4 (probe four orders of magnitude below the mesh), while
`exp_control_extraction.py` measured ~2.4 (probe at h/4). Both were right, and the reason is
Proposition 1: differentiating an interpolant by a symmetric difference at step eps costs an extra
O(eps^2) truncation term, so the total error is

    || grad_h V - grad V ||  =  O(h^k)  +  O(eps^2)

for a degree-k reconstruction. With eps = c * h^alpha the two terms are O(h^k) and O(h^{2 alpha}),
so the measured order is min(k, 2 alpha):

    alpha = 1     -> min(k, 2)      = 2 for a cubic reconstruction
    alpha = 1.5   -> min(k, 3)      = 3
    alpha = 2     -> min(k, 4)      = 3 (still capped by the reconstruction)

This script measures the order at four probe scalings and compares them with that prediction. It is a
falsifiable test of the proposition rather than an illustration of it: if the probe step were
innocent, every row would read ~3.4.

Usage: python src/exp_probe_scaling.py
Output: results/exp_probe_scaling.json
"""
from __future__ import annotations

import json
import math
import pathlib
import sys
import time

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from metrics import fit_order  # noqa: E402
from exp_control_extraction import EXTRACTORS, SYNTHETIC  # noqa: E402

RESULTS = SRC.parent / "results"

# Probe scalings to test. `alpha` is the exponent in eps = c * h^alpha; `c` is chosen so that the
# coarsest-mesh probe stays inside a cell for alpha = 1 (the paper's own configuration).
ALPHAS = [1.0, 1.25, 1.5, 2.0]
C = 0.25
GRIDS = [11, 15, 21, 27, 35, 45, 61]
METHODS = ["fd", "spline"]
FUNCTIONS = ["quartic", "exponential", "trigonometric"]


def order_of(track: list[tuple[float, float]]) -> tuple[float, float]:
    """Order from the longest ascending prefix of the error track (plateau points dropped).

    A high-degree reconstruction on a fine mesh can reach the floating-point floor, and a fit that
    includes the plateau reports the rounding rather than the method -- the trap that once produced a
    spurious order of -2.1 in this project.
    """
    pts = [(h, e) for h, e in track if e and e > 0 and math.isfinite(e)]
    if len(pts) < 3:
        return float("nan"), float("nan")
    floor = 1e-12 * pts[0][1]
    keep = [pts[0]]
    for h, e in pts[1:]:
        if e < floor:
            break
        if e > keep[-1][1] * 0.98:      # stopped falling -> plateau
            break
        keep.append((h, e))
    if len(keep) < 3:
        keep = pts[:3]
    return fit_order([p[0] for p in keep], [p[1] for p in keep])


def main() -> int:
    print("=" * 100)
    print("PROBE SCALING: does eps = c*h^alpha cap the gradient order at min(k, 2*alpha)?")
    print("=" * 100)
    print(f"  probe: eps = {C} * h^alpha      rms error, relative L2 scale (sqrt of the stored MSE)")

    out = {"probe_c": C, "alphas": ALPHAS, "grids": GRIDS, "rows": []}
    for fname in FUNCTIONS:
        f, df = SYNTHETIC[fname]
        print(f"\n  function = {fname}")
        for method in METHODS:
            if method == "spline" and fname == "quadratic":
                continue
            line = f"    {method:7s}"
            for alpha in ALPHAS:
                track = []
                for N in GRIDS:
                    h = 2.0 / (N - 1)
                    axes = [np.linspace(-1.0, 1.0, N)] * 2
                    mesh = np.meshgrid(*axes, indexing="ij")
                    Xn = np.stack([m.ravel() for m in mesh], axis=1)
                    snap = f(Xn).reshape(N, N)
                    rng = np.random.default_rng(5)
                    X = rng.uniform(-1.0, 1.0, size=(1200, 2))
                    exact = df(X)
                    scale_f = float(np.mean((exact - exact.mean()) ** 2)) or 1.0
                    # eps = (C * h^(alpha-1)) * dx, since the extractors multiply the scale by dx.
                    rel_scale = C * h ** (alpha - 1.0)
                    g = EXTRACTORS[method](axes, snap, X, rel_scale)
                    err = math.sqrt(float(np.mean((g - exact) ** 2) / scale_f))
                    track.append((h, err))
                order, r2 = order_of(track)
                predicted = min(3.0 if method == "spline" else 1.0, 2.0 * alpha)
                line += f"  a={alpha:<4} order={order:5.2f} (min(k,2a)={predicted:.1f})"
                out["rows"].append({"function": fname, "method": method, "alpha": alpha,
                                    "order": order, "fit_r2": r2, "predicted": predicted,
                                    "track": [[float(h), float(e)] for h, e in track]})
            print(line)

    print()
    print("=" * 100)
    print("VERDICT")
    print("=" * 100)
    ok = 0
    tot = 0
    for r in out["rows"]:
        if r["method"] != "spline" or r["order"] != r["order"]:
            continue
        tot += 1
        if abs(r["order"] - r["predicted"]) <= 0.5:
            ok += 1
    print(f"  cubic rows within 0.5 of the prediction min(k, 2*alpha): {ok}/{tot}")
    for alpha in ALPHAS:
        vals = [r["order"] for r in out["rows"]
                if r["method"] == "spline" and r["alpha"] == alpha and r["order"] == r["order"]]
        if vals:
            print(f"    alpha={alpha:<4}  cubic order {min(vals):.2f}-{max(vals):.2f}   "
                  f"predicted {min(3.0, 2 * alpha):.1f}")
    out["summary"] = {"cubic_rows_within_half": ok, "cubic_rows_total": tot}
    (RESULTS / "exp_probe_scaling.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/exp_probe_scaling.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
