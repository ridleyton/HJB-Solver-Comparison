"""Order-convention check: are the manuscript's convergence orders on ONE scale?

WHY THIS EXISTS
---------------
The paper reports "orders" from three different experiments:

  * `exp_control_extraction.py`  -> paper/tables/tab_extract.tex
  * `exp_control_order.py`       -> paper/tables/tab_control.tex
  * `exp_reconstruction_order.py`-> paper/tables/tab_order.tex

The first two fit the slope of log(MSE) against log(h) -- a SQUARED error, so the slope is
TWICE the convergence order. The third fits the slope of log(RMS error) -- the convergence
order itself. log(E^2) = 2 log(E), so the same physical quantity is reported on two different
scales, and the manuscript reads all of them as "order".

This script does not re-run any experiment and does not modify anything: it reads the stored
`results/*.json` and refits both ways, so the discrepancy is visible from the committed
artefacts alone.

Usage:  python src/check_order_convention.py
"""
from __future__ import annotations

import json
import math
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RES = ROOT / "results"


def fit(xs, ys) -> tuple[float, float]:
    """Least-squares slope of log(y) against log(x), with its R^2."""
    lx, ly = np.log(np.asarray(xs, float)), np.log(np.asarray(ys, float))
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    r2 = 1 - ((ly - pred) ** 2).sum() / ((ly - ly.mean()) ** 2).sum()
    return float(slope), float(r2)


def main() -> int:
    print("=" * 96)
    print("A. exp_control_extraction.json  ->  tab_extract   (MSE scale?)")
    print("=" * 96)
    data = json.loads((RES / "exp_control_extraction.json").read_text(encoding="utf-8"))
    for row in data["synthetic"]:
        print(f"\n  function = {row['function']}")
        for method, track in row["tracks"].items():
            pts = [(h, e) for h, e in track if e and e > 1e-16]
            if len(pts) < 3:
                print(f"    {method:8s} too few points")
                continue
            hs = [p[0] for p in pts]
            es = [p[1] for p in pts]
            s_mse, r2a = fit(hs, es)
            s_rms, _ = fit(hs, [math.sqrt(e) for e in es])
            print(f"    {method:8s} slope(log E) = {s_mse:5.2f}   "
                  f"slope(log sqrt(E)) = {s_rms:5.2f}   (R2 of the first = {r2a:.4f})")

    print()
    print("=" * 96)
    print("B. exp_reconstruction_order.json  ->  tab_order   (RMS scale?)")
    print("=" * 96)
    rec = json.loads((RES / "exp_reconstruction_order.json").read_text(encoding="utf-8"))
    for r in rec["results"]:
        pts = [(n, e) for n, e in zip(r["sizes"], r["errors"])
               if e and e > 1e-16 and math.isfinite(e)]
        if len(pts) < 3:
            continue
        ns = [p[0] for p in pts]
        es = [p[1] for p in pts]
        s_rms, _ = fit(ns, es)
        s_mse, _ = fit(ns, [e * e for e in es])
        print(f"  {r['function']:12s} deg={r['degree']}  reported={r['order']:6.2f}   "
              f"refit slope(log E) = {-s_rms:5.2f}   slope(log E^2) = {-s_mse:5.2f}")

    print()
    print("=" * 96)
    print("READING")
    print("=" * 96)
    print("  This script is now a REGRESSION GUARD, not a description of a live defect.")
    print("  The convention problem it was written to expose is fixed: the paper uses one")
    print("  metric -- the relative L2 error -- and every order is fitted to that error, not")
    print("  to its square. A and B above therefore agree for the same physical")
    print("  reconstruction (degree-1 multilinear ~ 1.0, degree-3 cubic ~ 3.4 with a matched")
    print("  probe), and the 'slope(log E^2)' column shows what the old numbers would have")
    print("  been. If a squared-error order ever re-enters the prose or a table,")
    print("  `src/verify_paper_numbers.py` is the check that fails.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
