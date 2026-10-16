"""The paper's error metric, defined once.

WHY THIS FILE EXISTS
--------------------
Until now the manuscript's "errors" came from three experiments that fitted two different
quantities:

  * `exp_control_extraction.py`, `exp_merton_extraction.py` and `exp_control_order.py` store a
    NORMALISED MEAN SQUARED ERROR, ``mean((g - g_ref)**2) / var(g_ref)``, and fit the slope of
    log(that) against log(h). Because log(E^2) = 2 log(E), the slope so obtained is TWICE the
    convergence order of the error itself.
  * `exp_reconstruction_order.py` and `exp_control_consequence.py` fit the slope of log of the
    error NORM, which is the convergence order.

The consequence is not cosmetic: the same multilinear control error was reported as "order 2.00"
in one table and "1.03-1.10" in another, and the headline control/value error ratio was a ratio
of squared errors, so it overstated the error ratio by its square root.

THE CONVENTION ADOPTED HERE
---------------------------
One metric, used everywhere in the paper:

    e  =  || a - a_ref ||_2 / || a_ref ||_2          (relative L2 error, a.k.a. the RMS error
                                                      normalised by the RMS of the reference)

and every convergence order is the least-squares slope of log(e) against log(h). R^2 stays in the
paper as a secondary, dimensionless accuracy number, but it is never used to fit an order.

HOW THE OLD NUMBERS ARE CONVERTED
---------------------------------
Exactly, with no re-running of any experiment, because the stored quantity is the square of the
new one:

    e_rms      = sqrt(e_mse)
    order_rms  = order_mse / 2          (log(sqrt(E)) = log(E)/2)
    ratio_rms  = sqrt(ratio_mse)

So `recompute_metrics.py` can rebuild every affected number from the committed `results/*.json`.
`check_order_convention.py` is the guard that keeps the two scales from being mixed up again.
"""
from __future__ import annotations

import math

import numpy as np


def rms_from_mse(mse: float) -> float:
    """Relative L2 (RMS) error from a stored normalised mean squared error."""
    return math.sqrt(mse) if mse is not None and mse >= 0 else float("nan")


def order_from_mse_order(order_mse: float) -> float:
    """Convergence order of the error from the slope fitted to its square."""
    return order_mse / 2.0 if order_mse is not None and order_mse == order_mse else float("nan")


def ratio_from_mse_ratio(ratio_mse: float) -> float:
    """Ratio of two errors from the ratio of their squares."""
    return math.sqrt(ratio_mse) if ratio_mse is not None and ratio_mse > 0 else float("nan")


def relative_l2(values: np.ndarray, reference: np.ndarray) -> float:
    """||values - reference||_2 / ||reference||_2 -- the metric that replaces the MSE."""
    v = np.asarray(values, dtype=float).ravel()
    r = np.asarray(reference, dtype=float).ravel()
    denom = float(np.linalg.norm(r))
    if denom == 0.0:
        return float("nan")
    return float(np.linalg.norm(v - r) / denom)


def fit_order(hs, errs) -> tuple[float, float]:
    """Least-squares slope of log(error) against log(h), and the fit's R^2.

    Returns (nan, nan) when there are fewer than two usable points. Points whose error is zero or
    non-finite are dropped, because a machine-precision plateau is not evidence of an order -- that
    trap produced a spurious order of -2.1 earlier in this project.
    """
    pts = [(float(h), float(e)) for h, e in zip(hs, errs)
           if h and h > 0 and e and e > 0 and math.isfinite(e)]
    if len(pts) < 2:
        return float("nan"), float("nan")
    lx = np.log([p[0] for p in pts])
    ly = np.log([p[1] for p in pts])
    A = np.vstack([lx, np.ones_like(lx)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = A @ np.array([slope, intercept])
    ss_tot = float(((ly - ly.mean()) ** 2).sum())
    r2 = 1.0 - float(((ly - pred) ** 2).sum()) / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


CONVENTION = {
    "error": "relative L2 norm: ||a - a_ref||_2 / ||a_ref||_2  (RMS error / RMS reference)",
    "order": "least-squares slope of log(error) against log(mesh spacing h)",
    "r2": "secondary, dimensionless accuracy number; never used to fit an order",
    "conversion_from_stored_mse": "error = sqrt(mse); order = order_mse / 2; ratio = sqrt(ratio_mse)",
}
