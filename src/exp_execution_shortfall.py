"""What is the reconstruction order worth, in money?

WHY THIS EXPERIMENT EXISTS
--------------------------
Everything the paper says about control accuracy is in relative L2 units. A practitioner does not
liquidate a relative L2 norm. This converts the paper's central result into the currency it actually
trades in, on the paper's own real-market calibration: for each of the nine instruments, the cost of
executing under a multilinear extraction versus a cubic one, in basis points of notional.

METHOD
------
The exact closed-form value function is placed on a grid (so the solve contributes no error at all and
the only difference between the arms is the reconstruction), each reconstruction produces a control,
and the deterministic closed loop dx = (Ax + Bu)dt is integrated under it. The objective is the
benchmark's:

    cost(u) = 1/2 int (lambda sigma^2 x'x + eta u'u) dt + 1/2 Q_T x_T'x_T

whose units ARE money once sigma and eta are relative: the impact term int eta u'u dt is the extra
spread/impact paid, as a fraction of notional, and 1e4 times it is basis points. The risk term is a
preference, not a cash cost, and is reported separately rather than folded in.

THE TERMINAL SINGULARITY, AND WHY THE HORIZON MUST NOT BE TRUNCATED
-------------------------------------------------------------------
With real spreads the optimal control at t -> T is large (at SPY's measured eta, |u*(T)| is 780
against 0.6-2.5 in the interior), so the last instants carry a large cost term. That does NOT spoil
the comparison, because both arms carry it: what is reported is the PAIRWISE difference between the
two reconstructions, which share the snapshots, the time interpolation and the terminal region.

Truncating the horizon is what breaks it. Cut the integral at T - delta and the terminal penalty that
pays for not liquidating is removed with it, so the slower (multilinear) policy looks cheaper than the
cubic one: measured, the pairwise difference goes from +8.4 bp at delta = 0 to -0.21 bp at
delta = 0.10 on SPY. That is the same artefact the paper documents for accuracy metrics at real
spreads --- the fix there was to drop the terminal sample from an ACCURACY measurement, where the
singularity is a property of the reference and not of the solver. Here the cost comparison needs it
kept. Reporting the sensitivity is the point of the delta sweep below.

Usage: python src/exp_execution_shortfall.py [--n-grid 21] [--delta 0.0]
Output: results/exp_execution_shortfall.json
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from exp_control_extraction import extract_cell, extract_spline  # noqa: E402
from exp_real_calibration import build, calibrated_assets  # noqa: E402

RESULTS = SRC.parent / "results"
ASSETS = ["SPY", "AMZN", "MSFT", "QQQ", "GOOGL", "OXY", "NVDA", "AAPL", "TSLA"]


def rollout(prob: ACProblem, control, x0: np.ndarray, t_end: float, n_steps: int = 4000):
    dt = t_end / n_steps
    xs = np.zeros((n_steps + 1, x0.size))
    us = np.zeros((n_steps + 1, x0.size))
    xs[0] = x0
    t = 0.0
    for k in range(n_steps):
        def rhs(x, tt):
            u = control(x.reshape(1, -1), tt)[0]
            return prob.A @ x + prob.B @ u, u
        k1, u1 = rhs(xs[k], t)
        k2, _ = rhs(xs[k] + 0.5 * dt * k1, t + 0.5 * dt)
        k3, _ = rhs(xs[k] + 0.5 * dt * k2, t + 0.5 * dt)
        k4, _ = rhs(xs[k] + dt * k3, t + dt)
        us[k] = u1
        xs[k + 1] = xs[k] + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
    return xs, us, dt


def costs(prob: ACProblem, xs: np.ndarray, us: np.ndarray, dt: float) -> dict:
    """Impact cost, risk penalty and terminal term, all as fractions of notional (x0 = 1 unit).

    Trapezoid on the midpoints, which is what a running cost sampled at the nodes deserves; the
    impact term on its own is NOT the objective (trading slower lowers int eta u'u while leaving
    inventory behind), so the signed statement the paper quotes is the excess of the TOTAL.
    """
    q = np.einsum("ij,jk,ik->i", xs, prob.Q, xs)
    r = np.einsum("ij,jk,ik->i", us, prob.R, us)
    impact = 0.5 * dt * float(np.sum(0.5 * (r[1:] + r[:-1])))
    risk = 0.5 * dt * float(np.sum(0.5 * (q[1:] + q[:-1])))
    term = 0.5 * float(xs[-1] @ prob.QT @ xs[-1])
    return {"impact": impact, "risk": risk, "terminal": term,
            "total": impact + risk + term}


def snapshot_control(prob: ACProblem, ref, axes, n_grid: int, extractor, n_snap: int = 41):
    """A time-interpolated control, as the paper's solvers store one.

    This is not a detail. Using the t = 0 snapshot at every time -- which the first version of this
    script did -- applies a frozen policy to a problem whose optimal control is strongly
    time-dependent, and the resulting 34% position error dwarfs the reconstruction effect the
    experiment exists to measure. The same interpolation the solver uses between stored snapshots is
    applied here.
    """
    ts = np.linspace(0.0, prob.T, n_snap)
    Xg = axes[0].reshape(-1, 1)
    snaps = [np.asarray(ref.value(Xg, float(t))).ravel() for t in ts]

    def control(X, t):
        X = np.atleast_2d(X)
        j = int(np.clip(np.searchsorted(ts, t), 1, n_snap - 1))
        w = (float(t) - ts[j - 1]) / (ts[j] - ts[j - 1])
        g = ((1.0 - w) * extractor(axes, snaps[j - 1], X, 1e-4)
             + w * extractor(axes, snaps[j], X, 1e-4))
        return g @ prob.B @ prob.R_inv.T
    return control


def measure_asset(name: str, c: dict, lam: float, delta: float, n_grid: int) -> dict:
    """Excess cost of each reconstruction, in basis points of notional, for one asset."""
    prob: ACProblem = build(name, c, lam, 1)
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    lo, hi = prob.domain()
    axes = [np.linspace(lo[0], hi[0], n_grid)]

    def control_star(X, t):
        return ref.control(np.atleast_2d(X), float(t))

    t_end = prob.T - delta
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    xs_s, us_s, dt = rollout(prob, control_star, x0, t_end)
    base = costs(prob, xs_s, us_s, dt)
    row = {"asset": name, "eta": c["eta"], "sigma": c["sigma"], "lam": lam, "delta": delta,
           "baseline_total": base["total"], "baseline_impact": base["impact"]}
    arms = {}
    for tag, extractor in (("multilinear", extract_cell), ("cubic", extract_spline)):
        ctrl = snapshot_control(prob, ref, axes, n_grid, extractor)
        xs, us, _ = rollout(prob, ctrl, x0, t_end)
        cst = costs(prob, xs, us, dt)
        arms[tag] = cst["total"]
        row[f"total_{tag}"] = cst["total"]
        row[f"excess_total_bp_{tag}"] = (cst["total"] - base["total"]) * 1e4
        row[f"excess_impact_bp_{tag}"] = (cst["impact"] - base["impact"]) * 1e4
        row[f"position_error_{tag}"] = float(np.max(np.abs(xs[:, 0] - xs_s[:, 0])))
    # THE QUOTABLE NUMBER is the PAIRWISE difference, not the excess over the exact optimum.
    # Both arms use the same snapshots and the same time interpolation, so the only thing that
    # differs between them is the spatial reconstruction; the exact optimum, by contrast, is
    # evaluated through the reference's own piecewise-linear P(t), which is itself slightly
    # suboptimal and can make a suboptimal control look better than "exact" at some horizons --
    # measured: the excess over the optimum changes sign (-0.11 bp at delta=0.02, +0.71 bp at
    # delta=0.05) while the pairwise difference does not.
    row["pairwise_bp"] = (arms["multilinear"] - arms["cubic"]) * 1e4
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-grid", type=int, default=21)
    ap.add_argument("--delta", type=float, default=0.0,
                    help="terminal slab excluded from the cost integral; 0 is the full horizon and "
                         "is the only setting in which the comparison is well posed (see above)")
    ap.add_argument("--lam", type=float, default=1e-4)
    ap.add_argument("--out", default="exp_execution_shortfall.json")
    args = ap.parse_args()

    print("=" * 104)
    print("EXECUTION SHORTFALL BY RECONSTRUCTION, on the paper's real-market calibration")
    print("=" * 104)
    print(f"  N={args.n_grid}, lambda={args.lam:g}, horizon [0, T-{args.delta:g}]")
    print("  headline: cost(multilinear) - cost(cubic), in bp of notional. Both arms share the")
    print("  snapshots and the time interpolation, so only the spatial reconstruction differs.")

    cal = calibrated_assets(ASSETS)
    out = {"n_grid": args.n_grid, "delta": args.delta, "lam": args.lam, "rows": []}
    print(f"\n  {'asset':>7} {'eta (bp)':>9} {'multilinear vs cubic':>21} {'excess over exact ml':>21} "
          f"{'excess over exact cu':>21}")
    for name in ASSETS:
        if name not in cal:
            continue
        row = measure_asset(name, cal[name], args.lam, args.delta, args.n_grid)
        out["rows"].append(row)
        print(f"  {name:>7} {cal[name]['eta']*1e4:>9.2f} {row['pairwise_bp']:>18.3f}bp "
              f"{row['excess_total_bp_multilinear']:>18.3f}bp "
              f"{row['excess_total_bp_cubic']:>18.3f}bp")

    # ---------------- robustness: the terminal cut and the (unmeasurable) risk aversion ---------
    print()
    print("  robustness on SPY: the terminal cut is the one that matters")
    print(f"    {'delta':>7} {'lam':>9} {'pairwise':>12} {'excess ml':>11} {'excess cu':>11}")
    sens = []
    for delta in (0.0, 0.02, 0.05, 0.10):
        for lam in (1e-6, 1e-4, 1e-2):
            r = measure_asset("SPY", cal["SPY"], lam, delta, args.n_grid)
            sens.append({"delta": delta, "lam": lam, "pairwise_bp": r["pairwise_bp"],
                         "excess_multilinear_bp": r["excess_total_bp_multilinear"],
                         "excess_cubic_bp": r["excess_total_bp_cubic"]})
            print(f"    {delta:>7.2f} {lam:>9.0e} {r['pairwise_bp']:>9.3f}bp "
                  f"{r['excess_total_bp_multilinear']:>8.3f}bp "
                  f"{r['excess_total_bp_cubic']:>8.3f}bp")
    out["sensitivity"] = sens

    if out["rows"]:
        pw = [r["pairwise_bp"] for r in out["rows"]]
        cu = [r["excess_total_bp_cubic"] for r in out["rows"]]
        out["summary"] = {
            "pairwise_bp_min": float(min(pw)), "pairwise_bp_max": float(max(pw)),
            "pairwise_bp_mean": float(np.mean(pw)),
            "cubic_excess_bp_max": float(max(cu)),
            "position_error_reduction": float(np.mean(
                [r["position_error_multilinear"] / r["position_error_cubic"] for r in out["rows"]])),
            "currency_at_10m_notional": float(np.mean(pw) * 1e-4 * 1e7),
        }
        print()
        print("=" * 104)
        print(f"  raising the reconstruction order is worth {min(pw):.2f} to {max(pw):.2f} bp of "
              f"notional (mean {np.mean(pw):.2f} bp) across the nine instruments")
        print(f"  the cubic arm itself costs at most {max(cu):.3f} bp more than the exact control, "
              f"i.e. it is at the numerical floor")
        print(f"  position error falls by {out['summary']['position_error_reduction']:.1f}x")
        print(f"  on a $10M order the mean figure is about "
              f"${out['summary']['currency_at_10m_notional']:,.0f}")
        print("=" * 104)
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
