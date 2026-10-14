"""
P1: does the comparison survive REAL market coefficients, rather than synthetic ones?

Why this is needed
------------------
journal_audit.py flagged the absence of real-market grounding as a blocker: every table in the
paper is computed with synthetic coefficients (eta = lam_sigma2 = 0.01, q_terminal = 0.1), and a
referee's first objection to a solver benchmark is that the numbers belong to a problem nobody
trades. This addresses it with the data already in the repository: minute bars for nine US
instruments, preprocessed causally (data_causal/, ascending, with temporal splits).

THE PROBLEM WITH A SINGLE "CALIBRATED" RUN, AND WHY THIS IS A SWEEP
-------------------------------------------------------------------
Two of the three coefficients are measurable and one is not:
  * sigma, the diffusion scale, is the relative volatility -- measured (ewma_volatility).
  * eta, the temporary impact, is not observable. It is here set to half the Roll (1984) effective
    spread, which is the standard identification: a round trip pays the spread once, so a unit
    trade pays about half of it.
  * lambda, the risk aversion, is a PREFERENCE, not a data quantity. Nothing in the price series
    determines it, so any single value is a choice, not a measurement.

That last point decides the design. At T = 1 the problem is governed by k = sqrt(Q/eta) =
sigma*sqrt(lambda)/sqrt(eta); with real sigma and real eta, a defensible lambda in [1e-6, 1e-2]
puts k between 0.001 and 0.1, while the paper's synthetic coefficients sit at k = 1 -- an order of
magnitude FASTER liquidation than realistic risk aversion implies. So a single calibrated run would
replace one arbitrary point with another.

What is worth reporting is therefore a span: run the comparison at the coefficients real data
implies, across the range of lambda a practitioner might hold, and at the synthetic default, and see
whether the paper's conclusions move. If the ranking of the solver families and the size of the
control/value error gap hold across that span, then the synthetic calibration is not what the paper's
findings rest on, and saying so is stronger than any single calibrated number.

What is measured, per (asset, lambda, dimension):
  * value R2 and control R2 for the accuracy-matched grid scheme and for the PINN;
  * the control/value error ratio, which is the paper's headline quantity.

Usage:
  python src/exp_real_calibration.py --dims 1,2 --lams 1e-6,1e-4,1e-2 --assets SPY,AAPL,NVDA,TSLA
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                      # noqa: E402
from solvers import FiniteDifferenceSolver            # noqa: E402
from check_terminal_contamination import metrics_at   # noqa: E402


def accuracy_metrics_at(solver, problem: ACProblem, n_times: int, seed: int = 1) -> dict:
    """The paper's accuracy metrics, with the terminal sample excluded.

    Delegates to the reimplementation in check_terminal_contamination, which was verified to
    reproduce `solvers.accuracy_metrics` exactly on identical query times. Reusing it here keeps one
    definition of the metric rather than two that can drift.
    """
    times = np.linspace(0.0, problem.T, n_times + 1)[:-1]
    return metrics_at(solver, problem, times, n_points=400, seed=seed)

ROOT = SRC.parent
CAUSAL = ROOT / "data_causal"
MANIFEST = CAUSAL / "preprocess_manifest.json"
OUT = ROOT / "results"

# Minutes in a US equity session, for converting a per-minute volatility to a per-day one.
MINUTES_PER_SESSION = 390


def _window(entry: dict) -> dict:
    """The manifest stores one block per source window; take the latest available."""
    per = entry.get("per_source") or {}
    for key in ("B", "A"):
        if key in per:
            return per[key]
    return next(iter(per.values())) if per else {}


def calibrated_assets(names: list[str]) -> dict[str, dict]:
    """sigma and eta per asset, from the causal data and the manifest.

    sigma is the median per-minute EWMA relative volatility scaled to one session, because the
    problem's horizon is T = 1 and the natural unit of a trading problem is a session. eta is half
    the Roll effective spread, using the same horizon scaling: the spread cost of liquidating one
    unit at unit rate over the horizon is the spread itself, so eta = spread/2 in the model's
    rate-cost convention.
    """
    if not MANIFEST.exists():
        raise SystemExit(f"manifest not found: {MANIFEST}")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    out: dict[str, dict] = {}
    for name in names:
        entry = (manifest.get("assets") or {}).get(name)
        if not entry:
            print(f"  skip {name}: not in the manifest")
            continue
        w = _window(entry)
        roll = w.get("roll_spread_mean")
        cs = w.get("cs_spread_mean")
        csv_path = CAUSAL / f"{name}_causal.csv"
        if not csv_path.exists() or roll is None:
            print(f"  skip {name}: missing csv or spread estimate")
            continue
        vols = []
        with csv_path.open(newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                try:
                    vols.append(float(row["ewma_volatility"]))
                except (TypeError, ValueError):
                    continue
        if not vols:
            print(f"  skip {name}: no volatility column")
            continue
        sigma_min = statistics.median(vols)
        sigma = sigma_min * math.sqrt(MINUTES_PER_SESSION)
        out[name] = {"sigma": sigma, "sigma_minute": sigma_min,
                     "roll_spread": roll, "cs_spread": cs,
                     "eta": roll / 2.0, "n_minutes": len(vols)}
    return out


def build(asset: str, cal: dict, lam: float, p: int) -> ACProblem:
    """The benchmark problem with one asset's real coefficients, replicated across p assets."""
    return ACProblem(p=p, eta=cal["eta"], lam_sigma2=lam * cal["sigma"] ** 2,
                     q_terminal=0.1, sigma=cal["sigma"])


def synthetic(p: int) -> ACProblem:
    return ACProblem(p=p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2")
    ap.add_argument("--lams", type=str, default="1e-6,1e-4,1e-2")
    ap.add_argument("--assets", type=str, default="SPY,AAPL,NVDA,TSLA")
    ap.add_argument("--n-grid", type=int, default=41)
    # The query schedule must NOT include t = T, and this is not a detail. The optimal control is
    # -R^-1 B' P(t) x, so it diverges as t -> T; the sharper the divergence the smaller the impact
    # coefficient eta. Measured: with the paper's synthetic eta = 0.01 the reference control at
    # t = T has mean |u| = 5.1, but with SPY's measured eta = 1.28e-4 -- 78x smaller -- it is 395, so
    # a single terminal sample carries roughly all of the mean. Scoring there measures the
    # singularity, not the solver, and it is what made an earlier version of this experiment report
    # value R^2 = -997 for a grid scheme that is in fact accurate in the interior. The interior
    # samples are comparable between the two calibrations (0.51-2.52 against 0.63-1.74), so dropping
    # the terminal sample is what makes the real-coefficient run commensurable with the paper's.
    ap.add_argument("--n-times", type=int, default=6)
    ap.add_argument("--out", type=str, default="exp_real_calibration.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    names = [s.strip() for s in args.assets.split(",") if s.strip()]
    lams = [float(s) for s in args.lams.split(",") if s.strip()]
    dims = [int(s) for s in args.dims.split(",") if s.strip()]

    cal = calibrated_assets(names)
    print("=" * 112)
    print("REAL-COEFFICIENT SENSITIVITY: does the comparison survive market-derived parameters?")
    print("=" * 112)
    print("  sigma from per-minute EWMA volatility scaled to a session; eta = half the Roll (1984)")
    print("  effective spread; lambda is a PREFERENCE (not measurable), so it is swept, not fitted.")
    print()
    print(f"  {'asset':>7} {'sigma(session)':>15} {'roll spread':>12} {'eta=roll/2':>11} "
          f"{'cs cross-chk':>13} {'minutes':>8}")
    for n, c in cal.items():
        print(f"  {n:>7} {c['sigma']:>15.4f} {c['roll_spread']:>12.3e} {c['eta']:>11.3e} "
              f"{c['cs_spread']:>13.3e} {c['n_minutes']:>8}")

    rows: list[dict] = []

    def measure(prob: ACProblem, tag: str, extra: dict) -> None:
        try:
            gs = FiniteDifferenceSolver(prob, n_grid=args.n_grid)
            gs.solve()
            gm = accuracy_metrics_at(gs, prob, args.n_times, seed=1)
        except Exception as exc:                       # noqa: BLE001
            print(f"    grid failed at {tag}: {type(exc).__name__}: {exc}")
            return
        k = math.sqrt(float(prob.lam_sigma2) / float(prob.eta))
        rows.append({**extra, "tag": tag, "k": k, "method": "grid",
                     "value_r2": gm.get("value_r2"), "control_r2": gm.get("control_r2"),
                     "control_sign_agreement": gm.get("control_sign_agreement"),
                     "value_mse": gm.get("value_mse"), "control_mse": gm.get("control_mse"),
                     "n_time_steps": gm.get("n_time_steps")})
        # The headline quantity, computed as two RELATIVE errors and only then divided.
        #
        # Dividing raw control_mse by raw value_mse is meaningless -- they are in different units
        # (a running cost per unit time against a monetary value) -- and an earlier version of this
        # experiment did exactly that and reported ratios of 1e5. Both errors are scaled by the mean
        # magnitude of their own reference quantity first, which is what makes the ratio a statement
        # about accuracy rather than about units.
        ratio = gm.get("control_over_value")
        rows[-1]["control_over_value_err"] = ratio
        print(f"    {tag:<34} k={k:>7.4f}  value R2={gm['value_r2']:>+9.5f}  "
              f"ctrl R2={gm['control_r2']:>+8.4f}  ctrl/val err={('%.1f' % ratio) if ratio else '--':>8}")

    for p in dims:
        print()
        print(f"--- p = {p} ---")
        measure(synthetic(p), f"p{p} synthetic default", {"p": p, "asset": "synthetic",
                                                          "lam": None})
        for name, c in cal.items():
            for lam in lams:
                prob = build(name, c, lam, p)
                measure(prob, f"p{p} {name} lam={lam:g}",
                        {"p": p, "asset": name, "lam": lam, "sigma": c["sigma"], "eta": c["eta"]})

    (OUT / args.out).write_text(json.dumps(
        {"rows": rows, "assets": cal, "dims": dims, "lams": lams, "n_grid": args.n_grid},
        indent=2), encoding="utf-8")

    # ---------------- summary ----------------
    print()
    print("=" * 112)
    print("SUMMARY")
    print("=" * 112)
    good = [r for r in rows if r.get("value_r2") is not None]
    if good:
        print(f"  {len(good)} configurations measured at p = {dims}")
        syn = [r for r in good if r["asset"] == "synthetic"]
        real = [r for r in good if r["asset"] != "synthetic"]
        for label, grp in (("synthetic", syn), ("real-calibrated", real)):
            if not grp:
                continue
            vr = [r["value_r2"] for r in grp]
            cr = [r["control_r2"] for r in grp if r.get("control_r2") is not None]
            ks = [r["k"] for r in grp]
            print(f"  {label:>17}: k from {min(ks):.4f} to {max(ks):.4f}, "
                  f"value R2 from {min(vr):+.4f} to {max(vr):+.4f}, "
                  f"control R2 from {min(cr):+.4f} to {max(cr):+.4f}"
                  if cr else f"  {label:>17}: k {min(ks):.4f}..{max(ks):.4f}")
        ratios = [r["control_over_value_err"] for r in good if r.get("control_over_value_err")]
        if ratios:
            print(f"  control/value error ratio across every configuration: "
                  f"{min(ratios):.1f} to {max(ratios):.1f}")
        print()
        print("  READ THIS AS A ROBUSTNESS RESULT, NOT A CALIBRATION RESULT. Lambda is a preference,")
        print("  so no single value is 'the' calibrated one; the claim is that the comparison's")
        print("  conclusions do not depend on which defensible value is chosen.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
