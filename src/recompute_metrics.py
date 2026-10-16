"""Rebuild every metric the paper quotes, on ONE scale, from the committed results.

Nothing is re-run: each stored quantity is a normalised mean squared error, and the paper's
canonical metric is its square root (see `metrics.py`). Where an experiment already stored an
error norm (`exp_reconstruction_order.py`, `exp_control_consequence.py`) it is used as-is.

Output: results/metrics_l2.json  -- the single source the manuscript's numbers are taken from.
Usage:  python src/recompute_metrics.py
"""
from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np

SRC = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from metrics import (  # noqa: E402
    CONVENTION, fit_order, order_from_mse_order, ratio_from_mse_ratio, rms_from_mse,
)

ROOT = SRC.parent
RESULTS = ROOT / "results"

# What counts as "below the floating-point floor" for an order fit. Same threshold the original
# experiments used, so the refit is on the same points and only the SCALE changes.
FLOOR = 1e-16


def load(name: str) -> dict:
    path = RESULTS / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pooled_orders(docs: list[dict], merton: dict | None = None) -> dict:
    """Gradient-error order per (method, function), pooled over the dimensions that were run."""
    out: dict[str, dict[str, dict]] = {}
    for doc in docs:
        for row in (doc or {}).get("synthetic", []) or []:
            for method, track in (row.get("tracks") or {}).items():
                pts = [(h, e) for h, e in track if e and e > FLOOR]
                if len(pts) < 3:
                    continue
                order_rms, r2 = fit_order([p[0] for p in pts],
                                          [rms_from_mse(p[1]) for p in pts])
                slot = out.setdefault(row["function"], {}).setdefault(
                    method, {"orders": [], "fit_r2": []})
                slot["orders"].append(order_rms)
                slot["fit_r2"].append(r2)
    for fn, per_method in out.items():
        for method, slot in per_method.items():
            vals = [v for v in slot["orders"] if v == v]
            slot["order_mean"] = float(np.mean(vals)) if vals else float("nan")
            slot["order_min"] = float(np.min(vals)) if vals else float("nan")
            slot["order_max"] = float(np.max(vals)) if vals else float("nan")
            slot["n_dims"] = len(vals)
            slot["fit_r2_mean"] = float(np.mean([r for r in slot["fit_r2"] if r == r])) \
                if any(r == r for r in slot["fit_r2"]) else float("nan")

    m_orders = ((merton or {}).get("meta") or {}).get("exact_node_orders") or {}
    if m_orders:
        out["Merton"] = {
            m: {
                "order_mean": order_from_mse_order(v.get("order")),
                "order_min": order_from_mse_order(v.get("order")),
                "order_max": order_from_mse_order(v.get("order")),
                "n_dims": 1,
                "fit_r2_mean": v.get("fit_r2"),
                "order_mse": v.get("order"),
            }
            for m, v in m_orders.items()
        }
    return out


def fixed_mesh_gain(docs: list[dict]) -> dict:
    """How much better a cubic reconstruction is than a multilinear one, at the SAME mesh."""
    per_function = {}
    for doc in docs:
        for row in (doc or {}).get("synthetic", []) or []:
            tracks = row.get("tracks") or {}
            fd = {round(h, 8): e for h, e in tracks.get("fd", []) if e and e > FLOOR}
            sp = {round(h, 8): e for h, e in tracks.get("spline", []) if e and e > FLOOR}
            common = sorted(set(fd) & set(sp))
            if not common:
                continue
            gains = [ratio_from_mse_ratio(fd[h] / sp[h]) for h in common]
            per_function.setdefault(row["function"], []).extend(gains)
    summary = {}
    for fn, g in per_function.items():
        g = [x for x in g if x == x]
        if g:
            summary[fn] = {"rms_ratio_min": float(np.min(g)), "rms_ratio_max": float(np.max(g)),
                           "n": len(g)}
    allg = [v for g in per_function.values() for v in g if v == v]
    if allg:
        summary["_overall"] = {"rms_ratio_min": float(np.min(allg)),
                               "rms_ratio_max": float(np.max(allg)), "n": len(allg)}
    return summary


def solved_grid_ratio(ext: dict, order: dict) -> dict:
    """The p=2, N=61 headline: multilinear vs cubic control error, and each against the value error.

    The value error comes from `exp_control_order.json` part_a_refinement (same problem, same grid),
    the control errors from the extraction experiment's solved-grid section. Reporting them as a
    ratio of RMS errors rather than of their squares is the whole point of this file.
    """
    grid = (ext or {}).get("solved_grid") or []
    ref = [r for r in ((order or {}).get("part_a_refinement") or [])
           if r.get("p") == 2 and r.get("N") == 61]
    if not grid or not ref:
        return {}
    row = max(grid, key=lambda r: r.get("N", 0))
    value_mse = ref[0].get("value_err")
    errs = row.get("errors") or {}
    out = {"p": 2, "N": row.get("N"), "value_mse": value_mse,
           "value_rms": rms_from_mse(value_mse)}
    for method in ("fd", "cell", "spline"):
        if errs.get(method):
            out[f"{method}_mse"] = errs[method]
            out[f"{method}_rms"] = rms_from_mse(errs[method])
    if out.get("fd_rms") and out.get("value_rms"):
        out["ratio_multilinear_rms"] = out["fd_rms"] / out["value_rms"]
        out["ratio_multilinear_mse"] = out["fd_mse"] / out["value_mse"]
    if out.get("spline_rms") and out.get("value_rms"):
        out["ratio_cubic_rms"] = out["spline_rms"] / out["value_rms"]
        out["ratio_cubic_mse"] = out["spline_mse"] / out["value_mse"]
    if out.get("fd_rms") and out.get("spline_rms"):
        out["gain_rms"] = out["fd_rms"] / out["spline_rms"]
        out["gain_mse"] = out["fd_mse"] / out["spline_mse"]
    return out


def control_order_table(order: dict, order_p3: dict | None = None) -> dict:
    """Per-dimension value/control errors and orders on the canonical scale."""
    rows = (order or {}).get("part_c_orders_at_best_probe") or []
    per_p: dict[int, dict] = {}
    for r in rows:
        p = int(r["p"])
        slot = per_p.setdefault(p, {"N": [], "h": [], "value_rms": [], "control_rms": [],
                                    "ratio_rms": []})
        slot["N"].append(int(r["N"]))
        slot["h"].append(float(r["h"]))
        slot["value_rms"].append(rms_from_mse(r.get("value_err")))
        slot["control_rms"].append(rms_from_mse(r.get("control_err")))
        slot["ratio_rms"].append(ratio_from_mse_ratio(r.get("ratio")))
    for p, slot in per_p.items():
        # Order is the slope of log(error) against log(h), with h the mesh spacing the row carries.
        o_v, r2_v = fit_order(slot["h"], slot["value_rms"])
        o_c, r2_c = fit_order(slot["h"], slot["control_rms"])
        slot["order_value_rms"] = o_v
        slot["order_control_rms"] = o_c
        slot["order_value_fit_r2"] = r2_v
        slot["order_control_fit_r2"] = r2_c
        slot["ratio_rms_first"] = slot["ratio_rms"][0]
        slot["ratio_rms_peak"] = max(slot["ratio_rms"])

    all_ratios = [x for s in per_p.values() for x in s["ratio_rms"] if x == x]
    headline = {
        "ratio_rms_min": float(np.min(all_ratios)) if all_ratios else float("nan"),
        "ratio_rms_max": float(np.max(all_ratios)) if all_ratios else float("nan"),
        "n_points": len(all_ratios),
    }
    fits = {}
    for k, v in ((load("exp_control_orders_fit.json").get("fits")) or {}).items():
        fits[k] = {
            "control_order_rms": order_from_mse_order(v.get("control_order")),
            "control_order_mse": v.get("control_order"),
            "control_fit_r2": v.get("control_fit_r2"),
            "value_order_rms": order_from_mse_order(v.get("value_order")),
            "value_order_mse": v.get("value_order"),
            "n_points": v.get("n_points"),
        }
    return {"per_p": per_p, "headline": headline, "least_squares_fits": fits}


def time_refinement(floor: dict) -> dict:
    """Mesh fixed, snapshot count swept: how much each error improves in the RMS metric."""
    rows = (floor or {}).get("part_d_snapshot_sweep") or []
    out = []
    for p in sorted({r["p"] for r in rows}):
        for N in sorted({r["N"] for r in rows if r["p"] == p}):
            rs = sorted([r for r in rows if r["p"] == p and r["N"] == N],
                        key=lambda r: r["n_snap"])
            if len(rs) < 2 or not rs[-1].get("value_err"):
                continue
            out.append({
                "p": p, "N": N, "n_snap_first": rs[0]["n_snap"], "n_snap_last": rs[-1]["n_snap"],
                "value_gain_rms": ratio_from_mse_ratio(rs[0]["value_err"] / rs[-1]["value_err"]),
                "control_gain_rms": ratio_from_mse_ratio(
                    rs[0]["control_err"] / rs[-1]["control_err"]),
                "value_gain_mse": rs[0]["value_err"] / rs[-1]["value_err"],
                "control_gain_mse": rs[0]["control_err"] / rs[-1]["control_err"],
            })
    return {"rows": out}


def main() -> int:
    ext = load("exp_control_extraction.json")
    ext3 = load("exp_control_extraction_p3.json")
    ext5 = load("exp_control_extraction_p5.json")
    merton = load("exp_merton_extraction.json")
    order = load("exp_control_order.json")
    floor = load("exp_control_floor.json")

    report = {
        "convention": CONVENTION,
        "note": ("Every error below is a relative L2 norm. Values were converted from the stored "
                 "normalised MSE by sqrt; orders that were fitted to the MSE were halved; ratios "
                 "of errors were square-rooted. No experiment was re-run."),
        "extraction_orders_rms": pooled_orders([ext, ext3, ext5], merton),
        "extraction_fixed_mesh_gain": fixed_mesh_gain([ext, ext3, ext5]),
        "solved_grid_p2": solved_grid_ratio(ext, order),
        "control_order": control_order_table(order),
        "time_refinement": time_refinement(floor),
        "already_on_this_scale": {
            "exp_reconstruction_order": "gradient orders by reconstruction degree, p=2, jittered mesh",
            "exp_control_consequence": "executed-path position and objective-loss orders",
        },
    }

    print("=" * 100)
    print("EXTRACTION ORDERS (relative L2 scale, pooled over p = 2, 3, 5)")
    print("=" * 100)
    print(f"  {'function':14s} {'method':8s} {'mean':>7s} {'range':>16s} {'n':>3s}")
    for fn, per_method in report["extraction_orders_rms"].items():
        for m, slot in per_method.items():
            rng = f"{slot['order_min']:.2f}-{slot['order_max']:.2f}"
            print(f"  {fn:14s} {m:8s} {slot['order_mean']:7.2f} {rng:>16s} {slot['n_dims']:3d}")

    print()
    print("=" * 100)
    print("FIXED-MESH GAIN: cubic vs multilinear (ratio of relative L2 errors)")
    print("=" * 100)
    for fn, s in report["extraction_fixed_mesh_gain"].items():
        print(f"  {fn:14s} {s['rms_ratio_min']:8.1f} - {s['rms_ratio_max']:8.1f}   (n={s['n']})")

    print()
    print("=" * 100)
    print("SOLVED GRID, p=2, N=61")
    print("=" * 100)
    for k, v in report["solved_grid_p2"].items():
        print(f"  {k:26s} {v if isinstance(v, str) else round(v, 8)}")

    print()
    print("=" * 100)
    print("CONTROL ORDER AND RATIOS (relative L2 scale)")
    print("=" * 100)
    for p, slot in sorted(report["control_order"]["per_p"].items()):
        print(f"  p={p}: order(value)={slot['order_value_rms']:.2f}  "
              f"order(control)={slot['order_control_rms']:.2f}  "
              f"ratio {slot['ratio_rms_first']:.1f} -> {slot['ratio_rms_peak']:.1f}")
    hl = report["control_order"]["headline"]
    print(f"  control/value ratio over all {hl['n_points']} points: "
          f"{hl['ratio_rms_min']:.1f} to {hl['ratio_rms_max']:.1f}")
    print("  least-squares fits (RMS):")
    for k, v in report["control_order"]["least_squares_fits"].items():
        print(f"    {k:36s} control {v['control_order_rms']:.2f} "
              f"(R2 {v['control_fit_r2']:.3f})   value {v['value_order_rms']:.2f}")

    print()
    print("=" * 100)
    print("TIME REFINEMENT, MESH FIXED (gain in relative L2 error)")
    print("=" * 100)
    for r in report["time_refinement"]["rows"]:
        print(f"  p={r['p']} N={r['N']:>3d}: value x{r['value_gain_rms']:.1f}  "
              f"control x{r['control_gain_rms']:.1f}")

    out = RESULTS / "metrics_l2.json"
    out.write_text(json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
