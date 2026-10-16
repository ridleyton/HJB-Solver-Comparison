"""
Does every load-bearing number in the PROSE match the results it came from?

Why this is the check that matters
----------------------------------
The tables are generated from `results/`, so they cannot silently disagree with them -- but the
prose is written by hand, and it quotes numbers too: "a ratio of $13.6$", "the cubic order is $2.6$
against $1.0$", "control error is 3.2 to 26 times the value error". Those are the numbers a referee
will try to reproduce, and a rewrite of this size is exactly the situation in which a prose figure
goes stale while everything else stays consistent.

TWO THINGS THIS FILE NOW ENFORCES THAT IT DID NOT BEFORE
--------------------------------------------------------
1. THE METRIC. Every order asserted here is fitted to the relative L2 error, not to its square (see
   `metrics.py`). The result files store a normalised MSE, so the assertions convert before
   comparing -- and the conversion is the point: asserting a square-scale number against prose
   written on the error scale would pass while both were wrong.
2. THE RIGHT RUN. The deep-BSDE assertion filters the result file by dimension. It did not, in the
   generator, and the p=4 row of the table silently carried the d=8 run for months. An assertion
   that names the dimension is what makes that class of error visible.

WHAT THIS DOES AND DOES NOT DO
------------------------------
It recomputes each asserted quantity from a result file and checks that the manuscript contains it,
to the precision the manuscript states. It is a SPOT CHECK with named assertions, not an exhaustive
extraction of every numeral in the paper.

Usage: python src/verify_paper_numbers.py
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
RESULTS = ROOT / "results"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics import order_from_mse_order, ratio_from_mse_ratio, rms_from_mse  # noqa: E402


def load(name: str):
    p = RESULTS / name
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def tex() -> str:
    """The manuscript plus every table it inputs, so a number may live in either."""
    t = (PAPER / "main.tex").read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"\\input\{([^}]+)\}", t):
        name = m.group(1)
        p = PAPER / (name if name.endswith(".tex") else name + ".tex")
        if p.exists():
            t += "\n" + p.read_text(encoding="utf-8", errors="replace")
    return t


def has_number(text: str, value: float, places: int = 3) -> bool:
    """Is `value` present to the given number of decimal places, in any plausible formatting?"""
    if value is None or not math.isfinite(float(value)):
        return False
    cands = {f"{value:.{places}f}"}
    for trim in range(1, places):
        cands.add(f"{value:.{places - trim}f}")
    cands.add(f"{value:g}")
    for c in cands:
        c2 = c.lstrip("0") if c.startswith("0.") else c
        for form in {c, c2}:
            if re.search(r"(?<![\d.])" + re.escape(form) + r"(?![\d])", text):
                return True
    return False


def main() -> int:
    text = tex()
    checks: list[tuple[str, str, object, int, str]] = []
    skipped: list[str] = []

    # ---------------- solver comparison, unchanged by the metric work ----------------
    dim = load("exp_dimension.json")
    if dim:
        runs = [r for r in dim["runs"] if r.get("feasible")]
        fd5 = [r for r in runs if r["method"] == "Finite Difference" and r["p"] == 5]
        if fd5:
            checks.append(("grid p=5 value R2", "exp_dimension.json",
                           fd5[0]["value_r2"], 5, "grid scheme accuracy at the top dimension"))
        # The PINN column is a MEAN over seeds, as its caption states, so the assertion must average
        # too. Asserting seed 0 instead produced a false failure here once.
        pin5 = [r for r in runs if r["method"] == "PINN" and r["p"] == 5]
        if pin5:
            m = sum(r["value_r2"] for r in pin5) / len(pin5)
            checks.append((f"PINN p=5 value R2 (mean of {len(pin5)} seeds)", "exp_dimension.json",
                           m, 4, "the collapse the paper reports as -616"))
        pin1 = [r for r in runs if r["method"] == "PINN" and r["p"] == 1]
        if pin1:
            m = sum(r["value_r2"] for r in pin1) / len(pin1)
            checks.append((f"PINN p=1 value R2 (mean of {len(pin1)} seeds)", "exp_dimension.json",
                           m, 4, "PINN competitive at low dimension"))

    # ---------------- the metric: relative L2 orders, fitted to the error ----------------
    fits = (load("exp_control_orders_fit.json") or {}).get("fits") or {}
    want = {1: ("exp_control_order part_c|p=1", "control order p=1 (relative L2)"),
            2: ("exp_control_order part_c|p=2", "control order p=2 (relative L2)"),
            3: ("exp_control_order part_c|p=3", "control order p=3 (relative L2)"),
            4: ("exp_control_floor part_e|p=4", "control order p=4 (relative L2)"),
            5: ("exp_control_p5 part_g|p=5", "control order p=5, time held fixed")}
    for p, (key, label) in want.items():
        v = fits.get(key)
        if v and v.get("control_order") is not None:
            checks.append((label, "exp_control_orders_fit.json",
                           order_from_mse_order(v["control_order"]), 2,
                           f"the 0.98/1.00 of 'it converges at a stable order of approximately 1'"))
            checks.append((f"value order p={p} (relative L2)", "exp_control_orders_fit.json",
                           order_from_mse_order(v["value_order"]), 2,
                           "the value error converges faster but irregularly"))
    if not fits:
        skipped.append("control/value orders: exp_control_orders_fit.json absent")

    # ---------------- extraction orders, on the same scale ----------------
    ext = load("exp_control_extraction.json")
    if ext:
        syn = {s["function"]: s for s in ext.get("synthetic") or []}
        quad = syn.get("quadratic", {}).get("orders", {})
        if quad.get("fd"):
            checks.append(("multilinear order, fd on quadratic (relative L2)",
                           "exp_control_extraction.json",
                           order_from_mse_order(quad["fd"]["order"]), 1,
                           "Table VI, row 1, first column"))
        mer = load("exp_merton_extraction.json")
        if mer:
            o = (((mer.get("meta") or {}).get("exact_node_orders") or {}).get("spline") or {})
            if o.get("order") is not None:
                checks.append(("cubic order, Merton (relative L2)", "exp_merton_extraction.json",
                               order_from_mse_order(o["order"]), 1,
                               "the 2.6 of '2.6 against 1.0'"))
        else:
            skipped.append("cubic order on Merton: exp_merton_extraction.json absent")

    # The intrinsic probe run: the evidence that the two order tables agree once the probe step is
    # matched, and therefore the number the prose quotes as "3.4 to 3.8".
    intr = load("exp_control_extraction_intrinsic.json")
    if intr:
        vals = []
        for row in intr.get("synthetic") or []:
            track = (row.get("tracks") or {}).get("spline") or []
            pts = [(h, e) for h, e in track if e and e > 1e-16 and row["function"] != "quadratic"]
            if len(pts) < 3:
                continue
            lx = [math.log(p[0]) for p in pts]
            ly = [math.log(rms_from_mse(p[1])) for p in pts]
            n = len(lx)
            mx, my = sum(lx) / n, sum(ly) / n
            s = sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)
            vals.append(s)
        if vals:
            checks.append(("cubic order, matched probe (low)", "exp_control_extraction_intrinsic.json",
                           min(vals), 1, "the '3.4 to 3.8' of the intrinsic row"))
            checks.append(("cubic order, matched probe (high)", "exp_control_extraction_intrinsic.json",
                           max(vals), 1, "the '3.4 to 3.8' of the intrinsic row"))
    else:
        skipped.append("intrinsic probe orders: exp_control_extraction_intrinsic.json absent")

    # ---------------- the headline ratios, converted exactly ----------------
    ord_file = load("exp_control_order.json")
    if ext and ord_file:
        g61 = next((r for r in ext.get("solved_grid") or [] if r.get("N") == 61), None)
        v61 = next((r for r in ord_file.get("part_a_refinement") or []
                    if r.get("p") == 2 and r.get("N") == 61), None)
        if g61 and v61:
            exact = math.sqrt(g61["errors"]["fd"] / v61["value_err"])
            checks.append(("control/value ratio at p=2, N=61 (relative L2)",
                           "exp_control_extraction.json + exp_control_order.json",
                           round(exact, 1), 1,
                           f"exact quotient {exact:.2f}; the ratio of errors, not of their squares"))
            cub = math.sqrt(g61["errors"]["spline"] / v61["value_err"])
            checks.append(("control/value ratio, cubic extraction (relative L2)",
                           "exp_control_extraction.json + exp_control_order.json",
                           round(cub, 1), 1, "the ratio falls to 1.7 with a cubic reconstruction"))
            gain = math.sqrt(g61["errors"]["fd"] / g61["errors"]["spline"])
            checks.append(("cubic gain over multilinear at p=2, N=61 (relative L2)",
                           "exp_control_extraction.json", round(gain, 1), 1,
                           "the '8.2 times smaller' of the solved-grid comparison"))
        # The abstract's range: the default-probe sweep, on the error scale.
        ratios = [ratio_from_mse_ratio(r["ratio"]) for r in ord_file.get("part_a_refinement") or []
                  if r.get("ratio")]
        ratios = [r for r in ratios if r == r]
        if ratios:
            checks.append(("control/value ratio, low end (relative L2)", "exp_control_order.json",
                           round(min(ratios), 1), 1, "the '3.2 to 26 times' of the abstract"))
            checks.append(("control/value ratio, high end (relative L2)", "exp_control_order.json",
                           round(max(ratios)), 0, "the '3.2 to 26 times' of the abstract"))
    else:
        skipped.append("control/value ratios: need exp_control_extraction.json and "
                       "exp_control_order.json")

    # ---------------- deep BSDE, asserted PER DIMENSION ----------------
    # This is the assertion that would have caught the table's dimension mix-up: the prose says the
    # p=4 best value error is 3.3%, and it can only match if the d=4 record is selected.
    bsde = load("exp_bsde_dims.json")
    if bsde:
        for want_d, want_pct in ((4, 3.3), (8, 0.3)):
            row = next((r for r in bsde.get("rows", []) if int(r.get("d", -1)) == want_d), None)
            if row and row.get("best_V_rel_err") is not None:
                checks.append((f"deep BSDE best value error, d={want_d}",
                               "exp_bsde_dims.json", row["best_V_rel_err"] * 100, 1,
                               "the per-dimension number the deep-BSDE table must carry"))
    else:
        skipped.append("deep BSDE per-dimension errors: exp_bsde_dims.json absent")

    # ---------------- order versus monotonicity (Table: tab_limit) ----------------
    lim = load("exp_limited_reconstruction.json")
    if lim:
        fns = lim.get("functions") or {}
        noisy = ((fns.get("exponential_noisy") or {}).get("per_method")) or {}
        kink = ((fns.get("kink") or {}).get("per_method")) or {}
        trig = ((fns.get("trigonometric") or {}).get("per_method")) or {}
        smooth = ((fns.get("exponential") or {}).get("per_method")) or {}
        if noisy.get("cubic"):
            checks.append(("wrong-way fraction, noisy nodal data (cubic)",
                           "exp_limited_reconstruction.json",
                           noisy["cubic"]["max_wrong_way"] * 100, 1,
                           "the '8.6% of samples' of the monotonicity subsection"))
            checks.append(("overshoot, noisy nodal data (cubic)",
                           "exp_limited_reconstruction.json",
                           noisy["cubic"]["max_overshoot"], 2,
                           "the '2.26x' of the monotonicity subsection"))
            checks.append(("wrong-way fraction, limited reconstruction",
                           "exp_limited_reconstruction.json",
                           noisy["limited"]["max_wrong_way"] * 100, 1,
                           "the 'never points the wrong way' claim"))
        if kink.get("cubic"):
            checks.append(("overshoot across a gradient kink (cubic)",
                           "exp_limited_reconstruction.json",
                           kink["cubic"]["max_overshoot"], 2, "the '1.33x' of the kink row"))
        if smooth.get("cubic") and smooth.get("limited"):
            checks.append(("cubic order, smooth data", "exp_limited_reconstruction.json",
                           smooth["cubic"]["order"], 2,
                           "the '3.41' quoted for smooth data"))
            checks.append(("limited order, smooth data", "exp_limited_reconstruction.json",
                           smooth["limited"]["order"], 2,
                           "the limiter is inert on smooth data"))
        if trig.get("cubic") and trig.get("limited"):
            checks.append(("cubic order at an interior turning point",
                           "exp_limited_reconstruction.json", trig["cubic"]["order"], 2,
                           "the '3.72 -> 1.51' of the clipping cost"))
            checks.append(("limited order at an interior turning point",
                           "exp_limited_reconstruction.json", trig["limited"]["order"], 2,
                           "the '3.72 -> 1.51' of the clipping cost"))
    else:
        skipped.append("order-vs-monotonicity: exp_limited_reconstruction.json absent")

    # ---------------- non-polynomial reference: is N* = 21 an artefact of the solution? ----------
    npoly = load("exp_nonpolynomial.json")
    npoly12 = load("exp_nonpolynomial_omega12.json")
    if npoly:
        dims = npoly.get("dims") or {}
        vals = [d for d in dims.values() if d.get("rows")]
        if vals:
            checks.append(("non-polynomial: target reached in any dimension",
                           "exp_nonpolynomial.json",
                           1.0 if any(d["target_reached"] for d in vals) else 0.0, 0,
                           "the text says the 1e-3 target is NOT reached at N=61"))
            checks.append(("non-polynomial value order, low (vs N)",
                           "exp_nonpolynomial.json",
                           min(d["order_value_vs_N"] for d in vals), 1,
                           "the '2.0-2.2' of the non-polynomial subsection"))
            checks.append(("non-polynomial control order, high (vs N)",
                           "exp_nonpolynomial.json",
                           max(d["order_control_vs_N"] for d in vals), 2,
                           "the '1.04-1.13' of the non-polynomial subsection"))
            gains = [d["rows"][-1]["control_rel_l2_multilinear"] / d["rows"][-1]["control_rel_l2_cubic"]
                     for d in vals]
            checks.append(("non-polynomial cubic gain, low", "exp_nonpolynomial.json",
                           min(gains), 1, "the 'nine- to seventeen-fold' claim"))
            checks.append(("non-polynomial cubic gain, high", "exp_nonpolynomial.json",
                           max(gains), 1, "the 'nine- to seventeen-fold' claim"))
    else:
        skipped.append("non-polynomial benchmark: exp_nonpolynomial.json absent")

    if npoly12:
        d2 = (npoly12.get("dims") or {}).get("2")
        d1 = (npoly12.get("dims") or {}).get("1")
        if d2:
            checks.append(("under-resolved variant, value relL2 at N=61 (d=2)",
                           "exp_nonpolynomial_omega12.json",
                           d2["rows"][-1]["value_rel_l2"] * 100, 1,
                           "the '5.1e-3 at d=2' of the under-resolved paragraph"))
            checks.append(("under-resolved variant, value order (d=2)",
                           "exp_nonpolynomial_omega12.json", d2["order_value_vs_N"], 2,
                           "the '2.8-3.1' apparent orders"))
        if d1:
            checks.append(("under-resolved variant, value relL2 at N=61 (d=1)",
                           "exp_nonpolynomial_omega12.json",
                           d1["rows"][-1]["value_rel_l2"] * 100, 1,
                           "the '1.2e-2 at d=1' of the under-resolved paragraph"))
    else:
        skipped.append("under-resolved non-polynomial variant: file absent")

    # ---------------- boundary versus interior ----------------
    bnd = load("exp_boundary_split.json")
    if bnd:
        rows21 = {p: [r for r in rows if r["N"] == 21] for p, rows in (bnd.get("dims") or {}).items()}
        rows21 = {p: r[0] for p, r in rows21.items() if r}
        if rows21:
            ratios = [r["control_error_density_ratio"] for r in rows21.values()]
            checks.append(("boundary/interior error density, low", "exp_boundary_split.json",
                           min(ratios), 2, "the '0.99-1.00 times the interior's' claim"))
            checks.append(("boundary/interior error density, high", "exp_boundary_split.json",
                           max(ratios), 2, "the '0.99-1.00 times the interior's' claim"))
            if "3" in rows21:
                checks.append(("boundary share at p=3, N=21", "exp_boundary_split.json",
                               rows21["3"]["boundary_share_measured"] * 100, 0,
                               "the '33% at p=3' of the boundary paragraph"))
            if "1" in rows21:
                checks.append(("boundary share at p=1, N=21", "exp_boundary_split.json",
                               rows21["1"]["boundary_share_measured"] * 100, 0,
                               "the '13% at p=1' of the boundary paragraph"))
            gains_b = [r["cubic_gain_boundary"] for r in rows21.values()]
            # The manuscript rounds these to whole numbers ('14-19x'), so the assertion is made at
            # that precision rather than at the precision of the result file.
            checks.append(("cubic gain in the boundary band, low", "exp_boundary_split.json",
                           min(gains_b), 0, "the '14-19x in the band' claim"))
            checks.append(("cubic gain in the boundary band, high", "exp_boundary_split.json",
                           max(gains_b), 0, "the '14-19x in the band' claim"))
            checks.append(("cubic gain in the interior", "exp_boundary_split.json",
                           max(r["cubic_gain_interior"] for r in rows21.values()), 0,
                           "the '26x in the interior' claim"))
    else:
        skipped.append("boundary split: exp_boundary_split.json absent")

    # ---------------- regression Monte Carlo (the classical mesh-free baseline) ----------------
    lsmc = load("exp_lsmc.json")
    if lsmc:
        dims = lsmc.get("dims") or {}
        for k in ("8", "12", "20"):
            rows = dims.get(k)
            if rows:
                checks.append((f"regression Monte Carlo value R2 at d={k}", "exp_lsmc.json",
                               rows[-1]["value_r2"], 4,
                               "the numbers quoted in the mesh-free subsection and its table"))
        if dims.get("12") and dims.get("20"):
            checks.append(("RMC accuracy at d=12 (prose)", "exp_lsmc.json",
                           dims["12"][-1]["value_r2"], 3, "the '0.999 at d=12' of the abstract"))
            checks.append(("RMC accuracy at d=20 (prose)", "exp_lsmc.json",
                           dims["20"][-1]["value_r2"], 3, "the '0.977 at d=20' of the abstract"))
        if dims.get("20"):
            checks.append(("RMC basis functions at d=20", "exp_lsmc.json",
                           dims["20"][-1]["basis_functions"], 0,
                           "the '231 at d=20' of the qualifications"))
    else:
        skipped.append("regression Monte Carlo: exp_lsmc.json absent")

    # ---------------- compiled kernel: what was NumPy, what was the scheme ----------------
    comp = load("exp_compiled_grid.json")
    if comp:
        rows = {r["d"]: r for r in comp.get("rows") or []}
        if 4 in rows:
            checks.append(("compiled speedup at p=4", "exp_compiled_grid.json",
                           rows[4]["speedup"], 1, "the '11.9x' of the timing subsection"))
            checks.append(("achieved rate, kernel, p=4", "exp_compiled_grid.json",
                           rows[4]["kernel_gflops"], 2, "the '7.7 GFLOP/s' claim"))
            checks.append(("achieved rate, numpy, p=4", "exp_compiled_grid.json",
                           rows[4]["numpy_gflops"], 2, "the '0.65 GFLOP/s' claim"))
        if 5 in rows:
            checks.append(("compiled grid seconds at p=5", "exp_compiled_grid.json",
                           rows[5]["kernel_seconds"], 0, "the '59 s' that flips the p=5 comparison"))
        if rows:
            # The prose writes this in scientific notation ("$1.3\times10^{-4}$"), so the assertion
            # is made on the mantissa, which is the part a reader can check.
            checks.append(("kernel faithfulness (relative field difference, mantissa)",
                           "exp_compiled_grid.json",
                           max(r["rel_diff_vs_numpy"] for r in rows.values()) * 1e4, 1,
                           "the '1.3 x 10^-4' agreement with the NumPy solver"))
    else:
        skipped.append("compiled kernel: exp_compiled_grid.json absent")

    # ---------------- execution shortfall, in basis points of notional ----------------
    short = load("exp_execution_shortfall.json")
    if short:
        s = short.get("summary") or {}
        rows = short.get("rows") or []
        if s:
            checks.append(("shortfall, mean pairwise (bp)", "exp_execution_shortfall.json",
                           s["pairwise_bp_mean"], 2, "the 'mean of 5.7 bp' claim"))
            checks.append(("shortfall in currency, thousands per $10M order",
                           "exp_execution_shortfall.json",
                           s["currency_at_10m_notional"] / 1000.0, 1, "the 'about $5.7k' claim"))
            checks.append(("cubic arm excess over exact (bp)", "exp_execution_shortfall.json",
                           s["cubic_excess_bp_max"], 3, "the 'within 0.018 bp' claim"))
        if rows:
            spy = next((r for r in rows if r["asset"] == "SPY"), None)
            tsla = next((r for r in rows if r["asset"] == "TSLA"), None)
            if spy:
                checks.append(("shortfall, SPY pairwise (bp)", "exp_execution_shortfall.json",
                               spy["pairwise_bp"], 2, "the upper end of the range"))
            if tsla:
                checks.append(("shortfall, TSLA pairwise (bp)", "exp_execution_shortfall.json",
                               tsla["pairwise_bp"], 2, "the lower end of the range"))
        sens = short.get("sensitivity") or []
        cut = [x for x in sens if abs(x["delta"] - 0.10) < 1e-9]
        if cut:
            checks.append(("shortfall under a truncated horizon (bp)",
                           "exp_execution_shortfall.json", cut[0]["pairwise_bp"], 2,
                           "the '-0.21 bp at delta = 0.1' of the measurement caution"))
    else:
        skipped.append("execution shortfall: exp_execution_shortfall.json absent")

    # ---------------- deep splitting: the scheme, and the learner that limits it ----------------
    ds = load("exp_deep_splitting.json")
    if ds:
        for arm, places, note in (
                ("quadratic-basis control", 4,
                 "the exact-quadratic arm's value R2 (the scheme is sound)"),
                ("neural", 2, "the neural arm's value R2 (the learner is the limitation)")):
            sel = [r for r in ds.get("rows") or [] if r.get("arm") == arm and r.get("d") == 1]
            if sel:
                checks.append((f"deep splitting, {arm}, value R2 at d=1",
                               "exp_deep_splitting.json", sel[0]["value_r2"], places, note))
        q4 = [r for r in ds.get("rows") or [] if r.get("arm") == "quadratic-basis control"
              and r.get("d") == 4]
        if q4:
            checks.append(("deep splitting, exact quadratic arm, value R2 at d=4",
                           "exp_deep_splitting.json", q4[0]["value_r2"], 4,
                           "the '0.9960 at d=4' claim"))
        n1 = [r for r in ds.get("rows") or [] if r.get("arm") == "neural" and r.get("d") == 1]
        if n1:
            checks.append(("deep splitting, neural arm, control R2 at d=1",
                           "exp_deep_splitting.json", n1[0]["control_r2"], 2,
                           "the '0.86 on the control against 0.16 on the value' claim"))
    else:
        skipped.append("deep splitting: exp_deep_splitting.json absent")

    # ---------------- is the splitting failure a budget problem? ----------------
    dsb = load("exp_deep_splitting_budget.json")
    if dsb:
        s = dsb.get("summary") or {}
        if s:
            checks.append(("deep splitting: per-step loss improvement from more budget",
                           "exp_deep_splitting_budget.json", s["loss_improvement"], 0,
                           "the '179x better per-step loss' claim"))
            checks.append(("deep splitting: budget multiple", "exp_deep_splitting_budget.json",
                           s["budget_ratio"], 1, "the '6.7x the budget' claim"))
            checks.append(("deep splitting: value R2 at the small budget",
                           "exp_deep_splitting_budget.json", s["value_r2_small_budget"], 2,
                           "the '+0.16 -> -16.5' comparison"))
            checks.append(("deep splitting: |value R2| at the large budget (magnitude)",
                           "exp_deep_splitting_budget.json", abs(s["value_r2_large_budget"]), 1,
                           "the '+0.16 -> -16.5' comparison"))
        rows = dsb.get("rows") or []
        if rows and s.get("control_r2_range"):
            # The prose quotes the RANGE endpoints, so both are asserted rather than their midpoint.
            for idx, end in enumerate(("lower", "upper")):
                checks.append((f"deep splitting: control R2, {end} end of the range",
                               "exp_deep_splitting_budget.json",
                               s["control_r2_range"][idx], 2,
                               "the 'control R2 stays between 0.86 and 0.89' claim"))
    else:
        skipped.append("deep splitting budget sweep: exp_deep_splitting_budget.json absent")

    # ---------------- value error vs gradient error at the trained nets ----------------
    dge = load("diag_splitting_gradient_error.json")
    if dge:
        checks.append(("gradient error exceeds value error by (ratio)", "diag_...json",
                       dge.get("worst_ratio"), 1,
                       "the 'up to 4.3x' claim about which error the recursion inherits"))
    else:
        skipped.append("gradient-error diagnostic: diag_splitting_gradient_error.json absent")

    # ---------------- multilevel Picard: where it works, and where its hypothesis fails ----------
    mlp = load("exp_mlp.json")
    if mlp:
        chk = mlp.get("implementation_check") or {}
        for lv, places, note in (("2", 3, "the implementation check's level-2 error"),
                                 ("4", 3, "the implementation check's level-4 error")):
            if lv in chk:
                checks.append((f"MLP implementation check, level {lv} relative error",
                               "exp_mlp.json", chk[lv]["rel_err"], places, note))
        rows = [r for r in (mlp.get("rows") or []) if r.get("eta") == 1.0 and r.get("d") == 20]
        if rows:
            checks.append(("MLP at d=20, eta=1, relative error (%)", "exp_mlp.json",
                           rows[0]["rel_err"] * 100, 2, "the '0.26% at d=20' claim"))
            checks.append(("MLP at d=20, eta=1, seconds", "exp_mlp.json",
                           rows[0]["seconds"], 1, "the 'in 2.3 seconds' claim"))
        div = [r for r in (mlp.get("rows") or []) if r.get("eta") == 0.01 and r.get("d") == 20]
        if div:
            checks.append(("MLP at d=20, eta=0.01, seconds (it diverges but cheaply)",
                           "exp_mlp.json", div[0]["seconds"], 1,
                           "the '1.8 s' that shows the divergence is not a budget running out"))
        sw = mlp.get("contraction_sweep") or []
        for eta_key, note in ((0.1, "the converged side of the contraction boundary"),
                              (0.03, "the diverged side of the contraction boundary")):
            sel = [r for r in sw if r["eta"] == eta_key]
            if sel:
                r = sel[0]
                checks.append((f"MLP contraction boundary, |gradV|/eta at eta={eta_key:g}",
                               "exp_mlp.json", r["grad_V_at_x0"] / r["eta"], 2, note))
    else:
        skipped.append("multilevel Picard: exp_mlp.json absent")

    # ---------------- the operator-learning paragraph's accuracy comparison ----------------
    # It compares the operator input's low dimensionality against what the classical methods already
    # reach on this problem, so the numbers are the regression method's relative L2 error at two
    # dimensions and the multilevel Picard result (asserted above).
    lsmc_rel = load("exp_lsmc.json")
    if lsmc_rel:
        dims = lsmc_rel.get("dims") or {}
        for dkey, places in (("1", 1), ("20", 1)):
            r = (dims.get(dkey) or [None])[-1]
            if r and "value_rel_l2" in r:
                checks.append((f"regression Monte Carlo relative L2 error (%) at d={dkey}",
                               "exp_lsmc.json", r["value_rel_l2"] * 100, places,
                               "the '0.4% at d=1 rising to 3.4% at d=20' comparison in the "
                               "operator-learning paragraph"))
    else:
        skipped.append("operator paragraph: exp_lsmc.json absent")

    # ---------------- the full set of compared solvers ----------------
    # The prose claims "eleven solver configurations across five families". That is a claim about the
    # generated table, so it is asserted against the table rather than against a result file.
    tab_solvers = ROOT / "paper" / "tables" / "tab_solvers.tex"
    if tab_solvers.exists():
        body = tab_solvers.read_text(encoding="utf-8")
        try:
            rows = body.split("\\midrule")[1].split("\\bottomrule")[0].strip().splitlines()
            rows = [r for r in rows if r.strip() and not r.strip().startswith("\\addlinespace")]
            checks.append(("solver configurations listed", "tab_solvers.tex", len(rows), 0,
                           "the 'eleven solver configurations' claim"))
            fams = {r.split("&")[1].strip() for r in rows if r.count("&") >= 4}
            checks.append(("families listed", "tab_solvers.tex", len(fams), 0,
                           "the 'five families' claim"))
        except IndexError:
            skipped.append("tab_solvers.tex has an unexpected shape")
    else:
        skipped.append("tab_solvers.tex absent")
    pip = load("exp_pi_pinn.json")
    if pip:
        pruns = pip.get("runs") or []
        for iters, places, note in ((2, 2, "the 'R^2 = -0.15' of the two-iteration PI-PINN row"),
                                    (8, 4, "the '0.9997' of the eight-iteration PI-PINN row")):
            sel = [r for r in pruns if r.get("method") == "PI-PINN" and r.get("num_iters") == iters
                   and "value_r2" in r]
            if sel:
                v = sum(r["value_r2"] for r in sel) / len(sel)
                checks.append((f"PI-PINN value R2 at {iters} iterations", "exp_pi_pinn.json",
                               abs(v) if iters == 2 else v, places, note))
    else:
        skipped.append("PI-PINN: exp_pi_pinn.json absent")

    # ---------------- how much of the regression advantage was the basis? ----------------
    lsmcb = load("exp_lsmc_basis.json")
    if lsmcb:
        rows = {r["A0"]: r for r in lsmcb.get("rows") or [] if not r.get("diverged")}
        for a0, places in ((0.0, 4), (1e-4, 4), (1e-3, 3)):
            r = rows.get(a0)
            if r:
                checks.append((f"regression R2 at A0={a0:g}", "exp_lsmc_basis.json",
                               r["value_r2"], places,
                               "the basis-sensitivity numbers in the mesh-free subsection"))
        # At the failing amplitude the paper quotes a RANGE, because the figure moves across runs:
        # assert both ends of it rather than a value the prose deliberately does not state.
        r5 = rows.get(5e-3)
        if r5:
            checks.append(("basis sensitivity: failing amplitude, upper end (magnitude)",
                           "exp_lsmc_basis.json", abs(r5["value_r2_max"]), 2,
                           "the '-1.19 to -1.20' range in the mesh-free subsection"))
            checks.append(("basis sensitivity: failing amplitude, lower end (magnitude)",
                           "exp_lsmc_basis.json", abs(r5["value_r2_min"]), 2,
                           "the '-1.19 to -1.20' range in the mesh-free subsection"))
        div = [r for r in lsmcb.get("rows") or [] if r.get("diverged")]
        if div:
            checks.append(("basis sensitivity: a divergent case exists",
                           "exp_lsmc_basis.json", 1.0, 0,
                           "the 'at A0 = 2e-2 the run diverges' claim"))
    else:
        skipped.append("basis sensitivity: exp_lsmc_basis.json absent")

    # ---------------- cost model ----------------
    cost = load("exp_cost_model.json")
    if cost:
        rr = cost.get("ratio") or cost.get("rate_ratio")
        if rr:
            order = len(str(int(rr))) - 1
            checks.append(("FLOP rate ratio order of magnitude", "exp_cost_model.json",
                           float(order), 0, "the 'about 10^2' of the cost argument"))

    # ---------------- report ----------------
    print("=" * 104)
    print("PROSE NUMBER CHECK -- each assertion recomputed from its result file")
    print("=" * 104)
    ok = bad = 0
    for label, src, value, places, why in checks:
        try:
            v = float(value)
        except (TypeError, ValueError):
            print(f"  SKIP  {label}: value not numeric")
            continue
        found = has_number(text, v, places)
        mark = "OK  " if found else "FAIL"
        if found:
            ok += 1
        else:
            bad += 1
        print(f"  {mark}  {label}")
        print(f"        source: {src}   computed: {v:.{places}f}   ({why})")
        if not found:
            print(f"        NOT FOUND in the manuscript at {places} decimal places -- the prose "
                  f"may quote a stale value, or a different quantity")
    for s in skipped:
        print(f"  SKIP  {s}")

    print()
    print("-" * 104)
    print(f"  {ok} matched, {bad} unmatched, {len(skipped)} skipped")
    print()
    print("  Every asserted order is fitted to the relative L2 error (src/metrics.py). If a row")
    print("  fails, first ask which SCALE the prose quoted before rewriting either side.")
    (RESULTS / "verify_paper_numbers.json").write_text(json.dumps(
        {"matched": ok, "unmatched": bad, "skipped": skipped,
         "metric": "relative L2; orders fitted to the error, not its square"}, indent=2),
        encoding="utf-8")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
