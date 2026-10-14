"""
Figure generation for the CISP-BMEI paper.

Every figure is built only from the corrected runs in `results/`. Nothing is copied from
the original thesis, and the six claims in the thesis that have no code provenance
(1D/2D accuracy tables, boundary-condition table, trading-behaviour table, the Spearman
rho = 1.0 claim, and the memory figures) are not reproduced anywhere.

Figures are sized for IEEE two-column: single column = 3.5 in, full width = 7.16 in.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
FIGDIR = Path("figures")

# IEEEtran-friendly defaults
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "lines.markersize": 4,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.4,
})

COLORS = {
    "Finite Difference": "#1f77b4",
    "Policy Iteration": "#ff7f0e",
    "PINN": "#2ca02c",
}
MARKERS = {"Finite Difference": "o", "Policy Iteration": "s", "PINN": "^"}

# Training price domain of the audited baseline (shaded in the OOD figure).
PRICE_LO, PRICE_HI = 20.0, 30.0


def save(fig, name: str) -> None:
    """Save as vector PDF (for LaTeX) and 300 dpi PNG (for preview).

    The creation timestamp is stripped so re-running the pipeline reproduces the files
    byte-for-byte, not merely pixel-for-pixel. Without this, matplotlib embeds the current
    date and every regeneration produces a different hash for identical content -- which
    defeats hash-based provenance checking.
    """
    FIGDIR.mkdir(parents=True, exist_ok=True)
    meta = {"Date": None, "Creator": "rework/src/make_figures.py", "Producer": None}
    for ext in ("pdf", "png"):
        fig.savefig(FIGDIR / f"{name}.{ext}", metadata=meta)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf|.png")


def _mean_sd(values):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    if v.size == 0:
        return np.nan, np.nan
    return float(v.mean()), float(v.std(ddof=1)) if v.size > 1 else 0.0


# ======================================================================================
def fig_accuracy_and_cost(dim_path: Path, grid_path: Path | None = None) -> None:
    """Single column, two panels: accuracy vs solve time, and cost vs dimension.

    These were two separate figures, but they answer one question between them and both are
    single-column-sized, so stacking them saves a figure's worth of space without losing
    content.
    """
    if not dim_path.exists():
        print(f"  skip accuracy/cost: {dim_path} missing")
        return
    data = json.loads(dim_path.read_text())
    runs = [r for r in data["runs"] if r.get("feasible", False)]

    groups: dict[tuple[str, int], dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        groups[(r["method"], r["p"])]["t"].append(r.get("solve_time_s"))
        groups[(r["method"], r["p"])]["r2"].append(r.get("value_r2"))

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6))
    YLO, YHI = 0.90, 1.005

    # ---- (a) accuracy against solve time -------------------------------------------
    ax = axes[0]
    offscale, seen = [], set()
    for (method, p), vals in sorted(groups.items()):
        t, tsd = _mean_sd(vals["t"])
        r2, r2sd = _mean_sd(vals["r2"])
        if not np.isfinite(t) or not np.isfinite(r2):
            continue
        col, mk = COLORS.get(method, "k"), MARKERS.get(method, "o")
        lab = method if method not in seen else None
        seen.add(method)
        if r2 < YLO:
            ax.plot([t], [YLO + 0.002], marker="v", color=col, markersize=4.5, label=lab)
            offscale.append((p, r2))
        else:
            ax.errorbar(t, r2, xerr=tsd, yerr=r2sd, fmt=mk, color=col, alpha=0.9,
                        capsize=2, lw=0.8, label=lab)
            ax.annotate(f"$p$={p}", (t, r2), textcoords="offset points", xytext=(3, -7),
                        fontsize=5.5, color=col)
    if offscale:
        vals = ", ".join(f"$p$={p}: {r2:.0f}" for p, r2 in offscale)
        ax.text(0.04, 0.06, f"$\\blacktriangledown$ solver failed ($R^2$)\n{vals}",
                transform=ax.transAxes, fontsize=5, color="0.25", ha="left", va="bottom")
    ax.set_xscale("log")
    ax.set_ylim(YLO, YHI)
    ax.set_xlabel("solve time (s, log scale)", fontsize=7)
    ax.set_ylabel(r"value $R^2$", fontsize=7)
    ax.legend(loc="center left", frameon=False, fontsize=5.5, handletextpad=0.3)
    ax.set_title("(a) accuracy vs solve time", fontsize=7)

    # ---- (b) cost against dimension, accuracy-matched -------------------------------
    ax2 = axes[1]
    if grid_path and grid_path.exists():
        g = json.loads(grid_path.read_text())
        summ = g.get("summary", {})
        gp = sorted(int(k) for k in summ)
        gpp = [p for p in gp if summ[str(p)].get("time_s")]
        ax2.plot(gpp, [summ[str(p)]["time_s"] for p in gpp], marker="o",
                 color=COLORS["Finite Difference"], lw=1.2, label="grid, matched")
    ps = sorted(groups.get("PINN", {}) if isinstance(groups, dict) else [])
    ps = sorted({p for (m, p) in groups if m == "PINN"})
    if ps:
        pin_t = [_mean_sd(groups[("PINN", p)]["t"])[0] for p in ps]
        pin_r2 = [float(np.mean(groups[("PINN", p)]["r2"])) for p in ps]
        ax2.plot(ps, pin_t, marker="^", color=COLORS["PINN"], lw=1.2, label="PINN")
        bad = [p for p, r in zip(ps, pin_r2) if r < 0.99]
        if bad:
            ax2.plot(bad, [pin_t[ps.index(p)] for p in bad], marker="x", ls="none",
                     color="#d62728", markersize=7, label="PINN inaccurate")
    ax2.set_yscale("log")
    ax2.set_xticks(sorted({r["p"] for r in runs}))
    ax2.set_xlabel("state dimension $p$", fontsize=7)
    ax2.set_ylabel("solve time (s)", fontsize=7)
    ax2.legend(loc="lower right", frameon=False, fontsize=5.5, handletextpad=0.3)
    ax2.set_title("(b) cost vs dimension, accuracy-matched", fontsize=7)

    fig.tight_layout()
    save(fig, "fig1_accuracy_and_cost")


def fig_cost_scaling(dim_path: Path, grid_path: Path | None = None) -> None:
    """Two panels: measured cost vs p, and the accuracy-matched comparison."""
    if not dim_path.exists():
        print(f"  skip cost-scaling: {dim_path} missing")
        return
    data = json.loads(dim_path.read_text())
    runs = [r for r in data["runs"] if r.get("feasible", False)]

    by_method: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    r2_by: dict[str, dict[int, list]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        by_method[r["method"]][r["p"]].append(r.get("solve_time_s"))
        r2_by[r["method"]][r["p"]].append(r.get("value_r2"))

    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.2))

    # ---- panel (a): cost vs p at the sweep's resolutions -----------------------------
    ax = axes[0]
    for method, per_p in sorted(by_method.items()):
        ps = sorted(per_p)
        means = [_mean_sd(per_p[p])[0] for p in ps]
        ax.plot(ps, means, marker=MARKERS.get(method, "o"), color=COLORS.get(method, "k"),
                label=method, lw=1.0)
    ax.set_yscale("log")
    ax.set_xlabel("state dimension $p$")
    ax.set_ylabel("solve time (s, log scale)")
    ax.set_xticks(sorted({r["p"] for r in runs}))
    ax.legend(loc="upper left", frameon=False)
    ax.set_title("(a) measured cost", fontsize=7)

    # ---- panel (b): accuracy-matched comparison --------------------------------------
    ax2 = axes[1]
    if grid_path and grid_path.exists():
        g = json.loads(grid_path.read_text())
        summ = g.get("summary", {})
        gp = sorted(int(k) for k in summ)
        gpp = [p for p in gp if summ[str(p)].get("time_s")]
        gt = [summ[str(p)]["time_s"] for p in gpp]
        ax2.plot(gpp, gt, marker="o", color=COLORS["Finite Difference"], lw=1.2,
                 label="grid, accuracy-matched")
    ps = sorted(by_method.get("PINN", {}))
    if ps:
        pin_t = [_mean_sd(by_method["PINN"][p])[0] for p in ps]
        pin_r2 = [float(np.mean(r2_by["PINN"][p])) for p in ps]
        ax2.plot(ps, pin_t, marker="^", color=COLORS["PINN"], lw=1.2, label="PINN")
        bad = [p for p, r in zip(ps, pin_r2) if r < 0.99]
        if bad:
            ax2.plot([p for p in bad], [pin_t[ps.index(p)] for p in bad], marker="x",
                     ls="none", color="#d62728", markersize=9,
                     label="PINN inaccurate ($R^2<0.99$)")
    ax2.set_yscale("log")
    ax2.set_xlabel("state dimension $p$")
    ax2.set_ylabel("solve time (s, log scale)")
    ax2.set_xticks(sorted({r["p"] for r in runs}))
    ax2.legend(loc="lower right", frameon=False, fontsize=6)
    ax2.set_title("(b) accuracy-matched", fontsize=7)

    fig.tight_layout()
    save(fig, "fig2_cost_scaling")


def fig_ood(ood_path: Path) -> None:
    """Single column: the spurious price-level dependence and its removal by rescaling."""
    if not ood_path.exists():
        print(f"  skip ood: {ood_path} missing")
        return
    data = json.loads(ood_path.read_text())
    conds = data.get("conditions", {})
    un = conds.get("unscaled_raw_price", [])
    rs = conds.get("price_rescaled_to_train_domain", [])
    if not un:
        print("  skip ood: no unscaled condition")
        return

    assets = [r["asset"] for r in un[0]["per_asset"]]
    prices = np.array([r["mean_price"] for r in un[0]["per_asset"]])
    order = np.argsort(prices)

    def agg(runs, key):
        out = []
        for i in range(len(assets)):
            vals = [r["per_asset"][i][key] for r in runs if i < len(r["per_asset"])]
            out.append(float(np.mean(vals)) if vals else np.nan)
        return np.array(out)

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6))

    ax = axes[0]
    u = agg(un, "grad_abs_mean")
    ax.plot(prices[order], u[order], "o-", color="#d62728", lw=1.0,
            label="raw price (as published)")
    if rs:
        r = agg(rs, "grad_abs_mean")
        ax.plot(prices[order], r[order], "s--", color="#1f77b4", lw=1.0,
                label="rescaled to training domain")
    ax.axvspan(PRICE_LO, PRICE_HI, color="0.85", zorder=0)
    ax.text(PRICE_LO + 0.5, ax.get_ylim()[1] * 0.5, "training\ndomain",
            fontsize=6, color="0.35", va="top")
    ax.set_yscale("log")
    ax.set_xlabel("asset price (USD)")
    ax.set_ylabel(r"mean $|\partial V/\partial S|$")
    ax.set_title("(a) local sensitivity vs price level", fontsize=7)
    ax.legend(loc="upper right", frameon=False, fontsize=6)

    ax = axes[1]
    for label, runs, col, mk in (
        ("raw price", un, "#d62728", "o"),
        ("rescaled", rs, "#1f77b4", "s"),
    ):
        if not runs:
            continue
        sp = [r.get("spearman_price_vs_abs_grad", np.nan) for r in runs]
        ax.bar(label, np.nanmean(sp), color=col, alpha=0.85, width=0.5)
        for i, s in enumerate(sp):
            ax.plot([label], [s], marker=mk, color="k", markersize=3)
    ax.axhline(0.0, color="k", lw=0.6)
    ax.set_ylabel(r"Spearman(price, $|\partial V/\partial S|$)")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("(b) rank dependence on price", fontsize=7)
    ax.set_xticks(range(2))
    ax.set_xticklabels(["raw price", "rescaled"], fontsize=7)

    fig.tight_layout()
    save(fig, "fig3_ood_no_trade")

    for cond, runs in (("unscaled", un), ("rescaled", rs)):
        if not runs:
            continue
        a = [r.get("spearman_price_vs_abs_grad", np.nan) for r in runs]
        print(f"  spearman(price, |dV/dS|) {cond:9s}: {np.nanmean(a):+.3f}")


def fig_ablation(abl_path: Path) -> None:
    """Full width: contribution of each defect to the reported performance."""
    if not abl_path.exists():
        print(f"  skip ablation: {abl_path} missing")
        return
    data = json.loads(abl_path.read_text())
    runs = data.get("runs", [])
    if not runs:
        print("  skip ablation: no runs")
        return

    order = ["as_published",
             "fix_D1_ordering", "fix_D2_same_bar_leak", "fix_D3_control_law",
             "fix_D3_control_sign", "fix_D4_dollar_scaling",
             "fix_D5_position_integration", "fix_D6_cost_units",
             "fix_D7_annualization", "fix_D8_input_scaling", "all_fixed"]
    labels = {
        "as_published": "as published",
        "fix_D1_ordering": "+ chronological order",
        "fix_D2_same_bar_leak": "+ lag position",
        "fix_D3_control_law": "+ control law (sign, $1/S$)",
        "fix_D3_control_sign": "+ control law (sign, $1/S$)",
        "fix_D4_dollar_scaling": "+ $1/S$ scaling",
        "fix_D5_position_integration": "+ integrate position",
        "fix_D6_cost_units": "+ cost units",
        "fix_D7_annualization": "+ annualisation",
        "fix_D8_input_scaling": "+ input scaling",
        "all_fixed": "all defects fixed",
    }

    agg: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in runs:
        a = r["agg"]
        for k in ("total_return", "sharpe_ratio", "annualized_volatility", "mean_abs_position"):
            agg[r["config"]][k].append(a.get(k, np.nan))

    names = [c for c in order if c in agg]
    y = np.arange(len(names))
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 3.0))

    for ax, key, xlabel, scale in (
        (axes[0], "total_return", "total return", 1.0),
        (axes[1], "sharpe_ratio", "Sharpe ratio", 1.0),
        (axes[2], "annualized_volatility", "annualised volatility", 1.0),
    ):
        m = [np.nanmean(agg[c][key]) * scale for c in names]
        s = [np.nanstd(agg[c][key], ddof=1) * scale if len(agg[c][key]) > 1 else 0.0 for c in names]
        cols = ["#d62728" if c == "as_published" else
                ("#2ca02c" if c == "all_fixed" else "#7f7f7f") for c in names]
        ax.barh(y, m, xerr=s, color=cols, alpha=0.85, height=0.7,
                error_kw={"lw": 0.6, "capsize": 2})
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([labels.get(c, c) for c in names])
        else:
            ax.set_yticklabels([])
        ax.axvline(0.0, color="k", lw=0.6)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
    fig.tight_layout()
    save(fig, "fig4_defect_ablation")


def main():
    print("generating figures from corrected results")
    fig_accuracy_and_cost(RESULTS / "exp_dimension.json",
                          RESULTS / "exp_grid_refinement_merged.json")
    fig_ood(RESULTS / "exp_ood.json")
    fig_ablation(RESULTS / "exp_ablation.json")


if __name__ == "__main__":
    main()
