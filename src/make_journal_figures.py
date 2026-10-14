"""
Journal figures, built only from result files that already exist.

Why these two, and why a separate script
----------------------------------------
The journal audit (journal_audit.py) counts zero figures in the manuscript. The conference version
was trimmed to its six-page camera-ready form and the figure went with the trim, which is fine for
a conference and not fine for a journal: a results paper with no figure asks the reader to
reconstruct every trend from a table.

The two figures produced here are the two the paper's own argument needs, and neither requires a new
experiment:

  fig_j1  THE HEADLINE MECHANISM. Control error against mesh spacing for the three reconstructions,
          with the measured convergence orders. This is the paper's main claim -- that the
          reconstruction order, not the value accuracy, governs control accuracy -- and it is the one
          result that cannot be read off a table, because the claim IS the slope.

  fig_j2  THE DEEP-BSDE STOPPING RULE. Value error and terminal residual on the same log axes,
          against training epoch, for the converged deep-BSDE runs, with the stopping threshold
          marked. This is the finding that makes the deep-BSDE baseline quotable at all: the
          residual descends monotonically while the value error bottoms out and then degrades, so
          training to a residual plateau overshoots. A table can give the numbers; only the figure
          shows that the two curves cross the way they do.

Everything is read from `results/`; nothing is recomputed and no number is typed in by hand, so a
figure cannot drift away from the result it depicts. Panels whose result file is absent are skipped
with a printed reason rather than silently omitted.

Sized for IEEE two-column: single column 3.5 in, full width 7.16 in.

Usage: python src/make_journal_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = Path("results")
FIGDIR = Path("figures")

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

# Labels for the three reconstructions, and the order each one is EXPECTED to show. The expected
# orders are not used to draw anything -- they are printed beside the measured ones so a reader can
# see the measurement agreeing or disagreeing with theory.
RECON = {
    "fd": ("Probing difference (2nd order)", "#1f77b4", "o"),
    "cell": ("Multilinear cell gradient (2nd order)", "#ff7f0e", "s"),
    "spline": ("Cubic spline derivative (4th order)", "#2ca02c", "^"),
}


def save(fig, name: str) -> None:
    """PDF for LaTeX plus a PNG preview; timestamps stripped so reruns are byte-identical."""
    FIGDIR.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        out = FIGDIR / f"{name}.{ext}"
        meta = {"CreationDate": None} if ext == "pdf" else {"Software": None}
        fig.savefig(out, metadata=meta)
        print(f"  wrote {out}")


def fig_control_order() -> bool:
    """Control error against mesh spacing, with measured convergence orders.

    Built from the SYNTHETIC tracks, not the exact-node tracks, and the distinction matters. On a
    quadratic the cubic spline is exact, so its errors sit at 1e-29: a convergence order fitted to
    those is a fit to machine noise (measured: -2.11). The synthetic tracks use functions that are
    not polynomials the spline can reproduce, which is the only setting in which a fourth-order
    claim means anything. The measured orders are read from the result file's own `orders` field
    rather than refitted here, so the figure annotates the same numbers the paper text quotes.
    """
    p = RESULTS / "exp_control_extraction.json"
    if not p.exists():
        print("  SKIP fig_j1: results/exp_control_extraction.json absent")
        return False
    d = json.loads(p.read_text(encoding="utf-8"))
    synth = d.get("synthetic") or []
    if not synth:
        print("  SKIP fig_j1: no synthetic tracks present")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})

    # --- left: one representative non-polynomial function, three reconstructions ---
    rep = next((s for s in synth if s["function"] == "exponential"), synth[0])
    ax = axes[0]
    for key, (label, color, marker) in RECON.items():
        tr = rep["tracks"].get(key)
        if not tr:
            continue
        h = np.array([pt[0] for pt in tr])
        e = np.array([pt[1] for pt in tr])
        o = (rep["orders"].get(key) or {}).get("order")
        shown = f"{label.split(' (')[0]} — order {o:.2f}" if o else label
        ax.loglog(h, e, marker=marker, color=color, lw=1.0, label=shown)
        hh = np.array([h[0], h[-1]])
        ax.loglog(hh, e[0] * (hh / h[0]) ** 2, ls=":", color=color, lw=0.7, alpha=0.7)
    ax.set_xlabel("mesh spacing $h$")
    ax.set_ylabel("control error")
    ax.set_title(f"Reconstruction order sets control error\ntest function: {rep['function']}",
                 fontsize=7.5)
    ax.legend(loc="lower right", framealpha=0.9)

    # --- right: is the order a property of the reconstruction, or of the example? ---
    ax = axes[1]
    funcs = [s["function"] for s in synth]
    keys = [k for k in RECON if any(s["orders"].get(k) for s in synth)]
    x = np.arange(len(funcs))
    w = 0.8 / max(len(keys), 1)
    for j, key in enumerate(keys):
        vals, xs = [], []
        for i, s in enumerate(synth):
            o = (s["orders"].get(key) or {}).get("order")
            if o:
                vals.append(o)
                xs.append(i + (j - (len(keys) - 1) / 2) * w)
        ax.bar(xs, vals, width=w, color=RECON[key][1], label=RECON[key][0].split(" (")[0])
    ax.axhline(2, ls="--", lw=0.8, color="0.35")
    ax.axhline(4, ls="--", lw=0.8, color="0.35")
    ax.text(len(funcs) - 0.5, 2.15, "order 2", fontsize=6.5, color="0.35", ha="right")
    ax.text(len(funcs) - 0.5, 4.15, "order 4", fontsize=6.5, color="0.35", ha="right")
    ax.set_xticks(x)
    ax.set_xticklabels(funcs, rotation=20, ha="right", fontsize=6.5)
    ax.set_ylabel("measured order")
    ax.set_ylim(0, 5.4)
    ax.set_title("Same order across test functions", fontsize=7.5)
    ax.legend(loc="upper left", framealpha=0.9, fontsize=6.5)

    fig.tight_layout()
    save(fig, "fig_j1_control_order")
    plt.close(fig)
    for s in synth:
        got = ", ".join(f"{k}={(s['orders'][k] or {}).get('order', float('nan')):.2f}"
                        for k in keys if s["orders"].get(k))
        print(f"    {s['function']}: {got}")
    return True


def fig_deep_bsde_stopping() -> bool:
    """Value error and terminal residual against epoch, with the stopping threshold marked."""
    series = []
    for fname, label in [("exp_bsde_ns400_conv.json", "n_steps=400"),
                         ("exp_bsde_ns200_conv.json", "n_steps=200"),
                         ("exp_bsde_ns50_conv.json", "n_steps=50")]:
        p = RESULTS / fname
        if not p.exists():
            continue
        rows = json.loads(p.read_text(encoding="utf-8")).get("rows", [])
        if not rows:
            continue
        tr = rows[0].get("trace") or []
        if len(tr) >= 3:
            series.append((label, tr, rows[0]))
    if not series:
        print("  SKIP fig_j2: no deep-BSDE convergence traces present")
        return False

    fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.5))
    colors = {"n_steps=400": "#1f77b4", "n_steps=200": "#ff7f0e", "n_steps=50": "#2ca02c"}

    # Left: the terminal residual, which DOES converge.
    ax = axes[0]
    for label, tr, _ in series:
        ep = [t["epoch"] for t in tr]
        fl = [t["terminal_rmse"] for t in tr]
        ax.loglog(ep, fl, marker="o", ms=3, lw=1.0, color=colors[label], label=label)
    # Exact floors are horizontal references: what the scheme itself could deliver at that dt.
    for label, _, row in series:
        ex = row.get("exact_floor_abs")
        if ex:
            ax.axhline(ex, ls=":", lw=0.7, color=colors[label], alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("terminal residual (absolute)")
    ax.set_title("Residual converges monotonically", fontsize=7.5)
    ax.legend(loc="upper right", framealpha=0.9)
    ax.text(0.03, 0.06, "dotted: exact-solution floor at that $n_{steps}$", transform=ax.transAxes,
            fontsize=6.5, color="0.35")

    # Right: the value error, which does NOT.
    ax = axes[1]
    for label, tr, _ in series:
        ep = [t["epoch"] for t in tr]
        ve = [t["value_rel_err"] for t in tr]
        ax.semilogx(ep, ve, marker="o", ms=3, lw=1.0, color=colors[label], label=label)
        k = int(np.argmin(ve))
        ax.plot(ep[k], ve[k], marker="*", ms=9, color=colors[label], zorder=5,
                markeredgecolor="k", markeredgewidth=0.3)
    # The stopping threshold is the residual level at which the value error bottoms out. It is read
    # from the traces, not assumed: take the median residual at the value minima.
    vals = []
    for _, tr, _ in series:
        k = int(np.argmin([t["value_rel_err"] for t in tr]))
        vals.append(tr[k]["terminal_rmse"])
    thr = float(np.median(vals))
    ax.set_xlabel("epoch")
    ax.set_ylabel("relative error in $V(x_0,0)$")
    ax.set_title("Value error bottoms out, then degrades", fontsize=7.5)
    ax.legend(loc="upper left", framealpha=0.9)
    ax.text(0.5, 0.06,
            f"stars: value-error minimum\nmedian residual there: {thr:.1e}",
            transform=ax.transAxes, fontsize=6.5, color="0.35", ha="center")

    fig.tight_layout()
    save(fig, "fig_j2_bsde_stopping")
    plt.close(fig)
    print(f"    stopping threshold (median residual at value minimum): {thr:.3e}")
    return True


def main() -> int:
    print("=" * 100)
    print("JOURNAL FIGURES -- read from results/, nothing recomputed or hand-typed")
    print("=" * 100)
    made = []
    if fig_control_order():
        made.append("fig_j1_control_order")
    if fig_deep_bsde_stopping():
        made.append("fig_j2_bsde_stopping")
    print()
    if made:
        print(f"  produced: {', '.join(made)}")
    else:
        print("  produced nothing -- see the SKIP reasons above")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
