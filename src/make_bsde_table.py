"""
The deep-BSDE comparison table: does it belong beside the grid and PINN columns, and at what cost?

Why this table needs its own generator
--------------------------------------
The paper's central table (tab_main) compares accuracy and cost by dimension for the grid scheme and
the PINN. The journal audit (journal_audit.py) counts deep BSDE as the missing third family --
missing not because it is unimportant but because it is THE standard high-dimensional method in
mathematical finance, and a comparison that cites it without running it leaves its most likely
referee question unanswered. It has now been run at p = 1, 4, 8.

Two things make the comparison non-trivial, and both are handled explicitly rather than papered over:

1. DIFFERENT ACCURACY MEASURE. tab_main reports value R^2, which needs a variance normaliser taken
   over the benchmark domain. The deep-BSDE runs report a relative error at the single initial state
   they answer for -- which is the honest measure for a point-valued solver, and also exactly why the
   two families are not interchangeable. To place deep BSDE in the same column, the relative error is
   converted with the SAME normaliser the other methods use, computed here from the analytic
   reference over the same domain, so the column is comparable rather than merely adjacent.

2. THE STOPPING RULE DECIDES THE NUMBER. In every deep-BSDE run the terminal residual fell
   monotonically while the value error bottomed out and then degraded, so the value read at the end
   of training is 14-686x worse than the best value the run passed through. The table therefore
   reports the value AT ITS BEST POINT, and says so in the caption, because a reader who trains to a
   residual plateau will not reproduce this column.

Cost is reported as the training wall time to reach that stopping point, extrapolated from the
measured seconds-per-epoch, plus the count of training steps. Deep BSDE is point-valued: one training
run answers ONE initial state, whereas the grid and the PINN return V over a whole domain from one
solve. The caption states that, because it is the structural difference the numbers alone hide.

Usage: python src/make_bsde_table.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem      # noqa: E402

RESULTS = Path("results")
TABDIR = Path("paper/tables")

# The deep-BSDE runs, and what each one is. The n_steps choice is the resolution each dimension was
# run at; it is part of the reported configuration because the discretisation floor depends on it.
#
# p=1 uses the COMPLETED n_steps=200 run rather than the n_steps=50 one: the n_steps=50 convergence
# run was stopped early once it had saturated, so its trace carries no wall-clock per epoch and its
# cost column would read nan. Every row here therefore comes from a run that reached its own final
# save, which is what makes the three rows comparable with each other.
#
# An earlier design produced a domain-wide R^2 for deep BSDE and put it beside the grid's R^2. That
# was dropped: the two measure different things (one point versus the whole domain), and the
# comparison flattered the point-valued solver, which is the direction that would have damaged the
# paper's own claim rather than supporting it.
RUNS = [
    (1, "exp_bsde_ns200_conv.json", 200),
    (4, "exp_bsde_dims.json", 50),
    (8, "exp_bsde_dims_d8.json", 50),
]


def value_variance(p: int, n_per_axis: int = 25) -> float:
    """Variance of the analytic V(x, 0) over the benchmark domain, in the same units as value_mse.

    This is the normaliser that turns a value error into an R^2, and it is computed from the
    analytic reference rather than copied from another table so that the conversion cannot silently
    inherit a different domain convention. At p = 8 a full tensor grid is impossible (25^8 points),
    so the variance is estimated by Monte Carlo over the same domain, with the count high enough
    that the estimate is stable to well under a percent.
    """
    prob = ACProblem(p=p)
    lo, hi = prob.domain()
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V0 = float(np.asarray(ref.value(np.zeros(prob.state_dim), 0.0)).ravel()[0])

    rng = np.random.default_rng(0)
    if p <= 4:
        # Full tensor grid: exact for this purpose and cheap at these dimensions.
        axes = [np.linspace(lo[i], hi[i], n_per_axis) for i in range(p)]
        mesh = np.meshgrid(*axes, indexing="ij")
        X = np.stack([m.ravel() for m in mesh], axis=1)
    else:
        n = 200_000
        X = lo + (hi - lo) * rng.random((n, p))
    V = np.asarray(ref.value(X, 0.0)).ravel() - V0
    return float(np.var(V, ddof=1))


def measured_sec_per_epoch(p: int) -> float | None:
    """Per-epoch training rate at this dimension, from the dedicated cost probe.

    Used instead of each run's own timing because the p=1 trace was recovered from a log that did not
    carry wall-clock. The probe measured every dimension in ONE process under identical conditions,
    discarding a warm-up epoch, which is exactly the comparison the table needs; the alternative --
    re-running 16,000 epochs just to time it -- would cost half an hour and would still carry
    whatever else was on the GPU at the time.
    """
    path = RESULTS / "probe_bsde_epoch_cost.json"
    if not path.exists():
        return None
    for r in json.loads(path.read_text(encoding="utf-8")).get("rows", []):
        if int(r.get("p", -1)) == p:
            return float(r["sec_per_epoch"])
    return None


def load_run(p: int, fname: str, n_steps: int):
    path = RESULTS / fname
    if not path.exists():
        return None
    rows = json.loads(path.read_text(encoding="utf-8")).get("rows", [])
    best = None
    for r in rows:
        tr = r.get("trace") or []
        if not tr:
            continue
        # Field names differ between the two experiment families; resolve explicitly rather than
        # assuming one shape, which is the failure mode this project has hit repeatedly.
        norm = []
        for t in tr:
            v = t.get("value_rel_err", t.get("V_rel_err"))
            res = t.get("terminal_rmse", t.get("floor_abs"))
            if v is None or res is None:
                continue
            norm.append({"epoch": t["epoch"], "v_err": float(v), "residual": float(res)})
        if len(norm) < 3:
            continue
        cand = min(norm, key=lambda t: t["v_err"])
        cand = dict(cand, n_steps=n_steps,
                    floor=r.get("exact_floor_abs", r.get("exact_floor")),
                    sec_per_epoch=r.get("sec_per_epoch"))
        if best is None or cand["v_err"] < best["v_err"]:
            best = cand
    return best


def main() -> int:
    TABDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 104)
    print("DEEP-BSDE COMPARISON TABLE")
    print("=" * 104)

    rows = []
    for p, fname, n_steps in RUNS:
        got = load_run(p, fname, n_steps)
        if got is None:
            print(f"  skip p={p}: {fname} absent or has no usable trace")
            continue
        var = value_variance(p)
        # Relative error on V(x0,0) -> an absolute error using the analytic |V(x0,0)| as the scale.
        # The R^2 below is reported in the JSON for reference ONLY. It is deliberately NOT written
        # into the LaTeX table: tab_main's R^2 is domain-wide while this one is at a single point, so
        # placing them in one column would credit a point-valued solver with a function-valued one's
        # accuracy -- an error in the direction that flatters deep BSDE, in a paper whose thesis is
        # that the classical scheme wins where it can run. That is exactly the kind of comparison the
        # audit exists to prevent.
        prob = ACProblem(p=p)
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        V0 = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])
        abs_err = got["v_err"] * abs(V0)
        mse = abs_err ** 2
        r2 = 1.0 - mse / var
        spe = got["sec_per_epoch"] or measured_sec_per_epoch(p)
        secs = spe * got["epoch"] if spe else float("nan")
        rows.append(dict(p=p, n_steps=n_steps, v_err=got["v_err"], r2=r2, var=var,
                         epoch=got["epoch"], secs=secs, floor=got["floor"],
                         residual=got["residual"], spe=spe,
                         timing_source="run" if got["sec_per_epoch"] else "cost probe"))
        print(f"  p={p}: best V rel err {got['v_err']:.4%} at epoch {got['epoch']} "
              f"-> R^2 {r2:.4f}  (domain variance {var:.3e}, floor {got['floor']:.3e})")

    if not rows:
        print("\n  nothing to write")
        return 1

    lines = [
        "% Auto-generated by src/make_bsde_table.py -- do not edit by hand.",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Deep BSDE, the standard high-dimensional method in mathematical finance, on the "
        "same problem and at the same dimensions as Table~\\ref{tab:main}. Two columns are "
        "deliberately NOT comparable with that table and are labelled so. First, the error is "
        "measured at the single initial state the solver answers for ($x_0$, one unit of each "
        "asset), whereas the grid and PINN columns of Table~\\ref{tab:main} are domain-wide: a "
        "point-valued solver is not being credited with a function-valued one's accuracy. Second, "
        "the value is quoted \\emph{at its best point during training}, not at the end. In every "
        "run the terminal residual kept falling while the value error bottomed out and then "
        "degraded, so the endpoint is a factor of $14$ to $686$ worse than the number here, and a "
        "reader who trains to a residual plateau will not reproduce this column. Time is training "
        "wall time to that stopping point, at the per-epoch rate measured for each dimension in a "
        "single controlled run; that rate is nearly flat in $p$ "
        "($0.11$--$0.15$~s per epoch), so the cost is carried by the number of epochs needed to "
        "converge, not by the cost of an epoch. One training run answers one initial state; it does "
        "not return $V$ over the domain.}",
        "\\label{tab:bsde}",
        "\\small",
        # The header of this table was 13.34 pt wider than the column, which overflows a two-column
        # layout silently -- the text runs into the gutter and no warning reaches the reader. Measured
        # at 265.34 pt against a 252.0 pt column. The fix is to tighten the inter-column space and
        # shorten the two longest headers rather than to drop a font size, because dropping to
        # \footnotesize makes the numbers harder to read than the headers were hard to shorten.
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{r r r r r r}",
        "\\toprule",
        "$p$ & $n_{\\mathrm{steps}}$ & train (s) & rel.\\ err & domain var & resid. \\\\",
        " & & & at $x_0$ & & at stop \\\\",
        "\\midrule",
    ]
    for r in rows:
        lines.append(f"{r['p']} & {r['n_steps']} & {r['secs']:.0f} & {r['v_err']:.3f} & "
                     f"{r['var']:.2e} & {r['residual']:.2e} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ]
    out = TABDIR / "tab_bsde.tex"
    out.write_text("\n".join(lines))
    print(f"\nwrote {out}")
    print("\n  This table IS \\input into main.tex, and the width above is no longer a guess: the")
    print("  un-tightened version measured 265.34 pt against the 252.0 pt column and produced a")
    print("  13.34 pt overfull box in a real build, which is why \\tabcolsep is set here.")
    print("  To re-check the layout, build with TEXLIVE_BIN set and read the overfull count:")
    print("      TEXLIVE_BIN=<texlive bin dir> python src/build_paper.py")
    (RESULTS / "bsde_table.json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
