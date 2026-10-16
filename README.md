# Value Accuracy Is Not Control Accuracy — rework

Reproducible code for the CISP-BMEI 2026 submission *"Value Accuracy Is Not Control Accuracy:
Reconstruction Order Governs Control Accuracy in Hamilton–Jacobi–Bellman Solvers."*

A from-scratch rework of an audited undergraduate thesis on SHoP-style HJB solvers for
high-frequency trading. **No file in the original thesis package is modified by anything here.**

**New here? Run this first, then read `CODE_GUIDE.md`.**

```powershell
cd <path-to-this-repository>
python src\verify_all.py        # ~5 s, CPU only; proves the checkout reproduces
```

`CODE_GUIDE.md` is the map: what each file does, what order to read them in, how to run the
experiments, VS Code tips, and a troubleshooting table. This README explains the *science*.

---

## Why a rework rather than a re-run

An audit of the original package found that its empirical results could not be reproduced as
published, for reasons that are documented in `../paper_assessment_for_IE_conference.md` and
summarised in the paper's Section IV. In brief:

| Defect | Where | Effect |
|---|---|---|
| D1 ordering | minute CSVs are newest-first and were never sorted | `price[i-1]` is the *next* minute; summed price return is +0.037950 in file order vs −0.021709 chronologically — the P&L sign is inverted |
| D2 same-bar leak | position from bar *i* multiplied by bar *i*'s return | the strategy sees the move it trades |
| D3 control sign | `-grad V / (2 kappa)` in three places | every "buy" signal (19.1% of minutes on SPY) has a negative rate: a strict pointwise inversion |
| D4 dollar scaling | the `1/(2 kappa S)` factor omits `S` | wrong control magnitude |
| D5 position vs rate | `position = clip(rate, -1, 1)` | position is not the integral of the rate, contrary to the model's own `dq = v dt` (the original's notation; the paper writes this `dx = u dt`) |
| D6 cost units | currency costs subtracted from a return series | costs swamp gross P&L by orders of magnitude |
| D7 annualisation | minute returns scaled by `sqrt(252)` | every volatility and Sharpe ratio is uninterpretable |
| D8 input scaling | trained on price ∈ [20, 30], evaluated at 40–50 | all cross-sectional conclusions are extrapolation artefacts |

Two further pipeline defects are also corrected: features were computed non-causally
(`shift(1)` on a newest-first file; VWAP as a global cumulative sum), and the "spread" input was
the high–low range, which is not a bid–ask spread.

Additionally, the original validated solver accuracy against Policy Iteration itself, making
that method zero-error by construction, and its implemented model had no inventory state.

---

## Design of the corrected benchmark

**Problem.** Multi-asset Almgren–Chriss optimal execution in linear–quadratic form:

```
state   x ∈ R^p   (inventory)
control u ∈ R^p   (trading rate)
dx = (A x + B u) dt + Σ dW

max_u  E[ -½ ∫(x'Qx + u'Ru) dt  -  ½ x_T' Q_T x_T ]
   R = η I    temporary market impact
   Q = λσ² I  inventory risk penalty
```

**Analytic reference.** With the quadratic ansatz `V = -½ x'P(t)x + c(t)`:

```
-Ṗ = Q + P A + A'P - P B R⁻¹B' P ,   P(T) = Q_T
 ċ = ½ tr(ΣΣ' P)                     ,   c(T) = 0
u*(x,t) = -R⁻¹B' P(t) x
```

Having `u*` in closed form is what makes the control **sign** falsifiable rather than assumed.

**Default parameters** (well scaled so the explicit time step is CFL-limited realistically):
`η = 0.01`, `λσ² = 0.01`, `Q_T = 0.1`, `σ = 0.01`, `T = 1` ⇒ `k = 1`.

---

## Layout

```
rework/
  CODE_GUIDE.md              HOW TO READ THE CODE -- start here
  src/
    verify_all.py            one command that proves the checkout reproduces
    --- the science -------------------------------------------------------------------
    ac_problem.py            benchmark problem, Riccati closed form, self-checks
    solvers.py               finite differences, Howard policy iteration, accuracy metrics
    pinn.py                  PINN (GPU)
    pinn_pi.py               policy-iteration PINN (reimplementation, see below)
    --- experiments (each writes results/<name>.json) ---------------------------------
    exp_dimension.py         Table I:  accuracy and cost vs state dimension p
    exp_grid_refinement.py   the accuracy-matched grid resolution per dimension
    exp_highdim.py           the p = 1..20 sweep on a non-separable problem
    exp_pinn_budget.py       is the PINN's failure at p>=4 a budget artefact?
    exp_eval_timing.py       per-query latency and the break-even analysis
    amortization_analysis.py one instance vs a parameter family
    exp_control_order.py     Tables II/III: control vs value convergence orders
    exp_control_extraction.py Table II: multilinear vs cubic reconstruction
    exp_control_floor.py     time refinement vs space refinement
    exp_control_p5.py        p=5 with the snapshot cap fixed
    exp_control_extend.py    extended sweeps + least-squares order fits
    exp_merton_extraction.py the non-polynomial (Merton) confirmation
    exp_pi_pinn.py           PI-PINN vs direct PINN vs grid (needs ../exploratory_hjb/)
    --- real data ---------------------------------------------------------------------
    preprocess_causal.py     causal preprocessing + Roll / Corwin-Schultz spreads
    backtest.py              corrected backtest engine + defect-ablation configurations
    exp_ood.py               out-of-domain evaluation case study
    exp_ablation.py          defect ablation on real data
    --- artifacts ---------------------------------------------------------------------
    make_figures.py          every figure, from results/*.json
    make_tables.py           every LaTeX table, from results/*.json
    build_paper.py           compile the manuscript (use --build-dir if a viewer locks it)
    check_paper.py           static checks on the .tex before compiling
    paper_length.py          words, floats, compiled page count
    --- tests and utilities -----------------------------------------------------------
    test_backtest.py         accounting vs hand-computed values
    test_experiment_paths.py code paths for the GPU experiments
    extract_lit.py           literature: PDF -> text (needs Ghostscript)
    scan_lit_timing.py       literature: how papers compare cost against solvers
    scan_lit_venues.py       literature: venue detection (see the caveat in the file)
    fetch_repo.py            fetch a GitHub repo without git (Schannel is broken here)
    bench_cpu_gpu.py         CPU vs GPU throughput for the PINN inner loop
  results/                   JSON results (committed, so tables rebuild in seconds)
  figures/                   paper figures (generated)
  paper/                     IEEEtran manuscript, bib, generated tables
  STATUS.md                  running log of what was measured and corrected
```

The manuscript is built and page-checked by `src/build_paper.py`; it looks for TeX Live on `PATH`
and for `IEEEtran.cls` inside `paper/`, and can be pointed elsewhere with the `TEXLIVE_BIN`,
`IEEE_CLS` and `IEEE_BST` environment variables.

### Market data are not distributed with this repository

The experiments that use real market data read `data/` (minute bars for nine US instruments, as
produced by the audited baseline's non-causal preprocessing) and `data_causal/` (the causally
preprocessed version, generated by `src/preprocess_causal.py`). **Neither directory is included
here --- not in the working tree, and not anywhere in this repository's history**, because
redistributing vendor market data is generally not permitted. Both paths are in `.gitignore` so that
a local copy cannot be committed by accident.

What *is* included is everything needed to reconstitute them and to check the calibration:

* `src/preprocess_causal.py` -- the full preprocessing, including the temporal split. The split
  dates are stated in the manuscript and in the script, so the split itself is reproducible even
  without the bars.
* `data_manifest/preprocess_manifest.json` -- the preprocessing's **metadata**: per-instrument row
  counts, date ranges, session filter, estimator names, and the per-instrument spread summaries
  (Roll and Corwin--Schultz). It contains no prices. It is what the manuscript's data statement
  refers to when it says the calibration can be re-checked without the bars.
* `results/exp_real_calibration.json` -- the calibration itself, per-instrument $\sigma$ and $\eta$,
  which is what `tab_calib` is generated from. `results/` also holds every other raw result file
  behind the tables, so no claim in the paper requires re-running anything.

Nothing in the verification suite needs the data: `python src/verify_all.py` runs to completion on
a clean checkout with no data present.

**To re-run the market-calibrated experiments** (the calibration sweep, the execution shortfall),
restore the bars to `data/` and `data_causal/` and run the script. Without them,
`src/exp_real_calibration.py` stops with an explanation rather than quietly reporting a
synthetic-only result. It did exactly that once --- running it on a checkout without the bars
overwrote the shipped result file with a weaker one --- which is why the guard exists.

**Note on `pinn_pi.py`.** The reference implementation of the policy-iteration PINN is JAX, and
JAX could not be installed in the environment these results were produced in, so `pinn_pi.py` is a
faithful PyTorch reimplementation built from the published source rather than their code running.
The file documents exactly what was taken from where, and the paper states the reimplementation
and its reduced budget wherever that comparison is reported.

---

## Verification

What is checked, by what, and what is NOT covered: see `VERIFICATION.md`. The short
version is `python src/verify_all.py` (23 checks, no GPU, no market data).

## Running

Environment: Python 3.11.5 with the exact pins in `requirements.txt` (torch 2.7.1+cu126,
numpy 1.24.3, scipy 1.11.1, pandas 2.0.3, matplotlib 3.7.2). A CUDA GPU is needed only for the
neural experiments; everything else -- including the whole verification suite -- runs on CPU.

```powershell
python -m pip install -r requirements.txt
```

Reproducing the *manuscript* additionally needs a LaTeX installation; `src/build_paper.py` expects
TeX Live and will report what it cannot find. That step is optional: every result in `results/` is
already checked in.

```powershell
# The simplest path: verify_all.py sets the two required variables itself.
python src\verify_all.py            # environment + unit checks + numeric smoke (~5 s)
python src\verify_all.py --full     # also trains a tiny PINN (needs the GPU, ~10 s)
```

For a direct run of any module you must set the variables yourself:

```powershell
# Anaconda's MKL and torch's libomp collide on Windows; this is required.
$env:KMP_DUPLICATE_LIB_OK="TRUE"
# PYTHONPATH must contain this repository's src/ directory. Run from the repository root:
$env:PYTHONPATH="$PWD\src"
# matplotlib's default font cache is not always writable; point it inside the checkout.
$env:MPLCONFIGDIR="$PWD\.mplconfig"

# 1. causal preprocessing and temporal split
python src\preprocess_causal.py

# 2. verification (fast, CPU)
python src\ac_problem.py            # analytic reference vs three closed forms
python src\test_backtest.py         # backtest accounting vs hand-computed values
python src\test_experiment_paths.py # code paths for the GPU experiments

# 3. experiments -- see CODE_GUIDE.md for the full list with runtimes
python src\exp_control_extraction.py --p 2 --grids 11,15,21,27,35,45,61 --also-pde --synthetic
python src\exp_dimension.py --dims 1,2,3,4,5 --epochs 6000 --seeds 0,1,2 --pi-max-p 3

# 4. paper artifacts
python src\make_tables.py
python src\make_figures.py
python src\build_paper.py --build-dir paper_build
```

`--pi-max-p` caps Howard policy iteration because it costs ~13–15 grid sweeps per solve; the
grid-cost scaling beyond that dimension is reported as an extrapolation.

---

## Verification performed

`ac_problem.py` checks the analytic reference against three independent closed forms:

| Check | Measured |
|---|---|
| Riccati `P(t)` vs finite-`Q_T` closed form | max err 9.8e-12 |
| `P(T) = Q_T` | exact |
| control law vs `-P(t)q/η` | max err 8.9e-16 |
| control negative for positive inventory | all sampled states |
| `Q_T→∞` limit recovers `ηk·coth(k(T-t))` | rel err 1.0e-9 |
| same limit recovers `q₀ sinh(k(T-t))/sinh(kT)` | max err 4.3e-9 |
| `p = 1,2,3,5,8`: `P` symmetric PSD | pass |

`test_backtest.py` checks the accounting against hand-computed values (17 checks), including
that the corrected annualisation matches a manual daily aggregation to 1e-12, and that the
control sign convention is right (positive gradient ⇒ positive rate under the HJB, negative
under the original defect).

`test_experiment_paths.py` checks the GPU experiments' plumbing on CPU, including the
rescaling chain rule and a sanity check that an unscaled gradient would fail the test by ~50×
(so the test has power).

### Bugs found and fixed *by* verification

These were all silent — none would have raised an exception:

1. **Grid solvers returned the `t=0` slice for every `t`** (no snapshot storage). A first
   accuracy measurement reported a spurious MSE of 3e-13 purely because the randomly drawn
   evaluation time happened to land near 0.
2. **Policy iteration collapsed the policy to a time-independent array.** The optimal policy
   here is `u*(x,t) = -R⁻¹B'P(t)x` with strongly time-dependent `P(t)`; the frozen-policy
   evaluation PDE was inconsistent near `t=T` and diverged.
3. **Upwind direction inverted.** Central differencing of a frozen advection term is
   unconditionally unstable; a von Neumann analysis showed `u > 0` needs the *forward* stencil.
4. **Corwin–Schultz used the average of the two daily squared ranges instead of the sum**,
   which makes `α` identically zero — the estimator returned 0 for every asset.
5. **`_forward` in the OOD experiment built 1-D tensors** that `_enc` concatenated on `dim=1`.

---

## What is *not* reproduced

The paper deliberately excludes six numerical claims from the original thesis that could not
be traced to any code path: the 1-D harmonic-oscillator and 2-D biharmonic accuracy tables
(including an `R²` metric that no function in the package computes), the terminal
boundary-condition table, the trading-behaviour table, the "Spearman ρ = 1.0" claim, and the
memory figures. None of them appears in the manuscript, and no value from them is cited.

## Building the manuscript

A **TeX Live** installation is required. The build finds `pdflatex` on `PATH`, or at the directory
named by the `TEXLIVE_BIN` environment variable, so no machine-specific location is recorded in this
repository:

```sh
export TEXLIVE_BIN=/usr/local/texlive/2026/bin/x86_64-linux   # only if not already on PATH
python src/build_paper.py            # stages paper_build/, compiles, reports pages and boxes
python src/paper_length.py           # words, floats, and the compiled page count
python src/diag_floats.py            # splits the page count into text vs floats
```

`build_paper.py` mirrors the submission sources into `paper_build/` rather than editing them,
because two substitutions are needed for a local build and neither should reach the submission:

* **`IEEEtran.bst` may be missing from the TeX tree** (and `tlmgr install` cannot always run). The
  build then substitutes TeX Live's `ieeetr.bst` — a standard IEEE Transactions style — so layout,
  length and float placement stay representative, but reference punctuation may differ slightly from
  the official template. The submission source keeps `\bibliographystyle{IEEEtran}`. Set `IEEE_BST`
  to override.
* **`\graphicspath`** is rewritten from `../figures/` to `./figures/`, matching the build layout.

`IEEEtran.cls` is likewise resolved from `IEEE_CLS` if set, otherwise from a copy placed in
`paper/`, otherwise from whatever the TeX distribution provides.

Current state: **11 pages**, 0 errors, no undefined references or citations, one 1.73 pt overfull
box (invisible). CISP-BMEI allows unlimited pages for initial submission; one registration covers
6 pages for the camera-ready, with extra pages charged. See `STATUS.md` for the cut list to reach 6.

## Known limitations of the local environment

* **`seaborn` is not installed** and is not used anywhere in the rework.
* **matplotlib needs `MPLCONFIGDIR` pointing inside the workspace** — the default
  `~/.matplotlib` font-cache lock is not writable in some sandboxes and raises `PermissionError`.
* **Ghostscript** is used only to rasterise the compiled PDF for visual inspection, and to turn
  reference PDFs into text. Nothing in the paper's results depends on it. `extract_lit.py` finds it
  via `GS_BIN` or `PATH`; a TeX-bundled ghostscript also needs `GS_LIB`.
* All neural timings come from one NVIDIA RTX 4090 Laptop GPU; grid timings from CPU. The
  asymmetry favours the neural method and is stated in the paper rather than adjusted away.
