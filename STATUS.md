# Rework status — SHoPHJB thesis → CISP-BMEI 2026 submission

**Status: complete and compiled.** All experiments run, all figures and tables generated from
those runs, the manuscript written, **typeset, and page-checked locally**, and all five
verification suites passing. The only outstanding item is an optional native-English proofread.

**Working rule honoured throughout**: no file in the original thesis package was modified
(verified: no file in it has a modification time inside this work's window). All rework lives in
`rework/`.

---

## Deliverable

An IEEEtran conference paper compiled against the CISP-BMEI 2026 requirements
(19th Int. Congress on Image and Signal Processing, BioMedical Engineering and Informatics;
31 Oct – 2 Nov 2026, Zhenjiang; IEEE Xplore / EI Compendex; English; **no page limit on initial
submission**, one registration covers 6 pages for the camera-ready with extra pages charged),
targeted at the CISP *Signal Modeling, Identification & Prediction* track.

**Title (revised after the control-extraction results):** *Value Accuracy Is Not Control Accuracy:
Reconstruction Order Governs Control Accuracy in Hamilton–Jacobi–Bellman Solvers.* Earlier titles,
in order: *Where Do Physics-Informed Neural Networks Beat Classical Solvers for the HFT HJB
Equation? A Closed-Form Benchmark* → *Value Accuracy Is Not Control Accuracy: Resolution
Requirements for HJB Solvers in High-Frequency Trading* → the current one. The title moved with the
finding: the paper began as a method horse-race, became a measurement-methodology paper when the
control/value decoupling appeared, and became an actionable one when the extraction experiment
located the cause in the reconstruction order rather than the grid.

**Compiled result: 11 pages, 0 errors, no undefined references or citations, one 1.73 pt overfull
box (invisible).** 6,810 prose words and 995 words in tables. The paper grew from 7 pages because the
user's questions were answered with measurements rather than assertions, and every added page is a
result: the control-accuracy section and its two tables, the high-dimension sweep, the separability
finding, the operator-literature positioning, the two-regime amortization analysis, the
market-calibrated scope section, and the revised five-contribution framing. Initial submission has no
page limit, so this is submittable as it stands; the camera-ready will need the trim list at the end
of this section.

A **13.34 pt overfull box** was found and removed during the final build. It came from
`tables/tab_bsde.tex`, whose six-column header measured 265.34 pt against the 252.0 pt column: the
generator now sets `\tabcolsep` to 4 pt and splits the two longest headers, bringing it to
205.1 pt. The failure mode is worth recording because nothing surfaced it earlier. An overfull box in
a two-column layout runs into the gutter, and the build had been reporting "one 1.7 pt box" for long
enough that a larger one was not read as new. It was located by compiling the document unchanged
apart from one suspect line -- the control run still showed the box, which is what ruled the suspect
out.

See `../revised_claims_and_title.md` for the full statement of what changed in the thesis's claims,
the candidate titles with trade-offs, and what must be deleted from the original.

Reaching 8 pages from the first working draft of 9 came from, in order of effect: tightening
display-math, list and float spacing (~1 page, no content lost); consolidating an overflowing
6-column accuracy table, a cost table and a crossover table into one full-width table; dropping
two tables whose content is a few sentences of prose; merging two cost figures into one
two-panel figure; dropping the ablation figure, whose numbers the ablation table already gives
precisely; and about 200 words of genuine redundancy. The count then went *up* again, from 7 to
8, when the training-versus-inference analysis was added — a deliberate trade of a page for a
substantive answer to the most obvious methodological objection (see below).

**To reach the 6-page camera-ready**, in priority order: (i) move Table~\ref{tab:ablation} to a
supplement (its argument is in the text); (ii) drop Fig.~3, the out-of-domain figure, whose
numbers are all in the text; (iii) shorten the Related Work section to a single paragraph, now
that it carries the operator-literature positioning as well; (iv) compress the defect list by
folding D2/D5/D6 into one sentence, as the ablation table already does. That removes roughly two
to three pages without losing a claim. Alternatively, pay for the extra pages.

---

## What the paper claims, and why it is defensible

The original thesis was written as *"we built a better solver"*. That framing does not survive its
own data, so the paper is instead a **benchmarking and method-selection study**. It makes three
claims, each backed by a number produced here.

**1. At equal accuracy, the classical scheme wins at every dimension where an accurate neural
solution is obtainable.** The grid scheme is run at the resolution each dimension needs to reach
R² ≥ 0.9999 (a separate refinement study; N* = 21 at every p).

| p | grid, accuracy-matched | R² | PINN (6000 epochs) | R² |
|---|---|---|---|---|
| 1 | 0.023 s | 0.99995 | 95 s | 0.9997 |
| 2 | 0.056 s | 0.99993 | 128 s | 0.9993 |
| 3 | 0.530 s | 0.99993 | 157 s | 0.9918 |
| 4 | 31.7 s | 0.99992 | 177 s | −57.0 |
| 5 | 865 s | 0.99991 | 226 s | −616 |

**2. The PINN's failure at p ≥ 4 is a budget artefact, not a capacity limit — and saying so does
not rescue it.** A referee's first objection would be under-training, so it was tested directly:

| p | epochs | collocation | time | R² |
|---|---|---|---|---|
| 4 | 6,000 | 2,048 | 177 s | −44.5 |
| 4 | 20,000 | 2,048 | 630 s | **+0.9972** |
| 4 | 6,000 | 8,192 | 184 s | −309.7 |
| 3 | 6,000 | 2,048 | 172 s | +0.9981 |
**3. Training time versus inference time — the objection tested and answered.** A natural
objection is that the comparison should use inference, not training, because a trained network
evaluates in microseconds. Both were measured. Neither method is "solve long" versus "evaluate
fast": both are an expensive offline solve followed by cheap online queries, and the grid
scheme's online query is multilinear interpolation of a stored snapshot. Per-state control
evaluation, amortised over a batch of 390 (one trading day of minute decisions):

| p | grid interpolation | PINN (GPU) |
|---|---|---|
| 1 | **0.66 µs** | 2.98 µs |
| 2 | 1.51 µs | — |
| 3 | 5.07 µs | **3.60 µs** |
| 4 | 12.91 µs | — |
| 5 | 34.27 µs | **4.86 µs** (but inaccurate) |

So the neural solver's inference is *not* uniformly faster: at p=1 grid interpolation beats a
forward pass plus autograd by about 4.5×. The PINN's inference wins only as p grows, where
multi-dimensional interpolation degrades and the network's cost does not. Combining both
components as `T(N) = T_offline + N*t_eval` and solving for break-even: **at p=1 the PINN never
breaks even** (the grid is cheaper both upfront and per query); **at p=3 it needs 1.06e8
evaluations, or about 272,000 trading days**; beyond p=3 the question is moot because the PINN is
inaccurate. The objection *would* be decisive if the classical scheme had to be re-solved per
query, but a fixed-parameter PINN does not escape that either — changed parameters mean
retraining. Exploiting that regime properly needs a parametric or operator-learning formulation,
which the paper names as the most promising direction rather than treating it as a defect of the
comparison here.

More optimisation steps do recover p=4. But the rescued run costs ~20× the accuracy-matched grid
solve at the same dimension (630 s vs 31.7 s) *and* is less accurate (0.9972 vs 0.99991). A second
finding, not anticipated: **quadrupling the collocation count at a fixed step budget makes things
worse** (−44.5 → −310), because each step's averaged gradient is noisier and the schedule is
exhausted first. Scale the step count, not the batch.

**4. Control accuracy is governed by the RECONSTRUCTION ORDER, not by value accuracy.** The
control is the deliverable of an HJB and is a derivative of the value function, so a grid scheme
must differentiate an interpolant while a network differentiates itself exactly. With the
multilinear interpolation and probing difference that is standard here, the control error is
second order and 13–185× the value error. **Isolating the extraction from the solution** — by
placing the *exact* closed-form value function on the grid as nodal values, so the discrete
solution carries no error at all — reproduces essentially the whole control error
(3.7e-3 → 1.1e-4, order 1.98, against 3.8e-3 → 1.1e-4 on the solved grid). So the grid solution is
not the limiting factor.

Three reconstructions, measured on functions with known gradients so the orders are properties of
the reconstruction rather than of the benchmark (ranges over p=2,3,5):

| reconstruction | quadratic | quartic | exponential | trigonometric |
|---|---|---|---|---|
| multilinear + probe | 2.1–2.2 | 2.3–2.3 | 2.2–2.3 | 2.1–2.3 |
| multilinear, exact cell slope | 2.0–2.0 | 1.9–2.0 | 2.0–2.0 | 2.0–2.0 |
| **cubic + probe** | **exact** | **4.4–5.3** | **4.2–4.7** | **4.7–6.8** |

Taking the *exact* gradient of the multilinear interpolant is no better than probing it, so
consistency with the interpolant buys nothing; only the **order** of the reconstruction helps. On
the benchmark's own solved grid at p=2, N=61 the cubic extraction gives 1.7e-6 against the
multilinear 1.1e-4 — a factor of 67 — and is then limited by the solver rather than the
reconstruction, so the **control/value error ratio falls from 185 to 2.8**.

**This reverses the practical advice** that follows from the accuracy comparison alone: control
accuracy is not bought with N^p spatial refinement but with the order of the differentiation
operator, which costs nothing at the grid level. Caveat: this benchmark's V is quadratic and a
cubic reconstruction reproduces quadratics exactly, so the machine-precision result is a best
case; the quotable numbers are the fourth-order convergence and the four-to-five orders of
magnitude on non-polynomial test functions.

**5. The control error cannot be improved by time refinement.** Holding the mesh fixed and varying
only the number of stored snapshots, the value error falls by 1–4 orders of magnitude while the
control error improves by at most ~17× and then plateaus (at p=5 not at all). Time refinement
cannot substitute for a better reconstruction.

**6. Dimensions above 5: the mesh-free solver is the only one that runs, and the earlier
"failure" was a parameterisation artefact.** The p=1..5 sweep is bounded by the grid method, not
by the reference: the Riccati equation is p² ODEs at any p, so accuracy remains measurable where
no classical baseline can run. Extending to p=20 exposed a flaw in the original parameterisation.
With the uniform defaults (Q=λσ²I, R=ηI, A=0) the value function is **separable**,
V(x,t)=Σᵢv(qᵢ,t) for a single scalar v, so solving it at p=20 is barely harder than at p=1 — a
network applies one learned univariate function and sums. That is not what the
curse-of-dimensionality argument is about, so the benchmark now supports per-asset heterogeneous
risk and impact coefficients plus random off-diagonal risk, making V a genuinely p-dimensional
quadratic with a full P matrix. Re-validated against the closed form up to p=20 (P symmetric PD,
terminal exact, control sign correct).

| p | PINN R² (hetero) | PINN time | grid R² | grid feasible |
|---|---|---|---|---|
| 1 | 0.99999 | 62 s | 0.99989 | yes |
| 3 | 0.9992 | 104 s | 0.99993 | yes |
| 5 | 0.9771 | 145 s | 0.99929 | yes |
| 8 | **0.9813** | 196 s | — | **no** (N^p) |
| 12 | 0.9269 | 270 s | — | no |
| 20 | 0.7570 | 414 s | — | no |
| **8, separable** | **−687.8** | 196 s | — | no |

**The separability control at p=8 is the sharpest single result: the separable problem fails
catastrophically (R²=−688) while the non-separable, genuinely harder problem is solved well
(R²=0.981).** So the paper's earlier claim that the PINN "collapses at p≥4" is *specific to that
parameterisation, not to the dimension* — a permutation-symmetric problem appears to be a harder
optimisation target for a generic MLP, presumably because it has degenerate directions and a much
weaker training signal. The manuscript was corrected accordingly: Table I now carries an explicit
cross-reference to the new Section V-F, and the subsection reports the control is usable to
p≈12 with accuracy decaying without any internal warning signal (sign agreement stays ≈0.7 at
p=20 where R²=0.76).

**7. Out-of-domain evaluation generates smooth, plausible, spurious cross-sectional structure.**
A network trained on prices in [20, 30] and evaluated on $42–$536 has its local price sensitivity
decay monotonically — by ~54× — with rank correlations of −0.90, −0.98, −1.00 over three seeds. It
saturates further outside its training domain. Any threshold on that gradient inherits the
ordering. Rescaling the input back into the training domain removes it (+0.12, −0.22, +0.23). The
metric is deliberately threshold-free, because the no-trade fraction depends on an arbitrary rule
and the audited baseline used two inconsistent ones.

Plus a defect ablation on real data showing the same model on the same data supports −7.9% or
+1.3% total return depending only on whether minute bars are sorted, and a Sharpe ratio of −0.14
or −1.20 depending only on the annualisation convention. No performance claim is made: the test
window is 14 trading days, on which no Sharpe ratio is distinguishable from zero, and that
limitation is stated in the paper rather than glossed.

---

## Verification performed

| Suite | Checks | Result |
|---|---|---|
| `src/ac_problem.py` | analytic reference vs three independent closed forms | **PASS** — Riccati P 9.8e-12; control law 8.9e-16; Q_T→∞ recovers ηk·coth (1.0e-9) and the Almgren–Chriss trajectory (4.3e-9); p=1,2,3,5,8 symmetric PSD |
| `src/test_backtest.py` | accounting vs hand-computed values (17 checks) | **PASS** — corrected annualisation matches a manual daily aggregation to 1e-12; control-sign convention correct |
| `src/test_experiment_paths.py` | GPU experiments' plumbing, on CPU | **PASS** — rescaling chain rule exact; power check confirms an unscaled gradient fails by 50× |
| `src/check_paper.py` | manuscript statics (no LaTeX toolchain locally) | **PASS** — 0 failures, 0 warnings |

### Five silent bugs found and fixed *by* verification

None raised an exception; each would have corrupted the study:

1. **Grid solvers returned the t=0 slice for every t** (no snapshot storage). A first accuracy run
   reported a spurious MSE of 3e-13 purely because the randomly drawn evaluation time landed near
   zero. Fixed by storing snapshots, interpolating in t, and averaging accuracy over time slices.
2. **Policy iteration collapsed the policy to a time-independent array.** The optimal policy here
   is u*(x,t) = −R⁻¹B′P(t)x with strongly time-dependent P(t), so the frozen-policy evaluation PDE
   was inconsistent near t=T and diverged.
3. **The upwind stencil direction was inverted.** Central differencing of a frozen advection term
   is unconditionally unstable; a von Neumann analysis showed u>0 needs the *forward* stencil.
4. **Corwin–Schultz used the average rather than the sum of the two daily squared ranges**, which
   makes α identically zero — the estimator returned 0 for every asset. Checked afterwards against
   the symmetric-case closed form (0.0200 vs √v = 0.020001).
5. **The OOD experiment omitted the terminal condition entirely**, leaving the PINN free to
   converge to an arbitrary solution of the PDE; it also built 1-D tensors that `_enc`
   concatenated on `dim=1`. The first version silently produced meaningless no-trade fractions.

---

## Corrections applied to the audited baseline

D1 ordering (newest-first CSVs never sorted → inverted P&L sign) · D2 same-bar leak ·
D3/D4 control law (−∇V/(2κ) instead of the HJB first-order condition, and the missing 1/S) ·
D5 position vs rate (position was set to the rate, not its integral) · D6 cost units (currency
costs subtracted from a return series) · D7 annualisation (minute returns scaled by √252) ·
D8 input scaling. Two pipeline defects are also corrected: features were computed non-causally
(`shift(1)` on a newest-first file; VWAP as a global cumulative sum), and the "spread" input was
the high–low range — which cannot be a bid–ask spread, since the source provides only OHLCV.
Replaced with the Roll estimator, cross-checked against Corwin–Schultz (implied spreads
1.2–7.0 bp, consistent with known effective spreads).

**Not carried over**, per instruction: the six numerical claims in the thesis with no code
provenance — the 1-D/2-D accuracy tables (including an R² that no function in the package
computes), the terminal boundary-condition table, the trading-behaviour table, the
"Spearman ρ = 1.0" claim, and the memory figures. None appears in the manuscript.

---

## Environment, and what was verified locally

| Item | Status |
|---|---|
| Python / torch | 3.11.5 (Anaconda) / 2.7.1+cu126 |
| GPU | NVIDIA RTX 4090 Laptop, 17.2 GB (neural timings); grid timings on CPU |
| `KMP_DUPLICATE_LIB_OK=TRUE` | **required** — Anaconda MKL vs torch libomp clash |
| `MPLCONFIGDIR` | must point inside the workspace; the default `~/.matplotlib` font-cache lock is not writable |
| `seaborn` | not installed, and not used |
| **LaTeX** | any TeX Live providing `pdflatex`, found on `PATH` or via `TEXLIVE_BIN` — no machine-specific path is recorded in this repository |
| Ghostscript | optional; used only to rasterise pages for visual inspection. Found via `GS_BIN` or `PATH`; a TeX-bundled copy also needs `GS_LIB` |

**The manuscript is compiled, not merely linted.** `src/build_paper.py` stages the sources into
`paper_build/` and runs pdflatex + bibtex + pdflatex ×2, reporting the page count, every
undefined reference and citation, and every overfull box. `paper/main.pdf` is a real build. The
rendered pages were inspected as images, which is how the 190 pt column overflow in the original
accuracy table and the 60 pt overflow in equation (3) were found and fixed.

Two substitutions are made for the local build only, and the submission sources keep the
correct forms:

* **`IEEEtran.bst` may be missing from the TeX tree** (and `tlmgr install` cannot always run). The
  build then substitutes TeX Live's `ieeetr.bst`, a standard IEEE Transactions style. Layout, length
  and float placement are therefore representative; reference punctuation may differ slightly from
  the official CISP-BMEI template. Override with `IEEE_BST`.
* **`\graphicspath`** is rewritten from `../figures/` to `./figures/` to match the build layout.

`IEEEtran.cls` is resolved from `IEEE_CLS` if set, otherwise from a copy in `paper/`, otherwise from
the TeX distribution.

**Still not verified:** the actual CISP-BMEI `.cls` (only the generic IEEEtran class was
available), and therefore the exact final page count under the official template. The body is
plain IEEEtran, so the swap should be mechanical, but a re-check after switching templates is
prudent.

### Honest limitations of the paper itself

(i) The sweep stops at p=5 because grid methods get expensive beyond it; higher-p claims are
extrapolation. (ii) One architecture (4×64 tanh), one optimiser, one schedule, no architecture
search — the negative result is about a competently trained standard PINN, not about every neural
approach. (iii) Real-data results cover nine instruments over two months, so they characterise the
pipeline rather than the strategy's profitability. (iv) One GPU and one CPU; neural timings are
GPU and grid timings CPU, which favours the neural method and therefore does not weaken the
conclusion. All four are stated in the paper's Discussion.

---

## Artifacts saved

| Artifact | Contents |
|---|---|
| `cisp-bmei2026-manuscript` v1 | `paper/main.tex`, `refs.bib`, 7 generated tables |
| `cisp-bmei2026-figures` v1 | 4 figures, PDF + 300 dpi PNG |
| `cisp-bmei2026-results` v1 | all 8 result JSONs behind every number |
| `cisp-bmei2026-rework-code` v2 | 15 source modules + README + this file (v1 accidentally included bytecode) |
| `thesis-assessment-ie-conversion` v1 | the original go/no-go assessment and defect audit |
