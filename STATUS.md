# Rework status — SHoPHJB thesis → journal submission

**Target (decided 2026-10-14)**: **Journal of Computational Physics** — JCR Q1 (PHYSICS, MATHEMATICAL,
rank 3/60), JIF 3.9. The conference submission is cancelled; the manuscript goes straight to a
journal. Selection evidence, the gap list and the roadmap are in `JOURNAL_Q1_TARGET_AND_GAPS.md`.

**Current stage: round 13 — the pre-submission read-through is done, and it found real defects rather
than only style. Remaining: author-only items.**

---

## Round 13 — the pre-submission proofread (2026-10-14)

Two of the three figures and one of the fourteen tables were **never cited anywhere in the text** ---
they were included, captioned, and never referred to. That is the kind of defect a proofread exists
for, and no content check in this repository could have found it: every number in them was correct,
every table was generated from a result file, and the orphan audit only asks whether a result file is
*read*, not whether the float it produced is *mentioned*.

What was found and fixed:

| defect | fix |
|---|---|
| `fig:acc_cost` never cited | referenced where accuracy and cost are discussed |
| `fig:ctrl_order` never cited | referenced where control error against mesh spacing is discussed |
| `tab:mlp` never cited | referenced in the MLP subsection (added last round with its section, not its citation) |
| `fig:bsde_stop` introduced in the architecture subsection, discussed in the BSDE one | moved next to the table it belongs to |
| `tab:main`, `tab:calib` introduced after their float | forward references added |
| terminology: 3 `grid solver` for the discretisation, 2 `neural method`, 2 `\S\ref`, 1 `approximately` | unified (one `grid solver` kept for an *implementation*, which is a real distinction) |
| 6 sentences over 80 words | split; worst is now 71 |

Also this round: the conclusion now carries the multilevel Picard result **with its condition** and the
one-term synthesis (the maximised Hamiltonian with coefficient $1/\eta$ limiting the grid, the
splitting recursion and MLP's contraction), which the abstract had and the conclusion did not.

**The checker needed three fixes before its findings could be trusted**, and each one had produced a
false positive that would have led to a pointless edit:

1. It scanned `main.tex` alone, so it saw **no tables at all** -- every table is `\input`. A check that
   silently skips fourteen floats reports success.
2. It measured citation distance from the float's **start**, and inlined table bodies run to ~1.6k
   characters, so two tables whose reference is the *next sentence* were flagged.
3. It counted **table captions and inline tabulars as prose**, so a 115-word "sentence" was really a
   paragraph plus a table.

The criterion that replaced distance is the right one: a float must be cited, and cited in the
subsection that introduces it. Distance was a proxy that produced two false positives and one true one.

```
python src/verify_all.py            -> 22 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 95 matched, 0 unmatched
python src/check_prose_consistency.py -> Nothing flagged
python src/check_tex.py             -> 0 structural problems; 60 cited keys of 60
python src/build_paper.py           -> 19 pages, 0 errors, 1 overfull box (1.73 pt)
```

New: `src/check_prose_consistency.py` (now part of the harness).

---

## Round 12 — operator learning: stated as a different axis, with a prediction (2026-10-14)

The user asked what the latest operator-learning methods are, and chose the discussion route rather
than adding a baseline. The paragraph is now in ``One problem or a family'', because that is the
subsection that already establishes the amortisation axis, and adding an operator row to
`tab_solvers` would have compared a family-trained surrogate against single-instance solvers.

What the paragraph says, in order:

1. **The field's direction**, cited: from fixed architectures --- spectral convolutions
   \cite{li2021fourier}, branch/trunk constructions \cite{lu2021learning},
   discretisation-invariance arguments \cite{raonic2023convolutional} --- towards models pretrained
   across equation families and specified at query time by examples rather than by weights
   \cite{yang2024pde}.
2. **Why it is a different axis**: an operator is trained on a family, so the honest comparison is
   amortised (train once, evaluate on held-out members, charge the classical methods a full re-solve
   each). That comparison is well posed here because every member has a closed-form reference.
3. **Why we did not run it**, stated as a property of the benchmark rather than a preference: the
   operator's input would be a low-dimensional parameter vector $(\lambda,\sigma,\eta,x_0)$, so the
   exercise tests learning a smooth map from a handful of numbers --- against methods already at
   $0.4\%$ (regression Monte Carlo) and $0.26\%$ (multilevel Picard) in seconds. Operator learning
   earns its place when the input is a \emph{function} (a volatility surface, an order-flow signal, a
   whole price path), which is the signalled and market-calibrated extension, and that is where we say
   we would place it.
4. **A prediction that ties it to the paper's theory**: Proposition~1 is not about grid schemes. Any
   solver whose delivered object is a derivative of a reconstructed field inherits the
   reconstruction's order and is capped by a differentiation step --- which includes an operator that
   outputs the value and obtains the control by differentiating it. So the recommendation of
   \S\ref{sec:control} transfers, and it is labelled a prediction because we did not measure it.

**Citation discipline.** Only what I could verify is cited. The DeepONet DOI and the in-context paper's
exact authors, volume and pages came from the Crossref record for `10.1016/j.jcp.2024.113379` (the
ICON paper, which itself is the JCP publication the discussion rests on); a NeurIPS foundation-model
entry was dropped because its author list could not be fetched here, and a wrong author list is worse
than one fewer citation.

```
python src/verify_all.py            -> 21 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 93 matched, 0 unmatched
python src/check_tex.py             -> 0 structural problems; 60 cited keys of 60
python src/build_paper.py           -> 19 pages, 0 errors, 1 overfull box (1.73 pt)
```

---

## Round 11 — is deep splitting's failure a budget problem? No (2026-10-14)

A background job from round 9 finished: the neural arm at $4\times64$ and 2000 epochs per step, roughly
$7\times$ the budget the paper reports. Its result closes the obvious escape hatch in the deep-splitting
claim:

| steps | epochs/step | arch | value $R^2$ | control $R^2$ | median step loss |
|---|---|---|---|---|---|
| 50 | 300 | $2\times32$ | $+0.16$ | $+0.864$ | $4.8\times10^{-2}$ |
| 50 | 2000 | $4\times64$ | $-16.5$ | $+0.886$ | $2.7\times10^{-4}$ |
| 100 | 2000 | $4\times64$ | $-4.8$ | $+0.863$ | $1.4\times10^{-4}$ |

At a **fixed** 50 steps, $6.7\times$ the gradient steps improves the median per-step fit by $179\times$
and moves the value $R^2$ from $+0.16$ to $-16.5$. A better fit to each step's target produces a worse
value function, while the control stays between $0.86$ and $0.89$ throughout. So the limitation is
neither budget nor capacity.

**And the mechanism is now measured rather than inferred.** `diag_splitting_gradient_error.py` compares,
at the trained networks, the *relative* error of the value against that of the recovered gradient ---
the quantity the recursion actually feeds forward. The gradient's relative error exceeds the value's by
$1.6$ to $4.3\times$ at the steps measured, with one exception near $t=0$. That is what a
mean-squared-error loss constraining the function rather than its derivative predicts, and the paper
now says "consistent with" rather than "because".

Two process points. The summary fields of the new result file were recomputed by `--resummarise` from
the stored rows rather than typed by hand (the sweep costs 24 minutes; derived data still has to come
from code). And one assertion failed first time because it checked the *midpoint* of a range the prose
quotes by its *endpoints* --- a reminder that the check is only as good as its agreement with the
sentence.

Also this round: the abstract now carries the multilevel Picard result ($0.26\%$ at $d=20$ in $2.3$\,s)
together with its condition, and the one-term synthesis --- the maximised Hamiltonian with coefficient
$1/\eta$ limiting the grid, the splitting recursion and MLP's contraction --- is stated there.

```
python src/verify_all.py            -> 21 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 93 matched, 0 unmatched  (was 86)
python src/check_orphan_results.py  -> OK
python src/build_paper.py           -> 19 pages, 0 errors, 1 overfull box (1.73 pt)
```

New: `src/exp_deep_splitting_budget.py`, `src/diag_splitting_gradient_error.py`,
`results/exp_deep_splitting_budget.json`, `results/diag_splitting_gradient_error.json`.

---

## Round 10 — multilevel Picard: the best high-dimensional result, and a shared limit (2026-10-14)

The last family that was cited and not run. `src/multilevel_picard.py` implements equation (5) of
Hutzenthaler–Jentzen–Kruse–Nguyen (arXiv:2009.02484) — the full-history recursive MLP approximation —
with one deviation stated rather than hidden: our nonlinearity consumes the **gradient**, so
$\nabla U_\ell$ comes from central differences (the practical route for gradient-dependent
nonlinearities; Neufeld–Wu's Beta-time-sampled variant is cited as the alternative).

**Where its hypothesis holds, it is the best method in the paper, and not narrowly.** MLP realises the
Picard iteration, whose Lipschitz constant here is $|\nabla V|/\eta$; at $\eta = 1$ that is $0.10$ and
the iteration converges:

| $d$ | 1 | 8 | 12 | **20** |
|---|---|---|---|---|
| rel. err | 0.64% | 0.43% | 0.17% | **0.26%** |
| seconds | 0.00 | 0.20 | 0.58 | **2.25** |

At $d=20$: 0.26% in 2.25 s on a CPU, against the PINN's $R^2 = 0.757$, regression Monte Carlo's 0.977
(12.5 s, hand-chosen basis) and a grid that cannot run at all.

**At the paper's own calibration it diverges — and the boundary is measured rather than asserted.**
The Picard Lipschitz constant is $1.25 > 1$; sweeping $\eta$ puts the transition between
$|\nabla V|/\eta = 0.56$ (converged) and $0.91$ (diverged). Three things make that a finding rather
than an excuse, and the first two exist because of last round's near-miss:

* the implementation is checked **where the hypothesis holds** (rel. err $6.7\times10^{-3} \to
  1.5\times10^{-3}$ from level 2 to 4), so a broken implementation cannot hide behind the divergence;
* the divergence is **not a budget**: the $d=20$ run that diverges takes 1.8 s;
* it is **not the finite-difference gradient**: $\varepsilon$ was varied over two orders of magnitude
  with no effect on the outcome — which is how the first hypothesis (gradient noise) was eliminated.

**The unifying statement this bought.** The maximised Hamiltonian
$\tfrac12\nabla V^\top BR^{-1}B^\top\nabla V$, with coefficient $1/\eta$, now limits **three**
methods in the paper: the grid's CFL step, the explicit deep-splitting recursion, and MLP's Picard
contraction. That is a cross-method claim about the benchmark class, and it is the kind of thing a
comparison paper is supposed to produce.

Also this round: a first MLP attempt divided the Monte Carlo noise by $2\varepsilon$ because the
up/down finite-difference evaluations used independent randomness. The fix (common random numbers) is
documented in the code, and the acceptance test that caught it is now the script's first action.

```
python src/verify_all.py            -> 21 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 86 matched, 0 unmatched  (was 79)
python src/check_tex.py             -> 0 structural problems; 58 cited keys of 58
python src/build_paper.py           -> 19 pages, 0 errors, 1 overfull box (1.73 pt)
```

New: `src/multilevel_picard.py`, `results/exp_mlp.json`, `paper/tables/tab_mlp.tex`, a new subsection
(``the classical probabilistic method, where it applies''), an MLP row in `tab_solvers` (12
configurations, 5 families), two bibliography entries, and M5b/M5c in `REFEREE_PREP.md`. The
"cited but not run" list is now empty.

---

## Round 9 — deep splitting, and a false negative that nearly shipped (2026-10-14)

The user asked for the addition that answers the strongest remaining objection: the deep-BSDE family
was represented only by its reference member. `src/deep_splitting.py` now implements the modern member
— one small network per time step, fitted backwards by regression on uncontrolled paths — with the
nonlinearity derived from this repository's conventions and cross-checked against the $dV$ identity the
deep-BSDE code already uses.

**The result, and the control experiment that makes it readable:**

| $d$ | deep splitting (neural regression) | same recursion, exact quadratic regression |
|---|---|---|
| 1 | value $R^2$ $+0.16$, control $+0.86$ | $+0.9987$ / $+0.9995$ |
| 4 | $-76.4$ / $-4.0$ | $+0.9960$ / $+0.9995$ |
| 8 | $-285.7$ / $-0.15$ | $+0.9939$ / $+0.9994$ |

Same recursion, same paths, same step count; only the per-step regression differs. So the scheme is
sound and the **learner** is the limitation — and the mechanism is this paper's own thesis: the
recursion feeds the *gradient* forward, $z = \Sigma^\top\nabla u$ enters $f$ with a coefficient
$M \sim 10^6$, so each per-step regression must be accurate in a derivative, while a mean-squared-error
loss constrains the function and not its derivative. The median per-step relative loss degrades from
$0.05$ at $d=1$ to $1.0$ at $d=8$. This is the value/control decoupling reappearing inside a neural
solver, and it is now in the paper (with both arms, so the attribution is visible to a referee).

### The part worth recording: I nearly published a false negative

Three separate times in this round my own diagnostic said "the splitting recursion is unstable on this
benchmark" — a clean, quotable, and **wrong** conclusion. Each time the cause was a quadratic-form
convention in my code, not a property of the method:

1. An algebraic recursion that ran its update **six times per step** (an `inner=6` default I had left
   in), integrating six time units instead of one and collapsing onto the Riccati fixed point.
2. `fitted_gradient` returned the gradient of $\tfrac12 x^\top Hx$ while `quad_features` emitted the
   raw products $x_ix_j$ — a factor of two, which quarters the feedback term $z^\top Mz$ and inflates
   the recursion's $P$ by $2.6\times$.
3. The same helper applied its factor of two to *off-diagonal* entries as well, which is right only for
   $d=1$. The self-test caught this one: $1.2\times10^{-10}$ for $d=1$ against $0.65$ for $d=3$.

What stopped it was the discipline the rest of this repository already uses, and it is worth stating
plainly because it is the cheapest lesson here: **the diagnostic now self-tests its gradient against a
finite difference before producing any number, and refuses to run otherwise.** Every one of the three
failures would have been caught by that check on the first run, and the near-miss is the reason the
paper reports *both* arms of the table: a negative result about someone else's method is only credible
if the same code succeeds when the only thing changed is the part under suspicion.

```
python src/verify_all.py            -> 21 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 79 matched, 0 unmatched  (was 75)
python src/check_tex.py             -> 0 structural problems; 56 cited keys of 56
python src/build_paper.py           -> 18 pages, 0 errors, 1 overfull box (1.73 pt)
```

The orphan-result audit added last round earned its keep immediately: it failed the harness on
`results/smoke.json`, a 200-step smoke test of this round's solver that nothing read. Removed.

New: `src/deep_splitting.py`, `src/exp_deep_splitting.py`, `src/diag_splitting_recursion.py`,
`results/exp_deep_splitting.json`, two rows in `paper/tables/tab_solvers.tex`, a new paragraph in
\S V (``and the family's modern member''), and M5/M5b in `REFEREE_PREP.md`.

---

## Round 8 — "did you only compare two methods?" (2026-10-14)

The question came from the user, and it found two real defects. The manuscript's headline table has
two columns (the two families that can be compared at *matched accuracy*), but it turned out that two
experiments had been **measured early and never reported at all**:

| experiment | file | why it matters |
|---|---|---|
| implicit LOD splitting scheme | `results/exp_implicit_grid.json` (30 rows, 3 problem cases) | the paper's cost claim rests on an *explicit* scheme, and the paper only *said* in the limitations that implicitness removes the restriction |
| PI-PINN, the second neural variant | `results/exp_pi_pinn.json` (16 runs, iteration sweep, 2 seeds) | the abstract already claimed "two PINN variants" — and no table ever showed the second one |

Both are now in the paper, via a new consolidated table (`tab:solvers`) that lists **nine solver
configurations across four families** with the operating point each was measured at, and both changed
what the paper says:

* the implicit scheme buys a factor of four rather than orders, because on this benchmark the
  stability restriction is set by the optimal **control**, not by the diffusion;
* PI-PINN at matched budget gives $R^2 = -0.15$ with two outer iterations and $0.9997$ with eight —
  the variant is conditional on a hyperparameter the direct variant does not have.

**A selection trap this audit exposed.** The accuracy-matched finite-difference run is *not* in
`exp_dimension.json`: that file holds the fixed-resolution sweep, whose $p=2$ row is a finer grid and
flatters the scheme by $10\times$ in both accuracy and time (0.99999 vs 0.99993; 0.518 s vs 0.056 s).
The first version of the new table silently picked the wrong one. Policy iteration likewise ran at
$N=41$, not at the $N=21$ the matched-accuracy column uses — the row now says so instead of implying
otherwise. Both are documented in the generator's docstring so the next edit cannot repeat them.

**Generalisation worth keeping:** the previous rounds verified that every number in the prose matched a
result file, which is a *consistency* check. It cannot catch an experiment that was run and never
written up. That gap is now closed by a check rather than by vigilance:
`src/check_orphan_results.py` fails if any `results/*.json` is read by nothing in `src/` or `paper/`,
with a short allowlist for smoke tests and merged intermediates. Its first run (after fixing an audit
bug of my own — LaTeX escapes underscores, so a naive scan reported 50 false orphans) found exactly
the three files that are legitimately unreported and **no further unreported experiments**. So the two
the question exposed were the only two.

```
python src/verify_all.py            -> 21 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 75 matched, 0 unmatched  (was 71)
python src/check_tex.py             -> 0 structural problems; 56 cited keys of 56
python src/build_paper.py           -> 18 pages, 0 errors, 1 overfull box (1.73 pt)
```

New: `paper/tables/tab_solvers.tex`, `src/check_orphan_results.py`, plus a new M0 entry in
`REFEREE_PREP.md`.

---

## Stage 2D — the paper reviewed as a referee would review it (2026-10-14)

`REFEREE_PREP.md` writes the report a skeptical JCP referee would write — ten major points, eight
minor — and answers each with the artifact that backs the answer, marking every one **Answered**,
**Partially** or **Open**. Writing it produced one new measurement, because the sharpest objection
could not be answered by assertion.

**The objection.** *"Your classical baseline is a straw man: you gave the regression the exact basis
and the neural solver no structure."* It is the strongest objection to the mesh-free result, and it is
correct in substance.

**The measurement** (`src/exp_lsmc_basis.py`, three seeded repeats per amplitude): the same regression,
the same quadratic basis, on the manufactured family whose value function becomes progressively less
quadratic.

| $A_0$ | regression $R^2$ |
|---|---|
| $0$ (basis exactly right) | $+0.9998$ |
| $10^{-4}$ | $+0.9994$ |
| $10^{-3}$ | $+0.956$ |
| $5\times10^{-3}$ | $-1.19$ to $-1.20$ |
| $2\times10^{-2}$ | **diverges** (3/3 repeats; the fitted policy drives the state out of the basis's region) |

$A_0 = 5\times10^{-3}$ is the amplitude of the non-polynomial term in the paper's *own* benchmark, so
the two experiments meet: the classical method that beats the neural solver at $d = 20$ fails on the
modified problem the paper constructs to answer a different objection. The paper now states the
qualified claim its evidence supports — the classical mesh-free method wins where the practitioner has
structural knowledge of the value function — and quantifies what that knowledge is worth.

**A reproducibility finding fell out of it.** At the failing amplitude the figure is not reproducible
across process runs: $-1.1941$, then $-1.13$ on a re-run of the identical script, while three seeded
repeats *inside* one run agree to $0.01$. The cause is threading non-determinism in the least-squares
solve being amplified by a chaotic regime — and the honest response is to quote a range in the paper
and say why, which is now done. The experiment was also removed from `verify_all.py` for the same
reason (a flaky check and a result-file overwrite are both worse than no check), with the reproduction
command recorded instead.

**A process mistake worth recording.** A one-line exit-code test of that script used its *default*
output path and overwrote the canonical result file with a two-iteration toy run, which then made a
prose assertion fail. It was caught immediately by `verify_paper_numbers.py` — the check did its job —
but the rule is: smoke-test with an explicit `--out`, never the default.

```
python src/verify_all.py            -> 20 passed, 0 failed, 1 skipped
python src/verify_paper_numbers.py  -> 71 matched, 0 unmatched  (was 65)
python src/journal_audit.py         -> 0 blockers, 2 author-only, 1 enhancement
```

New files: `src/exp_lsmc_basis.py`, `results/exp_lsmc_basis.json`, `REFEREE_PREP.md`.

---

## Consistency pass over the claims that changed (2026-10-14)

Every headline claim had been edited at least once during stages 0–2B, so the abstract, the
introduction's three findings, the decision guide and the conclusion were re-read against the final
numbers. Four stale statements were found and fixed, all of the same kind — a claim that was true
when written and had since been overtaken by a measurement:

| where | was | now |
|---|---|---|
| abstract, speed | "by a factor of $5$ to $50$ in time at $d=4$–$5$" | $5.6\times$ NumPy, $114\times$ compiled, at $d=4$ |
| abstract, scope of the high-$d$ claim | "measured where no classical baseline can run" | "where no *grid* can run" |
| introduction, first finding | "beyond that the mesh-free solver is the only method that runs" | the classical mesh-free method takes over, and is the more accurate of the two |
| conclusion | $5$–$50\times$ at $p=4$–$5$ | $5.6\times$ / $114\times$, and the $p=5$ sign flip stated |

The money result (2.7–8.4 bp) was added to the abstract, the introduction and the conclusion, since it
is the one number a practitioner will read first. `src/verify_paper_numbers.py` still reports
**65 matched, 0 unmatched** after the edits, which is the check that makes this kind of pass safe:
the prose cannot drift from `results/` without a failure.

```
python src/verify_all.py   -> 20 passed, 0 failed, 1 skipped   (the journal-package check is now in it)
python src/check_tex.py    -> 0 structural problems; 56 cited keys of 56
python src/journal_audit.py-> 0 blockers, 2 author-only items, 1 enhancement
```

---

## Stage 2C — the submission package (2026-10-14)

`src/build_journal.py` builds and checks everything a submission needs beyond the manuscript, and
enforces the two rules Elsevier checks mechanically: Highlights must be 3–5 bullets of at most 85
characters, and the declarations must exist as separate items.

| item | file | state |
|---|---|---|
| Highlights | `paper/journal/highlights.txt` | 5 bullets, longest **80** chars |
| Cover letter | `paper/journal/cover_letter.tex` → PDF | compiled, 2 pp |
| Declarations (CRediT, competing interest, data availability, funding, AI use) | `paper/journal/declarations.tex` → PDF | compiled, 2 pp |
| Nomenclature | `paper/journal/nomenclature.tex` → PDF | compiled, 1 p |
| Checklist + conversion recipe + DOI steps | `paper/journal/README.md` | written |

**On the journal's own class.** `elsarticle.cls` is not installed in the local TeX Live and could not
be fetched: every CTAN path tried returned 404 or an anti-bot page, and `tlmgr` refuses to install
until it self-updates. This is recorded rather than worked around, because it does not block anything:
Elsevier asks for free formatting at initial submission and for the class at revision. The conversion
recipe is written out in the package README — two class lines, the author block, the keyword
environment, the bibliography style — and nothing in the body depends on IEEEtran.

**What is deliberately left blank**, because only the authors can answer: funding, acknowledgements,
the generative-AI disclosure, the corresponding author, and the Zenodo release that turns the
repository URL into a DOI. The declarations file says so in each place rather than being filled with
plausible text.

```
python src/build_journal.py  ->  all required items present, highlights within limits,
                                 cover letter / declarations / nomenclature all compile
```

New files: `src/build_journal.py`, `paper/journal/{highlights.txt,cover_letter.tex,declarations.tex,nomenclature.tex,README.md}`,
`results/check_journal_package.json`.

---

## Stage 2A — the grid timings, separated into the scheme and the implementation (2026-10-14)

Every grid time in the paper comes from a NumPy implementation, and the paper conceded that a
compiled kernel would be faster. The concession is now a measurement: `src/exp_compiled_grid.py` runs
the same scheme as a fused parallel kernel (numba — the environment has no JAX) at identical work,
and checks that the two agree on the solved field.

| $p$ | NumPy (s) | kernel (s) | factor | rate, NumPy → kernel (GFLOP/s) |
|---|---|---|---|---|
| 1 | 0.023 | 0.009 | 2.5× | 0.01 → 0.01 |
| 2 | 0.054 | 0.015 | 3.7× | 0.14 → 0.53 |
| 3 | 0.581 | 0.076 | 7.7× | 0.58 → 4.41 |
| 4 | 18.476 | 1.558 | **11.9×** | 0.65 → 7.65 |
| 5 | 567.104 | 59.252 | 9.6× | 0.67 → 6.43 |

Field agreement: $1.3\times10^{-4}$ relative at every dimension — the edge closure, the interior is
identical (the kernel uses the direct second-order one-sided second derivative at the edges where the
NumPy path composes the first-derivative rule twice).

**The consequence is at $p=5$.** There, the NumPy grid was *slower* than the neural solver
(865 s against 226 s — the one dimension where the neural method won on time), and the compiled grid
is faster (59 s). At $p=4$ the accuracy-matched factor goes from 5.6× to 114×.

Two implementation notes worth keeping: the first version of this experiment timed the JIT
compilation itself and reported 0.0× in one dimension and 2.6× in the next (a warm-up call fixed it);
and the CFL bound must be sized from the *terminal* control, which is the largest one here — the same
mistake that broke the non-polynomial experiment in stage 1C.

## Stage 2B — what the reconstruction order is worth in money (2026-10-14)

`src/exp_execution_shortfall.py` runs the paper's real-market calibration through the executed closed
loop: the exact value function on the grid, each reconstruction driving the dynamics, and the
benchmark's own objective — whose units are money once $\sigma$ and $\eta$ are relative, so $10^4$
times the impact term is basis points of notional.

| instrument | $\eta$ (bp) | multilinear vs cubic (bp of notional) |
|---|---|---|
| SPY | 1.28 | 8.40 |
| AMZN | 1.46 | 7.43 |
| MSFT | 1.58 | 6.89 |
| QQQ | 1.60 | 6.81 |
| GOOGL | 1.97 | 5.59 |
| OXY | 2.24 | 4.96 |
| NVDA | 2.68 | 4.20 |
| AAPL | 2.73 | 4.14 |
| TSLA | 4.41 | 2.67 |

Mean **5.68 bp** (≈ \$5.7k on a \$10M order); the cubic arm is within **0.018 bp** of the exact
control, so the pairwise difference is a statement about the multilinear arm. The effect is largest on
the *most liquid* instrument — a tighter spread buys a larger optimal control, hence a larger
reconstruction error — which is the same ordering stage 1 found on the cost side.

**Two measurement traps, both found by checking rather than by assuming.** (i) Using the $t=0$
snapshot at every time applies a frozen policy to a problem whose optimal control is strongly
time-dependent: the resulting 34% position error dwarfed the effect being measured. Fixed by
interpolating between snapshots as the solver does. (ii) Truncating the integral at $T-\delta$ removes
the terminal penalty that pays for not liquidating, so the slower arm looks cheap and the pairwise
difference falls from $+8.4$ bp to $-0.21$ bp at $\delta = 0.1$. The full horizon is the well-posed
setting; the sensitivity is in the reproduction and in the caption. The figure moves by less than
0.002 bp across $\lambda \in [10^{-6}, 10^{-2}]$.

**Paper.** Two new tables (`tab_compiled`, `tab_shortfall`) and two new subsections; the "5–50×"
speed claim now reads 5.6×–114× depending on implementation, and the $p=5$ sign flip is stated.
**17 pages, 0 errors, 1 overfull box (1.73 pt)**.

```
python src/verify_all.py            -> 18 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py  -> 65 matched, 0 unmatched  (was 54)
python src/check_tex.py             -> 0 structural problems; 56 cited keys of 56
```

New files: `src/exp_compiled_grid.py`, `src/exp_execution_shortfall.py`,
`results/{exp_compiled_grid,exp_execution_shortfall}.json`,
`paper/tables/{tab_compiled,tab_shortfall}.tex`.

---

## Stage 1D — the classical mesh-free baseline, and a conclusion it changed (2026-10-14)

The paper's high-dimensional claim was that beyond $p \sim 5$ "the mesh-free solver is the only method
that runs" — true of the methods compared, and not true of methods in general, because the classical
family that also needs no mesh (simulation with backward regression) was cited and never run. A
referee would name it. It is now run.

**Method** (`src/exp_lsmc.py`): policy iteration with backward regression — sample states across the
domain, simulate, regress the realised future reward at every time slice on the quadratic features,
improve the policy to $u = R^{-1}B^\top\nabla V$ from the fitted quadratic, repeat. Same problem as
the high-dimensional sweep (`ACProblem(p=d, hetero=True, het_seed=1)`) and the same exact reference.

**Result** (Table `tab_meshfree`, 40k paths × 60 steps × 12 iterations, damping $\alpha = 0.5$):

| $d$ | PINN $R^2$ | **RMC $R^2$** | RMC $m$ | RMC (s, CPU) | PINN (s, GPU) |
|---|---|---|---|---|---|
| 1 | +1.0000 | **+1.0000** | 3 | 0.1 | 62 |
| 2 | +0.9994 | **+0.9998** | 6 | 0.5 | 85 |
| 3 | +0.9992 | **+0.9998** | 10 | 0.7 | 104 |
| 5 | +0.9771 | **+0.9995** | 21 | 1.5 | 145 |
| 8 | +0.9813 | **+0.9988** | 45 | 2.7 | 196 |
| 12 | +0.9269 | **+0.9989** | 91 | 5.1 | 270 |
| 20 | +0.7570 | **+0.9771** | 231 | 12.5 | 414 |

**The classical regression method is more accurate than the neural solver at every dimension where
both were measured, at a fraction of the cost.** The claim that the neural solver is what remains
beyond the grid's reach is false as stated, and the paper now says so in the abstract, the
introduction, the decision guide (which gained a row), Related Work and the conclusion.

**Qualifications, both in the paper.** (i) The basis is the exact structure of this benchmark's value
function — a real advantage the neural solver is not given — so this is a statement about the problem
class, and Section~\ref{sec:nonpoly} is the pointer to what happens when the solution stops being
quadratic. (ii) The neural numbers come from a fixed 8000-epoch budget at every dimension rather than
a per-dimension tuning, so the comparison is at stated budgets and not at matched accuracy.

**Three implementation facts, each of which the first version got wrong and none of which is free:**
paths must start *across the domain* (started from $x_0$ alone, the cloud has radius
$\sigma\sqrt{T}=0.01$ around a corner, the Hessian is meaningless, and the run diverged to $10^{33}$ in
two iterations); the greedy update must be *damped* (gain $1/\eta = 100$, so a Hessian error of
$10^{-3}$ is a feedback-gain error of $0.1$); and the regression must carry the exact terminal
condition, which is part of the problem rather than part of the answer.

**Paper.** New subsection "The mesh-free region is not the neural solver's alone" plus Table
`tab_meshfree`; references 54 → 56 (Kharroubi–Langrené–Pham; Bender–Steiner). **16 pages, 0 errors,
1 overfull box (1.73 pt)**.

```
python src/verify_all.py            -> 18 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py  -> 54 matched, 0 unmatched  (was 48)
python src/check_tex.py             -> 0 structural problems; 56 cited keys of 56
```

New files: `src/exp_lsmc.py`, `results/exp_lsmc.json`, `paper/tables/tab_meshfree.tex`.

---

## Stage 1C — is the comparison an artefact of the quadratic solution? (2026-10-14)

The objection a referee would raise first: every accuracy number rests on a value function that is a
quadratic form, a second-order scheme reproduces a quadratic almost exactly, and the accuracy-matched
resolution came out at $N^\star = 21$ in *every* dimension. That is a property of the solution, not of
the solvers.

**The construction.** Keep the solvers, change the solution. Impose
$V = -\tfrac12 x^\top P(t)x + A_0(T-t)\sin(\omega\,\mathbf{1}^\top x)$ (non-polynomial, non-separable,
and still equal to $-\tfrac12x^\top Q_Tx$ at $t=T$) and add to the running reward the source that
makes it exact. Because the source does not depend on $u$, the maximiser is unchanged and **the same
solver, stencil and CFL bound apply** with one added term — the hook is
`_hamiltonian_and_control(V, t)` in `solvers.py`, and problems without a `source` attribute take
exactly their previous code path (verified: FDM p=2 N=21 still gives value R² 0.999929).

**Result** (Table `tab_nonpoly`, `src/exp_nonpolynomial.py`):

| | d=1 | d=2 | d=3 |
|---|---|---|---|
| quadratic solution: $N^\star$ for the target | 21 | 21 | 21 |
| non-polynomial: value rel. $L^2$ at $N=61$ | 1.6e-3 | 2.0e-3 | 2.3e-3 |
| target $10^{-3}$ reached at $N=61$? | **no** | **no** | **no** |
| value order | 2.16 | 2.11 | 2.03 |
| control order | 1.13 | 1.08 | 1.04 |
| cubic gain over multilinear, same nodes | 9.3× | 16.6× | 15.5× |

* The dimension-free resolution **was** an artefact of the quadratic solution: the same target is not
  reached at $N=61$ anywhere, where the quadratic case reached it at $N=21$ with room to spare.
* The scheme is not degraded (value order still ≈2), and the control order is ≈1 as Proposition 1
  predicts for a piecewise-linear reconstruction. **The headline result survives**: the cubic
  reconstruction keeps a 9–17× advantage on the same nodes where a cubic is no longer exact.
* Honest qualification: with $\omega=3$ the dominant error is the CFL-tied *time* step, which is
  dimension-independent, so the failure to reach the target is about resolution, not about dimension.
  Pushing to $\omega=12$ (amplitude-matched, so the control magnitude is unchanged) raises the error
  at $N=61$ to $1.2\times10^{-2}$ ($d=1$) and $5.1\times10^{-3}$ ($d=2$) with apparent orders 2.8–3.1,
  and the cubic's advantage falls to $1.7$–$2.7\times$ — when the solve itself is wrong, a better
  reconstruction cannot rescue it. The source is a device, not a market friction, and the paper says so.

**Two failed first attempts, both worth keeping:** a hand-written reimplementation of the solver froze
the boundary values instead of using `edge_order=2` one-sided stencils and went unstable before
producing a number (hence: reuse the real solver); and the $\omega=12$ variant first destabilised the
explicit step because the CFL estimate sizes the step from the terminal control only.

## Stage 1E — where the error lives: boundary band versus interior (2026-10-14)

A point within one cell of the state box's edge lies where the stencils are one-sided and the
reconstruction has no node on the far side, and that band is 13% of the domain at $p=1$, 33% at $p=3$
and 41% at $p=5$ for $N=21$. If the band carried most of the error, every domain-wide average in the
paper would be a boundary statistic.

It does not. `src/exp_boundary_split.py`, at $N=21$:

| d | boundary share | control-error density, band / interior | cubic gain, band | cubic gain, interior |
|---|---|---|---|---|
| 1 | 13% | 0.99 | 13.9× | 25.8× |
| 2 | 24% | 1.00 | 16.9× | 25.8× |
| 3 | 33% | 1.00 | 18.6× | 25.8× |

The error density is uniform, so the band contributes exactly its share of the domain. Three notes
are in the paper: the solvers pad the domain by 0.25, so the *grid's* outer cells are never queried
(measured boundary share 0.000 when the mask was written that way — a fact about the padding, not
about the error); normalising the error *inside* each region flatters the boundary, because $|V|$ is
largest there, which is what the first version of this experiment reported; and one asymmetry does
survive — the cubic's advantage is smaller in the band (14–19×) than in the interior (26×).

**Paper.** Two new subsections (one in Results, one in the control-accuracy section) plus Table
`tab_nonpoly`; **15 pages, 0 errors, 1 overfull box (1.73 pt)**.

```
python src/verify_all.py            -> 18 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py  -> 48 matched, 0 unmatched  (was 33)
python src/check_tex.py             -> 0 structural problems; 54 cited keys of 54
```

New files: `src/exp_nonpolynomial.py`, `src/exp_boundary_split.py`,
`results/{exp_nonpolynomial,exp_nonpolynomial_omega12,exp_boundary_split}.json`,
`paper/tables/tab_nonpoly.tex`.

---

## Stage 1B — order versus monotonicity (2026-10-14)

Proposition 1 says raise the reconstruction order; the Barles--Souganidis framework says the scheme
must be monotone, and a high-order reconstruction is not. This stage separates the two and measures
the trade instead of asserting it.

**Where the conflict binds.** Three regimes, which this benchmark separates cleanly:
(i) *post-processing* — the LQ Hamiltonian maximiser is analytic, so the reconstruction never enters
the value scheme and its monotonicity is untouched (this is why Proposition 1 can be had without
reopening the convergence argument, and it is a property of the benchmark);
(ii) *policy iteration* — the extracted gradient defines the frozen policy, which does enter the
evaluation operator, but monotonicity is a property of how the frozen advection is discretised; our
implementation upwinds it, and an earlier central-difference version was unconditionally unstable
near $t=T$, which is how the requirement was found here;
(iii) *a general HJB* — $\nabla V$ enters the Hamiltonian discretisation itself, the conflict is
real, and it is why ENO/WENO exists.

**What was measured** (`src/exp_limited_reconstruction.py`, Table `tab_limit`). A monotonised-central
post-limiter built from the nodal one-sided slopes, against the unlimited cubic:

| data condition | cubic wrong-way | limited wrong-way | cubic overshoot | limited overshoot | order |
|---|---|---|---|---|---|
| smooth, monotone | 0.0% | 0.0% | 1.00× | 1.00× | 3.41 / 3.41 |
| smooth, interior turning point | 0.0% | 0.0% | 1.01× | 1.00× | 3.72 / **1.51** |
| gradient kink | 0.2% | 0.0% | 1.33× | 1.24× | — |
| 2% nodal noise (the solved-grid case) | **8.6%** | **0.0%** | **2.26×** | 1.64× | — |

"Wrong way" means the recovered gradient has the *opposite sign* to a significant true gradient —
for a control, a trade in the wrong direction, not a small error. Returning zero at a turning point
is the safe answer and is not counted.

**The finding is conditional, and the condition is checkable before any solve.** On smooth data the
limiter is inert and can only cost order (at an interior turning point, $3.72 \to 1.51$). On data
that are not smooth — a kink in the value function, a constraint, or nodal values carrying the
solver's own error — the unlimited cubic points the wrong way at up to 8.6% of samples and overshoots
by $2.26\times$, while the limited reconstruction never does. So: raise the order, and limit it where
the nodal data are not monotone.

One methodological note worth keeping: the first version of this experiment reported *no*
monotonicity problem at all, because the test functions were smooth and the metric counted a
limiter's deliberate zero at a turning point as a failure. Both had to be fixed before the trade was
visible — the failure mode needs non-smooth data or noisy nodal values to appear.

**Paper.** New subsection "Monotonicity: when a high-order reconstruction must be limited" plus
Table `tab_limit`; references grew 40 → 54 (ENO/WENO, gradient recovery, Barles--Souganidis,
Crandall--Lions, semi-Lagrangian, sparse grids, PDE-constrained optimal control, deep BSDE variants,
multilevel Picard), all 54 cited. Manuscript now **14 pages, 0 errors, 1 overfull box (1.73 pt)**.

```
python src/verify_all.py            -> 18 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py  -> 33 matched, 0 unmatched  (was 25)
python src/check_tex.py             -> 0 structural problems; 54 cited keys of 54
```

New files: `src/exp_limited_reconstruction.py`, `results/exp_limited_reconstruction.json`,
`paper/tables/tab_limit.tex`.

---

## Stage 1A — two propositions, and the measurement that confirms them (2026-10-14)

The paper's central claim was a numerical observation. It is now two propositions with a numerical
confirmation each, and the confirmation changed what the paper can say.

**Proposition 1 (reconstruction and probe).** A degree-$k$ reconstruction of a $C^{k+1}$ function
delivers gradient order $k$; a symmetric-difference probe at step $\varepsilon$ adds an independent
$O(\varepsilon^2)$, so the attainable order is $\min(k, 2\alpha)$ for $\varepsilon = c\,h^{\alpha}$,
and the reconstruction's own order survives only if $\varepsilon = o(h^{k/2})$.

This is what reconciles the two order tables that previously disagreed, and it turned the
disagreement into the actionable half of the recommendation: **a probe that shrinks only like $h$
caps a cubic reconstruction at order two.** New evidence:

| probe scaling | measured cubic order | predicted $\min(3, 2\alpha)$ |
|---|---|---|
| $\varepsilon = 0.25h$ | 2.10–2.37 | 2 |
| $\varepsilon = 0.25h^{1.25}$ | 2.85–3.29 | 2.5 (crossover) |
| $\varepsilon = 0.25h^{1.5}$ | 3.32–3.79 | 3 |
| $\varepsilon = 0.25h^{2}$ | 3.36–3.83 | 3 (reconstruction ceiling $\approx$ 3.4) |

(`src/exp_probe_scaling.py`, `results/exp_probe_scaling.json`; the multilinear rows stay flat at
0.92–1.16 for every $\alpha$, as $k=1$ predicts.)

**Proposition 2 (control error and objective loss).** For the LQ benchmark,
$J(u^\star) - J(u) = \tfrac12\mathbb{E}\int_0^T \delta u_t^\top R\,\delta u_t\,dt$ **exactly**, where
$\delta u_t = u(x_t,t) - u^\star(x_t,t)$ is evaluated along the path actually taken. Hence the
objective loss is $O(h^{2k})$ for a degree-$k$ reconstruction — a rate, not the bound the envelope
theorem would give — and the position error obeys the first-order bound
$\|x_t - x^\star_t\| \le T\|B\|\sup_t\|\delta u_t\|$.

`src/verify_suboptimality_identity.py` checks it against trajectories integrated under the paper's
own reconstructions:

* smooth controlled perturbation: agreement $4\times10^{-10}$ relative once the checker's reference
  is exact;
* the residual was traced, not waved away: it falls as $n_{\text{ref}}^{-1.9}$ when the reference's
  piecewise-linear $P(t)$ is refined, i.e. it was the *verifier's* interpolation error
  (`ACProblem.reference` uses `np.interp` on a 401-point Riccati solve — worth knowing independently);
* with a cubic reconstruction (continuous gradient) the identity agrees to $9\times10^{-9}$; with the
  multilinear one, whose gradient is piecewise **constant** and therefore makes the closed-loop
  dynamics only piecewise smooth, agreement degrades to $7\times10^{-5$. That asymmetry is a real
  additional cost of a piecewise-constant control and is now stated in the paper;
* the position-error bound holds in every case.

**What the paper gained.** A new Section "What Governs Control Accuracy" (Propositions 1 and 2 with
proof sketches, one page), the probe-scaling result, and a derived explanation of the measured loss
order $2.1$ (previously attributed to "the envelope theorem" — an appeal the identity replaces with
an equality). Manuscript now **13 pages, 0 errors, 1 overfull box (1.73 pt, invisible)**.

```
python src/verify_all.py   -> 18 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py -> 25 matched, 0 unmatched
```

New files: `src/exp_probe_scaling.py`, `src/verify_suboptimality_identity.py`,
`results/{exp_probe_scaling,check_suboptimality_identity}.json`.

---

## Stage 0 — one metric, and two data defects (2026-10-14)

Stage 0 existed because a journal referee would have found these before we did. Two defects, both
invisible to the existing checks, both now fixed and both now covered by a check that fails if they
come back.

### The metric: three tables, two scales

`exp_control_extraction.py`, `exp_merton_extraction.py` and `exp_control_order.py` store a
**normalised mean squared error** and fit the slope of `log(MSE)` against `log h`. Because
`log(E²) = 2 log(E)`, that slope is **twice** the convergence order. `exp_reconstruction_order.py`
and `exp_control_consequence.py` fit the error norm instead. The manuscript read all of them as
"order", so the same multilinear control error appeared as **order 2.00** in one table and
**1.03–1.10** in another, and the headline control/value ratio was a ratio of *squared* errors.

Fixed by defining one metric in `src/metrics.py` — the relative L² error, orders fitted to the error
— and converting every stored quantity exactly (`error = √MSE`, `order = order_mse/2`,
`ratio = √ratio_mse`). No experiment was re-run; `src/recompute_metrics.py` rebuilds everything into
`results/metrics_l2.json`, which is now the single source the prose numbers are taken from.

| where | was | now |
|---|---|---|
| control-error order, p = 1…5 | 2.00, 1.96, 1.96, 1.96, 1.96 | **1.00, 0.98, 0.98, 0.98, 0.98** |
| multilinear reconstruction | "second order" | **first order** |
| cubic reconstruction (practical probe ε = h/4) | "fourth order" (4.4–5.6) | **2.2–2.8** |
| cubic reconstruction (probe ≪ h) | not measured | **3.4–3.8** (new run) |
| control/value ratio, solved grid p=2 N=61 | 184 | **13.6** (cubic: 2.8 → **1.7**) |
| abstract's ratio range | 10 to 684 | **3.2 to 26** |
| "four to five orders of magnitude" (cubic vs multilinear, fixed mesh) | MSE ratio | **2.1–2.6 orders** in error norm |

**A finding fell out of this, which is why the probe row was added.** With a probe step far below the
mesh the cubic reconstruction reaches 3.4–3.8; with the practical ε = h/4 it reaches 2.2–2.8. The
probe step, not the reconstruction, caps the order — so the recommendation is not only to raise the
order but to differentiate consistently with it. `tab_extract` now carries both rows, which is what
reconciled the two order tables instead of hiding the disagreement.

### The deep-BSDE table carried the wrong dimension

`exp_bsde_dims.json` holds the d = 4 and the d = 8 run in one file, and `make_bsde_table.py`'s
`load_run` took the smallest error across **both**, ignoring the requested dimension. The p = 4 row
therefore carried the d = 8 run's accuracy, residual and floor.

| p = 4 row | was | now |
|---|---|---|
| rel. error at x₀ | 0.003 | **0.033** |
| residual at stop | 7.17e-01 | **5.21e+00** |
| training time (s) | 696 | **192** |

The prose was right and the table was wrong, so the two contradicted each other — and the error ran
in the direction that *flattered* deep BSDE and weakened the paper's own conclusion. Fixed by
filtering on the dimension, and `verify_paper_numbers.py` now asserts the per-dimension value errors,
so a future mix-up fails a check rather than reaching a table.

### Why every existing check passed anyway

`check_table_drift.py` regenerates each table from `results/` and compares — with a buggy generator it
reproduced the same wrong number faithfully. `verify_paper_numbers.py` asserted the prose, which was
correct. **"Reproducible" is not "correct": regenerating from a wrong upstream reproduces the error.**
The new checks assert the *quantity and its scale*, not just the file-to-file match.

### Verified after stage 0

```
python src/verify_all.py              -> 17 passed, 0 failed, 2 skipped
python src/verify_paper_numbers.py    -> 25 matched, 0 unmatched, 0 skipped   (was 7 assertions)
python src/check_tex.py               -> 0 structural problems
python src/check_order_convention.py  -> regression guard; the two scales now agree
python src/build_paper.py             -> 12 pages, 0 errors, 1 overfull box (1.73 pt, invisible)
make_tables.py / make_journal_figures.py -> byte-identical on re-run
```

Files changed in stage 0: `paper/main.tex`, `paper/tables/{tab_extract,tab_control,tab_guide,tab_bsde}.tex`,
`paper/main.pdf`, `figures/fig_j1_control_order.*`, `src/{metrics,recompute_metrics,check_order_convention}.py` (new),
`src/{make_tables,make_bsde_table,make_journal_figures,verify_paper_numbers,journal_audit,verify_all}.py`,
`results/{metrics_l2.json,exp_control_extraction_intrinsic.json}` (new).

---

# Historical: the conference-era work (kept because every measurement under it still stands)

**Status of that phase: complete and compiled.** All experiments run, all figures and tables generated from
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
