# Referee preparation — anticipated objections, and what backs each answer

Written in the voice of a skeptical reviewer for the *Journal of Computational Physics*, then answered
point by point. The purpose is to find the objections **before** a referee does, and to be honest about
which ones cannot be answered from what we have.

Legend for each answer:
**Answered** — there is a measurement in the paper or the repository;
**Partially** — answered with a stated limitation;
**Open** — we cannot answer it, and the paper says so.

---

## Major points

### M0. "How many methods did you actually compare? Your main table shows two."

*The referee's version:* a two-column headline table reads as a two-method paper, and the families you
cite (multilevel Picard, deep splitting, operator learning, sparse grids, semi-Lagrangian) are cited
rather than run.

*Our answer.* **Answered, and the question found two real defects** — this objection came from the
user, not from us, and it was the most productive single question asked about the manuscript.

Nine solver configurations across four families were measured, and Table `tab:solvers` now lists all
of them with the operating point each was measured at:

| family | solvers measured |
|---|---|
| grid | explicit finite differences (accuracy-matched), implicit LOD splitting, Howard policy iteration |
| neural | direct PINN ($4\times64$), PI-PINN (policy-iteration variant, iteration sweep) |
| probabilistic | deep BSDE (point-valued, dimension sweep) |
| classical mesh-free | regression Monte Carlo ($d = 1 \ldots 20$) |

Two of those rows had been **measured and never reported**: the implicit splitting scheme, and the
PI-PINN sweep — while the abstract already claimed "two PINN variants". Both are now in the paper, and
both changed what the paper says:

* The implicit scheme shows the CFL restriction is removable but that here it is set by the *control*
  rather than the diffusion: a $4\times$ step holds at $R^2 = 0.9997$ on the signalled problem and at
  $0.9966$ on the inventory-only one.
* The PI-PINN sweep shows the second neural variant needs its iterations resolved: at matched budget,
  two outer iterations give $R^2 = -0.15$ (worse than predicting the mean) and eight give $0.9997$.

*Residual risk, unchanged and now stated in one place:* the families that are cited and still not run
are multilevel Picard, deep splitting and operator learning (M5). What the audit fixed is the
*reporting* gap, not that one.

### M1. "This is a benchmark, not a contribution."

*The referee's version:* the paper measures existing methods on a known problem; CJP publishes methods
and analyses, not horse races.

*Our answer.* **Answered**, and this was the largest change made in preparing the submission. The paper
now contains two propositions rather than observations:

* **Proposition 1** gives the order a degree-$k$ reconstruction delivers and — the part that is not
  textbook — the interaction with the differentiation step: a symmetric-difference probe at
  $\varepsilon = c\,h^{\alpha}$ caps the attainable order at $\min(k, 2\alpha)$, so the reconstruction's
  own order survives only for $\varepsilon = o(h^{k/2})$. This is verified in two independent ways
  (Table `tab_extract`, and `exp_probe_scaling.py`, which sweeps $\alpha$ and recovers the predicted
  orders).
* **Proposition 2** is an exact identity: the suboptimality gap equals the $R$-weighted squared control
  deviation accumulated along the path actually taken. It converts "the control matters too" into a rate
  ($O(h^{2k})$ of objective for a degree-$k$ reconstruction), and it is verified numerically to
  $4\times10^{-10}$ relative (`verify_suboptimality_identity.py`).

The benchmark remains a benchmark; what changed is that it now tests a stated rate rather than
reporting a table.

*Residual risk.* A referee may still call Proposition 1 classical. It is close to classical *for
interpolation alone*; the probe-step interaction and its practical consequence are what we claim, and
the paper says explicitly that raising the order of a recovered derivative is not itself new.

### M2. "The most striking result — a classical method beating the neural one — is a straw man: you gave the regression the exact basis."

*The referee's version:* the value function is quadratic and the basis is quadratic; the neural solver
was given no such structure.

*Our answer.* **Answered**, and the answer is uncomfortable for us, which is why it is in the paper.
The qualification is now measured rather than asserted (`exp_lsmc_basis.py`): the same regression, the
same quadratic basis, on a family of problems whose value function becomes progressively less
quadratic.

| $A_0$ (amplitude of the non-polynomial term) | regression $R^2$ |
|---|---|
| $0$ (basis exactly right) | $+0.9998$ |
| $10^{-4}$ | $+0.9995$ |
| $10^{-3}$ | $+0.956$ |
| $5\times10^{-3}$ | $-1.19$ |
| $2\times10^{-2}$ | **diverges** (the fitted policy drives the state out of the basis's region) |

$A_0 = 5\times10^{-3}$ is the amplitude used in the paper's own non-polynomial benchmark, so the two
experiments meet: the classical method that beats the neural solver at $d = 20$ fails on the modified
problem this paper constructs to answer a different objection.

The paper states the conclusion in the form the evidence supports: the classical mesh-free method wins
where the practitioner has structural knowledge of the value function, and the neural solver is the
fallback where they do not. What is *not* claimed is that regression Monte Carlo dominates neural
solvers for high-dimensional HJB equations in general.

### M3. "The benchmark is linear-quadratic, so the classical scheme is flattered."

*The referee's version:* on a quadratic value function a second-order scheme is almost exact, and the
accuracy-matched resolution came out dimension-free ($N^\star = 21$ at every $d$), which is a property
of the solution, not of the solvers.

*Our answer.* **Answered** — this was the first objection we attacked, and it is Section
"is the comparison an artefact of the quadratic solution?" (Table `tab_nonpoly`). A manufactured
non-polynomial exact solution is imposed by adding a source to the running reward; the maximiser is
unchanged, so the *same solver, stencil and CFL bound* apply. The result: the same accuracy target is
not reached at $N=61$ in any dimension, where the quadratic benchmark reached it at $N=21$ with room to
spare. The headline control result survives where a cubic reconstruction is no longer exact (a
$9$–$17\times$ advantage over multilinear on the same nodes).

*Honest weakness, stated in the paper.* With $\omega = 3$ the dominant error is the CFL-tied time step,
which is dimension-independent, so the failure to reach the target is about resolution rather than
about dimension; and the source is a device producing an exact solution, not a market friction.

### M4. "The timing comparison is not like for like: NumPy CPU against GPU, and then a hand-written kernel against a NumPy baseline."

*The referee's version:* the speed claim is an artefact of comparing a compiled neural solver with an
interpreted grid one, and then of comparing your own kernel with your own slower implementation.

*Our answer.* **Answered** in two parts.
(i) The comparison is stated as GPU-for-neural against CPU-for-grid, which *favours the neural method*,
and the analytic FLOP count (which is hardware-free) is reported alongside. Counting work flatters the
neural method by about $10^2$, so the wall-clock claim is the conservative one.
(ii) `exp_compiled_grid.py` runs the same scheme as a fused parallel kernel at identical work and
checks that the two implementations agree on the solved field ($1.3\times10^{-4}$ relative). The
implementation factor is $2.5$–$11.9\times$, i.e. the paper's original timings were *pessimistic* for
the grid method, and at $p=5$ the compiled grid is faster than the neural solver where the NumPy one
was slower. Both numbers are reported (Table `tab_compiled`).

*Residual risk.* We did not run the neural solver on the CPU for a like-for-like device comparison, and
we did not tune the neural budget per dimension. Both are stated.

### M5. "Score against a strong baseline, not the reference member of the deep-BSDE family."

*The referee's version:* deep BSDE has variants (deep splitting, Malliavin-weighted, multilevel Picard)
that behave far better than the plain scheme; beating the plain one is not evidence.

*Our answer.* **Answered for deep splitting**, which is the variant a referee would name first.
`src/deep_splitting.py` implements it — one small network per time step, fitted backwards by
regression on uncontrolled paths, with the nonlinearity derived from this repository's conventions and
cross-checked against the $dV$ identity the deep-BSDE code already uses — and the scheme is validated
against the analytic conditional expectation before anything is compared (per-step deviation falls
$6.6\times10^{-2} \to 6.9\times10^{-4}$ from 5 to 80 steps, i.e. consistent at the expected order).

The result, and the part that makes it a finding rather than a number:

| arm (same recursion, same paths, $d=1$, 50 steps) | value $R^2$ | control $R^2$ |
|---|---|---|
| deep splitting, neural per-step regression | $+0.16$ | $+0.86$ |
| the same recursion, exact quadratic regression | $+0.9987$ | $+0.9995$ |

At $d=4$ and $d=8$ the neural arm gives $-76$ and $-286$ while the exact-quadratic arm gives $0.996$
and $0.994$. The recursion is therefore sound and the **learner** is what limits it — and the reason is
the paper's own thesis: the recursion feeds the *gradient* forward, $z = \Sigma^\top\nabla u$ enters
$f$ with a coefficient of order $10^6$, so the per-step regression must be accurate in a derivative,
while its mean-squared-error loss constrains the function and not its derivative. The median per-step
relative loss degrades from $0.05$ at $d=1$ to $1.0$ at $d=8$.

*Open:* nothing from this family's two named variants — deep splitting and multilevel Picard are both
now implemented, validated and reported (§M5 and §M5c). What remains unrun is the *tuned* deep-BSDE
literature (Malliavin-weighted, multilevel Picard for BSDEs rather than PDEs), which is a different
axis from the one this objection turns on.

### M5b. "Your negative result about deep splitting is a straw man."

*The referee's version:* an unsmiling reading of the above — the implementation is yours, and a
negative result from a home-made implementation is worth little.

*Our answer.* **Answered by construction, and this is why that table has two arms.** A negative result
about someone else's method is only credible if the same code, on the same paths with the same step
count, succeeds when the only thing changed is the part under suspicion. The exact-quadratic arm is
that control, and it succeeds. The budget is stated rather than implied ($50$ steps $\times$
$300$ epochs/step, $2\times32$ per step, $4096$ paths), and it is comparable to what the deep-BSDE runs
in the paper receive.

*What we do not claim:* that deep splitting is a weak method. We claim that on this benchmark its
per-step regression must be accurate in a derivative, and we show a gradient-blind implementation
failing and a gradient-exact one succeeding under otherwise identical conditions.

### M5c. "Then run multilevel Picard — it is the classical method with a complexity theory."

*The referee's version:* the strongest version of the "weak baseline" objection. MLP has published
complexity bounds proving it beats the curse of dimensionality, so if your benchmark defeats it, say
why.

*Our answer.* **Answered, and it produced the paper's strongest high-dimensional result and its
sharpest limitation, in the same table.** `src/multilevel_picard.py` implements equation (5) of
Hutzenthaler–Jentzen–Kruse–Nguyen, with the gradient obtained by central differences (the practical
route for gradient-dependent nonlinearities; the Beta-time-sampled variant of Neufeld–Wu is the
alternative and is cited).

Where its hypothesis holds it is the best method in the paper, and not narrowly. MLP realises the
Picard iteration, whose Lipschitz constant here is $|\nabla V|/\eta$. At $\eta = 1$:

| $d$ | rel. err | time |
|---|---|---|
| 1 | 0.64% | 0.00 s |
| 8 | 0.43% | 0.20 s |
| 12 | 0.17% | 0.58 s |
| **20** | **0.26%** | **2.25 s** |

At $d=20$: 0.26% in 2.25 s on a CPU, against the PINN's $R^2 = 0.757$, regression Monte Carlo's
0.977 (12.5 s, hand-chosen basis) and a grid that cannot run at all.

At the paper's own calibration ($\eta = 0.01$) it **diverges**, and the boundary is measured: the
Picard Lipschitz constant is $1.25 > 1$, and sweeping $\eta$ puts the transition between
$|\nabla V|/\eta = 0.56$ (converged) and $0.91$ (diverged). Three things make this a finding rather
than an excuse:

* the implementation is checked **where the hypothesis holds** — relative error falls
  $6.7\times10^{-3} \to 1.5\times10^{-3}$ from level 2 to level 4 — so a wrong implementation cannot
  hide behind it;
* the divergence is not a budget: the $d=20$ run that diverges takes 1.8 s;
* it is not the finite-difference gradient either: varying $\varepsilon$ over two orders of magnitude
  leaves the outcome unchanged.

It is the same $1/\eta$ stiffness that sets the grid's CFL limit and destabilises the explicit
splitting recursion. Three methods in the paper are limited by that one term, which is exactly the
kind of cross-method statement a benchmark paper should be able to make.

### M6. "The regression Monte Carlo baseline is not a general HJB solver; the basis was chosen for the problem."

*The referee's version:* same as M2 but sharper — the class of methods you compare against is
restricted to methods that already know the answer's structure, so the comparison says nothing about
the high-dimensional regime the paper is about.

*Our answer.* **Partially.** True, and stated. What makes it still worth publishing is that this is
exactly what a practitioner does: execution problems are linear-quadratic in the inventory by
construction (quadratic impact and quadratic risk are the modelling choices), so the structural
knowledge is available in the applications the paper is about. The basis-sensitivity experiment (M2)
bounds what happens as that assumption weakens.

### M7. "No nonlinear-impact, no jumps, no stochastic volatility; the HJB is a Riccati ODE in disguise."

*The referee's version:* the problem is linear-quadratic, so the grid solver is not being tested on the
kind of HJB the field cares about.

*Our answer.* **Partially.** The paper's object is the comparison at matched accuracy with a *trustworthy*
reference, which is why the reference is exact; the non-polynomial extension (M3) is the step taken in
that direction. A nonlinear-impact case (Guéant-type) with a semi-analytic reference is named as the
next step and not claimed.
*Open:* nothing in the paper measures a genuinely nonlinear HJB.

### M8. "The control-accuracy result is folklore: everyone knows a derivative of a numerical solution
loses an order."

*The referee's version:* gradient recovery is a classical subject; that differentiating loses an order
is textbook.

*Our answer.* **Answered** by positioning rather than by novelty claims. The paper cites ENO/WENO,
gradient recovery and post-processing, and the control-error literature in PDE-constrained optimisation,
and states that raising the recovery order is not new. What is specific here: (i) the control, not the
value, is the deliverable of an HJB, and the two errors are shown to be decoupled *by the reconstruction
choice* rather than by the solve; (ii) the probe-step cap, which is a genuine trap for practitioners;
(iii) the consequence in money (below); (iv) the fact that the decoupling can be fixed at zero grid
cost, which changes what a practitioner should do first.

### M9. "What is the economic significance? Relative $L^2$ errors are not money."

*The referee's version:* the accuracy discussion never reaches a quantity an execution desk cares about.

*Our answer.* **Answered** (Table `tab_shortfall`). On the paper's own real-market calibration, across
nine instruments, raising the reconstruction from multilinear to cubic is worth $2.7$–$8.4$ basis points
of notional (mean $5.7$\,bp, about \$5.7k on a \$10M order), with the largest effect on the most liquid
instrument. The cubic arm is within $0.018$\,bp of the exact control, so the figure is a statement about
the multilinear arm.

### M10. "Statistical significance: three seeds, and a $R^2$ of $-57 \pm 46$."

*The referee's version:* the neural results at the dimension where they fail are noise-dominated, and
the paper's conclusion rests on that failure.

*Our answer.* **Partially.** The variance is printed rather than hidden, the mean of a $R^2$ that is
heavy-tailed is explicitly discussed, and the architecture/budget ablation shows the failure is
recoverable — which is why the paper reports the rescued run and frames the result as a boundary rather
than a verdict. The honest summary is that the neural solver's *fixed-budget* accuracy is what is
measured.
*Open:* more seeds would tighten it, and it is the cheapest improvement available in revision.

---

## Minor points

| # | objection | answer |
|---|---|---|
| m1 | Truncated domain: what are the boundary conditions and does the box affect the answer? | The solver pads by 0.25; the query domain is the unpadded box. `exp_boundary_split.py` measures the error density in the outer band against the interior: ratio $0.99$–$1.00$ at $p=1,2,3$, so the domain-wide averages are not boundary statistics. |
| m2 | The CFL restriction makes the explicit scheme look bad; an implicit scheme removes it. | Stated, and measured: an implicit LOD variant exists in the code and is reported as removing the restriction at four times the explicit step. The paper does not generalise its cost claim to implicit schemes, and says so where the cost claim is made. |
| m3 | Real-market calibration but no P&L. | The paper is explicit that it makes no profitability claim, and the execution-shortfall experiment (M9) is a cost comparison, not a strategy backtest. |
| m4 | Why is the terminal sample excluded from some accuracy metrics? | Because the optimal control diverges at $t=T$ for a liquidation problem with a tight spread; the paper documents this and shows the failure mode ($R^2 = -997$ if it is left in). For the *cost* comparison the horizon is kept in full, for the opposite reason — truncation removes the terminal penalty and makes slow trading look cheap. Both are measured (`exp_execution_shortfall.py` sensitivity). |
| m5 | Reproducibility: can a referee check a number? | Every table is generated from `results/`; `verify_paper_numbers.py` asserts 65 prose claims against the result files; `verify_all.py` runs 20 checks on a clean checkout; the generators are byte-reproducible. |
| m6 | Notation and units. | A nomenclature exists in the submission package; all errors are relative $L^2$ norms and every order is fitted to the error rather than to its square, which was a defect in an earlier draft and is now enforced by a check. |
| m7 | Why IEEEtran and not the journal class? | Initial submission does not require `elsarticle`; the conversion is a fifteen-minute mechanical change, written out in the submission package. |
| m8 | The submission package lacks a funding statement and an AI-use disclosure. | Deliberately blank: only the authors can answer these. They are listed as author actions rather than filled with plausible text. |

---

## The three things we would fix first if the paper comes back

1. **A tuned modern high-dimensional baseline.** Replace the plain deep-BSDE reference member with
   deep splitting and/or multilevel Picard, which is the objection we cannot currently answer (M5).
2. **More seeds where the neural solver fails**, and a reported median rather than a mean of a
   heavy-tailed $R^2$ (M10).
3. **A nonlinear-impact example** with a semi-analytic reference, to move past the linear-quadratic
   structure the referee will keep pointing at (M7).

None of the three is a defect in what the paper claims; each is a limit on how far the claims reach,
and the paper states each limit where it binds.
