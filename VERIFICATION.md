# How the code in this repository is verified

Two different questions get confused, and they need different instruments:

1. **Does the code compute the right thing?** — correctness of the reference, of each solver, and of the
   experiments.
2. **Does the paper report what the code computed?** — consistency between the manuscript and
   `results/`.

A repository can pass (2) completely while failing (1) — every number in the prose can match a result
file that is itself wrong — and it can pass (1) while the paper quotes a stale number. This file
describes what covers each, and then what is **not** covered.

```
python src/verify_all.py            # 23 checks, ~5 minutes, no GPU, no market data needed
python src/verify_all.py --full     # adds the PINN at a budget where the paper's accuracy applies
```

---

## Layer 1 — the reference, because everything is measured against it

Nothing else can be trusted if the closed-form reference is wrong, and a reference is the one thing a
comparison paper cannot check against another solver. So it is checked against **three independent
closed forms** (`src/ac_problem.py` runs this itself, and the harness runs it), plus:

* the Riccati terminal condition `|P(T) - Q_T|`, which must be exactly 0 (`verify_all.py`);
* an independent numerical integration of the Riccati ODE with RK4, which reproduces the reference's
  `P(0)` to 6.8e-11 — this is how the sign convention of `dP/dt` was settled rather than argued.

## Layer 2 — each method is validated before it is compared

This is the discipline that matters most, and every method in the paper has its own instrument. The
common shape: check the method against something **known independently**, not against its own previous
iterate.

| method | instrument | what it would catch |
|---|---|---|
| finite differences | value/control R² against the exact solution at N=41 | a wrong stencil, sign, or boundary rule |
| policy iteration | same, at the paper's N=41 | a wrong maximiser or a non-converging sweep |
| PINN | `diag_alpha_pinn.py` evaluates the module's **own residual on the exact solution** and requires ~0; the residual differentiates its value function through autograd, so `dV/dt` is produced by the module | an error in A, B, Q, the control law, the `(∇V)'(Bu)` term, or the per-axis diffusion term |
| deep BSDE | `validate_on_exact()`: the forward recursion run with the **exact** θ and Z must have a vanishing terminal mismatch | a wrong driver, sign, or coupling to the state SDE |
| deep splitting | `validate_on_exact()`: the recursion with the exact solution against the **analytic conditional expectation** (not a single draw — a single draw measures the martingale, which is what the first version got wrong) | a wrong nonlinearity sign, the wrong quadrature endpoint, a wrong generator |
| multilevel Picard | an implementation check at weak nonlinearity (η=1), where the Picard map is a contraction and the method's hypothesis holds | a wrong telescoping weight, or a wrong time-sampling point — both of which still return plausible numbers elsewhere |
| regression Monte Carlo | the same scheme at A₀ = 0 must reproduce the value the paper already reports for the benchmark | a broken basis, damping, or regression |

## Layer 3 — one command over every method: `check_methods_smoke.py`

Layer 2 lives inside the experiment scripts, so a refactor could break a method without the harness
noticing. `check_methods_smoke.py` runs **every method on one small instance** and asserts *the claim
the paper makes about it* — not a uniform accuracy bar, because one uniform bar would fail on correct
code (two methods here have negative results, and two are learned methods whose claim is about a budget
far larger than a smoke test can afford).

```
finite differences   value R² > 0.95, control R² > 0.95, signs agree
policy iteration     value R² > 0.90 at N=41
PINN                 plumbing: runs, finite, |V| bounded   (accuracy is Layer 2 + --thorough)
deep BSDE            terminal residual falls: 1.1e+01 → 7.7e-03
deep splitting       scheme validation < 2e-2 at 100 steps; run completes finite
multilevel Picard    < 5% at η=1, and DIVERGES at η=0.01  (both asserted)
regression MC        value R² > 0.90
```

The last of those is deliberate. Two rows assert a **negative** result, so a future change that made
multilevel Picard converge at the paper's calibration, or made the splitting scheme fail its own
validation, is reported as a change in the paper's findings rather than silently absorbed.

## Layer 4 — independent agreement

* **Two implementations of one scheme**: the NumPy solver and the compiled numba kernel agree on the
  solved field to 1.3e-4 relative, so the timing comparison measures the implementation and not a
  difference in the algorithm.
* **Two algorithms on one problem**: finite differences and policy iteration are independent schemes;
  both are checked against the same reference.
* **Two computation paths for the deep BSDE statistics**: `check_bsde_reconcile.py`,
  `crosscheck_bsde_loops.py` re-derive the same quantities from different code.
* **Artefacts against their generators**: `check_table_drift.py` regenerates tables and compares them to
  the committed files; `make_tables.py` and `make_journal_figures.py` are byte-reproducible, and the
  submission package's PDFs are now byte-reproducible too (dates pinned via `SOURCE_DATE_EPOCH`).

## Layer 5 — the paper reports what the code produced

`verify_paper_numbers.py` asserts **95** claims in the manuscript against `results/`, each with the
source file named:

```
python src/verify_paper_numbers.py     # 95 matched, 0 unmatched, 0 skipped
```

This layer is a *consistency* check. It cannot tell whether a result file is right — only that the
sentence and the file say the same thing. Its value was demonstrated when the error convention changed:
every headline number moved and the assertions caught each stale sentence in turn.

## Layer 6 — checks that catch *classes* of error rather than instances

These exist because each one found something no other check could:

| check | the class it catches | what it actually found |
|---|---|---|
| `check_orphan_results.py` | a measurement that nothing reports | two experiments that had been run and never written up — one of them while the abstract already claimed its result |
| `check_prose_consistency.py` | floats nobody cites, terminology drift, unreadable sentences | three floats (two figures, one table) included and never referenced anywhere |
| `check_order_convention.py` | the same quantity measured on two scales | orders fitted to the squared error in three experiments and to the error in two, so one control error read as "2.00" and "1.03" in different tables |
| `check_release.py` | machine paths, credentials, build intermediates in a public release | an absolute TeX path in a fallback, which this repository's own comment forbids |
| `check_terminal_contamination.py` | a metric dominated by an artefact | value R² of −997 from the terminal sample alone |

## What is NOT verified

Stated because a verification document that only lists successes is not useful:

1. **A shared convention error.** If the reference and a solver are wrong in the same way, no assertion
   here catches it. This is the failure mode the project has hit most often — three times in the deep
   splitting work alone (a factor of two in a quadratic gradient, a factor applied only to diagonal
   terms, a recursion run six times per step). What mitigates it: the reference is cross-checked against
   three closed forms, the validations are written against analytic quantities rather than
   self-consistency, and diagnostics now self-test their gradients against finite differences before
   reporting anything.
2. **The PINN's accuracy at cheap budgets.** Measured on CPU: R² = −4.3 at 1000 epochs, +0.27 at 2000,
   +0.95 at 6000 with a larger collocation set. The default harness therefore asserts plumbing only, and
   `--full` asserts accuracy at 6000 epochs. A regression that halved the PINN's convergence rate would
   pass the default run.
3. **The model, as opposed to the code.** Whether Almgren–Chriss with quadratic impact is the right
   economic model is not a code question, and no check here addresses it.
4. **The market-calibrated experiments, as checked out.** The bars are not distributed (vendor data).
   Their result files are shipped and the calibration is re-checked from
   `data_manifest/preprocess_manifest.json`, but re-running the experiment requires the bars.
5. **GPU nondeterminism.** The neural results were produced on one GPU; the pinned seeds make CPU runs
   reproducible, but bitwise reproducibility of the CUDA kernels is not asserted.
6. **The negative results, beyond their existence.** The matrix asserts that deep splitting's neural arm
   and multilevel Picard at η = 0.01 behave as reported. It does not certify that no better-tuned
   implementation exists; the paper states the budgets and configurations it used for exactly that
   reason.

## Running it all

```powershell
cd <repository root>
$env:TEXLIVE_BIN = "<your TeX Live bin>"     # only if pdflatex is not on PATH
$env:PYTHONPATH   = "$PWD\src"

python src\verify_all.py                 # 23 checks: environment, reference, unit checks, every
                                         # method, prose numbers, orphan results, prose consistency
python src\verify_all.py --full          # adds the PINN at 6000 epochs (~3 min more)
python src\verify_paper_numbers.py       # 95 prose assertions alone
python src\check_methods_smoke.py        # the method claims matrix alone (~34s)
python src\check_release.py              # what must not be published
```

No GPU and no market data are required for any of the above.
