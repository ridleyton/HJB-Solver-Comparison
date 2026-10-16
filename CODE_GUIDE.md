# How to read this code

Written for someone opening this repository for the first time. It says what each file does, what
order to read them in, and where the interesting parts are. `README.md` explains the *science*;
this file explains the *code*.

---

## 0. First, prove it runs

```powershell
cd <path-to-this-repository>
python src\verify_all.py            # 23 checks, about 5 minutes, CPU only, no GPU, no market data
```

It sets the two environment variables that this machine needs (see §5), checks what is installed,
runs the unit-test modules, re-derives the paper's main numbers from scratch, then checks every
method against the claim the paper makes about it, the 95 prose assertions, the orphan-result audit
and the proofread pass. The headline part looks like this:

```
  closed form: |P(T) - Q_T| = 0.0e+00  (must be 0)
  FDM  p=2 N=21: 0.074s  value R2=+0.999929  control R2=+0.99601
  PI   p=2 N=15: 0.364s  value R2=+0.975005
  extraction on a quartic: multilinear 3.34e-02   cubic 3.75e-04   (relative L2; 89x better)
  [PASS] method claims matrix (every solver)   33.9s
  23 passed, 0 failed, 1 skipped
```

Add `--full` for the PINN at a budget where the paper's accuracy claim applies (6000 epochs, about
3 minutes on CPU). What each check covers, and what none of them covers, is in `VERIFICATION.md`.

If that prints `Everything verified`, the checkout is reproducible and you can trust what follows.
If it fails, fix that first — nothing else will be meaningful.

---

## 1. The mental model

Everything hangs off one small idea: **the benchmark has a closed-form answer, so every solver can
be scored against truth rather than against another solver.**

```
   ac_problem.py        the problem, AND its exact solution (a Riccati ODE)
        |
        +--> solvers.py       the grid and probabilistic solvers, all scored the same way
        |      FiniteDifferenceSolver     explicit time stepping on a tensor grid
        |      PolicyIterationSolver      Howard's algorithm on the same grid
        |      DeepBSDESolver             point-valued: trains V(x0) and Z along paths
        |
        +--> pinn.py          the direct neural solver
        +--> deep_splitting.py     one small network per time step (the modern BSDE variant)
        +--> multilevel_picard.py  the Picard iteration as a multilevel Monte Carlo scheme
        +--> exp_lsmc.py           regression Monte Carlo, the classical mesh-free baseline
        |
        +--> exp_*.py         drivers: sweep a parameter, call the solvers, write results/*.json
                 |
                 +--> make_tables.py / make_figures.py     read the JSON, emit LaTeX/PDF
                          |
                          +--> build_paper.py               compile the manuscript
```

`accuracy_metrics()` is the hinge. Every solver implements the same three-method interface — `solve()`, `value(X, t)`, `control(X, t)` — so the scoring code never knows which method it is
testing. If you understand that interface you understand the whole architecture.

---

## 2. Reading order

Read these seven, in this order. Each one is self-contained and they build on each other. The two
added most recently (⑤ and ⑥) are worth reading even if you skip the methods: both check themselves
against an analytic quantity before they are compared with anything.

### ①`src/ac_problem.py` (487 lines) — start here

The problem *and* the answer key. Read in this order:

| What | Why it matters |
|---|---|
| `ACProblem.__post_init__` | builds `Q`, `R`, `Q_T`, `A`, `B`. Note `hetero=True`: per-asset random coefficients, which is what makes the problem *non-separable* |
| `domain()` | the state box, `[-1,1]^p` |
| `hamiltonian_max()` | the sup over the control, done analytically: `u* = R⁻¹B'∇V`. **This is the only place the control is defined.** |
| `reference()` → `_RiccatiReference` | integrates `-Ṗ = Q + PA + A'P - PBR⁻¹B'P` backwards from `P(T)=Q_T` |
| `_RiccatiReference.P()` | the exact `P(t)`, interpolated from the integration |
| `_RiccatiReference.control()` | `u* = -R⁻¹B'P(t)x` — **the ground truth every solver is scored against** |
| the `__main__` block | four self-checks against three independent closed forms. Run it: `python src/ac_problem.py` |

**Key insight to take away:** the value function is quadratic, `V = -½x'P(t)x + c(t)`. That is why
the "is the value function a polynomial?" question matters for the paper — a cubic reconstruction
reproduces quadratics *exactly*, which is why the Merton experiment exists.

### ②`src/solvers.py` (779 lines) — the grid and probabilistic methods

| What | Why it matters |
|---|---|
| `_GridSolver._build_grid` | the tensor grid. `self.N = n_grid**p` — this is the curse of dimensionality, in one line |
| `_GridSolver._stable_dt` | the CFL bound. Note it has *both* a diffusion and an advection limit; the advection one usually binds here |
| `_GridSolver._sweep` | the backward time loop. `V = V + dt*(H + ½σ²ΔV)` — the "solve long" part |
| `_GridSolver._spatial_interp` | **multilinear** interpolation. The paper's central argument is that this, not the grid, limits control accuracy |
| `_GridSolver.control` | a central difference of that interpolant, with a fixed probe step. **This is the code the paper criticises** |
| `FiniteDifferenceSolver.solve` | a probe pass to estimate `|u|`, then the real pass |
| `PolicyIterationSolver` | Howard's algorithm: freeze the policy, solve the resulting *linear* PDE, improve. Read the docstring on `_sweep`'s `policy_snap` argument — the policy must be a function of `(x,t)`, and getting that wrong is a bug that was actually made here |
| `accuracy_metrics` | **the hinge.** Computes `value_r2`, `control_r2`, `control_sign_agreement`, `value_err`, `control_err` |

### ③`src/exp_control_extraction.py` (356 lines) — the paper's central experiment

Small file, big idea. Read the three extractors and compare them:

- `extract_fd` — central difference on a **multilinear** interpolant (what `solvers.py` does)
- `extract_cell` — the **exact** gradient of that same interpolant. Still second order. This is the
  proof that "being consistent with your interpolant" is not enough
- `extract_spline` — central difference on a **cubic** interpolant. Fourth order

Then read `to_exact_nodes`, which is the trick that makes the experiment decisive: it puts the
*analytic* value function on the grid, so the grid solution has zero error and everything that
remains belongs to the extraction step.

The `SYNTHETIC` dict at the bottom is why the result generalises — those test functions have known
gradients, so the measured orders are properties of the *reconstruction*, not of this benchmark.

### ④`src/pinn.py` (368 lines) — the neural method

| What | Why it matters |
|---|---|
| the ansatz | `V = V_T(x) + (T-t)N(x,t)`, which imposes the terminal condition *exactly* rather than as a penalty |
| the residual | the HJB residual, normalised by its value at initialisation so the loss is `O(1)` |
| collocation sampling | biased towards `t=T` with `(1-u)^time_bias`, where `|∇V|` is largest |
| `solve()` | **times the epoch loop.** This is why the paper's PINN numbers are *training* times, not inference |

### ⑤`src/deep_splitting.py` (473 lines) — one network per time step

The modern member of the deep-BSDE family, and the one place in this repository where a scheme is
validated against an **analytic conditional expectation** rather than against its own previous
iterate: `validate_on_exact()` runs the recursion with the exact solution and reports the deviation,
which is the check that caught two convention errors in the diagnostic that measured it. The same
recursion with an exact quadratic regression is the control arm that shows the limitation is the
learner rather than the scheme.

### ⑥`src/multilevel_picard.py` (351 lines) — the classical high-dimensional method

Equation (5) of Hutzenthaler–Jentzen–Kruse–Nguyen, with the gradient by central differences. Read
`implementation_check()` first: it runs at a weak nonlinearity, where the Picard map is a contraction
and the method's hypothesis holds, and refuses to report anything if it does not converge. The
contraction sweep next to it is what locates the boundary between where the method works and where it
diverges.

### ⑦`src/backtest.py` + `src/preprocess_causal.py` — the real-data half

Only needed if you care about the out-of-domain and defect-ablation results. `BacktestConfig` has
one flag per historical defect; `ablation_configs()` builds the ablation table. `preprocess_causal`
fixes the non-causal preprocessing (sorting, forward-looking VWAP, Corwin–Schultz).

---

## 3. What to read for what

| You want to understand | Read |
|---|---|
| the math of the problem | `ac_problem.py`, docstring + `__main__` |
| how a grid solver works | `solvers.py`: `_sweep`, `_stable_dt` |
| **why control accuracy ≠ value accuracy** | `exp_control_extraction.py` (whole file) |
| how the experiments are structured | any `exp_*.py` — they all follow the same shape |
| how a result becomes a table | `make_tables.py`, one `tab_*` function |
| how the paper is compiled | `build_paper.py` |
| the literature tooling | `extract_lit.py`, `scan_lit_*.py` (independent of the science) |

Skip unless you need them: `fetch_repo.py`, `fetch_files.py`, `bench_cpu_gpu.py`,
`print_lit_heads.py`, `diag_floats.py`, `check_sentence_length.py`. These are utilities written for
one-off tasks and are not part of the scientific pipeline.

---

## 4. Running the experiments

Each `exp_*.py` is standalone and writes `results/<name>.json`. They are all slow; start with the
cheap ones. `KMP_DUPLICATE_LIB_OK` and `MPLCONFIGDIR` are handled by `verify_all.py` but you must
set them yourself for a direct run (§5).

| Command | Time | What it produces |
|---|---|---|
| `python src/exp_grid_refinement.py` | ~2 min | the accuracy-matched grid resolution per dimension |
| `python src/exp_control_extraction.py --p 2 --grids 11,15,21,27,35,45,61 --also-pde --synthetic` | ~1 min | **Table II** — the paper's central evidence |
| `python src/exp_control_order.py --dims 1,2,3` | ~5 min | control vs value convergence orders |
| `python src/exp_dimension.py --dims 1,2,3,4,5 --epochs 6000 --seeds 0,1,2` | ~1 h | **Table I** — accuracy and cost vs dimension |
| `python src/exp_highdim.py --dims 1,2,3,5,8,12,20 --epochs 8000` | ~30 min | the p=20 sweep |
| `python src/exp_merton_extraction.py` | ~1 min | the non-polynomial confirmation |
| `python src/exp_control_floor.py` | ~20 min | time vs space refinement |
| `python src/exp_control_p5.py` | ~60 min | p=5 with the snapshot cap fixed |
| `python src/exp_pi_pinn.py` | ~60 min | the PI-PINN comparison (needs `exploratory_hjb/`) |
| `python src/make_tables.py && python src/make_figures.py` | ~10 s | rebuild every table and figure |
| `python src/build_paper.py --build-dir paper_build_new` | ~20 s | compile the manuscript |

**`--build-dir` exists for a reason.** If a PDF viewer has the output open, Windows locks
`main.pdf` and the build fails. Pass a fresh directory name rather than closing the viewer.

Full regeneration from nothing would take several hours. You do not need it: `results/*.json` is
committed, so `make_tables.py` reproduces the paper's tables in seconds.

---

## 5. Environment variables, and why the script sets them

Two variables are needed on this machine and neither is optional:

```powershell
$env:KMP_DUPLICATE_LIB_OK="TRUE"                        # Anaconda MKL vs torch libomp
$env:MPLCONFIGDIR="$PWD\.mplconfig"
$env:PYTHONPATH="$PWD\src"
```

- **`KMP_DUPLICATE_LIB_OK`** — Anaconda's MKL and PyTorch each ship an OpenMP runtime. Without this
  the process dies with `OMP: Error #15` the moment both are loaded. It is a workaround, not a fix.
- **`MPLCONFIGDIR`** — matplotlib writes a font cache to `~/.matplotlib`, which is not writable
  here, so the first plot raises `PermissionError`.
- **`PYTHONPATH`** — lets the modules import each other by bare name (`from ac_problem import ...`).

`verify_all.py` sets all three itself, which is the main reason to run it first.

---

## 6. Reading it in VS Code, and running it from there

Open the **`rework` folder** (`File → Open Folder`), not the whole thesis directory — otherwise the
file tree is dominated by data and PDFs.

**Already configured.** A local `.vscode/` folder (gitignored, because it carries this machine's paths
— see `.gitignore`) selects the interpreter, sets `TEXLIVE_BIN`/`PYTHONIOENCODING`/`MPLCONFIGDIR` for
every integrated terminal, hides build artefacts, and defines the tasks below. Recreate it on another
machine by pointing `python.defaultInterpreterPath` at that machine's interpreter; nothing else in the
repository depends on it.

**Run anything with `Ctrl+Shift+B`** (`Terminal → Run Task`), which lists:

| Task | What it does |
|---|---|
| `verify: all checks` | `verify_all.py`, the 23-check harness (default task) |
| `verify: method claims matrix` | every solver against the claim the paper makes about it, ~34s |
| `verify: paper numbers` | the 95 prose assertions |
| `build: tables` / `figures` / `manuscript PDF` / `submission package` | regenerate artefacts |

**Debug with `F5`** ("Debug the open script"): open any `exp_*.py`, set a breakpoint on a line, press
`F5`. The configuration passes the same environment variables, keeps `justMyCode` off so you can step
into the solvers, and runs in the integrated terminal so `print` output appears as it is produced.
Every script is `argparse`-driven, so `--help` in the terminal lists its knobs.

**Extensions.** Python, Pylance and debugpy are what make the above work and are the only ones this
repository assumes. Two optional additions:

* **Jupyter** (`ms-toolsai.jupyter`) — *not installed by default*, and without it the `# %%` / Run Cell
  workflow below does not work. With it, you can select a block in any `.py` file (or add an explicit
  `# %%` marker) and press `Shift+Enter` to run it against the live interpreter, which is the least
  intrusive way to poke at the code: no new files, nothing diverges from what is committed.
* **LaTeX Workshop** — navigation and preview for `paper/main.tex`. Note that the repository's own
  `build_paper.py` does more than a default recipe: it stages the build, checks the page count, the
  overfull boxes and the undefined references, and reports them. Prefer the task.

**Three features that will help most when reading:**

1. **Outline view** (`Ctrl+Shift+O`). Every file has a module docstring and one class per concept, so
   the outline is a usable table of contents. Typing `control` after `Ctrl+Shift+O` jumps to a method.

2. **`F12` (Go to Definition) on `accuracy_metrics`, `reference()` or `_sweep`.** The fastest way to
   follow the architecture: start in an `exp_*.py`, jump into the solver, jump into the problem.

3. **`Ctrl+Shift+F` (search in files) for a number.** Every value in the manuscript traces to a
   result file and every result file to a script, so searching for a figure like `0.9987` lands on the
   experiment that produced it. That is the intended way to audit a claim.

**Watch out for:** the files use `# ----` section banners, so folding (`Ctrl+K Ctrl+0`) gives a clean
skeleton of each file.

---

## 7. Where the code is subtle

These are the places where a careful reader will want to slow down. Each has an inline comment
explaining it, and most are bugs that were actually made and caught.

| Location | The subtlety |
|---|---|
| `solvers.py::_sweep` | the frozen policy must be a function of `(x,t)`, interpolated in time. A single time-independent array makes the policy-evaluation PDE inconsistent near `t=T` and it diverges |
| `solvers.py::_sweep` | upwinding direction. Central differencing of a frozen advection term is unconditionally unstable; von Neumann analysis gives `u>0` ⇒ **forward** stencil |
| `solvers.py::solve` | the snapshot memory cap. It silently halves `n_snap` as `N^p` grows, which coarsens *time* resolution and looks like a spatial error. `snap_cap` is a constructor argument for this reason |
| `solvers.py::control` | the probe step `eps`. It must scale with `dx`; a fixed step costs a factor of 40— 4 in control accuracy |
| `ac_problem.py` | `hetero=True` destroys separability. With uniform coefficients `V = Σᵢ v(xᵢ)` and a network barely has to learn anything |
| `preprocess_causal.py::corwin_schultz_rolling` | the two daily squared ranges must be **summed**, not averaged. Averaging makes `α ≠ 0` and the estimator returns 0 for every asset |
| `exp_ood.py` | the rescue path: rescaling the input, then converting the gradient back to price units by the chain rule. Drop the chain-rule factor and you get a plausible but wrong answer |
| `backtest.py` | the Jacobian `a` is computed in closed form. `np.gradient` divides by zero on repeated minute prices |

---

## 8. If something breaks

| Symptom | Cause | Fix |
|---|---|---|
| `OMP: Error #15` | `KMP_DUPLICATE_LIB_OK` unset | set it, or run through `verify_all.py` |
| `PermissionError: ...fontlist-v330.json.matplotlib-lock` | `MPLCONFIGDIR` unset | set it to a path inside the workspace |
| `ModuleNotFoundError: ac_problem` | `PYTHONPATH` unset | add `rework\src`, or run from `rework/` |
| `Cannot write into paper_build/` | a PDF viewer holds `main.pdf` | `--build-dir paper_build2` |
| `Can't find initialization file gs_init.ps` | Ghostscript needs `GS_LIB` | see `README.md`, "Known limitations" |
| a verification check fails | genuine | the failing check names the file; run that module directly to see full output |

---

## 9. Provenance

Every claim in the manuscript traces to a `results/*.json` file, and each artifact in `artifacts/`
records the command, inputs and hashes that produced it. If you want to know where a number came
from, search the paper for it and then grep `results/` — or read `STATUS.md`, which is the running
log of what was measured, what was corrected, and which of the author's own bugs were found by
verification.
