# How to read this code

Written for someone opening this repository for the first time. It says what each file does, what
order to read them in, and where the interesting parts are. `README.md` explains the *science*;
this file explains the *code*.

---

## 0. First, prove it runs

```powershell
cd <path-to-this-repository>
python src\verify_all.py            # ~5 seconds, CPU only
```

It sets the two environment variables that this machine needs (see 搂5), checks what is installed,
runs the three unit-test modules, then re-derives the paper's main numbers from scratch and prints
them:

```
  closed form: |P(T) - Q_T| = 0.0e+00  (must be 0)
  FDM  p=2 N=21: 0.178s  value R2=+0.999929  control R2=+0.99601
  PI   p=2 N=15: 0.800s  value R2=+0.975005
  extraction on a quartic: multilinear 1.11e-03   cubic 1.41e-07   (7926x better)
  15 passed, 0 failed, 1 skipped
```

Add `--full` to include a short PINN training run (needs the GPU, ~10 s).

If that prints `Everything verified`, the checkout is reproducible and you can trust what follows.
If it fails, fix that first 鈥?nothing else will be meaningful.

---

## 1. The mental model

Everything hangs off one small idea: **the benchmark has a closed-form answer, so every solver can
be scored against truth rather than against another solver.**

```
   ac_problem.py        the problem, AND its exact solution (a Riccati ODE)
        |
        +--> solvers.py       three classical solvers, all scored by accuracy_metrics()
        |      FiniteDifferenceSolver     explicit time stepping on a tensor grid
        |      PolicyIterationSolver      Howard's algorithm on the same grid
        |
        +--> pinn.py          one neural solver, same scoring
        |
        +--> exp_*.py         drivers: sweep a parameter, call the solvers, write results/*.json
                 |
                 +--> make_tables.py / make_figures.py     read the JSON, emit LaTeX/PDF
                          |
                          +--> build_paper.py               compile the manuscript
```

`accuracy_metrics()` is the hinge. Every solver implements the same three-method interface 鈥?`solve()`, `value(X, t)`, `control(X, t)` 鈥?so the scoring code never knows which method it is
testing. If you understand that interface you understand the whole architecture.

---

## 2. Reading order

Read these five, in this order. Each one is self-contained and they build on each other.

### 鈶?`src/ac_problem.py` (403 lines) 鈥?start here

The problem *and* the answer key. Read in this order:

| What | Why it matters |
|---|---|
| `ACProblem.__post_init__` | builds `Q`, `R`, `Q_T`, `A`, `B`. Note `hetero=True`: per-asset random coefficients, which is what makes the problem *non-separable* |
| `domain()` | the state box, `[-1,1]^p` |
| `hamiltonian_max()` | the sup over the control, done analytically: `u* = R鈦宦笲'鈭嘨`. **This is the only place the control is defined.** |
| `reference()` 鈫?`_RiccatiReference` | integrates `-峁?= Q + PA + A'P - PBR鈦宦笲'P` backwards from `P(T)=Q_T` |
| `_RiccatiReference.P()` | the exact `P(t)`, interpolated from the integration |
| `_RiccatiReference.control()` | `u* = -R鈦宦笲'P(t)x` 鈥?**the ground truth every solver is scored against** |
| the `__main__` block | four self-checks against three independent closed forms. Run it: `python src/ac_problem.py` |

**Key insight to take away:** the value function is quadratic, `V = -陆x'P(t)x + c(t)`. That is why
the "is the value function a polynomial?" question matters for the paper 鈥?a cubic reconstruction
reproduces quadratics *exactly*, which is why the Merton experiment exists.

### 鈶?`src/solvers.py` (445 lines) 鈥?the classical methods

| What | Why it matters |
|---|---|
| `_GridSolver._build_grid` | the tensor grid. `self.N = n_grid**p` 鈥?this is the curse of dimensionality, in one line |
| `_GridSolver._stable_dt` | the CFL bound. Note it has *both* a diffusion and an advection limit; the advection one usually binds here |
| `_GridSolver._sweep` | the backward time loop. `V = V + dt*(H + 陆蟽虏螖V)` 鈥?the "solve long" part |
| `_GridSolver._spatial_interp` | **multilinear** interpolation. The paper's central argument is that this, not the grid, limits control accuracy |
| `_GridSolver.control` | a central difference of that interpolant, with a fixed probe step. **This is the code the paper criticises** |
| `FiniteDifferenceSolver.solve` | a probe pass to estimate `|u|`, then the real pass |
| `PolicyIterationSolver` | Howard's algorithm: freeze the policy, solve the resulting *linear* PDE, improve. Read the docstring on `_sweep`'s `policy_snap` argument 鈥?the policy must be a function of `(x,t)`, and getting that wrong is a bug that was actually made here |
| `accuracy_metrics` | **the hinge.** Computes `value_r2`, `control_r2`, `control_sign_agreement`, `value_err`, `control_err` |

### 鈶?`src/exp_control_extraction.py` (356 lines) 鈥?the paper's central experiment

Small file, big idea. Read the three extractors and compare them:

- `extract_fd` 鈥?central difference on a **multilinear** interpolant (what `solvers.py` does)
- `extract_cell` 鈥?the **exact** gradient of that same interpolant. Still second order. This is the
  proof that "being consistent with your interpolant" is not enough
- `extract_spline` 鈥?central difference on a **cubic** interpolant. Fourth order

Then read `to_exact_nodes`, which is the trick that makes the experiment decisive: it puts the
*analytic* value function on the grid, so the grid solution has zero error and everything that
remains belongs to the extraction step.

The `SYNTHETIC` dict at the bottom is why the result generalises 鈥?those test functions have known
gradients, so the measured orders are properties of the *reconstruction*, not of this benchmark.

### 鈶?`src/pinn.py` (351 lines) 鈥?the neural method

| What | Why it matters |
|---|---|
| the ansatz | `V = V_T(x) + (T-t)N(x,t)`, which imposes the terminal condition *exactly* rather than as a penalty |
| the residual | the HJB residual, normalised by its value at initialisation so the loss is `O(1)` |
| collocation sampling | biased towards `t=T` with `(1-u)^time_bias`, where `|鈭嘨|` is largest |
| `solve()` | **times the epoch loop.** This is why the paper's PINN numbers are *training* times, not inference |

### 鈶?`src/backtest.py` + `src/preprocess_causal.py` 鈥?the real-data half

Only needed if you care about the out-of-domain and defect-ablation results. `BacktestConfig` has
one flag per historical defect; `ablation_configs()` builds the ablation table. `preprocess_causal`
fixes the non-causal preprocessing (sorting, forward-looking VWAP, Corwin鈥揝chultz).

---

## 3. What to read for what

| You want to understand | Read |
|---|---|
| the math of the problem | `ac_problem.py`, docstring + `__main__` |
| how a grid solver works | `solvers.py`: `_sweep`, `_stable_dt` |
| **why control accuracy 鈮?value accuracy** | `exp_control_extraction.py` (whole file) |
| how the experiments are structured | any `exp_*.py` 鈥?they all follow the same shape |
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
set them yourself for a direct run (搂5).

| Command | Time | What it produces |
|---|---|---|
| `python src/exp_grid_refinement.py` | ~2 min | the accuracy-matched grid resolution per dimension |
| `python src/exp_control_extraction.py --p 2 --grids 11,15,21,27,35,45,61 --also-pde --synthetic` | ~1 min | **Table II** 鈥?the paper's central evidence |
| `python src/exp_control_order.py --dims 1,2,3` | ~5 min | control vs value convergence orders |
| `python src/exp_dimension.py --dims 1,2,3,4,5 --epochs 6000 --seeds 0,1,2` | ~1 h | **Table I** 鈥?accuracy and cost vs dimension |
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

- **`KMP_DUPLICATE_LIB_OK`** 鈥?Anaconda's MKL and PyTorch each ship an OpenMP runtime. Without this
  the process dies with `OMP: Error #15` the moment both are loaded. It is a workaround, not a fix.
- **`MPLCONFIGDIR`** 鈥?matplotlib writes a font cache to `~/.matplotlib`, which is not writable
  here, so the first plot raises `PermissionError`.
- **`PYTHONPATH`** 鈥?lets the modules import each other by bare name (`from ac_problem import ...`).

`verify_all.py` sets all three itself, which is the main reason to run it first.

---

## 6. Reading it in VS Code

Open the **`rework` folder** (`File 鈫?Open Folder`), not the whole thesis directory 鈥?otherwise the
file tree is dominated by data and PDFs.

**Extensions worth installing:** Python (Microsoft), Pylance, and *Better Comments* (colour-codes
`# NOTE:` / `# BUG:` / `# TODO:` markers, which this code uses).

**The three features that will help most:**

1. **Outline view** (`Ctrl+Shift+O`, or the Outline panel in the Explorer sidebar). Every file here
   has a module docstring and one class per concept, so the outline is a usable table of contents.
   `Ctrl+Shift+O` then typing `control` jumps straight to a method.

2. **`F12` (Go to Definition) on `accuracy_metrics`, `reference()`, or `_sweep`.** This is the
   fastest way to follow the architecture: start in an `exp_*.py`, jump into the solver, jump into
   the problem.

3. **Run Cell / the integrated terminal.** Select a block and `Shift+Enter` in a `.py` file runs it
   in the interactive window 鈥?useful for poking at `ACProblem` without writing a script:
   ```python
   import sys; sys.path.insert(0, "src")
   from ac_problem import ACProblem
   prob = ACProblem(p=2, hetero=True, het_seed=1)
   prob.Q, prob.domain()          # inspect the matrices
   ```

**Watch out for:** the files use `# ----` section banners, so folding (`Ctrl+K Ctrl+0` to collapse
all) gives you a clean skeleton of each file.

**Not needed:** a debugger, though it works. The code is deterministic and vectorised; printing
shapes is usually faster than stepping.

---

## 7. Where the code is subtle

These are the places where a careful reader will want to slow down. Each has an inline comment
explaining it, and most are bugs that were actually made and caught.

| Location | The subtlety |
|---|---|
| `solvers.py::_sweep` | the frozen policy must be a function of `(x,t)`, interpolated in time. A single time-independent array makes the policy-evaluation PDE inconsistent near `t=T` and it diverges |
| `solvers.py::_sweep` | upwinding direction. Central differencing of a frozen advection term is unconditionally unstable; von Neumann analysis gives `u>0` 鈬?**forward** stencil |
| `solvers.py::solve` | the snapshot memory cap. It silently halves `n_snap` as `N^p` grows, which coarsens *time* resolution and looks like a spatial error. `snap_cap` is a constructor argument for this reason |
| `solvers.py::control` | the probe step `eps`. It must scale with `dx`; a fixed step costs a factor of 40鈥?4 in control accuracy |
| `ac_problem.py` | `hetero=True` destroys separability. With uniform coefficients `V = 危岬(x岬?` and a network barely has to learn anything |
| `preprocess_causal.py::corwin_schultz_rolling` | the two daily squared ranges must be **summed**, not averaged. Averaging makes `伪 鈮?0` and the estimator returns 0 for every asset |
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
from, search the paper for it and then grep `results/` 鈥?or read `STATUS.md`, which is the running
log of what was measured, what was corrected, and which of the author's own bugs were found by
verification.
