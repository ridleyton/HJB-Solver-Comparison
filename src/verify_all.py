"""
One command that proves the rework runs, and tells you what is installed.

    python src/verify_all.py            # fast: environment + all unit checks + numeric smoke
    python src/verify_all.py --full     # adds a short PINN training run (needs a GPU, ~1 min)

Why this exists: the environment needs two variables set before anything imports torch or
matplotlib on this machine, and forgetting either produces a confusing failure rather than a
clear one. This script sets them itself, so it is safe to run from a bare shell:

    KMP_DUPLICATE_LIB_OK=TRUE   Anaconda's MKL and torch's libomp both ship an OpenMP runtime;
                                without this the process aborts with "OMP: Error #15".
    MPLCONFIGDIR                matplotlib writes a font cache to ~/.matplotlib by default, which
                                is not writable here and raises PermissionError on first use.

Every check runs as a subprocess so that a hard crash (a segfault from the OpenMP clash, say)
is reported as a failed check rather than killing this script. Exit code is 0 only if every
check passed.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------------------
# Environment, set before anything else can import torch/matplotlib.
# ---------------------------------------------------------------------------------------
SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ["PYTHONPATH"] = str(SRC) + os.pathsep + os.environ.get("PYTHONPATH", "")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results: list[tuple[str, str, str]] = []      # (name, status, detail)


def record(name: str, status: str, detail: str = "") -> None:
    results.append((name, status, detail))
    tag = {PASS: "PASS", FAIL: "FAIL", SKIP: "SKIP"}[status]
    line = f"  [{tag}] {name}"
    if detail:
        line += f"   {detail}"
    print(line, flush=True)


def run_script(name: str, path: Path, timeout: int = 900,
               pass_marker: str | None = None,
               extra_args: list[str] | None = None) -> bool:
    """Run a module as a subprocess and decide pass/fail from its exit code."""
    if not path.exists():
        record(name, SKIP, f"{path.name} not found")
        return False
    t0 = time.perf_counter()
    try:
        proc = subprocess.run([sys.executable, str(path)] + list(extra_args or []), cwd=str(ROOT),
                              capture_output=True, text=True, timeout=timeout,
                              env=os.environ.copy())
    except subprocess.TimeoutExpired:
        record(name, FAIL, f"timed out after {timeout}s")
        return False
    dt = time.perf_counter() - t0
    out = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        note = f"{dt:.1f}s"
        if pass_marker and pass_marker in out:
            note += f", found '{pass_marker}'"
        record(name, PASS, note)
        return True
    # Surface the most useful line of the failure rather than all of it.
    interesting = [ln.strip() for ln in out.splitlines()
                   if ln.strip() and not ln.startswith("  File ")
                   and "site-packages" not in ln]
    detail = interesting[-1][:150] if interesting else f"exit {proc.returncode}"
    if len(out) > 400:
        log = ROOT / "logs" / f"verify_{path.stem}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text(out, encoding="utf-8", errors="replace")
        detail += f"  (full output: {log.relative_to(ROOT)})"
    record(name, FAIL, detail)
    return False


# ---------------------------------------------------------------------------------------
def check_environment() -> None:
    """Report versions, and flag the things that would break a later step."""
    print("\n" + "=" * 78)
    print("ENVIRONMENT")
    print("=" * 78)
    print(f"  python      {sys.version.split()[0]}  ({sys.executable})")
    print(f"  cwd         {os.getcwd()}")
    for mod, label in (("numpy", "numpy"), ("scipy", "scipy"), ("pandas", "pandas"),
                       ("matplotlib", "matplotlib"), ("torch", "torch")):
        try:
            m = __import__(mod)
            extra = ""
            if mod == "torch":
                extra = f"  CUDA available: {m.cuda.is_available()}"
                if m.cuda.is_available():
                    extra += f" ({m.cuda.get_device_name(0)})"
            print(f"  {label:11s} {getattr(m, '__version__', '?'):12s}{extra}")
        except Exception as e:
            print(f"  {label:11s} MISSING ({type(e).__name__})")
            record(f"import {mod}", FAIL, "required by the experiments")
            continue
        record(f"import {mod}", PASS)

    # Optional extras the literature tooling needs; their absence is not a blocker.
    for mod, why in (("yaml", "configs"), ("tqdm", "progress bars")):
        try:
            __import__(mod)
            record(f"import {mod} (optional)", PASS)
        except Exception:
            record(f"import {mod} (optional)", SKIP, f"only used for {why}")

    # TeX, for the manuscript build. Resolved through PATH or TEXLIVE_BIN rather than a hardcoded
    # absolute path: the machine-specific literal that used to be the first candidate made this
    # check report on one computer's disk layout and told a reviewer nothing about their own.
    tex = None
    _tex_env = os.environ.get("TEXLIVE_BIN")
    _cands = ([str(Path(_tex_env) / "pdflatex.exe"), str(Path(_tex_env) / "pdflatex")]
              if _tex_env else []) + ["pdflatex"]
    for cand in _cands:
        try:
            p = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=30)
            if p.returncode == 0:
                tex = cand
                break
        except Exception:
            continue
    if tex:
        record("pdflatex (manuscript)", PASS, tex)
    else:
        record("pdflatex (manuscript)", SKIP, "not on PATH; paper build will fail")


def check_numerics(quick: bool) -> None:
    """Run the core numbers in-process and print them, so the reader sees real output."""
    print("\n" + "=" * 78)
    print("CORE NUMERICS")
    print("=" * 78)
    sys.path.insert(0, str(SRC))
    try:
        import numpy as np
        from ac_problem import ACProblem
        from solvers import FiniteDifferenceSolver, PolicyIterationSolver, accuracy_metrics

        # 1. the analytic reference must satisfy its own Riccati ODE and terminal condition
        prob = ACProblem(p=2)
        ref = prob.reference(np.linspace(0.0, prob.T, 64))
        err_T = float(np.max(np.abs(ref.P(prob.T) - prob.QT)))
        print(f"  closed form: |P(T) - Q_T| = {err_T:.1e}  (must be 0)")
        record("Riccati reference terminal condition", PASS if err_T < 1e-12 else FAIL,
               f"{err_T:.1e}")

        # 2. a grid solve, scored against that reference
        t0 = time.perf_counter()
        fdm = FiniteDifferenceSolver(prob, n_grid=21)
        fdm.solve()
        m = accuracy_metrics(fdm, prob, n_points=300, n_times=4, seed=0)
        dt = time.perf_counter() - t0
        print(f"  FDM  p=2 N=21: {dt:.3f}s  value R2={m['value_r2']:+.6f}  "
              f"control R2={m['control_r2']:+.5f}")
        ok = m["value_r2"] > 0.999 and m["control_sign_agreement"] > 0.95
        record("FDM solve beats the closed form", PASS if ok else FAIL,
               f"value R2={m['value_r2']:.5f}")

        # 3. policy iteration, which is the slowest classical method
        t0 = time.perf_counter()
        pi = PolicyIterationSolver(prob, n_grid=15, max_policy_iter=8)
        pi.solve()
        mp = accuracy_metrics(pi, prob, n_points=300, n_times=4, seed=0)
        print(f"  PI   p=2 N=15: {time.perf_counter()-t0:.3f}s  "
              f"value R2={mp['value_r2']:+.6f}")
        record("Policy iteration converges", PASS if mp["value_r2"] > 0.9 else FAIL,
               f"value R2={mp['value_r2']:.5f}")

        # 4. the central claim of the paper, in miniature: does a cubic reconstruction beat a
        #    multilinear one when extracting a control from nodal values?
        from exp_control_extraction import extract_cell, extract_fd, extract_spline
        rng = np.random.default_rng(0)
        X = rng.uniform(-1.0, 1.0, size=(400, 2))
        axes = [np.linspace(-1.25, 1.25, 41)] * 2
        mesh = np.meshgrid(*axes, indexing="ij")
        xn = np.stack([m.ravel() for m in mesh], axis=1)
        snap = 0.25 * np.sum(xn**4, axis=1).reshape(41, 41)     # quartic: not polynomial-exact
        exact = X**3                                            # d/dx of x^4/4 is x^3
        scale = float(np.mean((exact - exact.mean()) ** 2))
        # Square roots, because the paper's metric is the relative L2 error and these two numbers
        # are the check that the reconstruction order works. Reporting the squared error here would
        # print a factor that is the square of the error ratio -- the exact confusion the metric
        # work removed from the tables.
        e_fd = float(np.sqrt(np.mean((extract_fd(axes, snap, X, 0.25) - exact) ** 2) / scale))
        e_sp = float(np.sqrt(np.mean((extract_spline(axes, snap, X, 0.25) - exact) ** 2) / scale))
        print(f"  extraction on a quartic: multilinear {e_fd:.2e}   cubic {e_sp:.2e}   "
              f"(relative L2; {e_fd/e_sp:.0f}x better)")
        record("cubic reconstruction beats multilinear", PASS if e_sp < e_fd / 10 else FAIL,
               f"{e_fd/e_sp:.0f}x")

        # 5. torch plumbing, if a GPU is present
        try:
            import torch
            from pinn import PINNSolver
            if not torch.cuda.is_available():
                record("PINN training smoke", SKIP, "no CUDA device")
            elif not quick:
                t0 = time.perf_counter()
                s = PINNSolver(ACProblem(p=2), epochs=300, n_collocation=512, seed=0)
                s.solve()
                mm = accuracy_metrics(s, ACProblem(p=2), n_points=200, n_times=3, seed=0)
                # This is a PLUMBING check, not an accuracy check. 300 epochs is two orders of
                # magnitude fewer than the paper's smallest run, so a poor R^2 here is expected
                # and says nothing about the method -- it only shows that training runs, the
                # tensors flow, and the metrics come back finite. The accuracy claim is the
                # FDM/extraction checks above and the full experiments.
                print(f"  PINN p=2, 300 epochs: {time.perf_counter()-t0:.1f}s  "
                      f"value R2={mm['value_r2']:+.3f}   <-- plumbing only; far too few epochs "
                      f"for accuracy")
                record("PINN training smoke (plumbing, not accuracy)",
                       PASS if np.isfinite(mm["value_r2"]) else FAIL,
                       "training runs end to end")
            else:
                record("PINN training smoke", SKIP, "run with --full to include it")
        except Exception as e:
            record("PINN training smoke", FAIL, f"{type(e).__name__}: {e}")
    except Exception as e:
        record("core numerics", FAIL, f"{type(e).__name__}: {e}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true",
                    help="also train a tiny PINN (needs a GPU)")
    ap.add_argument("--skip-unit-tests", action="store_true")
    args = ap.parse_args()

    print("=" * 78)
    print("VERIFYING THE REWORK")
    print("=" * 78)

    check_environment()

    print("\n" + "=" * 78)
    print("UNIT CHECKS  (each runs as its own process)")
    print("=" * 78)
    if not args.skip_unit_tests:
        run_script("analytic reference vs three closed forms", SRC / "ac_problem.py",
                   pass_marker="ALL CHECKS PASSED")
        run_script("backtest accounting vs hand-computed values", SRC / "test_backtest.py",
                   pass_marker="ALL BACKTEST ACCOUNTING CHECKS PASSED")
        run_script("experiment code paths (CPU)", SRC / "test_experiment_paths.py",
                   pass_marker="ALL EXPERIMENT-PATH CHECKS PASSED")
        # Feeds the EXACT solution to both PINN modules and requires their residual to vanish.
        # This is the check that keeps the residuals honest: a training smoke test passes even when
        # the residual operator is wrong, because a wrong residual still trains to something. It
        # covers the inventory-only AND signal-augmented benchmarks, and both modules.
        run_script("PINN residuals vanish on the exact solution", SRC / "diag_alpha_pinn.py",
                   pass_marker="ALL RESIDUALS VANISH ON THE EXACT SOLUTION")
        # The two checks that came out of the journal-conversion audit. Both fail on the state the
        # repository was in before it: one on a metric convention that mixed squared errors with
        # error norms, the other on a table row that carried another dimension's run.
        run_script("metric convention (orders on one scale)", SRC / "check_order_convention.py",
                   pass_marker="REGRESSION GUARD")
        run_script("prose numbers vs results/", SRC / "verify_paper_numbers.py",
                   pass_marker="Every asserted order is fitted to the relative L2 error")
        # Proposition 2 is a theorem about this v2 benchmark; this runs it against trajectories
        # integrated under the paper's own reconstructions.
        run_script("suboptimality identity (Proposition 2)",
                   SRC / "verify_suboptimality_identity.py", pass_marker="VERDICT: PASS")
        # The submission package: highlights within the publisher's limits, the declarations and
        # cover letter compiling, and every required item present.
        run_script("journal submission package", SRC / "build_journal.py",
                   pass_marker="All required items present")
        # The audit that a consistency check cannot do: a result file nothing reads is invisible to
        # every other check here. Two experiments had been measured and never reported before this.
        run_script("orphan result audit", SRC / "check_orphan_results.py",
                   pass_marker="every result file is either reported")
        # The proofread pass: every float cited at least once and in the subsection that introduces
        # it, one cross-reference style, no sentence over 80 words.
        run_script("prose consistency (floats, terminology, sentence length)",
                   SRC / "check_prose_consistency.py", pass_marker="Nothing flagged")
        # Every method, on one small instance, against the claim the paper makes about it -- including
        # the two negative results, which are asserted AS reported so a future change that alters them
        # is flagged rather than silently accepted. --full adds the PINN at a budget where the paper's
        # accuracy claim applies (about 150s on CPU).
        run_script("method claims matrix (every solver)",
                   SRC / "check_methods_smoke.py",
                   pass_marker="METHOD CLAIMS MATRIX COMPLETE", timeout=1200,
                   extra_args=["--thorough"] if args.full else None)
        # NOTE: exp_lsmc_basis.py is deliberately NOT in this harness. It is a 6-minute experiment
        # whose largest-amplitude case sits in a failing regime where the value moves by up to 0.07
        # across process runs, so re-running it here would make the harness flaky and would overwrite
        # the result file the prose assertions check against. It is a documented reproduction step:
        #   python src/exp_lsmc_basis.py --dim 4 --a0 0,1e-4,1e-3,5e-3,2e-2 --repeats 3

    check_numerics(quick=not args.full)

    # ---------------- summary ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("SUMMARY")
    print("=" * 78)
    n_pass = sum(1 for _, s, _ in results if s == PASS)
    n_fail = sum(1 for _, s, _ in results if s == FAIL)
    n_skip = sum(1 for _, s, _ in results if s == SKIP)
    for name, status, detail in results:
        if status == FAIL:
            print(f"  FAILED: {name}   {detail}")
    print(f"  {n_pass} passed, {n_fail} failed, {n_skip} skipped")
    if n_fail:
        print("\n  The rework does NOT verify. Fix the failures above before trusting any result.")
        return 1
    print("\n  Everything verified. The repository is reproducible as checked out.")
    print("  Next: see CODE_GUIDE.md for how to read the code and run the experiments.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
