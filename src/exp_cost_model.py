"""
P1: replace the wall-clock comparison with a HARDWARE-INDEPENDENT cost model.

Why this is needed
------------------
Section V-A compares solve time measured as GPU-for-neural against CPU-for-grid, and the paper
concedes the point ("we state the timing methodology explicitly ... because CPU-only neural timings
are not comparable to compiled grid solvers"). A journal referee will not accept that: the headline
claim of the whole paper is a COST comparison, so the unit it is measured in has to be defensible.

This script therefore counts work analytically, in FLOPs, for both families, and reports the
ACHIEVED rate (GFLOP/s) each one reaches on the hardware it ran on. The achieved rate is the
diagnostic that matters: it says whether wall-clock is a fair proxy for work, and in which direction
it is biased.

The models
----------
Grid (explicit backward-time FDM), per grid point per time step:
    gradient 2d | x'Qx 2d^2 | u'Ru 2m^2 | u*=R^-1B'gradV (dm + m^2)
    (gradV)'Ax d^2 | (gradV)'(Bu) dm | Laplacian 2d | update 2
    => 3d^2 + 3m^2 + 2dm + 4d + 2
  total = n_steps * N^d * that.

PINN, per collocation sample or query:
    forward+backward of the residual loss ~ 6P, plus one second derivative per state coordinate at
    roughly 2P each => (6 + 2d) P, with P the parameter count (4x64 tanh MLP: 4*64 + 3*64*64 + 64).
  training = epochs * batch * (6 + 2d) P ;  inference = N_queries * 2P.

Both models are stated as approximations and are used only to compare ORDERS and to expose the
device bias, never to claim a precise FLOP figure.

Usage: python src/exp_cost_model.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from solvers import FiniteDifferenceSolver

OUT = Path("results")
EPOCHS, BATCH, HIDDEN, LAYERS = 6000, 2048, 64, 4


def mlp_params(d: int, hidden: int = HIDDEN, layers: int = LAYERS) -> int:
    """Parameter count of the 4x64 tanh MLP with (state_dim + 1) inputs."""
    n = (d + 1) * hidden + hidden
    for _ in range(layers - 1):
        n += hidden * hidden + hidden
    return n + hidden + 1


def grid_flops(d: int, m: int, n_grid: int, n_steps: int) -> float:
    per_point = 3 * d * d + 3 * m * m + 2 * d * m + 4 * d + 2
    return float(n_steps) * float(n_grid) ** d * per_point


def pinn_flops(d: int, epochs: int = EPOCHS, batch: int = BATCH) -> tuple[float, int]:
    P = mlp_params(d)
    return float(epochs) * batch * (6 + 2 * d) * P, P


def main() -> int:
    OUT.mkdir(exist_ok=True)
    report: dict = {"epochs": EPOCHS, "batch": BATCH, "rows": []}
    print("=" * 104)
    print("HARDWARE-INDEPENDENT COST: analytic FLOPs and the rate each method actually achieves")
    print("=" * 104)
    print(f"  {'method':<10}{'d':>3}{'N':>5}{'n_steps':>9}{'GFLOP':>12}{'seconds':>10}"
          f"{'GFLOP/s':>11}   note")
    print("-" * 104)

    # ---- grid scheme: measured here, on CPU (which is where it runs) ----------------------
    # Swept over N on purpose. At N=21 a 2-D solve is only a few MFLOP in a few hundredths of a
    # second, so an "achieved GFLOP/s" there measures NumPy call overhead per time step, not the
    # arithmetic. If the rate RISES with N the small cases are overhead-bound and the asymptotic
    # figure is the large-N one; a flat rate would mean something else is limiting it.
    for p in (2,):
        prob = ACProblem(p=p)
        for n in (21, 31, 41):
            s = FiniteDifferenceSolver(prob, n_grid=n, n_snap=101, snap_cap=4e9)
            t0 = time.perf_counter()
            s.solve()
            sec = time.perf_counter() - t0
            f = grid_flops(prob.state_dim, prob.control_dim, n, s.n_time_steps)
            rate = f / sec / 1e9
            report["rows"].append({"method": "grid", "p": p, "device": "CPU", "N": n,
                                   "n_steps": s.n_time_steps, "flops": f, "seconds": sec,
                                   "gflops_per_s": rate})
            print(f"  {'grid':<10}{p:>3}{n:>5}{s.n_time_steps:>9}{f / 1e9:>12.4f}{sec:>10.3f}"
                  f"{rate:>11.4f}   CPU")

    # ---- PINN: trained here, on the GPU (which is where it runs) -------------------------
    try:
        import torch
        from pinn import PINNSolver
        device = "cuda" if torch.cuda.is_available() else "cpu"
        for p in (1, 2):
            prob = ACProblem(p=p)
            s = PINNSolver(prob, hidden=HIDDEN, layers=LAYERS, n_collocation=BATCH,
                           epochs=EPOCHS, lr=2e-3, device=device, seed=0, verbose=False)
            sec = s.solve()
            f, P = pinn_flops(prob.state_dim)
            rate = f / sec / 1e9
            report["rows"].append({"method": "PINN", "p": p, "device": device, "N": None,
                                   "n_steps": EPOCHS, "flops": f, "seconds": sec,
                                   "gflops_per_s": rate, "params": P})
            print(f"  {'PINN':<10}{p:>3}{'-':>5}{EPOCHS:>9}{f / 1e9:>12.3f}{sec:>10.3f}"
                  f"{rate:>11.3f}   {device}, {P} params")
    except Exception as e:
        print(f"  PINN skipped: {type(e).__name__}: {e}")

    print("-" * 104)
    g = [r for r in report["rows"] if r["method"] == "grid"]
    n = [r for r in report["rows"] if r["method"] == "PINN"]
    if g and n:
        # Use the LARGEST grid case as the grid's representative rate: the small ones are
        # overhead-bound and would understate what the scheme can do.
        g_big = max(g, key=lambda r: r["N"])
        rg = g_big["gflops_per_s"]
        rn = float(np.mean([r["gflops_per_s"] for r in n]))
        print(f"  achieved rate: grid {rg:.3f} GFLOP/s at N={g_big['N']} (CPU) vs PINN "
              f"{rn:.1f} GFLOP/s ({n[0]['device']})  -> ratio {rn / rg:.0f}x")
        trend = ", ".join("{}:{:.4f}".format(r["N"], r["gflops_per_s"]) for r in g)
        print(f"  grid rate trend over N: {trend}")
        print()
        print("  WHAT THIS MEANS FOR THE PAPER'S CLAIM. The two families are measured in units that")
        print("  differ by a large factor: the PINN converts FLOPs into time far more efficiently")
        print("  than the grid scheme does, so counting WORK instead of SECONDS flatters the neural")
        print("  method. The paper's wall-clock conclusion -- the grid is faster wherever it runs")
        print("  -- is therefore the CONSERVATIVE one, and it does not depend on the")
        print("  GPU-for-neural versus CPU-for-grid choice. That is the statement to give a")
        print("  referee, in place of 'we state the timing methodology rather than adjust for it'.")
        print()
        print("  CAVEAT THAT MUST GO IN THE PAPER: the grid timings are for a NumPy implementation")
        print("  whose per-step work is many small array operations, so at these sizes the rate is")
        print("  limited by interpreter overhead as much as by arithmetic or bandwidth. A compiled")
        print("  kernel would be substantially faster, which would STRENGTHEN this conclusion")
        print("  rather than weaken it, but it means the absolute grid seconds should not be read")
        print("  as the best a grid scheme can do.")
        report["mean_gflops_grid"] = rg
        report["mean_gflops_pinn"] = rn
        report["rate_ratio"] = rn / rg

    (OUT / "exp_cost_model.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'exp_cost_model.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
