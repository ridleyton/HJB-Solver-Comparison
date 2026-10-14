"""
Post-solve evaluation cost: is the neural solver's disadvantage only at training time?

The cost tables in the paper report *solve* time, which for the grid methods is the backward
time-stepping sweep and for the PINN is the full training run (`pinn.py:272-309`). That is the
right comparison when the equation has to be solved once, and the paper's Discussion says so.

It is not the whole story for deployment, because once a solution exists the operational
quantity is the latency of evaluating the optimal control at a state. This script measures
that separately, so the paper can report both numbers instead of leaving the reader to guess:

  * PINN  : one forward pass plus one autograd gradient (the control is u* = R^-1 B' grad V)
  * grid  : multilinear interpolation of the stored snapshot, plus central differences for
            the gradient

Both are measured for a single state and for a batch of 390 states, which is one trading day
of minute decisions for a single instrument. Measured on GPU and CPU for the PINN, CPU for the
grid method (the grid code is numpy-only).

Weights do not affect the operation count, so the measurement does not need a trained network;
the architecture and batch shapes are what matter.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from ac_problem import ACProblem
from pinn import PINNSolver
from solvers import FiniteDifferenceSolver

BATCH_ONE_DAY = 390          # minutes in a US regular session
N_REPEAT = 200


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def time_pinn(p: int, device: str, batch: int, repeat: int = N_REPEAT) -> float:
    """Seconds per control evaluation, amortised over `repeat` calls of a `batch` of states."""
    prob = ACProblem(p=p)
    net = PINNSolver(prob, hidden=64, layers=4, epochs=1, device=device, seed=0)
    dev = net.device
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(batch, p)).astype(np.float32)
    t = 0.5

    # Warm-up (includes the first CUDA kernel launches and autograd graph setup).
    for _ in range(10):
        net.control(X, t)
    _sync(dev)

    t0 = time.perf_counter()
    for _ in range(repeat):
        net.control(X, t)
    _sync(dev)
    return (time.perf_counter() - t0) / repeat


def time_grid(p: int, n_grid: int, batch: int, repeat: int = N_REPEAT,
              solve: bool = True) -> tuple[float, float]:
    """Seconds per (value, control) evaluation by interpolation on a stored grid.

    With ``solve=False`` the snapshot array is filled with dummy values instead of solving the
    HJB. Interpolation cost depends on the grid shape and the number of query points, not on
    the values, so the timing is unaffected -- and at p=5 a real solve costs 865 s, which would
    dominate this measurement for no benefit.
    """
    prob = ACProblem(p=p)
    s = FiniteDifferenceSolver(prob, n_grid=n_grid)
    if solve:
        s.solve()
    else:
        s._build_grid()
        n_snap = 9
        while n_snap > 2 and n_snap * s.N > 4e7:
            n_snap //= 2
        s.t_snap = np.linspace(0.0, prob.T, n_snap)
        s.snapshots = np.zeros((n_snap,) + s.shape)
        s.n_time_steps = 1
    rng = np.random.default_rng(0)
    X = rng.uniform(-1.0, 1.0, size=(batch, p))
    t = 0.5

    for _ in range(3):
        s.control(X, t)
    t0 = time.perf_counter()
    for _ in range(repeat):
        s.control(X, t)
    ctrl = (time.perf_counter() - t0) / repeat

    for _ in range(3):
        s.value(X, t)
    t0 = time.perf_counter()
    for _ in range(repeat):
        s.value(X, t)
    val = (time.perf_counter() - t0) / repeat
    return val, ctrl


def main() -> None:
    out = Path("results/exp_eval_timing.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    has_cuda = torch.cuda.is_available()
    gpu = torch.cuda.get_device_name(0) if has_cuda else None

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "torch": torch.__version__,
            "gpu": gpu,
            "repeat": N_REPEAT,
            "batch_one_day": BATCH_ONE_DAY,
            "note": ("Amortised latency of evaluating the optimal control after the solution "
                     "exists. Weights are randomly initialised: the operation count, and "
                     "therefore the timing, does not depend on the trained values."),
        },
        "runs": [],
    }

    print("post-solve evaluation latency")
    print("=" * 84)
    print(f"{'method':18s} {'p':>2s} {'batch':>6s} {'device':>5s} "
          f"{'per call (ms)':>14s} {'per state (us)':>15s}")

    for p in (1, 3, 5):
        for batch in (1, BATCH_ONE_DAY):
            # PINN on GPU and CPU
            for device in (["cuda", "cpu"] if has_cuda else ["cpu"]):
                t = time_pinn(p, device, batch)
                rec = {"method": "PINN", "p": p, "batch": batch, "device": device,
                       "seconds_per_call": t, "us_per_state": t / batch * 1e6}
                results["runs"].append(rec)
                print(f"{'PINN':18s} {p:>2d} {batch:>6d} {device:>5s} "
                      f"{t*1e3:>14.4f} {t/batch*1e6:>15.2f}")

    # Grid inference is cheap to measure at every dimension the benchmark covers, so run all
    # of them; only p=1 and p=3 have a usable PINN, so those are the only PINN entries.
    for p in (1, 2, 3, 4, 5):
        n_grid = {1: 21, 2: 21, 3: 21, 4: 21, 5: 21}.get(p, 21)
        # Solve for real only where it is cheap; p>=4 uses dummy snapshots (see time_grid).
        for batch in (1, BATCH_ONE_DAY):
            v, c = time_grid(p, n_grid, batch, solve=(p <= 3))
            rec = {"method": "Finite Difference (interpolation)", "p": p, "batch": batch,
                   "device": "cpu", "n_grid": n_grid,
                   "seconds_per_call_control": c, "us_per_state_control": c / batch * 1e6,
                   "seconds_per_call_value": v, "us_per_state_value": v / batch * 1e6}
            results["runs"].append(rec)
            print(f"{'grid (control)':18s} {p:>2d} {batch:>6d} {'cpu':>5s} "
                  f"{c*1e3:>14.4f} {c/batch*1e6:>15.2f}")

    # ---------------------------------------------------------------------------------
    # Break-even analysis. The right way to combine the two cost components is total cost
    # of ownership:  T(N) = solve_or_train + N * per_evaluation,  for N control
    # evaluations. If the PINN is to justify its training cost, it must win on the
    # per-evaluation term by enough to pay that cost back within a realistic N.
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 96)
    print("Break-even: evaluations needed for the PINN's cheaper inference to repay its "
          "training cost")
    print("=" * 96)
    # (p, grid solve s, PINN train s) -- PINN at the budget needed to be accurate.
    BUDGETS = {
        1: (0.023, 95.1, "PINN 6000 epochs, R2 0.9997"),
        2: (0.056, 128.2, "PINN 6000 epochs, R2 0.9993"),
        3: (0.530, 156.6, "PINN 6000 epochs, R2 0.9918"),
        4: (31.710, 630.0, "PINN 20000 epochs (rescued), R2 0.9972"),
    }
    per_day = BATCH_ONE_DAY * 1.0
    rows = []
    for p, (grid_solve, pinn_train, note) in BUDGETS.items():
        g = next((r for r in results["runs"] if r["method"].startswith("Finite")
                  and r["p"] == p and r["batch"] == BATCH_ONE_DAY), None)
        pn = next((r for r in results["runs"] if r["method"] == "PINN" and r["p"] == p
                   and r["batch"] == BATCH_ONE_DAY
                   and r["device"] == ("cuda" if has_cuda else "cpu")), None)
        if not g or not pn:
            continue
        g_us = g["us_per_state_control"]
        p_us = pn["us_per_state"]
        # T_grid(N) = T_pinn(N)  ->  grid_solve + g*N = train + p*N
        denom = (g_us - p_us) * 1e-6
        if denom <= 0:
            n_star = float("inf")
        else:
            n_star = (pinn_train - grid_solve) / denom
        days = n_star / per_day
        rows.append({"p": p, "grid_solve_s": grid_solve, "pinn_train_s": pinn_train,
                     "grid_us_per_state": g_us, "pinn_us_per_state": p_us,
                     "n_star_evaluations": (None if n_star == float("inf") else n_star),
                     "n_star_days": (None if n_star == float("inf") else days),
                     "note": note})
        if n_star == float("inf"):
            print(f"  p={p}: grid is cheaper BOTH upfront ({grid_solve:.3f}s vs "
                  f"{pinn_train:.1f}s) and per evaluation ({g_us:.2f} vs {p_us:.2f} us) "
                  f"-> the PINN never breaks even")
        else:
            print(f"  p={p}: break-even at {n_star:,.0f} evaluations "
                  f"({days:,.0f} trading days at {per_day:.0f}/day); "
                  f"grid {g_us:.2f} vs PINN {p_us:.2f} us/state; {note}")
    results["breakeven"] = rows

    # ---- ratio summary ---------------------------------------------------------------
    print("\n" + "=" * 84)
    print("Ratio: grid / PINN, single-state control evaluation")
    for p in (1, 3, 5):
        pin = next((r for r in results["runs"]
                    if r["method"] == "PINN" and r["p"] == p and r["batch"] == 1
                    and r["device"] == (["cuda", "cpu"][0] if has_cuda else "cpu")), None)
        grd = next((r for r in results["runs"]
                    if r["method"].startswith("Finite Difference") and r["p"] == p
                    and r["batch"] == 1), None)
        if pin and grd:
            g = grd["seconds_per_call_control"]
            ratio = pin["seconds_per_call"] / g
            print(f"  p={p}: PINN {pin['seconds_per_call']*1e6:8.2f} us  "
                  f"grid {g*1e6:8.2f} us  -> PINN is {ratio:6.1f}x slower per evaluation")
            results.setdefault("summary", {})[f"p{p}_pinn_over_grid"] = ratio

    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
