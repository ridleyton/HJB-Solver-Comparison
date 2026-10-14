"""
PROBE (cheap, decides the budget): what floor does the EXACT solution itself hit on the deep-BSDE
Euler recursion, as a function of n_steps?

Why this probe exists, and why it is the right first move
---------------------------------------------------------
The open question from the last session is whether the trained terminal-loss floor (about twice the
exact solution's) is TIME DISCRETISATION or the NETWORK. Nothing about training is needed to answer
half of it: if the exact solution is pushed through the SAME Euler recursion, the terminal mismatch
it leaves behind is by definition the discretisation floor -- no network, no optimiser, no epochs.

That number is a hard LOWER BOUND on what any network could achieve inside this recursion. So the
verdict has three parts, and this probe supplies the first two cheaply:

  1. floor_exact(n_steps)   -- what the scheme can do at all
  2. floor_exact trend       -- whether discretisation is even a live lever at these dt
  3. floor_trained(n_steps)  -- what the network actually reaches (the expensive part)

If floor_exact is already flat near 1e-3 while dt falls 8x, then refining time cannot close a 2x
gap and the answer is "the network is the limit". If it falls like dt (or dt^2), then part of the
gap is resolution and the sweep is worth its runtime.

This probe also measures the per-epoch cost, which is what determines the epoch budget of the
training sweep (cost per epoch is O(n_steps), so matching total work means epochs ~ 1/n_steps).

Usage: python src/probe_bsde_floor.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

# Must precede the torch import: Anaconda's MKL and torch each ship an OpenMP runtime and the
# second load aborts the process (OMP Error #15). Same workaround verify_all.py uses.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem          # noqa: E402
from deep_bsde import DeepBSDE            # noqa: E402

OUT = Path("results")


def exact_floor(prob: ACProblem, x0: np.ndarray, n_steps: int, n_paths: int, seed: int = 0,
                device: str = "cuda"):
    """Terminal mismatch left by the EXACT solution inside the Euler recursion.

    Self-consistency matters here: the Riccati reference is integrated on the SAME (n_steps+1)-point
    grid the recursion uses, so this measures the discretisation of the recursion alone and not a
    mismatch between two different time grids.

    Returns (rmse_absolute, rmse_normalised, terminal_payoff_rms, driver).
    """
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    T = float(prob.T)
    dt = T / n_steps
    # torch.tensor refuses arrays with negative strides, and riccati() returns y[:, ::-1], so copy.
    tf = lambda M: torch.tensor(np.ascontiguousarray(M), dtype=torch.float32, device=dev)  # noqa: E731

    # Exact solution on the recursion's own node set.
    t_nodes = np.linspace(0.0, T, n_steps + 1)
    P_np, c_np = prob.riccati(t_nodes)
    P = tf(P_np)                                  # (n_steps+1, d, d)
    A, B, Q, R_inv = tf(prob.A), tf(prob.B), tf(prob.Q), tf(prob.R_inv)
    QT, Sigma = tf(prob.QT), tf(prob.Sigma)

    V0 = float(prob.reference(t_nodes).value(x0, 0.0))
    V_scale = max(0.5 * float(np.max(np.abs(prob.QT))) * float(np.max(np.abs(x0))) ** 2, 1e-8)

    gen = torch.Generator(device=dev).manual_seed(seed)
    dW = torch.randn(n_steps, n_paths, prob.state_dim, device=dev, generator=gen) * np.sqrt(dt)

    x = torch.tensor(x0, dtype=torch.float32, device=dev).expand(n_paths, -1).clone()
    V = torch.full((n_paths, 1), V0, dtype=torch.float32, device=dev)
    t0 = time.perf_counter()
    with torch.no_grad():
        for n in range(n_steps):
            Pn = P[n]                                     # EXACT grad V = -P(t) x
            g = -(x @ Pn.T)                               # (N, d)
            u = (g @ B) @ R_inv.T
            run = 0.5 * torch.einsum("ni,ij,nj->n", x, Q, x).unsqueeze(1) \
                + 0.5 * torch.einsum("ni,ij,nj->n", g, B @ R_inv @ B.T, g).unsqueeze(1)
            V = V + run * dt + torch.einsum("ni,ni->n", g @ Sigma, dW[n]).unsqueeze(1)
            x = x + (x @ A.T + u @ B.T) * dt + dW[n] @ Sigma.T
    secs = time.perf_counter() - t0
    payoff = -0.5 * torch.einsum("ni,ij,nj->n", x, QT, x).unsqueeze(1)
    abs_rmse = float(((V - payoff).pow(2).mean()).sqrt())
    norm_rmse = float(((V - payoff).pow(2).mean() / V_scale ** 2).sqrt())
    return abs_rmse, norm_rmse, float(payoff.pow(2).mean().sqrt()), secs


def epoch_cost(prob: ACProblem, x0: np.ndarray, n_steps: int, n_paths: int, device: str) -> float:
    """Seconds for one forward+backward training epoch, so the epoch budget can be matched."""
    s = DeepBSDE(prob, x0, n_steps=n_steps, n_paths=n_paths, epochs=1, device=device,
                 log_every=0)
    s.solve()
    return s.solve_time / 1


def main() -> int:
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    V_true = float(prob.reference(np.linspace(0.0, prob.T, 401)).value(x0, 0.0))
    V_scale = max(0.5 * float(np.max(np.abs(prob.QT))) * float(np.max(np.abs(x0))) ** 2, 1e-8)

    print("=" * 96)
    print("PROBE: the EXACT solution's own floor in the deep-BSDE Euler recursion, vs n_steps")
    print("=" * 96)
    print(f"  device={device}  exact V(x0,0)={V_true:+.8e}  V_scale={V_scale:.4e}")
    print("  The exact floor is a HARD LOWER BOUND on any network trained inside this recursion.")
    print()
    print(f"  {'n_steps':>8} {'dt':>10} {'payoff RMS':>12} {'exact RMSE':>12} {'exact RATIO':>12} "
          f"{'secs':>7}")

    steps = [25, 50, 100, 200, 400, 800, 1600]
    rows = []
    for n in steps:
        a, nrm, prms, secs = exact_floor(prob, x0, n, 1024, device=device)
        rows.append({"n_steps": n, "dt": prob.T / n, "payoff_rms": prms, "exact_rmse_abs": a,
                     "exact_rmse_norm": nrm, "seconds": secs})
        print(f"  {n:>8} {prob.T / n:>10.5f} {prms:>12.4e} {a:>12.4e} {nrm:>12.4e} {secs:>7.2f}")

    print()
    print("  per-epoch TRAINING cost (forward+backward), which sets the matched-work budget:")
    print(f"  {'n_steps':>8} {'s/epoch':>10} {'epochs for 570s':>18}")
    costs = []
    for n in [50, 100, 200, 400]:
        c = epoch_cost(prob, x0, n, 1024, device)
        costs.append({"n_steps": n, "sec_per_epoch": c, "epochs_in_570s": int(570 / c)})
        print(f"  {n:>8} {c:>10.4f} {int(570 / c):>18}")

    (OUT / "probe_bsde_floor.json").write_text(json.dumps(
        {"V_true": V_true, "V_scale": V_scale, "floors": rows, "costs": costs,
         "device": device, "n_paths": 1024}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'probe_bsde_floor.json'}")

    # The honest reading, printed so it cannot be forgotten between sessions.
    print()
    print("-" * 96)
    print("READING:")
    f = {r["n_steps"]: r["exact_rmse_abs"] for r in rows}
    for lo, hi in [(50, 200), (200, 400), (400, 800)]:
        if lo in f and hi in f:
            order = math.log(f[lo] / f[hi]) / math.log(hi / lo) if f[hi] > 0 else float("nan")
            print(f"  exact floor {lo:>5} -> {hi:>5}: {f[lo]:.3e} -> {f[hi]:.3e}   "
                  f"observed order p={order:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
