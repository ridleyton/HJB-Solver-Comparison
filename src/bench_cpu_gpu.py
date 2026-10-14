"""
Measure throughput for the computation that dominates a PI-PINN step, on CPU and GPU.

Why this measures the right thing: the per-step cost of both our PINN and their JAX PI-PINN is
dominated by the same primitive -- a backward pass plus one second derivative per state
dimension, evaluated over a collocation batch. Their code additionally forms a full (N x N)
Hessian per sample (`hji_pi_pinn/problems/publisher_subscriber/problem.py`, `hess_x`) and takes
`trace(sigma sigma^T H)`, which is strictly more work than the Laplacian we take, so a ratio
measured here is a *lower* bound on their CPU penalty.

The batch size matters as much as the device: their `Ni = 1000 * N^2`, which is 9000 for N=3 and
121000 for N=11, against our 2048. We therefore measure at both.

This does not and cannot measure JAX itself (it is not installable here). It measures the
arithmetic, which is the part that determines the CPU/GPU ratio.
"""

from __future__ import annotations

import sys
import time

import torch

sys.path.insert(0, "src")
from ac_problem import ACProblem          # noqa: E402
from pinn_pi import PIPINN                # noqa: E402

STEPS = 150
BUDGET_S = 25.0        # time each case for about this long, so slow CPU cases still finish


def bench(device: str, p: int, n_points: int, batch_label: str) -> dict:
    prob = ACProblem(p=p, hetero=True, het_seed=1)
    s = PIPINN(prob, num_iters=1, num_epochs=STEPS, n_points=n_points,
               refresh_every=10**9, lr=1e-3, device=device, seed=0)

    gen = torch.Generator(device=s.device).manual_seed(0)
    x, t = s._sample(n_points, gen)
    pol = s._zero_policy(x)          # (N, control_dim) -- NOT zeros_like(x), which is the STATE

    def one_step():
        s.net.zero_grad(set_to_none=True)
        xx = x.detach().clone().requires_grad_(True)
        tt = t.detach().clone().requires_grad_(True)
        loss = (s._loss_pi(xx, tt, pol) / s.residual_scale).pow(2).mean()
        loss.backward()

    def sync():
        if s.device.type == "cuda":
            torch.cuda.synchronize()

    # Warm up enough to cover lazy init and any cudnn autotuning.
    for _ in range(5):
        one_step()
    sync()

    if s.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    # Time a single step, then choose a repeat count that fits the time budget. This keeps the
    # big-batch CPU cases from running for hours while still measuring them.
    sync(); t0 = time.perf_counter(); one_step(); sync()
    single = time.perf_counter() - t0
    reps = int(max(3, min(STEPS, BUDGET_S / max(single, 1e-6))))

    sync(); t0 = time.perf_counter()
    for _ in range(reps):
        one_step()
    sync()
    dt = time.perf_counter() - t0
    return {
        "device": device, "p": p, "batch": n_points, "batch_label": batch_label,
        "reps": reps, "steps_per_s": reps / dt, "ms_per_step": 1000 * dt / reps,
        "peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                    if s.device.type == "cuda" else None),
    }


def main() -> None:
    has_gpu = torch.cuda.is_available()
    print(f"torch {torch.__version__}  cuda={has_gpu}"
          + (f"  ({torch.cuda.get_device_name(0)})" if has_gpu else ""))
    print(f"steps per timing: {STEPS}")
    print("=" * 104)

    cases = [
        (3, 2048, "ours"),
        (3, 9000, "their Ni, N=3"),
        (11, 2048, "ours"),
        (11, 121000, "their Ni, N=11"),
    ]
    rows = []
    for p, nb, label in cases:
        for dev in (["cuda", "cpu"] if has_gpu else ["cpu"]):
            try:
                r = bench(dev, p, nb, label)
                rows.append(r)
                gm = (f"  peak {r['peak_gb']:.2f} GB" if r["peak_gb"] else "")
                print(f"  p={p:2d} batch={nb:>7d} ({label:>15s}) {dev:>4s}: "
                      f"{r['steps_per_s']:8.1f} steps/s  "
                      f"({r['ms_per_step']:8.2f} ms/step){gm}", flush=True)
            except Exception as e:
                print(f"  p={p} batch={nb} {dev} FAILED: {type(e).__name__}: {e}")

    print("\n" + "=" * 104)
    print("CPU penalty, and what it implies for their default 2.5e6-step run")
    print("=" * 104)
    by = {(r["p"], r["batch"], r["device"]): r for r in rows}
    for p, nb, label in cases:
        g = by.get((p, nb, "cuda"))
        c = by.get((p, nb, "cpu"))
        ratio = (g["steps_per_s"] / c["steps_per_s"]) if (g and c) else None
        if c:
            hours = 2.5e6 / c["steps_per_s"] / 3600
            note = f"CPU: {hours:8.1f} h for one 2.5e6-step run"
        else:
            note = "CPU: not measured"
        if ratio:
            note += f"   (CPU is {ratio:.0f}x slower)"
        print(f"  p={p:2d} batch={nb:>7d}: {note}")
        if c:
            h_red = 24000 / c["steps_per_s"] / 3600
            print(f"                    reduced 2.4e4-step run: {h_red*60:6.1f} min on CPU")


if __name__ == "__main__":
    main()
