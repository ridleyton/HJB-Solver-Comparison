"""
Deep-BSDE training cost per epoch as a function of dimension.

Why this exists
---------------
The comparison table needs the training wall time to each run's stopping point, but the p = 1 run's
trace was recovered from a log that did not carry wall-clock, so its cost cell would have read nan.
Re-running 16,000 epochs just to time it would cost half an hour and would still leave the timing
dependent on whatever else was on the GPU at the time.

Measuring per-epoch cost directly is both cheaper and more useful: it is a small, controlled
measurement, and the cost-versus-dimension relation is itself something the journal comparison
should state. The grid scheme's cost grows as N^p and the PINN's as a fixed network; deep BSDE's
per-epoch cost grows with the number of paths times the network width, and the number of epochs
needed to converge grows with dimension too. Reporting only the product hides which factor moved.

Method, and the one thing it must not do
----------------------------------------
A few epochs of training, timed under the SAME conditions for every dimension, in one process, so
the numbers are comparable with each other. Each dimension gets a warm-up epoch that is discarded,
because the first epoch of a fresh CUDA context includes allocation and kernel-selection costs that
have nothing to do with the steady-state rate -- including it would make low dimensions look
artificially expensive and flatten the trend.

This measures TRAINING cost only. Deep BSDE is point-valued, so its inference cost is a single
forward pass per query with no grid to interpolate; that is a different measurement and is not
claimed here.

Usage: python src/probe_bsde_epoch_cost.py [--dims 1,2,4,8,16] [--epochs 8]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np          # noqa: E402
import torch                # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402

OUT = Path("results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2,4,8,16")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 96)
    print("DEEP-BSDE TRAINING COST PER EPOCH vs DIMENSION (steady state, warm-up discarded)")
    print("=" * 96)
    print(f"  device={device}  epochs timed={args.epochs}  n_steps={args.n_steps}  "
          f"n_paths={args.n_paths}")
    print()
    print(f"  {'p':>3} {'s/epoch':>10} {'vs p=1':>9} {'params':>9} {'flops/epoch (est)':>18}")

    rows = []
    base = None
    for p in [int(v) for v in args.dims.split(",")]:
        prob = ACProblem(p=p)
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths,
                            epochs=args.epochs, device=device, seed=0, log_every=0,
                            eval_paths=64)
        opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)

        def one_epoch() -> None:
            s.dW = s._paths(args.n_paths, 12345)
            opt.zero_grad(set_to_none=True)
            xT, VT = s._rollout(s.theta, s._grad_net)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            loss = ((VT - g) / s.V_scale).pow(2).mean()
            loss.backward()
            opt.step()

        one_epoch()                                   # warm-up, discarded
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(args.epochs):
            one_epoch()
        if device == "cuda":
            torch.cuda.synchronize()
        per = (time.perf_counter() - t0) / args.epochs

        n_params = sum(q.numel() for q in s.net.parameters())
        if base is None:
            base = per
        # Forward+backward through the MLP is roughly 6 * params * paths multiply-adds, plus the
        # rollout's own d^2 work. Reported as an order-of-magnitude figure only, and labelled so.
        flops = 6 * n_params * args.n_paths * args.n_steps
        print(f"  {p:>3} {per:>10.5f} {per / base:>8.2f}x {n_params:>9} {flops:>18.3e}")
        rows.append({"p": p, "sec_per_epoch": per, "ratio_to_p1": per / base,
                     "n_params": n_params, "flops_per_epoch_est": flops,
                     "n_steps": args.n_steps, "n_paths": args.n_paths})

    (OUT / "probe_bsde_epoch_cost.json").write_text(json.dumps(
        {"rows": rows, "epochs_timed": args.epochs, "device": device}, indent=2), encoding="utf-8")

    print()
    print("-" * 96)
    if len(rows) >= 2:
        lo, hi = rows[0], rows[-1]
        import math
        order = math.log(hi["sec_per_epoch"] / lo["sec_per_epoch"]) / math.log(hi["p"] / lo["p"])
        print(f"  Per-epoch cost grows like p^{order:.2f} between p={lo['p']} and p={hi['p']}, i.e. "
              f"{hi['ratio_to_p1']:.1f}x from p={lo['p']} to p={hi['p']}.")
        print("  This is the PER-EPOCH rate only. The epochs needed to converge also grow with")
        print("  dimension, so the total training cost grows faster than this line suggests; the")
        print("  table's training-time column carries the product for each measured run.")
    print(f"\nwrote {OUT / 'probe_bsde_epoch_cost.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
