"""
CROSS-CHECK: two training loops over the identical solver disagree by a factor of ~14.

The contradiction, and why it has to be resolved before the verdict
-------------------------------------------------------------------
Every ingredient is shared: the same ACProblem, the same DeepBSDE class, the same Adam call, the
same per-epoch reseeding, the same 1024 training paths. Yet at epoch 500:

  * the sweep harness (exp_bsde_steps.py, DeepBSDEMonitor.solve_monitored) reports a normalised
    terminal RMSE of 1.2576e-02, converging to 1.845e-01 at epoch 7000;
  * the reconcile script (check_bsde_reconcile.py) reports 1.8138e-01 at epoch 500 -- and its whole
    trace reproduces the ARCHIVED run's logged values exactly (4.8543e-03 at epoch 3000, matching
    the archived 4.85e-03).

Two loops cannot both be right. One of them is measuring a different quantity or training a
different trajectory, and the difference between them is the difference between "the solver is
183x off the discretisation floor" and "the solver is 5x off it" -- i.e. between COST-ONLY and
CONTINUE. This script therefore runs BOTH loops side by side, epoch for epoch, printing the loss
each computes from IDENTICAL parameters and IDENTICAL Brownian increments. The first line where
they diverge localises the bug to a single operation.

This is the same failure mode the project has already hit four times (a plausible-looking
measurement that disagrees with another measurement of the same thing, where the disagreement
itself is the signal), so it gets a script rather than a guess.

Usage: python src/crosscheck_bsde_loops.py [--epochs 60] [--n-steps 50]
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np          # noqa: E402
import torch                # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402

OUT = Path("results")


def make(prob, x0, n_steps, n_paths, device):
    return DeepBSDEMonitor(prob, x0, n_steps=n_steps, n_paths=n_paths, epochs=1,
                           device=device, seed=0, log_every=0, eval_paths=8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=60)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0

    # Two independent monitors, identical in every constructor argument.
    sA = make(prob, x0, args.n_steps, args.n_paths, device)   # "sweep harness" convention
    sB = make(prob, x0, args.n_steps, args.n_paths, device)   # "archived deep_bsde.py" convention

    # Identical starting parameters, so any divergence is the loop and not the initialisation.
    sB.net.load_state_dict(copy.deepcopy(sA.net.state_dict()))
    with torch.no_grad():
        sB.theta.copy_(sA.theta)

    optA = torch.optim.Adam(list(sA.net.parameters()) + [sA.theta], lr=sA.lr)
    optB = torch.optim.Adam(list(sB.net.parameters()) + [sB.theta], lr=sB.lr)

    print("=" * 104)
    print("CROSS-CHECK: sweep-harness loop (A) vs archived-loop convention (B), same solver")
    print("=" * 104)
    print(f"  n_steps={args.n_steps}  n_paths={args.n_paths}  epochs={args.epochs}")
    print(f"  identical init: max|theta_A - theta_B| = "
          f"{abs(float(sA.theta) - float(sB.theta)):.3e}")
    print()
    print(f"  {'ep':>5} {'max|dW_A-dW_B|':>17} {'loss_A':>13} {'loss_B':>13} "
          f"{'ratio A/B':>10} {'max|theta_A-theta_B|':>22}")

    rows = []
    for ep in range(1, args.epochs + 1):
        # ---- A: the sweep harness convention (DeepBSDEMonitor._paths) ----
        sA.dW = sA._paths(args.n_paths, 1_000_000 + ep - 1)
        optA.zero_grad(set_to_none=True)
        xA, VA = sA._rollout(sA.theta, sA._grad_net)
        gA = -0.5 * torch.einsum("ni,ij,nj->n", xA, sA.QT, xA).unsqueeze(1)
        lossA = ((VA - gA) / sA.V_scale).pow(2).mean()
        lossA.backward()
        optA.step()

        # ---- B: the archived deep_bsde.py convention (dW=None then _brownian) ----
        sB.dW = None
        sB._brownian(torch.Generator(device=sB.device).manual_seed(1_000_000 + ep - 1))
        optB.zero_grad(set_to_none=True)
        xB, VB = sB._rollout(sB.theta, sB._grad_net)
        gB = -0.5 * torch.einsum("ni,ij,nj->n", xB, sB.QT, xB).unsqueeze(1)
        lossB = ((VB - gB) / sB.V_scale).pow(2).mean()
        lossB.backward()
        optB.step()

        dWdiff = float((sA.dW - sB.dW).abs().max())
        tdiff = abs(float(sA.theta) - float(sB.theta))
        la, lb = float(lossA.sqrt()), float(lossB.sqrt())
        rows.append({"epoch": ep, "dW_max_abs_diff": dWdiff, "lossA": la, "lossB": lb,
                     "theta_abs_diff": tdiff})
        if ep <= 5 or ep % 10 == 0:
            print(f"  {ep:>5} {dWdiff:>17.3e} {la:>13.4e} {lb:>13.4e} "
                  f"{la / lb:>10.3f} {tdiff:>22.3e}", flush=True)
            if dWdiff > 0 and ep <= 5:
                print(f"        ^^ THE BROWNIAN INCREMENTS ALREADY DIFFER at epoch {ep}")

    first = rows[0]
    print()
    print("-" * 104)
    if first["dW_max_abs_diff"] > 0:
        print(f"  LOCALISED: the Brownian increments differ at the FIRST epoch "
              f"(max abs diff {first['dW_max_abs_diff']:.3e}).")
        print("  So the two conventions do not generate the same paths, and everything downstream --")
        print("  loss, theta, the whole floor -- diverges from epoch 1. The bug is in path")
        print("  generation, not in the rollout or the loss.")
        print()
        print("  Checked against the archived code, which does:")
        print("      self.dW = None")
        print("      self._brownian(torch.Generator(device=...).manual_seed(1_000_000 + ep))")
        print("  ...i.e. it RESETS dW and draws a standard normal, scaled by sqrt(dt), INSIDE")
        print("  _brownian. Any harness that draws the increments itself must match that draw")
        print("  exactly, including the sqrt(dt) scaling and the generator's device.")
    else:
        print("  The Brownian increments agree exactly at every epoch, so the divergence is NOT in")
        print("  path generation. Compare the loss columns above to find the first differing epoch.")
    print(f"\n  final theta: A={float(sA.theta):.6e}  B={float(sB.theta):.6e}")
    print(f"  final losses: A={rows[-1]['lossA']:.4e}  B={rows[-1]['lossB']:.4e}  "
          f"ratio {rows[-1]['lossA'] / rows[-1]['lossB']:.2f}")

    (OUT / "crosscheck_bsde_loops.json").write_text(json.dumps(
        {"n_steps": args.n_steps, "n_paths": args.n_paths, "epochs": args.epochs,
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'crosscheck_bsde_loops.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
