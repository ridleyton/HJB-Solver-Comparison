"""
DECISIVE TEST: is the evaluator wrong, or is the training loop wrong?

Where this stands. Three facts are now established, and they cannot all be true at once:

  1. The archived training run is reproduced EXACTLY. Crosschecking the two loop conventions over
     identical paths gives identical losses, and at epoch 500 that loss is 1.8138e-01 -- the
     archived log records 1.8138e-01 at epoch 500. Training is faithful.
  2. Re-measuring the trained parameters on FRESH batches gives 5.0402e-03 normalised, i.e. 5.0x the
     exact floor. That is close to the archived plateau of 2.10e-03.
  3. The sweep harness reports 1.845e-01 normalised at epoch 7000 -- 37x LARGER, and it reports
     1.6015e-02 at epoch 100 where the training loss itself is 2.7e-01.

Fact 3 is internally impossible: an evaluation of trained parameters cannot be 17x SMALLER than the
loss the parameters were just trained to, on the same distribution. So the sweep's evaluator is
computing something different from what it claims.

The one structural difference between the two evaluators: the sweep measures through
DeepBSDEMonitor._terminal_batch, a re-implementation of DeepBSDE._rollout written to allow a batch
of any width (the solver's own rollout expands V from theta to self.n_paths and so refuses other
widths). A re-implementation that silently differs -- notably in the sign or the scale of the
value recursion -- would produce exactly this symptom.

This script settles it by evaluating ONE trained state THREE ways:
    (a) DeepBSDE._rollout            -- the solver's own, the definition of the loss;
    (b) _terminal_batch              -- the sweep's re-implementation;
    (c) _terminal_batch, independent batch -- to confirm (b) is at least self-consistent.
Any disagreement between (a) and (b) is a bug in (b), and every floor the sweep reported is void
until it is fixed.

Usage: python src/check_bsde_evaluator.py [--epochs 1000]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np          # noqa: E402
import torch                # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor                        # noqa: E402

OUT = Path("results")


def via_rollout(s: DeepBSDEMonitor, dW: torch.Tensor) -> float:
    """(a) The solver's own recursion. This IS the loss -- every other number must match it."""
    keep = s.dW
    s.dW = dW
    with torch.no_grad():
        xT, VT = s._rollout(s.theta, s._grad_net)
        payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        r = float(((VT - payoff).pow(2).mean()).sqrt())
    s.dW = keep
    return r


def via_terminal_batch(s: DeepBSDEMonitor, dW: torch.Tensor) -> float:
    """(b) The sweep's re-implementation, on the SAME batch."""
    keep = s.dW
    s.dW = dW
    with torch.no_grad():
        xT, VT = s._terminal_batch(dW.shape[1])
        payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        r = float(((VT - payoff).pow(2).mean()).sqrt())
    s.dW = keep
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1000)
    ap.add_argument("--n-paths", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0

    s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths, epochs=args.epochs,
                        device=device, seed=0, log_every=0, eval_paths=args.n_paths)

    # Train with the archived convention, which the crosscheck proved faithful.
    opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
    for ep in range(args.epochs):
        s.dW = None
        s._brownian(torch.Generator(device=s.device).manual_seed(1_000_000 + ep))
        opt.zero_grad(set_to_none=True)
        xT, VT = s._rollout(s.theta, s._grad_net)
        g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        loss = ((VT - g) / s.V_scale).pow(2).mean()
        loss.backward()
        opt.step()
        if (ep + 1) % 250 == 0:
            print(f"    trained {ep + 1} epochs, in-training normalised RMSE "
                  f"{float(loss.sqrt()):.4e}", flush=True)

    print()
    print("=" * 100)
    print("EVALUATOR TEST -- one trained state, three measurements of the same quantity")
    print("=" * 100)
    print(f"  {'batch':>28} {'(a) _rollout':>14} {'(b) _terminal_batch':>21} "
          f"{'ratio (b)/(a)':>14}")
    rows = []
    # _rollout expands V from theta to self.n_paths and therefore refuses any other width, so every
    # comparison row uses the training width -- the width the archived evaluation used too.
    for seed in (999_999, 424242, 7, 12345):
        gg = torch.Generator(device=s.device).manual_seed(seed)
        dW = torch.randn(s.n_steps, args.n_paths, s.d, device=s.device,
                         generator=gg) * np.sqrt(s.dt)
        a = via_rollout(s, dW)
        b = via_terminal_batch(s, dW)
        rows.append({"seed": seed, "width": args.n_paths, "rollout": a, "terminal_batch": b,
                     "ratio": b / a})
        print(f"  {'batch seed ' + str(seed):>28} {a:>14.4e} {b:>21.4e} {b / a:>14.4f}")

    # And once more at a width the solver's own rollout cannot reach, to prove the chunking path
    # (used by the 65536-path final estimate) is consistent with the unchunked one.
    gg = torch.Generator(device=s.device).manual_seed(31)
    wide = 3 * args.n_paths
    dW = torch.randn(s.n_steps, wide, s.d, device=s.device, generator=gg) * np.sqrt(s.dt)
    keep = s.dW
    s.dW = dW
    with torch.no_grad():
        xT, VT = s._terminal_batch(wide)
        payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        unchunked = float(((VT - payoff).pow(2).mean()).sqrt())
    s._terminal_batch = lambda n: DeepBSDEMonitor._terminal_batch_chunked(s, n, chunk=512)
    with torch.no_grad():
        xT, VT = s._terminal_batch(wide)
        payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        chunked = float(((VT - payoff).pow(2).mean()).sqrt())
    s.dW = keep
    print(f"  {'chunking check (3x width)':>28} {unchunked:>14.4e} {chunked:>21.4e} "
          f"{chunked / unchunked:>14.4f}")

    same = all(abs(r["ratio"] - 1.0) < 1e-3 for r in rows) and abs(chunked / unchunked - 1) < 1e-3
    print()
    if same:
        print("  (a) and (b) AGREE. The re-implementation is faithful, so the sweep's evaluator is")
        print("  NOT the source of the discrepancy and the search must continue on the training side.")
    else:
        print("  (a) and (b) DISAGREE. The sweep's evaluator is computing a different quantity from")
        print("  the solver's own loss. EVERY floor reported through it -- including all four")
        print("  matched-work points and the long-run control -- is void and must be recomputed.")
        print("  The verdict on discretisation vs network CANNOT be issued from those numbers.")

    (OUT / "check_bsde_evaluator.json").write_text(json.dumps(
        {"n_steps": args.n_steps, "epochs": args.epochs, "n_paths": args.n_paths,
         "rows": rows, "agree": same}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'check_bsde_evaluator.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
