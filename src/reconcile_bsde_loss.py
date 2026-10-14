"""
FINAL RECONCILIATION: the training loss and the evaluated loss disagree by ~20x at epoch 1000.

The contradiction, as it stands
-------------------------------
After training exactly 1000 epochs of the archived loop, on a single fixed set of Brownian
increments:

    the training step itself reports   loss.sqrt() = 1.1171e-01   (normalised)
    evaluating those SAME parameters on the SAME increments gives 5.5817e-03  (normalised)

Those are supposed to be the same number computed from the same parameters and the same paths. A
20x disagreement between a quantity and itself is not a modelling question, it is a bug, and until
it is located neither the training numbers nor the evaluation numbers can be trusted -- so the
whole discretisation-versus-network verdict is held back on it.

The two computations differ in exactly one respect, and it is a real one: the training loss is a
torch TENSOR with a graph, and the evaluation is under torch.no_grad() with a re-implemented
recursion. This script computes the loss BOTH ways inside one function, on one batch, from one
state, with no training in between, so the first differing digit localises the cause.

Usage: python src/reconcile_bsde_loss.py [--epochs 1000]
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

from ac_problem import ACProblem                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402

OUT = Path("results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--epochs", type=int, default=1000)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths, epochs=args.epochs,
                        device=device, seed=0, log_every=0, eval_paths=args.n_paths)

    print("=" * 100)
    print("FINAL RECONCILIATION: the training loss against the evaluated loss, same state, same batch")
    print("=" * 100)

    opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
    last_dW = None
    last_norm = None
    for ep in range(args.epochs):
        s.dW = None
        s._brownian(torch.Generator(device=s.device).manual_seed(1_000_000 + ep))
        # Snapshot the batch BEFORE it can be touched by anything downstream.
        batch = s.dW.clone()
        opt.zero_grad(set_to_none=True)
        xT, VT = s._rollout(s.theta, s._grad_net)
        g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        loss = ((VT - g) / s.V_scale).pow(2).mean()
        loss.backward()
        opt.step()
        last_dW = batch
        last_norm = float(loss.sqrt())

    print(f"  trained {args.epochs} epochs with the archived loop (reseeded per epoch)")
    print(f"  last epoch's TRAINING loss.sqrt() = {last_norm:.4e}   (this is what the archived "
          f"log printed)")
    print()

    # ---------------- now measure that same state on that same batch ----------------
    def measure(dW: torch.Tensor, label: str) -> dict:
        keep = s.dW
        s.dW = dW
        out = {"label": label, "width": int(dW.shape[1])}

        # (1) exactly the training expression, WITH grad
        s.net.zero_grad(set_to_none=True)
        xT, VT = s._rollout(s.theta, s._grad_net)
        gg = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        L = ((VT - gg) / s.V_scale).pow(2).mean()
        out["with_grad_normalised"] = float(L.sqrt())
        out["with_grad_absolute"] = float(L.sqrt()) * s.V_scale

        # (2) the same expression under no_grad
        with torch.no_grad():
            xT2, VT2 = s._rollout(s.theta, s._grad_net)
            gg2 = -0.5 * torch.einsum("ni,ij,nj->n", xT2, s.QT, xT2).unsqueeze(1)
            L2 = ((VT2 - gg2) / s.V_scale).pow(2).mean()
            out["no_grad_normalised"] = float(L2.sqrt())
            out["no_grad_absolute"] = float(L2.sqrt()) * s.V_scale

            # (3) the re-implemented recursion, same batch
            xT3, VT3 = s._terminal_batch(int(dW.shape[1]))
            gg3 = -0.5 * torch.einsum("ni,ij,nj->n", xT3, s.QT, xT3).unsqueeze(1)
            L3 = ((VT3 - gg3) / s.V_scale).pow(2).mean()
            out["terminal_batch_normalised"] = float(L3.sqrt())
            out["terminal_batch_absolute"] = float(L3.sqrt()) * s.V_scale

        # (4) are the two recursions producing the same terminal STATE at all?
        with torch.no_grad():
            out["max_abs_xT_diff"] = float((xT2 - xT3).abs().max())
            out["xT_rms"] = float(xT2.pow(2).mean().sqrt())

        s.dW = keep
        return out

    rows = [measure(last_dW, "last training batch")]
    fresh = torch.randn(s.n_steps, args.n_paths, s.d, device=s.device,
                        generator=torch.Generator(device=s.device).manual_seed(424242)) \
        * np.sqrt(s.dt)
    rows.append(measure(fresh, "fresh batch (seed 424242)"))

    for r in rows:
        print(f"  --- {r['label']} ({r['width']} paths) ---")
        print(f"      (1) training expression, with grad : {r['with_grad_normalised']:.4e} "
              f"normalised | {r['with_grad_absolute']:.4e} absolute")
        print(f"      (2) same, under no_grad            : {r['no_grad_normalised']:.4e} "
              f"normalised | {r['no_grad_absolute']:.4e} absolute")
        print(f"      (3) re-implemented recursion       : {r['terminal_batch_normalised']:.4e} "
              f"normalised | {r['terminal_batch_absolute']:.4e} absolute")
        print(f"      terminal STATE agreement: max|xT_diff| = {r['max_abs_xT_diff']:.3e} "
              f"against xT rms {r['xT_rms']:.3e}")
        print(f"      (2)/(1) = {r['no_grad_normalised'] / r['with_grad_normalised']:.6f}   "
              f"(3)/(2) = {r['terminal_batch_normalised'] / r['no_grad_normalised']:.6f}")
        print()

    A, B = rows[0], rows[1]
    print("-" * 100)
    consistent = (abs(A["no_grad_normalised"] / A["with_grad_normalised"] - 1) < 1e-4
                  and abs(A["terminal_batch_normalised"] / A["no_grad_normalised"] - 1) < 1e-4)
    if consistent:
        print("  All three computations agree on the same state and batch. So the evaluated floor is")
        print("  NOT an evaluator artefact, and the 20x gap between the TRAINING loss and the")
        print("  EVALUATED loss is a property of the two BATCHES, not of the two code paths:")
        print(f"    last training batch (reseeded at seed 1000000+epoch): "
              f"{A['with_grad_normalised']:.4e}")
        print(f"    fresh batch                                       : "
              f"{B['with_grad_normalised']:.4e}")
        print("  A batch drawn by `1_000_000 + ep` is an ordinary standard normal draw, so a 20x")
        print("  difference in loss between it and another ordinary draw means the loss is EXTREMELY")
        print("  heavy-tailed in the paths: a few paths dominate, and which batch you happen to draw")
        print("  decides the number. If so, the floor is a statement about the tail of the path")
        print("  distribution, and the estimator's stability -- not just its mean -- is the thing to")
        print("  report.")
    else:
        print("  The three computations DISAGREE, so one of the code paths is still wrong. The")
        print("  per-row ratios above say which: (2)/(1) tests the autograd path, (3)/(2) tests the")
        print("  re-implemented recursion.")
        print("  The terminal-STATE agreement line is the sharper diagnostic: if max|xT_diff| is")
        print("  NOT zero then the two recursions are stepping different states, and the value")
        print("  comparison downstream is meaningless.")

    (OUT / "reconcile_bsde_loss.json").write_text(json.dumps(
        {"n_steps": args.n_steps, "epochs": args.epochs, "training_loss_norm": last_norm,
         "measurements": rows}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'reconcile_bsde_loss.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
