"""
RECONCILIATION: the prior session's reported deep-BSDE floor does not match a re-measurement.

The discrepancy, stated precisely so it cannot be waved away
-----------------------------------------------------------
The archived deep-BSDE training run reports the terminal loss plateauing at 2.10e-03 in the
NORMALISED metric (the quantity its own code prints). Re-running the same measurement on the same
problem reproduces V_rel_err = 0.476 EXACTLY -- but the normalised terminal loss reads 1.845e-01,
a factor of 88 larger.

Both numbers cannot describe the same run. There are only three possibilities:

  (a) the re-measurement is wrong;
  (b) the archived run evaluated on FIXED Brownian paths and the reported floor is a path-specific
      residual, not the loss of the trained parameters under a fresh realisation;
  (c) the archived number was taken in different units.

(c) can be ruled out by arithmetic: the code prints sqrt of the normalised loss, and multiplying it
back by V_scale = 5e-2 gives 1.05e-04 absolute, which is 2.1x the exact floor. A factor of 88
cannot be produced by any single unit conversion of the same quantity.

This matters far more than a bookkeeping quarrel, because the whole verdict rule rests on the size
of the floor relative to the exact floor:
  * if the trained floor is 2.1x the exact floor, the solver is nearly at the discretisation limit
    and the answer leans CONTINUE;
  * if it is 183x, the solver is nowhere near it and the answer leans COST-ONLY.
Deciding between those on an unreconciled number would be exactly the kind of confident wrong
answer this project has already produced four times, so it gets settled by measurement.

The discriminating test
-----------------------
Evaluate the trained parameters BOTH ways, on the same state:
  * fixed batch: the same paths every epoch (what the archived run's evaluation saw), and
  * fresh batches: independent realisations, which is what "the loss of these parameters" means.
If fixed-path evaluation is dramatically lower, (b) is confirmed and the archived floor was a
path-specific residual. Both are reported here with enough paths that the difference cannot be
Monte Carlo noise.

Usage: python src/check_bsde_reconcile.py [--n-steps 50] [--epochs 3000]
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
from deep_bsde import DeepBSDE                    # noqa: E402
from probe_bsde_floor import exact_floor          # noqa: E402

OUT = Path("results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--probe-paths", type=int, default=8192)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])
    grad_true = np.atleast_2d(-np.atleast_2d(x0) @ ref.P(0.0)).ravel().astype(float)

    floor_abs, floor_norm, payoff_rms, _ = exact_floor(prob, x0, args.n_steps, args.n_paths,
                                                       device=device)
    print("=" * 100)
    print("RECONCILIATION: is the deep-BSDE floor 2.1x the exact floor, or 183x?")
    print("=" * 100)
    print(f"  n_steps={args.n_steps}  epochs={args.epochs}  n_paths={args.n_paths}")
    print(f"  exact floor: absolute {floor_abs:.4e}   normalised {floor_norm:.4e}")
    print(f"  (the archived run's own V_rel_err = 0.4763 is reproduced below, so the comparison is "
          f"like for like)")

    # The training loop below is copied from the archived deep_bsde.py verbatim, including the
    # per-epoch reseeding, so any difference in the measured floor is measurement, not training.
    s = DeepBSDE(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths, epochs=args.epochs,
                 device=device, seed=0, log_every=0)
    m = int(round(floor_abs / floor_norm))          # V_scale, recovered from the two floors
    print(f"  V_scale (implied by the two exact floors) = {m}  "
          f"[code computes it as 0.5*max|QT|*max|x0|^2 = {s.V_scale:.4e}]")

    gen = torch.Generator(device=s.device).manual_seed(0)
    s._brownian(gen)
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
        if (ep + 1) % 500 == 0:
            print(f"      epoch {ep + 1:>6}  normalised terminal RMSE {float(loss.sqrt()):.4e}"
                  f"   (in-training, one batch)", flush=True)

    # ---------------- the discriminating measurement ----------------
    # The batch width must equal the TRAINING width: _rollout expands V from theta to
    # self.n_paths, so it will not accept a batch of any other size. That constraint is itself a
    # hint about where the archived number came from -- its floor was read off a batch of exactly
    # this width, the training width.
    n_eval = args.n_paths

    def eval_floor(seed: int) -> float:
        """Absolute terminal RMSE of the trained parameters on a FRESH batch."""
        keep = s.dW
        gg = torch.Generator(device=s.device).manual_seed(seed)
        s.dW = torch.randn(s.n_steps, n_eval, s.d, device=s.device, generator=gg) * np.sqrt(s.dt)
        with torch.no_grad():
            xT, VT = s._rollout(s.theta, s._grad_net)
            payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            r = float(((VT - payoff).pow(2).mean()).sqrt())
        s.dW = keep
        return r

    def eval_on_batch(dW: torch.Tensor) -> tuple[float, float]:
        """(absolute, normalised) terminal RMSE on a SPECIFIC batch, as the archived code saw it."""
        keep = s.dW
        s.dW = dW
        with torch.no_grad():
            xT, VT = s._rollout(s.theta, s._grad_net)
            payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            rmse = float(((VT - payoff).pow(2).mean()).sqrt())
        s.dW = keep
        return rmse, rmse / s.V_scale

    # The archived code's own evaluation ran on whatever dW was in place -- the LAST TRAINING
    # BATCH, still holding the epoch-`epochs` draws. Reproduce exactly that readout.
    last_batch_abs, last_batch_norm = eval_on_batch(s.dW)

    # Fresh independent realisations: what "the loss of these parameters" means.
    fresh = [eval_floor(500_000 + i) for i in range(5)]
    fresh_mean = sum(fresh) / len(fresh)
    fresh_sd = float(np.std(fresh))

    V_hat = float(s.theta.detach()) * s.V_scale
    print()
    print("-" * 100)
    print(f"  trained V_hat = {V_hat:+.6e}   V_true = {V_true:+.6e}   "
          f"V_rel_err = {abs(V_hat - V_true) / abs(V_true):.4f}   "
          f"<- compare with the archived run's 0.4763")
    print()
    print(f"  ON THE ARCHIVED CODE'S OWN BATCH (the last training batch, {n_eval} paths, which is "
          f"exactly what its trailing `loss.sqrt()` printed):")
    print(f"     absolute {last_batch_abs:.4e}   NORMALISED {last_batch_norm:.4e}   "
          f"<- the archived run reported 2.10e-03")
    print()
    print(f"  FRESH batches, {n_eval} paths each, 5 seeds:")
    print(f"     {' '.join(f'{v:.4e}' for v in fresh)}")
    print(f"     mean {fresh_mean:.4e} +- {fresh_sd:.2e} absolute   "
          f"= {fresh_mean / s.V_scale:.4e} normalised")
    print(f"  exact floor: {floor_abs:.4e} absolute = {floor_norm:.4e} normalised")
    print(f"  ratio to exact floor:  last-batch {last_batch_abs / floor_abs:.1f}x   "
          f"fresh {fresh_mean / floor_abs:.1f}x")
    print(f"  archived run reported: 2.10e-03 normalised = {2.10e-03 * s.V_scale:.3e} absolute "
          f"= {2.10e-03 / floor_norm:.1f}x the exact floor")
    print()
    print("  VERDICT ON THE DISCREPANCY:")
    if last_batch_abs / fresh_mean > 1.5:
        print("    The archived floor is far lower on the batch it was computed from than on fresh")
        print("    realisations: it is a PER-BATCH residual, not the loss of the trained")
        print("    parameters. The fresh-batch floor is the quantity that transfers, and it is the")
        print("    one to carry forward. Any 'floor' quoted from a single reused batch -- including")
        print("    one quoted by the prior session -- understates the solver's error.")
    elif fresh_mean / last_batch_abs > 1.5:
        print("    The last training batch is HARDER than fresh batches, which would be odd and")
        print("    suggests the last epoch's draws are not representative. Use the fresh mean.")
    else:
        print("    Last-batch and fresh-batch evaluation agree within noise, so the archived floor")
        print("    is NOT a path-reuse artefact. The re-measurement is then the more trustworthy of")
        print("    the two and the archived 2.10e-03 is unreproducible until a cause is found.")

    (OUT / "check_bsde_reconcile.json").write_text(json.dumps({
        "n_steps": args.n_steps, "epochs": args.epochs, "n_paths": args.n_paths,
        "exact_floor_abs": floor_abs, "exact_floor_norm": floor_norm, "V_scale": s.V_scale,
        "V_hat": V_hat, "V_true": V_true,
        "value_rel_err": abs(V_hat - V_true) / abs(V_true),
        "last_batch_floor_abs": last_batch_abs, "last_batch_floor_norm": last_batch_norm,
        "fresh_batch_floors": fresh,
        "fresh_mean": fresh_mean, "fresh_sd": fresh_sd,
        "archived_reported_norm": 2.10e-03,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'check_bsde_reconcile.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
