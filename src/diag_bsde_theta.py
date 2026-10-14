"""
DIAGNOSTIC: why does the value at x0 stop improving, and does theta actually converge?

Found while reading the sweep trace, and worth chasing rather than noting and moving past: the
value error at x0 goes 0.59 -> 0.16 -> 0.27 -> 0.46 as training proceeds, while the terminal loss
falls monotonically. A single learned scalar theta = V(x0,0)/V_scale cannot behave like that if it
is converging. Either

  (a) theta is still wandering, in which case theta is under-identified by this objective and the
      value error is not a training-progress signal at all, or
  (b) theta has settled and something else in the readout is moving.

The distinction matters for the verdict, not just for tidiness. If theta is under-identified, then
an accuracy comparison against the grid is not merely unmet, it is ill-posed at x0 for this solver
as formulated -- which is a much stronger and more specific statement for the paper than "did not
reach target accuracy", and it points at the fix (a loss that constrains the value, e.g. a
martingale or log-variance loss) rather than at more epochs.

This runs ONE configuration, logs every quantity that could be moving, and prints the correlation
between theta and the terminal loss so the answer is read off numbers rather than inferred.

Usage: python src/diag_bsde_theta.py [--n-steps 50] [--epochs 1500] [--every 50]
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
from exp_bsde_steps import DeepBSDEMonitor, OUT   # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=1500)
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    # ref.value returns shape (1,) and ref.P returns shape (1,1) even at d = 1, so neither is a
    # scalar. Silently keeping the array shape makes every downstream "is it a number" assumption
    # wrong in a way that only shows up as a formatting error three lines later.
    V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])
    grad_true = np.atleast_2d(-np.atleast_2d(x0) @ ref.P(0.0)).ravel().astype(float)

    s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths,
                        epochs=args.epochs, device=device, seed=0, log_every=0, eval_paths=8192)
    print("=" * 100)
    print("DIAGNOSTIC: is theta converging, and what is the value error made of?")
    print("=" * 100)
    print(f"  n_steps={args.n_steps}  epochs={args.epochs}  V_scale={s.V_scale:.4e}  "
          f"V_scale*theta at init={float(s.theta.detach()) * s.V_scale:+.4e}")
    print(f"  exact V(x0,0)={V_true:+.8e}   exact grad V(x0,0)={grad_true[0]:+.6e}")
    print()
    print(f"  {'epoch':>6} {'theta(raw)':>13} {'V=V_s*theta':>14} {'V_rel_err':>10} "
          f"{'termRMSE':>11} {'mean(VT-g)':>12} {'grad_x0':>12} {'grad_rel':>9}")

    # A light re-implementation of the training loop rather than solve_monitored, because this
    # needs the batch MEAN of (V_T - g) -- the quantity that actually drives theta -- which the
    # monitored loop does not expose.
    gen = torch.Generator(device=s.device).manual_seed(0)
    s._brownian(gen)
    opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
    rows = []
    for ep in range(1, args.epochs + 1):
        s.dW = s._paths(s.n_paths, 1_000_000 + ep - 1)
        opt.zero_grad(set_to_none=True)
        xT, VT = s._rollout(s.theta, s._grad_net)
        g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
        mismatch = (VT - g)
        loss = ((mismatch / s.V_scale).pow(2)).mean()
        loss.backward()
        opt.step()
        if ep % args.every == 0 or ep == 1:
            V_hat, g_hat = s.value_grad_at_x0()
            rmse = s.terminal_rmse()
            rows.append({
                "epoch": ep, "theta_raw": float(s.theta.detach()),
                "V_hat": V_hat, "value_rel_err": abs(V_hat - V_true) / abs(V_true),
                "terminal_rmse": rmse,
                "mean_mismatch": float(mismatch.mean()),
                "sd_mismatch": float(mismatch.std()),
                "grad_x0": float(g_hat[0]),
                "grad_rel_err": float(abs(g_hat[0] - grad_true[0]) / abs(grad_true[0])),
            })
            r = rows[-1]
            print(f"  {ep:>6} {r['theta_raw']:>13.5e} {r['V_hat']:>14.6e} "
                  f"{r['value_rel_err']:>10.4f} {r['terminal_rmse']:>11.3e} "
                  f"{r['mean_mismatch']:>12.3e} {r['grad_x0']:>12.4e} "
                  f"{r['grad_rel_err']:>9.4f}", flush=True)

    # ---------------- read it off ----------------
    V_vals = np.array([r["V_hat"] for r in rows])
    rmse_vals = np.array([r["terminal_rmse"] for r in rows])
    miss = np.array([r["mean_mismatch"] for r in rows])
    print()
    print("-" * 100)
    print(f"  theta ran the value over [{V_vals.min():.4e}, {V_vals.max():.4e}] "
          f"(true {V_true:.4e}); spread {(V_vals.max() - V_vals.min()) / abs(V_true):.1%} of |V_true|")
    print(f"  late-run value drift (last half): "
          f"{abs(V_vals[len(V_vals) // 2:].max() - V_vals[len(V_vals) // 2:].min()) / abs(V_true):.1%}"
          f" of |V_true|")
    print(f"  corr(V_hat, terminal RMSE) = {np.corrcoef(V_vals, rmse_vals)[0, 1]:+.3f}")
    print(f"  mean(V_T - g) magnitude vs its own path-to-path sd: "
          f"{abs(miss[-1]) / (rows[-1]['sd_mismatch'] + 1e-30):.4f} "
          f"(the sd is the noise theta's gradient must be extracted from)")
    print(f"  path-to-path sd of the mismatch, last checkpoint: {rows[-1]['sd_mismatch']:.3e}"
          f"  vs terminal RMSE {rows[-1]['terminal_rmse']:.3e}")

    (OUT / "diag_bsde_theta.json").write_text(json.dumps(
        {"n_steps": args.n_steps, "epochs": args.epochs, "V_true": V_true,
         "grad_true": grad_true.tolist(), "V_scale": s.V_scale, "rows": rows}, indent=2),
        encoding="utf-8")
    print(f"\nwrote {OUT / 'diag_bsde_theta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
