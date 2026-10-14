"""
Does the deep-BSDE solver CONVERGE at d > 1? The question that gates the journal benchmark.

Where this stands
-----------------
The solver is validated and characterised at d = 1 only (deep-bsde-steps-verdict): there it reaches
a terminal residual within a factor ~1.6 of its own discretisation floor. The journal gap is that
deep BSDE -- not the PINN -- is the standard high-dimensional method in mathematical finance, and
the manuscript currently cites it once without running it, which is precisely the omission a
referee will name.

The tripwire (probe_bsde_dims.py) then found a blocker rather than a green light: at 120 epochs the
terminal residual is 2.49e-01 at d = 1 but 2.14e+03 at d = 4 and 1.48e+03 at d = 8, and it is not
falling. That is the signature of the instability the terminal-gradient anchor was introduced to
cure, reappearing as d rises -- the substituted optimal control feeds back with gain 1/eta and
|grad V| grows with the number of assets being liquidated, so a fixed amount of network error costs
more at higher d.

But 120 epochs is not a convergence test, so this runs one dimension far enough to answer the only
question that matters for planning: does the residual DESCEND, or is it stuck?

  * descends toward the exact floor  -> a full benchmark is worth its compute, and the paper gains
    the one baseline it is missing;
  * stuck orders of magnitude above  -> deep BSDE needs stability work FIRST, and the honest journal
    statement is about the regime where it is usable rather than a benchmark table.

Either answer is publishable; they differ in what has to be built next, which is why the question is
asked before any benchmark is launched.

Reported per checkpoint: the terminal residual in ABSOLUTE units against the exact floor at the same
dt, and the value error at x0. The value error is included because the d = 1 study showed the two
come apart -- the residual keeps falling while the value degrades -- so a residual that descends is
not by itself evidence that the solver is becoming useful.

Usage: python src/exp_bsde_dims.py --dims 1,4,8 --epochs 6000 [--with-x0-init]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np          # noqa: E402
import torch                # noqa: E402
import torch.nn as nn       # noqa: E402

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402

OUT = Path("results")


class DimDeepBSDE(DeepBSDEMonitor):
    """DeepBSDEMonitor plus the inputs the higher-dimensional problem needs.

    Two departures from the d = 1 configuration, both taken because the d = 1 one was tuned on a
    problem that does not have them:

    1. A TIME-INDEXED Z NETWORK. At d = 1 the input (x/tau, t) worked with a modest MLP. As d grows
       the state part of the input grows while the time part stays one coordinate, so the network
       can ignore time and fit a stationary map -- which cannot represent P(t) x for the Riccati
       P(t) that the reference actually has. The Fourier time features give the network a basis in
       which a smooth time dependence is cheap to represent.
    2. A LEARNABLE VALUE AT x0. theta was initialised at zero at d = 1 and the drift did the rest.
       At larger d the running cost is larger, so the value is larger and starting theta at zero
       means the very first terminal mismatches are dominated by a constant the network must first
       discover. Initialising theta at the EXACT value isolates the network's own error from that
       transient -- it is not information the deployed solver would have, so runs with and without
       it are both reported.
    """

    def __init__(self, *a, with_x0_init: bool = False, time_features: bool = True, **kw):
        super().__init__(*a, **kw)
        self.time_features = time_features
        if with_x0_init:
            ref = self.problem.reference(np.linspace(0.0, self.T, 401))
            with torch.no_grad():
                self.theta.fill_(float(np.asarray(ref.value(self.x0, 0.0)).ravel()[0])
                                 / self.V_scale)
        if time_features:
            # Rebuild the z network with time features appended, preserving the zeroed output layer
            # that the anchor relies on.
            hidden = 64
            layers = 4
            mods: list[nn.Module] = [nn.Linear(self.d + 1 + 4, hidden), nn.Tanh()]
            for _ in range(layers - 1):
                mods += [nn.Linear(hidden, hidden), nn.Tanh()]
            mods.append(nn.Linear(hidden, self.d))
            self.net = nn.Sequential(*mods).to(self.device)
            last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                last.weight.zero_()
                last.bias.zero_()

    def _grad_net(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if not self.time_features:
            return super()._grad_net(x, t)
        tau = t / self.T * 2.0 - 1.0
        feats = [x / 5.0, tau]
        for k in (1.0, 2.0):
            feats.append(torch.sin(math.pi * k * tau))
            feats.append(torch.cos(math.pi * k * tau))
        inp = torch.cat(feats, dim=1)
        z = self._z_terminal(x) + self.net(inp)
        return z @ self.Sigma_inv_T


def run_dim(d: int, args, device: str) -> dict:
    prob = ACProblem(p=d)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])

    s = DimDeepBSDE(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths, epochs=args.epochs,
                    device=device, seed=0, log_every=0, eval_paths=args.eval_paths,
                    with_x0_init=args.with_x0_init, time_features=not args.no_time_features)
    chk = s.validate_on_exact()
    print(f"\n  d={d}  V_true={V_true:+.6e}  exact floor {chk:.4e}  "
          f"theta0={float(s.theta.detach()) * s.V_scale:+.4e}  "
          f"({'x0-initialised' if args.with_x0_init else 'theta from zero'})")
    if not math.isfinite(chk):
        return {"d": d, "validated": False}

    ckpts = [e for e in (100, 250, 500, 1000, 2000, 4000, 6000, 8000, 12000, 16000, 20000)
             if e <= args.epochs]
    if args.epochs not in ckpts:
        ckpts.append(args.epochs)
    ckpts = sorted(set(ckpts))

    opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
    t0 = time.perf_counter()
    trace = []
    done = 0
    print(f"    {'epoch':>7} {'floor(abs)':>12} {'exact':>11} {'ratio':>9} {'V rel err':>10} "
          f"{'grad rel err':>13}")
    for target in ckpts:
        while done < target:
            s.dW = s._paths(args.n_paths, 1_000_000 + done)
            opt.zero_grad(set_to_none=True)
            xT, VT = s._rollout(s.theta, s._grad_net)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            loss = ((VT - g) / s.V_scale).pow(2).mean()
            loss.backward()
            opt.step()
            done += 1
        V_hat, g_hat = s.value_grad_at_x0()
        rmse = s.terminal_rmse()
        rec = {"epoch": done, "floor_abs": rmse, "exact_floor": chk, "ratio": rmse / chk,
               "V_hat": V_hat, "V_rel_err": abs(V_hat - V_true) / abs(V_true),
               "seconds": time.perf_counter() - t0}
        trace.append(rec)
        flag = "" if math.isfinite(rmse) else "   <-- NON-FINITE"
        print(f"    {done:>7} {rmse:>12.4e} {chk:>11.3e} {rmse / chk:>9.2f} "
              f"{rec['V_rel_err']:>10.3e} {'':>13}{flag}", flush=True)
        if not math.isfinite(rmse):
            break
    secs = time.perf_counter() - t0
    first, last = trace[0], trace[-1]
    return {"d": d, "V_true": V_true, "exact_floor": chk, "epochs": args.epochs,
            "sec_per_epoch": secs / max(done, 1), "theta0": None,
            "floor_first": first["floor_abs"], "floor_last": last["floor_abs"],
            "ratio_first": first["ratio"], "ratio_last": last["ratio"],
            "descent_factor": first["floor_abs"] / last["floor_abs"]
            if last["floor_abs"] > 0 else float("inf"),
            "best_V_rel_err": min(t["V_rel_err"] for t in trace),
            "best_V_epoch": min(trace, key=lambda t: t["V_rel_err"])["epoch"],
            "V_rel_err_last": last["V_rel_err"], "trace": trace}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,4,8")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--eval-paths", type=int, default=4096)
    ap.add_argument("--with-x0-init", action="store_true",
                    help="initialise theta at the exact V(x0,0) to isolate network error")
    ap.add_argument("--no-time-features", action="store_true",
                    help="use the d=1 network unchanged, for a like-for-like comparison")
    ap.add_argument("--out", type=str, default="exp_bsde_dims.json")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 104)
    print("DEEP BSDE AT HIGHER DIMENSION: does the residual descend?")
    print("=" * 104)
    print(f"  device={device}  epochs={args.epochs}  n_steps={args.n_steps}  "
          f"n_paths={args.n_paths}  x0_init={args.with_x0_init}  "
          f"time_features={not args.no_time_features}")

    rows = []
    for d in [int(v) for v in args.dims.split(",")]:
        rows.append(run_dim(d, args, device))
        (OUT / args.out).write_text(json.dumps({"rows": rows, "args": vars(args)}, indent=2),
                                    encoding="utf-8")

    print()
    print("=" * 104)
    print("SUMMARY")
    print("=" * 104)
    print(f"  {'d':>3} {'floor first':>13} {'floor last':>12} {'descent':>9} {'ratio last':>11} "
          f"{'best V err':>11} {'at epoch':>9} {'V err end':>10} {'s/epoch':>8}")
    for r in rows:
        if not r.get("exact_floor"):
            print(f"  {r['d']:>3}   driver non-finite -- needs stability work before benchmarking")
            continue
        print(f"  {r['d']:>3} {r['floor_first']:>13.3e} {r['floor_last']:>12.3e} "
              f"{r['descent_factor']:>8.1f}x {r['ratio_last']:>11.1f} {r['best_V_rel_err']:>11.3e} "
              f"{r['best_V_epoch']:>9} {r['V_rel_err_last']:>10.3e} {r['sec_per_epoch']:>8.4f}")

    print()
    print("-" * 104)
    for r in rows:
        if not r.get("exact_floor"):
            continue
        descends = r["descent_factor"] > 10
        reached = r["ratio_last"] < 10
        if reached:
            print(f"  d={r['d']}: reached within {r['ratio_last']:.1f}x of the exact floor. A "
                  f"benchmark here is worth its compute.")
        elif descends:
            print(f"  d={r['d']}: still descending ({r['descent_factor']:.0f}x over the budget), "
                  f"last ratio {r['ratio_last']:.0f}. Undertrained rather than stuck -- more epochs "
                  f"would tell, and the d=1 study shows this curve keeps falling for a long time.")
        else:
            print(f"  d={r['d']}: STUCK at {r['ratio_last']:.0f}x the exact floor. Stability work "
                  f"comes before any benchmark number.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
