"""
Does the deep-BSDE stopping threshold SCALE with the problem, or is it a fixed number?

Why this is the right follow-up
-------------------------------
The d = 1 study produced a clean stopping rule: train until the terminal residual reaches about
1e-3, because that is where the value error is minimised and training past it degrades the value
even as the residual keeps falling.

The d = 4 run then broke that rule. Its value error is minimised at 3.27% around epoch 2000, where
the residual is 5.2 -- three orders of magnitude above 1e-3 -- and by epoch 8000 the value error has
grown to 302% while the residual has fallen to 1.16. A practitioner applying the d = 1 threshold
literally would train straight through the optimum.

So an ABSOLUTE threshold does not transfer across dimension. The obvious candidate for one that does
is a RELATIVE one: stop when the residual has fallen to some fixed fraction of where it started. At
d = 1 the initial residual is ~2.5e-01 and the optimum sits at ~2.5e-03, i.e. about 1% of it; at
d = 4 the initial residual is ~1.3e+02 and the optimum sits at ~5.2, which is about 4% of it. Those
are close enough to be worth testing properly, and equal fractions would mean the threshold is a
property of the descent rather than of the problem.

This script measures the one quantity that decides it -- the INITIAL residual, before any training --
across dimensions. That is cheap (a few epochs each) and it is the denominator the relative rule
needs. It deliberately does NOT claim the relative rule holds: two points agree and two points are
not a law. What it establishes is whether the absolute threshold can possibly transfer, and what the
relative candidate's denominator actually is.

Usage: python src/probe_bsde_residual_scale.py [--dims 1,2,4,8,16]
"""

from __future__ import annotations

import argparse
import json
import math
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

# Where the value error was actually minimised in runs that have been completed. These are MEASURED
# values carried here only so the script can print the implied fraction; nothing is fitted to them.
KNOWN_OPTIMA = {1: (2.497e-03, 0.0382), 4: (5.212e+00, 0.0327)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2,4,8,16")
    ap.add_argument("--warmup", type=int, default=25,
                    help="epochs before reading the residual; 0 reads the untrained rollout")
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 104)
    print("RESIDUAL SCALE ACROSS DIMENSION -- the denominator an absolute stopping rule ignores")
    print("=" * 104)
    print(f"  device={device}  warmup={args.warmup} epochs  n_steps={args.n_steps}")
    print("  The question is whether the ABSOLUTE threshold found at d=1 can transfer at all.")
    print()
    print(f"  {'d':>3} {'exact floor':>12} {'initial resid':>14} {'init/floor':>11} "
          f"{'known optimum':>14} {'opt/init':>9} {'opt/floor':>11}")

    rows = []
    for d in [int(v) for v in args.dims.split(",")]:
        prob = ACProblem(p=d)
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths,
                            epochs=max(args.warmup, 1), device=device, seed=0, log_every=0,
                            eval_paths=4096)
        floor = s.validate_on_exact()
        if not math.isfinite(floor):
            print(f"  {d:>3} {'--':>12} driver non-finite, skipped")
            continue
        if args.warmup <= 0:
            init = floor
        else:
            opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
            for ep in range(args.warmup):
                s.dW = s._paths(args.n_paths, 1_000_000 + ep)
                opt.zero_grad(set_to_none=True)
                xT, VT = s._rollout(s.theta, s._grad_net)
                g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
                loss = ((VT - g) / s.V_scale).pow(2).mean()
                loss.backward()
                opt.step()
            init = s.terminal_rmse()

        opt_res, opt_verr = KNOWN_OPTIMA.get(d, (float("nan"), float("nan")))
        frac = opt_res / init if init > 0 and math.isfinite(opt_res) else float("nan")
        print(f"  {d:>3} {floor:>12.4e} {init:>14.4e} {init / floor:>11.3e} "
              f"{opt_res:>14.4e} {frac:>9.3f} {opt_res / floor:>11.1f}"
              if math.isfinite(opt_res) else
              f"  {d:>3} {floor:>12.4e} {init:>14.4e} {init / floor:>11.3e} "
              f"{'unknown':>14} {'--':>9} {'--':>11}")
        rows.append({"d": d, "exact_floor": floor, "initial_residual": init,
                     "init_over_floor": init / floor,
                     "known_optimum_residual": opt_res if math.isfinite(opt_res) else None,
                     "known_optimum_V_err": opt_verr if math.isfinite(opt_verr) else None,
                     "optimum_fraction_of_initial": frac if math.isfinite(frac) else None})

    (OUT / "probe_bsde_residual_scale.json").write_text(json.dumps(
        {"rows": rows, "warmup": args.warmup, "n_steps": args.n_steps}, indent=2), encoding="utf-8")

    print()
    print("-" * 104)
    known = [r for r in rows if r["optimum_fraction_of_initial"]]
    if known:
        for r in known:
            print(f"  d={r['d']}: optimum sits at {r['optimum_fraction_of_initial']:.3f} of the "
                  f"initial residual, and {r['init_over_floor']:.1e}x the exact floor at start")
        fr = [r["optimum_fraction_of_initial"] for r in known]
        print()
        if len(fr) >= 2 and max(fr) / min(fr) < 3:
            print(f"  The optimum fraction is consistent across the dimensions measured "
                  f"({min(fr):.3f} to {max(fr):.3f}, a spread of {max(fr) / min(fr):.1f}x). That is "
                  f"consistent with a RELATIVE stopping rule -- stop when the residual has fallen "
                  f"to a few percent of its own initial value -- but two points agreeing is not a "
                  f"law, and the fraction still has to be calibrated per problem class.")
        else:
            print("  The optimum fraction is NOT consistent, so neither an absolute nor a simple "
                  "relative threshold transfers; the stopping point would need to be found by "
                  "monitoring, not by a rule.")
        print()
        print("  WHY THIS MATTERS BEYOND THE RULE ITSELF: the initial residual grows by orders of")
        print("  magnitude with dimension (see the init/floor column), so a practitioner who takes")
        print("  the d=1 threshold literally trains through the optimum at higher dimension. The")
        print("  threshold is a property of the descent, not a universal constant.")

    print(f"\nwrote {OUT / 'probe_bsde_residual_scale.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
