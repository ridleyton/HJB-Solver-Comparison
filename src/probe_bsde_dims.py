"""
SMOKE TEST: does the validated deep-BSDE solver survive d > 1 at all?

Why this comes before any benchmark
-----------------------------------
The solver was validated and tuned entirely at d = 1 (see the deep-bsde-steps-verdict artifact).
Journal credibility is the reason this matters: every review of a neural-versus-classical HJB
comparison asks about deep BSDE, because it -- not the PINN -- is the standard high-dimensional
method in mathematical finance. The conference manuscript cites it once and does not run it.

But a benchmark number is worthless if the solver diverges at the dimension where it is supposed to
matter, and the whole reason this solver needed a terminal-gradient anchor is that its forward
recursion is explosively unstable when the network's gradient estimate is wrong. That instability
gets worse, not better, as d rises: the substituted optimal control feeds back through
B R^-1 B' with gain 1/eta, and |grad V| grows with the number of assets being liquidated.

So this is a TRIPWIRE, not a result. It trains for very few epochs at d = 1, 2, 4, 8 and reports
whether the rollout stays finite and whether the whole thing is stable, so that a full benchmark is
launched only on dimensions where it has a chance. It deliberately does not report accuracy: a
hundred epochs says nothing about accuracy, and presenting it as though it did would be the exact
error this project has already made four times.

Usage: python src/probe_bsde_dims.py [--epochs 150] [--dims 1,2,4,8]
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

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                  # noqa: E402
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402

OUT = Path("results")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", type=str, default="1,2,4,8")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--n-paths", type=int, default=1024)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 104)
    print("TRIPWIRE: is the deep-BSDE rollout finite at d > 1? (NOT an accuracy measurement)")
    print("=" * 104)
    print(f"  device={device}  epochs={args.epochs}  n_steps={args.n_steps}  "
          f"n_paths={args.n_paths}")
    print("  A hundred-odd epochs cannot establish accuracy. This only asks whether the recursion")
    print("  stays finite and whether the terminal residual is moving in the right direction.")
    print()
    print(f"  {'d':>3} {'n_assets':>8} {'state_dim':>9} {'V_true':>13} {'floor_exact':>12} "
          f"{'floor@100ep':>12} {'floor@end':>12} {'finite':>7} {'s/epoch':>8}")

    rows = []
    for d in [int(v) for v in args.dims.split(",")]:
        prob = ACProblem(p=d)
        x0 = np.zeros(prob.state_dim)
        x0[: prob.n_assets] = 1.0
        ref = prob.reference(np.linspace(0.0, prob.T, 401))
        V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])

        s = DeepBSDEMonitor(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths,
                            epochs=args.epochs, device=device, seed=0, log_every=0,
                            eval_paths=2048)
        chk = s.validate_on_exact()
        finite = math.isfinite(chk)
        if not finite:
            print(f"  {d:>3} {prob.n_assets:>8} {prob.state_dim:>9} {V_true:>13.4e} "
                  f"{'--':>12} {'--':>12} {'--':>12} {'NO':>7} {'--':>8}")
            rows.append({"d": d, "n_assets": prob.n_assets, "state_dim": prob.state_dim,
                         "V_true": V_true, "driver_finite": False})
            continue

        opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)
        t0 = time.perf_counter()
        marks = {}
        for ep in range(args.epochs):
            s.dW = s._paths(args.n_paths, 1_000_000 + ep)
            opt.zero_grad(set_to_none=True)
            xT, VT = s._rollout(s.theta, s._grad_net)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            loss = ((VT - g) / s.V_scale).pow(2).mean()
            loss.backward()
            opt.step()
            if ep + 1 in (100, args.epochs):
                marks[ep + 1] = float(loss.sqrt())
                if not math.isfinite(marks[ep + 1]):
                    break
        secs = time.perf_counter() - t0
        m100 = marks.get(100, float("nan"))
        mend = marks.get(args.epochs, float("nan"))
        ok = all(math.isfinite(v) for v in marks.values())
        print(f"  {d:>3} {prob.n_assets:>8} {prob.state_dim:>9} {V_true:>13.4e} "
              f"{chk:>12.4e} {m100:>12.4e} {mend:>12.4e} {str(ok):>7} {secs / args.epochs:>8.4f}")
        rows.append({"d": d, "n_assets": prob.n_assets, "state_dim": prob.state_dim,
                     "V_true": V_true, "driver_finite": True, "driver_floor": chk,
                     "loss_at_100": m100, "loss_at_end": mend, "finite": ok,
                     "sec_per_epoch": secs / args.epochs})

    (OUT / "probe_bsde_dims.json").write_text(json.dumps(
        {"rows": rows, "epochs": args.epochs, "n_steps": args.n_steps,
         "n_paths": args.n_paths, "device": device}, indent=2), encoding="utf-8")

    print()
    print("-" * 104)
    good = [r for r in rows if r.get("finite")]
    print(f"  finite at: {[r['d'] for r in good]}")
    bad = [r for r in rows if not r.get("finite")]
    if bad:
        print(f"  DIVERGED at: {[r['d'] for r in bad]} -- a benchmark there needs the stability work")
        print("  done first, not a longer run.")
    if good:
        worst = max(good, key=lambda r: r["sec_per_epoch"])
        print(f"  costliest finite dimension: d={worst['d']} at "
              f"{worst['sec_per_epoch']:.4f} s/epoch")
    print(f"\nwrote {OUT / 'probe_bsde_dims.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
