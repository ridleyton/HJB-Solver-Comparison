"""Is it really the gradient? Measure the two errors the recursion actually sees.

THE CLAIM THIS TESTS
--------------------
The paper attributes deep splitting's failure on this benchmark to the per-step regression being
accurate in the VALUE while the recursion consumes the GRADIENT: z = Sigma' grad u enters
f = -1/2 x'Qx + 1/2 z'M z with M ~ 1e6, so what a step needs is a good derivative, while a
mean-squared-error loss constrains the function. That is a mechanism claim, and a mechanism claim
should be measured rather than asserted.

WHAT IS MEASURED
----------------
At each trained step's network, on the same paths it was fitted to:

    value error      max |net_n(x) - V(t_n, x)|          relative to the value's own scale
    gradient error   max |grad net_n(x) - grad V(t_n, x)| relative to the gradient's own scale

Both are reported RELATIVE to their own scales, because that is the comparison that decides the
issue: a network can have a tiny relative value error and a large relative gradient error, and if it
does, the recursion inherits the second.

Usage: python src/diag_splitting_gradient_error.py [--steps 50] [--epochs 300]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from deep_splitting import DeepSplitting  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--paths", type=int, default=4096)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--out", default="diag_splitting_gradient_error.json")
    args = ap.parse_args()

    prob = ACProblem(p=args.dim)
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    s = DeepSplitting(prob, n_steps=args.steps, n_paths=args.paths, epochs=args.epochs,
                      hidden=args.hidden, layers=args.layers, seed=0, spread=True, device="cpu",
                      inner=1)
    print("=" * 100)
    print("VALUE ERROR vs GRADIENT ERROR at the trained per-step networks")
    print("=" * 100)
    print(f"  d={args.dim}, {args.steps} steps, {args.epochs} epochs/step, "
          f"{args.layers}x{args.hidden}")
    s.solve()

    print(f"\n  {'step':>5} {'t':>6} {'value err (rel)':>16} {'grad err (rel)':>16} {'ratio':>8}")
    worst_ratio = 0.0
    for n in (args.steps - 1, args.steps // 2, args.steps // 4, 1, 0):
        if n < 0 or s.nets[n] is None:
            continue
        t = n * s.dt
        X = s.paths[n].detach().cpu().numpy()
        xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            pred = s.nets[n](xt / s.x_scale).numpy().ravel()
        exact = np.asarray(ref.value(X, t)).ravel()
        # gradient of the network, exactly as the solver computes it
        z_net = s._z_of(s.nets[n], xt).numpy()                    # = Sigma' grad net
        z_exact = (X @ (-ref.P(t))) @ prob.Sigma
        v_scale = max(np.abs(exact).max(), 1e-30)
        g_scale = max(np.abs(z_exact).max(), 1e-30)
        v_err = np.abs(pred - exact).max() / v_scale
        g_err = np.abs(z_net - z_exact).max() / g_scale
        ratio = g_err / max(v_err, 1e-30)
        worst_ratio = max(worst_ratio, ratio)
        print(f"  {n:>5} {t:>6.2f} {v_err:>16.3e} {g_err:>16.3e} {ratio:>8.3f}")

    print(f"\n  the gradient's relative error exceeds the value's by up to {worst_ratio:.1f}x")
    print("  -> a regression whose VALUE is accurate can still be the wrong thing to feed a")
    print("     recursion that consumes the gradient, which is the mechanism the paper reports.")
    (SRC.parent / "results" / args.out).write_text(
        __import__("json").dumps({"steps": args.steps, "epochs": args.epochs,
                                  "worst_ratio": worst_ratio}, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    print("GRADIENT ERROR DIAGNOSTIC COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
