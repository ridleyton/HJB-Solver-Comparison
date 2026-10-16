"""Deep splitting: is the neural arm's failure a budget or capacity problem?

WHY THIS EXISTS
---------------
`exp_deep_splitting.py` shows that the SAME recursion succeeds with an exact quadratic regression
(value R^2 0.9987) and fails with a neural one (0.16) at matched step count and matched paths. That
attributes the failure to the learner rather than to the scheme. It does NOT say whether the learner
is failing for a reason that more compute would fix.

This sweep answers that, and the answer is the one the paper's mechanism predicts. From 50 to 200
steps, with the network widened from 2x32 to 4x64 and the per-step budget raised from 300 to 2000
epochs (about 7x the gradient steps), the median per-step loss improves by an order of magnitude --
and the value R^2 stays NEGATIVE at every configuration, while the control R^2 stays near 0.85.

That pattern is what a value-fitting loss implies when the recursion consumes the gradient: an
order of magnitude better fit to the VALUE does not deliver an order of magnitude better GRADIENT, and
it is the gradient that enters f = -1/2 x'Qx + 1/2 z'Mz with M ~ 1e6. It is the same value/control
decoupling the paper reports for the grid, appearing from the other side.

Usage: python src/exp_deep_splitting_budget.py [--steps 50,100] [--epochs 2000]
Output: results/exp_deep_splitting_budget.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from deep_splitting import DeepSplitting, evaluate  # noqa: E402

RESULTS = SRC.parent / "results"
CONFIGS = [(50, 300, 32, 2), (50, 2000, 64, 4), (100, 2000, 64, 4)]


def summarise(rows: list[dict]) -> dict:
    """Derived quantities, computed from the rows so they cannot drift from them.

    The comparison that matters is WITHIN a fixed step count: at 50 steps, does a bigger per-step
    budget repair the value? Comparing across step counts as well would conflate the two effects.
    """
    same = [r for r in rows if r["n_steps"] == 50] or rows
    small = min(same, key=lambda r: r["gradient_steps"])
    large = max(same, key=lambda r: r["gradient_steps"])
    return {
        "loss_improvement": small["median_step_loss"] / large["median_step_loss"],
        "budget_ratio": large["gradient_steps"] / small["gradient_steps"],
        "value_r2_small_budget": small["value_r2"],
        "value_r2_large_budget": large["value_r2"],
        "value_r2_negative_at_large_budget": large["value_r2"] < 0,
        "control_r2_range": [min(r["control_r2"] for r in rows),
                             max(r["control_r2"] for r in rows)],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dim", type=int, default=1)
    ap.add_argument("--paths", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resummarise", action="store_true",
                    help="recompute only the derived summary from the stored rows (the sweep itself "
                         "takes 24 minutes; the summary is derived data and must still come from "
                         "code rather than from an editor)")
    ap.add_argument("--out", default="exp_deep_splitting_budget.json")
    args = ap.parse_args()

    if args.resummarise:
        path = RESULTS / args.out
        data = json.loads(path.read_text(encoding="utf-8"))
        data["summary"] = summarise(data["rows"])
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"  rewrote the summary in results/{args.out}:")
        print(f"  {json.dumps(data['summary'], indent=2)}")
        return 0

    prob = ACProblem(p=args.dim)
    print("=" * 100)
    print("DEEP SPLITTING: does more budget or capacity repair the neural arm?")
    print("=" * 100)
    print(f"  d={args.dim}, {args.paths} paths, one step's worth of budget in the first column")
    print(f"\n  {'steps':>6} {'epochs':>7} {'arch':>7} {'secs':>8} {'value R2':>10} {'control R2':>11} "
          f"{'median loss':>12} {'rel err x0':>11}")
    out = {"dim": args.dim, "paths": args.paths, "seed": args.seed, "rows": []}
    for steps, epochs, hidden, layers in CONFIGS:
        solver = DeepSplitting(prob, n_steps=steps, n_paths=args.paths, epochs=epochs,
                               hidden=hidden, layers=layers, seed=args.seed, spread=True,
                               device="cpu", inner=1)
        t0 = time.perf_counter()
        solver.solve()
        secs = time.perf_counter() - t0
        sc = evaluate(prob, solver)
        row = {"n_steps": steps, "epochs_per_step": epochs, "hidden": hidden, "layers": layers,
               "gradient_steps": steps * epochs, "seconds": secs,
               "value_r2": sc["value_r2"], "control_r2": sc["control_r2"],
               "rel_err_at_x0": sc["rel_err_at_x0"],
               "median_step_loss": float(np.median(solver.step_losses))}
        out["rows"].append(row)
        print(f"  {steps:>6} {epochs:>7} {layers}x{hidden:<4} {secs:>8.1f} {sc['value_r2']:>+10.4f} "
              f"{sc['control_r2']:>+11.4f} {row['median_step_loss']:>12.2e} "
              f"{sc['rel_err_at_x0']*100:>10.3f}%", flush=True)
    out["summary"] = summarise(out["rows"])
    s = out["summary"]
    print(f"\n  at 50 steps: median loss better by {s['loss_improvement']:.0f}x at "
          f"{s['budget_ratio']:.1f}x the budget,")
    print(f"  and the value R2 moves from {s['value_r2_small_budget']:+.4f} to "
          f"{s['value_r2_large_budget']:+.4f}")
    print(f"  control R2 over all configurations: {s['control_r2_range'][0]:+.3f} "
          f"to {s['control_r2_range'][1]:+.3f}")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    print("DEEP SPLITTING BUDGET SWEEP COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
