"""
P2: architecture sensitivity. Does the boundary move when the neural solver is given a better one?

Why this is the ablation that matters
-------------------------------------
The paper reports its neural results at a single architecture -- a 4-layer, 64-wide tanh MLP -- and
says so in its own limitations. A referee reads that as the load-bearing weakness of any benchmark:
the boundary might sit where it does because the network was small, not because the solver family is
limited. The paper already answers the neighbouring question (a larger TRAINING BUDGET rescues p=4,
reported honestly rather than hidden), but budget and capacity are different axes, and only the first
was tested.

THE DESIGN, AND WHY IT IS NOT SIMPLY "MORE PARAMETERS". Two effects could be at work and they call
for different readings, so the sweep separates them:

  * width and depth increase capacity at a fixed budget -- if the boundary moves, capacity was the
    binding constraint;
  * width and depth increase cost -- so a fair comparison reports accuracy AND the parameter count
    and solve time alongside it, because a better score bought with 10x the compute is not evidence
    that the family is fine.

The sweep is run at p = 4, which is the dimension where the paper's fixed-budget PINN collapses
(value R^2 = -57) and where the larger-budget run recovers it (+0.9972). That is the point on the
boundary where an architecture effect would be most visible; a dimension where the solver already
succeeds would tell us nothing.

WHAT THIS DOES NOT SETTLE, stated up front. A sweep over MLPs does not exclude other families --
operator architectures, spectral features, curriculum or adaptive schemes. If capacity does not move
the boundary here, the honest conclusion is "not with this family or this budget", not "not at all".

Usage: python src/exp_arch_ablation.py --p 4 --epochs 6000,20000 --seeds 0,1
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem                    # noqa: E402
from pinn import PINNSolver                         # noqa: E402
from solvers import FiniteDifferenceSolver, accuracy_metrics   # noqa: E402

OUT = SRC.parent / "results"

# (hidden, layers, label). The baseline is the paper's 4x64; the others move one axis at a time and
# then both, so a capacity effect can be attributed rather than merely observed.
ARCHS = [
    (64, 4, "4x64 (paper baseline)"),
    (64, 2, "2x64 (shallower)"),
    (64, 8, "8x64 (deeper)"),
    (128, 4, "4x128 (wider)"),
    (256, 6, "6x256 (both)"),
]


def count_params(hidden: int, layers: int, in_dim: int) -> int:
    """Parameter count of the MLP the solver builds, for the cost column."""
    p = in_dim * hidden + hidden
    for _ in range(layers - 1):
        p += hidden * hidden + hidden
    p += hidden + 1
    return int(p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=4)
    ap.add_argument("--epochs", type=str, default="6000,20000")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--out", type=str, default="exp_arch_ablation.json")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    device = "cuda"
    prob = ACProblem(p=args.p)
    epochs_list = [int(s) for s in args.epochs.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]

    print("=" * 112)
    print(f"ARCHITECTURE SENSITIVITY at p = {args.p} (the dimension where the fixed budget collapses)")
    print("=" * 112)
    print("  Every cell is a full PINN solve scored against the analytic reference. The grid row is")
    print("  the accuracy-matched classical scheme at the same dimension, for reference.")
    print()

    # Grid reference, so the reader can see how far the neural family has to travel.
    grid_n = {1: 41, 2: 41, 3: 31, 4: 21, 5: 13}.get(args.p, 21)
    gs = FiniteDifferenceSolver(prob, n_grid=grid_n)
    gt = gs.solve()
    gm = accuracy_metrics(gs, prob, n_points=600, n_times=6, seed=11)
    print(f"  grid (accuracy-matched, N={grid_n}): value R2={gm['value_r2']:+.5f}  "
          f"ctrl R2={gm['control_r2']:+.5f}  {gt:.2f}s")
    print()

    rows: list[dict] = []
    print(f"  {'architecture':>22} {'epochs':>7} {'params':>9} {'time (s)':>9} "
          f"{'value R2':>12} {'ctrl R2':>11}")
    for epochs in epochs_list:
        for hidden, layers, label in ARCHS:
            r2s, cr2s, times = [], [], []
            for seed in seeds:
                try:
                    s = PINNSolver(prob, hidden=hidden, layers=layers, n_collocation=2048,
                                   epochs=epochs, lr=2e-3, device=device, seed=seed)
                    t = s.solve()
                    m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
                except Exception as exc:                        # noqa: BLE001
                    print(f"  {label:>22} {epochs:>7}  FAILED: {type(exc).__name__}: {exc}")
                    continue
                r2s.append(m["value_r2"]); cr2s.append(m["control_r2"]); times.append(t)
            if not r2s:
                continue
            in_dim = prob.state_dim + 1
            npar = count_params(hidden, layers, in_dim)
            rows.append({"p": args.p, "hidden": hidden, "layers": layers, "label": label,
                         "epochs": epochs, "params": npar,
                         "time_s": statistics.mean(times),
                         "value_r2_mean": statistics.mean(r2s),
                         "value_r2_sd": statistics.pstdev(r2s) if len(r2s) > 1 else 0.0,
                         "control_r2_mean": statistics.mean(cr2s),
                         "n_seeds": len(r2s)})
            print(f"  {label:>22} {epochs:>7} {npar:>9,} {statistics.mean(times):>9.1f} "
                  f"{statistics.mean(r2s):>+12.5f} {statistics.mean(cr2s):>+11.4f}")

    (OUT / args.out).write_text(json.dumps(
        {"rows": rows, "p": args.p, "grid": {"n_grid": grid_n, "time_s": gt,
                                             "value_r2": gm["value_r2"],
                                             "control_r2": gm["control_r2"]},
         "epochs": epochs_list, "seeds": seeds}, indent=2), encoding="utf-8")

    # ---------------- verdict ----------------
    print()
    print("-" * 112)
    if rows:
        best = max(rows, key=lambda r: r["value_r2_mean"])
        base = [r for r in rows if r["epochs"] == epochs_list[0] and "baseline" in r["label"]]
        print(f"  best configuration: {best['label']} at {best['epochs']} epochs, "
              f"value R2 = {best['value_r2_mean']:+.5f}, {best['params']:,} parameters, "
              f"{best['time_s']:.0f}s")
        if base:
            b = base[0]
            print(f"  paper baseline at the same budget ({b['epochs']} epochs): "
                  f"value R2 = {b['value_r2_mean']:+.5f}")
            if best["value_r2_mean"] > 0.99 and b["value_r2_mean"] < 0.9:
                print()
                print("  CAPACITY MOVES THE BOUNDARY. An architecture within the same MLP family does")
                print("  rescue this dimension at the fixed budget, so the paper's collapse at p=4 is")
                print("  a statement about the architecture it used, not about the budget alone. The")
                print("  cost column has to be read with it: a rescue bought with more parameters and")
                print("  more seconds is a different operating point, not a refutation.")
            elif best["value_r2_mean"] > 0.99:
                print()
                print("  The best architecture succeeds and the baseline also succeeds at this budget,")
                print("  so the boundary at this dimension is not architecture-limited.")
            else:
                print()
                print("  CAPACITY DOES NOT RESCUE THIS DIMENSION within the MLP family: no")
                print("  configuration in the sweep reaches value R2 > 0.99, across a")
                print(f"  {best['params'] / max(b['params'], 1):.1f}x range of parameter counts. The honest")
                print("  statement is therefore about this family and budget, not about neural solvers")
                print("  in general -- an operator or spectral architecture is not excluded here.")
    print(f"\nwrote {OUT / args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
