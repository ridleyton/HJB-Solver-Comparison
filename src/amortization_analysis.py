"""
Two different questions that the phrase "is the neural solver faster?" conflates.

The paper's timing table answers Q1. The neural-operator literature answers Q2, and its
inference-time framing is legitimate precisely because Q2 counts instances, not queries. This
script computes both, from the measured numbers, so the paper can state which regime each
conclusion belongs to instead of leaving the ambiguity to the reader.

Q1 -- ONE problem instance, many queries.
    The equation is solved once and queried at many states.
        grid  : T_solve + N * t_grid_query
        PINN  : T_train + N * t_pinn_query
    N* = (T_train - T_solve) / (t_grid_query - t_pinn_query)
    Measured: p=1 never breaks even; p=3 at 1.06e8 queries (~272,000 trading days).

Q2 -- MANY problem instances, one query each.
    A *family* of M related problems -- different risk aversions, horizons, volatilities, or
    assets. A classical solver must re-solve for every member; an operator maps the parameters
    to the solution and pays its cost once.
        grid     : M * T_solve
        operator : M_data * T_solve  +  T_train  +  M * t_infer
    M* = (M_data * T_solve + T_train) / (T_solve - t_infer)
    Note the data-generation term: an operator is usually trained on solutions produced by the
    very solver it is meant to replace, so those solves are part of its cost and are charged
    here. This is the accounting the FNO paper makes explicitly.

The second number is the one that decides whether operator learning is worth pursuing for the
HJB, and it is not the number the paper currently reports.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

RESULTS = Path("results")
FIELD_POINTS = 9261          # a 21^3 solution field at the accuracy-matched resolution
M_DATA_DEFAULT = 1000        # training-set size for an operator, in classical solves


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def main() -> None:
    grid = load("exp_grid_refinement_merged.json")           # T_solve per dimension
    dim = load("exp_dimension.json")                          # T_train per dimension
    budget = load("exp_pinn_budget.json")                     # rescued p=4 budget
    timing = load("exp_eval_timing.json")                     # per-query latencies

    # ---- assemble the per-dimension inputs, all measured ------------------------------
    solve = {}
    for k, v in (grid or {}).get("summary", {}).items():
        if v.get("time_s"):
            solve[int(k)] = v["time_s"]

    train = {}
    for r in (dim or {}).get("runs", []):
        if r.get("method") == "PINN" and r.get("feasible") and r.get("solve_time_s"):
            train.setdefault(r["p"], []).append(r["solve_time_s"])
    train = {p: sum(v) / len(v) for p, v in train.items()}
    # p=4 is only usable at the larger budget, so use that number instead of the failed one.
    for r in (budget or {}).get("runs", []):
        if r.get("p") == 4 and r.get("epochs") == 20000 and r.get("solve_time_s"):
            train[4] = r["solve_time_s"]

    q_grid, q_pinn = {}, {}
    for r in (timing or {}).get("runs", []):
        if r.get("batch") == 390:
            if r["method"].startswith("Finite"):
                q_grid[r["p"]] = r["us_per_state_control"]
            elif r["method"] == "PINN" and r["device"] == "cuda":
                q_pinn[r["p"]] = r["us_per_state"]

    print("Measured inputs")
    print(f"  {'p':>2s} {'T_solve grid (s)':>17s} {'T_train PINN (s)':>17s} "
          f"{'grid query (us)':>16s} {'PINN query (us)':>16s}")
    for p in sorted(solve):
        print(f"  {p:>2d} {solve[p]:>17.3f} {train.get(p, float('nan')):>17.1f} "
              f"{q_grid.get(p, float('nan')):>16.2f} {q_pinn.get(p, float('nan')):>16.2f}")

    # ---- Q1: one instance, many queries ---------------------------------------------
    print("\n" + "=" * 92)
    print("Q1 -- ONE instance, many state queries:  N* = (T_train - T_solve) / (q_grid - q_pinn)")
    print("=" * 92)
    q1 = {}
    for p in sorted(solve):
        if p not in train or p not in q_grid or p not in q_pinn:
            continue
        cheaper_upfront = train[p] < solve[p]          # PINN trains faster than the grid solves
        cheaper_per_query = q_pinn[p] < q_grid[p]      # PINN answers faster than grid lookup
        if cheaper_upfront and cheaper_per_query:
            print(f"  p={p}: the neural solver wins on BOTH terms "
                  f"(train {train[p]:.0f}s < solve {solve[p]:.0f}s; "
                  f"query {q_pinn[p]:.2f} < {q_grid[p]:.2f} us) -> dominates immediately, "
                  f"if it is accurate")
            q1[p] = 1
        elif not cheaper_upfront and not cheaper_per_query:
            print(f"  p={p}: the grid wins on BOTH terms "
                  f"(solve {solve[p]:.3f}s < train {train[p]:.0f}s; "
                  f"query {q_grid[p]:.2f} < {q_pinn[p]:.2f} us) -> never breaks even")
            q1[p] = None
        else:
            n = abs(train[p] - solve[p]) / abs(q_grid[p] - q_pinn[p]) * 1e6
            who = "neural" if cheaper_per_query else "grid"
            print(f"  p={p}: crossover at N* = {n:,.0f} queries "
                  f"({n/390:,.0f} trading days at 390/day); {who} solver better per query")
            q1[p] = n

    # ---- Q2: many instances, one query each -----------------------------------------
    print("\n" + "=" * 92)
    print("Q2 -- MANY instances, one query each:")
    print("      M* = (M_data * T_solve + T_train) / (T_solve - t_infer)")
    print(f"      assumptions: M_data = {M_DATA_DEFAULT} training solves, "
          f"field = {FIELD_POINTS} points, inference = measured per-state latency")
    print("=" * 92)
    q2 = {}
    for p in sorted(solve):
        if p not in train:
            continue
        # Inference cost of producing one whole solution field.
        us = q_pinn.get(p) or q_grid.get(p)
        t_infer = us * 1e-6 * FIELD_POINTS
        data_cost = M_DATA_DEFAULT * solve[p]
        denom = solve[p] - t_infer
        if denom <= 0:
            print(f"  p={p}: T_solve ({solve[p]:.3f}s) <= t_infer ({t_infer:.3f}s) "
                  f"-> the operator is cheaper from the first instance")
            q2[p] = 1
            continue
        m = (data_cost + train[p]) / denom
        print(f"  p={p}: T_solve={solve[p]:>8.3f}s  t_infer={t_infer:>7.4f}s  "
              f"data={data_cost:>9.0f}s  train={train[p]:>6.0f}s  ->  M* = {m:,.0f} instances")
        q2[p] = m

    # ---- what the numbers say --------------------------------------------------------
    print("\n" + "=" * 92)
    print("Reading")
    print("=" * 92)
    print("  Q1 is the regime of a single HJB solved once: the grid scheme wins, and the")
    print("  neural solver needs absurd query counts to catch up.")
    print()
    print("  Q2 is the regime the neural-operator literature reports, where the classical")
    print("  baseline is re-solved per instance. Here the picture inverts: the break-even")
    print("  instance count is small, and it FALLS as p rises, because T_solve grows and")
    print("  t_infer does not.")
    print()
    print("  Both statements are true. Which one applies is a property of the deployment")
    print("  model -- one problem, or a family of them -- not of the methods.")

    out = {"assumptions": {"M_data": M_DATA_DEFAULT, "field_points": FIELD_POINTS},
           "measured": {"T_solve_s": solve, "T_train_s": train,
                        "grid_query_us": q_grid, "pinn_query_us": q_pinn},
           "Q1_one_instance_many_queries": q1,
           "Q2_many_instances_one_query_each": q2}
    (RESULTS / "amortization_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"\nwrote {RESULTS / 'amortization_analysis.json'}")


if __name__ == "__main__":
    main()
