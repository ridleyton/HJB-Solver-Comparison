"""
Alpha benchmark: validate the reference against a grid solve, and test separability.

Two questions, both of which have to be answered before the alpha model can go in the paper.

1. IS THE REFERENCE RIGHT?  For the inventory-only benchmark the Riccati solution was validated
   against three independent closed forms. There is no elementary closed form once the alpha
   signal is present (the system is 2x2 with A != 0), so the validation has to be different: we
   solve the SAME HJB with the finite-difference solver -- an independent discretisation -- and
   check that the solver converges TO the reference as the mesh is refined. If the reference were
   a wrong solution branch, the solver would converge to something else and the R^2 would plateau
   below 1 instead of approaching it.

   The control-order column matters as much as the R^2: the paper's central claim is that the
   control error converges at the order of the reconstruction while the value error converges
   faster. If that survives the alpha extension it is a property of the differentiation operator
   rather than of the inventory-only model.

2. IS THE PROBLEM STILL SEPARABLE?  With uniform coefficients and A = 0 the value function is
   exactly separable, V = sum_i v(x_i), so a network solving it at high dimension is doing little
   more than summing one learned univariate function. The previous paper version broke that
   symmetry by RANDOMISING the coefficients, which is ad hoc. The alpha signal should break it
   STRUCTURALLY: the q-s cross term in Q and the (q, s) coupling in the dynamics make the Riccati
   solution P have a non-zero off-diagonal block even with perfectly uniform parameters. This
   script measures that block directly, so the claim can be made about the model rather than
   about a random draw.

Usage: python src/exp_alpha.py [--quick]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ac_problem import ACProblem
from exp_control_order import evaluate
from solvers import FiniteDifferenceSolver, accuracy_metrics

try:                                              # torch is optional for the grid-only sections
    from pinn import PINNSolver
    from pinn_pi import PIPINN
except Exception:                                 # pragma: no cover
    PINNSolver = None
    PIPINN = None

OUT = Path("results")


def separability_report(problem: ACProblem) -> dict:
    """Measure how far P is from block-diagonal, i.e. how far V is from separable.

    A separable V requires a diagonal P for all t. We report the Frobenius norm of the
    off-diagonal block(s) relative to the diagonal, at the time where it is largest.
    """
    t_grid = np.linspace(0.0, problem.T, 201)
    ref = problem.reference(t_grid)
    p = problem.p
    worst = 0.0
    for t in t_grid:
        P = ref.P(t)
        diag = np.linalg.norm(np.diag(P))
        off = np.linalg.norm(P - np.diag(np.diag(P)))
        worst = max(worst, off / max(diag, 1e-300))
    return {"offdiag_over_diag": worst, "state_dim": problem.state_dim}


def refinement(problem: ACProblem, grids: list[int], n_snap: int, label: str) -> list[dict]:
    rows = []
    for n in grids:
        t0 = time.perf_counter()
        s = FiniteDifferenceSolver(problem, n_grid=n, n_snap=n_snap, snap_cap=4e9)
        s.solve()
        # Use the SAME metric as exp_control_order.evaluate: normalised MSE (1 - R^2), i.e. the
        # error divided by the target's spread. Two traps this avoids, both of which bit earlier
        # versions of this script:
        #   * fitting an order to an R^2 rather than an error lets a saturating R^2 manufacture a
        #     spurious "high order" out of floating-point noise;
        #   * taking sqrt(MSE) HALVES the fitted slope, because the paper's order convention is
        #     defined on the MSE. A first version reported control order 1.06 this way where the
        #     validated successive-order measurement gives ~2.2 -- the same result, mis-scaled.
        m = evaluate(s, problem, n_points=1500, n_times=6, seed=1)
        wall = time.perf_counter() - t0
        v_err, c_err = m["value_err"], m["control_err"]
        rows.append({
            "label": label, "N": n, "value_r2": m["value_r2"], "control_r2": m["control_r2"],
            "value_err": v_err, "control_err": c_err,
            "ratio": c_err / v_err,
            "n_steps": s.n_time_steps, "n_snap_used": s.n_snap_used, "seconds": wall,
        })
        print(f"    N={n:<4} value R2={m['value_r2']:.6f}  ctrl R2={m['control_r2']:.6f}  "
              f"value_err={v_err:.3e}  ctrl_err={c_err:.3e}  ratio={c_err / v_err:.1f}  "
              f"{s.n_time_steps} steps  {wall:.1f}s")
    return rows


def fit_order(rows: list[dict], key: str) -> tuple[float, float]:
    """Least-squares order: slope of log(err) against log(h), with h = 1/N."""
    n = np.array([r["N"] for r in rows], dtype=float)
    e = np.array([r[key] for r in rows], dtype=float)
    ok = e > 0
    if ok.sum() < 3:
        return float("nan"), float("nan")
    x, y = np.log(1.0 / n[ok]), np.log(e[ok])
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return float(slope), float(r2)


def neural_section(seeds: list[int], epochs: int, device: str) -> list[dict]:
    """Train the neural solvers at matched state dimension, separable vs signal-augmented.

    The point of this section
    ------------------------
    Section V-C of the paper reports that a SEPARABLE problem fails catastrophically
    (R^2 = -688 at p = 8) while a non-separable one at the same dimension, architecture and budget
    is solved well (R^2 = 0.981), and that the non-separability was obtained by RANDOMISING
    per-asset coefficients. That is an uncomfortable way to make the point: a referee can reply
    that the hard case was manufactured, or that one random draw proves little.

    The signal extension breaks separability BY CONSTRUCTION, with uniform parameters. So it
    supplies an independent, principled instance of the same contrast, and this section tests it at
    MATCHED STATE DIMENSION: inventory-only p=2 against alpha with one asset (both d = 2), and
    inventory-only p=4 against alpha with two assets (both d = 4).

    The budget is deliberately identical to the dimension sweep (6000 epochs, 2048 collocation
    points, 4x64 tanh, lr 2e-3) so the numbers sit directly beside Table I. Reporting a different
    budget here would make the two incomparable, which is exactly the mistake that produced a
    spurious control-order number earlier in this project.
    """
    if PINNSolver is None:
        print("  torch unavailable -- skipping the neural section")
        return []

    print(f"  device={device}  epochs={epochs}  collocation=2048  seeds={seeds}")
    configs = [
        ("separable", ACProblem(p=2)),
        ("alpha", ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005)),
        ("separable", ACProblem(p=4)),
        ("alpha", ACProblem(p=2, alpha_kappa=1.0, alpha_sig=0.005)),
        ("separable", ACProblem(p=6)),
        ("alpha", ACProblem(p=3, alpha_kappa=1.0, alpha_sig=0.005)),
    ]
    rows: list[dict] = []
    for kind, prob in configs:
        for seed in seeds:
            try:
                s = PINNSolver(prob, hidden=64, layers=4, n_collocation=2048, epochs=epochs,
                               lr=2e-3, device=device, seed=seed)
                wall = s.solve()
                m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
            except Exception as e:                # a failed run is data, not a crash
                print(f"    {kind:<10} d={prob.state_dim} seed={seed}  FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                rows.append({"method": "PINN", "kind": kind, "state_dim": prob.state_dim,
                             "seed": seed, "feasible": False, "error": str(e)})
                continue
            rows.append({
                "method": "PINN", "kind": kind, "state_dim": prob.state_dim, "seed": seed,
                "feasible": True, "seconds": wall,
                "value_r2": m["value_r2"], "control_r2": m["control_r2"],
                "sign_agreement": m["control_sign_agreement"],
                "epochs": epochs, "n_collocation": 2048,
            })
            print(f"    {kind:<10} d={prob.state_dim} seed={seed}  {wall:7.1f}s  "
                  f"value R2={m['value_r2']:+.5f}  ctrl R2={m['control_r2']:+.5f}  "
                  f"sign={m['control_sign_agreement']:.3f}", flush=True)

    # The PI-PINN on the signal benchmark: it is the method the paper compares against, so the
    # alpha case needs it too, at the same budget.
    if PIPINN is not None:
        prob = ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005)
        for seed in seeds:
            try:
                s = PIPINN(prob, hidden=64, layers=4, num_iters=20, num_epochs=300,
                           device=device, seed=seed)
                wall = s.solve()
                m = accuracy_metrics(s, prob, n_points=600, n_times=6, seed=11)
            except Exception as e:
                print(f"    PI-PINN    d={prob.state_dim} seed={seed}  FAILED: "
                      f"{type(e).__name__}: {e}", flush=True)
                rows.append({"method": "PI-PINN", "kind": "alpha", "state_dim": prob.state_dim,
                             "seed": seed, "feasible": False, "error": str(e)})
                continue
            rows.append({
                "method": "PI-PINN", "kind": "alpha", "state_dim": prob.state_dim, "seed": seed,
                "feasible": True, "seconds": wall,
                "value_r2": m["value_r2"], "control_r2": m["control_r2"],
                "sign_agreement": m["control_sign_agreement"],
                "epochs": 20 * 300, "n_collocation": s.n_points,
            })
            print(f"    {'PI-PINN':<10} d={prob.state_dim} seed={seed}  {wall:7.1f}s  "
                  f"value R2={m['value_r2']:+.5f}  ctrl R2={m['control_r2']:+.5f}  "
                  f"sign={m['control_sign_agreement']:.3f}", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--neural", action="store_true",
                    help="also train the neural solvers (slow; minutes per run)")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--epochs", type=int, default=6000)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    if args.device:
        device = args.device
    else:
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            device = "cpu"

    OUT.mkdir(exist_ok=True)
    report: dict = {"separability": {}, "refinement": {}}

    print("=" * 78)
    print("1. SEPARABILITY: is V still separable with uniform coefficients?")
    print("=" * 78)
    cases = {
        "inventory-only, uniform (A=0)": ACProblem(p=2),
        "inventory-only, coupling=0.002": ACProblem(p=2, coupling=0.002),
        "alpha, uniform coefficients": ACProblem(p=2, alpha_kappa=1.0, alpha_sig=0.005),
    }
    for name, prob in cases.items():
        rep = separability_report(prob)
        report["separability"][name] = rep
        print(f"  {name:<34} state_dim={rep['state_dim']}  "
              f"||P_offdiag||/||P_diag|| max = {rep['offdiag_over_diag']:.3e}")
    print()
    print("  A separable V needs a diagonal P for all t, so the inventory-only row must be ~0")
    print("  and the alpha row must be clearly non-zero. That is the difference between")
    print("  breaking the symmetry by construction and breaking it by randomising parameters.")

    print()
    print("=" * 78)
    print("2. REFERENCE VALIDATION: does the grid solve converge TO the Riccati reference?")
    print("=" * 78)
    prob = ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005)
    print(f"  alpha benchmark, 1 asset, state dim {prob.state_dim}; "
          f"grid points = N^{prob.state_dim}")
    grids = [11, 15, 21, 25] if args.quick else [11, 15, 21, 25, 31, 35, 41]
    rows = refinement(prob, grids, n_snap=101, label="alpha p1")
    report["refinement"]["alpha_p1"] = rows
    vo, vr = fit_order(rows, "value_err")
    co, cr = fit_order(rows, "control_err")
    print(f"  fitted value order   = {vo:.2f}  (fit R2 {vr:.3f})")
    print(f"  fitted control order = {co:.2f}  (fit R2 {cr:.3f})")
    report["order"] = {"value": vo, "value_fit_r2": vr, "control": co, "control_fit_r2": cr}

    print()
    print("  same benchmark, inventory-only at the SAME state dimension for comparison:")
    prob0 = ACProblem(p=2)
    rows0 = refinement(prob0, grids, n_snap=101, label="inventory p2")
    report["refinement"]["inventory_p2"] = rows0
    co0, cr0 = fit_order(rows0, "control_err")
    print(f"  fitted control order = {co0:.2f}  (fit R2 {cr0:.3f})")
    report["order_inventory"] = {"control": co0, "control_fit_r2": cr0}

    print()
    print("=" * 78)
    print("3. ECONOMIC BEHAVIOUR: does a positive signal change the trade direction?")
    print("=" * 78)
    t_grid = np.linspace(0.0, prob.T, 201)
    ref = prob.reference(t_grid)
    for s in (-1.0, 0.0, 1.0):
        x = np.array([[1.0, s]])                   # inventory 1, signal s
        u = ref.control(x, 0.0)[0, 0]
        print(f"  inventory=1, signal={s:+.0f}  ->  optimal rate u* = {u:+.4f}")
    print("  Expected: u* becomes less negative as the signal rises, and can change sign --")
    print("  the trader holds inventory to harvest the alpha instead of liquidating it.")

    if args.neural:
        print()
        print("=" * 78)
        print("4. NEURAL SOLVERS at matched state dimension: separable vs signal-augmented")
        print("=" * 78)
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        report["neural"] = neural_section(seeds, args.epochs, device)
        # Summarise the matched pairs, which is the claim this section exists to test.
        print()
        for d in sorted({r["state_dim"] for r in report["neural"] if r.get("feasible")}):
            for kind in ("separable", "alpha"):
                vals = [r["value_r2"] for r in report["neural"]
                        if r.get("feasible") and r["state_dim"] == d and r["kind"] == kind
                        and r["method"] == "PINN"]
                if vals:
                    mean = float(np.mean(vals))
                    spread = float(np.std(vals))
                    print(f"  d={d}  {kind:<10} PINN value R2 = {mean:+.4f} +/- {spread:.4f}"
                          f"   (n={len(vals)})")
        print("  If the separable column is markedly worse than the alpha column at the same")
        print("  d, the parameterisation effect reproduces WITHOUT randomising any coefficient.")

    (OUT / "exp_alpha.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'exp_alpha.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
