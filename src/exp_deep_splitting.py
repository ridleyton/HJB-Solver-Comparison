"""The deep-splitting result for the paper: the scheme, the learner, and which one binds.

WHAT THIS PRODUCES
------------------
Two arms on the same benchmark, same recursion, same paths:

  * `neural`  -- the actual deep-splitting method: one small MLP per time step, fitted by Adam.
  * `quad`    -- the SAME recursion with the per-step regression solved exactly in the quadratic
                 basis. This is not a solver anyone would propose (it uses knowledge of the solution's
                 form); it is the control experiment that decides whether an inaccurate result is the
                 SCHEME's fault or the LEARNER's.

Getting that distinction wrong is how a paper ends up reporting "method X fails" when what failed was
the author's regression. Two convention bugs in an earlier version of this diagnostic produced exactly
that false conclusion, and the self-test in `diag_splitting_recursion.py` is what caught the second.

Usage: python src/exp_deep_splitting.py [--dims 1,4,8]
Output: results/exp_deep_splitting.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
from ac_problem import ACProblem  # noqa: E402
from deep_splitting import DeepSplitting, evaluate  # noqa: E402
from diag_splitting_recursion import fitted_gradient, quad_features, self_test  # noqa: E402
from metrics import relative_l2  # noqa: E402

RESULTS = SRC.parent / "results"


def run_quad(prob: ACProblem, steps: int, paths: int, seed: int = 0) -> dict:
    """The same recursion with an exact quadratic regression at every step."""
    s = DeepSplitting(prob, n_steps=steps, n_paths=paths, epochs=1, seed=seed, spread=True,
                      device="cpu")
    s._simulate(torch.Generator(device="cpu").manual_seed(seed))
    coefs: list[np.ndarray | None] = [None] * (steps + 1)
    for n in range(steps - 1, -1, -1):
        x_n = s.paths[n].numpy()
        x_next = s.paths[n + 1].numpy()
        if n == steps - 1:
            base = -0.5 * np.einsum("ni,ij,nj->n", x_next, prob.QT, x_next)
            z = x_n @ (-prob.QT) @ prob.Sigma
        else:
            base = quad_features(x_next) @ coefs[n + 1]
            z = fitted_gradient(coefs[n + 1], x_n, prob.state_dim) @ prob.Sigma
        target = base + s.dt * s._f(torch.tensor(x_n, dtype=torch.float32),
                                    torch.tensor(z, dtype=torch.float32)).numpy().ravel()
        coefs[n], *_ = np.linalg.lstsq(quad_features(x_n), target, rcond=None)

    rng = np.random.default_rng(3)
    lo, hi = prob.domain()
    X = rng.uniform(lo, hi, size=(4000, prob.state_dim))
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    pred = quad_features(X) @ coefs[0]
    exact = np.asarray(ref.value(X, 0.0)).ravel()
    u = fitted_gradient(coefs[0], X, prob.state_dim) @ prob.B @ prob.R_inv.T
    u_ref = ref.control(X, 0.0)
    return {
        "value_r2": float(1 - np.sum((pred - exact) ** 2) / np.sum((exact - exact.mean()) ** 2)),
        "value_rel_l2": relative_l2(pred, exact),
        "control_r2": float(1 - np.sum((u - u_ref) ** 2) / np.sum((u_ref - u_ref.mean()) ** 2)),
        "control_rel_l2": relative_l2(u, u_ref),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,4,8")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--paths", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="exp_deep_splitting.json")
    args = ap.parse_args()

    if not self_test():
        print("  SELF-TEST FAILED: the quadratic diagnostic's gradient disagrees with finite "
              "differences; refusing to produce numbers.")
        return 1

    dims = [int(s) for s in args.dims.split(",") if s.strip()]
    print("=" * 104)
    print("DEEP SPLITTING vs THE SAME RECURSION WITH AN EXACT QUADRATIC REGRESSION")
    print("=" * 104)
    print(f"  {args.steps} steps, {args.paths} paths, {args.layers}x{args.hidden} per step, "
          f"{args.epochs} epochs/step")
    out = {"args": vars(args), "rows": []}
    for d in dims:
        prob = ACProblem(p=d)
        print(f"\n  d={d}")
        solver = DeepSplitting(prob, n_steps=args.steps, n_paths=args.paths, epochs=args.epochs,
                               hidden=args.hidden, layers=args.layers, seed=args.seed,
                               spread=True, device="cpu")
        val = solver.validate_on_exact(n_steps_list=(args.steps, max(args.steps // 2, 2)))
        t0 = time.perf_counter()
        solver.solve()
        secs = time.perf_counter() - t0
        sc = evaluate(prob, solver)
        sc.update({"d": d, "arm": "neural", "seconds": secs,
                   "median_step_loss": float(np.median(solver.step_losses)),
                   "gradient_steps": args.steps * args.epochs, "validation": val})
        out["rows"].append(sc)
        print(f"    neural : value R2 {sc['value_r2']:+.4f}  control R2 {sc['control_r2']:+.4f}  "
              f"{secs:.0f}s  median step loss {sc['median_step_loss']:.2e}")
        t0 = time.perf_counter()
        qc = run_quad(prob, args.steps, args.paths, args.seed)
        qs = time.perf_counter() - t0
        qc.update({"d": d, "arm": "quadratic-basis control", "seconds": qs,
                   "validation": {"by_steps": {args.steps: val["by_steps"][args.steps]}}})
        out["rows"].append(qc)
        print(f"    quad   : value R2 {qc['value_r2']:+.4f}  control R2 {qc['control_r2']:+.4f}  "
              f"{qs:.0f}s")
    (RESULTS / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    print("DEEP SPLITTING COMPLETE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
