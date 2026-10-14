"""
REPRODUCIBILITY CHECK: does the same command give the same floor?

Found while running two convergence jobs concurrently (n_steps = 50 to 20000 epochs and n_steps =
400 to 12000 epochs): the n_steps = 50 floor at epoch 500 came out 9.06e-03 in one run and 5.58e-03
in another, for the same command, the same seed, and the same 1024 training paths reseeded from the
same integer stream. Both runs also reproduce the archived early trajectory, so neither is simply
"wrong" -- they are the same computation reaching different points.

That matters for the verdict in a specific way. This experiment compares a floor measured at
n_steps = 50 against a floor measured at n_steps = 400. If run-to-run scatter in the floor is of the
same size as the difference between the two resolutions, the comparison is not measuring resolution
at all. So the scatter has to be quantified before any ratio is interpreted, rather than assumed
small.

The runs are already done and their state files are on disk, so this re-measures them rather than
retraining: it loads each saved state and evaluates the floor on identical fresh batches. Identical
parameters give identical floors by construction, which separates two very different situations:

  * saved states differ in their floors -> the TRAINING diverged (a real reproducibility finding,
    and a limit on how finely resolution can be resolved by this design);
  * saved states agree in their floors -> the earlier disagreement was in the transient readings,
    not in the converged parameters.

Usage: python src/check_bsde_repro.py
"""

from __future__ import annotations

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
from exp_bsde_steps import DeepBSDEMonitor        # noqa: E402
from probe_bsde_floor import exact_floor          # noqa: E402

OUT = Path("results")


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])

    states = sorted(OUT.glob("bsde_state_ns*.pt"))
    if not states:
        print("no saved states found in results/ -- run exp_bsde_steps.py first")
        return 1

    print("=" * 100)
    print("REPRODUCIBILITY: floors of every saved state, re-measured on IDENTICAL batches")
    print("=" * 100)
    print("  Identical parameters give identical floors, so any difference below is a difference in")
    print("  the TRAINED PARAMETERS, not in the measurement.")
    print()

    # One fixed set of batches per n_steps, shared by every state at that n_steps.
    cache: dict[tuple[int, int], torch.Tensor] = {}
    rows = []
    print(f"  {'state file':>34} {'vintage':>26} {'ns':>5} {'floor(abs)':>12} {'ratio':>8} "
          f"{'V_rel_err':>10}")
    for p in states:
        blob = torch.load(p, map_location=device, weights_only=False)
        n_steps = int(blob["n_steps"])
        s = DeepBSDEMonitor(prob, x0, n_steps=n_steps, n_paths=1024, epochs=1, device=device,
                            seed=0, log_every=0, eval_paths=8192)
        s.net.load_state_dict(blob["net"])
        with torch.no_grad():
            s.theta.copy_(blob["theta"].to(device))

        # 5 batches of 8192 paths, identical for every state at this n_steps.
        floors = []
        for k in range(5):
            key = (n_steps, k)
            if key not in cache:
                g = torch.Generator(device=s.device).manual_seed(700_000 + k)
                cache[key] = torch.randn(n_steps, 8192, s.d, device=s.device,
                                         generator=g) * np.sqrt(s.dt)
            keep = s.dW
            s.dW = cache[key]
            with torch.no_grad():
                xT, VT = s._terminal_batch(8192)
                payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
                floors.append(float(((VT - payoff).pow(2).mean()).sqrt()))
            s.dW = keep
        mean = float(np.mean(floors))
        sd = float(np.std(floors))
        floor_abs, _, _, _ = exact_floor(prob, x0, n_steps, 1024, device=device)
        V_hat = float(s.theta.detach()) * s.V_scale
        print(f"  {p.name:>34} {blob.get('tag', '?'):>26} {n_steps:>5} {mean:>12.4e} "
              f"{mean / floor_abs:>8.1f} {abs(V_hat - V_true) / abs(V_true):>10.3e}")
        rows.append({"file": p.name, "tag": blob.get("tag"), "n_steps": n_steps,
                     "epochs": int(blob.get("epochs", -1)), "floor_mean": mean,
                     "floor_sd_across_batches": sd, "exact_floor": floor_abs,
                     "ratio": mean / floor_abs,
                     "V_rel_err": abs(V_hat - V_true) / abs(V_true)})

    # Group by n_steps so the comparison the verdict needs is explicit.
    print()
    print("-" * 100)
    print("  Floors grouped by resolution. This is the comparison the verdict rests on:")
    print()
    by_ns: dict[int, list[dict]] = {}
    for r in rows:
        by_ns.setdefault(r["n_steps"], []).append(r)
    print(f"  {'n_steps':>8} {'states':>7} {'floor min':>12} {'floor max':>12} {'spread':>9} "
          f"{'exact':>11} {'ratio min':>10} {'ratio max':>10}")
    for ns in sorted(by_ns):
        rs = by_ns[ns]
        fmin = min(r["floor_mean"] for r in rs)
        fmax = max(r["floor_mean"] for r in rs)
        ex = rs[0]["exact_floor"]
        print(f"  {ns:>8} {len(rs):>7} {fmin:>12.4e} {fmax:>12.4e} "
              f"{(fmax / fmin - 1):>8.1%} {ex:>11.3e} {fmin / ex:>10.1f} {fmax / ex:>10.1f}")

    print()
    print("  READING: compare the SPREAD column against the ratio differences BETWEEN resolutions.")
    print("  If the spread within one resolution is small next to the change in ratio across")
    print("  resolutions, the sweep can resolve resolution from noise and the verdict stands. If the")
    print("  spread is comparable, the design cannot separate them and the verdict must be")
    print("  qualified or the runs repeated with a fixed, deterministic execution.")

    (OUT / "check_bsde_repro.json").write_text(json.dumps(
        {"rows": rows, "V_true": V_true}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'check_bsde_repro.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
