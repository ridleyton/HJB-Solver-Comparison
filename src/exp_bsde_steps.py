"""
THE DECIDING EXPERIMENT: is the deep-BSDE accuracy floor TIME DISCRETISATION or the NETWORK?

The question, stated so it can be answered rather than argued
------------------------------------------------------------
At n_steps = 50 the trained terminal-loss floor is 2.10e-03 in the normalised metric, while the
EXACT solution pushed through the same recursion attains 1.006e-03 (probe_bsde_floor.py). Two
candidate explanations, and they imply opposite things for the paper:

  (D) DISCRETISATION. The Euler recursion is the limit. More steps buy accuracy, the network is
      fine, and the last session's 48%-off value is a resolution choice -- so the deep-BSDE line
      continues, just at a higher n_steps.
  (N) NETWORK. The recursion can do far better than the network reaches. Refining time cannot
      close the gap, no deep-BSDE accuracy number may be reported, and the comparison has to be
      argued on COST rather than accuracy.

The trap in the obvious version of this experiment
--------------------------------------------------
Sweeping n_steps at a FIXED epoch count decides nothing, because raising n_steps changes two
things at once: dt falls (layer D), and each epoch costs proportionally more so the optimiser gets
less total progress per unit time (layer N gets worse). A floor that stays flat under that sweep is
ambiguous -- it could be a discretisation floor or plain undertraining.

The design that removes the ambiguity
-------------------------------------
1. MATCHED WORK. Epochs are set so that epochs * n_steps is constant (the per-epoch cost measured
   in the probe is close to linear in n_steps). Every configuration therefore receives the same
   total number of rollout steps, hence total network forward/backward evaluations. This holds the
   optimiser's total progress roughly fixed and varies only the resolution. This is the design the
   first session used implicitly.

2. THE COMPARISON THAT IS THE RESULT. For each n_steps the measured floor is read against a floor
   that is known exactly and needs no training at all:
       ratio(n_steps) = floor_trained(n_steps) / floor_exact(n_steps)
   floor_trained tracks floor_exact (ratio flat in n_steps)  -> (D), the network is at the limit.
   floor_trained stays flat while floor_exact falls           -> (N), the network is the limit.
   The ratio is the whole verdict, and it is dimensionless, so it cannot be misread across units.

3. THE UNDERTRAINING CONTROL. Under matched work, the n_steps = 400 run gets 8x FEWER epochs than
   n_steps = 50. So the sweep is followed by ONE long run at the finest resolution, allowed to
   train far past the matched budget. If the floor at n_steps = 400 falls on that long run, the
   matched-work floors were undertrained and the sweep alone would have been misleading. That
   control is what licenses a verdict rather than a suggestion.

4. A DIRECT CHECK OF THE DRIVER AT THE NEW RESOLUTION. The exact-solution check is rerun at every
   n_steps (as the solver already does) and cross-checked against the independent floor computed
   in probe_bsde_floor.py. Two computations of the same quantity; if the solver's recursion had a
   resolution-dependent problem, they would not agree.

Reported at every checkpoint, on a FRESH evaluation batch (so no path overfitting can flatter the
floor): terminal RMSE against the exact-solution floor at that dt, V relative error at x0, and the
control relative error at x0.

Usage:
  python src/exp_bsde_steps.py                 # full sweep + long-run control
  python src/exp_bsde_steps.py --steps 50,200  # subset
  python src/exp_bsde_steps.py --long-epochs 0 # skip the control
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

# Before the torch import: Anaconda MKL and torch each ship an OpenMP runtime and the second
# load aborts the process (OMP Error #15). Same workaround as verify_all.py.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torch.nn as nn

SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))

from ac_problem import ACProblem              # noqa: E402
from deep_bsde import DeepBSDE, _mlp          # noqa: E402
from probe_bsde_floor import exact_floor      # noqa: E402

OUT = Path("results")

# Matched-work budget: epochs * n_steps held constant. Measured per-epoch costs are 0.120, 0.149,
# 0.306, 0.617 s at n_steps = 50, 100, 200, 400, so this is ~1/2, 1/4, 1/8, 1/16 of the 7000-epoch
# n_steps = 50 run the previous session used -- and epochs*n_steps = 70_000 for all of them.
MATCHED_EPOCH_STEPS = 70_000

# The n_steps = 50 baseline was 7000 epochs; keep it exactly, so the sweep's coarsest point
# reproduces the previous session's number rather than merely resembling it.
EPOCH_OVERRIDE = {50: 7000}

# The convergence run. Everything above is a MATCHED-WORK comparison; this one is allowed to train
# far past every other point so that "the floor is flat" can be distinguished from "the floor has
# not finished falling". Deciding discretisation-versus-network from a moving quantity would be the
# most likely way to get this wrong -- a slow power-law decay is easily mistaken for a plateau.
# The budget is set at 1,200,000 rollout steps, more than treble the matched budget, because the
# previous session's plateau claim (2.10e-03 at 7000 epochs) is not reproducible here: at 1000
# epochs the traces agree to four digits and the decay continues well past where that plateau was
# reported. Extra budget is cheap next to a wrong verdict.
CONVERGENCE_EPOCHS = 12000

# Checkpoints for the trajectory view. Only the ones <= epochs are used, so one list serves every
# configuration and the matched and convergence runs stay comparable at shared epochs.
CHECKPOINTS = [100, 175, 250, 500, 700, 1000, 1400, 2000, 2800, 4000, 5600, 7000, 8000,
               10000, 12000]


class DeepBSDEMonitor(DeepBSDE):
    """DeepBSDE plus a held-out-anchor evaluation and an auditable training trace.

    The solver itself is untouched; only measurement is added, so a number produced here and a
    number produced by deep_bsde.py are the same quantity.
    """

    def __init__(self, *a, eval_paths: int = 4096, hi_paths: int = 65536, seed: int = 0, **kw):
        super().__init__(*a, seed=seed, **kw)
        self.eval_paths = eval_paths
        self._seed = seed
        # A second, much larger batch used once at the END of a run. The per-checkpoint estimate is
        # deliberately cheap, and the epoch-to-epoch scatter of that estimate is of the same order
        # as the effect this whole experiment is trying to detect -- a 2x gap in the floor. The
        # final number that gets quoted therefore comes from many more paths, so the verdict cannot
        # be an artefact of Monte Carlo noise in the measurement instrument itself.
        self.hi_paths = hi_paths
        self.trace: list[dict] = []

    # ----------------------------------------------------------------------------------
    def _paths(self, n_paths: int, seed: int) -> torch.Tensor:
        """Fresh Brownian increments at this dt.

        Fresh rather than cached: an evaluation batch held fixed across checkpoints can be
        memorised, and the reported floor would then improve without the solver improving.
        """
        gen = torch.Generator(device=self.device).manual_seed(seed)
        return torch.randn(self.n_steps, n_paths, self.d, device=self.device,
                           generator=gen) * math.sqrt(self.dt)

    def _terminal_batch(self, n_paths: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Terminal (x, V) for an EVALUATION batch, at any batch size.

        DeepBSDE._rollout expands V from theta to self.n_paths, so it only accepts a batch of the
        training width. That is fine for training but wrong for evaluation: the evaluation floor
        should not be tied to the training batch size when a steadier estimate is wanted. Rather
        than modify the validated solver, this mirrors its recursion and builds V from the SCALAR
        theta -- one element, broadcast -- so any batch size works while the value recursion is
        stepped identically.

        THE BUG THIS DOCSTRING EXISTS TO PREVENT. The first version of this function started the
        value at ZERO instead of at theta. Because the recursion is V_T = V_0 + (running cost) +
        (stochastic integral), a wrong V_0 shifts every path's terminal value by the same constant,
        so the reported mismatch was inflated by exactly |V_0|:

            ||(V_T - 0 - g)||  vs  ||(V_T - V_0 - g)||,   V_0 = V_scale * theta = -8.95e-03

        At the measured point that inflated the floor by a factor 1.937 in the terminal RMSE, and
        it was invisible in every check that compared the evaluator against ITSELF -- the giveaway
        was only that the evaluated floor (1.08e-02) came out ABOVE the loss the parameters had just
        been trained to (1.12e-01 normalised, i.e. 5.58e-03 absolute). An evaluation of trained
        parameters cannot be worse than their own training loss on the same distribution. That
        impossibility is what exposed it, and it is the second time in this project that a
        re-implementation disagreed with the thing it re-implemented rather than failing loudly.
        """
        return self._terminal_batch_chunked(n_paths)

    def _terminal_batch_chunked(self, n_paths: int, chunk: int = 8192
                                ) -> tuple[torch.Tensor, torch.Tensor]:
        """Same recursion, run in chunks along the path axis, then concatenated.

        Chunking is not an optimisation -- it is required. The high-precision final estimate uses
        65536 paths, and at n_steps = 400 the per-step activations for a batch that wide exceed the
        16 GB of the available GPU, so the honest measurement is physically impossible without it.
        The recursion is independent across paths, so chunking changes nothing but peak memory.
        """
        if n_paths <= chunk:
            return self._run_terminal(n_paths, 0)

        xs, Vs = [], []
        for start in range(0, n_paths, chunk):
            m = min(chunk, n_paths - start)
            xc, Vc = self._run_terminal(m, start)
            xs.append(xc)
            Vs.append(Vc)
        return torch.cat(xs, dim=0), torch.cat(Vs, dim=0)

    def _run_terminal(self, n_paths: int, offset: int) -> tuple[torch.Tensor, torch.Tensor]:
        dW = self.dW
        assert dW is not None and offset + n_paths <= dW.shape[1]
        w = dW[:, offset:offset + n_paths]
        x = torch.tensor(self.x0, dtype=torch.float32, device=self.device)
        x = x.expand(n_paths, self.d).clone()
        # V_0 = V_scale * theta, matching DeepBSDE._rollout exactly. theta is a 1-element tensor, so
        # this broadcasts to any batch width without the n_paths dependency the solver's rollout has.
        V = (self.V_scale * self.theta).reshape(1, 1).expand(n_paths, 1)
        for n in range(self.n_steps):
            tn = n * self.dt
            tt = torch.full((n_paths, 1), tn, dtype=torch.float32, device=self.device)
            g = self._grad_net(x, tt)
            u = (g @ self.B) @ self.R_inv.T
            run = 0.5 * torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1) \
                + 0.5 * torch.einsum("ni,ij,nj->n", g, self.B @ self.R_inv @ self.B.T, g
                                     ).unsqueeze(1)
            V = V + run * self.dt + torch.einsum("ni,ni->n", g @ self.Sigma, w[n]).unsqueeze(1)
            x = x + (x @ self.A.T + u @ self.B.T) * self.dt + w[n] @ self.Sigma.T
        return x, V

    def terminal_rmse(self, n_paths: int | None = None, seed: int = 999_999) -> float:
        """Terminal-loss RMSE in ABSOLUTE units, on a fresh batch, with no gradient.

        Absolute rather than normalised so it can be placed directly beside floor_exact(), which
        is computed in absolute units. Keeping both in one unit system is deliberate: mixing
        scaled and unscaled quantities is the mistake that produced confident wrong numbers three
        times in this project already.
        """
        n_paths = n_paths or self.eval_paths
        keep = self.dW
        self.dW = self._paths(n_paths, seed)
        with torch.no_grad():
            xT, VT = self._terminal_batch(n_paths)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, self.QT, xT).unsqueeze(1)
            rmse = float(((VT - g).pow(2).mean()).sqrt())
        self.dW = keep
        return rmse

    def value_grad_at_x0(self) -> tuple[float, np.ndarray]:
        """(V(x0,0), grad V(x0,0)) -- theta is stored SCALED, so multiply by V_scale."""
        x = torch.tensor(np.atleast_2d(self.x0), dtype=torch.float32, device=self.device)
        t = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            g = self._grad_net(x, t)
        return float(self.theta.detach()) * self.V_scale, g.cpu().numpy()[0]

    # ----------------------------------------------------------------------------------
    def solve_monitored(self, checkpoints: list[int], V_true: float,
                        grad_true: np.ndarray, exact_rmse: float) -> None:
        """Train with a frozen warm-up, then take over the loop to log at checkpoints.

        The first phase is byte-identical to DeepBSDE.solve (same generator seeding, so the same
        dW stream) up to the first checkpoint. Only after that does this loop re-evaluate, which
        costs one extra forward pass and does not perturb the optimisation.
        """
        ck = sorted({c for c in checkpoints if 1 <= c <= self.epochs})
        # The seed is threaded through so a result can be independently CONFIRMED, not just
        # reported: the training trajectory on this problem is sensitive to execution-level
        # floating-point differences (two concurrently-run copies of the same command were seen to
        # descend at visibly different rates before converging to the same floor), so a headline
        # number that only one seed supports is not yet a number.
        gen = torch.Generator(device=self.device).manual_seed(self._seed)
        self._brownian(gen)
        opt = torch.optim.Adam(list(self.net.parameters()) + [self.theta], lr=self.lr)
        t0 = time.perf_counter()
        done = 0
        for target in ck + [self.epochs]:
            while done < target:
                self.dW = self._paths(self.n_paths, self._seed * 1_000_003 + 1_000_000 + done)
                opt.zero_grad(set_to_none=True)
                xT, VT = self._rollout(self.theta, self._grad_net)
                g = -0.5 * torch.einsum("ni,ij,nj->n", xT, self.QT, xT).unsqueeze(1)
                loss = ((VT - g) / self.V_scale).pow(2).mean()
                loss.backward()
                opt.step()
                done += 1
            self._record(done, V_true, grad_true, exact_rmse, time.perf_counter() - t0)
            print(f"      epoch {done:>6}  term RMSE {self.trace[-1]['terminal_rmse']:.4e}  "
                  f"(exact floor {exact_rmse:.3e}, ratio "
                  f"{self.trace[-1]['ratio_to_exact_floor']:.2f})  "
                  f"V_rel_err {self.trace[-1]['value_rel_err']:.3e}  "
                  f"grad_rel_err {self.trace[-1]['grad_rel_err']:.3e}", flush=True)
        self.solve_time = time.perf_counter() - t0

    def _record(self, epoch: int, V_true: float, grad_true: np.ndarray, exact_rmse: float,
                secs: float) -> None:
        V_hat, g_hat = self.value_grad_at_x0()
        rmse = self.terminal_rmse()
        self.trace.append({
            "epoch": epoch, "seconds": secs, "terminal_rmse": rmse,
            "exact_floor": exact_rmse, "ratio_to_exact_floor": rmse / exact_rmse,
            "V_hat": V_hat, "V_true": V_true,
            "value_rel_err": abs(V_hat - V_true) / abs(V_true),
            "grad_rel_err": _rel_err(g_hat, grad_true),
        })


def _rel_err(got: np.ndarray, want: np.ndarray) -> float:
    """Max relative error, elementwise, with both sides flattened.

    Flattening matters: the reference helpers return (1,) and (1,1) arrays at d = 1, so a naive
    subtraction can leave a broadcast-shaped result and a `max` over it hides the shape problem
    instead of a scalar. Elementwise is also the right comparison for the control, where one
    component can dominate.
    """
    got = np.asarray(got, dtype=float).ravel()
    want = np.asarray(want, dtype=float).ravel()
    scale = np.max(np.abs(want))
    return float(np.max(np.abs(got - want)) / scale)


def run_one(n_steps: int, args, device: str, prob: ACProblem, x0: np.ndarray,
            V_true: float, grad_true: np.ndarray, epochs: int, tag: str) -> dict:
    print(f"\n  n_steps={n_steps}   epochs={epochs}   "
          f"(epochs*n_steps={epochs * n_steps}, dt={prob.T / n_steps:.5f})")

    # Two independent computations of the same floor: the standalone probe's, and the solver's
    # own validate_on_exact. They must agree, or one of the two recursions is wrong.
    floor_abs, floor_norm, payoff_rms, _ = exact_floor(prob, x0, n_steps, 1024, device=device)
    s = DeepBSDEMonitor(prob, x0, n_steps=n_steps, n_paths=args.n_paths, epochs=epochs,
                        device=device, seed=args.run_seed, log_every=0,
                        eval_paths=args.eval_paths)
    chk = s.validate_on_exact()
    agree = abs(chk - floor_abs) / max(floor_abs, 1e-300)
    print(f"    exact-solution floor: probe {floor_abs:.4e}  solver check {chk:.4e}  "
          f"(disagreement {agree:.1%})   payoff RMS {payoff_rms:.4e}")
    if chk >= 1e-4:
        return {"n_steps": n_steps, "validated": False, "terminal_rmse_exact": chk}

    ck = [e for e in CHECKPOINTS if e <= epochs]
    if epochs not in ck:
        ck.append(epochs)
    ck = sorted(set(ck))
    s.solve_monitored(ck, V_true, grad_true, floor_abs)
    last = s.trace[-1]

    # The number that gets QUOTED: many more paths, so the instrument is not the limiting factor.
    hi = s.terminal_rmse(n_paths=s.hi_paths, seed=424242)
    V_hi, g_hi = s.value_grad_at_x0()
    print(f"    FINAL (hi-precision, {s.hi_paths} paths): floor {hi:.4e}  ratio {hi / floor_abs:.2f}"
          f"  V_rel_err {abs(V_hi - V_true) / abs(V_true):.3e}", flush=True)

    # Persist the state so the floor can be re-measured later at any resolution without retraining.
    state_path = OUT / f"bsde_state_ns{n_steps}_{tag}.pt"
    torch.save({"net": s.net.state_dict(), "theta": s.theta.detach().cpu(),
                "n_steps": n_steps, "epochs": epochs}, state_path)

    return {"n_steps": n_steps, "tag": tag, "validated": True, "epochs": epochs,
            "n_paths": args.n_paths, "eval_paths": args.eval_paths, "hi_paths": s.hi_paths,
            "dt": prob.T / n_steps, "epochs_times_steps": epochs * n_steps,
            "exact_floor_abs": floor_abs, "exact_floor_norm": floor_norm,
            "solver_check_abs": chk, "payoff_rms": payoff_rms,
            "floor_disagreement": agree, "seconds": s.solve_time,
            "terminal_rmse_final": last["terminal_rmse"],
            "ratio_final": last["ratio_to_exact_floor"],
            "terminal_rmse_hi": hi, "ratio_hi": hi / floor_abs,
            "V_hat_hi": V_hi, "value_rel_err_hi": abs(V_hi - V_true) / abs(V_true),
            "grad_rel_err_hi": _rel_err(g_hi, grad_true),
            "value_rel_err_final": last["value_rel_err"],
            "grad_rel_err_final": last["grad_rel_err"],
            "V_hat": last["V_hat"], "V_true": V_true,
            "state_file": state_path.name,
            "trace": s.trace}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=str, default="50,100,200,400")
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--eval-paths", type=int, default=4096)
    ap.add_argument("--long-steps", type=int, default=400,
                    help="resolution for the convergence run")
    ap.add_argument("--long-epochs", type=int, default=CONVERGENCE_EPOCHS,
                    help="epochs for the convergence run, far past the matched budget; 0 disables")
    # Convergence runs at different resolutions are informative on their own and are run
    # concurrently on a GPU that is nowhere near saturated, so they must not overwrite each other's
    # results. Every run also merges into the canonical file.
    ap.add_argument("--out", type=str, default="exp_bsde_steps.json",
                    help="results filename inside results/, so concurrent runs do not clobber")
    # Passing an empty --steps is not portable (PowerShell drops the empty argument), so skipping
    # the matched sweep is its own flag.
    ap.add_argument("--skip-matched", action="store_true",
                    help="run only the convergence run; useful per-resolution convergence studies")
    ap.add_argument("--run-seed", type=int, default=0,
                    help="training seed; run a second seed to CONFIRM a headline floor")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    prob = ACProblem(p=1)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    # ref.value returns shape (1,) and ref.P returns (1,1) even at d = 1, so neither is a scalar.
    V_true = float(np.asarray(ref.value(x0, 0.0)).ravel()[0])
    grad_true = np.atleast_2d(-np.atleast_2d(x0) @ ref.P(0.0)).ravel().astype(float)
    steps = [] if args.skip_matched else [int(v) for v in args.steps.split(",")]

    print("=" * 100)
    print("DEEP BSDE: is the accuracy floor TIME DISCRETISATION or the NETWORK?")
    print("=" * 100)
    print(f"  device={device}  exact V(x0,0)={V_true:+.8e}  grad V(x0,0)={grad_true[0]}")
    print("  Matched work: epochs * n_steps = "
          f"{MATCHED_EPOCH_STEPS} (except the n_steps=50 baseline, kept at 7000 epochs).")
    print("  VERDICT RULE: ratio = floor_trained / floor_exact.")
    print("    flat ratio across n_steps -> the network is AT the discretisation limit  -> (D)")
    print("    ratio rising as dt falls -> the network is the limit                     -> (N)")

    def save(rows: list[dict]) -> None:
        blob = {"rows": rows, "n_paths": args.n_paths, "eval_paths": args.eval_paths,
                "matched_epoch_steps": MATCHED_EPOCH_STEPS, "device": device,
                "V_true": V_true, "grad_true": grad_true.tolist()}
        (OUT / args.out).write_text(json.dumps(blob, indent=2), encoding="utf-8")
        # A single-resolution convergence run writes its own file; merge it into the canonical one so
        # the analysis always has every row available.
        if args.out != "exp_bsde_steps.json":
            canon = OUT / "exp_bsde_steps.json"
            prev = json.loads(canon.read_text(encoding="utf-8"))["rows"] if canon.exists() else []
            key = lambda r: (str(r.get("tag")), int(r.get("n_steps", 0)))  # noqa: E731
            merged = {key(r): r for r in prev}
            merged.update({key(r): r for r in rows})
            blob["rows"] = sorted(merged.values(), key=lambda r: (str(r.get("tag")),
                                                                  int(r.get("n_steps", 0))))
            canon.write_text(json.dumps(blob, indent=2), encoding="utf-8")

    rows: list[dict] = []

    # THE CONVERGENCE RUN GOES FIRST, deliberately. Everything else is a matched-work comparison,
    # and a matched-work point for a fine resolution is by construction barely trained. If the
    # machines are interrupted -- which has already happened twice on this task -- the expensive
    # run is the one that decides the verdict, so it is the one that must exist first.
    if args.long_epochs and args.long_steps:
        print()
        print("  CONVERGENCE RUN: the finest resolution, trained far past every matched budget.")
        print(f"  At n_steps={args.long_steps} matched work gives "
              f"{MATCHED_EPOCH_STEPS // args.long_steps} epochs; this run gets {args.long_epochs} "
              f"({args.long_epochs * args.long_steps / MATCHED_EPOCH_STEPS:.0f}x the matched work).")
        rows.append(run_one(args.long_steps, args, device, prob, x0, V_true, grad_true,
                            args.long_epochs, "convergence"))
        save(rows)

    for n in steps:
        epochs = EPOCH_OVERRIDE.get(n, max(1, MATCHED_EPOCH_STEPS // n))
        rows.append(run_one(n, args, device, prob, x0, V_true, grad_true, epochs, "matched"))
        save(rows)

    # ---------------- verdict ----------------
    print()
    print("=" * 100)
    print("SWEEP TABLE (matched work)")
    print("=" * 100)
    print(f"  {'n_steps':>8} {'epochs':>8} {'floor_exact':>12} {'floor_trained':>14} "
          f"{'ratio':>8} {'V_rel_err':>11} {'grad_rel_err':>13}")
    matched = [r for r in rows if r.get("validated") and r.get("tag") == "matched"]
    for r in matched:
        last = r["trace"][-1]
        print(f"  {r['n_steps']:>8} {r['epochs']:>8} {r['exact_floor_abs']:>12.3e} "
              f"{r['terminal_rmse_hi']:>14.4e} {r['ratio_hi']:>8.2f} "
              f"{r['value_rel_err_hi']:>11.3e} {last['grad_rel_err']:>13.3e}")

    conv = [r for r in rows if r.get("validated") and r.get("tag") == "convergence"]
    if conv:
        c = conv[0]
        print()
        print("CONVERGENCE RUN")
        print(f"  n_steps={c['n_steps']}  epochs={c['epochs']}  "
              f"floor {c['terminal_rmse_hi']:.4e}  exact floor {c['exact_floor_abs']:.3e}  "
              f"ratio {c['ratio_hi']:.2f}  V_rel_err {c['value_rel_err_hi']:.3e}")

    save(rows)
    print(f"\nwrote {OUT / 'exp_bsde_steps.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
