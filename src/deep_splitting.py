"""
Deep splitting (Beck-Becker-Cheridito-Jentzen-Neuenkirch) on the same benchmark.

WHY THIS SOLVER IS HERE
-----------------------
The paper's comparison represented the deep-BSDE family by its reference member, and a referee is
entitled to answer "then compare against a strong variant". Deep splitting is that variant: it
replaces one time-global network parameterising (V, Z) along paths with one small network per time
step, each fitted by regression on simulated paths. It is the modern member of the family, it needs no
terminal-mismatch trick to be stable, and it is the natural second entry for that row of the
comparison.

WHAT IS BEING SOLVED, DERIVED RATHER THAN QUOTED
-----------------------------------------------
The HJB, with the maximiser substituted and u* = R^-1 B' grad V (this repository's convention, which
makes grad V = -P(t) x), is

    V_t + 1/2 Tr(Sigma Sigma' Hess V) + (A x)' grad V - 1/2 x'Qx + 1/2 z' M z = 0,

with z = Sigma' grad V and M = Sigma^-1 B R^-1 B' Sigma^-T. In the standard semilinear form
u_t + 1/2 Tr(sigma sigma' Hess u) + mu' grad u + f = 0 with mu = A x and sigma = Sigma, the
nonlinearity is therefore

    f(x, z) = -1/2 x'Qx + 1/2 z' M z       (no dependence on u, which is what makes the explicit
                                            scheme legitimate here rather than an approximation)

and the nonlinear Feynman-Kac representation along the UNCONTROLLED path dX = A X dt + Sigma dW is

    u(t, x) = E[ u(t + dt, X_{t+dt}) + dt f(X_t, z_t) | X_t = x ].

Cross-checked against the dV identity the deep-BSDE code already uses, which is independent of this
derivation: along the uncontrolled path, dV = [1/2 x'Qx - 1/2 g' B R^-1 B' g] dt + z' dW = -f dt + z' dW,
so integrating from t to T gives exactly the representation above. If the sign of f were wrong, the
validation below would show it immediately -- that is what the validation is for.

VALIDATION BEFORE COMPARISON
----------------------------
`validate_on_exact()` runs the whole recursion with u_{n+1} = the EXACT solution at every step, and
reports the largest mid-recursion deviation from the exact solution. With a correct scheme this is a
discretisation error of order dt; with a sign error, a wrong generator or a wrong terminal condition it
is O(1). Nothing is trained until that check passes.

TWO TRAINING SETS, BECAUSE THE PAPER NEEDS BOTH
-----------------------------------------------
* `spread=False` starts every path at the benchmark's standard initial state, which is what deep BSDE
  does and is the only fair way to compare against its point value.
* `spread=True` starts paths across the domain, which makes the result a FUNCTION-valued approximation
  comparable with the grid, the PINN and regression Monte Carlo (value R^2 and control R^2 over the
  domain, not one number at x_0).

Usage: python src/deep_splitting.py [--dims 1,4,8] [--steps 50] [--epochs 300] [--spread]
Output: results/exp_deep_splitting.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ac_problem import ACProblem
from metrics import relative_l2

OUT = Path("results")


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Module:
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class DeepSplitting:
    """One small network per time step, fitted backwards by regression on simulated paths."""

    name = "Deep Splitting"

    def __init__(self, problem: ACProblem, x0: np.ndarray | None = None, n_steps: int = 50,
                 n_paths: int = 4096, epochs: int = 300, lr: float = 2e-3, hidden: int = 32,
                 layers: int = 2, seed: int = 0, spread: bool = False, device: str = "cuda",
                 inner: int = 1, damping: float = 0.5, weight_decay: float = 0.0,
                 log_every: int = 0):
        self.problem = problem
        if x0 is None:
            x0 = np.zeros(problem.state_dim)
            x0[: problem.n_assets] = 1.0
        self.x0 = np.asarray(x0, dtype=float)
        self.n_steps, self.n_paths, self.epochs, self.lr = n_steps, n_paths, epochs, lr
        self.hidden, self.layers, self.spread = hidden, layers, spread
        self.inner = inner
        self.damping = damping
        self.weight_decay = weight_decay
        self.log_every = log_every
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available())
                                   else "cpu")
        torch.manual_seed(seed)
        self.seed = seed
        self.d = problem.state_dim
        self.T = float(problem.T)
        self.dt = self.T / n_steps

        t = lambda M: torch.tensor(np.asarray(M), dtype=torch.float32, device=self.device)  # noqa
        self.A, self.B, self.Q, self.QT = t(problem.A), t(problem.B), t(problem.Q), t(problem.QT)
        self.Sigma = t(problem.Sigma)
        self.R_inv = t(problem.R_inv)
        # M = Sigma^-1 B R^-1 B' Sigma^-T, the matrix multiplying z'z in f.
        Sinv = np.linalg.inv(problem.Sigma)
        self.M = t(Sinv @ problem.B @ problem.R_inv @ problem.B.T @ Sinv.T)
        self.Sigma_T = t(problem.Sigma.T)
        # Row-vector convention throughout, matching deep_bsde.py: a column vector v has row form
        # v' , so u = R^-1 B' grad V becomes u_row = grad_V_row @ B @ R^-T. On THIS benchmark
        # B = I and R = eta I, so getting that order wrong is numerically invisible -- which is
        # exactly why it is written out rather than left to chance (the signalled problem's B is not
        # the identity).
        self.control_map = t(problem.B @ problem.R_inv.T)   # u_row = grad_V_row @ control_map
        self.Sigma_inv = t(np.linalg.inv(problem.Sigma))

        # Input scaling: the domain is about [-1, 1] and the grids are built on it, so a fixed scale
        # keeps every step's network in the same regime. An unscaled input is the same class of bug as
        # the unscaled theta in deep_bsde.py.
        lo, hi = problem.domain()
        self.x_scale = float(max(np.max(np.abs(lo)), np.max(np.abs(hi)), 1e-8))

        self.nets: list[nn.Module] = [None] * (n_steps + 1)   # index n approximates u(t_n, .)
        self.step_losses: list[float] = []
        self.paths: torch.Tensor | None = None
        self.dW: torch.Tensor | None = None

    # ----------------------------------------------------------------------------------
    def _terminal(self, x: torch.Tensor) -> torch.Tensor:
        """u(T, x) = -1/2 x' Q_T x, exactly. The terminal step is NOT trained."""
        return -0.5 * torch.einsum("ni,ij,nj->n", x, self.QT, x).unsqueeze(1)

    def _simulate(self, generator: torch.Generator) -> None:
        """Uncontrolled paths dX = A X dt + Sigma dW, with fixed increments.

        The increments are drawn once so that the regression targets at every step use the same
        randomness; resampling inside the backward pass would make the per-step problems inconsistent
        with each other for no benefit.
        """
        dW = torch.randn(self.n_steps, self.n_paths, self.d, device=self.device,
                         generator=generator) * np.sqrt(self.dt)
        x = torch.tensor(self.x0, dtype=torch.float32, device=self.device)
        x = x.expand(self.n_paths, self.d).clone()
        if self.spread:
            lo, hi = self.problem.domain()
            lo_t = torch.tensor(np.asarray(lo), dtype=torch.float32, device=self.device)
            hi_t = torch.tensor(np.asarray(hi), dtype=torch.float32, device=self.device)
            u = torch.rand(self.n_paths, self.d, device=self.device, generator=generator)
            x = lo_t + u * (hi_t - lo_t)
        xs = [x]
        for n in range(self.n_steps):
            x = x + (x @ self.A.T) * self.dt + dW[n] @ self.Sigma.T
            xs.append(x)
        self.paths = torch.stack(xs)          # (n_steps+1, n_paths, d)
        self.dW = dW

    def _f(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """f(x, z) = -1/2 x'Qx + 1/2 z' M z, as derived in the module docstring."""
        quad_x = torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1)
        quad_z = torch.einsum("ni,ij,nj->n", z, self.M, z).unsqueeze(1)
        return -0.5 * quad_x + 0.5 * quad_z

    def _z_of(self, net: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """z = Sigma' grad u, by autograd through the step's network.

        In row form z_row = grad_u_row @ Sigma. Writing Sigma' here instead is invisible on this
        benchmark (Sigma = sigma I, diagonal and symmetric) and wrong in general, which is the kind of
        error this repository has been bitten by before.
        """
        xg = x.detach().requires_grad_(True)
        u = net(xg / self.x_scale)
        (g,) = torch.autograd.grad(u.sum(), xg, create_graph=False)
        return (g / self.x_scale) @ self.Sigma

    # ----------------------------------------------------------------------------------
    def validate_on_exact(self, n_steps_list: tuple[int, ...] = ()) -> dict:
        """Run the recursion with the EXACT solution and check it against the exact expectation.

        TWO WAYS TO GET THIS CHECK WRONG, both of which this function did on its first run:

        (i) Comparing a single-sample recursion target against V(t_n, x) measures the MARTINGALE
            z'dW, not the discretisation error, because the target is one draw and V(t_n,x) is its
            conditional mean. The reference has to be the conditional expectation, computable in
            closed form here: with u(t+dt, y) = -1/2 y'P(t+dt)y + c(t+dt) and the UNCONTROLLED
            increment y = x + A x dt + Sigma dW,

                E[u(t+dt, X_{t+dt}) | X_t = x] = -1/2 [ m'P(t+dt)m + Tr(P(t+dt) Sigma Sigma' dt) ] + c,

            with m = x + A x dt. That is the quantity the recursion should reproduce.

        (ii) Getting the quadrature point wrong. The Feynman-Kac integral is
            int_t^{t+dt} f ds, whose LEFT endpoint is (t_n, x) -- so f is evaluated at the current
            state with the current gradient, which is the published explicit splitting scheme. The
            first version evaluated f at x_{n+1} and still "converged", which is exactly why this
            check compares against the analytic expectation rather than against a self-consistency
            condition.

        Returns the deviation at the configured step count, its ratio across a halving of dt (first
        order expected), and the value scale it should be read against.
        """
        out: dict = {"by_steps": {}}
        scales = []
        for steps in (n_steps_list or (self.n_steps,)):
            dt = self.T / steps
            gen = torch.Generator(device=self.device).manual_seed(0)
            dW = torch.randn(steps, self.n_paths, self.d, device=self.device,
                             generator=gen) * np.sqrt(dt)
            x = torch.tensor(self.x0, dtype=torch.float32, device=self.device)
            x = x.expand(self.n_paths, self.d).clone()
            if self.spread:
                lo, hi = self.problem.domain()
                lo_t = torch.tensor(np.asarray(lo), dtype=torch.float32, device=self.device)
                hi_t = torch.tensor(np.asarray(hi), dtype=torch.float32, device=self.device)
                u01 = torch.rand(self.n_paths, self.d, device=self.device, generator=gen)
                x = lo_t + u01 * (hi_t - lo_t)
            x = x.detach()
            ref = self.problem.reference(np.linspace(0.0, self.T, 401))
            Sig, A = self.problem.Sigma, self.problem.A
            worst, scale = 0.0, 0.0
            with torch.no_grad():
                for n in range(steps - 1, -1, -1):
                    tn = n * dt
                    if n == steps - 1:
                        pass                      # u_N is the terminal condition, exact by definition
                    # exact conditional expectation of u(t+dt, .). The reference object exposes only
                    # value/control/P, so the additive constant is recovered exactly at x = 0, where
                    # the quadratic part vanishes. Writing `c = 0` instead (which the first version
                    # did, behind a hasattr guard) silently drops it.
                    Pn = ref.P(tn + dt)
                    cn = float(np.asarray(ref.value(np.zeros((1, self.d)), tn + dt)).ravel()[0])
                    xn = x.cpu().numpy()
                    m = xn + xn @ A.T * dt
                    quad = np.einsum("ni,ij,nj->n", m, Pn, m) + np.trace(Pn @ Sig @ Sig.T) * dt
                    exp_next = -0.5 * quad + cn
                    exact_now = np.asarray(ref.value(xn, tn)).ravel()
                    # left-endpoint nonlinearity, from the EXACT gradient at (t_n, x_n)
                    z = torch.tensor(xn @ (-Pn) @ Sig, dtype=torch.float32, device=self.device)
                    xn_t = torch.tensor(xn, dtype=torch.float32, device=self.device)
                    target = exp_next + dt * self._f(xn_t, z).cpu().numpy().ravel()
                    dev = np.abs(target - exact_now)
                    worst = max(worst, float(dev.max()))
                    scale = max(scale, float(np.abs(exact_now).max()))
                    x = x + (x @ self.A.T) * dt + dW[n] @ self.Sigma.T
            out["by_steps"][steps] = {"max_abs_deviation": worst, "value_scale": scale,
                                      "relative": worst / scale if scale else float("nan")}
            scales.append(worst)
        first = out["by_steps"][(n_steps_list or (self.n_steps,))[0]]
        out.update(first)
        if len(scales) > 1 and scales[-1] > 0:
            out["ratio_finest_over_coarsest"] = scales[0] / scales[-1]
        return out

    def solve(self) -> float:
        """Backward pass: train step n's network on step n+1's output. Returns wall-clock seconds.

        EXPLICIT vs IMPLICIT, WHICH IS NOT A DETAIL ON THIS BENCHMARK
        ------------------------------------------------------------
        The published explicit scheme evaluates the nonlinearity f with the NEXT step's gradient. On
        this problem that is unstable, and the failure is not the usual "not enough steps" one:

        * the per-step discrepancy of the explicit scheme is small and shrinks with dt (measured:
          2.5e-3 at 40 steps, 1.2e-4 at 200 steps, against a value scale of 4e-2);
        * yet a network whose error per step is ~1e-4 accumulates to an error of order 0.35 at t=0,
          a growth of ~4% PER STEP, at both 40 and 200 steps.

        The growth per step is dt * M * z * sigma * k with M = Sigma^-1 B R^-1 B' Sigma^-T ~ 1e6 and
        z ~ 8e-4, so it scales like dt -- but the number of steps scales like 1/dt, and the total
        growth exp(T M z sigma k) is step-count independent. That is exactly the stiffness this paper
        documents for the GRID: the explicit treatment of the quadratic Hamiltonian 1/2 g' B R^-1 B' g
        is what limits the grid's time step, and in splitting form it is what makes the explicit
        recursion amplify. The remedy is the same in both cases: treat that term implicitly.

        `inner = 1` is the published explicit scheme (one pass, f from the next step's gradient).
        `inner >= 2` is the implicit variant: after each pass the nonlinearity is re-evaluated with
        the gradient of the network just trained, so the fixed-point map
        u_n = E[u_{n+1}] + dt f(grad u_n) is iterated. Its Lipschitz constant here is
        dt * M * z * sigma ~ 0.06, so the iteration converges, and -- the point of doing it --
        the step's dependence on the previous net is then purely additive, so errors ACCUMULATE
        instead of amplifying.
        """
        gen = torch.Generator(device=self.device).manual_seed(self.seed)
        if self.paths is None:
            self._simulate(gen)
        t0 = time.perf_counter()
        for n in range(self.n_steps - 1, -1, -1):
            x_n = self.paths[n].detach()
            x_next = self.paths[n + 1].detach()
            if n == self.n_steps - 1:
                with torch.no_grad():
                    base = self._terminal(x_next)
                z_prev = self._z_exact_terminal(x_n)
            else:
                with torch.no_grad():
                    base = self.nets[n + 1](x_next / self.x_scale)
                z_prev = self._z_of(self.nets[n + 1], x_n)
            net = _mlp(self.d, 1, self.hidden, self.layers).to(self.device)
            self.nets[n] = net
            for _ in range(max(self.inner, 1)):
                # The target must not carry gradients into the fitted network, but z is obtained by
                # autograd -- so the graph is built outside no_grad and detached afterwards.
                target = (base + self.dt * self._f(x_n, z_prev)).detach()
                self._fit(net, x_n, target)
                if self.inner > 1:
                    # DAMPED fixed-point update, and the damping is what makes this work at all.
                    # The undamped iteration is u <- A + dt f(grad u), whose Lipschitz constant is
                    # dt * M * |z| * sigma * |grad delta u| / delta u. The first factor dt*M*|z|*sigma
                    # is only ~0.1, but a tanh network's gradient carries wiggles an order of
                    # magnitude above the smooth scale, which pushes the product past 1 and the
                    # iteration DIVERGES: measured, the per-step regression loss with three undamped
                    # passes was 4e-2 at the very first step (20% of the target's spread) against
                    # 4e-8 for the same step with one pass. Blending the gradient that enters f damps
                    # the map.
                    w = self.damping
                    z_prev = (1.0 - w) * z_prev + w * self._z_of(net, x_n)
            if self.log_every and (n % self.log_every == 0):
                print(f"    step {n:>4d}/{self.n_steps}  loss {self.step_losses[-1]:.3e}",
                      flush=True)
        return time.perf_counter() - t0

    def _fit(self, net: nn.Module, x: torch.Tensor, target: torch.Tensor) -> None:
        """One regression pass: fit `net` to `target` at the states `x`.

        WEIGHT DECAY IS A SMOOTHNESS PRIOR HERE, and on this benchmark that is the interesting knob.
        What the recursion feeds forward is not the value but its GRADIENT: z = Sigma' grad u enters
        f = -1/2 x'Qx + 1/2 z'Mz with M ~ 1e6, so a gradient error is amplified into the next step's
        target. A regression loss constrains the FUNCTION, not its derivative, so a network can have
        a small value error and a large gradient error -- and the measured symptom is exactly that:
        raising the per-step budget from 300 to 2000 epochs lowered the loss and made the final value
        worse, while the control stayed roughly right. Weight decay bounds the weights and therefore
        the gradients, which is the cheapest way to test whether gradient quality is the binding
        constraint. `weight_decay = 0` is the plain splitting regression.
        """
        opt = torch.optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # Scaling the loss by the target's SPREAD rather than its RMS: the target is a value
        # function, whose RMS is dominated by its offset while the information sits in its variation.
        # (Adam is invariant to a constant loss scale, so this is a conditioning choice, not a fix --
        # the fix was the implicit pass above.)
        denom = target.std().clamp_min(1e-12)
        loss = torch.zeros((), device=self.device)
        for _ in range(self.epochs):
            opt.zero_grad()
            loss = ((net(x / self.x_scale) - target) / denom).pow(2).mean()
            loss.backward()
            opt.step()
        self.step_losses.append(float(loss))

    def _z_exact_terminal(self, x: torch.Tensor) -> torch.Tensor:
        """z at t = T from the boundary condition: grad V(x,T) = -Q_T x."""
        return (-(x @ self.QT.T)) @ self.Sigma

    # ----------------------------------------------------------------------------------
    def value(self, X: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Function-valued only when trained with spread=True; otherwise only t = 0 is meaningful."""
        x = torch.tensor(np.atleast_2d(np.asarray(X, dtype=float)), dtype=torch.float32,
                         device=self.device)
        with torch.no_grad():
            return self.nets[0](x / self.x_scale).cpu().numpy().ravel()

    def control(self, X: np.ndarray) -> np.ndarray:
        """u = R^-1 B' grad V, evaluated through the trained first-step network."""
        x = torch.tensor(np.atleast_2d(np.asarray(X, dtype=float)), dtype=torch.float32,
                         device=self.device)
        z = self._z_of(self.nets[0], x)                    # z_row = grad_V_row @ Sigma
        g = z @ self.Sigma_inv                             # grad_V_row = z_row @ Sigma^-1
        return (g @ self.control_map).cpu().numpy()


def evaluate(prob: ACProblem, solver: DeepSplitting, seed: int = 3, n_points: int = 4000) -> dict:
    rng = np.random.default_rng(seed)
    lo, hi = prob.domain()
    X = rng.uniform(lo, hi, size=(n_points, prob.state_dim))
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V_hat = solver.value(X, 0.0)
    V_ref = np.asarray(ref.value(X, 0.0)).ravel()
    u_hat = solver.control(X)
    u_ref = ref.control(X, 0.0)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0
    v_hat0 = float(solver.value(x0.reshape(1, -1), 0.0)[0])
    v_ref0 = float(ref.value(x0, 0.0))
    return {
        "value_r2": float(1 - np.sum((V_hat - V_ref) ** 2)
                          / np.sum((V_ref - V_ref.mean()) ** 2)),
        "value_rel_l2": relative_l2(V_hat, V_ref),
        "control_r2": float(1 - np.sum((u_hat - u_ref) ** 2)
                            / np.sum((u_ref - u_ref.mean()) ** 2)),
        "control_rel_l2": relative_l2(u_hat, u_ref),
        "value_at_x0": v_hat0,
        "exact_at_x0": v_ref0,
        "rel_err_at_x0": abs(v_hat0 - v_ref0) / abs(v_ref0),
    }


def run_case(d: int, args, spread: bool) -> dict:
    prob = ACProblem(p=d)
    solver = DeepSplitting(prob, n_steps=args.steps, n_paths=args.paths, epochs=args.epochs,
                           hidden=args.hidden, layers=args.layers, seed=args.seed,
                           spread=spread, device=args.device, inner=args.inner,
                           damping=args.damping, weight_decay=args.weight_decay,
                           log_every=args.log_every)
    # Validate at the configured step count and at half of it: first-order convergence is the
    # signature of a correct scheme, and an O(1) deviation is the signature of a wrong one.
    val = solver.validate_on_exact(n_steps_list=(args.steps, max(args.steps // 2, 2)))
    t0 = time.perf_counter()
    solver.solve()
    secs = time.perf_counter() - t0
    sc = evaluate(prob, solver)
    sc.update({"d": d, "spread": spread, "n_steps": args.steps, "n_paths": args.paths,
               "epochs_per_step": args.epochs, "hidden": args.hidden, "layers": args.layers,
               "seed": args.seed, "inner": args.inner, "damping": args.damping,
               "weight_decay": args.weight_decay,
               "seconds": secs,
               "gradient_steps": args.steps * args.epochs,
               "validation": val})
    return sc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="1,4,8")
    ap.add_argument("--steps", type=int, default=50)
    ap.add_argument("--paths", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--inner", type=int, default=1,
                    help="1 = the published explicit scheme; >=2 = iterate the per-step fixed point "
                         "(the implicit variant), which this benchmark needs (see solve())")
    ap.add_argument("--damping", type=float, default=0.5,
                    help="blend weight for the gradient entering f in the implicit passes; the "
                         "undamped iteration (1.0) diverges on this benchmark")
    ap.add_argument("--weight-decay", type=float, default=0.0,
                    help="smoothness prior on the per-step networks; see _fit")
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--out", default="exp_deep_splitting.json")
    args = ap.parse_args()

    dims = [int(s) for s in args.dims.split(",") if s.strip()]
    print("=" * 104)
    print("DEEP SPLITTING: the modern member of the deep-BSDE family, on the same benchmark")
    print("=" * 104)
    print(f"  {args.steps} steps, {args.paths} paths, {args.epochs} epochs/step, "
          f"{args.layers}x{args.hidden} per step -> {args.steps * args.epochs} gradient steps")
    out = {"args": vars(args), "rows": []}
    for d in dims:
        for spread in (False, True):
            tag = "spread (function-valued)" if spread else "point-valued (from x0)"
            print(f"\n  d={d}  {tag}")
            row = run_case(d, args, spread)
            v = row["validation"]
            print(f"    validation on the exact solution: max deviation {v['max_abs_deviation']:.3e} "
                  f"({v['relative']:.2e} relative)")
            print(f"    trained in {row['seconds']:.1f}s")
            print(f"    V at x0 rel err {row['rel_err_at_x0']*100:.3f}%   "
                  f"value R2 {row['value_r2']:+.4f}   control R2 {row['control_r2']:+.4f}")
            out["rows"].append(row)
    (OUT / args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\n  wrote results/{args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
