"""
Policy-iteration PINN (PI-PINN), a faithful PyTorch reimplementation of the scheme in
`HeeJunYang09/hji-pi-pinn-repro`.

Why a reimplementation rather than a call into their code
---------------------------------------------------------
Their repository is JAX (jax==0.4.30, jaxlib, optax) and runs on JAX-specified problems
(a 2D path-planning HJI and an N=3/5/11 publisher-subscriber game). Two obstacles made running
it directly impossible here:

1. **JAX is not installed and the link cannot install it.** Outbound TLS to GitHub works only
   through Python's own OpenSSL (git's Schannel backend fails outright), at roughly 10 KB/s.
   A CPU jaxlib wheel is tens of megabytes; the CUDA build is far larger.
2. Even with JAX, their problems are not ours.

So this module reproduces their *algorithm* on our benchmark. What is taken verbatim from their
source, with file references:

  * Architecture: a sine-activation MLP, `layers = [N+1, 64, 64, 64, 64, 1]`, Glorot-uniform
    weights, zero biases (`hji_pi_pinn/core/models.py`).
  * Optimiser: Adam at lr=1e-3 (`configs/ps_default.yaml`, `scripts/train_ps.py`).
  * Collocation: `Ni` points resampled every `refresh_every` epochs.
  * **The PI structure** (`scripts/train_ps.py`, lines 157-195): an outer loop over
    `num_iters`; inside it `num_epochs` optimiser steps with the policy *frozen*; at the end of
    each outer iteration, `policy = policy_update(params, batch)` -- i.e. policy improvement
    re-evaluates the Hamiltonian maximiser at the current value function's gradient.
  * The direct baseline is given the **same total number of optimiser steps**
    (`num_iters * num_epochs`), so PI is not advantaged by budget (`scripts/train_ps.py`,
    line 198: `total_epochs = num_iters * num_epochs`).

What this means mathematically, and why it should help
------------------------------------------------------
Freezing the policy makes the PDE that is being fitted *linear*: the quadratic term
`(grad V)' B R^-1 B' (grad V)` of the true HJB is replaced by the linear term
`(grad V)' B u_frozen`, with `u_frozen` held constant. Fitting a linear PDE is a much better
conditioned optimisation problem than fitting the HJB with its quadratic-in-gradient
Hamiltonian. The policy is then improved once per outer iteration. This is the same two-level
structure as classical Howard policy iteration, and the same structure as our grid
`PolicyIterationSolver` -- the experiment is therefore a clean 2x2: {grid, network} x
{nonlinear-direct, linear-policy-iteration}.

Budget note: their default is `num_iters=500 x num_epochs=5000 = 2.5e6` optimiser steps, about
three orders of magnitude more than the 6e3-8e3 used in our sweep, and roughly 10-40 hours per
run at our measured throughput. We therefore run PI-PINN at a *matched* total budget against the
direct PINN, and sweep the outer-loop split so that the comparison isolates the PI structure
rather than the budget. Any claim here is about PI at equal cost, not about PI at their settings.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from ac_problem import ACProblem


def _sin_mlp(in_dim: int, hidden: int, layers: int) -> nn.Sequential:
    """Sine-activation MLP with Glorot-uniform weights and zero biases.

    Matches `hji_pi_pinn/core/models.py`: every hidden layer is `sin(Wx + b)`, the output layer
    is linear, and biases start at zero.
    """
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden)]
    for _ in range(layers - 1):
        mods.append(nn.Linear(hidden, hidden))
    mods.append(nn.Linear(hidden, 1))
    net = nn.ModuleList(mods)

    class _SinNet(nn.Module):
        def __init__(self, blocks: nn.ModuleList):
            super().__init__()
            self.blocks = blocks

        def forward(self, z: torch.Tensor) -> torch.Tensor:
            x = z
            for i, lin in enumerate(self.blocks):
                x = lin(x)
                if i < len(self.blocks) - 1:
                    x = torch.sin(x)
            return x

    n = _SinNet(net)
    for m in n.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)      # Glorot uniform
            nn.init.zeros_(m.bias)
    return n


class PIPINN:
    """Policy-iteration PINN for the LQ optimal-execution HJB.

    Exposes the same interface as the other solvers: `solve()`, `value(X, t)`, `control(X, t)`.
    """

    name = "PI-PINN"

    def __init__(
        self,
        problem: ACProblem,
        hidden: int = 64,
        layers: int = 4,
        n_points: int | None = None,
        num_iters: int = 20,
        num_epochs: int = 300,
        refresh_every: int = 500,
        lr: float = 1e-3,
        domain_pad: float = 0.25,
        device: str = "cuda",
        seed: int = 0,
        terminal_weight: float = 10.0,
        time_bias: float = 3.0,
        log_every: int = 0,
        verbose: bool = False,
    ):
        self.problem = problem
        self.hidden = hidden
        self.layers = layers
        self.num_iters = num_iters
        self.num_epochs = num_epochs
        self.refresh_every = max(1, refresh_every)
        self.lr = lr
        self.device = torch.device(
            device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.seed = seed
        self.terminal_weight = terminal_weight
        self.time_bias = time_bias
        self.log_every = log_every
        self.verbose = verbose
        self.solve_time: float | None = None
        self.n_policy_updates = 0
        self.total_steps = num_iters * num_epochs

        # Their config uses Ni = 1000 * N^2; we keep the scaling but cap it, because the cap is
        # a budget choice that must be the same for every method we compare against.
        d = problem.state_dim
        default_pts = int(np.clip(1000 * d * d, 512, 4096))
        self.n_points = int(n_points or default_pts)

        torch.manual_seed(seed)
        np.random.seed(seed)

        lo, hi = problem.domain()
        self.lo = lo - domain_pad
        self.hi = hi + domain_pad
        self.center = (self.hi + self.lo) / 2.0
        self.halfwidth = (self.hi - self.lo) / 2.0
        x_max = float(np.max(np.abs(self.hi)))
        self.V_scale = max(0.5 * float(np.max(np.abs(problem.QT))) * x_max**2, 1e-8)

        self.net = _sin_mlp(d + 1, hidden, layers).to(self.device)

        self.Q = torch.tensor(problem.Q, dtype=torch.float32, device=self.device)
        self.R = torch.tensor(problem.R, dtype=torch.float32, device=self.device)
        self.R_inv = torch.tensor(problem.R_inv, dtype=torch.float32, device=self.device)
        self.A = torch.tensor(problem.A, dtype=torch.float32, device=self.device)
        self.B = torch.tensor(problem.B, dtype=torch.float32, device=self.device)
        self.QT = torch.tensor(problem.QT, dtype=torch.float32, device=self.device)
        # Per-axis diffusion 1/2 tr(Sigma Sigma' grad^2 V) = 1/2 sum_i D_i d2V/dx_i^2.
        # A single sigma^2 is correct only when every coordinate diffuses equally, which the
        # signal extension breaks (signal axes diffuse at 2 kappa).
        self.D = torch.tensor(np.diag(problem.Sigma @ problem.Sigma.T),
                              dtype=torch.float32, device=self.device)
        self.T = float(problem.T)
        self.residual_scale = 1.0

    # ----------------------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(self.lo, dtype=torch.float32, device=self.device)
        hw = torch.tensor(self.halfwidth, dtype=torch.float32, device=self.device)
        ct = torch.tensor(self.center, dtype=torch.float32, device=self.device)
        xn = (x - ct) / hw
        tn = t / self.T * 2.0 - 1.0
        return torch.cat([xn, tn], dim=1)

    def _value(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(self._encode(x, t)) * self.V_scale

    def _sample(self, n: int, gen: torch.Generator):
        lo = torch.tensor(self.lo, dtype=torch.float32, device=self.device)
        hi = torch.tensor(self.hi, dtype=torch.float32, device=self.device)
        x = lo + (hi - lo) * torch.rand(n, self.problem.state_dim, device=self.device,
                                        generator=gen)
        u = torch.rand(n, 1, device=self.device, generator=gen)
        t = (1.0 - (1.0 - u) ** self.time_bias) * self.T
        return x, t

    def _grad_and_lap(self, V: torch.Tensor, x: torch.Tensor):
        """Return (grad_x V, Hessian diagonal).

        The DIAGONAL, shape (N, d), not the trace: the HJB needs
        1/2 sum_i D_i d2V/dx_i^2 with D = diag(Sigma Sigma'), so the per-axis second derivatives
        must reach the caller separately and be contracted with D there.
        """
        d = self.problem.state_dim
        g = torch.autograd.grad(V.sum(), x, create_graph=True)[0]
        cols = []
        for i in range(d):
            d2 = torch.autograd.grad(g[:, i].sum(), x, create_graph=True, retain_graph=True)[0]
            cols.append(d2[:, i : i + 1])
        return g, torch.cat(cols, dim=1)

    # ----------------------------------------------------------------------------------
    # Losses: direct (nonlinear HJB) and PI (linear, policy frozen)
    # ----------------------------------------------------------------------------------
    def _loss_direct(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Residual of the true HJB -- the quadratic-in-gradient Hamiltonian is kept."""
        V = self.net(self._encode(x, t)) * self.V_scale
        g, hess = self._grad_and_lap(V, x)
        # u* = R^{-1} B' grad V; B multiplies grad V on the right and is not the identity in
        # general, so it must appear explicitly.
        u = (g @ self.B) @ self.R_inv.T
        xq = torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1)
        ur = torch.einsum("ni,ij,nj->n", u, self.R, u).unsqueeze(1)
        ax = torch.einsum("ni,ij,nj->n", g, self.A, x).unsqueeze(1)
        # (grad V)'(B u): form B u before contracting with grad V.
        bu = torch.einsum("ni,ni->n", g, u @ self.B.T).unsqueeze(1)
        H = -0.5 * (xq + ur) + ax + bu
        dVdt = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0]
        return dVdt + H + 0.5 * (hess @ self.D).unsqueeze(1)

    def _loss_pi(self, x: torch.Tensor, t: torch.Tensor, policy: torch.Tensor) -> torch.Tensor:
        """Residual of the *linear* PDE obtained by freezing the policy.

        This is the whole point of the method: the term (grad V)' B u is linear in grad V once u
        is fixed, so the quadratic Hamiltonian of `_loss_direct` disappears from the gradient
        landscape. Training on this is the policy-evaluation half of policy iteration.
        """
        V = self.net(self._encode(x, t)) * self.V_scale
        g, hess = self._grad_and_lap(V, x)
        xq = torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1)
        ur = torch.einsum("ni,ij,nj->n", policy, self.R, policy).unsqueeze(1)
        ax = torch.einsum("ni,ij,nj->n", g, self.A, x).unsqueeze(1)
        # The advection velocity is B * policy, not the policy itself: the policy is a trading
        # rate (dimension m) while the state has dimension d, and they coincide only when
        # B = I. An earlier version contracted grad V with the raw policy.
        bu = torch.einsum("ni,ni->n", g, policy @ self.B.T).unsqueeze(1)
        H = -0.5 * (xq + ur) + ax + bu
        dVdt = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0]
        return dVdt + H + 0.5 * (hess @ self.D).unsqueeze(1)

    def _zero_policy(self, x: torch.Tensor) -> torch.Tensor:
        """The initial admissible policy: no trading, shaped (N, control_dim).

        Deliberately NOT `torch.zeros_like(x)`. The policy is a trading rate per asset, so it has
        the CONTROL dimension, while x has the STATE dimension -- equal only while the benchmark is
        inventory-only. With the signal the state is (inventory, signal) and the two differ, at
        which point `zeros_like(x)` produces a control of the wrong width.
        """
        return torch.zeros(x.shape[0], self.problem.control_dim, device=self.device)

    def _policy_update(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Policy improvement: u = R^-1 B' grad V at the current parameters.

        The policy must be evaluated at each point's *own* time, so that the frozen policy used
        in the next policy-evaluation sweep is a function of (x, t). An earlier version of this
        evaluated every point at a fixed t = T/2, which makes the frozen policy time-independent
        and leaves the policy-evaluation PDE inconsistent with the batch it is trained on.
        """
        xr = x.detach().clone().requires_grad_(True)
        tr = t.detach().clone()
        V = self._value(xr, tr)
        g = torch.autograd.grad(V.sum(), xr)[0]
        return ((g @ self.B) @ self.R_inv.T).detach()

    def _policy_from(self, state: dict, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the policy from a *frozen* parameter set, then restore the live one.

        Their loop derives the frozen policy from the previous outer iteration's parameters, so
        the policy must be computable from a stored state dict rather than only from the live
        network.
        """
        live = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
        self.net.load_state_dict(state)
        try:
            with torch.enable_grad():
                return self._policy_update(x, t)
        finally:
            self.net.load_state_dict(live)

    def _terminal_loss(self, gen: torch.Generator) -> torch.Tensor:
        n = max(64, self.n_points // 4)
        lo = torch.tensor(self.lo, dtype=torch.float32, device=self.device)
        hi = torch.tensor(self.hi, dtype=torch.float32, device=self.device)
        x = lo + (hi - lo) * torch.rand(n, self.problem.state_dim, device=self.device,
                                        generator=gen)
        t = torch.full((n, 1), self.T, device=self.device)
        V = self._value(x, t)
        V_true = -0.5 * torch.einsum("ni,ij,nj->n", x, self.QT, x).unsqueeze(1)
        return ((V - V_true) ** 2).mean() / (self.V_scale**2)

    # ----------------------------------------------------------------------------------
    def solve(self, direct: bool = False) -> float:
        """Run PI (default) or the direct baseline at the same total step count."""
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        # Residual normalisation on a fixed probe batch, so the two modes are scaled alike.
        x0, t0 = self._sample(512, gen)
        x0 = x0.clone().requires_grad_(True); t0 = t0.clone().requires_grad_(True)
        with torch.enable_grad():
            r0 = (self._loss_direct(x0, t0) if direct
                  else self._loss_pi(x0, t0, self._zero_policy(x0))).detach()
            self.residual_scale = float(r0.pow(2).mean().sqrt().clamp_min(1e-12))
        del r0, x0, t0

        x, t = self._sample(self.n_points, gen)
        policy = self._zero_policy(x)
        if not direct:
            with torch.enable_grad():
                policy = self._policy_update(x, t)

        def epoch_step(batch_x, batch_t, pol):
            opt.zero_grad(set_to_none=True)
            bx = batch_x.detach().clone().requires_grad_(True)
            bt = batch_t.detach().clone().requires_grad_(True)
            res = (self._loss_direct(bx, bt) if direct
                   else self._loss_pi(bx, bt, pol))
            loss = (res / self.residual_scale).pow(2).mean()
            loss = loss + self.terminal_weight * self._terminal_loss(gen)
            loss.backward()
            opt.step()
            return float(loss.detach())

        t_start = time.perf_counter()
        step = 0
        last = 0.0
        if direct:
            # Same total number of optimiser steps as PI, matching their budget-parity design
            # (`scripts/train_ps.py:198`, total_epochs = num_iters * num_epochs).
            for _ in range(self.total_steps):
                if self.refresh_every and step > 0 and step % self.refresh_every == 0:
                    x, t = self._sample(self.n_points, gen)
                last = epoch_step(x, t, None)
                step += 1
        else:
            # Their PI loop, transcribed from `scripts/train_ps.py:172-190`:
            #   * `frozen` is the parameter set from the PREVIOUS outer iteration, captured at
            #     the top of each outer iteration;
            #   * whenever the collocation batch is refreshed mid-iteration, the policy is
            #     re-evaluated on the NEW batch using `frozen`, so the policy *function* is held
            #     fixed while its values track the current points;
            #   * at the end of each outer iteration the policy is improved from the current
            #     parameters on the current batch.
            # Two earlier versions of this loop were wrong and produced meaningless numbers: one
            # evaluated every point at a fixed t = T/2, and one kept a stale policy array across
            # a batch refresh. Both are the network analogue of storing a single time- and
            # point-independent control on the grid side -- the same mistake as one of the
            # baseline defects this paper documents.
            frozen = None
            for _ in range(self.num_iters):
                for _ in range(self.num_epochs):
                    ep = step % self.num_epochs
                    if self.refresh_every and ep > 0 and ep % self.refresh_every == 0:
                        x, t = self._sample(self.n_points, gen)
                        policy = (self._zero_policy(x) if frozen is None
                                  else self._policy_from(frozen, x, t))
                    last = epoch_step(x, t, policy)
                    step += 1
                frozen = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                policy = self._policy_from(frozen, x, t)
                self.n_policy_updates += 1
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.solve_time = time.perf_counter() - t_start
        return self.solve_time

    # ----------------------------------------------------------------------------------
    def value(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        tt = torch.full((X.shape[0], 1), float(t), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self._value(xt, tt).cpu().numpy().ravel()

    def control(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        tt = torch.full((X.shape[0], 1), float(t), dtype=torch.float32, device=self.device)
        V = self._value(xt, tt)
        g = torch.autograd.grad(V.sum(), xt)[0]
        return ((g @ self.B) @ self.R_inv.T).detach().cpu().numpy()


if __name__ == "__main__":
    from solvers import accuracy_metrics

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PI-PINN smoke test on {dev}")
    print("=" * 88)
    prob = ACProblem(p=2, hetero=True, het_seed=1)
    for label, direct, (ni, ne) in (("PI-PINN", False, (20, 300)),
                                    ("Direct PINN", True, (20, 300))):
        s = PIPINN(prob, num_iters=ni, num_epochs=ne, device=dev, seed=0)
        st = s.solve(direct=direct)
        m = accuracy_metrics(s, prob, n_points=400, n_times=5, seed=3)
        print(f"  {label:14s} steps={s.total_steps:6d} t={st:7.1f}s "
              f"value_R2={m['value_r2']:+.5f} ctrl_R2={m['control_r2']:+.5f} "
              f"sign={m['control_sign_agreement']:.3f} "
              f"policy_updates={s.n_policy_updates}")
