"""
Physics-informed neural network (PINN) solver for the LQ optimal-execution HJB.

This is the deep-learning arm of the comparison. It is trained on the HJB residual
    r(x,t) = dV/dt + H_max(x, grad V, t) + 1/2 tr(Sigma Sigma' d2V)   ->  0
with the terminal condition V(x,T) = -1/2 x' Q_T x imposed as a soft penalty.

Notes on faithfulness to the original work
------------------------------------------
* The Hamiltonian maximisation uses the *same* analytic maximiser as the grid solvers,
  ``u* = R^{-1} B' grad V`` (valid in closed form for the LQ problem). All methods are
  therefore compared on the same HJB, differing only in how the value function is
  represented and how the PDE is enforced.
* The original work's headline methodological claims -- "explicit Taylor series
  approximation", "matrix-based derivative evaluation", and a "spectral basis"
  representation -- have no implementation in the released code (verified by search:
  no Taylor/expansion, no Fourier/Chebyshev; ``utils/HopeGrad.py:86-118`` is a plain loop
  of ``torch.autograd.grad`` calls). This solver therefore does not claim to implement
  SHoP; it implements a standard PINN and, separately, benchmarks derivative-computation
  strategies as a function of derivative order and state dimension, which is the
  question the thesis's scalability argument actually turns on.

Input scaling
-------------
The state is mapped from the padded box domain to [-1, 1] and time to [0, 1] before the
network. This is deliberate: the original system trained on a normalised price domain
[20, 30] and then evaluated on raw prices of 40 to 550, i.e. up to 18x outside the
training range, which produced spurious cross-sectional "findings". Scaling is made an
explicit, inspectable part of the solver here.
"""

from __future__ import annotations

import time

import numpy as np
import torch
import torch.nn as nn

from ac_problem import ACProblem


def _mlp(in_dim: int, hidden: int, layers: int, activation: str = "tanh") -> nn.Sequential:
    act = {"tanh": nn.Tanh, "silu": nn.SiLU, "gelu": nn.GELU}[activation]
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), act()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), act()]
    mods += [nn.Linear(hidden, 1)]
    net = nn.Sequential(*mods)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class PINNSolver:
    """PINN for the LQ optimal-execution HJB.

    Parameters
    ----------
    deriv_mode : {'reverse_over_reverse', 'forward_over_reverse'}
        How the Laplacian is evaluated. ``reverse_over_reverse`` uses p nested
        ``autograd.grad`` calls; ``forward_over_reverse`` contracts the Hessian trace
        with forward-mode AD via ``torch.func``, which is typically cheaper for small p
        and is the modern alternative to the naive reverse-over-reverse recursion.
    """

    name = "PINN"

    def __init__(
        self,
        problem: ACProblem,
        hidden: int = 64,
        layers: int = 4,
        activation: str = "tanh",
        n_collocation: int = 2048,
        n_terminal: int = 256,
        epochs: int = 4000,
        lr: float = 1e-3,
        domain_pad: float = 0.25,
        device: str = "cuda",
        seed: int = 0,
        deriv_mode: str = "reverse_over_reverse",
        terminal_weight: float = 10.0,
        hard_terminal: bool = True,
        time_bias: float = 3.0,
        log_every: int = 0,
        verbose: bool = False,
    ):
        self.problem = problem
        self.hidden = hidden
        self.layers = layers
        self.activation = activation
        self.n_collocation = n_collocation
        self.n_terminal = n_terminal
        self.epochs = epochs
        self.lr = lr
        self.domain_pad = domain_pad
        self.seed = seed
        self.deriv_mode = deriv_mode
        self.terminal_weight = terminal_weight
        self.hard_terminal = hard_terminal
        self.time_bias = time_bias
        self.log_every = log_every
        self.verbose = verbose
        self.solve_time: float | None = None
        self.history: list[tuple[int, float]] = []
        self.residual_scale: float | None = None

        if device == "cuda" and not torch.cuda.is_available():
            device = "cpu"
        self.device = torch.device(device)

        torch.manual_seed(seed)
        np.random.seed(seed)

        d = problem.state_dim
        lo, hi = problem.domain()
        self.lo = lo - domain_pad
        self.hi = hi + domain_pad
        self.halfwidth = (self.hi - self.lo) / 2.0
        self.center = (self.hi + self.lo) / 2.0

        # Output scaling: |V| is O(0.5 * |Q_T| * max|x|^2); scaling improves conditioning.
        x_max = float(np.max(np.abs(self.hi)))
        self.V_scale = max(0.5 * float(np.max(np.abs(problem.QT))) * x_max**2, 1e-8)

        self.net = _mlp(d + 1, hidden, layers, activation).to(self.device)

        # Problem constants on device
        self.Q = torch.tensor(problem.Q, dtype=torch.float32, device=self.device)
        self.R = torch.tensor(problem.R, dtype=torch.float32, device=self.device)
        self.R_inv = torch.tensor(problem.R_inv, dtype=torch.float32, device=self.device)
        self.A = torch.tensor(problem.A, dtype=torch.float32, device=self.device)
        self.B = torch.tensor(problem.B, dtype=torch.float32, device=self.device)
        self.QT = torch.tensor(problem.QT, dtype=torch.float32, device=self.device)
        # Per-axis diffusion 1/2 tr(Sigma Sigma' grad^2 V) = 1/2 sum_i D_i d2V/dx_i^2.
        # A single sigma^2 is correct only when every coordinate diffuses equally.
        self.D = torch.tensor(np.diag(problem.Sigma @ problem.Sigma.T),
                              dtype=torch.float32, device=self.device)
        self.T = float(problem.T)

    # ----------------------------------------------------------------------------------
    # Encoding / decoding
    # ----------------------------------------------------------------------------------
    def _encode(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        lo = torch.tensor(self.lo, dtype=torch.float32, device=self.device)
        hw = torch.tensor(self.halfwidth, dtype=torch.float32, device=self.device)
        ct = torch.tensor(self.center, dtype=torch.float32, device=self.device)
        xn = (x - ct) / hw
        tn = (t - 0.0) / self.T * 2.0 - 1.0
        return torch.cat([xn, tn], dim=1)

    def _decode(self, out: torch.Tensor) -> torch.Tensor:
        return out * self.V_scale

    def _sample(self, n: int, generator: torch.Generator, terminal: bool = False):
        lo = torch.tensor(self.lo, dtype=torch.float32, device=self.device)
        hi = torch.tensor(self.hi, dtype=torch.float32, device=self.device)
        x = lo + (hi - lo) * torch.rand(n, self.problem.state_dim, device=self.device,
                                        generator=generator)
        if terminal:
            t = torch.full((n, 1), self.T, device=self.device)
        else:
            # Bias collocation points towards t = T, where |grad V| and hence |u*| are
            # largest (the terminal inventory penalty makes the problem stiff there).
            u = torch.rand(n, 1, device=self.device, generator=generator)
            t = (1.0 - (1.0 - u) ** self.time_bias) * self.T
        return x, t

    # ----------------------------------------------------------------------------------
    # Value function ansatz
    # ----------------------------------------------------------------------------------
    def _V_terminal(self, x: torch.Tensor) -> torch.Tensor:
        """Terminal condition V(x,T) = -1/2 x' Q_T x (scaled units)."""
        return (-0.5 * torch.einsum("ni,ij,nj->n", x, self.QT, x).unsqueeze(1)) / self.V_scale

    def _value_scaled(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Value function in scaled units.

        With ``hard_terminal=True`` the ansatz is
            V(x,t) = V_T(x) + (T - t) * N(x,t)
        so the terminal condition is satisfied *exactly* for any network N, and only the
        PDE residual has to be minimised. This is a standard PINN construction and matters
        here because the terminal penalty is what makes the problem stiff near t = T;
        leaving it as a soft penalty leaves a large residual on which the optimiser stalls.
        """
        n = self.net(self._encode(x, t))
        if not self.hard_terminal:
            return n
        return self._V_terminal(x) + (self.T - t) * n

    # ----------------------------------------------------------------------------------
    # Derivatives
    # ----------------------------------------------------------------------------------
    def _grad_and_lap(self, V: torch.Tensor, x: torch.Tensor):
        """Return (grad_x V, Hessian diagonal).

        Returns the DIAGONAL of the Hessian, shape (N, d), not its trace. The HJB carries
        1/2 tr(Sigma Sigma' grad^2 V) = 1/2 sum_i D_i d2V/dx_i^2 with D = diag(Sigma Sigma'), so
        the per-axis second derivatives have to reach the caller separately: a few lines further
        down they are contracted with D. Returning only the trace and multiplying it by a single
        sigma^2 is correct only when every state coordinate carries the same diffusion, which is
        true for the inventory-only benchmark and false as soon as the signal is present (the
        signal axes diffuse at 2 kappa, four orders of magnitude above inventory).

        The inner derivative must be built with ``create_graph=True`` for the second
        derivative to exist, so this cannot be used under ``torch.no_grad``. Callers that
        only want the value must detach the result (and must not call ``backward``).
        """
        d = self.problem.state_dim
        g = torch.autograd.grad(V.sum(), x, create_graph=True)[0]              # (N, d)
        if self.deriv_mode == "reverse_over_reverse":
            cols = []
            for i in range(d):
                d2 = torch.autograd.grad(
                    g[:, i].sum(), x, create_graph=True, retain_graph=True
                )[0][:, i : i + 1]
                cols.append(d2)
            hess_diag = torch.cat(cols, dim=1)                                 # (N, d)
        elif self.deriv_mode == "forward_over_reverse":
            # Hessian diagonal by forward-mode contraction of the gradient: for each
            # basis vector e_i, jvp(grad V, e_i) = H e_i, and the i-th diagonal entry is
            # (H e_i)_i.
            cols = []
            for i in range(d):
                e = torch.zeros_like(x)
                e[:, i] = 1.0
                _, hvp = torch.autograd.functional.jvp(
                    lambda z: torch.autograd.grad(V.sum(), z, create_graph=True)[0],
                    (x,), (e,), create_graph=True,
                )
                cols.append(hvp[:, i : i + 1])
            hess_diag = torch.cat(cols, dim=1)
        else:
            raise ValueError(f"unknown deriv_mode {self.deriv_mode!r}")
        return g, hess_diag

    def _residual(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        V = self._decode(self._value_scaled(x, t))
        g, hess_diag = self._grad_and_lap(V, x)

        # u* = R^{-1} B' grad V. B multiplies on the right of grad V (not B'), and B appears
        # explicitly rather than being the identity as it is in the inventory-only case.
        u = (g @ self.B) @ self.R_inv.T
        xq = torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1)
        ur = torch.einsum("ni,ij,nj->n", u, self.R, u).unsqueeze(1)
        ax = torch.einsum("ni,ij,nj->n", g, self.A, x).unsqueeze(1)
        # (grad V)'(B u): the B u product must be formed before contracting with grad V.
        bu = torch.einsum("ni,ni->n", g, u @ self.B.T).unsqueeze(1)
        H = -0.5 * (xq + ur) + ax + bu

        # dV/dt
        dVdt = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0]

        # HJB: dV/dt + H + 1/2 sum_i D_i d2V/dx_i^2 = 0
        lap_weighted = (hess_diag @ self.D).unsqueeze(1)
        return dVdt + H + 0.5 * lap_weighted

    # ----------------------------------------------------------------------------------
    # Training
    # ----------------------------------------------------------------------------------
    def solve(self, verbose: bool | None = None) -> float:
        verbose = self.verbose if verbose is None else verbose
        gen = torch.Generator(device=self.device)
        gen.manual_seed(self.seed)

        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01
        )

        # Fixed validation set to monitor the PDE residual.
        xv, tv = self._sample(2048, gen)

        # Normalise the residual by its magnitude at initialisation so the loss is O(1)
        # and the learning rate is not implicitly rescaled by the problem's units.
        with torch.enable_grad():
            x0, t0s = self._sample(1024, gen)
            if x0.shape[0]:
                pass
            x0 = x0.clone().requires_grad_(True)
            t0s = t0s.clone().requires_grad_(True)
            r0 = self._residual(x0, t0s).detach()
            self.residual_scale = float(r0.pow(2).mean().sqrt().clamp_min(1e-12))
            del r0, x0, t0s

        t_start = time.perf_counter()
        for ep in range(1, self.epochs + 1):
            self.net.train()
            opt.zero_grad(set_to_none=True)

            x, t = self._sample(self.n_collocation, gen)
            x = x.clone().requires_grad_(True)
            t = t.clone().requires_grad_(True)
            res = self._residual(x, t)
            loss_pde = (res / self.residual_scale).pow(2).mean()

            loss = loss_pde
            if not self.hard_terminal:
                xt, tt = self._sample(self.n_terminal, gen, terminal=True)
                Vt = self._decode(self._value_scaled(xt, tt))
                Vt_true = -0.5 * torch.einsum("ni,ij,nj->n", xt, self.QT, xt).unsqueeze(1)
                loss = loss + self.terminal_weight * ((Vt - Vt_true) ** 2).mean() / (self.V_scale**2)

            loss.backward()
            opt.step()
            sched.step()

            if self.log_every and ep % self.log_every == 0:
                # The residual needs gradients and hence cannot run under torch.no_grad.
                # We build the graph but never call backward on it, and detach the value.
                xv2 = xv.detach().clone().requires_grad_(True)
                tv2 = tv.detach().clone().requires_grad_(True)
                rv = self._residual(xv2, tv2)
                vl = float((rv.detach() / self.residual_scale).pow(2).mean())
                del rv, xv2, tv2
                self.history.append((ep, vl))
                if verbose:
                    print(f"    ep {ep:6d}  train={float(loss.detach()):.3e}  val_pde={vl:.3e}  "
                          f"lr={sched.get_last_lr()[0]:.2e}")

        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.solve_time = time.perf_counter() - t_start
        return self.solve_time

    # ----------------------------------------------------------------------------------
    # Evaluation
    # ----------------------------------------------------------------------------------
    def value(self, X: np.ndarray, t: float) -> np.ndarray:
        X = np.atleast_2d(X).astype(np.float32)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device)
        tt = torch.full((X.shape[0], 1), float(t), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            V = self._decode(self._value_scaled(xt, tt))
        return V.cpu().numpy().ravel()

    def control(self, X: np.ndarray, t: float) -> np.ndarray:
        """u* = R^{-1} B' grad V, evaluated by autograd on the trained network."""
        X = np.atleast_2d(X).astype(np.float32)
        xt = torch.tensor(X, dtype=torch.float32, device=self.device, requires_grad=True)
        tt = torch.full((X.shape[0], 1), float(t), dtype=torch.float32, device=self.device)
        V = self._decode(self._value_scaled(xt, tt))
        g = torch.autograd.grad(V.sum(), xt)[0]
        u = (g @ self.B) @ self.R_inv.T
        return u.detach().cpu().numpy()


if __name__ == "__main__":
    from solvers import accuracy_metrics

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"PINN smoke test on {dev}")
    print("=" * 78)
    for p in (1, 2):
        prob = ACProblem(p=p)
        s = PINNSolver(prob, hidden=64, layers=4, n_collocation=2048, epochs=6000,
                       lr=2e-3, device=dev, seed=0, log_every=1000, verbose=True)
        st = s.solve()
        m = accuracy_metrics(s, prob, n_points=500, n_times=6, seed=3)
        print(f"  p={p}: train={st:6.2f}s  value_MSE={m['value_mse']:.3e}  "
              f"value_R2={m['value_r2']:+.6f}  ctrl_R2={m['control_r2']:+.6f}  "
              f"sign_agree={m['control_sign_agreement']:.3f}  res_scale={s.residual_scale:.3e}")
        u = s.control(np.full((16, p), 0.5), 0.0)
        print(f"        control at (q=0.5, t=0): mean={u.mean():+.6f}  (reference "
              f"{prob.reference(np.linspace(0,1,101)).control(np.full((16,p),0.5),0.0).mean():+.6f})")
