"""
P1: a deep-BSDE baseline, and the validation that has to come before any comparison.

Why deep BSDE, and why validate it this way
-------------------------------------------
Deep BSDE (Han-E-Jentzen) is THE standard high-dimensional method for this class of problem in
finance, so a comparison paper that omits it is not credible. But it is also a different KIND of
solver from everything else in the paper, and getting that distinction right matters more than the
numbers:

  * the grid and the PINN are FUNCTION-valued: one solve gives V(x,t) over a whole domain;
  * deep BSDE is POINT-valued. It parameterises V(x_0,0) = theta and Z_t = sigma'grad V along the
    simulated paths, and it is trained so that the terminal mismatch vanishes. It therefore returns
    V and the control AT ONE INITIAL STATE, and answering a query at a different state means
    retraining.

That distinction is the whole point of including it: it is the concrete case of the "one problem or
a family" axis the paper already discusses.

THE VALIDATION, WHICH COMES FIRST. Before training anything, the forward recursion is run with the
EXACT theta and the EXACT z. If the driver, the sign convention or the coupling to the state SDE is
wrong, the terminal mismatch will not vanish and the error will be large and independent of every
training setting. This is the same instrument that caught the PINN residuals, and it is much
sharper than "the training loss went down".

Derivation of the driver actually used (so it can be checked rather than trusted). With
z = Sigma' grad V and the optimal control substituted, Ito gives
    dV = [ 1/2 x'Qx + 1/2 grad V' B R^-1 B' grad V ] dt + z' dW,
so the forward recursion is
    V_{n+1} = V_n + [1/2 x_n'Qx_n + 1/2 grad V_n' B R^-1 B' grad V_n] dt + z_n' dW_n
    x_{n+1} = x_n + (A x_n + B u*_n) dt + Sigma dW_n,      u*_n = R^-1 B' grad V_n
driven by the SAME dW, and the terminal condition is g(x) = -1/2 x'Q_T x.

Usage: python src/deep_bsde.py [--epochs N] [--dims 1,8,12]
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

OUT = Path("results")


def _mlp(in_dim: int, out_dim: int, hidden: int, layers: int) -> nn.Module:
    """z has the width of the STATE, not of the input: the input carries (x, t) so in_dim = d + 1
    while z = Sigma'grad V is a d-vector. Conflating the two is invisible when d = 1 and wrong
    for every d > 1."""
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods.append(nn.Linear(hidden, out_dim))
    return nn.Sequential(*mods)


class DeepBSDE:
    """Deep BSDE for the LQ optimal-execution HJB. Point-valued by construction."""

    name = "Deep BSDE"

    def __init__(self, problem: ACProblem, x0: np.ndarray, hidden: int = 64, layers: int = 4,
                 n_steps: int = 50, n_paths: int = 1024, epochs: int = 2000, lr: float = 3e-3,
                 device: str = "cuda", seed: int = 0, log_every: int = 0):
        self.problem = problem
        self.x0 = np.asarray(x0, dtype=float)
        self.n_steps, self.n_paths, self.epochs, self.lr = n_steps, n_paths, epochs, lr
        self.log_every = log_every
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available())
                                  else "cpu")
        torch.manual_seed(seed)
        self.d, self.m = problem.state_dim, problem.control_dim

        t = lambda M: torch.tensor(M, dtype=torch.float32, device=self.device)  # noqa: E731
        self.A, self.B, self.Q, self.R_inv = (t(problem.A), t(problem.B), t(problem.Q),
                                             t(problem.R_inv))
        self.QT, self.Sigma = t(problem.QT), t(problem.Sigma)
        # grad V = Sigma^{-T} z.  Sigma is diagonal here, so this is a reciprocal scaling.
        self.Sigma_inv_T = t(np.linalg.inv(problem.Sigma).T)
        self.T = float(problem.T)
        self.dt = self.T / n_steps

        self.net = _mlp(self.d + 1, self.d, hidden, layers).to(self.device)
        # Start the correction at EXACTLY zero: at initialisation z is then the exact terminal
        # value, so the rollout begins inside the stability region.
        #
        # A "small but nonzero" variant was tried and is WORSE, which is worth recording because it
        # looks like the obvious improvement. Scaling the default init by 0.01 gives a residual of
        # order 64 x 0.00125 ~ 0.08, while the anchor z_T = -Sigma Q_T x is only ~1e-3 -- so the
        # "small" residual is 80x LARGER than the thing it is correcting, the rollout destabilises
        # immediately, and the loss sits flat at 9.0e+04 instead of falling. Smallness has to be
        # measured against the ANCHOR, not against the layer's own default initialisation.
        #
        # The cost of zeroing is that d(loss)/d(W_earlier) carries a factor of the final weights, so
        # only the last layer learns at first. That starvation is self-resolving -- once the output
        # layer leaves zero the earlier layers receive gradient -- which is exactly what the falling
        # loss shows, so the fix is more epochs rather than a larger initial residual.
        last = [m for m in self.net if isinstance(m, nn.Linear)][-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
        # VALUE NORMALISATION, and it is not cosmetic. Here V(x0,0) is about -6e-3 while a plain
        # Adam step on an unnormalised theta at lr=1e-2 moves it by ~1e-2 -- larger than the answer
        # itself, so theta walks away and the run diverges (measured: relative error 2.5e+03, i.e.
        # theta ~ 15 against a true value of -6e-3). Parameterising theta = V_scale * theta_scaled
        # and forming the loss on (V - g)/V_scale puts the target at O(1) and makes the same lr
        # sensible. This is the identical trick the PINN already uses, and the same class of bug:
        # an unscaled quantity optimised with a scale-blind step.
        self.V_scale = max(0.5 * float(np.max(np.abs(problem.QT)))
                           * float(np.max(np.abs(self.x0))) ** 2, 1e-8)
        self.theta = nn.Parameter(torch.zeros(1, device=self.device))
        self.dW: torch.Tensor | None = None

    # ----------------------------------------------------------------------------------
    def _brownian(self, generator: torch.Generator) -> torch.Tensor:
        """Fix one set of increments so the terminal loss is a deterministic function of the
        parameters; common random numbers make the comparison between runs meaningful."""
        if self.dW is None:
            self.dW = torch.randn(self.n_steps, self.n_paths, self.d, device=self.device,
                                  generator=generator) * np.sqrt(self.dt)
        return self.dW

    def _rollout(self, theta: torch.Tensor, grad_fn) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward recursion of (x, V). `grad_fn(x,t)` returns grad V."""
        dW = self.dW
        x = torch.tensor(self.x0, dtype=torch.float32, device=self.device)
        x = x.expand(self.n_paths, self.d).clone()
        V = (self.V_scale * theta).expand(self.n_paths, 1).clone()
        for n in range(self.n_steps):
            tn = n * self.dt
            tt = torch.full((self.n_paths, 1), tn, dtype=torch.float32, device=self.device)
            g = grad_fn(x, tt)                                   # (N, d)
            u = (g @ self.B) @ self.R_inv.T                      # (N, m)
            run = 0.5 * torch.einsum("ni,ij,nj->n", x, self.Q, x).unsqueeze(1) \
                + 0.5 * torch.einsum("ni,ij,nj->n", g, self.B @ self.R_inv @ self.B.T, g
                                     ).unsqueeze(1)
            V = V + run * self.dt + torch.einsum("ni,ni->n", g @ self.Sigma, dW[n]).unsqueeze(1)
            x = x + (x @ self.A.T + u @ self.B.T) * self.dt + dW[n] @ self.Sigma.T
        return x, V

    def _z_terminal(self, x: torch.Tensor) -> torch.Tensor:
        """The EXACT z at t = T, straight from the boundary condition: z = Sigma' grad V and
        grad V(x,T) = -Q_T x, so z_T = -(x Q_T') Sigma.

        This is not information about the answer -- it is the terminal condition the problem
        already states -- and it is the anchor that makes the rollout stable at initialisation.
        """
        return (-(x @ self.QT.T)) @ self.Sigma

    def _grad_net(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """grad V from the parameterised z, as a RESIDUAL on top of the terminal value.

        WHY THE RESIDUAL FORM. The forward recursion carries 1/2 g' B R^-1 B' g dt with
        B R^-1 B' = I/eta, so the per-step growth factor is about |g|^2 dt / eta. With dt = 1/50
        and eta = 0.01 that is |g|^2: the exact |grad V| = |P(t) x| <= 0.1 is comfortably stable,
        but a randomly initialised network outputs |g| >> 1 and the value AND the state both run to
        infinity together -- measured as a terminal loss of ~1.9e+06 that did not decrease at all
        over 900 epochs. Anchoring z at the exact terminal value, and starting the correction at
        exactly zero (see the zeroed output layer in __init__), puts the initial rollout inside the
        stability region, which is what lets the loss start falling. At t = T the residual
        contributes nothing and z is exactly right.
        """
        inp = torch.cat([x / 5.0, t / self.T * 2.0 - 1.0], dim=1)
        z = self._z_terminal(x) + self.net(inp)
        return z @ self.Sigma_inv_T

    # ----------------------------------------------------------------------------------
    def validate_on_exact(self) -> float:
        """Terminal mismatch when theta and grad V are the EXACT ones. Must be ~0."""
        ref = self.problem.reference(np.linspace(0.0, self.T, 401))
        # theta is the SCALED parameter, so the exact value must be divided by V_scale to be
        # comparable. Passing it unscaled makes this check fail loudly for the wrong reason --
        # which is what the check is for, but it cost a round to notice.
        theta = torch.tensor([float(ref.value(self.x0, 0.0)) / self.V_scale],
                             dtype=torch.float32, device=self.device)

        def grad_exact(x, t):
            tt = float(t[0, 0])
            return -torch.tensor(np.atleast_2d(x.detach().cpu().numpy()) @ ref.P(tt),
                                 dtype=torch.float32, device=self.device)

        gen = torch.Generator(device=self.device).manual_seed(0)
        self._brownian(gen)
        with torch.no_grad():
            xT, VT = self._rollout(theta, grad_exact)
        g = -0.5 * torch.einsum("ni,ij,nj->n", xT, self.QT, xT).unsqueeze(1)
        # Returned in ABSOLUTE units. The loss is normalised internally, and a threshold expressed
        # in normalised units was compared against it twice by mistake; the terminal payoff is
        # O(0.05), so an absolute RMSE is the quantity that means something physical.
        return float((((VT - g) / self.V_scale).pow(2).mean()).sqrt()) * self.V_scale

    def solve(self) -> float:
        gen = torch.Generator(device=self.device).manual_seed(0)
        self._brownian(gen)
        opt = torch.optim.Adam(list(self.net.parameters()) + [self.theta], lr=self.lr)
        t0 = time.perf_counter()
        for ep in range(self.epochs):
            # RESAMPLE THE PATHS EVERY EPOCH. Holding them fixed (which is tempting, because it
            # makes the loss trace clean and runs comparable) lets the network absorb a constant
            # error in theta: theta shifts every path's terminal value by the same amount, and a
            # network that can see the same 1024 paths can cancel it. Measured: the terminal loss
            # fell to 4.9e-3 while theta stayed 28x off the exact value. Resampling makes theta
            # identifiable, because a constant offset can no longer be cancelled path by path.
            self.dW = None
            self._brownian(torch.Generator(device=self.device).manual_seed(1_000_000 + ep))
            opt.zero_grad(set_to_none=True)
            xT, VT = self._rollout(self.theta, self._grad_net)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, self.QT, xT).unsqueeze(1)
            loss = ((VT - g) / self.V_scale).pow(2).mean()
            loss.backward()
            opt.step()
            if self.log_every and (ep + 1) % self.log_every == 0:
                print(f"      epoch {ep + 1:>6}  terminal RMSE {float(loss.sqrt()):.4e}")
        self.solve_time = time.perf_counter() - t0
        return self.solve_time

    # ----------------------------------------------------------------------------------
    def value_at_x0(self) -> float:
        """theta is the SCALED parameter, so the physical value is V_scale * theta.

        Returning theta raw made V_rel_err read 27.7 while u_rel_err read 0.48 -- two numbers that
        cannot both describe the same solution, and the inconsistency is what exposed the missing
        factor. The correct reading is that the value and the control are wrong by comparable
        amounts, which is what a partially converged deep BSDE should look like.
        """
        return float(self.theta.detach()) * self.V_scale

    def control_at_x0(self) -> np.ndarray:
        x = torch.tensor(np.atleast_2d(self.x0), dtype=torch.float32, device=self.device)
        t = torch.zeros((1, 1), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            g = self._grad_net(x, t)
            return ((g @ self.B) @ self.R_inv.T).cpu().numpy()[0]


def run_case(d: int, args, device: str) -> dict:
    """One asset per state coordinate here, so d = p and the reference is directly comparable."""
    prob = ACProblem(p=d)
    x0 = np.zeros(prob.state_dim)
    x0[: prob.n_assets] = 1.0                    # liquidate one unit of each asset
    ref = prob.reference(np.linspace(0.0, prob.T, 401))
    V_true = float(ref.value(x0, 0.0))
    u_true = ref.control(np.atleast_2d(x0), 0.0)[0]

    print(f"\n  d={d}   exact V(x0,0)={V_true:+.8e}  exact u(x0,0)={u_true}")
    s = DeepBSDE(prob, x0, n_steps=args.n_steps, n_paths=args.n_paths, epochs=args.epochs,
                 device=device, seed=0, log_every=args.log_every)
    chk = s.validate_on_exact()
    print(f"    exact-solution check: terminal RMSE = {chk:.3e}   "
          f"{'OK' if chk < 1e-4 else 'FAIL -- driver or sign is wrong, do not train yet'}")
    if chk >= 1e-4:
        return {"d": d, "validated": False, "terminal_rmse_exact": chk}

    s.solve()
    V_hat, u_hat = s.value_at_x0(), s.control_at_x0()
    v_err = abs(V_hat - V_true) / max(abs(V_true), 1e-300)
    u_err = float(np.max(np.abs(u_hat - u_true)) / np.max(np.abs(u_true)))
    print(f"    trained {s.solve_time:.1f}s   V_rel_err={v_err:.3e}  u_rel_err={u_err:.3e}")
    return {"d": d, "validated": True, "terminal_rmse_exact": chk,
            "V_true": V_true, "V_hat": V_hat, "u_true": u_true.tolist(),
            "u_hat": u_hat.tolist(), "value_rel_err": v_err, "control_rel_err": u_err,
            "seconds": s.solve_time, "epochs": args.epochs, "n_paths": args.n_paths,
            "n_steps": args.n_steps}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=2000)
    ap.add_argument("--n-paths", type=int, default=1024)
    ap.add_argument("--n-steps", type=int, default=50)
    ap.add_argument("--log-every", type=int, default=0)
    ap.add_argument("--dims", type=str, default="1")
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 100)
    print("DEEP BSDE BASELINE -- and the validation that must pass before any comparison")
    print("=" * 100)
    print("  The forward recursion is first run with the EXACT theta and EXACT grad V. The terminal")
    print("  mismatch must vanish there; if it does not, the driver or its sign is wrong and no")
    print("  amount of training will fix it.")

    rows = [run_case(int(d), args, device) for d in args.dims.split(",")]
    (OUT / "deep_bsde.json").write_text(json.dumps(
        {"rows": rows, "epochs": args.epochs, "n_paths": args.n_paths,
         "n_steps": args.n_steps, "device": device}, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT / 'deep_bsde.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
