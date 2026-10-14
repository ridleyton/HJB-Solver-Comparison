"""
Diagnostic: does the refactored PINN residual actually solve the alpha HJB?

Why this test and not a training run
------------------------------------
Training a PINN on the alpha benchmark and looking at its R^2 would be a weak test: a residual that
is wrong in a subtle way still trains to *something*, and at a small budget the result is poor for
reasons that have nothing to do with the refactor. The decisive question is narrower and has an exact
answer: does the residual operator evaluate to ZERO on the exact solution?

So we replace the network with the exact value function from the Riccati reference and evaluate the
module's own `_residual` / `_loss_direct` on random states. The exact V satisfies the HJB
identically, so a correct residual returns ~0, and any error in A, B, Q, the control law
u* = R^-1 B' grad V, the (grad V)'(B u) term, or the per-axis diffusion term shows up as a finite
residual. `_residual` differentiates its value function with respect to t through autograd, so dV/dt
is produced by the module rather than supplied by the test.

HOW THE EXACT VALUE FUNCTION IS SUPPLIED -- and the trap that cost a round of debugging
----------------------------------------------------------------------------------------
The obvious approach is to tabulate P(t) and c(t) on a fine grid and interpolate. That FAILS here,
and not by a little: the derivative of a piecewise-linear interpolant carries O(h) error even though
the interpolant itself is O(h^2), so dV/dt comes out wrong by ~1e-2 and every case reports a
residual of a few percent -- including the inventory-only control case that had already passed
verification. An all-cases failure is the tell that the harness, not the code, is broken.

Instead we build V as a second-order Taylor expansion in t about a chosen t0:

    P(t) ~= P0 + (t-t0) Pdot0 + (t-t0)^2 Pddot0 / 2,   c(t) ~= c0 + (t-t0) cdot0 + ...

with Pdot, Pddot, cdot, cddot obtained ANALYTICALLY from the Riccati equation rather than by
differentiating an interpolant. Autograd then differentiates the polynomial exactly, and at t = t0
there is no truncation error at all: the value, its t-derivative and its Hessian are all exact, so
the module's residual must vanish to machine precision.

The inventory-only case is included as a control. It passed verification before the refactor, so a
failure there means the test is wrong.

Usage: python src/diag_alpha_pinn.py
"""

from __future__ import annotations

import numpy as np
import torch

from ac_problem import ACProblem
from pinn import PINNSolver
from pinn_pi import PIPINN


def exact_taylor(problem: ACProblem, t0: float, v_scale: float):
    """Return a torch function giving the exact V(x,t)/v_scale near t0, differentiable in t.

    The function must return SCALED units. Both modules decode by multiplying the network output by
    V_scale, so substituting the unscaled V makes every term that depends on grad V wrong by a
    factor of V_scale while the x'Qx term is untouched -- and because the residual is not
    scale-invariant, that shows up as a finite residual rather than as a harmless rescaling.
    """
    P, c = problem.riccati(np.array([t0]))
    Pt = P[0]
    ct = float(c[0])
    R_inv = problem.R_inv
    BB = problem.B @ R_inv @ problem.B.T

    # Riccati RHS and its time derivative, both analytic -- no interpolation anywhere.
    Pdot = -(problem.Q + Pt @ problem.A + problem.A.T @ Pt - Pt @ BB @ Pt)
    Pddot = -(Pdot @ problem.A + problem.A.T @ Pdot - Pdot @ BB @ Pt - Pt @ BB @ Pdot)
    SS = problem.Sigma @ problem.Sigma.T
    cdot = 0.5 * float(np.trace(SS @ Pt))
    cddot = 0.5 * float(np.trace(SS @ Pdot))

    P0 = torch.tensor(Pt, dtype=torch.float64)
    Pd0 = torch.tensor(Pdot, dtype=torch.float64)
    Pdd0 = torch.tensor(Pddot, dtype=torch.float64)

    def value_scaled(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xd = x.to(torch.float64)
        # tau MUST be measured from t0. Writing `tau = t` instead leaves the expansion evaluated
        # at tau = t0 rather than at tau = 0, so the "exact" function is not the exact solution at
        # all and every case reports a few percent of residual -- including the control case.
        tau = t.to(torch.float64).reshape(-1, 1) - t0
        Pm = P0 + tau[:, :, None] * Pd0 + 0.5 * tau[:, :, None] ** 2 * Pdd0
        cm = ct + tau.reshape(-1) * cdot + 0.5 * tau.reshape(-1) ** 2 * cddot
        quad = -0.5 * torch.einsum("ni,nij,nj->n", xd, Pm, xd)
        return (quad + cm).reshape(-1, 1) / v_scale

    return value_scaled


def check(build, problem: ACProblem, label: str, n_rand: int = 64, seed: int = 0) -> bool:
    torch.manual_seed(seed)
    solver = build(problem, seed)

    # The modules keep their matrices in float32, which is right for training but leaves only
    # ~1e-7 of headroom -- too coarse to distinguish "correct" from "slightly wrong" in a residual
    # that must vanish. Promote everything to float64 for this test only.
    for name in ("Q", "R", "R_inv", "A", "B", "QT", "D"):
        v = getattr(solver, name, None)
        if isinstance(v, torch.Tensor):
            setattr(solver, name, v.to(torch.float64))

    rng = np.random.default_rng(seed)
    lo, hi = problem.domain()
    v_scale = float(solver.V_scale)
    W = 0.0
    for t0 in (0.15, 0.5, 0.85):
        exact = exact_taylor(problem, t0, v_scale)

        # Substitute the exact solution. The hook differs by module: PINNSolver routes everything
        # through `_value_scaled` (which also carries its hard-terminal ansatz), whereas PIPINN's
        # `_loss_direct` calls `self.net` on the ENCODED input, so the encoder must be inverted.
        if hasattr(solver, "_value_scaled"):
            solver._value_scaled = exact
            residual_fn = solver._residual
        else:
            d = problem.state_dim
            hw = torch.tensor(np.asarray(solver.halfwidth), dtype=torch.float64)
            ct_ = torch.tensor(np.asarray(solver.center), dtype=torch.float64)
            T_ = float(problem.T)

            def net_exact(inp, _hw=hw, _ct=ct_, _d=d, _T=T_):
                # BOTH encodings must be inverted: x = xn*halfwidth + centre, and -- easily
                # forgotten, because it maps t=0.5 to tn=0 so the error is invisible at the
                # midpoint -- t = (tn+1)/2*T. Passing tn through as if it were t feeds the exact
                # solution a different time and produces a residual that looks like a real defect.
                x_ = inp[..., :_d] * _hw + _ct
                t_ = (inp[..., _d:] + 1.0) / 2.0 * _T
                return exact(x_, t_)

            solver.net = net_exact
            residual_fn = solver._loss_direct

        x = rng.uniform(lo - 0.25, hi + 0.25, size=(n_rand, problem.state_dim))
        xt = torch.tensor(x, dtype=torch.float64, requires_grad=True)
        tt = torch.full((n_rand, 1), t0, dtype=torch.float64, requires_grad=True)
        res = residual_fn(xt, tt).detach().numpy().ravel()
        W = max(W, float(np.max(np.abs(res))))

    # Threshold: the exact solution is exact to ~1e-16, but the residual is assembled from second
    # autograd derivatives in float64, which accumulate ~1e-8. Anything above 1e-7 is a real
    # defect; anything below is arithmetic.
    ok = W < 1e-7
    print(f"  {label:<46} state_dim={problem.state_dim}  max|residual| over 3 times = "
          f"{W:.3e}   {'OK' if ok else 'FAIL'}")
    return ok


def main() -> int:
    print("Residual of the refactored modules evaluated ON THE EXACT SOLUTION (must be ~0)")
    print("=" * 100)
    cases = [
        ("inventory-only, d=1   [control]", ACProblem(p=1)),
        ("inventory-only, d=2   [control]", ACProblem(p=2)),
        ("inventory-only, d=3, hetero", ACProblem(p=3, hetero=True, het_seed=1)),
        ("alpha, d=2 (1 asset + signal)", ACProblem(p=1, alpha_kappa=1.0, alpha_sig=0.005)),
        ("alpha, d=4 (2 assets + signals)", ACProblem(p=2, alpha_kappa=1.0, alpha_sig=0.005)),
        ("alpha, d=6, kappa=2", ACProblem(p=3, alpha_kappa=2.0, alpha_sig=0.002)),
    ]
    builders = (
        ("direct PINN",
         lambda prob, seed: PINNSolver(prob, hidden=16, layers=2, n_collocation=512,
                                       epochs=1, device="cpu", seed=seed, verbose=False)),
        ("PI-PINN",
         lambda prob, seed: PIPINN(prob, hidden=16, layers=2, n_points=512, num_iters=1,
                                   num_epochs=1, device="cpu", seed=seed, verbose=False)),
    )
    all_ok = True
    for label, prob in cases:
        for name, build in builders:
            all_ok &= check(build, prob, f"{label} / {name}")
    print()
    print("=" * 100)
    print("ALL RESIDUALS VANISH ON THE EXACT SOLUTION" if all_ok
          else "SOME RESIDUALS ARE NON-ZERO -- the refactor is wrong, not the training")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
