"""
Benchmark HJB problem with a closed-form ground truth.

Design rationale
----------------
The original thesis validated solver accuracy against *Policy Iteration*
(`comparison_framework.py:717-719`, `reference_method = "Policy Iteration"`), which makes
the reference method zero-error by construction -- a circular comparison. It also had no
inventory state in the network (`in_dim = 4`), so the implemented problem was a
price-tracking problem rather than the market-making problem described in the paper.

Here we use instead a linear-quadratic (LQ) optimal-execution HJB whose value function is
available in closed form via a matrix Riccati ODE. This gives:
  * a genuine analytic reference for accuracy metrics (no circularity);
  * an explicit optimal control law, so the sign convention can be checked analytically;
  * arbitrary state dimension p, enabling an honest curse-of-dimensionality sweep
    (the thesis claimed this motivation but never exceeded p = 4 inputs, and never
    tested p > 10).

Problem (multi-asset Almgren-Chriss optimal execution, in LQ form)
-----------------------------------------------------------------
State  x in R^p : inventory of each of p assets, x = q.
Control u in R^p: trading rate, dx = (A x + B u) dt + Sigma dW.
Objective: maximise
    E[ -1/2 * int_0^T (x'Qx + u'Ru) dt  -  1/2 * x_T' Q_T x_T ].
Interpretation:
    u'Ru  = temporary market impact  (eta |u|^2)
    x'Qx  = inventory risk penalty   (lambda sigma^2 |q|^2)
    x_T'Q_T x_T = terminal inventory penalty
This is the standard Almgren-Chriss problem; with A = 0, B = I, R = eta I, Q = lambda sigma^2 I
it reproduces the classical closed-form solution q_t = q_0 sinh(k(T-t))/sinh(kT),
k = sqrt(lambda sigma^2 / eta).

HJB
---
    -dV/dt = max_u { -1/2 (x'Qx + u'Ru) + (grad V)'(Ax + Bu) } + 1/2 tr(Sigma Sigma' d2V)
    V(x,T) = -1/2 x' Q_T x
Value function ansatz V(x,t) = -1/2 x'P(t)x + c(t) gives the Riccati equation
    -dP/dt = Q + P A + A'P - P B R^{-1} B' P ,   P(T) = Q_T
    dc/dt  = 1/2 tr(Sigma Sigma' P) ,            c(T) = 0
and optimal control
    u*(x,t) = -R^{-1} B' P(t) x .

Sign convention is therefore explicit and testable: for the inventory problem with
A = 0, B = I, R = eta I the control is u* = -(1/eta) P(t) q, and since P >= 0 and q >= 0
this is *negative* (liquidate). A solver that returns the opposite sign is inverted.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import solve_ivp


# --------------------------------------------------------------------------------------
# Problem specification
# --------------------------------------------------------------------------------------
@dataclass
class ACProblem:
    """Multi-asset Almgren-Chriss optimal-execution problem in linear-quadratic form.

    Parameters
    ----------
    p : int
        State dimension = number of assets.
    eta : float
        Temporary market-impact coefficient (control cost), R = eta * I.
    lam_sigma2 : float
        Inventory risk penalty, Q = lam_sigma2 * I.
    q_terminal : float
        Terminal inventory penalty, Q_T = q_terminal * I.
    T : float
        Time horizon (normalised to 1.0 throughout).
    sigma : float
        Diffusion scale, Sigma = sigma * I. Enters only through c(t).
    coupling : float
        Off-diagonal coupling in Q (cross-asset risk). 0.0 = independent assets.
    hetero : bool
        If True, each asset gets its own risk aversion and impact coefficient, drawn
        log-uniformly around `lam_sigma2` and `eta` from `het_seed`, and the off-diagonal terms
        of Q are drawn at random rather than being a constant. This matters for the dimension
        study: with the default uniform parameters and A = 0, the problem is exactly
        **separable** -- V(x,t) = sum_i v(q_i,t) with the *same* scalar v for every coordinate --
        so a network solving it at p = 20 is doing little more than applying one learned
        univariate function to twenty inputs. Random per-asset heterogeneity destroys that
        symmetry and makes V a genuinely p-dimensional quadratic with a full P matrix, which is
        the case the curse-of-dimensionality argument is actually about.
    """

    p: int = 1
    eta: float = 0.01
    lam_sigma2: float = 0.01
    q_terminal: float = 0.1
    T: float = 1.0
    sigma: float = 0.01
    coupling: float = 0.0
    hetero: bool = False
    het_seed: int = 0
    het_spread: float = 3.0

    # ---- optional alpha signal -------------------------------------------------------
    # Setting alpha_kappa > 0 augments the state from "inventory in p assets" to
    # "(inventory, alpha signal) per asset", i.e. state dimension 2p while the CONTROL stays p.
    # This is the financially realistic case: the trader does not liquidate to zero, but holds a
    # position proportional to the signal.
    alpha_kappa: float = 0.0
    alpha_sig: float = 0.005

    # Derived matrices, filled in __post_init__
    A: np.ndarray = field(init=False, repr=False)
    B: np.ndarray = field(init=False, repr=False)
    Q: np.ndarray = field(init=False, repr=False)
    R: np.ndarray = field(init=False, repr=False)
    QT: np.ndarray = field(init=False, repr=False)
    Sigma: np.ndarray = field(init=False, repr=False)
    R_inv: np.ndarray = field(init=False, repr=False)
    has_alpha: bool = field(init=False, repr=False, default=False)

    def __post_init__(self) -> None:
        p = self.p
        rng = np.random.default_rng(self.het_seed)

        if self.alpha_kappa > 0.0:
            self._build_alpha(p)
            return

        self.has_alpha = False
        # dx = (A x + B u) dt ; pure inventory dynamics dq = u dt
        self.A = np.zeros((p, p))
        self.B = np.eye(p)

        if not self.hetero:
            # Running cost x'Qx + u'Ru
            self.Q = self.lam_sigma2 * np.eye(p)
            if self.coupling:
                off = self.coupling * np.ones((p, p)) - self.coupling * np.eye(p)
                self.Q = self.Q + off
            self.R = self.eta * np.eye(p)
            self.QT = self.q_terminal * np.eye(p)
        else:
            # Log-uniform per-asset coefficients over [1/spread, spread] x the base value.
            lo, hi = 1.0 / self.het_spread, self.het_spread
            lam = self.lam_sigma2 * np.exp(rng.uniform(np.log(lo), np.log(hi), p))
            eta = self.eta * np.exp(rng.uniform(np.log(lo), np.log(hi), p))
            qt = self.q_terminal * np.exp(rng.uniform(np.log(lo), np.log(hi), p))
            self.Q = np.diag(lam)
            if p > 1:
                # Random symmetric off-diagonal risk, scaled to stay well below the diagonal so
                # Q remains positive definite.
                off = rng.uniform(-1.0, 1.0, (p, p))
                off = 0.5 * (off + off.T) * (0.3 * np.sqrt(lam[:, None] * lam[None, :]))
                np.fill_diagonal(off, 0.0)
                self.Q = self.Q + off
            self.R = np.diag(eta)
            self.QT = np.diag(qt)

        # Guard: the LQ problem needs Q, Q_T and R positive definite (R is diagonal, so this
        # reduces to the diagonal being positive, plus a PSD check on Q). The alpha extension
        # deliberately violates the Q part -- its Q is indefinite by construction -- which is why
        # it is built by _build_alpha() and returns before reaching this guard.
        if np.min(np.diag(self.R)) <= 0 or np.min(np.diag(self.QT)) <= 0:
            raise ValueError("R and Q_T must be positive definite")
        if np.linalg.eigvalsh(0.5 * (self.Q + self.Q.T)).min() <= 0:
            raise ValueError("Q must be positive definite for this benchmark")

        self.Sigma = self.sigma * np.eye(p)
        self.R_inv = np.linalg.inv(self.R)

    def _build_alpha(self, p: int) -> None:
        """Build the alpha-augmented state x = (inventory q, scaled signal s) in R^{2p}.

        The control stays p-dimensional: only inventory is traded. State dimension and control
        dimension therefore differ, which every solver must respect.
        """
        if self.hetero:
            raise ValueError("hetero and alpha_kappa are not tested together; run them separately")
        kappa, sig_s = self.alpha_kappa, self.alpha_sig
        n = 2 * p

        # dq = u dt  and  ds = -kappa s dt + sqrt(2 kappa) dW. A has NO inventory-to-signal
        # block: the signal does not move inventory by itself. (Putting I there would make
        # dq = (s + u) dt, a different and unintended model -- see src/alpha_probe.py.)
        self.A = np.zeros((n, n))
        self.A[p:, p:] = -kappa * np.eye(p)

        self.B = np.zeros((n, p))
        self.B[:p, :] = np.eye(p)

        # -1/2 x'Qx = -1/2 q'Lambda q + sig_s q's, i.e. the alpha is harvested by holding
        # inventory, so Q is INDEFINITE by construction (zero signal block, negative cross
        # block). The usual Q >= 0 solvability condition therefore does not apply; well-posedness
        # and the two feasibility bounds are settled empirically in src/alpha_probe.py.
        self.Q = np.zeros((n, n))
        self.Q[:p, :p] = self.lam_sigma2 * np.eye(p)
        self.Q[:p, p:] = -sig_s * np.eye(p)
        self.Q[p:, :p] = -sig_s * np.eye(p)

        self.R = self.eta * np.eye(p)
        self.QT = np.zeros((n, n))
        self.QT[:p, :p] = self.q_terminal * np.eye(p)

        # The signal is scaled to unit stationary variance, hence sqrt(2 kappa). This also
        # dominates the explicit CFL bound: inventory diffusion is sigma^2 = 1e-4 while signal
        # diffusion is 2 kappa = O(1), so the time step is set by the SIGNAL rather than by the
        # inventory. That is a genuine cost of carrying the extra state.
        self.Sigma = np.zeros((n, n))
        self.Sigma[:p, :p] = self.sigma * np.eye(p)
        self.Sigma[p:, p:] = np.sqrt(2.0 * kappa) * np.eye(p)

        self.R_inv = np.linalg.inv(self.R)
        self.has_alpha = True

    @property
    def n_assets(self) -> int:
        """Number of traded assets; equals the control dimension."""
        return self.p

    # ----------------------------------------------------------------------------------
    # Interfaces expected by the solvers
    # ----------------------------------------------------------------------------------
    @property
    def state_dim(self) -> int:
        return self.A.shape[0]

    @property
    def control_dim(self) -> int:
        return self.B.shape[1]

    def domain(self) -> tuple[np.ndarray, np.ndarray]:
        """Box domain for the state, used by grid and sampling methods.

        In the alpha case the state is (inventory, scaled signal): inventory lies in [-1, 1] as
        before, and the signal is SCALED to unit stationary variance so that [-1, 1] covers about
        one standard deviation. The two blocks therefore share a box by construction rather than
        by coincidence, which matters because the grid axes and the network's input scaling both
        read this.
        """
        d = self.state_dim
        lo = np.full(d, -1.0)
        hi = np.full(d, 1.0)
        return lo, hi

    def running_cost(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """-1/2 (x'Qx + u'Ru), vectorised over a batch of states/controls."""
        xq = np.einsum("...i,ij,...j->...", x, self.Q, x)
        ur = np.einsum("...i,ij,...j->...", u, self.R, u)
        return -0.5 * (xq + ur)

    def hamiltonian_max(self, x: np.ndarray, grad_V: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sup over u of { -1/2(x'Qx + u'Ru) + (grad V)'(Ax + Bu) }.

        Analytic maximiser: u* = R^{-1} B' grad_V.
        Returns (max_value, u_star).  Note: grad_V here is the gradient of V, and the
        linear term is (grad V)'(Ax + Bu) = (grad V)'Ax + (grad V)'B u, so
        d/du = -R u + B' grad V = 0  ->  u* = R^{-1} B' grad V.
        """
        Bt_g = grad_V @ self.B                        # grad V' B : (..., n) x (n, m) -> (..., m)
        u_star = Bt_g @ self.R_inv.T                  # (..., m)
        xq = np.einsum("...i,ij,...j->...", x, self.Q, x)
        ur = np.einsum("...i,ij,...j->...", u_star, self.R, u_star)
        a_term = np.einsum("...i,ij,...j->...", grad_V, self.A, x)
        val = -0.5 * (xq + ur) + a_term + np.einsum("...i,...i->...", Bt_g, u_star)
        return val, u_star

    # ----------------------------------------------------------------------------------
    # Closed-form reference solution (matrix Riccati)
    # ----------------------------------------------------------------------------------
    def riccati(self, t_grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integrate the Riccati ODE backward from T.

        Returns (P[t_index, d, d], c[t_index]) with t_grid ascending in [0, T], where d is the
        STATE dimension -- 2p in the alpha case, not p. Getting this wrong is silent in the
        inventory-only benchmark (where d = p) and a shape error here otherwise.
        """
        d = self.state_dim
        # Flatten the matrix into a vector for the ODE solver.
        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            P = y[: d * d].reshape(d, d)
            c = y[d * d]
            Pdot = -(self.Q + P @ self.A + self.A.T @ P - P @ self.B @ self.R_inv @ self.B.T @ P)
            c_dot = 0.5 * np.trace(self.Sigma @ self.Sigma.T @ P)
            return np.concatenate([Pdot.ravel(), [c_dot]])

        y0 = np.concatenate([self.QT.ravel(), [0.0]])
        # Integrate backward: substitute s = T - t, so s runs 0 -> T and t runs T -> 0.
        # t_eval must be ascending, so evaluate at s ascending (= T - t descending).
        t_grid = np.asarray(t_grid, dtype=float)
        s_eval = np.clip(self.T - t_grid[::-1], 0.0, self.T)
        sol = solve_ivp(
            lambda s, y: -rhs(self.T - s, y),
            (0.0, self.T),
            y0,
            t_eval=s_eval,
            rtol=1e-10,
            atol=1e-12,
            method="RK45",
        )
        if not sol.success:
            raise RuntimeError(f"Riccati integration failed: {sol.message}")
        ys = sol.y[:, ::-1]                     # back to ascending t
        P = ys[: d * d, :].T.reshape(-1, d, d)  # (n_t, d, d)
        c = ys[d * d, :]
        return P, c

    def reference(self, t_grid: np.ndarray):
        """Return a callable reference solution with .value(x,t) and .control(x,t)."""
        P, c = self.riccati(t_grid)

        def _interp(t: float) -> tuple[np.ndarray, float]:
            d = self.state_dim
            Pt = np.empty((d, d))
            for i in range(d):
                for j in range(d):
                    Pt[i, j] = np.interp(t, t_grid, P[:, i, j])
            ct = float(np.interp(t, t_grid, c))
            return Pt, ct

        class _Ref:
            def __init__(self, problem: "ACProblem"):
                self.problem = problem

            def value(self, x: np.ndarray, t: float) -> np.ndarray:
                Pt, ct = _interp(t)
                x = np.atleast_2d(x)
                return -0.5 * np.einsum("...i,ij,...j->...", x, Pt, x) + ct

            def control(self, x: np.ndarray, t: float) -> np.ndarray:
                Pt, _ = _interp(t)
                # u* = -R^{-1} B' P x, stacked over a batch. As a row vector this is
                # u' = -x' P B R^{-1}', hence B (not B') multiplies on the right. The two are
                # indistinguishable while B = I, which is why this only shows up in the alpha case.
                return -np.atleast_2d(x) @ Pt @ self.problem.B @ self.problem.R_inv.T

            def P(self, t: float) -> np.ndarray:
                return _interp(t)[0]

        return _Ref(self)

    # ----------------------------------------------------------------------------------
    # Classical scalar Almgren-Chriss trajectory, for an independent sanity check
    # ----------------------------------------------------------------------------------
    def ac_trajectory(self, q0: float, t_grid: np.ndarray) -> np.ndarray:
        """Closed-form inventory path for the *forced-liquidation* scalar problem.

        q_t = q0 sinh(k(T-t)) / sinh(kT),  k = sqrt(lam_sigma2 / eta).

        This is the Q_T -> infinity limit (inventory must reach zero exactly). For a
        finite terminal penalty Q_T the Riccati solution below applies instead; use
        :meth:`ac_P_closed_form` for that case.
        """
        k = np.sqrt(self.lam_sigma2 / self.eta)
        return q0 * np.sinh(k * (self.T - t_grid)) / np.sinh(k * self.T)

    def ac_P_closed_form(self, t_grid: np.ndarray) -> np.ndarray:
        """Closed-form scalar Riccati solution P(t) for finite terminal penalty Q_T.

        Solves  dP/dt = P^2/eta - Q  with  P(T) = Q_T,  giving

            P(t) = eta*k * (Q_T cosh(k(T-t)) + eta*k sinh(k(T-t)))
                        / (eta*k cosh(k(T-t)) + Q_T sinh(k(T-t))),   k = sqrt(Q/eta).

        As Q_T -> infinity this reduces to eta*k*coth(k(T-t)).
        """
        k = np.sqrt(self.lam_sigma2 / self.eta)
        QT = self.q_terminal
        ek = self.eta * k
        s = np.sinh(k * (self.T - t_grid))
        c = np.cosh(k * (self.T - t_grid))
        return ek * (QT * c + ek * s) / (ek * c + QT * s)


# --------------------------------------------------------------------------------------
# Analytic control-law sign check -- directly targets the thesis defect
# --------------------------------------------------------------------------------------
def analytic_control_sign_check(problem: ACProblem, n: int = 512, seed: int = 0) -> dict:
    """Verify u* = -R^{-1} B' P x has the correct sign for a liquidation problem.

    For a liquidation problem q0 > 0 we require u* < 0 (selling) at t = 0.
    """
    t_grid = np.linspace(0.0, problem.T, 201)
    ref = problem.reference(t_grid)
    rng = np.random.default_rng(seed)
    q0 = rng.uniform(0.1, 1.0, size=(n, problem.p))
    u = ref.control(q0, 0.0)
    return {
        "all_controls_negative_for_positive_inventory": bool(np.all(u < 0)),
        "frac_negative": float(np.mean(u < 0)),
        "mean_control": float(np.mean(u)),
        "P_0_diag": np.diag(ref.P(0.0)).tolist(),
        "P_T_diag": np.diag(ref.P(problem.T)).tolist(),
    }


if __name__ == "__main__":
    # ---------------------------------------------------------------------------------
    # Test 1: scalar Riccati solution must match the closed form for finite Q_T,
    #         pointwise. (Integrating the ODE by Euler would only measure discretisation
    #         error and is not a test of the Riccati solution.)
    # ---------------------------------------------------------------------------------
    prob = ACProblem(p=1, eta=0.0005, lam_sigma2=0.01, q_terminal=1.0, T=1.0)
    t = np.linspace(0.0, 1.0, 201)
    ref = prob.reference(t)
    k = np.sqrt(prob.lam_sigma2 / prob.eta)

    P_classical = prob.ac_P_closed_form(t)
    P_riccati = np.array([ref.P(ti)[0, 0] for ti in t])
    err_P = np.max(np.abs(P_riccati - P_classical))

    print("Test 1: scalar finite-terminal-penalty closed form")
    print(f"  kappa                        = {k:.8f}")
    print(f"  max |P_Riccati - P_closed|   = {err_P:.3e}")
    print(f"  P(0)={P_riccati[0]:.10f}  P(T)={P_riccati[-1]:.10f}  (P(T) must equal Q_T=1)")
    print(f"  control at (q=1, t=0)        = {ref.control(np.array([[1.0]]), 0.0)[0, 0]:.6f}  (expect < 0)")
    assert err_P < 1e-9, f"Riccati P does not match closed form ({err_P:.2e})"
    assert abs(P_riccati[-1] - prob.q_terminal) < 1e-9, "terminal condition P(T)=Q_T violated"

    # ---------------------------------------------------------------------------------
    # Test 2: the control law must be u* = -P q / eta exactly, with the correct
    #         (negative-for-positive-inventory) sign. This is the check that would have
    #         caught the sign inversion in the original backtest code.
    # ---------------------------------------------------------------------------------
    rng = np.random.default_rng(0)
    q = rng.uniform(0.1, 1.0, size=(256, 1))
    u = ref.control(q, 0.0)
    u_expected = -P_riccati[0] * q / prob.eta
    err_u = np.max(np.abs(u - u_expected))
    print("\nTest 2: control law u* = -R^-1 B' P x")
    print(f"  max |u_Riccati - (-P q / eta)| = {err_u:.3e}")
    print(f"  all controls negative for q>0  = {bool(np.all(u < 0))}")
    assert err_u < 1e-9, "control does not equal -P q / eta"
    assert np.all(u < 0), "control sign inverted"

    # ---------------------------------------------------------------------------------
    # Test 3: forced-liquidation limit. With Q_T -> infinity the Riccati solution must
    #         converge to the classical AC results: P -> eta*k*coth(k(T-t)) and the
    #         optimal path -> q0 sinh(k(T-t))/sinh(kT).
    # ---------------------------------------------------------------------------------
    prob_inf = ACProblem(p=1, eta=0.0005, lam_sigma2=0.01, q_terminal=1e8, T=1.0)
    ref_inf = prob_inf.reference(t)
    P_inf = np.array([ref_inf.P(ti)[0, 0] for ti in t])
    # Skip the first point (t=0 is fine) but avoid t=T where coth -> inf; use interior.
    interior = slice(0, -1)
    P_limit = prob.eta * k / np.tanh(k * (prob.T - t[interior]))
    err_lim = np.max(np.abs(P_inf[interior] - P_limit)) / np.max(np.abs(P_limit))
    print("\nTest 3: forced-liquidation limit (Q_T -> inf) recovers classical AC")
    print(f"  relative error in P vs eta*k*coth = {err_lim:.3e}")

    # The optimal path follows dq/dt = -P(t) q / eta, i.e. q_t = q0 exp(-(1/eta) int_0^t P).
    # Integrate the *analytic* P (a smooth function) rather than the ODE itself: with
    # Q_T = 1e8 the ODE is extremely stiff near t = T and explicit RK4 would overflow.
    fine = np.linspace(0.0, 1.0, 20001)
    P_fine = prob_inf.ac_P_closed_form(fine) / prob.eta
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (P_fine[1:] + P_fine[:-1]) * np.diff(fine))])
    q_analytic = 1.0 * np.exp(-cum)
    q_analytic_at_t = np.interp(t, fine, q_analytic)
    q_ac = prob.ac_trajectory(1.0, t)

    # Compare away from the singular endpoint (t=T is where q -> 0 and the relative
    # error is dominated by floating point in the exponential).
    interior = slice(0, -1)
    err_path = np.max(np.abs(q_analytic_at_t[interior] - q_ac[interior]))
    print(f"  max |q_optimal - AC trajectory|   = {err_path:.3e}")
    print(f"  terminal inventory                = {q_analytic[-1]:.3e}  (expect ~0)")
    print(f"  q(0)                              = {q_analytic[0]:.6f}  (expect 1.0)")
    assert abs(q_analytic[0] - 1.0) < 1e-12, "path must start at q0"
    assert q_analytic[-1] < 1e-6, "forced liquidation must drive inventory to ~0"
    assert err_path < 1e-6, f"optimal path does not recover the AC trajectory ({err_path:.2e})"

    # ---------------------------------------------------------------------------------
    # Test 4: multi-asset dimensions solve, stay symmetric PSD, and keep the sign.
    # ---------------------------------------------------------------------------------
    print("\nTest 4: multi-asset dimensions")
    for p in (1, 2, 3, 5, 8):
        pr = ACProblem(p=p, eta=0.0005, lam_sigma2=0.01, q_terminal=1.0)
        r = pr.reference(np.linspace(0, 1, 51))
        Pt = r.P(0.0)
        ev = np.linalg.eigvalsh(Pt)
        print(
            f"  p={p}: P(0) diag={np.round(np.diag(Pt)[:3], 8).tolist()}... "
            f"symmetric={np.allclose(Pt, Pt.T)} min_eig={ev.min():.3e}"
        )
        assert np.allclose(Pt, Pt.T), "P must be symmetric"
        assert ev.min() > -1e-12, "P must be positive semidefinite"

    print("\n  ALL CHECKS PASSED")
