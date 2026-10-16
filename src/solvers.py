"""
Numerical solvers for the LQ optimal-execution HJB, all validated against the
closed-form Riccati reference in :mod:`ac_problem`.

Common interface
----------------
Each solver exposes:
    solve()                -> wall-clock seconds (float)
    value(X, t)            -> np.ndarray, shape (n,)
    control(X, t)          -> np.ndarray, shape (n, p)

`X` is an (n, p) array of states, `t` a scalar in [0, T].

Defects in the original implementation addressed here by construction
--------------------------------------------------------------------
1. **Control sign.** The optimal control is taken from the analytic Hamiltonian
   maximisation for the LQ problem, ``u* = R^{-1} B' grad V``. For a liquidation problem
   this is *negative* for positive inventory. The original code hard-coded
   ``u = -grad V / (2 * market_impact)`` in three separate places
   (``comparison_framework.py:68``, ``main.py:272``, ``hft_backtest.py:107``) -- the
   negative of the correct control -- and additionally dropped the dollar scaling ``S``
   and the spread term. Here the control is derived once from the HJB and asserted
   against the closed form.

2. **Reference solution.** Accuracy is measured against the analytic Riccati solution
   rather than against Policy Iteration, which the original used as its own reference
   (``comparison_framework.py:717-719``) and which is therefore zero-error by
   construction.

3. **Time handling.** Solutions are stored as a set of time snapshots and interpolated
   in time; returning a single terminal snapshot for all ``t`` would silently destroy
   the accuracy metrics.

Stability
---------
The explicit backward scheme is subject to both a diffusion and an advection
(CFL) bound. The advection bound dominates here because the terminal inventory
penalty makes ``|u*|`` large near ``t = T``. ``dt`` is therefore chosen from the
measured maximum control magnitude, and the resulting number of time steps is
reported so the cost accounting is explicit.
"""

from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.linalg import solve_banded

from ac_problem import ACProblem


# ======================================================================================
# Shared grid machinery
# ======================================================================================
class _GridSolver:
    """Common tensor-product grid, snapshot storage and interpolation."""

    name = "grid solver"

    def __init__(self, problem: ACProblem, n_grid: int = 41, domain_pad: float = 0.25,
                 n_snap: int = 41, snap_cap: float = 4e7, time_scheme: str = "explicit",
                 n_steps: int | None = None):
        self.problem = problem
        self.n_grid = n_grid
        self.domain_pad = domain_pad
        self.n_snap = n_snap
        if time_scheme not in ("explicit", "lod"):
            raise ValueError(f"unknown time_scheme {time_scheme!r}")
        self.time_scheme = time_scheme
        self.n_steps_forced = n_steps
        # Cap on n_snap * N (snapshots times grid points, in float64 elements) before n_snap is
        # halved. The original 4e7 was far more conservative than this machine needs (52 GB
        # free) and, worse, it silently reduced the TIME resolution as N^p grew, which inflated
        # the value error at high dimension in a way that looked like a spatial effect -- see
        # exp_control_floor.py, where the cap explains the p=5 anomaly. Raise it explicitly
        # whenever time resolution has to be held fixed across a refinement sweep.
        self.snap_cap = float(snap_cap)
        self.n_snap_used: int | None = None
        self.solve_time: float | None = None
        self.n_time_steps: int | None = None
        self.snapshots: np.ndarray | None = None   # (n_snap, *shape), ascending in t
        self.t_snap: np.ndarray | None = None

    # ----------------------------------------------------------------------------------
    def _build_grid(self) -> None:
        prob = self.problem
        lo, hi = prob.domain()
        self.lo = lo - self.domain_pad
        self.hi = hi + self.domain_pad
        # STATE dimension, not the number of assets: with the alpha signal the two differ
        # (state 2p, control p) and every grid axis belongs to the state.
        d = prob.state_dim
        self.axes = [np.linspace(self.lo[i], self.hi[i], self.n_grid) for i in range(d)]
        self.dx = [self.axes[i][1] - self.axes[i][0] for i in range(d)]
        self.shape = tuple([self.n_grid] * d)
        mesh = np.meshgrid(*self.axes, indexing="ij")
        self.X = np.stack([m.ravel() for m in mesh], axis=1)   # (N, state_dim)
        self.N = self.X.shape[0]

    def _terminal(self) -> np.ndarray:
        prob = self.problem
        V = -0.5 * np.einsum("...i,ij,...j->...", self.X, prob.QT, self.X)
        return V.reshape(self.shape)

    def _stable_dt(self, max_u: float) -> float:
        """Largest dt satisfying both the diffusion and advection CFL bounds."""
        prob = self.problem
        dx_min = min(self.dx)
        # Diffusion: dt <= c dx^2 / max_i (Sigma Sigma')_ii. The bound must use the LARGEST
        # per-axis diffusion rather than a single sigma^2: in the alpha case the signal axes
        # carry sqrt(2 kappa), four orders of magnitude above inventory, and hardcoding sigma^2
        # would take a step far outside the stability region without failing loudly.
        diff = float(np.max(np.diag(prob.Sigma @ prob.Sigma.T)))
        dt_diff = 0.25 * dx_min**2 / max(diff, 1e-300)
        # Advection: dt <= c * dx / |u|_inf
        dt_adv = np.inf if max_u <= 0 else 0.25 * dx_min / max_u
        return min(dt_diff, dt_adv)

    def _hamiltonian_and_control(self, V: np.ndarray, t: float | None = None):
        """Return (H_max reshaped, u* (N, control_dim)) where
        H_max = max_u { -1/2(x'Qx + u'Ru) + (grad V)'(Ax + Bu) },
        u* = R^{-1} B' grad V.

        If the problem defines a `source(x, t)` -- a manufactured problem does, so that an exact but
        NON-POLYNOMIAL value function can be imposed (see exp_nonpolynomial.py) -- it is added to the
        running reward here. The maximiser is unchanged, because the source does not depend on u, so
        the stencil, the CFL bound and every other solver path are untouched. Problems without a
        source take exactly the code path they took before.
        """
        prob = self.problem
        d = prob.state_dim
        grad = np.stack(np.gradient(V, *self.dx, edge_order=2), axis=0)  # (d, *shape)
        g = grad.reshape(d, -1).T                                       # (N, d)
        # u* = R^{-1} B' grad V. B must appear explicitly: it is the identity only in the
        # inventory-only benchmark, and omitting it is invisible there.
        u = g @ prob.B @ prob.R_inv.T                                   # (N, control_dim)
        x = self.X
        Bu = u @ prob.B.T                                               # (N, d)
        xq = np.einsum("...i,ij,...j->...", x, prob.Q, x)
        ur = np.einsum("...i,ij,...j->...", u, prob.R, u)
        ax = np.einsum("...i,ij,...j->...", g, prob.A, x)
        bu = np.einsum("...i,...i->...", g, Bu)
        H = (-0.5 * (xq + ur) + ax + bu).reshape(self.shape)
        if t is not None:
            src = getattr(prob, "source", None)
            if src is not None:
                H = H + np.asarray(src(self.X, float(t))).reshape(self.shape)
        return H, u

    def _diffusion(self, V: np.ndarray) -> np.ndarray:
        """1/2 tr(Sigma Sigma' grad^2 V) = 1/2 sum_i D_i d2V/dx_i^2, D = diag(Sigma Sigma').

        The per-axis coefficient is D_i, not a single sigma^2. In the alpha case the signal axes
        diffuse at 2 kappa while inventory diffuses at sigma^2, so a hardcoded sigma^2 would drop
        the dominant diffusion term entirely.
        """
        prob = self.problem
        D = np.diag(prob.Sigma @ prob.Sigma.T)
        out = np.zeros(self.shape)
        for i in range(prob.state_dim):
            out += D[i] * np.gradient(
                np.gradient(V, self.dx[i], axis=i, edge_order=2), self.dx[i], axis=i,
                edge_order=2,
            )
        return 0.5 * out

    def _upwind_advection(self, V: np.ndarray, v: np.ndarray) -> np.ndarray:
        """sum_i v_i dV/dx_i with upwind differences.

        Needed for policy evaluation: once the policy is frozen, the whole term
        (grad V)'(A x + B u) is genuine *advection*, and central differencing combined with
        explicit Euler is unconditionally unstable for the advection equation. (The
        true-HJB sweep does not need this: there (grad V)'(B u*) = (grad V)'R^{-1}(grad V)
        is a positive-definite quadratic in grad V, i.e. a source term rather than a
        transport term, which is why the max-sweep is stable with central differences.)

        `v` is the FULL velocity of shape (N, state_dim), i.e. A x + B u. An earlier version
        took only the control `u` and central-differenced A x separately, which is inconsistent:
        with A != 0 the autonomous part also advects, and the explicit step then lies outside its
        stability region. (With the old A = 0 those two were the same thing, so the defect was
        invisible.)

        Zero-gradient boundary conditions are applied at the grid edges.
        """
        d = self.problem.state_dim
        n = self.n_grid
        v_grid = np.asarray(v).T.reshape(d, *self.shape)
        total = np.zeros(self.shape)
        for i in range(d):
            dx = self.dx[i]
            Vm = np.empty_like(V)   # Vm[j] = V[j-1], with Vm[0] = V[0]
            Vp = np.empty_like(V)   # Vp[j] = V[j+1], with Vp[n-1] = V[n-1]

            src_lo = [slice(None)] * d; src_lo[i] = slice(0, n - 1)
            dst_hi = [slice(None)] * d; dst_hi[i] = slice(1, n)
            Vm[tuple(dst_hi)] = V[tuple(src_lo)]

            src_hi = [slice(None)] * d; src_hi[i] = slice(1, n)
            dst_lo = [slice(None)] * d; dst_lo[i] = slice(0, n - 1)
            Vp[tuple(dst_lo)] = V[tuple(src_hi)]

            edge_lo = [slice(None)] * d; edge_lo[i] = 0
            edge_hi = [slice(None)] * d; edge_hi[i] = n - 1
            Vm[tuple(edge_lo)] = V[tuple(edge_lo)]
            Vp[tuple(edge_hi)] = V[tuple(edge_hi)]

            d_back = (V - Vm) / dx
            d_fwd = (Vp - V) / dx
            vi = np.take(v_grid, i, axis=0)
            # Stability direction: the scheme is V_new = V + dt*(v dV/dx + ...).
            # A von Neumann analysis of this form gives, with lambda = dt*v/dx,
            #   forward stencil  -> |g|^2 = 1 - 2*lambda*(1-cos th)*(1-lambda)  (stable 0<lambda<1)
            #   backward stencil -> stable for -1 < lambda < 0
            # i.e. v > 0 must use the FORWARD difference and v < 0 the BACKWARD one.
            total += np.where(vi > 0.0, d_fwd, d_back) * vi
        return total

    # ----------------------------------------------------------------------------------
    def _control_from_V(self, V: np.ndarray) -> np.ndarray:
        """u* = R^{-1} B' grad V on the grid, flattened to (N, control_dim)."""
        prob = self.problem
        grad = np.stack(np.gradient(V, *self.dx, edge_order=2), axis=0)
        return grad.reshape(prob.state_dim, -1).T @ prob.B @ prob.R_inv.T

    def _lod_step(self, rhs: np.ndarray, dt: float) -> np.ndarray:
        """Solve (I - dt L) V = rhs by per-axis implicit solves:  L = 1/2 sum_i D_i d2/dx_i^2.

        WHY. The explicit scheme's time step is restricted by the DIFFUSION, dt <= c dx^2 / D_i.
        In the inventory-only benchmark D = sigma^2 = 1e-4, which is harmless; the moment a signal
        is carried its axes diffuse at 2*kappa = O(1), four orders of magnitude more, and that
        single term sets the step and therefore the cost at every dimension. The restriction is a
        property of EXPLICIT time stepping, not of grid methods, so removing it is what decides
        whether "the grid is not runnable beyond p ~ 5" is a fact about grids or about this scheme.

        Method: locally one-dimensional (LOD) splitting, (I - dt L) ~= prod_i (I - dt L_i), so each
        factor is a tridiagonal solve along one axis. The Hamiltonian stays explicit, so the
        nonlinearity costs nothing extra. This is unconditionally stable in the diffusion and
        first-order accurate in dt for the splitting; it is a demonstration that the CFL
        restriction is removable, verified against the exact reference rather than asserted.

        Boundaries use a zero-second-derivative closure (the edge row is left as the identity, so
        no diffusion update is applied there), which keeps each tridiagonal system non-singular.
        """
        prob = self.problem
        D = np.diag(prob.Sigma @ prob.Sigma.T)
        out = rhs
        n = self.n_grid
        for i in range(prob.state_dim):
            r = 0.5 * D[i] * dt / self.dx[i] ** 2          # weight of the second difference
            ab = np.zeros((3, n))
            ab[1, :] = 1.0 + 2.0 * r
            ab[0, 1:] = -r
            ab[2, :-1] = -r
            ab[1, 0] = ab[1, -1] = 1.0                     # edge rows: identity (zero flux)
            ab[0, 1] = 0.0
            ab[2, -2] = 0.0
            moved = np.moveaxis(out, i, 0)
            shape = moved.shape
            sol = solve_banded((1, 1), ab, moved.reshape(shape[0], -1))
            out = np.moveaxis(sol.reshape(shape), 0, i)
        return out

    def _sweep(self, dt: float, snap_times: np.ndarray, policy_snap=None) -> np.ndarray:
        """Backward-time sweep from t=T to t=0, storing snapshots at `snap_times`.

        If `policy_snap` is None the Hamiltonian is maximised at every step (the true
        HJB). Otherwise it must be an array of shape (n_snap, N, p) giving the frozen
        policy u(x,t) at the snapshot times; it is interpolated linearly in t, so the
        policy is time-varying -- which is essential, because for this problem the
        optimal policy is u*(x,t) = -R^{-1} B' P(t) x with P strongly time-dependent.
        Freezing a single time-independent policy makes the policy-evaluation PDE
        inconsistent near t = T and diverges.
        """
        prob = self.problem
        V = self._terminal()
        n_steps = int(round(prob.T / dt))
        dt = prob.T / n_steps                      # exact step so we land on t=0
        out = np.empty((len(snap_times),) + self.shape)
        t_now = prob.T
        k = len(snap_times) - 1
        while k >= 0 and snap_times[k] >= t_now - 1e-12:
            out[k] = V
            k -= 1
        for step in range(n_steps):
            if policy_snap is None:
                H, _ = self._hamiltonian_and_control(V, t_now)
            else:
                # Interpolate the frozen policy in time.
                j = int(np.clip(np.searchsorted(snap_times, t_now), 1, len(snap_times) - 1))
                t0p, t1p = snap_times[j - 1], snap_times[j]
                w = 0.0 if t1p == t0p else float(np.clip((t_now - t0p) / (t1p - t0p), 0.0, 1.0))
                u = (1.0 - w) * policy_snap[j - 1] + w * policy_snap[j]
                x = self.X
                xq = np.einsum("...i,ij,...j->...", x, prob.Q, x)
                ur = np.einsum("...i,ij,...j->...", u, prob.R, u)
                # The frozen-policy PDE advects with the FULL velocity A x + B u, and all of it
                # is upwinded. Upwinding only B u while central-differencing A x (what an earlier
                # version did) is inconsistent, and with A != 0 it leaves the explicit step
                # outside its stability region. The two coincide when A = 0, which is exactly why
                # the defect was invisible in the inventory-only benchmark.
                v = x @ prob.A.T + u @ prob.B.T
                adv = self._upwind_advection(V, v)
                H = (-0.5 * (xq + ur)).reshape(self.shape) + adv
            if self.time_scheme == "lod":
                # Implicit diffusion, explicit Hamiltonian: (I - dt L) V_new = V + dt H(V).
                V = self._lod_step(V + dt * H, dt)
            else:
                V = V + dt * (H + self._diffusion(V))
            t_now -= dt
            while k >= 0 and snap_times[k] >= t_now - 1e-12:
                out[k] = V
                k -= 1
        self.n_time_steps = n_steps
        return out

    # ----------------------------------------------------------------------------------
    def _spatial_interp(self, snap: np.ndarray, X: np.ndarray) -> np.ndarray:
        rgi = RegularGridInterpolator(self.axes, snap, method="linear",
                                      bounds_error=False, fill_value=None)
        return rgi(np.clip(X, self.lo, self.hi))

    def value(self, X: np.ndarray, t: float) -> np.ndarray:
        """Interpolate the value function in space and time (linear in t)."""
        X = np.atleast_2d(X)
        ts = self.t_snap
        t = float(np.clip(t, ts[0], ts[-1]))
        j = int(np.clip(np.searchsorted(ts, t), 1, len(ts) - 1))
        t0, t1 = ts[j - 1], ts[j]
        w = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        V0 = self._spatial_interp(self.snapshots[j - 1], X)
        V1 = self._spatial_interp(self.snapshots[j], X)
        return (1.0 - w) * V0 + w * V1

    def control(self, X: np.ndarray, t: float) -> np.ndarray:
        """u* = R^{-1} B' grad V, with grad V by central differences in space.

        THIS IS THE ROUTINE THE PAPER IS ABOUT. Everything a grid solver is asked to *do* comes
        out of here, and the control is a derivative of the value function, so its accuracy is
        limited by how that derivative is taken -- not by how well the value function was solved.

        Three things are worth knowing before reading the body.

        (1) `self.value` interpolates the stored grid with a MULTILINEAR scheme, which is
            piecewise linear: its gradient is piecewise constant, a stair-step. So the gradient
            reported here is the slope of the cell the query point falls in, not the slope at the
            point itself.

        (2) The probe step `eps` is a FIXED 1e-4, independent of the grid spacing -- the worst of
            the available choices. `exp_control_extraction.py` measures the alternatives: a step
            scaled to the mesh (eps ~ dx/4) is 40-74x more accurate at no cost, and a CUBIC
            interpolant with the same probe is four to five orders of magnitude more accurate
            again. The fixed probe is kept here because this is the configuration the audited
            baseline used and the paper's comparison is against it; for the good behaviour use
            `exp_control_extraction.EXTRACTORS`.

        (3) Consequence upstream: the error here converges at order ~2.0 in the mesh spacing while
            the value error converges faster, so the control/value error ratio *grows* under
            refinement (2 -> 179 across N = 11..61 at p=2). Neither refining the grid nor refining
            time fixes it; the reconstruction order does.
        """
        X = np.atleast_2d(X)
        eps = 1e-4                      # probe step -- deliberately not scaled to dx; see (2)
        g = np.zeros_like(X)
        for i in range(self.problem.state_dim):
            # Perturb the query point along each axis and take a symmetric difference. Because
            # eps is tiny and the interpolant is piecewise linear, both perturbed points almost
            # always land in the same cell, so this returns that cell's slope.
            Xp = X.copy(); Xp[:, i] += eps
            Xm = X.copy(); Xm[:, i] -= eps
            g[:, i] = (self.value(Xp, t) - self.value(Xm, t)) / (2 * eps)
        # u* = R^{-1} B' grad V. R is diagonal in this benchmark, but R^{-1} is used rather than a
        # reciprocal so the same line is correct for a general positive-definite control cost; B
        # must appear explicitly, because it is the identity only in the inventory-only case.
        return g @ self.problem.B @ self.problem.R_inv.T


# ======================================================================================
# Finite difference
# ======================================================================================
class FiniteDifferenceSolver(_GridSolver):
    """Explicit backward-time finite differences; cost and memory scale as N^p.

    This is the method whose grid cost is the object of the dimensionality study: the
    original thesis asserted the curse of dimensionality but never measured it.
    """

    name = "Finite Difference"

    def solve(self) -> float:
        self._build_grid()
        if self.n_steps_forced is None:
            # First pass with a conservative dt to estimate the control magnitude.
            probe_dt = self._stable_dt(max_u=0.0)
            n_probe = max(int(np.ceil(self.problem.T / probe_dt)), 2)
            probe_dt = self.problem.T / n_probe
            self.t_snap = np.linspace(0.0, self.problem.T, 3)
            _ = self._sweep(probe_dt, self.t_snap)
            # Estimate |u*|_inf from the terminal-condition control, which bounds the
            # magnitude near t = T (largest in this problem).
            u_T = (-self.problem.QT @ self.X.T).T @ self.problem.B @ self.problem.R_inv.T
            max_u = float(np.max(np.abs(u_T))) * 1.05
            dt = self._stable_dt(max_u)
            n_steps = max(int(np.ceil(self.problem.T / dt)), 2)
        else:
            # Step count forced by the caller. Used to exercise the implicit (LOD) scheme at
            # steps far beyond the explicit CFL limit, which is the whole point of having it;
            # the probe sweep is skipped because it exists only to bound the explicit dt.
            n_steps = max(int(self.n_steps_forced), 2)
        dt = self.problem.T / n_steps

        # Cap snapshot memory at self.snap_cap float64 elements.
        n_snap = self.n_snap
        while n_snap > 5 and n_snap * self.N > self.snap_cap:
            n_snap //= 2
        self.n_snap_used = int(n_snap)
        self.t_snap = np.linspace(0.0, self.problem.T, n_snap)

        t0 = time.perf_counter()
        self.snapshots = self._sweep(dt, self.t_snap)
        self.solve_time = time.perf_counter() - t0
        return self.solve_time


# ======================================================================================
# Policy iteration
# ======================================================================================
class PolicyIterationSolver(_GridSolver):
    """Howard policy iteration on the same grid.

    Outer loop (policy improvement):  u_{k+1} = R^{-1} B' grad V_k
    Inner loop (policy evaluation):   backward-time solve of the linear PDE obtained by
                                      freezing u, iterated until the policy stops moving.

    Differs from the original implementation, which returned a stored ``policy`` array
    interpolated by nearest neighbour (``comparison_framework.py:420-433``) and was then
    used as the accuracy reference for every other method.
    """

    name = "Policy Iteration"

    def __init__(self, problem: ACProblem, n_grid: int = 41, domain_pad: float = 0.25,
                 n_snap: int = 41, max_policy_iter: int = 20, tol: float = 1e-10,
                 snap_cap: float = 4e7):
        super().__init__(problem, n_grid, domain_pad, n_snap, snap_cap=snap_cap)
        self.max_policy_iter = max_policy_iter
        self.tol = tol
        self.n_policy_iters = 0

    def solve(self) -> float:
        self._build_grid()
        u_T = (-self.problem.QT @ self.X.T).T @ self.problem.B @ self.problem.R_inv.T
        max_u = float(np.max(np.abs(u_T))) * 1.05
        dt = self._stable_dt(max_u)
        n_steps = max(int(np.ceil(self.problem.T / dt)), 2)
        dt = self.problem.T / n_steps

        n_snap = self.n_snap
        while n_snap > 5 and n_snap * self.N > self.snap_cap:
            n_snap //= 2
        self.n_snap_used = int(n_snap)
        self.t_snap = np.linspace(0.0, self.problem.T, n_snap)

        t0 = time.perf_counter()
        # Initial admissible policy: no trading, at every snapshot time.
        # Control dimension, NOT state dimension: the policy is a trading rate per asset.
        policy_snap = np.zeros((len(self.t_snap), self.N, self.problem.control_dim))
        snap = None
        for outer in range(self.max_policy_iter):
            snap = self._sweep(dt, self.t_snap, policy_snap=policy_snap)
            # Policy improvement: u(x,t) = R^{-1} B' grad V(x,t) at each snapshot time,
            # so the improved policy remains a function of both state and time.
            policy_new = np.stack([self._control_from_V(snap[k]) for k in range(len(self.t_snap))])
            delta = float(np.max(np.abs(policy_new - policy_snap)))
            policy_snap = policy_new
            self.n_policy_iters = outer + 1
            if delta < self.tol:
                break
        self.solve_time = time.perf_counter() - t0
        self.snapshots = snap
        return self.solve_time


# ======================================================================================
# Accuracy metrics against the analytic reference
# ======================================================================================
def accuracy_metrics(solver, problem: ACProblem, n_points: int = 2000,
                     n_times: int = 8, seed: int = 0) -> dict:
    """Score a solver against the analytic Riccati solution. THE HINGE OF THE WHOLE CODEBASE.

    Every solver -- finite differences, policy iteration, the PINN, and the PI-PINN -- exposes the
    same three methods (`solve`, `value`, `control`), so this function scores all of them without
    knowing which is which. If you understand the interface and this function, you understand the
    architecture; everything else is a driver that sweeps a parameter and writes JSON.

    It returns TWO R^2 values, and the paper's central claim is precisely that they disagree:

      value_r2 / control_r2      R^2 against the closed form. `value_r2` is the number the
                                 literature reports. `control_r2` is the number that matters,
                                 because the control is what gets executed.
      value_err / control_err    the same errors normalised by the spread of the target, i.e.
                                 1 - R^2. These are the quantities the convergence-order fits in
                                 exp_control_*.py regress on, because an order must be fitted to
                                 an error, not to an R^2 (an R^2 that saturates at 1 will show a
                                 spurious "high order" out of floating-point noise).
      control_sign_agreement     fraction of states where the control has the right sign. Reported
                                 because a solver can get the sign right while being badly wrong
                                 in magnitude -- which is exactly what a negative R^2 with high
                                 sign agreement means.

    Errors are averaged over several times, not just t=0, so that a solver which ignores the time
    dimension cannot score well. `seed` fixes the query points, so two solvers compared with the
    same seed are scored on identical states.
    """
    rng = np.random.default_rng(seed)
    lo, hi = problem.domain()
    X = rng.uniform(lo, hi, size=(n_points, problem.state_dim))
    times = np.linspace(0.0, problem.T, n_times)

    t_grid = np.linspace(0.0, problem.T, 401)
    ref = problem.reference(t_grid)

    V_hat, V_ref, u_hat, u_ref = [], [], [], []
    for t in times:
        V_hat.append(solver.value(X, t))
        V_ref.append(ref.value(X, t))
        u_hat.append(solver.control(X, t))
        u_ref.append(ref.control(X, t))
    V_hat = np.concatenate(V_hat); V_ref = np.concatenate(V_ref)
    u_hat = np.concatenate(u_hat); u_ref = np.concatenate(u_ref)

    def _stats(a, b):
        e = a - b
        ss_res = float(np.sum(e**2))
        ss_tot = float(np.sum((b - b.mean()) ** 2))
        return (
            float(np.mean(e**2)),
            float(np.mean(np.abs(e))),
            float(np.max(np.abs(e))),
            (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        )

    mse_v, mae_v, mx_v, r2_v = _stats(V_hat, V_ref)
    mse_u, mae_u, mx_u, r2_u = _stats(u_hat.ravel(), u_ref.ravel())
    return {
        "value_mse": mse_v, "value_mae": mae_v, "value_max_err": mx_v, "value_r2": r2_v,
        "control_mse": mse_u, "control_mae": mae_u, "control_max_err": mx_u, "control_r2": r2_u,
        "relative_value_l2": float(
            np.linalg.norm(V_hat - V_ref) / (np.linalg.norm(V_ref) + 1e-300)
        ),
        "control_sign_agreement": float(np.mean(np.sign(u_hat) == np.sign(u_ref))),
        "n_time_steps": getattr(solver, "n_time_steps", None),
    }


# ======================================================================================
# Deep BSDE: the third family, and the only one that is point-valued
# ======================================================================================


class DeepBSDESolver:
    """Deep BSDE (Han-E-Jentzen) adapted to the common interface, with the one difference kept loud.

    WHY THIS ADAPTER AND NOT A UNIFORM SUBCLASS
    -------------------------------------------
    The grid solvers and the PINN are FUNCTION-valued: one solve returns ``value(X, t)`` over any set
    of states. Deep BSDE is POINT-valued by construction. It parameterises ``V(x_0, 0)`` and ``Z``
    along simulated paths, so one training run answers one initial state, and there is no value
    function to query elsewhere. A subclass that silently returned something plausible for arbitrary
    ``X`` would be the single most damaging bug available in this repository: it would let a
    point-valued solver be scored as though it had solved the domain, in the direction that flatters
    the neural family, in a paper whose thesis is the opposite.

    So ``value`` and ``control`` here accept ONLY the state the solver was trained at, and raise
    otherwise. The limitation is enforced rather than documented, because documentation is what
    failed the first four times this project compared quantities that were not the same quantity.

    THE STOPPING RULE, AND WHY IT IS NOT DEFAULTED
    ----------------------------------------------
    In every run measured (see ``analysis_bsde_dims.md``) the terminal residual fell monotonically
    while the error in ``V(x0,0)`` bottomed out and then grew -- by a factor of 14 to 686 between the
    best point and the end of the run. Training to a residual plateau therefore overshoots, and the
    answer is only meaningful if training stopped near the value-error minimum.

    But no single threshold does that, and this is the part that is easy to get wrong. The residual
    level at the optimum, as a fraction of where the residual started, was 26% at d=1, 3.9% at d=4
    and 0.8% at d=8 -- a spread of 31x, and not monotone in any way that can be extrapolated. A
    caller who fixes ``stop_ratio`` at, say, 2% and applies it at d=1 stops roughly 30x too early and
    gets a value error of 66% instead of the 3.8% the same configuration reaches if trained properly.
    That was observed here, not hypothesised: the first version of this class defaulted to 2% and
    stopped at epoch 50 at d=1.

    So ``stop_ratio`` defaults to None, which trains the full budget and reproduces the measured
    runs. Setting it is a deliberate act, and the class reports ``start_residual``, ``stop_residual``
    and ``epochs_run`` so the choice is visible in the result rather than buried in a default.

    Interface deviation, stated so it cannot surprise a caller: ``solve()`` returns the wall-clock
    seconds for the run that was performed, and ``value``/``control`` return arrays of length 1.
    """

    name = "Deep BSDE"
    point_valued = True

    def __init__(self, problem: ACProblem, x0: np.ndarray | None = None, n_steps: int = 200,
                 n_paths: int = 1024, epochs: int = 8000, lr: float = 3e-3,
                 hidden: int = 64, layers: int = 4, seed: int = 0,
                 stop_ratio: float | None = None, log_every: int = 0,
                 device: str = "cuda"):
        self.problem = problem
        if x0 is None:
            # The benchmark's standard query: liquidate one unit of each asset.
            x0 = np.zeros(problem.state_dim)
            x0[: problem.n_assets] = 1.0
        self.x0 = np.asarray(x0, dtype=float)
        self.n_steps, self.n_paths, self.epochs = n_steps, n_paths, epochs
        self.stop_ratio, self.log_every, self.device = stop_ratio, log_every, device
        self._kw = dict(n_steps=n_steps, n_paths=n_paths, epochs=epochs, lr=lr, hidden=hidden,
                        layers=layers, seed=seed, device=device)
        self._solver = None
        self.start_residual: float | None = None
        self.stop_residual: float | None = None
        self.epochs_run: int | None = None
        self.solve_time: float | None = None

    # ----------------------------------------------------------------------------------
    def solve(self) -> float:
        """Train, stopping when the terminal residual has fallen to ``stop_ratio`` of its start.

        The early-stopping check is evaluated on a HELD-OUT batch rather than the training batch, so
        the criterion cannot be met by fitting the realisation currently in memory. When
        ``stop_ratio`` is None the full ``epochs`` budget is used, which reproduces the runs in
        ``analysis_bsde_dims.md`` and is what a caller should do if they intend to inspect the
        value/residual divergence themselves.
        """
        import torch                                    # local: keeps torch optional for the grid path

        from deep_bsde import DeepBSDE                    # reuse the validated implementation
        s = DeepBSDE(self.problem, self.x0, log_every=self.log_every, **self._kw)
        self._solver = s

        gen = torch.Generator(device=s.device).manual_seed(0)
        s._brownian(gen)
        opt = torch.optim.Adam(list(s.net.parameters()) + [s.theta], lr=s.lr)

        # ONE fixed held-out batch, used for every stopping decision, drawn once before training.
        #
        # Not a fresh batch per check: the terminal residual is estimated by Monte Carlo, so a fresh
        # draw each time compares two numbers whose difference includes estimation noise, and near
        # the threshold that noise decides whether training stops. Fixing the batch makes the
        # comparison monotone in the thing being measured. It is held out from training, so the
        # criterion still cannot be met by fitting the realisation currently in memory -- which is the
        # property that matters, and the reason a fixed TRAINING batch would be wrong.
        #
        # The batch width must equal n_paths: the base solver's rollout expands V from theta to
        # self.n_paths and will not accept another width. Using the training width therefore costs
        # nothing, and Widening the criterion batch would mean reimplementing the rollout -- which is
        # how the evaluation bug that inflated these floors by a factor of ~2 got in the first time.
        holdout = self._residual_batch(s, torch, n_paths=self.n_paths, seed=987_654_321)

        def residual() -> float:
            return self._residual(s, torch, holdout)

        t0 = time.perf_counter()
        done = 0
        for ep in range(self.epochs):
            # The base solver's own convention: reset dW and redraw, reseeded per epoch so theta is
            # identifiable (a fixed path set lets the network absorb a constant offset in theta).
            s.dW = None
            s._brownian(torch.Generator(device=s.device).manual_seed(1_000_000 + ep))
            opt.zero_grad(set_to_none=True)
            xT, VT = s._rollout(s.theta, s._grad_net)
            g = -0.5 * torch.einsum("ni,ij,nj->n", xT, s.QT, xT).unsqueeze(1)
            loss = ((VT - g) / s.V_scale).pow(2).mean()
            loss.backward()
            opt.step()
            done = ep + 1
            if ep == 0:
                self.start_residual = residual()
            if self.stop_ratio is not None and self.start_residual:
                # Checking every epoch would cost a full extra rollout per epoch; every 50 is ample
                # for a criterion whose own optimum moves by hundreds of epochs.
                if done % 50 == 0:
                    cur = residual()
                    if cur <= self.stop_ratio * self.start_residual:
                        self.stop_residual = cur
                        break
            if self.log_every and done % self.log_every == 0:
                print(f"      epoch {done:>6}  normalised resid "
                      f"{float(loss.sqrt()):.4e}", flush=True)
        self.solve_time = time.perf_counter() - t0
        self.epochs_run = done
        if self.stop_residual is None:
            self.stop_residual = residual()
        return self.solve_time

    # ----------------------------------------------------------------------------------
    @staticmethod
    def _residual_batch(solver, torch, n_paths: int, seed: int):
        """A fixed set of Brownian increments at the solver's own dt."""
        gen = torch.Generator(device=solver.device).manual_seed(seed)
        return torch.randn(solver.n_steps, n_paths, solver.d, device=solver.device,
                           generator=gen) * (solver.dt ** 0.5)

    @staticmethod
    def _residual(solver, torch, batch) -> float:
        """Absolute terminal RMSE of the current parameters on the given batch."""
        keep = solver.dW
        solver.dW = batch
        with torch.no_grad():
            xT, VT = solver._rollout(solver.theta, solver._grad_net)
            payoff = -0.5 * torch.einsum("ni,ij,nj->n", xT, solver.QT, xT).unsqueeze(1)
            r = float(((VT - payoff).pow(2).mean()).sqrt())
        solver.dW = keep
        return r

    # ----------------------------------------------------------------------------------
    def _check_states(self, X: np.ndarray) -> np.ndarray:
        """Accept only the trained state; refuse everything else, loudly.

        This is the whole point of the adapter. Returning a number for an arbitrary X would present a
        point-valued solver as a function-valued one.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        if X.shape[0] != 1 or not np.allclose(X[0], self.x0):
            raise ValueError(
                f"{self.name} is point-valued: it was trained at x0={self.x0.tolist()} and cannot be "
                f"queried at {X.shape[0]} other state(s). Retraining is required for a different "
                f"initial state. This refusal is deliberate -- see the class docstring.")
        return X

    def value(self, X: np.ndarray, t: float = 0.0) -> np.ndarray:
        if abs(t) > 1e-12:
            raise ValueError(f"{self.name} answers t=0 only (asked for t={t}).")
        self._check_states(X)
        return np.array([self._solver.value_at_x0()], dtype=float)

    def control(self, X: np.ndarray, t: float = 0.0) -> np.ndarray:
        if abs(t) > 1e-12:
            raise ValueError(f"{self.name} answers t=0 only (asked for t={t}).")
        self._check_states(X)
        return np.atleast_2d(self._solver.control_at_x0())

    def report(self) -> dict:
        """Everything a reader needs to judge the answer, including the stopping choice."""
        return {"name": self.name, "point_valued": True, "x0": self.x0.tolist(),
                "n_steps": self.n_steps, "epochs_run": self.epochs_run,
                "start_residual": self.start_residual, "stop_residual": self.stop_residual,
                "stop_ratio": self.stop_ratio, "seconds": self.solve_time}


if __name__ == "__main__":
    print("Solver validation against the analytic Riccati reference")
    print("=" * 86)
    ok = True
    for p in (1, 2, 3):
        prob = ACProblem(p=p)                     # well-scaled defaults
        print(f"\np = {p}   (eta={prob.eta}, lam_sigma2={prob.lam_sigma2}, "
              f"Q_T={prob.q_terminal}, kappa={np.sqrt(prob.lam_sigma2/prob.eta):.3f})")
        for cls, kw in ((FiniteDifferenceSolver, {"n_grid": 41}),
                        (PolicyIterationSolver, {"n_grid": 41, "max_policy_iter": 15})):
            s = cls(prob, **kw)
            st = s.solve()
            m = accuracy_metrics(s, prob, n_points=600, seed=1)
            extra = ""
            if isinstance(s, PolicyIterationSolver):
                extra = f" pol_iters={s.n_policy_iters}"
            print(f"  {s.name:20s} solve={st:7.3f}s steps={m['n_time_steps']:5d} "
                  f"value_MSE={m['value_mse']:.3e} value_R2={m['value_r2']:+.6f} "
                  f"ctrl_R2={m['control_r2']:+.6f} sign_agree={m['control_sign_agreement']:.3f}"
                  f"{extra}")
            u = s.control(np.full((16, p), 0.5), 0.0)
            if not np.all(u < 0):
                print(f"    !! {s.name}: control sign inverted")
                ok = False
            if not (m["value_r2"] > 0.99):
                print(f"    !! {s.name}: value R2 below 0.99")
                ok = False
    print("\n" + ("ALL SOLVER CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
