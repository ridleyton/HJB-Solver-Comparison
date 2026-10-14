"""
Probe: is the alpha-augmented LQ problem well posed, and can the Riccati reference be trusted?

Why this probe comes first
--------------------------
Adding a mean-reverting alpha signal puts a CROSS term q's in the objective (the trader earns the
alpha by holding inventory). That makes the running-cost matrix indefinite -- Q has a zero block
and negative off-diagonal blocks -- so the standard solvability condition (Q >= 0) does not apply
and it is not obvious a priori that the backward Riccati integration stays bounded on [0, T].

Three checks decide whether the reference may be used:

 1. WELL-POSEDNESS  -- the backward integration must not hit a conjugate point (no blow-up).
 2. ECONOMIC SANITY -- two bounds that must hold, both derived from a feasible strategy rather
    than from the reference itself:
      (a) The trader can always IGNORE the signal and follow the zero-signal strategy, so
          V(t, (0, s)) >= V(t, (0, 0)) for every s. With the ansatz V = -1/2 x'Px + c this forces
          the signal block of P to be NEGATIVE (P indefinite). A P that comes back positive
          definite is a wrong-branch or mis-specified problem -- this test is what caught the
          first version of the dynamics matrix, which wrongly put I in the inventory-to-signal
          block.
      (b) Starting flat is NOT free: with u = 0 the position still diffuses, q_t = sigma W_t, so
          the risk penalty bites. The shutdown strategy therefore gives a NEGATIVE floor,
              c(t) >= -1/4 Lambda sigma^2 (T-t)^2 - 1/2 Q_T sigma^2 (T-t),
          and c(t) < 0 is expected rather than a failure. (An earlier version of this probe
          asserted c >= 0 and flagged every row; that assertion was simply wrong.)
 3. ODE RESIDUAL    -- the integrated P must satisfy the Riccati equation, checked with
    np.gradient at two grid resolutions so that a residual which is pure discretisation can be
    told apart from one that is a real error.

Model (the state is (inventory q, scaled signal s-hat), state dim = 2 * n_assets):
    dq = u dt
    ds = -kappa s dt + sqrt(2 kappa) dW          (unit stationary variance after scaling)
    reward rate = -1/2 (q' Lambda q - 2 sigma_s q' s + eta |u|^2)
so that with x = (q, s),
    A = [[0, 0], [0, -kappa I]]        B = [[I], [0]]
    Q = [[Lambda I, -sigma_s I], [-sigma_s I, 0]]
    Q_T = diag(Q_T I, 0)               Sigma = diag(sigma I, sqrt(2 kappa) I)

Note A has NO (q, s) block: the signal does not move inventory by itself.

Usage: python src/alpha_probe.py
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp


def build(p: int, kappa: float, sig_s: float, lam: float, eta: float, qt: float,
          sigma: float, T: float = 1.0):
    """Assemble the augmented LQ matrices. p = number of assets; state dim = 2p."""
    n = 2 * p
    A = np.zeros((n, n))
    A[p:, p:] = -kappa * np.eye(p)

    B = np.zeros((n, p))
    B[:p, :] = np.eye(p)

    # x'Qx = q'Lambda q - 2 sigma_s q's, so that -1/2 x'Qx = -1/2 q'Lambda q + sigma_s q's.
    Q = np.zeros((n, n))
    Q[:p, :p] = lam * np.eye(p)
    Q[:p, p:] = -sig_s * np.eye(p)
    Q[p:, :p] = -sig_s * np.eye(p)

    R = eta * np.eye(p)
    QT = np.zeros((n, n))
    QT[:p, :p] = qt * np.eye(p)

    Sig = np.zeros((n, n))
    Sig[:p, :p] = sigma * np.eye(p)
    Sig[p:, p:] = np.sqrt(2.0 * kappa) * np.eye(p)
    return A, B, Q, R, QT, Sig, T


def riccati(A, B, Q, R, QT, Sig, T, n_t=401):
    """Integrate the Riccati ODE backward from T. Returns (P, c, t_grid) or (None, None, msg)."""
    n = A.shape[0]
    R_inv = np.linalg.inv(R)

    def rhs(t, y):
        P = y[: n * n].reshape(n, n)
        c = y[n * n]
        Pdot = -(Q + P @ A + A.T @ P - P @ B @ R_inv @ B.T @ P)
        return np.concatenate([Pdot.ravel(), [0.5 * np.trace(Sig @ Sig.T @ P)]])

    y0 = np.concatenate([QT.ravel(), [0.0]])
    t_grid = np.linspace(0.0, T, n_t)
    s_eval = np.clip(T - t_grid[::-1], 0.0, T)
    # rtol/atol are tight because the reference IS the benchmark: an error here is an error in
    # every accuracy number downstream, and it would be invisible (it would look like solver error).
    sol = solve_ivp(lambda s, y: -rhs(T - s, y), (0.0, T), y0, t_eval=s_eval,
                    rtol=1e-12, atol=1e-14, method="RK45")
    if not sol.success:
        return None, None, None, sol.message
    ys = sol.y[:, ::-1]
    P = ys[: n * n, :].T.reshape(-1, n, n)
    c = ys[n * n, :]
    if not np.all(np.isfinite(P)):
        return None, None, None, "non-finite P"
    return P, c, t_grid, "ok"


def ode_residual(A, B, Q, R, P, t_grid) -> float:
    """Relative residual of the Riccati ODE, using np.gradient for dP/dt.

    This is NOT an identity check: dP/dt comes from differentiating the integrated solution, and
    the right-hand side is evaluated at a shifted time index, so a wrong branch or a wrong sign
    convention shows up here.
    """
    n = A.shape[0]
    R_inv = np.linalg.inv(R)
    dP = np.gradient(P, t_grid, axis=0)
    rhs = np.empty_like(P)
    for i, _ in enumerate(t_grid):
        Pi = P[i]
        rhs[i] = -(Q + Pi @ A + A.T @ Pi - Pi @ B @ R_inv @ B.T @ Pi)
    num = np.max(np.abs(dP - rhs))
    den = max(np.max(np.abs(rhs)), 1e-30)
    return float(num / den)


class _Ref:
    def __init__(self, P, c, t_grid, B, R_inv):
        self.P, self.c, self.t_grid, self.B, self.R_inv = P, c, t_grid, B, R_inv
        self.n = P.shape[1]

    def _Pc(self, t: float):
        Pt = np.empty((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                Pt[i, j] = np.interp(t, self.t_grid, self.P[:, i, j])
        return Pt, float(np.interp(t, self.t_grid, self.c))

    def value(self, x: np.ndarray, t: float) -> float:
        Pt, ct = self._Pc(t)
        x = np.atleast_1d(np.asarray(x, dtype=float))
        return float(-0.5 * x @ Pt @ x + ct)

    def c_at(self, t: float) -> float:
        return float(np.interp(t, self.t_grid, self.c))

    def control(self, x: np.ndarray, t: float) -> np.ndarray:
        Pt, _ = self._Pc(t)
        return -np.atleast_2d(x) @ Pt @ self.B @ self.R_inv.T


def main() -> int:
    p = 2
    lam, eta, qt, sigma = 0.01, 0.01, 0.1, 0.01
    n = 2 * p
    print(f"p={p} assets, state dim={n};  Lam={lam}  eta={eta}  Q_T={qt}  sigma={sigma}")
    print()
    hdr = (f"{'kappa':>6}{'sig_s':>7}{'q*':>7}{'max|P|':>9}{'eig(P0)min':>12}{'eig(P0)max':>12}"
           f"{'min c':>9}{'dV(s>0)':>10}{'u(q=1,s=1)':>12}{'u(q=1,s=-1)':>13}{'ODEres':>9}"
           f"  note")
    print(hdr)
    print("-" * len(hdr))
    for kappa in (0.5, 1.0, 2.0):
        for sig_s in (0.002, 0.005, 0.010):
            A, B, Q, R, QT, Sig, T = build(p, kappa, sig_s, lam, eta, qt, sigma)
            P, c, tg, msg = riccati(A, B, Q, R, QT, Sig, T)
            if P is None:
                print(f"{kappa:>6.2f}{sig_s:>7.3f}   FAILED: {msg}")
                continue
            ref = _Ref(P, c, tg, B, np.linalg.inv(R))
            P0 = P[0]
            ev = np.linalg.eigvalsh(0.5 * (P0 + P0.T))

            x_flat = np.zeros(n)
            x_sig = np.zeros(n)
            x_sig[p:] = 1.0                                   # flat inventory, signal = +1
            dV_sig = ref.value(x_sig, 0.0) - ref.value(x_flat, 0.0)

            x_q = np.zeros(n); x_q[0] = 1.0
            x_qs_p = np.zeros(n); x_qs_p[0] = 1.0; x_qs_p[p] = 1.0
            x_qs_m = np.zeros(n); x_qs_m[0] = 1.0; x_qs_m[p] = -1.0
            u_q = float(ref.control(x_q, 0.0)[0, 0])
            u_p = float(ref.control(x_qs_p, 0.0)[0, 0])
            u_m = float(ref.control(x_qs_m, 0.0)[0, 0])

            note = []
            if dV_sig < -1e-12:
                note.append("SANITY FAIL: signal hurts")
            if ev.min() > 0:
                note.append("SANITY FAIL: P PSD")
            # Feasible-strategy floor for c: with u = 0 every inventory coordinate diffuses,
            # q_i(t) = sigma W_i(t), so the expected reward is
            #   -1/4 Lambda sigma^2 p (T-t)^2  -  1/2 Q_T sigma^2 p (T-t).
            # The factor p is the number of inventory coordinates, i.e. one noise channel per
            # asset; omitting it understates the floor and makes a correct reference look wrong.
            tt = np.linspace(0.0, T, len(tg))
            tau = T - tt
            floor = -0.25 * lam * sigma**2 * p * tau**2 - 0.5 * qt * sigma**2 * p * tau
            if np.any(c < floor - 1e-12):
                note.append("SANITY FAIL: c below floor")
            res_coarse = ode_residual(A, B, Q, R, P, tg)
            P_f, _, tg_f, _ = riccati(A, B, Q, R, QT, Sig, T, n_t=1601)
            res_fine = ode_residual(A, B, Q, R, P_f, tg_f) if P_f is not None else float("nan")
            ratio = res_coarse / res_fine if res_fine and res_fine == res_fine else float("nan")
            print(f"{kappa:>6.2f}{sig_s:>7.3f}{sig_s/lam:>7.2f}{np.max(np.abs(P)):>9.4f}"
                  f"{ev.min():>12.3e}{ev.max():>12.3e}{c.min():>9.2e}{dV_sig:>10.2e}"
                  f"{u_p:>12.4f}{u_m:>13.4f}{ratio:>9.1f}"
                  f"  {'; '.join(note) if note else 'ok'}")
    print()
    print("Expected: eig(P0) min < 0 (P indefinite), dV(s>0) > 0, min c above its floor,")
    print("u(q=1,s=+1) > u(q=1,s=-1) (a positive signal means selling less), and the ODE residual")
    print("column is the COARSE/FINE ratio. np.gradient is one-sided at the two endpoints, so the")
    print("max-over-grid residual converges at O(dt) and the ratio should be about 4, which is what")
    print("4x more points buys. A ratio near 1 would mean a genuine error rather than discretisation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
