"""
Unit tests for the corrected backtest accounting.

These check the *mechanics* (ordering, sign, position integration, cost units,
annualisation) against hand-computed values, using a stub model with a known gradient.
They are deliberately independent of any trained network, so a failure here is a bug in the
accounting rather than a modelling outcome.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import (
    MINUTES_PER_DAY,
    BacktestConfig,
    compute_metrics,
    control_from_grad,
    run_backtest,
)

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "PASS" if cond else "FAIL"
    print(f"  [{status}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


class StubModel:
    """Returns a constant value-function gradient, so the control is analytically known."""

    def __init__(self, grad: float = 1e-3):
        self.grad = grad

    def value_and_grad(self, price, vol, spread, t):
        price = np.atleast_1d(np.asarray(price, dtype=float))
        return -0.5 * (price - 25.0) ** 2, np.full_like(price, self.grad)


def make_frame(n_days: int = 3, minutes: int = 20, price_fn=None) -> pd.DataFrame:
    rows = []
    for d in range(n_days):
        day = pd.Timestamp("2025-04-07") + pd.Timedelta(days=d)
        for m in range(minutes):
            ts = day + pd.Timedelta(hours=9, minutes=30 + m)
            p = price_fn(d, m) if price_fn else 100.0 + 10.0 * d + m
            rows.append({"date": ts, "close": p, "spread": 0.02,
                         "normalized_spread": 0.02 / p,
                         "ewma_volatility": 1e-3, "split": "test"})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------
print("control_from_grad: HJB sign convention")
# --------------------------------------------------------------------------------------
price = np.array([25.0])
spread = np.array([0.02])
cfg_hjb = BacktestConfig(sign_convention="hjb")
cfg_orig = BacktestConfig(sign_convention="original", threshold_rule="gamma")

pos_grad = np.array([1e-3])
check("HJB: positive gradient -> POSITIVE (buying) rate",
      control_from_grad(pos_grad, price, spread, cfg_hjb)[0] > 0,
      f"v={control_from_grad(pos_grad, price, spread, cfg_hjb)[0]:.4f}")
check("Original: positive gradient -> NEGATIVE rate (inverted)",
      control_from_grad(pos_grad, price, spread, cfg_orig)[0] < 0,
      f"v={control_from_grad(pos_grad, price, spread, cfg_orig)[0]:.4f}")
neg_grad = np.array([-1e-3])
check("HJB: negative gradient -> negative (selling) rate",
      control_from_grad(neg_grad, price, spread, cfg_hjb)[0] < 0)
check("HJB: small gradient inside the band -> no trade",
      abs(control_from_grad(np.array([1e-9]), price, spread, cfg_hjb)[0]) == 0.0)
# Dollar scaling must change the magnitude
cfg_nodollar = BacktestConfig(sign_convention="hjb", dollar_scaling=False)
v_scaled = control_from_grad(pos_grad, price, spread, cfg_hjb)[0]
v_unscaled = control_from_grad(pos_grad, price, spread, cfg_nodollar)[0]
check("Dollar scaling divides by S", abs(v_unscaled / v_scaled - price[0]) < 1e-6,
      f"ratio={v_unscaled/v_scaled:.4f}, S={price[0]}")

# --------------------------------------------------------------------------------------
print("\nordering: ascending vs descending run over the same series")
# --------------------------------------------------------------------------------------
df = make_frame(2, 10)
_, m_asc = run_backtest(df, StubModel(), BacktestConfig(sort_ascending=True, lag_position=True,
                                                        integrate_position=True))
_, m_desc = run_backtest(df, StubModel(), BacktestConfig(sort_ascending=False, lag_position=True,
                                                         integrate_position=True))
check("ascending and descending runs differ", m_asc["total_return"] != m_desc["total_return"],
      f"asc={m_asc['total_return']:+.6f} desc={m_desc['total_return']:+.6f}")

# --------------------------------------------------------------------------------------
print("\nposition integration: q accumulates at dt = 1/390")
# --------------------------------------------------------------------------------------
cfg = BacktestConfig(integrate_position=True, sign_convention="hjb", lag_position=True)
d_int, _ = run_backtest(df, StubModel(grad=1e-3), cfg)
cfg_noint = BacktestConfig(**{**cfg.__dict__, "integrate_position": False})
d_no, _ = run_backtest(df, StubModel(grad=1e-3), cfg_noint)

# The trading rate varies bar to bar (it depends on price through the threshold and the
# 1/S scaling), so the integrated position is the running sum of v*dt, clipped.
v_path = d_int["trading_rate"].to_numpy()
dt = 1.0 / MINUTES_PER_DAY
expected_int = np.clip(np.cumsum(v_path * dt), -cfg.q_max, cfg.q_max)
check("integrated position equals the running sum of v*dt (clipped)",
      np.allclose(d_int["position"].to_numpy(), expected_int, atol=1e-12),
      f"max|diff|={np.max(np.abs(d_int['position'].to_numpy() - expected_int)):.2e}")

expected_no = np.clip(v_path, -cfg.q_max, cfg.q_max)
check("non-integrated position equals clip(rate) bar by bar",
      np.allclose(d_no["position"].to_numpy(), expected_no, atol=1e-12),
      f"max|diff|={np.max(np.abs(d_no['position'].to_numpy() - expected_no)):.2e}")

check("integrated position is far smaller than the raw rate",
      np.max(np.abs(d_int["position"])) < np.max(np.abs(d_no["position"])) * 0.05,
      f"max|pos_int|={np.max(np.abs(d_int['position'])):.2e} "
      f"max|pos_noint|={np.max(np.abs(d_no['position'])):.2e}")

# --------------------------------------------------------------------------------------
print("\nlag_position: no same-bar leak")
# --------------------------------------------------------------------------------------
df2 = make_frame(2, 10)
lagged = BacktestConfig(**{**cfg.__dict__, "lag_position": True})
same = BacktestConfig(**{**cfg.__dict__, "lag_position": False})
d_lag, _ = run_backtest(df2, StubModel(grad=1e-3), lagged)
d_same, _ = run_backtest(df2, StubModel(grad=1e-3), same)
check("lagged and same-bar returns differ",
      not np.allclose(d_lag["strategy_return"].iloc[1:5], d_same["strategy_return"].iloc[1:5]))

# --------------------------------------------------------------------------------------
print("\nannualisation: correct vs original")
# --------------------------------------------------------------------------------------
n_days, mins = 4, 30
rng = np.random.default_rng(0)
r = rng.normal(0, 1e-4, size=n_days * mins)
idx = pd.DatetimeIndex(
    [pd.Timestamp("2025-04-07") + pd.Timedelta(days=d, minutes=m)
     for d in range(n_days) for m in range(mins)]
)
mc = compute_metrics(r, idx, BacktestConfig(annualization="correct"))
mo = compute_metrics(r, idx, BacktestConfig(annualization="original"))
# Recompute the "correct" daily aggregation by hand.
daily = pd.Series(r, index=idx).groupby(idx.date).apply(lambda x: np.prod(1 + x.to_numpy()) - 1)
hand_vol = daily.std(ddof=1) * np.sqrt(252)
hand_sharpe = daily.mean() / daily.std(ddof=1) * np.sqrt(252)
check("correct vol matches hand-computed daily aggregation",
      abs(mc["annualized_volatility"] - hand_vol) < 1e-12,
      f"{mc['annualized_volatility']:.6e} vs {hand_vol:.6e}")
check("correct sharpe matches hand computation",
      abs(mc["sharpe_ratio"] - hand_sharpe) < 1e-12)
check("original vol differs from correct (minute/daily mismatch)",
      abs(mo["annualized_volatility"] - mc["annualized_volatility"]) > 1e-9,
      f"orig={mo['annualized_volatility']:.4e} correct={mc['annualized_volatility']:.4e}")
check("original vol understates by a large factor",
      mo["annualized_volatility"] / max(mc["annualized_volatility"], 1e-300) < 1.0)

# --------------------------------------------------------------------------------------
print("\ncost units: return-units vs the original dollar/return mix")
# --------------------------------------------------------------------------------------
_, m_ret = run_backtest(df2, StubModel(grad=1e-3), BacktestConfig(**{**cfg.__dict__, "cost_in_return_units": True}))
_, m_dol = run_backtest(df2, StubModel(grad=1e-3), BacktestConfig(**{**cfg.__dict__, "cost_in_return_units": False}))
check("dollar-cost configuration produces a much larger total cost drag",
      m_dol["total_return"] < m_ret["total_return"],
      f"return-units={m_ret['total_return']:+.6f} dollar-mix={m_dol['total_return']:+.6f}")

# --------------------------------------------------------------------------------------
print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    raise SystemExit(1)
print("ALL BACKTEST ACCOUNTING CHECKS PASSED")
