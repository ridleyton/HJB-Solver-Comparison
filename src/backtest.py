"""
Corrected backtest engine, and a defect ablation over the original pipeline.

Why an ablation
---------------
The original backtest contained several independent defects, each of which changes the
reported performance (`hft_backtest.py`, `comparison_framework.py:890-999`):

  D1 ordering      the shipped CSVs are newest-first and were never sorted, so
                   ``price[i-1]`` is the *later* minute. On the shipped SPY file the
                   summed price return over the file order is +0.037950 against -0.021709
                   in true chronological order: the P&L sign is inverted.
  D2 same-bar leak position is derived from bar *i*'s close and then multiplied by the
                   return from bar *i* to bar *i-1*.
  D3 control sign  ``optimal_rate = -grad V / (2 * market_impact)`` in all three call
                   sites, whereas the first-order condition of the thesis's own HJB
                   (`main.tex:363-379`) gives a *positive* rate when ``grad V > 0``. On the
                   shipped SPY file every minute classified as a "buy signal" (19.1%) has a
                   negative trading rate -- a strict, pointwise sign inversion.
  D4 dollar scale  the HJB control carries a ``1/(2 kappa S)`` factor; the code omits the
                   ``S``.
  D5 position      ``position = clip(trading_rate, -1, 1)``, so "position" is the rate
                   itself rather than its integral. The thesis's own model says
                   ``dq = v dt`` (`main.tex:352`).
  D6 cost units    ``cost = gamma |trade| price`` is in dollars while
                   ``gross = position * price_return`` is a return; costs therefore swamp
                   the signal by orders of magnitude.
  D7 annualisation ``(1+total)^(252/N_minutes)`` and ``std(minute)*sqrt(252)`` mix a
                   minute-level series with daily annualisation factors, so every reported
                   volatility and Sharpe ratio is uninterpretable.
  D8 input scale   the network was trained on price in [20, 30] and evaluated on raw
                   prices of 40 to 550.

Fixing these one at a time, and then all together, shows how much of the original result
was an accounting artefact rather than a property of the strategy. That quantification is
the empirical core of the methodological contribution.

Usage: see `--config single` / `all` below.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

MINUTES_PER_DAY = 390.0
TRADING_DAYS = 252.0


# ======================================================================================
# Configuration: one flag per defect
# ======================================================================================
@dataclass
class BacktestConfig:
    # D1: process bars in ascending time order (correct) or file order (original defect)
    sort_ascending: bool = True
    # D2: use the previous bar's position for the current bar's return
    lag_position: bool = True
    # D3: 'hjb' = control from the HJB first-order condition (correct); 'original' = -V_S/(2k)
    sign_convention: str = "hjb"
    # D4: include the 1/S scaling in the control
    dollar_scaling: bool = True
    # Threshold rule for the no-trade band
    threshold_rule: str = "gamma_spread_price"   # or 'gamma' (hft_backtest.py:110)
    # D5: integrate the rate into inventory
    integrate_position: bool = True
    # D6: express costs in return units
    cost_in_return_units: bool = True
    # D7: correct annualisation
    annualization: str = "correct"               # or 'original'
    # D8: map price into the training domain [20, 30] before evaluation
    scale_inputs: bool = True
    # Inventory limits
    q_max: float = 1.0
    # Cost parameters (from side_info.json)
    gamma: float = 0.0001
    kappa: float = 0.0005
    train_price_lo: float = 20.0
    train_price_hi: float = 30.0

    @staticmethod
    def as_published() -> "BacktestConfig":
        return BacktestConfig(
            sort_ascending=False, lag_position=False, sign_convention="original",
            dollar_scaling=False, threshold_rule="gamma", integrate_position=False,
            cost_in_return_units=False, annualization="original", scale_inputs=False,
        )

    @staticmethod
    def all_fixed() -> "BacktestConfig":
        return BacktestConfig()


# ======================================================================================
# Performance metrics
# ======================================================================================
def compute_metrics(strategy_returns: np.ndarray, minute_index: pd.DatetimeIndex,
                    cfg: BacktestConfig) -> dict:
    """Standard performance metrics.

    Correct annualisation aggregates minute returns into daily returns first, then scales
    the daily series by 252. The original instead exponentiated by ``252 / N_minutes`` and
    scaled the minute standard deviation by ``sqrt(252)``.
    """
    r = np.asarray(strategy_returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return {"total_return": 0.0, "annualized_return": 0.0,
                "annualized_volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}

    total = float(np.prod(1.0 + r) - 1.0)
    cum = np.cumprod(1.0 + r)
    dd = float((cum / np.maximum.accumulate(cum) - 1.0).min())

    if cfg.annualization == "original":
        # As published: mixes minute counts with a daily factor.
        ann = float((1.0 + total) ** (TRADING_DAYS / len(r)) - 1.0) if total > -1 else -1.0
        vol = float(np.std(r, ddof=1) * np.sqrt(TRADING_DAYS)) if len(r) > 1 else 0.0
        sharpe = float(ann / vol) if vol > 0 else 0.0
    else:
        # Aggregate to daily returns on the trading dates actually present.
        days = pd.Series(r, index=minute_index).groupby(minute_index.date)
        daily = days.apply(lambda x: float(np.prod(1.0 + x.to_numpy()) - 1.0))
        n_days = len(daily)
        ann = float((1.0 + total) ** (TRADING_DAYS / n_days) - 1.0) if (n_days > 0 and total > -1) else -1.0
        vol = float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if n_days > 1 else 0.0
        sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(TRADING_DAYS)) if (n_days > 1 and daily.std(ddof=1) > 0) else 0.0

    return {
        "total_return": total, "annualized_return": ann,
        "annualized_volatility": vol, "sharpe_ratio": sharpe, "max_drawdown": dd,
    }


# ======================================================================================
# Control law
# ======================================================================================
def control_from_grad(grad_V: np.ndarray, price: np.ndarray, spread: np.ndarray,
                      cfg: BacktestConfig) -> np.ndarray:
    """Trading rate implied by the value-function gradient.

    Correct form (thesis's own first-order condition, main.tex:371-379):
        v = (V_S - gamma S spread) / (2 kappa S)      if V_S >  gamma S spread
        v = 0                                          if |V_S| <= gamma S spread
        v = (V_S + gamma S spread) / (2 kappa S)      if V_S < -gamma S spread
    so a positive gradient (value rising in price) produces a *positive* (buying) rate.

    Original form: v = -V_S / (2 kappa), with the threshold taken either as
    ``gamma * normalized_spread * close`` (main.py:275) or the flat constant ``gamma``
    (hft_backtest.py:110) -- the two implementations disagree.
    """
    g = np.asarray(grad_V, dtype=float)
    S = np.asarray(price, dtype=float)
    sp = np.asarray(spread, dtype=float)

    if cfg.sign_convention == "original":
        thr = (cfg.gamma * sp * S) if cfg.threshold_rule == "gamma_spread_price" \
            else np.full_like(S, cfg.gamma)
        v = -g / (2.0 * cfg.kappa)
        v = np.where(np.abs(g) > thr, v, 0.0)
        return v

    thr = cfg.gamma * S * sp
    denom = (2.0 * cfg.kappa * S) if cfg.dollar_scaling else (2.0 * cfg.kappa)
    v = np.where(g > thr, (g - thr) / denom,
                 np.where(g < -thr, (g + thr) / denom, 0.0))
    return v


# ======================================================================================
# Backtest engine
# ======================================================================================
def run_backtest(df: pd.DataFrame, model, cfg: BacktestConfig) -> tuple[pd.DataFrame, dict]:
    """Run one backtest configuration.

    Parameters
    ----------
    df : DataFrame with columns date, close, spread, normalized_spread, and whatever the
         model needs.
    model : object with ``value_and_grad(price, vol, spread, t) -> (V, dV/dS)``.
    """
    d = df.copy()
    if cfg.sort_ascending:
        d = d.sort_values("date").reset_index(drop=True)
    else:
        # Reproduce the shipped file order (newest-first) without sorting.
        d = d.sort_values("date", ascending=False).reset_index(drop=True)

    price = d["close"].to_numpy(dtype=float)
    vol = d["ewma_volatility"].to_numpy(dtype=float)
    spread = d["spread"].to_numpy(dtype=float)
    spread_prop = d["normalized_spread"].to_numpy(dtype=float)
    t_norm = np.full_like(price, 0.5)

    eval_price = price.astype(float)
    jac = 1.0
    if cfg.scale_inputs:
        # Affine-map the realised price range into the training domain, then undo the
        # mapping on the returned gradient so the control is expressed in true units.
        # The Jacobian is the affine slope, computed in closed form: taking it by
        # np.gradient(eval_price, price) divides by zero whenever two consecutive minutes
        # share a price, which happens often at minute frequency.
        lo, hi = float(np.min(price)), float(np.max(price))
        if hi - lo < 1e-12:
            hi = lo + 1.0
        a = (cfg.train_price_hi - cfg.train_price_lo) / (hi - lo)
        eval_price = cfg.train_price_lo + (price - lo) * a
        jac = a

    # Models may return (V, dV/dS) or (V, dV/dS, mapped_price); accept both.
    out = model.value_and_grad(eval_price, vol, spread, t_norm)
    V, g_eval = out[0], out[1]
    grad_V = g_eval * jac                      # chain rule back to true price units

    v = control_from_grad(grad_V, price, spread if cfg.dollar_scaling else spread_prop, cfg)

    dt = 1.0 / MINUTES_PER_DAY
    n = len(d)
    position = np.zeros(n)
    trade = np.zeros(n)
    ret = np.zeros(n)
    q = 0.0
    prev_pos = 0.0

    for i in range(n):
        if cfg.integrate_position:
            q = float(np.clip(q + v[i] * dt, -cfg.q_max, cfg.q_max))
            pos = q
        else:
            pos = float(np.clip(v[i], -cfg.q_max, cfg.q_max))
        trade[i] = pos - prev_pos

        if i == 0:
            position[i] = pos
            prev_pos = pos
            continue

        # Return over the interval ending at bar i.
        if cfg.sort_ascending:
            pr = price[i] / price[i - 1] - 1.0        # bar i-1 -> bar i (past -> present)
        else:
            pr = price[i] / price[i - 1] - 1.0        # reversed order: present -> past

        # Position earning the return: with lag_position the position was set on the
        # previous bar (no same-bar information); otherwise the current bar's own position
        # is used, which leaks the current close into the current return.
        pos_for_return = prev_pos if cfg.lag_position else pos

        # Costs
        dq = abs(trade[i])
        if cfg.cost_in_return_units:
            # Half-spread paid on turnover, plus a quadratic impact term, both as returns.
            cost = 0.5 * spread_prop[i] * dq + cfg.kappa * dq * dq
        else:
            # As published: dollars divided into a return series.
            cost = cfg.gamma * dq * price[i] + cfg.kappa * dq * dq * price[i]

        ret[i] = pos_for_return * pr - cost
        position[i] = pos
        prev_pos = pos

    d["position"] = position
    d["trade"] = trade
    d["trading_rate"] = v
    d["price_sensitivity"] = grad_V
    d["strategy_return"] = ret

    idx = pd.DatetimeIndex(d["date"])
    metrics = compute_metrics(ret, idx, cfg)
    metrics["mean_abs_position"] = float(np.mean(np.abs(position)))
    metrics["turnover"] = float(np.sum(np.abs(trade)))
    metrics["no_trade_fraction"] = float(np.mean(np.abs(v) < 1e-12))
    metrics["gross_sum"] = float(np.sum(position * (np.diff(price, prepend=price[0]) / price)))
    return d, metrics


# ======================================================================================
# Driver
# ======================================================================================
def ablation_configs() -> dict[str, BacktestConfig]:
    """As-published, then each defect fixed in isolation, then all fixed."""
    base = BacktestConfig.as_published()
    out: dict[str, BacktestConfig] = {"as_published": base}

    fixes = {
        "fix_D1_ordering": ("sort_ascending", True),
        "fix_D2_same_bar_leak": ("lag_position", True),
        # D3 (sign) and D4 (missing 1/S) are not separable: the released control,
        # v = -V_S / (2 kappa), omits S altogether, so "fixing D4 alone" changes nothing.
        # They are therefore treated as one control-law defect and fixed together.
        "fix_D3_control_law": ("sign_convention", "hjb"),
        "fix_D5_position_integration": ("integrate_position", True),
        "fix_D6_cost_units": ("cost_in_return_units", True),
        "fix_D7_annualization": ("annualization", "correct"),
        "fix_D8_input_scaling": ("scale_inputs", True),
    }
    for name, (field, cval) in fixes.items():
        cfg = BacktestConfig(**{**asdict(base), field: cval})
        if field == "sign_convention":
            # Adopting the HJB control also adopts its threshold and its 1/S scaling.
            cfg.threshold_rule = "gamma_spread_price"
            cfg.dollar_scaling = True
        out[name] = cfg
    out["all_fixed"] = BacktestConfig.all_fixed()
    return out
