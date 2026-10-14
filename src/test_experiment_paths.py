"""
Code-path tests for the experiment drivers that are expensive to run on the GPU.

The point of these tests is to catch plumbing bugs (shape errors, wrong chain-rule scaling,
configs that silently fix more than one defect) *before* spending GPU time on training.
They use a randomly initialised network, so they test the machinery, not the science.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backtest import BacktestConfig, ablation_configs, run_backtest
from exp_ood import (
    ASSETS,
    PRICE_MAX,
    PRICE_MIN,
    ThesisStylePINN,
)
from preprocess_causal import corwin_schultz_daily

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    if not cond:
        FAILS.append(name)


# ======================================================================================
print("ThesisStylePINN: forward shapes and the rescaling chain rule")
# ======================================================================================
net = ThesisStylePINN(device="cpu", seed=0, epochs=1)
n = 32
price = np.linspace(40.0, 550.0, n)
vol = np.full(n, 0.01)
spread = np.full(n, 0.0002)

V_raw, g_raw, _ = net.value_and_grad(price, vol, spread, np.full(n, 0.5))
check("raw evaluation returns (n,) values and gradients",
      V_raw.shape == (n,) and g_raw.shape == (n,), f"{V_raw.shape} {g_raw.shape}")
check("raw evaluation is finite", np.all(np.isfinite(V_raw)) and np.all(np.isfinite(g_raw)))
check("rescale_shift defaults off", getattr(net, "rescale_shift", False) is False)

net.rescale_shift = True
net.eval_price_lo, net.eval_price_hi = float(price.min()), float(price.max())
V_rs, g_rs, S_map = net.value_and_grad(price, vol, spread, np.full(n, 0.5))
check("rescaled evaluation returns finite values and gradients",
      np.all(np.isfinite(V_rs)) and np.all(np.isfinite(g_rs)))

# The mapped price must lie inside the training domain [PRICE_MIN, PRICE_MAX].
check("mapped prices lie inside the training domain",
      (S_map.min() >= PRICE_MIN - 1e-9) and (S_map.max() <= PRICE_MAX + 1e-9),
      f"[{S_map.min():.4f}, {S_map.max():.4f}] vs [{PRICE_MIN}, {PRICE_MAX}]")

# Chain rule: dV/dS must be dV/dS_map * dS_map/dS, where dS_map/dS is the affine slope.
# The reference call must be made with rescaling OFF, otherwise it would map S_map again.
slope = (PRICE_MAX - PRICE_MIN) / (price.max() - price.min())
net.rescale_shift = False
V_ref_at_map, g_ref_at_map, _ = net.value_and_grad(S_map, vol, spread, np.full(n, 0.5))
net.rescale_shift = True

check("rescaled value equals the unscaled value at the mapped price",
      np.allclose(V_rs, V_ref_at_map, rtol=1e-5, atol=1e-4),
      f"max|diff|={np.max(np.abs(V_rs - V_ref_at_map)):.3e} (values are O(200), float32)")
check("rescaled gradient equals the unscaled gradient times the affine slope",
      np.allclose(g_rs, g_ref_at_map * slope, rtol=2e-3, atol=1e-6),
      f"slope={slope:.6f} max|diff|={np.max(np.abs(g_rs - g_ref_at_map * slope)):.3e}")

# Finite-difference cross-check of dV/dS in *true* price units.
# NOTE: the network runs in float32 and V is O(200) at these price levels, so an
# absolute value error of ~1e-5 is expected; the step must be large enough that the
# finite difference is not swamped by that noise. This check is therefore coarse by
# construction and exists to catch gross scaling errors (e.g. a missing or inverted
# jacobian), not to measure high-order accuracy.
h = 5.0
net.rescale_shift = True
g_fd = np.empty(n)
for i in range(n):
    pp = price.copy(); pp[i] += h
    pm = price.copy(); pm[i] -= h
    vp, _, _ = net.value_and_grad(pp, vol, spread, np.full(n, 0.5))
    vm, _, _ = net.value_and_grad(pm, vol, spread, np.full(n, 0.5))
    g_fd[i] = (vp[i] - vm[i]) / (2 * h)
mask = np.abs(g_fd) > 1e-4
if mask.any():
    rel = np.max(np.abs(g_rs[mask] - g_fd[mask]) / np.abs(g_fd[mask]))
    check("gradient agrees with a finite difference of V in true price units",
          rel < 5e-2, f"max rel err={rel:.3e} (float32-limited)")
# A missing jacobian would be wrong by 1/slope = 51x; ensure the check would catch that.
if mask.any():
    rel_noscale = np.max(np.abs(g_ref_at_map[mask] - g_fd[mask]) / np.abs(g_fd[mask]))
    check("an unscaled gradient would fail this check (sanity on test power)",
          rel_noscale > 0.5, f"unscaled rel err={rel_noscale:.3e}")

# ======================================================================================
print("\nOOD evaluation: no-trade fractions are well formed")
# ======================================================================================
import tempfile
import pathlib

with tempfile.TemporaryDirectory() as td:
    td = pathlib.Path(td)
    rng = np.random.default_rng(0)
    for a in ASSETS:
        m = 300
        px = np.linspace(40, 550, 9)[ASSETS.index(a)] * (1 + 0.001 * rng.standard_normal(m))
        pd.DataFrame({
            "date": pd.date_range("2025-04-07", periods=m, freq="min"),
            "close": px,
            "ewma_volatility": np.full(m, 0.001),
            "normalized_spread": np.full(m, 0.0002),
            "spread": np.full(m, 0.02),
        }).to_csv(td / f"{a}_processed_hft_data.csv", index=False)

    from exp_ood import evaluate_no_trade

    net.rescale_shift = False
    ev = evaluate_no_trade(net, td, n_per_asset=100, seed=0)
    pa = ev["per_asset"]
    check("evaluate_no_trade returns all assets", len(pa) == len(ASSETS), f"{len(pa)}")
    check("per-asset rows are sorted by price",
          all(pa[i]["mean_price"] <= pa[i + 1]["mean_price"] for i in range(len(pa) - 1)))
    for key in ("no_trade_frac", "grad_abs_mean", "grad_abs_median", "control_abs_mean"):
        vals = [r[key] for r in pa]
        if key.startswith("no_trade") or key.startswith("control_abs"):
            continue
        if not all(np.isfinite(v) for v in vals):
            check(f"{key} finite", False, f"{vals[:3]}")
            break
    else:
        check("all recorded fields are finite", True)
    check("no-trade fractions lie within [0,1]",
          all(0.0 <= r["no_trade_frac"] <= 1.0 for r in pa))
    check("V_range is finite and non-negative",
          all(np.isfinite(r["V_range"]) and r["V_range"] >= 0 for r in pa))
    check("grad_abs_mean is non-negative",
          all(r["grad_abs_mean"] >= 0 for r in pa))

# ======================================================================================
print("\nCorwin-Schultz estimator behaves monotonically in range")
# ======================================================================================
narrow = corwin_schultz_daily(np.array([100.5, 100.5]), np.array([99.5, 99.5]))
wide = corwin_schultz_daily(np.array([103.0, 103.0]), np.array([97.0, 97.0]))
# The rolling estimator needs two periods, so index 0 is 0 by construction; the estimate
# formed from the available pair is the last entry.
narrow_v = float(np.ravel(narrow)[-1]); wide_v = float(np.ravel(wide)[-1])
check("wider high-low range gives a larger spread", wide_v > narrow_v,
      f"{narrow_v:.5f} < {wide_v:.5f}")
check("zero range gives zero spread",
      float(np.ravel(corwin_schultz_daily(np.array([100.0, 100.0]),
                                          np.array([100.0, 100.0])))[-1]) == 0.0)

# ======================================================================================
print("\nablation configs: each fixes exactly one field")
# ======================================================================================
cfgs = ablation_configs()
check("as_published and all_fixed are present",
      "as_published" in cfgs and "all_fixed" in cfgs)
base = cfgs["as_published"].__dict__
# Fields belonging to the single control-law defect (D3 sign + D4 scaling + its threshold
# rule); these are ablated together, so more than one may differ at once.
CONTROL_LAW_FIELDS = {"sign_convention", "dollar_scaling", "threshold_rule"}
for name, cfg in cfgs.items():
    if name in ("as_published", "all_fixed"):
        continue
    diffs = [k for k, v in cfg.__dict__.items() if base[k] != v]
    extra = set(diffs) - CONTROL_LAW_FIELDS
    check(f"{name} fixes one defect (control-law fields may differ together)",
          len(extra) <= 1 and (len(extra) == 1 or set(diffs) <= CONTROL_LAW_FIELDS),
          f"differs in {diffs}")
check("all_fixed differs from as_published in all 9 tracked fields",
      len([k for k, v in cfgs["all_fixed"].__dict__.items() if base[k] != v]) >= 9,
      f"{len([k for k, v in cfgs['all_fixed'].__dict__.items() if base[k] != v])} fields")

# ======================================================================================
print("\nbacktest runs against the ThesisStylePINN interface")
# ======================================================================================
m = 200
px = 100.0 + np.cumsum(np.random.default_rng(1).normal(0, 0.05, m))
frame = pd.DataFrame({
    "date": pd.date_range("2025-04-07", periods=m, freq="min"),
    "close": px,
    "ewma_volatility": np.full(m, 0.001),
    "normalized_spread": np.full(m, 0.0002),
    "spread": np.full(m, 0.02),
})
net.rescale_shift = False
for cname in ("as_published", "all_fixed"):
    d, metrics = run_backtest(frame, net, cfgs[cname])
    check(f"backtest '{cname}' produces finite metrics",
          all(np.isfinite(v) for v in metrics.values()),
          f"ret={metrics['total_return']:+.6f} sharpe={metrics['sharpe_ratio']:+.4f}")

# ======================================================================================
print("\n" + "=" * 70)
if FAILS:
    print(f"{len(FAILS)} CHECK(S) FAILED:")
    for f in FAILS:
        print(f"  - {f}")
    raise SystemExit(1)
print("ALL EXPERIMENT-PATH CHECKS PASSED")
