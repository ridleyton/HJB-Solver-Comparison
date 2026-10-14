"""
Experiment 3: out-of-domain evaluation as a source of spurious cross-sectional results.

The defect being studied
------------------------
The original system trained its network on a normalised price domain of [20, 30] with a
terminal target of 25.0 (`comparison_framework.py:21-26,521`, `utils/PDEs.py:20-21,213`,
`side_info.json`), then evaluated that network on the **raw** close price of real assets
(`hft_backtest.py:55,85`, `comparison_framework.py:897,928`, `main.py:209`):

    OXY $40 | GOOGL $162 | NVDA $110 | AMZN $189 | AAPL $209 |
    TSLA $285 | MSFT $391 | QQQ $472 | SPY $550

i.e. up to ~18x above the top of the training domain, with the volatility and spread
channels also outside their training ranges (verified: close 482-571, ewma vol
7.5e-5 to 1.8e-2, spread 0.06 to 35.0, against training ranges [20,30], [0.01,0.05],
[0.001,0.01]).

The thesis reported that the fraction of "no-trade" minutes was almost perfectly monotone
in the asset's price level (OXY 0.0% ... SPY 80.9%) and interpreted this as the model
preferring to stay passive in liquid index instruments, going so far as to report a
"perfect rank correlation (Spearman rho = 1.0)" between inactivity and outcome
(`main.tex:2279,2327,2371`).

This module reproduces the *mechanism*. It trains the same style of network on the same
training domain and the same HJB, evaluates it on the real price levels above, and shows
that the implied no-trade fraction is monotone in price purely as a consequence of
extrapolation distance -- then shows that scaling the input into the training domain
removes the pattern.

HJB used (matching the thesis, `main.tex:363-369`)
--------------------------------------------------
    dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + max_v [ v dV/dS - kappa v^2 S - gamma |v| S * spread ] = 0
    V(S,T) = -1/2 (S - 25)^2

with the control law from the thesis's first-order condition (`main.tex:371-379`):

    v* = (V_S - gamma S spread) / (2 kappa S)      if V_S >  gamma S spread
       = 0                                          if |V_S| <= gamma S spread
       = (V_S + gamma S spread) / (2 kappa S)      if V_S < -gamma S spread

Note this is the *correct* sign (v* > 0, i.e. buying, when V_S > 0), and is deliberately
different from the released code, which computed `-V_S / (2 kappa)` in all three places.
The sign defect is documented separately; this experiment isolates the scaling defect.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# --------------------------------------------------------------------------------------
# Problem constants, taken from the original configuration (side_info.json)
# --------------------------------------------------------------------------------------
KAPPA = 0.0005          # market_impact_factor
GAMMA = 0.0001          # transaction_cost_factor
PRICE_MIN, PRICE_MAX = 20.0, 30.0
VOL_MIN, VOL_MAX = 0.01, 0.05
SPREAD_MIN, SPREAD_MAX = 0.001, 0.01
T_HORIZON = 1.0
TARGET_PRICE = 25.0
SIGMA_SCALE = 0.01      # volatility_factor


def _mlp(in_dim: int, hidden: int, layers: int) -> nn.Sequential:
    mods: list[nn.Module] = [nn.Linear(in_dim, hidden), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(hidden, hidden), nn.Tanh()]
    mods += [nn.Linear(hidden, 1)]
    for m in mods:
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
    return nn.Sequential(*mods)


class ThesisStylePINN:
    """PINN for the thesis's price-tracking HJB, with a selectable input scaling.

    ``scale_inputs=True``  maps S into [PRICE_MIN, PRICE_MAX] (i.e. the domain the network
                           was trained on) before evaluation -- the correct treatment.
    ``scale_inputs=False`` feeds the raw price, reproducing the original defect.
    """

    def __init__(self, hidden: int = 64, layers: int = 4, device: str = "cuda",
                 seed: int = 0, epochs: int = 20000, lr: float = 1e-3,
                 n_collocation: int = 2048, scale_inputs: bool = False,
                 verbose: bool = False, log_every: int = 0):
        self.device = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
        self.epochs, self.lr = epochs, lr
        self.n_collocation = n_collocation
        self.scale_inputs = scale_inputs
        self.verbose, self.log_every = verbose, log_every
        self.seed = seed
        self.solve_time = None
        self.history: list[tuple[int, float]] = []
        self.residual_scale = 1.0

        torch.manual_seed(seed); np.random.seed(seed)
        self.net = _mlp(4, hidden, layers).to(self.device)
        # V is O(0.5 * (S-25)^2) with S up to 30, so ~12.5; use 625 as a round scale.
        self.V_scale = 625.0

    # ----------------------------------------------------------------------------------
    def _enc(self, S, vol, spread, t):
        """Normalise inputs. Unscaled evaluation feeds raw S -- the original defect."""
        Sn = (S - PRICE_MIN) / (PRICE_MAX - PRICE_MIN)
        vn = (vol - VOL_MIN) / (VOL_MAX - VOL_MIN)
        pn = (spread - SPREAD_MIN) / (SPREAD_MAX - SPREAD_MIN)
        return torch.cat([Sn, vn, pn, t], dim=1)

    def _sample(self, n, gen):
        S = PRICE_MIN + (PRICE_MAX - PRICE_MIN) * torch.rand(n, 1, device=self.device, generator=gen)
        vol = VOL_MIN + (VOL_MAX - VOL_MIN) * torch.rand(n, 1, device=self.device, generator=gen)
        sp = SPREAD_MIN + (SPREAD_MAX - SPREAD_MIN) * torch.rand(n, 1, device=self.device, generator=gen)
        t = torch.rand(n, 1, device=self.device, generator=gen) * T_HORIZON
        return S, vol, sp, t

    def _residual(self, S, vol, spread, t):
        """HJB residual only: dV/dt + 1/2 sigma^2 S^2 d2V/dS2 + max_v H."""
        inp = self._enc(S, vol, spread, t)
        V = self.net(inp) * self.V_scale
        gS = torch.autograd.grad(V.sum(), S, create_graph=True)[0]
        gSS = torch.autograd.grad(gS.sum(), S, create_graph=True, retain_graph=True)[0]
        gT = torch.autograd.grad(V.sum(), t, create_graph=True, retain_graph=True)[0]

        sigma = vol * SIGMA_SCALE
        # max_v [ v V_S - kappa v^2 S - gamma |v| S spread ]
        thr = GAMMA * S * spread
        v = torch.where(gS > thr, (gS - thr) / (2 * KAPPA * S),
                        torch.where(gS < -thr, (gS + thr) / (2 * KAPPA * S),
                                    torch.zeros_like(gS)))
        H = v * gS - KAPPA * v**2 * S - GAMMA * torch.abs(v) * S * spread
        return gT + 0.5 * sigma**2 * S**2 * gSS + H

    def _terminal_loss(self, n: int, gen) -> torch.Tensor:
        """Soft penalty enforcing V(S, T) = -1/2 (S - 25)^2.

        This is essential: the HJB residual alone does not determine the solution, because
        the terminal condition is what pins down which solution of the PDE is wanted. An
        earlier version of this module omitted it, and the network then converged to an
        arbitrary solution of the PDE, which made the out-of-domain comparison meaningless.
        """
        S = PRICE_MIN + (PRICE_MAX - PRICE_MIN) * torch.rand(
            n, 1, device=self.device, generator=gen)
        vol = VOL_MIN + (VOL_MAX - VOL_MIN) * torch.rand(n, 1, device=self.device, generator=gen)
        sp = SPREAD_MIN + (SPREAD_MAX - SPREAD_MIN) * torch.rand(
            n, 1, device=self.device, generator=gen)
        t = torch.full((n, 1), T_HORIZON, device=self.device)
        V = self.net(self._enc(S, vol, sp, t)) * self.V_scale
        V_true = -0.5 * (S - TARGET_PRICE) ** 2
        return ((V - V_true) ** 2).mean() / (self.V_scale**2)

    def solve(self, terminal_weight: float = 10.0):
        gen = torch.Generator(device=self.device); gen.manual_seed(self.seed)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=self.epochs, eta_min=self.lr * 0.01)
        # Normalise the PDE residual by its initial magnitude so the loss is O(1).
        S0, v0, sp0, t0 = self._sample(512, gen)
        S0 = S0.clone().requires_grad_(True); t0 = t0.clone().requires_grad_(True)
        with torch.enable_grad():
            r0 = self._residual(S0, v0, sp0, t0).detach()
            self.residual_scale = float(r0.pow(2).mean().sqrt().clamp_min(1e-12))
        del r0, S0, v0, sp0, t0

        t_start = time.perf_counter()
        for ep in range(1, self.epochs + 1):
            opt.zero_grad(set_to_none=True)
            S, vol, sp, t = self._sample(self.n_collocation, gen)
            S = S.clone().requires_grad_(True); t = t.clone().requires_grad_(True)
            res = self._residual(S, vol, sp, t)
            loss = (res / self.residual_scale).pow(2).mean()
            loss = loss + terminal_weight * self._terminal_loss(self.n_collocation // 4, gen)
            loss.backward(); opt.step(); sched.step()
            if self.log_every and ep % self.log_every == 0:
                self.history.append((ep, float(loss.detach())))
                if self.verbose:
                    print(f"    ep {ep:6d} loss={float(loss.detach()):.3e}")
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        self.solve_time = time.perf_counter() - t_start
        return self.solve_time

    # ----------------------------------------------------------------------------------
    def _forward(self, S, vol, spread, t):
        S = np.atleast_1d(np.asarray(S, dtype=np.float32)).reshape(-1)
        vol = np.atleast_1d(np.asarray(vol, dtype=np.float32)).reshape(-1)
        spread = np.atleast_1d(np.asarray(spread, dtype=np.float32)).reshape(-1)
        t = np.atleast_1d(np.asarray(t, dtype=np.float32)).reshape(-1)
        dev = self.device

        # ---- the scaling decision ------------------------------------------------
        if getattr(self, "rescale_shift", False):
            # Correct treatment: affine-map the evaluation price back into the training
            # domain, then undo the map on the returned gradient via the chain rule.
            S_lo, S_hi = self.eval_price_lo, self.eval_price_hi
            S_mapped = PRICE_MIN + (S - S_lo) / (S_hi - S_lo) * (PRICE_MAX - PRICE_MIN)
            S_map_t = torch.tensor(S_mapped, dtype=torch.float32, device=dev).reshape(-1, 1)
            S_map_t.requires_grad_(True)
            jac = (PRICE_MAX - PRICE_MIN) / (S_hi - S_lo)     # dS_mapped/dS
        else:
            S_map_t = torch.tensor(S, dtype=torch.float32, device=dev).reshape(-1, 1)
            S_map_t.requires_grad_(True)
            jac = 1.0

        vol_t = torch.tensor(vol, dtype=torch.float32, device=dev).reshape(-1, 1)
        sp_t = torch.tensor(spread, dtype=torch.float32, device=dev).reshape(-1, 1)
        t_t = torch.tensor(t, dtype=torch.float32, device=dev).reshape(-1, 1)

        inp = self._enc(S_map_t, vol_t, sp_t, t_t)
        V = self.net(inp) * self.V_scale
        g = torch.autograd.grad(V.sum(), S_map_t)[0] * jac
        return (V.detach().cpu().numpy().ravel(),
                g.detach().cpu().numpy().ravel(),
                S_mapped if getattr(self, "rescale_shift", False) else S)

    def value_and_grad(self, price, vol, spread, t):
        return self._forward(price, vol, spread, t)


# --------------------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------------------
ASSETS = ["OXY", "NVDA", "GOOGL", "AMZN", "AAPL", "TSLA", "MSFT", "QQQ", "SPY"]


def initial_prices(data_dir: Path) -> dict[str, float]:
    out = {}
    for a in ASSETS:
        f = data_dir / f"{a}_processed_hft_data.csv"
        if f.exists():
            import csv
            with f.open() as fh:
                r = csv.DictReader(fh)
                out[a] = float(next(r)["close"])
    return out


def evaluate_no_trade(net: ThesisStylePINN, data_dir: Path, n_per_asset: int = 400,
                      seed: int = 0) -> dict:
    """Measure how the model's local sensitivity depends on the asset's price level.

    The primary quantity is the magnitude of the value-function's price sensitivity,
    ``|dV/dS|``, evaluated at each instrument's realised prices. A threshold-free measure is
    used deliberately: the no-trade fraction depends on an arbitrary threshold rule, and the
    audited baseline in fact used two inconsistent ones. If out-of-domain evaluation is
    distorting the model, then ``|dV/dS|`` should fall monotonically with price level,
    because the network saturates further outside its training domain. Any threshold applied
    to such a gradient inherits that ordering.

    The same measurement is repeated with the price affinely mapped into the training domain,
    which should remove the dependence.
    """
    import pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    for a in ASSETS:
        f = data_dir / f"{a}_processed_hft_data.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f)
        if len(df) > n_per_asset:
            df = df.iloc[rng.choice(len(df), n_per_asset, replace=False)]
        price = df["close"].to_numpy()
        vol = df["ewma_volatility"].to_numpy()
        spread = df["normalized_spread"].to_numpy()
        t = np.full_like(price, 0.5)
        V, g, _ = net.value_and_grad(price, vol, spread, t)

        # Implied control from the HJB first-order condition, for reference.
        thr = GAMMA * price * spread
        denom = 2.0 * KAPPA * price
        v = np.where(g > thr, (g - thr) / denom,
                     np.where(g < -thr, (g + thr) / denom, 0.0))

        rows.append({
            "asset": a,
            "mean_price": float(price.mean()),
            "V_min": float(V.min()), "V_max": float(V.max()),
            "V_range": float(V.max() - V.min()),
            "grad_abs_mean": float(np.abs(g).mean()),
            "grad_abs_median": float(np.median(np.abs(g))),
            "control_abs_mean": float(np.abs(v).mean()),
            "no_trade_frac": float(np.mean(np.abs(v) < 1e-12)),
        })
    rows.sort(key=lambda d: d["mean_price"])
    return {"per_asset": rows}


def spearman(x, y) -> float:
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx**2).sum() * (ry**2).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--out", type=str, default="results/exp_ood.json")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out); out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_domain_price": [PRICE_MIN, PRICE_MAX],
            "train_domain_vol": [VOL_MIN, VOL_MAX],
            "train_domain_spread": [SPREAD_MIN, SPREAD_MAX],
            "target_price": TARGET_PRICE, "kappa": KAPPA, "gamma": GAMMA,
            "epochs": args.epochs, "seeds": seeds,
            "eval_assets": ASSETS,
            "note": ("Conditions: (1) unscaled evaluation on raw prices (reproduces the "
                     "original defect); (2) evaluation with price affine-mapped into the "
                     "training domain (the fix)."),
        },
        "conditions": {},
    }

    for cond, rescale in (("unscaled_raw_price", False), ("price_rescaled_to_train_domain", True)):
        print(f"\n=== condition: {cond} ===", flush=True)
        per_seed = []
        for sd in seeds:
            net = ThesisStylePINN(device=args.device, seed=sd, epochs=args.epochs,
                                  scale_inputs=True, log_every=args.epochs,
                                  verbose=False)
            st = net.solve()
            print(f"  seed {sd}: trained in {st:.1f}s", flush=True)
            # Map prices into the training domain when rescaling is requested.
            all_prices = initial_prices(data_dir)
            net.rescale_shift = rescale
            if rescale:
                net.eval_price_lo = min(all_prices.values()) if all_prices else PRICE_MIN
                net.eval_price_hi = max(all_prices.values()) if all_prices else PRICE_MAX
            ev = evaluate_no_trade(net, data_dir, seed=100 + sd)
            ev["seed"] = sd
            ev["train_time_s"] = st
            pa = ev["per_asset"]
            pr = [r["mean_price"] for r in pa]
            ev["spearman_price_vs_abs_grad"] = spearman(pr, [r["grad_abs_mean"] for r in pa])
            ev["spearman_price_vs_V_range"] = spearman(pr, [r["V_range"] for r in pa])
            ev["spearman_price_vs_no_trade"] = spearman(pr, [r["no_trade_frac"] for r in pa])
            ev["grad_abs_mean"] = float(np.mean([r["grad_abs_mean"] for r in pa]))
            per_seed.append(ev)
            print(f"    spearman(price, |dV/dS|) = {ev['spearman_price_vs_abs_grad']:+.3f}"
                  f"   (price, no-trade) = {ev['spearman_price_vs_no_trade']:+.3f}"
                  f"   mean |dV/dS| = {ev['grad_abs_mean']:.4e}", flush=True)
        results["conditions"][cond] = per_seed
        out_path.write_text(json.dumps(results, indent=2))

    # Summary table (mean over seeds) for the paper.
    print("\nPrice sensitivity by asset (mean over seeds)")
    print(f"{'asset':6s} {'price':>8s} {'|dV/dS| unscaled':>18s} {'|dV/dS| rescaled':>18s}")
    un = results["conditions"]["unscaled_raw_price"]
    rs = results["conditions"]["price_rescaled_to_train_domain"]
    for i, r in enumerate(un[0]["per_asset"]):
        a = r["asset"]
        u = np.mean([s["per_asset"][i]["grad_abs_mean"] for s in un])
        rr = np.mean([s["per_asset"][i]["grad_abs_mean"] for s in rs])
        print(f"{a:6s} {r['mean_price']:8.1f} {u:18.4e} {rr:18.4e}")

    out_path.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
