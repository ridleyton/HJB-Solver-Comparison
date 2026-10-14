"""
Driver: defect ablation of the original backtest pipeline.

Trains the thesis-style network once per seed (price-tracking HJB on a [20, 30] training
domain, matching the original configuration), then runs every ablation configuration from
`backtest.ablation_configs()` on the out-of-sample test window of each asset.

Outputs a JSON of metrics per (configuration, asset, seed) plus a printed summary, so the
contribution of each individual defect to the reported performance is visible.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import BacktestConfig, ablation_configs, run_backtest
from exp_ood import ThesisStylePINN

ASSETS = ["AAPL", "AMZN", "GOOGL", "MSFT", "NVDA", "OXY", "QQQ", "SPY", "TSLA"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="data_causal")
    ap.add_argument("--epochs", type=int, default=20000)
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--split", type=str, default="test",
                    help="which split to backtest on (train/validation/test)")
    ap.add_argument("--out", type=str, default="results/exp_ablation.json")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    configs = ablation_configs()

    results = {
        "meta": {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": args.epochs, "seeds": seeds, "split": args.split,
            "data_dir": str(data_dir),
            "note": ("Each configuration starts from 'as_published' and fixes exactly one "
                     "defect, except 'all_fixed' which fixes all of them."),
            "configs": {k: v.__dict__ for k, v in configs.items()},
        },
        "runs": [],
    }

    # Cache the per-asset data
    data: dict[str, pd.DataFrame] = {}
    for a in ASSETS:
        f = data_dir / f"{a}_causal.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"])
        df = df[df["split"] == args.split].reset_index(drop=True)
        if len(df) > 10:
            data[a] = df
    print(f"assets with {args.split} data: {list(data)}", flush=True)

    for sd in seeds:
        print(f"\n### seed {sd}: training thesis-style PINN", flush=True)
        net = ThesisStylePINN(device=args.device, seed=sd, epochs=args.epochs,
                              log_every=args.epochs, verbose=False)
        net.rescale_shift = False          # scaling is the backtest's job, via the config
        t_train = net.solve()
        print(f"  trained in {t_train:.1f}s", flush=True)

        for cfg_name, cfg in configs.items():
            per_asset = {}
            for a, df in data.items():
                try:
                    _, m = run_backtest(df, net, cfg)
                except Exception as e:
                    print(f"    {cfg_name}/{a} FAILED: {type(e).__name__}: {e}", flush=True)
                    continue
                per_asset[a] = m
            if not per_asset:
                continue
            agg = {
                k: float(np.mean([m[k] for m in per_asset.values()]))
                for k in next(iter(per_asset.values()))
                if isinstance(next(iter(per_asset.values()))[k], (int, float))
            }
            results["runs"].append({
                "seed": sd, "config": cfg_name, "split": args.split,
                "agg": agg, "per_asset": per_asset,
            })
            print(f"  {cfg_name:28s} ret={agg['total_return']:+.4f} "
                  f"sharpe={agg['sharpe_ratio']:+.4f} vol={agg['annualized_volatility']:.4f} "
                  f"|pos|={agg['mean_abs_position']:.4f} turnover={agg['turnover']:.2f}",
                  flush=True)
            out_path.write_text(json.dumps(results, indent=2))

    out_path.write_text(json.dumps(results, indent=2))

    # ---- summary table ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"Defect ablation summary (mean over seeds and assets, split={args.split})")
    print("=" * 100)
    print(f"{'configuration':30s} {'total_ret':>11s} {'sharpe':>9s} {'ann_vol':>9s} "
          f"{'maxDD':>9s} {'|pos|':>8s} {'turnover':>10s}")
    order = ["as_published"] + [k for k in configs if k.startswith("fix_")] + ["all_fixed"]
    for name in order:
        rows = [r["agg"] for r in results["runs"] if r["config"] == name]
        if not rows:
            continue
        g = lambda k: float(np.mean([r[k] for r in rows]))
        print(f"{name:30s} {g('total_return'):>+11.4f} {g('sharpe_ratio'):>+9.4f} "
              f"{g('annualized_volatility'):>9.4f} {g('max_drawdown'):>+9.4f} "
              f"{g('mean_abs_position'):>8.4f} {g('turnover'):>10.2f}")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
