#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Optimize notification-focused strategy parameters.

Goal:
- Keep signal quality robust across years.
- Tune long/short confirmation thresholds separately.
- Tune signal cooldown to reduce noisy duplicate alerts.
"""

import contextlib
import io
import itertools
import multiprocessing as mp
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, STRATEGY_PARAMS, TRADING_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


@dataclass
class YearResult:
    total_return: float
    total_trades: int
    win_rate: float
    max_drawdown: float
    profit_factor: float
    long_signals: int
    short_signals: int


g_cache = None
g_years = None
g_base = None
g_train_years = None
g_val_years = None


def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    data_dir = os.path.join(PROJECT_ROOT, "data")
    files = [
        "ETHUSDT_15m_2020_ccxt.csv",
        "ETHUSDT_15m_2021_ccxt.csv",
        "ETHUSDT_15m_2022_ccxt.csv",
        "ETHUSDT_15m_2023_now.csv",
    ]
    dfs = []
    for f in files:
        p = os.path.join(data_dir, f)
        if os.path.exists(p):
            dfs.append(pd.read_csv(p, index_col="ts", parse_dates=True))

    if not dfs:
        raise FileNotFoundError("No data files found under data/")

    df_15m = pd.concat(dfs).sort_index()
    df_15m = df_15m[~df_15m.index.duplicated(keep="first")]

    df_4h = df_15m.resample("4H").agg(
        {"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"}
    ).dropna()
    return df_15m, df_4h


def precompute_cache(df_15m: pd.DataFrame, df_4h: pd.DataFrame, years: List[int]) -> Dict:
    cache = {}
    smc = SMCIndicators(SMC_PARAMS)
    base = {**TRADING_PARAMS, **STRATEGY_PARAMS}
    strategy = SMCStrategyFinal(base)

    for y in years:
        y15 = df_15m[df_15m.index.year == y].copy()
        y4 = df_4h[df_4h.index.year == y].copy()
        if len(y15) < 100:
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            y4 = smc.calculate_all(y4)
            y4 = strategy.identify_key_structure_points(
                y4, min_move_pct=base.get("min_structure_move_pct", 0.02)
            )

        cache[y] = {
            "df_15m": y15,
            "df_4h_base": y4,
        }
        print(f"  {y}: 15m={len(y15)}, 4h={len(y4)}")
    return cache


def simulate_trades(df_1h: pd.DataFrame, params: Dict) -> YearResult:
    trades = []
    enable_be = params.get("enable_breakeven", True)
    be_trigger = params.get("breakeven_trigger_r", 1.5)
    be_pct = params.get("breakeven_profit_pct", 0.005)

    signals = df_1h[df_1h["signal"] != 0]
    for idx, row in signals.iterrows():
        direction = "long" if row["signal"] == 1 else "short"
        entry = float(row["entry_price"])
        sl = float(row["stop_loss"])
        tp = float(row["take_profit"])
        orig_sl = sl
        be_moved = False

        for _, candle in df_1h[df_1h.index > idx].iterrows():
            h = float(candle["h"])
            l = float(candle["l"])

            if enable_be and not be_moved:
                risk = abs(entry - orig_sl)
                if risk > 0:
                    target = entry + (be_trigger * risk) if direction == "long" else entry - (be_trigger * risk)
                    hit = (direction == "long" and h >= target) or (direction == "short" and l <= target)
                    if hit:
                        sl = entry * (1 + be_pct) if direction == "long" else entry * (1 - be_pct)
                        be_moved = True

            if direction == "long":
                if l <= sl:
                    trades.append((sl - entry) / entry * 100.0)
                    break
                if h >= tp:
                    trades.append((tp - entry) / entry * 100.0)
                    break
            else:
                if h >= sl:
                    trades.append((entry - sl) / entry * 100.0)
                    break
                if l <= tp:
                    trades.append((entry - tp) / entry * 100.0)
                    break

    if not trades:
        return YearResult(
            total_return=0.0,
            total_trades=0,
            win_rate=0.0,
            max_drawdown=0.0,
            profit_factor=0.0,
            long_signals=int((df_1h["signal"] == 1).sum()),
            short_signals=int((df_1h["signal"] == -1).sum()),
        )

    equity = 1.0
    peak = 1.0
    mdd = 0.0
    wins = 0
    total_profit = 0.0
    total_loss = 0.0

    for pnl in trades:
        equity *= 1.0 + pnl / 100.0
        if pnl > 0:
            wins += 1
            total_profit += pnl
        else:
            total_loss += abs(pnl)
        peak = max(peak, equity)
        mdd = max(mdd, (peak - equity) / peak * 100.0)

    return YearResult(
        total_return=(equity - 1.0) * 100.0,
        total_trades=len(trades),
        win_rate=(wins / len(trades) * 100.0),
        max_drawdown=mdd,
        profit_factor=(total_profit / total_loss if total_loss > 0 else 99.99),
        long_signals=int((df_1h["signal"] == 1).sum()),
        short_signals=int((df_1h["signal"] == -1).sum()),
    )


def worker_init(cache, years, base, train_years, val_years):
    global g_cache, g_years, g_base, g_train_years, g_val_years
    g_cache = cache
    g_years = years
    g_base = base
    g_train_years = train_years
    g_val_years = val_years


def evaluate_config(config: Dict):
    params = {**g_base, **config}
    strategy = SMCStrategyFinal(params)
    results: Dict[int, YearResult] = {}

    for y in g_years:
        if y not in g_cache:
            continue
        d15 = g_cache[y]["df_15m"]
        d4 = g_cache[y]["df_4h_base"].copy()
        with contextlib.redirect_stdout(io.StringIO()):
            d4 = strategy.identify_high_quality_ob_4h(d4)
            d1 = strategy.generate_signals_mtf(d15, d4, signal_lookback=None)
        results[y] = simulate_trades(d1, params)

    train_total = sum(results[y].total_return for y in g_train_years if y in results)
    val_total = sum(results[y].total_return for y in g_val_years if y in results)
    train_min = min((results[y].total_return for y in g_train_years if y in results), default=0.0)
    val_min = min((results[y].total_return for y in g_val_years if y in results), default=0.0)
    neg_years = sum(1 for y in g_years if y in results and results[y].total_return < 0)
    avg_pf = np.mean([results[y].profit_factor for y in results if results[y].profit_factor > 0]) if results else 0.0
    total_shorts = sum(results[y].short_signals for y in results)

    # Bias toward robustness and out-of-sample stability for notification quality
    score = (
        train_total * 0.15
        + train_min * 1.8
        + val_total * 0.45
        + val_min * 3.0
        + avg_pf * 12.0
        + min(total_shorts, 120) * 0.08
        - neg_years * 20.0
    )
    return score, config, results


def print_result(rank: int, score: float, cfg: Dict, results: Dict[int, YearResult]) -> None:
    print(
        f"#{rank} score={score:.2f} "
        f"long_body={cfg['entry_candle_body_ratio']:.2f} "
        f"long_vol={cfg['entry_volume_threshold']:.2f} "
        f"short_body={cfg['entry_candle_body_ratio_short']:.2f} "
        f"short_vol={cfg['entry_volume_threshold_short']:.2f} "
        f"cooldown={cfg['signal_cooldown_bars']}"
    )
    for y in sorted(results.keys()):
        r = results[y]
        print(
            f"  {y}: ret={r.total_return:7.2f}% trades={r.total_trades:3d} "
            f"wr={r.win_rate:5.1f}% pf={r.profit_factor:4.2f} "
            f"L/S={r.long_signals}/{r.short_signals}"
        )


def main():
    print("=" * 80)
    print("Notification Strategy Optimizer")
    print("=" * 80)

    df_15m, df_4h = load_data()
    years = sorted(set(df_15m.index.year))
    years = [y for y in years if len(df_15m[df_15m.index.year == y]) > 100]

    train_years = [y for y in years if y >= 2023]
    val_years = [y for y in years if y <= 2022]
    print(f"Years: {years}")
    print(f"Train years: {train_years}")
    print(f"Val years: {val_years}")

    print("\nPrecomputing base SMC cache...")
    cache = precompute_cache(df_15m, df_4h, years)

    base = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    grid = {
        "entry_candle_body_ratio": [0.45, 0.50, 0.55],
        "entry_volume_threshold": [0.45, 0.50, 0.55],
        "entry_candle_body_ratio_short": [0.55, 0.60, 0.65],
        "entry_volume_threshold_short": [0.55, 0.60, 0.65],
        "signal_cooldown_bars": [0, 1, 2],
    }

    keys = list(grid.keys())
    combos = list(itertools.product(*(grid[k] for k in keys)))
    configs = []
    for combo in combos:
        cfg = {}
        for k, v in zip(keys, combo):
            cfg[k] = v
        configs.append(cfg)

    print(f"\nTotal configs: {len(configs)}")

    workers = max(1, mp.cpu_count() - 1)
    print(f"Workers: {workers}")

    start = time.time()
    best = []

    with mp.Pool(
        processes=workers,
        initializer=worker_init,
        initargs=(cache, years, base, train_years, val_years),
    ) as pool:
        done = 0
        for score, cfg, results in pool.imap_unordered(evaluate_config, configs, chunksize=8):
            done += 1
            best.append((score, cfg, results))
            best.sort(key=lambda x: x[0], reverse=True)
            if len(best) > 20:
                best = best[:20]

            if done % 20 == 0:
                elapsed = time.time() - start
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(configs) - done) / rate if rate > 0 else 0.0
                print(
                    f"[{done}/{len(configs)}] "
                    f"elapsed={elapsed/60:.1f}m rate={rate:.2f}/s eta={eta/60:.1f}m "
                    f"best={best[0][0]:.2f}"
                )
                sys.stdout.flush()

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} minutes")
    best.sort(key=lambda x: x[0], reverse=True)

    print("\nTop 5 configs:")
    for i, (score, cfg, results) in enumerate(best[:5], 1):
        print_result(i, score, cfg, results)

    top_score, top_cfg, top_results = best[0]
    output_dir = os.path.join(PROJECT_ROOT, "output")
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "notify_best_params.json")
    with open(json_path, "w", encoding="utf-8") as f:
        payload = {
            "score": top_score,
            "params": top_cfg,
            "years": {
                str(y): {
                    "total_return": top_results[y].total_return,
                    "total_trades": top_results[y].total_trades,
                    "win_rate": top_results[y].win_rate,
                    "max_drawdown": top_results[y].max_drawdown,
                    "profit_factor": top_results[y].profit_factor,
                    "long_signals": top_results[y].long_signals,
                    "short_signals": top_results[y].short_signals,
                }
                for y in sorted(top_results.keys())
            },
        }
        import json
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"\nSaved best params: {json_path}")


if __name__ == "__main__":
    mp.freeze_support()
    main()
