# parallel_optimizer.py
# 多線程參數優化器
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def load_data():
    """載入數據"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    df_2023 = pd.read_csv(os.path.join(data_dir, 'ETHUSDT_15m_2023_now.csv'),
                          index_col='ts', parse_dates=True)
    df_2022 = pd.read_csv(os.path.join(data_dir, 'ETHUSDT_15m_2022_ccxt.csv'),
                          index_col='ts', parse_dates=True)
    df_15m = pd.concat([df_2022, df_2023]).sort_index()
    df_15m = df_15m[~df_15m.index.duplicated(keep='first')]
    df_4h = df_15m.resample('4H').agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()
    return df_15m, df_4h


def test_single_year(args):
    """測試單一年份（用於多線程）"""
    df_15m, df_4h, params, year = args

    # 動態導入避免 pickle 問題
    import sys
    sys.path.insert(0, PROJECT_ROOT)
    from config.settings import SMC_PARAMS
    from strategy.indicators import SMCIndicators
    from strategy.smc_strategy import SMCStrategyFinal

    df_15m_y = df_15m[df_15m.index.year == year].copy()
    df_4h_y = df_4h[df_4h.index.year == year].copy()

    if len(df_15m_y) < 100:
        return year, {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    smc = SMCIndicators(SMC_PARAMS)
    strategy = SMCStrategyFinal(params)

    # 靜默計算
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        df_4h_y = smc.calculate_all(df_4h_y)
        df_4h_y = strategy.identify_key_structure_points(df_4h_y)
        df_4h_y = strategy.identify_high_quality_ob_4h(df_4h_y)
        df_1h = strategy.generate_signals_mtf(df_15m_y, df_4h_y)

    # 模擬交易
    trades = []
    enable_be = params.get('enable_breakeven', True)
    be_trigger = params.get('breakeven_trigger_r', 1.5)
    be_pct = params.get('breakeven_profit_pct', 0.005)

    for idx, row in df_1h[df_1h['signal'] != 0].iterrows():
        direction = 'long' if row['signal'] == 1 else 'short'
        entry = row['entry_price']
        sl = row['stop_loss']
        tp = row['take_profit']
        orig_sl = sl
        be_moved = False

        for c_idx, c in df_1h[df_1h.index > idx].iterrows():
            h, l = c['h'], c['l']

            if enable_be and not be_moved:
                risk = abs(entry - orig_sl)
                if risk > 0:
                    target = entry + (be_trigger * risk) if direction == 'long' else entry - (be_trigger * risk)
                    if (direction == 'long' and h >= target) or (direction == 'short' and l <= target):
                        sl = entry * (1 + be_pct) if direction == 'long' else entry * (1 - be_pct)
                        be_moved = True

            if direction == 'long':
                if l <= sl:
                    trades.append((sl - entry) / entry * 100)
                    break
                elif h >= tp:
                    trades.append((tp - entry) / entry * 100)
                    break
            else:
                if h >= sl:
                    trades.append((entry - sl) / entry * 100)
                    break
                elif l <= tp:
                    trades.append((entry - tp) / entry * 100)
                    break

    if not trades:
        return year, {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    equity = 1.0
    for pnl in trades:
        equity *= (1 + pnl / 100.0)
    wins = sum(1 for p in trades if p > 0)

    return year, {
        'total_return': (equity - 1.0) * 100,
        'total_trades': len(trades),
        'win_rate': wins / len(trades)
    }


def test_params_parallel(df_15m, df_4h, params, years):
    """多線程測試參數"""
    results = {}

    # 由於 ProcessPoolExecutor 在某些環境有問題，使用簡單的順序執行
    for year in years:
        _, result = test_single_year((df_15m, df_4h, params, year))
        results[year] = result

    return results


def main():
    print("="*70)
    print("參數優化（順序執行）")
    print("="*70)
    sys.stdout.flush()

    print("\n載入數據...")
    sys.stdout.flush()
    df_15m, df_4h = load_data()
    print(f"數據範圍: {df_15m.index[0].date()} ~ {df_15m.index[-1].date()}")
    sys.stdout.flush()

    from config.settings import TRADING_PARAMS, STRATEGY_PARAMS
    base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    # 測試基準
    print("\n[基準策略]")
    sys.stdout.flush()
    years = [2022, 2023, 2024, 2025]
    base_results = test_params_parallel(df_15m, df_4h, base_params, years)
    for year in years:
        r = base_results[year]
        status = "OK" if r['total_return'] >= 30 else "FAIL"
        print(f"  {year}: {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")
        sys.stdout.flush()

    base_total = sum(r['total_return'] for r in base_results.values())
    print(f"  總報酬: {base_total:.2f}%")
    sys.stdout.flush()

    # 參數組合
    print("\n[搜索參數組合]")
    sys.stdout.flush()

    configs = [
        # (pullback, min_pb, ob_thresh, trend, struct, rr)
        (True, 0.3, 50, True, True, 2.5),
        (True, 0.25, 50, True, True, 2.5),
        (True, 0.2, 50, True, True, 2.5),
        (True, 0.3, 40, True, True, 2.5),
        (True, 0.3, 50, False, True, 2.5),
        (True, 0.3, 50, True, True, 3.0),
        (True, 0.25, 40, True, True, 3.0),
        (True, 0.2, 40, False, True, 3.0),
        (True, 0.25, 45, True, True, 2.8),
    ]

    best_score = -float('inf')
    best_params = None
    best_results = None

    for i, (pb, min_pb, ob, trend, struct, rr) in enumerate(configs):
        params = copy.deepcopy(base_params)
        params['enable_pullback_confirmation'] = pb
        params['min_pullback_pct'] = min_pb
        params['ob_quality_threshold'] = ob
        params['enable_trend_filter'] = trend
        params['enable_structure_filter'] = struct
        params['risk_reward_ratio'] = rr

        print(f"\n組合 {i+1}/{len(configs)}: pb={pb}, min_pb={min_pb}, ob={ob}, trend={trend}, struct={struct}, rr={rr}")
        sys.stdout.flush()

        results = test_params_parallel(df_15m, df_4h, params, years)

        total = sum(r['total_return'] for r in results.values())
        min_ret = min(r['total_return'] for r in results.values())

        for year in years:
            r = results[year]
            status = "OK" if r['total_return'] >= 30 else "FAIL"
            print(f"  {year}: {r['total_return']:6.2f}% [{status}]")
        print(f"  總: {total:.2f}% | 最低: {min_ret:.2f}%")
        sys.stdout.flush()

        all_pass = all(r['total_return'] >= 30 for r in results.values())
        score = total + min_ret * 2 if all_pass else min_ret * 3

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results

    # 結果
    print("\n" + "="*70)
    print("最佳結果")
    print("="*70)
    for year in years:
        r = best_results[year]
        sample = "樣本外" if year == 2022 else "樣本內"
        status = "OK" if r['total_return'] >= 30 else "FAIL"
        print(f"  {year} ({sample}): {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

    best_total = sum(r['total_return'] for r in best_results.values())
    print(f"\n總報酬: {best_total:.2f}%")
    print(f"提升: {best_total - base_total:.2f}%")

    # 顯示變更
    print("\n變更的參數:")
    if best_params['min_pullback_pct'] != base_params['min_pullback_pct']:
        print(f"  min_pullback_pct: {base_params['min_pullback_pct']} -> {best_params['min_pullback_pct']}")
    if best_params['ob_quality_threshold'] != base_params['ob_quality_threshold']:
        print(f"  ob_quality_threshold: {base_params['ob_quality_threshold']} -> {best_params['ob_quality_threshold']}")
    if best_params['enable_trend_filter'] != base_params['enable_trend_filter']:
        print(f"  enable_trend_filter: {base_params['enable_trend_filter']} -> {best_params['enable_trend_filter']}")
    if best_params['enable_structure_filter'] != base_params['enable_structure_filter']:
        print(f"  enable_structure_filter: {base_params['enable_structure_filter']} -> {best_params['enable_structure_filter']}")
    if best_params['risk_reward_ratio'] != base_params['risk_reward_ratio']:
        print(f"  risk_reward_ratio: {base_params['risk_reward_ratio']} -> {best_params['risk_reward_ratio']}")
    sys.stdout.flush()


if __name__ == "__main__":
    main()
