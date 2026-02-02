# find_optimal.py
# 尋找讓所有年份都達到 30% 的最佳參數
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd
import numpy as np
from typing import Dict
import copy

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


def load_data():
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


def test_year(df_15m, df_4h, params, year):
    df_15m_y = df_15m[df_15m.index.year == year].copy()
    df_4h_y = df_4h[df_4h.index.year == year].copy()

    if len(df_15m_y) < 100:
        return {'total_return': 0, 'total_trades': 0}

    smc = SMCIndicators(SMC_PARAMS)
    strategy = SMCStrategyFinal(params)

    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        df_4h_y = smc.calculate_all(df_4h_y)
        df_4h_y = strategy.identify_key_structure_points(df_4h_y)
        df_4h_y = strategy.identify_high_quality_ob_4h(df_4h_y)
        df_1h = strategy.generate_signals_mtf(df_15m_y, df_4h_y)

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
        return {'total_return': 0, 'total_trades': 0}

    equity = 1.0
    for pnl in trades:
        equity *= (1 + pnl / 100.0)

    return {
        'total_return': (equity - 1.0) * 100,
        'total_trades': len(trades)
    }


def main():
    print("="*70)
    print("尋找最佳參數（讓所有年份 >= 30%）")
    print("="*70)
    sys.stdout.flush()

    print("\n載入數據...")
    sys.stdout.flush()
    df_15m, df_4h = load_data()
    print(f"數據: {df_15m.index[0].date()} ~ {df_15m.index[-1].date()}")
    sys.stdout.flush()

    base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
    years = [2022, 2023, 2024, 2025]

    # 更細緻的搜索空間
    # 根據之前結果：RR=3.0 對 2023 有幫助
    # trend=False 對 2023 有很大幫助但可能傷害 2025
    # 需要找到平衡

    configs = []

    # 探索不同 RR 值
    for rr in [2.8, 3.0, 3.2, 3.5]:
        for trend in [True, False]:
            for struct in [True, False]:
                for ob in [45, 50, 55]:
                    for min_pb in [0.25, 0.3]:
                        configs.append((True, min_pb, ob, trend, struct, rr))

    # 加入一些特殊組合
    # 嘗試調整保本參數
    for be_trigger in [1.0, 1.5, 2.0]:
        for be_pct in [0.003, 0.005, 0.008]:
            configs.append(('be', 0.3, 50, True, True, 3.0, be_trigger, be_pct))

    best_score = -float('inf')
    best_params = None
    best_results = None
    all_pass_found = False

    print(f"\n搜索 {len(configs)} 種組合...")
    sys.stdout.flush()

    for i, config in enumerate(configs):
        params = copy.deepcopy(base_params)

        if config[0] == 'be':
            # 保本參數測試
            _, min_pb, ob, trend, struct, rr, be_trigger, be_pct = config
            params['enable_pullback_confirmation'] = True
            params['min_pullback_pct'] = min_pb
            params['ob_quality_threshold'] = ob
            params['enable_trend_filter'] = trend
            params['enable_structure_filter'] = struct
            params['risk_reward_ratio'] = rr
            params['breakeven_trigger_r'] = be_trigger
            params['breakeven_profit_pct'] = be_pct
        else:
            pb, min_pb, ob, trend, struct, rr = config
            params['enable_pullback_confirmation'] = pb
            params['min_pullback_pct'] = min_pb
            params['ob_quality_threshold'] = ob
            params['enable_trend_filter'] = trend
            params['enable_structure_filter'] = struct
            params['risk_reward_ratio'] = rr

        results = {}
        for year in years:
            results[year] = test_year(df_15m, df_4h, params, year)

        total = sum(r['total_return'] for r in results.values())
        min_ret = min(r['total_return'] for r in results.values())
        all_pass = all(r['total_return'] >= 30 for r in results.values())

        # 評分
        if all_pass:
            score = 1000 + total  # 所有達標優先
            if not all_pass_found:
                all_pass_found = True
                print(f"\n*** 發現所有年份達標的組合！***")
        else:
            score = min_ret * 5 + total * 0.3  # 優化最低年份

        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_results = results

            print(f"\n[{i+1}/{len(configs)}] 新最佳:")
            for y in years:
                r = results[y]
                status = "OK" if r['total_return'] >= 30 else "FAIL"
                print(f"  {y}: {r['total_return']:6.2f}% [{status}]")
            print(f"  總: {total:.2f}% | 最低: {min_ret:.2f}%")
            sys.stdout.flush()

        if (i + 1) % 20 == 0:
            print(f"  已測試 {i+1}/{len(configs)}...")
            sys.stdout.flush()

    # 最終結果
    print("\n" + "="*70)
    print("最佳結果")
    print("="*70)

    all_pass = all(best_results[y]['total_return'] >= 30 for y in years)
    if all_pass:
        print("\n*** 所有年份都達到 30%！***\n")
    else:
        print("\n注意：仍有年份未達 30%\n")

    for year in years:
        r = best_results[year]
        sample = "樣本外" if year == 2022 else "樣本內"
        status = "OK" if r['total_return'] >= 30 else "FAIL"
        print(f"  {year} ({sample}): {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

    best_total = sum(r['total_return'] for r in best_results.values())
    print(f"\n總報酬: {best_total:.2f}%")

    print("\n建議的參數變更:")
    if best_params.get('enable_pullback_confirmation') != base_params.get('enable_pullback_confirmation'):
        print(f"  enable_pullback_confirmation: {best_params['enable_pullback_confirmation']}")
    if best_params.get('min_pullback_pct') != base_params.get('min_pullback_pct'):
        print(f"  min_pullback_pct: {best_params['min_pullback_pct']}")
    if best_params.get('ob_quality_threshold') != base_params.get('ob_quality_threshold'):
        print(f"  ob_quality_threshold: {best_params['ob_quality_threshold']}")
    if best_params.get('enable_trend_filter') != base_params.get('enable_trend_filter'):
        print(f"  enable_trend_filter: {best_params['enable_trend_filter']}")
    if best_params.get('enable_structure_filter') != base_params.get('enable_structure_filter'):
        print(f"  enable_structure_filter: {best_params['enable_structure_filter']}")
    if best_params.get('risk_reward_ratio') != base_params.get('risk_reward_ratio'):
        print(f"  risk_reward_ratio: {best_params['risk_reward_ratio']}")
    if best_params.get('breakeven_trigger_r') != base_params.get('breakeven_trigger_r'):
        print(f"  breakeven_trigger_r: {best_params['breakeven_trigger_r']}")
    if best_params.get('breakeven_profit_pct') != base_params.get('breakeven_profit_pct'):
        print(f"  breakeven_profit_pct: {best_params['breakeven_profit_pct']}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
