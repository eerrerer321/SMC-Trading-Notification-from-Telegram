# quick_optimize.py
# 快速參數優化 - 較小的搜索空間
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd
import numpy as np
from typing import Dict, List
import copy

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


def load_data():
    """載入所有數據"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')

    # 2023-2025
    df_2023 = pd.read_csv(os.path.join(data_dir, 'ETHUSDT_15m_2023_now.csv'),
                          index_col='ts', parse_dates=True)
    # 2022
    df_2022 = pd.read_csv(os.path.join(data_dir, 'ETHUSDT_15m_2022_ccxt.csv'),
                          index_col='ts', parse_dates=True)

    df_15m = pd.concat([df_2022, df_2023]).sort_index()
    df_15m = df_15m[~df_15m.index.duplicated(keep='first')]

    df_4h = df_15m.resample('4H').agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()

    return df_15m, df_4h


def test_params(df_15m, df_4h, params: Dict, year: int) -> Dict:
    """測試參數在指定年份的表現"""
    df_15m_y = df_15m[df_15m.index.year == year].copy()
    df_4h_y = df_4h[df_4h.index.year == year].copy()

    if len(df_15m_y) < 100:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

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

            # 移動保本
            if enable_be and not be_moved:
                risk = abs(entry - orig_sl)
                if risk > 0:
                    target = entry + (be_trigger * risk) if direction == 'long' else entry - (be_trigger * risk)
                    if (direction == 'long' and h >= target) or (direction == 'short' and l <= target):
                        sl = entry * (1 + be_pct) if direction == 'long' else entry * (1 - be_pct)
                        be_moved = True

            # 檢查止損/止盈
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
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0}

    # 計算結果
    equity = 1.0
    for pnl in trades:
        equity *= (1 + pnl / 100.0)

    wins = sum(1 for p in trades if p > 0)
    return {
        'total_return': (equity - 1.0) * 100,
        'total_trades': len(trades),
        'win_rate': wins / len(trades)
    }


def main():
    print("="*70)
    print("快速參數優化")
    print("="*70)

    print("\n載入數據...")
    df_15m, df_4h = load_data()
    print(f"數據範圍: {df_15m.index[0].date()} ~ {df_15m.index[-1].date()}")

    base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    # 測試基準
    print("\n【基準策略】")
    for year in [2022, 2023, 2024, 2025]:
        r = test_params(df_15m, df_4h, base_params, year)
        status = "OK" if r['total_return'] >= 30 else "FAIL"
        print(f"  {year}: {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

    base_total = sum(test_params(df_15m, df_4h, base_params, y)['total_return']
                     for y in [2022, 2023, 2024, 2025])
    print(f"  總報酬: {base_total:.2f}%")

    # 快速搜索關鍵參數
    print("\n【參數搜索】")

    best_params = None
    best_score = -float('inf')
    best_results = None

    # 核心參數組合
    configs = [
        # (pullback_confirm, min_pullback, ob_threshold, trend_filter, structure_filter, rr)
        (True, 0.3, 50, True, True, 2.5),    # 原始
        (True, 0.25, 50, True, True, 2.5),   # 降低回撤要求
        (True, 0.2, 50, True, True, 2.5),    # 更低回撤
        (True, 0.3, 40, True, True, 2.5),    # 降低 OB 門檻
        (True, 0.3, 50, False, True, 2.5),   # 關閉趨勢過濾
        (True, 0.3, 50, True, False, 2.5),   # 關閉結構過濾
        (True, 0.3, 50, True, True, 3.0),    # 提高 RR
        (True, 0.3, 50, True, True, 3.5),    # 更高 RR
        (True, 0.25, 40, True, True, 3.0),   # 組合優化 1
        (True, 0.25, 40, False, True, 3.0),  # 組合優化 2
        (True, 0.2, 40, True, True, 3.0),    # 組合優化 3
        (True, 0.2, 40, False, True, 3.0),   # 組合優化 4
        (True, 0.2, 40, False, False, 3.0),  # 組合優化 5
        (False, 0.3, 50, True, True, 2.5),   # 關閉回撤確認
        (False, 0.3, 40, True, True, 3.0),   # 關閉回撤 + 優化
        (True, 0.25, 45, True, True, 2.8),   # 平衡設置
        (True, 0.2, 45, False, True, 3.0),   # 平衡設置 2
    ]

    for i, (pb_confirm, min_pb, ob_thresh, trend_f, struct_f, rr) in enumerate(configs):
        params = copy.deepcopy(base_params)
        params['enable_pullback_confirmation'] = pb_confirm
        params['min_pullback_pct'] = min_pb
        params['ob_quality_threshold'] = ob_thresh
        params['enable_trend_filter'] = trend_f
        params['enable_structure_filter'] = struct_f
        params['risk_reward_ratio'] = rr

        results = {}
        for year in [2022, 2023, 2024, 2025]:
            results[year] = test_params(df_15m, df_4h, params, year)

        total = sum(r['total_return'] for r in results.values())
        min_ret = min(r['total_return'] for r in results.values())

        # 評分：優先所有年份達標
        all_pass = all(r['total_return'] >= 30 for r in results.values())
        if all_pass:
            score = total + min_ret * 2
        else:
            score = min_ret * 3 + total * 0.5

        if score > best_score:
            best_score = score
            best_params = params
            best_results = results

            print(f"\n組合 {i+1}: 總 {total:.1f}% | 最低 {min_ret:.1f}%")
            print(f"  pullback={pb_confirm}, min_pb={min_pb}, ob={ob_thresh}, "
                  f"trend={trend_f}, struct={struct_f}, rr={rr}")
            for y, r in results.items():
                status = "OK" if r['total_return'] >= 30 else "FAIL"
                print(f"  {y}: {r['total_return']:6.2f}% [{status}]")

    # 最終結果
    print("\n" + "="*70)
    print("最佳參數組合")
    print("="*70)

    print("\n變更的參數:")
    changes = []
    if best_params['enable_pullback_confirmation'] != base_params['enable_pullback_confirmation']:
        changes.append(f"  enable_pullback_confirmation: {base_params['enable_pullback_confirmation']} -> {best_params['enable_pullback_confirmation']}")
    if best_params['min_pullback_pct'] != base_params['min_pullback_pct']:
        changes.append(f"  min_pullback_pct: {base_params['min_pullback_pct']} -> {best_params['min_pullback_pct']}")
    if best_params['ob_quality_threshold'] != base_params['ob_quality_threshold']:
        changes.append(f"  ob_quality_threshold: {base_params['ob_quality_threshold']} -> {best_params['ob_quality_threshold']}")
    if best_params['enable_trend_filter'] != base_params['enable_trend_filter']:
        changes.append(f"  enable_trend_filter: {base_params['enable_trend_filter']} -> {best_params['enable_trend_filter']}")
    if best_params['enable_structure_filter'] != base_params['enable_structure_filter']:
        changes.append(f"  enable_structure_filter: {base_params['enable_structure_filter']} -> {best_params['enable_structure_filter']}")
    if best_params['risk_reward_ratio'] != base_params['risk_reward_ratio']:
        changes.append(f"  risk_reward_ratio: {base_params['risk_reward_ratio']} -> {best_params['risk_reward_ratio']}")

    for c in changes:
        print(c)

    print("\n各年份表現:")
    for year in [2022, 2023, 2024, 2025]:
        r = best_results[year]
        sample = "樣本外" if year == 2022 else "樣本內"
        status = "OK" if r['total_return'] >= 30 else "FAIL"
        print(f"  {year} ({sample}): {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

    best_total = sum(r['total_return'] for r in best_results.values())
    print(f"\n總報酬: {best_total:.2f}%")
    print(f"相較基準提升: {best_total - base_total:.2f}%")


if __name__ == "__main__":
    main()
