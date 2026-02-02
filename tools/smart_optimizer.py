# smart_optimizer.py
# 智能策略優化器 - 支持樣本內外測試，避免過擬合
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
import json
import copy

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加父目錄到路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


class SmartOptimizer:
    """智能策略優化器"""

    def __init__(self):
        """初始化"""
        data_dir = os.path.join(PROJECT_ROOT, 'data')

        # 載入 2023-2025 數據（樣本內）
        print("載入 2023-2025 數據（樣本內）...")
        df_2023_now = pd.read_csv(
            os.path.join(data_dir, 'ETHUSDT_15m_2023_now.csv'),
            index_col='ts', parse_dates=True
        )
        print(f"  {len(df_2023_now)} 根 15m K線")

        # 載入 2022 數據（樣本外）
        print("載入 2022 數據（樣本外）...")
        df_2022 = pd.read_csv(
            os.path.join(data_dir, 'ETHUSDT_15m_2022_ccxt.csv'),
            index_col='ts', parse_dates=True
        )
        print(f"  {len(df_2022)} 根 15m K線")

        # 合併數據
        self.df_15m = pd.concat([df_2022, df_2023_now]).sort_index()
        self.df_15m = self.df_15m[~self.df_15m.index.duplicated(keep='first')]

        # 聚合為 4H
        self.df_4h = self.df_15m.resample('4H').agg({
            'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
        }).dropna()

        print(f"總共: {len(self.df_15m)} 根 15m, {len(self.df_4h)} 根 4H K線")
        print(f"時間範圍: {self.df_15m.index[0].date()} ~ {self.df_15m.index[-1].date()}")

    def test_params(self, params: Dict, years: List[int], verbose=False) -> Dict[int, Dict]:
        """測試參數在指定年份的表現"""
        results = {}

        for year in years:
            df_15m = self.df_15m[self.df_15m.index.year == year].copy()
            df_4h = self.df_4h[self.df_4h.index.year == year].copy()

            if len(df_15m) < 100 or len(df_4h) < 50:
                results[year] = {'total_return': 0, 'total_trades': 0, 'win_rate': 0,
                                'max_drawdown': 0, 'profit_factor': 0}
                continue

            # 初始化
            smc_indicators = SMCIndicators(SMC_PARAMS)
            strategy = SMCStrategyFinal(params)

            # 計算指標（靜默）
            import contextlib
            import io as _io
            with contextlib.redirect_stdout(_io.StringIO()):
                df_4h = smc_indicators.calculate_all(df_4h)
                df_4h = strategy.identify_key_structure_points(
                    df_4h, min_move_pct=params.get('min_structure_move_pct', 0.02)
                )
                df_4h = strategy.identify_high_quality_ob_4h(df_4h)
                df_1h = strategy.generate_signals_mtf(df_15m, df_4h)

            # 模擬交易
            trades = self._simulate_trades(df_1h, params)
            results[year] = self._calculate_results(trades)

            if verbose:
                r = results[year]
                status = "OK" if r['total_return'] >= 30 else "FAIL"
                print(f"  {year}: {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | "
                      f"勝率 {r['win_rate']*100:5.1f}% | MDD {r['max_drawdown']*100:5.1f}% | [{status}]")

        return results

    def _simulate_trades(self, df_1h, params):
        """模擬交易"""
        trades = []
        enable_breakeven = params.get('enable_breakeven', True)
        breakeven_trigger_r = params.get('breakeven_trigger_r', 1.5)
        breakeven_profit_pct = params.get('breakeven_profit_pct', 0.005)

        signals = df_1h[df_1h['signal'] != 0]

        for idx, signal_row in signals.iterrows():
            direction = 'long' if signal_row['signal'] == 1 else 'short'
            entry_price = signal_row['entry_price']
            stop_loss = signal_row['stop_loss']
            take_profit = signal_row['take_profit']
            original_stop_loss = stop_loss

            trade = {
                'direction': direction,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'original_stop_loss': original_stop_loss,
                'pnl_pct': None,
                'breakeven_moved': False
            }

            future_candles = df_1h[df_1h.index > idx]

            for candle_idx, candle in future_candles.iterrows():
                high = candle['h']
                low = candle['l']

                if enable_breakeven and not trade['breakeven_moved']:
                    risk = abs(entry_price - original_stop_loss)
                    if risk > 0:
                        if direction == 'long':
                            target = entry_price + (breakeven_trigger_r * risk)
                            if high >= target:
                                trade['stop_loss'] = entry_price * (1 + breakeven_profit_pct)
                                trade['breakeven_moved'] = True
                        else:
                            target = entry_price - (breakeven_trigger_r * risk)
                            if low <= target:
                                trade['stop_loss'] = entry_price * (1 - breakeven_profit_pct)
                                trade['breakeven_moved'] = True

                if direction == 'long':
                    if low <= trade['stop_loss']:
                        trade['pnl_pct'] = (trade['stop_loss'] - entry_price) / entry_price * 100
                        break
                    elif high >= take_profit:
                        trade['pnl_pct'] = (take_profit - entry_price) / entry_price * 100
                        break
                else:
                    if high >= trade['stop_loss']:
                        trade['pnl_pct'] = (entry_price - trade['stop_loss']) / entry_price * 100
                        break
                    elif low <= take_profit:
                        trade['pnl_pct'] = (entry_price - take_profit) / entry_price * 100
                        break

            if trade['pnl_pct'] is not None:
                trades.append(trade)

        return trades

    def _calculate_results(self, trades):
        """計算結果"""
        if not trades:
            return {'total_trades': 0, 'win_rate': 0, 'total_return': 0,
                   'max_drawdown': 0, 'profit_factor': 0}

        total_trades = len(trades)
        wins = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(wins) / total_trades

        equity = 1.0
        peak = 1.0
        max_dd = 0.0
        total_profit = 0.0
        total_loss = 0.0

        for t in trades:
            equity *= (1 + t['pnl_pct'] / 100.0)
            peak = max(peak, equity)
            max_dd = max(max_dd, (peak - equity) / peak)

            if t['pnl_pct'] > 0:
                total_profit += t['pnl_pct']
            else:
                total_loss += abs(t['pnl_pct'])

        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': (equity - 1.0) * 100,
            'max_drawdown': max_dd,
            'profit_factor': total_profit / total_loss if total_loss > 0 else 99.99
        }

    def optimize(self, param_grid: Dict, in_sample_years: List[int],
                 out_sample_years: List[int], target_return: float = 30.0):
        """
        參數優化

        Args:
            param_grid: 參數搜索空間
            in_sample_years: 樣本內年份（用於優化）
            out_sample_years: 樣本外年份（用於驗證）
            target_return: 每年目標報酬率
        """
        print("\n" + "="*70)
        print("智能參數優化")
        print(f"樣本內: {in_sample_years} | 樣本外: {out_sample_years}")
        print(f"目標: 每年 >= {target_return}%")
        print("="*70)

        # 基準測試
        base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
        print("\n【基準策略】")
        base_results = self.test_params(base_params, in_sample_years + out_sample_years, verbose=True)
        base_total = sum(r['total_return'] for r in base_results.values())
        print(f"基準總報酬: {base_total:.2f}%")

        # 參數搜索
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"\n搜索 {len(all_combinations)} 種參數組合...")

        best_params = None
        best_score = -float('inf')
        best_in_sample = None
        best_out_sample = None

        for i, combo in enumerate(all_combinations):
            test_params = copy.deepcopy(base_params)
            for name, value in zip(param_names, combo):
                test_params[name] = value

            # 樣本內測試
            in_results = self.test_params(test_params, in_sample_years)

            # 快速過濾：如果樣本內表現太差就跳過
            in_min = min(r['total_return'] for r in in_results.values())
            if in_min < target_return * 0.5:  # 至少達到目標的一半
                continue

            # 計算評分
            in_total = sum(r['total_return'] for r in in_results.values())
            in_all_pass = all(r['total_return'] >= target_return for r in in_results.values())

            # 優先選擇所有年份都達標的
            if in_all_pass:
                score = in_total + in_min * 2
            else:
                score = in_min

            if score > best_score:
                # 樣本外驗證
                out_results = self.test_params(test_params, out_sample_years)
                out_total = sum(r['total_return'] for r in out_results.values())
                out_min = min(r['total_return'] for r in out_results.values())

                # 樣本外也要有基本表現
                if out_min > 0:
                    best_score = score
                    best_params = test_params.copy()
                    best_in_sample = in_results
                    best_out_sample = out_results

                    print(f"\n[{i+1}/{len(all_combinations)}] 發現更佳組合:")
                    for name, value in zip(param_names, combo):
                        if value != base_params.get(name):
                            print(f"  {name}: {base_params.get(name)} -> {value}")
                    print(f"  樣本內: {in_total:.2f}% (min: {in_min:.2f}%)")
                    print(f"  樣本外: {out_total:.2f}% (min: {out_min:.2f}%)")

            if (i + 1) % 100 == 0:
                print(f"  已測試 {i+1}/{len(all_combinations)} 組合...")

        # 最終結果
        if best_params:
            print("\n" + "="*70)
            print("最佳參數組合")
            print("="*70)

            print("\n【樣本內表現】")
            for year in in_sample_years:
                r = best_in_sample[year]
                status = "OK" if r['total_return'] >= target_return else "FAIL"
                print(f"  {year}: {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

            print("\n【樣本外驗證】")
            for year in out_sample_years:
                r = best_out_sample[year]
                status = "OK" if r['total_return'] >= target_return else "FAIL"
                print(f"  {year}: {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | [{status}]")

            total_all = sum(r['total_return'] for r in best_in_sample.values()) + \
                       sum(r['total_return'] for r in best_out_sample.values())
            print(f"\n總報酬（所有年份）: {total_all:.2f}%")
            print(f"相較基準提升: {total_all - base_total:.2f}%")

            # 保存參數
            changed = {}
            for key in param_grid.keys():
                if best_params.get(key) != base_params.get(key):
                    changed[key] = {'old': base_params.get(key), 'new': best_params.get(key)}

            output_path = os.path.join(PROJECT_ROOT, 'tools', 'best_params.json')
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(changed, f, indent=2, ensure_ascii=False)
            print(f"\n最佳參數已保存至: {output_path}")

            return best_params, best_in_sample, best_out_sample

        print("\n未找到符合條件的參數組合")
        return None, None, None


def main():
    """主函數"""
    optimizer = SmartOptimizer()

    # 定義搜索空間（更全面的參數）
    param_grid = {
        # 核心過濾條件
        'enable_pullback_confirmation': [True, False],
        'min_pullback_pct': [0.2, 0.25, 0.3, 0.35],

        # OB 質量
        'ob_quality_threshold': [40, 50, 60],

        # 過濾器開關
        'enable_trend_filter': [True, False],
        'enable_structure_filter': [True, False],

        # 風險報酬
        'risk_reward_ratio': [2.0, 2.5, 3.0, 3.5],

        # RSI
        'rsi_long_max': [80, 85, 90],
        'rsi_short_min': [10, 15, 20],
    }

    # 執行優化
    # 樣本內: 2023-2025, 樣本外: 2022
    best_params, in_results, out_results = optimizer.optimize(
        param_grid=param_grid,
        in_sample_years=[2023, 2024, 2025],
        out_sample_years=[2022],
        target_return=30.0
    )


if __name__ == "__main__":
    main()
