# optimizer.py
# SMC 策略優化器 - 尋找最佳參數組合
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import itertools
import json

# 設定 stdout 編碼
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


@dataclass
class BacktestResult:
    """回測結果"""
    year: int
    total_trades: int
    win_count: int
    loss_count: int
    win_rate: float
    total_return: float  # 複利報酬率
    max_drawdown: float
    profit_factor: float
    avg_r: float


class SMCOptimizer:
    """SMC 策略優化器"""

    def __init__(self, data_path: str):
        """初始化優化器"""
        print("正在載入數據...")
        self.df_15m = pd.read_csv(data_path, index_col='ts', parse_dates=True)
        print(f"載入 {len(self.df_15m)} 根 15m K線")
        print(f"時間範圍: {self.df_15m.index[0]} ~ {self.df_15m.index[-1]}")

        # 聚合為 4H
        self.df_4h = self.df_15m.resample('4H').agg({
            'o': 'first',
            'h': 'max',
            'l': 'min',
            'c': 'last',
            'v': 'sum'
        }).dropna()
        print(f"聚合為 {len(self.df_4h)} 根 4H K線")

    def run_backtest(self, params: Dict, year: Optional[int] = None,
                     start_date: Optional[str] = None,
                     end_date: Optional[str] = None) -> BacktestResult:
        """
        執行回測

        Args:
            params: 策略參數
            year: 指定年份（可選）
            start_date: 開始日期（可選）
            end_date: 結束日期（可選）
        """
        # 過濾數據
        df_15m = self.df_15m.copy()
        df_4h = self.df_4h.copy()

        if year:
            df_15m = df_15m[df_15m.index.year == year]
            df_4h = df_4h[df_4h.index.year == year]
        elif start_date and end_date:
            df_15m = df_15m[start_date:end_date]
            df_4h = df_4h[start_date:end_date]

        if len(df_15m) < 100 or len(df_4h) < 50:
            return BacktestResult(
                year=year or 0, total_trades=0, win_count=0, loss_count=0,
                win_rate=0, total_return=0, max_drawdown=0, profit_factor=0, avg_r=0
            )

        # 初始化指標計算器和策略
        smc_indicators = SMCIndicators(SMC_PARAMS)
        strategy = SMCStrategyFinal(params)

        # 計算 SMC 指標
        df_4h = smc_indicators.calculate_all(df_4h)

        # 識別關鍵結構位
        min_move_pct = params.get('min_structure_move_pct', 0.02)
        df_4h = strategy.identify_key_structure_points(df_4h, min_move_pct=min_move_pct)

        # 識別高質量 OB
        df_4h = strategy.identify_high_quality_ob_4h(df_4h)

        # 生成訊號
        df_1h = strategy.generate_signals_mtf(df_15m, df_4h)

        # 模擬交易
        trades = self._simulate_trades(df_1h, params)

        # 計算結果
        return self._calculate_results(trades, year or 0)

    def _simulate_trades(self, df_1h: pd.DataFrame, params: Dict) -> List[Dict]:
        """模擬交易"""
        trades = []

        enable_breakeven = params.get('enable_breakeven', True)
        breakeven_trigger_r = params.get('breakeven_trigger_r', 1.5)
        breakeven_profit_pct = params.get('breakeven_profit_pct', 0.005)

        signals = df_1h[df_1h['signal'] != 0]

        for idx, signal_row in signals.iterrows():
            direction = 'long' if signal_row['signal'] == 1 else 'short'
            entry_time = idx
            entry_price = signal_row['entry_price']
            stop_loss = signal_row['stop_loss']
            take_profit = signal_row['take_profit']
            original_stop_loss = stop_loss

            trade = {
                'direction': direction,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'original_stop_loss': original_stop_loss,
                'exit_time': None,
                'exit_price': None,
                'exit_reason': None,
                'pnl_pct': None,
                'breakeven_moved': False
            }

            future_candles = df_1h[df_1h.index > entry_time]

            for candle_idx, candle in future_candles.iterrows():
                high = candle['h']
                low = candle['l']

                # 移動保本
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

                # 檢查止損/止盈
                if direction == 'long':
                    if low <= trade['stop_loss']:
                        trade['exit_time'] = candle_idx
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'sl'
                        trade['pnl_pct'] = (trade['exit_price'] - entry_price) / entry_price * 100
                        break
                    elif high >= take_profit:
                        trade['exit_time'] = candle_idx
                        trade['exit_price'] = take_profit
                        trade['exit_reason'] = 'tp'
                        trade['pnl_pct'] = (trade['exit_price'] - entry_price) / entry_price * 100
                        break
                else:
                    if high >= trade['stop_loss']:
                        trade['exit_time'] = candle_idx
                        trade['exit_price'] = trade['stop_loss']
                        trade['exit_reason'] = 'sl'
                        trade['pnl_pct'] = (entry_price - trade['exit_price']) / entry_price * 100
                        break
                    elif low <= take_profit:
                        trade['exit_time'] = candle_idx
                        trade['exit_price'] = take_profit
                        trade['exit_reason'] = 'tp'
                        trade['pnl_pct'] = (entry_price - trade['exit_price']) / entry_price * 100
                        break

            if trade['exit_time'] is not None:
                trades.append(trade)

        return trades

    def _calculate_results(self, trades: List[Dict], year: int) -> BacktestResult:
        """計算回測結果"""
        if not trades:
            return BacktestResult(
                year=year, total_trades=0, win_count=0, loss_count=0,
                win_rate=0, total_return=0, max_drawdown=0, profit_factor=0, avg_r=0
            )

        total_trades = len(trades)
        wins = [t for t in trades if t['pnl_pct'] and t['pnl_pct'] > 0]
        losses = [t for t in trades if t['pnl_pct'] and t['pnl_pct'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # 複利報酬
        equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0
        total_profit = 0.0
        total_loss = 0.0
        r_multiples = []

        for t in trades:
            if t['pnl_pct']:
                pnl_ratio = t['pnl_pct'] / 100.0
                equity *= (1 + pnl_ratio)

                if equity > peak_equity:
                    peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                if t['pnl_pct'] > 0:
                    total_profit += t['pnl_pct']
                else:
                    total_loss += abs(t['pnl_pct'])

                # R 倍數
                risk = abs(t['entry_price'] - t['original_stop_loss'])
                if risk > 0:
                    actual_profit = t['pnl_pct'] / 100.0 * t['entry_price']
                    r_multiple = actual_profit / risk
                    r_multiples.append(r_multiple)

        total_return = (equity - 1.0) * 100
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
        avg_r = np.mean(r_multiples) if r_multiples else 0

        return BacktestResult(
            year=year,
            total_trades=total_trades,
            win_count=win_count,
            loss_count=loss_count,
            win_rate=win_rate,
            total_return=total_return,
            max_drawdown=max_drawdown,
            profit_factor=profit_factor if profit_factor != float('inf') else 99.99,
            avg_r=avg_r
        )

    def test_current_strategy(self) -> Dict[int, BacktestResult]:
        """測試目前策略在各年份的表現"""
        print("\n" + "="*80)
        print("測試目前策略")
        print("="*80)

        current_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
        results = {}

        for year in [2023, 2024, 2025]:
            print(f"\n回測 {year} 年...")
            result = self.run_backtest(current_params, year=year)
            results[year] = result

            print(f"  交易數: {result.total_trades}")
            print(f"  勝率: {result.win_rate*100:.1f}%")
            print(f"  報酬率: {result.total_return:.2f}%")
            print(f"  最大回撤: {result.max_drawdown*100:.2f}%")
            print(f"  獲利因子: {result.profit_factor:.2f}")

        return results

    def optimize(self, param_grid: Dict, target_return: float = 30.0) -> Tuple[Dict, Dict[int, BacktestResult]]:
        """
        參數優化

        Args:
            param_grid: 參數搜索空間
            target_return: 目標年報酬率

        Returns:
            (最佳參數, 各年份結果)
        """
        print("\n" + "="*80)
        print("開始參數優化")
        print(f"目標: 每年報酬率 >= {target_return}%")
        print("="*80)

        # 生成所有參數組合
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        all_combinations = list(itertools.product(*param_values))

        print(f"總共 {len(all_combinations)} 種參數組合")

        best_params = None
        best_results = None
        best_score = -float('inf')

        for i, combo in enumerate(all_combinations):
            # 構建參數字典
            test_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
            for name, value in zip(param_names, combo):
                test_params[name] = value

            # 測試各年份
            year_results = {}
            valid = True
            total_return = 0
            min_return = float('inf')

            for year in [2023, 2024, 2025]:
                result = self.run_backtest(test_params, year=year)
                year_results[year] = result
                total_return += result.total_return
                min_return = min(min_return, result.total_return)

                # 檢查是否有足夠交易
                if result.total_trades < 5:
                    valid = False
                    break

            if not valid:
                continue

            # 計算評分：優先保證每年都達標，其次看總報酬
            all_years_pass = all(r.total_return >= target_return for r in year_results.values())

            if all_years_pass:
                score = total_return + min_return * 2  # 獎勵最差年份也好
            else:
                score = min_return  # 未達標時，優化最差年份

            if score > best_score:
                best_score = score
                best_params = test_params.copy()
                best_results = year_results

                # 顯示進度
                print(f"\n[{i+1}/{len(all_combinations)}] 發現更佳組合:")
                for name, value in zip(param_names, combo):
                    print(f"  {name}: {value}")
                print(f"  2023: {year_results[2023].total_return:.2f}%")
                print(f"  2024: {year_results[2024].total_return:.2f}%")
                print(f"  2025: {year_results[2025].total_return:.2f}%")
                print(f"  總報酬: {total_return:.2f}%")

            # 進度指示
            if (i + 1) % 50 == 0:
                print(f"  已測試 {i+1}/{len(all_combinations)} 組合...")

        return best_params, best_results


def main():
    """主函數"""
    data_path = os.path.join(os.path.dirname(__file__), 'data', 'ETHUSDT_15m_2023_now.csv')
    optimizer = SMCOptimizer(data_path)

    # 1. 測試目前策略
    print("\n" + "="*80)
    print("步驟 1: 測試目前策略作為基準")
    print("="*80)
    baseline_results = optimizer.test_current_strategy()

    baseline_total = sum(r.total_return for r in baseline_results.values())
    print(f"\n基準總報酬: {baseline_total:.2f}%")

    # 2. 參數優化
    print("\n" + "="*80)
    print("步驟 2: 參數優化")
    print("="*80)

    # 定義搜索空間
    param_grid = {
        # 回撤確認
        'enable_pullback_confirmation': [True, False],
        'min_pullback_pct': [0.2, 0.3, 0.4],

        # OB 質量
        'ob_quality_threshold': [40, 50, 60],

        # 趨勢過濾
        'enable_trend_filter': [True, False],

        # 結構過濾
        'enable_structure_filter': [True, False],

        # 風險報酬比
        'risk_reward_ratio': [2.0, 2.5, 3.0],

        # RSI 過濾
        'rsi_long_max': [80, 85, 90],
        'rsi_short_min': [10, 15, 20],
    }

    best_params, best_results = optimizer.optimize(param_grid, target_return=30.0)

    if best_results:
        print("\n" + "="*80)
        print("最佳參數組合")
        print("="*80)

        # 顯示與基準的差異
        changed_params = {}
        base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
        for key in param_grid.keys():
            if best_params.get(key) != base_params.get(key):
                changed_params[key] = {
                    'old': base_params.get(key),
                    'new': best_params.get(key)
                }

        print("\n變更的參數:")
        for key, values in changed_params.items():
            print(f"  {key}: {values['old']} -> {values['new']}")

        print("\n各年份表現:")
        for year, result in best_results.items():
            status = "達標" if result.total_return >= 30 else "未達標"
            print(f"  {year}: {result.total_return:.2f}% ({result.total_trades} 筆交易, "
                  f"勝率 {result.win_rate*100:.1f}%, MDD {result.max_drawdown*100:.1f}%) [{status}]")

        best_total = sum(r.total_return for r in best_results.values())
        print(f"\n優化後總報酬: {best_total:.2f}%")
        print(f"相較基準提升: {best_total - baseline_total:.2f}%")

        # 儲存最佳參數
        output_path = os.path.join(os.path.dirname(__file__), 'optimized_params.json')
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(changed_params, f, indent=2, ensure_ascii=False)
        print(f"\n最佳參數已儲存至: {output_path}")
    else:
        print("\n未找到符合條件的參數組合")


if __name__ == "__main__":
    main()
