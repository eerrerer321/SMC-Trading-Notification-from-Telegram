# test_strategy.py
# 快速測試策略各年份表現
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# 添加父目錄到路徑
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


def test_year(df_15m, df_4h, year, params=None, verbose=True):
    """測試指定年份"""
    if params is None:
        params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    # 過濾數據
    df_15m_year = df_15m[df_15m.index.year == year].copy()
    df_4h_year = df_4h[df_4h.index.year == year].copy()

    if len(df_15m_year) < 100:
        return None

    # 初始化
    smc_indicators = SMCIndicators(SMC_PARAMS)
    strategy = SMCStrategyFinal(params)

    # 計算 SMC 指標（靜默模式）
    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        df_4h_year = smc_indicators.calculate_all(df_4h_year)
        min_move_pct = params.get('min_structure_move_pct', 0.02)
        df_4h_year = strategy.identify_key_structure_points(df_4h_year, min_move_pct=min_move_pct)
        df_4h_year = strategy.identify_high_quality_ob_4h(df_4h_year)
        df_1h = strategy.generate_signals_mtf(df_15m_year, df_4h_year)

    # 模擬交易
    trades = simulate_trades(df_1h, params)

    # 計算結果
    result = calculate_results(trades)
    result['year'] = year

    if verbose:
        status = "OK" if result['total_return'] >= 30 else "FAIL"
        print(f"  {year}: {result['total_return']:6.2f}% | "
              f"{result['total_trades']:3d} 筆 | "
              f"勝率 {result['win_rate']*100:5.1f}% | "
              f"MDD {result['max_drawdown']*100:5.1f}% | "
              f"PF {result['profit_factor']:5.2f} | [{status}]")

    return result


def simulate_trades(df_1h, params):
    """模擬交易（使用 /chart 相同邏輯）"""
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


def calculate_results(trades):
    """計算結果"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_return': 0,
            'max_drawdown': 0, 'profit_factor': 0
        }

    total_trades = len(trades)
    wins = [t for t in trades if t['pnl_pct'] > 0]
    win_rate = len(wins) / total_trades

    # 複利報酬
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


def test_all_years(params=None, verbose=True):
    """測試所有年份"""
    data_path = os.path.join(PROJECT_ROOT, 'data', 'ETHUSDT_15m_2023_now.csv')

    if verbose:
        print("載入數據...")
    df_15m = pd.read_csv(data_path, index_col='ts', parse_dates=True)
    df_4h = df_15m.resample('4H').agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()

    if verbose:
        print(f"數據: {df_15m.index[0].date()} ~ {df_15m.index[-1].date()}")
        print("-" * 70)

    results = {}
    for year in [2023, 2024, 2025]:
        result = test_year(df_15m, df_4h, year, params, verbose)
        if result:
            results[year] = result

    if verbose:
        print("-" * 70)
        total = sum(r['total_return'] for r in results.values())
        all_pass = all(r['total_return'] >= 30 for r in results.values())
        status = "ALL PASS" if all_pass else "NEED IMPROVEMENT"
        print(f"總報酬: {total:.2f}% | {status}")

    return results


if __name__ == "__main__":
    print("="*70)
    print("SMC 策略年度測試")
    print("="*70)
    test_all_years()
