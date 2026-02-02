# test_all_years.py
# 測試所有年份 (2020-2025) 的策略表現
# -*- coding: utf-8 -*-

import sys
import os
import io
import pandas as pd

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


def load_all_data():
    """載入所有年份數據"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')

    dfs = []
    files = [
        'ETHUSDT_15m_2020_ccxt.csv',
        'ETHUSDT_15m_2021_ccxt.csv',
        'ETHUSDT_15m_2022_ccxt.csv',
        'ETHUSDT_15m_2023_now.csv',
    ]

    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            df = pd.read_csv(path, index_col='ts', parse_dates=True)
            dfs.append(df)
            print(f"  載入 {f}: {len(df)} 根 K 線")

    if not dfs:
        return None, None

    # 合併所有數據
    df_15m = pd.concat(dfs).sort_index()
    df_15m = df_15m[~df_15m.index.duplicated(keep='first')]

    # 轉換為 4H
    df_4h = df_15m.resample('4H').agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()

    return df_15m, df_4h


def test_year(df_15m, df_4h, year, params=None):
    """測試指定年份"""
    df_15m_y = df_15m[df_15m.index.year == year].copy()
    df_4h_y = df_4h[df_4h.index.year == year].copy()

    if len(df_15m_y) < 100:
        return None

    if params is None:
        params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

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
                    trades.append({'pnl': (sl - entry) / entry * 100, 'be': be_moved})
                    break
                elif h >= tp:
                    trades.append({'pnl': (tp - entry) / entry * 100, 'be': False})
                    break
            else:
                if h >= sl:
                    trades.append({'pnl': (entry - sl) / entry * 100, 'be': be_moved})
                    break
                elif l <= tp:
                    trades.append({'pnl': (entry - tp) / entry * 100, 'be': False})
                    break

    if not trades:
        return {
            'total_return': 0,
            'total_trades': 0,
            'win_rate': 0,
            'max_drawdown': 0,
            'profit_factor': 0
        }

    # 計算績效
    equity = 1.0
    peak = 1.0
    max_dd = 0
    total_profit = 0
    total_loss = 0
    wins = 0

    for t in trades:
        pnl = t['pnl']
        equity *= (1 + pnl / 100.0)

        if pnl > 0:
            wins += 1
            total_profit += pnl
        else:
            total_loss += abs(pnl)

        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'total_return': (equity - 1.0) * 100,
        'total_trades': len(trades),
        'win_rate': wins / len(trades) * 100,
        'max_drawdown': max_dd,
        'profit_factor': total_profit / total_loss if total_loss > 0 else 99.99
    }


def main():
    print("="*70)
    print("SMC 策略全年份測試 (2020-2025)")
    print("="*70)
    print("\n載入數據...")
    sys.stdout.flush()

    df_15m, df_4h = load_all_data()

    if df_15m is None:
        print("❌ 數據載入失敗")
        return

    print(f"\n總數據: {len(df_15m)} 根 15m K 線")
    print(f"時間範圍: {df_15m.index[0].date()} ~ {df_15m.index[-1].date()}")
    sys.stdout.flush()

    params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    print("\n" + "-"*70)
    print("年份       報酬率    交易數  勝率    MDD     PF    狀態")
    print("-"*70)
    sys.stdout.flush()

    years = [2020, 2021, 2022, 2023, 2024, 2025]
    results = {}
    total_return = 0
    all_pass = True

    for year in years:
        r = test_year(df_15m, df_4h, year, params)

        if r is None:
            print(f"  {year}:  無數據")
            continue

        results[year] = r
        total_return += r['total_return']

        status = "OK" if r['total_return'] >= 30 else "FAIL"
        if r['total_return'] < 30:
            all_pass = False

        # 標記樣本外數據
        sample_type = "樣本外" if year <= 2022 else "樣本內"

        print(f"  {year} ({sample_type}): {r['total_return']:6.2f}% | {r['total_trades']:3d} 筆 | "
              f"{r['win_rate']:5.1f}% | {r['max_drawdown']:5.1f}% | {r['profit_factor']:4.2f} | [{status}]")
        sys.stdout.flush()

    print("-"*70)

    # 分類統計
    in_sample_return = sum(results[y]['total_return'] for y in [2023, 2024, 2025] if y in results)
    out_sample_return = sum(results[y]['total_return'] for y in [2020, 2021, 2022] if y in results)

    print(f"\n樣本內 (2023-2025) 總報酬: {in_sample_return:.2f}%")
    print(f"樣本外 (2020-2022) 總報酬: {out_sample_return:.2f}%")
    print(f"全部年份總報酬: {total_return:.2f}%")

    if all_pass:
        print("\n✅ ALL PASS - 所有年份都達到 30% 以上！")
    else:
        failed_years = [y for y in years if y in results and results[y]['total_return'] < 30]
        print(f"\n⚠️ 未達標年份: {failed_years}")

    sys.stdout.flush()


if __name__ == "__main__":
    main()
