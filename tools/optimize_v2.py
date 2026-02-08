# optimize_v2.py
# 高效兩階段參數優化 - 多進程並行 + 樣本外驗證
# -*- coding: utf-8 -*-

import sys
import os
import io
import copy
import time
import pickle
import tempfile
import pandas as pd
import numpy as np
import multiprocessing as mp

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal

# ============ 多進程 Worker 全域變數 ============
g_cache = None
g_years = None
g_base_params = None


def worker_init(cache_path, years, base_params):
    """Worker 進程初始化：載入預計算快取"""
    global g_cache, g_years, g_base_params
    with open(cache_path, 'rb') as f:
        g_cache = pickle.load(f)
    g_years = years
    g_base_params = base_params


def test_year_fast(cache, year, params):
    """快速測試 - 使用預計算快取，只重跑 OB 識別和信號生成"""
    if year not in cache:
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0,
                'max_drawdown': 0, 'profit_factor': 0}

    df_15m_y = cache[year]['df_15m']
    df_4h_y = cache[year]['df_4h_base'].copy()

    strategy = SMCStrategyFinal(params)

    import contextlib
    import io as _io
    with contextlib.redirect_stdout(_io.StringIO()):
        df_4h_y = strategy.identify_high_quality_ob_4h(df_4h_y)
        df_1h = strategy.generate_signals_mtf(df_15m_y, df_4h_y)

    # 模擬交易
    trades = []
    enable_be = params.get('enable_breakeven', True)
    be_trigger = params.get('breakeven_trigger_r', 1.5)
    be_pct = params.get('breakeven_profit_pct', 0.005)

    signals = df_1h[df_1h['signal'] != 0]
    future_data = df_1h

    for idx, row in signals.iterrows():
        direction = 'long' if row['signal'] == 1 else 'short'
        entry = row['entry_price']
        sl = row['stop_loss']
        tp = row['take_profit']
        orig_sl = sl
        be_moved = False

        for c_idx, c in future_data[future_data.index > idx].iterrows():
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
        return {'total_return': 0, 'total_trades': 0, 'win_rate': 0,
                'max_drawdown': 0, 'profit_factor': 0}

    equity = 1.0
    peak = 1.0
    max_dd = 0
    total_profit = 0
    total_loss = 0
    wins = 0

    for pnl in trades:
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


def worker_eval(config):
    """Worker: 評估一組參數"""
    global g_cache, g_years, g_base_params
    params = {**g_base_params, **config}
    results = {}
    for year in g_years:
        results[year] = test_year_fast(g_cache, year, params)

    total = sum(r['total_return'] for r in results.values())
    min_ret = min(r['total_return'] for r in results.values())
    neg_years = sum(1 for r in results.values() if r['total_return'] < 0)
    pf_list = [r['profit_factor'] for r in results.values() if r['profit_factor'] > 0]
    avg_pf = np.mean(pf_list) if pf_list else 0

    score = (total * 0.3) + (min_ret * 5.0) + (avg_pf * 20) - (neg_years * 30)
    return config, results, score, total, min_ret


def evaluate_sequential(cache, params, years):
    """單進程評估（用於驗證）"""
    results = {}
    for year in years:
        results[year] = test_year_fast(cache, year, params)
    total = sum(r['total_return'] for r in results.values())
    min_ret = min(r['total_return'] for r in results.values())
    neg_years = sum(1 for r in results.values() if r['total_return'] < 0)
    pf_list = [r['profit_factor'] for r in results.values() if r['profit_factor'] > 0]
    avg_pf = np.mean(pf_list) if pf_list else 0
    score = (total * 0.3) + (min_ret * 5.0) + (avg_pf * 20) - (neg_years * 30)
    return results, score, total, min_ret


def load_data(files):
    """載入指定 CSV 檔案"""
    data_dir = os.path.join(PROJECT_ROOT, 'data')
    dfs = []
    for f in files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            dfs.append(pd.read_csv(path, index_col='ts', parse_dates=True))
        else:
            print(f"  [警告] 找不到: {f}")
    if not dfs:
        return None, None
    df_15m = pd.concat(dfs).sort_index()
    df_15m = df_15m[~df_15m.index.duplicated(keep='first')]
    df_4h = df_15m.resample('4H').agg({
        'o': 'first', 'h': 'max', 'l': 'min', 'c': 'last', 'v': 'sum'
    }).dropna()
    return df_15m, df_4h


def precompute_per_year(df_15m, df_4h, years):
    """預計算每年的 SMC 指標（不依賴優化參數，只算一次）"""
    smc = SMCIndicators(SMC_PARAMS)
    base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}
    strategy = SMCStrategyFinal(base_params)

    cache = {}
    import contextlib
    import io as _io

    for year in years:
        df_15m_y = df_15m[df_15m.index.year == year].copy()
        df_4h_y = df_4h[df_4h.index.year == year].copy()
        if len(df_15m_y) < 100:
            continue

        with contextlib.redirect_stdout(_io.StringIO()):
            df_4h_y = smc.calculate_all(df_4h_y)
            df_4h_y = strategy.identify_key_structure_points(df_4h_y,
                        min_move_pct=base_params.get('min_structure_move_pct', 0.02))

        cache[year] = {
            'df_15m': df_15m_y,
            'df_4h_base': df_4h_y,
        }
        print(f"  {year}: 15m={len(df_15m_y)}, 4h={len(df_4h_y)} (已快取)")
    return cache


def run_parallel(configs, cache, years, base_params, phase_name, num_workers=None):
    """多進程並行評估"""
    if num_workers is None:
        num_workers = max(1, mp.cpu_count() - 1)

    print(f"{phase_name}: {len(configs)} 組合, {num_workers} 進程並行")
    sys.stdout.flush()

    # 儲存快取到臨時檔案
    cache_path = os.path.join(tempfile.gettempdir(), 'smc_opt_cache.pkl')
    with open(cache_path, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  快取已存入: {cache_path} ({os.path.getsize(cache_path) / 1024 / 1024:.1f} MB)")
    sys.stdout.flush()

    start_time = time.time()

    try:
        with mp.Pool(processes=num_workers,
                      initializer=worker_init,
                      initargs=(cache_path, years, base_params)) as pool:

            all_results = []
            best_score = -float('inf')
            done = 0

            # 使用 imap_unordered 讓結果即時回傳
            for config, results, score, total, min_ret in pool.imap_unordered(worker_eval, configs, chunksize=10):
                all_results.append((config, results, score, total, min_ret))
                done += 1

                if score > best_score:
                    best_score = score
                    print(f"\n  [{done}/{len(configs)}] 新最佳 (score={score:.1f}):")
                    for y in years:
                        r = results[y]
                        print(f"    {y}: {r['total_return']:7.2f}% | {r['total_trades']:3d}筆 | "
                              f"WR:{r['win_rate']:4.1f}% | MDD:{r['max_drawdown']:5.1f}% | PF:{r['profit_factor']:.2f}")
                    print(f"    總: {total:.2f}% | 最低: {min_ret:.2f}%")
                    sys.stdout.flush()

                if done % 200 == 0:
                    elapsed = time.time() - start_time
                    rate = done / elapsed
                    remaining = (len(configs) - done) / rate if rate > 0 else 0
                    print(f"  進度 {done}/{len(configs)} | "
                          f"{rate:.1f} 組/秒 | 剩餘 ~{remaining:.0f}s")
                    sys.stdout.flush()

    finally:
        # 清理臨時檔案
        if os.path.exists(cache_path):
            os.remove(cache_path)

    elapsed = time.time() - start_time
    print(f"\n{phase_name}完成！耗時 {elapsed:.0f}s ({elapsed/60:.1f}min)")
    sys.stdout.flush()

    return all_results


def print_results_table(results, years, label=""):
    """印出結果表格"""
    if label:
        print(f"\n{label}")
    print(f"  {'年份':<8} {'報酬率':>8} {'交易數':>6} {'勝率':>6} {'MDD':>6} {'PF':>6}")
    print("  " + "-" * 50)
    for y in years:
        r = results[y]
        print(f"  {y}: {r['total_return']:7.2f}% | {r['total_trades']:3d}筆 | "
              f"{r['win_rate']:5.1f}% | {r['max_drawdown']:5.1f}% | {r['profit_factor']:.2f}")
    total = sum(r['total_return'] for r in results.values())
    avg_wr = np.mean([r['win_rate'] for r in results.values()])
    avg_pf = np.mean([r['profit_factor'] for r in results.values() if r['profit_factor'] > 0]) if any(r['profit_factor'] > 0 for r in results.values()) else 0
    print(f"  {'合計':>6}: {total:7.2f}% | avg WR: {avg_wr:.1f}% | avg PF: {avg_pf:.2f}")


def main():
    print("=" * 70)
    print("高效參數優化（多進程並行 + 樣本外驗證）")
    print("=" * 70)
    print(f"CPU 核心數: {mp.cpu_count()}")
    sys.stdout.flush()

    # ========== 載入訓練資料 (2023-2025) ==========
    print("\n載入訓練資料 (2023-2025)...")
    df_15m_train, df_4h_train = load_data(['ETHUSDT_15m_2023_now.csv'])
    if df_15m_train is None:
        print("錯誤：無法載入訓練資料")
        return

    train_years = sorted(set(df_15m_train.index.year))
    train_years = [y for y in train_years if len(df_15m_train[df_15m_train.index.year == y]) > 100]
    print(f"訓練年份: {train_years}")
    print(f"資料範圍: {df_15m_train.index[0]} ~ {df_15m_train.index[-1]}")
    sys.stdout.flush()

    # 預計算訓練資料
    print("\n預計算 SMC 指標（每年只算一次）...")
    train_cache = precompute_per_year(df_15m_train, df_4h_train, train_years)
    print("預計算完成！")
    sys.stdout.flush()

    base_params = {**TRADING_PARAMS, **STRATEGY_PARAMS}

    # ========== 第一階段：核心參數搜索 ==========
    print("\n" + "=" * 70)
    print("第一階段：核心參數搜索（訓練集 2023-2025）")
    print("=" * 70)
    sys.stdout.flush()

    phase1_configs = []
    for rr in [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]:
        for trend in [True, False]:
            for struct in [True, False]:
                for ob in [35, 40, 45, 50, 55, 60]:
                    for min_pb in [0.15, 0.20, 0.25, 0.30, 0.35]:
                        for neutral in [True, False]:
                            phase1_configs.append({
                                'risk_reward_ratio': rr,
                                'enable_trend_filter': trend,
                                'enable_structure_filter': struct,
                                'ob_quality_threshold': ob,
                                'min_pullback_pct': min_pb,
                                'allow_neutral_market': neutral,
                            })

    phase1_results = run_parallel(
        phase1_configs, train_cache, train_years, base_params,
        "第一階段"
    )

    # 取前 5 名
    phase1_results.sort(key=lambda x: x[2], reverse=True)
    top_n = 5
    top_configs = phase1_results[:top_n]

    print(f"\n第一階段前 {top_n} 名:")
    for rank, (config, results, score, total, min_ret) in enumerate(top_configs):
        print(f"  #{rank+1}: 總={total:7.2f}% | 最低={min_ret:7.2f}% | score={score:.1f}")
        changed = {k: v for k, v in config.items() if v != base_params.get(k)}
        for k, v in changed.items():
            print(f"      {k}: {v}")
    sys.stdout.flush()

    # ========== 第二階段：風控微調 ==========
    print("\n" + "=" * 70)
    print("第二階段：風控參數微調（訓練集 2023-2025）")
    print("=" * 70)
    sys.stdout.flush()

    phase2_configs = []
    for base_config, _, _, _, _ in top_configs:
        for be_trigger in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5]:
            for be_pct in [0.002, 0.003, 0.005, 0.008, 0.01]:
                for body_ratio in [0.4, 0.5, 0.6, 0.7]:
                    for vol_thresh in [0.5, 0.6, 0.7]:
                        config = copy.deepcopy(base_config)
                        config['breakeven_trigger_r'] = be_trigger
                        config['breakeven_profit_pct'] = be_pct
                        config['entry_candle_body_ratio'] = body_ratio
                        config['entry_volume_threshold'] = vol_thresh
                        phase2_configs.append(config)

    # 也測試關閉保本
    for base_config, _, _, _, _ in top_configs:
        config = copy.deepcopy(base_config)
        config['enable_breakeven'] = False
        phase2_configs.append(config)

    phase2_results = run_parallel(
        phase2_configs, train_cache, train_years, base_params,
        "第二階段"
    )

    # 找最佳結果
    phase2_results.sort(key=lambda x: x[2], reverse=True)

    if not phase2_results:
        print("未找到有效結果")
        return

    best_config, best_results, best_score, best_total, best_min_ret = phase2_results[0]

    print("\n" + "=" * 70)
    print("訓練集最佳結果（2023-2025）")
    print("=" * 70)
    print_results_table(best_results, train_years)

    # ========== 樣本外驗證 (2020-2022) ==========
    print("\n" + "=" * 70)
    print("樣本外驗證（2020-2022）- 檢查是否過擬合")
    print("=" * 70)
    sys.stdout.flush()

    print("\n載入驗證資料 (2020-2022)...")
    df_15m_val, df_4h_val = load_data([
        'ETHUSDT_15m_2020_ccxt.csv',
        'ETHUSDT_15m_2021_ccxt.csv',
        'ETHUSDT_15m_2022_ccxt.csv',
    ])

    if df_15m_val is not None:
        val_years = sorted(set(df_15m_val.index.year))
        val_years = [y for y in val_years if len(df_15m_val[df_15m_val.index.year == y]) > 100]
        print(f"驗證年份: {val_years}")

        print("預計算驗證集 SMC 指標...")
        val_cache = precompute_per_year(df_15m_val, df_4h_val, val_years)

        # 用最佳參數跑驗證集
        best_params = {**base_params, **best_config}
        val_results, val_score, val_total, val_min_ret = evaluate_sequential(
            val_cache, best_params, val_years
        )

        print_results_table(val_results, val_years, "樣本外績效 (2020-2022):")

        # 過擬合分析
        print("\n" + "-" * 50)
        print("過擬合分析:")
        train_avg_ret = best_total / len(train_years)
        val_avg_ret = val_total / len(val_years) if val_years else 0
        print(f"  訓練集年均報酬: {train_avg_ret:.2f}%")
        print(f"  驗證集年均報酬: {val_avg_ret:.2f}%")

        if val_avg_ret > 0 and train_avg_ret > 0:
            degradation = (1 - val_avg_ret / train_avg_ret) * 100
            print(f"  績效衰退: {degradation:.1f}%")
            if degradation < 30:
                print("  評估: 良好 - 策略泛化能力可接受")
            elif degradation < 50:
                print("  評估: 警告 - 有輕微過擬合跡象")
            else:
                print("  評估: 危險 - 可能嚴重過擬合")
        elif val_avg_ret <= 0:
            print("  評估: 危險 - 驗證集虧損，嚴重過擬合")

        # 也測試前 5 名在驗證集的表現
        print("\n前 5 名組合在驗證集的表現:")
        print(f"  {'排名':<4} {'訓練總報酬':>10} {'驗證總報酬':>10} {'驗證最低':>8} {'狀態':>8}")
        print("  " + "-" * 50)

        for rank, (cfg, _, s, t, m) in enumerate(phase2_results[:5]):
            p = {**base_params, **cfg}
            vr, vs, vt, vm = evaluate_sequential(val_cache, p, val_years)
            val_neg = sum(1 for r in vr.values() if r['total_return'] < 0)
            status = "OK" if vt > 0 and val_neg <= 1 else "注意"
            print(f"  #{rank+1}:  {t:9.2f}%   {vt:9.2f}%   {vm:7.2f}%   {status}")
    else:
        print("  [跳過] 無法載入 2020-2022 驗證資料")

    # ========== 最終結果 ==========
    print("\n" + "=" * 70)
    print("最終最佳參數")
    print("=" * 70)

    print("\n建議的 STRATEGY_PARAMS 參數更新:")
    changed = {k: v for k, v in sorted(best_config.items())
               if v != base_params.get(k)}

    if changed:
        for k, v in changed.items():
            old = base_params.get(k, 'N/A')
            print(f"    '{k}': {v},  # 原: {old}")
    else:
        print("  (無需變更，現有參數已是最佳)")

    print("\n所有最佳參數:")
    for k, v in sorted(best_config.items()):
        print(f"    '{k}': {v},")

    sys.stdout.flush()


if __name__ == '__main__':
    mp.freeze_support()  # Windows 必要
    main()
