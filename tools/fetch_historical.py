# fetch_historical.py
# 抓取歷史K線數據 (2020-2021)
# -*- coding: utf-8 -*-

import sys
import os
import io
import ccxt
import pandas as pd
from datetime import datetime
import time

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)


def fetch_year_data(exchange, symbol: str, year: int, timeframe: str = '15m'):
    """抓取指定年份的K線數據"""

    # 時間周期毫秒數
    tf_ms = {
        '1m': 60 * 1000,
        '5m': 5 * 60 * 1000,
        '15m': 15 * 60 * 1000,
        '1h': 60 * 60 * 1000,
        '4h': 4 * 60 * 60 * 1000,
    }[timeframe]

    # 設定時間範圍
    start_time = int(datetime(year, 1, 1, 0, 0, 0).timestamp() * 1000)
    end_time = int(datetime(year, 12, 31, 23, 59, 59).timestamp() * 1000)

    print(f"\n抓取 {year} 年數據...")
    print(f"  時間範圍: {datetime(year, 1, 1)} ~ {datetime(year, 12, 31)}")

    all_data = []
    current_time = start_time
    limit_per_request = 1500
    batch_count = 0

    while current_time < end_time:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_time,
                limit=limit_per_request
            )

            if not ohlcv:
                break

            all_data.extend(ohlcv)
            batch_count += 1

            # 更新時間為最後一根K線之後
            current_time = ohlcv[-1][0] + tf_ms

            if batch_count % 5 == 0:
                print(f"  已抓取 {len(all_data)} 根 K 線...")
                sys.stdout.flush()

            # 避免 API 限流
            time.sleep(0.3)

            # 如果返回數據少於請求數量，表示已到最新
            if len(ohlcv) < limit_per_request:
                break

        except Exception as e:
            print(f"  ⚠️ 錯誤: {e}")
            time.sleep(2)
            continue

    if not all_data:
        return None

    # 轉換為 DataFrame
    df = pd.DataFrame(all_data, columns=['ts', 'o', 'h', 'l', 'c', 'v'])
    df['ts'] = pd.to_datetime(df['ts'], unit='ms')
    df.set_index('ts', inplace=True)
    df = df.drop_duplicates()
    df.sort_index(inplace=True)

    # 過濾只保留該年份
    df = df[df.index.year == year]

    print(f"  ✅ 完成: {len(df)} 根 K 線")
    print(f"     範圍: {df.index[0]} ~ {df.index[-1]}")

    return df


def main():
    print("="*70)
    print("抓取歷史K線數據 (2020-2021)")
    print("="*70)
    sys.stdout.flush()

    # 初始化交易所
    print("\n連接 Binance...")
    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'}
    })

    symbol = "ETH/USDT"
    data_dir = os.path.join(PROJECT_ROOT, 'data')

    for year in [2020, 2021]:
        df = fetch_year_data(exchange, symbol, year, '15m')

        if df is not None and len(df) > 0:
            filename = f"ETHUSDT_15m_{year}_ccxt.csv"
            filepath = os.path.join(data_dir, filename)
            df.to_csv(filepath)
            print(f"  💾 已儲存: {filename}")
        else:
            print(f"  ❌ {year} 年數據抓取失敗")

        sys.stdout.flush()

    print("\n" + "="*70)
    print("完成！")
    print("="*70)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
