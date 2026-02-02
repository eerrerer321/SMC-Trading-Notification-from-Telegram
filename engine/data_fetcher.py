# engine/data_fetcher.py
# 實時數據獲取模組
# -*- coding: utf-8 -*-

import ccxt
import pandas as pd
import time
from typing import Optional, Tuple
from datetime import datetime, timedelta

class LiveDataFetcher:
    """实时数据获取器"""

    def __init__(self, exchange_name: str, api_key: str = None,
                 api_secret: str = None):
        """
        初始化交易所连接

        Args:
            exchange_name: 'binance' | 'bybit'
            api_key: API 密钥（可选，只查询数据不需要）
            api_secret: API 密钥（可选）
        """
        self.exchange_name = exchange_name

        # 初始化交易所
        if exchange_name == 'binance':
            self.exchange = ccxt.binance({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}  # 使用合约市场
            })
        elif exchange_name == 'bybit':
            self.exchange = ccxt.bybit({
                'apiKey': api_key,
                'secret': api_secret,
                'enableRateLimit': True,
                'options': {'defaultType': 'swap'}
            })
        else:
            raise ValueError(f"不支持的交易所: {exchange_name}")

        print(f"✅ 已連接到 {exchange_name.upper()}")

    def fetch_historical_data(self, symbol: str, timeframe: str,
                             limit: int = 2000) -> pd.DataFrame:
        """
        获取历史K线数据（带重试机制）

        Args:
            symbol: 交易对 (如 'ETHUSDT')
            timeframe: 时间周期 (如 '15m', '1h', '4h')
            limit: 获取数量

        Returns:
            DataFrame with columns: ['ts', 'o', 'h', 'l', 'c', 'v']
        """
        max_retries = 3
        retry_delay = 2

        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    limit=limit
                )

                df = pd.DataFrame(
                    ohlcv,
                    columns=['ts', 'o', 'h', 'l', 'c', 'v']
                )

                # 转换时间戳为 datetime
                df['ts'] = pd.to_datetime(df['ts'], unit='ms')
                df.set_index('ts', inplace=True)

                return df

            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️ 獲取數據失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                    print(f"   {wait_time}秒後重試...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ 獲取數據失敗 (已重試{max_retries}次): {e}")
                    return None

        return None

    def aggregate_to_4h(self, df_15m: pd.DataFrame) -> pd.DataFrame:
        """
        将 15 分钟数据聚合为 4 小时

        Args:
            df_15m: 15分钟数据

        Returns:
            4小时数据
        """
        df_4h = df_15m.resample('4H').agg({
            'o': 'first',
            'h': 'max',
            'l': 'min',
            'c': 'last',
            'v': 'sum'
        }).dropna()

        return df_4h

    def fetch_historical_data_extended(self, symbol: str, timeframe: str,
                                        days: int) -> pd.DataFrame:
        """
        分批獲取長期歷史K線數據（突破單次 API 限制）

        Args:
            symbol: 交易對 (如 'ETHUSDT')
            timeframe: 時間周期 (如 '15m', '1h', '4h')
            days: 需要獲取的天數

        Returns:
            DataFrame with columns: ['ts', 'o', 'h', 'l', 'c', 'v']
        """
        # 計算時間周期的毫秒數
        timeframe_ms = {
            '1m': 60 * 1000,
            '5m': 5 * 60 * 1000,
            '15m': 15 * 60 * 1000,
            '30m': 30 * 60 * 1000,
            '1h': 60 * 60 * 1000,
            '4h': 4 * 60 * 60 * 1000,
            '1d': 24 * 60 * 60 * 1000,
        }

        if timeframe not in timeframe_ms:
            print(f"⚠️ 不支援的時間周期: {timeframe}")
            return None

        tf_ms = timeframe_ms[timeframe]
        limit_per_request = 1500  # 每次請求的 K 線數量（保守值）

        # 計算需要的總 K 線數量
        total_candles_needed = int((days * 24 * 60 * 60 * 1000) / tf_ms) + 100  # 多獲取一些

        # 計算需要分幾次請求
        num_requests = (total_candles_needed + limit_per_request - 1) // limit_per_request

        print(f"  需要獲取約 {total_candles_needed} 根 {timeframe} K線")
        print(f"  將分 {num_requests} 次請求獲取...")

        all_data = []
        end_time = int(datetime.now().timestamp() * 1000)  # 當前時間（毫秒）

        for i in range(num_requests):
            try:
                # 計算本次請求的結束時間
                since = end_time - (limit_per_request * tf_ms)

                ohlcv = self.exchange.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit_per_request
                )

                if ohlcv:
                    all_data = ohlcv + all_data  # 將新資料放在前面
                    # 更新結束時間為本次資料的最早時間
                    end_time = ohlcv[0][0] - tf_ms

                    if i < num_requests - 1:
                        print(f"    批次 {i + 1}/{num_requests}: 獲取 {len(ohlcv)} 根 K線")
                        time.sleep(0.5)  # 避免 API 限流

                # 檢查是否已經獲取足夠的資料
                if len(all_data) >= total_candles_needed:
                    break

            except Exception as e:
                print(f"⚠️ 批次 {i + 1} 獲取失敗: {e}")
                time.sleep(2)
                continue

        if not all_data:
            return None

        # 轉換為 DataFrame
        df = pd.DataFrame(
            all_data,
            columns=['ts', 'o', 'h', 'l', 'c', 'v']
        )

        # 移除重複的時間戳
        df = df.drop_duplicates(subset=['ts'], keep='last')

        # 轉換時間戳為 datetime
        df['ts'] = pd.to_datetime(df['ts'], unit='ms')
        df.set_index('ts', inplace=True)
        df.sort_index(inplace=True)

        # 過濾到指定時間範圍
        cutoff_time = datetime.now() - timedelta(days=days)
        df = df[df.index >= cutoff_time]

        print(f"  ✅ 最終獲取 {len(df)} 根 {timeframe} K線")
        print(f"     時間範圍: {df.index[0]} ~ {df.index[-1]}")

        return df

    def get_latest_kline_data(self, symbol: str, base_timeframe: str = '15m',
                             trading_timeframe: str = '4h',
                             lookback: int = 2000) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        获取最新的K线数据（包括基础周期和交易周期）

        Args:
            symbol: 交易对
            base_timeframe: 基础时间周期
            trading_timeframe: 交易时间周期
            lookback: 回看数量

        Returns:
            (df_base, df_trading) - 基础周期数据和交易周期数据
        """
        # 获取基础周期数据
        df_base = self.fetch_historical_data(symbol, base_timeframe, lookback)

        if df_base is None or len(df_base) == 0:
            return None, None

        # 聚合为交易周期
        if base_timeframe == '15m' and trading_timeframe == '4h':
            df_trading = self.aggregate_to_4h(df_base)
        else:
            # 如果交易周期与基础周期相同，直接返回
            if base_timeframe == trading_timeframe:
                df_trading = df_base.copy()
            else:
                # 否则直接从交易所获取交易周期数据
                df_trading = self.fetch_historical_data(
                    symbol, trading_timeframe, limit=500
                )

        return df_base, df_trading

    def get_current_price(self, symbol: str) -> Optional[float]:
        """获取当前价格（带重试机制）"""
        max_retries = 3
        retry_delay = 1  # 价格查询失败用较短延迟

        for attempt in range(max_retries):
            try:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    time.sleep(wait_time)
                else:
                    print(f"❌ 獲取價格失敗 (已重試{max_retries}次): {e}")
                    return None

        return None

    def check_new_candle(self, symbol: str, timeframe: str,
                        last_candle_time: Optional[datetime]) -> bool:
        """
        检查是否有新的K线生成（带重试机制）

        Args:
            symbol: 交易对
            timeframe: 时间周期
            last_candle_time: 上次检查的K线时间

        Returns:
            是否有新K线
        """
        max_retries = 3
        retry_delay = 2  # 初始延迟2秒

        for attempt in range(max_retries):
            try:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=2)

                if len(ohlcv) == 0:
                    return False

                # 获取最新完成的K线时间（倒数第二根）
                latest_closed_candle_time = pd.to_datetime(ohlcv[-2][0], unit='ms')

                # 如果是第一次检查
                if last_candle_time is None:
                    return True

                # 比较时间
                return latest_closed_candle_time > last_candle_time

            except Exception as e:
                if attempt < max_retries - 1:
                    # 指数退避重试
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"⚠️ 檢查新K線失敗 (嘗試 {attempt + 1}/{max_retries}): {e}")
                    print(f"   {wait_time}秒後重試...")
                    time.sleep(wait_time)
                else:
                    # 最后一次尝试失败
                    print(f"❌ 檢查新K線失敗 (已重試{max_retries}次): {e}")
                    print(f"   可能原因: 網路連接問題或API服務臨時中斷")
                    return False

        return False

    def get_balance(self, currency: str = 'USDT') -> Optional[float]:
        """
        获取账户余额

        Args:
            currency: 币种

        Returns:
            可用余额
        """
        try:
            balance = self.exchange.fetch_balance()
            return float(balance['free'].get(currency, 0))
        except Exception as e:
            print(f"❌ 獲取餘額失敗: {e}")
            return None


# ============ 測試用 ============
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import EXCHANGE

    print("測試實時數據獲取...")

    fetcher = LiveDataFetcher(
        exchange_name=EXCHANGE,
        api_key=None,
        api_secret=None
    )

    symbol = "ETHUSDT"
    print(f"\n獲取 {symbol} 15 分鐘數據...")

    df_15m, df_4h = fetcher.get_latest_kline_data(
        symbol=symbol,
        base_timeframe='15m',
        trading_timeframe='4h',
        lookback=500
    )

    if df_15m is not None and df_4h is not None:
        print(f"✅ 15 分鐘數據: {len(df_15m)} 根 K 線")
        print(f"✅ 4 小時數據: {len(df_4h)} 根 K 線")
        print(f"\n最新 4 小時 K 線:")
        print(df_4h.tail(3))

        price = fetcher.get_current_price(symbol)
        if price:
            print(f"\n✅ 當前價格: ${price:,.2f}")

    else:
        print("❌ 數據獲取失敗！")
