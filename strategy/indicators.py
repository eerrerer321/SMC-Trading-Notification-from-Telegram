# strategy/indicators.py
# SMC (Smart Money Concepts) 核心指標實現
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict

class SMCIndicators:
    """SMC 指標計算類"""

    def __init__(self, params: dict):
        self.params = params

    # ============ Market Structure - 市場結構 ============
    def detect_swing_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測 Swing Highs 和 Swing Lows

        Swing High: 中間的高點比左右都高
        Swing Low: 中間的低點比左右都低
        """
        df = df.copy()
        strength = self.params['swing_strength']

        # 初始化
        df['swing_high'] = np.nan
        df['swing_low'] = np.nan

        for i in range(strength, len(df) - strength):
            # Swing High: 中間高點比前後都高
            is_swing_high = True
            for j in range(1, strength + 1):
                if df['h'].iloc[i] <= df['h'].iloc[i - j] or \
                   df['h'].iloc[i] <= df['h'].iloc[i + j]:
                    is_swing_high = False
                    break

            if is_swing_high:
                df.loc[df.index[i], 'swing_high'] = df['h'].iloc[i]

            # Swing Low: 中間低點比前後都低
            is_swing_low = True
            for j in range(1, strength + 1):
                if df['l'].iloc[i] >= df['l'].iloc[i - j] or \
                   df['l'].iloc[i] >= df['l'].iloc[i + j]:
                    is_swing_low = False
                    break

            if is_swing_low:
                df.loc[df.index[i], 'swing_low'] = df['l'].iloc[i]

        return df

    def identify_market_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        識別市場結構：HH, HL, LH, LL

        上升趨勢: Higher Highs (HH) + Higher Lows (HL)
        下降趨勢: Lower Highs (LH) + Lower Lows (LL)
        """
        df = df.copy()
        df = self.detect_swing_points(df)

        # 初始化
        df['structure'] = 'ranging'  # 'bullish', 'bearish', 'ranging'
        df['hh'] = False  # Higher High
        df['hl'] = False  # Higher Low
        df['lh'] = False  # Lower High
        df['ll'] = False  # Lower Low

        # 獲取所有 swing points
        swing_highs = df[df['swing_high'].notna()].copy()
        swing_lows = df[df['swing_low'].notna()].copy()

        # 識別 HH/LH
        for i in range(1, len(swing_highs)):
            prev_high = swing_highs['swing_high'].iloc[i - 1]
            curr_high = swing_highs['swing_high'].iloc[i]
            idx = swing_highs.index[i]

            if curr_high > prev_high:
                df.loc[idx, 'hh'] = True
            else:
                df.loc[idx, 'lh'] = True

        # 識別 HL/LL
        for i in range(1, len(swing_lows)):
            prev_low = swing_lows['swing_low'].iloc[i - 1]
            curr_low = swing_lows['swing_low'].iloc[i]
            idx = swing_lows.index[i]

            if curr_low > prev_low:
                df.loc[idx, 'hl'] = True
            else:
                df.loc[idx, 'll'] = True

        # 判斷趨勢（用滾動窗口）
        lookback = self.params['swing_lookback']
        for i in range(lookback, len(df)):
            recent = df.iloc[i - lookback:i + 1]

            hh_count = recent['hh'].sum()
            hl_count = recent['hl'].sum()
            lh_count = recent['lh'].sum()
            ll_count = recent['ll'].sum()

            # 上升趨勢：有 HH 和 HL
            if hh_count > 0 and hl_count > 0 and hh_count >= lh_count:
                df.loc[df.index[i], 'structure'] = 'bullish'
            # 下降趨勢：有 LH 和 LL
            elif lh_count > 0 and ll_count > 0 and ll_count >= hl_count:
                df.loc[df.index[i], 'structure'] = 'bearish'
            else:
                df.loc[df.index[i], 'structure'] = 'ranging'

        return df

    # ============ Order Blocks - 訂單區塊 ============
    def find_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        找出 Order Blocks（大資金交易密集區）

        Bullish OB: 最後一根下跌 K 線（在反轉上漲之前）
        Bearish OB: 最後一根上漲 K 線（在反轉下跌之前）
        """
        df = df.copy()
        df = self.detect_swing_points(df)

        # 初始化
        df['bullish_ob_top'] = np.nan
        df['bullish_ob_bottom'] = np.nan
        df['bearish_ob_top'] = np.nan
        df['bearish_ob_bottom'] = np.nan
        df['ob_age'] = 0  # OB 年齡（幾根 K 線前形成的）

        lookback = self.params['ob_lookback']

        for i in range(lookback, len(df)):
            # Bullish Order Block: 在 swing low 之前的最後一根下跌 K 線
            if not pd.isna(df['swing_low'].iloc[i]):
                # 向前找最後一根下跌 K 線
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df['c'].iloc[j] < df['o'].iloc[j]:  # 下跌 K 線
                        df.loc[df.index[i], 'bullish_ob_top'] = df['h'].iloc[j]
                        df.loc[df.index[i], 'bullish_ob_bottom'] = df['l'].iloc[j]
                        break

            # Bearish Order Block: 在 swing high 之前的最後一根上漲 K 線
            if not pd.isna(df['swing_high'].iloc[i]):
                # 向前找最後一根上漲 K 線
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df['c'].iloc[j] > df['o'].iloc[j]:  # 上漲 K 線
                        df.loc[df.index[i], 'bearish_ob_top'] = df['h'].iloc[j]
                        df.loc[df.index[i], 'bearish_ob_bottom'] = df['l'].iloc[j]
                        break

        # 向前填充 OB（OB 在未被突破前一直有效）
        max_age = self.params['ob_max_age']

        # Bullish OB 向前填充
        last_bullish_ob_top = np.nan
        last_bullish_ob_bottom = np.nan
        age = 0

        for i in range(len(df)):
            # 如果有新的 Bullish OB
            if not pd.isna(df['bullish_ob_top'].iloc[i]):
                last_bullish_ob_top = df['bullish_ob_top'].iloc[i]
                last_bullish_ob_bottom = df['bullish_ob_bottom'].iloc[i]
                age = 0
            # 如果 OB 被突破（價格跌破 OB 底部）
            elif not pd.isna(last_bullish_ob_bottom) and df['c'].iloc[i] < last_bullish_ob_bottom:
                last_bullish_ob_top = np.nan
                last_bullish_ob_bottom = np.nan
                age = 0
            # OB 太老了
            elif age > max_age:
                last_bullish_ob_top = np.nan
                last_bullish_ob_bottom = np.nan
                age = 0
            else:
                # 填充有效的 OB
                if not pd.isna(last_bullish_ob_top):
                    df.loc[df.index[i], 'bullish_ob_top'] = last_bullish_ob_top
                    df.loc[df.index[i], 'bullish_ob_bottom'] = last_bullish_ob_bottom
                age += 1

        # Bearish OB 向前填充（同理）
        last_bearish_ob_top = np.nan
        last_bearish_ob_bottom = np.nan
        age = 0

        for i in range(len(df)):
            if not pd.isna(df['bearish_ob_top'].iloc[i]):
                last_bearish_ob_top = df['bearish_ob_top'].iloc[i]
                last_bearish_ob_bottom = df['bearish_ob_bottom'].iloc[i]
                age = 0
            elif not pd.isna(last_bearish_ob_top) and df['c'].iloc[i] > last_bearish_ob_top:
                last_bearish_ob_top = np.nan
                last_bearish_ob_bottom = np.nan
                age = 0
            elif age > max_age:
                last_bearish_ob_top = np.nan
                last_bearish_ob_bottom = np.nan
                age = 0
            else:
                if not pd.isna(last_bearish_ob_top):
                    df.loc[df.index[i], 'bearish_ob_top'] = last_bearish_ob_top
                    df.loc[df.index[i], 'bearish_ob_bottom'] = last_bearish_ob_bottom
                age += 1

        return df

    # ============ BOS/CHoCH - 結構突破 ============
    def detect_bos_choch(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        檢測 BOS (Break of Structure) 和 CHoCH (Change of Character)

        BOS: 突破前高/低，趨勢延續
        CHoCH: 反向突破，趨勢反轉
        """
        df = df.copy()
        df = self.detect_swing_points(df)

        # 初始化
        df['bos'] = False
        df['choch'] = False
        df['bos_direction'] = ''  # 'bullish' | 'bearish'

        buffer = self.params['structure_buffer']

        # 獲取所有 swing points
        swing_highs = df[df['swing_high'].notna()].copy()
        swing_lows = df[df['swing_low'].notna()].copy()

        # 追蹤最近的 swing high/low
        last_swing_high = None
        last_swing_low = None
        current_trend = 'ranging'

        for i in range(len(df)):
            current_close = df['c'].iloc[i]

            # 更新最近的 swing points
            if not pd.isna(df['swing_high'].iloc[i]):
                last_swing_high = df['swing_high'].iloc[i]

            if not pd.isna(df['swing_low'].iloc[i]):
                last_swing_low = df['swing_low'].iloc[i]

            # 檢測突破
            if last_swing_high is not None and last_swing_low is not None:
                # Bullish BOS: 上升趨勢中突破前高
                if current_trend in ['bullish', 'ranging'] and \
                   current_close > last_swing_high * (1 + buffer):
                    df.loc[df.index[i], 'bos'] = True
                    df.loc[df.index[i], 'bos_direction'] = 'bullish'

                    if current_trend == 'bearish':
                        df.loc[df.index[i], 'choch'] = True

                    current_trend = 'bullish'

                # Bearish BOS: 下降趨勢中突破前低
                elif current_trend in ['bearish', 'ranging'] and \
                     current_close < last_swing_low * (1 - buffer):
                    df.loc[df.index[i], 'bos'] = True
                    df.loc[df.index[i], 'bos_direction'] = 'bearish'

                    if current_trend == 'bullish':
                        df.loc[df.index[i], 'choch'] = True

                    current_trend = 'bearish'

        return df

    # ============ Fair Value Gap (FVG) - 可選 ============
    def find_fvg(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        找出 Fair Value Gaps（公允價值缺口）

        三根 K 線之間的 imbalance（缺口）
        """
        df = df.copy()

        # 初始化
        df['bullish_fvg_top'] = np.nan
        df['bullish_fvg_bottom'] = np.nan
        df['bearish_fvg_top'] = np.nan
        df['bearish_fvg_bottom'] = np.nan

        min_size = self.params['fvg_min_size']

        for i in range(2, len(df)):
            # Bullish FVG: K 線 1 的高點 < K 線 3 的低點
            gap = df['l'].iloc[i] - df['h'].iloc[i - 2]
            if gap > 0 and gap / df['c'].iloc[i] > min_size:
                df.loc[df.index[i], 'bullish_fvg_top'] = df['l'].iloc[i]
                df.loc[df.index[i], 'bullish_fvg_bottom'] = df['h'].iloc[i - 2]

            # Bearish FVG: K 線 1 的低點 > K 線 3 的高點
            gap = df['l'].iloc[i - 2] - df['h'].iloc[i]
            if gap > 0 and gap / df['c'].iloc[i] > min_size:
                df.loc[df.index[i], 'bearish_fvg_top'] = df['l'].iloc[i - 2]
                df.loc[df.index[i], 'bearish_fvg_bottom'] = df['h'].iloc[i]

        return df

    # ============ 整合所有 SMC 指標 ============
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """計算所有 SMC 指標"""
        print("計算 SMC 指標...")
        print("  - Market Structure (Swing Points, HH/HL/LH/LL)")
        df = self.identify_market_structure(df)

        print("  - Order Blocks (Bullish/Bearish OB)")
        df = self.find_order_blocks(df)

        print("  - BOS/CHoCH (結構突破)")
        df = self.detect_bos_choch(df)

        if self.params['fvg_enabled']:
            print("  - Fair Value Gaps (FVG)")
            df = self.find_fvg(df)

        print("[OK] SMC 指標計算完成！")
        return df


# ============ 測試用 ============
if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import SMC_PARAMS, DATA_PATH

    # 讀取數據
    df = pd.read_csv(DATA_PATH, index_col='ts', parse_dates=True)
    print(f"讀取數據: {len(df)} 根 K 線")

    # 聚合為 4 小時
    df_4h = df.resample('4H').agg({
        'o': 'first',
        'h': 'max',
        'l': 'min',
        'c': 'last',
        'v': 'sum'
    }).dropna()
    print(f"聚合為 4 小時: {len(df_4h)} 根 K 線")

    # 計算 SMC 指標
    smc = SMCIndicators(SMC_PARAMS)
    df_4h = smc.calculate_all(df_4h)

    # 統計
    print("\n" + "="*80)
    print("SMC 指標統計")
    print("="*80)
    print(f"Swing Highs: {df_4h['swing_high'].notna().sum()}")
    print(f"Swing Lows: {df_4h['swing_low'].notna().sum()}")
    print(f"Bullish OBs: {df_4h['bullish_ob_top'].notna().sum()}")
    print(f"Bearish OBs: {df_4h['bearish_ob_top'].notna().sum()}")
    print(f"BOS: {df_4h['bos'].sum()}")
    print(f"CHoCH: {df_4h['choch'].sum()}")

    # 顯示最近的 SMC 信號
    print("\n最近 10 根 K 線的 SMC 信號:")
    cols = ['c', 'structure', 'bos', 'bos_direction']
    print(df_4h[cols].tail(10))
