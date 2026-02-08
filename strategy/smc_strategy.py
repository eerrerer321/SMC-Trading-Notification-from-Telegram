# strategy/smc_strategy.py
# SMC 交易策略 - 多時間框架版本
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

class SMCStrategyFinal:
    """SMC 交易策略 - 最终优化版本（独立实现）"""

    def __init__(self, trading_params: dict):
        self.params = trading_params

    # ========== 基础指标计算 ==========
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 ATR"""
        high = df['h']
        low = df['l']
        close = df['c'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """计算 RSI"""
        delta = df['c'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_ema(self, df: pd.DataFrame, period: int) -> pd.Series:
        """计算 EMA"""
        return df['c'].ewm(span=period, adjust=False).mean()

    # ========== 4H: 关键结构位识别 ==========
    def identify_key_structure_points(self, df: pd.DataFrame, min_move_pct: float = 0.02) -> pd.DataFrame:
        """
        4H时间周期：识别关键的结构位（过滤小波动）
        min_move_pct: 最小结构移动百分比（默认2%）

        注意：只使用向後看的條件（已確認的歷史數據），
        不使用未來數據，消除前瞻偏差 (Look-ahead Bias)。
        - Swing High: 檢查「價格從前低漲到此高點」的幅度
        - Swing Low: 檢查「價格從前高跌到此低點」的幅度
        """
        df = df.copy()
        df['key_swing_high'] = np.nan
        df['key_swing_low'] = np.nan

        for i in range(len(df)):
            if not pd.isna(df['swing_high'].iloc[i]):
                current_high = df['swing_high'].iloc[i]

                # 只向後看：價格從前低漲到此高點的幅度
                start_idx = max(0, i - 10)
                prev_low = df['l'].iloc[start_idx:i].min() if i > 0 else current_high

                move_up = (current_high - prev_low) / prev_low if prev_low > 0 else 0

                if move_up >= min_move_pct:
                    df.loc[df.index[i], 'key_swing_high'] = current_high

            if not pd.isna(df['swing_low'].iloc[i]):
                current_low = df['swing_low'].iloc[i]

                # 只向後看：價格從前高跌到此低點的幅度
                start_idx = max(0, i - 10)
                prev_high = df['h'].iloc[start_idx:i].max() if i > 0 else current_low

                move_down = (prev_high - current_low) / prev_high if prev_high > 0 else 0

                if move_down >= min_move_pct:
                    df.loc[df.index[i], 'key_swing_low'] = current_low

        return df

    # ========== 4H: 高质量OB识别 ==========
    def identify_high_quality_ob_4h(self, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        4H时间周期：识别高质量的Order Block
        质量评分基于：成交量、反转速度、OB大小
        """
        df = df_4h.copy()
        df['hq_bullish_ob_top'] = np.nan
        df['hq_bullish_ob_bottom'] = np.nan
        df['hq_bearish_ob_top'] = np.nan
        df['hq_bearish_ob_bottom'] = np.nan
        df['ob_strength'] = np.nan

        # 计算成交量moving average
        df['volume_ma'] = df['v'].rolling(window=20).mean()

        lookback = int(self.params.get('ob_lookback', 20))
        ob_quality_threshold = float(self.params.get('ob_quality_threshold', 60))
        ob_volume_threshold = float(self.params.get('ob_volume_percentile', 0.70))
        vol_threshold_multiplier = (ob_volume_threshold / 0.70) if ob_volume_threshold > 0 else 1.0

        for i in range(lookback, len(df)):
            # === Bullish OB ===
            if not pd.isna(df['key_swing_low'].iloc[i]):
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df['c'].iloc[j] < df['o'].iloc[j]:  # 下跌K线
                        ob_candle = df.iloc[j]
                        next_candle = df.iloc[j + 1] if j + 1 < len(df) else None

                        if next_candle is None:
                            continue

                        strength = 0

                        # 成交量异常（40分）
                        vol_ma = df['volume_ma'].iloc[j]
                        if not pd.isna(vol_ma) and vol_ma > 0:
                            vol_ratio = ob_candle['v'] / vol_ma
                            if vol_ratio >= 2.0 * vol_threshold_multiplier:
                                strength += 40
                            elif vol_ratio >= 1.5 * vol_threshold_multiplier:
                                strength += 30
                            elif vol_ratio >= 1.2 * vol_threshold_multiplier:
                                strength += 20

                        # 反转迅速（30分）
                        if next_candle['c'] > next_candle['o']:
                            reversal_strength = (next_candle['c'] - next_candle['o']) / (ob_candle['h'] - ob_candle['l'])
                            if reversal_strength >= 1.0:
                                strength += 30
                            elif reversal_strength >= 0.7:
                                strength += 20
                            elif reversal_strength >= 0.5:
                                strength += 10

                        # OB大小合理（30分）
                        ob_size = ob_candle['h'] - ob_candle['l']
                        avg_range = (df['h'] - df['l']).rolling(20).mean().iloc[j]
                        if not pd.isna(avg_range) and avg_range > 0:
                            size_ratio = ob_size / avg_range
                            if 0.8 <= size_ratio <= 2.0:
                                strength += 30
                            elif 0.5 <= size_ratio <= 2.5:
                                strength += 20

                        # 只保留高质量OB
                        if strength >= ob_quality_threshold:
                            df.loc[df.index[i], 'hq_bullish_ob_top'] = ob_candle['h']
                            df.loc[df.index[i], 'hq_bullish_ob_bottom'] = ob_candle['l']
                            df.loc[df.index[i], 'ob_strength'] = strength
                        break

            # === Bearish OB ===
            if not pd.isna(df['key_swing_high'].iloc[i]):
                for j in range(i - 1, max(0, i - lookback), -1):
                    if df['c'].iloc[j] > df['o'].iloc[j]:  # 上涨K线
                        ob_candle = df.iloc[j]
                        next_candle = df.iloc[j + 1] if j + 1 < len(df) else None

                        if next_candle is None:
                            continue

                        strength = 0

                        vol_ma = df['volume_ma'].iloc[j]
                        if not pd.isna(vol_ma) and vol_ma > 0:
                            vol_ratio = ob_candle['v'] / vol_ma
                            if vol_ratio >= 2.0 * vol_threshold_multiplier:
                                strength += 40
                            elif vol_ratio >= 1.5 * vol_threshold_multiplier:
                                strength += 30
                            elif vol_ratio >= 1.2 * vol_threshold_multiplier:
                                strength += 20

                        if next_candle['c'] < next_candle['o']:
                            reversal_strength = (next_candle['o'] - next_candle['c']) / (ob_candle['h'] - ob_candle['l'])
                            if reversal_strength >= 1.0:
                                strength += 30
                            elif reversal_strength >= 0.7:
                                strength += 20
                            elif reversal_strength >= 0.5:
                                strength += 10

                        ob_size = ob_candle['h'] - ob_candle['l']
                        avg_range = (df['h'] - df['l']).rolling(20).mean().iloc[j]
                        if not pd.isna(avg_range) and avg_range > 0:
                            size_ratio = ob_size / avg_range
                            if 0.8 <= size_ratio <= 2.0:
                                strength += 30
                            elif 0.5 <= size_ratio <= 2.5:
                                strength += 20

                        if strength >= ob_quality_threshold:
                            df.loc[df.index[i], 'hq_bearish_ob_top'] = ob_candle['h']
                            df.loc[df.index[i], 'hq_bearish_ob_bottom'] = ob_candle['l']
                            df.loc[df.index[i], 'ob_strength'] = strength
                        break

        # 向前填充OB
        self._forward_fill_ob(df, 'hq_bullish_ob_top', 'hq_bullish_ob_bottom', 'bullish')
        self._forward_fill_ob(df, 'hq_bearish_ob_top', 'hq_bearish_ob_bottom', 'bearish')

        return df

    def _forward_fill_ob(self, df: pd.DataFrame, top_col: str, bottom_col: str, ob_type: str):
        """向前填充OB直到被突破"""
        last_top = np.nan
        last_bottom = np.nan
        last_strength = np.nan
        age = 0
        max_age = 50

        for i in range(len(df)):
            if not pd.isna(df[top_col].iloc[i]):
                last_top = df[top_col].iloc[i]
                last_bottom = df[bottom_col].iloc[i]
                last_strength = df['ob_strength'].iloc[i] if 'ob_strength' in df.columns else np.nan
                age = 0
            elif not pd.isna(last_top):
                # 检查是否被突破
                if ob_type == 'bullish':
                    if df['c'].iloc[i] < last_bottom:
                        last_top = np.nan
                        last_bottom = np.nan
                        last_strength = np.nan
                        age = 0
                        continue
                else:  # bearish
                    if df['c'].iloc[i] > last_top:
                        last_top = np.nan
                        last_bottom = np.nan
                        last_strength = np.nan
                        age = 0
                        continue

                # 检查年龄
                if age > max_age:
                    last_top = np.nan
                    last_bottom = np.nan
                    last_strength = np.nan
                    age = 0
                    continue

                # 填充
                df.loc[df.index[i], top_col] = last_top
                df.loc[df.index[i], bottom_col] = last_bottom
                if not pd.isna(last_strength):
                    df.loc[df.index[i], 'ob_strength'] = last_strength
                age += 1

    # ========== 1H: 进场确认 ==========
    def check_entry_confirmation_1h(self, df_1h: pd.DataFrame, idx: int, direction: str) -> bool:
        """
        1H 時間週期：進場確認

        條件：
        - K 線實體比例 ≥60%
        - 成交量 ≥平均的 70%
        - 前一根 K 線支持方向
        """
        if idx < 2:
            return False

        current = df_1h.iloc[idx]
        prev1 = df_1h.iloc[idx - 1]

        # 计算平均成交量
        avg_volume = df_1h['v'].rolling(20).mean().iloc[idx]

        body_ratio_threshold = float(self.params.get('entry_candle_body_ratio', 0.6))
        volume_threshold = float(self.params.get('entry_volume_threshold', 0.70))

        if direction == 'long':
            # 当前K线必须是强烈的阳线
            body = current['c'] - current['o']
            candle_range = current['h'] - current['l']

            if body <= 0 or candle_range == 0:
                return False

            body_ratio = body / candle_range
            if body_ratio < body_ratio_threshold:
                return False

            if not pd.isna(avg_volume) and current['v'] < avg_volume * volume_threshold:
                return False

            # 前一根K线应该显示买盘力量
            has_support = False
            if prev1['c'] > prev1['o']:  # 前一根是阳线
                has_support = True
            elif (prev1['o'] - prev1['l']) / (prev1['h'] - prev1['l']) >= 0.6:  # 锤子线
                has_support = True

            return has_support

        else:  # short
            body = current['o'] - current['c']
            candle_range = current['h'] - current['l']

            if body <= 0 or candle_range == 0:
                return False

            body_ratio = body / candle_range
            if body_ratio < 0.6:
                return False

            # 成交量確認：≥平均的 70%
            if not pd.isna(avg_volume) and current['v'] < avg_volume * 0.7:
                return False

            has_resistance = False
            if prev1['c'] < prev1['o']:  # 前一根是阴线
                has_resistance = True
            elif (prev1['h'] - prev1['o']) / (prev1['h'] - prev1['l']) >= 0.6:  # 射击之星
                has_resistance = True

            return has_resistance

    def check_ob_test_1h(self, df_1h: pd.DataFrame, idx: int, ob_top: float, ob_bottom: float, direction: str) -> bool:
        """
        1H时间周期：检查是否正确测试了4H的OB
        """
        if idx < 1:
            return False

        ob_mid = (ob_top + ob_bottom) / 2

        if direction == 'long':
            for i in range(max(0, idx - 1), idx + 1):
                low = df_1h['l'].iloc[i]
                if low <= ob_mid:
                    if df_1h['c'].iloc[i] >= ob_bottom:
                        return True

        else:  # short
            for i in range(max(0, idx - 1), idx + 1):
                high = df_1h['h'].iloc[i]
                if high >= ob_mid:
                    if df_1h['c'].iloc[i] <= ob_top:
                        return True

        return False

    def check_pullback_to_ob(self, df_1h: pd.DataFrame, idx: int, ob_top: float, ob_bottom: float, direction: str) -> bool:
        """
        確認價格是從高點/低點回撤到 OB（而非追高/殺低）

        v2.0 更新：增加「首次觸及 OB」模式，在強勢趨勢中允許順勢進場

        SMC 核心邏輯：
        - 模式 1（回撤進場）：價格先離開 OB → 回撤到 OB → 進場
        - 模式 2（首次觸及）：價格首次進入 OB 區域，且 OB 是新形成的

        Args:
            df_1h: 1H K線數據
            idx: 當前 K 線索引
            ob_top: OB 頂部價格
            ob_bottom: OB 底部價格
            direction: 'long' 或 'short'

        Returns:
            True 如果確認是有效進場機會
        """
        # 回看週期
        pullback_lookback = int(self.params.get('pullback_lookback', 20))
        min_pullback_pct = float(self.params.get('min_pullback_pct', 0.3))
        # 新增：允許首次觸及 OB 進場
        allow_first_touch = bool(self.params.get('allow_first_touch_ob', True))
        # 新增：首次觸及需要的最小 K 線動量
        first_touch_momentum = float(self.params.get('first_touch_momentum', 0.01))

        if idx < pullback_lookback:
            return False

        current_price = df_1h['c'].iloc[idx]
        current_low = df_1h['l'].iloc[idx]
        current_high = df_1h['h'].iloc[idx]
        current_open = df_1h['o'].iloc[idx]

        lookback_start = max(0, idx - pullback_lookback)
        lookback_data = df_1h.iloc[lookback_start:idx]

        if direction == 'long':
            # === 做多 ===
            recent_high = lookback_data['h'].max()

            # 模式 1：標準回撤進場
            if recent_high > ob_top:
                if current_low > ob_top:
                    return False
                total_distance = recent_high - ob_bottom
                if total_distance <= 0:
                    return False
                pullback_distance = recent_high - current_low
                pullback_ratio = pullback_distance / total_distance
                if pullback_ratio >= min_pullback_pct and current_price >= ob_bottom:
                    return True

            # 模式 2：首次觸及 OB（強勢趨勢中）
            if allow_first_touch:
                # 當前 K 線是陽線且有一定動量
                candle_momentum = (current_price - current_open) / current_open if current_open > 0 else 0
                if candle_momentum >= first_touch_momentum:
                    # 價格從下方首次進入 OB 區域
                    if current_low <= ob_top and current_price >= ob_bottom:
                        # 確認之前價格在 OB 下方
                        recent_closes = lookback_data['c'].tail(5)
                        if len(recent_closes) > 0 and (recent_closes < ob_bottom).any():
                            return True

        else:  # short
            # === 做空 ===
            recent_low = lookback_data['l'].min()

            # 模式 1：標準回撤進場
            if recent_low < ob_bottom:
                if current_high < ob_bottom:
                    return False
                total_distance = ob_top - recent_low
                if total_distance <= 0:
                    return False
                pullback_distance = current_high - recent_low
                pullback_ratio = pullback_distance / total_distance
                if pullback_ratio >= min_pullback_pct and current_price <= ob_top:
                    return True

            # 模式 2：首次觸及 OB（強勢下跌趨勢中）
            if allow_first_touch:
                # 當前 K 線是陰線且有一定動量
                candle_momentum = (current_open - current_price) / current_open if current_open > 0 else 0
                if candle_momentum >= first_touch_momentum:
                    # 價格從上方首次進入 OB 區域
                    if current_high >= ob_bottom and current_price <= ob_top:
                        # 確認之前價格在 OB 上方
                        recent_closes = lookback_data['c'].tail(5)
                        if len(recent_closes) > 0 and (recent_closes > ob_top).any():
                            return True

        return False

    # ========== 止损止盈计算 ==========
    def calculate_position_size(self, balance: float, entry_price: float, stop_loss: float) -> float:
        """计算仓位大小"""
        sizing_mode = self.params.get('position_sizing', 'risk_based')
        leverage = float(self.params.get('leverage', 1.0))

        if sizing_mode == 'fixed_percent':
            fixed_notional_pct = float(self.params.get('fixed_notional_pct', 0.10))
            notional = balance * fixed_notional_pct
            exposure = notional * leverage
            if entry_price <= 0:
                return 0
            return exposure / entry_price

        risk_amount = balance * float(self.params.get('risk_per_trade', 0.02))
        price_risk = abs(entry_price - stop_loss)

        if price_risk == 0:
            return 0

        position_size = (risk_amount * leverage) / price_risk

        return position_size

    def calculate_stop_loss_mtf(self, df_4h: pd.DataFrame, current_time: pd.Timestamp, direction: str,
                                 ob_bottom: float, ob_top: float) -> Optional[float]:
        """
        基於 4H 關鍵結構位計算止損

        參數：
        - lookback: 10
        - buffer: 5%
        """
        df_4h_before = df_4h[df_4h.index <= current_time]

        if len(df_4h_before) == 0:
            return ob_bottom * 0.995 if direction == 'long' else ob_top * 1.005

        if direction == 'long':
            # 找最近的key swing low
            lookback_rows = min(10, len(df_4h_before))
            recent_4h = df_4h_before.iloc[-lookback_rows:]

            for i in range(len(recent_4h) - 1, -1, -1):
                if not pd.isna(recent_4h['key_swing_low'].iloc[i]):
                    key_low = recent_4h['key_swing_low'].iloc[i]
                    candle_range = recent_4h['h'].iloc[i] - recent_4h['l'].iloc[i]
                    buffer = candle_range * 0.05  # 5% buffer
                    return key_low - buffer

            return ob_bottom * 0.995

        else:  # short
            lookback_rows = min(10, len(df_4h_before))
            recent_4h = df_4h_before.iloc[-lookback_rows:]

            for i in range(len(recent_4h) - 1, -1, -1):
                if not pd.isna(recent_4h['key_swing_high'].iloc[i]):
                    key_high = recent_4h['key_swing_high'].iloc[i]
                    candle_range = recent_4h['h'].iloc[i] - recent_4h['l'].iloc[i]
                    buffer = candle_range * 0.05  # 5% buffer
                    return key_high + buffer

            return ob_top * 1.005

    def calculate_take_profit(self, entry_price: float, stop_loss: float, direction: str) -> float:
        """
        計算止盈

        使用 RR 比率（預設 2.5）
        """
        risk = abs(entry_price - stop_loss)
        rr_ratio = self.params.get('risk_reward_ratio', 2.5)
        reward = risk * rr_ratio

        if direction == 'long':
            return entry_price + reward
        else:
            return entry_price - reward

    # ========== 主要交易逻辑 ==========
    def generate_signals_mtf(self, df_15m: pd.DataFrame, df_4h_with_smc: pd.DataFrame,
                             signal_lookback: Optional[int] = None) -> pd.DataFrame:
        """
        多時間框架信號生成

        特性：
        - RSI 過濾：Long <85, Short >15
        - 成交量過濾：≥平均的 70%
        - 趨勢過濾（EMA90 vs EMA200）
        - 市場結構過濾
        """
        print("\n" + "="*80)
        print("SMC 多時間框架信號生成")
        print("="*80)

        # Step 1: 聚合为1H
        print("\nStep 1: Aggregating 15m data to 1H...")
        df_1h = df_15m.resample('1H').agg({
            'o': 'first',
            'h': 'max',
            'l': 'min',
            'c': 'last',
            'v': 'sum'
        }).dropna()
        print(f"  Created {len(df_1h)} 1H candles")

        # Step 2: 计算1H辅助指标
        print("\nStep 2: Calculating 1H auxiliary indicators...")
        df_1h['rsi'] = self.calculate_rsi(df_1h, period=14)
        df_1h['atr'] = self.calculate_atr(df_1h, period=14)

        # Step 3: 从4H获取趋势和OB信息
        print("\nStep 3: Mapping 4H structure to 1H timeframe...")

        # 计算4H的EMA（用于趋势判断）
        df_4h_with_smc['ema90'] = self.calculate_ema(df_4h_with_smc, period=90)
        df_4h_with_smc['ema200'] = self.calculate_ema(df_4h_with_smc, period=200)

        # 初始化1H信号列
        df_1h['signal'] = 0
        df_1h['entry_price'] = np.nan
        df_1h['stop_loss'] = np.nan
        df_1h['take_profit'] = np.nan
        df_1h['ob_source'] = None

        # 将4H信息映射到1H
        df_1h['structure_4h'] = None
        df_1h['bullish_ob_4h_top'] = np.nan
        df_1h['bullish_ob_4h_bottom'] = np.nan
        df_1h['bearish_ob_4h_top'] = np.nan
        df_1h['bearish_ob_4h_bottom'] = np.nan
        df_1h['trend_4h'] = None

        for i in range(len(df_1h)):
            current_time_1h = df_1h.index[i]
            df_4h_at_time = df_4h_with_smc[df_4h_with_smc.index <= current_time_1h]

            if len(df_4h_at_time) == 0:
                continue

            latest_4h = df_4h_at_time.iloc[-1]

            # 映射4H信息到1H
            df_1h.loc[df_1h.index[i], 'structure_4h'] = latest_4h['structure']

            if not pd.isna(latest_4h['hq_bullish_ob_top']):
                df_1h.loc[df_1h.index[i], 'bullish_ob_4h_top'] = latest_4h['hq_bullish_ob_top']
                df_1h.loc[df_1h.index[i], 'bullish_ob_4h_bottom'] = latest_4h['hq_bullish_ob_bottom']

            if not pd.isna(latest_4h['hq_bearish_ob_top']):
                df_1h.loc[df_1h.index[i], 'bearish_ob_4h_top'] = latest_4h['hq_bearish_ob_top']
                df_1h.loc[df_1h.index[i], 'bearish_ob_4h_bottom'] = latest_4h['hq_bearish_ob_bottom']

            # 判断4H趋势
            if latest_4h['ema90'] > latest_4h['ema200']:
                df_1h.loc[df_1h.index[i], 'trend_4h'] = 'bull'
            elif latest_4h['ema90'] < latest_4h['ema200']:
                df_1h.loc[df_1h.index[i], 'trend_4h'] = 'bear'
            else:
                df_1h.loc[df_1h.index[i], 'trend_4h'] = 'neutral'

        print(f"  Mapped 4H structure to {len(df_1h)} 1H candles")

        # Step 4: 在1H上生成信号
        print("\nStep 4: Generating entry signals on 1H timeframe...")

        long_count = 0
        short_count = 0

        enable_trend_filter = bool(self.params.get('enable_trend_filter', True))
        enable_structure_filter = bool(self.params.get('enable_structure_filter', True))
        allow_neutral_market = bool(self.params.get('allow_neutral_market', True))
        enable_rsi_filter = bool(self.params.get('enable_rsi_filter', True))
        rsi_long_max = float(self.params.get('rsi_long_max', 85))
        rsi_short_min = float(self.params.get('rsi_short_min', 15))

        signal_cooldown_bars = int(self.params.get('signal_cooldown_bars', 0))
        last_signal_bar_by_ob: Dict[str, int] = {}

        # 即時監控模式：限制信號掃描範圍，只掃描最近 N 根 K 線
        # 回測模式（signal_lookback=None）：掃描所有 K 線
        if signal_lookback is not None and signal_lookback > 0:
            scan_start = max(50, len(df_1h) - signal_lookback)
            print(f"  [即時模式] 只掃描最近 {signal_lookback} 根 1H K線 (index {scan_start}~{len(df_1h)-1})")
        else:
            scan_start = 50

        for i in range(scan_start, len(df_1h)):
            current_1h = df_1h.iloc[i]
            current_time = df_1h.index[i]

            # === 做多信号 ===
            if not pd.isna(current_1h['bullish_ob_4h_top']):
                ob_top = current_1h['bullish_ob_4h_top']
                ob_bottom = current_1h['bullish_ob_4h_bottom']
                ob_source = f"4H_Bull_OB_{ob_bottom:.2f}-{ob_top:.2f}"

                if signal_cooldown_bars > 0:
                    last_i = last_signal_bar_by_ob.get(ob_source)
                    if last_i is not None and (i - last_i) < signal_cooldown_bars:
                        continue

                if enable_structure_filter:
                    if current_1h['structure_4h'] == 'bearish':
                        continue
                    if (not allow_neutral_market) and current_1h['structure_4h'] == 'ranging':
                        continue

                if enable_trend_filter:
                    if current_1h['trend_4h'] == 'bear':
                        continue
                    if (not allow_neutral_market) and current_1h['trend_4h'] == 'neutral':
                        continue

                # 条件3：1H正确测试了4H OB
                if not self.check_ob_test_1h(df_1h, i, ob_top, ob_bottom, 'long'):
                    continue

                # 条件3.5：回撤確認（防止追高）
                enable_pullback_confirm = self.params.get('enable_pullback_confirmation', True)
                if enable_pullback_confirm:
                    if not self.check_pullback_to_ob(df_1h, i, ob_top, ob_bottom, 'long'):
                        continue

                # 条件4：1H进场确认（包含成交量70%检查）
                if not self.check_entry_confirmation_1h(df_1h, i, 'long'):
                    continue

                if enable_rsi_filter and (not pd.isna(current_1h['rsi'])):
                    if current_1h['rsi'] > rsi_long_max:
                        continue

                # 生成做多信号
                entry_price = current_1h['c']
                stop_loss = self.calculate_stop_loss_mtf(df_4h_with_smc, current_time, 'long', ob_bottom, ob_top)

                if stop_loss is not None and stop_loss < entry_price:
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, 'long')

                    df_1h.loc[df_1h.index[i], 'signal'] = 1
                    df_1h.loc[df_1h.index[i], 'entry_price'] = entry_price
                    df_1h.loc[df_1h.index[i], 'stop_loss'] = stop_loss
                    df_1h.loc[df_1h.index[i], 'take_profit'] = take_profit
                    df_1h.at[df_1h.index[i], 'ob_source'] = ob_source
                    if signal_cooldown_bars > 0:
                        last_signal_bar_by_ob[ob_source] = i
                    long_count += 1

            # === 做空信号 ===
            elif not pd.isna(current_1h['bearish_ob_4h_top']):
                ob_top = current_1h['bearish_ob_4h_top']
                ob_bottom = current_1h['bearish_ob_4h_bottom']
                ob_source = f"4H_Bear_OB_{ob_bottom:.2f}-{ob_top:.2f}"

                if signal_cooldown_bars > 0:
                    last_i = last_signal_bar_by_ob.get(ob_source)
                    if last_i is not None and (i - last_i) < signal_cooldown_bars:
                        continue

                if enable_structure_filter:
                    if current_1h['structure_4h'] == 'bullish':
                        continue
                    if (not allow_neutral_market) and current_1h['structure_4h'] == 'ranging':
                        continue

                if enable_trend_filter:
                    if current_1h['trend_4h'] == 'bull':
                        continue
                    if (not allow_neutral_market) and current_1h['trend_4h'] == 'neutral':
                        continue

                if not self.check_ob_test_1h(df_1h, i, ob_top, ob_bottom, 'short'):
                    continue

                # 回撤確認（防止殺低）
                enable_pullback_confirm = self.params.get('enable_pullback_confirmation', True)
                if enable_pullback_confirm:
                    if not self.check_pullback_to_ob(df_1h, i, ob_top, ob_bottom, 'short'):
                        continue

                if not self.check_entry_confirmation_1h(df_1h, i, 'short'):
                    continue

                if enable_rsi_filter and (not pd.isna(current_1h['rsi'])):
                    if current_1h['rsi'] < rsi_short_min:
                        continue

                entry_price = current_1h['c']
                stop_loss = self.calculate_stop_loss_mtf(df_4h_with_smc, current_time, 'short', ob_bottom, ob_top)

                if stop_loss is not None and stop_loss > entry_price:
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, 'short')

                    df_1h.loc[df_1h.index[i], 'signal'] = -1
                    df_1h.loc[df_1h.index[i], 'entry_price'] = entry_price
                    df_1h.loc[df_1h.index[i], 'stop_loss'] = stop_loss
                    df_1h.loc[df_1h.index[i], 'take_profit'] = take_profit
                    df_1h.at[df_1h.index[i], 'ob_source'] = ob_source
                    if signal_cooldown_bars > 0:
                        last_signal_bar_by_ob[ob_source] = i
                    short_count += 1

        print(f"\n信號生成完成")
        print(f"  做多信號: {long_count}")
        print(f"  做空信號: {short_count}")
        print(f"  總信號數: {long_count + short_count}")
        print("="*80)

        return df_1h
