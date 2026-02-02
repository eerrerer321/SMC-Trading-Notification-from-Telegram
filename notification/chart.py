# notification/chart.py
# SMC 交易圖表生成模組 - 使用 Plotly 繪製互動式 K 線圖
# -*- coding: utf-8 -*-

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass

# 添加路徑
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
except ImportError:
    print("請安裝 plotly: pip install plotly")
    raise

try:
    import kaleido
except ImportError:
    print("請安裝 kaleido 以支援 PNG 匯出: pip install kaleido")


@dataclass
class SimulatedTrade:
    """模擬交易記錄"""
    trade_id: int
    direction: str  # 'long' | 'short'
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # 'tp' | 'sl' | 'breakeven_sl' | 'open'
    pnl_pct: Optional[float] = None
    breakeven_moved: bool = False
    breakeven_time: Optional[datetime] = None


class SMCChartGenerator:
    """SMC 交易圖表生成器"""

    def __init__(self):
        # 導入必要模組
        from config.settings import (
            EXCHANGE, SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS
        )
        from engine.data_fetcher import LiveDataFetcher
        from strategy.indicators import SMCIndicators
        from strategy.smc_strategy import SMCStrategyFinal

        self.exchange = EXCHANGE
        self.smc_params = SMC_PARAMS
        self.trading_params = TRADING_PARAMS.copy()
        self.plan_e_params = STRATEGY_PARAMS

        # 初始化資料獲取器
        self.data_fetcher = LiveDataFetcher(
            exchange_name=EXCHANGE,
            api_key=None,
            api_secret=None
        )

        # 初始化 SMC 指標計算器
        self.smc_indicators = SMCIndicators(self.smc_params)

        # 初始化策略
        self.strategy = SMCStrategyFinal(self.trading_params)

    def parse_time_range(self, time_str: str) -> Tuple[int, Optional[datetime], Optional[datetime]]:
        """
        解析時間範圍字串，返回 (天數, 開始日期, 結束日期)

        支援格式：
        - '7d' -> 過去 7 天
        - '30d' -> 過去 30 天
        - '2w' -> 過去 14 天
        - '1m' -> 過去 30 天
        - '3m' -> 過去 90 天
        - '2025-01-01 2025-01-31' -> 指定日期範圍
        - '2025-01-01~2025-01-31' -> 指定日期範圍（用 ~ 分隔）

        Returns:
            (days, start_date, end_date)
            - 如果是相對時間（如 7d），start_date 和 end_date 為 None
            - 如果是日期範圍，返回計算出的天數和實際日期
        """
        time_str = time_str.strip()

        # 檢查是否為日期範圍格式
        date_separators = ['~', ' ']
        for sep in date_separators:
            if sep in time_str:
                parts = [p.strip() for p in time_str.split(sep) if p.strip()]
                if len(parts) == 2:
                    try:
                        # 嘗試解析日期
                        start_date = self._parse_date(parts[0])
                        end_date = self._parse_date(parts[1])

                        if start_date and end_date:
                            # 確保開始日期早於結束日期
                            if start_date > end_date:
                                start_date, end_date = end_date, start_date

                            # 計算天數
                            days = (end_date - start_date).days + 1

                            # 限制最長一年（365 天）
                            if days > 365:
                                print(f"⚠️ 查詢範圍超過一年，將限制為 365 天")
                                days = 365
                                start_date = end_date - timedelta(days=364)

                            return days, start_date, end_date
                    except Exception:
                        pass

        # 相對時間格式
        time_str_lower = time_str.lower()

        if time_str_lower.endswith('d'):
            try:
                days = int(time_str_lower[:-1])
            except ValueError:
                days = 30
        elif time_str_lower.endswith('w'):
            try:
                days = int(time_str_lower[:-1]) * 7
            except ValueError:
                days = 30
        elif time_str_lower.endswith('m'):
            try:
                days = int(time_str_lower[:-1]) * 30
            except ValueError:
                days = 30
        else:
            # 預設為天數
            try:
                days = int(time_str_lower)
            except ValueError:
                days = 30  # 預設 30 天

        # 限制最長一年
        if days > 365:
            print(f"⚠️ 查詢範圍超過一年，將限制為 365 天")
            days = 365

        return days, None, None

    def _parse_date(self, date_str: str) -> Optional[datetime]:
        """
        解析日期字串

        支援格式：
        - '2025-01-01'
        - '2025/01/01'
        - '20250101'
        """
        date_str = date_str.strip()

        # 嘗試多種格式
        formats = [
            '%Y-%m-%d',
            '%Y/%m/%d',
            '%Y%m%d',
            '%Y.%m.%d',
        ]

        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        return None

    def fetch_historical_data(self, symbol: str, days: int,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        獲取歷史 K 線資料

        Args:
            symbol: 交易對
            days: 天數
            start_date: 開始日期（可選，用於指定日期範圍）
            end_date: 結束日期（可選，用於指定日期範圍）

        Returns:
            (df_15m, df_4h) - 15分鐘和4小時資料
        """
        if start_date and end_date:
            print(f"正在獲取 {symbol} {start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')} 的資料...")
        else:
            print(f"正在獲取 {symbol} 過去 {days} 天的資料...")

        # 如果是指定日期範圍，計算從「現在」到「開始日期」的天數
        if start_date and end_date:
            # 計算從現在到開始日期的天數（確保能獲取足夠的歷史資料）
            days_from_now = (datetime.now() - start_date).days + 10  # 多獲取一些
            actual_days = max(days, days_from_now)
        else:
            actual_days = days

        # 計算需要的 15 分鐘 K 線數量
        # 每天 24 小時 * 4 (15分鐘一根) = 96 根/天
        candles_needed = actual_days * 96 + 500

        # 如果需要的 K 線數量超過單次 API 限制，使用分批獲取
        if candles_needed > 1500:
            print(f"  資料量較大，使用分批獲取模式...")
            df_15m = self.data_fetcher.fetch_historical_data_extended(
                symbol=symbol,
                timeframe='15m',
                days=actual_days
            )

            if df_15m is None or len(df_15m) == 0:
                raise Exception("無法獲取歷史資料")

            # 聚合為 4H
            df_4h = self.data_fetcher.aggregate_to_4h(df_15m)

        else:
            # 使用原本的單次獲取方式
            df_15m, df_4h = self.data_fetcher.get_latest_kline_data(
                symbol=symbol,
                base_timeframe='15m',
                trading_timeframe='4h',
                lookback=candles_needed
            )

            if df_15m is None or df_4h is None:
                raise Exception("無法獲取歷史資料")

            # 過濾到指定時間範圍
            cutoff_time = datetime.now() - timedelta(days=actual_days)
            df_15m = df_15m[df_15m.index >= cutoff_time]
            df_4h = df_4h[df_4h.index >= cutoff_time]

        # 如果指定了日期範圍，進一步過濾
        if start_date and end_date:
            # 結束日期需要包含當天的所有資料（到 23:59:59）
            end_date_inclusive = end_date + timedelta(days=1)

            df_15m = df_15m[(df_15m.index >= start_date) & (df_15m.index < end_date_inclusive)]
            df_4h = df_4h[(df_4h.index >= start_date) & (df_4h.index < end_date_inclusive)]

        print(f"  獲取 15m K線: {len(df_15m)} 根")
        print(f"  獲取 4H K線: {len(df_4h)} 根")

        return df_15m, df_4h

    def generate_signals(self, df_15m: pd.DataFrame, df_4h: pd.DataFrame) -> pd.DataFrame:
        """
        使用現有策略計算訊號

        Returns:
            df_1h - 包含訊號的 1 小時資料
        """
        print("正在計算 SMC 指標...")

        # 計算 4H SMC 指標
        df_4h = self.smc_indicators.calculate_all(df_4h)

        # 識別關鍵結構位
        min_move_pct = self.plan_e_params.get('min_structure_move_pct', 0.02)
        df_4h = self.strategy.identify_key_structure_points(df_4h, min_move_pct=min_move_pct)

        # 識別高質量 OB
        df_4h = self.strategy.identify_high_quality_ob_4h(df_4h)

        # 生成交易訊號（1H）
        print("正在生成交易訊號...")
        df_1h = self.strategy.generate_signals_mtf(df_15m, df_4h)

        return df_1h

    def simulate_trades(self, df_1h: pd.DataFrame, df_4h: pd.DataFrame) -> List[SimulatedTrade]:
        """
        模擬交易結果

        根據訊號進場後，逐根 K 線檢查是否觸及 SL/TP
        支援移動止損邏輯

        Returns:
            交易列表
        """
        trades: List[SimulatedTrade] = []
        trade_id = 1

        # 獲取移動保本參數
        enable_breakeven = bool(self.trading_params.get('enable_breakeven', True))
        breakeven_trigger_r = float(self.trading_params.get('breakeven_trigger_r', 1.5))
        breakeven_profit_pct = float(self.trading_params.get('breakeven_profit_pct', 0.005))

        # 找出所有訊號
        signals = df_1h[df_1h['signal'] != 0].copy()

        print(f"找到 {len(signals)} 個交易訊號，開始模擬...")

        for idx, signal_row in signals.iterrows():
            direction = 'long' if signal_row['signal'] == 1 else 'short'
            entry_time = idx
            entry_price = signal_row['entry_price']
            stop_loss = signal_row['stop_loss']
            take_profit = signal_row['take_profit']
            original_stop_loss = stop_loss

            # 創建交易記錄
            trade = SimulatedTrade(
                trade_id=trade_id,
                direction=direction,
                entry_time=entry_time.to_pydatetime() if hasattr(entry_time, 'to_pydatetime') else entry_time,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )

            # 從進場後的下一根 K 線開始檢查
            future_candles = df_1h[df_1h.index > entry_time]

            for candle_idx, candle in future_candles.iterrows():
                candle_time = candle_idx.to_pydatetime() if hasattr(candle_idx, 'to_pydatetime') else candle_idx
                high = candle['h']
                low = candle['l']

                # 檢查移動止損
                if enable_breakeven and not trade.breakeven_moved:
                    risk = abs(entry_price - original_stop_loss)
                    if risk > 0:
                        if direction == 'long':
                            target = entry_price + (breakeven_trigger_r * risk)
                            if high >= target:
                                trade.stop_loss = entry_price * (1 + breakeven_profit_pct)
                                trade.breakeven_moved = True
                                trade.breakeven_time = candle_time
                        else:  # short
                            target = entry_price - (breakeven_trigger_r * risk)
                            if low <= target:
                                trade.stop_loss = entry_price * (1 - breakeven_profit_pct)
                                trade.breakeven_moved = True
                                trade.breakeven_time = candle_time

                # 檢查止損/止盈（同一根 K 線先檢查不利方向）
                if direction == 'long':
                    # 做多：先檢查止損
                    if low <= trade.stop_loss:
                        trade.exit_time = candle_time
                        trade.exit_price = trade.stop_loss
                        trade.exit_reason = 'breakeven_sl' if trade.breakeven_moved else 'sl'
                        trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
                        break
                    elif high >= take_profit:
                        trade.exit_time = candle_time
                        trade.exit_price = take_profit
                        trade.exit_reason = 'tp'
                        trade.pnl_pct = (trade.exit_price - entry_price) / entry_price * 100
                        break
                else:  # short
                    # 做空：先檢查止損
                    if high >= trade.stop_loss:
                        trade.exit_time = candle_time
                        trade.exit_price = trade.stop_loss
                        trade.exit_reason = 'breakeven_sl' if trade.breakeven_moved else 'sl'
                        trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100
                        break
                    elif low <= take_profit:
                        trade.exit_time = candle_time
                        trade.exit_price = take_profit
                        trade.exit_reason = 'tp'
                        trade.pnl_pct = (entry_price - trade.exit_price) / entry_price * 100
                        break

            # 如果交易尚未結束
            if trade.exit_time is None:
                trade.exit_reason = 'open'
                # 計算浮動盈虧（使用最後一根 K 線收盤價）
                last_close = df_1h['c'].iloc[-1]
                if direction == 'long':
                    trade.pnl_pct = (last_close - entry_price) / entry_price * 100
                else:
                    trade.pnl_pct = (entry_price - last_close) / entry_price * 100

            trades.append(trade)
            trade_id += 1

        return trades

    def create_chart(self, df_4h: pd.DataFrame, trades: List[SimulatedTrade],
                     symbol: str = "ETHUSDT", days: int = 30,
                     time_desc: str = None) -> go.Figure:
        """
        創建 Plotly K 線圖並標記交易訊號

        Args:
            df_4h: 4 小時 K 線資料
            trades: 模擬交易列表
            symbol: 交易對
            days: 天數範圍
            time_desc: 時間描述（如 "過去 30 天" 或 "2025/01/01 - 2025/01/31"）

        Returns:
            Plotly Figure 物件
        """
        # 時間描述
        if time_desc is None:
            time_desc = f"過去 {days} 天"

        # 創建子圖（K 線 + 成交量）
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            subplot_titles=(f'{symbol} 4H K線圖 - {time_desc}', '成交量'),
            row_heights=[0.8, 0.2]
        )

        # 繪製 K 線圖
        fig.add_trace(
            go.Candlestick(
                x=df_4h.index,
                open=df_4h['o'],
                high=df_4h['h'],
                low=df_4h['l'],
                close=df_4h['c'],
                name='K線',
                increasing_line_color='#26a69a',
                decreasing_line_color='#ef5350'
            ),
            row=1, col=1
        )

        # 繪製成交量
        colors = ['#26a69a' if c >= o else '#ef5350'
                  for c, o in zip(df_4h['c'], df_4h['o'])]
        fig.add_trace(
            go.Bar(
                x=df_4h.index,
                y=df_4h['v'],
                name='成交量',
                marker_color=colors,
                opacity=0.7
            ),
            row=2, col=1
        )

        # ========== 繪製 Order Block 區域 ==========
        # 收集所有 OB 區域（避免重複繪製相同的 OB）
        drawn_bullish_obs = set()  # (top, bottom) 已繪製的 Bullish OB
        drawn_bearish_obs = set()  # (top, bottom) 已繪製的 Bearish OB

        for i in range(len(df_4h)):
            row_data = df_4h.iloc[i]
            current_time = df_4h.index[i]

            # Bullish OB（綠色半透明區域）
            if 'hq_bullish_ob_top' in df_4h.columns and not pd.isna(row_data.get('hq_bullish_ob_top')):
                ob_top = row_data['hq_bullish_ob_top']
                ob_bottom = row_data['hq_bullish_ob_bottom']
                ob_key = (round(ob_top, 2), round(ob_bottom, 2))

                if ob_key not in drawn_bullish_obs:
                    drawn_bullish_obs.add(ob_key)
                    # 找出這個 OB 的有效時間範圍
                    ob_start = current_time
                    ob_end = current_time

                    # 向後找到 OB 結束的時間
                    for j in range(i, len(df_4h)):
                        future_row = df_4h.iloc[j]
                        if pd.isna(future_row.get('hq_bullish_ob_top')) or \
                           round(future_row['hq_bullish_ob_top'], 2) != ob_key[0]:
                            break
                        ob_end = df_4h.index[j]

                    # 繪製 OB 區域
                    fig.add_shape(
                        type="rect",
                        x0=ob_start, x1=ob_end,
                        y0=ob_bottom, y1=ob_top,
                        fillcolor="rgba(0, 200, 83, 0.15)",  # 綠色半透明
                        line=dict(color="rgba(0, 200, 83, 0.5)", width=1),
                        row=1, col=1
                    )
                    # 標註 OB
                    fig.add_annotation(
                        x=ob_start,
                        y=ob_top,
                        text=f"Bull OB",
                        showarrow=False,
                        font=dict(size=9, color="#00c853"),
                        xanchor="left",
                        yanchor="bottom",
                        row=1, col=1
                    )

            # Bearish OB（紅色半透明區域）
            if 'hq_bearish_ob_top' in df_4h.columns and not pd.isna(row_data.get('hq_bearish_ob_top')):
                ob_top = row_data['hq_bearish_ob_top']
                ob_bottom = row_data['hq_bearish_ob_bottom']
                ob_key = (round(ob_top, 2), round(ob_bottom, 2))

                if ob_key not in drawn_bearish_obs:
                    drawn_bearish_obs.add(ob_key)
                    ob_start = current_time
                    ob_end = current_time

                    for j in range(i, len(df_4h)):
                        future_row = df_4h.iloc[j]
                        if pd.isna(future_row.get('hq_bearish_ob_top')) or \
                           round(future_row['hq_bearish_ob_top'], 2) != ob_key[0]:
                            break
                        ob_end = df_4h.index[j]

                    fig.add_shape(
                        type="rect",
                        x0=ob_start, x1=ob_end,
                        y0=ob_bottom, y1=ob_top,
                        fillcolor="rgba(255, 23, 68, 0.15)",  # 紅色半透明
                        line=dict(color="rgba(255, 23, 68, 0.5)", width=1),
                        row=1, col=1
                    )
                    fig.add_annotation(
                        x=ob_start,
                        y=ob_bottom,
                        text=f"Bear OB",
                        showarrow=False,
                        font=dict(size=9, color="#ff1744"),
                        xanchor="left",
                        yanchor="top",
                        row=1, col=1
                    )

        # ========== 標記交易訊號 ==========
        for trade in trades:
            entry_time = trade.entry_time
            entry_price = trade.entry_price

            # 進場標記
            if trade.direction == 'long':
                # 做多進場 - 綠色向上三角形
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=15,
                            color='#00c853',
                            line=dict(width=2, color='white')
                        ),
                        name=f'做多 #{trade.trade_id}',
                        hovertemplate=(
                            f'<b>做多進場 #{trade.trade_id}</b><br>'
                            f'時間: %{{x}}<br>'
                            f'價格: $%{{y:,.2f}}<br>'
                            f'止損: ${trade.stop_loss:,.2f}<br>'
                            f'止盈: ${trade.take_profit:,.2f}<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )
            else:
                # 做空進場 - 紅色向下三角形
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time],
                        y=[entry_price],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=15,
                            color='#ff1744',
                            line=dict(width=2, color='white')
                        ),
                        name=f'做空 #{trade.trade_id}',
                        hovertemplate=(
                            f'<b>做空進場 #{trade.trade_id}</b><br>'
                            f'時間: %{{x}}<br>'
                            f'價格: $%{{y:,.2f}}<br>'
                            f'止損: ${trade.stop_loss:,.2f}<br>'
                            f'止盈: ${trade.take_profit:,.2f}<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # 移動止損標記
            if trade.breakeven_moved and trade.breakeven_time:
                fig.add_trace(
                    go.Scatter(
                        x=[trade.breakeven_time],
                        y=[trade.stop_loss],
                        mode='markers',
                        marker=dict(
                            symbol='diamond',
                            size=10,
                            color='#2196f3',
                            line=dict(width=1, color='white')
                        ),
                        name=f'移動止損 #{trade.trade_id}',
                        hovertemplate=(
                            f'<b>移動止損 #{trade.trade_id}</b><br>'
                            f'時間: %{{x}}<br>'
                            f'新止損: $%{{y:,.2f}}<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

            # 出場標記
            if trade.exit_time and trade.exit_price:
                if trade.exit_reason == 'tp':
                    # 止盈 - 金色星形
                    marker_config = dict(
                        symbol='star',
                        size=15,
                        color='#ffd700',
                        line=dict(width=2, color='white')
                    )
                    label = '止盈'
                elif trade.exit_reason in ['sl', 'breakeven_sl']:
                    # 止損 - 灰色 X
                    marker_config = dict(
                        symbol='x',
                        size=12,
                        color='#9e9e9e' if trade.exit_reason == 'sl' else '#64b5f6',
                        line=dict(width=3)
                    )
                    label = '保本止損' if trade.exit_reason == 'breakeven_sl' else '止損'
                else:
                    continue

                fig.add_trace(
                    go.Scatter(
                        x=[trade.exit_time],
                        y=[trade.exit_price],
                        mode='markers',
                        marker=marker_config,
                        name=f'{label} #{trade.trade_id}',
                        hovertemplate=(
                            f'<b>{label} #{trade.trade_id}</b><br>'
                            f'時間: %{{x}}<br>'
                            f'價格: $%{{y:,.2f}}<br>'
                            f'盈虧: {trade.pnl_pct:+.2f}%<extra></extra>'
                        ),
                        showlegend=False
                    ),
                    row=1, col=1
                )

                # 連接進出場的線段
                line_color = '#00c853' if trade.pnl_pct > 0 else '#ff1744'
                fig.add_trace(
                    go.Scatter(
                        x=[entry_time, trade.exit_time],
                        y=[entry_price, trade.exit_price],
                        mode='lines',
                        line=dict(color=line_color, width=1, dash='dot'),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=1, col=1
                )

        # 添加圖例說明
        legend_items = [
            ('triangle-up', '#00c853', '做多進場'),
            ('triangle-down', '#ff1744', '做空進場'),
            ('star', '#ffd700', '止盈出場'),
            ('x', '#9e9e9e', '止損出場'),
            ('diamond', '#2196f3', '移動止損'),
            ('square', 'rgba(0, 200, 83, 0.3)', 'Bullish OB'),
            ('square', 'rgba(255, 23, 68, 0.3)', 'Bearish OB'),
        ]

        for i, (symbol_type, color, name) in enumerate(legend_items):
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode='markers',
                    marker=dict(symbol=symbol_type, size=12, color=color),
                    name=name,
                    showlegend=True
                ),
                row=1, col=1
            )

        # 更新佈局
        fig.update_layout(
            title=dict(
                text=f'SMC 交易回顧 - {symbol} 4H',
                font=dict(size=20)
            ),
            xaxis_rangeslider_visible=False,
            height=800,
            template='plotly_dark',
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor='rgba(0,0,0,0.5)'
            ),
            hovermode='x unified'
        )

        # 更新 Y 軸
        fig.update_yaxes(title_text="價格 (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="成交量", row=2, col=1)

        return fig

    def generate_trade_summary(self, trades: List[SimulatedTrade], days: int,
                               start_date: Optional[datetime] = None,
                               end_date: Optional[datetime] = None) -> str:
        """
        生成交易明細文字訊息

        Args:
            trades: 模擬交易列表
            days: 時間範圍（天）
            start_date: 開始日期（可選）
            end_date: 結束日期（可選）

        Returns:
            格式化的文字訊息
        """
        # 使用新方法取得分段訊息，然後合併
        messages = self.generate_trade_summary_parts(trades, days, start_date, end_date)
        return "\n".join(messages)

    def generate_trade_summary_parts(self, trades: List[SimulatedTrade], days: int,
                                      start_date: Optional[datetime] = None,
                                      end_date: Optional[datetime] = None,
                                      max_chars: int = 4000) -> List[str]:
        """
        生成交易明細文字訊息（分段版本，適用於 Telegram）

        Telegram 單則訊息限制為 4096 字元，此方法會：
        1. 按交易邊界分段（不會切斷單筆交易）
        2. 統計摘要永遠作為最後一段獨立訊息

        Args:
            trades: 模擬交易列表
            days: 時間範圍（天）
            start_date: 開始日期（可選）
            end_date: 結束日期（可選）
            max_chars: 單則訊息最大字元數（預設 4000，保留緩衝）

        Returns:
            訊息列表，每則訊息不超過 max_chars
        """
        # 構建時間描述
        if start_date and end_date:
            time_desc = f"{start_date.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}"
        else:
            time_desc = f"過去 {days} 天"

        if not trades:
            return [f"📊 {time_desc} 內無交易訊號"]

        # 計算統計數據
        completed_trades = [t for t in trades if t.exit_reason != 'open']
        open_trades = [t for t in trades if t.exit_reason == 'open']
        wins = [t for t in completed_trades if t.pnl_pct and t.pnl_pct > 0]
        losses = [t for t in completed_trades if t.pnl_pct and t.pnl_pct <= 0]

        # 名目加總（原本的計算方式）
        total_pnl_nominal = sum(t.pnl_pct for t in completed_trades if t.pnl_pct)
        win_rate = len(wins) / len(completed_trades) * 100 if completed_trades else 0

        # 複利計算（正確的報酬計算）
        equity = 1.0
        peak_equity = 1.0
        max_drawdown = 0.0
        total_profit = 0.0
        total_loss = 0.0

        for t in completed_trades:
            if t.pnl_pct:
                pnl_ratio = t.pnl_pct / 100.0
                equity *= (1 + pnl_ratio)

                # 追蹤最大回撤
                if equity > peak_equity:
                    peak_equity = equity
                drawdown = (peak_equity - equity) / peak_equity
                if drawdown > max_drawdown:
                    max_drawdown = drawdown

                # 計算獲利因子
                if t.pnl_pct > 0:
                    total_profit += t.pnl_pct
                else:
                    total_loss += abs(t.pnl_pct)

        total_pnl_compound = (equity - 1.0) * 100  # 複利報酬百分比
        profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

        # 生成各筆交易的文字（每筆獨立）
        trade_texts = []
        for trade in trades:
            trade_lines = self._format_single_trade(trade)
            trade_texts.append("\n".join(trade_lines))

        # 生成統計摘要（獨立）
        stats_lines = [
            "─" * 20,
            "",
            "📊 <b>統計摘要</b>",
            f"• 總交易數：{len(trades)} 筆"
        ]
        if completed_trades:
            stats_lines.append(f"• 已完成：{len(completed_trades)} 筆")
            stats_lines.append(f"• 勝率：{win_rate:.1f}% ({len(wins)}勝{len(losses)}敗)")
            stats_lines.append(f"• 總盈虧：{total_pnl_nominal:+.2f}%（名目）")
            stats_lines.append(f"• 複利報酬：{total_pnl_compound:+.2f}%")
            stats_lines.append(f"• 最大回撤：{max_drawdown*100:.2f}%")
            if profit_factor != float('inf'):
                stats_lines.append(f"• 獲利因子：{profit_factor:.2f}")
            else:
                stats_lines.append(f"• 獲利因子：∞（無虧損）")
        if open_trades:
            stats_lines.append(f"• 持倉中：{len(open_trades)} 筆")
        stats_text = "\n".join(stats_lines)

        # 組合訊息：按交易邊界分段
        messages = []
        header = f"📊 <b>{time_desc} 交易回顧</b>\n"

        current_message = header
        current_part = 1
        total_trades = len(trade_texts)

        for i, trade_text in enumerate(trade_texts):
            # 嘗試將這筆交易加入當前訊息
            potential_message = current_message + "\n" + trade_text if current_message != header else current_message + trade_text

            if len(potential_message) <= max_chars:
                current_message = potential_message
            else:
                # 當前訊息已滿，儲存並開始新訊息
                if current_message != header:
                    messages.append(current_message)
                    current_part += 1

                # 開始新訊息（帶分頁標記）
                page_header = f"📊 <b>{time_desc} 交易回顧（續 {current_part}）</b>\n"
                current_message = page_header + trade_text

        # 儲存最後的交易訊息
        if current_message and current_message != header:
            messages.append(current_message)

        # 統計摘要作為獨立的最後一則訊息
        messages.append(stats_text)

        return messages

    def _format_single_trade(self, trade: 'SimulatedTrade') -> List[str]:
        """
        格式化單筆交易為文字行

        Args:
            trade: 交易記錄

        Returns:
            格式化的文字行列表
        """
        direction_emoji = "📈" if trade.direction == 'long' else "📉"
        direction_text = "做多" if trade.direction == 'long' else "做空"

        lines = [
            f"\n{direction_emoji} <b>交易 #{trade.trade_id} - {direction_text}</b>",
            f"• 時間：{trade.entry_time.strftime('%Y-%m-%d %H:%M')}",
            f"• 進場：{trade.entry_price:,.2f} USDT",
            f"• 止損：{trade.stop_loss:,.2f} USDT｜止盈：{trade.take_profit:,.2f} USDT"
        ]

        if trade.breakeven_moved:
            lines.append(f"• 📍 已移動止損（{trade.breakeven_time.strftime('%m-%d %H:%M') if trade.breakeven_time else 'N/A'}）")

        if trade.exit_reason == 'open':
            lines.append(f"• 結果：⏳ 持倉中（浮動 {trade.pnl_pct:+.2f}%）")
        elif trade.exit_reason == 'tp':
            lines.append(f"• 出場：{trade.exit_price:,.2f} USDT")
            lines.append(f"• 結果：✅ 止盈 ({trade.pnl_pct:+.2f}%)")
        elif trade.exit_reason == 'breakeven_sl':
            lines.append(f"• 出場：{trade.exit_price:,.2f} USDT")
            lines.append(f"• 結果：🔷 保本止損 ({trade.pnl_pct:+.2f}%)")
        else:
            lines.append(f"• 出場：{trade.exit_price:,.2f} USDT")
            lines.append(f"• 結果：❌ 止損 ({trade.pnl_pct:+.2f}%)")

        return lines

    def generate_chart_for_telegram(self, symbol: str, time_range: str,
                                     output_dir: str = None) -> Tuple[str, str, List[SimulatedTrade]]:
        """
        為 Telegram 生成圖表和訊息

        Args:
            symbol: 交易對
            time_range: 時間範圍字串（如 '7d', '30d', '3m', '2025-01-01~2025-01-31'）
            output_dir: 輸出目錄（預設為專案根目錄下的 output）

        Returns:
            (圖片路徑, 文字訊息, 交易列表)
        """
        # 解析時間範圍
        days, start_date, end_date = self.parse_time_range(time_range)

        # 設定輸出目錄
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)

        # 獲取資料
        df_15m, df_4h = self.fetch_historical_data(symbol, days, start_date, end_date)

        # 生成訊號
        df_1h = self.generate_signals(df_15m, df_4h)

        # 模擬交易
        trades = self.simulate_trades(df_1h, df_4h)

        # 構建時間範圍描述
        if start_date and end_date:
            time_desc = f"{start_date.strftime('%Y-%m-%d')}~{end_date.strftime('%Y-%m-%d')}"
            chart_title_time = f"{start_date.strftime('%Y/%m/%d')} - {end_date.strftime('%Y/%m/%d')}"
        else:
            time_desc = f"{days}d"
            chart_title_time = f"過去 {days} 天"

        # 創建圖表
        fig = self.create_chart(df_4h, trades, symbol, days, chart_title_time)

        # 儲存圖片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        image_path = os.path.join(output_dir, f'smc_chart_{symbol}_{days}d_{timestamp}.png')

        fig.write_image(image_path, width=1400, height=800, scale=2)
        print(f"圖表已儲存: {image_path}")

        # 生成文字訊息
        summary = self.generate_trade_summary(trades, days, start_date, end_date)

        return image_path, summary, trades


# ============ 測試用 ============
if __name__ == "__main__":
    print("="*80)
    print("SMC 圖表生成器測試")
    print("="*80)

    generator = SMCChartGenerator()

    # 測試生成 7 天的圖表
    try:
        image_path, summary, trades = generator.generate_chart_for_telegram(
            symbol="ETHUSDT",
            time_range="7d"
        )

        print("\n" + "="*80)
        print("測試結果")
        print("="*80)
        print(f"圖片路徑: {image_path}")
        print(f"交易數量: {len(trades)}")
        print("\n文字訊息預覽:")
        print(summary.replace('<b>', '').replace('</b>', ''))

    except Exception as e:
        print(f"測試失敗: {e}")
        import traceback
        traceback.print_exc()
