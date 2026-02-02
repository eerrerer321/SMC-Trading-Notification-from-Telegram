# monitor.py
# SMC 實時監控主程式
# -*- coding: utf-8 -*-

import sys
import os
import time
import pandas as pd
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

# 添加路徑
sys.path.insert(0, os.path.dirname(__file__))

# 導入配置
from config.settings import (
    SYMBOL, BASE_TIMEFRAME, TRADING_TIMEFRAME, EXCHANGE,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS, ENABLE_TELEGRAM,
    CHECK_INTERVAL, DATA_LOOKBACK, TRADING_MODE,
    SMC_PARAMS, TRADING_PARAMS, STRATEGY_PARAMS,
    NOTIFY_ON_SIGNAL, NOTIFY_ON_STOP_LOSS, NOTIFY_ON_TAKE_PROFIT, NOTIFY_ON_ERROR
)

# 導入核心組件
from engine.data_fetcher import LiveDataFetcher
from notification.telegram import TelegramNotifier
from notification.chart import SMCChartGenerator
from strategy.indicators import SMCIndicators
from strategy.smc_strategy import SMCStrategyFinal


@dataclass
class PaperPosition:
    """模擬持倉"""
    position_id: str
    side: str  # 'long' | 'short'
    entry_time: datetime
    entry_price: float
    stop_loss: float
    take_profit: float
    breakeven_moved: bool
    original_stop_loss: float


class SMCLiveMonitor:
    """SMC 實時監控系統"""

    def __init__(self):
        # 初始化數據獲取器
        self.data_fetcher = LiveDataFetcher(
            exchange_name=EXCHANGE,
            api_key=None,
            api_secret=None
        )

        # 初始化 Telegram 通知
        if ENABLE_TELEGRAM and TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_IDS:
            self.notifier = TelegramNotifier(
                bot_token=TELEGRAM_BOT_TOKEN,
                chat_ids=TELEGRAM_CHAT_IDS,
                enabled=True
            )
        else:
            self.notifier = None

        # 初始化 SMC 指標計算器
        self.smc_indicators = SMCIndicators(SMC_PARAMS)

        # 初始化策略
        self.strategy = SMCStrategyFinal(TRADING_PARAMS)

        # 初始化圖表生成器（延遲載入）
        self.chart_generator: Optional[SMCChartGenerator] = None

        self.positions: List[PaperPosition] = []
        self.next_position_seq = 1

        # 最後檢查時間
        self.last_check_time = None
        self.last_signal_time = None
        self.is_first_run = True
        self.last_daily_report_date = None

        # 註冊 Telegram 指令
        if self.notifier:
            self._register_telegram_commands()

        print(f"✅ SMC 實時監控系統初始化完成")
        print(f"   交易對：{SYMBOL}")
        print(f"   檢查間隔：{CHECK_INTERVAL} 秒")

    def _register_telegram_commands(self) -> None:
        """註冊 Telegram 指令處理器"""
        self.notifier.register_command('chart', self._handle_chart_command)
        self.notifier.register_command('status', self._handle_status_command)
        self.notifier.register_command('help', self._handle_help_command)

    def _handle_chart_command(self, chat_id: str, args: str) -> None:
        """處理 /chart 指令"""
        time_range = args.strip() if args.strip() else "7d"

        self.notifier._send_to_chat(
            chat_id,
            f"📊 正在生成 {SYMBOL} 過去 {time_range} 的交易圖表...\n請稍候。"
        )

        try:
            if self.chart_generator is None:
                self.chart_generator = SMCChartGenerator()

            image_path, summary, trades = self.chart_generator.generate_chart_for_telegram(
                symbol=SYMBOL,
                time_range=time_range
            )

            caption = f"📊 {SYMBOL} 過去 {time_range} 交易回顧\n共 {len(trades)} 筆交易"
            self.notifier._send_photo_to_chat(chat_id, image_path, caption)

            days, start_date, end_date = self.chart_generator.parse_time_range(time_range)
            message_parts = self.chart_generator.generate_trade_summary_parts(
                trades, days, start_date, end_date
            )

            for part in message_parts:
                self.notifier._send_to_chat(chat_id, part)

            print(f"✅ 已發送圖表到 {chat_id}")

        except Exception as e:
            error_msg = f"❌ 生成圖表時出錯：{str(e)}"
            print(error_msg)
            self.notifier._send_to_chat(chat_id, error_msg)

    def _handle_status_command(self, chat_id: str, args: str) -> None:
        """處理 /status 指令"""
        try:
            current_price = self.data_fetcher.get_current_price(SYMBOL)

            df_15m, df_4h = self.data_fetcher.get_latest_kline_data(
                symbol=SYMBOL,
                base_timeframe=BASE_TIMEFRAME,
                trading_timeframe=TRADING_TIMEFRAME,
                lookback=500
            )

            structure = "unknown"
            if df_4h is not None and len(df_4h) > 0:
                df_4h = self.smc_indicators.calculate_all(df_4h)
                structure = df_4h.iloc[-1].get('structure', 'unknown')

            message = f"""
📊 <b>系統狀態</b>

💰 <b>市場資訊</b>
• 交易對: {SYMBOL}
• 當前價格: ${current_price:,.2f}
• 市場結構: {structure}
• 更新時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📋 <b>持倉資訊</b>
"""
            if self.positions:
                for pos in self.positions:
                    side_emoji = "🟢" if pos.side == "long" else "🔴"
                    if pos.side == "long":
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
                    pnl_emoji = "📈" if pnl_pct > 0 else "📉"

                    message += f"\n{side_emoji} {pos.position_id} - {pos.side.upper()}"
                    message += f"\n  • 進場: ${pos.entry_price:,.2f}"
                    message += f"\n  • {pnl_emoji} 浮動盈虧: {pnl_pct:+.2f}%"
            else:
                message += "• 無持倉"

            message += "\n\n✅ 系統運行正常"

            self.notifier._send_to_chat(chat_id, message.strip())

        except Exception as e:
            self.notifier._send_to_chat(chat_id, f"❌ 獲取狀態時出錯：{str(e)}")

    def _handle_help_command(self, chat_id: str, args: str) -> None:
        """處理 /help 指令"""
        message = """
📖 <b>SMC 監控系統指令說明</b>

📊 <b>/chart [時間]</b>
生成交易回顧圖表
• /chart 7d - 過去 7 天
• /chart 30d - 過去 30 天
• /chart 3m - 過去 3 個月
• /chart 2025-01-01~2025-01-31

📋 <b>/status</b>
顯示當前系統狀態和持倉資訊

❓ <b>/help</b>
顯示此幫助訊息
        """
        self.notifier._send_to_chat(chat_id, message.strip())

    def process_new_candle(self):
        """處理新 K 線，檢測交易信號"""
        try:
            df_15m, df_4h = self.data_fetcher.get_latest_kline_data(
                symbol=SYMBOL,
                base_timeframe=BASE_TIMEFRAME,
                trading_timeframe=TRADING_TIMEFRAME,
                lookback=DATA_LOOKBACK
            )

            if df_15m is None or df_4h is None:
                print("❌ 數據獲取失敗")
                return

            # 計算 SMC 指標（4H）
            df_4h = self.smc_indicators.calculate_all(df_4h)

            # 識別關鍵結構位和高質量 OB
            df_4h = self.strategy.identify_key_structure_points(
                df_4h,
                min_move_pct=STRATEGY_PARAMS.get('min_structure_move_pct', 0.02)
            )
            df_4h = self.strategy.identify_high_quality_ob_4h(df_4h)

            # 生成交易信號（1H）
            df_1h = self.strategy.generate_signals_mtf(df_15m, df_4h)

            # 檢查最新信號
            latest_signals = df_1h[df_1h['signal'] != 0].tail(5)

            if len(latest_signals) > 0:
                latest = latest_signals.iloc[-1]
                signal_time = latest_signals.index[-1]

                if self.is_first_run:
                    self.last_signal_time = signal_time
                    print(f"  ℹ️  初始化：已記錄最後信號時間 {signal_time}")
                elif signal_time > self.last_signal_time:
                    self.last_signal_time = signal_time
                    self.send_signal_notification(latest, signal_time, df_1h)

            if self.is_first_run:
                self.is_first_run = False
                if len(latest_signals) == 0:
                    print(f"  ℹ️  初始化：無歷史信號")

            # 獲取當前價格和市場狀態
            current_price = self.data_fetcher.get_current_price(SYMBOL)
            latest_4h = df_4h.iloc[-1]
            structure = latest_4h['structure']

            if current_price is not None:
                self.update_positions(current_price)

            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 監控正常")
            print(f"  價格: ${current_price:,.2f}")
            print(f"  4H 結構: {structure}")
            print(f"  最近信號數: {len(latest_signals)}")

        except Exception as e:
            error_msg = f"處理數據時出錯: {str(e)}"
            print(f"❌ {error_msg}")
            if self.notifier and NOTIFY_ON_ERROR:
                self.notifier.notify_error(error_msg)

    def _new_position_id(self, side: str) -> str:
        prefix = 'L' if side == 'long' else 'S'
        pid = f"{prefix}{self.next_position_seq:04d}"
        self.next_position_seq += 1
        return pid

    def update_positions(self, current_price: float) -> None:
        """更新持倉狀態（止損/止盈/移動保本）"""
        if not self.positions:
            return

        trigger_r = float(STRATEGY_PARAMS.get('breakeven_trigger_r', 1.5))
        profit_pct = float(STRATEGY_PARAMS.get('breakeven_profit_pct', 0.005))
        enable_be = bool(STRATEGY_PARAMS.get('enable_breakeven', True))

        for pos in list(self.positions):
            side = pos.side

            # 止損/止盈檢查
            if side == 'long':
                if current_price <= pos.stop_loss:
                    pnl_pct = (pos.stop_loss - pos.entry_price) / pos.entry_price * 100.0
                    if self.notifier and NOTIFY_ON_STOP_LOSS:
                        self.notifier.notify_stop_loss(
                            side='long', entry_price=pos.entry_price,
                            exit_price=pos.stop_loss, pnl_pct=pnl_pct,
                            position_id=pos.position_id
                        )
                    self.positions.remove(pos)
                    continue

                if current_price >= pos.take_profit:
                    pnl_pct = (pos.take_profit - pos.entry_price) / pos.entry_price * 100.0
                    if self.notifier and NOTIFY_ON_TAKE_PROFIT:
                        self.notifier.notify_take_profit(
                            side='long', entry_price=pos.entry_price,
                            exit_price=pos.take_profit, pnl_pct=pnl_pct,
                            position_id=pos.position_id
                        )
                    self.positions.remove(pos)
                    continue
            else:
                if current_price >= pos.stop_loss:
                    pnl_pct = (pos.entry_price - pos.stop_loss) / pos.entry_price * 100.0
                    if self.notifier and NOTIFY_ON_STOP_LOSS:
                        self.notifier.notify_stop_loss(
                            side='short', entry_price=pos.entry_price,
                            exit_price=pos.stop_loss, pnl_pct=pnl_pct,
                            position_id=pos.position_id
                        )
                    self.positions.remove(pos)
                    continue

                if current_price <= pos.take_profit:
                    pnl_pct = (pos.entry_price - pos.take_profit) / pos.entry_price * 100.0
                    if self.notifier and NOTIFY_ON_TAKE_PROFIT:
                        self.notifier.notify_take_profit(
                            side='short', entry_price=pos.entry_price,
                            exit_price=pos.take_profit, pnl_pct=pnl_pct,
                            position_id=pos.position_id
                        )
                    self.positions.remove(pos)
                    continue

            # 移動保本
            if enable_be and not pos.breakeven_moved:
                risk = abs(pos.entry_price - pos.original_stop_loss)
                if risk > 0:
                    if side == 'long':
                        target = pos.entry_price + (trigger_r * risk)
                        if current_price >= target:
                            new_sl = pos.entry_price * (1.0 + profit_pct)
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            pos.breakeven_moved = True
                            if self.notifier:
                                self.notifier.notify_move_stop_loss(
                                    side='long', entry_price=pos.entry_price,
                                    old_stop_loss=old_sl, new_stop_loss=new_sl,
                                    position_id=pos.position_id,
                                    reason=f"到達 {trigger_r}R，移動停損到 +{profit_pct*100:.2f}%"
                                )
                    else:
                        target = pos.entry_price - (trigger_r * risk)
                        if current_price <= target:
                            new_sl = pos.entry_price * (1.0 - profit_pct)
                            old_sl = pos.stop_loss
                            pos.stop_loss = new_sl
                            pos.breakeven_moved = True
                            if self.notifier:
                                self.notifier.notify_move_stop_loss(
                                    side='short', entry_price=pos.entry_price,
                                    old_stop_loss=old_sl, new_stop_loss=new_sl,
                                    position_id=pos.position_id,
                                    reason=f"到達 {trigger_r}R，移動停損到 -{profit_pct*100:.2f}%"
                                )

    def send_signal_notification(self, signal_bar, signal_time, df_1h):
        """發送信號通知"""
        direction = 'long' if signal_bar['signal'] == 1 else 'short'
        entry_price = signal_bar['entry_price']
        stop_loss = signal_bar['stop_loss']
        take_profit = signal_bar['take_profit']

        rsi = signal_bar['rsi'] if not pd.isna(signal_bar['rsi']) else 50
        atr = signal_bar['atr'] if not pd.isna(signal_bar['atr']) else 0
        structure = signal_bar['structure_4h']
        ob_info = signal_bar['ob_source'] if signal_bar['ob_source'] else ""

        print(f"\n🔔 發現新 {direction.upper()} 信號！")
        print(f"   時間: {signal_time}")
        print(f"   進場: ${entry_price:,.2f}")
        print(f"   止損: ${stop_loss:,.2f}")
        print(f"   止盈: ${take_profit:,.2f}")

        position_id = self._new_position_id(direction)
        self.positions.append(
            PaperPosition(
                position_id=position_id,
                side=direction,
                entry_time=signal_time.to_pydatetime() if hasattr(signal_time, 'to_pydatetime') else datetime.now(),
                entry_price=float(entry_price),
                stop_loss=float(stop_loss),
                take_profit=float(take_profit),
                breakeven_moved=False,
                original_stop_loss=float(stop_loss),
            )
        )

        if self.notifier and NOTIFY_ON_SIGNAL:
            if direction == 'long':
                self.notifier.notify_long_signal(
                    price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                    atr=atr, rsi=rsi, structure=structure,
                    ob_info=f"• Order Block: {ob_info}" if ob_info else "",
                    position_id=position_id
                )
            else:
                self.notifier.notify_short_signal(
                    price=entry_price, stop_loss=stop_loss, take_profit=take_profit,
                    atr=atr, rsi=rsi, structure=structure,
                    ob_info=f"• Order Block: {ob_info}" if ob_info else "",
                    position_id=position_id
                )

    def check_and_send_daily_report(self) -> None:
        """檢查並發送每日狀態報告（早上 8 點）"""
        if not self.notifier:
            return

        from datetime import timezone
        tw_tz = timezone(timedelta(hours=8))
        now_tw = datetime.now(tw_tz)

        current_hour = now_tw.hour
        current_date = now_tw.date()

        if current_hour == 8 and self.last_daily_report_date != current_date:
            try:
                current_price = self.data_fetcher.get_current_price(SYMBOL)
                if current_price is None:
                    return

                df_15m, df_4h = self.data_fetcher.get_latest_kline_data(
                    symbol=SYMBOL,
                    base_timeframe=BASE_TIMEFRAME,
                    trading_timeframe=TRADING_TIMEFRAME,
                    lookback=DATA_LOOKBACK
                )

                if df_4h is not None and len(df_4h) > 0:
                    df_4h = self.smc_indicators.calculate_all(df_4h)
                    structure = df_4h.iloc[-1].get('structure', 'unknown')
                else:
                    structure = 'unknown'

                self.notifier.notify_daily_status(
                    current_price=current_price,
                    structure=structure,
                    positions=self.positions
                )

                self.last_daily_report_date = current_date
                print(f"\n📧 已發送每日狀態報告")

            except Exception as e:
                print(f"❌ 發送每日報告時出錯: {str(e)}")

    def run(self):
        """運行主監控循環"""
        print("\n" + "=" * 80)
        print("🚀 SMC 實時監控系統啟動")
        print("=" * 80)

        try:
            current_price = self.data_fetcher.get_current_price(SYMBOL)
            print(f"\n✅ 數據連接成功")
            print(f"   交易對: {SYMBOL}")
            print(f"   當前價格: ${current_price:,.2f}")
            print(f"   數據來源: {EXCHANGE.capitalize()}")
        except Exception as e:
            print(f"\n❌ 無法獲取價格數據: {str(e)}")
            return

        if self.notifier:
            config_info = {
                'symbol': SYMBOL,
                'mode': TRADING_MODE,
                'interval': CHECK_INTERVAL,
                'risk': TRADING_PARAMS['risk_per_trade'] * 100
            }
            self.notifier.notify_startup(config_info, current_price=current_price, positions=self.positions)
            self.notifier.start_polling()
            print(f"✅ Telegram 指令監聽已啟動")
            print(f"   支援指令: /chart, /status, /help")

        print(f"\n開始監控 {SYMBOL}...")
        print(f"按 Ctrl+C 停止\n")

        try:
            while True:
                has_new_candle = self.data_fetcher.check_new_candle(
                    symbol=SYMBOL,
                    timeframe='1h',
                    last_candle_time=self.last_check_time
                )

                if has_new_candle or self.last_check_time is None:
                    print(f"\n{'=' * 80}")
                    print(f"檢測到新 1H K 線，開始分析...")
                    self.process_new_candle()
                    self.last_check_time = datetime.now()

                self.check_and_send_daily_report()
                time.sleep(CHECK_INTERVAL)

        except KeyboardInterrupt:
            print("\n\n" + "=" * 80)
            print("⏹️  監控系統已停止")
            if self.notifier:
                self.notifier.stop_polling()
            print("=" * 80)
        except Exception as e:
            error_msg = f"監控系統錯誤: {str(e)}"
            print(f"\n❌ {error_msg}")
            if self.notifier:
                self.notifier.stop_polling()
                self.notifier.notify_error(error_msg)


def main():
    """主函數"""
    if ENABLE_TELEGRAM:
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_IDS:
            print("⚠️  警告：請先配置 Telegram Bot Token / Chat IDs！")
            print("   編輯 config/settings.py 或設置環境變數")
            response = input("\n是否繼續運行（不發送通知）？[y/N]: ")
            if response.lower() != 'y':
                return

    monitor = SMCLiveMonitor()
    monitor.run()


if __name__ == "__main__":
    main()
