# notification/telegram.py
# Telegram 通知系統
# -*- coding: utf-8 -*-

import os
import requests
import threading
import time
from datetime import datetime
from typing import Optional, Callable, Dict, Any

class TelegramNotifier:
    """Telegram 通知管理器"""

    def __init__(self, bot_token: str, chat_id=None, chat_ids=None, enabled: bool = True):
        self.bot_token = bot_token
        self.enabled = enabled
        self.base_url = f"https://api.telegram.org/bot{bot_token}"

        # 支持单个 chat_id（向后兼容）或多个 chat_ids
        if chat_ids is not None:
            # 如果提供了 chat_ids（列表）
            if isinstance(chat_ids, list):
                self.chat_ids = chat_ids
            else:
                # 如果是字符串，转换为列表
                self.chat_ids = [str(chat_ids)]
        elif chat_id is not None:
            # 如果只提供了单个 chat_id，转换为列表
            self.chat_ids = [str(chat_id)]
        else:
            self.chat_ids = []

        # 保留 chat_id 属性以保持向后兼容
        self.chat_id = self.chat_ids[0] if self.chat_ids else None

        # 指令處理器
        self._command_handlers: Dict[str, Callable] = {}
        self._polling_thread: Optional[threading.Thread] = None
        self._polling_active = False
        self._last_update_id = 0

    def send_message(self, text: str, parse_mode: str = "HTML") -> bool:
        """
        发送 Telegram 消息到所有配置的 chat_ids

        Args:
            text: 消息内容
            parse_mode: 'HTML' | 'Markdown'

        Returns:
            是否至少成功发送给一个接收者
        """
        if not self.enabled:
            return False

        if not self.chat_ids:
            print("⚠️ 没有配置 Telegram Chat IDs")
            return False

        success_count = 0
        url = f"{self.base_url}/sendMessage"

        # 循环发送给所有 chat_ids
        for chat_id in self.chat_ids:
            try:
                payload = {
                    'chat_id': chat_id,
                    'text': text,
                    'parse_mode': parse_mode
                }

                response = requests.post(url, json=payload, timeout=10)

                if response.status_code == 200:
                    success_count += 1
                else:
                    print(f"❌ Telegram 发送失败 (Chat ID: {chat_id}): {response.text}")

            except Exception as e:
                print(f"❌ Telegram 发送错误 (Chat ID: {chat_id}): {e}")

        # 只要有一个发送成功就返回 True
        return success_count > 0

    def notify_long_signal(self, price: float, stop_loss: float,
                          take_profit: float, atr: float, rsi: float,
                          structure: str, ob_info: str = "",
                          position_id: Optional[str] = None,
                          current_price: Optional[float] = None,
                          breakeven_trigger_price: Optional[float] = None,
                          breakeven_new_sl: Optional[float] = None,
                          max_deviation_pct: float = 0.02) -> bool:
        """發送做多信號通知"""

        risk_reward = abs((take_profit - price) / (price - stop_loss))

        pos_line = f"• 倉位ID: {position_id}" if position_id else ""

        # 即時價格與偏離資訊
        price_section = ""
        if current_price is not None:
            deviation_pct = (current_price - price) / price * 100
            deviation_abs = current_price - price
            if abs(deviation_pct) > max_deviation_pct * 100:
                price_section = f"""
🔔 <b>即時價格</b>
• 當前市價: ${current_price:,.2f}
• ⚠️ 偏離信號價: {deviation_pct:+.2f}% (${deviation_abs:+,.2f})
• ❗ 價格已大幅偏離，請謹慎評估是否進場"""
            else:
                price_section = f"""
🔔 <b>即時價格</b>
• 當前市價: ${current_price:,.2f}
• 偏離信號價: {deviation_pct:+.2f}% (${deviation_abs:+,.2f})"""

        # 移動止損資訊
        breakeven_section = ""
        if breakeven_trigger_price is not None and breakeven_new_sl is not None:
            breakeven_section = f"""
🧲 <b>移動止損計劃</b>
• 觸發價格: ${breakeven_trigger_price:,.2f} (+{(breakeven_trigger_price-price)/price*100:.2f}%)
• 觸發後止損移至: ${breakeven_new_sl:,.2f} (+{(breakeven_new_sl-price)/price*100:.2f}%)"""

        message = f"""
🟢 <b>SMC 做多信號</b> 🟢

📊 <b>基本資訊</b>
• 交易對: ETHUSDT
{pos_line}
• 信號價: ${price:,.2f}
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{price_section}

📈 <b>技術分析</b>
• 市場結構: {structure}
• RSI: {rsi:.2f}
• ATR: {atr:.2f} ({atr/price*100:.3f}%)
{ob_info}

💰 <b>交易計劃</b>
• 進場價: ${price:,.2f}
• 止損價: ${stop_loss:,.2f} (-{(price-stop_loss)/price*100:.2f}%)
• 止盈價: ${take_profit:,.2f} (+{(take_profit-price)/price*100:.2f}%)
• 風險報酬比: 1:{risk_reward:.2f}
{breakeven_section}

⚠️ <b>風險提示</b>
請確認市場環境後再進場！
        """

        return self.send_message(message.strip())

    def notify_move_stop_loss(self, side: str, entry_price: float,
                              old_stop_loss: float, new_stop_loss: float,
                              position_id: Optional[str] = None,
                              reason: str = "移動停損") -> bool:
        emoji = "🟢" if side == "long" else "🔴"
        pos_line = f"\n🏷️ <b>倉位ID</b>: {position_id}" if position_id else ""

        message = f"""
🧲 <b>移動停損觸發</b>

{emoji} <b>持倉類型</b>: {side.upper()}
{pos_line}

🧾 <b>調整內容</b>
• 進場價: ${entry_price:,.2f}
• 原止損: ${old_stop_loss:,.2f}
• 新止損: ${new_stop_loss:,.2f}
• 原因: {reason}
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        return self.send_message(message.strip())

    def notify_short_signal(self, price: float, stop_loss: float,
                           take_profit: float, atr: float, rsi: float,
                           structure: str, ob_info: str = "",
                           position_id: Optional[str] = None,
                           current_price: Optional[float] = None,
                           breakeven_trigger_price: Optional[float] = None,
                           breakeven_new_sl: Optional[float] = None,
                           max_deviation_pct: float = 0.02) -> bool:
        """發送做空信號通知"""

        risk_reward = abs((price - take_profit) / (stop_loss - price))

        pos_line = f"• 倉位ID: {position_id}" if position_id else ""

        # 即時價格與偏離資訊
        price_section = ""
        if current_price is not None:
            deviation_pct = (current_price - price) / price * 100
            deviation_abs = current_price - price
            if abs(deviation_pct) > max_deviation_pct * 100:
                price_section = f"""
🔔 <b>即時價格</b>
• 當前市價: ${current_price:,.2f}
• ⚠️ 偏離信號價: {deviation_pct:+.2f}% (${deviation_abs:+,.2f})
• ❗ 價格已大幅偏離，請謹慎評估是否進場"""
            else:
                price_section = f"""
🔔 <b>即時價格</b>
• 當前市價: ${current_price:,.2f}
• 偏離信號價: {deviation_pct:+.2f}% (${deviation_abs:+,.2f})"""

        # 移動止損資訊
        breakeven_section = ""
        if breakeven_trigger_price is not None and breakeven_new_sl is not None:
            breakeven_section = f"""
🧲 <b>移動止損計劃</b>
• 觸發價格: ${breakeven_trigger_price:,.2f} (-{(price-breakeven_trigger_price)/price*100:.2f}%)
• 觸發後止損移至: ${breakeven_new_sl:,.2f} (-{(price-breakeven_new_sl)/price*100:.2f}%)"""

        message = f"""
🔴 <b>SMC 做空信號</b> 🔴

📊 <b>基本資訊</b>
• 交易對: ETHUSDT
{pos_line}
• 信號價: ${price:,.2f}
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{price_section}

📉 <b>技術分析</b>
• 市場結構: {structure}
• RSI: {rsi:.2f}
• ATR: {atr:.2f} ({atr/price*100:.3f}%)
{ob_info}

💰 <b>交易計劃</b>
• 進場價: ${price:,.2f}
• 止損價: ${stop_loss:,.2f} (+{(stop_loss-price)/price*100:.2f}%)
• 止盈價: ${take_profit:,.2f} (-{(price-take_profit)/price*100:.2f}%)
• 風險報酬比: 1:{risk_reward:.2f}
{breakeven_section}

⚠️ <b>風險提示</b>
請確認市場環境後再進場！
        """

        return self.send_message(message.strip())

    def notify_stop_loss(self, side: str, entry_price: float,
                        exit_price: float, pnl_pct: float,
                        position_id: Optional[str] = None) -> bool:
        """發送止損通知"""

        emoji = "🟢" if side == "long" else "🔴"
        pos_line = f"\n🏷️ <b>倉位ID</b>: {position_id}" if position_id else ""

        message = f"""
🛑 <b>止損觸發</b> 🛑

{emoji} <b>持倉類型</b>: {side.upper()}
{pos_line}

💸 <b>交易結果</b>
• 進場價: ${entry_price:,.2f}
• 出場價: ${exit_price:,.2f}
• 盈虧: {pnl_pct:+.2f}%
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

📝 建議檢查策略並調整參數
        """

        return self.send_message(message.strip())

    def notify_take_profit(self, side: str, entry_price: float,
                          exit_price: float, pnl_pct: float,
                          position_id: Optional[str] = None) -> bool:
        """發送止盈通知"""

        emoji = "🟢" if side == "long" else "🔴"
        pos_line = f"\n🏷️ <b>倉位ID</b>: {position_id}" if position_id else ""

        message = f"""
🎉 <b>止盈觸發</b> 🎉

{emoji} <b>持倉類型</b>: {side.upper()}
{pos_line}

💰 <b>交易結果</b>
• 進場價: ${entry_price:,.2f}
• 出場價: ${exit_price:,.2f}
• 盈利: {pnl_pct:+.2f}%
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

✅ 恭喜獲利！
        """

        return self.send_message(message.strip())

    def notify_close_position(self, side: str, entry_price: float,
                             exit_price: float, pnl_pct: float,
                             reason: str = "手動平倉",
                             position_id: Optional[str] = None) -> bool:
        """發送平倉通知"""

        emoji = "🟢" if side == "long" else "🔴"
        result_emoji = "✅" if pnl_pct > 0 else "❌"
        pos_line = f"\n🏷️ <b>倉位ID</b>: {position_id}" if position_id else ""

        message = f"""
{result_emoji} <b>平倉通知</b>

{emoji} <b>持倉類型</b>: {side.upper()}
{pos_line}

💸 <b>交易結果</b>
• 進場價: ${entry_price:,.2f}
• 出場價: ${exit_price:,.2f}
• 盈虧: {pnl_pct:+.2f}%
• 原因: {reason}
• 時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        """

        return self.send_message(message.strip())

    def notify_error(self, error_msg: str) -> bool:
        """發送錯誤通知"""

        message = f"""
⚠️ <b>系統錯誤</b> ⚠️

❌ <b>錯誤訊息</b>
{error_msg}

🕐 <b>時間</b>
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

請檢查系統日誌！
        """

        return self.send_message(message.strip())

    def notify_startup(self, config_info: dict, current_price: float = None, positions: list = None) -> bool:
        """發送啟動通知"""

        price_info = ""
        if current_price:
            price_info = f"""
📊 <b>當前市場</b>
• 當前價格: ${current_price:,.2f}
• 數據來源: Binance
• 連接狀態: ✅ 正常

"""

        # 持仓信息
        position_info = ""
        if positions and len(positions) > 0:
            position_info = "\n📋 <b>當前持倉</b>\n"
            for pos in positions:
                side_emoji = "🟢" if pos.side == "long" else "🔴"
                pnl = ""
                if current_price:
                    if pos.side == "long":
                        pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                    else:
                        pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100
                    pnl = f" (浮動盈虧: {pnl_pct:+.2f}%)"

                position_info += f"{side_emoji} {pos.position_id} - {pos.side.upper()}\n"
                position_info += f"  • 進場: ${pos.entry_price:,.2f}{pnl}\n"
                position_info += f"  • 止損: ${pos.stop_loss:,.2f}\n"
                position_info += f"  • 止盈: ${pos.take_profit:,.2f}\n\n"
        else:
            position_info = "\n📋 <b>當前持倉</b>\n• 無持倉\n\n"

        message = f"""
🚀 <b>SMC 監控系統啟動</b> 🚀

⚙️ <b>配置資訊</b>
• 交易對: {config_info.get('symbol', 'N/A')}
• 交易模式: {config_info.get('mode', 'N/A')}
• 檢查間隔: {config_info.get('interval', 'N/A')} 秒
• 風險比例: {config_info.get('risk', 'N/A')}%

{price_info}{position_info}────────────────────
🆕 <b>2025/01/17 新功能</b>

📊 <b>/chart [時間]</b> - 查詢歷史交易
輸入指令生成 K 線圖，標記所有交易訊號

相對時間：
• /chart 7d - 過去 7 天
• /chart 30d - 過去 30 天
• /chart 3m - 過去 3 個月

指定日期範圍（最長一年）：
• /chart 2025-01-01~2025-01-31
• /chart 2024-06-01 2024-12-31

📋 <b>/status</b> - 查詢系統狀態
❓ <b>/help</b> - 查看所有指令

────────────────────
✅ 系統正在監控市場...
        """

        return self.send_message(message.strip())

    def notify_heartbeat(self, current_price: float, structure: str,
                        last_check_time: str) -> bool:
        """發送心跳通知（可選，用於確認系統運行）"""

        message = f"""
💓 <b>系統心跳</b>

📊 當前狀態
• 價格: ${current_price:,.2f}
• 市場結構: {structure}
• 最後檢查: {last_check_time}

✅ 系統運行正常
        """

        return self.send_message(message.strip())

    def send_photo(self, photo_path: str, caption: str = "", parse_mode: str = "HTML") -> bool:
        """
        發送圖片到所有配置的 chat_ids

        Args:
            photo_path: 圖片檔案路徑
            caption: 圖片說明
            parse_mode: 'HTML' | 'Markdown'

        Returns:
            是否至少成功發送給一個接收者
        """
        if not self.enabled:
            return False

        if not self.chat_ids:
            print("⚠️ 沒有配置 Telegram Chat IDs")
            return False

        if not os.path.exists(photo_path):
            print(f"⚠️ 圖片檔案不存在: {photo_path}")
            return False

        success_count = 0
        url = f"{self.base_url}/sendPhoto"

        for chat_id in self.chat_ids:
            try:
                with open(photo_path, 'rb') as photo_file:
                    files = {'photo': photo_file}
                    data = {
                        'chat_id': chat_id,
                        'caption': caption[:1024] if caption else "",  # Telegram 限制 1024 字元
                        'parse_mode': parse_mode
                    }

                    response = requests.post(url, data=data, files=files, timeout=30)

                    if response.status_code == 200:
                        success_count += 1
                    else:
                        print(f"❌ Telegram 發送圖片失敗 (Chat ID: {chat_id}): {response.text}")

            except Exception as e:
                print(f"❌ Telegram 發送圖片錯誤 (Chat ID: {chat_id}): {e}")

        return success_count > 0

    def register_command(self, command: str, handler: Callable[[str, str], None]) -> None:
        """
        註冊指令處理器

        Args:
            command: 指令名稱（不含斜線，如 'chart'）
            handler: 處理函數，接收 (chat_id, args) 參數
        """
        self._command_handlers[command.lower()] = handler
        print(f"✅ 已註冊指令: /{command}")

    def _process_update(self, update: dict) -> None:
        """處理單個 Telegram Update"""
        if 'message' not in update:
            return

        message = update['message']
        chat_id = str(message.get('chat', {}).get('id', ''))
        text = message.get('text', '')

        # 檢查是否是指令
        if text.startswith('/'):
            parts = text[1:].split(maxsplit=1)
            command = parts[0].lower()

            # 移除 @bot_username（如果有）
            if '@' in command:
                command = command.split('@')[0]

            args = parts[1] if len(parts) > 1 else ""

            if command in self._command_handlers:
                try:
                    print(f"📩 收到指令: /{command} {args} (from {chat_id})")
                    self._command_handlers[command](chat_id, args)
                except Exception as e:
                    print(f"❌ 處理指令 /{command} 時出錯: {e}")
                    self._send_to_chat(chat_id, f"⚠️ 處理指令時出錯: {str(e)}")

    def _send_to_chat(self, chat_id: str, text: str, parse_mode: str = "HTML") -> bool:
        """發送訊息到特定 chat_id"""
        if not self.enabled:
            return False

        try:
            url = f"{self.base_url}/sendMessage"
            payload = {
                'chat_id': chat_id,
                'text': text,
                'parse_mode': parse_mode
            }
            response = requests.post(url, json=payload, timeout=10)
            return response.status_code == 200
        except Exception as e:
            print(f"❌ 發送訊息到 {chat_id} 失敗: {e}")
            return False

    def _send_photo_to_chat(self, chat_id: str, photo_path: str, caption: str = "",
                            parse_mode: str = "HTML") -> bool:
        """發送圖片到特定 chat_id"""
        if not self.enabled:
            return False

        if not os.path.exists(photo_path):
            return False

        try:
            url = f"{self.base_url}/sendPhoto"
            with open(photo_path, 'rb') as photo_file:
                files = {'photo': photo_file}
                data = {
                    'chat_id': chat_id,
                    'caption': caption[:1024] if caption else "",
                    'parse_mode': parse_mode
                }
                response = requests.post(url, data=data, files=files, timeout=30)
                return response.status_code == 200
        except Exception as e:
            print(f"❌ 發送圖片到 {chat_id} 失敗: {e}")
            return False

    def _polling_loop(self) -> None:
        """Polling 迴圈，在背景執行緒中運行（含指數退避）"""
        print("🔄 開始 Telegram 指令監聽...")
        consecutive_errors = 0

        while self._polling_active:
            try:
                url = f"{self.base_url}/getUpdates"
                params = {
                    'offset': self._last_update_id + 1,
                    'timeout': 30,
                    'allowed_updates': ['message']
                }

                response = requests.get(url, params=params, timeout=35)

                if response.status_code == 200:
                    data = response.json()
                    if data.get('ok') and data.get('result'):
                        for update in data['result']:
                            update_id = update.get('update_id', 0)
                            if update_id > self._last_update_id:
                                self._last_update_id = update_id
                                self._process_update(update)
                    # 成功後重置錯誤計數
                    if consecutive_errors > 0:
                        print("✅ Telegram 連線已恢復")
                    consecutive_errors = 0

            except requests.exceptions.Timeout:
                # Long polling timeout，正常現象
                pass
            except Exception as e:
                consecutive_errors += 1
                # 指數退避：5s → 10s → 20s → 40s → 最多 60s
                wait_time = min(5 * (2 ** (consecutive_errors - 1)), 60)
                # 只在首次和每 10 次錯誤時印出，避免刷屏
                if consecutive_errors == 1 or consecutive_errors % 10 == 0:
                    print(f"⚠️ Polling 連線失敗（第 {consecutive_errors} 次）: {e}")
                    print(f"   下次重試等待 {wait_time}s")
                time.sleep(wait_time)

    def start_polling(self) -> None:
        """開始監聽 Telegram 指令（在背景執行緒中）"""
        if self._polling_active:
            print("⚠️ Polling 已經在運行中")
            return

        if not self.enabled or not self.bot_token:
            print("⚠️ Telegram 未啟用或未配置 Bot Token")
            return

        self._polling_active = True
        self._polling_thread = threading.Thread(target=self._polling_loop, daemon=True)
        self._polling_thread.start()

    def stop_polling(self) -> None:
        """停止監聽 Telegram 指令"""
        self._polling_active = False
        if self._polling_thread:
            self._polling_thread.join(timeout=5)
            self._polling_thread = None
        print("⏹️ 已停止 Telegram 指令監聽")

    def notify_daily_status(self, current_price: float, structure: str, positions: list = None) -> bool:
        """發送每日狀態通知"""

        # 持仓信息
        position_info = ""
        if positions and len(positions) > 0:
            position_info = "\n📋 <b>當前持倉</b>\n"
            for pos in positions:
                side_emoji = "🟢" if pos.side == "long" else "🔴"

                # 计算浮动盈亏
                if pos.side == "long":
                    pnl_pct = (current_price - pos.entry_price) / pos.entry_price * 100
                else:
                    pnl_pct = (pos.entry_price - current_price) / pos.entry_price * 100

                pnl_emoji = "📈" if pnl_pct > 0 else "📉" if pnl_pct < 0 else "➖"

                position_info += f"{side_emoji} <b>{pos.position_id}</b> - {pos.side.upper()}\n"
                position_info += f"  • 進場價: ${pos.entry_price:,.2f}\n"
                position_info += f"  • 止損價: ${pos.stop_loss:,.2f}\n"
                position_info += f"  • 止盈價: ${pos.take_profit:,.2f}\n"
                position_info += f"  • {pnl_emoji} 浮動盈虧: {pnl_pct:+.2f}%\n\n"
        else:
            position_info = "\n📋 <b>當前持倉</b>\n• 無持倉\n\n"

        message = f"""
🌅 <b>每日狀態報告</b>

📊 <b>市場狀態</b>
• 交易對: ETHUSDT
• 當前價格: ${current_price:,.2f}
• 市場結構: {structure}
• 報告時間: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

{position_info}✅ 系統運行正常，持續監控中...
        """

        return self.send_message(message.strip())


# ============ 測試用 ============
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_IDS, ENABLE_TELEGRAM

    notifier = TelegramNotifier(
        bot_token=TELEGRAM_BOT_TOKEN,
        chat_ids=TELEGRAM_CHAT_IDS,
        enabled=ENABLE_TELEGRAM
    )

    print("測試 Telegram 通知...")

    config = {
        'symbol': 'ETHUSDT',
        'mode': 'notify_only',
        'interval': 60,
        'risk': 1.0
    }

    success = notifier.notify_startup(config)

    if success:
        print("✅ Telegram 通知測試成功！")
    else:
        print("❌ Telegram 通知測試失敗！請檢查 BOT_TOKEN 和 CHAT_ID")
