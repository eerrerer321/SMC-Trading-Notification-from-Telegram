# config/settings.py
# SMC 交易系統統一配置文件
# -*- coding: utf-8 -*-

import os
import pandas as pd

# ============ 環境變數載入 ============
def _load_dotenv(dotenv_path: str) -> None:
    """從 .env 檔案載入環境變數"""
    if not os.path.exists(dotenv_path):
        return
    try:
        with open(dotenv_path, 'r', encoding='utf-8') as f:
            for raw_line in f:
                line = raw_line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    continue
                key, value = line.split('=', 1)
                key = key.strip().lstrip('\ufeff')
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def _get_env_first(*keys: str, default: str = "") -> str:
    """從多個環境變數名稱中取得第一個有值的"""
    for k in keys:
        v = os.getenv(k)
        if v is not None and str(v).strip() != "":
            return str(v).strip()
    return default


# 載入 .env 檔案
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
_ENV_CANDIDATES = [
    os.getenv('SMC_ENV_PATH', '').strip(),
    os.path.join(_PROJECT_ROOT, '.env'),
    os.path.join(_PROJECT_ROOT, '.env.local'),
]
for _p in _ENV_CANDIDATES:
    if _p and os.path.exists(_p):
        _load_dotenv(_p)


# ============ 交易對配置 ============
SYMBOL = "ETHUSDT"
BASE_TIMEFRAME = "15m"      # 基礎數據週期
TRADING_TIMEFRAME = "4h"    # 交易分析週期
DATA_PATH = "ETHUSDT_15m_2023_now.csv"


# ============ 交易所配置 ============
EXCHANGE = "binance"  # 'binance' | 'bybit'
BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.getenv('BINANCE_API_SECRET', '')
BYBIT_API_KEY = os.getenv('BYBIT_API_KEY', '')
BYBIT_API_SECRET = os.getenv('BYBIT_API_SECRET', '')


# ============ Telegram 配置 ============
TELEGRAM_BOT_TOKEN = _get_env_first(
    'TELEGRAM_BOT_TOKEN', 'TG_BOT_TOKEN', 'BOT_TOKEN', default=''
)

_chat_ids_str = _get_env_first(
    'TELEGRAM_CHAT_IDS', 'TELEGRAM_CHAT_ID', 'TG_CHAT_ID', 'CHAT_ID', default=''
)
TELEGRAM_CHAT_IDS = [cid.strip() for cid in _chat_ids_str.split(',') if cid.strip()]

ENABLE_TELEGRAM = os.getenv('ENABLE_TELEGRAM', 'true').lower() in ('1', 'true', 'yes')
NOTIFY_ON_SIGNAL = True
NOTIFY_ON_STOP_LOSS = True
NOTIFY_ON_TAKE_PROFIT = True
NOTIFY_ON_ERROR = True


# ============ 監控設置 ============
CHECK_INTERVAL = 180  # 檢查間隔（秒）
DATA_LOOKBACK = 2000  # 回看 K 線數量
TRADING_MODE = 'notify_only'  # 'notify_only' | 'semi_auto' | 'full_auto'


# ============ SMC 指標參數 ============
SMC_PARAMS = {
    'swing_lookback': 5,        # Swing 回看週期
    'swing_strength': 2,        # Swing 強度（左右各需要幾根 K 線）
    'ob_lookback': 20,          # Order Block 回看範圍
    'ob_buffer': 0.001,         # Order Block 緩衝區（0.1%）
    'ob_max_age': 50,           # OB 最大有效期（根 K 線）
    'bos_confirmation': 1,      # BOS 確認需要幾根 K 線收在結構外
    'structure_buffer': 0.0005, # 結構突破緩衝（0.05%）
    'fvg_min_size': 0.002,      # FVG 最小尺寸（0.2%）
    'fvg_enabled': True,        # 是否啟用 FVG
}


# ============ 策略參數 ============
STRATEGY_PARAMS = {
    # 策略資訊
    'strategy_name': 'SMC MTF Strategy',
    'strategy_version': 'v2.0',

    # 市場結構過濾
    'enable_trend_filter': False,        # [優化] 關閉趨勢過濾，增加進場機會
    'enable_structure_filter': True,     # 啟用市場結構過濾
    'min_structure_move_pct': 0.02,      # 最小結構移動百分比 2%

    # Order Block 質量過濾
    'ob_quality_threshold': 45,          # [優化] 降低 OB 門檻 50 -> 45
    'ob_volume_percentile': 0.70,        # 成交量要求

    # RSI 過濾
    'enable_rsi_filter': True,
    'rsi_long_max': 85,                  # 做多：RSI 不超過 85
    'rsi_short_min': 15,                 # 做空：RSI 不低於 15
    'rsi_period': 14,

    # 進場確認（1H）
    'entry_candle_body_ratio': 0.5,      # [優化] K 線實體比例 ≥50%（原 60%）
    'entry_volume_threshold': 0.50,      # [優化] 成交量門檻：≥平均的50%（原 70%）

    # 止損設置
    'stop_loss_lookback': 10,            # 止損回看週期
    'stop_loss_buffer_pct': 0.08,        # 止損緩衝 8%

    # 止盈設置
    'risk_reward_ratio': 4.5,            # [優化] 風險報酬比 1:4.5（原 3.5）

    # 移動保本機制
    'enable_breakeven': True,
    'breakeven_trigger_r': 2.5,          # [優化] 到達 2.5R 時觸發（原 1.5）
    'breakeven_profit_pct': 0.003,       # [優化] 移動到 +0.3% 利潤位置（原 0.5%）

    # 倉位管理
    'max_open_trades': 10,
    'allow_neutral_market': True,        # 允許中性市場交易

    # 回撤確認（防止追高殺低）
    'enable_pullback_confirmation': True,
    'pullback_lookback': 20,
    'min_pullback_pct': 0.20,            # [優化] 降低回撤要求 0.25 -> 0.20

    # 首次觸及 OB 進場
    'allow_first_touch_ob': False,
    'first_touch_momentum': 0.008,

    # 即時監控：信號時效控制
    'max_signal_age_bars': 2,            # 只掃描最近 N 根 1H K線產生信號（防止回溯信號）
    'max_price_deviation_pct': 0.01,     # 價格偏離超過 2% 時標記為「已偏離」
}


# ============ 交易參數 ============
TRADING_PARAMS = {
    # 進場條件
    'require_trend': True,
    'require_ob': True,
    'require_bos': True,
    'require_fvg': False,

    # 風險管理
    'risk_per_trade': float(os.getenv('RISK_PER_TRADE', '0.017')),  # 每筆交易風險 1.7%
    'max_open_trades': int(os.getenv('MAX_OPEN_POSITIONS', '10')),
    'leverage': int(os.getenv('LEVERAGE', '1')),

    # 止損/止盈
    'stop_loss_type': 'structure',       # 'structure' | 'atr' | 'fixed'
    'atr_buffer_multiplier': 0.5,
    'stop_loss_buffer': 0.05,
    'take_profit_type': 'rr',            # 'rr' | 'structure' | 'fixed'

    # 資金管理
    'initial_balance': 10000,
    'position_sizing': 'fixed_percent',  # 'risk_based' | 'fixed_percent'
    'fixed_notional_pct': 0.10,
}

# 合併策略參數到交易參數
TRADING_PARAMS.update(STRATEGY_PARAMS)


# ============ 回測參數 ============
BACKTEST_PARAMS = {
    'trading_fee': 0.0006,      # 手續費 0.06%
    'slippage': 0.0005,         # 滑價 0.05%
    'use_4h_candle': True,
}


# ============ 輸出設定 ============
OUTPUT_FILES = {
    'trades_csv': 'smc_trades.csv',
    'equity_csv': 'smc_equity.csv',
    'summary_txt': 'smc_summary.txt',
}


def get_project_root() -> str:
    """獲取專案根目錄"""
    return _PROJECT_ROOT


def print_config():
    """打印當前配置"""
    print("=" * 80)
    print("SMC 交易系統配置")
    print("=" * 80)
    print(f"交易對: {SYMBOL}")
    print(f"基礎週期: {BASE_TIMEFRAME} → 交易週期: {TRADING_TIMEFRAME}")
    print(f"\nSMC 指標參數:")
    for key, val in SMC_PARAMS.items():
        print(f"  {key}: {val}")
    print(f"\n策略參數:")
    for key, val in STRATEGY_PARAMS.items():
        print(f"  {key}: {val}")
    print("=" * 80)


if __name__ == "__main__":
    print_config()
