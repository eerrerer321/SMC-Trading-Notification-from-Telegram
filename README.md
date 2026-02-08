# SMC 加密貨幣交易系統

> **Smart Money Concepts (SMC) 自動監控與通知系統**
> 基於機構資金行為的多時間框架交易策略

---

## 系統概述

### 核心功能

1. **實時監控**: 定時檢查市場，檢測新 K 線時觸發分析
2. **SMC 信號檢測**: 多時間框架分析（15m → 1H → 4H）
3. **Telegram 通知**: 繁體中文通知，包含完整進出場計劃
4. **風險管理**: 自動計算止損/止盈，智能移動保本
5. **僅通知模式**: 不自動交易，人工確認後再進場

### 通知策略績效參考（ETHUSDT 2020-2025，參數重優化）

**訓練集（2023-2025）- 用於參數優化**

| 年份 | 報酬率 | 交易數 | 勝率 | MDD | PF |
|------|--------|--------|------|-----|-----|
| 2023 | **+61.78%** | 81 | 33.3% | 22.7% | 1.61 |
| 2024 | **+50.84%** | 88 | 37.5% | 32.7% | 1.49 |
| 2025 | **+182.11%** | 97 | 40.2% | 32.1% | 1.87 |
| **合計** | **+294.73%** | | avg 37.0% | | avg 1.66 |

**驗證集（2020-2022）- 未參與優化，檢驗泛化能力**

| 年份 | 報酬率 | 交易數 | 勝率 | MDD | PF |
|------|--------|--------|------|-----|-----|
| 2020 | **+315.79%** | 83 | 27.7% | 47.2% | 2.09 |
| 2021 | **+20.09%** | 106 | 28.3% | 57.9% | 1.19 |
| 2022 | **+6.79%** | 108 | 29.6% | 36.1% | 1.14 |
| **合計** | **+342.67%** | | avg 28.5% | | avg 1.47 |

**六年總計**: 訓練集 +294.73% / 驗證集 +342.67% / 全部 **+637.40%**

> **注意**: 歷史績效不代表未來表現，僅供參考。本段為通知策略參數優化後的歷史模擬結果（`output/notify_best_params.json`）。策略採高盈虧比（1:4.5）低勝率風格，2022 已轉為小幅正報酬但仍屬弱勢年份。

---

## 專案結構

```
SMC - 帶圖表版/
├── config/                    # 配置參數
│   ├── __init__.py
│   └── settings.py            # 所有參數設定（SMC、交易、Telegram等）
│
├── strategy/                  # 交易策略
│   ├── __init__.py
│   ├── indicators.py          # SMC 技術指標（Swing Points, OB, BOS/CHoCH, FVG）
│   └── smc_strategy.py        # SMC 交易策略（信號生成邏輯）
│
├── engine/                    # 引擎
│   ├── __init__.py
│   ├── data_fetcher.py        # 數據獲取（CCXT 交易所連接）
│
├── notification/              # 通知
│   ├── __init__.py
│   ├── telegram.py            # Telegram 通知
│   └── chart.py               # 圖表生成（Plotly）
│
├── tools/                     # 開發工具
│   ├── test_all_years.py      # 全年份回測
│   ├── find_optimal.py        # 參數優化搜索
│   ├── optimize_notify_strategy.py  # 通知策略參數優化
│   └── fetch_historical.py    # 歷史數據抓取
│
├── data/                      # 歷史數據
├── output/                    # 輸出圖表
│
├── monitor.py                 # 主程式：實時監控
├── review.py                  # 主程式：歷史回顧
├── 启动监控.bat               # Windows 啟動腳本
│
├── .env                       # 環境變數（Telegram 配置）
├── .env.example               # 環境變數範例
└── requirements.txt           # Python 依賴套件
```

---

## 快速開始

### 1. 安裝依賴

```bash
pip install -r requirements.txt
```

### 2. 配置環境變數

複製 `.env.example` 為 `.env`，填入你的 Telegram 配置：

```bash
# Telegram 配置
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_IDS=your_chat_id_here

# 可選：多個 Chat ID 用逗號分隔
# TELEGRAM_CHAT_IDS=123456789,987654321
```

#### 如何獲取 Telegram 配置？

1. **Bot Token**: 在 Telegram 搜尋 `@BotFather`，發送 `/newbot` 創建機器人
2. **Chat ID**: 在 Telegram 搜尋 `@userinfobot`，點擊 START 獲取你的 Chat ID

### 3. 啟動監控

**方法 1: 雙擊啟動（Windows）**
```
雙擊 启动监控.bat
```

**方法 2: 命令行**
```bash
python monitor.py
```

### 4. 歷史回顧

查看過去的交易信號和統計：

```bash
# 回顧最近 30 天（預設）
python review.py

# 回顧最近 7 天
python review.py 7d

# 回顧最近 3 個月
python review.py 3m

# 回顧指定日期範圍
python review.py 2025-01-01~2025-01-15

# 只顯示統計，不生成圖表
python review.py 30d --no-chart

# 指定其他交易對
python review.py 30d --symbol BTCUSDT
```

---

## SMC 交易策略

### 什麼是 SMC？

**Smart Money Concepts (SMC)** 是一種追蹤「聰明錢」（機構資金）行為的交易方法。核心理念是：機構資金進出市場會留下「痕跡」，散戶可以透過識別這些痕跡來跟隨機構的方向交易。

### 核心概念

#### 1. Market Structure（市場結構）

- **Higher High (HH)** + **Higher Low (HL)** → 上升趨勢
- **Lower High (LH)** + **Lower Low (LL)** → 下降趨勢

#### 2. Order Block (OB, 訂單區塊)

機構大量下單的價格區域，價格回測此區會獲得支撐/阻力。

#### 3. Break of Structure (BOS)

價格突破前一個重要的高點或低點，確認趨勢延續。

#### 4. Change of Character (CHoCH)

價格突破結構的反向點，預示趨勢可能反轉。

### 多時間框架分析

```
15分鐘 (15m) ──聚合→ 1小時 (1H) ──聚合→ 4小時 (4H)
    ↓                    ↓                  ↓
 基礎數據            進場時機            市場結構
```

| 時間框架 | 作用 |
|---------|------|
| **4H** | 大方向判斷（市場結構、OB 識別、趨勢方向） |
| **1H** | 進場時機（價格回測 OB 確認、RSI/成交量過濾） |
| **15m** | 數據基礎（原始 OHLCV 數據） |

---

## 交易邏輯

### 做多條件（全部滿足）

| 條件 | 說明 |
|------|------|
| 4H 趨勢 | 上升趨勢或中性市場 |
| 4H 結構 | 市場結構為 bullish |
| 高質量 OB | 存在符合質量門檻的看漲 OB |
| 價格回測 | 1H 低點觸及 OB 範圍 |
| RSI 過濾 | RSI < 85 |
| 成交量確認 | 成交量 ≥ 平均的 50% |
| K 線實體 | 實體比例 ≥ 50% |

### 出場策略

- **止損**: 過去 10 根 K 線最低點 - 8% 緩衝
- **止盈**: 風險 × 4.5 (RR 比 1:4.5)
- **移動保本**: 獲利達 2.5R 時，止損移至 +0.3% 位置

---

## 參數配置

所有參數集中在 `config/settings.py`：

### SMC 指標參數

```python
SMC_PARAMS = {
    'swing_lookback': 5,       # Swing 點回看週期
    'swing_strength': 2,       # Swing 點強度
    'ob_lookback': 20,         # OB 回看範圍
    'ob_max_age': 50,          # OB 最大有效期
    'structure_buffer': 0.0005, # 結構突破緩衝
    'fvg_min_size': 0.002,     # FVG 最小尺寸
    'fvg_enabled': True,       # 啟用 FVG
}
```

### 策略參數

```python
STRATEGY_PARAMS = {
    'enable_trend_filter': False,    # [優化] 關閉趨勢過濾
    'ob_quality_threshold': 45,      # OB 質量門檻
    'min_pullback_pct': 0.20,        # [優化] 回撤要求 0.25→0.20
    'risk_reward_ratio': 4.5,        # [優化] 風險報酬比 3.5→4.5
    'allow_neutral_market': True,    # 允許中性市場交易
    'enable_pullback_confirmation': True,  # 啟用回測確認
    'entry_candle_body_ratio': 0.45,  # [優化] long 實體比例
    'entry_volume_threshold': 0.55,   # [優化] long 成交量門檻
    'entry_candle_body_ratio_short': 0.60,  # short 實體比例
    'entry_volume_threshold_short': 0.60,   # short 成交量門檻
    'signal_cooldown_bars': 2,        # [優化] 同 OB 信號冷卻
    'breakeven_trigger_r': 2.5,      # [優化] 保本觸發 1.5R→2.5R
    'breakeven_profit_pct': 0.003,   # [優化] 保本後利潤 0.5%→0.3%
    'stop_loss_buffer_pct': 0.08,    # 止損緩衝 8%
}
```

### 交易參數

```python
TRADING_PARAMS = {
    'risk_per_trade': 0.017,   # 每筆風險 1.7%
    'max_open_trades': 10,     # 最大持倉數
    'leverage': 1,             # 槓桿倍數
}
```

---

## 風險管理

### 倉位計算

```
單筆風險金額 = 總資金 × 1.7%
持倉大小 = 單筆風險金額 / (進場價 - 止損價)
```

### 風險配置建議

| 類型 | risk_per_trade | 說明 |
|------|---------------|------|
| 保守型 | 1.0% - 1.5% | 新手建議 |
| 標準型 | 1.5% - 2.0% | 一般配置 |
| 積極型 | 2.0% - 2.5% | 經驗豐富者 |

---

## Telegram 通知

### 通知類型

1. **系統啟動**: 監控開始運行
2. **做多/做空信號**: 包含進場價、止損、止盈
3. **止損/止盈觸發**: 交易結束通知
4. **錯誤通知**: 系統異常

### 通知範例

```
🟢 SMC 做多信號 🟢

📊 基本資訊
• 交易對: ETHUSDT
• 價格: $3,365.02
• 時間: 2025-12-09 16:00:00

📈 技術分析
• 市場結構: bullish
• RSI: 72.5
• Order Block: 4H_Bull_OB_3320.50-3340.00

💰 交易計劃
• 進場價: $3,365.02
• 止損價: $3,096.22 (-7.99%)
• 止盈價: $4,306.84 (+27.98%)
• 風險報酬比: 1:4.50
```

---

## 常見問題

### Q: 需要交易所 API 密鑰嗎？

**A**: 不需要。系統使用 Binance 公開 API 獲取市場數據，完全免費。

### Q: 可以監控其他幣種嗎？

**A**: 可以。修改 `config/settings.py` 中的 `SYMBOL`：

```python
SYMBOL = "BTCUSDT"  # 改為 BTC
```

### Q: 系統會自動交易嗎？

**A**: 不會。系統僅發送 Telegram 通知，需要手動確認後在交易所下單。

### Q: 為什麼勝率只有約 30-40%？

**A**: 勝率不是唯一重要的指標。策略採高盈虧比風格，配合 4.5 倍的風險報酬比，即使勝率 35%，期望值仍為正：
```
0.35 × 4.5 - 0.65 × 1 = +0.925（每筆平均賺 0.925 倍風險）
```

---

## 風險警告

⚠️ **重要聲明**

1. **歷史表現不代表未來結果**
2. 系統僅供通知參考，非投資建議
3. 加密貨幣市場波動極大，可能造成重大損失
4. 建議先用小資金測試，熟悉後再調整
5. 永遠不要投入無法承受損失的資金

---

## 更新日誌

### v2.3 - 2026-02-09

- **Notification strategy re-optimization**:
  - Added `tools/optimize_notify_strategy.py` (243 combinations, multiprocessing)
  - Optimization finished in ~44 minutes, output saved to `output/notify_best_params.json`
- **Performance update (ETHUSDT 2020-2025)**:
  - Train (2023-2025): **+294.73%**
  - Validation (2020-2022): **+342.67%**
  - Total (6 years): **+637.40%**
- **Updated key parameters**:
  - `entry_candle_body_ratio`: 0.50 -> **0.45**
  - `entry_volume_threshold`: 0.50 -> **0.55**
  - `entry_volume_threshold_short`: 0.70 -> **0.60**
  - `signal_cooldown_bars`: 1 -> **2**

### v2.2 - 2026-02-08

- **修復前瞻偏差 (Look-ahead Bias)**:
  - `indicators.py`: 擺動點移至確認位置 `i + strength`，不再使用未來資料
  - `smc_strategy.py`: `identify_key_structure_points()` 只使用向後看條件
  - 回測結果更接近實盤表現
- **信號時效修復**:
  - 新增 `signal_lookback` 參數，即時監控只掃描最近 N 根 K 線
  - 通知加入即時市場價格、價格偏離警告、移動止損觸發價格
- **參數重新優化（兩階段 + 樣本外驗證）**:
  - 訓練集 2023-2025（3,485 組合，11 進程並行搜索）
  - 驗證集 2020-2022（未參與優化，檢驗泛化能力）
  - 績效衰退 35.6%，策略泛化能力可接受
  - 優化後參數:
    - `risk_reward_ratio`: 3.5 → **4.5**（更大盈虧比）
    - `breakeven_trigger_r`: 1.5 → **2.5**（保本觸發更晚，讓利潤奔跑）
    - `breakeven_profit_pct`: 0.005 → **0.003**（保本位更緊）
    - `entry_candle_body_ratio`: 0.6 → **0.5**（放寬進場條件）
    - `entry_volume_threshold`: 0.7 → **0.5**（放寬成交量門檻）
    - `min_pullback_pct`: 0.25 → **0.20**（降低回撤要求）
- **新增工具**: `tools/optimize_v2.py` - 多進程並行參數優化 + 樣本外驗證

### v2.1 - 2026-02-03

- **參數優化**: 透過 105 種參數組合搜索，找到所有年份 (2020-2025) 報酬率均 ≥30% 的最佳配置
- **優化內容**:
  - 關閉趨勢過濾 (`enable_trend_filter: False`) - 增加進場機會
  - 降低 OB 門檻 (`ob_quality_threshold: 50→45`)
  - 降低回撤要求 (`min_pullback_pct: 0.3→0.25`)
  - 提高風險報酬比 (`risk_reward_ratio: 2.5→3.5`)
- **新增工具** (`tools/` 資料夾):
  - `test_all_years.py` - 全年份測試
  - `find_optimal.py` - 參數優化搜索
  - `fetch_historical.py` - 歷史數據抓取

### v2.0 - 2026-01-19

- 重構專案結構，按功能明確分類
- 統一配置文件至 `config/settings.py`
- 新增 `review.py` 歷史回顧工具
- 優化程式碼結構，移除重複功能

### v1.0 - 2025-12-10

- 完整 SMC 交易策略實現
- 多時間框架分析（15m/1H/4H）
- 實時監控系統
- Telegram 通知（繁體中文）
- 智能移動保本機制

---

## 授權

本專案僅供個人學習和研究使用。
