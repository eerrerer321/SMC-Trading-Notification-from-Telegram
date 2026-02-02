# review.py
# SMC 歷史回顧主程式 - 生成交易圖表和統計報告
# -*- coding: utf-8 -*-

import sys
import os
import io
import argparse
from datetime import datetime, timedelta

# 設定 stdout 編碼以支援 Unicode（解決 Windows 終端機的 emoji 顯示問題）
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

sys.path.insert(0, os.path.dirname(__file__))

from config.settings import SYMBOL
from notification.chart import SMCChartGenerator


def main():
    """主函數"""
    parser = argparse.ArgumentParser(description='SMC 歷史交易回顧')
    parser.add_argument(
        'time_range',
        nargs='?',
        default='30d',
        help='時間範圍（例如：7d, 30d, 3m, 2025-01-01~2025-01-31）'
    )
    parser.add_argument(
        '--symbol',
        default=SYMBOL,
        help=f'交易對（預設：{SYMBOL}）'
    )
    parser.add_argument(
        '--output',
        default=None,
        help='輸出圖片路徑（預設：自動生成）'
    )
    parser.add_argument(
        '--no-chart',
        action='store_true',
        help='不生成圖表，只顯示統計'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("📊 SMC 歷史交易回顧")
    print("=" * 80)
    print(f"交易對: {args.symbol}")
    print(f"時間範圍: {args.time_range}")
    print("=" * 80)
    print()

    try:
        generator = SMCChartGenerator()

        # 解析時間範圍
        days, start_date, end_date = generator.parse_time_range(args.time_range)
        print(f"正在獲取數據（約 {days} 天）...")

        # 獲取數據並生成信號
        df_15m, df_4h = generator.fetch_historical_data(args.symbol, days)

        if df_15m is None or df_4h is None:
            print("❌ 數據獲取失敗！")
            return

        print(f"  15m K 線: {len(df_15m)} 根")
        print(f"  4H K 線: {len(df_4h)} 根")
        print()

        # 生成信號
        print("正在分析交易信號...")
        df_1h = generator.generate_signals(df_15m, df_4h)

        # 模擬交易
        print("正在模擬交易...")
        trades = generator.simulate_trades(df_1h, df_4h)

        # 過濾時間範圍
        if start_date and end_date:
            trades = [t for t in trades if start_date <= t.entry_time <= end_date]
        elif days:
            cutoff = datetime.now() - timedelta(days=days)
            trades = [t for t in trades if t.entry_time >= cutoff]

        print(f"\n找到 {len(trades)} 筆交易")
        print()

        # 計算統計
        if trades:
            completed = [t for t in trades if t.exit_reason != 'open']
            open_trades = [t for t in trades if t.exit_reason == 'open']
            wins = [t for t in completed if t.pnl_pct and t.pnl_pct > 0]
            losses = [t for t in completed if t.pnl_pct and t.pnl_pct <= 0]

            total_pnl = sum(t.pnl_pct for t in completed if t.pnl_pct)
            win_rate = len(wins) / len(completed) * 100 if completed else 0

            # 複利計算
            equity = 1.0
            peak_equity = 1.0
            max_drawdown = 0.0
            total_profit = 0.0
            total_loss = 0.0

            for t in completed:
                if t.pnl_pct:
                    pnl_ratio = t.pnl_pct / 100.0
                    equity *= (1 + pnl_ratio)

                    if equity > peak_equity:
                        peak_equity = equity
                    drawdown = (peak_equity - equity) / peak_equity
                    if drawdown > max_drawdown:
                        max_drawdown = drawdown

                    if t.pnl_pct > 0:
                        total_profit += t.pnl_pct
                    else:
                        total_loss += abs(t.pnl_pct)

            compound_return = (equity - 1.0) * 100
            profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')

            print("=" * 80)
            print("📊 統計摘要")
            print("=" * 80)
            print(f"  總交易數: {len(trades)} 筆")
            print(f"  已完成: {len(completed)} 筆")
            print(f"  持倉中: {len(open_trades)} 筆")
            print(f"  勝率: {win_rate:.1f}% ({len(wins)} 勝 {len(losses)} 敗)")
            print(f"  名目報酬: {total_pnl:+.2f}%")
            print(f"  複利報酬: {compound_return:+.2f}%")
            print(f"  最大回撤: {max_drawdown * 100:.2f}%")
            if profit_factor != float('inf'):
                print(f"  獲利因子: {profit_factor:.2f}")
            else:
                print(f"  獲利因子: ∞（無虧損）")
            print("=" * 80)
            print()

            # 顯示每筆交易
            print("交易明細:")
            print("-" * 80)
            for i, t in enumerate(trades):
                direction_emoji = "🟢" if t.direction == 'long' else "🔴"
                direction_text = "做多" if t.direction == 'long' else "做空"

                if t.exit_reason == 'open':
                    status = "持倉中"
                    pnl_str = ""
                elif t.exit_reason == 'tp':
                    status = "止盈"
                    pnl_str = f" ({t.pnl_pct:+.2f}%)"
                elif t.exit_reason in ('sl', 'breakeven_sl'):
                    status = "止損" if t.exit_reason == 'sl' else "保本止損"
                    pnl_str = f" ({t.pnl_pct:+.2f}%)"
                else:
                    status = t.exit_reason
                    pnl_str = f" ({t.pnl_pct:+.2f}%)" if t.pnl_pct else ""

                print(f"{direction_emoji} #{i+1} {direction_text} | "
                      f"進場: {t.entry_time.strftime('%Y-%m-%d %H:%M')} ${t.entry_price:,.2f} | "
                      f"{status}{pnl_str}")

            print("-" * 80)

        # 生成圖表
        if not args.no_chart and trades:
            print("\n正在生成圖表...")

            if args.output:
                output_path = args.output
            else:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"output/smc_review_{args.time_range}_{timestamp}.png"

            os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else 'output', exist_ok=True)

            fig = generator.create_chart(df_4h, trades)
            fig.write_image(output_path, width=1600, height=900, scale=2)

            print(f"✅ 圖表已保存至: {output_path}")

    except Exception as e:
        print(f"❌ 發生錯誤: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
