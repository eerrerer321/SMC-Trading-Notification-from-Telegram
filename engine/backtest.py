# engine/backtest.py
# SMC 回測引擎
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from typing import List, Dict
from datetime import datetime

class TradeFinal:
    """交易记录类"""

    def __init__(self, entry_time, entry_price, direction, size,
                 stop_loss, take_profit):
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.size = size
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.original_stop_loss = stop_loss  # 保存原始止损用于计算R倍数

        # 资金占用（保证金/占用资金）
        self.margin = 0.0

        # 出场信息
        self.exit_time = None
        self.exit_price = None
        self.exit_reason = None

        # 盈亏信息
        self.pnl = 0
        self.pnl_percent = 0
        self.return_multiple = 0

    def close(self, exit_time, exit_price, exit_reason):
        """平仓"""
        self.exit_time = exit_time
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        # 计算盈亏
        if self.direction == 'long':
            self.pnl = (exit_price - self.entry_price) * self.size
            self.pnl_percent = (exit_price - self.entry_price) / self.entry_price
        else:  # short
            self.pnl = (self.entry_price - exit_price) * self.size
            self.pnl_percent = (self.entry_price - exit_price) / self.entry_price

        # 计算 R 倍数 - 使用原始止损
        risk = abs(self.entry_price - self.original_stop_loss)
        if risk > 0:
            actual_profit = exit_price - self.entry_price if self.direction == 'long' else self.entry_price - exit_price
            self.return_multiple = actual_profit / risk
        else:
            self.return_multiple = 0

    def to_dict(self):
        """转换为字典"""
        return {
            'entry_time': self.entry_time,
            'entry_price': self.entry_price,
            'direction': self.direction,
            'size': self.size,
            'stop_loss': self.original_stop_loss,
            'take_profit': self.take_profit,
            'exit_time': self.exit_time,
            'exit_price': self.exit_price,
            'exit_reason': self.exit_reason,
            'pnl': self.pnl,
            'pnl_percent': self.pnl_percent,
            'return_multiple': self.return_multiple
        }


class SMCBacktestFinal:
    """
    SMC 回測引擎

    特性：
    - 智能移動保本：到達 1.5R 時，止損移至入場價 +0.5% 利潤
    - 確保保本退出時也能覆蓋手續費
    """

    def __init__(self, strategy, backtest_params: dict, trading_params: dict):
        self.strategy = strategy
        self.backtest_params = backtest_params
        self.trading_params = trading_params

        self.initial_balance = trading_params.get('initial_balance', 10000)
        self.balance = float(self.initial_balance)
        self.used_margin = 0.0

        self.open_trades: List[TradeFinal] = []
        self.trades: List[TradeFinal] = []
        self.equity_curve = []

        # 記錄每個交易是否已經移動到保本
        self.breakeven_moved = {}

    def get_unrealized_pnl(self, mark_price: float) -> float:
        """计算当前未实现盈亏（使用 mark price）"""
        pnl = 0.0
        for trade in self.open_trades:
            if trade.direction == 'long':
                pnl += (mark_price - trade.entry_price) * trade.size
            else:
                pnl += (trade.entry_price - mark_price) * trade.size
        return float(pnl)

    def get_equity(self, mark_price: float) -> float:
        """账户权益 = 可用余额 + 已占用保证金 + 未实现盈亏"""
        return float(self.balance + self.used_margin + self.get_unrealized_pnl(mark_price))

    def apply_costs(self, trade_value: float) -> float:
        """计算交易成本（手续费 + 滑价）"""
        fee = trade_value * self.backtest_params.get('trading_fee', 0.0006)
        slippage = trade_value * self.backtest_params.get('slippage', 0.0005)
        return fee + slippage

    def open_trade(self, timestamp, entry_price, direction, stop_loss, take_profit):
        """开仓"""
        max_trades = self.trading_params.get('max_open_trades', 10)
        if len(self.open_trades) >= max_trades:
            return

        # 计算仓位大小
        equity = self.get_equity(entry_price)
        position_size = self.strategy.calculate_position_size(
            equity, entry_price, stop_loss
        )

        if position_size <= 0:
            return

        # 检查保证金/占用资金
        leverage = float(self.trading_params.get('leverage', 1))
        trade_value = float(position_size * entry_price)
        required_margin = trade_value / leverage if leverage > 0 else trade_value

        if required_margin > self.balance:
            return

        # 扣除手续费
        costs = self.apply_costs(trade_value)
        if costs > self.balance - required_margin:
            return
        self.balance -= (required_margin + costs)
        self.used_margin += required_margin

        # 创建交易
        trade = TradeFinal(
            entry_time=timestamp,
            entry_price=entry_price,
            direction=direction,
            size=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit
        )

        trade.margin = required_margin

        self.open_trades.append(trade)

    def close_trade(self, trade: TradeFinal, timestamp, exit_price, exit_reason):
        """平仓"""
        trade.close(timestamp, exit_price, exit_reason)

        # 扣除手续费
        trade_value = trade.size * exit_price
        costs = self.apply_costs(trade_value)

        # 更新余额（释放保证金 + 结算盈亏）
        self.balance += trade.margin + trade.pnl - costs
        self.used_margin -= trade.margin

        # 记录交易
        self.trades.append(trade)
        self.open_trades.remove(trade)

        # 清理breakeven记录
        trade_id = id(trade)
        if trade_id in self.breakeven_moved:
            del self.breakeven_moved[trade_id]

    def check_exit_conditions(self, trade: TradeFinal, current_bar) -> tuple:
        """
        檢查止損/止盈，支持智能移動保本

        特性：
        - 到達 1.5R 時，止損移至入場價 + 0.5% 利潤（Long）
        - 到達 1.5R 時，止損移至入場價 - 0.5% 利潤（Short）
        - 確保保本退出時也能覆蓋手續費

        Returns:
            (should_exit, exit_price, exit_reason)
        """
        high = current_bar['h']
        low = current_bar['l']

        # === 智能移動保本機制 ===
        trade_id = id(trade)

        if trade_id not in self.breakeven_moved:
            self.breakeven_moved[trade_id] = False

        if not self.breakeven_moved[trade_id]:
            # 計算是否到達 1.5R
            risk = abs(trade.entry_price - trade.original_stop_loss)
            target_1_5r = trade.entry_price + (1.5 * risk) if trade.direction == 'long' else trade.entry_price - (1.5 * risk)

            if trade.direction == 'long':
                # 做多：檢查是否觸及 1.5R
                if high >= target_1_5r:
                    # 移動止損到入場價 + 0.5%
                    new_stop = trade.entry_price * (1 + 0.005)  # +0.5%
                    trade.stop_loss = new_stop
                    self.breakeven_moved[trade_id] = True
            else:  # short
                # 做空：檢查是否觸及 1.5R
                if low <= target_1_5r:
                    # 移動止損到入場價 - 0.5%
                    new_stop = trade.entry_price * (1 - 0.005)  # -0.5%
                    trade.stop_loss = new_stop
                    self.breakeven_moved[trade_id] = True

        # === 检查止损和止盈 ===
        if trade.direction == 'long':
            # 先检查止损（使用low）
            if low <= trade.stop_loss:
                return (True, trade.stop_loss, 'sl')
            # 再检查止盈（使用high）
            if high >= trade.take_profit:
                return (True, trade.take_profit, 'tp')
        else:  # short
            # 先检查止损（使用high）
            if high >= trade.stop_loss:
                return (True, trade.stop_loss, 'sl')
            # 再检查止盈（使用low）
            if low <= trade.take_profit:
                return (True, trade.take_profit, 'tp')

        return (False, None, None)

    def run(self, df: pd.DataFrame) -> Dict:
        """運行回測"""
        print("\n" + "="*80)
        print("開始回測 SMC 策略")
        print("="*80)
        print(f"初始資金: ${self.initial_balance:,.2f}")
        print(f"槓桿倍數: {self.trading_params.get('leverage', 1)}x")
        print(f"每筆風險: {self.trading_params.get('risk_per_trade', 0.02)*100}%")
        print(f"風險報酬比: 1:{self.trading_params.get('risk_reward_ratio', 2.5)}")
        print("="*80)

        self.equity_curve = []

        for i in range(len(df)):
            current_bar = df.iloc[i]
            timestamp = df.index[i]

            # 检查现有持仓的止损/止盈（包括智能移动保本）
            for trade in self.open_trades[:]:
                should_exit, exit_price, exit_reason = self.check_exit_conditions(
                    trade, current_bar
                )
                if should_exit:
                    self.close_trade(trade, timestamp, exit_price, exit_reason)

            # 检查新信号
            signal = current_bar['signal']

            if signal == 1:  # 做多信号
                self.open_trade(
                    timestamp=timestamp,
                    entry_price=current_bar['entry_price'],
                    direction='long',
                    stop_loss=current_bar['stop_loss'],
                    take_profit=current_bar['take_profit']
                )

            elif signal == -1:  # 做空信号
                self.open_trade(
                    timestamp=timestamp,
                    entry_price=current_bar['entry_price'],
                    direction='short',
                    stop_loss=current_bar['stop_loss'],
                    take_profit=current_bar['take_profit']
                )

            # 记录权益曲线
            mark_price = float(current_bar['c'])
            equity = self.get_equity(mark_price)
            self.equity_curve.append({
                'timestamp': timestamp,
                'equity': equity,
                'balance': self.balance,
                'used_margin': self.used_margin,
                'open_trades': len(self.open_trades)
            })

        # 关闭所有剩余持仓
        if len(self.open_trades) > 0:
            last_price = df['c'].iloc[-1]
            last_time = df.index[-1]
            for trade in self.open_trades[:]:
                self.close_trade(trade, last_time, last_price, 'manual')

        print(f"\n[OK] 回测完成！")
        print(f"总交易次数: {len(self.trades)}")
        print(f"最终资金: ${self.balance:,.2f}")

        return self.calculate_metrics()

    def calculate_metrics(self) -> Dict:
        """计算绩效指标"""
        if len(self.trades) == 0:
            return {}

        # 转换为DataFrame
        trades_data = [t.to_dict() for t in self.trades]
        trades_df = pd.DataFrame(trades_data)

        # 基本统计
        total_trades = len(trades_df)
        wins = trades_df[trades_df['pnl'] > 0]
        losses = trades_df[trades_df['pnl'] <= 0]

        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades if total_trades > 0 else 0

        # 盈亏统计
        total_pnl = trades_df['pnl'].sum()
        total_return = (self.balance - self.initial_balance) / self.initial_balance

        avg_win = wins['pnl'].mean() if len(wins) > 0 else 0
        avg_loss = losses['pnl'].mean() if len(losses) > 0 else 0

        # R倍数统计
        avg_r = trades_df['return_multiple'].mean()

        # 盈利因子
        total_profit = wins['pnl'].sum() if len(wins) > 0 else 0
        total_loss = abs(losses['pnl'].sum()) if len(losses) > 0 else 0
        profit_factor = total_profit / total_loss if total_loss > 0 else 0

        # 期望值
        expectancy = (win_rate * avg_win) + ((1 - win_rate) * avg_loss)

        # 最大回撤
        equity_df = pd.DataFrame(self.equity_curve)
        if 'equity' in equity_df.columns:
            dd_base = equity_df['equity']
        else:
            dd_base = equity_df['balance']
        equity_df['cummax'] = dd_base.cummax()
        equity_df['drawdown'] = (dd_base - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()

        # Sharpe Ratio
        if len(trades_df) > 1:
            returns = pd.to_numeric(trades_df['pnl_percent'], errors='coerce').dropna()
            std = float(returns.std(ddof=0)) if len(returns) > 1 else 0.0
            sharpe_ratio = float(returns.mean()) / std if std >= 1e-9 else 0.0
        else:
            sharpe_ratio = 0

        metrics = {
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'avg_r': avg_r,
            'profit_factor': profit_factor,
            'expectancy': expectancy,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'final_balance': self.balance,
            'trades_df': trades_df,
            'equity_df': equity_df
        }

        return metrics

    def print_report(self, metrics: Dict):
        """打印回測報告"""
        print("\n" + "="*80)
        print("回測績效報告")
        print("="*80)

        print(f"\n【交易統計】")
        print(f"  總交易次數: {metrics['total_trades']}")
        print(f"  獲利次數: {metrics['win_count']}")
        print(f"  虧損次數: {metrics['loss_count']}")
        print(f"  勝率: {metrics['win_rate']*100:.2f}%")

        print(f"\n【盈虧分析】")
        print(f"  總盈虧: ${metrics['total_pnl']:,.2f}")
        print(f"  總報酬率: {metrics['total_return']*100:.2f}%")
        print(f"  平均獲利: ${metrics['avg_win']:,.2f}")
        print(f"  平均虧損: ${metrics['avg_loss']:,.2f}")
        print(f"  獲利因子: {metrics['profit_factor']:.2f}")
        print(f"  期望值: ${metrics['expectancy']:,.2f}")
        print(f"  平均 R 倍數: {metrics['avg_r']:.2f}R")

        print(f"\n【風險指標】")
        print(f"  最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")

        print(f"\n【最終結果】")
        print(f"  初始資金: ${self.initial_balance:,.2f}")
        print(f"  最終資金: ${metrics['final_balance']:,.2f}")
        print(f"  淨利潤: ${metrics['final_balance'] - self.initial_balance:,.2f}")

        print("="*80)
