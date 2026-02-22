#!/usr/bin/env python3
"""
多股票回测脚本

基于 const.py 中的 STOCK_LIST 股票池进行回测
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path

# 导入常量配置
from const import (
    STOCK_LIST, INITIAL_CAPITAL, MAX_POSITION, MIN_POSITION,
    SINGLE_TRADE_RATIO, COMMISSION_RATE, STAMP_TAX_RATE, MIN_TRADE_UNIT,
    STOCK_DB_PATH, get_full_stock
)

# 输出文件路径
OUTPUT_FILE = Path(__file__).parent / "eval_output.txt"

# 数据库配置›
DB_PATH = STOCK_DB_PATH
TABLE_NAME = "stock_daily"

# ==================== 回测参数配置（可修改） ====================

# 回测时间范围
START_DATE = "2023-06-01"    # 回测开始日期 (YYYY-MM-DD)
END_DATE = "2024-06-01"      # 回测结束日期 (YYYY-MM-DD)

# 股票池（从 const.py 导入，也可在此覆盖）
# 如果 STOCK_LIST 为空，则自动获取全部股票
STOCK_POOL = STOCK_LIST


def get_stock_pool():
    """获取股票池"""
    if STOCK_POOL and len(STOCK_POOL) > 0:
        print(f"使用自定义股票池，共 {len(STOCK_POOL)} 只")
        return STOCK_POOL
    else:
        print("STOCK_LIST 为空，自动获取全部股票列表...")
        code_list, code_name_map = get_full_stock()
        print(f"获取到 {len(code_list)} 只股票")
        return code_list, code_name_map

# 初始资金 (RMB)
INITIAL_CASH = INITIAL_CAPITAL

# 每只股票最大持仓比例（相对于总资金）
MAX_STOCK_POSITION = 0.2     # 单只股票最多占用 20% 资金

# ==================== 回测类定义 ====================


@dataclass
class StockPosition:
    """股票持仓"""
    code: str
    shares: int = 0
    avg_cost: float = 0.0

    @property
    def market_value(self, price: float = 0) -> float:
        return self.shares * price


@dataclass
class DailyRecord:
    """每日记录"""
    date: str
    positions: Dict[str, StockPosition] = field(default_factory=dict)
    prices: Dict[str, float] = field(default_factory=dict)
    trades: List[Dict] = field(default_factory=list)
    total_commission: float = 0.0
    total_stamp_tax: float = 0.0
    cash: float = 0.0
    total_value: float = 0.0
    cumulative_return: float = 0.0


class MultiStockBacktestEngine:
    """多股票回测引擎"""

    def __init__(self, stock_pool: List[str], start_date: str, end_date: str, initial_cash: float, code_name_map: Dict[str, str] = None):
        self.stock_pool = stock_pool
        self.start_date = start_date
        self.end_date = end_date
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions: Dict[str, StockPosition] = {code: StockPosition(code) for code in stock_pool}
        self.records: List[DailyRecord] = []
        self.trade_count = 0
        self.daily_prices: Dict[str, Dict[str, float]] = {}  # date -> {code: price}
        self.code_name_map = code_name_map or {}  # 代码到名称的映射
        self.output_file = None  # 输出文件句柄

    def get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        return self.code_name_map.get(code, code)

    def open_output_file(self):
        """打开输出文件"""
        self.output_file = open(OUTPUT_FILE, 'w', encoding='utf-8')

    def close_output_file(self):
        """关闭输出文件"""
        if self.output_file:
            self.output_file.close()

    def print_and_write(self, text: str = ""):
        """同时打印到控制台和写入文件"""
        print(text)
        if self.output_file:
            self.output_file.write(text + '\n')
            self.output_file.flush()

    def load_all_data(self):
        """加载所有股票的历史数据"""
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        for code in self.stock_pool:
            cursor.execute(f"""
                SELECT date, close FROM {TABLE_NAME}
                WHERE code = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
            """, (code, self.start_date, self.end_date))

            for row in cursor.fetchall():
                date = row["date"]
                price = row["close"]
                if date not in self.daily_prices:
                    self.daily_prices[date] = {}
                self.daily_prices[date][code] = price

        conn.close()

    def get_trading_days(self) -> List[str]:
        """获取所有交易日列表（取所有股票交易日的并集）"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # 获取在日期范围内有任意股票数据的所有交易日
        cursor.execute(f"""
            SELECT DISTINCT date FROM {TABLE_NAME}
            WHERE code IN ({','.join(['?' for _ in self.stock_pool])})
            AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, self.stock_pool + [self.start_date, self.end_date])

        days = [row[0] for row in cursor.fetchall()]
        conn.close()
        return days

    def calculate_ma(self, code: str, date: str, period: int) -> float:
        """计算移动平均线"""
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT close FROM {TABLE_NAME}
            WHERE code = ? AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """, (code, date, period))

        prices = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(prices) < period:
            return prices[0] if prices else 0
        return sum(prices) / len(prices)

    def generate_signal(self, code: str, date: str, price: float) -> Tuple[str, int, str]:
        """
        简单的多因子信号生成
        返回: (信号类型, 建议股数, 交易依据)
        """
        # 获取历史数据计算均线
        ma20 = self.calculate_ma(code, date, 20)
        ma60 = self.calculate_ma(code, date, 60)

        if ma20 == 0 or ma60 == 0:
            return "HOLD", 0, "数据不足，无法计算均线"

        position = self.positions[code]

        # 计算均线乖离率
        ma20_deviation = (price - ma20) / ma20 * 100 if ma20 > 0 else 0
        ma60_deviation = (price - ma60) / ma60 * 100 if ma60 > 0 else 0

        # 判断趋势
        trend = "UP" if ma20 > ma60 else "DOWN"

        # 简单策略：价格突破20日均线买入，跌破卖出
        if price > ma20 * 1.02 and position.shares == 0:
            # 买入信号
            max_invest = self.initial_cash * MAX_STOCK_POSITION
            shares = int(max_invest / price / MIN_TRADE_UNIT) * MIN_TRADE_UNIT
            reason = f"【买入信号】价格({price:.2f})突破MA20({ma20:.2f}) 2%以上，MA20乖离率{ma20_deviation:+.2f}%，趋势{trend}"
            return "BUY", shares, reason
        elif price < ma20 * 0.98 and position.shares > 0:
            # 卖出信号
            # 计算持仓盈亏
            cost = position.avg_cost
            pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
            reason = f"【卖出信号】价格({price:.2f})跌破MA20({ma20:.2f}) 2%以上，持仓成本{cost:.2f}，当前盈亏{pnl_pct:+.2f}%"
            return "SELL", position.shares, reason

        # 持仓中但未触发卖出信号
        if position.shares > 0:
            cost = position.avg_cost
            pnl_pct = (price - cost) / cost * 100 if cost > 0 else 0
            return "HOLD", 0, f"【持仓观望】价格({price:.2f})在MA20({ma20:.2f})附近，持仓成本{cost:.2f}，当前盈亏{pnl_pct:+.2f}%"

        return "HOLD", 0, f"【空仓观望】价格({price:.2f})未突破MA20({ma20:.2f})，MA20乖离率{ma20_deviation:+.2f}%"

    def execute_trade(self, code: str, signal: str, shares: int, price: float) -> Dict:
        """执行交易"""
        result = {
            "code": code,
            "signal": signal,
            "shares": 0,
            "price": price,
            "amount": 0.0,
            "commission": 0.0,
            "stamp_tax": 0.0,
            "success": False
        }

        if signal == "HOLD" or shares <= 0:
            return result

        position = self.positions[code]

        if signal == "BUY":
            # 检查资金
            max_affordable = int(self.cash / price / MIN_TRADE_UNIT) * MIN_TRADE_UNIT
            actual_shares = min(shares, max_affordable)

            if actual_shares <= 0:
                return result

            trade_amount = actual_shares * price
            commission = max(trade_amount * COMMISSION_RATE, 5)
            total_cost = trade_amount + commission

            if total_cost > self.cash:
                return result

            # 更新持仓
            old_shares = position.shares
            old_cost = position.avg_cost * old_shares
            position.shares += actual_shares
            position.avg_cost = (old_cost + trade_amount) / position.shares

            self.cash -= total_cost
            self.trade_count += 1

            result["shares"] = actual_shares
            result["amount"] = trade_amount
            result["commission"] = commission
            result["success"] = True

        elif signal == "SELL":
            actual_shares = min(shares, position.shares)

            if actual_shares <= 0:
                return result

            trade_amount = actual_shares * price
            commission = max(trade_amount * COMMISSION_RATE, 5)
            stamp_tax = trade_amount * STAMP_TAX_RATE

            # 更新持仓
            position.shares -= actual_shares
            if position.shares == 0:
                position.avg_cost = 0

            self.cash += trade_amount - commission - stamp_tax
            self.trade_count += 1

            result["shares"] = actual_shares
            result["amount"] = trade_amount
            result["commission"] = commission
            result["stamp_tax"] = stamp_tax
            result["success"] = True

        return result

    def run(self):
        """运行回测"""
        # 打开输出文件
        self.open_output_file()

        trading_days = self.get_trading_days()

        if len(trading_days) == 0:
            self.print_and_write(f"错误: 未找到股票池在 {self.start_date} 到 {self.end_date} 的交易数据")
            self.close_output_file()
            return

        self.print_and_write(f"\n{'='*100}")
        self.print_and_write(f"多股票回测开始")
        self.print_and_write(f"{'='*100}")
        if len(self.stock_pool) <= 20:
            self.print_and_write(f"股票池: {', '.join([f'{c}({self.get_stock_name(c)})' for c in self.stock_pool])}")
        else:
            sample = [f"{c}({self.get_stock_name(c)})" for c in self.stock_pool[:5]]
            self.print_and_write(f"股票池: {', '.join(sample)} ... (共 {len(self.stock_pool)} 只)")
        self.print_and_write(f"股票数量: {len(self.stock_pool)}")
        self.print_and_write(f"回测区间: {self.start_date} 至 {self.end_date}")
        self.print_and_write(f"交易日数: {len(trading_days)}")
        self.print_and_write(f"初始资金: {self.initial_cash:,.2f} RMB")
        self.print_and_write(f"{'='*100}\n")

        for date in trading_days:
            daily_prices = self.daily_prices.get(date, {})
            if not daily_prices:
                continue

            daily_trades = []
            total_commission = 0.0
            total_stamp_tax = 0.0

            # 对每只股票生成信号并执行交易
            for code in self.stock_pool:
                if code not in daily_prices:
                    continue

                price = daily_prices[code]
                signal, shares, reason = self.generate_signal(code, date, price)

                if signal != "HOLD":
                    trade = self.execute_trade(code, signal, shares, price)
                    if trade["success"]:
                        trade["reason"] = reason  # 添加交易依据
                        trade["name"] = self.get_stock_name(code)  # 添加股票名称
                        daily_trades.append(trade)
                        total_commission += trade["commission"]
                        total_stamp_tax += trade["stamp_tax"]

            # 计算当日总市值
            total_market_value = 0.0
            for code, position in self.positions.items():
                if code in daily_prices:
                    total_market_value += position.shares * daily_prices[code]

            total_value = self.cash + total_market_value
            cumulative_return = (total_value - self.initial_cash) / self.initial_cash * 100

            # 记录
            record = DailyRecord(
                date=date,
                positions={code: StockPosition(code, pos.shares, pos.avg_cost)
                          for code, pos in self.positions.items()},
                prices=daily_prices.copy(),
                trades=daily_trades.copy(),
                total_commission=total_commission,
                total_stamp_tax=total_stamp_tax,
                cash=self.cash,
                total_value=total_value,
                cumulative_return=cumulative_return
            )
            self.records.append(record)

            # 打印当日详情（只打印有交易的日子）
            if daily_trades:
                self.print_daily_summary(record)

        # 打印回测结果汇总
        self.print_final_summary()

        # 打印保存路径（在关闭文件前）
        self.print_and_write(f"\n回测结果已保存到: {OUTPUT_FILE}")

        # 关闭输出文件
        self.close_output_file()

    def print_daily_summary(self, record: DailyRecord):
        """打印每日持仓详情"""
        self.print_and_write(f"\n{'='*100}")
        self.print_and_write(f"日期: {record.date}")
        self.print_and_write(f"{'='*100}")

        # 打印持仓详情（只显示有持仓的股票）
        self.print_and_write(f"\n【持仓详情】")
        print(f"{'代码':<12} {'名称':<12} {'持仓股数':>10} {'当前价格':>10} {'市值':>14} {'成本价':>10} {'盈亏':>12}")
        self.output_file.write(f"{'代码':<12} {'名称':<12} {'持仓股数':>10} {'当前价格':>10} {'市值':>14} {'成本价':>10} {'盈亏':>12}\n")
        print("-" * 90)
        self.output_file.write("-" * 90 + "\n")

        total_market_value = 0.0
        has_position = False
        for code in self.stock_pool:
            position = record.positions.get(code)
            price = record.prices.get(code, 0)

            if position and position.shares > 0:
                has_position = True
                market_value = position.shares * price
                total_market_value += market_value
                cost = position.avg_cost
                pnl = (price - cost) / cost * 100 if cost > 0 else 0
                name = self.get_stock_name(code)[:10]  # 截断名称
                line = f"{code:<12} {name:<12} {position.shares:>10} {price:>10.2f} {market_value:>14,.2f} {cost:>10.2f} {pnl:>11.2f}%"
                print(line)
                self.output_file.write(line + "\n")

        if not has_position:
            print("  (无持仓)")
            self.output_file.write("  (无持仓)\n")

        print("-" * 90)
        self.output_file.write("-" * 90 + "\n")
        print(f"{'合计持仓市值':<24} {total_market_value:>14,.2f}")
        self.output_file.write(f"{'合计持仓市值':<24} {total_market_value:>14,.2f}\n")

        # 打印交易记录
        if record.trades:
            self.print_and_write(f"\n【当日交易】")
            for trade in record.trades:
                op = "买入" if trade["signal"] == "BUY" else "卖出"
                fee = trade["commission"] + trade["stamp_tax"]
                name = trade.get('name', self.get_stock_name(trade['code']))
                self.print_and_write(f"\n  {'='*90}")
                self.print_and_write(f"  代码: {trade['code']} ({name})")
                self.print_and_write(f"  操作: {op} {trade['shares']} 股 @ {trade['price']:.2f}")
                self.print_and_write(f"  金额: {trade['amount']:,.2f} RMB")
                self.print_and_write(f"  手续费: {fee:.2f} RMB")
                self.print_and_write(f"  依据: {trade.get('reason', '无')}")
                self.print_and_write(f"  {'='*90}")

        # 打印资金状况
        self.print_and_write(f"\n【资金状况】")
        self.print_and_write(f"  剩余现金:     {record.cash:>15,.2f} RMB")
        self.print_and_write(f"  持仓市值:     {total_market_value:>15,.2f} RMB")
        self.print_and_write(f"  总资产:       {record.total_value:>15,.2f} RMB")
        self.print_and_write(f"  当日手续费:   {record.total_commission + record.total_stamp_tax:>15,.2f} RMB")
        self.print_and_write(f"  累计收益率:   {record.cumulative_return:>15.2f}%")

    def calculate_stock_statistics(self) -> Dict[str, Dict]:
        """计算每只股票的详细统计信息"""
        stats = {}

        for code in self.stock_pool:
            stock_trades = []
            buy_records = []  # 记录每次买入

            # 收集该股票的所有交易
            for record in self.records:
                for trade in record.trades:
                    if trade["code"] == code and trade["success"]:
                        stock_trades.append({
                            "date": record.date,
                            "signal": trade["signal"],
                            "shares": trade["shares"],
                            "price": trade["price"],
                            "amount": trade["amount"],
                            "commission": trade["commission"],
                            "stamp_tax": trade["stamp_tax"]
                        })

            # 计算买入卖出统计
            total_buy_amount = 0.0
            total_sell_amount = 0.0
            total_buy_shares = 0
            total_sell_shares = 0
            total_commission = 0.0
            total_stamp_tax = 0.0
            trade_count = 0

            for trade in stock_trades:
                if trade["signal"] == "BUY":
                    total_buy_amount += trade["amount"]
                    total_buy_shares += trade["shares"]
                    total_commission += trade["commission"]
                else:
                    total_sell_amount += trade["amount"]
                    total_sell_shares += trade["shares"]
                    total_commission += trade["commission"]
                    total_stamp_tax += trade["stamp_tax"]
                trade_count += 1

            # 计算最终持仓盈亏
            last_record = self.records[-1]
            final_position = last_record.positions.get(code)
            final_price = last_record.prices.get(code, 0)

            realized_pnl = 0.0  # 已实现盈亏
            unrealized_pnl = 0.0  # 未实现盈亏
            final_market_value = 0.0
            final_cost = 0.0

            if final_position and final_position.shares > 0:
                final_market_value = final_position.shares * final_price
                final_cost = final_position.shares * final_position.avg_cost
                unrealized_pnl = final_market_value - final_cost

            # 已实现盈亏 = 卖出金额 - 对应买入成本
            # 简化计算：总卖出 - 总买入 + 当前持仓成本
            total_invested = total_buy_amount
            total_returned = total_sell_amount
            current_cost = final_cost

            # 整体盈亏 = 卖出金额 + 当前市值 - 买入金额 - 交易成本
            total_pnl = total_returned + final_market_value - total_invested - total_commission - total_stamp_tax

            # 计算持仓天数
            hold_days = 0
            if stock_trades:
                first_trade_date = stock_trades[0]["date"]
                last_date = last_record.date
                hold_days = (datetime.strptime(last_date, "%Y-%m-%d") -
                           datetime.strptime(first_trade_date, "%Y-%m-%d")).days

            stats[code] = {
                "trade_count": trade_count,
                "buy_times": sum(1 for t in stock_trades if t["signal"] == "BUY"),
                "sell_times": sum(1 for t in stock_trades if t["signal"] == "SELL"),
                "total_buy_amount": total_buy_amount,
                "total_sell_amount": total_sell_amount,
                "total_buy_shares": total_buy_shares,
                "total_sell_shares": total_sell_shares,
                "total_commission": total_commission,
                "total_stamp_tax": total_stamp_tax,
                "final_shares": final_position.shares if final_position else 0,
                "final_price": final_price,
                "final_market_value": final_market_value,
                "final_cost": final_cost,
                "unrealized_pnl": unrealized_pnl,
                "total_pnl": total_pnl,
                "hold_days": hold_days,
                "avg_buy_price": total_buy_amount / total_buy_shares if total_buy_shares > 0 else 0,
                "avg_sell_price": total_sell_amount / total_sell_shares if total_sell_shares > 0 else 0,
            }

        return stats

    def print_final_summary(self):
        """打印最终回测汇总"""
        if not self.records:
            return

        last_record = self.records[-1]

        # 计算统计数据
        total_return = (last_record.total_value - self.initial_cash) / self.initial_cash * 100
        max_value = max(r.total_value for r in self.records)
        min_value = min(r.total_value for r in self.records)
        max_drawdown = min(r.cumulative_return for r in self.records)

        # 计算总交易成本
        total_commission = sum(r.total_commission for r in self.records)
        total_stamp_tax = sum(r.total_stamp_tax for r in self.records)

        # 计算每只股票详细统计
        stock_stats = self.calculate_stock_statistics()

        # 过滤出有交易的股票
        traded_stats = {code: stat for code, stat in stock_stats.items() if stat['trade_count'] > 0}

        self.print_and_write(f"\n{'='*120}")
        self.print_and_write(f"回测结果汇总")
        self.print_and_write(f"{'='*120}")
        self.print_and_write(f"初始资金:        {self.initial_cash:>15,.2f} RMB")
        self.print_and_write(f"最终资产:        {last_record.total_value:>15,.2f} RMB")
        self.print_and_write(f"总收益率:        {total_return:>15.2f}%")
        self.print_and_write(f"最大回撤:        {max_drawdown:>15.2f}%")
        self.print_and_write(f"最高资产:        {max_value:>15,.2f} RMB")
        self.print_and_write(f"最低资产:        {min_value:>15,.2f} RMB")
        self.print_and_write(f"-" * 60)
        self.print_and_write(f"总交易次数:      {self.trade_count:>15} 次")
        self.print_and_write(f"总佣金:          {total_commission:>15,.2f} RMB")
        self.print_and_write(f"总印花税:        {total_stamp_tax:>15,.2f} RMB")
        self.print_and_write(f"总交易成本:      {total_commission + total_stamp_tax:>15,.2f} RMB")
        self.print_and_write(f"-" * 60)

        # 打印有交易的股票详情
        if traded_stats:
            self.print_and_write(f"\n【有交易的股票详情】 (共 {len(traded_stats)} 只)")
            self.print_and_write(f"{'='*120}")

            # 按总盈亏排序
            sorted_stats = sorted(traded_stats.items(), key=lambda x: x[1]["total_pnl"], reverse=True)

            for code, stat in sorted_stats:
                name = self.get_stock_name(code)
                self.print_and_write(f"\n【{code} {name}】")
                self.print_and_write(f"  交易次数:      买入 {stat['buy_times']} 次 / 卖出 {stat['sell_times']} 次")
                self.print_and_write(f"  买入金额:      {stat['total_buy_amount']:>15,.2f} RMB ({stat['total_buy_shares']} 股, 均价 {stat['avg_buy_price']:.2f})")
                self.print_and_write(f"  卖出金额:      {stat['total_sell_amount']:>15,.2f} RMB ({stat['total_sell_shares']} 股, 均价 {stat['avg_sell_price']:.2f})")
                self.print_and_write(f"  交易成本:      佣金 {stat['total_commission']:,.2f} + 印花税 {stat['total_stamp_tax']:,.2f} = {stat['total_commission'] + stat['total_stamp_tax']:,.2f} RMB")

                if stat['final_shares'] > 0:
                    self.print_and_write(f"  最终持仓:      {stat['final_shares']} 股")
                    self.print_and_write(f"  持仓市值:      {stat['final_market_value']:>15,.2f} RMB (价格 {stat['final_price']:.2f})")
                    self.print_and_write(f"  持仓成本:      {stat['final_cost']:>15,.2f} RMB")
                    self.print_and_write(f"  未实现盈亏:    {stat['unrealized_pnl']:>15,.2f} RMB ({stat['unrealized_pnl']/stat['final_cost']*100 if stat['final_cost'] > 0 else 0:+.2f}%)")

                pnl_pct = (stat['total_pnl'] / stat['total_buy_amount'] * 100) if stat['total_buy_amount'] > 0 else 0
                self.print_and_write(f"  总盈亏:        {stat['total_pnl']:>15,.2f} RMB ({pnl_pct:+.2f}%)")
                self.print_and_write(f"  持仓天数:      {stat['hold_days']} 天")
                self.print_and_write("-" * 80)

            # 打印汇总表格
            self.print_and_write(f"\n【盈亏排名汇总】")
            self.print_and_write(f"{'='*120}")
            self.print_and_write(f"{'排名':<6} {'代码':<10} {'名称':<12} {'买入金额':>12} {'卖出金额':>12} {'当前市值':>12} {'交易成本':>10} {'总盈亏':>12} {'盈亏比例':>8} {'交易次数':>6}")
            self.print_and_write("-" * 120)

            for rank, (code, stat) in enumerate(sorted_stats, 1):
                name = self.get_stock_name(code)[:10]
                pnl_pct = (stat['total_pnl'] / stat['total_buy_amount'] * 100) if stat['total_buy_amount'] > 0 else 0
                self.print_and_write(f"{rank:<6} {code:<10} {name:<12} {stat['total_buy_amount']:>12,.0f} {stat['total_sell_amount']:>12,.0f} "
                      f"{stat['final_market_value']:>12,.0f} {stat['total_commission'] + stat['total_stamp_tax']:>10,.0f} "
                      f"{stat['total_pnl']:>12,.0f} {pnl_pct:>7.1f}% {stat['trade_count']:>6}")

            self.print_and_write(f"{'='*120}")

        # 统计信息
        profitable_stocks = sum(1 for s in traded_stats.values() if s['total_pnl'] > 0)
        losing_stocks = sum(1 for s in traded_stats.values() if s['total_pnl'] < 0)
        total_pnl = sum(s['total_pnl'] for s in traded_stats.values())

        self.print_and_write(f"\n【整体统计】")
        self.print_and_write(f"  有交易股票数:  {len(traded_stats)} / {len(stock_stats)}")
        self.print_and_write(f"  盈利股票数:    {profitable_stocks} / {len(traded_stats)}")
        self.print_and_write(f"  亏损股票数:    {losing_stocks} / {len(traded_stats)}")
        self.print_and_write(f"  股票总盈亏:    {total_pnl:,.2f} RMB")
        self.print_and_write(f"  现金剩余:      {last_record.cash:,.2f} RMB")
        self.print_and_write(f"  总资产:        {last_record.total_value:,.2f} RMB")
        self.print_and_write(f"{'='*120}\n")


def main():
    """主函数"""
    # 获取股票池和名称映射
    stock_pool, code_name_map = get_stock_pool()

    # 股票数量提示
    print(f"\n股票池数量: {len(stock_pool)} 只")
    if len(stock_pool) > 500:
        print(f"注意: 股票数量较多({len(stock_pool)}只)，回测可能需要较长时间")

    engine = MultiStockBacktestEngine(
        stock_pool=stock_pool,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_cash=INITIAL_CASH,
        code_name_map=code_name_map
    )
    engine.load_all_data()
    engine.run()


if __name__ == "__main__":
    main()
