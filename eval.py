#!/usr/bin/env python3
"""
多股票回测脚本

基于 const.py 中的 STOCK_LIST 股票池进行回测
集成 calc.py 的风险控制功能
"""

import sqlite3
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

# 导入常量配置
from const import (
    STOCK_LIST, INITIAL_CAPITAL, MAX_POSITION, MIN_POSITION,
    SINGLE_TRADE_RATIO, COMMISSION_RATE, STAMP_TAX_RATE, MIN_TRADE_UNIT,
    STOCK_DB_PATH, get_full_stock, MAX_WORKERS,
    USE_INDUSTRY_ALPHA, INDUSTRY_ALPHA_WEIGHT,
    LIQUIDITY_MIN_AVG_AMOUNT_20D, MIN_LISTING_DAYS,
)

from tradable_filter import TradableFilterConfig, check_tradable

# 导入 calc.py 的风险管理和信号生成功能
from calc import RiskManager, get_trade_signal, Signal

# 导入市场情绪分析
from market_sentiment import get_market_sentiment, get_position_limit

# 导入行业 Alpha 因子（可选）
try:
    from industry_alpha import IndustryAlphaCalculator, get_industry_alpha_rotation
    INDUSTRY_ALPHA_AVAILABLE = True
except ImportError:
    INDUSTRY_ALPHA_AVAILABLE = False

# 输出文件路径
OUTPUT_FILE = Path(__file__).parent / "eval_output.txt"

# 数据库配置›
DB_PATH = STOCK_DB_PATH
TABLE_NAME = "stock_daily"

# ==================== 回测参数配置（可修改） ====================

# 回测时间范围
START_DATE = "2023-01-01"    # 回测开始日期 (YYYY-MM-DD)
END_DATE = "2026-02-13"      # 回测结束日期 (YYYY-MM-DD)

# 股票池（从 const.py 导入，也可在此覆盖）
# 如果 STOCK_LIST 为空，则自动获取全部股票
STOCK_POOL = STOCK_LIST


def get_stock_pool():
    """获取股票池和代码名称映射"""
    if STOCK_POOL and len(STOCK_POOL) > 0:
        print(f"使用自定义股票池，共 {len(STOCK_POOL)} 只")
        # 尝试从文件加载名称映射
        code_name_map = {}
        try:
            from pathlib import Path
            stock_file = Path(STOCK_LIST_FILE)
            if stock_file.exists():
                with open(stock_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith('#') or line.startswith('-'):
                            continue
                        parts = line.split(maxsplit=1)
                        if len(parts) >= 2:
                            code_name_map[parts[0].strip()] = parts[1].strip()
        except Exception:
            pass
        return STOCK_POOL, code_name_map
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
    """股票持仓（扩展版，支持风控）"""
    code: str
    shares: int = 0
    avg_cost: float = 0.0
    entry_date: str = ""           # 入场日期
    highest_price: float = 0.0     # 持仓期间最高价（用于移动止损）
    hold_days: int = 0             # 持仓天数
    tp_stage: int = 0              # 分批止盈阶段：0未触发；1已触发8%；2已触发15%

    @property
    def market_value(self, price: float = 0) -> float:
        return self.shares * price

    def update_highest_price(self, current_price: float):
        """更新持仓期间最高价"""
        if current_price > self.highest_price:
            self.highest_price = current_price

    def increment_hold_days(self):
        """增加持仓天数"""
        self.hold_days += 1


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
    """多股票回测引擎（集成风控和市场情绪）"""

    def __init__(self, stock_pool: List[str], start_date: str, end_date: str, initial_cash: float, 
                 code_name_map: Dict[str, str] = None, risk_manager: RiskManager = None,
                 enable_market_sentiment: bool = True):
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
        
        # 初始化风险管理器（使用传入的或创建默认实例）
        self.risk_manager = risk_manager or RiskManager()
        
        # 市场情绪控制
        self.enable_market_sentiment = enable_market_sentiment
        self.current_position_limit = 1.0  # 当前仓位上限（根据市场情绪动态调整）
        self.sentiment_history: Dict[str, Dict] = {}  # 记录每日情绪

        # 可交易池过滤缓存：date -> {code: TradableResult}
        self.tradable_cache: Dict[str, Dict[str, object]] = {}

        # 行业 Alpha 因子
        self.use_industry_alpha = USE_INDUSTRY_ALPHA and INDUSTRY_ALPHA_AVAILABLE
        self.industry_alpha_calculator = None
        self.industry_alpha_cache: Dict[str, Dict] = {}  # 日期 -> 行业Alpha数据
        if self.use_industry_alpha:
            try:
                self.industry_alpha_calculator = IndustryAlphaCalculator()
                print(f"行业 Alpha 因子已启用 (权重: {INDUSTRY_ALPHA_WEIGHT})")
            except Exception as e:
                print(f"行业 Alpha 因子初始化失败: {e}")
                self.use_industry_alpha = False

    def get_stock_name(self, code: str) -> str:
        """获取股票名称"""
        return self.code_name_map.get(code, code)

    def get_current_total_position_ratio(self, daily_prices: Dict[str, float]) -> float:
        """计算当前总仓位比例"""
        total_market_value = 0.0
        for code, position in self.positions.items():
            if position.shares > 0 and code in daily_prices:
                total_market_value += position.shares * daily_prices[code]
        
        total_value = self.cash + total_market_value
        if total_value <= 0:
            return 0.0
        return total_market_value / total_value

    def update_market_sentiment(self, date: str) -> Dict:
        """更新市场情绪并返回情绪数据"""
        if not self.enable_market_sentiment:
            self.current_position_limit = 1.0
            return {"score": 1.0, "position_ratio": 1.0, "description": "情绪控制已关闭"}
        
        try:
            sentiment = get_market_sentiment(date)
            self.current_position_limit = sentiment["position_ratio"]
            self.sentiment_history[date] = sentiment
            return sentiment
        except Exception as e:
            # 如果情绪分析失败，默认允许满仓
            self.current_position_limit = 1.0
            return {"score": 1.0, "position_ratio": 1.0, "description": f"情绪分析失败: {e}"}

    def can_open_new_position(self, daily_prices: Dict[str, float], planned_invest: float) -> bool:
        """检查是否可以开新仓（检查总仓位限制）"""
        current_ratio = self.get_current_total_position_ratio(daily_prices)
        planned_ratio = planned_invest / (self.cash + sum(
            self.positions[c].shares * p for c, p in daily_prices.items() 
            if self.positions[c].shares > 0
        ))
        
        return (current_ratio + planned_ratio) <= self.current_position_limit

    @staticmethod
    def _calculate_single_signal_static(args: Tuple) -> Dict:
        """
        计算单只股票的信号（静态方法，用于多进程）
        
        参数:
            args: (code, date, price, position_shares, position_avg_cost, 
                   position_highest_price, position_hold_days, initial_cash)
        
        返回:
            信号信息字典
        """
        (code, date, price, position_shares, position_avg_cost,
         position_highest_price, position_hold_days, initial_cash) = args
        
        try:
            # 导入必要的模块（在子进程中）
            from calc import get_trade_signal, RiskManager
            from const import STOCK_DB_PATH, MAX_POSITION, MIN_POSITION, SINGLE_TRADE_RATIO
            
            # 创建独立的风险管理器实例
            risk_manager = RiskManager()
            
            # 调用信号生成
            decision, _, _ = get_trade_signal(
                code=code,
                date=date,
                hold=position_shares,
                initial_capital=initial_cash,
                max_position=MAX_POSITION,
                min_position=MIN_POSITION,
                single_trade_ratio=SINGLE_TRADE_RATIO,
                risk_manager=risk_manager,
                entry_price=position_avg_cost if position_shares > 0 else 0,
                highest_price=position_highest_price if position_shares > 0 else price,
                hold_days=position_hold_days if position_shares > 0 else 0,
                db_path=STOCK_DB_PATH,
                table_name="stock_daily"
            )
            
            return {
                'code': code,
                'signal': decision.signal.value,
                'shares': decision.shares,
                'reason': decision.reason,
                'score': decision.confidence,
                'price': price
            }
        except Exception as e:
            return {
                'code': code,
                'signal': 'HOLD',
                'shares': 0,
                'reason': f'信号生成失败: {str(e)}',
                'score': 0,
                'price': price
            }

    def _prepare_signal_args(self, date: str, daily_prices: Dict[str, float]) -> List[Tuple]:
        """准备多进程计算的参数列表"""
        args_list = []
        for code, price in daily_prices.items():
            position = self.positions[code]
            args = (
                code,
                date,
                price,
                position.shares,
                position.avg_cost,
                position.highest_price,
                position.hold_days,
                self.initial_cash
            )
            args_list.append(args)
        return args_list

    def _calculate_signals_parallel(self, date: str, daily_prices: Dict[str, float]) -> List[Dict]:
        """
        并行计算所有股票的信号
        
        返回:
            信号信息列表
        """
        # 准备参数列表
        args_list = self._prepare_signal_args(date, daily_prices)
        
        signals = []
        
        # 确定工作进程数
        workers = min(MAX_WORKERS, cpu_count(), len(args_list))
        
        if workers <= 1 or len(args_list) < 10:
            # 股票数量少或只有一个worker，使用单进程
            for args in args_list:
                signal_info = self._calculate_single_signal_static(args)
                signals.append(signal_info)
        else:
            # 使用多进程并行计算（静默模式，不输出进度）
            with ProcessPoolExecutor(max_workers=workers) as executor:
                future_to_code = {
                    executor.submit(self._calculate_single_signal_static, args): args[0] 
                    for args in args_list
                }
                
                for future in as_completed(future_to_code):
                    code = future_to_code[future]
                    try:
                        signal_info = future.result()
                        signals.append(signal_info)
                    except Exception as e:
                        # 如果某个进程失败，记录错误并返回HOLD
                        signals.append({
                            'code': code,
                            'signal': 'HOLD',
                            'shares': 0,
                            'reason': f'多进程计算失败: {str(e)}',
                            'score': 0,
                            'price': daily_prices[code]
                        })
        
        return signals

    def _sort_signals_by_priority(self, signals: List[Dict]) -> List[Dict]:
        """
        按优先级排序信号
        
        排序规则:
        1. SELL 信号优先（止损/止盈）
        2. BUY 信号按评分降序（优先买入评分高的）
        3. HOLD 信号最后
        """
        def sort_key(signal_info):
            signal = signal_info['signal']
            score = signal_info['score']
            
            if signal == 'SELL':
                # 卖出信号优先级最高，按评分降序（先止损）
                return (0, -score)
            elif signal == 'BUY':
                # 买入信号按评分降序
                return (1, -score)
            else:
                # HOLD 信号最后
                return (2, 0)
        
        return sorted(signals, key=sort_key)

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

    def generate_signal(self, code: str, date: str, price: float) -> Tuple[str, int, str]:
        """
        使用 calc.py 的多因子策略生成信号（含风控和行业Alpha）
        返回: (信号类型, 建议股数, 交易依据)
        """
        position = self.positions[code]

        # ========== 可交易池过滤（仅影响 BUY；SELL 一律放行以便退出） ==========
        try:
            day_cache = self.tradable_cache.get(date)
            if day_cache is None:
                day_cache = {}
                self.tradable_cache[date] = day_cache

            tr = day_cache.get(code)
            if tr is None:
                cfg = TradableFilterConfig(
                    amount_window=20,
                    min_avg_amount=LIQUIDITY_MIN_AVG_AMOUNT_20D,
                    min_listing_days=MIN_LISTING_DAYS,
                )
                # 复用单个连接计算，避免每票反复 connect
                conn = getattr(self, "_tradable_conn", None)
                if conn is None:
                    self._tradable_conn = sqlite3.connect(DB_PATH)
                    conn = self._tradable_conn
                tr = check_tradable(conn, code, date, cfg)
                day_cache[code] = tr

            if not tr.tradable and position.shares <= 0:
                return "HOLD", 0, f"不可交易池过滤: {tr.reason}"
        except Exception:
            # 过滤异常时不阻断交易信号（保守：让策略继续跑）
            pass

        # 准备行业 Alpha 因子参数
        industry_alpha_score = 0.0
        industry_rank = 999
        use_alpha = False

        if self.use_industry_alpha and self.industry_alpha_calculator:
            try:
                # 获取该股票所属行业的 Alpha 数据
                stock_alpha = self.industry_alpha_calculator.calculate_stock_alpha_in_industry(code, date)
                if stock_alpha:
                    industry_alpha_score = stock_alpha.industry_alpha
                    industry_rank = stock_alpha.industry_rank
                    use_alpha = True
            except Exception as e:
                # 获取失败时不使用 Alpha 因子
                pass

        try:
            # 使用 calc.py 的 get_trade_signal 函数获取交易决策
            # 传入正确的股票数据库配置和行业 Alpha 因子
            decision, _, _ = get_trade_signal(
                code=code,
                date=date,
                hold=position.shares,
                initial_capital=self.initial_cash,
                max_position=MAX_POSITION,
                min_position=MIN_POSITION,
                single_trade_ratio=SINGLE_TRADE_RATIO,
                risk_manager=self.risk_manager,
                entry_price=position.avg_cost if position.shares > 0 else 0,
                highest_price=position.highest_price if position.shares > 0 else price,
                hold_days=position.hold_days if position.shares > 0 else 0,
                tp_stage=position.tp_stage if position.shares > 0 else 0,
                db_path=DB_PATH,           # 使用 eval.py 的数据库路径
                table_name=TABLE_NAME,      # 使用 eval.py 的表名
                industry_alpha_score=industry_alpha_score,
                industry_rank=industry_rank,
                use_industry_alpha=use_alpha,
                industry_alpha_weight=INDUSTRY_ALPHA_WEIGHT
            )

            # 将 Signal 枚举转换为字符串
            signal_str = decision.signal.value

            return signal_str, decision.shares, decision.reason

        except Exception as e:
            # 数据不足或其他错误时返回 HOLD
            return "HOLD", 0, f"信号生成失败: {str(e)}"

    def execute_trade(self, code: str, signal: str, shares: int, price: float, date: str) -> Dict:
        """执行交易（含风控信息更新）"""
        result = {
            "code": code,
            "signal": signal,
            "shares": 0,
            "price": price,
            "amount": 0.0,
            "commission": 0.0,
            "stamp_tax": 0.0,
            "success": False,
            "reason": "",
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
            
            # 风控信息：首次买入时记录入场日期和最高价
            if old_shares == 0:
                position.entry_date = date
                position.highest_price = price
                position.hold_days = 0
                position.tp_stage = 0  # 新开仓重置分批止盈阶段

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
                # 清仓时重置风控信息
                position.avg_cost = 0
                position.entry_date = ""
                position.highest_price = 0
                position.hold_days = 0
                position.tp_stage = 0

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

            # 更新市场情绪（每日开盘前）
            sentiment = self.update_market_sentiment(date)
            current_position_ratio = self.get_current_total_position_ratio(daily_prices)

            # 更新所有持仓的最高价和持仓天数（用于移动止损和时间止损）
            for code, position in self.positions.items():
                if position.shares > 0 and code in daily_prices:
                    position.update_highest_price(daily_prices[code])
                    position.increment_hold_days()

            # 并行计算所有股票的信号和评分
            signals = self._calculate_signals_parallel(date, daily_prices)
            
            # 按评分排序：卖出信号优先（止损），然后按买入评分降序
            sorted_signals = self._sort_signals_by_priority(signals)

            # 按优先级顺序执行交易
            for signal_info in sorted_signals:
                code = signal_info['code']
                signal = signal_info['signal']
                shares = signal_info['shares']
                reason = signal_info['reason']
                score = signal_info['score']
                price = daily_prices[code]

                if signal != "HOLD":
                    # 买入前检查仓位限制
                    if signal == "BUY":
                        planned_invest = shares * price
                        if not self.can_open_new_position(daily_prices, planned_invest):
                            # 超出仓位限制，跳过买入
                            reason += f" [跳过: 仓位限制 {self.current_position_limit*100:.0f}%，当前 {current_position_ratio*100:.1f}%]"
                            continue

                    trade = self.execute_trade(code, signal, shares, price, date)
                    if trade["success"]:
                        trade["reason"] = reason
                        # 分批止盈阶段推进（根据 reason 标签；不依赖额外字段回传，改动最小）
                        try:
                            pos = self.positions[code]
                            if "【分批止盈1】" in reason:
                                pos.tp_stage = max(pos.tp_stage, 1)
                            elif "【分批止盈2】" in reason:
                                pos.tp_stage = max(pos.tp_stage, 2)
                        except Exception:
                            pass
                        trade["name"] = self.get_stock_name(code)
                        trade['score'] = score  # 记录评分
                        daily_trades.append(trade)
                        total_commission += trade["commission"]
                        total_stamp_tax += trade["stamp_tax"]
                        
                        # 更新当前仓位比例
                        current_position_ratio = self.get_current_total_position_ratio(daily_prices)

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

        # 关闭过滤缓存的连接
        try:
            conn = getattr(self, "_tradable_conn", None)
            if conn is not None:
                conn.close()
                self._tradable_conn = None
        except Exception:
            pass

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

        # 显示市场情绪
        sentiment = self.sentiment_history.get(record.date, {})
        if sentiment:
            self.print_and_write(f"\n【市场情绪】")
            self.print_and_write(f"  情绪得分: {sentiment.get('score', 0):.2f}")
            self.print_and_write(f"  仓位限制: {sentiment.get('position_ratio', 1.0)*100:.0f}%")
            self.print_and_write(f"  情绪描述: {sentiment.get('description', 'N/A')}")
            self.print_and_write(f"  ETF状态: MA5上方{sentiment.get('above_ma5_count', 0)}/{sentiment.get('total_etfs', 5)}, "
                               f"MA20上方{sentiment.get('above_ma20_count', 0)}/{sentiment.get('total_etfs', 5)}, "
                               f"趋势向上{sentiment.get('up_trend_count', 0)}/{sentiment.get('total_etfs', 5)}, "
                               f"MA20拐头{sentiment.get('ma20_rising_count', 0)}/{sentiment.get('total_etfs', 5)}")

        # 显示行业 Alpha 信息（如果启用）
        if self.use_industry_alpha and self.industry_alpha_calculator:
            try:
                alpha_signal = self.industry_alpha_calculator.get_industry_alpha_signal(record.date)
                if alpha_signal:
                    self.print_and_write(f"\n【行业Alpha】")
                    self.print_and_write(f"  市场信号: {alpha_signal.get('market_signal', 'N/A')}")
                    self.print_and_write(f"  推荐行业: {', '.join(alpha_signal.get('top_industries', [])[:3])}")
                    self.print_and_write(f"  回避行业: {', '.join(alpha_signal.get('avoid_industries', [])[:3])}")
            except Exception as e:
                pass  # 获取失败时静默处理

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

    # 创建风险管理器（可根据需要调整参数）
    risk_manager = RiskManager(
        # 已改为 ATR/波动自适应止损：min_stop_loss_pct 仅作为“最小止损宽度”下限
        min_stop_loss_pct=0.10,  # 最小止损宽度下限（10%）
        trail_stop_pct=0.10,     # 10%移动止损
        atr_multiplier=2.0,      # ATR止损倍数
        time_stop_days=10,       # 10天时间止损
        max_position=0.1,        # 单只股票最大仓位10%（降低集中度）
        max_total_position=0.8   # 总最大仓位80%
    )

    engine = MultiStockBacktestEngine(
        stock_pool=stock_pool,
        start_date=START_DATE,
        end_date=END_DATE,
        initial_cash=INITIAL_CASH,
        code_name_map=code_name_map,
        risk_manager=risk_manager,
        enable_market_sentiment=True  # 启用市场情绪控制
    )
    engine.load_all_data()
    engine.run()


if __name__ == "__main__":
    main()
