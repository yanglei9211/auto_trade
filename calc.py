#!/usr/bin/env python3
"""
ETF 交易信号计算模块

提供多因子交易策略的计算功能，可被回测脚本导入使用
"""

import sqlite3
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

# ==================== 配置部分 ====================
DB_PATH = "/Users/yanglei/Documents/sqlite/sqlite-data/etf_data.db"
TABLE_NAME = "etf_daily"

# 默认交易参数（可在回测脚本中覆盖）
DEFAULT_INITIAL_CAPITAL = 100000  # 初始资金（用于计算仓位）
DEFAULT_MAX_POSITION = 0.95       # 最大仓位比例
DEFAULT_MIN_POSITION = 0.0        # 最小仓位比例
DEFAULT_SINGLE_TRADE_RATIO = 0.2  # 单次交易占总资金比例


class Signal(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class TradeDecision:
    signal: Signal
    shares: int
    reason: str
    confidence: float  # 置信度 0-1


@dataclass
class MarketData:
    date: str
    open: float
    close: float
    high: float
    low: float
    volume: int
    amount: float
    amplitude: float
    pct_change: float
    change: float
    turnover: float


class FactorCalculator:
    """多因子计算器"""

    def __init__(self, data: List[MarketData]):
        self.data = data
        self.closes = np.array([d.close for d in data])
        self.volumes = np.array([d.volume for d in data])
        self.highs = np.array([d.high for d in data])
        self.lows = np.array([d.low for d in data])

    def ma(self, period: int) -> float:
        """移动平均线"""
        if len(self.closes) < period:
            return self.closes[-1] if len(self.closes) > 0 else 0
        return np.mean(self.closes[-period:])

    def ema(self, period: int) -> float:
        """指数移动平均线"""
        if len(self.closes) < period:
            return self.closes[-1] if len(self.closes) > 0 else 0
        alpha = 2 / (period + 1)
        ema = self.closes[-period]
        for price in self.closes[-period+1:]:
            ema = alpha * price + (1 - alpha) * ema
        return ema

    def rsi(self, period: int = 14) -> float:
        """相对强弱指数"""
        if len(self.closes) < period + 1:
            return 50
        deltas = np.diff(self.closes[-period-1:])
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)
        avg_gain = np.mean(gains) if len(gains) > 0 else 0
        avg_loss = np.mean(losses) if len(losses) > 0 else 0
        if avg_loss == 0:
            return 100
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def macd(self) -> Tuple[float, float, float]:
        """MACD指标，返回 (macd_line, signal_line, histogram)"""
        if len(self.closes) < 26:
            return 0, 0, 0
        ema12 = self.ema(12)
        ema26 = self.ema(26)
        macd_line = ema12 - ema26
        # 简化计算 signal line (9日EMA of MACD)
        signal_line = macd_line * 0.2  # 简化处理
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    def bollinger_bands(self, period: int = 20, std_dev: float = 2.0) -> Tuple[float, float, float]:
        """布林带，返回 (upper, middle, lower)"""
        if len(self.closes) < period:
            middle = self.closes[-1] if len(self.closes) > 0 else 0
            return middle, middle, middle
        middle = np.mean(self.closes[-period:])
        std = np.std(self.closes[-period:])
        upper = middle + std_dev * std
        lower = middle - std_dev * std
        return upper, middle, lower

    def atr(self, period: int = 14) -> float:
        """平均真实波幅"""
        if len(self.data) < period:
            return 0
        tr_list = []
        for i in range(-period, 0):
            if i == -len(self.data):
                continue
            high_low = self.highs[i] - self.lows[i]
            high_close = abs(self.highs[i] - self.closes[i-1])
            low_close = abs(self.lows[i] - self.closes[i-1])
            tr_list.append(max(high_low, high_close, low_close))
        return np.mean(tr_list) if tr_list else 0

    def volume_ma(self, period: int = 20) -> float:
        """成交量均线"""
        if len(self.volumes) < period:
            return np.mean(self.volumes) if len(self.volumes) > 0 else 0
        return np.mean(self.volumes[-period:])

    def volatility(self, period: int = 20) -> float:
        """波动率（收益率标准差）"""
        if len(self.closes) < period + 1:
            return 0
        returns = np.diff(self.closes[-period-1:]) / self.closes[-period-1:-1]
        return np.std(returns) * np.sqrt(252)  # 年化波动率

    def trend_strength(self, period: int = 20) -> float:
        """趋势强度 (-1 到 1)"""
        if len(self.closes) < period:
            return 0
        ma_val = self.ma(period)
        current = self.closes[-1]
        # 价格在均线之上为正，之下为负
        distance = (current - ma_val) / ma_val if ma_val != 0 else 0
        # 限制在 -1 到 1 之间
        return max(-1, min(1, distance * 5))


class Strategy:
    """多因子交易策略"""

    def __init__(self, data: List[MarketData], current_price: float,
                 initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                 max_position: float = DEFAULT_MAX_POSITION,
                 min_position: float = DEFAULT_MIN_POSITION,
                 single_trade_ratio: float = DEFAULT_SINGLE_TRADE_RATIO):
        self.factor = FactorCalculator(data)
        self.current_price = current_price
        self.signals: Dict[str, float] = {}
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.min_position = min_position
        self.single_trade_ratio = single_trade_ratio

    def calculate_all_factors(self) -> Dict[str, float]:
        """计算所有因子得分 (-1 到 1)"""
        # 1. 趋势因子 (基于均线)
        ma5 = self.factor.ma(5)
        ma20 = self.factor.ma(20)
        ma60 = self.factor.ma(60)

        trend_short = 1 if self.current_price > ma5 else -1 if self.current_price < ma5 else 0
        trend_mid = 1 if self.current_price > ma20 else -1 if self.current_price < ma20 else 0
        trend_long = 1 if self.current_price > ma60 else -1 if self.current_price < ma60 else 0

        # 2. RSI 因子
        rsi = self.factor.rsi(14)
        rsi_signal = 0
        if rsi < 30:
            rsi_signal = 1  # 超卖，买入
        elif rsi > 70:
            rsi_signal = -1  # 超买，卖出
        else:
            rsi_signal = (50 - rsi) / 20  # 归一化到 -1 到 1

        # 3. MACD 因子
        macd_line, signal_line, histogram = self.factor.macd()
        macd_signal = 1 if histogram > 0 and macd_line > 0 else -1 if histogram < 0 and macd_line < 0 else 0
        if abs(histogram) < 0.001:
            macd_signal = histogram * 100  # 弱信号

        # 4. 布林带因子
        upper, middle, lower = self.factor.bollinger_bands(20)
        bb_position = (self.current_price - lower) / (upper - lower) if upper != lower else 0.5
        bb_signal = 1 if bb_position < 0.1 else -1 if bb_position > 0.9 else (0.5 - bb_position) * 2

        # 5. 成交量因子
        vol_ma = self.factor.volume_ma(20)
        current_vol = self.factor.volumes[-1] if len(self.factor.volumes) > 0 else 0
        vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1
        vol_signal = 1 if vol_ratio > 2 else -1 if vol_ratio < 0.5 else (vol_ratio - 1)

        # 6. 波动率因子 (高波动率降低仓位)
        volatility = self.factor.volatility(20)
        vol_adj = -1 if volatility > 0.3 else 1 if volatility < 0.15 else 0

        # 7. 动量因子
        if len(self.factor.closes) >= 10:
            momentum = (self.current_price - self.factor.closes[-10]) / self.factor.closes[-10] * 10
            momentum_signal = max(-1, min(1, momentum))
        else:
            momentum_signal = 0

        self.signals = {
            "trend_short": trend_short * 0.15,
            "trend_mid": trend_mid * 0.20,
            "trend_long": trend_long * 0.10,
            "rsi": rsi_signal * 0.20,
            "macd": macd_signal * 0.15,
            "bollinger": bb_signal * 0.10,
            "volume": vol_signal * 0.05,
            "volatility": vol_adj * 0.03,
            "momentum": momentum_signal * 0.02,
        }

        return self.signals

    def get_composite_score(self) -> float:
        """获取综合得分 (-1 到 1)"""
        if not self.signals:
            self.calculate_all_factors()
        return sum(self.signals.values())

    def generate_signal(self, current_hold: int) -> TradeDecision:
        """生成交易信号"""
        score = self.get_composite_score()

        # 根据得分确定信号
        if score > 0.3:
            signal = Signal.BUY
        elif score < -0.3:
            signal = Signal.SELL
        else:
            signal = Signal.HOLD

        # 根据当前持仓调整
        if signal == Signal.BUY and current_hold > 0:
            # 已有持仓，检查是否加仓
            if score < 0.5:
                signal = Signal.HOLD
        elif signal == Signal.SELL and current_hold == 0:
            signal = Signal.HOLD

        # 计算交易股数
        shares = 0
        if signal == Signal.BUY:
            # 计算可买入金额
            available_capital = self.initial_capital * (1 - current_hold * self.current_price / self.initial_capital)
            trade_amount = min(available_capital * 0.3, self.initial_capital * self.single_trade_ratio)
            shares = int(trade_amount / self.current_price / 100) * 100  # 手数取整
        elif signal == Signal.SELL:
            # 卖出部分仓位
            if current_hold > 0:
                shares = min(current_hold, int(current_hold * 0.5 / 100) * 100)
                if shares == 0:
                    shares = current_hold  # 全部卖出

        # 生成理由
        reason_parts = [f"{k}={v:+.2f}" for k, v in self.signals.items() if abs(v) > 0.05]
        reason = f"综合得分: {score:+.3f}, 主要因子: {', '.join(reason_parts[:4])}"

        confidence = min(abs(score) * 2, 1.0)

        return TradeDecision(signal, shares, reason, confidence)


def load_historical_data(code: str, before_date: str) -> List[MarketData]:
    """加载历史数据"""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT * FROM {TABLE_NAME}
        WHERE code = ? AND date < ?
        ORDER BY date ASC
    """, (code, before_date))

    rows = cursor.fetchall()
    conn.close()

    data = []
    for row in rows:
        data.append(MarketData(
            date=row["date"],
            open=row["open"],
            close=row["close"],
            high=row["high"],
            low=row["low"],
            volume=row["volume"],
            amount=row["amount"],
            amplitude=row["amplitude"],
            pct_change=row["pct_change"],
            change=row["change"],
            turnover=row["turnover"]
        ))

    return data


def parse_date(date_str: str) -> str:
    """统一日期格式为 YYYY-MM-DD"""
    if len(date_str) == 8:
        return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
    return date_str


def get_trade_signal(code: str, date: str, hold: int,
                     initial_capital: float = DEFAULT_INITIAL_CAPITAL,
                     max_position: float = DEFAULT_MAX_POSITION,
                     min_position: float = DEFAULT_MIN_POSITION,
                     single_trade_ratio: float = DEFAULT_SINGLE_TRADE_RATIO) -> Tuple[TradeDecision, List[MarketData], float]:
    """
    获取交易信号（供回测脚本调用）

    参数:
        code: ETF代码
        date: 日期 (YYYY-MM-DD)
        hold: 当前持仓
        initial_capital: 初始资金
        max_position: 最大仓位比例
        min_position: 最小仓位比例
        single_trade_ratio: 单次交易比例

    返回:
        (交易决策, 历史数据列表, 当前价格)
    """
    # 加载历史数据
    data = load_historical_data(code, date)

    if len(data) < 20:
        raise ValueError(f"历史数据不足 ({len(data)} 天)，无法生成有效信号")

    # 当前价格
    current_price = data[-1].close

    # 计算因子并生成信号
    strategy = Strategy(data, current_price, initial_capital, max_position, min_position, single_trade_ratio)
    decision = strategy.generate_signal(hold)

    return decision, data, current_price


def calculate_factors(code: str, date: str) -> Dict[str, float]:
    """
    计算所有因子得分（供分析使用）

    参数:
        code: ETF代码
        date: 日期 (YYYY-MM-DD)

    返回:
        因子字典
    """
    data = load_historical_data(code, date)

    if len(data) < 20:
        raise ValueError(f"历史数据不足 ({len(data)} 天)")

    current_price = data[-1].close
    strategy = Strategy(data, current_price)
    return strategy.calculate_all_factors()
