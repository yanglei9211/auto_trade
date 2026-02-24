#!/usr/bin/env python3
"""
股票交易信号计算模块

提供多因子交易策略的计算功能，可被回测脚本导入使用
"""

import sqlite3
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
from enum import Enum

# ==================== 配置部分 ====================
# 默认使用股票数据库，可在调用时通过参数覆盖
DEFAULT_DB_PATH = "/Users/yanglei/Documents/sqlite/sqlite-data/stock_data.db"
DEFAULT_TABLE_NAME = "stock_daily"

# 兼容旧代码
DB_PATH = DEFAULT_DB_PATH
TABLE_NAME = DEFAULT_TABLE_NAME

# 默认交易参数（可在回测脚本中覆盖）
DEFAULT_INITIAL_CAPITAL = 100000  # 初始资金（用于计算仓位）
DEFAULT_MAX_POSITION = 0.95       # 最大仓位比例
DEFAULT_MIN_POSITION = 0.0        # 最小仓位比例
DEFAULT_SINGLE_TRADE_RATIO = 0.2   # 单次交易占总资金比例

# 默认止损参数
# 说明：止损优先使用“ATR自适应止损”（entry - ATR_MULTIPLIER * ATR）。
# 为防止 ATR 异常偏小导致止损过紧，增加一个“最小止损比例”下限。
DEFAULT_MIN_STOP_LOSS_PCT = 0.10   # 最小止损比例下限 (10%)
DEFAULT_TRAIL_STOP_PCT = 0.10      # 移动止损比例 (10%)
DEFAULT_ATR_MULTIPLIER = 2.0       # ATR止损倍数
DEFAULT_TIME_STOP_DAYS = 5         # 时间止损天数（缩短至5天，提高资金效率）
DEFAULT_TAKE_PROFIT_PCT = 0.15     # 止盈比例 (15%)

# 分批止盈阈值（更对称、更容易实现）
TP1_PROFIT_PCT = 0.08              # 盈利8%：先止盈1/3
TP2_PROFIT_PCT = 0.15              # 盈利15%：再止盈1/3
TP_SELL_FRACTION = 1/3             # 每次卖出比例


class RiskManager:
    """
    风险管理器
    
    提供止损和仓位动态调整功能
    """

    def __init__(
        self,
        min_stop_loss_pct: float = DEFAULT_MIN_STOP_LOSS_PCT,
        trail_stop_pct: float = DEFAULT_TRAIL_STOP_PCT,
        atr_multiplier: float = DEFAULT_ATR_MULTIPLIER,
        time_stop_days: int = DEFAULT_TIME_STOP_DAYS,
        max_position: float = 0.2,
        max_total_position: float = 0.8
    ):
        """
        初始化风险管理器
        
        参数:
            min_stop_loss_pct: 最小止损比例下限 (如 0.05 表示至少允许亏损5%再止损；防止ATR过小)
            trail_stop_pct: 移动止损比例 (如 0.10 表示从最高点回落10%止损)
            atr_multiplier: ATR止损倍数
            time_stop_days: 时间止损天数 (持仓N天未盈利则卖出)
            max_position: 单只股票最大仓位比例
            max_total_position: 总最大仓位比例
        """
        self.min_stop_loss_pct = min_stop_loss_pct
        self.trail_stop_pct = trail_stop_pct
        self.atr_multiplier = atr_multiplier
        self.time_stop_days = time_stop_days
        self.max_position = max_position
        self.max_total_position = max_total_position

    def should_stop_loss(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        atr: float = 0,
        hold_days: int = 0,
        score: float = 0.0,
        time_stop_requires_weak_score: bool = True,
        time_stop_score_threshold: float = 0.0,
    ) -> Tuple[bool, str]:
        """
        检查是否触发止损
        
        参数:
            entry_price: 入场价格
            current_price: 当前价格
            highest_price: 持仓期间最高价
            atr: ATR值 (可选)
            hold_days: 持仓天数 (可选)
            score: 策略综合得分（用于时间止损的“弱趋势”判定）
            time_stop_requires_weak_score: 是否要求 score 偏弱才触发时间止损
            time_stop_score_threshold: score 阈值（默认 0.0，表示 score<=0 才触发时间止损）

        返回:
            (是否止损, 止损原因)
        """
        loss_pct = (entry_price - current_price) / entry_price

        # 1) ATR/波动自适应止损（主止损）
        # 目标：替代“固定5%”成为默认止损逻辑。
        # - ATR > 0 时：止损价 = entry - k * ATR
        # - 但 ATR 可能异常偏小（或数据不足导致为0），因此设置最小止损比例下限。
        if atr > 0:
            atr_stop_price = entry_price - self.atr_multiplier * atr
            min_pct_stop_price = entry_price * (1 - self.min_stop_loss_pct)
            stop_price = min(atr_stop_price, min_pct_stop_price)  # 价格更低 => 止损更宽松
            if current_price < stop_price:
                return True, f"ATR自适应止损 (stop={stop_price:.2f}, ATR={atr:.2f})"

        # 2) 兜底：若 ATR 不可用，则使用最小止损比例下限
        if atr <= 0 and loss_pct >= self.min_stop_loss_pct:
            return True, f"止损(兜底{self.min_stop_loss_pct*100:.1f}%)"

        # 3) 移动止损 (保护利润)
        if highest_price > entry_price:
            drawdown_pct = (highest_price - current_price) / highest_price
            if drawdown_pct >= self.trail_stop_pct:
                return True, f"移动止损 (回撤{drawdown_pct*100:.1f}%)"

        # 4) 时间止损（方案A：仅在“得分偏弱/趋势不佳”时触发，避免震荡市被频繁洗出）
        if hold_days >= self.time_stop_days and current_price <= entry_price:
            if (not time_stop_requires_weak_score) or (score <= time_stop_score_threshold):
                return True, f"时间止损 (持仓{hold_days}天未盈利, score={score:+.3f})"

        return False, ""

    def should_take_profit(
        self,
        entry_price: float,
        current_price: float,
        highest_price: float,
        hold_days: int = 0
    ) -> Tuple[bool, str]:
        """（保留兼容）检查是否触发“全卖”止盈

        说明：新版本主要使用“分批止盈”，因此这里不再承担主要止盈职责。
        目前仅保留为兼容逻辑（可后续删除）。
        """
        profit_pct = (current_price - entry_price) / entry_price

        # 兼容：极端情况下仍可全卖止盈（比如盈利很高且回撤显著）
        if profit_pct >= DEFAULT_TAKE_PROFIT_PCT:
            if highest_price > entry_price:
                drawdown_pct = (highest_price - current_price) / highest_price
                if drawdown_pct >= 0.10:
                    return True, f"止盈(兼容) 盈利{profit_pct*100:.1f}%, 回撤{drawdown_pct*100:.1f}%"

        return False, ""

    def calculate_position_size(
        self,
        confidence: float,
        volatility: float,
        atr: float,
        available_capital: float,
        current_price: float
    ) -> int:
        """
        动态计算仓位
        
        参数:
            confidence: 信号置信度 (0-1)
            volatility: 波动率 (年化)
            atr: ATR值
            available_capital: 可用资金
            current_price: 当前价格
            
        返回:
            建议买入股数 (手取整)
        """
        # 1. 基础仓位 = 置信度
        base_ratio = confidence

        # 2. 波动率调整：波动越大仓位越小
        if volatility < 0.15:
            vol_adj = 1.0
        elif volatility > 0.30:
            vol_adj = 0.5
        else:
            vol_adj = 1.0 - (volatility - 0.15) / 0.15 * 0.5

        # 3. ATR调整：ATR越大仓位越小 (控制单日风险)
        if atr > 0:
            atr_ratio = atr / current_price
            if atr_ratio < 0.02:
                atr_adj = 1.0
            elif atr_ratio > 0.04:
                atr_adj = 0.5
            else:
                atr_adj = 1.0 - (atr_ratio - 0.02) / 0.02 * 0.5
        else:
            atr_adj = 1.0

        # 4. 最终仓位
        position_ratio = base_ratio * vol_adj * atr_adj

        # 5. 限制最大仓位
        position_ratio = min(position_ratio, self.max_position)

        # 6. 计算股数
        amount = available_capital * position_ratio
        shares = int(amount / current_price / 100) * 100  # 手取整

        return max(shares, 0)

    def calculate_sell_shares(
        self,
        current_hold: int,
        entry_price: float,
        current_price: float,
        force_full: bool = False
    ) -> int:
        """
        计算卖出股数
        
        参数:
            current_hold: 当前持仓
            entry_price: 持仓成本
            current_price: 当前价格
            force_full: 是否强制全卖
            
        返回:
            建议卖出股数
        """
        if current_hold <= 0:
            return 0

        if force_full:
            return current_hold

        # 盈利超过20%可考虑分批卖出
        profit_pct = (current_price - entry_price) / entry_price
        if profit_pct > 0.20:
            return min(current_hold, int(current_hold * 0.5 / 100) * 100)
        elif profit_pct > 0.10:
            return min(current_hold, int(current_hold * 0.25 / 100) * 100)
        else:
            return current_hold


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
        """MACD指标，返回 (macd_line, signal_line, histogram)

        标准定义：
          DIF = EMA12(close) - EMA26(close)
          DEA = EMA9(DIF)
          HIST = DIF - DEA
        """
        if len(self.closes) < 26 + 9:
            return 0, 0, 0

        # 计算EMA序列（用于DEA）
        def ema_series(values: np.ndarray, period: int) -> np.ndarray:
            alpha = 2 / (period + 1)
            out = np.zeros_like(values, dtype=float)
            out[0] = values[0]
            for i in range(1, len(values)):
                out[i] = alpha * values[i] + (1 - alpha) * out[i-1]
            return out

        closes = self.closes.astype(float)
        ema12_s = ema_series(closes, 12)
        ema26_s = ema_series(closes, 26)
        dif_s = ema12_s - ema26_s
        dea_s = ema_series(dif_s, 9)

        macd_line = float(dif_s[-1])
        signal_line = float(dea_s[-1])
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
                 single_trade_ratio: float = DEFAULT_SINGLE_TRADE_RATIO,
                 risk_manager: RiskManager = None):
        """
        初始化策略
        
        参数:
            data: 历史市场数据
            current_price: 当前价格
            initial_capital: 初始资金
            max_position: 最大仓位比例
            min_position: 最小仓位比例
            single_trade_ratio: 单次交易比例
            risk_manager: 风险管理器 (可选)
        """
        self.factor = FactorCalculator(data)
        self.current_price = current_price
        self.signals: Dict[str, float] = {}
        self.initial_capital = initial_capital
        self.max_position = max_position
        self.min_position = min_position
        self.single_trade_ratio = single_trade_ratio
        
        # 使用传入的风险管理器或创建默认实例
        self.risk_manager = risk_manager or RiskManager()
        
        # 计算当前波动率和ATR供仓位管理使用
        self.volatility = self.factor.volatility(20)
        self.atr = self.factor.atr(14)

    def calculate_all_factors(self) -> Dict[str, float]:
        """计算所有因子得分 (-1 到 1)"""
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
            "rsi": rsi_signal * 0.15,
            "macd": macd_signal * 0.10,
            "bollinger": bb_signal * 0.08,
            "volume": vol_signal * 0.05,
            "volatility": vol_adj * 0.03,
            "momentum": momentum_signal * 0.04,
            "industry_alpha": 0.0,  # 将由外部注入行业 Alpha 因子
        }

        return self.signals

    def apply_industry_alpha(self, industry_alpha_score: float, industry_rank: int, alpha_weight: float = 0.10):
        """
        应用行业 Alpha 因子到综合得分

        参数:
            industry_alpha_score: 行业 Alpha 得分 (-1 到 1)
            industry_rank: 行业排名 (1-5为推荐行业)
            alpha_weight: 行业Alpha权重（由外部传入；默认0.10）
        """

        # 排名越靠前，Alpha 因子得分越高
        if industry_rank <= 3:
            rank_multiplier = 1.0
        elif industry_rank <= 5:
            rank_multiplier = 0.5
        else:
            rank_multiplier = -0.5  # 排名靠后的行业扣分

        self.signals["industry_alpha"] = industry_alpha_score * alpha_weight * rank_multiplier

    def get_industry_boost_factor(self, industry_rank: int) -> float:
        """
        获取行业排名带来的选股 boost 因子

        用于在选股时优先选择强势行业中的股票
        """
        if industry_rank <= 3:
            return 1.2  # 前3名行业，得分提升20%
        elif industry_rank <= 5:
            return 1.1  # 前5名行业，得分提升10%
        elif industry_rank <= 10:
            return 1.0  # 正常
        else:
            return 0.8  # 排名靠后的行业，得分降低20%

    def get_composite_score(self) -> float:
        """获取综合得分 (-1 到 1)"""
        if not self.signals:
            self.calculate_all_factors()
        return sum(self.signals.values())

    def generate_signal(self, current_hold: int, entry_price: float = 0,
                        highest_price: float = 0, hold_days: int = 0,
                        tp_stage: int = 0) -> TradeDecision:
        """
        生成交易信号 (含风险管理)
        
        参数:
            current_hold: 当前持仓股数
            entry_price: 持仓成本价 (可选，用于止损判断)
            highest_price: 持仓期间最高价 (可选，用于移动止损)
            hold_days: 持仓天数 (可选，用于时间止损)
            tp_stage: 分批止盈阶段（0未触发；1已触发8%；2已触发15%）

        返回:
            TradeDecision: 交易决策
        """
        score = self.get_composite_score()
        
        # ========== 止损检查 ==========
        if current_hold > 0 and entry_price > 0:
            # 使用风险管理器检查止损
            should_stop, stop_reason = self.risk_manager.should_stop_loss(
                entry_price=entry_price,
                current_price=self.current_price,
                highest_price=highest_price if highest_price > 0 else self.current_price,
                atr=self.atr,
                hold_days=hold_days,
                score=score,
                time_stop_requires_weak_score=True,
                time_stop_score_threshold=-0.1,
            )
            
            if should_stop:
                # 触发止损，强制卖出
                shares = self.risk_manager.calculate_sell_shares(
                    current_hold=current_hold,
                    entry_price=entry_price,
                    current_price=self.current_price,
                    force_full=True
                )
                reason = f"【止损】{stop_reason}, 综合得分: {score:+.3f}"
                confidence = 0.9  # 止损信号置信度高
                return TradeDecision(Signal.SELL, shares, reason, confidence)

        # ========== 分批止盈（8%卖1/3；15%再卖1/3；剩余交给移动止损/趋势退出） ==========
        if current_hold > 0 and entry_price > 0:
            profit_pct = (self.current_price - entry_price) / entry_price

            # 用 tp_stage 做分批止盈状态机，防止在阈值之上连续多天重复卖出。

            # 估算分批卖出股数（手取整）
            def _sell_frac(hold: int, frac: float) -> int:
                raw = int(hold * frac)
                return max(int(raw / 100) * 100, 0)

            # 第一档：盈利>=8%，卖出约1/3（仅触发一次）
            if tp_stage <= 0 and profit_pct >= TP1_PROFIT_PCT and current_hold >= 300:
                sell_shares = _sell_frac(current_hold, TP_SELL_FRACTION)
                if sell_shares > 0:
                    reason = f"【分批止盈1】盈利{profit_pct*100:.1f}%>=8%，卖出1/3({sell_shares})，综合得分: {score:+.3f}"
                    return TradeDecision(Signal.SELL, sell_shares, reason, 0.8)

            # 第二档：盈利>=15%，再卖出约1/3（仅触发一次）
            if tp_stage == 1 and profit_pct >= TP2_PROFIT_PCT and current_hold >= 300:
                sell_shares = _sell_frac(current_hold, TP_SELL_FRACTION)
                if sell_shares > 0:
                    reason = f"【分批止盈2】盈利{profit_pct*100:.1f}%>=15%，再卖出1/3({sell_shares})，综合得分: {score:+.3f}"
                    return TradeDecision(Signal.SELL, sell_shares, reason, 0.85)

        # ========== 兼容：全卖止盈（保留但弱化） ==========
        if current_hold > 0 and entry_price > 0:
            should_take_profit, profit_reason = self.risk_manager.should_take_profit(
                entry_price=entry_price,
                current_price=self.current_price,
                highest_price=highest_price if highest_price > 0 else self.current_price,
                hold_days=hold_days
            )
            if should_take_profit:
                shares = self.risk_manager.calculate_sell_shares(
                    current_hold=current_hold,
                    entry_price=entry_price,
                    current_price=self.current_price,
                    force_full=True
                )
                reason = f"【止盈(兼容)】{profit_reason}, 综合得分: {score:+.3f}"
                confidence = 0.9
                return TradeDecision(Signal.SELL, shares, reason, confidence)

        # ========== 正常信号生成 ==========
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

        # ========== 计算交易股数 (含动态仓位) ==========
        shares = 0
        confidence = min(abs(score) * 2, 1.0)
        
        if signal == Signal.BUY:
            # 可用资金
            used_capital = current_hold * self.current_price
            available_capital = self.initial_capital - used_capital
            
            if available_capital > 0:
                # 使用风险管理器计算动态仓位
                shares = self.risk_manager.calculate_position_size(
                    confidence=confidence,
                    volatility=self.volatility,
                    atr=self.atr,
                    available_capital=available_capital,
                    current_price=self.current_price
                )
                
                # 如果计算结果为0，使用原来的fallback逻辑
                if shares == 0:
                    trade_amount = min(available_capital * 0.3, 
                                     self.initial_capital * self.single_trade_ratio)
                    shares = int(trade_amount / self.current_price / 100) * 100
                    
        elif signal == Signal.SELL:
            if current_hold > 0:
                # 使用风险管理器计算卖出股数
                shares = self.risk_manager.calculate_sell_shares(
                    current_hold=current_hold,
                    entry_price=entry_price if entry_price > 0 else self.current_price,
                    current_price=self.current_price,
                    force_full=False
                )
                if shares == 0:
                    shares = current_hold

        # 生成理由
        reason_parts = []
        
        # 添加因子信息
        factor_info = [f"{k}={v:+.2f}" for k, v in self.signals.items() if abs(v) > 0.05]
        if factor_info:
            reason_parts.append(f"因子: {', '.join(factor_info[:3])}")
        
        # 添加仓位调整信息
        if signal == Signal.BUY and shares > 0:
            reason_parts.append(f"波动率: {self.volatility*100:.1f}%, ATR: {self.atr:.2f}")
        
        # 添加风险提示
        if current_hold > 0 and entry_price > 0:
            loss_pct = (self.current_price - entry_price) / entry_price
            if loss_pct > 0:
                reason_parts.append(f"持仓亏损: {loss_pct*100:.1f}%")
        
        reason = f"综合得分: {score:+.3f}, {', '.join(reason_parts)}"

        return TradeDecision(signal, shares, reason, confidence)


def load_historical_data(code: str, before_date: str, 
                         db_path: str = None, table_name: str = None) -> List[MarketData]:
    """加载历史数据"""
    # 使用传入的参数或默认值
    db = db_path or DEFAULT_DB_PATH
    table = table_name or DEFAULT_TABLE_NAME
    
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute(f"""
        SELECT * FROM {table}
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
                     single_trade_ratio: float = DEFAULT_SINGLE_TRADE_RATIO,
                     risk_manager: RiskManager = None,
                     entry_price: float = 0,
                     highest_price: float = 0,
                     hold_days: int = 0,
                     tp_stage: int = 0,
                     db_path: str = None,
                     table_name: str = None,
                     industry_alpha_score: float = 0.0,
                     industry_rank: int = 999,
                     use_industry_alpha: bool = False,
                     industry_alpha_weight: float = 0.10) -> Tuple[TradeDecision, List[MarketData], float]:
    """
    获取交易信号（供回测脚本调用）

    参数:
        code: 股票代码
        date: 日期 (YYYY-MM-DD)
        hold: 当前持仓
        initial_capital: 初始资金
        max_position: 最大仓位比例
        min_position: 最小仓位比例
        single_trade_ratio: 单次交易比例
        risk_manager: 风险管理器 (可选)
        entry_price: 持仓成本价 (可选，用于止损)
        highest_price: 持仓期间最高价 (可选，用于移动止损)
        hold_days: 持仓天数 (可选，用于时间止损)
        db_path: 数据库路径 (可选，默认使用股票数据库)
        table_name: 表名 (可选，默认使用stock_daily)
        industry_alpha_score: 行业 Alpha 得分 (可选)
        industry_rank: 行业排名 (可选)
        use_industry_alpha: 是否使用行业 Alpha 因子 (可选)
        tp_stage: 分批止盈阶段（0未触发；1已触发8%；2已触发15%）

    返回:
        (交易决策, 历史数据列表, 当前价格)
    """
    # 加载历史数据
    data = load_historical_data(code, date, db_path, table_name)

    if len(data) < 20:
        raise ValueError(f"历史数据不足 ({len(data)} 天)，无法生成有效信号")

    # 当前价格
    current_price = data[-1].close

    # 计算因子并生成信号
    strategy = Strategy(data, current_price, initial_capital, max_position, 
                        min_position, single_trade_ratio, risk_manager)

    # 应用行业 Alpha 因子
    if use_industry_alpha and industry_alpha_score != 0.0:
        strategy.apply_industry_alpha(industry_alpha_score, industry_rank, alpha_weight=industry_alpha_weight)

    decision = strategy.generate_signal(hold, entry_price, highest_price, hold_days, tp_stage)

    # 如果启用了行业 Alpha，在理由中追加信息
    if use_industry_alpha and industry_rank <= 10:
        industry_tag = f"[行业排名{industry_rank}]"
        decision.reason = f"{industry_tag} {decision.reason}"

    return decision, data, current_price


def calculate_factors(code: str, date: str) -> Dict[str, float]:
    """
    计算所有因子得分（供分析使用）

    参数:
        code: 股票代码
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
