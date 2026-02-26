#!/usr/bin/env python3
"""
市场情绪分析模块

基于 ETF_POOL 中的ETF数据计算市场情绪，用于控制仓位
"""

import sqlite3
from typing import List, Dict, Tuple
from dataclasses import dataclass

from const import STOCK_DB_PATH, ETF_POOL


@dataclass
class ETFSignal:
    """单只ETF的信号数据"""
    code: str
    price: float
    ma5: float
    ma20: float
    trend: str  # "UP", "DOWN", "NEUTRAL"
    above_ma5: bool
    above_ma20: bool
    ma20_rising: bool  # MA20是否在向上倾斜
    golden_cross: bool  # MA5上穿MA20金叉信号


class MarketSentimentAnalyzer:
    """市场情绪分析器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or STOCK_DB_PATH
        self.table_name = "etf_daily"
        self.etf_pool = ETF_POOL

    def calculate_ma(self, code: str, date: str, period: int) -> float:
        """计算ETF的移动平均线"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT close FROM {self.table_name}
            WHERE code = ? AND date < ?
            ORDER BY date DESC
            LIMIT ?
        """, (code, date, period))

        prices = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(prices) < period:
            return prices[0] if prices else 0
        return sum(prices) / len(prices)

    def get_etf_signal(self, code: str, date: str) -> ETFSignal:
        """获取单只ETF的信号"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当日价格
        cursor.execute(f"""
            SELECT close FROM {self.table_name}
            WHERE code = ? AND date = ?
        """, (code, date))

        row = cursor.fetchone()
        conn.close()

        if not row:
            # 无当日数据：返回中性信号，避免 dataclass 字段不匹配导致运行时异常
            return ETFSignal(
                code=code,
                price=0.0,
                ma5=0.0,
                ma20=0.0,
                trend="NEUTRAL",
                above_ma5=False,
                above_ma20=False,
                ma20_rising=False,
                golden_cross=False,
            )

        price = row[0]
        ma5 = self.calculate_ma(code, date, 5)
        ma20 = self.calculate_ma(code, date, 20)
        ma20_prev = self.calculate_ma(code, date, 25)  # 5天前的MA20，用于判断方向

        # 判断MA20是否在向上倾斜（5天内上涨超过0.5%）
        ma20_rising = (ma20 - ma20_prev) / ma20_prev > 0.005 if ma20_prev > 0 else False

        # 判断金叉：MA5上穿MA20（简化判断：当前MA5>MA20且5天前MA5<=MA20）
        ma5_prev = self.calculate_ma(code, date, 10)  # 近似判断
        golden_cross = ma5 > ma20 and ma5_prev <= ma20 * 1.02 if ma20 > 0 else False

        # 综合判断趋势（结合位置和方向）
        if ma5 > ma20 * 1.005:  # 放宽到0.5%
            if ma20_rising:
                trend = "UP"
            else:
                trend = "WEAK_UP"  # 价格在均线上但均线未拐头
        elif ma5 < ma20 * 0.995:
            if not ma20_rising:
                trend = "DOWN"
            else:
                trend = "WEAK_DOWN"
        else:
            trend = "NEUTRAL"

        return ETFSignal(
            code=code,
            price=price,
            ma5=ma5,
            ma20=ma20,
            trend=trend,
            above_ma5=price > ma5 if ma5 > 0 else False,
            above_ma20=price > ma20 if ma20 > 0 else False,
            ma20_rising=ma20_rising,
            golden_cross=golden_cross
        )

    def analyze_sentiment(self, date: str) -> Dict:
        """
        分析市场情绪

        返回:
            {
                "score": 0.0-1.0,  # 情绪得分
                "position_ratio": 0.0-1.0,  # 建议仓位比例
                "description": "情绪描述",
                "details": [ETFSignal, ...],
                "above_ma20_count": int,
                "above_ma60_count": int,
                "up_trend_count": int
            }
        """
        signals = []
        above_ma5_count = 0
        above_ma20_count = 0
        ma20_rising_count = 0
        golden_cross_count = 0
        up_trend_count = 0
        weak_up_count = 0

        for code in self.etf_pool:
            signal = self.get_etf_signal(code, date)
            signals.append(signal)

            if signal.above_ma5:
                above_ma5_count += 1
            if signal.above_ma20:
                above_ma20_count += 1
            if signal.ma20_rising:
                ma20_rising_count += 1
            if signal.golden_cross:
                golden_cross_count += 1
            if signal.trend == "UP":
                up_trend_count += 1
            elif signal.trend == "WEAK_UP":
                weak_up_count += 1

        total = len(self.etf_pool)

        # 组合方案权重（参考tt.md推荐方案）
        # MA5上方25% + MA20上方20% + 趋势向上20% + MA20拐头20% + 金叉15%
        score = (above_ma5_count / total * 0.25 +
                 above_ma20_count / total * 0.20 +
                 up_trend_count / total * 0.20 +
                 ma20_rising_count / total * 0.20 +
                 golden_cross_count / total * 0.15)

        # 根据得分确定建议仓位（降低阈值，更容易建仓）
        if score >= 0.55:
            position_ratio = 0.95  # 高仓位
            description = "强势市场，建议高仓位"
        elif score >= 0.35:
            position_ratio = 0.7  # 中等仓位
            description = "震荡偏多，建议中等仓位"
        elif score >= 0.2:
            position_ratio = 0.4  # 低仓位
            description = "震荡偏空，建议低仓位"
        else:
            position_ratio = 0.2  # 极低仓位
            description = "弱势市场，建议极低仓位"

        return {
            "score": score,
            "position_ratio": position_ratio,
            "description": description,
            "details": signals,
            "above_ma5_count": above_ma5_count,
            "above_ma20_count": above_ma20_count,
            "ma20_rising_count": ma20_rising_count,
            "golden_cross_count": golden_cross_count,
            "up_trend_count": up_trend_count,
            "weak_up_count": weak_up_count,
            "total_etfs": total
        }


def get_market_sentiment(date: str) -> Dict:
    """
    获取指定日期的市场情绪（便捷函数）

    参数:
        date: 日期 (YYYY-MM-DD)

    返回:
        情绪分析结果字典
    """
    analyzer = MarketSentimentAnalyzer()
    return analyzer.analyze_sentiment(date)


def get_position_limit(date: str) -> float:
    """
    获取指定日期的建议仓位上限

    参数:
        date: 日期 (YYYY-MM-DD)

    返回:
        建议仓位比例 (0.0-1.0)
    """
    result = get_market_sentiment(date)
    return result["position_ratio"]


if __name__ == "__main__":
    # 测试
    test_date = "2024-01-15"
    result = get_market_sentiment(test_date)

    print(f"市场情绪分析 ({test_date})")
    print(f"{'='*60}")
    print(f"情绪得分: {result['score']:.2f}")
    print(f"建议仓位: {result['position_ratio']*100:.0f}%")
    print(f"情绪描述: {result['description']}")
    print(f"\nETF详情:")
    print(f"  MA5上方: {result['above_ma5_count']}/{result['total_etfs']}")
    print(f"  MA20上方: {result['above_ma20_count']}/{result['total_etfs']}")
    print(f"  MA20拐头: {result['ma20_rising_count']}/{result['total_etfs']}")
    print(f"  金叉信号: {result['golden_cross_count']}/{result['total_etfs']}")
    print(f"  趋势向上: {result['up_trend_count']}/{result['total_etfs']}")
    print(f"\n各ETF状态:")
    for s in result['details']:
        status = "📈" if s.trend == "UP" else "📉" if s.trend == "DOWN" else "➡️"
        print(f"  {s.code}: {status} 价格={s.price:.2f}, MA5={s.ma5:.2f}, MA20={s.ma20:.2f}, 趋势={s.trend}")
