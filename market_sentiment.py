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
    ma20: float
    ma60: float
    trend: str  # "UP", "DOWN", "NEUTRAL"
    above_ma20: bool
    above_ma60: bool


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
            return ETFSignal(code=code, price=0, ma20=0, ma60=0, trend="NEUTRAL",
                           above_ma20=False, above_ma60=False)

        price = row[0]
        ma20 = self.calculate_ma(code, date, 20)
        ma60 = self.calculate_ma(code, date, 60)

        # 判断趋势
        if ma20 > ma60 * 1.02:
            trend = "UP"
        elif ma20 < ma60 * 0.98:
            trend = "DOWN"
        else:
            trend = "NEUTRAL"

        return ETFSignal(
            code=code,
            price=price,
            ma20=ma20,
            ma60=ma60,
            trend=trend,
            above_ma20=price > ma20 if ma20 > 0 else False,
            above_ma60=price > ma60 if ma60 > 0 else False
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
        above_ma20_count = 0
        above_ma60_count = 0
        up_trend_count = 0

        for code in self.etf_pool:
            signal = self.get_etf_signal(code, date)
            signals.append(signal)

            if signal.above_ma20:
                above_ma20_count += 1
            if signal.above_ma60:
                above_ma60_count += 1
            if signal.trend == "UP":
                up_trend_count += 1

        total = len(self.etf_pool)

        # 计算情绪得分 (0-1)
        # 权重: MA20上方占40%, MA60上方占30%, 趋势向上占30%
        score = (above_ma20_count / total * 0.4 +
                 above_ma60_count / total * 0.3 +
                 up_trend_count / total * 0.3)

        # 根据得分确定建议仓位
        if score >= 0.7:
            position_ratio = 0.9  # 高仓位
            description = "强势市场，建议高仓位"
        elif score >= 0.5:
            position_ratio = 0.6  # 中等仓位
            description = "震荡偏多，建议中等仓位"
        elif score >= 0.3:
            position_ratio = 0.3  # 低仓位
            description = "震荡偏空，建议低仓位"
        else:
            position_ratio = 0.1  # 极低仓位
            description = "弱势市场，建议极低仓位或空仓"

        return {
            "score": score,
            "position_ratio": position_ratio,
            "description": description,
            "details": signals,
            "above_ma20_count": above_ma20_count,
            "above_ma60_count": above_ma60_count,
            "up_trend_count": up_trend_count,
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
    print(f"  MA20上方: {result['above_ma20_count']}/{result['total_etfs']}")
    print(f"  MA60上方: {result['above_ma60_count']}/{result['total_etfs']}")
    print(f"  趋势向上: {result['up_trend_count']}/{result['total_etfs']}")
    print(f"\n各ETF状态:")
    for s in result['details']:
        status = "📈" if s.trend == "UP" else "📉" if s.trend == "DOWN" else "➡️"
        print(f"  {s.code}: {status} 价格={s.price:.2f}, MA20={s.ma20:.2f}, 趋势={s.trend}")
