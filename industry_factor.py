#!/usr/bin/env python3
"""
行业因子计算模块

提供行业轮动和相对强度计算功能
"""

import sqlite3
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from const import STOCK_DB_PATH, INDUSTRY_LIST


@dataclass
class IndustryStrength:
    """行业强度数据"""
    industry: str
    return_20d: float      # 20日涨跌幅
    return_60d: float      # 60日涨跌幅
    rank: int              # 强度排名
    above_ma20: bool       # 是否在MA20上方
    trend: str             # 趋势方向


@dataclass
class StockRelativeStrength:
    """个股相对行业强度"""
    code: str
    industry: str
    stock_return_20d: float    # 个股20日涨幅
    industry_return_20d: float # 行业20日涨幅
    relative_strength: float   # 相对强度（个股-行业）
    rank_in_industry: int      # 行业内排名


class IndustryAnalyzer:
    """行业分析器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or STOCK_DB_PATH
        self.industry_table = "industry_daily"
        self.mapping_table = "stock_industry"

    def get_stock_industry(self, code: str) -> Optional[str]:
        """获取个股所属行业"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(f"""
            SELECT industry FROM {self.mapping_table} WHERE code = ?
        """, (code,))
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def calculate_industry_return(self, industry: str, date: str, days: int = 20) -> float:
        """计算行业N日涨跌幅"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 获取当日收盘价
        cursor.execute(f"""
            SELECT close FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (industry, date))
        current_row = cursor.fetchone()

        if not current_row:
            conn.close()
            return 0.0

        current_price = current_row[0]

        # 获取N日前收盘价
        cursor.execute(f"""
            SELECT close FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT 1 OFFSET ?
        """, (industry, date, days))
        past_row = cursor.fetchone()

        conn.close()

        if not past_row or past_row[0] == 0:
            return 0.0

        return (current_price - past_row[0]) / past_row[0]

    def calculate_industry_ma(self, industry: str, date: str, period: int = 20) -> float:
        """计算行业MA"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT close FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT ?
        """, (industry, date, period))

        prices = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(prices) < period:
            return prices[0] if prices else 0
        return sum(prices) / len(prices)

    def get_all_industry_strength(self, date: str) -> List[IndustryStrength]:
        """
        获取所有行业的强度排名

        返回:
            按强度排序的行业列表
        """
        industries = []

        for industry in INDUSTRY_LIST:
            return_20d = self.calculate_industry_return(industry, date, 20)
            return_60d = self.calculate_industry_return(industry, date, 60)

            # 获取当前价格
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(f"""
                SELECT close FROM {self.industry_table}
                WHERE industry = ? AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (industry, date))
            row = cursor.fetchone()
            conn.close()

            if not row:
                continue

            current_price = row[0]
            ma20 = self.calculate_industry_ma(industry, date, 20)

            above_ma20 = current_price > ma20 if ma20 > 0 else False

            # 判断趋势
            if return_20d > 0.05 and return_60d > 0.1:
                trend = "STRONG_UP"
            elif return_20d > 0:
                trend = "UP"
            elif return_20d < -0.05:
                trend = "DOWN"
            else:
                trend = "NEUTRAL"

            # 综合得分（20日涨幅权重60%，60日涨幅权重40%）
            composite_score = return_20d * 0.6 + return_60d * 0.4

            industries.append({
                'industry': industry,
                'return_20d': return_20d,
                'return_60d': return_60d,
                'composite_score': composite_score,
                'above_ma20': above_ma20,
                'trend': trend
            })

        # 按综合得分排序
        industries.sort(key=lambda x: x['composite_score'], reverse=True)

        # 添加排名
        result = []
        for rank, ind in enumerate(industries, 1):
            result.append(IndustryStrength(
                industry=ind['industry'],
                return_20d=ind['return_20d'],
                return_60d=ind['return_60d'],
                rank=rank,
                above_ma20=ind['above_ma20'],
                trend=ind['trend']
            ))

        return result

    def get_top_industries(self, date: str, top_n: int = 5) -> List[str]:
        """获取排名前N的行业名称"""
        strengths = self.get_all_industry_strength(date)
        return [s.industry for s in strengths[:top_n]]

    def calculate_stock_relative_strength(self, code: str, date: str) -> Optional[StockRelativeStrength]:
        """
        计算个股相对其所属行业的强度

        返回:
            StockRelativeStrength 或 None（如果无法计算）
        """
        industry = self.get_stock_industry(code)
        if not industry:
            return None

        # 获取个股20日涨幅（从stock_daily表）
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT close FROM stock_daily
            WHERE code = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (code, date))
        current_row = cursor.fetchone()

        if not current_row:
            conn.close()
            return None

        current_price = current_row[0]

        cursor.execute("""
            SELECT close FROM stock_daily
            WHERE code = ? AND date <= ?
            ORDER BY date DESC LIMIT 1 OFFSET 20
        """, (code, date))
        past_row = cursor.fetchone()

        conn.close()

        if not past_row or past_row[0] == 0:
            return None

        stock_return = (current_price - past_row[0]) / past_row[0]

        # 获取行业20日涨幅
        industry_return = self.calculate_industry_return(industry, date, 20)

        # 计算相对强度
        relative_strength = stock_return - industry_return

        return StockRelativeStrength(
            code=code,
            industry=industry,
            stock_return_20d=stock_return,
            industry_return_20d=industry_return,
            relative_strength=relative_strength,
            rank_in_industry=0  # 需要在行业内部计算排名
        )

    def get_industry_rotation_signal(self, date: str) -> Dict:
        """
        获取行业轮动信号

        返回:
            {
                "top_industries": ["白酒", "电力", ...],
                "avoid_industries": ["房地产", ...],
                "industry_scores": [...],
                "signal": "进攻"/"防守"/"中性"
            }
        """
        strengths = self.get_all_industry_strength(date)

        if not strengths:
            return {
                "top_industries": [],
                "avoid_industries": [],
                "industry_scores": [],
                "signal": "中性"
            }

        # 前5名作为推荐行业
        top_industries = [s.industry for s in strengths[:5]]

        # 后5名作为回避行业
        avoid_industries = [s.industry for s in strengths[-5:]]

        # 判断整体信号
        top_score = strengths[0].return_20d if strengths else 0
        avg_score = sum(s.return_20d for s in strengths) / len(strengths) if strengths else 0

        if top_score > 0.1 and avg_score > 0.05:
            signal = "进攻"
        elif avg_score < -0.05:
            signal = "防守"
        else:
            signal = "中性"

        return {
            "top_industries": top_industries,
            "avoid_industries": avoid_industries,
            "industry_scores": strengths,
            "signal": signal
        }


# 便捷函数
def get_industry_rotation(date: str) -> Dict:
    """获取指定日期的行业轮动信号"""
    analyzer = IndustryAnalyzer()
    return analyzer.get_industry_rotation_signal(date)


def get_stock_industry_relative_strength(code: str, date: str) -> Optional[float]:
    """获取个股相对行业强度（便捷函数）"""
    analyzer = IndustryAnalyzer()
    result = analyzer.calculate_stock_relative_strength(code, date)
    return result.relative_strength if result else None


if __name__ == "__main__":
    # 测试
    test_date = "2024-06-01"

    print(f"行业轮动分析 ({test_date})")
    print(f"{'='*60}")

    signal = get_industry_rotation(test_date)

    print(f"\n市场信号: {signal['signal']}")
    print(f"\n推荐行业 (Top 5):")
    for i, ind in enumerate(signal['top_industries'], 1):
        print(f"  {i}. {ind}")

    print(f"\n回避行业 (Bottom 5):")
    for i, ind in enumerate(signal['avoid_industries'], 1):
        print(f"  {i}. {ind}")

    print(f"\n行业排名详情:")
    for score in signal['industry_scores'][:10]:
        trend_icon = "📈" if score.trend == "STRONG_UP" else "📉" if score.trend == "DOWN" else "➡️"
        print(f"  {score.rank:2d}. {score.industry:12s} {trend_icon} "
              f"20日: {score.return_20d*100:+.1f}%, 60日: {score.return_60d*100:+.1f}%")
