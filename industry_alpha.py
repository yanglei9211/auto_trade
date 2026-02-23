#!/usr/bin/env python3
"""
行业 Alpha 因子模块

基于行业数据构建多维度 Alpha 因子，用于获取超额收益
"""

import sqlite3
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

from const import STOCK_DB_PATH, INDUSTRY_LIST
from industry_factor import IndustryAnalyzer


@dataclass
class IndustryAlphaFactor:
    """行业 Alpha 因子数据"""
    industry: str
    # 动量因子
    momentum_20d: float      # 20日动量
    momentum_60d: float      # 60日动量
    momentum_120d: float     # 120日动量
    # 相对强度因子
    rs_vs_market: float      # 相对大盘强度
    rs_rank: int             # 相对强度排名
    # 趋势因子
    trend_score: float       # 趋势得分 (0-1)
    above_ma20: bool
    above_ma60: bool
    # 波动率因子
    volatility: float        # 20日波动率
    # 资金流向因子
    volume_ratio: float      # 成交量比率
    # 综合 Alpha 得分
    alpha_score: float       # 综合 Alpha 得分
    alpha_rank: int          # Alpha 排名


@dataclass
class StockIndustryAlpha:
    """个股行业 Alpha 数据"""
    code: str
    industry: str
    # 个股表现
    stock_return_20d: float
    stock_volatility: float
    # 行业表现
    industry_alpha: float    # 行业 Alpha 得分
    industry_rank: int       # 行业排名 (1-5为推荐)
    # 个股在行业内的相对表现
    relative_strength: float # 相对行业强度
    rs_percentile: float     # 行业内分位数 (0-1)
    # 综合选股得分
    selection_score: float   # 选股得分 (越高越推荐)


class IndustryAlphaCalculator:
    """行业 Alpha 计算器"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path or STOCK_DB_PATH
        self.analyzer = IndustryAnalyzer(db_path)
        self.industry_table = "industry_daily"
        self.mapping_table = "stock_industry"

    def calculate_industry_volatility(self, industry: str, date: str, days: int = 20) -> float:
        """计算行业波动率"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(f"""
            SELECT close FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT ?
        """, (industry, date, days + 1))

        prices = [row[0] for row in cursor.fetchall()]
        conn.close()

        if len(prices) < days:
            return 0.0

        # 计算日收益率的标准差
        prices = np.array(prices[::-1])  # 正序排列
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # 年化波动率

        return volatility

    def calculate_volume_ratio(self, industry: str, date: str) -> float:
        """计算成交量比率（当前20日均量 / 前20日均量）"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # 最近20日成交量
        cursor.execute(f"""
            SELECT volume FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT 20
        """, (industry, date))
        recent_volumes = [row[0] for row in cursor.fetchall()]

        # 前20日成交量
        cursor.execute(f"""
            SELECT volume FROM {self.industry_table}
            WHERE industry = ? AND date <= ?
            ORDER BY date DESC LIMIT 20 OFFSET 20
        """, (industry, date))
        past_volumes = [row[0] for row in cursor.fetchall()]

        conn.close()

        if not recent_volumes or not past_volumes:
            return 1.0

        recent_avg = np.mean(recent_volumes)
        past_avg = np.mean(past_volumes)

        return recent_avg / past_avg if past_avg > 0 else 1.0

    def calculate_market_return(self, date: str, days: int = 20) -> float:
        """计算大盘（沪深300 proxy）N日涨幅"""
        # 使用多个行业的平均作为市场基准
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        returns = []
        for industry in ["银行", "证券Ⅱ", "白酒Ⅱ", "电力", "半导体"]:
            cursor.execute(f"""
                SELECT close FROM {self.industry_table}
                WHERE industry = ? AND date <= ?
                ORDER BY date DESC LIMIT 1 OFFSET {days}
            """, (industry, date))
            past_row = cursor.fetchone()

            cursor.execute(f"""
                SELECT close FROM {self.industry_table}
                WHERE industry = ? AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (industry, date))
            current_row = cursor.fetchone()

            if past_row and current_row and past_row[0] > 0:
                ret = (current_row[0] - past_row[0]) / past_row[0]
                returns.append(ret)

        conn.close()

        return np.mean(returns) if returns else 0.0

    def calculate_industry_alpha_factors(self, date: str) -> List[IndustryAlphaFactor]:
        """
        计算所有行业的 Alpha 因子

        返回:
            按 Alpha 得分排序的行业列表
        """
        industries = []
        market_return_20d = self.calculate_market_return(date, 20)

        for industry in INDUSTRY_LIST:
            # 基础动量
            mom_20d = self.analyzer.calculate_industry_return(industry, date, 20)
            mom_60d = self.analyzer.calculate_industry_return(industry, date, 60)
            mom_120d = self.analyzer.calculate_industry_return(industry, date, 120)

            # 相对大盘强度
            rs_vs_market = mom_20d - market_return_20d

            # 趋势判断
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
            ma20 = self.analyzer.calculate_industry_ma(industry, date, 20)
            ma60 = self.analyzer.calculate_industry_ma(industry, date, 60)

            above_ma20 = current_price > ma20 if ma20 > 0 else False
            above_ma60 = current_price > ma60 if ma60 > 0 else False

            # 趋势得分 (0-1)
            trend_score = 0.0
            if above_ma20:
                trend_score += 0.3
            if above_ma60:
                trend_score += 0.3
            if mom_20d > 0:
                trend_score += 0.2
            if mom_60d > 0:
                trend_score += 0.2

            # 波动率
            volatility = self.calculate_industry_volatility(industry, date, 20)

            # 成交量比率
            volume_ratio = self.calculate_volume_ratio(industry, date)

            # 综合 Alpha 得分（多因子加权）
            # 动量40% + 相对强度25% + 趋势20% + 波动率调整10% + 成交量5%
            alpha_score = (
                mom_20d * 0.25 +           # 短期动量
                mom_60d * 0.15 +           # 中期动量
                rs_vs_market * 0.25 +      # 相对强度
                trend_score * 0.10 +       # 趋势得分
                (1 - volatility) * 0.05 +  # 低波动加分
                (volume_ratio - 1) * 0.05  # 放量加分
            )

            industries.append({
                'industry': industry,
                'momentum_20d': mom_20d,
                'momentum_60d': mom_60d,
                'momentum_120d': mom_120d,
                'rs_vs_market': rs_vs_market,
                'trend_score': trend_score,
                'above_ma20': above_ma20,
                'above_ma60': above_ma60,
                'volatility': volatility,
                'volume_ratio': volume_ratio,
                'alpha_score': alpha_score
            })

        # 按 Alpha 得分排序
        industries.sort(key=lambda x: x['alpha_score'], reverse=True)

        # 添加排名
        result = []
        for rank, ind in enumerate(industries, 1):
            result.append(IndustryAlphaFactor(
                industry=ind['industry'],
                momentum_20d=ind['momentum_20d'],
                momentum_60d=ind['momentum_60d'],
                momentum_120d=ind['momentum_120d'],
                rs_vs_market=ind['rs_vs_market'],
                rs_rank=rank,
                trend_score=ind['trend_score'],
                above_ma20=ind['above_ma20'],
                above_ma60=ind['above_ma60'],
                volatility=ind['volatility'],
                volume_ratio=ind['volume_ratio'],
                alpha_score=ind['alpha_score'],
                alpha_rank=rank
            ))

        return result

    def get_top_alpha_industries(self, date: str, top_n: int = 5) -> List[str]:
        """获取 Alpha 得分排名前N的行业"""
        factors = self.calculate_industry_alpha_factors(date)
        return [f.industry for f in factors[:top_n]]

    def calculate_stock_alpha_in_industry(self, code: str, date: str) -> Optional[StockIndustryAlpha]:
        """
        计算个股在其行业内的 Alpha 表现

        用于：在强势行业中选择强势股
        """
        industry = self.analyzer.get_stock_industry(code)
        if not industry:
            return None

        # 获取行业 Alpha 因子
        industry_factors = self.calculate_industry_alpha_factors(date)
        industry_factor = next((f for f in industry_factors if f.industry == industry), None)

        if not industry_factor:
            return None

        # 获取个股20日涨幅
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT close FROM stock_daily
            WHERE code = ? AND date <= ?
            ORDER BY date DESC LIMIT 1
        """, (code, date))
        current_row = cursor.fetchone()

        cursor.execute("""
            SELECT close FROM stock_daily
            WHERE code = ? AND date <= ?
            ORDER BY date DESC LIMIT 1 OFFSET 20
        """, (code, date))
        past_row = cursor.fetchone()

        # 计算个股波动率
        cursor.execute("""
            SELECT close FROM stock_daily
            WHERE code = ? AND date <= ?
            ORDER BY date DESC LIMIT 21
        """, (code, date))
        prices = [row[0] for row in cursor.fetchall()]

        conn.close()

        if not current_row or not past_row or past_row[0] == 0:
            return None

        stock_return = (current_row[0] - past_row[0]) / past_row[0]

        # 计算个股波动率
        if len(prices) >= 21:
            prices = np.array(prices[::-1])
            returns = np.diff(prices) / prices[:-1]
            stock_volatility = np.std(returns) * np.sqrt(252)
        else:
            stock_volatility = 0.0

        # 相对行业强度
        relative_strength = stock_return - industry_factor.momentum_20d

        # 获取行业内所有股票计算分位数
        cursor = sqlite3.connect(self.db_path).cursor()
        cursor.execute(f"""
            SELECT s.code FROM {self.mapping_table} m
            JOIN stock_daily s ON m.code = s.code
            WHERE m.industry = ? AND s.date <= ?
            GROUP BY s.code
        """, (industry, date))
        stocks_in_industry = [row[0] for row in cursor.fetchall()]
        cursor.connection.close()

        # 计算行业内所有股票的涨幅
        rs_list = []
        for stock_code in stocks_in_industry[:50]:  # 限制数量
            cursor = sqlite3.connect(self.db_path).cursor()
            cursor.execute("""
                SELECT close FROM stock_daily
                WHERE code = ? AND date <= ?
                ORDER BY date DESC LIMIT 1 OFFSET 20
            """, (stock_code, date))
            past = cursor.fetchone()
            cursor.execute("""
                SELECT close FROM stock_daily
                WHERE code = ? AND date <= ?
                ORDER BY date DESC LIMIT 1
            """, (stock_code, date))
            current = cursor.fetchone()
            cursor.connection.close()

            if past and current and past[0] > 0:
                ret = (current[0] - past[0]) / past[0]
                rs_list.append(ret - industry_factor.momentum_20d)

        # 计算分位数
        if rs_list:
            rs_percentile = sum(1 for rs in rs_list if rs <= relative_strength) / len(rs_list)
        else:
            rs_percentile = 0.5

        # 综合选股得分
        # 行业 Alpha 40% + 相对强度 35% + 低波动 15% + 行业内排名 10%
        selection_score = (
            industry_factor.alpha_score * 0.40 +
            relative_strength * 0.35 +
            (1 - stock_volatility) * 0.15 +
            rs_percentile * 0.10
        )

        return StockIndustryAlpha(
            code=code,
            industry=industry,
            stock_return_20d=stock_return,
            stock_volatility=stock_volatility,
            industry_alpha=industry_factor.alpha_score,
            industry_rank=industry_factor.alpha_rank,
            relative_strength=relative_strength,
            rs_percentile=rs_percentile,
            selection_score=selection_score
        )

    def get_industry_alpha_signal(self, date: str) -> Dict:
        """
        获取行业 Alpha 轮动信号

        返回:
            {
                "top_industries": [...],      # 推荐行业（Alpha前5）
                "avoid_industries": [...],    # 回避行业（Alpha后5）
                "industry_factors": [...],    # 所有行业因子
                "market_signal": "进攻"/"防守"/"中性",
                "top_picks": [...]            # 各推荐行业的首选个股
            }
        """
        factors = self.calculate_industry_alpha_factors(date)

        if not factors:
            return {
                "top_industries": [],
                "avoid_industries": [],
                "industry_factors": [],
                "market_signal": "中性",
                "top_picks": []
            }

        # 前5名作为推荐行业
        top_factors = factors[:5]
        top_industries = [f.industry for f in top_factors]

        # 后5名作为回避行业
        avoid_industries = [f.industry for f in factors[-5:]]

        # 判断市场整体信号
        avg_alpha = np.mean([f.alpha_score for f in factors])
        top_alpha = top_factors[0].alpha_score if top_factors else 0

        if top_alpha > 0.1 and avg_alpha > 0.03:
            market_signal = "进攻"
        elif avg_alpha < -0.03:
            market_signal = "防守"
        else:
            market_signal = "中性"

        return {
            "top_industries": top_industries,
            "avoid_industries": avoid_industries,
            "industry_factors": factors,
            "market_signal": market_signal,
            "top_picks": []  # 可在后续扩展
        }


# 便捷函数
def get_industry_alpha_rotation(date: str) -> Dict:
    """获取行业 Alpha 轮动信号（便捷函数）"""
    calculator = IndustryAlphaCalculator()
    return calculator.get_industry_alpha_signal(date)


def get_stock_industry_selection_score(code: str, date: str) -> Optional[float]:
    """获取个股行业选股得分（便捷函数）"""
    calculator = IndustryAlphaCalculator()
    result = calculator.calculate_stock_alpha_in_industry(code, date)
    return result.selection_score if result else None


if __name__ == "__main__":
    # 测试
    test_date = "2024-06-01"

    print(f"行业 Alpha 因子分析 ({test_date})")
    print(f"{'='*80}")

    calculator = IndustryAlphaCalculator()
    factors = calculator.calculate_industry_alpha_factors(test_date)

    print(f"\nAlpha 排名前10行业:")
    print(f"{'排名':<4} {'行业':<12} {'Alpha':>8} {'20日':>8} {'60日':>8} {'RS':>8} {'趋势':>6}")
    print("-" * 80)
    for f in factors[:10]:
        print(f"{f.alpha_rank:<4} {f.industry:<12} {f.alpha_score:>+8.3f} "
              f"{f.momentum_20d*100:>+7.1f}% {f.momentum_60d*100:>+7.1f}% "
              f"{f.rs_vs_market*100:>+7.1f}% {f.trend_score:>6.2f}")

    # 测试个股
    print(f"\n\n个股行业 Alpha 测试 (600519):")
    stock_alpha = calculator.calculate_stock_alpha_in_industry("600519", test_date)
    if stock_alpha:
        print(f"  行业: {stock_alpha.industry}")
        print(f"  行业 Alpha: {stock_alpha.industry_alpha:+.3f}")
        print(f"  行业排名: {stock_alpha.industry_rank}")
        print(f"  个股20日涨幅: {stock_alpha.stock_return_20d*100:+.1f}%")
        print(f"  相对强度: {stock_alpha.relative_strength*100:+.1f}%")
        print(f"  行业内分位: {stock_alpha.rs_percentile*100:.0f}%")
        print(f"  选股得分: {stock_alpha.selection_score:+.3f}")
