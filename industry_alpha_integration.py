#!/usr/bin/env python3
"""
行业 Alpha 因子整合方案

将行业 Alpha 因子集成到交易信号计算和回测流程中
"""

from typing import Dict, List, Optional, Tuple
from industry_alpha import IndustryAlphaCalculator, get_industry_alpha_rotation
from calc import Strategy, Signal, TradeDecision


class IndustryEnhancedStrategy:
    """
    行业 Alpha 增强策略

    在原有策略基础上，增加行业轮动 Alpha 因子
    """

    def __init__(self, base_strategy: Strategy, date: str, code: str,
                 db_path: str = None):
        """
        初始化行业增强策略

        参数:
            base_strategy: 基础策略实例
            date: 当前日期
            code: 股票代码
            db_path: 数据库路径
        """
        self.base_strategy = base_strategy
        self.date = date
        self.code = code
        self.alpha_calculator = IndustryAlphaCalculator(db_path)

        # 缓存行业 Alpha 数据
        self._industry_alpha_cache = None
        self._stock_alpha_cache = None

    def _get_industry_alpha(self) -> Optional[Dict]:
        """获取该股票所属行业的 Alpha 数据"""
        if self._industry_alpha_cache is not None:
            return self._industry_alpha_cache

        # 获取所有行业 Alpha 因子
        factors = self.alpha_calculator.calculate_industry_alpha_factors(self.date)

        # 获取该股票所属行业
        industry = self.alpha_calculator.analyzer.get_stock_industry(self.code)
        if not industry:
            return None

        # 找到该行业的 Alpha 因子
        for factor in factors:
            if factor.industry == industry:
                self._industry_alpha_cache = {
                    'industry': industry,
                    'alpha_score': factor.alpha_score,
                    'alpha_rank': factor.alpha_rank,
                    'momentum_20d': factor.momentum_20d,
                    'rs_vs_market': factor.rs_vs_market
                }
                return self._industry_alpha_cache

        return None

    def _get_stock_alpha(self) -> Optional[Dict]:
        """获取个股 Alpha 数据"""
        if self._stock_alpha_cache is not None:
            return self._stock_alpha_cache

        stock_alpha = self.alpha_calculator.calculate_stock_alpha_in_industry(
            self.code, self.date
        )

        if stock_alpha:
            self._stock_alpha_cache = {
                'selection_score': stock_alpha.selection_score,
                'relative_strength': stock_alpha.relative_strength,
                'rs_percentile': stock_alpha.rs_percentile,
                'industry_rank': stock_alpha.industry_rank
            }

        return self._stock_alpha_cache

    def calculate_enhanced_score(self) -> float:
        """
        计算行业增强后的综合得分

        策略逻辑：
        1. 基础得分来自技术分析（原策略）
        2. 行业 Alpha 因子加成（强势行业加分，弱势行业扣分）
        3. 个股在行业内的相对表现加成
        """
        # 基础得分
        base_score = self.base_strategy.get_composite_score()

        # 获取行业 Alpha
        industry_alpha = self._get_industry_alpha()
        if not industry_alpha:
            return base_score

        # 应用行业 Alpha 因子
        self.base_strategy.apply_industry_alpha(
            industry_alpha['alpha_score'],
            industry_alpha['alpha_rank']
        )

        # 获取行业 boost 因子
        boost = self.base_strategy.get_industry_boost_factor(
            industry_alpha['alpha_rank']
        )

        # 获取个股 Alpha
        stock_alpha = self._get_stock_alpha()
        if stock_alpha:
            # 个股在行业内的相对强度加成
            # 如果个股跑赢行业，额外加分
            rs_boost = 1 + (stock_alpha['rs_percentile'] - 0.5) * 0.2
            boost *= rs_boost

        # 重新计算综合得分（含行业 Alpha）
        enhanced_score = self.base_strategy.get_composite_score() * boost

        return enhanced_score

    def generate_enhanced_signal(self, current_hold: int, entry_price: float = 0,
                                  highest_price: float = 0, hold_days: int = 0) -> TradeDecision:
        """
        生成行业增强后的交易信号
        """
        # 计算增强得分
        enhanced_score = self.calculate_enhanced_score()

        # 获取行业 Alpha 信息（用于理由说明）
        industry_alpha = self._get_industry_alpha()
        stock_alpha = self._get_stock_alpha()

        # 生成基础信号
        base_decision = self.base_strategy.generate_signal(
            current_hold, entry_price, highest_price, hold_days
        )

        # 如果基础信号是 HOLD，但行业 Alpha 很强，考虑提升为 BUY
        if base_decision.signal == Signal.HOLD and enhanced_score > 0.3:
            if industry_alpha and industry_alpha['alpha_rank'] <= 5:
                # 强势行业中的股票，信号提升
                shares = self.base_strategy.risk_manager.calculate_position_size(
                    price=self.base_strategy.current_price,
                    atr=self.base_strategy.atr,
                    cash=self.base_strategy.initial_capital * 0.2,  # 单只股票最多20%
                    volatility=self.base_strategy.volatility
                )

                reason = f"{base_decision.reason} | 行业Alpha增强: {industry_alpha['industry']}排名{industry_alpha['alpha_rank']}"
                if stock_alpha:
                    reason += f", 个股相对强度{stock_alpha['relative_strength']*100:+.1f}%"

                return TradeDecision(Signal.BUY, shares, reason, min(enhanced_score, 1.0))

        # 如果基础信号是 BUY，但行业 Alpha 很弱，考虑降级为 HOLD
        elif base_decision.signal == Signal.BUY:
            if industry_alpha and industry_alpha['alpha_rank'] > 30:
                # 弱势行业中的股票，信号降级
                return TradeDecision(
                    Signal.HOLD, 0,
                    f"{base_decision.reason} | 行业Alpha过滤: {industry_alpha['industry']}排名{industry_alpha['alpha_rank']}(靠后)",
                    base_decision.confidence * 0.5
                )

        return base_decision


class IndustryRotationStrategy:
    """
    纯行业轮动策略

    完全基于行业 Alpha 因子进行选股，不依赖技术分析
    """

    def __init__(self, db_path: str = None):
        self.alpha_calculator = IndustryAlphaCalculator(db_path)

    def select_stocks(self, date: str, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        基于行业 Alpha 选择股票

        策略：
        1. 选出 Alpha 排名前5的行业
        2. 在每个行业中选出相对强度最高的前4只股票
        3. 返回最终股票列表

        参数:
            date: 日期
            top_n: 返回股票数量

        返回:
            [(股票代码, 选股得分), ...]
        """
        selected = []

        # 获取行业 Alpha 排名
        industry_factors = self.alpha_calculator.calculate_industry_alpha_factors(date)
        top_industries = industry_factors[:5]  # 前5名行业

        for industry_factor in top_industries:
            industry = industry_factor.industry

            # 获取该行业的所有股票
            conn = self.alpha_calculator.analyzer.db_path
            import sqlite3
            conn = sqlite3.connect(conn)
            cursor = conn.cursor()

            cursor.execute(f"""
                SELECT code FROM {self.alpha_calculator.mapping_table}
                WHERE industry = ?
            """, (industry,))
            stocks = [row[0] for row in cursor.fetchall()]
            conn.close()

            # 计算每只股票的行业 Alpha
            stock_scores = []
            for code in stocks[:20]:  # 限制每行业检查20只
                stock_alpha = self.alpha_calculator.calculate_stock_alpha_in_industry(
                    code, date
                )
                if stock_alpha:
                    stock_scores.append((code, stock_alpha.selection_score))

            # 按得分排序，取前4只
            stock_scores.sort(key=lambda x: x[1], reverse=True)
            selected.extend(stock_scores[:4])

        # 最终排序并返回
        selected.sort(key=lambda x: x[1], reverse=True)
        return selected[:top_n]


# 使用示例
def example_usage():
    """使用示例"""
    from datetime import datetime

    date = "2024-06-01"

    # 1. 获取行业 Alpha 轮动信号
    print("=" * 60)
    print("行业 Alpha 轮动信号")
    print("=" * 60)

    signal = get_industry_alpha_rotation(date)
    print(f"市场信号: {signal['market_signal']}")
    print(f"推荐行业: {', '.join(signal['top_industries'])}")
    print(f"回避行业: {', '.join(signal['avoid_industries'])}")

    # 2. 纯行业轮动选股
    print("\n" + "=" * 60)
    print("行业轮动选股结果")
    print("=" * 60)

    rotation_strategy = IndustryRotationStrategy()
    selected = rotation_strategy.select_stocks(date, top_n=10)

    print(f"{'排名':<4} {'代码':<10} {'得分':>8}")
    print("-" * 30)
    for i, (code, score) in enumerate(selected, 1):
        print(f"{i:<4} {code:<10} {score:>+8.3f}")

    # 3. 行业增强策略示例
    print("\n" + "=" * 60)
    print("行业增强策略示例 (600519)")
    print("=" * 60)

    # 这里需要实际的市场数据来创建 Strategy 实例
    # 示例代码展示如何整合
    print("使用方式:")
    print("  strategy = Strategy(data, current_price)")
    print("  enhanced = IndustryEnhancedStrategy(strategy, date, '600519')")
    print("  score = enhanced.calculate_enhanced_score()")
    print("  decision = enhanced.generate_enhanced_signal(...)")


if __name__ == "__main__":
    example_usage()
