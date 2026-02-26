#!/usr/bin/env python3
"""
数据缓存模块

提供股票历史数据的内存缓存，避免频繁查询数据库
"""

import sqlite3
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

from const import STOCK_DB_PATH


@dataclass
class StockDataCache:
    """股票数据缓存"""
    code: str
    dates: List[str]
    opens: List[float]
    closes: List[float]
    highs: List[float]
    lows: List[float]
    volumes: List[int]
    amounts: List[float]
    amplitudes: List[float]
    pct_changes: List[float]
    changes: List[float]
    turnovers: List[float]

    def get_data_before(self, date: str, min_days: int = 20) -> Optional[Dict]:
        """
        获取指定日期之前的历史数据

        参数:
            date: 日期 (YYYY-MM-DD)
            min_days: 最少需要的天数

        返回:
            数据字典，如果数据不足返回None
        """
        try:
            idx = self.dates.index(date)
        except ValueError:
            # 日期不存在，找最近的前一个交易日
            idx = -1
            for i, d in enumerate(self.dates):
                if d >= date:
                    break
                idx = i
            if idx < 0:
                return None

        if idx < min_days:
            return None

        return {
            "dates": self.dates[:idx],
            "opens": self.opens[:idx],
            "closes": self.closes[:idx],
            "highs": self.highs[:idx],
            "lows": self.lows[:idx],
            "volumes": self.volumes[:idx],
            "amounts": self.amounts[:idx],
            "amplitudes": self.amplitudes[:idx],
            "pct_changes": self.pct_changes[:idx],
            "changes": self.changes[:idx],
            "turnovers": self.turnovers[:idx],
        }


class DataCacheManager:
    """数据缓存管理器"""

    def __init__(self, db_path: str = None, cache_dir: str = None):
        self.db_path = db_path or STOCK_DB_PATH
        self.cache_dir = Path(cache_dir) if cache_dir else Path(__file__).parent / ".cache"
        self.cache_dir.mkdir(exist_ok=True)
        self._cache: Dict[str, StockDataCache] = {}
        self._avg_amount_cache: Dict[str, Dict[str, float]] = {}  # code -> {date: avg_amount}

    def _get_cache_file(self, start_date: str, end_date: str) -> Path:
        """获取缓存文件路径"""
        return self.cache_dir / f"stock_cache_{start_date}_{end_date}.pkl"

    def load_from_db(self, stock_codes: List[str], start_date: str, end_date: str, progress_callback=None) -> Dict[str, StockDataCache]:
        """
        从数据库加载股票数据到内存缓存

        参数:
            stock_codes: 股票代码列表
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            progress_callback: 进度回调函数 (current, total)

        返回:
            缓存字典
        """
        print(f"[DataCache] 开始从数据库加载 {len(stock_codes)} 只股票数据...")
        start_time = datetime.now()

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        total = len(stock_codes)
        for i, code in enumerate(stock_codes):
            cursor.execute(
                """
                SELECT date, open, close, high, low, volume, amount,
                       amplitude, pct_change, change, turnover
                FROM stock_daily
                WHERE code = ? AND date >= ? AND date <= ?
                ORDER BY date ASC
                """,
                (code, start_date, end_date),
            )

            rows = cursor.fetchall()
            if len(rows) >= 20:  # 至少20天数据才缓存
                cache = StockDataCache(
                    code=code,
                    dates=[row["date"] for row in rows],
                    opens=[row["open"] for row in rows],
                    closes=[row["close"] for row in rows],
                    highs=[row["high"] for row in rows],
                    lows=[row["low"] for row in rows],
                    volumes=[row["volume"] for row in rows],
                    amounts=[row["amount"] for row in rows],
                    amplitudes=[row["amplitude"] for row in rows],
                    pct_changes=[row["pct_change"] for row in rows],
                    changes=[row["change"] for row in rows],
                    turnovers=[row["turnover"] for row in rows],
                )
                self._cache[code] = cache

                # 预计算20日平均成交额
                self._avg_amount_cache[code] = {}
                for j in range(20, len(rows)):
                    date = rows[j]["date"]
                    avg_amount = sum(rows[j - 19 : j + 1][k]["amount"] for k in range(20)) / 20
                    self._avg_amount_cache[code][date] = avg_amount

            if progress_callback:
                progress_callback(i + 1, total)
            elif (i + 1) % 500 == 0:
                print(f"[DataCache] 已加载 {i+1}/{total} 只股票...")

        conn.close()

        elapsed = (datetime.now() - start_time).total_seconds()
        print(f"[DataCache] 加载完成，共 {len(self._cache)} 只股票，耗时 {elapsed:.1f} 秒")
        return self._cache

    def save_to_disk(self, start_date: str, end_date: str):
        """保存缓存到磁盘"""
        cache_file = self._get_cache_file(start_date, end_date)
        with open(cache_file, "wb") as f:
            pickle.dump(
                {
                    "cache": self._cache,
                    "avg_amount": self._avg_amount_cache,
                },
                f,
            )
        print(f"[DataCache] 缓存已保存到 {cache_file}")

    def load_from_disk(self, start_date: str, end_date: str) -> bool:
        """从磁盘加载缓存"""
        cache_file = self._get_cache_file(start_date, end_date)
        if not cache_file.exists():
            return False

        print(f"[DataCache] 从磁盘加载缓存 {cache_file}...")
        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            self._cache = data["cache"]
            self._avg_amount_cache = data["avg_amount"]
        print(f"[DataCache] 已加载 {len(self._cache)} 只股票缓存")
        return True

    def get_stock_data(self, code: str, date: str, min_days: int = 20) -> Optional[Dict]:
        """
        获取股票历史数据

        参数:
            code: 股票代码
            date: 日期
            min_days: 最少需要的天数

        返回:
            数据字典或None
        """
        cache = self._cache.get(code)
        if not cache:
            return None
        return cache.get_data_before(date, min_days)

    def get_avg_amount(self, code: str, date: str) -> float:
        """获取20日平均成交额"""
        return self._avg_amount_cache.get(code, {}).get(date, 0.0)

    def get_cached_codes(self) -> List[str]:
        """获取已缓存的股票代码列表"""
        return list(self._cache.keys())


# 全局缓存实例
_global_cache: Optional[DataCacheManager] = None


def get_global_cache() -> DataCacheManager:
    """获取全局缓存实例"""
    global _global_cache
    if _global_cache is None:
        _global_cache = DataCacheManager()
    return _global_cache


def init_cache(stock_codes: List[str], start_date: str, end_date: str, use_disk_cache: bool = True) -> DataCacheManager:
    """
    初始化缓存

    参数:
        stock_codes: 股票代码列表
        start_date: 开始日期
        end_date: 结束日期
        use_disk_cache: 是否使用磁盘缓存

    返回:
        缓存管理器实例
    """
    global _global_cache
    _global_cache = DataCacheManager()

    # 尝试从磁盘加载
    if use_disk_cache and _global_cache.load_from_disk(start_date, end_date):
        # 检查是否包含所有需要的股票
        cached_codes = set(_global_cache.get_cached_codes())
        needed_codes = set(stock_codes)
        if needed_codes.issubset(cached_codes):
            return _global_cache
        else:
            missing = needed_codes - cached_codes
            print(f"[DataCache] 磁盘缓存缺少 {len(missing)} 只股票，重新加载...")

    # 从数据库加载
    _global_cache.load_from_db(stock_codes, start_date, end_date)

    if use_disk_cache:
        _global_cache.save_to_disk(start_date, end_date)

    return _global_cache
