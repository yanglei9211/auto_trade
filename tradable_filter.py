#!/usr/bin/env python3
"""tradable_filter.py

可交易池过滤：
1) 流动性过滤：近N日成交额均值 > 阈值
2) ST/退市过滤：name 包含 'ST' / '退' 等关键词（基于 stock_daily.name）
3) 次新过滤：上市 < MIN_LISTING_DAYS（在本项目数据条件下，用“该票在DB中最早交易日距离当前日”近似）

设计目标：
- 过滤逻辑尽量独立，可在 eval.py / 实盘等入口复用
- 默认使用 SQLite stock_daily 表（字段：code,date,name,amount...）

注意：
- 次新过滤用的是“数据库中最早出现日期”作为上市日近似；若未来接入真实上市日期，应替换。
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple


def _to_date(s: str) -> datetime:
    # 兼容 'YYYY-MM-DD' 和 'YYYYMMDD'
    if '-' in s:
        return datetime.strptime(s, "%Y-%m-%d")
    return datetime.strptime(s, "%Y%m%d")


@dataclass
class TradableFilterConfig:
    amount_window: int = 20
    min_avg_amount: float = 50_000_000  # 5000万
    min_listing_days: int = 60


@dataclass
class TradableResult:
    tradable: bool
    reason: str = ""
    avg_amount: float = 0.0
    listing_days: int = 999999


def get_stock_base_info(
    conn: sqlite3.Connection,
    code: str,
) -> Tuple[str, str]:
    """返回 (name, first_date_in_db)；若缺失返回 ("", "")."""
    cur = conn.cursor()
    cur.execute(
        "SELECT name, MIN(date) FROM stock_daily WHERE code=?",
        (code,),
    )
    row = cur.fetchone()
    if not row or row[0] is None or row[1] is None:
        return "", ""
    return str(row[0]), str(row[1])


def calc_avg_amount(
    conn: sqlite3.Connection,
    code: str,
    date: str,
    window: int = 20,
) -> float:
    """计算截至 date（含）最近 window 个交易日成交额均值。"""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT amount FROM stock_daily
        WHERE code=? AND date<=? AND amount IS NOT NULL
        ORDER BY date DESC
        LIMIT ?
        """,
        (code, date, window),
    )
    rows = cur.fetchall()
    if not rows:
        return 0.0
    vals = [float(r[0]) for r in rows if r[0] is not None]
    return sum(vals) / len(vals) if vals else 0.0


def is_st_or_delist(name: str) -> bool:
    if not name:
        return False
    # 常见：*ST、ST、退市/退 等
    bad_kw = ["ST", "退"]
    return any(k in name.upper() for k in bad_kw)


def listing_days_in_db(first_date: str, asof_date: str) -> int:
    if not first_date:
        return 0
    d0 = _to_date(first_date)
    d1 = _to_date(asof_date)
    return max((d1 - d0).days, 0)


def check_tradable(
    conn: sqlite3.Connection,
    code: str,
    date: str,
    cfg: TradableFilterConfig,
) -> TradableResult:
    name, first_date = get_stock_base_info(conn, code)

    if is_st_or_delist(name):
        return TradableResult(False, reason=f"ST/退市过滤(name={name})")

    ld = listing_days_in_db(first_date, date)
    if ld < cfg.min_listing_days:
        return TradableResult(False, reason=f"次新过滤(listing_days={ld} < {cfg.min_listing_days})", listing_days=ld)

    avg_amt = calc_avg_amount(conn, code, date, cfg.amount_window)
    if avg_amt < cfg.min_avg_amount:
        return TradableResult(False, reason=f"流动性过滤(avg20={avg_amt:.0f} < {cfg.min_avg_amount})", avg_amount=avg_amt, listing_days=ld)

    return TradableResult(True, avg_amount=avg_amt, listing_days=ld)


def batch_tradable_mask(
    db_path: str,
    codes: Iterable[str],
    date: str,
    cfg: TradableFilterConfig,
) -> Dict[str, TradableResult]:
    conn = sqlite3.connect(db_path)
    try:
        out: Dict[str, TradableResult] = {}
        for c in codes:
            out[c] = check_tradable(conn, c, date, cfg)
        return out
    finally:
        conn.close()
