#!/usr/bin/env python3
"""tradable_filter_report.py

按票统计“可交易池过滤”结果：
- 输出通过率
- 输出被拦截的股票列表（code, name, reason, avg20_amount, listing_days）

用法示例：
  python3 tradable_filter_report.py --date 2025-02-13 --min-avg-amount 50000000 --min-listing-days 60 --limit 200

说明：
- 仅统计“无持仓时是否可开仓”的过滤条件（与 eval.py 一致：有持仓仍允许 SELL/退出）。
"""

from __future__ import annotations

import argparse
import sqlite3
from dataclasses import asdict
from typing import List, Tuple

from const import STOCK_DB_PATH, MIN_LISTING_DAYS, LIQUIDITY_MIN_AVG_AMOUNT_20D
from tradable_filter import TradableFilterConfig, check_tradable, get_stock_base_info


def get_all_codes(conn: sqlite3.Connection) -> List[str]:
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT code FROM stock_daily")
    return [r[0] for r in cur.fetchall()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="as-of date, YYYY-MM-DD")
    ap.add_argument("--min-avg-amount", type=float, default=LIQUIDITY_MIN_AVG_AMOUNT_20D)
    ap.add_argument("--min-listing-days", type=int, default=MIN_LISTING_DAYS)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--limit", type=int, default=300, help="max blocked rows to print")
    args = ap.parse_args()

    cfg = TradableFilterConfig(
        amount_window=args.window,
        min_avg_amount=args.min_avg_amount,
        min_listing_days=args.min_listing_days,
    )

    conn = sqlite3.connect(STOCK_DB_PATH)
    try:
        codes = get_all_codes(conn)

        total = len(codes)
        tradable = 0
        blocked = []  # (code,name,reason,avg,listing_days)

        for code in codes:
            name, _ = get_stock_base_info(conn, code)
            r = check_tradable(conn, code, args.date, cfg)
            if r.tradable:
                tradable += 1
            else:
                blocked.append((code, name, r.reason, r.avg_amount, r.listing_days))

        blocked.sort(key=lambda x: x[2])

        print("=== Tradable filter report ===")
        print(f"asof_date={args.date} window={cfg.amount_window} min_avg_amount={cfg.min_avg_amount:.0f} min_listing_days={cfg.min_listing_days}")
        print(f"total={total} tradable={tradable} blocked={len(blocked)} pass_rate={tradable/total*100:.2f}%")
        print()

        # 按原因聚合
        by_reason = {}
        for _, _, reason, *_ in blocked:
            key = reason.split('(')[0]
            by_reason[key] = by_reason.get(key, 0) + 1

        print("-- blocked by reason --")
        for k, v in sorted(by_reason.items(), key=lambda kv: -kv[1]):
            print(f"{k}: {v}")

        print("\n-- blocked samples --")
        for i, (code, name, reason, avg_amt, ld) in enumerate(blocked[: args.limit], 1):
            avg_txt = f"avg20={avg_amt:.0f}" if avg_amt else "avg20=NA"
            print(f"{i:>4}. {code} {name}\t{reason}\t{avg_txt}\tlisting_days={ld}")

        if len(blocked) > args.limit:
            print(f"... ({len(blocked)-args.limit} more blocked not shown; use --limit to increase)")

    finally:
        conn.close()


if __name__ == "__main__":
    main()
