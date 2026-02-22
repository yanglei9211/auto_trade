#!/usr/bin/env python3
"""
同步 ETF 日线数据到 SQLite 数据库

用法:
    # 方式1: 同步 ETF_POOL 中所有 ETF（使用 const.py 中的起止时间）
    python sync_etf.py

    # 方式2: 同步指定日期范围的 ETF_POOL
    python sync_etf.py <start_date> <end_date>

    # 方式3: 同步单个指定 ETF
    python sync_etf.py <start_date> <end_date> <code>

参数:
    start_date: 开始日期 (YYYYMMDD)，可选，默认使用 const.DEFAULT_START_DATE
    end_date: 结束日期 (YYYYMMDD)，可选，默认使用 const.DEFAULT_END_DATE
    code: ETF 代码，可选，默认同步 ETF_POOL 中的所有 ETF

示例:
    python sync_etf.py                    # 同步 ETF_POOL 所有 ETF，使用默认时间
    python sync_etf.py 20260201 20260215  # 同步 ETF_POOL 所有 ETF，指定时间
    python sync_etf.py 20260201 20260215 510050  # 同步指定 ETF
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path

import akshare as ak
import pandas as pd

# 导入常量配置
from const import STOCK_DB_PATH, ETF_POOL, DEFAULT_START_DATE, DEFAULT_END_DATE

# 数据库配置 - ETF数据写入stock_data.db的独立表
DB_PATH = STOCK_DB_PATH
TABLE_NAME = "etf_daily"


def init_database():
    """初始化数据库，创建表结构"""
    # 确保目录存在
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            code TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL,
            close REAL,
            high REAL,
            low REAL,
            volume INTEGER,
            amount REAL,
            amplitude REAL,
            pct_change REAL,
            change REAL,
            turnover REAL,
            PRIMARY KEY (code, date)
        )
    """)

    # 创建索引
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_etf_code_date ON {TABLE_NAME}(code, date);
    """)

    conn.commit()
    conn.close()


def fetch_etf_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从 akshare 获取 ETF 日线数据"""
    df = ak.fund_etf_hist_em(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"
    )
    return df


def save_to_database(df, code: str):
    """将数据保存到 SQLite 数据库"""
    if df.empty:
        print(f"警告: 没有数据需要保存")
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 列名映射
    column_mapping = {
        "日期": "date",
        "开盘": "open",
        "收盘": "close",
        "最高": "high",
        "最低": "low",
        "成交量": "volume",
        "成交额": "amount",
        "振幅": "amplitude",
        "涨跌幅": "pct_change",
        "涨跌额": "change",
        "换手率": "turnover",
    }

    # 重命名列
    df = df.rename(columns=column_mapping)

    # 添加 code 列
    df["code"] = code

    # 选择需要的列
    columns = [
        "code", "date", "open", "close", "high", "low",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover"
    ]
    df = df[columns]

    # 转换日期格式
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    # 插入数据
    inserted_count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (code, date, open, close, high, low, volume, amount, amplitude, pct_change, change, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                row["code"], row["date"], row["open"], row["close"],
                row["high"], row["low"], row["volume"], row["amount"],
                row["amplitude"], row["pct_change"], row["change"], row["turnover"]
            ))
            inserted_count += 1
        except sqlite3.Error as e:
            print(f"插入数据失败: {e}")
            continue

    conn.commit()
    conn.close()

    return inserted_count


def sync_single_etf(code: str, start_date: str, end_date: str) -> int:
    """
    同步单个 ETF 的数据
    
    返回:
        插入的记录数
    """
    print(f"\n{'='*60}")
    print(f"同步 ETF: {code}")
    print(f"日期范围: {start_date} - {end_date}")
    print(f"{'='*60}")

    try:
        # 获取 ETF 数据
        print("正在从 akshare 获取数据...")
        df = fetch_etf_data(code, start_date, end_date)
        print(f"获取到 {len(df)} 条数据")

        if df.empty:
            print("警告: 没有获取到数据")
            return 0

        # 保存到数据库
        print("正在保存到数据库...")
        inserted = save_to_database(df, code)
        print(f"成功保存 {inserted} 条数据")

        # 显示统计信息
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE code = ?", (code,))
        total_count = cursor.fetchone()[0]
        conn.close()

        print(f"该 ETF 在数据库中共有 {total_count} 条记录")
        return inserted

    except Exception as e:
        print(f"错误: {e}")
        return 0


def main():
    # 解析命令行参数
    if len(sys.argv) == 1:
        # 方式1: 无参数，使用 const 中的默认时间和 ETF_POOL
        start_date = DEFAULT_START_DATE
        end_date = DEFAULT_END_DATE
        etf_codes = ETF_POOL
        print("使用默认配置:")
        print(f"  起止时间: {start_date} - {end_date}")
        print(f"  ETF池子: {', '.join(etf_codes)}")
    elif len(sys.argv) == 3:
        # 方式2: 指定起止时间，同步 ETF_POOL 中所有 ETF
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        etf_codes = ETF_POOL
        print("使用指定时间，同步 ETF_POOL:")
        print(f"  起止时间: {start_date} - {end_date}")
        print(f"  ETF池子: {', '.join(etf_codes)}")
    elif len(sys.argv) == 4:
        # 方式3: 同步单个指定 ETF
        start_date = sys.argv[1]
        end_date = sys.argv[2]
        etf_codes = [sys.argv[3]]
        print("同步指定 ETF:")
        print(f"  起止时间: {start_date} - {end_date}")
        print(f"  ETF代码: {etf_codes[0]}")
    else:
        print("用法:")
        print("  python sync_etf.py                    # 同步 ETF_POOL，使用默认时间")
        print("  python sync_etf.py <start> <end>      # 同步 ETF_POOL，指定时间")
        print("  python sync_etf.py <start> <end> <code> # 同步指定 ETF")
        sys.exit(1)

    # 验证日期格式
    try:
        datetime.strptime(start_date, "%Y%m%d")
        datetime.strptime(end_date, "%Y%m%d")
    except ValueError:
        print("错误: 日期格式应为 YYYYMMDD")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"开始同步 ETF 数据")
    print(f"{'='*60}")
    print(f"数据库: {DB_PATH}")

    # 初始化数据库
    init_database()

    # 同步所有指定的 ETF
    total_inserted = 0
    for code in etf_codes:
        inserted = sync_single_etf(code, start_date, end_date)
        total_inserted += inserted

    # 最终统计
    print(f"\n{'='*60}")
    print(f"同步完成!")
    print(f"总共插入 {total_inserted} 条记录")
    print(f"{'='*60}")

    # 显示每个 ETF 的总记录数
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    print("\n各 ETF 数据汇总:")
    for code in etf_codes:
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE code = ?", (code,))
        count = cursor.fetchone()[0]
        print(f"  {code}: {count} 条记录")
    conn.close()


if __name__ == "__main__":
    main()
