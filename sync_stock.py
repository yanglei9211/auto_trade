#!/usr/bin/env python3
"""
同步股票日线数据到 SQLite 数据库

用法:
    python sync_stock.py

配置:
    修改文件中的 STOCK_LIST、START_DATE、END_DATE 变量

示例:
    # 同步贵州茅台、比亚迪、宁德时代的数据
    STOCK_LIST = ["600519", "002594", "300750"]
    START_DATE = "20250101"
    END_DATE = "20260215"
"""

import sqlite3
from pathlib import Path
import akshare as ak
import pandas as pd
from typing import List, Tuple

# 导入常量配置
from const import STOCK_LIST, DEFAULT_START_DATE, DEFAULT_END_DATE, STOCK_DB_PATH, get_full_stock

# ==================== 配置部分（可修改） ====================

# 股票代码列表（从 const.py 导入，也可在此覆盖）
# 如果 STOCK_LIST 为空列表，则自动获取全部股票
# STOCK_LIST = STOCK_LIST

# 同步日期范围 (YYYYMMDD)（从 const.py 导入，也可在此覆盖）
START_DATE = DEFAULT_START_DATE
END_DATE = DEFAULT_END_DATE


def get_stock_list():
    """获取要同步的股票列表和名称映射"""
    if STOCK_LIST and len(STOCK_LIST) > 0:
        print(f"使用自定义股票列表，共 {len(STOCK_LIST)} 只")
        # 获取名称映射
        _, code_name_map = get_full_stock()
        return STOCK_LIST, code_name_map
    else:
        print("STOCK_LIST 为空，自动获取全部股票列表...")
        code_list, code_name_map = get_full_stock()
        print(f"获取到 {len(code_list)} 只股票")
        return code_list, code_name_map

# 数据库配置
DB_PATH = STOCK_DB_PATH
TABLE_NAME = "stock_daily"

# ==================== 数据库操作 ====================


def init_database(conn: sqlite3.Connection):
    """初始化数据库表结构"""
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT NOT NULL,
            name        TEXT,
            date        TEXT NOT NULL,
            open        REAL,
            close       REAL,
            high        REAL,
            low         REAL,
            volume      INTEGER,
            amount      REAL,
            amplitude   REAL,
            pct_change  REAL,
            change      REAL,
            turnover    REAL,
            created_at  TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(code, date)
        ) STRICT;
    """)
    # 创建索引以提高查询性能
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_stock_code_date ON {TABLE_NAME}(code, date);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_stock_date ON {TABLE_NAME}(date);
    """)
    conn.commit()


def get_stock_name(code: str, code_name_map: dict) -> str:
    """获取股票名称"""
    return code_name_map.get(code, code)  # 如果找不到名称，返回代码


def fetch_stock_data(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从 akshare 获取股票日线数据"""
    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )
    return df


def save_to_database(conn: sqlite3.Connection, code: str, name: str, df: pd.DataFrame) -> int:
    """将数据保存到 SQLite 数据库"""
    cursor = conn.cursor()

    # 重命名列以匹配数据库字段
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
        "换手率": "turnover"
    }
    df = df.rename(columns=column_mapping)

    # 插入或替换数据
    inserted_count = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (code, name, date, open, close, high, low, volume, amount, amplitude, pct_change, change, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code,
                name,
                row["date"],
                row["open"],
                row["close"],
                row["high"],
                row["low"],
                int(row["volume"]),
                row["amount"],
                row["amplitude"],
                row["pct_change"],
                row["change"],
                row["turnover"]
            ))
            inserted_count += 1
        except sqlite3.Error as e:
            print(f"    插入数据失败: {e}, 日期: {row['date']}")

    conn.commit()
    return inserted_count


def sync_single_stock(conn: sqlite3.Connection, code: str, start_date: str, end_date: str, code_name_map: dict) -> Tuple[int, str]:
    """同步单只股票数据"""
    print(f"\n正在同步股票: {code}")

    try:
        # 获取股票名称
        name = get_stock_name(code, code_name_map)
        print(f"  股票名称: {name}")

        # 获取股票数据
        df = fetch_stock_data(code, start_date, end_date)

        if df.empty:
            print(f"  警告: 未获取到数据")
            return 0, name

        print(f"  获取到 {len(df)} 条数据")
        print(f"  数据范围: {df['日期'].iloc[0]} 至 {df['日期'].iloc[-1]}")

        # 保存到数据库
        inserted_count = save_to_database(conn, code, name, df)
        print(f"  成功保存 {inserted_count} 条数据")

        return inserted_count, name

    except Exception as e:
        print(f"  错误: {e}")
        return 0, "未知"


def get_sync_summary(conn: sqlite3.Connection, stock_list: List[str]) -> dict:
    """获取同步汇总信息"""
    cursor = conn.cursor()
    summary = {}

    for code in stock_list:
        cursor.execute(f"""
            SELECT COUNT(*), MIN(date), MAX(date) 
            FROM {TABLE_NAME} 
            WHERE code = ?
        """, (code,))
        result = cursor.fetchone()
        summary[code] = {
            "count": result[0],
            "min_date": result[1],
            "max_date": result[2]
        }

    return summary


def main():
    # 确保目录存在
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)

    # 获取股票列表和名称映射
    stock_list, code_name_map = get_stock_list()

    print("=" * 60)
    print("股票数据同步")
    print("=" * 60)
    if len(stock_list) <= 20:
        print(f"股票列表: {', '.join(stock_list)}")
    else:
        print(f"股票列表: {', '.join(stock_list[:10])} ... (共 {len(stock_list)} 只)")
    print(f"开始日期: {START_DATE}")
    print(f"结束日期: {END_DATE}")
    print(f"数据库: {DB_PATH}")
    print("=" * 60)

    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)

        # 初始化表结构
        init_database(conn)
        print("数据库初始化完成")

        # 同步每只股票
        total_inserted = 0
        success_stocks = []
        failed_stocks = []

        for i, code in enumerate(stock_list, 1):
            print(f"\n[{i}/{len(stock_list)}] ", end="")
            count, name = sync_single_stock(conn, code, START_DATE, END_DATE, code_name_map)
            if count > 0:
                total_inserted += count
                success_stocks.append((code, name, count))
            else:
                failed_stocks.append(code)

        # 打印汇总
        print("\n" + "=" * 60)
        print("同步完成汇总")
        print("=" * 60)

        if success_stocks:
            print(f"\n成功同步 {len(success_stocks)} 只股票:")
            for code, name, count in success_stocks[:20]:  # 只显示前20只
                print(f"  {code} ({name}): {count} 条记录")
            if len(success_stocks) > 20:
                print(f"  ... 还有 {len(success_stocks) - 20} 只")

        if failed_stocks:
            print(f"\n同步失败 {len(failed_stocks)} 只股票:")
            for code in failed_stocks[:20]:
                print(f"  {code}")
            if len(failed_stocks) > 20:
                print(f"  ... 还有 {len(failed_stocks) - 20} 只")

        # 数据库统计
        summary = get_sync_summary(conn, stock_list)
        print(f"\n数据库统计:")
        total_records = 0
        for code, info in summary.items():
            total_records += info["count"]
        print(f"  总股票数: {len(stock_list)}")
        print(f"  成功同步: {len(success_stocks)}")
        print(f"  同步失败: {len(failed_stocks)}")
        print(f"  总记录数: {total_records} 条")

        print(f"\n总计: {total_records} 条记录")
        print("=" * 60)

    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            print("\n数据库连接已关闭")


if __name__ == "__main__":
    main()
