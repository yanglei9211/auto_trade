#!/usr/bin/env python3
"""
同步 ETF 日线数据到 SQLite 数据库

用法:
    python sync_etf.py <start_date> <end_date> <code>

参数:
    start_date: 开始日期，格式 YYYYMMDD
    end_date: 结束日期，格式 YYYYMMDD
    code: ETF 代码，例如 510050

示例:
    python sync_etf.py 20260201 20260215 510050
"""

import sys
import sqlite3
from pathlib import Path
import akshare as ak
import pandas as pd

# 导入常量配置
from const import ETF_DB_PATH

# ==================== 配置部分 ====================
DB_PATH = ETF_DB_PATH
TABLE_NAME = "etf_daily"

# 确保目录存在
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def init_database(conn: sqlite3.Connection):
    """初始化数据库表结构"""
    cursor = conn.cursor()
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            code        TEXT NOT NULL,
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
        CREATE INDEX IF NOT EXISTS idx_etf_code_date ON {TABLE_NAME}(code, date);
    """)
    conn.commit()


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


def save_to_database(conn: sqlite3.Connection, code: str, df: pd.DataFrame):
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
                (code, date, open, close, high, low, volume, amount, amplitude, pct_change, change, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                code,
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
            print(f"插入数据失败: {e}, 日期: {row['date']}")

    conn.commit()
    return inserted_count


def main():
    # 检查参数
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)

    start_date = sys.argv[1]
    end_date = sys.argv[2]
    code = sys.argv[3]

    # 验证日期格式
    if len(start_date) != 8 or len(end_date) != 8:
        print("错误: 日期格式应为 YYYYMMDD，例如 20260201")
        sys.exit(1)

    print(f"开始同步 ETF 数据:")
    print(f"  代码: {code}")
    print(f"  开始日期: {start_date}")
    print(f"  结束日期: {end_date}")
    print(f"  数据库: {DB_PATH}")

    try:
        # 连接数据库
        conn = sqlite3.connect(DB_PATH)

        # 初始化表结构
        init_database(conn)
        print("数据库初始化完成")

        # 获取 ETF 数据
        print(f"正在获取 {code} 的日线数据...")
        df = fetch_etf_data(code, start_date, end_date)

        if df.empty:
            print("未获取到数据，请检查代码和日期范围")
            return

        print(f"获取到 {len(df)} 条数据")
        print("\n数据预览:")
        print(df.to_string())

        # 保存到数据库
        inserted_count = save_to_database(conn, code, df)
        print(f"\n成功保存 {inserted_count} 条数据到数据库")

        # 查询已保存的数据量
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE code = ?", (code,))
        total_count = cursor.fetchone()[0]
        print(f"该 ETF 在数据库中共有 {total_count} 条记录")

    except ak.AkShareException as e:
        print(f"获取数据失败: {e}")
        sys.exit(1)
    except sqlite3.Error as e:
        print(f"数据库错误: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"发生错误: {e}")
        sys.exit(1)
    finally:
        if 'conn' in locals():
            conn.close()
            print("\n数据库连接已关闭")


if __name__ == "__main__":
    main()
