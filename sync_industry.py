#!/usr/bin/env python3
"""
同步行业指数日线数据到 SQLite 数据库

用法:
    python sync_industry.py                    # 同步所有行业，使用默认时间
    python sync_industry.py <start> <end>      # 同步所有行业，指定时间

数据存储:
    - stock_data.db / industry_daily: 行业指数日线数据
    - stock_data.db / stock_industry: 个股行业映射表
"""

import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import akshare as ak
import pandas as pd

from const import STOCK_DB_PATH, DEFAULT_START_DATE, DEFAULT_END_DATE

# 数据库配置
DB_PATH = STOCK_DB_PATH
TABLE_NAME = "industry_daily"
MAPPING_TABLE = "stock_industry"

# 重点关注的行业列表（使用akshare标准名称，与const.py保持一致）
from const import INDUSTRY_LIST


def init_database():
    """初始化数据库表结构"""
    Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # 行业指数日线数据表
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            industry TEXT NOT NULL,
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
            PRIMARY KEY (industry, date)
        )
    """)

    # 个股行业映射表
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {MAPPING_TABLE} (
            code TEXT PRIMARY KEY,
            industry TEXT,
            industry_code TEXT,
            update_date TEXT
        )
    """)

    # 创建索引
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_industry_date ON {TABLE_NAME}(industry, date);
    """)
    cursor.execute(f"""
        CREATE INDEX IF NOT EXISTS idx_stock_industry ON {MAPPING_TABLE}(industry);
    """)

    conn.commit()
    conn.close()


def get_all_industries() -> List[str]:
    """获取所有可用的行业名称"""
    try:
        df = ak.stock_board_industry_name_em()
        return df['板块名称'].tolist()
    except Exception as e:
        print(f"获取行业列表失败: {e}")
        return []


def find_valid_industries(wanted_list: List[str], all_industries: List[str]) -> List[str]:
    """
    从想要的行业列表中，找出实际存在的行业名称
    
    参数:
        wanted_list: 想要的行业列表
        all_industries: 所有可用的行业列表
    
    返回:
        实际存在的行业列表
    """
    valid = []
    not_found = []
    
    for wanted in wanted_list:
        if wanted in all_industries:
            valid.append(wanted)
        else:
            # 尝试找相似名称
            similar = [a for a in all_industries if wanted.replace('Ⅱ', '').replace('Ⅲ', '') in a 
                      or a.replace('Ⅱ', '').replace('Ⅲ', '') in wanted]
            if similar:
                # 使用第一个相似的
                valid.append(similar[0])
                print(f"  提示: '{wanted}' 未找到，使用 '{similar[0]}'")
            else:
                not_found.append(wanted)
    
    if not_found:
        print(f"  警告: 以下行业未找到对应名称: {not_found}")
    
    return valid


def fetch_industry_data(industry: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从akshare获取行业指数日线数据"""
    try:
        # 转换日期格式为 YYYYMMDD
        start = start_date.replace("-", "")
        end = end_date.replace("-", "")
        
        df = ak.stock_board_industry_hist_em(
            symbol=industry,
            start_date=start,
            end_date=end,
            period="日k",
            adjust=""
        )
        return df
    except Exception as e:
        print(f"  获取 {industry} 数据失败: {e}")
        return pd.DataFrame()


def save_industry_data(df: pd.DataFrame, industry: str) -> int:
    """保存行业数据到数据库"""
    if df.empty:
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

    df = df.rename(columns=column_mapping)
    df["industry"] = industry

    columns = [
        "industry", "date", "open", "close", "high", "low",
        "volume", "amount", "amplitude", "pct_change", "change", "turnover"
    ]
    df = df[columns]
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(f"""
                INSERT OR REPLACE INTO {TABLE_NAME} 
                (industry, date, open, close, high, low, volume, amount, 
                 amplitude, pct_change, change, turnover)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, tuple(row))
            inserted += 1
        except Exception as e:
            print(f"  插入数据失败: {e}")
            continue

    conn.commit()
    conn.close()
    return inserted


def sync_stock_industry_mapping():
    """同步个股行业映射表"""
    print("\n同步个股行业映射...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    total_mapped = 0
    for industry in INDUSTRY_LIST:
        try:
            df = ak.stock_board_industry_cons_em(symbol=industry)
            if df.empty:
                continue

            for _, row in df.iterrows():
                code = row['代码']
                name = row['名称']

                try:
                    cursor.execute(f"""
                        INSERT OR REPLACE INTO {MAPPING_TABLE} 
                        (code, industry, update_date)
                        VALUES (?, ?, ?)
                    """, (code, industry, datetime.now().strftime("%Y-%m-%d")))
                    total_mapped += 1
                except Exception as e:
                    continue

            print(f"  {industry}: {len(df)} 只股票")

        except Exception as e:
            print(f"  {industry} 映射失败: {e}")
            continue

    conn.commit()
    conn.close()
    print(f"共映射 {total_mapped} 只股票到行业")


def sync_single_industry(industry: str, start_date: str, end_date: str) -> int:
    """同步单个行业的数据"""
    print(f"\n同步行业: {industry}")

    df = fetch_industry_data(industry, start_date, end_date)
    if df.empty:
        print(f"  无数据")
        return 0

    inserted = save_industry_data(df, industry)
    print(f"  保存 {inserted} 条记录")

    # 显示统计
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE industry = ?", (industry,))
    total = cursor.fetchone()[0]
    conn.close()
    print(f"  该行业共 {total} 条历史记录")

    return inserted


def main():
    # 解析参数
    if len(sys.argv) == 1:
        start_date = DEFAULT_START_DATE
        end_date = DEFAULT_END_DATE
        print("使用默认时间范围")
    elif len(sys.argv) == 3:
        start_date = sys.argv[1]
        end_date = sys.argv[2]
    else:
        print("用法: python sync_industry.py [start_date] [end_date]")
        print("示例: python sync_industry.py 20240101 20250101")
        sys.exit(1)

    print(f"{'='*60}")
    print(f"行业数据同步")
    print(f"{'='*60}")
    print(f"时间范围: {start_date} - {end_date}")
    print(f"数据库: {DB_PATH}")

    # 获取所有可用行业
    print("\n获取可用行业列表...")
    all_industries = get_all_industries()
    if not all_industries:
        print("错误: 无法获取行业列表")
        sys.exit(1)
    print(f"akshare共有 {len(all_industries)} 个行业")

    # 找出有效的行业名称
    valid_industries = find_valid_industries(INDUSTRY_LIST, all_industries)
    print(f"有效行业: {len(valid_industries)}/{len(INDUSTRY_LIST)}")

    # 初始化数据库
    init_database()

    # 同步行业指数数据
    print(f"\n{'='*60}")
    print("同步行业指数数据...")
    total_inserted = 0
    for industry in valid_industries:
        inserted = sync_single_industry(industry, start_date, end_date)
        total_inserted += inserted

    # 同步个股行业映射
    sync_stock_industry_mapping()

    # 最终统计
    print(f"\n{'='*60}")
    print("同步完成!")
    print(f"目标行业: {len(INDUSTRY_LIST)} 个")
    print(f"有效行业: {len(valid_industries)} 个")
    print(f"共插入: {total_inserted} 条行业指数记录")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(DISTINCT industry) FROM {TABLE_NAME}")
    industry_count = cursor.fetchone()[0]
    cursor.execute(f"SELECT COUNT(*) FROM {MAPPING_TABLE}")
    mapping_count = cursor.fetchone()[0]
    conn.close()

    print(f"数据库中行业: {industry_count} 个")
    print(f"个股映射: {mapping_count} 只")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
