#!/usr/bin/env python3
"""检查数据库中行业数据的完整性"""

import sqlite3
from const import STOCK_DB_PATH

def check_industry_data():
    conn = sqlite3.connect(STOCK_DB_PATH)
    cursor = conn.cursor()
    
    print("="*60)
    print("行业数据完整性检查")
    print("="*60)
    print(f"数据库: {STOCK_DB_PATH}")
    
    # 1. 检查行业日线数据表
    print("\n【1. 行业日线数据表 (industry_daily)】")
    cursor.execute("SELECT COUNT(*) FROM industry_daily")
    total_records = cursor.fetchone()[0]
    print(f"总记录数: {total_records}")
    
    cursor.execute("SELECT COUNT(DISTINCT industry) FROM industry_daily")
    industry_count = cursor.fetchone()[0]
    print(f"行业数量: {industry_count}")
    
    # 每个行业的记录数
    cursor.execute("""
        SELECT industry, COUNT(*) as cnt, MIN(date) as start_date, MAX(date) as end_date
        FROM industry_daily
        GROUP BY industry
        ORDER BY cnt DESC
    """)
    rows = cursor.fetchall()
    print(f"\n各行业数据情况:")
    print(f"{'行业':<15} {'记录数':>8} {'开始日期':<12} {'结束日期':<12}")
    print("-" * 50)
    for row in rows:
        print(f"{row[0]:<15} {row[1]:>8} {row[2]:<12} {row[3]:<12}")
    
    # 2. 检查日期范围
    print("\n【2. 日期范围检查】")
    cursor.execute("SELECT MIN(date), MAX(date) FROM industry_daily")
    min_date, max_date = cursor.fetchone()
    print(f"最早日期: {min_date}")
    print(f"最晚日期: {max_date}")
    
    # 3. 检查个股行业映射表
    print("\n【3. 个股行业映射表 (stock_industry)】")
    cursor.execute("SELECT COUNT(*) FROM stock_industry")
    mapping_count = cursor.fetchone()[0]
    print(f"映射记录数: {mapping_count}")
    
    cursor.execute("SELECT COUNT(DISTINCT industry) FROM stock_industry")
    mapped_industry_count = cursor.fetchone()[0]
    print(f"有映射的行业数: {mapped_industry_count}")
    
    # 每个行业的股票数量
    cursor.execute("""
        SELECT industry, COUNT(*) as cnt
        FROM stock_industry
        GROUP BY industry
        ORDER BY cnt DESC
    """)
    rows = cursor.fetchall()
    print(f"\n各行业股票数量:")
    for row in rows:
        print(f"  {row[0]}: {row[1]} 只")
    
    # 4. 数据质量检查
    print("\n【4. 数据质量检查】")
    
    # 检查是否有空值
    cursor.execute("""
        SELECT COUNT(*) FROM industry_daily 
        WHERE open IS NULL OR close IS NULL OR high IS NULL OR low IS NULL
    """)
    null_count = cursor.fetchone()[0]
    print(f"价格字段为空的记录: {null_count}")
    
    # 检查成交量为0的记录
    cursor.execute("""
        SELECT COUNT(*) FROM industry_daily WHERE volume = 0
    """)
    zero_volume = cursor.fetchone()[0]
    print(f"成交量为0的记录: {zero_volume}")
    
    # 5. 检查数据连续性（每个行业应该有多少个交易日）
    print("\n【5. 数据连续性检查】")
    cursor.execute("""
        SELECT industry, COUNT(DISTINCT date) as unique_dates
        FROM industry_daily
        GROUP BY industry
        HAVING unique_dates < 500
        ORDER BY unique_dates
    """)
    incomplete = cursor.fetchall()
    if incomplete:
        print("数据可能不完整的行业 (<500天):")
        for row in incomplete:
            print(f"  {row[0]}: {row[1]} 天")
    else:
        print("所有行业数据天数都 >= 500 天，数据完整")
    
    conn.close()
    print("\n" + "="*60)
    print("检查完成")
    print("="*60)

if __name__ == "__main__":
    check_industry_data()
