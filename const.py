#!/usr/bin/env python3
"""
常量配置文件

存放项目中使用的公共常量，如股票列表、默认日期等
"""

# ==================== 股票列表 ====================
# 用于数据同步和回测的股票代码列表
STOCK_LIST = []
STOCK_LIST_BAK = [
    "600809",   
    "002594",   # 比亚迪
    "300750",   # 宁德时代
    "000858",   # 五粮液
    "600036",   # 招商银行
    "688981",
    "601138",
    "000568",
    "605117",
    "600118",
    "600343",
    "688102"
]

# ==================== ETF 池子（用于辅助判断市场情绪）====================
ETF_POOL = [
    "510050",   # 上证50 ETF
    "510300",   # 沪深300 ETF
    "159915",   # 创业板 ETF
    "513010",   # 恒生科技 ETF
    "510500",   # 中证500 ETF
]

# ==================== 默认日期配置 ====================
# 默认同步开始日期 (YYYYMMDD)
DEFAULT_START_DATE = "20240101"

# 默认同步结束日期 (YYYYMMDD)
DEFAULT_END_DATE = "20260213"

# ==================== 数据库路径配置 ====================
# SQLite 数据库基础目录
DB_BASE_PATH = "/Users/yanglei/Documents/sqlite/sqlite-data"

# ETF 数据库路径
ETF_DB_PATH = f"{DB_BASE_PATH}/etf_data.db"

# 股票数据库路径
STOCK_DB_PATH = f"{DB_BASE_PATH}/stock_data.db"

# ==================== 交易参数配置 ====================
# 初始资金 (RMB)
INITIAL_CAPITAL = 100000

# 最大仓位比例
MAX_POSITION = 0.95

# 最小仓位比例
MIN_POSITION = 0.0

# 单次交易占总资金比例
SINGLE_TRADE_RATIO = 0.2

# 佣金率
COMMISSION_RATE = 0.0003

# 印花税率
STAMP_TAX_RATE = 0.001

# 最小交易单位（股）
MIN_TRADE_UNIT = 100

# 全量股票列表文件路径
STOCK_LIST_FILE = "/Users/yanglei/Documents/moltbot/scripts/stock_list.txt"

# 多进程配置
MAX_WORKERS = 8  # 最大并行进程数


def get_full_stock():
    """
    读取 stock_list.txt 文件，获取全量股票列表

    返回:
        Tuple[List[str], Dict[str, str]]: (股票代码列表, 代码到名称的映射字典)

    示例:
        code_list, code_name_map = get_full_stock()
        print(f"总股票数: {len(code_list)}")
        print(f"600000 名称: {code_name_map.get('600000')}")
    """
    from pathlib import Path
    from typing import List, Dict, Tuple

    code_list: List[str] = []
    code_name_map: Dict[str, str] = {}

    stock_file = Path(STOCK_LIST_FILE)

    if not stock_file.exists():
        print(f"警告: 股票列表文件不存在: {STOCK_LIST_FILE}")
        return code_list, code_name_map

    with open(stock_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行和注释行
            if not line or line.startswith('#') or line.startswith('-'):
                continue

            # 解析每行: "代码 名称"
            parts = line.split(maxsplit=1)
            if len(parts) >= 2:
                code = parts[0].strip()
                name = parts[1].strip()
                code_list.append(code)
                code_name_map[code] = name

    return code_list, code_name_map
