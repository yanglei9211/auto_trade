#!/usr/bin/env python3
"""
回测结果可视化脚本

读取 eval_output.txt 中的回测数据，绘制总资产与上证指数的对比图
"""

import re
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import akshare as ak
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from const import STOCK_DB_PATH

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_backtest_output(file_path: str) -> List[Tuple[str, float]]:
    """
    解析回测输出文件，提取每日总资产
    
    返回:
        [(日期, 总资产), ...]
    """
    data = []
    current_date = None
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # 匹配日期行: "日期: 2025-01-02"
            date_match = re.match(r'日期:\s*(\d{4}-\d{2}-\d{2})', line)
            if date_match:
                current_date = date_match.group(1)
            
            # 匹配总资产行: "总资产: 99,995.00 RMB"
            if current_date and '总资产:' in line:
                # 提取数字（处理千分位逗号）
                amount_match = re.search(r'总资产:\s*([\d,]+\.?\d*)', line)
                if amount_match:
                    amount_str = amount_match.group(1).replace(',', '')
                    try:
                        total_value = float(amount_str)
                        data.append((current_date, total_value))
                    except ValueError:
                        pass
    
    return data


def get_index_data_from_db(index_code: str, start_date: str, end_date: str) -> List[Tuple[str, float]]:
    """
    从本地数据库获取指数数据
    
    参数:
        index_code: 指数代码 (如 '510300' 沪深300ETF, '510500' 中证500ETF)
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
        [(日期, 收盘价), ...]
    """
    try:
        conn = sqlite3.connect(STOCK_DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, close FROM etf_daily 
            WHERE code = ? AND date >= ? AND date <= ?
            ORDER BY date ASC
        """, (index_code, start_date, end_date))
        
        rows = cursor.fetchall()
        conn.close()
        
        if rows:
            return [(date, float(close)) for date, close in rows]
        else:
            return []
    
    except Exception as e:
        print(f"  从数据库获取指数 {index_code} 失败: {e}")
        return []


def get_all_index_data(start_date: str, end_date: str) -> Dict[str, List[Tuple[str, float]]]:
    """
    获取所有指数数据（沪深300、中证500）
    
    参数:
        start_date: 开始日期 (YYYY-MM-DD)
        end_date: 结束日期 (YYYY-MM-DD)
    
    返回:
        {
            'hs300': [(日期, 收盘价), ...],  # 沪深300
            'zz500': [(日期, 收盘价), ...],  # 中证500
        }
    """
    result = {}
    
    # 沪深300 ETF (510300)
    print(f"  获取沪深300数据...")
    hs300_data = get_index_data_from_db('510300', start_date, end_date)
    if hs300_data:
        result['hs300'] = hs300_data
        print(f"    成功获取 {len(hs300_data)} 条数据")
    else:
        print(f"    未找到数据")
    
    # 中证500 ETF (510500)
    print(f"  获取中证500数据...")
    zz500_data = get_index_data_from_db('510500', start_date, end_date)
    if zz500_data:
        result['zz500'] = zz500_data
        print(f"    成功获取 {len(zz500_data)} 条数据")
    else:
        print(f"    未找到数据")
    
    return result


def normalize_data(data: List[Tuple[str, float]], base_value: float = 100.0) -> List[Tuple[str, float]]:
    """
    将数据归一化到基准值（用于对比涨跌幅）
    
    参数:
        data: [(日期, 值), ...]
        base_value: 归一化基准值，默认100
    
    返回:
        [(日期, 归一化值), ...]
    """
    if not data:
        return []
    
    first_value = data[0][1]
    if first_value == 0:
        return data
    
    normalized = []
    for date, value in data:
        normalized_value = (value / first_value) * base_value
        normalized.append((date, normalized_value))
    
    return normalized


def plot_comparison(backtest_data: List[Tuple[str, float]], 
                   index_data_dict: Dict[str, List[Tuple[str, float]]],
                   initial_capital: float = 100000.0,
                   output_path: str = None):
    """
    绘制回测结果与指数对比图（三条线：策略、沪深300、中证500）
    
    参数:
        backtest_data: 回测数据 [(日期, 总资产), ...]
        index_data_dict: 指数数据字典 {'hs300': [...], 'zz500': [...]}
        initial_capital: 初始资金
        output_path: 输出图片路径
    """
    if not backtest_data:
        print("错误: 没有回测数据可供绘图")
        return
    
    # 归一化数据（都以100为起点）
    backtest_normalized = normalize_data(backtest_data, 100.0)
    
    # 转换日期格式
    backtest_dates = [datetime.strptime(d, '%Y-%m-%d') for d, _ in backtest_normalized]
    backtest_values = [v for _, v in backtest_normalized]
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), 
                                    gridspec_kw={'height_ratios': [3, 1]})
    
    # ========== 主图：归一化对比（三条线） ==========
    # 策略总资产 - 蓝色实线
    ax1.plot(backtest_dates, backtest_values, 'b-', linewidth=2.5, 
             label='策略总资产', marker='o', markersize=3)
    
    # 沪深300 - 红色虚线
    if 'hs300' in index_data_dict:
        hs300_normalized = normalize_data(index_data_dict['hs300'], 100.0)
        hs300_dates = [datetime.strptime(d, '%Y-%m-%d') for d, _ in hs300_normalized]
        hs300_values = [v for _, v in hs300_normalized]
        ax1.plot(hs300_dates, hs300_values, 'r--', linewidth=2, 
                 label='沪深300', alpha=0.8)
        hs300_final = hs300_values[-1]
        hs300_return = (hs300_final - 100) / 100 * 100
    else:
        hs300_return = None
    
    # 中证500 - 绿色点划线
    if 'zz500' in index_data_dict:
        zz500_normalized = normalize_data(index_data_dict['zz500'], 100.0)
        zz500_dates = [datetime.strptime(d, '%Y-%m-%d') for d, _ in zz500_normalized]
        zz500_values = [v for _, v in zz500_normalized]
        ax1.plot(zz500_dates, zz500_values, 'g-.', linewidth=2, 
                 label='中证500', alpha=0.8)
        zz500_final = zz500_values[-1]
        zz500_return = (zz500_final - 100) / 100 * 100
    else:
        zz500_return = None
    
    ax1.axhline(y=100, color='gray', linestyle=':', alpha=0.5, label='基准线')
    ax1.set_ylabel('归一化值 (起点=100)', fontsize=12)
    ax1.set_title('回测结果 vs 沪深300 vs 中证500', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 添加收益率标注
    final_value = backtest_values[-1]
    total_return = (final_value - 100) / 100 * 100
    
    # 策略收益标注（蓝色）
    ax1.annotate(f'策略: {total_return:+.1f}%', 
                 xy=(backtest_dates[-1], final_value),
                 xytext=(10, 15), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', fc='blue', alpha=0.3),
                 fontsize=11, color='blue', fontweight='bold')
    
    # 沪深300收益标注（红色）
    if hs300_return is not None:
        ax1.annotate(f'沪深300: {hs300_return:+.1f}%', 
                     xy=(hs300_dates[-1], hs300_final),
                     xytext=(10, -10), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.3),
                     fontsize=11, color='red')
    
    # 中证500收益标注（绿色）
    if zz500_return is not None:
        ax1.annotate(f'中证500: {zz500_return:+.1f}%', 
                     xy=(zz500_dates[-1], zz500_final),
                     xytext=(10, -35), textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.5', fc='green', alpha=0.3),
                     fontsize=11, color='green')
    
    # ========== 子图：实际总资产 ==========
    actual_values = [v for _, v in backtest_data]
    ax2.fill_between(backtest_dates, actual_values, alpha=0.3, color='blue')
    ax2.plot(backtest_dates, actual_values, 'b-', linewidth=1.5, label='总资产')
    ax2.axhline(y=initial_capital, color='gray', linestyle='--', alpha=0.5, label='初始资金')
    
    ax2.set_xlabel('日期', fontsize=12)
    ax2.set_ylabel('总资产 (RMB)', fontsize=12)
    ax2.legend(loc='upper left', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 格式化x轴日期
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    # 格式化y轴为货币格式
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'¥{x/10000:.1f}万'))
    
    plt.tight_layout()
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存到: {output_path}")
    else:
        plt.show()
    
    # 返回收益数据供统计使用
    return {
        'strategy': total_return,
        'hs300': hs300_return,
        'zz500': zz500_return
    }


def main():
    """主函数"""
    # 默认读取最新的 eval_output.txt
    script_dir = Path(__file__).parent
    eval_output_path = script_dir / "eval_output.txt"
    
    if not eval_output_path.exists():
        print(f"错误: 找不到回测结果文件: {eval_output_path}")
        print("请确保 eval.py 回测已完成并生成 eval_output.txt")
        return
    
    print(f"正在解析回测结果: {eval_output_path}")
    backtest_data = parse_backtest_output(str(eval_output_path))
    
    if not backtest_data:
        print("错误: 无法从输出文件中提取数据")
        return
    
    print(f"提取到 {len(backtest_data)} 个交易日的数据")
    print(f"日期范围: {backtest_data[0][0]} 至 {backtest_data[-1][0]}")
    
    # 获取日期范围
    start_date = backtest_data[0][0]
    end_date = backtest_data[-1][0]
    
    # 获取指数数据（沪深300 + 中证500）
    print(f"\n正在获取指数数据 ({start_date} 至 {end_date})...")
    index_data_dict = get_all_index_data(start_date, end_date)
    
    if not index_data_dict:
        print("警告: 未能获取任何指数数据，将只显示策略曲线")
    
    # 计算回测收益
    initial_value = backtest_data[0][1]
    final_value = backtest_data[-1][1]
    total_return = (final_value - initial_value) / initial_value * 100
    
    print(f"\n回测统计:")
    print(f"  初始资金: ¥{initial_value:,.2f}")
    print(f"  最终资产: ¥{final_value:,.2f}")
    print(f"  策略收益率: {total_return:+.2f}%")
    
    # 计算各指数收益
    returns = plot_comparison(backtest_data, index_data_dict, initial_value, None)
    
    if returns:
        if returns['hs300'] is not None:
            print(f"  沪深300: {returns['hs300']:+.2f}%")
            print(f"  超额收益(沪深300): {total_return - returns['hs300']:+.2f}%")
        if returns['zz500'] is not None:
            print(f"  中证500: {returns['zz500']:+.2f}%")
            print(f"  超额收益(中证500): {total_return - returns['zz500']:+.2f}%")
    
    # 保存图表（带时间戳，避免覆盖）
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = script_dir / f"backtest_comparison_{timestamp}.png"
    plot_comparison(backtest_data, index_data_dict, initial_value, str(output_path))


if __name__ == "__main__":
    main()
