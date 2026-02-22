#!/usr/bin/env python3
"""
获取 A 股所有股票列表并保存到文件

用法:
    python get_full_stock_list.py

输出:
    在脚本同目录下生成 stock_list.txt 文件，每行包含: 代码 名称
"""

import akshare as ak
import pandas as pd
from pathlib import Path

# 输出文件路径
OUTPUT_FILE = Path(__file__).parent / "stock_list.txt"


def get_all_stocks():
    """获取沪深 A 股所有股票列表（包括主板、创业板、科创板）"""
    print("正在获取 A 股股票列表...")

    # 分别获取沪市和深市股票
    print("  获取沪市主板股票...")
    sh_df = ak.stock_info_sh_name_code()
    sh_df = sh_df[['证券代码', '证券简称']].copy()
    sh_df.columns = ['code', 'name']

    print("  获取深市股票（主板+创业板）...")
    sz_df = ak.stock_info_sz_name_code()
    sz_df = sz_df[['A股代码', 'A股简称']].copy()
    sz_df.columns = ['code', 'name']

    print("  获取科创板股票...")
    kc_df = ak.stock_zh_kcb_spot()
    kc_df = kc_df[['代码', '名称']].copy()
    kc_df.columns = ['code', 'name']
    # 去掉 sh 前缀
    kc_df['code'] = kc_df['code'].str.replace('sh', '')

    # 合并所有股票
    df = pd.concat([sh_df, sz_df, kc_df], ignore_index=True)
    # 去重
    df = df.drop_duplicates(subset=['code'], keep='first')
    print(f"  共获取 {len(df)} 只股票")
    return df


def save_stock_list(df):
    """保存股票列表到文件"""
    print(f"\n正在保存到文件: {OUTPUT_FILE}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write("# A股股票列表\n")
        f.write(f"# 总数: {len(df)} 只\n")
        f.write("# 格式: 代码 名称\n")
        f.write("-" * 40 + "\n")

        for _, row in df.iterrows():
            code = row['code']
            name = row['name']
            f.write(f"{code} {name}\n")

    print(f"成功保存 {len(df)} 只股票")


def print_statistics(df):
    """打印统计信息"""
    print(f"\n{'='*60}")
    print(f"A股股票统计")
    print(f"{'='*60}")
    print(f"总股票数: {len(df)}")

    # 按市场分类统计
    sh_main = len(df[df['code'].str.startswith('600')]) + len(df[df['code'].str.startswith('601')]) + \
              len(df[df['code'].str.startswith('603')]) + len(df[df['code'].str.startswith('605')])
    sh_kc = len(df[df['code'].str.startswith('688')])
    sh_b = len(df[df['code'].str.startswith('900')])

    sz_main = len(df[df['code'].str.startswith('000')]) + len(df[df['code'].str.startswith('001')])
    sz_cy = len(df[df['code'].str.startswith('300')]) + len(df[df['code'].str.startswith('301')])
    sz_b = len(df[df['code'].str.startswith('200')])

    beijing = len(df[df['code'].str.startswith('8')]) + len(df[df['code'].str.startswith('4')])

    print(f"\n市场分布:")
    print(f"  沪市主板:   {sh_main} 只")
    print(f"  科创板:     {sh_kc} 只")
    print(f"  深市主板:   {sz_main} 只")
    print(f"  创业板:     {sz_cy} 只")
    print(f"  北交所:     {beijing} 只")
    print(f"  B股:        {sh_b + sz_b} 只")
    print(f"  合计:       {len(df)} 只")

    # 显示前10只股票
    print(f"\n前10只股票:")
    print(f"{'代码':<10} {'名称':<15}")
    print("-" * 30)
    for _, row in df.head(10).iterrows():
        print(f"{row['code']:<10} {row['name']:<15}")

    print(f"{'='*60}\n")


def main():
    """主函数"""
    print("="*60)
    print("A股股票列表获取工具")
    print("="*60)

    try:
        # 获取股票列表
        df = get_all_stocks()

        # 打印统计
        print_statistics(df)

        # 保存到文件
        save_stock_list(df)

        print(f"文件已保存: {OUTPUT_FILE}")
        print("完成!")

    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
