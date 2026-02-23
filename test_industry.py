#!/usr/bin/env python3
"""测试akshare行业接口"""

import akshare as ak

print("测试获取行业列表...")
try:
    df = ak.stock_board_industry_name_em()
    print(f"成功！共 {len(df)} 个行业")
    print("\n前20个行业名称:")
    for i, name in enumerate(df['板块名称'].head(20), 1):
        print(f"  {i}. {name}")
    
    # 检查我们想要的行业是否在列表中
    our_industries = [
        "白酒", "电力", "银行", "证券", "保险",
        "半导体", "新能源汽车", "光伏", "医药"
    ]
    
    print("\n\n检查我们的行业列表:")
    all_names = df['板块名称'].tolist()
    for ind in our_industries:
        if ind in all_names:
            print(f"  ✓ {ind} - 存在")
        else:
            print(f"  ✗ {ind} - 不存在")
            # 查找相似名称
            similar = [n for n in all_names if ind in n or n in ind]
            if similar:
                print(f"    相似名称: {similar[:3]}")
    
    # 测试获取一个行业的数据
    print("\n\n测试获取'白酒'数据...")
    test_df = ak.stock_board_industry_hist_em(
        symbol="白酒",
        period="daily",
        start_date="20240101",
        end_date="20240131"
    )
    print(f"成功！获取到 {len(test_df)} 条数据")
    print(test_df.head())
    
except Exception as e:
    print(f"错误: {e}")
    import traceback
    traceback.print_exc()
