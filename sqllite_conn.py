import sqlite3
from pathlib import Path

# ==================== 配置部分 ====================
DB_PATH = "/Users/yanglei/Documents/sqlite/sqlite-data/mydatabase.db"  # 与 docker 挂载路径对应
TABLE_NAME = "test_users"

# 确保目录存在（如果手动运行脚本时还没创建文件夹）
Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)


def main():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # 让查询结果可以像字典一样访问
    cursor = conn.cursor()

    try:
        # 检查表是否存在
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (TABLE_NAME,)
        )
        table_exists = cursor.fetchone() is not None

        if not table_exists:
            print(f"表 {TABLE_NAME} 不存在，正在创建并插入数据...")

            # 创建表（使用 STRICT 模式，推荐现代写法）
            cursor.execute(f"""
                CREATE TABLE {TABLE_NAME} (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    username    TEXT NOT NULL,
                    age         INTEGER,
                    email       TEXT,
                    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
                ) STRICT;
            """)

            # 插入 10 条测试数据
            users = [
                (f"user_{i}", 20 + i, f"user{i}@example.com")
                for i in range(1, 11)
            ]
            cursor.executemany(
                f"INSERT INTO {TABLE_NAME} (username, age, email) VALUES (?, ?, ?)",
                users
            )
            conn.commit()
            print(f"成功插入 {len(users)} 条数据")
        else:
            print(f"表 {TABLE_NAME} 已存在，跳过创建和插入，直接查询")

        # 查询 id = 5 的记录（假设自增从 1 开始，第 5 条）
        cursor.execute(f"SELECT * FROM {TABLE_NAME} WHERE id = 5")
        row = cursor.fetchone()

        if row:
            print("\n查询结果 (id=5):")
            print(f"  ID       : {row['id']}")
            print(f"  用户名   : {row['username']}")
            print(f"  年龄     : {row['age']}")
            print(f"  邮箱     : {row['email']}")
            print(f"  创建时间 : {row['created_at']}")
        else:
            print("没有找到 id=5 的记录（可能数据未正确插入）")

    except sqlite3.Error as e:
        print(f"SQLite 错误: {e}")

    finally:
        conn.close()
        print("\n数据库连接已关闭")


if __name__ == "__main__":
    main()