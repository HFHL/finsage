import sqlite3

# 创建数据库文件并连接
conn = sqlite3.connect('reranker_data.db')

# 创建表
cursor = conn.cursor()

# SQL创建表结构
cursor.execute('''

ALTER TABLE chunks ADD COLUMN rewriten_q TEXT;

''')

# 提交并关闭连接
conn.commit()
conn.close()

print("Database and table created successfully.")