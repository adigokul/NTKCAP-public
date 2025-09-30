import pandas as pd
import sqlite3

# 讀取CSV文件
df = pd.read_csv(r'C:\Users\Hermes\Desktop\icd10\ICD_10_cm.csv')

# 連接到SQLite數據庫（如果不存在則創建）
conn = sqlite3.connect('icd10.db')

# 將DataFrame寫入數據庫
df.to_sql('icd10', conn, if_exists='replace', index=False)

# 創建全文搜索索引
conn.execute('CREATE VIRTUAL TABLE icd10_fts USING fts5(Code,USE,CM2023_英文名稱,CM2023_中文名稱,狀態)')
conn.execute('INSERT INTO icd10_fts SELECT Code,USE,CM2023_英文名稱,CM2023_中文名稱,狀態 FROM icd10')

conn.commit()
conn.close()

print("ICD-10代碼資料導入完成並建立全文搜索索引。")
