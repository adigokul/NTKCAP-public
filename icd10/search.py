import sqlite3

# 連接到SQLite數據庫
conn = sqlite3.connect('icd10.db')

# 定義模糊搜索函數
def fuzzy_search(query):
    cursor = conn.execute("SELECT Code, Description FROM icd10_fts WHERE Description MATCH ?", (f'{query}*',))
    results = cursor.fetchall()
    return results

# 使用者輸入
while True:
    user_input = input('symptoms (enter -1 to exit): ')
    if user_input == '-1':
        break

    # 執行模糊搜尋
    search_results = fuzzy_search(user_input)

    # 顯示結果
    if search_results:
        for result in search_results:
            print(f"Code: {result[0]}, Description: {result[1]}")
    else:
        print("No matching results found.")

conn.close()
