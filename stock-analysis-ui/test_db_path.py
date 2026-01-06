import sqlite3
import os
print(f"CWD: {os.getcwd()}")
print(f"DB exists at 'stock_analysis.db': {os.path.exists('stock_analysis.db')}")
print(f"DB exists at 'src/stock_analysis.db': {os.path.exists('src/stock_analysis.db')}")

if os.path.exists('src/stock_analysis.db'):
    c = sqlite3.connect('src/stock_analysis.db')
    print('Popular count:', c.execute('SELECT COUNT(*) FROM symbol_lists WHERE list_type="popular"').fetchone()[0])
    c.close()
