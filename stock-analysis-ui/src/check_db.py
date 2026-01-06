import sqlite3
c = sqlite3.connect('stock_analysis.db')
print('Tables:', c.execute('SELECT name FROM sqlite_master WHERE type="table"').fetchall())
print('symbol_lists count:', c.execute('SELECT COUNT(*) FROM symbol_lists').fetchone()[0])
print('popular count:', c.execute('SELECT COUNT(*) FROM symbol_lists WHERE list_type="popular"').fetchone()[0])
c.close()
