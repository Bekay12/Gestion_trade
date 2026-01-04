import sqlite3

conn = sqlite3.connect('signaux/optimization_hist.db')
cursor = conn.cursor()

# Check table info
cursor.execute("PRAGMA table_info(optimization_runs)")
cols = cursor.fetchall()
print('Colonnes disponibles:')
for col in cols:
    print(f'  {col[1]} ({col[2]})')

conn.close()
