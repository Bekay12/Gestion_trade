import sqlite3
from config import OPTIMIZATION_DB_PATH

conn = sqlite3.connect(OPTIMIZATION_DB_PATH)
cursor = conn.cursor()

# Check table info
cursor.execute("PRAGMA table_info(optimization_runs)")
cols = cursor.fetchall()
print('Colonnes disponibles:')
for col in cols:
    print(f'  {col[1]} ({col[2]})')

conn.close()
