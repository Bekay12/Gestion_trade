import sqlite3
from math import floor

TARGET = 200
DB = 'stock_analysis.db'

def main():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()

    # Popular counts
    cur.execute("SELECT COALESCE(sector,'Unknown'), COUNT(*) FROM symbols WHERE list_type='popular' AND is_active=1 GROUP BY COALESCE(sector,'Unknown')")
    pop = dict(cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM symbols WHERE list_type='popular' AND is_active=1")
    total_pop = cur.fetchone()[0]

    # Optimization counts
    cur.execute("SELECT COALESCE(sector,'Unknown'), COUNT(*) FROM symbols WHERE list_type='optimization' AND is_active=1 GROUP BY COALESCE(sector,'Unknown')")
    opt = dict(cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM symbols WHERE list_type='optimization' AND is_active=1")
    total_opt = cur.fetchone()[0]

    # Targets per sector
    fracs = []
    targets = {}
    for sec, cnt in pop.items():
        p = cnt / total_pop if total_pop else 0
        raw = p * TARGET
        base = floor(raw)
        frac = raw - base
        targets[sec] = base
        fracs.append((frac, sec))
    needed = TARGET - sum(targets.values())
    for frac, sec in sorted(fracs, reverse=True)[:max(0, needed)]:
        targets[sec] += 1

    added_total = 0
    per_sector_added = {}

    for sec in sorted(pop.keys()):
        have = opt.get(sec, 0)
        want = targets.get(sec, 0)
        gap = max(0, want - have)
        if gap == 0:
            per_sector_added[sec] = 0
            continue
        cur.execute(
            """
            SELECT symbol, sector, market_cap_range, market_cap_value
            FROM symbols
            WHERE list_type='popular' AND is_active=1 AND COALESCE(sector,'Unknown')=?
              AND symbol NOT IN (SELECT symbol FROM symbols WHERE list_type='optimization' AND is_active=1)
            ORDER BY symbol ASC
            LIMIT ?
            """,
            (sec, gap)
        )
        rows = cur.fetchall()
        to_add = rows[:gap]
        for sym, sector, cap_range, cap_val in to_add:
            cur.execute(
                """
                INSERT OR REPLACE INTO symbols (symbol, sector, market_cap_range, market_cap_value, list_type, last_checked, is_active)
                VALUES (?, ?, ?, ?, 'optimization', CURRENT_TIMESTAMP, 1)
                """,
                (sym, sector, cap_range, cap_val)
            )
        per_sector_added[sec] = len(to_add)
        added_total += len(to_add)

    conn.commit()
    conn.close()
    print({"added_total": added_total, "per_sector_added": per_sector_added})

if __name__ == "__main__":
    main()
