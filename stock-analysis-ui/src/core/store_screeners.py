"""
Moteur de screening store-based — VUES UNIQUES sans équivalent Finviz.

Après consolidation sur Finviz (market-wide) pour les stratégies classiques,
il ne reste ici que les 2 screeners que Finviz ne sait PAS reproduire :

  • Combined / Profils : classification bi-score (Big Growth × Secure) en
    Dual Champion / Pure Growth / Pure Safe / Balanced.
  • Golden Cross récent : régime SMA50 > SMA200 avec croisement récent.

Ils lisent le store Parquet via `market_store.get_latest_features()` (1 requête
DuckDB, 0 requête yfinance) et restent donc limités au catalogue local.
Sortie : dict {title, headers, rows} pour ScreenerResultsDialog (symbole en col 0).
"""
import pandas as pd

from market_store import get_latest_features

MAX_ROWS = 80


def _num(df, col):
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series([float("nan")] * len(df), index=df.index, dtype="float64")


def _flag(df, col):
    if col in df.columns:
        return df[col].fillna(False).astype(bool)
    return pd.Series([False] * len(df), index=df.index)


def _cell(row, col, nd=2, is_int=False):
    if col not in row or pd.isna(row[col]):
        return None
    v = row[col]
    try:
        return int(v) if is_int else round(float(v), nd)
    except (TypeError, ValueError):
        return str(v)


def _result(title, headers, rows):
    return {"title": f"{title} — {len(rows)} résultat(s)", "headers": headers, "rows": rows}


def _empty(title):
    return {"title": title, "headers": [], "rows": []}


def screen_combined(symbols=None):
    """Combined — profils Dual Champion / Pure Growth / Pure Safe / Balanced.
    Croise le score Big Growth (c1..c5) et le score Secure (secure_score / s1..s7)."""
    df = get_latest_features(symbols)
    if df.empty:
        return _empty("Combined")
    bg = sum(_flag(df, f"c{i}_ok").astype(int) for i in range(1, 6))
    if "secure_score" in df.columns:
        sec = _num(df, "secure_score").fillna(0)
    else:
        sec = sum(_flag(df, f"s{i}_ok").astype(int) for i in range(1, 8))
    df = df.assign(_bg=bg, _sec=sec)

    def _profile(g, s):
        if g >= 3 and s >= 5:
            return "Dual Champion"
        if g >= 4 and s < 3:
            return "Pure Growth"
        if s >= 5 and g < 3:
            return "Pure Safe"
        if g >= 3 and s >= 3:
            return "Balanced"
        return None

    _order = {"Dual Champion": 0, "Pure Safe": 1, "Pure Growth": 2, "Balanced": 3}
    rows = []
    for _, r in df.iterrows():
        prof = _profile(int(r["_bg"]), int(r["_sec"]))
        if prof:
            rows.append([r["symbol"], prof, int(r["_bg"]), int(r["_sec"])])
    rows.sort(key=lambda x: (_order.get(x[1], 9), -x[2], -x[3]))
    rows = rows[:MAX_ROWS]
    return _result(
        "Combined — profils (Dual Champion / Pure Safe / Pure Growth / Balanced)",
        ["Symbole", "Profil", "Growth /5", "Safe /7"],
        rows,
    )


def screen_golden_cross(max_gap_pct: float = 5.0, symbols=None):
    """Golden Cross récent : SMA50 > SMA200 avec écart faible (croisement récent),
    prix au-dessus des deux moyennes."""
    df = get_latest_features(symbols)
    if df.empty:
        return _empty("Golden Cross")
    price = _num(df, "price"); s50 = _num(df, "sma50"); s200 = _num(df, "sma200")
    gap = (s50 / s200 - 1.0) * 100.0
    cond = (s50 > s200) & (gap <= max_gap_pct) & (price > s50)
    sel = df[cond.fillna(False)].assign(_g=gap).sort_values("_g").head(MAX_ROWS)
    rows = []
    for _, r in sel.iterrows():
        rows.append([
            r["symbol"], _cell(r, "price", 2), _cell(r, "sma50", 2),
            _cell(r, "sma200", 2), _cell(r, "_g", 2),
        ])
    return _result(
        f"Golden Cross récent (SMA50 > SMA200, écart ≤ {max_gap_pct:.0f}%)",
        ["Symbole", "Prix", "SMA50", "SMA200", "Écart %"],
        rows,
    )


# Registre nom → fonction (uniquement les vues sans équivalent Finviz).
SCREENERS = {
    "combined": screen_combined,
    "golden_cross": screen_golden_cross,
}
