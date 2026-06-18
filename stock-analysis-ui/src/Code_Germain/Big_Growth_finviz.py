#!/usr/bin/env python3

from finvizfinance.screener.overview import Overview
from finvizfinance.screener.financial import Financial
from finvizfinance.screener.valuation import Valuation
from finvizfinance.screener.technical import Technical
from finvizfinance.screener.performance import Performance

import pandas as pd

BASE_FILTERS = {
    "Country": "USA",
    "Average Volume": "Over 200K",
    "Price": "Over $10",
}

MIN_SCORE = 4

def fetch_view(cls, filters, order="Ticker", limit=3000):
    obj = cls()
    obj.set_filter(filters_dict=filters)
    return obj.screener_view(order=order, limit=limit, ascend=True, verbose=0)

def normalize_ticker_col(df):
    for col in df.columns:
        if str(col).strip().lower() == "ticker":
            return df.rename(columns={col: "Ticker"})
    return df

def to_numeric_safe(series):
    return (
        series.astype(str)
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("x", "", regex=False)
        .str.strip()
        .replace({"-": None, "N/A": None, "nan": None})
        .pipe(pd.to_numeric, errors="coerce")
    )

def build_dataset():
    overview = normalize_ticker_col(fetch_view(Overview, BASE_FILTERS))
    financial = normalize_ticker_col(fetch_view(Financial, BASE_FILTERS))
    valuation = normalize_ticker_col(fetch_view(Valuation, BASE_FILTERS))
    technical = normalize_ticker_col(fetch_view(Technical, BASE_FILTERS))
    performance = normalize_ticker_col(fetch_view(Performance, BASE_FILTERS))

    df = overview.merge(financial, on="Ticker", how="inner", suffixes=("", "_fin"))
    df = df.merge(valuation, on="Ticker", how="left", suffixes=("", "_val"))
    df = df.merge(technical, on="Ticker", how="left", suffixes=("", "_tech"))
    df = df.merge(performance, on="Ticker", how="left", suffixes=("", "_perf"))

    return df

def score_big_growth(df):
    df = df.copy()

    if "Sales Q/Q" in df.columns:
        df["sales_qq"] = to_numeric_safe(df["Sales Q/Q"])
    else:
        df["sales_qq"] = pd.NA

    if "Gross Margin" in df.columns:
        df["gross_margin"] = to_numeric_safe(df["Gross Margin"])
    else:
        df["gross_margin"] = pd.NA

    if "P/E" in df.columns:
        df["pe"] = to_numeric_safe(df["P/E"])
    else:
        df["pe"] = pd.NA

    if "PEG" in df.columns:
        df["peg"] = to_numeric_safe(df["PEG"])
    else:
        df["peg"] = pd.NA

    if "Perf 3M" in df.columns:
        df["perf_3m"] = to_numeric_safe(df["Perf 3M"])
    else:
        df["perf_3m"] = pd.NA

    if "Perf 6M" in df.columns:
        df["perf_6m"] = to_numeric_safe(df["Perf 6M"])
    else:
        df["perf_6m"] = pd.NA

    if "Rel Volume" in df.columns:
        df["rel_volume"] = to_numeric_safe(df["Rel Volume"])
    else:
        df["rel_volume"] = pd.NA

    if "52W High" in df.columns:
        df["from_52w_high"] = to_numeric_safe(df["52W High"])
    else:
        df["from_52w_high"] = pd.NA

    df["C1_ok"] = df["sales_qq"].fillna(-999) > 20
    df["C2_ok"] = df["gross_margin"].fillna(-999) > 30

    pe_ok = df["pe"].between(0, 25, inclusive="neither").fillna(False)
    peg_ok = df["peg"].between(0, 1.5, inclusive="neither").fillna(False)
    high_ok = (df["from_52w_high"].fillna(999) <= -25)

    df["C3_hits"] = pe_ok.astype(int) + peg_ok.astype(int) + high_ok.astype(int)
    df["C3_ok"] = df["C3_hits"] >= 2

    perf3_ok = df["perf_3m"].between(8, 60, inclusive="both").fillna(False)
    perf6_ok = (df["perf_6m"].fillna(999) < 150)
    df["C4_ok"] = perf3_ok & perf6_ok

    df["C5_ok"] = df["rel_volume"].fillna(0) > 1.2

    df["score"] = (
        df["C1_ok"].astype(int) +
        df["C2_ok"].astype(int) +
        df["C3_ok"].astype(int) +
        df["C4_ok"].astype(int) +
        df["C5_ok"].astype(int)
    )

    df = df[df["score"] >= MIN_SCORE].copy()
    df = df.sort_values(["score", "sales_qq", "perf_3m"], ascending=[False, False, False])

    return df

def main():
    df = build_dataset()
    scored = score_big_growth(df)

    if scored.empty:
        print("Aucun stock avec score >= 4/5")
        return

    output_cols = [
        "Ticker", "Company", "Sector", "Industry", "Price",
        "score", "sales_qq", "gross_margin", "C3_hits",
        "perf_3m", "perf_6m", "rel_volume",
        "C1_ok", "C2_ok", "C3_ok", "C4_ok", "C5_ok"
    ]
    output_cols = [c for c in output_cols if c in scored.columns]

    print(scored[output_cols].to_string(index=False))
    scored.to_csv("big_growth_finviz_4of5.csv", index=False)
    print("\nRésultats sauvegardés dans big_growth_finviz_4of5.csv")

if __name__ == "__main__":
    main()