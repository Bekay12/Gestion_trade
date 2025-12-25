import pandas as pd
from pathlib import Path
import requests

def normalize(symbol: str) -> str:
    return symbol.strip().upper().replace('.', '-')

url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

def main():
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        resp = requests.get(url, headers=headers, timeout=20)
        resp.raise_for_status()
        html = resp.text

        tables = pd.read_html(html)
        df = None
        name = None
        for t in tables:
            found = None
            for c in t.columns:
                if str(c).strip().lower().startswith('symbol') or str(c).strip().lower().startswith('ticker'):
                    found = (t, c)
                    break
            if found:
                df, name = found
                break
        if df is None:
            print("No symbol column found")
            return 1
        syms = [normalize(str(x)) for x in df[name].astype(str).tolist() if str(x).strip()]
        syms = list(dict.fromkeys(syms))
        Path('sp500_symbols.txt').write_text('\n'.join(syms), encoding='utf-8')
        print(f"Wrote {len(syms)} symbols to sp500_symbols.txt")
        return 0
    except Exception as e:
        print("Fetch failed:", e)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
