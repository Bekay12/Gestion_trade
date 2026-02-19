import yfinance as yf

# Test direct avec yfinance
symbol = 'AAPL'
print(f"Testing {symbol}...")
ticker = yf.Ticker(symbol)
info = ticker.info

print(f"\nreturnOnEquity: {info.get('returnOnEquity')}")
print(f"trailingPE: {info.get('trailingPE')}")
print(f"pegRatio: {info.get('pegRatio')}")
print(f"revenueGrowth: {info.get('revenueGrowth')}")

# Calcul manuel
roe = info.get('returnOnEquity')
if roe:
    roe_pct = float(roe) * 100
    print(f"\nROE calculé: {roe_pct:.2f}%")
else:
    print("\n⚠️ Pas de ROE disponible")

# PEG
trailing_pe = info.get('trailingPE')
rev_growth = info.get('revenueGrowth')
if rev_growth:
    rev_growth_pct = float(rev_growth) * 100
    print(f"Revenue Growth: {rev_growth_pct:.2f}%")
    
    if trailing_pe and rev_growth_pct > 0:
        peg_calc = float(trailing_pe) / rev_growth_pct
        print(f"PEG calculé: {peg_calc:.2f}")
else:
    print("⚠️ Pas de Revenue Growth")

peg_direct = info.get('pegRatio')
if peg_direct:
    print(f"PEG direct: {peg_direct:.2f}")
