import sys
sys.path.insert(0, '.')
from qsi import compute_financial_derivatives

symbols = ['RDDT', 'RZLV', 'SEMR', 'SEZL', 'SG', 'SLSR', 'TIGR', 'TNGX', 'TRVI', 'TSLA']

for symbol in symbols:
    print(f"\n{'='*60}")
    print(f"Testing {symbol}...")
    d = compute_financial_derivatives(symbol)
    roe = d.get('roe_val', 0)
    peg = d.get('peg_ratio', 0)
    print(f"  ROE: {roe:.2f}%")
    print(f"  PEG: {peg:.2f}")
