import sys
sys.path.insert(0, '.')
from qsi import compute_financial_derivatives

print('Testing compute_financial_derivatives...')
d = compute_financial_derivatives('AAPL')
print(f"ROE: {d.get('roe_val', 0)}")
print(f"PEG: {d.get('peg_ratio', 0)}")
print(f"Keys: {list(d.keys())}")
