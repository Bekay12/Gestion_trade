import pandas as pd
import numpy as np
import os
import sys

PROJECT_SRC = os.path.abspath(os.path.dirname(__file__))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from qsi import get_trading_signal

# Create mock price and volume series
rng = pd.date_range(end=pd.Timestamp.today(), periods=60, freq='D')
prices = pd.Series(np.linspace(100, 110, len(rng)) + np.random.normal(0, 0.5, len(rng)), index=rng)
volumes = pd.Series(np.random.randint(100000, 200000, len(rng)), index=rng)

result = get_trading_signal(prices, volumes, domaine='Technology', return_derivatives=True)
print('Result keys:', len(result))
print('Signal:', result[0])
print('Last price:', result[1])
print('Derivatives:', result[-1])
