import sys
import os
import time

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), ''))
print(time.strftime('%Y-%m-%d %H:%M:%S'), 'PROJECT_SRC ->', PROJECT_SRC, flush=True)
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

print(time.strftime('%Y-%m-%d %H:%M:%S'), 'About to import qsi', flush=True)
try:
    import qsi
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'SUCCESS: imported qsi from', getattr(qsi, '__file__', str(qsi)), flush=True)
except Exception as e:
    print(time.strftime('%Y-%m-%d %H:%M:%S'), 'FAILED to import qsi:', type(e).__name__, e, flush=True)
    raise
