import os
import sys
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Mêmes garde-fous que l'UI desktop (ui/main_window.py) : éviter les
# segfaults de l'accélération C et les appels réseau de recommandations
# pendant les imports/tests qui n'en ont pas explicitement besoin.
os.environ.setdefault('QSI_DISABLE_C_ACCELERATION', '1')
os.environ.setdefault('QSI_CONSENSUS_OFFLINE', '1')
