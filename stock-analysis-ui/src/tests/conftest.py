import os
import shutil
import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).resolve().parent.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Mêmes garde-fous que l'UI desktop (ui/main_window.py) : éviter les
# segfaults de l'accélération C et les appels réseau de recommandations
# pendant les imports/tests qui n'en ont pas explicitement besoin.
os.environ.setdefault('QSI_DISABLE_C_ACCELERATION', '1')
os.environ.setdefault('QSI_CONSENSUS_OFFLINE', '1')
# Filet de sécurité : aucun test ne doit déclencher la complétion des profils
# d'instruments, qui consommerait le budget de requêtes yfinance. Les tests qui
# veulent l'exercer lèvent la variable eux-mêmes ET simulent yf.Ticker.
os.environ.setdefault('QSI_DISABLE_PROFILE_FETCH', '1')


@pytest.fixture(scope='session', autouse=True)
def isolate_real_database(tmp_path_factory):
    """
    --------------------------------------------------------------------------
    Objectif:
        Empecher tout test d'ecrire dans la base reelle. Redirige DB_PATH vers
        une COPIE temporaire de la vraie base, pour toute la session.

        Necessaire : sync_txt_to_sqlite() se termine par
        `DELETE FROM symbol_lists WHERE list_type = ? AND symbol NOT IN (...)`.
        Les tests appellent cette fonction avec des fixtures de deux ou trois
        symboles ; sans isolation, le DELETE vide les vraies listes de
        l'utilisateur (constate le 2026-07-30 : 2822 symboles populaires
        reduits a 7, restaures depuis les fichiers .txt).

        C'est une COPIE et non une base vide : sur une base vide, les 2822
        symboles seraient inconnus et declencheraient autant de requetes
        yfinance, alors que le budget de requetes est une contrainte dure.

    Entrees:
        tmp_path_factory (TempPathFactory): fixture pytest de repertoire temporaire

    Sorties:
        db_copy (Path): chemin de la base temporaire utilisee par les tests
    --------------------------------------------------------------------------
    """
    import config

    real_db = Path(config.DB_PATH)
    db_copy = tmp_path_factory.mktemp('db') / 'stock_analysis.db'
    if real_db.exists():
        shutil.copy2(real_db, db_copy)

    mp = pytest.MonkeyPatch()
    mp.setattr(config, 'DB_PATH', str(db_copy), raising=False)
    mp.setattr(config, 'MARKET_DATA_DB_PATH', str(db_copy), raising=False)

    # `from config import DB_PATH` copie la VALEUR dans le module importateur :
    # patcher config seul ne suffit pas. Les modules de test en font autant
    # (`from symbol_manager import DB_PATH`) et capturent la valeur avant meme
    # que cette fixture ne s'execute. On reecrit donc toute copie deja prise,
    # ou qu'elle se trouve — sinon un test ecrit dans la copie mais relit la
    # vraie base, et ses assertions portent sur le mauvais fichier.
    real_db_str = str(real_db)
    for module in list(sys.modules.values()):
        if module is None or not hasattr(module, '__dict__'):
            continue
        for attr in ('DB_PATH', 'MARKET_DATA_DB_PATH'):
            if getattr(module, attr, None) == real_db_str:
                mp.setattr(module, attr, str(db_copy), raising=False)

    yield db_copy
    mp.undo()
