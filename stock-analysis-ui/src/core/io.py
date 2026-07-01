"""
I/O des signaux : sauvegarde CSV évolutive.
Fonctions candidates à migrer ici progressivement :
  download_stock_data, get_cached_data, analyze_cache_status,
  load_symbol_lists, save_symbols_to_txt, load_symbols_from_txt,
  log_new_symbols, auto_register_analyzed_symbols.
"""
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_to_evolutive_csv(signals, filename="signaux_trading.csv"):
    """
    Sauvegarde les signaux dans un CSV évolutif qui conserve l'historique.
    - Crée le fichier s'il n'existe pas
    - Ajoute de nouveaux signaux
    - Met à jour les signaux existants
    - Conserve l'historique des changements
    """
    if not signals:
        return

    df_new = pd.DataFrame(signals)
    if df_new.empty:
        return

    detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df_new['detection_time'] = detection_time

    script_dir = Path(__file__).parent.parent  # remonte à src/
    signals_dir = script_dir / "signaux"
    file_path = signals_dir / filename

    if file_path.exists():
        try:
            df_old = pd.read_csv(file_path)
            df_combined = pd.concat([df_old, df_new], ignore_index=True)
            df_combined = df_combined.sort_values(
                by=['detection_time', 'Symbole', 'Fiabilite'],
                ascending=[True, False]
            )
            df_clean = df_combined.drop_duplicates(
                subset=['Symbole', 'Signal', 'Prix', 'RSI'],
                keep='first'
            )
        except Exception as e:
            print(f"⚠️ Erreur lecture CSV: {e}")
            df_clean = df_new
    else:
        df_clean = df_new

    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        base_name = Path(filename).stem
        archive_file = signals_dir / f"{base_name}_{timestamp}.csv"
        df_clean.to_csv(archive_file, index=False)
        df_clean.to_csv(file_path, index=False)
        print(f"💾 Signaux sauvegardés: {file_path} (archive: {archive_file})")
    except Exception as e:
        print(f"🚨 Erreur sauvegarde CSV: {e}")
