"""
Workers : threads Qt et helpers standalone extraits de main_window.py.
Aucune dépendance sur MainWindow — importable indépendamment.
"""
from PyQt5.QtCore import QThread, pyqtSignal, QObject
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
import sys
import os
import yfinance as yf

import qsi

PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

try:
    from symbol_manager import get_symbols_by_list_type, get_recent_symbols, get_symbol_info_from_db
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    SYMBOL_MANAGER_AVAILABLE = False
    get_symbol_info_from_db = None
    get_symbols_by_list_type = None
    get_recent_symbols = None


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _fetch_yf_info_with_timeout(symbol: str, timeout_sec: float = 2.0) -> dict:
    """Récupère yfinance.info avec timeout court pour éviter les blocages prolongés."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(lambda: yf.Ticker(symbol).info)
    try:
        info = future.result(timeout=timeout_sec)
        return info if isinstance(info, dict) else {}
    except (FuturesTimeoutError, Exception):
        return {}
    finally:
        executor.shutdown(wait=True, cancel_futures=False)


def _is_valid_ticker_info(symbol: str, info: dict) -> bool:
    """Vérifie si un ticker a au moins des données de prix dans yfinance.info."""
    try:
        has_market_price = (
            info.get('regularMarketPrice') is not None
            or info.get('currentPrice') is not None
            or info.get('lastPrice') is not None
        )
        has_market_cap = (
            info.get('marketCap') is not None
            or info.get('totalAssets') is not None
        )
        has_symbol = info.get('symbol') == symbol
        return bool(has_symbol and (has_market_price or has_market_cap))
    except Exception:
        return False


def _get_sector_cache_first(symbol: str) -> str:
    """Récupère le secteur en priorité DB/cache local, puis fallback yfinance timeout court."""
    try:
        if get_symbol_info_from_db:
            db_info = get_symbol_info_from_db(symbol) or {}
            sector = db_info.get('sector')
            if sector and sector != 'Inconnu':
                return sector
    except Exception:
        pass

    try:
        if getattr(qsi, 'get_pickle_cache', None) is not None:
            fin_cache = qsi.get_pickle_cache(symbol, 'financial', ttl_hours=24 * 365)
            if isinstance(fin_cache, dict):
                sector = fin_cache.get('sector')
                if sector and sector != 'Inconnu':
                    return sector
    except Exception:
        pass

    if getattr(qsi, 'OFFLINE_MODE', False):
        return 'Inconnu'

    info = _fetch_yf_info_with_timeout(symbol, timeout_sec=2.0)
    sector = info.get('sector', 'Inconnu') if isinstance(info, dict) else 'Inconnu'
    return sector if sector else 'Inconnu'


# ---------------------------------------------------------------------------
# Background threads
# ---------------------------------------------------------------------------

class AnalysisThread(QThread):
    """Runs analyse_signaux_populaires in a subprocess for memory isolation."""
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, symbols, mes_symbols, period="12mo", analysis_id=0, min_holding_days=7):
        super().__init__()
        self.symbols = symbols
        self.mes_symbols = mes_symbols
        self.period = period
        self._stop_requested = False
        self.analysis_id = analysis_id
        self.min_holding_days = max(1, int(min_holding_days))
        self._process = None

    def run(self):
        import subprocess, pickle, tempfile
        args_file = result_file = None
        try:
            self.progress.emit("Analyse en cours...")
            fiab_threshold = 30
            args = {
                'task': 'analyse_signaux',
                'symbols': self.symbols,
                'mes_symbols': self.mes_symbols,
                'period': self.period,
                'taux_reussite_min': fiab_threshold,
                'min_holding_days': self.min_holding_days,
            }
            fd_a, args_file = tempfile.mkstemp(suffix='_args.pkl')
            fd_r, result_file = tempfile.mkstemp(suffix='_result.pkl')
            os.close(fd_a); os.close(fd_r)
            with open(args_file, 'wb') as f:
                pickle.dump(args, f)

            worker_script = os.path.join(PROJECT_SRC, '_subprocess_worker.py')
            self._process = subprocess.Popen(
                [sys.executable, '-u', worker_script, args_file, result_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1, cwd=PROJECT_SRC,
            )
            for line in self._process.stdout:
                if self._stop_requested:
                    self._process.terminate()
                    break
                print(line, end='', flush=True)
            self._process.wait()

            if self._stop_requested:
                return
            if self._process.returncode != 0:
                self.error.emit(f"Analyse subprocess failed (code {self._process.returncode})")
                return
            with open(result_file, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('success'):
                result = payload['result']
                result['_analysis_id'] = self.analysis_id
                self.result_ready.emit(result)
            else:
                self.error.emit(payload.get('error', 'Unknown subprocess error'))
        except Exception as e:
            if not self._stop_requested:
                self.error.emit(str(e))
        finally:
            for fp in (args_file, result_file):
                if fp:
                    try: os.unlink(fp)
                    except OSError: pass

    def stop(self):
        self._stop_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()


class DownloadThread(QThread):
    """Runs download_stock_data in a subprocess for memory isolation."""
    result_ready = pyqtSignal(dict)
    error = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(self, symbols, period="12mo", do_backtest=False, analysis_id=0, min_holding_days=7, best_parameters_snapshot=None):
        super().__init__()
        self.symbols = symbols
        self.period = period
        self._stop_requested = False
        self.analysis_id = analysis_id
        self.min_holding_days = max(1, int(min_holding_days))
        self._process = None

    def run(self):
        import subprocess, pickle, tempfile
        args_file = result_file = None
        try:
            self.progress.emit("Téléchargement des données...")
            args = {
                'task': 'download',
                'symbols': self.symbols,
                'period': self.period,
            }
            fd_a, args_file = tempfile.mkstemp(suffix='_args.pkl')
            fd_r, result_file = tempfile.mkstemp(suffix='_result.pkl')
            os.close(fd_a); os.close(fd_r)
            with open(args_file, 'wb') as f:
                pickle.dump(args, f)

            worker_script = os.path.join(PROJECT_SRC, '_subprocess_worker.py')
            self._process = subprocess.Popen(
                [sys.executable, '-u', worker_script, args_file, result_file],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding='utf-8', errors='replace', bufsize=1, cwd=PROJECT_SRC,
            )
            for line in self._process.stdout:
                if self._stop_requested:
                    self._process.terminate()
                    break
                print(line, end='', flush=True)
            self._process.wait()

            if self._stop_requested:
                return
            if self._process.returncode != 0:
                self.error.emit(f"Download subprocess failed (code {self._process.returncode})")
                return
            with open(result_file, 'rb') as f:
                payload = pickle.load(f)
            if payload.get('success'):
                result = payload['result']
                result['_analysis_id'] = self.analysis_id
                self.result_ready.emit(result)
            else:
                self.error.emit(payload.get('error', 'Unknown subprocess error'))
        except Exception as e:
            if not self._stop_requested:
                self.error.emit(str(e))
        finally:
            for fp in (args_file, result_file):
                if fp:
                    try: os.unlink(fp)
                    except OSError: pass

    def stop(self):
        self._stop_requested = True
        if self._process and self._process.poll() is None:
            self._process.terminate()


class ParquetSyncThread(QThread):
    """Synchronise les symboles analysés dans le store Parquet en arrière-plan.

    Déclenché depuis on_analysis_complete() après chaque analyse UI.
    Premier appel : télécharge 3 ans → stocke tout (~750 lignes/symbole).
    Appels suivants : ne télécharge que le delta depuis la dernière entrée.
    """
    sync_done = pyqtSignal(int, int)   # (ok_count, err_count)

    def __init__(self, symbols, parent=None):
        super().__init__(parent)
        self.symbols = list(symbols)

    def run(self):
        try:
            from market_store import refresh_symbol_incremental
        except Exception as exc:
            print(f"[Parquet] ERREUR import market_store : {exc}")
            import traceback; traceback.print_exc()
            self.sync_done.emit(0, len(self.symbols))
            return

        MAX_WORKERS = 6

        print(f"[Parquet] démarrage sync de {len(self.symbols)} symboles (workers={MAX_WORKERS})…")
        ok = err = 0

        def _sync_one(symbol):
            r = refresh_symbol_incremental(symbol)
            return symbol, r.get('status', '?'), r.get('feature_rows', 0), None

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_sync_one, sym): sym for sym in self.symbols}
            for future in as_completed(futures):
                if self.isInterruptionRequested():
                    for f in futures:
                        f.cancel()
                    break
                sym = futures[future]
                try:
                    symbol, status, rows, _ = future.result()
                    print(f"[Parquet] {symbol}: {status} ({rows} lignes features)")
                    ok += 1
                except Exception as exc:
                    print(f"[Parquet] {sym} ERREUR: {exc}")
                    err += 1

        print(f"[Parquet] sync terminé : {ok} ok, {err} erreurs sur {len(self.symbols)} symboles")
        self.sync_done.emit(ok, err)

    def stop(self):
        self.requestInterruption()
        if not self.wait(2000):
            self.terminate()
            self.wait(500)


class LogCapture:
    """Captures stdout/stderr and writes both to QTextEdit and to original stdout/stderr.
    Uses a Qt signal to marshal UI writes back to the GUI thread."""
    class _Emitter(QObject):
        message = pyqtSignal(str)

    def __init__(self, text_edit):
        self.text_edit = text_edit
        self.original_stdout = sys.__stdout__
        self.original_stderr = sys.__stderr__
        self._emitter = self._Emitter()
        self._emitter.message.connect(self._append_message)

    def _append_message(self, message):
        """Append only from GUI thread via queued Qt signal delivery."""
        try:
            self.text_edit.append(message)
        except Exception:
            pass

    def write(self, message):
        """Write message to QTextEdit and original stdout."""
        try:
            if message and message.strip():
                self._emitter.message.emit(message.rstrip())
                self.original_stdout.write(message)
                self.original_stdout.flush()
        except Exception:
            try:
                self.original_stdout.write(message)
                self.original_stdout.flush()
            except Exception:
                pass

    def flush(self):
        try:
            self.original_stdout.flush()
        except Exception:
            pass

    def isatty(self):
        return False
