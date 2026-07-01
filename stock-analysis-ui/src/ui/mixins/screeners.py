"""
ScreenersMixin — méthodes de découverte de symboles via screeners externes.
Toutes ces méthodes sont des méthodes de MainWindow, déplacées ici pour
réduire la taille de main_window.py.
"""
from PyQt5.QtWidgets import QApplication, QMessageBox, QProgressDialog
from PyQt5.QtCore import Qt
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError, as_completed
from datetime import datetime
import yfinance as yf


class ScreenersMixin:

    def _select_all_items(self, list_widget):
        list_widget.selectAll()

    def _present_screener_results(self, title, headers, rows):
        """Ouvre un dialog interactif (table triable + cases à cocher) et injecte
        les symboles cochés dans le champ d'analyse. Retourne la liste injectée
        (vide si l'utilisateur annule)."""
        from ui.dialogs import ScreenerResultsDialog
        dlg = ScreenerResultsDialog(title, headers, rows, parent=self)
        if dlg.exec_() != ScreenerResultsDialog.Accepted:
            return []
        selected = list(dict.fromkeys(dlg.selected_symbols()))
        if selected:
            self.symbol_input.setText(", ".join(selected))
            if hasattr(self, "_status"):
                self._status(f"{len(selected)} symbole(s) injecté(s) dans le champ d'analyse")
        return selected

    def _compute_daily_top_movers(self, top_n=30):
        """Récupère les top movers Yahoo (day_gainers/day_losers) via yfinance.screen,
        sans dépendre des listes locales."""
        now = datetime.now()
        cached = getattr(self, '_daily_movers_cache', None)
        if isinstance(cached, dict):
            ts = cached.get('timestamp')
            if isinstance(ts, datetime):
                if (now - ts).total_seconds() <= 300:
                    return cached

        n = max(1, int(top_n))

        def _extract_entries(screen_payload):
            quotes = []
            if isinstance(screen_payload, dict):
                quotes = screen_payload.get('quotes') or []
            extracted = []
            for q in quotes:
                if not isinstance(q, dict):
                    continue
                symbol = str(q.get('symbol') or '').strip().upper()
                if not symbol:
                    continue
                pct = q.get('regularMarketChangePercent')
                if isinstance(pct, dict):
                    pct = pct.get('raw', pct.get('fmt'))
                try:
                    pct_val = float(pct)
                except Exception:
                    continue
                extracted.append((symbol, pct_val))
            return extracted

        def _fetch_screeners():
            winners_payload = yf.screen('day_gainers', count=n)
            losers_payload = yf.screen('day_losers', count=n)
            return _extract_entries(winners_payload), _extract_entries(losers_payload)

        executor = ThreadPoolExecutor(max_workers=1)
        future = executor.submit(_fetch_screeners)
        try:
            winners, losers = future.result(timeout=12)
        except FuturesTimeoutError:
            raise TimeoutError("Timeout Yahoo Finance: impossible de récupérer les top movers dans le délai imparti.")
        finally:
            executor.shutdown(wait=False, cancel_futures=True)

        cache_payload = {
            'timestamp': now,
            'winners': winners[:n],
            'losers': losers[:n],
        }
        self._daily_movers_cache = cache_payload
        return cache_payload

    def _show_top_daily_movers(self, mover_type):
        """Affiche les top movers du jour et injecte les symboles dans le champ d'analyse."""
        mover_type = (mover_type or '').strip().lower()
        if mover_type not in {'winners', 'losers'}:
            mover_type = 'winners'

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            top_data = self._compute_daily_top_movers(top_n=30)
        except TimeoutError as e:
            QMessageBox.warning(self, "Timeout", str(e))
            return
        except Exception as e:
            QMessageBox.warning(self, "Erreur", f"Impossible de charger les top movers Yahoo: {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        entries = top_data.get(mover_type, []) if isinstance(top_data, dict) else []
        if not entries:
            QMessageBox.information(
                self,
                "Information",
                "Aucune donnée intraday/journalière disponible pour calculer les top movers.",
            )
            return

        title = "Top 30 Winners du jour" if mover_type == 'winners' else "Top 30 Losers du jour"
        rows = [(sym, round(pct, 2)) for sym, pct in entries]
        self._present_screener_results(title, ["Symbole", "Variation (%)"], rows)

    def _show_yahoo_screener(self):
        """Charge jusqu'à 30 symboles du screener Yahoo sélectionné et les injecte dans le champ d'analyse."""
        screener_key = self.screener_combo.currentData()
        screener_label = self.screener_combo.currentText()
        if not screener_key:
            return

        if screener_key.startswith("_store_"):
            self._show_store_screener(screener_key[len("_store_"):])
            return

        if screener_key.startswith("_fvw_"):
            self._show_finviz_market_screener(screener_key[len("_fvw_"):])
            return

        if screener_key == "_events_48h":
            self._show_events_48h_screener()
            return
        if screener_key == "_events_48h_mes_coko":
            self._show_events_48h_screener(list_sources=("mes_list", "coko_list"))
            return
        if screener_key == "_finviz_gapper":
            self._show_finviz_gapper_screener()
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            payload = yf.screen(screener_key, count=30)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "Erreur", f"Impossible de charger le screener '{screener_label}': {e}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        quotes = (payload or {}).get('quotes') or []
        entries = []
        for q in quotes:
            if not isinstance(q, dict):
                continue
            symbol = str(q.get('symbol') or '').strip().upper()
            if not symbol:
                continue
            pct = q.get('regularMarketChangePercent')
            if isinstance(pct, dict):
                pct = pct.get('raw', pct.get('fmt'))
            try:
                pct_val = float(pct)
            except Exception:
                pct_val = None
            entries.append((symbol, pct_val))

        entries = entries[:30]
        if not entries:
            QMessageBox.information(self, "Yahoo Screener", f"Aucun résultat pour le screener « {screener_label} ».")
            return

        rows = [(sym, (round(pct, 2) if pct is not None else None)) for sym, pct in entries]
        self._present_screener_results(
            f"Yahoo Screener — {screener_label} (max 30)",
            ["Symbole", "Variation (%)"],
            rows,
        )

    def _show_store_screener(self, engine_key):
        """Exécute une vue store-only (Combined/Profils ou Golden Cross) et présente
        le résultat. engine_key ∈ core.store_screeners.SCREENERS. 0 requête yfinance."""
        try:
            from core.store_screeners import SCREENERS
        except Exception as e:
            QMessageBox.warning(self, "Screener", f"Moteur de screening indisponible : {e}")
            return
        fn = SCREENERS.get(engine_key)
        if fn is None:
            QMessageBox.warning(self, "Screener", f"Screener inconnu : {engine_key}")
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = fn()
        except Exception as e:
            QApplication.restoreOverrideCursor()
            import traceback
            QMessageBox.warning(self, "Screener", f"Erreur pendant le screening :\n{e}\n\n{traceback.format_exc()}")
            return
        finally:
            QApplication.restoreOverrideCursor()

        rows = res.get("rows") or []
        if not rows:
            QMessageBox.information(
                self, "Screener",
                f"{res.get('title', 'Screener')}\n\nAucun résultat dans le store local (catalogue).\n"
                "Lance des analyses pour remplir le store Parquet."
            )
            return
        self._present_screener_results(res["title"], res["headers"], rows)

    def _show_finviz_market_screener(self, preset_key):
        """Screener Finviz *market-wide* : découvre des titres dans tout le marché US
        (hors catalogue local). 1 requête Finviz, 0 requête yfinance."""
        progress = QProgressDialog(
            "Interrogation de Finviz (marché entier)…", None, 0, 0, self
        )
        progress.setWindowTitle("Finviz — Screener marché")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        try:
            from core.finviz_screeners import run_preset
            res = run_preset(preset_key, limit=100)
        except Exception as e:
            progress.close()
            import traceback
            QMessageBox.warning(
                self, "Finviz",
                f"Erreur pendant le screening Finviz :\n{e}\n\n{traceback.format_exc()}"
            )
            return
        finally:
            progress.close()

        rows = res.get("rows") or []
        if not rows:
            QMessageBox.information(
                self, "Finviz",
                f"{res.get('title', 'Screener Finviz')}\n\nAucun résultat renvoyé par Finviz."
            )
            return
        if hasattr(self, "_status"):
            self._status(f"Finviz : {len(rows)} titres trouvés sur le marché entier")
        self._present_screener_results(res["title"], res["headers"], rows)

    def _show_finviz_gapper_screener(self):
        """Screener Finviz via finvizfinance :
          • Market Cap  : < $300M (Micro + Nano)
          • Prix        : $1 – $20
          • Vol. moyen  : > 500K
          • Vol. relatif: > 2×
          • Float Short : > 10%
          • Gap Up      : ≥ 5%
        """
        from finvizfinance.screener.overview import Overview

        progress = QProgressDialog(
            "Interrogation de Finviz en cours…",
            "Annuler", 0, 0, self
        )
        progress.setWindowTitle("Finviz Gapper Screener")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        FILTERS = {
            "Market Cap.":     "-Micro (under $300mln)",
            "Price":           "Under $20",
            "Average Volume":  "Over 500K",
            "Relative Volume": "Over 2",
            "Float Short":     "Over 10%",
            "Gap":             "Up 5%",
        }

        df = None
        try:
            import finvizfinance.util as _fv_util
            from curl_cffi.requests import Session as CurlSession
            _orig_session = _fv_util.session
            _fv_util.session = CurlSession(impersonate="chrome")
            try:
                fov = Overview()
                fov.set_filter(filters_dict=FILTERS)
                df = fov.screener_view(order="Change", limit=100, ascend=False)
            finally:
                _fv_util.session = _orig_session
        except Exception as e:
            progress.close()
            import traceback
            QMessageBox.warning(
                self, "Finviz Gapper — Erreur",
                f"Erreur lors de la requête Finviz :\n\n{e}\n\n" + traceback.format_exc()
            )
            return

        # NB : ne pas tester progress.wasCanceled() APRÈS close() — close() met
        # le flag à True, ce qui ferait toujours sortir avant l'affichage.
        progress.close()

        if df is None or len(df) == 0:
            QMessageBox.information(
                self, "Finviz Gapper",
                "Finviz ne retourne aucun résultat pour ces filtres.\n\n"
                "Ce screener nécessite que :\n"
                "  • le marché US soit ouvert (9h30-16h ET)\n"
                "  • des actions aient gappé de >= 5% aujourd'hui"
            )
            return

        def _pct(v):
            try:
                return float(str(v).replace("%", "").replace(",", "").strip())
            except Exception:
                return None

        def _num(v):
            s = str(v).replace(",", "").strip()
            for suffix, mult in [("B", 1e9), ("M", 1e6), ("K", 1e3)]:
                if s.upper().endswith(suffix):
                    try:
                        return float(s[:-1]) * mult
                    except Exception:
                        return None
            try:
                return float(s)
            except Exception:
                return None

        results = []
        for _, row in df.iterrows():
            sym = str(row.get("Ticker") or "").strip().upper()
            if not sym:
                continue
            chg   = _pct(row.get("Change"))
            # Finviz renvoie Change en fraction (0.4776) → convertir en % (47.76).
            chg   = round(chg * 100, 2) if chg is not None else None
            price = _num(row.get("Price"))
            vol   = _num(row.get("Volume"))
            cap   = _num(row.get("Market Cap"))
            cap_m = round(cap / 1_000_000, 1) if cap else None
            results.append((sym, chg, price, cap_m, vol))

        if not results:
            QMessageBox.information(self, "Finviz Gapper",
                                    "Finviz a répondu mais aucun symbole exploitable n'a été trouvé.")
            return

        top30 = results[:30]
        rows = [
            (sym, chg, price, cap_m, (int(vol) if vol is not None else None))
            for sym, chg, price, cap_m, vol in top30
        ]
        self._present_screener_results(
            f"Finviz Gapper (Cap<$300M, $1-$20, Gap>=5%) — {len(top30)} résultat(s)",
            ["Symbole", "Gap (%)", "Prix ($)", "Cap (M$)", "Volume"],
            rows,
        )

    def _show_events_48h_screener(self, list_sources=None):
        """Interroge Yahoo Finance (calendar) sur les symboles des listes demandées et affiche
        ceux qui ont un earnings ou un ex-dividende dans les 48 h a venir.
        list_sources: tuple de noms d'attributs QListWidget a utiliser.
                      None = toutes les listes + Parquet store."""
        from datetime import datetime, timedelta, date as date_type
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        all_symbols: set = set()
        _default_sources = ("popular_list", "mes_list", "coko_list", "random_list", "recent_list")
        _sources = list_sources if list_sources is not None else _default_sources

        for attr in _sources:
            lw = getattr(self, attr, None)
            if lw is None:
                continue
            for i in range(lw.count()):
                item = lw.item(i)
                sym = (item.data(Qt.UserRole) or item.text() or "").strip().upper()
                if sym:
                    all_symbols.add(sym)

        if list_sources is None:
            try:
                from market_store import PARQUET_DIR
                features_dir = PARQUET_DIR / "features"
                if features_dir.exists():
                    for d in features_dir.iterdir():
                        if d.is_dir():
                            s = d.name.replace("symbol=", "")
                            if s:
                                all_symbols.add(s.upper())
            except Exception:
                pass

        if not all_symbols:
            src_label = "Mes + Coko" if list_sources is not None else "vos listes"
            QMessageBox.information(self, "Evenements 48h",
                                    f"Aucun symbole dans {src_label}.")
            return

        symbols_list = sorted(all_symbols)
        total = len(symbols_list)
        src_label = "Mes + Coko" if list_sources is not None else "tous symboles"

        now = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        cutoff = now + timedelta(hours=48)
        FRESH_DAYS = 10   # une ligne Parquet plus récente que ça est considérée fiable
        results = []

        def _within_window(date_value):
            """Retourne datetime si date_value tombe dans [now, cutoff], sinon None."""
            try:
                if isinstance(date_value, date_type) and not isinstance(date_value, datetime):
                    evt = datetime(date_value.year, date_value.month, date_value.day)
                else:
                    evt = pd.to_datetime(date_value, errors="coerce")
                    if evt is None or pd.isna(evt):
                        return None
                    evt = evt.to_pydatetime()
                evt = evt.replace(tzinfo=None, hour=0, minute=0, second=0, microsecond=0)
                return evt if now <= evt <= cutoff else None
            except Exception:
                return None

        # ── 1. Source locale Parquet (0 requête yfinance) ────────────────
        covered = set()
        try:
            from market_store import get_latest_event_dates
            local = get_latest_event_dates(symbols_list)
        except Exception:
            local = None

        if local is not None and not local.empty:
            now_full = datetime.utcnow()
            for _, row in local.iterrows():
                sym = str(row["symbol"]).upper()
                fd = pd.to_datetime(row.get("feature_date"), errors="coerce")
                is_fresh = (fd is not None and not pd.isna(fd)
                            and (now_full - fd.to_pydatetime()).days <= FRESH_DAYS)
                if not is_fresh:
                    continue  # donnée trop ancienne → laisser au fallback yfinance
                covered.add(sym)
                ed = _within_window(row.get("next_earnings_date"))
                if ed:
                    results.append((sym, "Earnings", ed.strftime("%Y-%m-%d")))
                xd = _within_window(row.get("next_ex_dividend_date"))
                if xd:
                    results.append((sym, "Ex-Dividende", xd.strftime("%Y-%m-%d")))

        # ── 2. Fallback yfinance pour les symboles non couverts localement ──
        missing = [s for s in symbols_list if s not in covered]
        cancelled = False
        if missing:
            from PyQt5.QtWidgets import QProgressDialog
            progress = QProgressDialog(
                f"Yahoo Finance pour {len(missing)} symbole(s) absent(s) du store "
                f"({len(covered)} résolus localement)…",
                "Annuler", 0, len(missing), self
            )
            progress.setWindowTitle("Evenements 48h")
            progress.setWindowModality(Qt.WindowModal)
            progress.setMinimumDuration(0)
            progress.setValue(0)
            QApplication.processEvents()

            def _check_symbol(sym):
                hits = []
                try:
                    t = yf.Ticker(sym)
                    cal = t.calendar
                    if cal is None:
                        return hits
                    if hasattr(cal, 'to_dict'):
                        cal = cal.iloc[0].to_dict() if not cal.empty else {}
                    if not isinstance(cal, dict):
                        return hits
                    mapping = {
                        "Earnings Date":    "Earnings",
                        "Ex-Dividend Date": "Ex-Dividende",
                        "Dividend Date":    "Dividende",
                    }
                    for key, label in mapping.items():
                        evt_dt = None
                        raw = cal.get(key)
                        if raw is None:
                            continue
                        candidates = raw if isinstance(raw, (list, tuple)) else [raw]
                        for c in candidates:
                            evt_dt = _within_window(c)
                            if evt_dt:
                                hits.append((sym, label, evt_dt.strftime("%Y-%m-%d")))
                                break
                except Exception:
                    pass
                return hits

            done = [0]
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(_check_symbol, sym): sym for sym in missing}
                for future in as_completed(futures):
                    if progress.wasCanceled():
                        cancelled = True
                        break
                    try:
                        results.extend(future.result())
                    except Exception:
                        pass
                    done[0] += 1
                    progress.setValue(done[0])
                    QApplication.processEvents()
            progress.close()

        if cancelled:
            return

        if not results:
            QMessageBox.information(
                self, "Evenements 48h",
                f"Aucun evenement (earnings / dividende) trouve dans les 48h\n"
                f"pour les {total} symboles ({len(covered)} via le store local, "
                f"{len(missing)} via Yahoo Finance)."
            )
            return

        if hasattr(self, "_status"):
            self._status(
                f"Événements 48h : {len(covered)} résolus localement, "
                f"{len(missing)} via yfinance"
            )
        rows = [(sym, evt, dt) for sym, evt, dt in sorted(results, key=lambda x: (x[2], x[0]))]
        self._present_screener_results(
            f"Événements 48h — {len({r[0] for r in results})} symbole(s)",
            ["Symbole", "Événement", "Date"],
            rows,
        )
