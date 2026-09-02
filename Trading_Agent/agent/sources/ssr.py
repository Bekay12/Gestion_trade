"""
ssr - Restriction Rule 201 depuis le fichier quotidien public de NASDAQ Trader.

Resout la moitie du verrou de R9. La documentation supposait cette donnee
payante ; elle est en realite publiee librement, chaque jour de bourse, sous
forme de fichier texte.

Semantique du fichier, qui n'est pas evidente : le fichier date du jour D
contient les declenchements de D-1 ET de D. C'est exactement la fenetre pendant
laquelle la restriction court — declenchee un jour, elle reste active le
lendemain entier. La presence d'un symbole dans le fichier du jour D signifie
donc « sous restriction le jour D ».

Une donnee indisponible remonte None, jamais False : le moteur doit pouvoir
distinguer « pas sous restriction » de « je ne sais pas ».
"""

from __future__ import annotations

import csv
import io
import logging
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

FILE_URL = "https://www.nasdaqtrader.com/dynamic/symdir/shorthalts/shorthalts{ymd}.txt"
USER_AGENT = "Mozilla/5.0 (compatible; paper-trading-journal/1.0)"
# Nombre de jours ouvres a remonter quand le fichier du jour n'existe pas encore.
MAX_LOOKBACK = 4


class SsrUnavailable(RuntimeError):
    """Le fichier n'a pu etre obtenu pour aucune date de la fenetre."""


class SsrCalendar:
    """Acces au calendrier des restrictions, avec cache disque."""

    def __init__(self, cache_dir: Path | None = None, timeout: int = 20) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer le client.

        Inputs:
            cache_dir (Path | None): repertoire de cache
            timeout (int): delai reseau en secondes

        Outputs:
            None
        ----------------------------------------------------------------------
        """
        self.cache_dir = cache_dir or (
            Path(__file__).resolve().parents[2] / "docu" / "_cache" / "ssr"
        )
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _fetch(self, day: date) -> str | None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Recuperer le fichier brut d'une journee, cache d'abord.

        Inputs:
            day (date): journee de bourse

        Outputs:
            text (str | None): contenu, None si la journee n'a pas de fichier
        ----------------------------------------------------------------------
        """
        ymd = day.strftime("%Y%m%d")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / f"{ymd}.txt"

        # Un fichier de journee passee ne change plus : cache definitif.
        if cached.exists() and (day < date.today() or _is_fresh(cached)):
            return cached.read_text(encoding="utf-8", errors="replace")

        try:
            response = self.session.get(FILE_URL.format(ymd=ymd), timeout=self.timeout)
        except requests.RequestException as error:
            logger.warning("[SSR] %s injoignable : %s", ymd, error)
            return cached.read_text(encoding="utf-8", errors="replace") if cached.exists() else None

        if response.status_code != 200 or not response.text.strip():
            return None

        text = response.text
        cached.write_text(text, encoding="utf-8")
        return text

    def for_day(self, day: date) -> set[str] | None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Symboles sous restriction Rule 201 pour une journee donnee.

        Inputs:
            day (date): journee de bourse

        Outputs:
            symbols (set[str] | None): symboles, None si le fichier est absent
        ----------------------------------------------------------------------
        """
        text = self._fetch(day)
        if text is None:
            return None
        return parse_file(text)

    def active_symbols(self, day: date | None = None) -> tuple[set[str], date]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Symboles sous restriction, en remontant si le fichier du jour n'est
            pas encore publie. Le repli conserve la validite : un declenchement
            de la veille court encore aujourd'hui.

        Inputs:
            day (date | None): journee visee, defaut aujourd'hui

        Outputs:
            symbols (set[str]): symboles sous restriction
            source_day (date): journee du fichier effectivement utilise

        Raises:
            SsrUnavailable: si aucun fichier n'est trouve dans la fenetre
        ----------------------------------------------------------------------
        """
        day = day or date.today()
        for back in range(MAX_LOOKBACK + 1):
            probe = day - timedelta(days=back)
            symbols = self.for_day(probe)
            if symbols is not None:
                if back:
                    logger.info("[SSR] repli sur le fichier du %s", probe)
                return symbols, probe
        raise SsrUnavailable(
            f"aucun fichier NASDAQ trouve entre {day - timedelta(days=MAX_LOOKBACK)} et {day}"
        )

    def is_active(self, symbol: str, day: date | None = None) -> bool | None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Repondre pour un titre. None signifie « indeterminable », ce que le
            moteur traite comme un refus et non comme une absence de
            restriction.

        Inputs:
            symbol (str): ticker
            day (date | None): journee visee

        Outputs:
            active (bool | None): True, False, ou None si donnee indisponible
        ----------------------------------------------------------------------
        """
        try:
            symbols, _ = self.active_symbols(day)
        except SsrUnavailable:
            return None
        return symbol.strip().upper() in symbols


def _is_fresh(path: Path, minutes: int = 30) -> bool:
    """Le fichier du jour courant s'enrichit en seance : cache court."""
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(minutes=minutes)


def parse_file(text: str) -> set[str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Extraire les symboles du fichier NASDAQ. Le nom de societe peut
        contenir des virgules et etre entre guillemets, d'ou le passage par un
        lecteur CSV. La derniere ligne est un horodatage de generation, sans
        virgule : elle est ecartee par le controle du nombre de colonnes.

    Inputs:
        text (str): contenu brut du fichier

    Outputs:
        symbols (set[str]): symboles en majuscules
    --------------------------------------------------------------------------
    """
    symbols: set[str] = set()
    reader = csv.reader(io.StringIO(text))
    for row in reader:
        if len(row) < 4:
            continue                      # ligne d'horodatage finale
        symbol = row[0].strip().upper()
        if not symbol or symbol == "SYMBOL":
            continue                      # en-tete
        symbols.add(symbol)
    return symbols
