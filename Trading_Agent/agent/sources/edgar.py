"""
edgar - Client SEC EDGAR : historique des depots et detection de dilution.

Couvre la fonction F4 de docu/methode/03-outils.md, la plus citee du corpus
(101 mentions) et la seule dont la gratuite soit certaine. Elle porte la
distinction qui fait la methode : un S-3 actif signale une AUTORISATION
d'emettre, un 424B signale que l'emission a EU LIEU. Un titre qui s'envole
alors qu'un S-3 dort est un candidat a la dilution, pas une histoire de
croissance.

La SEC impose un en-tete User-Agent nominatif et limite le debit a dix requetes
par seconde. L'en-tete est lu dans la variable d'environnement SEC_USER_AGENT :
il n'est jamais devine, parce qu'il engage l'identite de l'appelant.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
FILING_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"

# La SEC plafonne a dix requetes par seconde ; on reste tres en dessous.
MIN_INTERVAL = 0.15
CACHE_TTL_HOURS = 12

# Un S-3 ordinaire porte une capacite d'emission pendant trois ans.
SHELF_VALIDITY_DAYS = 365 * 3
# Au-dela, un 424B ne dit plus rien de la seance en cours.
OFFERING_RECENT_DAYS = 30

DILUTIVE_FORMS = ("S-3", "S-3ASR", "S-1")
OFFERING_FORMS = ("424B1", "424B2", "424B3", "424B4", "424B5", "424B7", "424B8")
EVENT_FORMS = ("8-K",)

# F8 : les agregateurs du marche (OpenInsider, WhaleWisdom) ne sont que des
# vitrines de ces formulaires. La source est ici, publique et gratuite.
INSIDER_FORMS = ("4", "4/A")                       # operations des dirigeants
STAKE_FORMS = ("SC 13D", "SC 13D/A", "SC 13G", "SC 13G/A")   # franchissements de 5 %
INSTITUTIONAL_FORMS = ("13F-HR", "13F-HR/A")       # positions des gestionnaires
# Au-dela, une operation de dirigeant ne dit plus rien de la seance.
INSIDER_RECENT_DAYS = 90
STAKE_RECENT_DAYS = 30


class EdgarError(RuntimeError):
    """Erreur d'acces a EDGAR, distinguee d'une absence de resultat."""


@dataclass(frozen=True)
class Filing:
    """Un depot reglementaire."""

    form: str
    filed: date
    accession: str
    document: str
    cik: str
    description: str = ""

    @property
    def url(self) -> str:
        """Lien direct vers le document principal du depot."""
        return FILING_URL.format(
            cik=self.cik.lstrip("0"),
            accession=self.accession.replace("-", ""),
            document=self.document,
        )

    def age_days(self, today: date | None = None) -> int:
        return ((today or date.today()) - self.filed).days


@dataclass
class DilutionReport:
    """
    Lecture des depots sous l'angle de la dilution. `risk` resume, mais ce sont
    les champs qui portent l'information exploitable.
    """

    symbol: str
    shelf_active: bool
    latest_shelf: Filing | None
    recent_offerings: list[Filing]
    recent_events: list[Filing]

    @property
    def risk(self) -> str:
        """Trois niveaux : une emission constatee prime sur une capacite dormante."""
        if self.recent_offerings:
            return "eleve"
        if self.shelf_active:
            return "modere"
        return "faible"

    def summary(self) -> str:
        if self.recent_offerings:
            latest = self.recent_offerings[0]
            return (f"{len(self.recent_offerings)} emission(s) recente(s), "
                    f"derniere {latest.form} du {latest.filed}")
        if self.shelf_active and self.latest_shelf:
            return f"capacite d'emission dormante ({self.latest_shelf.form} du {self.latest_shelf.filed})"
        return "aucun signal de dilution dans la fenetre examinee"


@dataclass
class OwnershipReport:
    """
    Lecture des depots sous l'angle de la detention : dirigeants, blocs de plus
    de 5 %, gestionnaires. Couvre F8 sans passer par un agregateur tiers.
    """

    symbol: str
    insider_filings: list[Filing]
    stake_filings: list[Filing]
    institutional_filings: list[Filing]

    @property
    def insider_active(self) -> bool:
        """Des dirigeants ont declare des operations dans la fenetre recente."""
        return bool(self.insider_filings)

    def summary(self) -> str:
        parts: list[str] = []
        if self.stake_filings:
            latest = self.stake_filings[0]
            parts.append(f"franchissement de seuil ({latest.form} du {latest.filed})")
        if self.insider_filings:
            parts.append(f"{len(self.insider_filings)} declaration(s) de dirigeant")
        if self.institutional_filings:
            parts.append(f"{len(self.institutional_filings)} depot(s) 13F")
        return " ; ".join(parts) or "aucun mouvement de detention declare recemment"


def assess_ownership(
    symbol: str, filings: list[Filing], today: date | None = None
) -> OwnershipReport:
    """
    --------------------------------------------------------------------------
    Purpose:
        Extraire les signaux de detention. Un franchissement de 5 % est le plus
        parlant des trois : il signale qu'un intervenant a pris une position
        significative, information que le cours n'a pas forcement integree.

    Inputs:
        symbol (str): ticker, pour tracabilite
        filings (list[Filing]): depots, ordre indifferent
        today (date | None): date de reference

    Outputs:
        report (OwnershipReport): depots classes, du plus recent au plus ancien
    --------------------------------------------------------------------------
    """
    today = today or date.today()

    def recent(forms: tuple[str, ...], window: int) -> list[Filing]:
        return sorted(
            (f for f in filings if f.form in forms and f.age_days(today) <= window),
            key=lambda f: f.filed, reverse=True,
        )

    return OwnershipReport(
        symbol=symbol.upper(),
        insider_filings=recent(INSIDER_FORMS, INSIDER_RECENT_DAYS),
        stake_filings=recent(STAKE_FORMS, STAKE_RECENT_DAYS),
        institutional_filings=recent(INSTITUTIONAL_FORMS, INSIDER_RECENT_DAYS),
    )


class EdgarClient:
    """Client HTTP avec cache disque et limitation de debit."""

    def __init__(self, cache_dir: Path | None = None, user_agent: str | None = None) -> None:
        """
        ----------------------------------------------------------------------
        Purpose:
            Preparer un client. L'en-tete nominatif exige par la SEC vient de
            l'environnement, jamais d'une valeur devinee.

        Inputs:
            cache_dir (Path | None): repertoire de cache
            user_agent (str | None): en-tete, a defaut SEC_USER_AGENT

        Outputs:
            None

        Raises:
            EdgarError: si aucun User-Agent n'est disponible
        ----------------------------------------------------------------------
        """
        agent = user_agent or os.environ.get("SEC_USER_AGENT", "").strip()
        if not agent:
            raise EdgarError(
                "SEC_USER_AGENT absent. La SEC exige un en-tete nominatif : "
                "definir SEC_USER_AGENT=\"Nom Prenom contact@domaine\" avant usage."
            )
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": agent, "Accept-Encoding": "gzip, deflate"})
        self.cache_dir = cache_dir or (Path(__file__).resolve().parents[2] / "docu" / "_cache" / "edgar")
        self._last_call = 0.0

    def _get(self, url: str, cache_key: str, ttl_hours: int = CACHE_TTL_HOURS) -> dict:
        """
        ----------------------------------------------------------------------
        Purpose:
            Recuperer un document JSON, en servant le cache s'il est frais.

        Inputs:
            url (str): adresse
            cache_key (str): nom du fichier de cache
            ttl_hours (int): duree de validite du cache

        Outputs:
            payload (dict): document decode

        Raises:
            EdgarError: sur echec reseau ou statut non 200
        ----------------------------------------------------------------------
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached = self.cache_dir / f"{cache_key}.json"
        if cached.exists():
            age = datetime.now() - datetime.fromtimestamp(cached.stat().st_mtime)
            if age < timedelta(hours=ttl_hours):
                return json.loads(cached.read_text(encoding="utf-8"))

        elapsed = time.monotonic() - self._last_call
        if elapsed < MIN_INTERVAL:
            time.sleep(MIN_INTERVAL - elapsed)

        try:
            response = self.session.get(url, timeout=20)
        except requests.RequestException as error:
            raise EdgarError(f"EDGAR injoignable : {error}") from error
        finally:
            self._last_call = time.monotonic()

        if response.status_code != 200:
            raise EdgarError(f"EDGAR a repondu {response.status_code} pour {url}")

        payload = response.json()
        cached.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def resolve_cik(self, ticker: str) -> str:
        """
        ----------------------------------------------------------------------
        Purpose:
            Traduire un ticker en identifiant CIK sur dix chiffres.

        Inputs:
            ticker (str): symbole boursier

        Outputs:
            cik (str): identifiant zero-padde

        Raises:
            EdgarError: si le ticker est introuvable
        ----------------------------------------------------------------------
        """
        payload = self._get(TICKERS_URL, "company_tickers", ttl_hours=24 * 7)
        wanted = ticker.strip().upper()
        for entry in payload.values():
            if entry.get("ticker", "").upper() == wanted:
                return str(entry["cik_str"]).zfill(10)
        raise EdgarError(f"ticker {wanted} introuvable dans le referentiel SEC")

    def filings(self, ticker: str, limit: int = 200) -> list[Filing]:
        """
        ----------------------------------------------------------------------
        Purpose:
            Lire l'historique recent des depots d'un emetteur.

        Inputs:
            ticker (str): symbole boursier
            limit (int): nombre maximal de depots retournes

        Outputs:
            filings (list[Filing]): du plus recent au plus ancien
        ----------------------------------------------------------------------
        """
        cik = self.resolve_cik(ticker)
        payload = self._get(SUBMISSIONS_URL.format(cik=cik), f"sub_{cik}", ttl_hours=6)
        recent = (payload.get("filings") or {}).get("recent") or {}

        forms = recent.get("form", [])
        dates = recent.get("filingDate", [])
        accessions = recent.get("accessionNumber", [])
        documents = recent.get("primaryDocument", [])
        descriptions = recent.get("primaryDocDescription", [])

        out: list[Filing] = []
        for i in range(min(len(forms), len(dates), len(accessions), limit)):
            try:
                filed = date.fromisoformat(dates[i])
            except (ValueError, TypeError):
                continue
            out.append(Filing(
                form=forms[i],
                filed=filed,
                accession=accessions[i],
                document=documents[i] if i < len(documents) else "",
                cik=cik,
                description=descriptions[i] if i < len(descriptions) else "",
            ))
        logger.info("[EDGAR] %s : %d depots lus", ticker.upper(), len(out))
        return out


def assess_dilution(
    symbol: str, filings: list[Filing], today: date | None = None
) -> DilutionReport:
    """
    --------------------------------------------------------------------------
    Purpose:
        Lire une liste de depots sous l'angle de la dilution, en separant la
        capacite d'emettre de l'emission constatee. C'est la distinction que
        le corpus place au centre de la selection.

    Inputs:
        symbol (str): ticker, pour tracabilite
        filings (list[Filing]): depots, ordre indifferent
        today (date | None): date de reference, defaut aujourd'hui

    Outputs:
        report (DilutionReport): signaux structures
    --------------------------------------------------------------------------
    """
    today = today or date.today()

    shelves = sorted(
        (f for f in filings if f.form in DILUTIVE_FORMS),
        key=lambda f: f.filed, reverse=True,
    )
    latest_shelf = shelves[0] if shelves else None
    shelf_active = bool(
        latest_shelf and latest_shelf.age_days(today) <= SHELF_VALIDITY_DAYS
    )

    offerings = sorted(
        (f for f in filings
         if f.form in OFFERING_FORMS and f.age_days(today) <= OFFERING_RECENT_DAYS),
        key=lambda f: f.filed, reverse=True,
    )
    events = sorted(
        (f for f in filings
         if f.form in EVENT_FORMS and f.age_days(today) <= 7),
        key=lambda f: f.filed, reverse=True,
    )

    return DilutionReport(
        symbol=symbol.upper(),
        shelf_active=shelf_active,
        latest_shelf=latest_shelf,
        recent_offerings=offerings,
        recent_events=events,
    )
