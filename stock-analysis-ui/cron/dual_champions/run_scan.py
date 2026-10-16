#!/usr/bin/env python3
"""
run_scan - Veille Dual Champions, cron local, canal de notification separe du
cycle gaps (demande du 13.10.2026).

Deux horloges independantes, memorisees dans state.json, verifiees a chaque
invocation (cron toutes les 5 minutes - meme granularite et meme raison que
Trading_Agent/gaps/cron/run_scheduled.py : tolere les imprecisions de cron) :

    toutes les 30 min, 07:30-18:00 ET jours ouvres (premarket + seance +
    after-hours, elargi le 13.10.2026 de 2h avant/apres la seance 09:30-16:00) :
        tire les 50 prochains tickers d'une file melangee ("le tour"), les
        analyse via Combined_scan.analyze_safe, accumule les Dual Champions
        (et Dual Champions*) trouves.
    toutes les 2h, qu'il y ait eu un ou plusieurs tours depuis :
        notifie l'accumulateur, puis le vide.

La file est completee chaque fois qu'elle descend sous la taille d'un lot,
par trois sources combinees et dedupliquees (demande du 13.10.2026 : "pas
seulement Finviz, aussi popular stock par Yahoo ou random Yahoo finance") :
    - Finviz, preset dual_star (valorisation/croissance/securite pre-filtrees)
    - Yahoo most_actives (yf.screen, la notion Yahoo de "populaire")
    - Un echantillon aleatoire de sp500_symbols.txt + popular_symbols.txt
      (univers large non pre-filtre - le "random Yahoo Finance" demande)

"Aleatoire mais jamais deux fois avant le tour complet, et pas de redondance
inutile" (demandes du 13.10.2026) : le melange (random.shuffle) donne
l'aleatoire ; la garantie de non-repetition porte sur TOUTE la journee, pas
seulement sur la file courante - etat["deja_aujourdhui"] accumule chaque
ticker scanne depuis minuit ET, et toute nouvelle decouverte en est filtree
avant d'etre ajoutee a la file. Sans ca (bug corrige le 13.10.2026), Yahoo
most_actives renvoyant presque toujours les memes megacaps, une redecouverte
frequente reinjectait MRVL/META/ACN plusieurs fois le meme jour. Reinitialise
a minuit ET (changement de jour) : un ticker peut repasser le lendemain, pas
le meme jour.

Cout, la raison du plafond horaire : Combined_scan.analyze_safe coute environ
4 requetes yfinance par titre (meme cout que core.combined_finviz.run_finviz_
combined, deja documente dans ce depot). A 50 titres toutes les 30 min sur la
fenetre 07:30-18:00 ET (10h30), plafond theorique ~4200 requetes yfinance/jour
- en hausse par rapport aux ~2600/jour de la fenetre 09:30-16:00 initiale, et
toujours au-dessus de ce que le cycle gaps consomme seul. Le budget de
requetes yfinance est une contrainte dure du depot (CLAUDE.md racine) ; revoir
la frequence ou la taille du lot si ca declenche les limites de debit deja
mesurees ailleurs.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

CRON_DIR = Path(__file__).resolve().parent
SRC = CRON_DIR.parent.parent / "src"
STATE_FILE = CRON_DIR / "state.json"
# Le protocole permanent (Results/Screeners/DualChampions_Watch.csv) est
# gere par core.dual_champions_archive, partage avec le scan manuel a la
# demande (core.combined_finviz) - voir _archiver_detections plus bas.

sys.path.insert(0, str(CRON_DIR))
sys.path.insert(0, str(SRC))
os.chdir(SRC)  # core.* et Combined_scan supposent CWD = src/ (chemins relatifs)

from notify import notify  # noqa: E402

NY = ZoneInfo("America/New_York")
TAILLE_LOT = 50
TAILLE_ALEATOIRE = 200  # tire de sp500_symbols.txt + popular_symbols.txt a chaque tour
TAILLE_YAHOO_POPULAIRE = 100  # yf.screen("most_actives"), plafond du nombre demande
INTERVALLE_SCAN = timedelta(minutes=30)
INTERVALLE_NOTIFY = timedelta(hours=2)
TOLERANCE = timedelta(minutes=4)

_POOL_ALEATOIRE: list[str] | None = None


def _ny_now() -> datetime:
    return datetime.now(NY)


def _marche_ouvert(now: datetime) -> bool:
    """Fenetre de scan : 2h avant l'ouverture (premarket) a 2h apres la cloture
    (after-hours), soit 07:30-18:00 ET - elargie le 13.10.2026 sur demande ("2
    Stunden vor und nach der Markteroffnung"). "Apres l'ouverture" n'aurait
    rien elargi (9h30+2h=11h30 est deja en seance) ; lu comme premarket +
    after-hours, l'extension usuelle autour d'une seance US."""
    if now.weekday() >= 5:
        return False
    ouverture = now.replace(hour=7, minute=30, second=0, microsecond=0)
    cloture = now.replace(hour=18, minute=0, second=0, microsecond=0)
    return ouverture <= now <= cloture


def _etat_defaut() -> dict:
    return {"file": [], "deja_aujourdhui": [], "jour": None,
            "accumulateur": [], "dernier_scan": None, "dernier_notify": None}


def _charger_etat() -> dict:
    if STATE_FILE.exists():
        try:
            etat = json.load(open(STATE_FILE))
            return {**_etat_defaut(), **etat}
        except Exception:
            pass
    return _etat_defaut()


def _sauver_etat(etat: dict) -> None:
    STATE_FILE.write_text(json.dumps(etat, indent=2, ensure_ascii=False))


def _archiver_detections(horodatage: str, trouves: list[list]) -> list[list]:
    """Delegue a core.dual_champions_archive, protocole permanent PARTAGE
    avec le scan manuel a la demande (core.combined_finviz.run_finviz_combined)
    - demande du 13.10.2026, apres qu'OXY/XOM aient ete re-presentees comme
    nouvelles par la veille cron alors qu'elles avaient deja ete montrees le
    meme jour via un scan manuel. Rend les lignes reellement ecrites (le
    reste etait redondant), pour que l'appelant puisse signaler ce qui est
    vraiment nouveau."""
    try:
        from core.combined_finviz import HEADERS
        from core.dual_champions_archive import archiver
        return archiver(horodatage, "Dual Champions Watch (cron 30min)", trouves, HEADERS)
    except Exception as e:
        print(f"[DUALCHAMP] archivage echoue: {e}", file=sys.stderr)
        return []


def _charger_pool_aleatoire() -> list[str]:
    """sp500_symbols.txt + popular_symbols.txt : univers large, non pre-filtre
    par aucun critere fondamental - le bassin pour le tirage "random"."""
    global _POOL_ALEATOIRE
    if _POOL_ALEATOIRE is None:
        pool: set[str] = set()
        for nom in ("sp500_symbols.txt", "popular_symbols.txt"):
            chemin = SRC / nom
            if chemin.exists():
                pool.update(l.strip() for l in chemin.read_text().splitlines() if l.strip())
        _POOL_ALEATOIRE = sorted(pool)
    return _POOL_ALEATOIRE


def _decouvrir_univers() -> list[str]:
    """Trois sources combinees et dedupliquees : Finviz (pre-filtre fondamental),
    Yahoo most_actives (notion Yahoo de "populaire"), et un echantillon aleatoire
    d'un univers large (sp500 + liste "popular" du depot). Chaque source echoue
    silencieusement pour les autres - une panne Finviz ne doit pas vider le tour
    si Yahoo repond, et inversement."""
    tickers: set[str] = set()

    try:
        from core.finviz_screeners import run_preset
        fv = run_preset("dual_star", limit=500)
        tickers.update(r[0] for r in (fv.get("rows") or []) if r and r[0])
    except Exception as e:
        print(f"[DUALCHAMP] decouverte Finviz echouee: {e}", file=sys.stderr)

    try:
        import yfinance as yf
        res = yf.screen("most_actives", count=TAILLE_YAHOO_POPULAIRE)
        tickers.update(q["symbol"] for q in res.get("quotes", []) if q.get("symbol"))
    except Exception as e:
        print(f"[DUALCHAMP] decouverte Yahoo most_actives echouee: {e}", file=sys.stderr)

    try:
        pool = _charger_pool_aleatoire()
        tickers.update(random.sample(pool, min(TAILLE_ALEATOIRE, len(pool))))
    except Exception as e:
        print(f"[DUALCHAMP] echantillon aleatoire echoue: {e}", file=sys.stderr)

    univers = list(tickers)
    random.shuffle(univers)
    return univers


def _scanner_lot(tickers: list[str]) -> list[list]:
    from core import scan_fondamentaux as sf
    import Combined_scan as cs
    from core.combined_finviz import lignes

    if len(sf.EUR_RATES) == 1:  # taux de change : un seul appel groupe
        sf.charger_taux_eur(verbose=False)
    resultats = [cs.analyze_safe(t) for t in tickers]
    return lignes(resultats)  # deja trie, etoile en tete, None filtres


def _formatter_resume(lignes_cumulees: list[list]) -> str:
    """Tri : etoiles avant Dual Champion simples (mise en evidence demandee),
    puis meilleur score total (Growth+Safe, Combined_scan.score_total) en
    tete dans chaque groupe - pas l'ordre chronologique des lots scannes, qui
    n'aurait aucun sens pour le lecteur. Score total non conserve dans la
    ligne accumulee (seuls G et S le sont) ; reconstruit ici par l[4]+l[5],
    identique au calcul de Combined_scan.py (total_score = growth+safe)."""
    if not lignes_cumulees:
        return "aucun Dual Champion detecte sur ce cycle de 2h"

    def _ligne(l: list) -> str:
        return (f"{l[0]} ({l[2]}) — G: {l[4]}/5 | S: {l[5]}/7 | "
                f"momentum: {l[7]}% | dividende: {l[8]}%")

    def _tri(groupe: list[list]) -> list[list]:
        return sorted(groupe, key=lambda l: (-(l[4] + l[5]), l[0]))

    etoiles = _tri([l for l in lignes_cumulees if l[3] == "💎 Dual Champion*"])
    simples = _tri([l for l in lignes_cumulees if l[3] == "💎 Dual Champion"])
    parties = []
    if etoiles:
        parties.append(f"⭐ DUAL CHAMPION* ({len(etoiles)})")
        parties += [_ligne(l) for l in etoiles]
    if simples:
        if parties:
            parties.append("")
        parties.append(f"Dual Champion ({len(simples)})")
        parties += [_ligne(l) for l in simples]
    return "\n".join(parties)


def main() -> int:
    now = _ny_now()
    etat = _charger_etat()

    aujourdhui = now.date().isoformat()
    if etat.get("jour") != aujourdhui:
        # Nouveau jour : on oublie ce qui a ete scanne hier, un ticker peut
        # repasser. Dans la meme journee, en revanche, plus jamais - voir
        # _decouvrir_univers() plus bas.
        etat["jour"] = aujourdhui
        etat["file"] = []
        etat["deja_aujourdhui"] = []

    if _marche_ouvert(now):
        dernier_scan = (datetime.fromisoformat(etat["dernier_scan"])
                        if etat.get("dernier_scan") else None)
        if dernier_scan is None or now - dernier_scan >= INTERVALLE_SCAN - TOLERANCE:
            if len(etat["file"]) < TAILLE_LOT:
                # Filtre contre la file ET contre tout ce qui a deja ete
                # scanne aujourd'hui (pas seulement la file courante) - sans
                # ca, une redecouverte frequente (Yahoo most_actives renvoie
                # presque toujours les memes megacaps) reinjecte un ticker
                # deja scanne plus tot dans la journee. Bug signale le
                # 13.10.2026 : MRVL/META/ACN rescannes plusieurs fois le
                # meme jour.
                deja = set(etat["file"]) | set(etat["deja_aujourdhui"])
                etat["file"] += [t for t in _decouvrir_univers() if t not in deja]
            lot, etat["file"] = etat["file"][:TAILLE_LOT], etat["file"][TAILLE_LOT:]
            if lot:
                etat["deja_aujourdhui"] += lot
                try:
                    resultats = _scanner_lot(lot)
                    trouves = [r for r in resultats if r[3].startswith("💎")]
                    # L'accumulateur ne recoit que ce que l'archiveur a
                    # reellement ecrit (deja filtre pour exclure ce qui est
                    # connu depuis aujourd'hui, par CE scan ou par l'outil
                    # manuel) - PAS "trouves" brut. Bug corrige le 13.10.2026 :
                    # OVV/PAGS/SAVE.ST notifies deux fois le meme jour parce
                    # que l'accumulateur ignorait ce filtrage et reprenait
                    # tout ce qui etait "💎" dans le lot, deja connu ou pas.
                    etat["accumulateur"] += _archiver_detections(
                        now.isoformat(timespec="seconds"), trouves)
                except Exception as e:
                    notify("Dual Champions - erreur de scan", f"{type(e).__name__}: {e}",
                          priority="high")
            etat["dernier_scan"] = now.isoformat()

    dernier_notify = (datetime.fromisoformat(etat["dernier_notify"])
                      if etat.get("dernier_notify") else None)
    if dernier_notify is None or now - dernier_notify >= INTERVALLE_NOTIFY - TOLERANCE:
        if dernier_notify is not None:  # pas de notification vide au tout premier demarrage
            notify("Dual Champions - cycle de 2h", _formatter_resume(etat["accumulateur"]))
        etat["accumulateur"] = []
        etat["dernier_notify"] = now.isoformat()

    _sauver_etat(etat)
    return 0


if __name__ == "__main__":
    sys.exit(main())
