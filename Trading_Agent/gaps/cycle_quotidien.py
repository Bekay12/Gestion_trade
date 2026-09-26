#!/usr/bin/env python3
"""
cycle_quotidien - orchestre les phases du cycle de gaps, sans heure codee en dur.

POURQUOI IL DECIDE LUI-MEME DE LA PHASE

Le cycle est defini en heure de New York, la machine vit a Paris, et les deux
zones ne changent pas d'heure le meme week-end. Mesure du 26.09.2026:

    07h00 ET le 26.09.2026 = 13h00 a Paris
    07h00 ET le 05.11.2026 = 13h00 a Paris
    07h00 ET le 20.03.2027 = 12h00 a Paris

Une entree cron en heure locale deriverait donc d'une heure deux fois par an, et
la detection tomberait hors de sa fenetre sans que rien ne le signale. Ce script
est appele souvent (toutes les quinze minutes), lit la fenetre de session reelle
et decide s'il agit. La crontab ne porte aucune heure metier.

POURQUOI CRON ET NON UN TIMER SYSTEMD

`loginctl show-user` rend `Linger=no` sur ce poste: les timers utilisateur
s'arretent a la deconnexion, donc une detection de 13h00 serait manquee chaque
fois que la session est fermee. Le service cron, lui, est actif en permanence.
Verifie le 26.09.2026.

PHASES

    premarket   07h00-09h30 ET   detection des gaps        -> detections/
    cloture     16h05-17h00 ET   cassures puis notation    -> detections/
    (autre)                      rien

Chaque phase est idempotente: un fichier temoin par phase et par jour empeche un
second passage quand cron rappelle un quart d'heure plus tard.

CE QUE CE SCRIPT NE FAIT PAS, ET NE FERA PAS

Il ne verifie aucun catalyseur. Le non negociable de la methode demande une
lecture humaine, et deux journees de mesure l'ont confirme: le 24.09.2026 la
liste batie sur les seuls signaux structurels valait -7,46 % en moyenne. Le
script produit la liste des depots a lire, avec leurs URL, et s'arrete la.

Aufruf:
    python3 gaps/cycle_quotidien.py                # decide et agit
    python3 gaps/cycle_quotidien.py --phase premarket --forcer
    python3 gaps/cycle_quotidien.py --etat          # que ferait-il maintenant
    python3 gaps/cycle_quotidien.py --installer-cron
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from datetime import datetime
from zoneinfo import ZoneInfo

RACINE = os.path.dirname(os.path.abspath(__file__))
DETECTIONS = os.path.join(RACINE, "detections")
TEMOINS = os.path.join(RACINE, ".temoins")
JOURNAL = os.path.join(RACINE, "cycle.log")

# Mis a vrai par --simuler: les commandes sont affichees, jamais executees.
SIMULATION = False

VENV = "/home/berkam/Projets/Gestion_trade/.venv_new/bin/python"
SKILL = os.path.expanduser("~/.claude/skills/gap-trading-germain/scripts")

NY = ZoneInfo("America/New_York")

# Fenetres en minutes depuis minuit, heure de New York.
FENETRES = {
    "premarket": (7 * 60, 9 * 60 + 30),
    # 16h05 et non 16h00: les barres de cloture mettent quelques minutes a se
    # stabiliser chez yfinance, et une notation sur des cours non definitifs
    # produirait des notes fausses sans rien signaler.
    "cloture":   (16 * 60 + 5, 17 * 60),
}

LIMITE = 30            # voir SKILL.md: jamais moins de 30 sans raison explicite
MIN_GAP = 5


def maintenant_ny() -> datetime:
    return datetime.now(NY)


def phase_courante(t: datetime = None) -> str | None:
    """Phase que la fenetre de session autorise, ou None."""
    t = t or maintenant_ny()
    if t.weekday() >= 5:          # samedi, dimanche
        return None
    m = t.hour * 60 + t.minute
    for nom, (debut, fin) in FENETRES.items():
        if debut <= m < fin:
            return nom
    return None


def _temoin(phase: str, jour: str) -> str:
    return os.path.join(TEMOINS, f"{jour}_{phase}.done")


def deja_fait(phase: str, jour: str) -> bool:
    return os.path.exists(_temoin(phase, jour))


def marquer(phase: str, jour: str, resume: str) -> None:
    if SIMULATION:
        return
    os.makedirs(TEMOINS, exist_ok=True)
    with open(_temoin(phase, jour), "w", encoding="utf-8") as f:
        f.write(f"{datetime.now(NY).isoformat()}\n{resume}\n")


def journaliser(ligne: str) -> None:
    """Journal append-only. Un cycle non surveille sans trace est un cycle muet."""
    if SIMULATION:
        print(f"  [journal] {ligne}")
        return
    with open(JOURNAL, "a", encoding="utf-8") as f:
        f.write(f"{datetime.now(NY).isoformat()}  {ligne}\n")


def lancer(argv: list, etiquette: str) -> tuple[int, str]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Execute une phase et rend son code de retour avec sa sortie.

        Aucune exception n'est avalee: un echec est journalise avec son code et
        la fin de sa sortie, et la phase n'est PAS marquee comme faite, de sorte
        que le passage suivant reessaie.

    Inputs:
        argv (list): commande complete
        etiquette (str): nom lisible de la phase, pour le journal

    Outputs:
        (code, sortie) (tuple): code de retour, sortie fusionnee
    --------------------------------------------------------------------------
    """
    if SIMULATION:
        print(f"  [simulation] {etiquette}")
        print(f"               {' '.join(argv)}")
        return 0, ""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        journaliser(f"{etiquette} : DELAI DEPASSE apres 1800 s")
        return 124, "delai depasse"
    except Exception as e:
        journaliser(f"{etiquette} : ECHEC {type(e).__name__}: {e}")
        return 1, str(e)
    sortie = (p.stdout or "") + (p.stderr or "")
    if p.returncode != 0:
        journaliser(f"{etiquette} : code {p.returncode}\n"
                    f"    {sortie.strip()[-400:]}")
    return p.returncode, sortie


def phase_premarket(jour: str) -> int:
    cible = os.path.join(DETECTIONS, f"{jour}_premarket.json")
    code, sortie = lancer([VENV, os.path.join(SKILL, "gap_scan.py"),
                           "--mode", "premarket", "--min-gap", str(MIN_GAP),
                           "--limit", str(LIMITE), "--json", cible],
                          "detection premarket")
    if code == 0:
        a_lire = sortie.count("https://www.sec.gov")
        journaliser(f"detection premarket : OK -> {os.path.basename(cible)}"
                    f" ({a_lire} depot(s) a lire)")
        marquer("premarket", jour, f"-> {cible}")
    return code


def phase_cloture(jour: str) -> int:
    """Cassures puis notation. La notation tourne meme si le scan de cassures
    echoue: elle porte sur la detection du matin, qui ne depend pas de lui."""
    cible = os.path.join(DETECTIONS, f"{jour}_cassure.json")
    code_c, _ = lancer([VENV, os.path.join(SKILL, "cassure_scan.py"),
                        "--min-hausse", str(MIN_GAP), "--limit", str(LIMITE),
                        "--json", cible], "scan cassures")
    journaliser(f"scan cassures : {'OK' if code_c == 0 else 'echec'}")

    detection = os.path.join(DETECTIONS, f"{jour}_premarket.json")
    if not os.path.exists(detection):
        journaliser("notation : aucune detection du matin a noter")
        marquer("cloture", jour, "cassures seules, pas de detection a noter")
        return code_c
    code_n, sortie = lancer([VENV, os.path.join(RACINE, "evaluer_gaps.py"),
                             "--fichier", detection, "--jours", "1"],
                            "notation J1")
    taux = [l for l in sortie.splitlines() if "justesse" in l.lower()]
    journaliser(f"notation J1 : {'OK' if code_n == 0 else 'echec'}"
                + (f" — {taux[0].strip()}" if taux else ""))
    if code_c == 0 and code_n == 0:
        marquer("cloture", jour, "cassures + notation")
    return max(code_c, code_n)


CRONTAB = ("*/15 * * * * "
           f"{VENV} {os.path.join(RACINE, 'cycle_quotidien.py')} "
           f">> {os.path.join(RACINE, 'cron.log')} 2>&1")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Cycle quotidien des gaps")
    p.add_argument("--phase", choices=sorted(FENETRES), default=None,
                   help="forcer une phase precise")
    p.add_argument("--forcer", action="store_true",
                   help="ignorer la fenetre de session et le temoin du jour")
    p.add_argument("--etat", action="store_true",
                   help="dire ce qui serait fait, sans rien faire")
    p.add_argument("--simuler", action="store_true",
                   help="afficher les commandes et le journal, sans rien executer")
    p.add_argument("--installer-cron", action="store_true")
    a = p.parse_args(argv)
    global SIMULATION
    SIMULATION = a.simuler

    if a.installer_cron:
        print("Ligne a ajouter a la crontab (aucune heure metier: le script decide) :")
        print(f"\n  {CRONTAB}\n")
        print("Installation :")
        print(f"  (crontab -l 2>/dev/null; echo '{CRONTAB}') | crontab -")
        print("\nElle n'est PAS installee automatiquement: une tache planifiee qui")
        print("appelle le reseau et ecrit dans le depot releve de ta decision.")
        return 0

    t = maintenant_ny()
    jour = t.date().isoformat()
    phase = a.phase or phase_courante(t)

    if a.etat:
        print(f"New York   : {t:%Y-%m-%d %H:%M %Z}  ({t:%A})")
        print(f"Phase      : {phase or 'aucune (hors fenetre ou week-end)'}")
        for nom, (d, f) in sorted(FENETRES.items()):
            etat = "faite" if deja_fait(nom, jour) else "en attente"
            print(f"  {nom:10} {d//60:02d}h{d%60:02d}-{f//60:02d}h{f%60:02d} ET"
                  f"   {etat}")
        return 0

    if phase is None:
        return 0                  # hors fenetre: sortie silencieuse, cron appelle souvent
    if deja_fait(phase, jour) and not a.forcer:
        return 0

    journaliser(f"phase {phase} ({t:%H:%M %Z})")
    code = phase_premarket(jour) if phase == "premarket" else phase_cloture(jour)
    if code == 0 and phase == "premarket":
        journaliser("RAPPEL : aucun catalyseur verifie. Lire les depots signales "
                    "avant toute liste courte (mesure du 24.09.2026 : -7,46 %).")
    return code


if __name__ == "__main__":
    sys.exit(main())
