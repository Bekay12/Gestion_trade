"""
dual_champions_archive - protocole permanent partage entre le scan manuel a
la demande (core.combined_finviz.run_finviz_combined) et la veille cron
(cron/dual_champions/run_scan.py), pour que les deux outils aient la meme
memoire.

Pourquoi ce module existe : demande du 13.10.2026, apres qu'OXY (et XOM) ont
ete presentes comme nouvelles trouvailles par la veille cron alors qu'elles
avaient deja ete montrees le meme jour via un scan manuel - les deux outils
utilisaient la meme logique de notation mais n'avaient aucune memoire
partagee. Ce module est maintenant le seul endroit qui lit et ecrit
Results/Screeners/DualChampions_Watch.csv ; les deux appelants l'importent au
lieu de dupliquer la logique chacun de son cote.

UN seul fichier permanent, jamais date, jamais recree (demande explicite :
"je ne veux pas un nouveau fichier par jour"). Dedup sur deux criteres,
quel que soit l'outil qui a produit la detection :
    - au plus une ligne par (ticker, date)
    - aucune nouvelle ligne si (profil, G, S) du ticker est identique a sa
      DERNIERE ligne connue, meme si cette derniere ligne date d'un jour
      different.

La signature ne retient QUE (profil, G, S) depuis le 14.10.2026 (demande
explicite : "tant que les scores G et S sont identiques ils ne doivent plus
etre affiches"). Avant cette date elle incluait aussi G3, momentum et
dividende ; momentum et dividende se recalculent chaque jour a partir du
cours et ne sont quasiment jamais identiques deux jours de suite, donc un
ticker dont le score reel n'avait pas change (ex. OXY G:4/5 S:5/7 inchange
du 13 au 14.10) reapparaissait chaque jour comme "nouveau". G3 est volontai-
rement exclu aussi : il est deja reflete dans le profil (l'etoile depend de
G3/G4/S4, cf. scan_fondamentaux.est_etoile), donc un changement de profil a
G/S constants reste detecte via le profil, pas besoin de G3 en plus.
"""
from __future__ import annotations

import csv
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parents[1] / "Results" / "Screeners"
FICHIER = RESULTS_DIR / "DualChampions_Watch.csv"


def charger_derniers(fichier: Path | None = None) -> dict[str, tuple[str, tuple]]:
    """Dernier etat connu par ticker : {ticker: (date AAAA-MM-JJ, signature)}.
    "ligne" = [run_at, screener, ticker, nom, pays, profil, G, S, G3, mom, div] ;
    signature = (profil, G, S) uniquement, soit (row[5], row[6], row[7]) -
    voir le docstring du module pour pourquoi G3/momentum/dividende en sont
    exclus depuis le 14.10.2026.

    fichier=None resout FICHIER a l'APPEL, pas a la definition de la fonction
    - un defaut `fichier: Path = FICHIER` figerait la valeur une fois pour
      toutes au chargement du module ; un monkeypatch de dual_champions_
      archive.FICHIER dans un test n'aurait alors plus aucun effet (piege
      rencontre le 13.10.2026 : des tests hors-reseau ecrivaient quand meme
      dans le vrai fichier de production)."""
    if fichier is None:
        fichier = FICHIER
    dernier: dict[str, tuple[str, tuple]] = {}
    if fichier.exists():
        with fichier.open(newline="", encoding="utf-8-sig") as f:
            for row in csv.reader(f):
                if not row or row[0] == "run_at":
                    continue
                dernier[row[2]] = (row[0][:10], (row[5], row[6], row[7]))
    return dernier


def deja_connu(ticker: str, ligne: list, aujourdhui: str,
               dernier: dict[str, tuple[str, tuple]]) -> bool:
    """Vrai si cette detection ne contient rien de nouveau par rapport au
    dernier etat connu de ce ticker (meme jour, ou meme (profil, G, S)
    qu'avant - momentum/dividende/G3 ignores, voir le docstring du module)."""
    precedent = dernier.get(ticker)
    if not precedent:
        return False
    # ligne = [ticker, nom, pays, profil, G, S, G3, mom, div]
    signature = (str(ligne[3]), str(ligne[4]), str(ligne[5]))
    return precedent[0] == aujourdhui or precedent[1] == signature


def archiver(horodatage: str, screener: str, trouves: list[list],
             headers: list[str], fichier: Path | None = None) -> list[list]:
    """
    --------------------------------------------------------------------------
    Purpose:
        Ajoute au protocole permanent uniquement ce qui est reellement
        nouveau ou modifie ; echoue silencieusement (une archive manquee ne
        doit jamais faire echouer un scan deja fait).

    Inputs:
        horodatage (str): ISO, utilise aussi comme cle de date (10 premiers
            caracteres)
        screener (str): nom de l'outil appelant, pour tracer la provenance
        trouves (list[list]): lignes [ticker, nom, pays, profil, G, S, G3,
            mom, div] - typiquement core.combined_finviz.lignes() filtre aux
            profils "💎"
        headers (list[str]): entete des colonnes (core.combined_finviz.HEADERS)
        fichier (Path | None): surchargeable pour les tests ; None resout
            FICHIER a l'appel (voir le docstring de charger_derniers pour le
            piege du defaut fige evite ici)

    Outputs:
        ecrites (list[list]): le sous-ensemble de "trouves" reellement
            ajoute au fichier (les autres etaient redondantes) - utile pour
            distinguer "nouveau" de "deja connu" cote appelant
    --------------------------------------------------------------------------
    """
    if not trouves:
        return []
    if fichier is None:
        fichier = FICHIER
    aujourdhui = horodatage[:10]
    dernier = charger_derniers(fichier)
    a_ecrire = [l for l in trouves if not deja_connu(l[0], l, aujourdhui, dernier)]
    if not a_ecrire:
        return []
    try:
        fichier.parent.mkdir(parents=True, exist_ok=True)
        nouveau = not fichier.exists()
        with fichier.open("a", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            if nouveau:
                w.writerow(["run_at", "screener", *headers])
            for l in a_ecrire:
                w.writerow([horodatage, screener, *l])
    except Exception:
        return []
    return a_ecrire
