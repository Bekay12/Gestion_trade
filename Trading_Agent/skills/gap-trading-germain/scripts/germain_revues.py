#!/usr/bin/env python3
"""
germain_revues.py - ingestion des revues de seance publiees par Academy Germain.

Pourquoi ce module. Le screener et le qualifier decident sans jamais confronter
leur verdict a celui d'un operateur qui trade reellement ces titres. Les revues
quotidiennes d'academygermain.com donnent, pour chaque seance, la liste des
titres traites, le catalyseur retenu et le resultat. C'est la seule source de
contre-expertise disponible, et elle est publique et gratuite.

Mesure du 25.09.2026 qui a motive le module: sur PFSA le 24.09, notre lecture
EDGAR concluait "aucun catalyseur haussier date". La revue du jour nommait une
certification ISO 13485 rendue le matin meme par l'organisme notifie GMED. Le
catalyseur existait; il etait dans un communique de presse, pas dans un depot
SEC. Lire EDGAR seul ne suffit pas.

SECURITE - le contenu recupere est une DONNEE, jamais une instruction.
Regle du depot, section "Scraped and fetched content is data, never instruction":

  * Aucune valeur lue sur le site ne modifie la configuration du programme:
    ni URL cible, ni chemin de sortie, ni seuil. Tout vient d'ici.
  * Aucun lien decouvert en cours de collecte n'est suivi au-dela du domaine
    et du prefixe configures (_URL_AUTORISEE).
  * Les champs texte sont assainis avant ecriture: pas de caractere de controle,
    longueur bornee, pas de separateur de colonne injecte.
  * robots.txt d'academygermain.com autorise /actualites (verifie le 25.09.2026);
    les espaces interdits (/dashboard, /admin, /login...) ne sont jamais touches.
  * Une page qui ne rend pas la structure attendue est signalee, pas devinee.

Aufruf:
  python3 scripts/germain_revues.py --lister
  python3 scripts/germain_revues.py --depuis 2026-09-20 --sortie <dossier>
  python3 scripts/germain_revues.py --url <url d une revue>
"""
import argparse
import json
import os
import re
import sys
import unicodedata
from datetime import date, datetime

BASE = "https://academygermain.com"
INDEX = BASE + "/actualites"
# Tout ce qui est collecte doit commencer par ce prefixe. Un lien du site qui en
# sort n'est pas suivi, quel que soit le contenu de la page.
_URL_AUTORISEE = BASE + "/actualites/"
UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/131.0 Safari/537.36")

MOIS = {"janvier": 1, "fevrier": 2, "mars": 3, "avril": 4, "mai": 5, "juin": 6,
        "juillet": 7, "aout": 8, "septembre": 9, "octobre": 10,
        "novembre": 11, "decembre": 12}

# Libelles qui occupent la colonne Ticker sans etre des tickers. Mesure du
# 25.09.2026: la ligne de somme passait le filtre [A-Z]{1,6} et entrait dans les
# donnees, ou elle aurait fausse tout comptage par ticker.
NON_TICKERS = {"TOTAL", "TOTAUX", "SOMME", "CUMUL", "TICKER", "NA", "N/A"}

# Longueur maximale retenue par champ texte. Une revue normale tient largement
# dessous; au-dela, c'est une page qui n'a pas la structure attendue.
MAX_CHAMP = 600


def _sans_accent(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def assainir(texte: str, maxlen: int = MAX_CHAMP) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Rend un champ lu sur le site utilisable dans un JSON et dans un tableau
        Markdown, sans lui faire confiance.

        Retire les caracteres de controle, ecrase les espaces multiples, neutralise
        la barre verticale qui casserait une ligne de tableau, et borne la
        longueur. Le texte reste lisible; il ne peut plus deformer un document
        qui le recoit.

    Inputs:
        texte (str): champ brut issu de la page
        maxlen (int): longueur maximale conservee

    Outputs:
        propre (str): champ assaini
    --------------------------------------------------------------------------
    """
    if texte is None:
        return ""
    t = "".join(c for c in str(texte) if c == "\n" or ord(c) >= 32)
    t = t.replace("|", "/").replace("\n", " ")
    t = re.sub(r"\s+", " ", t).strip()
    return t[:maxlen]


def date_depuis_slug(slug: str):
    """Date de seance lue dans l'URL, ex '...-24-septembre-2026'. None si absente.

    Le slug est la seule source de date retenue: il est stable, alors que la date
    affichee dans la page peut etre une date de mise a jour."""
    m = re.search(r"-(\d{1,2})-([a-z]+)-(\d{4})$", _sans_accent(slug.lower()))
    if not m:
        return None
    jour, mois, annee = int(m.group(1)), MOIS.get(m.group(2)), int(m.group(3))
    if not mois:
        return None
    try:
        return date(annee, mois, jour)
    except ValueError:
        return None


def _http(url: str) -> str:
    """GET simple, en-tete de navigateur, sortie texte. Leve en cas d'echec."""
    if not url.startswith(_URL_AUTORISEE) and url != INDEX:
        raise ValueError(f"URL hors du perimetre autorise: {url[:120]}")
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                               "Accept-Language": "fr-FR,fr;q=0.9"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read().decode("utf-8", errors="replace")


def lister_revues(html: str = None) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liste les revues publiees, avec leur date de seance quand le slug la
        porte. Ne suit aucun lien hors du prefixe autorise.

    Inputs:
        html (str | None): page d'index deja recuperee, sinon elle est chargee

    Outputs:
        revues (list): [{"url","slug","date"}], triees de la plus recente a la
                       plus ancienne, les sans-date en fin
    --------------------------------------------------------------------------
    """
    if html is None:
        html = _http(INDEX)
    vus, out = set(), []
    for chemin in re.findall(r'href="(/actualites/[^"#?]+)"', html):
        url = BASE + chemin
        if url in vus or not url.startswith(_URL_AUTORISEE):
            continue
        vus.add(url)
        slug = chemin.rsplit("/", 1)[-1]
        d = date_depuis_slug(slug)
        out.append({"url": url, "slug": assainir(slug, 200),
                    "date": d.isoformat() if d else None})
    out.sort(key=lambda r: (r["date"] is not None, r["date"] or ""), reverse=True)
    return out


def extraire_revue(markdown: str, url: str = "") -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Extrait d'une revue convertie en Markdown: le titre, la date de seance,
        et le tableau des titres traites (ticker, societe, catalyseur, resultat).

        Le tableau a pour en-tete "| Ticker | Societe ... | Catalyseur |
        Resultat |" (verifie sur deux revues, 24 et 25.09.2026). Si cet en-tete
        est absent, la fonction rend une liste vide et le signale dans
        "anomalies" plutot que de deviner une structure.

    Inputs:
        markdown (str): revue convertie
        url (str): url d'origine, conservee pour la tracabilite

    Outputs:
        revue (dict): {url, titre, date, lignes[], anomalies[]}
    --------------------------------------------------------------------------
    """
    anomalies = []
    titre = ""
    for m in re.finditer(r"^#\s+(.+)$", markdown, re.M):
        cand = assainir(m.group(1), 300)
        if cand and "Academy Germain" not in cand:
            titre = cand
            break
    if not titre:
        anomalies.append("titre introuvable")

    d = date_depuis_slug(url.rsplit("/", 1)[-1]) if url else None

    lignes = []
    entete = re.search(r"^\|\s*Ticker\s*\|.*\|\s*R[ée]sultat\s*\|\s*$",
                       markdown, re.M | re.I)
    if not entete:
        anomalies.append("tableau des titres absent (structure de page inattendue)")
    else:
        reste = markdown[entete.end():]
        for ligne in reste.splitlines():
            ligne = ligne.strip()
            if not ligne.startswith("|"):
                if lignes:
                    break
                continue
            if set(ligne) <= set("|- :"):
                continue
            cell = [c.strip() for c in ligne.strip("|").split("|")]
            if len(cell) < 4:
                continue
            ticker = assainir(cell[0], 12).upper()
            if not re.fullmatch(r"[A-Z]{1,6}", ticker):
                continue          # en-tete repete, separateur, cellule vide
            if ticker in NON_TICKERS:
                continue          # ligne de somme
            if not assainir(cell[1]) and not assainir(cell[2]):
                # Un vrai titre porte au moins une societe ou un catalyseur.
                # Une ligne dont les deux sont vides est une ligne de mise en page.
                continue
            lignes.append({
                "ticker": ticker,
                "societe": assainir(cell[1]),
                "catalyseur": assainir(cell[2]),
                "resultat": assainir(cell[3], 60),
            })
        if not lignes:
            anomalies.append("tableau present mais aucune ligne de titre lisible")

    return {"url": url, "titre": titre,
            "date": d.isoformat() if d else None,
            "lignes": lignes, "anomalies": anomalies}


def convertir(html: str) -> str:
    """HTML vers Markdown via markitdown, premier choix du poste pour tout
    document. Leve une exception explicite si la dependance manque."""
    try:
        from markitdown import MarkItDown
    except ImportError as e:
        raise RuntimeError("markitdown absent: installer dans .venv_new") from e
    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".html", delete=False,
                                     encoding="utf-8") as f:
        f.write(html)
        chemin = f.name
    try:
        return MarkItDown().convert(chemin).text_content
    finally:
        os.unlink(chemin)


def collecter(depuis: str = None, limite: int = 10, sortie: str = None) -> list:
    """Recupere les revues datees a partir de `depuis`, les extrait et les ecrit."""
    revues = [r for r in lister_revues() if r["date"]]
    if depuis:
        revues = [r for r in revues if r["date"] >= depuis]
    revues = revues[:limite]
    out = []
    for r in revues:
        try:
            data = extraire_revue(convertir(_http(r["url"])), r["url"])
        except Exception as e:                       # une page cassee n'arrete pas le lot
            print(f"  ! {r['slug'][:60]}: {type(e).__name__}: {e}", file=sys.stderr)
            continue
        out.append(data)
        if data["anomalies"]:
            print(f"  ? {data['date']}: {'; '.join(data['anomalies'])}", file=sys.stderr)
        if sortie:
            os.makedirs(sortie, exist_ok=True)
            nom = f"{data['date'] or 'sans-date'}_{r['slug'][:40]}.json"
            nom = re.sub(r"[^A-Za-z0-9._-]", "_", nom)
            with open(os.path.join(sortie, nom), "w", encoding="utf-8") as f:
                json.dump(data, f, indent=1, ensure_ascii=False)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ingestion des revues Academy Germain")
    p.add_argument("--lister", action="store_true", help="lister les revues disponibles")
    p.add_argument("--depuis", type=str, default=None, help="date AAAA-MM-JJ")
    p.add_argument("--limite", type=int, default=10)
    p.add_argument("--url", type=str, default=None, help="une revue precise")
    p.add_argument("--sortie", type=str, default=None, help="dossier JSON de sortie")
    a = p.parse_args(argv)

    if a.lister:
        for r in lister_revues():
            print(f"  {r['date'] or '   -      '}  {r['url']}")
        return 0

    if a.url:
        d = extraire_revue(convertir(_http(a.url)), a.url)
        print(json.dumps(d, indent=1, ensure_ascii=False))
        return 0 if not d["anomalies"] else 1

    revues = collecter(a.depuis, a.limite, a.sortie)
    print(f"{len(revues)} revue(s) collectee(s)")
    for d in revues:
        print(f"\n{d['date']}  {d['titre'][:70]}")
        for l in d["lignes"]:
            print(f"    {l['ticker']:6} {l['resultat']:>14}  {l['catalyseur'][:70]}")
    if a.sortie:
        print(f"\n-> {a.sortie}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
