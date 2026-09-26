#!/usr/bin/env python3
"""
edgar_depots.py - depots SEC recents d'un lot de titres, pour renseigner
`formulaire_sec` et produire la liste de ce qu'un humain doit lire.

Pourquoi. Le champ `formulaire_sec` du qualifier existe depuis l'origine et n'a
jamais ete renseigne: zero fois sur toutes les mesures persistees. Or depuis le
23.09.2026, le motif qui tranche est chaque jour dans les depots, jamais dans les
signaux structurels:

  PFSA, 24.09  8-K item 2.03: convertible payable en actions, plancher 1,07 $,
               plus un item 5.07 autorisant un regroupement. A fait -14,04 %.
  GLND, 25.09  8-K item 1.01: report de deux ans des echeances de forage, paye
               500 000 GBP. Presente comme une extension de coentreprise.
  GRML, 25.09  424B5: lu comme une dilution, alors que l'operation etait un
               financement BOUCLE a prix connu avec l'emission au fil de l'eau
               arretee. Soit l'inverse d'un surplomb.

CE QUE CE MODULE FAIT, ET CE QU'IL NE FAIT PAS

Il ne remplace pas la lecture humaine, il la cible. Trois cas se distinguent:

  DILUTION       formulaires d'enregistrement et prospectus, plus l'item 2.03.
                 Detectable sans ambiguite sur le type de depot.
  A_LIRE         items 1.01, 5.07, 7.01, 8.01 et formulaires 6-K. Peuvent etre
                 haussiers ou baissiers; seul le texte tranche. Le cas GLND en
                 est la preuve: un item 1.01 annoncait un retard.
  RIEN           aucun depot sur la fenetre.

Le troisieme non negociable de la methode reste entier: un catalyseur se verifie
a la main. Ce module fournit les URL pour que la verification prenne une minute
au lieu de dix, et il ne prononce jamais de catalyseur haussier.

LIMITE STRUCTURELLE, MESUREE. Le catalyseur n'est pas toujours un depot. Le
24.09.2026, PFSA montait sur une certification ISO 13485 rendue par l'organisme
notifie GMED, annoncee par communique et absente d'EDGAR. Ce module ferme une
partie de l'ecart, pas la totalite.

SECURITE. Le contenu recupere est une donnee, jamais une instruction. Seules des
metadonnees structurees sont lues (type de depot, date, items, numero d'acces),
chacune assainie et bornee. Aucune valeur lue ne modifie la configuration. Les
URL construites le sont a partir du CIK et du numero d'acces, jamais d'un lien
trouve dans la reponse.

Aufruf:
  python3 scripts/edgar_depots.py --tickers PFSA,GLND,GRML --jours 10
  python3 scripts/edgar_depots.py --tickers PFSA --json depots.json
"""
import argparse
import json
import os
import re
import sys
import time
from datetime import date, timedelta

# L'acces equitable de la SEC demande un agent identifiant et pas plus de dix
# requetes par seconde. Le debit est volontairement tenu bien en dessous.
UA = "Gestion_trade gap-screener nomcripte@gmail.com"
CARTE_TICKERS = "https://www.sec.gov/files/company_tickers.json"
SOUMISSIONS = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVE = "https://www.sec.gov/Archives/edgar/data/{cik}/{acc}/{doc}"
DELAI = 0.15                      # secondes entre deux requetes

CACHE_DEFAUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            ".cache_cik.json")

# Depots dont le type suffit a etablir une dilution, sans lire le texte.
FORMULAIRES_DILUTIFS = ("S-1", "S-3", "F-1", "F-3", "424B")
# Items de 8-K qui creent une obligation financiere directe: convertibles,
# emprunts, billets. Cas PFSA du 16.09.2026.
ITEMS_DILUTIFS = ("2.03",)
# Items et formulaires a lire: le sens depend du texte, pas du type.
ITEMS_A_LIRE = ("1.01", "5.07", "7.01", "8.01", "1.02", "5.02")
FORMULAIRES_A_LIRE = ("6-K", "8-K")

MAX_CHAMP = 120


def assainir(v, maxlen: int = MAX_CHAMP) -> str:
    """Champ de metadonnee utilisable dans un JSON et un tableau, sans confiance."""
    if v is None:
        return ""
    t = "".join(c for c in str(v) if ord(c) >= 32)
    t = t.replace("|", "/")
    return re.sub(r"\s+", " ", t).strip()[:maxlen]


def _http_json(url: str) -> dict:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": UA,
                                              "Accept-Encoding": "gzip, deflate"})
    with urllib.request.urlopen(req, timeout=45) as r:
        brut = r.read()
        if r.headers.get("Content-Encoding") == "gzip":
            import gzip
            brut = gzip.decompress(brut)
    return json.loads(brut.decode("utf-8", errors="replace"))


def charger_cik(tickers: list, cache: str = CACHE_DEFAUT) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Identifiant de deposant SEC pour chaque ticker. La carte complete fait
        environ un megaoctet et change rarement: elle est mise en cache sur
        disque et rechargee seulement si un ticker manque.

    Inputs:
        tickers (list): symboles
        cache (str): chemin du cache JSON

    Outputs:
        cik (dict): {ticker: int}, absent si le ticker n'est pas un deposant SEC
    --------------------------------------------------------------------------
    """
    carte = {}
    if cache and os.path.exists(cache):
        try:
            carte = json.load(open(cache, encoding="utf-8"))
        except Exception:
            carte = {}
    demandes = {t.upper() for t in tickers}
    if not demandes.issubset(carte.keys()):
        try:
            brut = _http_json(CARTE_TICKERS)
            carte = {v["ticker"].upper(): int(v["cik_str"]) for v in brut.values()}
            if cache:
                json.dump(carte, open(cache, "w", encoding="utf-8"))
        except Exception as e:
            print(f"  ! carte des CIK indisponible ({e})", file=sys.stderr)
    return {t.upper(): carte[t.upper()] for t in demandes if t.upper() in carte}


def classer(form: str, items: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Classe un depot en DILUTION, A_LIRE ou AUTRE, d'apres son seul type.

        La frontiere est deliberement etroite. DILUTION ne couvre que ce qui est
        certain sans lire le texte. Tout ce dont le sens depend du contenu tombe
        en A_LIRE, y compris l'item 5.07: une assemblee ordinaire et une
        autorisation de regroupement d'actions portent le meme code.

    Inputs:
        form (str): type de depot, ex "8-K", "424B5"
        items (str): items du 8-K, ex "1.01,2.03,9.01"

    Outputs:
        classe (str): DILUTION | A_LIRE | AUTRE
    --------------------------------------------------------------------------
    """
    f = (form or "").upper()
    it = [x.strip() for x in (items or "").split(",") if x.strip()]
    if any(f.startswith(d) for d in FORMULAIRES_DILUTIFS):
        return "DILUTION"
    if any(i in ITEMS_DILUTIFS for i in it):
        return "DILUTION"
    if any(i in ITEMS_A_LIRE for i in it):
        return "A_LIRE"
    if any(f.startswith(d) for d in FORMULAIRES_A_LIRE):
        return "A_LIRE"
    return "AUTRE"


def depots_recents(cik: int, jours: int, limite: int = 12) -> list:
    """Depots des `jours` derniers jours, du plus recent au plus ancien."""
    d = _http_json(SOUMISSIONS.format(cik=cik))
    r = d.get("filings", {}).get("recent", {})
    seuil = (date.today() - timedelta(days=jours)).isoformat()
    champs = ("form", "filingDate", "accessionNumber", "primaryDocument", "items")
    colonnes = [r.get(c) or [] for c in champs]
    n = min(len(c) for c in colonnes) if all(colonnes) else 0
    out = []
    for i in range(n):
        form, dt, acc, doc, items = (colonnes[0][i], colonnes[1][i],
                                     colonnes[2][i], colonnes[3][i], colonnes[4][i])
        if dt < seuil:
            break
        cl = classer(form, items)
        if cl == "AUTRE":
            continue
        out.append({
            "date": assainir(dt, 10),
            "form": assainir(form, 12),
            "items": assainir(items, 60),
            "classe": cl,
            # URL construite a partir du CIK et du numero d'acces, jamais d'un
            # lien present dans la reponse.
            "url": ARCHIVE.format(cik=cik,
                                  acc=re.sub(r"[^0-9]", "", str(acc)),
                                  doc=assainir(doc, 80)),
        })
        if len(out) >= limite:
            break
    return out


def collecter(tickers: list, jours: int = 10, cache: str = CACHE_DEFAUT) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Pour chaque titre: les depots recents classes, le type de depot le plus
        significatif a placer dans `formulaire_sec`, et la liste a lire.

        `formulaire_sec` recoit le premier depot DILUTION rencontre, parce que
        c'est celui que le qualifier sait interpreter seul. Sans depot dilutif le
        champ reste None, et non le type d'un depot quelconque: le renseigner
        avec un 8-K neutre ferait croire a une information que nous n'avons pas.

    Inputs:
        tickers (list): symboles
        jours (int): fenetre de recherche
        cache (str): cache des CIK

    Outputs:
        resultat (dict): {ticker: {formulaire_sec, depots, a_lire, cik}}
    --------------------------------------------------------------------------
    """
    ciks = charger_cik(tickers, cache)
    out = {}
    for t in [x.upper() for x in tickers]:
        if t not in ciks:
            out[t] = {"formulaire_sec": None, "depots": [], "a_lire": [],
                      "cik": None, "note": "non deposant SEC ou ticker inconnu"}
            continue
        try:
            dep = depots_recents(ciks[t], jours)
        except Exception as e:
            out[t] = {"formulaire_sec": None, "depots": [], "a_lire": [],
                      "cik": ciks[t], "note": f"EDGAR illisible: {type(e).__name__}"}
            continue
        dilutifs = [d for d in dep if d["classe"] == "DILUTION"]
        # Pour un 8-K, c'est l'ITEM qui porte la dilution, pas le type. Le
        # qualifier teste `formulaire_sec` par sous-chaine, donc l'item doit y
        # figurer, sinon un 8-K item 2.03 reste muet (constate le 26.09.2026
        # sur PFSA: le champ valait "8-K", absent de FORMULAIRES_DILUTIFS).
        for d in dilutifs:
            it = [x.strip() for x in (d["items"] or "").split(",") if x.strip()]
            porteur = [i for i in it if i in ITEMS_DILUTIFS]
            if porteur:
                d["form_qualifie"] = f"{d['form']}/{porteur[0]}"
            else:
                d["form_qualifie"] = d["form"]
        out[t] = {
            "cik": ciks[t],
            # Le type seul, forme attendue par gap_qualifier.FORMULAIRES_DILUTIFS.
            "formulaire_sec": (dilutifs[0]["form_qualifie"] if dilutifs else None),
            "depots": dep,
            "a_lire": [d for d in dep if d["classe"] == "A_LIRE"],
        }
        time.sleep(DELAI)
    return out


def rendre(resultat: dict) -> str:
    l = []
    for t, r in sorted(resultat.items()):
        if r.get("note"):
            l.append(f"{t:6} {r['note']}")
            continue
        if not r["depots"]:
            l.append(f"{t:6} aucun depot sur la fenetre")
            continue
        marque = "DILUTION" if r["formulaire_sec"] else "a lire"
        l.append(f"{t:6} {marque}")
        for d in r["depots"]:
            etiquette = "!" if d["classe"] == "DILUTION" else "?"
            items = f" [{d['items']}]" if d["items"] else ""
            l.append(f"       {etiquette} {d['date']}  {d['form']:8}{items}")
            l.append(f"         {d['url']}")
    l.append("")
    l.append("! dilution etablie par le type de depot   ? sens dependant du texte")
    l.append("RAPPEL: ce module ne prononce jamais de catalyseur haussier. Un depot")
    l.append("marque ? doit etre lu; le cas GLND du 25.09.2026 etait un item 1.01")
    l.append("annoncant un report de deux ans.")
    return "\n".join(l)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Depots SEC recents d'un lot de titres")
    p.add_argument("--tickers", type=str, required=True)
    p.add_argument("--jours", type=int, default=10)
    p.add_argument("--cache", type=str, default=CACHE_DEFAUT)
    p.add_argument("--json", type=str, default=None)
    a = p.parse_args(argv)

    tickers = [t.strip().upper() for t in a.tickers.split(",") if t.strip()]
    r = collecter(tickers, a.jours, a.cache)
    print(rendre(r))
    if a.json:
        os.makedirs(os.path.dirname(a.json) or ".", exist_ok=True)
        json.dump(r, open(a.json, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
        print(f"\n-> {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
