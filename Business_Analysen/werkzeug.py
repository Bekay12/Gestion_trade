"""Gemeinsames Werkzeug aller Unternehmensanalysen in diesem Ordner.

Zwei Aufgaben, die jede Analyse braucht und die keine von Hand erledigen darf:

  index(ordner)   Seitenweiser Textauszug jeder PDF-Quelle. Seitenzahlen aus
                  Zeilennummern eines Gesamtauszugs abzuleiten erzeugt falsche
                  Zitate; hier wird jede Seite einzeln extrahiert, damit die
                  GEDRUCKTE Seite aus der Kopf- oder Fusszeile ablesbar ist.

  pruefe(...)     Guard: jeder Wert muss im Text der zitierten Seite vorkommen.
                  Urteilsvermoegen versagt bei dieser Menge, ein Skript nicht.

Aufruf:  python3 werkzeug.py index <Unterordner>
"""
import re
import subprocess
import sys
from pathlib import Path

BASIS = Path(__file__).resolve().parent


def seitenzahl(pdf: Path) -> int:
    aus = subprocess.run(['pdfinfo', str(pdf)], capture_output=True, text=True).stdout
    return int(re.search(r'Pages:\s+(\d+)', aus).group(1))


def index(ordner: str) -> None:
    """Jede PDF in <ordner>/quellen seitenweise nach <ordner>/seiten/<name>/ auszugsweise ablegen."""
    ziel_basis = BASIS / ordner / 'seiten'
    for pdf in sorted((BASIS / ordner / 'quellen').glob('*.pdf')):
        ziel = ziel_basis / pdf.stem
        ziel.mkdir(parents=True, exist_ok=True)
        n = seitenzahl(pdf)
        if len(list(ziel.glob('*.txt'))) == n:
            print(f'  {pdf.name}: {n} Seiten (bereits im Index)')
            continue
        # EIN Aufruf statt n Aufrufen: pdftotext trennt Seiten mit \f. Der
        # seitenweise Aufruf kostet bei 620 Seiten Minuten, dieser Sekunden,
        # und liefert denselben Text.
        roh = subprocess.run(['pdftotext', '-layout', str(pdf), '-'],
                             capture_output=True, text=True).stdout
        seiten_texte = roh.split('\f')
        if seiten_texte and not seiten_texte[-1].strip():
            seiten_texte.pop()
        if len(seiten_texte) != n:
            print(f'  ACHTUNG {pdf.name}: {len(seiten_texte)} Textseiten gegen '
                  f'{n} PDF-Seiten — Seitenbezug waere unsicher, Einzelabruf')
            for i in range(1, n + 1):
                (ziel / f'{i:04d}.txt').write_text(subprocess.run(
                    ['pdftotext', '-layout', '-f', str(i), '-l', str(i), str(pdf), '-'],
                    capture_output=True, text=True).stdout)
        else:
            for i, txt in enumerate(seiten_texte, start=1):
                (ziel / f'{i:04d}.txt').write_text(txt)
        print(f'  {pdf.name}: {n} Seiten')


def schreibweisen(wert):
    """Formen, in denen ein Geschaeftsbericht denselben Wert setzen kann.

    Ohne diese Varianten meldet die Pruefung bei jedem gruppierten Betrag und
    jedem negativen Wert einen Fehlalarm: Berichte schreiben 5,763 fuer 5763,
    (54) fuer -54, und europaeische Berichte zusaetzlich 5.763 und -54.
    """
    formen = set()
    if isinstance(wert, str):
        return {wert}
    if isinstance(wert, int) or (isinstance(wert, float) and float(wert).is_integer()):
        n = int(wert)
        formen |= {str(n), f'{n:,}', f'{n:,}'.replace(',', '.'), f'({n})',
                   f'({n:,})', f'({n:,}', f'({n}', f'{n:,}'.replace(',', ' ')}
        if n < 0:
            formen |= {f'({abs(n)})', f'({abs(n):,})', f'-{abs(n)}', f'–{abs(n)}'}
    else:
        f = float(wert)
        for txt in (f'{f}', f'{f:.1f}', f'{f:.2f}', f'{f:.3f}'):
            formen |= {txt, txt.replace('.', ','), f'({txt})', f'({txt.replace(".", ",")})'}
        # Dezimalzahlen ueber tausend werden gruppiert gesetzt: 5.924,1 oder
        # 5,924.1. Ohne diese Formen meldet der Guard bei jedem grossen
        # Betrag mit Nachkommastelle einen Fehlalarm.
        for nk in (1, 2):
            ganz = int(abs(round(f, nk)))
            rest_txt = f'{abs(round(f, nk)):.{nk}f}'.split('.')[1]
            gruppiert = f'{ganz:,}'
            vz = '-' if f < 0 else ''
            formen |= {
                f'{vz}{gruppiert}.{rest_txt}',
                f'{vz}{gruppiert.replace(",", ".")},{rest_txt}',
                f'({gruppiert}.{rest_txt})',
                f'({gruppiert.replace(",", ".")},{rest_txt})',
            }
    # Typographie allemande des nombres négatifs : tiret demi-cadratin suivi
    # d'une espace insécable ou ordinaire (« – 378 », « – 0,54 »). Sans ces
    # formes, tout montant négatif d'un rapport allemand remonte comme non
    # vérifié, ce qui est un faux positif et non un fauxs citation.
    negatives = set()
    for f in list(formen):
        if f.startswith('-'):
            corps = f[1:]
            negatives |= {f'– {corps}', f'–{corps}', f'− {corps}', f'−{corps}',
                          f'- {corps}', f'—{corps}', f'— {corps}'}
    formen |= negatives
    return {f for f in formen if f}


def pruefe(werte: dict, dateien: dict, ordner: str) -> int:
    """werte: {name: (wert, quelle, gedruckte_seite)} ; dateien: {quelle: pdf-stem}"""
    seiten = BASIS / ordner / 'seiten'
    fehler = []
    for name, (wert, quelle, seite) in werte.items():
        pfad = seiten / dateien[quelle] / f'{int(seite):04d}.txt'
        if not pfad.exists():
            fehler.append(f'  {name}: Seite {seite} von {quelle} nicht im Index')
            continue
        text = pfad.read_text()
        if not any(f in text for f in schreibweisen(wert)):
            fehler.append(f'  {name} = {wert}: nicht auf {quelle}, S. {seite}')
    print(f'{len(werte) - len(fehler)} von {len(werte)} Werten auf der zitierten Seite belegt')
    if fehler:
        print('NICHT BELEGT:')
        print('\n'.join(fehler))
    return 1 if fehler else 0


if __name__ == '__main__':
    if len(sys.argv) >= 3 and sys.argv[1] == 'index':
        for ordner in sys.argv[2:]:
            print(f'{ordner}:')
            index(ordner)
    else:
        print(__doc__)


# ── Dépôts EDGAR : un seul fichier HTML, pages imprimées à reconstituer ─────
def index_edgar(ordner: str) -> None:
    """Découpe chaque dépôt .htm aux sauts de page et écrit une page par fichier.

    Les pages imprimées d'un dépôt EDGAR sont délimitées par les éléments <hr>,
    et le numéro imprimé est la dernière ligne avant la coupure. Déduire la page
    d'un numéro de ligne dans un dump intégral produit des citations fausses :
    c'est la règle 1 de la skill, et le motif du contrôle de couverture ci-dessous.
    """
    import html as _html

    ziel_basis = BASIS / ordner / 'seiten'
    for depot in sorted((BASIS / ordner / 'quellen').glob('*.htm')):
        brut = depot.read_text(errors='replace')
        blocs = re.split(r'<hr\b[^>]*>', brut, flags=re.I)
        ziel = ziel_basis / depot.stem
        ziel.mkdir(parents=True, exist_ok=True)
        avec_numero = 0
        for i, bloc in enumerate(blocs, start=1):
            texte = re.sub(r'<[^>]+>', '\n', bloc)
            texte = _html.unescape(texte)
            texte = re.sub(r'[ \t ]+', ' ', texte)
            texte = '\n'.join(l.strip() for l in texte.splitlines() if l.strip())
            (ziel / f'{i:04d}.txt').write_text(texte)
            if _numero_dans(texte.splitlines()) is not None:
                avec_numero += 1
        part = avec_numero / len(blocs) * 100 if blocs else 0
        drapeau = '' if part >= 60 else '  ⚠️ pied de page non reconnu ?'
        print(f'  {depot.name}: {len(blocs)} blocs, '
              f'{avec_numero} portent un numéro imprimé ({part:.0f} %){drapeau}')


def _numero_dans(lignes):
    """Numéro de page imprimé dans les trois dernières lignes du bloc, ou None.

    Le pied de page n'a pas la même forme d'un émetteur à l'autre. Honeywell
    écrit le numéro PUIS le nom de la société ; ESAB et Pool le mettent en
    dernier. Ne chercher que la dernière ligne rendait 0 % des pages de
    Honeywell citables — et c'est précisément le contrôle de couverture qui l'a
    révélé, pas la lecture du fichier.
    """
    for ligne in reversed([l.strip() for l in lignes if l.strip()][-3:]):
        if re.fullmatch(r'\d{1,3}', ligne):
            return int(ligne)
    return None


def page_imprimee(ordner: str, stem: str, bloc: int):
    """Numéro imprimé lu sur le bloc, ou None. Jamais déduit."""
    f = BASIS / ordner / 'seiten' / stem / f'{bloc:04d}.txt'
    if not f.exists():
        return None
    return _numero_dans(f.read_text().splitlines())
