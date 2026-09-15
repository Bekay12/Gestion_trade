"""Extraction ciblée de lignes chiffrées dans les blocs EDGAR découpés.

Le format des dépôts est régulier : l'étiquette sur une ligne, puis les valeurs
des exercices sur les lignes suivantes, entrecoupées de '$' et de parenthèses.
Cette fonction lit ce motif et rien d'autre ; elle ne devine jamais une valeur
absente, elle rend None, ce qui laisse le guard de citation faire son travail.
"""
import re
import sys
from pathlib import Path

BASIS = Path(__file__).resolve().parent
NOMBRE = re.compile(r'\(?\$?\s*-?[\d,]+(\.\d+)?\)?')


def _recoller(lignes):
    """Recolle les intitulés dont la première lettre est isolée.

    Le HTML de certains émetteurs (Pool Corp) place la lettre initiale d'un
    intitulé dans son propre élément : après suppression des balises, « Net
    sales » devient « N » puis « et sales ». Sans ce recollage, aucune
    étiquette de ces dépôts n'est trouvée et chaque valeur remonte None — un
    échec silencieux qui ressemble à une donnée absente.
    """
    out, i = [], 0
    while i < len(lignes):
        cour = lignes[i].strip()
        suiv = lignes[i + 1].strip() if i + 1 < len(lignes) else ''
        if len(cour) == 1 and cour.isalpha() and suiv and suiv[0].islower():
            out.append(cour + suiv)
            i += 2
        else:
            out.append(cour)
            i += 1
    return out


def valeurs(ordner, stem, bloc, etiquette, n=3):
    f = BASIS / ordner / 'seiten' / stem / f'{int(bloc):04d}.txt'
    lignes = _recoller(f.read_text().splitlines())
    for i, l in enumerate(lignes):
        if l.strip().lower() == etiquette.strip().lower():
            out, j, negatif = [], i + 1, False
            while j < len(lignes) and len(out) < n:
                s = lignes[j].strip()
                if s == '(':
                    negatif = True
                elif s in ('$', ')'):
                    pass
                elif NOMBRE.fullmatch(s):
                    v = float(s.strip('()$ ').replace(',', ''))
                    out.append(-v if negatif else v)
                    negatif = False
                elif s == '—':
                    out.append(0.0)
                elif out:
                    break
                j += 1
            return out
    return None


if __name__ == '__main__':
    ordner, stem, bloc = sys.argv[1], sys.argv[2], sys.argv[3]
    for etq in sys.argv[4:]:
        print(f'  {etq[:48]:50s} {valeurs(ordner, stem, bloc, etq)}')
