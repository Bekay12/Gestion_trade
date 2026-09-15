"""Guard : chaque valeur doit figurer dans le texte du bloc de sa page imprimée."""
from daten_us import BLOCS, DOKUMENTE, WERTE
from werkzeug import pruefe

code = 0
for firma in WERTE:
    print(f'\n=== {firma}')
    dateien = {q: stamm for q, (stamm, _) in DOKUMENTE[firma].items()}
    # la page IMPRIMÉE citée est convertie en numéro de bloc EDGAR
    werte_bloc = {k: (w, q, BLOCS[firma][s]) for k, (w, q, s) in WERTE[firma].items()}
    code |= pruefe(werte_bloc, dateien, firma)
raise SystemExit(code)
