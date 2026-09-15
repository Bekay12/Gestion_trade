"""Guard ueber alle drei Analysen. Vor jedem Satz Prosa auszufuehren."""
import sys
from daten import DOKUMENTE, WERTE
from werkzeug import pruefe

code = 0
for firma in WERTE:
    print(f'\n=== {firma}')
    dateien = {q: stamm for q, (stamm, _) in DOKUMENTE[firma].items()}
    versatz = {q: v for q, (_, v) in DOKUMENTE[firma].items()}
    # Die gedruckte Seite wird fuer die Pruefung auf die PDF-Seite zurueckgerechnet.
    werte_pdf = {k: (w, q, s + versatz[q]) for k, (w, q, s) in WERTE[firma].items()}
    code |= pruefe(werte_pdf, dateien, firma)
raise SystemExit(code)
