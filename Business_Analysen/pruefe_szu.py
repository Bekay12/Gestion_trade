from daten_szu import DOKUMENTE, WERTE
from werkzeug import pruefe
dateien = {q: s for q, (s, _) in DOKUMENTE.items()}
versant = {q: v for q, (_, v) in DOKUMENTE.items()}
# la page imprimée citée est reconvertie en page PDF pour la vérification
w = {k: (val, q, s + versant[q]) for k, (val, q, s) in WERTE.items()}
raise SystemExit(pruefe(w, dateien, 'Suedzucker'))
