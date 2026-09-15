from daten_agi import DOKUMENTE, WERTE
from werkzeug import pruefe
raise SystemExit(pruefe(WERTE, {q: s for q, (s, _) in DOKUMENTE.items()}, 'Alamos'))
