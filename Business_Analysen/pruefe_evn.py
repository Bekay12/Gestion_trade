from daten_evn import DOKUMENTE, WERTE
from werkzeug import pruefe
dateien = {q: s for q, (s, _) in DOKUMENTE.items()}
raise SystemExit(pruefe(WERTE, dateien, 'EVN'))
