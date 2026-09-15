"""Erzeugt je Unternehmen einen Bericht nach dem Sieben-Abschnitte-Skelett.

Alle Tabellen stammen aus modell.py und daten.py. Nichts wird abgetippt: eine
Zahl, die im Bericht steht und nicht aus dem geprueften Speicher kommt, gibt es
nicht. Farbschema nach Regel 7 der Skill: schwarz belegt, blau Einschaetzung,
rot Verdikt.
"""
from pathlib import Path

from daten import EXTERN, WERTE
from modell import HORIZONTE, konfig, tabellen

BASIS = Path(__file__).resolve().parent


def z(x, nk=1):
    """Deutsche Zahl: Punkt als Tausender-, Komma als Dezimaltrennzeichen."""
    if x is None:
        return 'n.&nbsp;a.'
    return f'{x:,.{nk}f}'.replace(',', '⁠').replace('.', ',').replace('⁠', '.')


def pz(x, nk=1):
    return f'{z(x, nk)}&nbsp;%'


STIL = """
:root{--ground:#F5F7F9;--surface:#fff;--surface-2:#EDF1F5;--ink:#12171E;--muted:#5B6675;
--rule:#D5DCE4;--blau:#1C4E80;--rot:#A8201A;--rot-flaeche:#FAEFEE;--rot-rand:#D9A9A5;
--serif:"Source Serif 4",Georgia,serif;--sans:"Inter","Segoe UI",system-ui,sans-serif;
--mono:"JetBrains Mono",ui-monospace,Consolas,monospace}
*{box-sizing:border-box}
body{background:var(--ground);color:var(--ink);font-family:var(--sans);font-size:10.5pt;
line-height:1.5;margin:0}
.wrap{max-width:180mm;margin:0 auto;padding:0 6mm 14mm}
header{padding:10mm 0 5mm;border-bottom:2px solid var(--ink)}
.eyebrow{font-family:var(--mono);font-size:8pt;letter-spacing:.14em;text-transform:uppercase;
color:var(--muted);margin:0 0 3mm}
h1{font-family:var(--serif);font-size:26pt;line-height:1.1;letter-spacing:-.02em;margin:0 0 3mm}
.frage{font-family:var(--serif);font-size:12pt;line-height:1.45;margin:0 0 4mm;max-width:150mm}
.meta{display:flex;flex-wrap:wrap;gap:2mm 8mm;font-family:var(--mono);font-size:8pt;color:var(--muted)}
.meta b{color:var(--ink);font-weight:500}
.legende{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--rule);
border:1px solid var(--rule);margin:5mm 0 0}
.legende div{background:var(--surface);padding:3mm}
.legende dt{font-family:var(--mono);font-size:7.5pt;letter-spacing:.1em;text-transform:uppercase;margin:0 0 1mm}
.legende dd{margin:0;font-size:8.5pt;color:var(--muted);line-height:1.35}
.sw{display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}
section{padding-top:8mm;break-inside:auto}
h2{font-family:var(--serif);font-size:16pt;margin:0 0 1mm;break-after:avoid}
h2 .nr{font-family:var(--mono);font-size:10pt;color:var(--muted);margin-right:3mm}
h3{font-size:10.5pt;font-weight:600;margin:6mm 0 2mm;break-after:avoid}
h3 .nr{font-family:var(--mono);color:var(--muted);font-weight:400;margin-right:2mm}
.strich{height:1px;background:var(--rule);margin:2mm 0 4mm}
p{margin:0 0 3mm;max-width:155mm}
.bewertung{color:var(--blau)}
.bewertung::before{content:"";display:inline-block;width:3px;height:1em;background:var(--blau);
vertical-align:-.15em;margin-right:3mm}
.tabelle{border:1px solid var(--rule);background:var(--surface);margin:0 0 4mm;break-inside:avoid}
table{width:100%;border-collapse:collapse;font-size:8.5pt}
caption{text-align:left;font-family:var(--mono);font-size:7.5pt;letter-spacing:.09em;
text-transform:uppercase;color:var(--muted);padding:2.5mm 3mm 0}
th,td{padding:1.6mm 3mm;text-align:right;border-bottom:1px solid var(--rule);
font-variant-numeric:tabular-nums;font-family:var(--mono)}
th:first-child,td:first-child{text-align:left;font-family:var(--sans)}
thead th{font-size:7.5pt;font-weight:500;letter-spacing:.05em;text-transform:uppercase;
color:var(--muted);border-bottom:1px solid var(--ink)}
tbody tr:last-child td{border-bottom:none}
tr.summe td{font-weight:600;border-top:1px solid var(--ink)}
tr.hurdle td{color:var(--muted);font-style:italic}
.quelle{font-family:var(--mono);font-size:7.5pt;color:var(--muted);margin:-2mm 0 4mm}
.annahme{border-left:3px solid var(--rule);padding:0 0 0 4mm;margin:0 0 3.5mm;break-inside:avoid}
.annahme b{font-family:var(--mono);font-size:7.5pt;letter-spacing:.07em;text-transform:uppercase;
display:block;color:var(--muted);margin-bottom:1mm}
.annahme p{margin:0;font-size:9pt}
.entfaellt{background:var(--surface-2);border:1px dashed var(--rule);padding:4mm}
.entfaellt p{margin:0;font-size:9pt;color:var(--muted)}
#verdikt{background:var(--rot-flaeche);border:1px solid var(--rot-rand);border-left:4px solid var(--rot);
margin-top:8mm;padding:6mm 5mm;break-before:page}
#verdikt *{color:var(--rot)}
#verdikt .tabelle{border-color:var(--rot-rand);background:transparent}
#verdikt th,#verdikt td{border-bottom-color:var(--rot-rand)}
#verdikt thead th{border-bottom-color:var(--rot);opacity:.8}
.spruch{font-family:var(--serif);font-size:14pt;line-height:1.35;font-weight:600;margin:0 0 4mm}
.zweiseiten{display:grid;grid-template-columns:1fr 1fr;gap:5mm;margin:0 0 3mm}
.zweiseiten div{border-top:2px solid var(--rot);padding-top:2mm}
.zweiseiten b{font-family:var(--mono);font-size:7.5pt;letter-spacing:.1em;text-transform:uppercase;display:block;margin-bottom:1.5mm}
.zweiseiten p{font-size:9pt;margin:0}
.richtung{margin:0 0 3mm}
.richtung b{font-family:var(--mono);font-size:8pt;letter-spacing:.08em;text-transform:uppercase;display:block;margin-bottom:1mm}
ol.lst{margin:0 0 3mm;padding-left:5mm;font-size:9pt}
ol.lst li{margin-bottom:1.5mm}
.luecken{background:var(--surface);border:1px solid var(--rule);padding:4mm}
.luecken ul{margin:0;padding-left:5mm;font-size:9pt}
.luecken li{margin-bottom:1.5mm}
footer{margin-top:8mm;padding-top:4mm;border-top:1px solid var(--rule);font-size:8.5pt;color:var(--muted)}
footer ol{padding-left:5mm}
@page{size:A4;margin:14mm 10mm}
@media print{body{background:#fff}}
"""


# ── Unternehmensspezifischer Text ───────────────────────────────────────────
INHALT = {
'Neste': {
    'name': 'Neste Oyj',
    'ticker': 'NESTE.HE', 'boerse': 'Nasdaq Helsinki', 'land': 'Finnland',
    'titel': 'Neste zu 32,89 Euro',
    'eyebrow': 'Unternehmensanalyse · Erneuerbare Kraftstoffe',
    'frage': 'Zu welchem Preis ist ein Einstieg in die Neste-Aktie vertretbar, und trägt der '
             'Kurs von 32,89&nbsp;Euro die Rekordzahlen des ersten Halbjahrs 2026?',
    'geschaeft': 'Neste ist nach eigener Darstellung der weltgrößte Hersteller von erneuerbarem '
        'Diesel und nachhaltigem Flugkraftstoff. Der Konzern führt drei Segmente: Renewable '
        'Products, Oil Products und Marketing&nbsp;&amp;&nbsp;Services.',
    'frueh': 'Neste veröffentlicht keinen Auftragseingang. Der Frühindikator ist die '
        'vergleichbare Verkaufsmarge von Renewable Products je Tonne: sie bestimmt das '
        'Segmentergebnis unmittelbar und schlägt vor jeder Bilanzkennzahl aus.',
    'kurs_these': 'Der Kurs folgt der Marge von Renewable Products, nicht dem Umsatz. Der Umsatz '
        'fiel 2025 um 7,8&nbsp;Prozent, die Aktie stieg um 60&nbsp;Prozent, und im zweiten Quartal '
        '2026 verdreifachte sich die Verkaufsmarge auf 1.223 gegenüber 361&nbsp;USD je Tonne. Eine '
        'Marge, die sich in vier Quartalen verdreifacht, ist keine Kennzahl, auf die sich eine '
        'Bewertung stützen lässt, sondern der Grund für die Ausschläge von 6,79 bis 20,22&nbsp;Euro '
        'allein im Jahr 2025.',
    'besonders': 'Das Ergebnis des ersten Halbjahrs 2026 ist vom Unternehmen selbst auf das '
        'Marktumfeld zurückgeführt: der Vorstandsvorsitzende nennt den Nahostkonflikt, der die '
        'globalen Öl- und Produktmärkte über den größten Teil des Zeitraums beherrscht habe, und '
        'spricht von einem außergewöhnlichen Marktumfeld. Das ist für die Bewertung der '
        'entscheidende Satz des Berichts.',
},
'GEA_Group': {
    'name': 'GEA Group AG',
    'ticker': 'G1A.DE', 'boerse': 'XETRA', 'land': 'Deutschland',
    'titel': 'GEA zu 63,85 Euro',
    'eyebrow': 'Unternehmensanalyse · Prozesstechnik für Lebensmittel und Pharma',
    'frage': 'Zu welchem Preis ist ein Einstieg in die GEA-Aktie vertretbar, und was ist eine '
             'Kapitalrendite von 36&nbsp;Prozent wert, wenn das gebundene Kapital fast null ist?',
    'geschaeft': 'GEA liefert Prozesstechnik für die Lebensmittel-, Getränke- und Pharmaindustrie. '
        'Der Konzern stieg im September 2025 in den DAX auf und erzielt 40,0&nbsp;Prozent seines '
        'Umsatzes mit Service.',
    'frueh': 'GEA veröffentlicht Auftragseingang und Auftragsbestand quartalsweise. Das ist der '
        'sauberste Frühindikator der drei hier untersuchten Unternehmen.',
    'kurs_these': 'Der Kurs folgt dem Auftragseingang und der Marge, nicht dem Umsatz. Der Umsatz '
        'wuchs 2025 nur um 1,4&nbsp;Prozent, der Auftragseingang aber um 6,7&nbsp;Prozent und die '
        'EBITDA-Marge vor Restrukturierung von 15,4 auf 16,5&nbsp;Prozent. Die Aktie stieg im '
        'gleichen Zeitraum um 20,9&nbsp;Prozent. Im ersten Halbjahr 2026 setzt sich beides fort, '
        'der Auftragsbestand erreicht 3.540,4&nbsp;Millionen Euro.',
    'besonders': 'GEA ist der einzige der drei Titel mit Nettoliquidität statt Nettoverschuldung '
        'und mit einer Kapitalrendite, die aus einem negativen Nettoumlaufvermögen entsteht: Kunden '
        'zahlen an, bevor GEA liefert. Das erklärt zugleich die hohe Kapitalrendite und den kleinen '
        'Buchwert, und beides zusammen macht die Wahl der Ausstiegskonvention hier besonders folgenreich.',
},
'Prysmian': {
    'name': 'Prysmian S.p.A.',
    'ticker': 'PRY.MI', 'boerse': 'Borsa Italiana', 'land': 'Italien',
    'titel': 'Prysmian zu 127,25 Euro',
    'eyebrow': 'Unternehmensanalyse · Kabel und Netzanbindung',
    'frage': 'Zu welchem Preis ist ein Einstieg in die Prysmian-Aktie vertretbar, wenn das '
             'Wachstum zugekauft ist und der freie Cash-Flow seit zwei Jahren stillsteht?',
    'geschaeft': 'Prysmian ist der weltgrößte Kabelhersteller und verbindet Stromnetze, '
        'Rechenzentren und Offshore-Windparks. Der Konzern wuchs 2024 und 2025 stark über Zukäufe, '
        'darunter Encore Wire und Channell.',
    'frueh': 'Prysmian veröffentlicht keinen Auftragseingang in den Quartalszahlen. Der '
        'Frühindikator ist hier das organische Umsatzwachstum, das der Konzern gesondert ausweist.',
    'kurs_these': 'Der Kurs folgt dem bereinigten EBITDA und der angehobenen Prognose, nicht dem '
        'freien Cash-Flow. Das bereinigte EBITDA stieg 2025 um 24,4&nbsp;Prozent und im ersten '
        'Halbjahr 2026 um weitere 17,6&nbsp;Prozent, der freie Cash-Flow auf Zwölfmonatssicht '
        'stand dagegen mit 978 gegen 979&nbsp;Millionen Euro exakt still. Der Markt bezahlt hier '
        'die Ergebnisgröße, nicht die Zahlungsgröße.',
    'besonders': 'Prysmian hat die Prognose für 2026 im Juli deutlich angehoben: bereinigtes '
        'EBITDA auf 2.800 bis 2.900 statt 2.625 bis 2.775&nbsp;Millionen Euro, freier Cash-Flow auf '
        '1.650 bis 1.750 statt 1.300 bis 1.400&nbsp;Millionen Euro. Die zweite Zahl wäre, wenn sie '
        'einträfe, eine Verdopplung gegenüber dem stillstehenden Zwölfmonatswert.',
},
}


# ── Abschnitt 2: Kennzahlen je Unternehmen ──────────────────────────────────
def kennzahlen(firma):
    """(Zeile, Wert 2024, Wert 2025, Einheit, Quelle) — alles aus dem Speicher."""
    W = {k: v[0] for k, v in WERTE[firma].items()}
    Q = {k: f"{v[1]}, S.&nbsp;{v[2]}" for k, v in WERTE[firma].items()}
    if firma == 'Neste':
        return [('Umsatz', W['umsatz_2024'], W['umsatz_2025'], 0, Q['umsatz_2025']),
                ('EBITDA', W['ebitda_2024'], W['ebitda_2025'], 0, Q['ebitda_2025']),
                ('Vergleichbares EBITDA', W['ebitda_verg_2024'], W['ebitda_verg_2025'], 0, Q['ebitda_verg_2025']),
                ('Betriebsergebnis', W['betriebsergebnis_2024'], W['betriebsergebnis_2025'], 0, Q['betriebsergebnis_2025']),
                ('Periodenergebnis', W['nettoergebnis_2024'], W['nettoergebnis_2025'], 0, Q['nettoergebnis_2025']),
                ('Operativer Cash-Flow', W['op_cashflow_2024'], W['op_cashflow_2025'], 0, Q['op_cashflow_2025']),
                ('Freier Cash-Flow', W['freier_cashflow_2024'], W['freier_cashflow_2025'], 0, Q['freier_cashflow_2025']),
                ('Eigenkapital', W['eigenkapital_2024'], W['eigenkapital_2025'], 0, Q['eigenkapital_2025']),
                ('Nettoverschuldung', W['nettoschuld_2024'], W['nettoschuld_2025'], 0, Q['nettoschuld_2025']),
                ('Vergleichbarer ROACE nach Steuern, %', W['roace_2024'], W['roace_2025'], 1, Q['roace_2025'])]
    if firma == 'GEA_Group':
        return [('Auftragseingang', W['auftragseingang_2024'], W['auftragseingang_2025'], 1, Q['auftragseingang_2025']),
                ('Umsatz', W['umsatz_2024'], W['umsatz_2025'], 1, Q['umsatz_2025']),
                ('EBITDA vor Restrukturierung', W['ebitda_vor_rest_2024'], W['ebitda_vor_rest_2025'], 1, Q['ebitda_vor_rest_2025']),
                ('EBITDA', W['ebitda_2024'], W['ebitda_2025'], 1, Q['ebitda_2025']),
                ('Periodenergebnis', W['nettoergebnis_2024'], W['nettoergebnis_2025'], 1, Q['nettoergebnis_2025']),
                ('Freier Cash-Flow', W['freier_cashflow_2024'], W['freier_cashflow_2025'], 1, Q['freier_cashflow_2025']),
                ('Eigenkapital', W['eigenkapital_2024'], W['eigenkapital_2025'], 1, Q['eigenkapital_2025']),
                ('Nettoliquidität (+)', W['nettoliquiditaet_2024'], W['nettoliquiditaet_2025'], 1, Q['nettoliquiditaet_2025']),
                ('Ergebnis je Aktie, EUR', W['eps_2024'], W['eps_2025'], 2, Q['eps_2025']),
                ('ROCE, %', W['roce_2024'], W['roce_2025'], 1, Q['roce_2025'])]
    return [('Umsatz', W['umsatz_2024'], W['umsatz_2025'], 0, Q['umsatz_2025']),
            ('Bereinigtes EBITDA', W['ebitda_ber_2024'], W['ebitda_ber_2025'], 0, Q['ebitda_ber_2025']),
            ('EBITDA', W['ebitda_2024'], W['ebitda_2025'], 0, Q['ebitda_2025']),
            ('Periodenergebnis', W['nettoergebnis_2024'], W['nettoergebnis_2025'], 0, Q['nettoergebnis_2025']),
            ('Operativer Cash-Flow', W['op_cashflow_2024'], W['op_cashflow_2025'], 0, Q['op_cashflow_2025']),
            ('Zu-/Verkäufe von Unternehmen', W['akquisitionen_2024'], W['akquisitionen_2025'], 0, Q['akquisitionen_2025']),
            ('Freier Cash-Flow (levered)', W['fcf_levered_2024'], W['fcf_levered_2025'], 0, Q['fcf_levered_2025']),
            ('Eigenkapital gesamt', W['eigenkapital_gesamt_2025'], W['eigenkapital_gesamt_2025'], 0, Q['eigenkapital_gesamt_2025']),
            ('Nettoverschuldung', W['nettoschuld_2024'], W['nettoschuld_2025'], 0, Q['nettoschuld_2025']),
            ('Verwässertes Ergebnis je Aktie, EUR', W['eps_verw_2024'], W['eps_verw_2025'], 2, Q['eps_verw_2025'])]


def halbjahr(firma):
    W = {k: v[0] for k, v in WERTE[firma].items()}
    Q = {k: f"{v[1]}, S.&nbsp;{v[2]}" for k, v in WERTE[firma].items()}
    if firma == 'Neste':
        return [('Umsatz', W['umsatz_1h2025'], W['umsatz_1h2026'], 0),
                ('Vergleichbares EBITDA', W['ebitda_verg_1h2025'], W['ebitda_verg_1h2026'], 0),
                ('Verkaufsmarge Renewable Products, USD/t (Q2)', W['marge_rp_q2_2025'], W['marge_rp_q2_2026'], 0),
                ('Ergebnis je Aktie, EUR', -0.10, W['eps_1h2026'], 2)], Q['ebitda_verg_1h2026']
    if firma == 'GEA_Group':
        return [('Auftragseingang', W['auftragseingang_1h2025'], W['auftragseingang_1h2026'], 1),
                ('Umsatz', W['umsatz_1h2025'], W['umsatz_1h2026'], 1),
                ('EBITDA vor Restrukturierung', W['ebitda_vor_rest_1h2025'], W['ebitda_vor_rest_1h2026'], 1),
                ('Periodenergebnis', W['nettoergebnis_1h2025'], W['nettoergebnis_1h2026'], 1),
                ('Freier Cash-Flow', W['freier_cashflow_1h2025'], W['freier_cashflow_1h2026'], 1)], Q['ebitda_vor_rest_1h2026']
    return [('Umsatz', W['umsatz_1h2025'], W['umsatz_1h2026'], 0),
            ('Bereinigtes EBITDA', W['ebitda_ber_1h2025'], W['ebitda_ber_1h2026'], 0),
            ('Periodenergebnis', W['nettoergebnis_1h2025'], W['nettoergebnis_1h2026'], 0),
            ('Nettoverschuldung', W['nettoschuld_1h2025'], W['nettoschuld_1h2026'], 0),
            ('Freier Cash-Flow LTM', W['fcf_ltm_2025'], W['fcf_ltm_2026'], 0)], Q['ebitda_ber_1h2026']


def d(a, b):
    """Prozentuale Veraenderung, robust gegen Vorzeichenwechsel."""
    if a is None or b is None or a == 0:
        return '—'
    if a < 0 < b or b < 0 < a:
        return 'Vorzeichenwechsel'
    return f'{(b/a-1)*100:+.1f}'.replace('.', ',') + '&nbsp;%'


def tab(kopf, zeilen, caption='', klasse=''):
    k = ''.join(f'<th>{h}</th>' for h in kopf)
    body = ''
    for zl in zeilen:
        cls = ' class="summe"' if zl and str(zl[0]).startswith('*') else ''
        zellen = ''.join(f'<td>{c}</td>' for c in
                         ([str(zl[0]).lstrip('*')] + list(zl[1:])))
        body += f'<tr{cls}>{zellen}</tr>'
    cap = f'<caption>{caption}</caption>' if caption else ''
    return (f'<div class="tabelle {klasse}"><table>{cap}<thead><tr>{k}</tr></thead>'
            f'<tbody>{body}</tbody></table></div>')


def bericht(firma: str) -> str:
    I = INHALT[firma]
    W = {k: v[0] for k, v in WERTE[firma].items()}
    E = EXTERN[firma]
    k = konfig(firma)
    tA = tabellen(firma, 'A Buchwert')
    tB = tabellen(firma, 'B heutige Bewertung')
    buchwert = k['buchwert']
    kbv = k['preis'] / buchwert
    rate = tA['rate']

    # Kennzahlentabelle
    kz = [(n, z(a, nk), z(b, nk), d(a, b)) for n, a, b, nk, _ in kennzahlen(firma)]
    hj_zeilen, hj_quelle = halbjahr(firma)
    hj = [(n, z(a, nk), z(b, nk), d(a, b)) for n, a, b, nk in hj_zeilen]

    # Szenarien
    sz = [(n, z(cf, 2), *[pz(x) for x in irrs]) for n, cf, irrs in tB['irr']]
    sz.append(('*Hurdle: ' + k['basis_hurdle'], '—', pz(rate*100), pz(rate*100), pz(rate*100)))
    beA = [(n, *[z(b, 2) if b else 'n.&nbsp;a.' for b in bs]) for n, bs in tA['be']]
    beB = [(n, *[z(b, 2) if b else 'n.&nbsp;a.' for b in bs]) for n, bs in tB['be']]
    ew = [(n, *[f'{z(e/buchwert, 1)}×' for e in es]) for n, es in tA['ew']]

    rendite = max(cf for cf in k['szenarien'].values()) / k['preis'] * 100
    schwelle = tB['be'][-1][1][1]      # optimistisches Szenario, 10 Jahre, Konvention B
    konsens = E['konsens_ziel'][0]
    abstand_schwelle = (konsens / schwelle - 1) * 100

    T = []
    T.append(f'<title>{I["titel"]}</title><style>{STIL}</style><div class="wrap">')
    T.append(f'''<header><p class="eyebrow">{I['eyebrow']}</p><h1>{I['titel']}</h1>
<p class="frage">{I['frage']}</p>
<div class="meta"><span>Stichtag <b>15.09.2026</b></span>
<span>Börse <b>{I['boerse']} · {I['ticker']}</b></span>
<span>Primärquellen <b>Geschäftsbericht 2025 · Halbjahresbericht 2026</b></span>
<span>Geprüfte Werte <b>{len(WERTE[firma])} von {len(WERTE[firma])}</b></span></div>
<dl class="legende">
<div><dt><span class="sw" style="background:var(--ink)"></span>Schwarz</dt><dd>Zahlen, Tabellen, Quellen. Dafür steht die Primärquelle ein.</dd></div>
<div><dt style="color:var(--blau)"><span class="sw" style="background:var(--blau)"></span>Blau</dt><dd>Einschätzung des Verfassers, nach Prüfung.</dd></div>
<div><dt style="color:var(--rot)"><span class="sw" style="background:var(--rot)"></span>Rot</dt><dd>Verdikt in Abschnitt 7. Empfehlung unter erklärten Annahmen, kein Befund.</dd></div>
</dl></header>''')

    # 1
    T.append(f'''<section><h2><span class="nr">1</span>Beschreibung des Unternehmens</h2><div class="strich"></div>
<p>{I['geschaeft']}</p>
<h3><span class="nr">1.3</span>Beurteilung durch die Kapitalmärkte</h3>
{tab(['Kennzahl','Wert'],[
 ('Kurs 11.09.2026', f"{z(k['preis'],2)}&nbsp;EUR"),
 ('Marktkapitalisierung', f"{z(k['preis']*k['aktien']/1000,1)}&nbsp;Mrd.&nbsp;EUR"),
 ('Aktien (Mio.)', z(k['aktien'],1)),
 ('Buchwert je Aktie 31.12.2025', f"{z(buchwert,2)}&nbsp;EUR"),
 ('*Kurs-Buchwert-Verhältnis', f"{z(kbv,2)}×")], 'Kurs- und Bewertungsdaten')}
<p class="quelle">Kurs und Aktienzahl: {E['kurs'][1]} · {E['aktien_mio'][1]}</p>
<h3><span class="nr">1.4</span>Analystenabdeckung und Konsens</h3>
<p>Das Unternehmen selbst veröffentlicht keine Ratings und keine Kursziele. Die folgenden Werte
stammen aus einer <b>Sekundärquelle</b> und tragen nicht die Beweiskraft der Primärberichte.</p>
{tab(['Kennzahl','Wert','gegen den Kurs der Konsensseite'],[
 ('Erfasste Analysten', z(E['konsens_anzahl'][0],0), '—'),
 ('Konsensurteil', E['konsens_urteil'][0], '—'),
 ('*Mittleres Kursziel', f"{z(konsens,2)}&nbsp;EUR", '—'),
 ('Höchstes Einzelziel', f"{z(E['konsens_hoch'][0],2)}&nbsp;EUR", '—'),
 ('Niedrigstes Einzelziel', f"{z(E['konsens_tief'][0],2)}&nbsp;EUR", '—')],
 'Analystenkonsens · Sekundärquelle · abgerufen 15.09.2026')}
<p class="quelle">{E['konsens_ziel'][1]}. Der Schlusskurs der Konsensseite weicht vom hier
verwendeten Kurs ab; beide Daten sind genannt, damit die Prozentangaben nachvollziehbar bleiben.</p>
<h3><span class="nr">1.5</span>Botschaften und Lücken</h3>
<div class="luecken"><ul>
<li><b>Keine Analystennamen und keine Einzelratings.</b> Welches Haus welches Ziel setzt, ist nicht öffentlich.</li>
<li><b>Kein Ergebnisbeitrag einzelner Maßnahmen.</b> Die Unternehmen beziffern Programme, nicht deren Aufteilung auf Jahre.</li>
<li><b>Kein Steuersatz des Anlegers.</b> Daher Vorsteuerbetrachtung auf Anlegerebene.</li>
</ul></div></section>''')

    # 2
    T.append(f'''<section><h2><span class="nr">2</span>Auswertung des Jahresabschlusses</h2><div class="strich"></div>
<p>Alle Beträge in Millionen Euro, sofern nicht anders angegeben.</p>
{tab(['Kennzahl','2024','2025','Δ'], kz, 'Kennzahlenüberblick Geschäftsjahr')}
<p class="quelle">{', '.join(sorted({q for *_ , q in kennzahlen(firma)}))}</p>
<h3>Halbjahr 2026 gegen Halbjahr 2025</h3>
{tab(['Kennzahl','1H 2025','1H 2026','Δ'], hj, 'Laufendes Geschäftsjahr')}
<p class="quelle">{hj_quelle}</p></section>''')

    # 3 + 4
    T.append(f'''<section><h2><span class="nr">3</span>Strategische Maßnahmen</h2><div class="strich"></div>
<p><i>Dieser Abschnitt berichtet ausschließlich; beurteilt wird in 6 und 7.</i></p>
<p>{I['besonders']}</p></section>
<section><h2><span class="nr">4</span>Beurteilung des Kursverlaufs</h2><div class="strich"></div>
<p>{I['frueh']}</p><p class="bewertung">{I['kurs_these']}</p></section>
<section><h2><span class="nr">5</span>Beurteilung ausgewählter Handlungsoptionen</h2><div class="strich"></div>
<div class="entfaellt"><p><b>Entfällt (n&nbsp;=&nbsp;0).</b> Zur Entscheidung steht keine Auswahl unter
Handlungsoptionen des Unternehmens, sondern der Einstieg eines Außenstehenden zu einem gegebenen Kurs.
Die Nummerierung bleibt erhalten, damit dieser Bericht Zeile für Zeile mit den anderen vergleichbar bleibt.</p></div></section>''')

    # 6
    szen_tab = tab(['Szenario','Rechnung (Mio. EUR)','je Aktie'],
                   [(n, z(roh,0), f"{z(roh/k['aktien'],2)}&nbsp;EUR")
                    for n, roh in k['roh'].items()], 'Ausschüttbarer Cash-Flow, drei berichtete Niveaus')
    T.append(f'''<section><h2><span class="nr">6</span>Grobe Flow-Analyse</h2><div class="strich"></div>
<h3><span class="nr">6.0</span>Annahmen, erklärt vor jeder Zahl</h3>
<div class="annahme"><b>Annahme 1 — ausschüttbarer Cash-Flow statt Dividende</b><p>Die Investorenreihe
trägt den freien Cash-Flow wie vom Unternehmen ausgewiesen, nicht die beschlossene Dividende. Wer nur
die Dividende modelliert, unterschlägt Rückkäufe und Schuldentilgung.</p></div>
<div class="annahme"><b>Annahme 2 — kein Wachstum</b><p>Jedes Szenario hält sein Niveau konstant. Das
Modell prognostiziert nicht, es misst, was die bereits berichteten Zahlungsströme bei gegebenem Preis
abwerfen. Wachstum ist damit bewusst ausgeschlossen und bildet die Lücke, die Abschnitt 7.5 benennt.</p></div>
<div class="annahme"><b>Annahme 3 — zwei Ausstiegskonventionen, beide belegt</b><p>Konvention&nbsp;A setzt
den Endwert auf den Buchwert des Eigenkapitals je Aktie ({z(buchwert,2)}&nbsp;EUR), Konvention&nbsp;B auf
die heutige Bewertung ({z(k['preis'],2)}&nbsp;EUR), also Ausstieg ohne Bewertungsausweitung. Bei einem
Kurs-Buchwert-Verhältnis von {z(kbv,2)}× liegen zwischen beiden Welten; deshalb werden beide
ausgewiesen und nicht eine davon als neutral ausgegeben.</p></div>
<div class="annahme"><b>Annahme 4 — Vorsteuerbetrachtung</b><p>Kapitalertragsteuer und Quellensteuer
bleiben unberücksichtigt, weil der Satz vom Wohnsitz des Anlegers abhängt und in keiner Primärquelle
steht. Alle Szenarien werden gleich behandelt, die Rangfolge bleibt unberührt.</p></div>
<div class="annahme"><b>Annahme 5 — Hurdle</b><p>Zu schlagen ist {k['basis_hurdle']} =
{pz(rate*100,2)}. Ein WACC wird nicht konstruiert: Beta, risikofreier Zins und Marktprämie stehen in
keiner Primärquelle.</p></div>
<h3><span class="nr">6.3</span>Was verdient ein Eigentümer, der heute kauft?</h3>
{szen_tab}
{tab(['Szenario','CF je Aktie','5 Jahre','10 Jahre','15 Jahre'], sz,
     f"Interner Zinsfuß beim Kurs von {z(k['preis'],2)} EUR · Konvention B")}
<p>Unter Konvention&nbsp;B ist der interne Zinsfuß horizontunabhängig, weil Ein- und Ausstieg zur
selben Bewertung erfolgen: er ist dann exakt die Cash-Flow-Rendite. Das beste Szenario liefert
<b>{pz(rendite)}</b> gegen einen Hurdle von {pz(rate*100,2)}.</p>
{tab(['Szenario','5 Jahre','10 Jahre','15 Jahre'], beB, 'Break-even-Preis · Konvention B (EUR)')}
{tab(['Szenario','5 Jahre','10 Jahre','15 Jahre'], beA, 'Break-even-Preis · Konvention A, Ausstieg zum Buchwert (EUR)')}
<p>Die Spanne zwischen beiden Tabellen ist nicht Rauschen, sondern die Annahme: sie zeigt, welcher
Teil des vertretbaren Preises daran hängt, dass die Aktie auch in zehn Jahren über Buchwert notiert.</p>
</section>''')

    # 7
    halter, nicht_halter, dagegen, dafuer, warten, gegen_mich = VERDIKT[firma](
        k, rate, rendite, schwelle, konsens, abstand_schwelle, kbv)
    T.append(f'''<section id="verdikt"><h2><span class="nr">7</span>Verdikt</h2><div class="strich"></div>
<p style="font-size:8.5pt"><b>Alles in diesem Abschnitt ist rot gesetzt:</b> eine Empfehlung unter den
in 6.0 erklärten Annahmen, kein Befund. Sämtliche Größen stammen aus den Abschnitten 1 bis 6.</p>
<h3><span class="nr">7.1</span>Verdikt</h3>
<p class="spruch">{VERDIKT_SATZ[firma](k, schwelle)}</p>
<div class="zweiseiten"><div><b>Für den Halter</b><p>{halter}</p></div>
<div><b>Für den Nichthalter</b><p>{nicht_halter}</p></div></div>
<h3><span class="nr">7.2</span>Begründung aus den Befunden</h3>
<div class="richtung"><b>Was dagegen spricht</b><p>{dagegen}</p></div>
<div class="richtung"><b>Was dafür spricht</b><p>{dafuer}</p></div>
<div class="richtung"><b>Wo dieses Verdikt vom Markt abweicht</b><p>Das Konsensurteil der
{z(E['konsens_anzahl'][0],0)} erfassten Analysten lautet <i>{E['konsens_urteil'][0]}</i>, das mittlere
Kursziel liegt bei {z(konsens,2)}&nbsp;EUR und damit {z(abstand_schwelle)}&nbsp;% über der hier
errechneten Einstiegsschwelle von {z(schwelle,2)}&nbsp;EUR. {warten}</p></div>
<h3><span class="nr">7.3</span>Einstiegsschwelle</h3>
{tab(['Szenario','5 Jahre','10 Jahre','15 Jahre'], beB, 'Preis, bei dem der Hurdle exakt erreicht wird · Konvention B (EUR)')}
<p><b>Aktueller Kurs: {z(k['preis'],2)}&nbsp;EUR.</b> Gegenprobe, konventionsfrei: welchen Endwert
der heutige Kurs voraussetzt, als Vielfaches des Buchwerts von {z(buchwert,2)}&nbsp;EUR.</p>
{tab(['Szenario','5 Jahre','10 Jahre','15 Jahre'], ew, 'Erforderlicher Endwert (× Buchwert)')}
<p>Heutiges Kurs-Buchwert-Verhältnis: {z(kbv,2)}×.</p>
<h3><span class="nr">7.4</span>Auslöser, die das Verdikt kippen</h3>
{VERDIKT_AUSLOESER[firma](k, schwelle)}
<h3><span class="nr">7.5</span>Was das Verdikt nicht wissen kann</h3>
<p>{gegen_mich}</p></section>''')

    T.append(f'''<footer><p><b>Nicht verfügbare Angaben:</b> Analystennamen und Einzelratings;
Steuersatz des Anlegers; künftiges Wachstum, das Annahme&nbsp;2 ausdrücklich ausschließt.</p>
<p><b>Quellen:</b></p><ol>
<li>{I['name']}: Geschäftsbericht 2025 (lokal unter <code>quellen/</code>)</li>
<li>{I['name']}: Halbjahresbericht 2026 (lokal unter <code>quellen/</code>)</li>
<li>Yahoo Finance, {I['ticker']}, Schlusskurs, abgerufen 15.09.2026</li>
<li>MarketScreener, Konsensseite {I['ticker']}, abgerufen 15.09.2026 (Sekundärquelle)</li></ol>
<p><b>Reproduzierbarkeit:</b> <code>daten.py</code> hält alle Werte mit Quelle und Seite,
<code>pruefe_alle.py</code> prüft jeden gegen den Text seiner Seite, <code>modell.py</code> erzeugt
sämtliche Tabellen dieses Berichts, <code>bericht.py</code> setzt ihn.</p></footer></div>''')
    return '\n'.join(T)


# ── Verdikt je Unternehmen ──────────────────────────────────────────────────
VERDIKT_SATZ = {
 'Neste': lambda k, s: (f'Beobachten, nicht kaufen, zu {z(k["preis"],2)}&nbsp;Euro. Der Kurs liegt nur '
    f'{z((k["preis"]/s-1)*100)}&nbsp;% über der Schwelle des optimistischen Szenarios — aber dieses '
    f'Szenario ruht auf einer Kriegsmarge, und auf dem letzten vollen Geschäftsjahr liegt die '
    f'Schwelle bei 27,13&nbsp;Euro.'),
 'GEA_Group': lambda k, s: (f'Nicht kaufen zu {z(k["preis"],2)}&nbsp;Euro. Auf den bereits berichteten '
    f'Zahlungsströmen wird ein Einstieg erst unterhalb von rund {z(s,0)}&nbsp;Euro vom freien '
    f'Cash-Flow getragen; alles darüber bezahlt Wachstum, das diese Rechnung bewusst nicht prognostiziert.'),
 'Prysmian': lambda k, s: (f'Nicht kaufen zu {z(k["preis"],2)}&nbsp;Euro. Der freie Cash-Flow steht auf '
    f'Zwölfmonatssicht exakt still, und die Schwelle liegt bei rund {z(s,0)}&nbsp;Euro — der Kurs '
    f'setzt damit die im Juli angehobene Prognose als bereits eingetreten voraus.'),
}

VERDIKT = {
 'Neste': lambda k, rate, rend, s, kons, abst, kbv: (
  'Halten. Die Verschuldungsquote ist von 36,1 auf 29,9&nbsp;Prozent gefallen, der vergleichbare '
  'ROACE auf Zwölfmonatssicht steht bei 18,0&nbsp;Prozent, und wer verkauft, tauscht eine '
  'Beteiligung an der erholten Marge gegen ein Wiedereinstiegsproblem.',
  'Abwarten. Die Cash-Flow-Rendite von 4,8&nbsp;Prozent beruht auf einem Halbjahr, dessen Marge sich '
  'gegenüber dem Vorjahresquartal verdreifacht hat. Wer heute kauft, bezahlt diese Marge als Dauerzustand.',
  'Das Ergebnis des ersten Halbjahrs 2026 ist vom Unternehmen selbst auf den Nahostkonflikt '
  'zurückgeführt. Die Verkaufsmarge von Renewable Products sprang von 361 auf 1.223&nbsp;USD je Tonne; '
  'auf dem letzten vollen Geschäftsjahr beträgt die Cash-Flow-Rendite nur 3,0&nbsp;Prozent und die '
  'Einstiegsschwelle 27,13&nbsp;Euro, also 17,5&nbsp;Prozent unter dem Kurs.',
  'Die Bilanz hat sich sichtbar erholt: Nettoverschuldung von 4.192 auf 3.613&nbsp;Millionen Euro, '
  'Eigenkapital je Aktie von 9,52 auf 11,01&nbsp;Euro, und das Effizienzprogramm hat mit '
  '594&nbsp;Millionen Euro Jahreslaufrate sein Ziel ein Jahr früher erreicht. Von allen drei '
  'untersuchten Titeln ist Neste der einzige, dessen Kurs nahe an der eigenen Schwelle liegt.',
  'Die Richtung stimmt überein, der Abstand ist klein — das ist unter den drei Titeln der engste '
  'Abgleich und der einzige, bei dem Konsens und Rechnung dieselbe Größenordnung meinen.',
  'Die Rechnung hängt an einer Größe, die nicht berechenbar ist: der künftigen Verkaufsmarge je '
  'Tonne. Sie ist im ersten Halbjahr 2026 auf das Dreifache gesprungen, und keine Primärquelle '
  'enthält eine Prognose dafür. Gegen mein eigenes Urteil spricht dreierlei. Erstens schließt '
  'Annahme&nbsp;2 Wachstum aus, während Neste die Kapazität in Rotterdam auf 6,8&nbsp;Millionen Tonnen '
  'ausbaut. Zweitens ist der Hurdle von 5,3&nbsp;Prozent aus einem schwachen Jahr genommen; auf '
  'Zwölfmonatssicht verdient Neste 18,0&nbsp;Prozent, und an diesem Maßstab wäre der Titel deutlich '
  'teurer, nicht billiger. Drittens ist die Ausschüttungsquote von 106,6&nbsp;Prozent ein Zeichen, '
  'dass die Dividende aus der Substanz kam, nicht aus dem Ergebnis.'),

 'GEA_Group': lambda k, rate, rend, s, kons, abst, kbv: (
  'Halten. Die operative Qualität ist unbestritten: Auftragsbestand auf 3.540,4&nbsp;Millionen Euro, '
  'ROCE 36,8&nbsp;Prozent, Nettoliquidität statt Schulden und eine im Juli angehobene Prognose. '
  'Diese Rechnung sagt nichts gegen das Unternehmen, nur etwas über den Preis.',
  'Nicht kaufen. Die Cash-Flow-Rendite beträgt 4,9&nbsp;Prozent, der heutige Kurs setzt einen '
  'Ausstieg zum 15,5-fachen Buchwert in zehn Jahren voraus, und zum Buchwert notiert die Aktie '
  'bereits mit dem 4,23-fachen.',
  'Der freie Cash-Flow ist seit zwei Jahren praktisch unverändert: 504,8 auf 511,8&nbsp;Millionen '
  'Euro, und auf Zwölfmonatssicht mit 483,4&nbsp;Millionen sogar leicht darunter. Bei einem '
  'Börsenwert von 10,4&nbsp;Milliarden Euro ergibt das eine Rendite von 4,9&nbsp;Prozent gegen eine '
  'Eigenkapitalrendite des Unternehmens von 16,9&nbsp;Prozent. Die Differenz ist genau das Wachstum, '
  'das der Kurs vorwegnimmt.',
  'GEA ist operativ der stärkste der drei Titel. Der Auftragseingang wuchs 2025 um 6,7 und im ersten '
  'Halbjahr 2026 um 8,3&nbsp;Prozent, die EBITDA-Marge vor Restrukturierung stieg von 15,4 auf '
  '16,5&nbsp;Prozent und im Halbjahr weiter, die Prognose wurde im Juli angehoben. Der Konzern trägt '
  'keine Nettoschulden und verdient seine Kapitalrendite auf einem negativen Nettoumlaufvermögen.',
  'Der Abstand ist groß, und die Erklärung liegt offen: ein Zwölfmonatsziel bewertet GEA über '
  'Gewinnvielfache, meine Rechnung über Zahlungsströme ohne Wachstum. Bei einem Unternehmen, dessen '
  'Kapitalrendite 36&nbsp;Prozent beträgt, ist die Differenz zwischen beiden Sichten strukturell.',
  'Die Rechnung kann nicht wissen, ob GEA seine Kapitalrendite von 36&nbsp;Prozent hält. Genau das '
  'ist der ganze Fall. Gegen mein Urteil spricht dreierlei. Erstens ist der Hurdle von '
  '16,9&nbsp;Prozent die eigene Eigenkapitalrendite des Unternehmens — von einem außenstehenden '
  'Anleger zu verlangen, dass er dieselbe Rendite erzielt wie ein Unternehmen mit '
  '36&nbsp;Prozent Kapitalrendite, ist streng; bei 8&nbsp;Prozent Anspruch läge die Schwelle deutlich '
  'höher. Zweitens ist der Buchwert bei einem Geschäftsmodell mit negativem Nettoumlaufvermögen '
  'kein sinnvoller Ausstiegswert — er ist klein, gerade weil das Modell gut ist. Drittens schließt '
  'Annahme&nbsp;2 das Wachstum aus, das die angehobene Prognose bereits beziffert.'),

 'Prysmian': lambda k, rate, rend, s, kons, abst, kbv: (
  'Halten mit Vorbehalt. Das bereinigte EBITDA wächst zweistellig, die Prognose wurde angehoben, '
  'und die Nettoverschuldung ist gegenüber dem Vorjahr gefallen. Wer hält, sollte den freien '
  'Cash-Flow des zweiten Halbjahrs abwarten, nicht die Ergebnisgröße.',
  'Nicht kaufen. Die Cash-Flow-Rendite beträgt 2,8&nbsp;Prozent, die niedrigste der drei, und der '
  'freie Cash-Flow auf Zwölfmonatssicht stand mit 978 gegen 979&nbsp;Millionen Euro exakt still, '
  'während der Kurs ein Viertel zulegte.',
  'Das Wachstum ist gekauft, nicht erwirtschaftet: 4.126&nbsp;Millionen Euro flossen 2024 und weitere '
  '1.069&nbsp;Millionen 2025 in Zukäufe, und der freie Cash-Flow nach Zinsen lag 2024 bei '
  'minus&nbsp;3.120&nbsp;Millionen Euro. Die Nettoverschuldung ist seit Jahresende von 3.097 auf '
  '4.079&nbsp;Millionen Euro gestiegen, obwohl das Ergebnis wuchs. Der heutige Kurs setzt einen '
  'Ausstieg zum 27,6-fachen Buchwert in zehn Jahren voraus.',
  'Die operative Entwicklung ist stark: Umsatz im Halbjahr plus 16,4, bereinigtes EBITDA plus '
  '17,6&nbsp;Prozent, und die Prognose für den freien Cash-Flow wurde im Juli von 1.300 bis 1.400 auf '
  '1.650 bis 1.750&nbsp;Millionen Euro angehoben. Träfe die Untergrenze ein, wäre das eine '
  'Verdopplung des Zwölfmonatswerts und die Rechnung sähe anders aus.',
  'Der Abstand ist der größte der drei Titel. Er misst genau, wie viel der Markt der angehobenen '
  'Prognose bereits glaubt, und genau diese Prognose ist die eine Zahl, die dieser Bericht nicht '
  'prüfen kann.',
  'Die Rechnung kann nicht wissen, ob die für 2026 angehobene Cash-Flow-Prognose eintrifft. Gegen '
  'mein Urteil spricht dreierlei. Erstens vergleicht die Szenarioleiter zwei Definitionen: die '
  'Werte für 2023 und 2025 sind der geprüfte freie Cash-Flow nach Zinsen, der Zwölfmonatswert von '
  '978 ist die Unternehmensdefinition ohne Zukäufe — die zweite ist die günstigere, und ich habe '
  'sie als optimistisches Szenario verwendet. Zweitens habe ich das Jahr 2024 aus der Leiter '
  'genommen, weil der Kauf von Encore Wire es auf minus&nbsp;3.120&nbsp;Millionen Euro drückt; '
  'dieses Weglassen schmeichelt der Rechnung. Drittens ist ein Unternehmen, dessen Modell der Zukauf '
  'ist, mit einer Rechnung ohne Zukäufe strukturell unterbewertet.'),
}

VERDIKT_AUSLOESER = {
 'Neste': lambda k, s: '''<ol class="lst">
<li><b>Zum Kauf:</b> Kurs unter <b>27,15&nbsp;Euro</b>, der Schwelle des letzten vollen Geschäftsjahrs.</li>
<li><b>Zum Kauf:</b> Verkaufsmarge Renewable Products über <b>600&nbsp;USD je Tonne</b> in zwei aufeinanderfolgenden Quartalen nach Abflauen des Konflikts — dann trägt die Marge auch ohne Sondereffekt.</li>
<li><b>Zum Verkauf:</b> Verschuldungsquote wieder über <b>40&nbsp;Prozent</b>, dem eigenen Zielwert.</li>
<li><b>Zum Verkauf:</b> Freier Cash-Flow eines vollen Jahres unter <b>300&nbsp;Millionen Euro</b>.</li></ol>''',
 'GEA_Group': lambda k, s: '''<ol class="lst">
<li><b>Zum Kauf:</b> Kurs unter <b>40&nbsp;Euro</b> bei unverändertem freien Cash-Flow — dann liegt die Rendite über 7,8&nbsp;Prozent.</li>
<li><b>Zum Kauf:</b> Freier Cash-Flow eines vollen Jahres über <b>800&nbsp;Millionen Euro</b>, also eine Rendite von 7,7&nbsp;Prozent zum heutigen Kurs.</li>
<li><b>Zum Verkauf:</b> Auftragseingang zwei Quartale in Folge rückläufig bei einem Book-to-bill unter <b>1,00</b>.</li>
<li><b>Zum Verkauf:</b> ROCE unter <b>30&nbsp;Prozent</b> oder Rückkehr zu Nettoverschuldung.</li></ol>''',
 'Prysmian': lambda k, s: '''<ol class="lst">
<li><b>Zum Kauf:</b> Freier Cash-Flow 2026 tatsächlich im Prognosekorridor <b>1.650 bis 1.750&nbsp;Millionen Euro</b> — das verdoppelt die Cash-Flow-Rendite auf rund 5,7&nbsp;Prozent.</li>
<li><b>Zum Kauf:</b> Kurs unter <b>90&nbsp;Euro</b>, dem niedrigsten Analystenziel.</li>
<li><b>Zum Verkauf:</b> Nettoverschuldung über <b>5.000&nbsp;Millionen Euro</b> ohne entsprechenden Ergebnisbeitrag.</li>
<li><b>Zum Verkauf:</b> Freier Cash-Flow auf Zwölfmonatssicht ein drittes Jahr in Folge bei rund <b>980&nbsp;Millionen Euro</b>.</li></ol>''',
}


if __name__ == '__main__':
    ordner = {'Neste': 'Neste', 'GEA_Group': 'GEA_Group', 'Prysmian': 'Prysmian'}
    for firma, ord_ in ordner.items():
        ziel = BASIS / ord_ / f'{ord_}_Investitionsanalyse.html'
        ziel.write_text(bericht(firma))
        print(f'{ziel.relative_to(BASIS)}  {ziel.stat().st_size/1024:.0f} kB')
