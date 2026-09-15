"""Flow-Analyse Yara (n = 0, Variante ohne Handlungsoptionen).

Die Entscheidung ist nicht 'welche Option finanzieren', sondern 'zu welchem
Preis einsteigen'. Abschnitt 6 stellt deshalb zwei Fragen, die ALTERNATIVEN
sind, nie Summanden (Regel 12: derselbe Cash kann nicht zweimal ausgegeben
werden):

  6.1  Wie schnell traegt Yara die Schulden inklusive Gulf Coast Ammonia?
       Annahme: jeder freie Dollar tilgt. Das ist eine KAPAZITAETSRECHNUNG,
       keine Prognose.
  6.3  Was verdient ein Eigentuemer, der heute kauft?
       Annahme: nur der planmaessige Dienst wird geleistet, der Rest wird
       ausgeschuettet.

IRR und Barwert stammen aus scripts/cashflow_irr.py der Skill, damit die
Bisektion und die Abbruchtoleranz dieselben sind wie im Referenzfall.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / '.claude/skills/startup-investment-analyzer/scripts'))
from cashflow_irr import irr, npv          # noqa: E402

from werte import EXTERN, WERTE            # noqa: E402

W = {k: v[0] for k, v in WERTE.items()}
AKTIEN = W['aktien_ausstehend'] / 1e6      # Millionen Stueck

# --- Einstiegspreis ---------------------------------------------------------
KURS_NOK = EXTERN['kurs_aktuell_nok'][0]
USD_NOK = EXTERN['usd_nok_aktuell'][0]
PREIS_USD = KURS_NOK / USD_NOK

# --- Endwert ----------------------------------------------------------------
# Annahme: Endwert = Buchwert des Eigenkapitals je Aktie zum 31.12.2025,
# konstant gehalten. Belegt (GB2025 S.192), unabhaengig vom Einstiegspreis und
# konservativ: die Aktie notiert ueber Buchwert, ein Ausstieg zum Buchwert
# unterstellt also keine Bewertungsausweitung. Ein Multiplikator waere eine
# erfundene Zahl und verstiesse gegen Regel 5.
BUCHWERT_JE_AKTIE = W['ek_anteilseigner_2025'] / AKTIEN

# --- Hurdle -----------------------------------------------------------------
# Regel 9: der IRR braucht etwas zu schlagen, und der Hurdle muss belegt sein.
# Basis ist die Eigenkapitalrendite 2025 OHNE den Waehrungsumrechnungsgewinn,
# weil dieser Gewinn nicht operativ ist und die berichtete Rendite aufblaeht.
HURDLES = {
    'EK-Rendite 2025 ohne Waehrungsgewinn':
        (W['nettoergebnis_2025'] - W['waehrungsgewinn_2025']) / W['ek_anteilseigner_2025'],
    'EK-Rendite 2025 berichtet':
        W['nettoergebnis_2025'] / W['ek_anteilseigner_2025'],
    'ROIC 2025 (GB2025 S.17)':
        W['roic_2025'] / 100,
}
HURDLE_BASIS = 'EK-Rendite 2025 ohne Waehrungsgewinn'

# --- Ausschuettbarer Cash-Flow je Aktie -------------------------------------
# Regel 12: die Investorenreihe traegt den AUSSCHUETTBAREN Cash-Flow, nicht die
# Dividende allein; wer nur die erklaerte Dividende modelliert, unterschlaegt
# den Rueckkauf. Definition hier: operativer Cash-Flow abzueglich
# Investitions-Cash-Flow, beides wie berichtet.
FCF_2024 = W['cf_operativ_2024'] - W['cf_invest_2024']
FCF_2025 = W['cf_operativ_2025'] - W['cf_invest_2025']
FCF_1H2025 = 1207 - 428          # Q2-2026 S.1, Vergleichsspalte 1H 2025
FCF_1H2026 = W['cf_operativ_1h2026'] - W['cf_invest_1h2026']
FCF_LTM = FCF_2025 - FCF_1H2025 + FCF_1H2026

SZENARIEN = {
    'pessimistisch (GJ 2024, Zyklustief)': FCF_2024 / AKTIEN,
    'Basis (GJ 2025, letztes volles Jahr)': FCF_2025 / AKTIEN,
    'optimistisch (LTM bis 30.06.2026)':    FCF_LTM / AKTIEN,
}
HORIZONTE = [5, 10, 15]


def reihe(preis: float, cf_je_aktie: float, jahre: int) -> list:
    """Zahlungsreihe je Aktie: Kauf in t=0, Ausschuettung in t=1..N, Endwert in N."""
    flows = [-preis] + [cf_je_aktie] * jahre
    flows[-1] += BUCHWERT_JE_AKTIE
    return flows


def break_even(cf_je_aktie: float, jahre: int, rate: float) -> float:
    """Preis, bei dem der IRR den Hurdle exakt erreicht (Bisektion ueber den Preis).

    Der Barwert faellt streng monoton im Preis, die Klammer ist also gutartig.
    """
    lo, hi = 0.01, 1000.0
    for _ in range(200):
        mitte = (lo + hi) / 2
        if npv(reihe(mitte, cf_je_aktie, jahre), rate) > 0:
            lo = mitte
        else:
            hi = mitte
    return (lo + hi) / 2


def tabelle_irr() -> str:
    rate = HURDLES[HURDLE_BASIS]
    zeilen = ["| Szenario | Ausschüttbarer CF je Aktie | 5 Jahre | 10 Jahre | 15 Jahre |",
              "|---|---|---|---|---|"]
    for name, cf in SZENARIEN.items():
        werte = [f"{irr(reihe(PREIS_USD, cf, h)) * 100:.1f} %" for h in HORIZONTE]
        zeilen.append(f"| {name} | {cf:.2f} USD | " + " | ".join(werte) + " |")
    zeilen.append(f"| *Hurdle* | *{HURDLE_BASIS}* | "
                  + " | ".join([f"*{rate*100:.1f} %*"] * 3) + " |")
    return "\n".join(zeilen)


def tabelle_breakeven(rate: float) -> str:
    zeilen = ["| Szenario | Wachstum | 5 Jahre | 10 Jahre | 15 Jahre |", "|---|---|---|---|---|"]
    for name, cf in SZENARIEN.items():
        preise = [break_even(cf, h, rate) for h in HORIZONTE]
        zeilen.append(f"| {name} | 0 % p.a. (konstant) | "
                      + " | ".join(f"{p:.2f} USD / {p*USD_NOK:.0f} NOK" for p in preise) + " |")
    return "\n".join(zeilen)


def tabelle_entschuldung() -> str:
    netto = W['nettoschuld_1h2026']
    pro_forma = netto + W['gca_kaufpreis'] * 1000
    ebitda_ltm = W['ebitda_bsp_2025'] - W['ebitda_bsp_1h2025'] + W['ebitda_bsp_1h2026']
    zeilen = [
        "| Kennzahl | Wert | Quelle / Rechnung |", "|---|---|---|",
        f"| Nettoverschuldung 30.06.2026 | {netto:,} Mio. USD | Q2-2026, S. 34 |",
        f"| Kaufpreis Gulf Coast Ammonia | {W['gca_kaufpreis']*1000:,.0f} Mio. USD | Q2-2026, S. 26 |",
        f"| Pro-forma-Nettoverschuldung | {pro_forma:,.0f} Mio. USD | Summe der beiden Zeilen |",
        f"| EBITDA excl. Sondereffekte LTM | {ebitda_ltm:,} Mio. USD | 2.803 − 1.290 + 1.802 |",
        f"| Pro forma Nettoschuld / EBITDA | {pro_forma/ebitda_ltm:.2f} | Quotient |",
        "| Zielkorridor des Unternehmens | 1,5 – 2,0 | GB2025, S. 38 |",
        f"| Freier Cash-Flow LTM | {FCF_LTM:,} Mio. USD | 988 − 779 + 867 |",
        f"| Jahre bis Nettoverschuldung null | {pro_forma/FCF_LTM:.1f} | Kapazität, keine Prognose |",
    ]
    return "\n".join(zeilen)


if __name__ == '__main__':
    print(f"Einstiegspreis        {PREIS_USD:7.2f} USD  ({KURS_NOK} NOK / {USD_NOK})")
    print(f"Buchwert je Aktie     {BUCHWERT_JE_AKTIE:7.2f} USD")
    for name, r in HURDLES.items():
        print(f"Hurdle {name:38s} {r*100:5.2f} %")
    print("\n--- IRR beim heutigen Preis ---")
    print(tabelle_irr())
    print("\n--- Break-even-Preis (Hurdle: " + HURDLE_BASIS + ") ---")
    print(tabelle_breakeven(HURDLES[HURDLE_BASIS]))
    print("\n--- Entschuldungskapazitaet ---")
    print(tabelle_entschuldung())


def erforderlicher_endwert(cf_je_aktie: float, jahre: int, rate: float) -> float:
    """Endwert je Aktie, den der HEUTIGE Preis voraussetzt, um den Hurdle zu erreichen.

    Dreht die Frage um: statt 'welcher Preis ist vertretbar' heisst sie hier
    'welchen Ausstiegswert muss der Markt in N Jahren zahlen, damit der heutige
    Preis den Hurdle traegt'. Das macht sichtbar, wie viel des Kurses NICHT von
    den Ausschuettungen getragen wird.
    """
    barwert_ausschuettung = sum(cf_je_aktie / (1 + rate) ** t for t in range(1, jahre + 1))
    return (PREIS_USD - barwert_ausschuettung) * (1 + rate) ** jahre


def tabelle_endwert(rate: float) -> str:
    zeilen = ["| Szenario | 5 Jahre | 10 Jahre | 15 Jahre |", "|---|---|---|---|"]
    for name, cf in SZENARIEN.items():
        felder = []
        for h in HORIZONTE:
            ew = erforderlicher_endwert(cf, h, rate)
            felder.append(f"{ew:.0f} USD ({ew / BUCHWERT_JE_AKTIE:.1f}× Buchwert)")
        zeilen.append(f"| {name} | " + " | ".join(felder) + " |")
    return "\n".join(zeilen)
