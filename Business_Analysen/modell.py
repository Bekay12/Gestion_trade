"""Eigentuemerrechnung der drei Dual Champions (n = 0, optionslose Variante).

Die Zahlungsreihe je Aktie ist: Kaufpreis in t=0, ausschuettbarer Cash-Flow in
t=1..N, Endwert in N. IRR, Barwert, Break-even-Preis und erforderlicher Endwert
kommen aus scripts/cashflow_irr.py der Skill, ueber das dort vorgesehene Feld
`terminal_value` - ohne Endwert beantwortete das Modell eine andere Frage und
lieferte fuer jedes Unternehmen einen Break-even weit unter dem Marktpreis.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path.home() / '.claude/skills/startup-investment-analyzer/scripts'))
from cashflow_irr import (break_even_investment, full_cashflows, irr,  # noqa: E402
                          required_terminal_value)

from daten import EXTERN, WERTE  # noqa: E402

HORIZONTE = [5, 10, 15]


def w(firma, name):
    return WERTE[firma][name][0]


def konfig(firma: str) -> dict:
    """Kennzahlen je Unternehmen, alle aus dem geprueften Wertespeicher."""
    preis = EXTERN[firma]['kurs'][0]
    aktien = EXTERN[firma]['aktien_mio'][0]

    if firma == 'Neste':
        # Freier Cash-Flow wie von Neste selbst ausgewiesen (GB2025 S. 79:
        # 759 / -341) und rollierend: 759 - 0 (1H25) + 450 (1H26).
        szenarien = {
            'pessimistisch — GJ 2024 (Zyklustief)': w(firma, 'freier_cashflow_2024'),
            'Basis — GJ 2025 (letztes volles Jahr)': w(firma, 'freier_cashflow_2025'),
            'optimistisch — LTM bis 30.06.2026':
                w(firma, 'freier_cashflow_2025') - w(firma, 'fcf_1h2025') + w(firma, 'fcf_1h2026'),
        }
        buchwert = w(firma, 'ek_je_aktie_2025')
        hurdles = {
            'Vergleichbarer ROACE 2025 (GB2025 S. 240)': w(firma, 'roace_2025') / 100,
            'Eigenkapitalrendite 2025 (GB2025 S. 240)': w(firma, 'roe_2025') / 100,
            'Vergleichbarer ROACE LTM 30.06.2026': w(firma, 'roace_ltm_2026') / 100,
        }
        basis_hurdle = 'Vergleichbarer ROACE 2025 (GB2025 S. 240)'

    elif firma == 'GEA_Group':
        szenarien = {
            'pessimistisch — LTM bis 30.06.2026':
                w(firma, 'freier_cashflow_2025') - w(firma, 'freier_cashflow_1h2025')
                + w(firma, 'freier_cashflow_1h2026'),
            'Basis — GJ 2024': w(firma, 'freier_cashflow_2024'),
            'optimistisch — GJ 2025 (letztes volles Jahr)': w(firma, 'freier_cashflow_2025'),
        }
        buchwert = w(firma, 'eigenkapital_2025') / aktien
        hurdles = {
            'Eigenkapitalrendite 2025 (414,0 / 2.453,4)':
                w(firma, 'nettoergebnis_2025') / w(firma, 'eigenkapital_2025'),
            'ROCE 2025 (GB2025 S. 2)': w(firma, 'roce_2025') / 100,
        }
        basis_hurdle = 'Eigenkapitalrendite 2025 (414,0 / 2.453,4)'

    else:  # Prysmian
        szenarien = {
            'pessimistisch — GJ 2023 (levered)': w(firma, 'fcf_levered_2023'),
            'Basis — GJ 2025 (levered)': w(firma, 'fcf_levered_2025'),
            'optimistisch — LTM 30.06.2026 (Unternehmensdefinition)': w(firma, 'fcf_ltm_2026'),
        }
        buchwert = w(firma, 'eigenkapital_2025') / aktien
        hurdles = {
            'Eigenkapitalrendite 2025 (1.294 / 6.680)':
                w(firma, 'nettoergebnis_2025') / w(firma, 'eigenkapital_gesamt_2025'),
            'Eigenkapitalrendite auf Konzernanteil (1.294 / 6.474)':
                w(firma, 'nettoergebnis_2025') / w(firma, 'eigenkapital_2025'),
        }
        basis_hurdle = 'Eigenkapitalrendite 2025 (1.294 / 6.680)'

    return {'preis': preis, 'aktien': aktien, 'buchwert': buchwert,
            'szenarien': {k: v / aktien for k, v in szenarien.items()},
            'roh': szenarien, 'hurdles': hurdles, 'basis_hurdle': basis_hurdle}


def reihe_cfg(preis, cf, endwert, hurdle, jahre):
    return {
        'currency': 'EUR je Aktie', 'opening_liquidity': 0.0,
        'horizon_start': 2026, 'horizon_end': 2026 + jahre,
        'schedule_years': [2026], 'sensitivity_horizons': [2026 + jahre],
        'revenue_base': 1.0, 'earnings_base': 1.0,
        'hurdle': hurdle * 100,
        'hurdle_source': 'siehe Annahme 8 des jeweiligen Berichts',
        'options': {'K': {'name': 'Kauf', 'investment': {'2026': preis},
                          'earnings_uplift': cf, 'uplift_start': 2027,
                          'terminal_value': endwert}},
    }


def endwerte(k: dict) -> dict:
    """Zwei belegte Ausstiegskonventionen. Die Wahl dominiert das Ergebnis.

    A  Buchwert des Eigenkapitals je Aktie, konstant. Unabhaengig vom
       Einstiegspreis und hart konservativ. Fuer ein Unternehmen, das nahe am
       Buchwert notiert, ist das eine milde Annahme; fuer ein kapitalarmes
       Unternehmen mit hoher Kapitalrendite ist sie es nicht, denn dessen
       Buchwert ist klein, WEIL es wenig Kapital bindet.
    B  Heutiges Kurs-Buchwert-Vielfaches, auf den konstant gehaltenen Buchwert
       angewandt, also Ausstieg zur heutigen Bewertung ohne Ausweitung.

    Beide werden ausgewiesen; die Differenz ist der Teil des Urteils, der von
    der Annahme kommt und nicht von den Zahlen.
    """
    return {'A Buchwert': k['buchwert'],
            'B heutige Bewertung': k['preis']}


def tabellen(firma: str, konvention: str = 'A Buchwert') -> dict:
    k = konfig(firma)
    k['buchwert'] = endwerte(k)[konvention]
    rate = k['hurdles'][k['basis_hurdle']]
    irr_zeilen, be_zeilen, ew_zeilen = [], [], []
    for name, cf in k['szenarien'].items():
        irrs, bes, ews = [], [], []
        for h in HORIZONTE:
            c = reihe_cfg(k['preis'], cf, k['buchwert'], rate, h)
            irrs.append(irr(full_cashflows(c, 'K')) * 100)
            be = break_even_investment(c, 'K')
            bes.append(be)
            ews.append(required_terminal_value(c, 'K'))
        irr_zeilen.append((name, cf, irrs))
        be_zeilen.append((name, bes))
        ew_zeilen.append((name, ews))
    return {'k': k, 'rate': rate, 'irr': irr_zeilen, 'be': be_zeilen, 'ew': ew_zeilen}


if __name__ == '__main__':
    import itertools
    for firma, konv in itertools.product(['Neste', 'GEA_Group', 'Prysmian'],
                                          ['A Buchwert', 'B heutige Bewertung']):
        t = tabellen(firma, konv); k = t['k']
        print(f"\n### Endwertkonvention {konv} = {k['buchwert']:.2f} EUR je Aktie")
        print(f"\n{'='*100}\n{firma}   Kurs {k['preis']:.2f} EUR   "
              f"Aktien {k['aktien']:.1f} Mio   Buchwert je Aktie {k['buchwert']:.2f} EUR")
        print(f"  Hurdle (Basis): {k['basis_hurdle']} = {t['rate']*100:.2f} %")
        for n, r in k['hurdles'].items():
            print(f"     Sensitivitaet: {n} = {r*100:.2f} %")
        print(f"\n  {'Szenario':52s} {'CF/Aktie':>9s} {'IRR 5J':>8s} {'10J':>8s} {'15J':>8s}")
        for name, cf, irrs in t['irr']:
            print(f"  {name:52s} {cf:8.2f}  " + "  ".join(f"{x:6.1f}%" for x in irrs))
        print(f"\n  {'Break-even-Preis':52s} {'5J':>9s} {'10J':>9s} {'15J':>9s}")
        for name, bes in t['be']:
            print(f"  {name:52s} " + "  ".join(
                ('    n.a.' if b is None else f"{b:7.2f}") for b in bes))
        print(f"\n  {'Erforderlicher Endwert (x Buchwert)':52s} {'5J':>9s} {'10J':>9s} {'15J':>9s}")
        for name, ews in t['ew']:
            print(f"  {name:52s} " + "  ".join(f"{e/k['buchwert']:6.1f}x " for e in ews))
