#!/usr/bin/env python3
"""
Offline unit tests for cashflow_irr.py. Stdlib only, no network, no config file.

Run:
    python3 scripts/Test/test_cashflow_irr.py

The listed-equity tests below exist because the module was written for
investment projects, where the money spent is gone and only the option's own
uplift returns. A listed share is the other shape: the stake still has a value
at the end of the horizon, and that value dominates the result on any horizon a
reader cares about. Modelling it through `working_capital` (tied up at the
start, released at the end) happens to produce the right final flow, but it
also produces a wrong outflow at t=0 and it scales with the investment, which
silently corrupts `break_even_investment`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cashflow_irr as m           # noqa: E402

FEHLER = []


def pruefe(bedingung: bool, name: str, detail: str = "") -> None:
    if bedingung:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FEHLER.append(name)


def _aktien_cfg(preis: float = 49.43, cf: float = 4.22,
                endwert: float = 34.25, hurdle: float = 11.34) -> dict:
    """Listed share: one outflow at t=0, a constant distribution, an exit value."""
    return {
        "currency": "USD je Aktie",
        "opening_liquidity": 0.0,
        "horizon_start": 2026,
        "horizon_end": 2036,
        "schedule_years": [2026, 2027, 2028],
        "sensitivity_horizons": [2031, 2036, 2041],
        "revenue_base": 1.0,
        "earnings_base": 1.0,
        "hurdle": hurdle,
        "hurdle_source": "return on equity 2025 excl. FX gain, annual report p. 17 / p. 192",
        "options": {
            "K": {
                "name": "Kauf zum heutigen Kurs",
                "investment": {"2026": preis},
                "earnings_uplift": cf,
                "uplift_start": 2027,
                "terminal_value": endwert,
            }
        },
    }


def test_endwert_faellt_nur_im_letzten_jahr_an() -> None:
    cfg = _aktien_cfg()
    flows = m.full_cashflows(cfg, "K")
    pruefe(abs(flows[0] - (-49.43)) < 1e-9, "Endwert veraendert t=0 nicht", f"{flows[0]}")
    pruefe(abs(flows[1] - 4.22) < 1e-9, "Endwert veraendert Zwischenjahre nicht", f"{flows[1]}")
    pruefe(abs(flows[-1] - (4.22 + 34.25)) < 1e-9,
           "Endwert liegt im letzten Jahr auf der Ausschuettung", f"{flows[-1]}")


def test_endwert_skaliert_nicht_mit_der_investition() -> None:
    """Der Kern: der Ausstiegswert haengt nicht am Einstiegspreis.

    Skaliert er mit, findet die Bisektion den Preis, bei dem BEIDE Seiten
    wachsen, und der Break-even-Preis ist beliebig falsch.
    """
    cfg = _aktien_cfg()
    einfach = m.full_cashflows(cfg, "K", investment_scale=1.0)
    doppelt = m.full_cashflows(cfg, "K", investment_scale=2.0)
    pruefe(abs(doppelt[0] - 2 * einfach[0]) < 1e-9,
           "Investition skaliert", f"{doppelt[0]}")
    pruefe(abs(doppelt[-1] - einfach[-1]) < 1e-9,
           "Endwert skaliert NICHT", f"{doppelt[-1]} vs {einfach[-1]}")


def test_break_even_preis_erreicht_den_hurdle_exakt() -> None:
    """Die Probe aufs Exempel: den gefundenen Preis einsetzen und den IRR messen."""
    cfg = _aktien_cfg()
    preis = m.break_even_investment(cfg, "K", 2036)
    pruefe(preis is not None, "Break-even-Preis existiert")
    if preis is None:
        return
    probe = _aktien_cfg(preis=preis)
    gemessen = m.irr(m.full_cashflows(probe, "K", 2036))
    pruefe(abs(gemessen - 0.1134) < 1e-4,
           "IRR beim Break-even-Preis trifft den Hurdle",
           f"IRR {gemessen:.6f} bei Preis {preis:.4f}")
    pruefe(preis < 49.43,
           "Break-even liegt unter dem heutigen Kurs", f"{preis:.2f}")


def test_reihe_ohne_vorzeichenwechsel_gilt_nicht_als_unter_hurdle() -> None:
    """Regression: die Klammer von break_even_investment war einseitig.

    Bei verschwindender Investition hat die Reihe keinen Vorzeichenwechsel mehr,
    weil nur noch Zufluesse uebrig bleiben. irr() wirft dann, und der Abfang
    deutete das als 'unter jedem positiven Hurdle' - das genaue Gegenteil: eine
    Reihe aus lauter Zufluessen hat eine unendliche Rendite. Fuer Projekte fiel
    das nie auf, weil dort ohne Investition auch kein Ertrag entsteht. Beim
    Aktienfall bleiben Ausschuettung und Endwert stehen, und die Funktion gab
    None zurueck statt eines Preises.
    """
    cfg = _aktien_cfg()
    winzig = m.full_cashflows(cfg, "K", 2036, investment_scale=1e-6)
    pruefe(all(f > 0 for f in winzig[1:]) and winzig[0] > -1e-3,
           "Aufbau: Reihe ist bei verschwindender Investition praktisch reine Zuflussreihe")
    pruefe(m.break_even_investment(cfg, "K", 2036) is not None,
           "Break-even wird gefunden statt als 'unter Hurdle' verworfen")


def test_erforderlicher_endwert_ist_die_umkehrung() -> None:
    """Gegenprobe: welchen Ausstiegswert setzt der HEUTIGE Preis voraus?

    Der zurueckgegebene Wert, als terminal_value eingesetzt, muss den Hurdle
    exakt treffen. Das ist die Frage, die ein Leser tatsaechlich stellt, wenn
    der Break-even-Preis weit unter dem Kurs liegt.
    """
    cfg = _aktien_cfg()
    noetig = m.required_terminal_value(cfg, "K", 2036)
    pruefe(noetig > 34.25, "Erforderlicher Endwert liegt ueber dem angesetzten",
           f"{noetig:.2f}")
    probe = _aktien_cfg(endwert=noetig)
    gemessen = m.irr(m.full_cashflows(probe, "K", 2036))
    pruefe(abs(gemessen - 0.1134) < 1e-4,
           "IRR mit erforderlichem Endwert trifft den Hurdle",
           f"IRR {gemessen:.6f}")


def test_konfiguration_ohne_endwert_bleibt_unveraendert() -> None:
    """Rueckwaertskompatibilitaet: bestehende Konfigurationen duerfen sich nicht bewegen."""
    cfg = dict(m.EXAMPLE)
    erwartet = [
        m.uplift(cfg["options"]["A"], j)
        - m._year_amount(cfg["options"]["A"].get("working_capital", {}), j)
        - m._year_amount(cfg["options"]["A"].get("investment", {}), j)
        for j in range(cfg["horizon_start"], cfg["horizon_end"] + 1)
    ]
    erwartet[-1] += sum(float(v) for v in cfg["options"]["A"].get("working_capital", {}).values())
    tatsaechlich = m.full_cashflows(cfg, "A")
    pruefe(all(abs(a - b) < 1e-9 for a, b in zip(erwartet, tatsaechlich)),
           "Reihe ohne terminal_value unveraendert")


def test_endwert_taucht_in_der_tabelle_auf() -> None:
    """Was in die Rechnung eingeht, muss auch in der gerenderten Tabelle stehen."""
    cfg = _aktien_cfg()
    cfg["schedule_years"] = [2026, 2027, 2036]
    zeilen = m.render_schedule(cfg, "K", "markdown")
    pruefe("34.2" in zeilen or "34,2" in zeilen,
           "Endwert erscheint in der Zahlungsreihe", zeilen.replace("\n", " | ")[:200])


def test_erforderliches_wachstum_ist_die_zweite_umkehrung() -> None:
    """Der Preis wird in die Wachstumsrate uebersetzt, die er voraussetzt.

    break_even_investment() beantwortet "bis zu welchem Preis", und das ist die
    halbe Frage. Steht der Kurs weit ueber dieser Schwelle, will der Leser
    wissen, WAS der Markt annimmt - nicht nur, dass er mehr annimmt als man
    selbst. Die Wachstumsrate ist die ehrlichste Form dieser Antwort, weil sie
    gegen die eigene Historie des Unternehmens pruefbar ist, waehrend eine
    Prognose des Verfassers es nicht waere.
    """
    cfg = _aktien_cfg()
    g = m.required_growth(cfg, "K", 2036)
    pruefe(g is not None, "Wachstumsrate existiert")
    if g is None:
        return

    # Probe: mit dieser Rate wachsende Ausschuettungen treffen den Hurdle exakt.
    flows = m.full_cashflows(cfg, "K", 2036)
    basis = cfg["options"]["K"]["earnings_uplift"]
    gewachsen = [flows[0]] + [
        basis * (1 + g) ** t for t in range(1, 2036 - 2026 + 1)]
    gewachsen[-1] += cfg["options"]["K"]["terminal_value"]
    pruefe(abs(m.irr(gewachsen) - 0.1134) < 1e-4,
           "IRR mit erforderlichem Wachstum trifft den Hurdle",
           f"IRR {m.irr(gewachsen):.6f} bei g={g:.4f}")


def test_erforderliches_wachstum_faellt_mit_dem_preis() -> None:
    """Richtungsprobe: ein billigerer Einstieg verlangt weniger Wachstum.

    Ohne diese Probe koennte ein Vorzeichenfehler in der Bisektion unbemerkt
    bleiben - die Rate waere plausibel, aber falsch herum.
    """
    teuer = m.required_growth(_aktien_cfg(preis=60.0), "K", 2036)
    billig = m.required_growth(_aktien_cfg(preis=30.0), "K", 2036)
    pruefe(teuer is not None and billig is not None, "beide Raten existieren")
    if teuer is None or billig is None:
        return
    pruefe(billig < teuer, "billiger Einstieg verlangt weniger Wachstum",
           f"{billig:.4f} gegen {teuer:.4f}")


def test_wachstum_entfaellt_wenn_der_preis_den_hurdle_schon_traegt() -> None:
    """Deckt der Zahlungsstrom den Hurdle bereits, ist die Rate negativ oder null.

    Die Funktion darf dann nicht None liefern: 'kein Wachstum noetig' ist eine
    Aussage, kein Fehlschlag, und der Bericht soll sie treffen koennen.
    """
    g = m.required_growth(_aktien_cfg(preis=12.0), "K", 2036)
    pruefe(g is not None and g <= 0,
           "sehr billiger Einstieg verlangt kein Wachstum", f"{g}")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            print(name)
            fn()
    print()
    if FEHLER:
        print(f"{len(FEHLER)} Test(s) fehlgeschlagen: {', '.join(FEHLER)}")
        raise SystemExit(1)
    print("alle Tests bestanden")
