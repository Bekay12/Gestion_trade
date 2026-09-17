#!/usr/bin/env python3
"""
kennzahlen.py - erzeugt data/kennzahlen.tex, die einzige Zahlenquelle des
Dokuments.

Jeder Rohwert steht genau einmal in ROH, zusammen mit Quelle und gedruckter
Seite. Abgeleitete Groessen (Margen, Veraenderungsraten, Eigenkapitalrendite,
Buchwert je Aktie) werden hier gerechnet und nicht getippt; damit koennen Text
und Tabellen einander nicht widersprechen, und keine Rate entsteht aus
vorgerundeten Werten.

Die gedruckten Seitenzahlen stammen aus refs/*.txt, erzeugt von
scripts/edgar_seiten.py aus den EDGAR-Einreichungen (Form 40-F und 6-K).
scripts/pruefe_seiten.py prueft jeden Wert gegen den Text seiner Seite.

Roemische Endungen an den Namen der Mehrjahresreihe (UmsatzXXV = 2025) sind
keine Marotte: LaTeX-Makronamen duerfen keine Ziffern enthalten.

Aufruf: python3 scripts/kennzahlen.py
Ausgabe: data/kennzahlen.tex
"""
import csv
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Rohwerte. Betraege in Mio. USD, Unzen in Stueck, Kosten und Preise in USD je
# Unze, sofern der Name nichts anderes sagt.
# Schluessel -> (Wert, Quellenhinweis)
# --------------------------------------------------------------------------
ROH = {
    # === Teil 1: Unternehmen =============================================
    # --- Circular 2026 ---------------------------------------------------
    "AktienUmlauf":        (419965411, "Circular 2026, S. 5"),
    "VanEckAktien":        (42040749, "Circular 2026, S. 6"),
    "VanEckAnteil":        (10.01, "Circular 2026, S. 6"),
    "DirektorenGesamt":    (11, "Circular 2026, S. 70"),
    "DirektorenUnabh":     (10, "Circular 2026, S. 70"),

    # === Teil 2: Jahresabschluss 2025 ====================================
    # --- Konzern-GuV, Abschluss 2025, S. 7 -------------------------------
    "Umsatz":              (1808.8, "Abschluss 2025, S. 7"),
    "UmsatzVJ":            (1346.9, "Abschluss 2025, S. 7"),
    "Herstellkosten":      (809.5, "Abschluss 2025, S. 7"),
    "HerstellkostenVJ":    (751.1, "Abschluss 2025, S. 7"),
    "Abschreibung":        (209.7, "Abschluss 2025, S. 7"),
    "AbschreibungVJ":      (218.4, "Abschluss 2025, S. 7"),
    "Exploration":         (26.3, "Abschluss 2025, S. 7"),
    "Verwaltung":          (39.3, "Abschluss 2025, S. 7"),
    "AktienVerguetung":    (55.0, "Abschluss 2025, S. 7"),
    # Wertaufholung und Veraeusserungsgewinn sind die beiden Einmaleffekte,
    # die das Ergebnis 2025 tragen. Sie stehen hier einzeln, weil Teil 2 die
    # Eigenkapitalrendite mit und ohne sie ausweisen muss.
    "Wertaufholung":       (218.8, "Abschluss 2025, S. 7"),
    "WertaufholungVJ":     (57.1, "Abschluss 2025, S. 7"),
    "BetriebsErgebnis":    (1097.5, "Abschluss 2025, S. 7"),
    "BetriebsErgebnisVJ":  (561.9, "Abschluss 2025, S. 7"),
    "VeraeusserungsGewinn": (231.0, "Abschluss 2025, S. 7"),
    "DerivateVerlust":     (230.5, "Abschluss 2025, S. 7"),
    "DerivateVerlustVJ":   (24.2, "Abschluss 2025, S. 7"),
    "ErgebnisVorSteuern":  (1089.7, "Abschluss 2025, S. 7"),
    "ErgebnisVorSteuernVJ": (502.2, "Abschluss 2025, S. 7"),
    "SteuerLaufend":       (120.5, "Abschluss 2025, S. 7"),
    "SteuerLatent":        (83.4, "Abschluss 2025, S. 7"),
    "Konzernergebnis":     (885.8, "Abschluss 2025, S. 7"),
    "KonzernergebnisVJ":   (284.3, "Abschluss 2025, S. 7"),

    # --- Konzernbilanz, Abschluss 2025, S. 6 -----------------------------
    "Liquiditaet":         (623.1, "Abschluss 2025, S. 6"),
    "LiquiditaetVJ":       (327.2, "Abschluss 2025, S. 6"),
    "Umlaufvermoegen":     (1135.5, "Abschluss 2025, S. 6"),
    "Bergbauvermoegen":    (4957.5, "Abschluss 2025, S. 6"),
    "Bilanzsumme":         (6384.6, "Abschluss 2025, S. 6"),
    "BilanzsummeVJ":       (5336.1, "Abschluss 2025, S. 6"),
    "KurzfristigeSchulden": (567.6, "Abschluss 2025, S. 6"),
    # Achtung: "Total Liabilities" sind die GESAMTVERBINDLICHKEITEN und nicht
    # die Finanzschulden. Die Finanzschulden stehen eine Zeile darueber und
    # betragen ein Zehntel davon; die beiden zu verwechseln verkehrt die
    # Verschuldungslage des Unternehmens ins Gegenteil.
    "Gesamtverbindlichkeiten": (1938.8, "Abschluss 2025, S. 6"),
    "GesamtverbindlichkeitenVJ": (1751.9, "Abschluss 2025, S. 6"),
    "Finanzschulden":      (200.0, "Abschluss 2025, S. 6"),
    "FinanzschuldenVJ":    (250.0, "Abschluss 2025, S. 6"),
    "Leasing":             (11.2, "Abschluss 2025, S. 6"),
    "LeasingKurz":         (11.8, "Abschluss 2025, S. 6"),
    "Rekultivierung":      (153.4, "Abschluss 2025, S. 6"),
    "Eigenkapital":        (4445.8, "Abschluss 2025, S. 6"),
    "EigenkapitalVJ":      (3584.2, "Abschluss 2025, S. 6"),
    "Gewinnruecklage":     (217.2, "Abschluss 2025, S. 6"),

    # --- Kapitalflussrechnung, Abschluss 2025, S. 9 ----------------------
    "OpCashflow":          (795.3, "Abschluss 2025, S. 9"),
    "OpCashflowVJ":        (661.1, "Abschluss 2025, S. 9"),
    "Sachinvestitionen":   (507.1, "Abschluss 2025, S. 9"),
    "SachinvestitionenVJ": (417.6, "Abschluss 2025, S. 9"),
    "ZinsAktiviert":       (17.1, "Abschluss 2025, S. 9"),
    "VeraeusserungErloes": (160.0, "Abschluss 2025, S. 9"),
    "DividendeGezahlt":    (39.5, "Abschluss 2025, S. 9"),
    "DividendeGezahltVJ":  (35.1, "Abschluss 2025, S. 9"),
    "Rueckkauf":           (38.8, "Abschluss 2025, S. 9"),
    "TilgungFazilitaet":   (50.0, "Abschluss 2025, S. 9"),
    # Die vier Posten, die den Unterschied zwischen dem Flow-Modell und dem
    # tatsaechlichen freien Cash-Flow 2025 ausmachen. Sie stehen einzeln hier,
    # weil Abschnitt 5.0 das Modell gegen das Ist-Jahr eicht und die
    # Abweichung benennt, statt sie wegzurechnen.
    "HedgeAusbuchung":     (113.5, "Abschluss 2025, S. 9"),
    "GoldVorauszahlung":   (50.0, "Abschluss 2025, S. 9"),
    "UmlaufvermoegenSteuern": (129.0, "Abschluss 2025, S. 9"),

    # --- Anhang: Fazilitaet, Steuern, Dividende --------------------------
    "FazilitaetUnbeansprucht": (550.0, "Abschluss 2025, S. 31"),
    "Steuersatz":          (25.0, "Abschluss 2025, S. 34"),
    "DividendeErklaert":   (42.1, "Abschluss 2025, S. 37"),
    "DividendeQuartalNeu": (0.04, "Abschluss 2025, S. 37"),
    "DividendeErhoehung":  (60, "Abschluss 2025, S. 37"),

    # --- Kennzahlenseite MD&A 2025, S. 4 ---------------------------------
    "BereinigtesErgebnis":   (587.1, "MD&A 2025, S. 4"),
    "BereinigtesErgebnisVJ": (328.9, "MD&A 2025, S. 4"),
    "FreierCashflow":        (351.7, "MD&A 2025, S. 4"),
    "FreierCashflowVJ":      (272.3, "MD&A 2025, S. 4"),
    "Goldpreis":             (3372, "MD&A 2025, S. 4"),
    "GoldpreisVJ":           (2379, "MD&A 2025, S. 4"),
    "EPSbasic":              (2.11, "MD&A 2025, S. 4"),
    "EPSdil":                (2.10, "MD&A 2025, S. 4"),
    # Der Vorjahreswert der AISC steht im Bericht 2025 bei 1252 und im
    # Bericht 2024 bei 1281. Beide sind belegt; die Abweichung ist eine
    # Neuberechnung, kein Fehler, und wird in Teil 2 benannt.
    "AISCVJneu":             (1252, "MD&A 2025, S. 4"),
    "GoldVerkauft":          (531230, "MD&A 2025, S. 4"),
    "GoldVerkauftVJ":        (560234, "MD&A 2025, S. 4"),

    # --- Ausblick, MD&A 2025, S. 12 --------------------------------------
    "PrognoseProdUnten":   (570, "MD&A 2025, S. 12"),
    "PrognoseProdOben":    (650, "MD&A 2025, S. 12"),
    "PrognoseAiscUnten":   (1500, "MD&A 2025, S. 12"),
    "PrognoseAiscOben":    (1600, "MD&A 2025, S. 12"),
    "PrognoseCapexUnten":  (910, "MD&A 2025, S. 12"),
    "PrognoseCapexOben":   (1000, "MD&A 2025, S. 12"),
    "PrognoseWachstumsCapexUnten": (657, "MD&A 2025, S. 12"),
    "PrognoseWachstumsCapexOben":  (720, "MD&A 2025, S. 12"),
    "PrognoseErhaltCapexUnten":    (193, "MD&A 2025, S. 12"),
    "PrognoseErhaltCapexOben":     (220, "MD&A 2025, S. 12"),

    # --- Dreijahresprognose und Wachstumsprojekte, MD&A 2025, S. 7 -------
    # Diese Groessen sind vom Unternehmen veroeffentlicht und stehen damit
    # auf derselben Stufe wie die erzielten Goldpreise: Sie in die Rechnung
    # zu nehmen, ist keine eigene Prognose. Ohne sie belastet das Modell
    # fuenfzehn Jahre mit dem Wachstumskapital und erhaelt nie die Produktion,
    # die es kauft - ein Fehler, keine Vorsicht.
    "ProgAchtProdUnten":   (755, "MD&A 2025, S. 7"),
    "ProgAchtProdOben":    (835, "MD&A 2025, S. 7"),
    "ProgAchtProdDelta":   (46, "MD&A 2025, S. 7"),
    "ProgAchtAISCRueckgang": (18, "MD&A 2025, S. 7"),
    "ProgSechsProdDelta":  (12, "MD&A 2025, S. 7"),
    "Reserven":            (15.9, "MD&A 2025, S. 7"),
    "ReservenDelta":       (32, "MD&A 2025, S. 7"),
    "ReservenGehalt":      (1.87, "MD&A 2025, S. 7"),
    "ResourcenMI":         (5.5, "MD&A 2025, S. 7"),
    "IGDKapitalwert":      (12.2, "MD&A 2025, S. 7"),
    "IGDProduktion":       (534000, "MD&A 2025, S. 7"),
    "IGDAISC":             (1025, "MD&A 2025, S. 7"),
    "IGDGoldpreis":        (4500, "MD&A 2025, S. 7"),
    "SchachtTiefe":        (1350, "MD&A 2025, S. 7"),
    # Kleine Prozentangaben aus demselben Abschnitt. Der Wachhund liesse sie
    # als zweistellige Ganzzahlen durch; sie sind aber belegpflichtige Werte
    # aus einer Quelle und gehoeren deshalb hierher wie jeder andere.
    "SchachtAnteil":       (98, "MD&A 2025, S. 7"),
    "IGDReservenPlus":     (30, "MD&A 2025, S. 7"),
    "IGDZins":             (5, "MD&A 2025, S. 7"),
    "ReservenGehaltPlus":  (5, "MD&A 2025, S. 7"),
    "ProduktionRueckgangRund": (4, "MD&A 2025, S. 12"),
    "ReservenJahreFolge":  (7, "MD&A 2025, S. 13"),
    "LynnLakeProduktion":  (186000, "MD&A 2025, S. 13"),
    "LynnLakeAISC":        (829, "MD&A 2025, S. 13"),
    # --- Erhaltungskapital 2025, MD&A 2025, S. 41 ------------------------
    "ErhaltungsKapital":   (144.6, "MD&A 2025, S. 41"),
    "ErhaltungsKapitalVJ": (110.1, "MD&A 2025, S. 41"),
    "ExplorationAktiviert": (60, "MD&A 2025, S. 12"),

    # === Teil 3/4: Halbjahr 2026 =========================================
    # --- Kennzahlenseite MD&A Q2 2026, S. 4 ------------------------------
    "HJUmsatz":            (1190.8, "MD&A Q2 2026, S. 4"),
    "HJUmsatzVJ":          (771.2, "MD&A Q2 2026, S. 4"),
    "HJKonzernergebnis":   (461.8, "MD&A Q2 2026, S. 4"),
    "HJKonzernergebnisVJ": (174.6, "MD&A Q2 2026, S. 4"),
    "HJBereinigt":         (479.6, "MD&A Q2 2026, S. 4"),
    "HJBereinigtVJ":       (203.9, "MD&A Q2 2026, S. 4"),
    "HJFreierCashflow":    (245.2, "MD&A Q2 2026, S. 4"),
    "HJFreierCashflowVJ":  (64.5, "MD&A Q2 2026, S. 4"),
    "HJProduktion":        (254500, "MD&A Q2 2026, S. 4"),
    "HJProduktionVJ":      (262200, "MD&A Q2 2026, S. 4"),
    "HJGoldpreis":         (4660, "MD&A Q2 2026, S. 4"),
    "HJGoldpreisVJ":       (3027, "MD&A Q2 2026, S. 4"),
    "HJCashkosten":        (1268, "MD&A Q2 2026, S. 4"),
    "HJAISC":              (1793, "MD&A Q2 2026, S. 4"),
    "HJAISCVJ":            (1565, "MD&A Q2 2026, S. 4"),
    "HJOpCashflow":        (474.3, "MD&A Q2 2026, S. 4"),
    "HJOpCashflowVJ":      (279.1, "MD&A Q2 2026, S. 4"),

    # --- Prognose 2026, angehoben, MD&A Q2 2026, S. 6 --------------------
    "PrognoseCashUnten":   (1175, "MD&A Q2 2026, S. 6"),
    "PrognoseCashOben":    (1275, "MD&A Q2 2026, S. 6"),
    "PrognoseAiscUntenNeu": (1775, "MD&A Q2 2026, S. 6"),
    "PrognoseAiscObenNeu":  (1875, "MD&A Q2 2026, S. 6"),

    # --- Bilanz zum 30.06.2026, Abschluss Q2 2026, S. 2 ------------------
    "HJBilanzsumme":       (6712.9, "Abschluss Q2 2026, S. 2"),
    "HJEigenkapital":      (4804.9, "Abschluss Q2 2026, S. 2"),
    "HJLiquiditaet":       (636.9, "Abschluss Q2 2026, S. 2"),
    "HJGesamtverbindlichkeiten": (1908.0, "Abschluss Q2 2026, S. 2"),

    # === Mehrjahresreihe 2016-2025 =======================================
    # Jede Zahl stammt aus dem Bericht IHRES Jahres, nicht aus der
    # Vergleichsspalte eines spaeteren - sonst traegt die Reihe stillschweigend
    # spaetere Neuberechnungen. scripts/reihe.py prueft die Kette: Die
    # Vorjahresspalte eines Berichts muss dem Berichtsjahr des Vorberichts
    # gleichen. Von 45 solchen Gleichungen geht genau eine nicht auf, und
    # dieser eine Bruch ist ein Befund und kein Fehler (siehe Teil 2).
    # --- Umsatz, je aus dem MD&A des Berichtsjahres, S. 4 ---
    "UmsatzXVI":                 (482.2, "MD&A 2016, S. 4"),
    "UmsatzXVII":                (542.8, "MD&A 2017, S. 4"),
    "UmsatzXVIII":               (651.8, "MD&A 2018, S. 4"),
    "UmsatzXIX":                 (683.1, "MD&A 2019, S. 4"),
    "UmsatzXX":                  (748.1, "MD&A 2020, S. 4"),
    "UmsatzXXI":                 (823.6, "MD&A 2021, S. 4"),
    "UmsatzXXII":                (821.2, "MD&A 2022, S. 4"),
    "UmsatzXXIII":               (1023.3, "MD&A 2023, S. 4"),
    "UmsatzXXIV":                (1346.9, "MD&A 2024, S. 4"),
    "UmsatzXXV":                 (1808.8, "MD&A 2025, S. 4"),
    # --- OpCF, je aus dem MD&A des Berichtsjahres, S. 4 ---
    "OpCFXVI":                   (135.7, "MD&A 2016, S. 4"),
    "OpCFXVII":                  (163.5, "MD&A 2017, S. 4"),
    "OpCFXVIII":                 (213.9, "MD&A 2018, S. 4"),
    "OpCFXIX":                   (260.4, "MD&A 2019, S. 4"),
    "OpCFXX":                    (368.4, "MD&A 2020, S. 4"),
    "OpCFXXI":                   (356.5, "MD&A 2021, S. 4"),
    "OpCFXXII":                  (298.5, "MD&A 2022, S. 4"),
    "OpCFXXIII":                 (472.7, "MD&A 2023, S. 4"),
    "OpCFXXIV":                  (661.1, "MD&A 2024, S. 4"),
    "OpCFXXV":                   (795.3, "MD&A 2025, S. 4"),
    # --- Produktion, je aus dem MD&A des Berichtsjahres, S. 4 ---
    "ProduktionXVI":             (392000, "MD&A 2016, S. 4"),
    "ProduktionXVII":            (429400, "MD&A 2017, S. 4"),
    "ProduktionXVIII":           (505000, "MD&A 2018, S. 4"),
    "ProduktionXIX":             (494500, "MD&A 2019, S. 4"),
    "ProduktionXX":              (426800, "MD&A 2020, S. 4"),
    "ProduktionXXI":             (457200, "MD&A 2021, S. 4"),
    "ProduktionXXII":            (460400, "MD&A 2022, S. 4"),
    "ProduktionXXIII":           (529300, "MD&A 2023, S. 4"),
    "ProduktionXXIV":            (567000, "MD&A 2024, S. 4"),
    "ProduktionXXV":             (545400, "MD&A 2025, S. 4"),
    # --- AISC, je aus dem MD&A des Berichtsjahres, S. 4 ---
    "AISCXVI":                   (1010, "MD&A 2016, S. 4"),
    "AISCXVII":                  (933, "MD&A 2017, S. 4"),
    "AISCXVIII":                 (989, "MD&A 2018, S. 4"),
    "AISCXIX":                   (951, "MD&A 2019, S. 4"),
    "AISCXX":                    (1046, "MD&A 2020, S. 4"),
    "AISCXXI":                   (1135, "MD&A 2021, S. 4"),
    "AISCXXII":                  (1204, "MD&A 2022, S. 4"),
    "AISCXXIII":                 (1160, "MD&A 2023, S. 4"),
    "AISCXXIV":                  (1281, "MD&A 2024, S. 4"),
    "AISCXXV":                   (1524, "MD&A 2025, S. 4"),
    # --- Cashkosten, je aus dem MD&A des Berichtsjahres, S. 4 ---
    "CashkostenXVI":             (797, "MD&A 2016, S. 4"),
    "CashkostenXVII":            (770, "MD&A 2017, S. 4"),
    "CashkostenXVIII":           (802, "MD&A 2018, S. 4"),
    "CashkostenXIX":             (720, "MD&A 2019, S. 4"),
    "CashkostenXX":              (761, "MD&A 2020, S. 4"),
    "CashkostenXXI":             (794, "MD&A 2021, S. 4"),
    "CashkostenXXII":            (884, "MD&A 2022, S. 4"),
    "CashkostenXXIII":           (850, "MD&A 2023, S. 4"),
    "CashkostenXXIV":            (927, "MD&A 2024, S. 4"),
    "CashkostenXXV":             (1077, "MD&A 2025, S. 4"),
    # --- Sachinvestitionen, je aus dem Abschluss des Berichtsjahres ---
    "CapexXVI":                  (146.5, "Abschluss 2016, S. 8"),
    "CapexXVII":                 (162.5, "Abschluss 2017, S. 8"),
    "CapexXVIII":                (221.5, "Abschluss 2018, S. 8"),
    "CapexXIX":                  (263.6, "Abschluss 2019, S. 9"),
    "CapexXX":                   (246.1, "Abschluss 2020, S. 9"),
    "CapexXXI":                  (348.6, "Abschluss 2021, S. 9"),
    "CapexXXII":                 (313.7, "Abschluss 2022, S. 9"),
    "CapexXXIII":                (348.9, "Abschluss 2023, S. 9"),
    "CapexXXIV":                 (417.6, "Abschluss 2024, S. 9"),
    "CapexXXV":                  (507.1, "Abschluss 2025, S. 9"),
    # --- Eigenkapital zum Jahresende, je aus dem Abschluss ---
    "EigenkapitalXVI":            (1759.4, "Abschluss 2016, S. 5"),
    "EigenkapitalXVII":           (2681.2, "Abschluss 2017, S. 5"),
    "EigenkapitalXVIII":          (2602.3, "Abschluss 2018, S. 5"),
    "EigenkapitalXIX":            (2695.3, "Abschluss 2019, S. 6"),
    "EigenkapitalXX":             (2851.5, "Abschluss 2020, S. 6"),
    "EigenkapitalXXI":            (2735.6, "Abschluss 2021, S. 6"),
    "EigenkapitalXXII":           (2721.1, "Abschluss 2022, S. 6"),
    "EigenkapitalXXIII":          (2923.5, "Abschluss 2023, S. 6"),
    "EigenkapitalXXIV":           (3584.2, "Abschluss 2024, S. 6"),
    "EigenkapitalXXV":            (4445.8, "Abschluss 2025, S. 6"),
}

# --------------------------------------------------------------------------
# Sekundaerquellen. Getrennt gehalten, weil sie KEINE gedruckte Seite tragen
# und deshalb vom Wachhund in pruefe_seiten.py nicht belegt werden koennen.
# Die Trennung ist keine Formalie: Ein Leser muss sehen koennen, welche Zahl
# einer Pruefung gegen ein Primaerdokument standhielte und welche nicht.
# Schluessel -> (Wert, Quelle mit Abrufdatum)
# --------------------------------------------------------------------------
EXTERN = {
    # Analystenkonsens. Er steht in KEINEM Bericht des Unternehmens - kein
    # Emittent veroeffentlicht Ratings oder Kursziele ueber sich selbst -, ist
    # aber jederzeit beschaffbar. Ihn als "nicht verfuegbar" zu fuehren waere
    # eine unfertige Suche und keine benannte Luecke.
    #
    # Aktualitaetspruefung nach Regel 5: Ein Kursziel mit ausgewiesenem
    # Aufschlag verraet den Kurs, auf den es gerechnet wurde.
    #   46,25 / (1 + 0,3150) = 35,17
    # und 35,17 ist genau der auf derselben Seite genannte Schlusskurs vom
    # 15.09.2026. Der Konsens ist damit aktuell und nicht veraltet.
    "KonsensZahl":     (13, "stockanalysis.com/stocks/agi/forecast/, S&P Global, abgerufen 16.09.2026"),
    "KonsensZiel":     (46.25, "stockanalysis.com/stocks/agi/forecast/, abgerufen 16.09.2026"),
    "KonsensAufschlag": (31.50, "stockanalysis.com/stocks/agi/forecast/, abgerufen 16.09.2026"),
    "KonsensZielTief": (38.0, "stockanalysis.com/stocks/agi/forecast/, abgerufen 16.09.2026"),
    "KonsensZielHoch": (60.0, "stockanalysis.com/stocks/agi/forecast/, abgerufen 16.09.2026"),
    "KonsensKurs":     (35.17, "stockanalysis.com/stocks/agi/forecast/, Schlusskurs 15.09.2026"),
}

# --------------------------------------------------------------------------
# Abgeleitete Groessen. Jede wird gerechnet, keine getippt.
# (Makroname, Ausdruck) - der Ausdruck sieht die Rohwerte unter ihrem Namen.
# --------------------------------------------------------------------------
JAHRE = list(range(2016, 2026))
ROEMISCH = {2016: "XVI", 2017: "XVII", 2018: "XVIII", 2019: "XIX", 2020: "XX",
            2021: "XXI", 2022: "XXII", 2023: "XXIII", 2024: "XXIV", 2025: "XXV"}

ABGELEITET = [
    # --- Teil 2: Veraenderungsraten -------------------------------------
    ("UmsatzDelta", "(Umsatz / UmsatzVJ - 1) * 100"),
    ("ErgebnisDelta", "(Konzernergebnis / KonzernergebnisVJ - 1) * 100"),
    ("BereinigtDelta", "(BereinigtesErgebnis / BereinigtesErgebnisVJ - 1) * 100"),
    ("ProduktionDelta", "(ProduktionXXV / ProduktionXXIV - 1) * 100"),
    ("GoldpreisDelta", "(Goldpreis / GoldpreisVJ - 1) * 100"),
    ("AISCDelta", "(AISCXXV / AISCVJneu - 1) * 100"),
    ("OpCashflowDelta", "(OpCashflow / OpCashflowVJ - 1) * 100"),
    ("FreierCashflowDelta", "(FreierCashflow / FreierCashflowVJ - 1) * 100"),
    # Die AISC-Neuberechnung: Differenz zwischen dem, was der Bericht 2024
    # fuer 2024 auswies, und dem, was der Bericht 2025 als Vorjahr fuehrt.
    ("AISCNeuberechnung", "AISCXXIV - AISCVJneu"),

    # --- Teil 2: operatives Ergebnis vor Abschreibung --------------------
    # Das Betriebsergebnis 2025 enthaelt eine Wertaufholung, die kein
    # Zahlungsstrom ist. Sie wird herausgerechnet, damit die Marge das
    # operative Geschaeft misst und nicht eine Bilanzkorrektur.
    ("EBITDA", "BetriebsErgebnis + Abschreibung - Wertaufholung"),
    ("EBITDAVJ", "BetriebsErgebnisVJ + AbschreibungVJ - WertaufholungVJ"),
    ("EBITDAMarge", "(BetriebsErgebnis + Abschreibung - Wertaufholung) / Umsatz * 100"),
    ("EBITDAMargeVJ",
     "(BetriebsErgebnisVJ + AbschreibungVJ - WertaufholungVJ) / UmsatzVJ * 100"),

    # --- Teil 2: Bilanz und Rendite -------------------------------------
    # Nettoliquiditaet, nicht Nettoverschuldung: Die Finanzschulden liegen
    # unter dem Kassenbestand. Das Vorzeichen ist der Befund.
    ("Nettoliquiditaet", "Liquiditaet - Finanzschulden"),
    ("NettoliquiditaetVJ", "LiquiditaetVJ - FinanzschuldenVJ"),
    ("Eigenkapitalquote", "Eigenkapital / Bilanzsumme * 100"),
    ("EigenkapitalMittel", "(Eigenkapital + EigenkapitalVJ) / 2"),
    # Die Eigenkapitalrendite wird zweimal ausgewiesen. Der berichtete Wert
    # enthaelt die Wertaufholung und den Veraeusserungsgewinn; beide sind
    # einmalig. Die H\"urde des Anlageurteils ruht auf dem bereinigten Wert,
    # weil eine einmalige Bilanzkorrektur keinen Massstab fuer eine
    # Daueranlage abgibt.
    ("Eigenkapitalrendite", "Konzernergebnis / ((Eigenkapital + EigenkapitalVJ) / 2) * 100"),
    ("EigenkapitalrenditeBer",
     "BereinigtesErgebnis / ((Eigenkapital + EigenkapitalVJ) / 2) * 100"),
    ("BuchwertJeAktie", "Eigenkapital / (AktienUmlauf / 1e6)"),
    ("FreierCashflowJeAktie", "FreierCashflow / (AktienUmlauf / 1e6)"),
    ("ErgebnisJeAktieGerechnet", "Konzernergebnis / (AktienUmlauf / 1e6)"),
    ("AktienMio", "AktienUmlauf / 1e6"),

    # --- Teil 2: Investitionen ------------------------------------------
    ("InvestitionsQuote", "Sachinvestitionen / OpCashflow * 100"),
    ("PrognoseCapexMitte", "(PrognoseCapexUnten + PrognoseCapexOben) / 2"),
    ("PrognoseWachstumsCapexMitte",
     "(PrognoseWachstumsCapexUnten + PrognoseWachstumsCapexOben) / 2"),
    ("PrognoseErhaltCapexMitte",
     "(PrognoseErhaltCapexUnten + PrognoseErhaltCapexOben) / 2"),
    ("PrognoseProdMitte", "(PrognoseProdUnten + PrognoseProdOben) / 2"),
    ("PrognoseProdDelta", "((PrognoseProdUnten + PrognoseProdOben) / 2 * 1000 / ProduktionXXV - 1) * 100"),

    # --- Teil 3/4: Halbjahr 2026 ----------------------------------------
    ("HJUmsatzDelta", "(HJUmsatz / HJUmsatzVJ - 1) * 100"),
    ("HJProduktionDelta", "(HJProduktion / HJProduktionVJ - 1) * 100"),
    ("HJGoldpreisDelta", "(HJGoldpreis / HJGoldpreisVJ - 1) * 100"),
    ("HJAISCDelta", "(HJAISC / HJAISCVJ - 1) * 100"),
    ("HJFreierCashflowDelta", "(HJFreierCashflow / HJFreierCashflowVJ - 1) * 100"),
    ("HJNettoliquiditaet", "HJLiquiditaet - Finanzschulden"),
    # Die Marge je Unze ist bei einem Goldproduzenten der eigentliche
    # Fruehindikator: Sie trennt, was der Goldpreis beitraegt, von dem, was
    # das Unternehmen selbst steuert.
    ("MargeJeUnze", "Goldpreis - AISCXXV"),
    ("MargeJeUnzeVJ", "GoldpreisVJ - AISCVJneu"),
    ("HJMargeJeUnze", "HJGoldpreis - HJAISC"),
    ("HJMargeJeUnzeVJ", "HJGoldpreisVJ - HJAISCVJ"),
    ("MargeJeUnzeDelta", "((Goldpreis - AISCXXV) / (GoldpreisVJ - AISCVJneu) - 1) * 100"),
    ("HJMargeJeUnzeDelta",
     "((HJGoldpreis - HJAISC) / (HJGoldpreisVJ - HJAISCVJ) - 1) * 100"),

    # --- Teil 1: Analystenkonsens ---------------------------------------
    # Der aus Kursziel und Aufschlag zurueckgerechnete Bezugskurs. Weicht er
    # vom tatsaechlichen Kurs ab, ist das Ziel veraltet und faellt aus der
    # Betrachtung - es wird nicht eingemittelt.
    ("KonsensBezugskurs", "KonsensZiel / (1 + KonsensAufschlag / 100)"),
    ("KonsensAbstandKurs", "(KonsensZiel / KursSchlussXXVI - 1) * 100"),
    ("KonsensTiefAbstandKurs", "(KonsensZielTief / KursSchlussXXVI - 1) * 100"),

    # --- Teil 1: Eigentuemer --------------------------------------------
    ("VanEckGerechnet", "VanEckAktien / AktienUmlauf * 100"),
    ("StreubesitzRest", "100 - VanEckAnteil"),
    ("AnteilGesamt", "VanEckAnteil + (100 - VanEckAnteil)"),
    ("StreubesitzAktien", "AktienUmlauf - VanEckAktien"),
    ("DirektorenUnabhAnteil", "DirektorenUnabh / DirektorenGesamt * 100"),
]


def _reihe(praefix: str, werte: dict) -> list:
    """Gibt die Mehrjahresreihe [(Jahr, Wert)] zu einem Praefix zurueck."""
    return [(j, werte[praefix + ROEMISCH[j]]) for j in JAHRE
            if praefix + ROEMISCH[j] in werte]


def median(zahlen: list) -> float:
    """Median einer Liste; bei gerader Laenge das Mittel der beiden mittleren."""
    s = sorted(zahlen)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def perzentil(wert: float, zahlen: list) -> float:
    """
    --------------------------------------------------------------------------
    Purpose:
        Gibt an, an welcher Stelle der eigenen Geschichte ein Wert steht.
        Das ist die Angabe, die ein Niveaumodell braucht und die ihm sonst
        fehlt: Drei aus den eigenen Berichten entnommene Auspraegungen sind
        drei NIVEAUS, und ein Niveau sagt nichts darueber, wo im Zyklus es
        liegt. Am Zyklustief ist jedes davon niedrig, die Leiter ist
        niedrig, die Einstiegsschwelle ist niedrig - und das Urteil lautet
        "zu teuer" genau dann, wenn der Wert billig ist.

    Inputs:
        wert (float): der einzuordnende Wert
        zahlen (list): die Reihe, in die er eingeordnet wird

    Outputs:
        rang (float): Anteil der Reihe in Prozent, der unter dem Wert liegt
    --------------------------------------------------------------------------
    """
    if not zahlen:
        return 0.0
    return sum(1 for z in zahlen if z < wert) / len(zahlen) * 100


def _kurse() -> dict:
    """Liest die von scripts/fetch_kurs.py und vergleich_kurs.py erzeugten Reihen."""
    aus = {}
    pfad = os.path.join(ROOT, "data", "kurs_jahr.csv")
    with open(pfad, encoding="utf-8") as fh:
        for zeile in csv.DictReader(z for z in fh if not z.startswith("#")):
            j = int(zeile["jahr"])
            aus[f"KursSchluss{ROEMISCH.get(j, 'XXVI')}"] = float(zeile["schluss_usd"])
            aus[f"KursHoch{ROEMISCH.get(j, 'XXVI')}"] = float(zeile["hoch_usd"])
            aus[f"KursTief{ROEMISCH.get(j, 'XXVI')}"] = float(zeile["tief_usd"])
    pfad = os.path.join(ROOT, "data", "vergleich.csv")
    if os.path.exists(pfad):
        zeilen = [z for z in open(pfad, encoding="utf-8") if not z.startswith("#")]
        reihen = list(csv.DictReader(zeilen))
        hoch = max(reihen, key=lambda z: float(z["AGI"]))
        tief = min(reihen, key=lambda z: float(z["AGI"]))
        letzt = reihen[-1]
        for sym, name in (("AGI", "Aktie"), ("GLD", "Gold"), ("GDX", "Sektor")):
            h, t, e = float(hoch[sym]), float(tief[sym]), float(letzt[sym])
            aus[f"Einbruch{name}"] = (t / h - 1) * 100
            aus[f"Abstand{name}"] = (e / h - 1) * 100
        aus["EinbruchTagHoch"] = hoch["datum"]
        aus["EinbruchTagTief"] = tief["datum"]
        aus["KursStand"] = letzt["datum"]
    return aus


def main() -> None:
    werte = {k: v[0] for k, v in ROH.items()}
    werte.update({k: v[0] for k, v in EXTERN.items()})
    kurse = _kurse()
    werte.update({k: v for k, v in kurse.items() if isinstance(v, float)})

    zeilen = [
        "% ERZEUGT von scripts/kennzahlen.py - NICHT von Hand aendern.",
        "% Jeder Rohwert traegt seine Quelle und seine GEDRUCKTE Seite; jede",
        "% abgeleitete Groesse ist gerechnet und nicht getippt. Geprueft von",
        "% scripts/pruefe_seiten.py.",
        "",
        "% --- Rohwerte ------------------------------------------------------",
    ]
    for name, (wert, quelle) in ROH.items():
        w = f"{wert:.10g}"
        zeilen.append(f"\\newcommand{{\\{name}}}{{{w}}}% {quelle}")

    if EXTERN:
        zeilen += ["", "% --- Sekundaerquellen (ohne gedruckte Seite) ------------------------"]
        for name, (wert, quelle) in EXTERN.items():
            zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.10g}}}% {quelle}")

    zeilen += ["", "% --- Abgeleitete Groessen ------------------------------------------"]
    for name, ausdruck in ABGELEITET:
        wert = eval(ausdruck, {"__builtins__": {}}, werte)  # noqa: S307
        werte[name] = wert
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.10g}}}% = {ausdruck}")

    # --- Mehrjahresreihe: Median und Perzentil des juengsten Wertes -------
    zeilen += ["", "% --- Mehrjahresreihe 2016-2025: Lage des juengsten Wertes ----------"]
    for praefix, makro in (("FreierCashflowReihe", None), ("OpCF", "OpCF"),
                           ("Umsatz", "Umsatz"), ("Produktion", "Produktion"),
                           ("AISC", "AISC"), ("Capex", "Capex")):
        if makro is None:
            continue
        reihe = _reihe(praefix, werte)
        if len(reihe) < 5:
            continue
        zahlen = [w for _, w in reihe]
        m = median(zahlen)
        p = perzentil(zahlen[-1], zahlen)
        werte[f"{makro}Median"] = m
        werte[f"{makro}Perzentil"] = p
        zeilen.append(f"\\newcommand{{\\{makro}Median}}{{{m:.10g}}}"
                      f"% Median {reihe[0][0]}-{reihe[-1][0]}")
        zeilen.append(f"\\newcommand{{\\{makro}Perzentil}}{{{p:.10g}}}"
                      f"% Rang von {reihe[-1][0]} in der eigenen Reihe")

    # --- Freier Cash-Flow der Reihe: OpCF abzueglich Sachinvestitionen ----
    fcf = [(j, werte[f"OpCF{ROEMISCH[j]}"] - werte[f"Capex{ROEMISCH[j]}"])
           for j in JAHRE if f"OpCF{ROEMISCH[j]}" in werte and f"Capex{ROEMISCH[j]}" in werte]
    zeilen.append("")
    zeilen.append("% Freier Cash-Flow je Jahr, gerechnet aus operativem Cash-Flow")
    zeilen.append("% abzueglich Sachinvestitionen - beide belegt, die Differenz nicht")
    zeilen.append("% getippt. Die vom Unternehmen selbst ausgewiesene Groesse weicht")
    zeilen.append("% davon ab, weil sie in einzelnen Jahren anders abgegrenzt wurde;")
    zeilen.append("% die einheitliche Rechnung macht die zehn Jahre vergleichbar.")
    for j, w in fcf:
        zeilen.append(f"\\newcommand{{\\FCFReihe{ROEMISCH[j]}}}{{{w:.10g}}}")
        werte[f"FCFReihe{ROEMISCH[j]}"] = w
    zahlen = [w for _, w in fcf]
    werte["FCFMedian"] = median(zahlen)
    werte["FCFPerzentil"] = perzentil(zahlen[-1], zahlen)
    werte["FCFMin"] = min(zahlen)
    werte["FCFMax"] = max(zahlen)
    for k in ("FCFMedian", "FCFPerzentil", "FCFMin", "FCFMax"):
        zeilen.append(f"\\newcommand{{\\{k}}}{{{werte[k]:.10g}}}")

    # --- Kursreihe und Einbruch 2026 -------------------------------------
    zeilen += ["", "% --- Kursreihe (Nasdaq, Abrufdatum im Kopf von data/kurs_jahr.csv) -"]
    for name, wert in kurse.items():
        if isinstance(wert, float):
            zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.10g}}}")
        else:
            tag = "{}.{}.{}".format(*reversed(wert.split("-")))
            zeilen.append(f"\\newcommand{{\\{name}}}{{{tag}}}")

    # --- Kennzahlen, die den aktuellen Kurs brauchen ----------------------
    kurs = kurse["KursSchlussXXVI"]
    werte["Kurs"] = kurs
    nach = [
        ("Kurs", kurs),
        ("Marktkapitalisierung", kurs * werte["AktienMio"]),
        ("KursBuchwert", kurs / werte["BuchwertJeAktie"]),
        ("KursGewinn", kurs / werte["ErgebnisJeAktieGerechnet"]),
        ("Dividendenrendite", werte["DividendeQuartalNeu"] * 4 / kurs * 100),
        ("FCFRendite", werte["FreierCashflowJeAktie"] / kurs * 100),
    ]
    zeilen += ["", "% --- Bewertung zum aktuellen Kurs ----------------------------------"]
    for name, wert in nach:
        werte[name] = wert
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.10g}}}")

    # --- Tabellenkoerper der Zehnjahresreihe ------------------------------
    # Erzeugt und nicht getippt: Die Tabelle in Teil 2 fuehrt damit
    # zwangslaeufig dieselben Zahlen wie die Makros des Fliesstextes.
    def _d(x, stellen=0):
        return f"{x:,.{stellen}f}".replace(",", "@").replace(".", ",").replace("@", ".")

    korpus = []
    for j in JAHRE:
        r = ROEMISCH[j]
        korpus.append(
            f"{j} & {_d(werte['Umsatz' + r], 1)} & {_d(werte['Produktion' + r])} & "
            f"{_d(werte['AISC' + r])} & {_d(werte['OpCF' + r], 1)} & "
            f"{_d(werte['Capex' + r], 1)} & {_d(werte['FCFReihe' + r], 1)} & "
            f"{_d(werte['Eigenkapital' + r], 1)}\\\\")
    with open(os.path.join(ROOT, "data", "reihe_tabelle.tex"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(korpus) + "\n")

    # --- Tabellenkoerper der Kursreihe -------------------------------------
    korpus = []
    with open(os.path.join(ROOT, "data", "kurs_jahr.csv"), encoding="utf-8") as fh:
        for z in csv.DictReader(x for x in fh if not x.startswith("#")):
            korpus.append(f"{z['jahr']} & {_d(float(z['schluss_usd']), 2)} & "
                          f"{_d(float(z['hoch_usd']), 2)} & {_d(float(z['tief_usd']), 2)}\\\\")
    with open(os.path.join(ROOT, "data", "kurs_tabelle.tex"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(korpus) + "\n")

    # --- Lage des heutigen Kurses in der Spanne des laufenden Jahres -------
    zeilen += ["", "% --- Lage des heutigen Kurses ---------------------------------------"]
    for name, ausdruck in (
            ("KursAbstandTief", "(Kurs / KursTiefXXVI - 1) * 100"),
            ("KursAbstandHoch", "(Kurs / KursHochXXVI - 1) * 100"),
            ("KursSeitFuenfundzwanzig", "(Kurs / KursSchlussXXV - 1) * 100"),
            ("KursSeitEinundzwanzig", "(Kurs / KursSchlussXXI - 1) * 100"),
            # Der Teil des Einbruchs, den weder der Goldpreis noch der Sektor
            # erklaert - ausgeschrieben statt im Text in Worten genannt, damit
            # auch er aus der Zahlenschicht kommt.
            ("EinbruchUeberschuss", "EinbruchSektor - EinbruchAktie"),
            ("AbstandUeberschuss", "AbstandSektor - AbstandAktie")):
        wert = eval(ausdruck, {"__builtins__": {}}, werte)  # noqa: S307
        werte[name] = wert
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.10g}}}% = {ausdruck}")

    ziel = os.path.join(ROOT, "data", "kennzahlen.tex")
    os.makedirs(os.path.dirname(ziel), exist_ok=True)
    with open(ziel, "w", encoding="utf-8") as fh:
        fh.write("\n".join(zeilen) + "\n")
    print(f"[KENNZAHLEN] {len(ROH)} Rohwerte, {len(ABGELEITET)} abgeleitete, "
          f"-> data/kennzahlen.tex")


if __name__ == "__main__":
    main()
