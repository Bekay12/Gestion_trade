#!/usr/bin/env python3
"""
kennzahlen.py - erzeugt data/kennzahlen.tex, die einzige Zahlenquelle des
Dokuments.

Jeder Rohwert steht genau einmal in ROH, zusammen mit Quelle und gedruckter
Seite. Abgeleitete Groessen (Margen, Veraenderungsraten, Verschuldungsgrad)
werden hier gerechnet und nicht getippt; damit koennen Text und Tabellen
einander nicht widersprechen und keine Rate entsteht aus vorgerundeten
Werten.

Die gedruckten Seitenzahlen stammen aus refs/*.txt, erzeugt von
scripts/edgar_pages.py aus den EDGAR-Einreichungen.

Aufruf: python3 scripts/kennzahlen.py
Ausgabe: data/kennzahlen.tex
"""
import csv
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# Rohwerte. Betraege in Mio. USD, sofern nicht anders benannt.
# Schluessel -> (Wert, Quellenhinweis)
# --------------------------------------------------------------------------
ROH = {
    # --- Konzern-GuV, Form 10-K 2025, S. 49 -----------------------------
    "UmsatzGesamt":        (5902, "10-K 2025, S. 49"),
    "UmsatzGesamtVJ":      (4805, "10-K 2025, S. 49"),
    "UmsatzGesamtVVJ":     (4257, "10-K 2025, S. 49"),
    "ProvisionsErtraege":  (5763, "10-K 2025, S. 49"),
    "ProvisionsErtraegeVJ": (4705, "10-K 2025, S. 49"),
    "InvestErtrag":        (139, "10-K 2025, S. 49"),
    "InvestErtragVJ":      (100, "10-K 2025, S. 49"),
    "Personalaufwand":     (2935, "10-K 2025, S. 49"),
    "PersonalaufwandVJ":   (2406, "10-K 2025, S. 49"),
    "SonstBetrAufwand":    (959, "10-K 2025, S. 49"),
    "SonstBetrAufwandVJ":  (710, "10-K 2025, S. 49"),
    "AbschrImmat":         (312, "10-K 2025, S. 49"),
    "AbschrImmatVJ":       (178, "10-K 2025, S. 49"),
    "AbschrSach":          (55, "10-K 2025, S. 49"),
    "AbschrSachVJ":        (44, "10-K 2025, S. 49"),
    "Zinsaufwand":         (297, "10-K 2025, S. 49"),
    "ZinsaufwandVJ":       (193, "10-K 2025, S. 49"),
    "ErgebnisVorSteuern":  (1371, "10-K 2025, S. 49"),
    "ErgebnisVorSteuernVJ": (1303, "10-K 2025, S. 49"),
    "Ertragsteuern":       (304, "10-K 2025, S. 49"),
    "ErtragsteuernVJ":     (301, "10-K 2025, S. 49"),
    "Konzernergebnis":     (1054, "10-K 2025, S. 49"),
    "KonzernergebnisVJ":   (993, "10-K 2025, S. 49"),
    "KonzernergebnisVVJ":  (871, "10-K 2025, S. 49"),
    "EPSbasic":            (3.37, "10-K 2025, S. 49"),
    "EPSbasicVJ":          (3.48, "10-K 2025, S. 49"),
    "EPSdil":              (3.16, "10-K 2025, S. 49"),
    "EPSdilVJ":            (3.46, "10-K 2025, S. 49"),

    # --- MD&A, Form 10-K 2025, S. 36 ------------------------------------
    "Kernprovisionen":     (5508, "10-K 2025, S. 36"),
    "KernprovisionenVJ":   (4539, "10-K 2025, S. 36"),
    "Gewinnbeteiligung":   (255, "10-K 2025, S. 36"),
    "GewinnbeteiligungVJ": (166, "10-K 2025, S. 36"),
    "OrganischWachstum":   (2.8, "10-K 2025, S. 36 (in Prozent)"),
    "OrganischWachstumVJ": (10.4, "10-K 2025, S. 36 (in Prozent)"),
    "OrganischNeugeschaeft": (126, "10-K 2025, S. 36"),
    "UmsatzAusZukaeufenMDA": (836, "10-K 2025, S. 36"),
    "WaehrungseffektMDA":  (18, "10-K 2025, S. 36"),
    "DesinvestitionMDA":   (11, "10-K 2025, S. 36"),

    # --- Ueberleitung EBITDAC, Form 10-K 2025, S. 39 (2025) und S. 40 (2024)
    "EBITDAC":             (2060, "10-K 2025, S. 39"),
    "EBITDACVJ":           (1720, "10-K 2025, S. 40"),
    "EBITDACber":          (2121, "10-K 2025, S. 39"),
    "EBITDACberVJ":        (1689, "10-K 2025, S. 40"),
    "AkqIntegrationskosten": (113, "10-K 2025, S. 39"),
    "EscrowBewertung":     (-54, "10-K 2025, S. 39"),
    "VeraeusserungsergebnisVJ": (-31, "10-K 2025, S. 40"),
    "RetailUmsatz":        (3406, "10-K 2025, S. 39"),
    "RetailUmsatzVJ":      (2729, "10-K 2025, S. 40"),
    "RetailEBITDACber":    (1022, "10-K 2025, S. 39"),
    "RetailEBITDACberVJ":  (818, "10-K 2025, S. 40"),
    "SpezUmsatz":          (2409, "10-K 2025, S. 39"),
    "SpezUmsatzVJ":        (2016, "10-K 2025, S. 40"),
    "SpezEBITDACber":      (1038, "10-K 2025, S. 39"),
    "SpezEBITDACberVJ":    (862, "10-K 2025, S. 40"),
    "RetailOrganisch":     (2.8, "10-K 2025, S. 41 (in Prozent)"),
    "SpezOrganisch":       (2.8, "10-K 2025, S. 42 (in Prozent)"),
    "SpezOrganischVJ":     (17.8, "10-K 2025, S. 42 (in Prozent)"),
    "RetailOrganischVJ":   (5.8, "10-K 2025, S. 41 (in Prozent)"),

    # --- Konzernbilanz, Form 10-K 2025, S. 51 ---------------------------
    "Zahlungsmittel":      (1079, "10-K 2025, S. 51"),
    "ZahlungsmittelVJ":    (675, "10-K 2025, S. 51"),
    "TreuhandMittel":      (2471, "10-K 2025, S. 51"),
    "Goodwill":            (15087, "10-K 2025, S. 51"),
    "GoodwillVJ":          (7970, "10-K 2025, S. 51"),
    "ImmatVermoegen":      (4906, "10-K 2025, S. 51"),
    "ImmatVermoegenVJ":    (1814, "10-K 2025, S. 51"),
    "Bilanzsumme":         (29991, "10-K 2025, S. 51"),
    "BilanzsummeVJ":       (17612, "10-K 2025, S. 51"),
    "Eigenkapital":        (12573, "10-K 2025, S. 51"),
    "EigenkapitalVJ":      (6437, "10-K 2025, S. 51"),
    "AktienUmlauf":        (336, "10-K 2025, S. 51 (Mio. Stueck)"),
    "AktienUmlaufVJ":      (286, "10-K 2025, S. 51 (Mio. Stueck)"),

    # --- Kapitalflussrechnung, Form 10-K 2025, S. 53 --------------------
    "OperativerCF":        (1450, "10-K 2025, S. 53"),
    "OperativerCFVJ":      (1174, "10-K 2025, S. 53"),
    "OperativerCFVVJ":     (1010, "10-K 2025, S. 53"),
    "Sachinvestitionen":   (68, "10-K 2025, S. 53"),
    "SachinvestitionenVJ": (82, "10-K 2025, S. 53"),
    "UnternehmenskaeufeCF": (7854, "10-K 2025, S. 53"),
    "UnternehmenskaeufeCFVJ": (890, "10-K 2025, S. 53"),
    "Dividendenzahlung":   (193, "10-K 2025, S. 53"),
    "DividendenzahlungVJ": (154, "10-K 2025, S. 53"),
    "EarnoutZahlung":      (143, "10-K 2025, S. 53"),
    "EarnoutZahlungVJ":    (117, "10-K 2025, S. 53"),
    "AktienrueckkaufCF":   (100, "10-K 2025, S. 53"),

    # --- Finanzverbindlichkeiten, Form 10-K 2025, S. 67 -----------------
    "Gesamtverschuldung":  (7613, "10-K 2025, S. 67"),
    "GesamtverschuldungVJ": (3824, "10-K 2025, S. 67"),
    "Anleihen":            (6650, "10-K 2025, S. 67"),
    "AnleihenVJ":          (2850, "10-K 2025, S. 67"),
    "Bankkredite":         (313, "10-K 2025, S. 67"),

    # --- Vertragliche Verpflichtungen, Form 10-K 2025, S. 46 ------------
    "TilgungGesamt":       (7682, "10-K 2025, S. 46"),
    "TilgungJahrEins":     (719, "10-K 2025, S. 46"),
    "TilgungJahrZweiDrei": (813, "10-K 2025, S. 46"),
    "TilgungJahrVierFuenf": (1150, "10-K 2025, S. 46"),
    "TilgungDanach":       (5000, "10-K 2025, S. 46"),
    "ZinsverpflichtungGesamt": (4181, "10-K 2025, S. 46"),
    "ZinsverpflichtungJahrEins": (376, "10-K 2025, S. 46"),
    "ZinsverpflichtungZweiDrei": (649, "10-K 2025, S. 46"),
    "ZinsverpflichtungVierFuenf": (561, "10-K 2025, S. 46"),
    "ZinsverpflichtungDanach": (2595, "10-K 2025, S. 46"),
    "EarnoutMaximum":      (842, "10-K 2025, S. 46"),
    "EarnoutBilanziert":   (541, "10-K 2025, S. 46"),
    "EscrowVerbindlichkeit": (616, "10-K 2025, S. 46"),

    # --- Unternehmenszusammenschluesse, Form 10-K 2025, S. 63 -----------
    "AccessionKaufpreis":  (9608, "10-K 2025, S. 63"),
    "AccessionBarzahlung": (8293, "10-K 2025, S. 63"),
    "AccessionAktien":     (613, "10-K 2025, S. 63"),
    "AccessionGoodwill":   (6547, "10-K 2025, S. 63"),
    "AccessionImmat":      (3221, "10-K 2025, S. 63"),
    "KaeufeKaufpreis":     (10112, "10-K 2025, S. 63"),
    "KaeufeGoodwill":      (6885, "10-K 2025, S. 63"),
    "UmsatzAusKaeufen":    (747, "10-K 2025, S. 63"),
    "ErgebnisAusKaeufen":  (77, "10-K 2025, S. 63"),
    "NutzungsdauerKunden": (14, "10-K 2025, S. 63 (Jahre)"),

    # --- Pro-forma-Angaben, Form 10-K 2025, S. 64 -----------------------
    "ProFormaUmsatz":      (6947, "10-K 2025, S. 64"),
    "ProFormaUmsatzVJ":    (6637, "10-K 2025, S. 64"),
    "ProFormaErgebnis":    (1220, "10-K 2025, S. 64"),
    "ProFormaErgebnisVJ":  (1080, "10-K 2025, S. 64"),
    "ProFormaEPSdil":      (3.45, "10-K 2025, S. 64"),
    "ProFormaEPSdilVJ":    (3.20, "10-K 2025, S. 64"),

    # --- Finanzierung der Uebernahme, Form 10-K 2025, S. 45 und S. 46 ---
    "AnleihevolumenNeu":   (4200, "10-K 2025, S. 45"),
    "KuponMin":            (4.600, "10-K 2025, S. 45 (in Prozent)"),
    "KuponMax":            (6.250, "10-K 2025, S. 45 (in Prozent)"),
    "AnleiheTranchen":     (6, "10-K 2025, S. 45"),
    "RueckkaufAufstockung": (1251, "10-K 2025, S. 28"),
    "KapitalerhoehungStueck": (43137254, "10-K 2025, S. 46 (Stueck)"),
    "KapitalerhoehungKurs": (102.00, "10-K 2025, S. 46 (USD je Aktie)"),
    "KapitalerhoehungNetto": (4315, "10-K 2025, S. 46"),
    "VerkaeuferaktienWert": (1045, "10-K 2025, S. 46"),
    "VerkaeuferaktienKurs": (110.57, "10-K 2025, S. 46 (USD je Aktie)"),
    "RueckkaufRahmen":     (1400, "10-K 2025, S. 28 (Restrahmen 31.12.2025)"),
    "RueckkaufStueck":     (1255970, "10-K 2025, S. 28 (Stueck 2025)"),
    "RueckkaufKurs":       (79.62, "10-K 2025, S. 28 (USD je Aktie)"),

    # --- Beschaeftigte, Form 10-K 2025, S. 12 und S. 8 ------------------
    "Mitarbeiter":         (22888, "10-K 2025, S. 7"),
    "MitarbeiterRetail":   (14531, "10-K 2025, S. 6"),
    "MitarbeiterSpez":     (7905, "10-K 2025, S. 7"),
    "AkquisitionenAnzahl": (43, "10-K 2025, S. 8"),
    "AkquisitionenKoepfe": (5794, "10-K 2025, S. 8"),

    # --- Aktie und Aktionaere -------------------------------------------
    "AktienRegisterStueck": (340420023, "10-K 2025, S. 28 (Stand 10.02.2026)"),
    "AktionaereRegister":  (2446, "10-K 2025, S. 28"),
    "HyattBrownStueck":    (35997546, "Proxy Statement 2026, S. 76 (Stueck)"),
    "HyattBrownAnteil":    (10.60, "Proxy Statement 2026, S. 76 (in Prozent)"),
    "PowellBrownStueck":   (5320524, "Proxy Statement 2026, S. 76 (Stueck)"),
    "PowellBrownAnteil":   (1.57, "Proxy Statement 2026, S. 76 (in Prozent)"),
    "OrganeStueck":        (44574701, "Proxy Statement 2026, S. 76 (Stueck)"),
    "OrganeAnteil":        (13.13, "Proxy Statement 2026, S. 76 (in Prozent)"),
    "OrganePersonen":      (21, "Proxy Statement 2026, S. 76"),
    "VanguardStueck":      (37330892, "Proxy Statement 2026, S. 76 (Stueck)"),
    "VanguardAnteil":      (10.99, "Proxy Statement 2026, S. 76 (in Prozent)"),
    "CapitalWorldStueck":  (17503659, "Proxy Statement 2026, S. 76 (Stueck)"),
    "CapitalWorldAnteil":  (5.15, "Proxy Statement 2026, S. 76 (in Prozent)"),
    "AnalystenHaeuser":    (18, "Investor-Relations-Seite, Analyst Coverage"),
    "AufsichtsratMitglieder": (14, "Proxy Statement 2026, S. 76"),
    "VorstandsmitgliederAnzahl": (9, "10-K 2025, S. 10"),

    # --- Dividende je Aktie (USD) ---------------------------------------
    "DividendeJeAktie":    (0.62, "Investor-Relations-Seite, Dividend History"),
    "DividendeJeAktieVJ":  (0.54, "Investor-Relations-Seite, Dividend History"),
    "DividendeJeAktieVVJ": (0.48, "Investor-Relations-Seite, Dividend History"),
    "DividendeQuartalNeu": (0.165, "10-K 2025, S. 44 (Beschluss vom 21.01.2026)"),

    # --- Halbjahr 2026, Form 10-Q Q2 2026, S. 29 und S. 7 ---------------
    "HJUmsatz":            (3577, "10-Q Q2 2026, S. 29"),
    "HJUmsatzVJ":          (2689, "10-Q Q2 2026, S. 29"),
    "HJErgebnis":          (714, "10-Q Q2 2026, S. 29"),
    "HJErgebnisVJ":        (563, "10-Q Q2 2026, S. 29"),
    "HJEBITDACber":        (1329, "10-Q Q2 2026, S. 29"),
    "HJEBITDACberMarge":   (37.2, "10-Q Q2 2026, S. 29 (in Prozent)"),
    "HJOrganisch":         (-0.3, "10-Q Q2 2026, S. 29 (in Prozent)"),
    "HJOrganischVJ":       (5.1, "10-Q Q2 2026, S. 29 (in Prozent)"),
    "QzweiOrganisch":      (-0.7, "10-Q Q2 2026, S. 29 (in Prozent)"),
    "QzweiOrganischVJ":    (3.6, "10-Q Q2 2026, S. 29 (in Prozent)"),
    "QzweiZinsaufwand":    (100, "10-Q Q2 2026, S. 29"),
    "HJAbschrImmat":       (226, "10-Q Q2 2026, S. 29"),
    "HJAbschrSach":        (35, "10-Q Q2 2026, S. 29"),
    "HJZinsaufwand":       (199, "10-Q Q2 2026, S. 29"),
    "HJOperativerCF":      (608, "10-Q Q2 2026, S. 9"),
    "HJSachinvestitionen": (38, "10-Q Q2 2026, S. 9"),
    "HJDividendenzahlung": (112, "10-Q Q2 2026, S. 9"),
    "HJRueckkauf":         (500, "10-Q Q2 2026, S. 9"),
    "HJEarnoutZahlung":    (184, "10-Q Q2 2026, S. 9"),
    "HJZahlungsmittel":    (918, "10-Q Q2 2026, S. 7"),
    "HJFKkurz":            (413, "10-Q Q2 2026, S. 7"),
    "HJFKlang":            (7346, "10-Q Q2 2026, S. 7"),
    "HJEigenkapital":      (12608, "10-Q Q2 2026, S. 7"),
    "HJAktienUmlauf":      (330, "10-Q Q2 2026, S. 7 (Mio. Stueck)"),
    "HJEigeneAktien":      (1348, "10-Q Q2 2026, S. 7"),
    "HJEigeneAktienVJ":    (848, "10-Q Q2 2026, S. 7"),
}

# Abgeleitete Groessen: Schluessel -> (Ausdruck ueber ROH, Erlaeuterung)
ABGELEITET = [
    ("UmsatzDelta", "(UmsatzGesamt/UmsatzGesamtVJ-1)*100",
     "Umsatzwachstum 2025 gegenueber 2024 in Prozent"),
    ("ProvisionsDelta", "(ProvisionsErtraege/ProvisionsErtraegeVJ-1)*100",
     "Wachstum der Provisionsertraege in Prozent"),
    ("KernprovisionenDelta", "(Kernprovisionen/KernprovisionenVJ-1)*100",
     "Wachstum der Kernprovisionen in Prozent"),
    ("EBITDACMarge", "EBITDAC/UmsatzGesamt*100",
     "berichtete EBITDAC-Marge 2025 in Prozent"),
    ("EBITDACMargeVJ", "EBITDACVJ/UmsatzGesamtVJ*100",
     "berichtete EBITDAC-Marge 2024 in Prozent"),
    ("EBITDACberMarge", "EBITDACber/UmsatzGesamt*100",
     "bereinigte EBITDAC-Marge 2025 in Prozent"),
    ("EBITDACberMargeVJ", "EBITDACberVJ/UmsatzGesamtVJ*100",
     "bereinigte EBITDAC-Marge 2024 in Prozent"),
    ("EBITDACDelta", "(EBITDAC/EBITDACVJ-1)*100",
     "Veraenderung des berichteten EBITDAC in Prozent"),
    ("EBITDACberDelta", "(EBITDACber/EBITDACberVJ-1)*100",
     "Veraenderung des bereinigten EBITDAC in Prozent"),
    ("BereinigungsSpanne", "EBITDACber-EBITDAC",
     "Abstand bereinigtes zu berichtetem EBITDAC 2025"),
    ("BereinigungsSpanneVJ", "EBITDACberVJ-EBITDACVJ",
     "Abstand bereinigtes zu berichtetem EBITDAC 2024"),
    ("ErgebnisDelta", "(Konzernergebnis/KonzernergebnisVJ-1)*100",
     "Veraenderung des Konzernergebnisses in Prozent"),
    ("EPSdilDelta", "(EPSdil/EPSdilVJ-1)*100",
     "Veraenderung des verwaesserten Ergebnisses je Aktie in Prozent"),
    ("Steuerquote", "Ertragsteuern/ErgebnisVorSteuern*100",
     "Konzernsteuerquote 2025 in Prozent"),
    ("Nettoverschuldung", "Gesamtverschuldung-Zahlungsmittel",
     "Nettoverschuldung 31.12.2025; Treuhandmittel bleiben ausser Ansatz"),
    ("NettoverschuldungVJ", "GesamtverschuldungVJ-ZahlungsmittelVJ",
     "Nettoverschuldung 31.12.2024"),
    ("Verschuldungsgrad", "(Gesamtverschuldung-Zahlungsmittel)/EBITDACber",
     "Nettoverschuldung je Einheit bereinigtes EBITDAC 2025"),
    ("VerschuldungsgradVJ", "(GesamtverschuldungVJ-ZahlungsmittelVJ)/EBITDACberVJ",
     "derselbe Quotient fuer 2024"),
    ("VerschuldungsgradProForma", "(Gesamtverschuldung-Zahlungsmittel)/(ProFormaUmsatz*EBITDACberMarge/100)",
     "Verschuldungsgrad, gerechnet auf das bereinigte EBITDAC der Pro-forma-Erloese"),
    ("EBITDACProForma", "ProFormaUmsatz*EBITDACberMarge/100",
     "bereinigtes EBITDAC auf Pro-forma-Erloesbasis, Marge 2025 fortgeschrieben"),
    ("HJNettoverschuldung", "HJFKkurz+HJFKlang-HJZahlungsmittel",
     "Nettoverschuldung 30.06.2026"),
    ("FreierCF", "OperativerCF-Sachinvestitionen",
     "freier Cash-Flow 2025 vor Akquisitionen"),
    ("FreierCFVJ", "OperativerCFVJ-SachinvestitionenVJ",
     "freier Cash-Flow 2024"),
    ("FreierCFnachDividende", "OperativerCF-Sachinvestitionen-Dividendenzahlung",
     "freier Cash-Flow 2025 nach Dividende"),
    ("EKRendite", "Konzernergebnis/Eigenkapital*100",
     "Eigenkapitalrendite auf den Bestand zum 31.12.2025"),
    ("EKRenditeDurchschnitt", "Konzernergebnis/((Eigenkapital+EigenkapitalVJ)/2)*100",
     "Eigenkapitalrendite auf das durchschnittliche Eigenkapital 2025"),
    ("EKRenditeVJ", "KonzernergebnisVJ/EigenkapitalVJ*100",
     "Eigenkapitalrendite auf den Bestand zum 31.12.2024"),
    ("GoodwillQuote", "Goodwill/Bilanzsumme*100",
     "Anteil des Geschaefts- oder Firmenwerts an der Bilanzsumme in Prozent"),
    ("ImmaterielleQuote", "(Goodwill+ImmatVermoegen)/Bilanzsumme*100",
     "Anteil Goodwill plus erworbener immaterieller Vermoegenswerte"),
    ("GoodwillEKDeckung", "Goodwill/Eigenkapital",
     "Goodwill je Einheit Eigenkapital"),
    ("EKQuote", "Eigenkapital/Bilanzsumme*100",
     "Eigenkapitalquote 31.12.2025 in Prozent"),
    ("Zinsdeckung", "EBITDACber/Zinsaufwand",
     "bereinigtes EBITDAC je Einheit Zinsaufwand 2025"),
    ("ZinsdeckungVJ", "EBITDACberVJ/ZinsaufwandVJ",
     "dasselbe fuer 2024"),
    ("HJUmsatzDelta", "(HJUmsatz/HJUmsatzVJ-1)*100",
     "Umsatzwachstum erstes Halbjahr 2026 in Prozent"),
    ("AktienZuwachs", "(AktienUmlauf/AktienUmlaufVJ-1)*100",
     "Zuwachs der umlaufenden Aktien 2025 in Prozent"),
    ("VerwaesserungStueck", "AktienUmlauf-AktienUmlaufVJ",
     "Zuwachs der umlaufenden Aktien in Mio. Stueck"),
    ("KursDeltaZweiZwei", "(KursSchlussZFII/KursSchlussZFI-1)*100",
     "Kursveraenderung 2022 in Prozent"),
    ("KursDeltaZweiDrei", "(KursSchlussZFIII/KursSchlussZFII-1)*100",
     "Kursveraenderung 2023 in Prozent"),
    ("KursDeltaZweiVier", "(KursSchlussZFIV/KursSchlussZFIII-1)*100",
     "Kursveraenderung 2024 in Prozent"),
    ("KursDeltaZweiFuenf", "(KursSchlussZFV/KursSchlussZFIV-1)*100",
     "Kursveraenderung 2025 in Prozent"),
    ("KursDeltaLaufend", "(KursSchlussZFVI/KursSchlussZFV-1)*100",
     "Kursveraenderung 2026 bis zum Abrufdatum in Prozent"),
    ("KursAbstandHoch", "(KursSchlussZFVI/KursHochZFV-1)*100",
     "Abstand des aktuellen Kurses zum Hoechstkurs 2025 in Prozent"),
    ("KursAbstandTief", "(KursSchlussZFVI/KursTiefZFVI-1)*100",
     "Abstand des aktuellen Kurses zum Jahrestief 2026 in Prozent"),
    ("Marktkapitalisierung", "KursSchlussZFV*AktienUmlauf",
     "Marktkapitalisierung 31.12.2025; beide Groessen tragen dasselbe Datum"),
    ("MarktkapitalisierungVJ", "KursSchlussZFIV*AktienUmlaufVJ",
     "Marktkapitalisierung 31.12.2024"),
    ("KGV", "KursSchlussZFV/EPSdil",
     "Kurs-Gewinn-Verhaeltnis 31.12.2025 auf verwaessertem Ergebnis je Aktie"),
    ("KGVVJ", "KursSchlussZFIV/EPSdilVJ",
     "dasselbe zum 31.12.2024"),
    ("KursBuchwert", "(KursSchlussZFV*AktienUmlauf)/Eigenkapital",
     "Verhaeltnis Marktkapitalisierung zu bilanziellem Eigenkapital 31.12.2025"),
    ("Dividendenrendite", "DividendeJeAktie/KursSchlussZFV*100",
     "Dividendenrendite auf den Schlusskurs 31.12.2025 in Prozent"),
    ("DividendenrenditeVJ", "DividendeJeAktieVJ/KursSchlussZFIV*100",
     "Dividendenrendite auf den Schlusskurs 31.12.2024 in Prozent"),
    ("DividendeDelta", "(DividendeJeAktie/DividendeJeAktieVJ-1)*100",
     "Erhoehung der Dividende je Aktie 2025 in Prozent"),
    ("Ausschuettungsquote", "DividendeJeAktie/EPSdil*100",
     "Ausschuettungsquote auf das verwaesserte Ergebnis je Aktie in Prozent"),
    # Der Jahresumsatz der Zukaeufe 2025 steht nirgends direkt; er ergibt
    # sich aus zwei veroeffentlichten Angaben: dem Pro-forma-Mehrumsatz
    # (Volljahreswirkung abzueglich des bereits konsolidierten Teils) plus
    # dem tatsaechlich konsolidierten Umsatzbeitrag. Die Formel steht im
    # Dokument, damit der Leser sie nachrechnen kann.
    ("ZukaeufeUmsatzJahr", "ProFormaUmsatz-UmsatzGesamt+UmsatzAusKaeufen",
     "rechnerischer Jahresumsatz der 2025 erworbenen Gesellschaften"),
    ("ProFormaMehrumsatz", "ProFormaUmsatz-UmsatzGesamt",
     "Mehrumsatz der Pro-forma-Rechnung gegenueber dem berichteten Umsatz"),
    ("KaufpreisJeUmsatz", "KaeufeKaufpreis/(ProFormaUmsatz-UmsatzGesamt+UmsatzAusKaeufen)",
     "Kaufpreis aller Zukaeufe 2025 je Einheit ihres Jahresumsatzes"),
    ("EigenkapitalZuwachs", "(Eigenkapital/EigenkapitalVJ-1)*100",
     "Zuwachs des bilanziellen Eigenkapitals 2025 in Prozent"),
    ("GoodwillZuwachs", "Goodwill-GoodwillVJ",
     "Zuwachs des Geschaefts- oder Firmenwerts 2025"),
    ("HJEigeneAktienZuwachs", "HJEigeneAktien-HJEigeneAktienVJ",
     "Aktienrueckkauf im ersten Halbjahr 2026 zu Anschaffungskosten"),
    ("RetailMarge", "RetailEBITDACber/RetailUmsatz*100",
     "bereinigte EBITDAC-Marge des Retail-Segments 2025 in Prozent"),
    ("RetailMargeVJ", "RetailEBITDACberVJ/RetailUmsatzVJ*100",
     "dieselbe Marge 2024"),
    ("SpezMarge", "SpezEBITDACber/SpezUmsatz*100",
     "bereinigte EBITDAC-Marge der Specialty Distribution 2025 in Prozent"),
    ("SpezMargeVJ", "SpezEBITDACberVJ/SpezUmsatzVJ*100",
     "dieselbe Marge 2024"),
    ("OperativerCFDelta", "(OperativerCF/OperativerCFVJ-1)*100",
     "Veraenderung des operativen Cash-Flows 2025 in Prozent"),
    ("MargeDeltaBer", "EBITDACberMarge-EBITDACberMargeVJ",
     "Veraenderung der bereinigten EBITDAC-Marge in Prozentpunkten"),
    ("MargeDeltaBerichtet", "EBITDACMargeVJ-EBITDACMarge",
     "Rueckgang der berichteten EBITDAC-Marge in Prozentpunkten"),
]


# LaTeX-Makronamen duerfen keine Ziffern enthalten; \KursSchlussZFV ist
# kein gueltiger Bezeichner. Die Jahreszahl wird deshalb im Namen durch ein
# roemisches Kuerzel ersetzt: ZF = zwanziger Jahre, danach die Endziffer.
JAHRKUERZEL = {"2021": "ZFI", "2022": "ZFII", "2023": "ZFIII",
               "2024": "ZFIV", "2025": "ZFV", "2026": "ZFVI"}


def _kurse() -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest data/kurs_jahr.csv und data/kurs_extrema.csv und gibt die
        Kurswerte als Makrowerte zurueck. Der Makroname traegt statt der
        Jahreszahl das Kuerzel aus JAHRKUERZEL.

    Inputs:
        keine

    Outputs:
        werte (dict[str, tuple]): Schluessel -> (Wert, Quellenhinweis)
    --------------------------------------------------------------------------
    """
    werte = {}
    quelle = "Nasdaq, Historical Quotes BRO, abgerufen am 28.08.2026"
    pfad = os.path.join(ROOT, "data", "kurs_jahr.csv")
    with open(pfad, encoding="utf-8") as fh:
        zeilen = [z for z in fh if not z.startswith("#")]
    for r in csv.DictReader(zeilen):
        j = JAHRKUERZEL[r["jahr"]]
        werte[f"KursSchluss{j}"] = (float(r["schluss_usd"]), quelle)
        werte[f"KursHoch{j}"] = (float(r["hoch_usd"]), quelle)
        werte[f"KursTief{j}"] = (float(r["tief_usd"]), quelle)
    pfad = os.path.join(ROOT, "data", "kurs_extrema.csv")
    with open(pfad, encoding="utf-8") as fh:
        zeilen = [z for z in fh if not z.startswith("#")]
    for r in csv.DictReader(zeilen):
        tag = r["datum"]
        j, m, t = tag.split("-")
        werte[f"Kurstag{r['art'].capitalize()}{JAHRKUERZEL[r['jahr']]}"] = (
            f"{int(t)}.{int(m)}.{j}", quelle)
    return werte


def main() -> None:
    roh = dict(ROH)
    kurse = _kurse()
    zahlen = {k: v[0] for k, v in roh.items()}
    zahlen.update({k: v[0] for k, v in kurse.items()
                   if isinstance(v[0], (int, float))})

    abgeleitet = []
    for name, ausdruck, kommentar in ABGELEITET:
        wert = eval(ausdruck, {"__builtins__": {}}, zahlen)  # nur eigene Namen
        zahlen[name] = wert
        abgeleitet.append((name, wert, ausdruck, kommentar))

    zeilen = [
        "% ==================================================================",
        "% kennzahlen.tex - ERZEUGT von scripts/kennzahlen.py. Nicht haendisch",
        "% aendern; jede Aenderung dort vornehmen und das Skript neu laufen",
        "% lassen. Betraege in Mio. USD, sofern der Kommentar nichts anderes",
        "% sagt. Die Seitenzahlen sind die des gedruckten Dokuments.",
        "% ==================================================================",
        "",
        "% --- Rohwerte aus den Primaerquellen ------------------------------",
    ]
    for name, (wert, quelle) in roh.items():
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert}}}   % {quelle}")
    zeilen += ["", "% --- Kursreihe (siehe scripts/fetch_kurs.py) ----------------------"]
    for name, (wert, quelle) in kurse.items():
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert}}}   % {quelle}")
    zeilen += ["", "% --- Abgeleitete Groessen: gerechnet, nicht getippt ---------------"]
    for name, wert, ausdruck, kommentar in abgeleitet:
        zeilen.append(f"\\newcommand{{\\{name}}}{{{round(wert, 4)}}}"
                      f"   % = {ausdruck}; {kommentar}")
    # --- Selbstauskunft des Dokuments -------------------------------------
    # Der Abschnitt "Methode und Grenzen" nennt, wie viele Angaben geprueft
    # werden. Diese Zahl wird gezaehlt und nicht getippt, sonst laeuft sie
    # beim naechsten hinzugefuegten Rohwert aus dem Tritt. Sie gehoert
    # bewusst NICHT nach ROH: sie beschreibt das Dokument, nicht das
    # Unternehmen, und hat deshalb keine Seitenangabe, die pruefbar waere.
    belegt = sum(1 for _, quelle in roh.values()
                 if re.match(r"^.+?,\s*S\.\s*\d+", quelle))
    zeilen += ["", "% --- Selbstauskunft des Dokuments ---------------------------------",
               f"\\newcommand{{\\PruefungAngaben}}{{{belegt}}}"
               "   % gezaehlt: Rohwerte mit Seitenangabe"]
    zeilen.append("")

    ziel = os.path.join(ROOT, "data", "kennzahlen.tex")
    with open(ziel, "w", encoding="utf-8") as fh:
        fh.write("\n".join(zeilen))
    print(f"[KENNZAHLEN] {len(roh)} Rohwerte, {len(kurse)} Kurswerte, "
          f"{len(abgeleitet)} abgeleitete Groessen -> data/kennzahlen.tex")


if __name__ == "__main__":
    main()
