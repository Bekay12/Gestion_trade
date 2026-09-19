#!/usr/bin/env python3
"""
kennzahlen.py - erzeugt data/kennzahlen.tex, die einzige Zahlenquelle des
Dokuments.

Jeder Rohwert steht genau einmal in ROH, mit Quelle und gedruckter Seite.
Abgeleitete Groessen werden hier gerechnet und nie getippt, damit Text und
Tabellen einander nicht widersprechen koennen und keine Rate aus
vorgerundeten Werten entsteht.

Drei Bloecke, und die Trennung ist Absicht:

    ROH      belegbar auf einer gedruckten Seite; pruefe_seiten.py geht sie
             Wert fuer Wert durch.
    EXTERN   Sekundaerquellen (Kurse, Branchenindex, Wechselkurs). Sie tragen
             keine gedruckte Seite, stehen deshalb ausserhalb der Pruefung und
             werden im Dokument als Sekundaerquelle ausgewiesen.
    ABGELEITET  gerechnet aus den beiden vorigen.

Die Mehrjahresreihen kommen nicht von Hand, sondern aus reihe_azn.py und
reihe_rdy.py: dieselben Werte, die dort 148 Kettengleichungen bestehen.
Roemische Endungen (AznUmsatzXXV = 2025) sind keine Marotte - LaTeX-Makro-
namen duerfen keine Ziffern enthalten.

Waehrungen: AstraZeneca in Mio. USD, Dr. Reddy's in Mio. INR. Die beiden
werden NIE vermischt; jede Groesse traegt die Waehrung im Namen der Tabelle,
in der sie erscheint.

Aufruf: python3 scripts/kennzahlen.py
Ausgabe: data/kennzahlen.tex
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reihe_azn                                            # noqa: E402
import reihe_rdy                                            # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ROEMISCH = {1: "I", 4: "IV", 5: "V", 9: "IX", 10: "X", 40: "XL", 50: "L"}


def roemisch(n: int) -> str:
    """Jahreszahl als roemische Endung: 2025 -> XXV, 2018 -> XVIII."""
    n, aus = n % 100, ""
    for wert in sorted(_ROEMISCH, reverse=True):
        while n >= wert:
            aus += _ROEMISCH[wert]
            n -= wert
    return aus


def reihen_roh() -> dict:
    """Mehrjahresreihen beider Emittenten als Rohwerte mit Seitenbeleg."""
    aus = {}
    for vorsatz, modul in (("Azn", reihe_azn), ("Rdy", reihe_rdy)):
        for (name, jahr), lesungen in modul.sammle().items():
            wert, bericht, seite = lesungen[0]
            aus[f"{vorsatz}{name}{roemisch(jahr)}"] = (wert, f"{bericht}, S. {seite}")
    return aus


# --------------------------------------------------------------------------
# Rohwerte, die nicht Teil einer Reihe sind. Betraege AZN in Mio. USD,
# RDY in Mio. INR, sofern der Name nichts anderes sagt.
# --------------------------------------------------------------------------
ROH = {
    # === AstraZeneca, Geschaeftsjahr 2025 ================================
    # --- Gesamtergebnisrechnung, Geschaeftsbericht 2025, S. 125 ----------
    "AznProduktumsatz":      (55573, "Geschaeftsbericht 2025, S. 125"),
    "AznAllianzumsatz":      (3067, "Geschaeftsbericht 2025, S. 125"),
    "AznHerstellkosten":     (10633, "Geschaeftsbericht 2025, S. 125"),
    "AznRohertrag":          (48106, "Geschaeftsbericht 2025, S. 125"),
    "AznForschung":          (14232, "Geschaeftsbericht 2025, S. 125"),
    "AznVertrieb":           (19933, "Geschaeftsbericht 2025, S. 125"),
    "AznFinanzaufwand":      (1694, "Geschaeftsbericht 2025, S. 125"),
    "AznVorsteuerergebnis":  (12402, "Geschaeftsbericht 2025, S. 125"),
    "AznSteuern":            (2169, "Geschaeftsbericht 2025, S. 125"),
    "AznAktienVerwaessert":  (1562, "Geschaeftsbericht 2025, S. 125"),
    # --- Bilanz, Geschaeftsbericht 2025, S. 126 -------------------------
    "AznSachanlagen":        (12962, "Geschaeftsbericht 2025, S. 126"),
    "AznGoodwill":           (21242, "Geschaeftsbericht 2025, S. 126"),
    "AznImmatVermoegen":     (37846, "Geschaeftsbericht 2025, S. 126"),
    "AznVorraete":           (6557, "Geschaeftsbericht 2025, S. 126"),
    "AznForderungen":        (15177, "Geschaeftsbericht 2025, S. 126"),
    "AznLiquiditaet":        (5711, "Geschaeftsbericht 2025, S. 126"),
    "AznBilanzsumme":        (114074, "Geschaeftsbericht 2025, S. 126"),
    "AznBilanzsummeVJ":      (104035, "Geschaeftsbericht 2025, S. 126"),
    # Achtung: "Interest-bearing loans and borrowings" sind die FINANZschulden,
    # nicht die Gesamtverbindlichkeiten. Die Gesamtverbindlichkeiten stehen
    # mit 65.355 und sind fast doppelt so hoch; die beiden zu verwechseln
    # verkehrt die Verschuldungslage ins Gegenteil.
    "AznFinanzschuldenKurz": (3104, "Geschaeftsbericht 2025, S. 126"),
    "AznFinanzschuldenLang": (24715, "Geschaeftsbericht 2025, S. 126"),
    "AznVerbindlichkeiten":  (65355, "Geschaeftsbericht 2025, S. 126"),
    "AznEigenkapital":       (48719, "Geschaeftsbericht 2025, S. 126"),
    "AznEigenkapitalVJ":     (40871, "Geschaeftsbericht 2025, S. 126"),
    # --- Kennzahlenuebersicht und Finanzteil ----------------------------
    "AznCoreBetriebsergebnis":   (18478, "Geschaeftsbericht 2025, S. 51"),
    "AznCoreBetriebsergebnisVJ": (16928, "Geschaeftsbericht 2025, S. 51"),
    "AznCoreErgebnisJeAktie":    (9.16, "Geschaeftsbericht 2025, S. 51"),
    "AznCoreErgebnisJeAktieVJ":  (8.21, "Geschaeftsbericht 2025, S. 51"),
    "AznEbitda":                 (19476, "Geschaeftsbericht 2025, S. 59"),
    "AznEbitdaVJ":               (16691, "Geschaeftsbericht 2025, S. 59"),
    "AznNettoverschuldung":      (23374, "Geschaeftsbericht 2025, S. 60"),
    "AznNettoverschuldungVJ":    (24570, "Geschaeftsbericht 2025, S. 60"),
    "AznDividendeJeAktie":       (3.20, "Geschaeftsbericht 2025, S. 63"),
    # Erklaert fuer das Jahr (3,20) gegen im Jahr gezahlt und verbucht
    # (3,13). Der Unterschied ist der Versatz zwischen Erklaerung und
    # Zahlung und keine Abweichung; beide Werte stehen im Bericht.
    "AznDividendeGezahltJeAktie": (3.13, "Geschaeftsbericht 2025, S. 169"),
    "AznDividendeDeckungCore":   (2.86, "Geschaeftsbericht 2025, S. 63"),
    "AznAktienAusgegeben":       (1550907927, "Geschaeftsbericht 2025, S. 169"),
    "AznNennwert":               (0.25, "Geschaeftsbericht 2025, S. 169"),
    "AznSteuerZahlung":          (2845, "Geschaeftsbericht 2025, S. 59"),
    # Zahlungen auf bedingte Kaufpreise frueherer Zukaeufe. Sie sind
    # wiederkehrend und werden im freien Mittelzufluss dieses Berichts
    # bewusst NICHT abgezogen; die Annahme dazu steht im Text.
    "AznBedingteKaufpreise":     (1164, "Geschaeftsbericht 2025, S. 128"),
    # --- Eigentuemerstruktur, Geschaeftsbericht 2025, S. 225 ------------
    "AznBlackRock":          (6.50, "Geschaeftsbericht 2025, S. 225"),
    "AznCapitalGroup":       (4.97, "Geschaeftsbericht 2025, S. 225"),
    "AznWellington":         (4.20, "Geschaeftsbericht 2025, S. 225"),
    "AznInvestorAB":         (3.33, "Geschaeftsbericht 2025, S. 225"),
    # --- Halbjahr 2026, Form 6-K vom 27.07.2026 -------------------------
    "AznUmsatzHJ":           (30672, "Halbjahresbericht 2026, S. 1"),
    "AznCoreErgebnisJeAktieHJ": (5.21, "Halbjahresbericht 2026, S. 1"),
    "AznErgebnisJeAktieHJ":  (3.60, "Halbjahresbericht 2026, S. 1"),

    # === Dr. Reddy's, Geschaeftsjahr zum 31.03.2026 ======================
    # --- Bilanz, Form 20-F 2026, S. 111 ---------------------------------
    "RdyLiquiditaet":        (15368, "Form 20-F 2026, S. 111"),
    "RdyWertpapiere":        (72446, "Form 20-F 2026, S. 111"),
    "RdyForderungen":        (101219, "Form 20-F 2026, S. 111"),
    "RdyVorraete":           (76531, "Form 20-F 2026, S. 111"),
    "RdyBilanzsumme":        (579346, "Form 20-F 2026, S. 111"),
    "RdyBilanzsummeVJ":      (492989, "Form 20-F 2026, S. 111"),
    "RdyVerbindlichkeiten":  (198889, "Form 20-F 2026, S. 111"),
    "RdyEigenkapital":       (380457, "Form 20-F 2026, S. 111"),
    "RdyEigenkapitalVJ":     (337166, "Form 20-F 2026, S. 111"),
    "RdySchuldenKurz":       (59135, "Form 20-F 2026, S. 111"),
    "RdySchuldenLangKurz":   (6003, "Form 20-F 2026, S. 111"),
    "RdySchuldenLang":       (12203, "Form 20-F 2026, S. 111"),
    # --- Gewinn- und Verlustrechnung, Form 20-F 2026, S. 112 ------------
    "RdyBetriebsergebnis":   (50551, "Form 20-F 2026, S. 112"),
    "RdyBetriebsergebnisVJ": (71843, "Form 20-F 2026, S. 112"),
    "RdyFinanzertrag":       (7870, "Form 20-F 2026, S. 112"),
    "RdyFinanzaufwand":      (-3738, "Form 20-F 2026, S. 112"),
    "RdySteuern":            (12351, "Form 20-F 2026, S. 112"),
    "RdyWertminderung":      (3519, "Form 20-F 2026, S. 112"),
    "RdyErgebnisJeAktie":    (51.48, "Form 20-F 2026, S. 112"),
    "RdyErgebnisJeAktieVJ":  (67.88, "Form 20-F 2026, S. 112"),
    "RdyErgebnisJeAktieVVJ": (66.93, "Form 20-F 2026, S. 112"),
    # --- Eigentuemer und Aktie, Form 20-F 2026 --------------------------
    "RdyAktienAusgegeben":   (834656970, "Form 20-F 2026, S. 81"),
    "RdyPromoterAnteil":     (26.63, "Form 20-F 2026, S. 81"),
    "RdyLicAnteil":          (12.16, "Form 20-F 2026, S. 81"),
    "RdyAdrAnteil":          (11.94, "Form 20-F 2026, S. 82"),
    "RdyDividendeJeAktie":   (8, "Form 20-F 2026, S. 82"),
    "RdyUmrechnungskurs":    (93.83, "Form 20-F 2026, S. 119"),
    # Die Preissenkung auf Lenalidomid nach dem Vergleich mit der
    # Klaegerseite; die einzige bezifferte Angabe zum Auslaufen dieses
    # Geschaefts, die der Bericht ueberhaupt macht.
    "RdySsaEffekt":          (4530, "Form 20-F 2026, S. 158"),
    # Waehrungsbereinigter Umsatzrueckgang, vom Unternehmen selbst genannt.
    "RdyOrganisch":          (-2, "Form 20-F 2026, S. 54"),
    "RdyNrtKaufpreis":       (56121, "Form 20-F 2026, S. 192"),
    # --- Halbjahr 2026, Kapitalflussrechnung, S. 33 ---------------------
    # Diese sechs Werte bilden die letzten zwoelf Monate. Sie stehen hier und
    # nicht in flow.py, damit auch sie ihre gedruckte Seite tragen.
    "AznOperativerCfHJ":     (6224, "Halbjahresbericht 2026, S. 33"),
    "AznOperativerCfHJVJ":   (7099, "Halbjahresbericht 2026, S. 33"),
    "AznSachInvestitionHJ":  (-1310, "Halbjahresbericht 2026, S. 33"),
    "AznSachInvestitionHJVJ": (-1088, "Halbjahresbericht 2026, S. 33"),
    "AznImmatInvestitionHJ": (-3333, "Halbjahresbericht 2026, S. 33"),
    "AznImmatInvestitionHJVJ": (-1804, "Halbjahresbericht 2026, S. 33"),
    "AznImmatAbgangHJ":      (165, "Halbjahresbericht 2026, S. 33"),
    "AznImmatAbgangHJVJ":    (95, "Halbjahresbericht 2026, S. 33"),
    "AznLeasingTilgungHJ":   (-182, "Halbjahresbericht 2026, S. 33"),
    "AznLeasingTilgungHJVJ": (-184, "Halbjahresbericht 2026, S. 33"),

    # --- Organe, Belegschaft, Rechtsrisiken -----------------------------
    "AznBoardSitzungen":     (9, "Geschaeftsbericht 2025, S. 67"),
    # Der Verguetungsbericht druckt in Tausend Pfund; so wird der Wert
    # gespeichert und erst bei der Ausgabe in Millionen gewandelt.
    "AznCeoVerguetung":      (17696, "Geschaeftsbericht 2025, S. 97"),
    "AznCfoVerguetung":      (7846, "Geschaeftsbericht 2025, S. 97"),
    "AznPayRatio":           (176, "Geschaeftsbericht 2025, S. 110"),
    "AznPayRatioOhneLti":    (63, "Geschaeftsbericht 2025, S. 111"),
    "AznMitarbeiter":        (96100, "Geschaeftsbericht 2025, S. 27"),
    "AznMitarbeiterVJ":      (94300, "Geschaeftsbericht 2025, S. 27"),
    "AznMitarbeiterFuE":     (16100, "Geschaeftsbericht 2025, S. 27"),
    "AznRueckstellungRecht": (376, "Geschaeftsbericht 2025, S. 160"),
    "AznRueckstellungRechtVJ": (859, "Geschaeftsbericht 2025, S. 160"),
    "AznEigeneAktien":       (147547, "Geschaeftsbericht 2025, S. 169"),
    "AznZustimmungNotierung": (99.36, "Geschaeftsbericht 2025, S. 50"),

    # --- Therapiegebiete und Regionen, Geschaeftsbericht 2025 -----------
    "AznOnkologie":          (25619, "Geschaeftsbericht 2025, S. 13"),
    "AznOnkologieVJ":        (22353, "Geschaeftsbericht 2025, S. 13"),
    "AznCvrm":               (12861, "Geschaeftsbericht 2025, S. 17"),
    "AznCvrmVJ":             (12517, "Geschaeftsbericht 2025, S. 17"),
    "AznRespImmun":          (8866, "Geschaeftsbericht 2025, S. 17"),
    "AznRespImmunVJ":        (7876, "Geschaeftsbericht 2025, S. 17"),
    "AznImpfstoffe":         (1268, "Geschaeftsbericht 2025, S. 17"),
    "AznImpfstoffeVJ":       (1462, "Geschaeftsbericht 2025, S. 17"),
    "AznSeltene":            (9126, "Geschaeftsbericht 2025, S. 23"),
    "AznSelteneVJ":          (8768, "Geschaeftsbericht 2025, S. 23"),
    "AznUmsatzUsa":          (25450, "Geschaeftsbericht 2025, S. 32"),
    "AznUmsatzUsaVJ":        (23235, "Geschaeftsbericht 2025, S. 32"),
    "AznUmsatzEuropa":       (12739, "Geschaeftsbericht 2025, S. 32"),
    "AznUmsatzSchwellen":    (15303, "Geschaeftsbericht 2025, S. 32"),
    "AznUmsatzChina":        (6654, "Geschaeftsbericht 2025, S. 33"),
    "AznUmsatzChinaVJ":      (6413, "Geschaeftsbericht 2025, S. 33"),
    "AznFarxiga":            (8492, "Geschaeftsbericht 2025, S. 18"),
    "AznTagrisso":           (7254, "Geschaeftsbericht 2025, S. 13"),
    "AznImfinzi":            (6063, "Geschaeftsbericht 2025, S. 13"),
    "AznBrilinta":           (823, "Geschaeftsbericht 2025, S. 18"),
    "AznFarxigaHJ":          (4042, "Halbjahresbericht 2026, S. 9"),
    "AznFarxigaUsaHJ":       (668, "Halbjahresbericht 2026, S. 9"),
    # --- Massnahmen und Pipeline ----------------------------------------
    "AznForschungCore":      (13822, "Geschaeftsbericht 2025, S. 55"),
    "AznPipeline":           (197, "Geschaeftsbericht 2025, S. 30"),
    "AznSpaetphase":         (20, "Geschaeftsbericht 2025, S. 30"),
    "AznZulassungen":        (43, "Geschaeftsbericht 2025, S. 28"),
    "AznEsoBiotec":          (425, "Geschaeftsbericht 2025, S. 62"),
    "AznFibroGen":           (221, "Geschaeftsbericht 2025, S. 62"),
    "AznCspc":               (110, "Geschaeftsbericht 2025, S. 63"),
    "AznDizal":              (600, "Halbjahresbericht 2026, S. 5"),
    "AznRestrukturierung":   (237, "Geschaeftsbericht 2025, S. 59"),
    # Betraege in Mrd. USD, so wie der Bericht sie nennt.
    "AznInvestitionUsa":     (50, "Geschaeftsbericht 2025, S. 4"),
    "AznVirginia":           (4.5, "Geschaeftsbericht 2025, S. 4"),
    "AznMaryland":           (2, "Geschaeftsbericht 2025, S. 4"),
    "AznAmbition":           (80, "Geschaeftsbericht 2025, S. 10"),
    # Prognosetreue: das jeweils erreichte Umsatzwachstum in Prozent, aus dem
    # Bericht des Jahres selbst. Die Prognosen sind Worte ("high-teens") und
    # stehen deshalb im Text, nicht in der Zahlenschicht.
    "AznIstXXII":            (19, "Geschaeftsbericht 2022, S. 60"),
    "AznIstXXIII":           (3, "Geschaeftsbericht 2023, S. 58"),
    "AznIstXXIV":            (18, "Geschaeftsbericht 2024, S. 67"),
    "AznIstXXV":             (9, "Geschaeftsbericht 2025, S. 55"),

    # === Dr. Reddy's, Ergaenzungen =======================================
    # --- Segmente und Maerkte, Form 20-F 2026, S. 54 --------------------
    "RdyGlobalGenerics":     (299033, "Form 20-F 2026, S. 54"),
    "RdyGlobalGenericsVJ":   (289552, "Form 20-F 2026, S. 54"),
    "RdyPsai":               (34773, "Form 20-F 2026, S. 54"),
    "RdyPsaiVJ":             (33846, "Form 20-F 2026, S. 54"),
    "RdyNordamerika":        (113737, "Form 20-F 2026, S. 54"),
    "RdyNordamerikaVJ":      (145164, "Form 20-F 2026, S. 54"),
    "RdyEuropa":             (55501, "Form 20-F 2026, S. 54"),
    "RdyEuropaVJ":           (35882, "Form 20-F 2026, S. 54"),
    "RdyIndien":             (62186, "Form 20-F 2026, S. 54"),
    "RdyIndienVJ":           (53734, "Form 20-F 2026, S. 54"),
    "RdyRussland":           (34786, "Form 20-F 2026, S. 54"),
    "RdyNrtUmsatz":          (28189, "Form 20-F 2026, S. 54"),
    "RdyNrtUmsatzVJ":        (12020, "Form 20-F 2026, S. 54"),
    "RdyRohertragGg":        (169698, "Form 20-F 2026, S. 57"),
    "RdyRohertragGgVJ":      (179606, "Form 20-F 2026, S. 57"),
    "RdyRohertragGgMarge":   (56.7, "Form 20-F 2026, S. 57"),
    "RdyRohertragGgMargeVJ": (62.0, "Form 20-F 2026, S. 57"),
    "RdyOnkologie":          (63131, "Form 20-F 2026, S. 141"),
    "RdyOnkologieVJ":        (85798, "Form 20-F 2026, S. 141"),
    "RdyNerven":             (61292, "Form 20-F 2026, S. 141"),
    "RdyUmsatzUsa":          (117892, "Form 20-F 2026, S. 141"),
    "RdyUmsatzUsaVJ":        (149351, "Form 20-F 2026, S. 141"),
    # --- Massnahmen, Standorte, Belegschaft -----------------------------
    "RdyMitarbeiter":        (27527, "Form 20-F 2026, S. 77"),
    "RdyMitarbeiterVJ":      (27811, "Form 20-F 2026, S. 77"),
    "RdyAndaEingereicht":    (15, "Form 20-F 2026, S. 55"),
    "RdyAndaKumuliert":      (339, "Form 20-F 2026, S. 55"),
    "RdyAndaAnhaengig":      (77, "Form 20-F 2026, S. 55"),
    "RdyAnlagenBau":         (15409, "Form 20-F 2026, S. 47"),
    "RdyInvestVerpflichtung": (9716, "Form 20-F 2026, S. 62"),
    "RdyNestleInvest":       (7344, "Form 20-F 2026, S. 192"),
    "RdyNrtBarzahlung":      (51407, "Form 20-F 2026, S. 192"),
    "RdyNrtGoodwill":        (7170, "Form 20-F 2026, S. 192"),
    # --- Quartal zum 30.06.2026, S. 39 bzw. 16 --------------------------
    "RdyQuartalGewinn":      (4348, "Quartalsbericht 2027, S. 39"),
    "RdyQuartalGewinnVJ":    (14096, "Quartalsbericht 2027, S. 39"),
    "RdyQuartalCf":          (-927, "Quartalsbericht 2027, S. 39"),
    "RdyQuartalCfVJ":        (14629, "Quartalsbericht 2027, S. 39"),
    "RdyQuartalInvest":      (-1231, "Quartalsbericht 2027, S. 39"),
    "RdyQuartalInvestVJ":    (-10115, "Quartalsbericht 2027, S. 39"),
    "RdySemaglutidRueckstellung": (2397, "Quartalsbericht 2027, S. 16"),
    "RdyKursTage":           (108, "Quartalsbericht 2027, S. 39"),
    "RdyKursTageVJ":         (93, "Quartalsbericht 2027, S. 39"),
    # --- Organe, Form 20-F 2026 -----------------------------------------
    "RdyDirektoren":         (10, "Form 20-F 2026, S. 72"),
    "RdyDirektorenUnabh":    (8, "Form 20-F 2026, S. 72"),
    "RdyAusschuesse":        (7, "Form 20-F 2026, S. 73"),
    "RdyVerguetungChairman": (100.66, "Form 20-F 2026, S. 71"),
    "RdyVerguetungCoChairman": (158.17, "Form 20-F 2026, S. 71"),
    "RdyVerguetungCeo":      (230.33, "Form 20-F 2026, S. 71"),
    "RdyVerguetungCfo":      (72.48, "Form 20-F 2026, S. 71"),
    "RdyDeckelVollzeit":     (0.75, "Form 20-F 2026, S. 70"),
    "RdyRecordHolder":       (449665, "Form 20-F 2026, S. 82"),
    "RdyAktienVorSplit":     (166818266, "Form 20-F 2026, S. 157"),
    "RdyAktienNachSplit":    (834455365, "Form 20-F 2026, S. 157"),
}

# --------------------------------------------------------------------------
# Sekundaerquellen. Keine gedruckte Seite, deshalb ausserhalb der
# Seitenpruefung und im Dokument als Sekundaerquelle ausgewiesen.
# --------------------------------------------------------------------------
EXTERN = {
    "AznKurs":     (166.14, "Nasdaq, Historical Quotes AZN, Schluss 17.09.2026, "
                            "abgerufen 18.09.2026"),
    "RdyKurs":     (12.19, "Nasdaq, Historical Quotes RDY, Schluss 17.09.2026, "
                           "abgerufen 18.09.2026"),
    # Huerde: Zehnjahresrendite des Branchenindex, den der Auftraggeber
    # gewaehlt hat. Genommen wird der INDEX, nicht der ihn abbildende Fonds:
    # 10,16 % gegen 9,74 % - die haertere Latte, und die, die die Branche
    # tatsaechlich erwirtschaftet hat.
    "Huerde":      (10.16, "iShares U.S. Pharmaceuticals ETF, Average Annual Total "
                           "Returns, Benchmark (Dow Jones U.S. Select Pharmaceuticals "
                           "Index), 10 Jahre zum 30.06.2026, ishares.com, "
                           "abgerufen 17.09.2026"),
    "HuerdeFonds": (9.74, "dieselbe Quelle, Fonds statt Index (NAV)"),
    # Analystenkonsens. Er steht in keinem Geschaeftsbericht und ist deshalb
    # keine Luecke, sondern eine Sekundaerquelle. Der Stalenessssatz aus der
    # Regel fuer Sekundaerdaten ist gerechnet und besteht: das aus Kursziel
    # und Aufschlag zurueckgerechnete Bezugsniveau trifft auf den Cent den
    # Schlusskurs vom 17.09.2026 (AZN 166,14; RDY 12,19).
    "AznKonsensZiel":  (212.90, "stockanalysis.com/stocks/azn/forecast, Konsens aus "
                                "11 Schaetzungen (S&P Global), abgerufen 18.09.2026"),
    "AznKonsensAufschlag": (28.14, "dieselbe Quelle"),
    "AznKonsensHaeuser": (11, "dieselbe Quelle"),
    "RdyKonsensZiel":  (13.36, "stockanalysis.com/stocks/rdy/forecast, Konsens aus "
                               "5 Schaetzungen (S&P Global), abgerufen 18.09.2026"),
    "RdyKonsensAufschlag": (9.60, "dieselbe Quelle"),
    "RdyKonsensHaeuser": (5, "dieselbe Quelle"),
    "Wechselkurs": (95.94, "Europaeische Zentralbank, Referenzkurse 17.09.2026, "
                           "Kreuzkurs aus EUR/USD 1,1481 und EUR/INR 110,1475"),
}


def _kurse() -> dict:
    with open(os.path.join(ROOT, "data", "kurs_stand.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Abgeleitete Groessen. Ausdruecke werden der Reihe nach ausgewertet und
# duerfen auf vorher berechnete zugreifen.
# --------------------------------------------------------------------------
ABGELEITET = [
    # --- AstraZeneca ----------------------------------------------------
    ("AznUmsatzWachstum",   "(AznUmsatzXXV / AznUmsatzXXIV - 1) * 100"),
    ("AznErgebnisWachstum", "(AznJahresueberschussXXV / AznJahresueberschussXXIV - 1) * 100"),
    ("AznEbitdaMarge",      "AznEbitda / AznUmsatzXXV * 100"),
    ("AznEbitdaMargeVJ",    "AznEbitdaVJ / AznUmsatzXXIV * 100"),
    ("AznCoreMarge",        "AznCoreBetriebsergebnis / AznUmsatzXXV * 100"),
    ("AznBetriebsMarge",    "AznBetriebsergebnisXXV / AznUmsatzXXV * 100"),
    ("AznEigenkapitalrendite", "AznJahresueberschussXXV / AznEigenkapital * 100"),
    ("AznEigenkapitalquote", "AznEigenkapital / AznBilanzsumme * 100"),
    ("AznVerschuldungsgrad", "AznNettoverschuldung / AznEbitda"),
    ("AznBuchwertJeAktie",  "AznEigenkapital / (AznAktienAusgegeben / 1e6)"),
    ("AznKursGewinn",       "AznKurs / AznErgebnisJeAktieXXV"),
    ("AznKursGewinnCore",   "AznKurs / AznCoreErgebnisJeAktie"),
    ("AznKursBuchwert",     "AznKurs / AznBuchwertJeAktie"),
    ("AznDividendenrendite", "AznDividendeJeAktie / AznKurs * 100"),
    ("AznMarktwert",        "AznKurs * AznAktienAusgegeben / 1e6"),
    ("AznMarktwertMrd",     "AznMarktwert / 1000"),
    ("AznForschungsquote",  "AznForschung / AznUmsatzXXV * 100"),
    ("AznBetriebsergebnisWachstum",
     "(AznBetriebsergebnisXXV / AznBetriebsergebnisXXIV - 1) * 100"),
    ("AznCoreWachstum",     "(AznCoreBetriebsergebnis / AznCoreBetriebsergebnisVJ - 1) * 100"),
    ("AznEbitdaWachstum",   "(AznEbitda / AznEbitdaVJ - 1) * 100"),
    ("AznEpsWachstum",      "(AznErgebnisJeAktieXXV / AznErgebnisJeAktieXXIV - 1) * 100"),
    ("AznNettoverschuldungWachstum",
     "(AznNettoverschuldung / AznNettoverschuldungVJ - 1) * 100"),
    ("AznEigenkapitalWachstum", "(AznEigenkapital / AznEigenkapitalVJ - 1) * 100"),
    # --- Dr. Reddy's ----------------------------------------------------
    ("RdyUmsatzWachstum",   "(RdyUmsatzXXVI / RdyUmsatzXXV - 1) * 100"),
    ("RdyErgebnisWachstum", "(RdyJahresueberschussXXVI / RdyJahresueberschussXXV - 1) * 100"),
    ("RdyRohertragsmarge",  "RdyRohertragXXVI / RdyUmsatzXXVI * 100"),
    ("RdyRohertragsmargeVJ", "RdyRohertragXXV / RdyUmsatzXXV * 100"),
    ("RdyBetriebsMarge",    "RdyBetriebsergebnis / RdyUmsatzXXVI * 100"),
    ("RdyBetriebsMargeVJ",  "RdyBetriebsergebnisVJ / RdyUmsatzXXV * 100"),
    ("RdyEigenkapitalrendite", "RdyJahresueberschussXXVI / RdyEigenkapital * 100"),
    ("RdyEigenkapitalquote", "RdyEigenkapital / RdyBilanzsumme * 100"),
    ("RdyFinanzschulden",   "RdySchuldenKurz + RdySchuldenLangKurz + RdySchuldenLang"),
    ("RdyNettokasse",       "RdyLiquiditaet + RdyWertpapiere - RdyFinanzschulden"),
    ("RdyBuchwertJeAktie",  "RdyEigenkapital / (RdyAktienAusgegeben / 1e6)"),
    ("RdyKursRupien",       "RdyKurs * Wechselkurs"),
    ("RdyKursGewinn",       "RdyKursRupien / RdyErgebnisJeAktie"),
    ("RdyKursBuchwert",     "RdyKursRupien / RdyBuchwertJeAktie"),
    ("RdyDividendenrendite", "RdyDividendeJeAktie / RdyKursRupien * 100"),
    ("RdyAusschuettungsquote", "RdyDividendeJeAktie / RdyErgebnisJeAktie * 100"),
    ("RdyMarktwertRupien",  "RdyKursRupien * RdyAktienAusgegeben / 1e6"),
    ("RdyMarktwertMrd",     "RdyMarktwertRupien / 1000"),
    ("RdyForschungsquote",  "RdyForschungXXVI / RdyUmsatzXXVI * 100"),
    ("RdySsaAnteil",        "RdySsaEffekt / RdyUmsatzXXVI * 100"),
    # --- Kursbezogene Groessen ------------------------------------------
    ("AznCeoVerguetungMio", "AznCeoVerguetung / 1000"),
    ("AznCfoVerguetungMio", "AznCfoVerguetung / 1000"),
    ("AznAbstandHoch",      "(AznKurs / AznKursHochXXVI - 1) * 100"),
    ("AznRenditeLaufend",   "(AznKurs / AznKursSchlussXXV - 1) * 100"),
    ("AznRenditeZehn",      "((AznKurs / AznKursSchlussXVI) ** 0.1 - 1) * 100"),
    ("RdyAbstandHoch",      "(RdyKurs / RdyKursHochXXVI - 1) * 100"),
    ("RdyRenditeLaufend",   "(RdyKurs / RdyKursSchlussXXV - 1) * 100"),
    ("RdyRenditeZehn",      "((RdyKurs / RdyKursSchlussXVI) ** 0.1 - 1) * 100"),
    # --- Segment- und Marktveraenderungen -------------------------------
    ("AznOnkologieWachstum", "(AznOnkologie / AznOnkologieVJ - 1) * 100"),
    ("AznSelteneWachstum",  "(AznSeltene / AznSelteneVJ - 1) * 100"),
    ("AznCvrmWachstum",     "(AznCvrm / AznCvrmVJ - 1) * 100"),
    ("AznRespImmunWachstum", "(AznRespImmun / AznRespImmunVJ - 1) * 100"),
    ("AznImpfstoffeWachstum", "(AznImpfstoffe / AznImpfstoffeVJ - 1) * 100"),
    ("AznAmbitionRate",     "((AznAmbition * 1000 / AznUmsatzXXV) ** 0.2 - 1) * 100"),
    # Bewertung am Jahreshoch, gerechnet auf dasselbe bereinigte Ergebnis je
    # Aktie wie heute. Nur so trennt die Groesse Bewertung von Ertrag; mit
    # einem anderen Nenner waere sie eine Mischung aus beidem.
    ("AznKgvHoch",          "AznKursHochXXVI / AznCoreErgebnisJeAktie"),
    ("AznUsaAnteil",        "AznUmsatzUsa / AznUmsatzXXV * 100"),
    ("AznChinaAnteil",      "AznUmsatzChina / AznUmsatzXXV * 100"),
    ("AznFarxigaAnteil",    "AznFarxiga / AznUmsatzXXV * 100"),
    ("AznFarxigaHJWachstum", "(AznFarxigaHJ / (AznFarxiga / 2) - 1) * 100"),
    ("RdyNordamerikaWachstum", "(RdyNordamerika / RdyNordamerikaVJ - 1) * 100"),
    ("RdyEuropaWachstum",   "(RdyEuropa / RdyEuropaVJ - 1) * 100"),
    ("RdyIndienWachstum",   "(RdyIndien / RdyIndienVJ - 1) * 100"),
    ("RdyOnkologieWachstum", "(RdyOnkologie / RdyOnkologieVJ - 1) * 100"),
    ("RdyOnkologieRueckgang", "RdyOnkologieVJ - RdyOnkologie"),
    ("RdyNordamerikaAnteil", "RdyNordamerika / RdyGlobalGenerics * 100"),
    ("RdyUsaAnteil",        "RdyUmsatzUsa / RdyUmsatzXXVI * 100"),
    ("RdyQuartalGewinnWachstum", "(RdyQuartalGewinn / RdyQuartalGewinnVJ - 1) * 100"),
    ("RdyNrtWachstum",      "(RdyNrtUmsatz / RdyNrtUmsatzVJ - 1) * 100"),
    ("RdyGlobalGenericsAnteil", "RdyGlobalGenerics / RdyUmsatzXXVI * 100"),
    ("RdyPsaiWachstum",     "(RdyPsai / RdyPsaiVJ - 1) * 100"),
    ("RdyRohertragGgWachstum", "(RdyRohertragGg / RdyRohertragGgVJ - 1) * 100"),
    ("RdyBetriebsergebnisWachstum",
     "(RdyBetriebsergebnis / RdyBetriebsergebnisVJ - 1) * 100"),
    ("RdyEpsWachstum",      "(RdyErgebnisJeAktie / RdyErgebnisJeAktieVJ - 1) * 100"),
    ("RdyForschungWachstum", "(RdyForschungXXVI / RdyForschungXXV - 1) * 100"),
    ("RdyEigenkapitalWachstum", "(RdyEigenkapital / RdyEigenkapitalVJ - 1) * 100"),
    ("RdyKgvHoch",          "RdyKursHochXXVI * Wechselkurs / RdyErgebnisJeAktie"),
    ("RdyOnkologieAnteilVJ", "RdyOnkologieVJ / RdyUmsatzXXV * 100"),
    ("RdyRohertragWachstum", "(RdyRohertragXXVI / RdyRohertragXXV - 1) * 100"),
    ("RdyGlobalGenericsWachstum", "(RdyGlobalGenerics / RdyGlobalGenericsVJ - 1) * 100"),
    ("RdyNordamerikaRueckgang", "RdyNordamerikaVJ - RdyNordamerika"),
    ("RdyEuropaZuwachs",    "RdyEuropa - RdyEuropaVJ"),
    ("RdyInvestitionen",    "-(RdySachInvestitionXXVI + RdyImmatInvestitionXXVI)"),
    ("RdyQuartalCfBetrag",  "-RdyQuartalCf"),
    # --- Pruefung der Sekundaerdaten -------------------------------------
    ("AznAnteileGemeldet",  "AznBlackRock + AznCapitalGroup + AznWellington + AznInvestorAB"),
    ("AznStreubesitz",      "100 - AznAnteileGemeldet"),
    ("RdyStreubesitz",      "100 - RdyPromoterAnteil"),
    ("RdyStreubesitzOhneLic", "100 - RdyPromoterAnteil - RdyLicAnteil"),
    ("RdySumme",            "RdyPromoterAnteil + RdyLicAnteil + RdyStreubesitzOhneLic"),
    ("AznKonsensBezug",     "AznKonsensZiel / (1 + AznKonsensAufschlag / 100)"),
    ("RdyKonsensBezug",     "RdyKonsensZiel / (1 + RdyKonsensAufschlag / 100)"),
]


def rechne() -> dict:
    """Wertetafel aus ROH, EXTERN, den Kursreihen und ABGELEITET."""
    tafel = {k: v for k, (v, _) in ROH.items()}
    tafel.update({k: v for k, (v, _) in EXTERN.items()})
    tafel.update({k: v for k, (v, _) in reihen_roh().items()})
    # Kursgroessen: Jahresschluss, Jahreshoch und Jahrestief je Titel. Sie
    # stammen aus data/kurs_stand.json, erzeugt von scripts/kurse.py, und sind
    # Sekundaerdaten wie die Kurse selbst.
    for kuerzel, vorsatz in (("azn", "Azn"), ("rdy", "Rdy")):
        k = _kurse()[kuerzel]
        for feld, name in (("jahre", "KursSchluss"), ("hoch", "KursHoch"),
                           ("tief", "KursTief")):
            for jahr, wert in k[feld].items():
                tafel[f"{vorsatz}{name}{roemisch(int(jahr))}"] = wert
    for name, ausdruck in ABGELEITET:
        tafel[name] = eval(ausdruck, {"__builtins__": {}}, tafel)   # noqa: S307
    return tafel


def main() -> int:
    tafel = rechne()
    kurse = _kurse()
    zeilen = ["% erzeugt von scripts/kennzahlen.py - NICHT von Hand aendern",
              "% Rohwerte: siehe ROH in scripts/kennzahlen.py, je mit gedruckter Seite",
              ""]
    # Der blanke Wert, nicht \num{...}: Die Ausgabemakros (\Pct, \MioUSD,
    # \USDje) legen selbst ein \num darum, und ein zweites innen bricht
    # siunitx mit "Invalid number" ab.
    for name in sorted(tafel):
        wert = tafel[name]
        herkunft = ROH.get(name) or EXTERN.get(name)
        beleg = f"% {herkunft[1]}" if herkunft else "% abgeleitet"
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.6g}}}{beleg}")
    # Der Stichtag der Kursreihe ist kein Zahlenwert, sondern ein Datum; er
    # steht deshalb hier und nicht in der Tafel. Die Kursgroessen selbst
    # kommen bereits aus rechne() und duerfen hier NICHT ein zweites Mal
    # ausgegeben werden - LaTeX bricht bei einem doppelten \newcommand ab.
    for kuerzel, vorsatz in (("azn", "Azn"), ("rdy", "Rdy")):
        # Deutsches Datumsformat: Die Reihe liefert das Datum in ISO-Form,
        # und "2026-09-17" mitten im deutschen Satz ist ein Fremdkoerper.
        jahr, monat, tag = str(kurse[kuerzel]["letzter_tag"]).split("-")
        zeilen.append(f"\\newcommand{{\\{vorsatz}KursTag}}{{{tag}.{monat}.{jahr}}}")
    ziel = os.path.join(ROOT, "data", "kennzahlen.tex")
    open(ziel, "w").write("\n".join(zeilen) + "\n")
    print(f"[KENNZAHLEN] {len(ROH)} Einzelwerte, {len(reihen_roh())} Reihenwerte, "
          f"{len(EXTERN)} Sekundaerwerte, {len(ABGELEITET)} abgeleitet "
          f"-> data/kennzahlen.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())


# --------------------------------------------------------------------------
# Sekundaerquellen. Keine gedruckte Seite, deshalb ausserhalb der
# Seitenpruefung und im Dokument als Sekundaerquelle ausgewiesen.
# --------------------------------------------------------------------------
EXTERN = {
    "AznKurs":     (166.14, "Nasdaq, Historical Quotes AZN, Schluss 17.09.2026, "
                            "abgerufen 18.09.2026"),
    "RdyKurs":     (12.19, "Nasdaq, Historical Quotes RDY, Schluss 17.09.2026, "
                           "abgerufen 18.09.2026"),
    # Huerde: Zehnjahresrendite des Branchenindex, den der Auftraggeber
    # gewaehlt hat. Genommen wird der INDEX, nicht der ihn abbildende Fonds:
    # 10,16 % gegen 9,74 % - die haertere Latte, und die, die die Branche
    # tatsaechlich erwirtschaftet hat.
    "Huerde":      (10.16, "iShares U.S. Pharmaceuticals ETF, Average Annual Total "
                           "Returns, Benchmark (Dow Jones U.S. Select Pharmaceuticals "
                           "Index), 10 Jahre zum 30.06.2026, ishares.com, "
                           "abgerufen 17.09.2026"),
    "HuerdeFonds": (9.74, "dieselbe Quelle, Fonds statt Index (NAV)"),
    # Analystenkonsens. Er steht in keinem Geschaeftsbericht und ist deshalb
    # keine Luecke, sondern eine Sekundaerquelle. Der Stalenessssatz aus der
    # Regel fuer Sekundaerdaten ist gerechnet und besteht: das aus Kursziel
    # und Aufschlag zurueckgerechnete Bezugsniveau trifft auf den Cent den
    # Schlusskurs vom 17.09.2026 (AZN 166,14; RDY 12,19).
    "AznKonsensZiel":  (212.90, "stockanalysis.com/stocks/azn/forecast, Konsens aus "
                                "11 Schaetzungen (S&P Global), abgerufen 18.09.2026"),
    "AznKonsensAufschlag": (28.14, "dieselbe Quelle"),
    "AznKonsensHaeuser": (11, "dieselbe Quelle"),
    "RdyKonsensZiel":  (13.36, "stockanalysis.com/stocks/rdy/forecast, Konsens aus "
                               "5 Schaetzungen (S&P Global), abgerufen 18.09.2026"),
    "RdyKonsensAufschlag": (9.60, "dieselbe Quelle"),
    "RdyKonsensHaeuser": (5, "dieselbe Quelle"),
    "Wechselkurs": (95.94, "Europaeische Zentralbank, Referenzkurse 17.09.2026, "
                           "Kreuzkurs aus EUR/USD 1,1481 und EUR/INR 110,1475"),
}


def _kurse() -> dict:
    with open(os.path.join(ROOT, "data", "kurs_stand.json")) as f:
        return json.load(f)


# --------------------------------------------------------------------------
# Abgeleitete Groessen. Ausdruecke werden der Reihe nach ausgewertet und
# duerfen auf vorher berechnete zugreifen.
# --------------------------------------------------------------------------
ABGELEITET = [
    # --- AstraZeneca ----------------------------------------------------
    ("AznUmsatzWachstum",   "(AznUmsatzXXV / AznUmsatzXXIV - 1) * 100"),
    ("AznErgebnisWachstum", "(AznJahresueberschussXXV / AznJahresueberschussXXIV - 1) * 100"),
    ("AznEbitdaMarge",      "AznEbitda / AznUmsatzXXV * 100"),
    ("AznEbitdaMargeVJ",    "AznEbitdaVJ / AznUmsatzXXIV * 100"),
    ("AznCoreMarge",        "AznCoreBetriebsergebnis / AznUmsatzXXV * 100"),
    ("AznBetriebsMarge",    "AznBetriebsergebnisXXV / AznUmsatzXXV * 100"),
    ("AznEigenkapitalrendite", "AznJahresueberschussXXV / AznEigenkapital * 100"),
    ("AznEigenkapitalquote", "AznEigenkapital / AznBilanzsumme * 100"),
    ("AznVerschuldungsgrad", "AznNettoverschuldung / AznEbitda"),
    ("AznBuchwertJeAktie",  "AznEigenkapital / (AznAktienAusgegeben / 1e6)"),
    ("AznKursGewinn",       "AznKurs / AznErgebnisJeAktieXXV"),
    ("AznKursGewinnCore",   "AznKurs / AznCoreErgebnisJeAktie"),
    ("AznKursBuchwert",     "AznKurs / AznBuchwertJeAktie"),
    ("AznDividendenrendite", "AznDividendeJeAktie / AznKurs * 100"),
    ("AznMarktwert",        "AznKurs * AznAktienAusgegeben / 1e6"),
    ("AznMarktwertMrd",     "AznMarktwert / 1000"),
    ("AznForschungsquote",  "AznForschung / AznUmsatzXXV * 100"),
    ("AznBetriebsergebnisWachstum",
     "(AznBetriebsergebnisXXV / AznBetriebsergebnisXXIV - 1) * 100"),
    ("AznCoreWachstum",     "(AznCoreBetriebsergebnis / AznCoreBetriebsergebnisVJ - 1) * 100"),
    ("AznEbitdaWachstum",   "(AznEbitda / AznEbitdaVJ - 1) * 100"),
    ("AznEpsWachstum",      "(AznErgebnisJeAktieXXV / AznErgebnisJeAktieXXIV - 1) * 100"),
    ("AznNettoverschuldungWachstum",
     "(AznNettoverschuldung / AznNettoverschuldungVJ - 1) * 100"),
    ("AznEigenkapitalWachstum", "(AznEigenkapital / AznEigenkapitalVJ - 1) * 100"),
    # --- Dr. Reddy's ----------------------------------------------------
    ("RdyUmsatzWachstum",   "(RdyUmsatzXXVI / RdyUmsatzXXV - 1) * 100"),
    ("RdyErgebnisWachstum", "(RdyJahresueberschussXXVI / RdyJahresueberschussXXV - 1) * 100"),
    ("RdyRohertragsmarge",  "RdyRohertragXXVI / RdyUmsatzXXVI * 100"),
    ("RdyRohertragsmargeVJ", "RdyRohertragXXV / RdyUmsatzXXV * 100"),
    ("RdyBetriebsMarge",    "RdyBetriebsergebnis / RdyUmsatzXXVI * 100"),
    ("RdyBetriebsMargeVJ",  "RdyBetriebsergebnisVJ / RdyUmsatzXXV * 100"),
    ("RdyEigenkapitalrendite", "RdyJahresueberschussXXVI / RdyEigenkapital * 100"),
    ("RdyEigenkapitalquote", "RdyEigenkapital / RdyBilanzsumme * 100"),
    ("RdyFinanzschulden",   "RdySchuldenKurz + RdySchuldenLangKurz + RdySchuldenLang"),
    ("RdyNettokasse",       "RdyLiquiditaet + RdyWertpapiere - RdyFinanzschulden"),
    ("RdyBuchwertJeAktie",  "RdyEigenkapital / (RdyAktienAusgegeben / 1e6)"),
    ("RdyKursRupien",       "RdyKurs * Wechselkurs"),
    ("RdyKursGewinn",       "RdyKursRupien / RdyErgebnisJeAktie"),
    ("RdyKursBuchwert",     "RdyKursRupien / RdyBuchwertJeAktie"),
    ("RdyDividendenrendite", "RdyDividendeJeAktie / RdyKursRupien * 100"),
    ("RdyAusschuettungsquote", "RdyDividendeJeAktie / RdyErgebnisJeAktie * 100"),
    ("RdyMarktwertRupien",  "RdyKursRupien * RdyAktienAusgegeben / 1e6"),
    ("RdyMarktwertMrd",     "RdyMarktwertRupien / 1000"),
    ("RdyForschungsquote",  "RdyForschungXXVI / RdyUmsatzXXVI * 100"),
    ("RdySsaAnteil",        "RdySsaEffekt / RdyUmsatzXXVI * 100"),
    # --- Kursbezogene Groessen ------------------------------------------
    ("AznCeoVerguetungMio", "AznCeoVerguetung / 1000"),
    ("AznCfoVerguetungMio", "AznCfoVerguetung / 1000"),
    ("AznAbstandHoch",      "(AznKurs / AznKursHochXXVI - 1) * 100"),
    ("AznRenditeLaufend",   "(AznKurs / AznKursSchlussXXV - 1) * 100"),
    ("AznRenditeZehn",      "((AznKurs / AznKursSchlussXVI) ** 0.1 - 1) * 100"),
    ("RdyAbstandHoch",      "(RdyKurs / RdyKursHochXXVI - 1) * 100"),
    ("RdyRenditeLaufend",   "(RdyKurs / RdyKursSchlussXXV - 1) * 100"),
    ("RdyRenditeZehn",      "((RdyKurs / RdyKursSchlussXVI) ** 0.1 - 1) * 100"),
    # --- Segment- und Marktveraenderungen -------------------------------
    ("AznOnkologieWachstum", "(AznOnkologie / AznOnkologieVJ - 1) * 100"),
    ("AznSelteneWachstum",  "(AznSeltene / AznSelteneVJ - 1) * 100"),
    ("AznCvrmWachstum",     "(AznCvrm / AznCvrmVJ - 1) * 100"),
    ("AznRespImmunWachstum", "(AznRespImmun / AznRespImmunVJ - 1) * 100"),
    ("AznImpfstoffeWachstum", "(AznImpfstoffe / AznImpfstoffeVJ - 1) * 100"),
    ("AznAmbitionRate",     "((AznAmbition * 1000 / AznUmsatzXXV) ** 0.2 - 1) * 100"),
    # Bewertung am Jahreshoch, gerechnet auf dasselbe bereinigte Ergebnis je
    # Aktie wie heute. Nur so trennt die Groesse Bewertung von Ertrag; mit
    # einem anderen Nenner waere sie eine Mischung aus beidem.
    ("AznKgvHoch",          "AznKursHochXXVI / AznCoreErgebnisJeAktie"),
    ("AznUsaAnteil",        "AznUmsatzUsa / AznUmsatzXXV * 100"),
    ("AznChinaAnteil",      "AznUmsatzChina / AznUmsatzXXV * 100"),
    ("AznFarxigaAnteil",    "AznFarxiga / AznUmsatzXXV * 100"),
    ("AznFarxigaHJWachstum", "(AznFarxigaHJ / (AznFarxiga / 2) - 1) * 100"),
    ("RdyNordamerikaWachstum", "(RdyNordamerika / RdyNordamerikaVJ - 1) * 100"),
    ("RdyEuropaWachstum",   "(RdyEuropa / RdyEuropaVJ - 1) * 100"),
    ("RdyIndienWachstum",   "(RdyIndien / RdyIndienVJ - 1) * 100"),
    ("RdyOnkologieWachstum", "(RdyOnkologie / RdyOnkologieVJ - 1) * 100"),
    ("RdyOnkologieRueckgang", "RdyOnkologieVJ - RdyOnkologie"),
    ("RdyNordamerikaAnteil", "RdyNordamerika / RdyGlobalGenerics * 100"),
    ("RdyUsaAnteil",        "RdyUmsatzUsa / RdyUmsatzXXVI * 100"),
    ("RdyQuartalGewinnWachstum", "(RdyQuartalGewinn / RdyQuartalGewinnVJ - 1) * 100"),
    ("RdyNrtWachstum",      "(RdyNrtUmsatz / RdyNrtUmsatzVJ - 1) * 100"),
    ("RdyGlobalGenericsAnteil", "RdyGlobalGenerics / RdyUmsatzXXVI * 100"),
    ("RdyPsaiWachstum",     "(RdyPsai / RdyPsaiVJ - 1) * 100"),
    ("RdyRohertragGgWachstum", "(RdyRohertragGg / RdyRohertragGgVJ - 1) * 100"),
    ("RdyBetriebsergebnisWachstum",
     "(RdyBetriebsergebnis / RdyBetriebsergebnisVJ - 1) * 100"),
    ("RdyEpsWachstum",      "(RdyErgebnisJeAktie / RdyErgebnisJeAktieVJ - 1) * 100"),
    ("RdyForschungWachstum", "(RdyForschungXXVI / RdyForschungXXV - 1) * 100"),
    ("RdyEigenkapitalWachstum", "(RdyEigenkapital / RdyEigenkapitalVJ - 1) * 100"),
    ("RdyKgvHoch",          "RdyKursHochXXVI * Wechselkurs / RdyErgebnisJeAktie"),
    ("RdyOnkologieAnteilVJ", "RdyOnkologieVJ / RdyUmsatzXXV * 100"),
    ("RdyRohertragWachstum", "(RdyRohertragXXVI / RdyRohertragXXV - 1) * 100"),
    ("RdyGlobalGenericsWachstum", "(RdyGlobalGenerics / RdyGlobalGenericsVJ - 1) * 100"),
    ("RdyNordamerikaRueckgang", "RdyNordamerikaVJ - RdyNordamerika"),
    ("RdyEuropaZuwachs",    "RdyEuropa - RdyEuropaVJ"),
    ("RdyInvestitionen",    "-(RdySachInvestitionXXVI + RdyImmatInvestitionXXVI)"),
    ("RdyQuartalCfBetrag",  "-RdyQuartalCf"),
    # --- Pruefung der Sekundaerdaten -------------------------------------
    ("AznAnteileGemeldet",  "AznBlackRock + AznCapitalGroup + AznWellington + AznInvestorAB"),
    ("AznStreubesitz",      "100 - AznAnteileGemeldet"),
    ("RdyStreubesitz",      "100 - RdyPromoterAnteil"),
    ("RdyStreubesitzOhneLic", "100 - RdyPromoterAnteil - RdyLicAnteil"),
    ("RdySumme",            "RdyPromoterAnteil + RdyLicAnteil + RdyStreubesitzOhneLic"),
    ("AznKonsensBezug",     "AznKonsensZiel / (1 + AznKonsensAufschlag / 100)"),
    ("RdyKonsensBezug",     "RdyKonsensZiel / (1 + RdyKonsensAufschlag / 100)"),
]


def rechne() -> dict:
    """Wertetafel aus ROH, EXTERN, den Kursreihen und ABGELEITET."""
    tafel = {k: v for k, (v, _) in ROH.items()}
    tafel.update({k: v for k, (v, _) in EXTERN.items()})
    tafel.update({k: v for k, (v, _) in reihen_roh().items()})
    # Kursgroessen: Jahresschluss, Jahreshoch und Jahrestief je Titel. Sie
    # stammen aus data/kurs_stand.json, erzeugt von scripts/kurse.py, und sind
    # Sekundaerdaten wie die Kurse selbst.
    for kuerzel, vorsatz in (("azn", "Azn"), ("rdy", "Rdy")):
        k = _kurse()[kuerzel]
        for feld, name in (("jahre", "KursSchluss"), ("hoch", "KursHoch"),
                           ("tief", "KursTief")):
            for jahr, wert in k[feld].items():
                tafel[f"{vorsatz}{name}{roemisch(int(jahr))}"] = wert
    for name, ausdruck in ABGELEITET:
        tafel[name] = eval(ausdruck, {"__builtins__": {}}, tafel)   # noqa: S307
    return tafel


def main() -> int:
    tafel = rechne()
    kurse = _kurse()
    zeilen = ["% erzeugt von scripts/kennzahlen.py - NICHT von Hand aendern",
              "% Rohwerte: siehe ROH in scripts/kennzahlen.py, je mit gedruckter Seite",
              ""]
    # Der blanke Wert, nicht \num{...}: Die Ausgabemakros (\Pct, \MioUSD,
    # \USDje) legen selbst ein \num darum, und ein zweites innen bricht
    # siunitx mit "Invalid number" ab.
    for name in sorted(tafel):
        wert = tafel[name]
        herkunft = ROH.get(name) or EXTERN.get(name)
        beleg = f"% {herkunft[1]}" if herkunft else "% abgeleitet"
        zeilen.append(f"\\newcommand{{\\{name}}}{{{wert:.6g}}}{beleg}")
    # Der Stichtag der Kursreihe ist kein Zahlenwert, sondern ein Datum; er
    # steht deshalb hier und nicht in der Tafel. Die Kursgroessen selbst
    # kommen bereits aus rechne() und duerfen hier NICHT ein zweites Mal
    # ausgegeben werden - LaTeX bricht bei einem doppelten \newcommand ab.
    for kuerzel, vorsatz in (("azn", "Azn"), ("rdy", "Rdy")):
        # Deutsches Datumsformat: Die Reihe liefert das Datum in ISO-Form,
        # und "2026-09-17" mitten im deutschen Satz ist ein Fremdkoerper.
        jahr, monat, tag = str(kurse[kuerzel]["letzter_tag"]).split("-")
        zeilen.append(f"\\newcommand{{\\{vorsatz}KursTag}}{{{tag}.{monat}.{jahr}}}")
    ziel = os.path.join(ROOT, "data", "kennzahlen.tex")
    open(ziel, "w").write("\n".join(zeilen) + "\n")
    print(f"[KENNZAHLEN] {len(ROH)} Einzelwerte, {len(reihen_roh())} Reihenwerte, "
          f"{len(EXTERN)} Sekundaerwerte, {len(ABGELEITET)} abgeleitet "
          f"-> data/kennzahlen.tex")
    return 0


if __name__ == "__main__":
    sys.exit(main())
