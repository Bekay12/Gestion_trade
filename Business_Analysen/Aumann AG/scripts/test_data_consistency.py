"""
Tests gegen das Auseinanderlaufen der Zahlenschicht.

Acht Kurswerte stehen technisch erzwungen zweimal im Repo: als Makro in
data/kennzahlen.tex (fuer Fliesstext und Tabellen) und als Spalte in einer
CSV-Datei (fuer pgfplots, das keine LaTeX-Makros lesen kann). Die Regel in
data/CLAUDE.md - jeder Zahlenwert genau einmal - laesst sich hier nicht
einhalten; diese Tests sichern die Doppelung stattdessen ab.

Reine Standardbibliothek, kein Netz, kein LaTeX-Lauf, kein SyncTeX.
"""
import csv
import os
import re
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KENNZAHLEN = os.path.join(ROOT, "data", "kennzahlen.tex")

_NEWCOMMAND = re.compile(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}")


def parse_macros(path: str) -> dict:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest die \\newcommand-Definitionen einer .tex-Datei in ein
        Dictionary. Kommentarzeilen und Zeilenkommentare hinter der
        Definition werden ignoriert.

    Inputs:
        path (str): Pfad zu data/kennzahlen.tex.

    Outputs:
        macros (dict[str, str]): Makroname ohne Backslash -> Rohwert.
    --------------------------------------------------------------------------
    """
    macros = {}
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            code = raw.split("%")[0]
            for name, value in _NEWCOMMAND.findall(code):
                macros[name] = value
    return macros


def _norm(value: str) -> str:
    """
    --------------------------------------------------------------------------
    Purpose:
        Vereinheitlicht einen Kurswert auf zwei Nachkommastellen, damit
        "10.1", "10.10" und "10.100" als gleich gelten.

    Inputs:
        value (str): Rohwert aus Makro oder CSV-Spalte.

    Outputs:
        normalised (str): Wert mit genau zwei Nachkommastellen.
    --------------------------------------------------------------------------
    """
    return f"{float(value):.2f}"


def _read_csv(name: str) -> list:
    """
    --------------------------------------------------------------------------
    Purpose:
        Liest eine CSV-Datei aus data/ und ueberspringt die
        Kopfkommentarzeilen, die mit "#" beginnen.

    Inputs:
        name (str): Dateiname innerhalb von data/.

    Outputs:
        rows (list[dict]): Zeilen als Dictionary je Spaltenname.
    --------------------------------------------------------------------------
    """
    path = os.path.join(ROOT, "data", name)
    with open(path, encoding="utf-8") as fh:
        nutzdaten = (z for z in fh if not z.lstrip().startswith("#"))
        return list(csv.DictReader(nutzdaten))


class TestParseMacros(unittest.TestCase):
    def test_parse_macros_ignoriert_kommentare(self):
        import tempfile
        text = ("% \\newcommand{\\Auskommentiert}{99.99}\n"
                "\\newcommand{\\KursA}{10.10}   % GB 2024, S. 13\n"
                "\\newcommand{\\KursB}{9.42}\n")
        with tempfile.TemporaryDirectory() as tmp:
            pfad = os.path.join(tmp, "k.tex")
            with open(pfad, "w", encoding="utf-8") as fh:
                fh.write(text)
            macros = parse_macros(pfad)
        self.assertEqual(macros, {"KursA": "10.10", "KursB": "9.42"})


class TestZahlenschichtKonsistenz(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.macros = parse_macros(KENNZAHLEN)

    # Roemische Jahresziffern des Projekts: ZFI = 2021 ... ZFV = 2025
    JAHRE = {"2021": "ZFI", "2022": "ZFII", "2023": "ZFIII",
             "2024": "ZFIV", "2025": "ZFV"}

    # Erwartete Stuetzpunkte: Schlusskurs 2021 als Ausgangspunkt, danach je
    # Jahr 2022-2025 Hoch, Tief und Schlusskurs.
    ARTEN = {"S": "KursSilvester", "H": "KursHoch", "T": "KursTief"}

    def test_kurse_stimmen_mit_makros(self):
        # Fuer 2022 bis 2025 tragen die \KursSilvester-Makros die Werte der
        # Geschaeftsberichte (Quelle und gedruckte Seite im Zeilenkommentar
        # von kennzahlen.tex). Dieser Test ist damit zugleich die Gegenprobe,
        # die die externe Kursquelle ueberhaupt zitierfaehig macht: Weicht
        # die Yahoo-Reihe von den Berichten ab, faellt er.
        rows = _read_csv("kursverlauf.csv")
        # Ohne diese Zusicherung bestuende der Test auch bei leerer Datei.
        self.assertEqual(len(rows), 13, "dreizehn Stuetzpunkte erwartet")
        for row in rows:
            makro = self.ARTEN[row["art"]] + self.JAHRE[row["jahr"]]
            self.assertEqual(
                _norm(row["kurs"]), _norm(self.macros[makro]),
                f"kursverlauf.csv {row['datum']} ({row['art']}) weicht von "
                f"\\{makro} ab")

    def test_alle_makros_kommen_in_der_abbildung_vor(self):
        # Gegenrichtung des vorigen Tests: Ein Makro, das die Tabelle zeigt,
        # die Abbildung aber nicht, faellt sonst niemandem auf.
        rows = _read_csv("kursverlauf.csv")
        vorhanden = {self.ARTEN[r["art"]] + self.JAHRE[r["jahr"]] for r in rows}
        erwartet = {"KursSilvesterZFI"}
        for ziffer in ("ZFII", "ZFIII", "ZFIV", "ZFV"):
            erwartet |= {p + ziffer for p in self.ARTEN.values()}
        self.assertEqual(vorhanden, erwartet)

    def test_stuetzpunkte_sind_streng_nach_datum_geordnet(self):
        # Die Invariante der Abbildung. In 2023 und 2025 lag das Tief vor dem
        # Hoch; wer die Zeilen nach dem Schema Hoch-Tief sortiert, laesst die
        # Linie rueckwaerts laufen - pgfplots zeichnet das kommentarlos.
        rows = _read_csv("kursverlauf.csv")
        daten = [r["datum"] for r in rows]
        self.assertEqual(daten, sorted(daten), "Datumsreihenfolge verletzt")
        self.assertEqual(len(set(daten)), len(daten), "Datum doppelt vergeben")
        # Die x-Koordinate muss dieselbe Ordnung tragen wie das Datum.
        xs = [float(r["x"]) for r in rows]
        self.assertEqual(xs, sorted(xs), "x-Koordinate folgt nicht dem Datum")
        for row in rows:
            self.assertEqual(
                row["datum"][:4], row["jahr"],
                f"{row['datum']}: Datum liegt nicht im Jahr {row['jahr']}")


if __name__ == "__main__":
    unittest.main()
