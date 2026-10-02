"""
Tests fuer check_literals.py - Wachhund gegen hartkodierte Zahlen
in sections/*.tex. Reine Standardbibliothek.
"""
import unittest
from check_literals import find_literals


class TestFindLiterals(unittest.TestCase):
    def test_flags_decimal_number(self):
        hits = find_literals(r"Der Umsatz betrug 246,8 Mio. EUR.")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 1)

    def test_flags_large_integer(self):
        hits = find_literals(r"Der Auftragseingang lag bei 2135 TEUR.")
        self.assertEqual(len(hits), 1)

    def test_flags_german_thousands_integer(self):
        # Deutsches Tausendertrennzeichen (Punkt) darf die Erkennung nicht
        # umgehen: 12.345 muss wie 12345 als hartkodiert gelten.
        hits = find_literals(r"Der Betrag lag bei 12.345 EUR.")
        self.assertEqual(len(hits), 1)

    def test_flags_german_thousands_integer_six_digits(self):
        hits = find_literals(r"Bilanzsumme: 246.800 EUR.")
        self.assertEqual(len(hits), 1)

    def test_flags_german_thousands_with_decimal(self):
        # Tausenderpunkt UND Dezimalkomma zugleich: 1.234,5 ist ein
        # einziger Treffer, nicht zwei.
        hits = find_literals(r"Der Umsatz betrug 1.234,5 Mio. EUR.")
        self.assertEqual(len(hits), 1)

    def test_allows_year(self):
        hits = find_literals(r"Im Gesch\"aftsjahr 2024 sowie 2026 und 1998.")
        self.assertEqual(hits, [])

    def test_allows_macro_call(self):
        hits = find_literals(r"Der Umsatz betrug \MioEUR{\UmsatzZFV}.")
        self.assertEqual(hits, [])

    def test_allows_label_and_ref(self):
        hits = find_literals(r"\label{tab:cf-a} siehe Tabelle \ref{tab:cf-a}")
        self.assertEqual(hits, [])

    def test_allows_generated_table_body(self):
        # \tabellenkoerper liest eine erzeugte Tabelle aus data/ ein; der
        # Dateiname ist ein Pfad, kein Zahlenwert des Dokuments.
        hits = find_literals(r"\tabellenkoerper{data/vergleich2028}")
        self.assertEqual(hits, [])

    def test_allows_digits_inside_url_argument(self):
        # Das zweite Argument von \quelleWeb ist eine URL und steht nicht in
        # \url{}. Archiv- und Nachrichtenadressen enthalten lange
        # Ziffernfolgen, die kein Zahlenwert des Dokuments sind.
        hits = find_literals(
            r"Text.\quelleWeb{Internet Archive}"
            r"{http://web.archive.org/web/20250914061406/https://x.example/}"
            r"{31.07.2026}")
        self.assertEqual(hits, [])

    def test_url_argument_does_not_mask_number_in_same_line(self):
        # Die Ausnahme darf nur das Klammerargument entfernen, nicht die Zeile.
        hits = find_literals(
            r"Der Umsatz betrug 246,8 Mio.\quelleWeb{Q}{https://x.example/1234}{31.07.2026}")
        self.assertEqual(hits, [(1, "246,8")])

    def test_ignores_comment_lines(self):
        hits = find_literals("% FAKTENBASIS: Umsatz 246,8 Mio. EUR\nText ohne Zahl.")
        self.assertEqual(hits, [])

    def test_ignores_inline_comment_tail(self):
        hits = find_literals(r"Text ohne Zahl. % Beleg: 246,8 Mio. EUR")
        self.assertEqual(hits, [])

    def test_respects_escaped_percent(self):
        hits = find_literals(r"Die Marge stieg um 12,5 \% gegen\"uber dem Vorjahr.")
        self.assertEqual(len(hits), 1)

    def test_reports_correct_line_number(self):
        text = "Erste Zeile.\nZweite Zeile.\nDritte Zeile mit 246,8 Mio.\n"
        hits = find_literals(text)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0][0], 3)

    def test_allows_small_integers(self):
        # Aufzaehlungen wie "drei Optionen (A, B, C)" oder "Abschnitt 5.4"
        # sollen nicht anschlagen.
        hits = find_literals(r"Siehe Abschnitt 5.4 sowie die 3 Optionen.")
        self.assertEqual(hits, [])

    def test_skip_marker_suppresses_all_hits(self):
        # Eine Datei mit dem Marker "% check-literals: skip" darf auch eine
        # sonst anschlagende Zahl enthalten (z.B. eine Matrikelnummer).
        text = "% check-literals: skip\nMatrikelnummer: 885893\n"
        hits = find_literals(text)
        self.assertEqual(hits, [])

    def test_without_skip_marker_number_still_flagged(self):
        # Gleicher Text ohne den Marker: die Zahl muss weiterhin anschlagen,
        # damit der vorige Test wirklich den Marker prueft und nicht nur
        # eine zufaellig unerkannte Zahl.
        text = "Matrikelnummer: 885893\n"
        hits = find_literals(text)
        self.assertEqual(len(hits), 1)


if __name__ == "__main__":
    unittest.main()
