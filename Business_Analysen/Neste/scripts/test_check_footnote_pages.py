#!/usr/bin/env python3
"""
Tests fuer check_footnote_pages.py - Erkennung wiederholter Fussnoten auf
derselben gedruckten Seite. Deckt nur die reinen Funktionen ab
(parse_citations, group_duplicates); resolve_page ruft SyncTeX gegen eine
gebaute PDF auf und ist damit build-abhaengig, nicht Teil dieser schnellen,
offline laufenden Suite (vgl. testing.md).
"""
import os
import tempfile
import unittest

from check_footnote_pages import group_duplicates, parse_citations


class TestParseCitations(unittest.TestCase):
    def _write(self, tmpdir, name, content):
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
        return path

    def test_findet_gb_aufruf_mit_zeile(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex", "Text.\\Q{GBXXIV}{2} Weiter.\n")
            hits = parse_citations(tmp)
        self.assertEqual(hits, [("GBXXIV S.2", "01-x.tex", 1)])

    def test_findet_mehrere_aufrufe_auf_verschiedenen_zeilen(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex",
                        "Erste.\\Q{GBXXIV}{2}\nZweite.\\Q{GBXXIV}{5}\n")
            hits = parse_citations(tmp)
        self.assertEqual([h[2] for h in hits], [1, 2])

    def test_ignoriert_kommentarzeile(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex",
                        "% FAKTENBASIS: \\Q{GBXXIV}{2}\nEcht.\\Q{GBXXIV}{5}\n")
            hits = parse_citations(tmp)
        self.assertEqual(hits, [("GBXXIV S.5", "01-x.tex", 2)])

    def test_quellefs_und_quelleweb_erzeugen_eigene_schluessel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "01-x.tex",
                "A.\\QL{k}{HJ}{2}\n"
                "B.\\QW{Titel}{quelle:x}{29.07.2026}\n")
            hits = parse_citations(tmp)
        self.assertEqual(
            [h[0] for h in hits],
            ["HJ S.2",
             "Web quelle:x"])

    def test_verschiedene_webquellen_erzeugen_verschiedene_schluessel(self):
        # Zwei verschiedene Webquellen auf derselben Seite sind kein
        # Duplikat - sie drucken zwei verschiedene Fussnotentexte.
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "01-x.tex",
                "A.\\QW{Kurse}{quelle:kurse}{31.07.2026}\n"
                "B.\\QW{Konsens}{quelle:konsens}{31.07.2026}\n")
            hits = parse_citations(tmp)
        keys = [h[0] for h in hits]
        self.assertEqual(len(set(keys)), 2)

    def test_gleiche_webquelle_erzeugt_gleichen_schluessel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "01-x.tex",
                "A.\\QW{Kurse}{quelle:kurse}{31.07.2026}\n"
                "B.\\QW{Kurse}{quelle:kurse}{31.07.2026}\n")
            hits = parse_citations(tmp)
        keys = [h[0] for h in hits]
        self.assertEqual(len(set(keys)), 1)

    def test_mehrere_aufrufe_in_einer_zeile(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex",
                        "\\Q{GBXXIII}{14}\\Q{GBXXIV}{13}\\Q{HJ}{13}\n")
            hits = parse_citations(tmp)
        self.assertEqual(len(hits), 3)
        self.assertTrue(all(h[2] == 1 for h in hits))


class TestGroupDuplicates(unittest.TestCase):
    def test_keine_duplikate_ohne_wiederholung(self):
        entries = [(3, "GB 2024 S.2", "a.tex", 1), (4, "GB 2024 S.5", "a.tex", 2)]
        self.assertEqual(group_duplicates(entries), {})

    def test_erkennt_duplikat_auf_derselben_seite(self):
        entries = [
            (9, "GB 2024 S.10", "a.tex", 11),
            (9, "GB 2024 S.10", "a.tex", 25),
        ]
        dup = group_duplicates(entries)
        self.assertIn(9, dup)
        self.assertEqual(dup[9]["GB 2024 S.10"], [("a.tex", 11), ("a.tex", 25)])

    def test_ignoriert_gleichen_schluessel_auf_verschiedenen_seiten(self):
        entries = [
            (10, "GB 2024 S.5", "a.tex", 113),
            (11, "GB 2024 S.5", "a.tex", 131),
        ]
        self.assertEqual(group_duplicates(entries), {})

    def test_mehrere_schluessel_auf_derselben_seite_unabhaengig(self):
        entries = [
            (9, "GB 2024 S.9", "a.tex", 1),
            (9, "GB 2024 S.9", "a.tex", 2),
            (9, "GB 2024 S.10", "a.tex", 3),
        ]
        dup = group_duplicates(entries)
        self.assertEqual(set(dup[9]), {"GB 2024 S.9"})

    def test_dreifaches_duplikat_wird_vollstaendig_gemeldet(self):
        entries = [
            (6, "GB 2024 S.2", "a.tex", 35),
            (6, "GB 2024 S.2", "a.tex", 46),
            (6, "GB 2024 S.2", "a.tex", 58),
        ]
        dup = group_duplicates(entries)
        self.assertEqual(len(dup[6]["GB 2024 S.2"]), 3)


if __name__ == "__main__":
    unittest.main()
