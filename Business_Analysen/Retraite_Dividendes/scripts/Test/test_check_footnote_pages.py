#!/usr/bin/env python3
"""
Tests fuer check_footnote_pages.py - Erkennung wiederholter Fussnoten auf
derselben gedruckten Seite. Deckt nur die reinen Funktionen ab
(parse_citations, group_duplicates); resolve_page ruft SyncTeX gegen eine
gebaute PDF auf und ist damit build-abhaengig, nicht Teil dieser schnellen,
offline laufenden Suite (vgl. testing.md).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
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

    def test_qh_aufruf_erzeugt_eigenen_schluessel(self):
        # tache 10: \QH{cle} (data/quellen.json) muss als Zitation erkannt
        # werden, sonst meldet der Pruefer nach Aufgabe 10 dauerhaft "0
        # Zitationen" (siehe Kommentar bei _CALL).
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex", "Le taux reduit\\QH{kv_satz_ermaessigt} s'applique.\n")
            hits = parse_citations(tmp)
        self.assertEqual(hits, [("QH kv_satz_ermaessigt", "01-x.tex", 1)])

    def test_citation_apres_un_pourcentage_echappe_reste_visible(self):
        # tache 10 (chapitres 02-03): l'ancien split("%") coupait la ligne au "\%"
        # d'un pourcentage et rendait invisible la citation qui le suit; un vrai
        # commentaire (% non echappe) reste ignore.
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex",
                        "Taux de 14~\\%\\QH{kv_satz_ermaessigt}. % \\QH{soli_satz}\n")
            hits = parse_citations(tmp)
        self.assertEqual(hits, [("QH kv_satz_ermaessigt", "01-x.tex", 1)])

    def test_qp_aufruf_erzeugt_eigenen_schluessel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "01-x.tex", "Selon Finanztip\\QP{finanztip_geldanlage}.\n")
            hits = parse_citations(tmp)
        self.assertEqual(hits, [("QP finanztip_geldanlage", "01-x.tex", 1)])

    def test_gleiche_qh_cle_erzeugt_gleichen_schluessel(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "01-x.tex",
                "A.\\QH{kv_satz_ermaessigt}\nB.\\QH{kv_satz_ermaessigt}\n")
            hits = parse_citations(tmp)
        keys = [h[0] for h in hits]
        self.assertEqual(len(set(keys)), 1)

    def test_qh_und_qp_und_q_koexistieren_in_derselben_zeile(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "01-x.tex",
                "\\Q{GBXXIV}{2}\\QH{kv_satz_ermaessigt}\\QP{finanztip_geldanlage}\n")
            hits = parse_citations(tmp)
        self.assertEqual(
            [h[0] for h in hits],
            ["GBXXIV S.2", "QH kv_satz_ermaessigt", "QP finanztip_geldanlage"])


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
