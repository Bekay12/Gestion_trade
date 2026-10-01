#!/usr/bin/env python3
"""
Tests fuer check_footnote_groups.py - Erkennung von Fussnotengruppen, deren
Wiederholung nach einem verschobenen Seitenumbruch nicht mehr auf der Seite
ihrer Primaerzitation steht. Deckt nur die reinen Funktionen ab
(parse_groups, find_offset_groups); resolve_page ruft SyncTeX gegen eine
gebaute PDF auf und ist damit build-abhaengig, nicht Teil dieser schnellen,
offline laufenden Suite (vgl. testing.md).
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import tempfile
import unittest

from check_footnote_groups import find_offset_groups, parse_groups


class TestParseGroups(unittest.TestCase):
    def _write(self, tmpdir, name, content):
        path = os.path.join(tmpdir, name)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)

    def test_merken_und_erneut_werden_dem_schluessel_zugeordnet(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "a.tex",
                        "Text.\\QL{k1}{GBXXV}{10}\n"
                        "Mehr.\\QR{k1}\n")
            groups = parse_groups(tmp)
        self.assertEqual(groups["k1"],
                         [("QL", "a.tex", 1), ("QR", "a.tex", 2)])

    def test_kommentarzeilen_werden_ignoriert(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "a.tex", "% \\QL{k1}{HJ}{2}\nText.\\QR{k2}\n")
            groups = parse_groups(tmp)
        self.assertNotIn("k1", groups)
        self.assertEqual(len(groups["k2"]), 1)

    def test_manueller_label_anker_fuer_qh_wird_wie_ql_erkannt(self):
        # tache 10: \QH/\QP setzen selbst kein \label; wer eine Wiederholung
        # auf derselben Seite buendeln will, ankert manuell mit
        # \footnote{\label{fn:<anker>}\QHtexte{<cle>}} - siehe Kommentar bei
        # \QH in preamble.tex und _ANCHOR in diesem Skript.
        with tempfile.TemporaryDirectory() as tmp:
            self._write(
                tmp, "a.tex",
                "Text.\\footnote{\\label{fn:monancre}\\QHtexte{kv_satz_ermaessigt}}\n"
                "Mehr.\\QR{monancre}\n")
            groups = parse_groups(tmp)
        self.assertEqual(groups["monancre"],
                         [("QL", "a.tex", 1), ("QR", "a.tex", 2)])

    def test_pourcentage_echappe_ne_coupe_pas_la_ligne(self):
        # tache 10 (chapitres 02-03): "\%" est un pourcentage, pas un commentaire;
        # la repetition qui le suit sur la meme ligne doit rester visible.
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "a.tex",
                        "Taux 25~\\%.\\QL{k1}{GBXXV}{10}\n"
                        "Encore 5~\\%\\QR{k1} % commentaire \\QR{k2}\n")
            groups = parse_groups(tmp)
        self.assertEqual(groups["k1"], [("QL", "a.tex", 1), ("QR", "a.tex", 2)])
        self.assertNotIn("k2", groups)


class TestFindOffsetGroups(unittest.TestCase):
    def test_gruppe_auf_einer_seite_ist_fehlerfrei(self):
        dated = {"k1": [("QL", "a.tex", 1, 9), ("QR", "a.tex", 5, 9)]}
        self.assertEqual(find_offset_groups(dated), {})

    def test_wiederholung_auf_folgeseite_wird_gemeldet(self):
        dated = {"k1": [("QL", "a.tex", 1, 9), ("QR", "a.tex", 5, 10)]}
        fehler = find_offset_groups(dated)
        self.assertEqual(fehler["k1"][0], 9)
        self.assertEqual(fehler["k1"][1], [("QR", "a.tex", 5, 10)])

    def test_erneut_ohne_anker_wird_gemeldet(self):
        dated = {"k1": [("QR", "a.tex", 5, 10)]}
        self.assertEqual(find_offset_groups(dated)["k1"][0], -1)


if __name__ == "__main__":
    unittest.main()
