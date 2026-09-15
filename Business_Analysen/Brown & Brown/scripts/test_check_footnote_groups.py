#!/usr/bin/env python3
"""
Tests fuer check_footnote_groups.py - Erkennung von Fussnotengruppen, deren
Wiederholung nach einem verschobenen Seitenumbruch nicht mehr auf der Seite
ihrer Primaerzitation steht. Deckt nur die reinen Funktionen ab
(parse_groups, find_offset_groups); resolve_page ruft SyncTeX gegen eine
gebaute PDF auf und ist damit build-abhaengig, nicht Teil dieser schnellen,
offline laufenden Suite (vgl. testing.md).
"""
import os
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
                        "Text.\\quelleGB{2024}{10}\\quelleMerken{k1}\n"
                        "Mehr.\\quelleErneut{k1}\n")
            groups = parse_groups(tmp)
        self.assertEqual(groups["k1"],
                         [("Merken", "a.tex", 1), ("Erneut", "a.tex", 2)])

    def test_kommentarzeilen_werden_ignoriert(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write(tmp, "a.tex", "% \\quelleMerken{k1}\nText.\\quelleErneut{k2}\n")
            groups = parse_groups(tmp)
        self.assertNotIn("k1", groups)
        self.assertEqual(len(groups["k2"]), 1)


class TestFindOffsetGroups(unittest.TestCase):
    def test_gruppe_auf_einer_seite_ist_fehlerfrei(self):
        dated = {"k1": [("Merken", "a.tex", 1, 9), ("Erneut", "a.tex", 5, 9)]}
        self.assertEqual(find_offset_groups(dated), {})

    def test_wiederholung_auf_folgeseite_wird_gemeldet(self):
        dated = {"k1": [("Merken", "a.tex", 1, 9), ("Erneut", "a.tex", 5, 10)]}
        fehler = find_offset_groups(dated)
        self.assertEqual(fehler["k1"][0], 9)
        self.assertEqual(fehler["k1"][1], [("Erneut", "a.tex", 5, 10)])

    def test_erneut_ohne_anker_wird_gemeldet(self):
        dated = {"k1": [("Erneut", "a.tex", 5, 10)]}
        self.assertEqual(find_offset_groups(dated)["k1"][0], -1)


if __name__ == "__main__":
    unittest.main()
