import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import presse   # noqa: E402


class TestZitate(unittest.TestCase):
    def test_nur_woertliche_zitate(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, "a.md"), "w", encoding="utf-8") as f:
                f.write("Die Dividende ist   kein Zins.\nMehr Text.")
            eintraege = [{"id": "a", "zitat": "Die Dividende ist kein Zins."},
                         {"id": "a", "zitat": "Dividenden sind sicher."}]
            ok, raus = presse.pruefen(eintraege, d)
            self.assertEqual([e["zitat"] for e in ok], ["Die Dividende ist kein Zins."])
            self.assertEqual(len(raus), 1)

    def test_fehlende_refs_datei_wird_abgelehnt(self):
        # Une entree dont le fichier refs/presse/<id>.md n'existe pas doit etre
        # rejetee proprement (dans "raus"), jamais lever d'exception.
        with tempfile.TemporaryDirectory() as d:
            eintraege = [{"id": "inexistant", "zitat": "Une citation quelconque."}]
            ok, raus = presse.pruefen(eintraege, d)
            self.assertEqual(ok, [])
            self.assertEqual(len(raus), 1)
            self.assertEqual(raus[0]["id"], "inexistant")

    def test_id_hors_motif_leve_une_erreur(self):
        # Revue finale, constat 7: "id" sert tel quel dans os.path.join(rohordner,
        # f"{id}.md") (presse.py) et dans \csname/\label (rechnung_retraite.py). Un id
        # qui sort de ^[a-z0-9_]+$ (traversee de chemin, ou caractere qui casserait
        # \csname/\label a la compilation) doit etre refuse avant le join/l'usage
        # LaTeX, jamais silencieusement joint ou imprime.
        with tempfile.TemporaryDirectory() as d:
            eintraege = [{"id": "../../etc/passwd", "zitat": "Une citation."}]
            with self.assertRaises(ValueError):
                presse.pruefen(eintraege, d)


if __name__ == "__main__":
    unittest.main()
