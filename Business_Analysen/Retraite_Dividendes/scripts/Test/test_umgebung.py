"""Verifie l'environnement avant la nuit: bibliotheques, LaTeX, donnees Shiller joignables."""
import importlib
import shutil
import unittest


class TestUmgebung(unittest.TestCase):
    def test_bibliotheques(self):
        for m in ("pandas", "numpy", "yfinance", "xlrd", "requests"):
            importlib.import_module(m)

    def test_latex(self):
        for prog in ("latexmk", "pdflatex", "pdftotext"):
            self.assertIsNotNone(shutil.which(prog), prog)


if __name__ == "__main__":
    unittest.main()
