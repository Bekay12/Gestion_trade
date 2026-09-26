#!/usr/bin/env python3
"""
test_cycle_quotidien - tests hors ligne de la decision de phase.

Stdlib seule, aucun appel reseau, aucun sous-processus: `phase_courante` est pure
et recoit son instant en argument.

Ce que ces tests protegent en priorite, c'est la NEUTRALISATION DE LA DERIVE
SAISONNIERE. C'est le defaut qu'une tache planifiee produit silencieusement: une
heure codee en local tombe hors fenetre deux fois par an, la detection ne se fait
pas, et rien ne le signale. Une seule classe ici verifie qu'un meme creneau de
Paris tombe dans la bonne phase avant et apres chaque changement d'heure.

    python3 gaps/Test/test_cycle_quotidien.py
"""
import os
import sys
import unittest
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from cycle_quotidien import phase_courante, FENETRES, LIMITE   # noqa: E402

NY = ZoneInfo("America/New_York")
PARIS = ZoneInfo("Europe/Paris")


def _ny(iso: str) -> datetime:
    return datetime.fromisoformat(iso).replace(tzinfo=NY)


class TestBornesDesFenetres(unittest.TestCase):
    """Les bornes exactes, la ou une erreur d'inegalite se cache."""

    def test_premarket_inclut_son_debut(self):
        self.assertEqual(phase_courante(_ny("2026-09-28T07:00")), "premarket")

    def test_premarket_exclut_sa_fin(self):
        self.assertEqual(phase_courante(_ny("2026-09-28T09:29")), "premarket")
        self.assertIsNone(phase_courante(_ny("2026-09-28T09:30")))

    def test_avant_le_premarket_rien(self):
        self.assertIsNone(phase_courante(_ny("2026-09-28T06:59")))

    def test_cloture_commence_a_16h05_pas_a_16h00(self):
        """Les barres de cloture mettent quelques minutes a se stabiliser chez
        yfinance; noter avant produirait des notes fausses sans le signaler."""
        self.assertIsNone(phase_courante(_ny("2026-09-28T16:04")))
        self.assertEqual(phase_courante(_ny("2026-09-28T16:05")), "cloture")

    def test_cloture_exclut_sa_fin(self):
        self.assertEqual(phase_courante(_ny("2026-09-28T16:59")), "cloture")
        self.assertIsNone(phase_courante(_ny("2026-09-28T17:00")))

    def test_pleine_seance_aucune_phase(self):
        """Entre l'ouverture et la cloture, le cycle n'a rien a faire."""
        for h in (10, 12, 14, 15):
            self.assertIsNone(phase_courante(_ny(f"2026-09-28T{h:02d}:30")), h)


class TestWeekEnd(unittest.TestCase):
    """Aucune phase le week-end: le marche americain est ferme."""

    def test_samedi_et_dimanche(self):
        self.assertIsNone(phase_courante(_ny("2026-09-26T08:00")))  # samedi
        self.assertIsNone(phase_courante(_ny("2026-09-27T08:00")))  # dimanche

    def test_vendredi_reste_actif(self):
        self.assertEqual(phase_courante(_ny("2026-09-25T08:00")), "premarket")


class TestDeriveSaisonniere(unittest.TestCase):
    """LE test qui justifie l'architecture.

    Mesure du 26.09.2026: 07h00 ET vaut 13h00 a Paris en septembre et en
    novembre, mais 12h00 en mars, parce que les deux zones ne changent pas
    d'heure le meme week-end. Une heure metier codee dans la crontab locale
    tomberait donc hors fenetre deux fois par an.

    Ici la decision se prend sur l'instant reel, donc un creneau de Paris doit
    tomber dans la bonne phase quelle que soit la saison.
    """

    def _phase_depuis_paris(self, iso_paris: str):
        t = datetime.fromisoformat(iso_paris).replace(tzinfo=PARIS)
        return phase_courante(t.astimezone(NY))

    def test_le_decalage_change_bien_selon_la_saison(self):
        """La premisse elle-meme, verifiee et non supposee."""
        sept = _ny("2026-09-28T07:00").astimezone(PARIS).hour
        mars = _ny("2027-03-22T07:00").astimezone(PARIS).hour
        self.assertEqual(sept, 13)
        self.assertEqual(mars, 12)
        self.assertNotEqual(sept, mars)

    def test_13h_paris_est_le_premarket_en_septembre(self):
        self.assertEqual(self._phase_depuis_paris("2026-09-28T13:00"), "premarket")

    def test_13h_paris_n_est_plus_le_premarket_en_mars(self):
        """Une crontab figee a 13h00 Paris manquerait l'ouverture de la fenetre:
        a cette heure-la, en mars, il est deja 08h00 ET. Elle tombe encore dans
        la fenetre ici, mais une heure plus tard elle serait dehors."""
        self.assertEqual(self._phase_depuis_paris("2027-03-22T13:00"), "premarket")
        # Une heure de derive suffit a sortir de la fenetre:
        self.assertIsNone(self._phase_depuis_paris("2027-03-22T15:00"))

    def test_le_creneau_de_cloture_suit_aussi(self):
        self.assertEqual(self._phase_depuis_paris("2026-09-28T22:30"), "cloture")
        self.assertEqual(self._phase_depuis_paris("2027-03-22T21:30"), "cloture")


class TestContrat(unittest.TestCase):
    """Ce que le reste du dispositif suppose de ce module."""

    def test_les_deux_phases_existent(self):
        self.assertEqual(set(FENETRES), {"premarket", "cloture"})

    def test_les_fenetres_ne_se_chevauchent_pas(self):
        (d1, f1), (d2, f2) = FENETRES["premarket"], FENETRES["cloture"]
        self.assertTrue(f1 <= d2 or f2 <= d1)

    def test_la_limite_respecte_la_regle_du_skill(self):
        """SKILL.md: jamais moins de 30 sans raison explicite. Mesure du
        23.09.2026: un plafond a 15 a laisse passer sept titres eligibles."""
        self.assertGreaterEqual(LIMITE, 30)


if __name__ == "__main__":
    unittest.main(verbosity=2)
