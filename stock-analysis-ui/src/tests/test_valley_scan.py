"""Verrouille la logique du détecteur de creux.

Honnêteté sur la méthode : contrairement aux corrections des autres scanners,
ce fichier a été écrit APRÈS le code, pas avant. Les tests ci-dessous ont donc
été vérifiés par mutation — chacun a été confronté à une version délibérément
cassée du code pour s'assurer qu'il échoue — mais ils n'ont pas la force d'un
test rouge écrit en premier.

Les seuils ne sont pas inventés : ils viennent du rejeu des trois épisodes
mesurés le 15.09.2026 via l'option --asof.
  GEA au 01.11.2023      recul 3 mois −15,6 % contre −6,8 % pour l'indice,
                         part propre 67,6 %  → divergence attendue
  Neste au 01.03.2025    recul 3 mois −39,8 % quand l'indice montait de 8,6 %,
                         part propre 128,6 % → refus attendu
  Prysmian au 01.03.2025 part propre 339,4 % → refus attendu
"""
import numpy as np
import pandas as pd
import pytest

import Valley_scan as vs


def test_indice_choisi_selon_la_place_de_cotation():
    assert vs.indice_de('G1A.DE') == '^GDAXI'
    assert vs.indice_de('PRY.MI') == 'FTSEMIB.MI'
    assert vs.indice_de('NESTE.HE') == '^OMXH25'
    assert vs.indice_de('AAPL') == vs.INDICE_DEFAUT
    # Oslo n'a plus d'indice exploitable chez Yahoo : repli documenté.
    assert vs.indice_de('YAR.OL') == '^STOXX'


def _series(valeurs, debut='2025-01-01'):
    idx = pd.date_range(debut, periods=len(valeurs), freq='D')
    return pd.Series([float(v) for v in valeurs], index=idx)


def test_fenetre_courte_separe_le_marche_de_lentreprise():
    """Un titre qui suit exactement son indice n'a pas de part propre."""
    n = 80
    titre = _series(np.linspace(100, 90, n))      # −10 %
    indice = _series(np.linspace(100, 90, n))     # −10 %
    r = vs._fenetre_courte(titre, indice, beta=1.0, jours=63, nom='3m')
    assert r['mouv_3m_%'] < 0
    assert r['fraction_propre_3m_%'] == pytest.approx(0.0, abs=2.0)


def test_fenetre_courte_detecte_une_baisse_entierement_propre():
    """Titre en baisse, indice en hausse : la part propre dépasse 100 %.

    C'est exactement la configuration de Neste en février 2025, et la raison
    pour laquelle le signal de divergence doit refuser ce cas.
    """
    n = 80
    titre = _series(np.linspace(100, 60, n))      # −40 %
    indice = _series(np.linspace(100, 109, n))    # +9 %
    r = vs._fenetre_courte(titre, indice, beta=1.0, jours=63, nom='3m')
    assert r['fraction_propre_3m_%'] > 100.0


def _mesures(baisse_3a, mouv_3m, frac_3m, au_dessus_du_bas=50.0):
    return {'baisse_%': baisse_3a, 'mouv_3m_%': mouv_3m,
            'fraction_propre_3m_%': frac_3m,
            'au_dessus_du_bas_%': au_dessus_du_bas}


def test_divergence_quand_la_baisse_vient_du_marche():
    """Le cas GEA : recul partagé avec l'indice, indicateur intact."""
    signal, note, _ = vs.classer(_mesures(-33.6, -15.6, 67.6),
                                 {'indicateur_casse': None}, 20.0, 20.0)
    assert signal == '🕳️ DIVERGENCE'
    assert note > 3.0


def test_pas_de_divergence_quand_la_baisse_est_entierement_propre():
    """Le cas Neste et le cas Prysmian : le signal doit REFUSER.

    C'est le test qui compte le plus. Un détecteur qui classe ces deux-là en
    creux est pire qu'inutile : il recommande d'acheter une dégradation réelle.
    """
    for frac in (128.6, 339.4):
        signal, _, _ = vs.classer(_mesures(-83.4, -39.8, frac),
                                  {'indicateur_casse': None}, 20.0, 20.0)
        assert signal != '🕳️ DIVERGENCE', f"part propre {frac} classée à tort"


def test_indicateur_casse_bloque_la_divergence():
    """Même avec une baisse partagée, une activité qui recule n'est pas un creux."""
    signal, _, _ = vs.classer(_mesures(-33.6, -15.6, 40.0),
                              {'indicateur_casse': True}, 20.0, 20.0)
    assert signal == '⚠️ PIÈGE'


def test_inflexion_demande_cours_bas_ET_activite_qui_repart():
    proche = _mesures(-60.0, -2.0, 90.0, au_dessus_du_bas=8.0)
    loin = _mesures(-60.0, -2.0, 90.0, au_dessus_du_bas=45.0)
    assert vs.classer(proche, {'indicateur_retourne': True}, 20.0, 20.0)[0] == '↗️ INFLEXION'
    assert vs.classer(loin, {'indicateur_retourne': True}, 20.0, 20.0)[0] != '↗️ INFLEXION'
    assert vs.classer(proche, {'indicateur_retourne': False}, 20.0, 20.0)[0] != '↗️ INFLEXION'


def test_seuil_de_part_propre_est_reglable():
    """Le seuil de 75 % est calibré, pas gravé : il doit rester un paramètre."""
    m = _mesures(-33.6, -15.6, 67.6)
    assert vs.classer(m, {'indicateur_casse': None}, 20.0, 20.0, 75.0)[0] == '🕳️ DIVERGENCE'
    assert vs.classer(m, {'indicateur_casse': None}, 20.0, 20.0, 60.0)[0] != '🕳️ DIVERGENCE'


def test_titre_sans_baisse_nest_pas_un_candidat():
    signal, note, _ = vs.classer(_mesures(-5.0, 2.0, 50.0, au_dessus_du_bas=80.0),
                                 {}, 20.0, 20.0)
    assert signal == '—' and note == 0.0


def test_inflexion_exige_aussi_une_progression_sur_un_an():
    """Défaut mesuré sur le tirage de 200 titres du 15.09.2026.

    POOL affichait la séquence trimestrielle 1451 → 982 → 1138 → 1823 et
    déclenchait le signal d'inflexion. Ce n'est pas un retournement, c'est la
    saison : POOL vend du matériel de piscine. Deux hausses séquentielles
    consécutives sont le comportement NORMAL d'une activité saisonnière au
    premier semestre, et le signal les confondait avec une reprise.

    La reprise séquentielle reste nécessaire, mais elle ne suffit plus : le
    dernier trimestre doit aussi dépasser le même trimestre de l'an passé.
    """
    m = _mesures(-59.0, -11.9, 115.0, au_dessus_du_bas=0.0)
    # Rebond saisonnier sur une activité qui recule d'une année sur l'autre
    saisonnier = {'indicateur_retourne': True, 'ca_var_a1_%': -4.9, 'indicateur_casse': False}
    assert vs.classer(m, saisonnier, 20.0, 20.0)[0] != '↗️ INFLEXION'
    # Même rebond, mais l'activité progresse aussi sur un an
    vrai = {'indicateur_retourne': True, 'ca_var_a1_%': 4.3, 'indicateur_casse': False}
    assert vs.classer(m, vrai, 20.0, 20.0)[0] == '↗️ INFLEXION'


def test_inflexion_tolere_une_progression_annuelle_inconnue():
    """Sans donnée annuelle, le signal ne doit pas disparaître en silence.

    Le mode rétrospectif ne dispose d'aucun fondamental : exiger une valeur
    rendrait le signal structurellement impossible à rejouer.
    """
    m = _mesures(-59.0, -11.9, 115.0, au_dessus_du_bas=0.0)
    inconnu = {'indicateur_retourne': True, 'ca_var_a1_%': float('nan')}
    assert vs.classer(m, inconnu, 20.0, 20.0)[0] == '↗️ INFLEXION'


# ── Garde-fou de trésorerie ────────────────────────────────────────────────
# Valeurs réelles, recoupées le 15.09.2026 entre yfinance et les 10-K 2025
# (flux de trésorerie libre en M USD, du plus récent au plus ancien) :
#   Pool Corp   310 / 600 / 828 / 441   dette 1535 / 1272 / 1364 / 1661
#   ESAB        213 / 304 / 282 / 174   dette 1345 / 1163 / 1117 / 1313
#   Honeywell  5422 / 5226 / 4599 / 4508 dette 35563 / 32077 / 21536 / 20537
# Les deux premiers ont été classés « inflexion » par le détecteur alors que
# leur trésorerie se dégradait. C'est ce que ce garde-fou doit empêcher.

def test_tresorerie_effondree_disqualifie_linflexion():
    """Pool Corp : chiffre d'affaires en hausse, trésorerie divisée par 2,7.

    Le signal reposait sur le seul chiffre d'affaires trimestriel. Une activité
    dont le flux libre passe de 828 à 310 M USD en deux ans n'est pas en
    reprise, quelle que soit la saisonnalité du carnet.
    """
    etat = vs.etat_tresorerie([310, 600, 828, 441], [1535, 1272, 1364, 1661])
    assert etat['tresorerie_degradee'] is True
    m = _mesures(-59.0, -11.9, 115.0, au_dessus_du_bas=0.0)
    f = {'indicateur_retourne': True, 'ca_var_a1_%': 2.2, **etat}
    assert vs.classer(m, f, 20.0, 20.0)[0] == '⚠️ PIÈGE'


def test_tresorerie_en_baisse_avec_dette_en_hausse_disqualifie_aussi():
    """ESAB : flux libre à 70 % de son sommet, mais dette en hausse.

    Le seul rapport au sommet ne suffisait pas à l'écarter (0,70 exactement).
    La seconde condition — trésorerie qui recule pendant que la dette monte —
    est ce qui distingue une année creuse d'une dégradation financée.
    """
    etat = vs.etat_tresorerie([213, 304, 282, 174], [1345, 1163, 1117, 1313])
    assert etat['tresorerie_degradee'] is True
    m = _mesures(-50.0, -26.3, 116.2, au_dessus_du_bas=8.7)
    f = {'indicateur_retourne': True, 'ca_var_a1_%': 12.9, **etat}
    assert vs.classer(m, f, 20.0, 20.0)[0] == '⚠️ PIÈGE'


def test_tresorerie_en_progression_laisse_passer_linflexion():
    """Honeywell : flux libre au plus haut de la fenêtre, le signal doit tenir."""
    etat = vs.etat_tresorerie([5422, 5226, 4599, 4508], [35563, 32077, 21536, 20537])
    assert etat['tresorerie_degradee'] is False
    m = _mesures(-22.6, -12.8, 115.3, au_dessus_du_bas=15.7)
    f = {'indicateur_retourne': True, 'ca_var_a1_%': 4.3, **etat}
    assert vs.classer(m, f, 20.0, 20.0)[0] == '↗️ INFLEXION'


def test_tresorerie_inconnue_ne_bloque_pas():
    """Sans série de trésorerie, le garde-fou doit s'abstenir, pas interdire.

    Le mode rétrospectif n'a aucun fondamental : un garde-fou qui refuse par
    défaut rendrait le signal impossible à rejouer.
    """
    etat = vs.etat_tresorerie([], [])
    assert etat['tresorerie_degradee'] is None
    m = _mesures(-59.0, -11.9, 115.0, au_dessus_du_bas=0.0)
    assert vs.classer(m, {'indicateur_retourne': True, **etat}, 20.0, 20.0)[0] == '↗️ INFLEXION'


def test_une_seule_annee_ne_permet_aucun_jugement():
    """Une seule observation ne permet ni de constater une baisse ni de l'exclure."""
    assert vs.etat_tresorerie([310], [1535])['tresorerie_degradee'] is None
