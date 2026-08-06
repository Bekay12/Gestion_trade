"""
Verrouillage de l'arithmetique du budget d'evaluations.

`popsize` de SciPy est un MULTIPLICATEUR : la population vaut
popsize * dimension. En passant 200 avec 36 dimensions, l'optimisateur lancait
7 200 individus par generation, soit environ 1,45 million d'evaluations par
groupe au lieu des 75 000 du budget. Le menu affichait par ailleurs une
estimation calculee sur 3 500 quand le lancement utilisait 30 000.
"""
import pytest

from core import optim_budget as budget


@pytest.mark.parametrize("precision, attendu", [(1, 15000), (2, 30000), (3, 60000)])
def test_budget_effectif_par_precision(precision, attendu) -> None:
    """Une seule regle d'echelle, partagee par le menu et le lancement."""
    assert budget.budget_effectif(30000, precision) == attendu


def test_budget_effectif_refuse_une_precision_inconnue() -> None:
    with pytest.raises(ValueError):
        budget.budget_effectif(30000, 7)


@pytest.mark.parametrize("enveloppe", [200, 2000, 30000, 75000])
@pytest.mark.parametrize("dimension", [14, 25, 36])
def test_differential_ne_depasse_pas_le_budget(enveloppe, dimension) -> None:
    """population * (maxiter + 1) reste sous l'enveloppe."""
    multiplicateur, maxiter = budget.plan_differential(enveloppe, dimension)

    assert multiplicateur >= 1
    assert maxiter >= 1
    prevu = budget.evaluations_differential(multiplicateur, maxiter, dimension)
    assert prevu <= enveloppe, f"{prevu} > {enveloppe}"


@pytest.mark.parametrize("dimension", [14, 25, 36])
def test_differential_utilise_une_part_utile_du_budget(dimension) -> None:
    """Un plan qui n'utiliserait que 1 % du budget serait inutile.

    Couvre les trois dimensions que l'optimiseur produit reellement (14 params
    de base, 25 avec les extras prix ou fondamentaux, 36 avec les deux), pas
    seulement 36 : mesure faite a 30 000, la fraction utilisee est >= 0.99
    pour chacune (voir rapport de tache), le plancher 0.5 ci-dessous garde
    donc une marge confortable sans coller a la valeur mesuree.
    """
    multiplicateur, maxiter = budget.plan_differential(30000, dimension)
    prevu = budget.evaluations_differential(multiplicateur, maxiter, dimension)

    assert prevu >= 30000 * 0.5


def test_regression_popsize_multiplicateur() -> None:
    """Regression B4 : l'ancien code passait popsize=200 avec 36 dimensions,
    soit 7 200 individus par generation."""
    multiplicateur, maxiter = budget.plan_differential(75000, 36)
    population = multiplicateur * 36

    assert population <= 75000
    assert population < 7200


@pytest.mark.parametrize("enveloppe", [200, 2000, 30000])
def test_plans_des_autres_strategies(enveloppe) -> None:
    """Chaque strategie annonce un nombre d'evaluations sous l'enveloppe, et
    une part significative n'est pas gaspillee.

    Fractions reellement utilisees sur les enveloppes parametrees ci-dessus
    (mesurees, voir rapport de tache) :
        genetique : 0.98 (200), 0.99 (2000), 0.577 (30000)
        pso       : 0.98 (200), 0.99 (2000), 1.0   (30000)
    Le pire cas mesure est 0.577 (genetique a 30000) : le plancher 0.5
    ci-dessous mord donc sur une implementation degenerescente (par exemple
    un plan qui renverrait toujours le minimum autorise) sans jamais faire
    echouer le comportement reel des deux fonctions.

    `plan_lhs` est plafonne a 1200 echantillons : au-dela de ce plafond, la
    fraction utile de l'enveloppe chute structurellement (0.04 a 30000) sans
    que ce soit une degenerescence. Une borne uniforme de 0.5 le ferait donc
    echouer a tort des que l'enveloppe depasse 2400. Le cas est traite a part
    : sous le plafond, `plan_lhs` doit utiliser l'enveloppe en entier ; au-dela,
    il doit utiliser le plafond en entier, jamais moins.
    """
    population, generations = budget.plan_genetique(enveloppe)
    assert population * generations <= enveloppe
    assert population >= 4 and generations >= 1
    assert population * generations >= enveloppe * 0.5

    particules, iterations = budget.plan_pso(enveloppe)
    assert particules * (iterations + 1) <= enveloppe
    assert particules >= 4 and iterations >= 1
    assert particules * (iterations + 1) >= enveloppe * 0.5

    echantillons = budget.plan_lhs(enveloppe)
    assert 0 < echantillons <= enveloppe
    plafond_lhs = 1200
    assert echantillons == min(plafond_lhs, enveloppe)


def test_repartition_hybride_somme_a_un() -> None:
    """En mode hybrid, le budget est reparti, pas pris en entier par chacune."""
    assert set(budget.REPARTITION_HYBRIDE) == {'differential', 'pso', 'lhs'}
    assert sum(budget.REPARTITION_HYBRIDE.values()) == pytest.approx(1.0)


def test_hybride_total_sous_le_budget() -> None:
    """La somme des trois plans reste sous le budget global."""
    enveloppe, dimension = 30000, 36
    part = budget.REPARTITION_HYBRIDE
    mult, maxiter = budget.plan_differential(int(enveloppe * part['differential']), dimension)
    particules, iterations = budget.plan_pso(int(enveloppe * part['pso']))
    echantillons = budget.plan_lhs(int(enveloppe * part['lhs']))

    total = (budget.evaluations_differential(mult, maxiter, dimension)
             + particules * (iterations + 1)
             + echantillons)
    assert total <= enveloppe
