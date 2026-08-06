"""
Arithmetique du budget d'evaluations de l'optimisateur hybride.

Une seule regle d'echelle selon la precision, et un plan par strategie qui
annonce son nombre d'evaluations. Le menu et le lancement lisent les memes
fonctions, ce qui supprime l'ecart entre le budget affiche et le budget utilise.

`popsize` de SciPy est un multiplicateur : la population vaut
popsize * dimension. L'ignorer faisait passer un groupe de 75 000 evaluations
prevues a environ 1,45 million.
"""
from __future__ import annotations

# Facteur d'echelle du budget selon le nombre de decimales cherchees.
FACTEUR_PRECISION: dict[int, float] = {1: 0.5, 2: 1.0, 3: 2.0}

# Repartition du budget entre strategies en mode hybrid.
REPARTITION_HYBRIDE: dict[str, float] = {
    'differential': 0.4,
    'pso': 0.3,
    'lhs': 0.3,
}


def budget_effectif(budget_base: int, precision: int) -> int:
    """
    --------------------------------------------------------------------------
    Objectif:
        Budget d'evaluations pour une precision donnee. Seule regle d'echelle
        du projet : l'ancien code la comptait deux fois, une fois dans le CLI
        et une fois dans optimize_sector_coefficients_hybrid.

    Inputs:
        budget_base (int): budget de reference
        precision (int): 1, 2 ou 3 decimales

    Outputs:
        budget (int)
    --------------------------------------------------------------------------
    """
    if precision not in FACTEUR_PRECISION:
        raise ValueError(
            f"precision {precision} inconnue, attendu {sorted(FACTEUR_PRECISION)}")
    return int(budget_base * FACTEUR_PRECISION[precision])


def plan_differential(enveloppe: int, dimension: int) -> tuple[int, int]:
    """
    --------------------------------------------------------------------------
    Objectif:
        Choisir (multiplicateur popsize, maxiter) tels que le nombre
        d'evaluations reste sous l'enveloppe.

    Inputs:
        enveloppe (int): evaluations autorisees
        dimension (int): taille du vecteur

    Outputs:
        (multiplicateur, maxiter) (tuple[int, int]): chacun au moins 1
    --------------------------------------------------------------------------
    """
    dimension = max(1, int(dimension))
    enveloppe = max(4, int(enveloppe))
    # Cible : une population large mais qui laisse au moins 10 generations.
    multiplicateur = max(1, min(15, enveloppe // (dimension * 10)))
    population = multiplicateur * dimension
    maxiter = max(1, enveloppe // population - 1)
    return multiplicateur, maxiter


def evaluations_differential(multiplicateur: int, maxiter: int, dimension: int) -> int:
    """Evaluations qu'un plan DE consommera : population initiale plus iterations."""
    return multiplicateur * dimension * (maxiter + 1)


def plan_genetique(enveloppe: int) -> tuple[int, int]:
    """(population, generations) sous l'enveloppe."""
    enveloppe = max(4, int(enveloppe))
    population = max(4, min(300, int(enveloppe ** 0.5)))
    generations = max(1, min(100, enveloppe // population))
    return population, generations


def plan_pso(enveloppe: int) -> tuple[int, int]:
    """(particules, iterations) sous l'enveloppe, l'initialisation comptant."""
    enveloppe = max(4, int(enveloppe))
    particules = max(4, min(100, int(enveloppe ** 0.5)))
    iterations = max(1, enveloppe // particules - 1)
    return particules, iterations


def plan_lhs(enveloppe: int) -> int:
    """Nombre d'echantillons, un par evaluation."""
    return max(1, min(1200, int(enveloppe)))
