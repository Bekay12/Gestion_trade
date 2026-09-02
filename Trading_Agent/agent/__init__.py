"""
agent - Moteur de decision et portefeuille papier.

Implemente les regles R1 a R13 de docu/methode/01-regles.md et le pipeline de
docu/methode/02-logique-decision.md, sur un portefeuille simule.

Principe de conception, repris de docu/methode/05-specification-agent.md : les
regles bloquantes REFUSENT, elles n'avertissent pas, et une donnee absente vaut
refus. Le moteur ne decide jamais d'entrer a la place de l'operateur : il
qualifie, calcule les tailles, et oppose un refus a ce qui viole une regle.
"""

from .gate import evaluate
from .models import ClosedTrade, Position, RiskProfile, Snapshot, TradePlan, Verdict
from .portfolio import PaperPortfolio

__all__ = [
    "evaluate",
    "PaperPortfolio",
    "RiskProfile",
    "Snapshot",
    "TradePlan",
    "Verdict",
    "Position",
    "ClosedTrade",
]
