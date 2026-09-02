# agent/

Moteur de décision et portefeuille papier. Implémente les règles R1 à R13 de
[../docu/methode/01-regles.md](../docu/methode/01-regles.md).

## Carte des modules

| Fichier | Contenu | Règles |
|---|---|---|
| `models.py` | Types partagés, tous les champs indisponibles en `\| None` | — |
| `rules.py` | Calculs purs, sans effet de bord | R1, R2, R4, R7, R8 |
| `gate.py` | Le crible, produit un `Verdict` | R3, R5, R6, R9, R10, R11, R12 |
| `portfolio.py` | Exécutions simulées, stops, journal, drawdown | — |
| `live.py` | Session temps réel, CLI | orchestration |
| `demo.py` | Quatre scénarios pédagogiques | — |
| `sync_ibkr.py` | Alimente le pont depuis TWS | R7 |
| `scan_market.py` | Liste de pré-marché : découvrir → filtrer → qualifier | R6 |
| `ibkr_doctor.py` | Preflight : ce que le compte sait réellement faire | — |
| `eastern.py` | Heure de l'Est et état du marché — **définition unique** | R10 |

L'état de couverture réel — quelles fonctions F1 à F8 sont automatisées, quelles
couches de la spécification avancent — vit dans
[../docu/methode/07-etat-automatisation.md](../docu/methode/07-etat-automatisation.md).
Le mettre à jour dès qu'un module de `sources/` change d'état.

## Invariants à ne pas casser

**`Verdict.allowed` est vrai uniquement si `blocks` est vide.** Ne jamais
ajouter un motif de refus dans `warnings` pour « laisser passer quand même ».
Si une condition doit bloquer, elle va dans `blocks`.

**R9 bloque.** C'est la seule optimisation originale issue du croisement des
sources. La rétrograder en avertissement viderait le module de sa raison
d'être.

**Le moteur ne décide jamais d'entrer.** Il qualifie et refuse. `evaluate()`
retourne un verdict ; c'est l'appelant humain qui ouvre la position.

**L'ordre des décisions de dimensionnement ne s'inverse pas** : risque accepté
→ niveau d'invalidation → taille. La taille est une conséquence, jamais un
choix premier.

## Pièges rencontrés, à ne pas réintroduire

**Précision flottante dans `position_size`.** `5.00 - 4.80` vaut
`0.20000000000000018` en binaire ; une division entière brute rendait 499
actions au lieu de 500. L'erreur était systématique et toujours dans le même
sens, donc elle sous-dimensionnait chaque position. D'où l'arrondi au
millionième avant plancher. Ne pas « simplifier » ce code.

**`halted` était codé en dur à `False`**, si bien que R6 ne rejetait jamais un
titre suspendu. Le champ vient désormais du courtier. Toute source qui ne le
fournit pas doit laisser `False` *et* le documenter, pas prétendre savoir.

**Le décalage horaire de l'Est était écrit deux fois**, dans `live.py` et dans
le préflight. Une constante horaire dupliquée finit toujours par diverger : elle
vit maintenant dans `eastern.py`, seul endroit qui la définit, avec les bornes
de séance importées de R10. Ne pas la réécrire ailleurs.

**Un diagnostic qui ignore l'heure ment sur la cause.** Un préflight lancé un
dimanche a conclu à un abonnement de données absent alors que le marché était
simplement fermé — hors séance il n'y a ni carnet ni transaction. Tout contrôle
qui interprète une donnée manquante doit d'abord demander à `eastern.py` si le
marché est ouvert.

## Ajouter une règle

1. Le calcul pur va dans `rules.py`, avec ses bornes de validité.
2. La décision va dans `gate.py`, en citant la règle dans le message
   (`"R9 : ..."`) — les messages sont lus par un humain sous pression.
3. Un test par chemin de refus dans `Test/`.
4. Répercuter dans [../docu/methode/01-regles.md](../docu/methode/01-regles.md)
   **et** dans le questionnaire si une question devient fausse.

Le numéro de règle est un contrat entre le code, la documentation et le
questionnaire. Ne pas renuméroter.

## Style

Docstrings au format du projet (Purpose / Inputs / Outputs). Type hints
partout. Journalisation préfixée : `[GATE]`, `[PF]`, `[MARKET]`, `[BROKER]`,
`[EDGAR]`, `[SSR]`. Pas d'accents dans les commentaires et docstrings.
