# docu/portefeuille/

**Données vivantes du compte papier.** Contrairement au reste de `docu/`, ce
dossier n'est pas de la documentation : c'est de l'état, écrit et relu par
[../../agent/](../../agent/).

| Fichier | Écrit par | Édition manuelle |
|---|---|---|
| `journal.json` | `PaperPortfolio.save()` | **Non** |
| `journal-demo.json` | Archive de `demo.py` | Non |
| `broker_snapshot.json` | Pont courtier | Non |
| `watchlist.json` | **Toi** | **Oui, c'est sa raison d'être** |

## journal.json — ne pas éditer à la main

Porte le capital, les positions ouvertes, l'historique des opérations et le
sommet de capital. Il est relu au démarrage de chaque session : sans reprise
d'état, le drawdown (R4) et la série de pertes consécutives (R12) repartiraient
de zéro à chaque lancement et ne voudraient plus rien dire.

Une modification à la main fausse ces deux règles silencieusement. Pour repartir
de zéro, **renommer** le fichier plutôt que le vider — l'historique a de la
valeur.

`journal-demo.json` est l'état laissé par `demo.py`, écarté pour qu'il ne
pollue pas le compte réel. `demo.py` écrit dans `journal.json` : le relancer
écrase le compte en cours.

## watchlist.json — le seul fichier que tu remplis

Porte les plans de trade **et** le statut d'emprunt.

```json
{"symbol":"CDTG","direction":"short","catalyst":"…",
 "entry":1.83,"stop":2.05,"targets":[1.40],
 "invalidation":"…","borrow":"hard","borrow_rate":0.85}
```

`borrow` vaut `"easy"`, `"hard"`, `"none"` ou `null`. **Aucune source publique
ne diffuse cette donnée** — elle se relève chez le courtier. Laissée à `null`,
toute position vendeuse est refusée par R7 et R9. C'est voulu, pas une panne.

`catalyst` doit tenir en une phrase : c'est le test d'admission de R11. S'il ne
tient pas, le titre n'a rien à faire dans la liste.

Le fichier est créé avec un modèle au premier lancement s'il n'existe pas.

## broker_snapshot.json — périssable

Cotations temps réel déposées par un producteur externe. **Au-delà de 15
minutes, une entrée est ignorée** plutôt que servie : un prix périmé est plus
dangereux qu'un prix absent, parce qu'il a l'air exploitable.

```powershell
.venv\Scripts\python.exe agent\live.py broker   # ce qui est frais, ce qui ne l'est pas
```

Ce fichier existe parce que le connecteur courtier vit dans une session
d'assistant, pas dans le processus Python. Une connexion TWS/IB Gateway le
remplacera sans que le moteur change.

## Ne pas versionner

État local et positions. Rien ici n'a vocation à quitter la machine.
