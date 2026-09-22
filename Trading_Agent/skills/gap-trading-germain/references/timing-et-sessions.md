# Quand lancer le scan, et pourquoi

Réponse courte : **deux fenêtres, deux usages différents**, pas une même analyse
répétée deux fois.

## Les fenêtres

| Fenêtre | Heure ET | Heure France (CEST) | Rôle |
|---|---|---|---|
| Veille au soir | 18h–22h | 00h–04h | Préparation longue, calendrier earnings |
| **Pré-marché tôt** | 7h00–8h00 | 13h00–14h00 | Premier balayage, futures et news overnight |
| **Pré-marché cœur** | 8h00–9h30 | **14h00–15h30** | Fenêtre principale : réactions earnings, données macro, flux institutionnel |
| Ouverture | 9h30–10h00 | 15h30–16h00 | Exécution. Plus d'analyse : la moitié des comblements se joue ici |
| Milieu de séance | 11h–15h | 17h–21h | Peu d'intérêt pour les gaps |
| **Clôture** | 16h00–17h00 | **22h00–23h00** | Le gap a-t-il tenu ? Bascule intraday → intraweek |

## Pourquoi le pré-marché est la fenêtre principale

La recherche publiée est convergente : la fenêtre 7h–9h30 ET est celle où les
scanners de gap et les alertes de volume identifient les mouvements du jour, et
elle est **supérieure au scan après clôture** pour décider d'un trade intraday. La
raison est mécanique : le gap se forme sur les news de la nuit et les publications
d'avant-bourse ; à 9h30 ET, la formation est terminée et l'information est publique.

Cadence conseillée par les praticiens : commencer vers 7h00 ET, disposer d'une
liste courte de cinq à quinze noms à 8h30, classée **par volume et non par
pourcentage de hausse** — le volume indique quel mouvement a un adossement
institutionnel.

Pour un opérateur en France, cette fenêtre tombe l'après-midi (13h–15h30), ce qui
est praticable sans contrainte nocturne. C'est un avantage réel du marché US vu
d'Europe, et la raison pour laquelle ce skill privilégie le mode `premarket`.

## Pourquoi la clôture reste indispensable

Le scan de clôture ne sert pas à trader le jour même : il sert à **mesurer la
règle des 30 minutes après coup**. Un gap qui a survécu à la séance sans être
comblé est passé du côté de la continuation, et c'est à ce moment seulement que la
question intraweek se pose.

C'est aussi le moment où la discrimination squeeze / pump devient tranchable : le
pump « retombe généralement sous le niveau d'avant-pump » le jour même, tandis que
le squeeze « continue sur plusieurs jours » (Academy Germain, Partie 7 Ch.03). À
16h00 ET, on sait lequel des deux on regarde ; à 9h30, non.

## Ce que ça implique pour le skill

- `--mode premarket` produit une **watchlist**, pas un verdict. Les horizons
  annoncés y sont provisoires : le catalyseur est connu, la tenue ne l'est pas.
- `--mode close` produit un **verdict**, parce que la variable manquante (le gap
  a-t-il tenu ?) est enfin observable. C'est le mode qui autorise une conclusion
  intraweek.
- Lancer le mode `close` sans avoir lancé le mode `premarket` le matin même reste
  utile, mais le rapport doit dire que le catalyseur a été reconstitué après coup
  plutôt qu'identifié avant l'ouverture.

## Limite de données à déclarer

Finviz gratuit ne donne pas les données pré-marché ; la formation le note
explicitement (Finviz Elite, ~25 $/mois, les ajoute). Conséquence concrète :
avant 9h30 ET, le champ `Gap` de Finviz reflète une couverture partielle. Le
rapport le signale au lieu de présenter la liste comme exhaustive.

Sources externes consultées le 22.09.2026 :
[Scanz — Gap and Go](https://scanz.com/gap-and-go-strategy/),
[Tradewink — Premarket Trading Guide](https://tradewink.com/learn/pre-market-trading-guide),
[TradingStats — When Do Gaps Fill](https://tradingstats.net/when-do-gaps-fill/).
