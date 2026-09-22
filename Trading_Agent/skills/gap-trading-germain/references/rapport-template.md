# Format de sortie

Trois blocs, dans cet ordre, pour chaque titre retenu. Rien d'autre : ce rapport
se lit entre 14h et 15h30, pas le soir au calme.

Couleurs de responsabilité, comme dans `startup-investment-analyzer` :
fait constaté (neutre) · **évaluation du rédacteur** · *verdict*.

---

## 1. Analyse du gap et du contexte

Une demi-page maximum. Doit contenir, dans l'ordre :

| Élément | Exigence |
|---|---|
| Le gap | Taille en %, **et** en multiple d'ATR (le % seul ne dit rien) |
| Le catalyseur | Nature, date, source (EDGAR + numéro de formulaire, ou média nommé) |
| Le titre | Capitalisation, float, short interest, volume moyen |
| La technique | RVOL du moment, position par rapport au VWAP, plus-haut/plus-bas du jour |
| La qualité du catalyseur | Haussier réel / baissier déguisé (S-3, 424B) / spéculatif / introuvable |

Si le catalyseur est introuvable, l'écrire ainsi : *« catalyseur non identifié »*.
Ne jamais écrire *« probablement lié à… »*.

## 2. Signaux à valider avant de passer l'ordre

Une check-list, pas un paragraphe. Chaque ligne est vérifiable en quelques
secondes devant l'écran, et chacune peut annuler le trade à elle seule.

```
[ ] Catalyseur identifié, daté, et de nature haussière (pas un S-3 / 424B)
[ ] RVOL ≥ 2 (≥ 3 pour viser la continuation)
[ ] Cours AU-DESSUS du VWAP        ← si non : distribution, pas gap haussier
[ ] Volume progressif, pas un pic unique au sommet
[ ] Short interest élevé AVANT le mouvement (si thèse squeeze)
[ ] Aucun signal de pump : pas de pic de volume au plus haut, pas de chute
    rapide post-sommet, dépôts SEC récents présents
[ ] Liquidité : volume moyen > 500 K
[ ] Niveau psychologique / résistance identifié au-dessus de l'entrée
[ ] Taille de position réduite si nano cap ou low float
```

Les cases non cochables faute de donnée sont marquées `?`, pas cochées par défaut.

## 3. Horizon de réalisation

Le tableau qui répond à « combien de temps ». L'horizon découle de la
classification, il ne se choisit pas librement :

| Verdict | Horizon annoncé | Ce qui le fixe |
|---|---|---|
| `FADE` | **Jour** — souvent la première heure | Comblement attendu ; l'essentiel se joue dans les 30 premières minutes |
| `CONTINUATION` | **Jour**, prolongeable | Gap tenu + RVOL ≥ 3 + catalyseur ; réévaluer à la clôture |
| `SQUEEZE` | **Plusieurs jours à une semaine** | « Le mouvement continue sur plusieurs jours » (Germain, P7 Ch.03) |
| `PUMP_RISK` | **Aucun** | Ne pas entrer. Si déjà en position : sortie, pas d'horizon |

Trois obligations d'écriture :

1. **Nommer le déclencheur de sortie**, pas seulement la durée. Exemple : « sortie
   au retour sous le VWAP » est un horizon opérationnel ; « environ deux jours »
   n'en est pas un.
2. **Dire ce qui invaliderait l'horizon.** Un gap `SQUEEZE` qui se comble dans
   l'heure n'était pas un squeeze : l'horizon tombe avec sa prémisse.
3. **Aucun horizon supérieur à la semaine.** Au-delà, ce n'est plus un trade de
   gap ; la question devient une question de valeur, et c'est un autre skill.

---

## Modèle condensé (un titre)

```
### XXXX — Nom (cap XX M$, float X,X M)

GAP        +12,4 % · 1,8 × ATR · au-dessus du VWAP
CATALYSEUR 8-K du 21.09.2026 : contrat de distribution signé — haussier réel
TECHNIQUE  RVOL 6,2 · volume 4,1 M (moy. 660 K) · short interest 18,3 %
VERDICT    SQUEEZE — short interest élevé avant le mouvement, volume progressif

À VALIDER AVANT L'ORDRE
[x] Catalyseur haussier daté      [x] RVOL ≥ 3        [x] Au-dessus du VWAP
[x] Volume progressif             [x] SI > 10 % avant [ ] Résistance identifiée ?
[x] Aucun signal de pump          [x] Vol. moyen > 500 K

HORIZON    Plusieurs jours (3 à 5 séances)
SORTIE     Retour durable sous le VWAP, ou épuisement du volume (RVOL < 1,5)
INVALIDE SI Comblement du gap dans la première heure → la thèse squeeze tombe
```
