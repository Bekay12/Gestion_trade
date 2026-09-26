# Notre méthode face à la pratique de Germain

Source : « DayTrade Recap du 14 au 18 septembre 2026 », chaîne Academy Germain.
Sous-titres français générés automatiquement, extraits via `yt-dlp` le 25.09.2026,
4 439 mots. **La transcription est bruitée** : « créding » pour trading, « VAP »
pour VWAP, « que level » pour key level, « Cpop » pour CPOP. Les citations sont
reproduites telles quelles et interprétées avec prudence.

Nature de la vidéo : **un récapitulatif de trades**, pas un cours. Germain y
commente sa semaine écran par écran.

## 1. La phrase qui réoriente tout

> « moi je suis un vendeur à découvert. Je cherche des entreprises qui ont des
> problèmes financiers. »

**Germain est vendeur à découvert.** Notre screener et son classificateur sont
construits pour une entrée acheteuse : `CONTINUATION` et `SQUEEZE` sont des
verdicts haussiers, `FADE` annonce un comblement, `PUMP_RISK` dit « ne pas
jouer ». Aucune classe ne dit « vendre à découvert ».

Or `PUMP_RISK` décrit exactement son terrain de chasse.

## 2. Ses configurations, telles qu'il les décrit

| Marqueur | Citation | Notre équivalent |
|---|---|---|
| Regroupement d'actions | CPOP, ENMG : « c'est un stock qui a eu à faire un reverse split » | **Absent du code** |
| Aucune nouvelle fraîche | « deux qui n'avaient pas de news fraîches » | `catalyseur = None`, mais lu comme une inconnue |
| Détresse financière | « des entreprises qui ont des problèmes financiers » | Alertes nano cap et low float, indirectement |
| Dilution, convertibles | MSS : paiement en 636 308 actions plus notes convertibles | `FORMULAIRES_DILUTIFS`, jamais renseigné |
| Front side / back side | « le backside c'est quand les vendeurs ont le contrôle », séparés par le VWAP | Alerte « cours sous le VWAP », mais approximée |
| Continuation jour 2 | RETO : « un stock de continuation de 2e jour » | Mode `close`, partiellement |
| Épuisement des acheteurs | « je cherchais le mouvement quand les acheteurs s'épuisent » | **Absent** |

Deux outils qu'il utilise et que nous n'avons pas du tout : le **carnet d'ordres
et la profondeur de marché** (« le level 1 », « la profondeur du marché ») et le
**time and sales**. Ce sont des données temps réel de courtier, pas de screener.

## 3. Sa gestion de position

Rien à voir avec la nôtre. Sur XHLD il entre vers 15, sort à 14,50 puis 14,
réattaque à 12, sort à 10 puis à 9. Sur CPOP : « chaque rebond qu'il y avait je
shortais ». Plusieurs allers-retours sur le même titre dans la séance.

Notre skill impose un horizon annoncé **avant** l'entrée et un déclencheur de
sortie unique. Sa pratique est du scalp répété sur une thèse tenue à la journée.

Résultat annoncé sur la semaine : environ 8 500 à 9 000 $, pour une à deux
heures par jour. Une journée sans trade le 17, pour raison personnelle.

## 4. Le test sur nos propres données

Si l'on avait vendu à découvert chaque verdict `PUMP_RISK` des 23 et 24.09,
à l'ouverture, débouclé à la clôture :

| | |
|---|---|
| Observations | 15 |
| Rendement moyen | **+4,13 %** |
| Médiane | +2,95 % |
| Gagnants | 10 sur 15 |

**Trois réserves qui interdisent d'en tirer une stratégie :**

1. **Une seule ligne porte 55 % de la moyenne.** GRML au 23.09 rapporte +36,02 %.
   Sans elle, la moyenne tombe à **+1,86 %**.
2. Ni coût d'emprunt, ni disponibilité du borrow, ni slippage ne sont modélisés.
   Sur des nano caps à faible flottant, le borrow est cher quand il existe.
3. Quinze observations sur deux journées. Ce n'est pas un échantillon.

La classe `INSUFFISANT` rendrait **−0,30 %** à ce même exercice, et resterait de
toute façon inexécutable : shorter demande le même volume qu'acheter.

## 5. Ce que le dépôt sait déjà faire

`Trading_Agent/docu/portefeuille/watchlist.json` contient **deux plans vendeurs**
et un champ dédié :

> « borrow: 'easy' | 'hard' | 'none', relevé chez le courtier. Absent (null) =
> toute position vendeuse refusée (R7/R9). Aucune source publique ne diffuse
> cette donnée. »

L'infrastructure vendeuse existe donc, avec ses règles R7 et R9. **C'est le skill
de gaps qui est resté acheteur**, pas le système.

## 6. Ce que je changerais, par ordre de rendement

1. **Ajouter une direction au verdict.** `PUMP_RISK` devient actionnable comme
   vente à découvert lorsque le borrow est disponible, au lieu d'un simple refus.
   Le classificateur ne changerait pas de logique, seulement de conclusion.
2. **Renseigner `formulaire_sec` par un appel EDGAR automatique.** Deux jours de
   suite, le motif décisif était dans les dépôts : PFSA le 24.09 (convertible
   payable en actions, −14,0 %), GLND le 25.09 (report de forage de deux ans).
   Le champ existe déjà et reste toujours `None`.
3. **Détecter le regroupement d'actions.** Marqueur central chez lui, absent chez
   nous. Repérable par le 8-K item 5.07 et par un saut de prix sans volume.
4. **Le vrai VWAP via IBKR**, déjà identifié comme chantier prioritaire. Il
   devient plus important encore : le front side / back side est son arbitre
   principal, et notre VWAP approximé porte 50 % des erreurs mesurées.

## 7. Ce que cette comparaison ne peut pas établir

- La transcription est automatique et déforme le vocabulaire technique. Sa
  « stratégie préférée », évoquée comme « la fin de mon plus au premier rebond »,
  n'est pas reconstituable de façon fiable.
- Une semaine de récapitulatif ne donne ni son taux de réussite, ni sa taille de
  position, ni son risque par trade.
- Il renvoie à des revues quotidiennes écrites sur son site, qui contiennent le
  détail de chaque trade. Elles n'ont pas été consultées.
