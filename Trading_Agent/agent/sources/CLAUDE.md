# agent/sources/

Couche d'acquisition. Chaque module couvre une fonction de
[../../docu/methode/03-outils.md](../../docu/methode/03-outils.md).

| Module | Fonction | Coût |
|---|---|---|
| `edgar.py` | F4 dépôts réglementaires, F8 détention | Gratuit |
| `ssr.py` | F5 restriction Rule 201 | Gratuit |
| `market.py` | F1 prix/volume, F5 flottant, F3 actualité | Gratuit, différé |
| `broker.py` | Pont vers un instantané courtier temps réel | Connecteur |
| `ibkr.py` | **Producteur TWS/Gateway** : carnet, halt, **emprunt + taux** | Compte IBKR |
| `ib_ticks.py` | Traduction pure des ticks — seuils, unites, garde-fous | — |
| `ib_ports.py` | Où TWS écoute. Sonde les quatre ports par défaut | — |
| `scanner.py` | **F1 découverte**, bornes importées de R6 | Compte IBKR |
| `scan_filters.py` | Filtres avancés : Rule 201, non empruntable, halt | — |

## La règle absolue de cette couche

**Une donnée qu'on ne peut pas obtenir reste `None` et remonte telle quelle.**
Aucun module ne comble un trou par une valeur plausible, une moyenne, ou un
champ voisin. Le moteur traite `None` comme un refus ; le remplacer par une
approximation transforme un refus en autorisation, silencieusement.

Cas concret à ne jamais réintroduire : **ne pas substituer `sharesOutstanding`
à `floatShares`.** Le flottant exclut les titres bloqués. La substitution
fausserait R6, R8 et R9 d'un coup, et le résultat aurait l'air correct.

## Sémantiques non évidentes

**`ssr.py` — le fichier du jour D contient les déclenchements de D-1 *et* de D.**
Ce n'est pas une redondance : c'est exactement la fenêtre pendant laquelle la
Rule 201 court (jour de déclenchement + lendemain entier). Présence dans le
fichier du jour D ⇒ sous restriction le jour D. Le repli sur la veille reste
donc valide quand le fichier du jour n'est pas encore publié.

La dernière ligne du fichier est un horodatage de génération sans virgule —
écartée par le contrôle du nombre de colonnes. Le nom de société peut contenir
des virgules et être entre guillemets, d'où le passage par un lecteur CSV.

**`edgar.py` — S-3 et 424B ne disent pas la même chose.** Un S-3 actif signale
une *autorisation* d'émettre (dilution possible) ; un 424B récent signale que
l'émission a *eu lieu*. Confondre les deux annule l'intérêt du module.

L'en-tête `SEC_USER_AGENT` est obligatoire côté SEC : sans en-tête, la réponse
est 403. Le client refuse de démarrer plutôt que d'en inventer un — c'est un
choix qui engage l'identité de l'appelant.

**`market.py` — les cours publics sont différés**, et le volume public ne
couvre pas la séance de pré-marché : il sert parfois celui de la veille, ce qui
produit un volume relatif faux et flatteur. Le pont courtier corrige les deux.

**`broker.py` — le connecteur IBKR vit dans une session d'assistant, pas dans
ce processus Python.** Un script ne peut pas l'appeler. Le pont lit un fichier
JSON qu'un producteur externe alimente. Une connexion TWS/IB Gateway pourra le
remplacer sans que le moteur change.

La fraîcheur est vérifiée, jamais supposée : au-delà de 15 minutes, l'instantané
est ignoré. **Un prix périmé est plus dangereux qu'un prix absent**, parce qu'il
a l'air exploitable.

Le connecteur répond avec des **tirets** là où les champs demandés portent des
**soulignés** (`bid_ask` → `bid-ask`). Lire la mauvaise forme rend tout `None`
en silence. Le drapeau `is_close` sur `last` signale une clôture reprise, pas
une transaction de la séance.

**`ibkr.py` — trois pièges du protocole TWS, vérifiés sur sa documentation.**

Le **volume des actions américaines porte un multiplicateur 100** (ticks 8 et
21). Servi brut, le volume relatif de R6 vaut un centième du réel et le crible
rejette tout sans que rien n'ait l'air cassé.

Le **tick 49 (`halted`) vaut -1 pour « statut indisponible »**, ce qui n'est pas
« non suspendu ». Sa valeur 0 n'est d'ailleurs renvoyée que si le contrat figure
dans une liste TWS : l'ignorance est donc le cas courant, pas l'exception. Elle
remonte `False` selon la convention du projet, **mais journalisée** — jamais en
silence.

**Le tick de volume différé (74) arrive en virgule fixe, échelle 10⁶.**
Élucidé par mesure le 2026-08-23 : TWS envoie lui-même `48 591 578 764 254` pour
AAPL — le **callback brut** porte déjà cette valeur, alors que le tick 89 arrive
juste par le *même chemin de code*. Ce n'est donc ni `ib_async`, ni la locale du
système : c'est l'encodage du tick. Corroboré sur cinq titres couvrant trois
ordres de grandeur.

`session_volume()` décode — mais **seulement si le brut est impossible ET que le
décodé devient plausible**. Hors de cette fenêtre elle remonte `None`, le moteur
retombe sur le volume public, et le RVOL de R6 reste calculable. Ne pas
transformer cette fenêtre en mise à l'échelle systématique : une inference
silencieuse sur le volume fausserait le RVOL dans le sens **permissif**, et un
filtre qui accepte tout ne refuse plus rien.

Mesure limitée au flux **différé**. Le comportement en direct n'est pas vérifié.

**Un compte sans abonnement ne reçoit RIEN en direct**, pas une version
dégradée : erreur 10089 et zéro tick. Le repli sur le différé est automatique
(`auto_delayed`), journalisé, et `realtime` reste `False`.

Les **seuils du tick 46** (> 2,5 facile, > 1,5 difficile, ≤ 1,5 non empruntable)
viennent de la documentation TWS, pas d'un arbitrage local. Ne pas les ajuster
« au feeling » : ce sont eux qui alimentent la condition *hard to borrow* de R9.

**`scanner.py` — il ne crée aucun seuil, il les importe.** Prix, volume moyen
et capitalisation viennent de `gate.py`, donc de R6. Recopier un chiffre ici
ferait diverger le balayage et le crible en silence, et le seul symptôme serait
une liste de candidats que le crible refuse tous. Un test lit le fichier source
et échoue si un seuil R6 y réapparaît en dur.

**Le scanner ne qualifie pas.** Un symbole remonté est un symbole à passer au
crible, pas un candidat. Il ne calcule pas non plus le volume **relatif** — IB
filtre sur un volume absolu ; le rapport au volume habituel, qui est le filtre
décisif de R6, ne se mesure que sur le `Snapshot`.

**Un filtre refusé rend une liste VIDE, pas une erreur.** TWS a répondu
« erreur 10360, Scan filter floatSharesBelow is not allowed » sur le compte de
test — par le canal d'erreurs, pas par une exception, et la requête a rendu zéro
symbole. Un marché calme ressemble trait pour trait à ça. `scan()` écoute donc
`errorEvent` pendant la requête et lève si un code de `ERREURS_BALAYAGE`
apparaît. Sans ce relevé, la liste de pré-marché se viderait en silence.

**Les balayages intrajournaliers ne rendent rien d'exploitable hors séance.**
Ils ne lèvent pas : ils rendent un **ordre alphabétique**, faute de classement à
calculer. Observé le 2026-08-23 — `rvol` a rendu AADX, AAL, AAOG, AAOX, AAOZ.
`SCANS_INTRAJOURNALIERS` déclenche un avertissement quand le marché est fermé.

**Un balayage vide et une source morte ne se confondent pas.** `sweep()` lève
`ScanUnavailable` quand *tous* les balayages ont échoué, et ne rend une liste
vide que si le marché est réellement calme. Sans cette distinction, TWS fermé
ressemblerait à une séance sans candidat — la panne déguisée en information.

## Ce qui reste hors de portée

**Le statut d'emprunt et son taux — levé, sous condition.** La documentation les
annonçait sans source : c'était vrai du connecteur MCP, faux d'une connexion
TWS. `ibkr.py` les obtient (tick générique 236 pour le statut,
`reqHistoricalData(whatToShow="FEE_RATE")` pour le taux). **La condition est que
TWS ou Gateway tourne.** Sans lui, la saisie manuelle dans
`../../docu/portefeuille/watchlist.json` reste la voie, et une position vendeuse
sans statut d'emprunt est toujours refusée — comportement voulu, pas panne.

## Ajouter une source

Répercuter l'état de couverture dans
[../../docu/methode/07-etat-automatisation.md](../../docu/methode/07-etat-automatisation.md)
— tableau 1, et le compteur qui l'ouvre. Un module livré sans cette mise à jour
laisse l'état des lieux affirmer qu'une fonction est manuelle alors qu'elle ne
l'est plus.

Respecter le contrat : retourner `None` plutôt qu'approximer, journaliser sous
`[NOM]`, ne jamais lever pour une donnée manquante (lever seulement quand la
source elle-même est inaccessible), et livrer une suite hors ligne avec le
réseau simulé.
