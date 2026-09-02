# agent/Test/

Suites unitaires. **Toutes hors ligne** : ni réseau, ni clé, ni compte
courtier, ni fichier du projet.

```powershell
.venv\Scripts\python.exe agent\Test\test_agent.py     # 43 — moteur, portefeuille
.venv\Scripts\python.exe agent\Test\test_edgar.py     # 22 — dépôts, dilution, détention
.venv\Scripts\python.exe agent\Test\test_sources.py   # 21 — SSR, marché
.venv\Scripts\python.exe agent\Test\test_broker.py    # 17 — pont courtier, spread, halt
.venv\Scripts\python.exe agent\Test\test_ibkr.py      # 40 — producteur TWS, emprunt, taux
.venv\Scripts\python.exe agent\Test\test_scanner.py   # 31 — découverte F1, bornes R6 importées
.venv\Scripts\python.exe agent\Test\test_doctor.py    # 11 — preflight, heures de marché
```

`test_ibkr.py`, `test_scanner.py` et `test_doctor.py` remplacent le client
`ib_async` par un faux : ni TWS, ni Gateway, ni compte, ni réseau. Ce qu'ils
protègent n'est pas la plomberie du protocole mais les traductions qui, fausses,
passeraient inaperçues — l'unité du volume américain en tête.

Deux tests méritent d'être compris avant d'être modifiés :

**`test_aucune_borne_n_est_ecrite_en_dur_dans_le_module`** lit le fichier source
de `scanner.py` et échoue si un seuil de R6 y réapparaît en dur. C'est ce qui
garantit que le balayage et le crible ne divergeront pas en silence.

**`test_la_sonde_ne_touche_pas_le_reseau_reel`** existe parce que le piège
ci-dessous est retombé. Après une découpe de module, un test patchait
`probe_port` dans `ibkr` alors que la fonction vivait désormais dans
`ib_ports` : la vraie sonde tournait, et comme Gateway écoutait, elle a répondu.
**Patcher là où la fonction est DÉFINIE, pas là où elle est ré-exportée.**

`unittest`, pas `pytest` — rien à installer. Chaque fichier insère la racine du
projet dans `sys.path` pour être lançable directement.

## Le piège d'isolation, déjà tombé une fois

`MarketSource()` construit par défaut un **vrai** `BrokerBridge`, qui lit
`docu/portefeuille/broker_snapshot.json`. Un test qui instancie `MarketSource`
sans injecter de pont lit donc les cotations réelles du projet, et ses fixtures
sont silencieusement écrasées par le marché.

C'est arrivé : un test annonçait une clôture à 1,46, le fichier réel imposait
1,39, et le test a échoué alors que le code avait raison.

**Toujours injecter `absent_bridge()`** (défini dans `test_sources.py`) ou un
`BrokerBridge` pointant sur un répertoire temporaire. Même vigilance pour
`SsrCalendar` et le cache EDGAR : passer un `cache_dir` temporaire.

## Ce qui doit rester couvert

Les tests qui comptent le plus ne vérifient pas des calculs mais des **refus** :

- une donnée d'emprunt absente bloque une position vendeuse ;
- un statut Rule 201 inconnu bloque, il ne laisse pas passer ;
- la configuration piège de R9 bloque quand trois conditions sont réunies ;
- un titre suspendu bloque ;
- les bornes de compte (perte journalière, pertes consécutives, drawdown)
  bloquent.

Si une de ces assertions passe au vert après une refonte, vérifier que la règle
bloque encore réellement — pas qu'on a assoupli le test.

## Simuler le réseau

`requests.Session.get` est remplacé par `unittest.mock.patch.object`, jamais
appelé pour de vrai. yfinance est remplacé en réassignant `_ticker_factory`,
ce qui évite d'importer la bibliothèque dans les tests.

Les échantillons de données (`SAMPLE` du fichier NASDAQ, `IBKR_LIVE`) sont des
extraits **fidèles au format réellement observé**, guillemets et lignes
parasites compris. Ne pas les « nettoyer » : c'est justement ce qui casse en
production.
