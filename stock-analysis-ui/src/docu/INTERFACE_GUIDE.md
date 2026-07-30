# 📈 Guide d'Utilisation - Interface Web Stock Analysis

## Aperçu

L'interface web Stock Analysis permet d'accéder à toutes les fonctionnalités de l'application de bureau directement depuis un navigateur web. Elle utilise le même moteur d'analyse que la version PyQt5.

## 🎯 4 Onglets Principaux

### 1️⃣ Onglet "Analyser" (🔍 Analyser)

**Fonction:** Analyser un symbole unique pour obtenir un signal de trading

**Utilisation:**
- Entrez un symbole boursier (ex: AAPL, MSFT, GOOGL, TSLA)
- Sélectionnez la période historique (1M, 3M, 6M, 1A, 2A, 5A)
- Cliquez sur **"Analyser"**

**Résultats affichés:**
- **Signal:** BUY (Achat), SELL (Vente) ou HOLD (Attendre)
- **Prix Actuel:** Dernier prix connu
- **Fiabilité:** Pourcentage de confiance du signal (0-100%)
- **RSI:** Indice de Force Relative (0-100)
- **Tendance:** Haussière, Baissière ou Latérale
- **Domaine:** Secteur d'activité de la compagnie
- **Volume:** Volume moyen d'échange

**Exemple:**
```
AAPL → SIGNAL: BUY | PRIX: $185.50 | FIABILITÉ: 78%
```

---

### 2️⃣ Onglet "Listes" (📋 Listes)

**Fonction:** Gérer les listes de symboles à analyser

**3 Types de Listes:**

#### 📌 Symboles Populaires
- Symboles les plus analysés
- Affiche: AAPL, MSFT, GOOGL, TSLA, AMZN...
- Usage: Analyse rapide des valeurs connues

#### 👤 Mes Symboles
- Votre liste personnalisée
- Permet d'ajouter/retirer vos symboles favoris
- Format d'ajout: "MSFT, NFLX, CRM" (séparés par des virgules)

#### ⚙️ Liste Optimisation
- Symboles pour backtesting
- Utilisée dans l'onglet "Backtest"
- Jusqu'à 50 symboles maximum

**Exemple d'ajout:**
```
Entrez: NVIDIA, AMDQI, INTEL
Cliquez: "+ Ajouter aux Populaires"
✅ 3 symboles ajoutés!
```

---

### 3️⃣ Onglet "Batch" (📊 Batch)

**Fonction:** Analyser plusieurs symboles en une seule requête

**Utilisation:**
- Entrez jusqu'à **20 symboles** (séparés par des virgules)
- Sélectionnez la période
- Cliquez sur **"Analyser Lot"**

**Résultats:**
Tableau complet avec tous les symboles et leurs signaux

**Exemple:**
```
Symboles: AAPL, MSFT, GOOGL, NVDA, META, NVDA
Période: 1 Ans
↓
Résultats:
┌─────────┬────────┬───────┬───────────┐
│ Symbole │ Signal │ Prix  │ Fiabilité │
├─────────┼────────┼───────┼───────────┤
│ AAPL    │ BUY    │ $185  │    78%    │
│ MSFT    │ BUY    │ $415  │    85%    │
│ GOOGL   │ SELL   │ $142  │    62%    │
│ NVDA    │ BUY    │ $874  │    91%    │
└─────────┴────────┴───────┴───────────┘
```

---

### 4️⃣ Onglet "Backtest" (🔬 Backtest)

**Fonction:** Tester une stratégie de trading sur l'historique

**Paramètres:**
- **Symbole:** Un seul symbole à backtester
- **Période:** Étendue historique de test
- **MA Rapide:** Moyenne mobile rapide (par défaut: 9)
- **MA Lente:** Moyenne mobile lente (par défaut: 21)

**Résultats affichés:**
- **Gain Total:** Profit/perte en pourcentage
- **Win Rate:** % de trades gagnants
- **Nb Trades:** Nombre de transactions
- **Gagnants:** Nombre de trades rentables

**Exemple:**
```
Symbole: AAPL
Période: 1 Ans
MA Rapide: 9
MA Lente: 21
↓
RÉSULTATS:
├─ Gain Total:    +12.45%
├─ Win Rate:       65.5%
├─ Nb Trades:      47
└─ Gagnants:       31
```

---

## 📊 Tableau de Bord Principal

Le haut de page affiche les **statistiques globales:**

- **Signaux Total:** Nombre total de signaux enregistrés
- **Signaux Achat:** Nombre de signaux BUY
- **Signaux Vente:** Nombre de signaux SELL
- **Fiabilité Moyenne:** Pourcentage moyen de confiance

---

## ⌨️ Raccourcis Clavier

- **ENTRÉE** dans le champ "Symbole" → Lance l'analyse
- **Tab** → Navigue entre les champs du formulaire
- **Clic sur les onglets** → Change de section

---

## 🔧 Interprétation des Signaux

### Signal: BUY (Achat) 🟢
- **Confiance:** > 70% = **Fiable**
- **Action:** Acheter ou ajouter à la position
- **Indicateurs:** RSI faible, tendance haussière

### Signal: SELL (Vente) 🔴
- **Confiance:** > 70% = **Fiable**
- **Action:** Vendre ou fermer la position
- **Indicateurs:** RSI élevé, tendance baissière

### Signal: HOLD (Attendre) 🟡
- **Confiance:** Variable
- **Action:** Observer, pas de trading
- **Indicateurs:** Marché indécis

---

## 💡 Conseils d'Utilisation

1. **Vérifiez les niveaux de fiabilité** - Ne tradez que si > 70%
2. **Diversifiez** - Utilisez plusieurs symboles (Batch)
3. **Testez d'abord** - Utilisez Backtest avant d'investir
4. **Analysez les tendances** - Regardez les périodes (1M vs 5A)
5. **Surveillez le volume** - Volume élevé = Signal plus fiable

---

## 📡 API Endpoints (Pour les développeurs)

L'interface web utilise ces endpoints REST:

```
POST   /api/analyze           - Analyser un symbole
POST   /api/analyze-popular   - Analyser listes populaires
POST   /api/analyze-batch     - Analyser multiples symboles
GET    /api/lists             - Récupérer les listes
POST   /api/lists/<type>      - Ajouter/retirer symboles
POST   /api/backtest          - Exécuter un backtest
GET    /api/signals           - Récupérer les signaux
GET    /api/stats             - Obtenir les stats globales
GET    /health                - Vérifier l'état du serveur
```

---

## 🐛 Dépannage

### "Aucun signal fiable trouvé"
- Le symbole n'existe pas ou les données manquent
- Essayez avec une période plus longue
- Vérifiez que le symbole est correct (ex: AAPL vs AAL)

### "Erreur de connexion"
- Vérifiez votre connexion Internet
- Le serveur peut être en redémarrage
- Attendez 30 secondes et réessayez

### Les onglets ne changent pas
- Videz le cache du navigateur (Ctrl+Shift+Del)
- Rechargez la page (F5)
- Essayez avec un autre navigateur

---

## 📞 Support

Pour plus d'informations ou signaler un problème:
- Consultez le README.md du projet
- Vérifiez les logs de l'API
- Testez avec curl: `curl https://stock-analysis-api-8dz1.onrender.com/health`

---

**Version:** 1.0  
**Dernière mise à jour:** Janvier 2025  
**Lien de déploiement:** https://stock-analysis-api-8dz1.onrender.com/
