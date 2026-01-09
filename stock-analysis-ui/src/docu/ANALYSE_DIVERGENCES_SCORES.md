# 🔍 ANALYSE PROFONDE: Différences de Scores entre "Analyse" et "Analyse & Backtest"

**Date:** 22 janvier 2026  
**Analysé par:** Audit du code source  
**Statut:** ✅ 3 problèmes majeurs identifiés

---

## 📊 COMPARAISON DES DONNÉES (Logs vs Capture d'écran)

### Exemples de divergences observées:

| Action | Logs (Analyse & Backtest) | Capture (Analyse Simple) | Divergence |
|--------|---------------------------|-------------------------|-----------|
| IMNM | Score=5.30, CapRange=Unknown | Score=7.78, CapRange=Mid | ❌ ParamKey différente |
| OCS | Score=-0.10, CapRange=Unknown | Score=4.55, CapRange=Small | ❌ ParamKey différente |
| ARGX | Score=9.30, CapRange=? | Score=7.9, CapRange=Large | ❌ Coeffs différents |
| HROW | Score=3.30, CapRange=? | Score=3.3, CapRange=Small | ✅ Concordance |

---

## 🔴 PROBLÈME #1: Cap_range retourne "Unknown" au lieu de consulter la DB

### Localisation:
- [qsi.py#L1238-L1265](src/qsi.py#L1238)

### Code problématique:
```python
def get_cap_range_for_symbol(symbol: str) -> str:
    """Tente de récupérer le range de market cap via le cache financier."""
    try:
        if get_pickle_cache is not None:
            d = get_pickle_cache(symbol, 'financial', ttl_hours=24*365)
            if d is not None and isinstance(d, dict):
                mc_b = float(d.get('market_cap_val', 0.0) or 0.0)
                # Utiliser la fonction consolidée de symbol_manager
                from symbol_manager import classify_cap_range
                return classify_cap_range(mc_b)
    except Exception:
        pass
    return 'Unknown'  # ❌ TROP PESSIMISTE
```

### Problème:
- Si le cache pickle n'existe pas ou si l'exception est levée → retour immédiat `'Unknown'`
- **NE CHERCHE PAS** dans la base de données `symbols.db` ou la table SQLite
- Résultat: IMNM et OCS retournent `"Unknown"` au lieu de `"Mid"` et `"Small"`

### Conséquence:
```
Avec cap_range="Unknown":
  ParamKey cherchée = "Healthcare_Unknown" → PAS TROUVÉE
  Fallback = "Healthcare" seul
  
Avec cap_range="Mid":
  ParamKey cherchée = "Healthcare_Mid" → TROUVÉE ✅
  Utilise les bons paramètres optimisés
```

**Résultat:** Scores divergent car les **coefficients et seuils utilisés sont différents**

---

## 🔴 PROBLÈME #2: Aucune tentative de fallback vers la DB ou yfinance

### Localisation:
- [main_window.py#L970-L1000](src/ui/main_window.py#L970)

### Code actuel (incohérent):
```python
cap_range = qsi.get_cap_range_for_symbol(symbol)  # Retourne souvent "Unknown"

# ✅ Il existe un fallback pour cap_range dans analyse_backtest_batch
if CAP_FALLBACK_ENABLED and (cap_range == "Unknown" or not cap_range):
    best_params_all = qsi.extract_best_parameters()
    for fallback_cap in ["Large", "Mid", "Mega"]:
        test_key = f"{domaine}_{fallback_cap}"
        if test_key in best_params_all:
            cap_range = fallback_cap
            break
```

### Le problème:
- Le fallback essaie juste les cap_range génériques (`["Large", "Mid", "Mega"]`)
- **NE CHERCHE JAMAIS DANS LA DB** pour le cap_range réel du symbole
- Si la DB a "Small" mais les params optimisés n'ont que "Large" → on utilise "Large" par erreur

### Différence avec le Backtest:
Dans les logs, le backtest utilise `cap_range="Unknown"` alors que:
1. La capture d'écran (analyse simple) affiche le vrai cap_range
2. Le code de **fallback** pour cap_range existe déjà mais...
3. ...est appliqué APRÈS `get_trading_signal()` au lieu de AVANT

---

## 🔴 PROBLÈME #3: Incohérence domaine entre yfinance et DB

### Localisation:
- [main_window.py#L964](src/ui/main_window.py#L964)
- [optimisateur_hybride.py#L76](src/optimisateur_hybride.py#L76)

### Code problématique:
```python
# Dans main_window.py (analyse simple)
try:
    if qsi.OFFLINE_MODE:
        fin_cache = qsi.get_pickle_cache(symbol, 'financial', ttl_hours=24*365)
        domaine = fin_cache.get('sector', 'Inconnu') if fin_cache else "Inconnu"
    else:
        info = yf.Ticker(symbol).info  # ⚠️ Appel en ligne potentiellement lent
        domaine = info.get("sector", "Inconnu")
except Exception:
    domaine = "Inconnu"
```

### Les problèmes:
1. **Mode offline vs online:** Si cache expiré → revient à "Inconnu"
2. **Secteurs non standardisés:** yfinance peut retourner "Health Care" et la DB "Healthcare"
3. **Pas de normalisation:** Les noms de secteurs ne sont pas unifiés

### Exemple dans les logs:
```
⚪ IREN: Signal=NEUTRE, Score=0.11, 
        Domaine=Financial Services,  ← yfinance
        CapRange=Large, 
        ParamKey=Financial Services_Large
        
Mais si la DB avait "Finance" ou "Financials":
  ParamKey ne trouverait rien → Fallback
```

---

## ✅ SOLUTIONS RECOMMANDÉES

### Solution 1: Améliorer `get_cap_range_for_symbol()`

**Ajouter un fallback vers la DB SQLite:**

```python
def get_cap_range_for_symbol(symbol: str) -> str:
    """Récupère le cap_range avec stratégie complète:
    1. Cache pickle
    2. DB SQLite (symbols.db)
    3. Calcul via yfinance.info
    4. Unknown
    """
    # 1️⃣ Essayer le cache
    if get_pickle_cache is not None:
        d = get_pickle_cache(symbol, 'financial', ttl_hours=24*365)
        if d is not None and isinstance(d, dict):
            mc_b = float(d.get('market_cap_val', 0.0) or 0.0)
            if mc_b > 0:
                from symbol_manager import classify_cap_range
                return classify_cap_range(mc_b)
    
    # 2️⃣ Essayer la DB SQLite
    try:
        from symbol_manager import classify_cap_range_for_symbol
        cap = classify_cap_range_for_symbol(symbol)
        if cap and cap != 'Unknown':
            return cap
    except Exception:
        pass
    
    # 3️⃣ Essayer yfinance en ligne
    try:
        if not OFFLINE_MODE:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            market_cap = ticker.info.get('marketCap')
            if market_cap:
                from symbol_manager import classify_cap_range
                return classify_cap_range(market_cap / 1e9)
    except Exception:
        pass
    
    return 'Unknown'
```

---

### Solution 2: Appliquer le fallback AVANT `get_trading_signal()`

**Dans [main_window.py#L970-L1000](src/ui/main_window.py#L970):**

```python
# AVANT
cap_range = qsi.get_cap_range_for_symbol(symbol)  # Peut retourner Unknown

# ✅ AJOUTER le fallback PROACTIF (pas réactif après)
if cap_range == "Unknown":
    # Chercher dans la DB d'abord
    try:
        from symbol_manager import get_symbol_by_name
        db_sym = get_symbol_by_name(symbol)
        if db_sym and db_sym.get('cap_range'):
            cap_range = db_sym['cap_range']
    except:
        pass
    
    # Si toujours Unknown, essayer les fallbacks standards
    if cap_range == "Unknown" and domaine in best_params_all:
        for fallback_cap in ["Large", "Mid", "Small", "Mega"]:
            test_key = f"{domaine}_{fallback_cap}"
            if test_key in best_params_all:
                cap_range = fallback_cap
                break

# ✅ PUIS appeler get_trading_signal avec le cap_range correct
sig, score, ... = get_trading_signal(
    prices, volumes, domaine=domaine,
    cap_range=cap_range,  # ← Correct maintenant!
    seuil_achat=seuil_achat_opt,
    seuil_vente=seuil_vente_opt
)
```

---

### Solution 3: Normaliser les noms de secteurs

**Créer une fonction de normalisation:**

```python
def normalize_sector_name(sector: str) -> str:
    """Normalise les noms de secteurs depuis yfinance vers la DB."""
    normalization = {
        'Health Care': 'Healthcare',
        'Financials': 'Financial Services',
        'Information Technology': 'Technology',
        'Industrials': 'Industrial',
        'Real Estate': 'Real Estate',
        # ... etc
    }
    return normalization.get(sector, sector)

# À utiliser dans:
# 1. main_window.py: domaine = normalize_sector_name(info.get("sector", "Inconnu"))
# 2. api.py: domaine = normalize_sector_name(info.get("sector", "Inconnu"))
# 3. optimisateur_hybride.py: return normalize_sector_name(sector)
```

---

## 🎯 IMPACT ESTIMÉ

### Avant correction:
- ❌ Analyse simple: scores corrects (utilise cap_range de la DB)
- ❌ Analyse & Backtest: scores divergents (cap_range=Unknown)
- ❌ Incohérence UI/Backtest

### Après correction:
- ✅ Analyse simple: scores corrects
- ✅ Analyse & Backtest: scores identiques
- ✅ Cohérence complète

---

## 📝 CHECKLIST DE CORRECTION

- [ ] Implémenter fallback vers DB dans `get_cap_range_for_symbol()`
- [ ] Ajouter normalisation des secteurs
- [ ] Tester avec les 5 symboles de divergence (IMNM, OCS, ARGX, HROW, PRCT)
- [ ] Vérifier logs "ParamKey" pour chaque symbole
- [ ] Comparer scores avant/après sur un batch complet

---

## 🔗 FICHIERS À MODIFIER

1. **src/qsi.py** - Lignes 1238-1265 (fonction `get_cap_range_for_symbol`)
2. **src/ui/main_window.py** - Lignes 970-1000 (boucle d'analyse)
3. **src/api.py** - Lignes 320-340 (analyse API)
4. **Créer:** `src/sector_normalizer.py` (nouveau module)

---

## 🚀 PRIORITÉ

**HAUTE** - Cela explique toutes les divergences observées entre Analyse simple et Backtest
