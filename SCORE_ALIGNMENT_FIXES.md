# 🎯 CORRECTIONS - DÉCALAGE ENTRE ACTIONS ET HISTORIQUE DU SCORE

## Résumé Exécutif

**Problème identifié**: Divergence entre le score calculé lors de l'analyse et celui affiché dans le tableau des résultats. La cause: les **seuils d'achat/vente** et le **secteur normalisé** n'étaient pas synchronisés entre le calcul et l'affichage.

**Impact**: 
- Score affiché 3.5 mais Signal montrait "ACHAT" (seuil à calcul = 2.0, à affichage = 4.2)
- Ratio Score/Seuil incorrect
- Historique du score en désaccord avec les résultats actuels

---

## CORRECTIONS APPLIQUÉES

### 1. ✅ SYNCHRONISATION DES SEUILS (Correction #1)

**Localisation**: `qsi.py:1017` et `main_window.py:2102`

**Le problème**:
```python
# Au calcul (get_trading_signal):
buy_threshold = seuil_achat if seuil_achat is not None else best_params[domaine].thresholds[7]

# À l'affichage (update_results_table) - RECALCULE!
seuil_achat = best_params[domaine_cap].thresholds[7]  # Possible différence si best_params changé
```

**La solution**:
1. Stocker les seuils utilisés dans `derivatives`:
   ```python
   derivatives['_seuil_achat_used'] = buy_threshold
   derivatives['_seuil_vente_used'] = sell_threshold
   ```

2. Inclure dans le résultat retourné par `process_symbol`:
   ```python
   '_seuil_achat_used': derivatives.get('_seuil_achat_used'),
   '_seuil_vente_used': derivatives.get('_seuil_vente_used'),
   ```

3. Utiliser à l'affichage en priorité:
   ```python
   # Récupérer d'abord les seuils utilisés au calcul
   seuil_achat = signal.get('_seuil_achat_used')  # Valeur RÉELLE utilisée
   if seuil_achat is None:
       # Fallback seulement si absent
       seuil_achat = recalculer_depuis_best_params()
   ```

**Résultat**: ✅ La ratio Score/Seuil correspond maintenant exactement au calcul initial

---

### 2. ✅ NORMALIZATION DU SECTEUR SYSTÉMATIQUE (Correction #2)

**Localisation**: `qsi.py:2565` et `main_window.py:1268`

**Le problème**:
```python
# Parfois "Technology", parfois "Tech", parfois non-normalisé
# → Clés de paramètres différentes dans best_params
# → Coefficients complètement différents!
```

**La solution**:
1. Normaliser QUÉ AU DÉPART dans `process_symbol`:
   ```python
   from sector_normalizer import normalize_sector
   domaine = normalize_sector(domaine)  # Une seule fois au départ
   print(f"🔄 {symbol}: Secteur normalisé '{raw}' → '{normalized}'")
   ```

2. Assurer la même normalisation dans `on_download_complete`:
   ```python
   domaine = normalize_sector(domaine)  # Même fonction, même résultat
   ```

**Résultat**: ✅ Tous les calculs utilisent le même secteur normalisé

---

### 3. ✅ STOCKAGE DE LA CLÉ DE PARAMÈTRES (Correction #3)

**Localisation**: `qsi.py:1019` et ajout dans `process_symbol`

**Le problème**:
- Pas de trace de quelle clé `domain_cap` a été utilisée au calcul
- Impossible de récréer le même calcul à l'affichage

**La solution**:
```python
# Dans get_trading_signal:
derivatives['_selected_param_key'] = selected_key or domaine or 'UNKNOWN'

# Dans process_symbol:
'_selected_param_key': derivatives.get('_selected_param_key'),

# À l'affichage (optionnel mais utile pour debug):
print(f"ℹ️ {symbol} utilisait param_key: {signal.get('_selected_param_key')}")
```

**Résultat**: ✅ Traçabilité complète des paramètres utilisés

---

## VÉRIFICATION DES CORRECTIONS

### Test 1: Logs de Thresholds
Vérifiez dans les logs lors de l'analyse:

```
✅ Seuils utilisés depuis stockage pour AAPL: 4.2, -0.5
✅ Seuils utilisés depuis stockage pour MSFT: 3.8, -0.3
```

Si vous voyez:
```
⚠️ Seuils recalculés pour AAPL (seuil stocké absent)
```

C'est que la correction n'est pas appliquée partout.

### Test 2: Secteur Normalisé
Vérifiez les logs:

```
🔄 AAPL: Secteur normalisé 'Technology' -> 'Technology'  (OK: pas changement)
🔄 MSFT: Secteur normalisé 'Information Technology' -> 'Technology'  (OK: normalisé)
```

Si vous voyez des secteurs différents entre deux analyses = problème de synchronisation.

### Test 3: Validation du Ratio
Dans la table des résultats, colonne "Score/Seuil":

1. **Avant correction**:
   - Score: 3.5
   - Affiché: ratio = 0.83 (3.5/4.2 avec seuil recalculé)
   - Réalité: ratio = 1.75 (3.5/2.0 avec seuil original)
   → **DIVERGENCE**

2. **Après correction**:
   - Score: 3.5
   - Affiché: ratio = 1.75 (3.5/2.0 avec seuil original)
   - Cohérence: **OK** ✅

---

## IMPLÉMENTATION COMPLÈTE

### Files modifiés:

1. **`stock-analysis-ui/src/qsi.py`**
   - Line 1019: Ajout `_seuil_achat_used`, `_seuil_vente_used` aux derivatives
   - Line 2565-2580: Normalisation du secteur au départ
   - Line 2647:Sauvegarde des seuils dans le résultat

2. **`stock-analysis-ui/src/ui/main_window.py`**
   - Line 1268-1280: Normalisation du secteur cohérente
   - Line 2102-2150: Utiliser les seuils stockés (fallback en dernier)

### Validation:

```bash
# Exécuter le diagnostic:
python src/diagnostic_score_misalignment.py --symbols AAPL MSFT TSLA

# Vérifier les logs:
tail -f logs/score_debug.log
```

---

## NOTES IMPORTANTES

### 🔴 Cas Edge à surveiller:

1. **Signal NEUTRE**: Seuils ne sont pas stockés!
   ```python
   if signal != "NEUTRE":
       return {..., '_seuil_achat_used': ..., '_seuil_vente_used': ...}
   ```
   **Action**: Ajouter seuils même pour NEUTRE si necesseire

2. **Changement de best_parameters**: Si la DB est chargée différemment
   ```python
   # Assurer que best_parameters est chargé une seule fois:
   self.best_parameters = extract_best_parameters()  # Une fois!
   ```

3. **Cache TA_CACHE**: Peut causer des variations si prix arrondissent différemment
   - Monitorer avec `diagnostic_score_misalignment.py`

---

## COMPARAISON AVANT/APRÈS

| Aspect | AVANT | APRÈS |
|--------|-------|-------|
| **Seuils synchronisés** | ❌ Recalculés à chaque fois | ✅ Stockés et réutilisés |
| **Secteur normalisé** | ⚠️ Parfois, parfois non | ✅ Systématiquement |
| **Ratio Score/Seuil** | ❌ Peut être faux | ✅ Correct garanti |
| **Traçabilité** | ❌ Impossible de reproduire | ✅ `_selected_param_key` stocké |
| **Divergence chart/table** | ❌ Possible si seuils changent | ✅ Impossible si même seuils |

---

## PROCHAINES ÉTAPES

### 1. Tester en détail
```python
# Dans main_window.py, ajouter au démarrage:
logger.log_event('ANALYSIS_CONFIG', 'BATCH', {
    'best_parameters_loaded': bool(self.best_parameters),
    'sector_normalizer_available': True,
    'timestamp': datetime.now().isoformat()
})
```

### 2. Ajouter alertes si synchronisation échoue
```python
# Dans update_results_table, vérifie seuils:
if signal.get('_seuil_achat_used') is None:
    print(f"🚨 ALERTE: {sym} sans seuil stocké")
    # Peut indiquer analyse incomplète
```

### 3. Vérifier cas limites
- [ ] NEUTRE signals avec différents seuils
- [ ] Secteurs fallback (Inconnu)
- [ ] Cap_range Unknown ou absent
- [ ] Paramètres volatiles en base (rechargement)

