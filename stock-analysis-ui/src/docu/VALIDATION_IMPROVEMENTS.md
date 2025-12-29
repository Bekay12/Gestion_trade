# Amélioration Validation & Anti-Overfitting

## 📋 Plan d'implémentation

### Phase 1: Nettoyage des groupes ✅ COMPLÈTE
- [x] Filtrer les cellules secteur×cap avec <5 symboles
- [x] Essayer de compléter avec symboles populaires du même secteur
- [x] Limiter à 12 symboles max (échantillonnage aléatoire si >12)
- [x] Logger les groupes ignorés dans `ignored_groups.log`
- [x] Paramètre configurable: `MIN_SYMBOLS_PER_GROUP = 5`, `MAX_SYMBOLS_PER_GROUP = 12`

### Phase 2: Validation temporelle
- [ ] Implémenter split train/val (18-24 mois train, 3-6 mois val)
- [ ] Hold-out final: réserver 3-6 derniers mois
- [ ] Seuil de validation: gain_per_trade >= seuil, max_drawdown <= seuil
- [ ] Rejeter les paramètres qui échouent la validation

### Phase 3: Régularisation
- [ ] Resserrer bornes: coeffs [0.5, 2.5], thresholds plus étroits
- [ ] Pénaliser configs avec trop peu de trades (<3 par an)
- [ ] Ajouter objectif composite: gain - penalty(complexity)

### Phase 4: Cache & Performance
- [ ] Préchauffer DERIV_CACHE si price_features actives
- [ ] Limiter n_jobs = min(cpu_count - 1, len(stocks))
- [ ] Vérifier TTL fundamentals avant fetch

## 🎯 Métriques cibles post-amélioration
- Overfitting: <20% (au lieu de 35-40%)
- Groupes: tous ≥5 symboles
- Validation: out-of-sample obligatoire avec seuil
- Hold-out: test final sur derniers mois

## 📊 Configuration actuelle
```python
MIN_SYMBOLS_PER_GROUP = 5
TRAIN_MONTHS = 18
VAL_MONTHS = 6
HOLDOUT_MONTHS = 3
MIN_GAIN_PER_TRADE = 1.0  # Seuil minimal $
MAX_DRAWDOWN_PCT = 15.0   # Seuil maximal %
MIN_TRADES_PER_YEAR = 3
```
