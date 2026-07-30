# 🎯 Implémentation Sélection Hybride : FIXE (mes_symbols) + ALÉATOIRE (popular)

**Date:** 2025-01-XX  
**Fichier modifié:** `optimisateur_hybride.py`

---

## 🎯 Objectif

Remplacer la logique de sélection 3-tiers (completion only) par une stratégie hybride qui :
1. **Partie FIXE (60%)** : Priorité absolue aux symboles de `mes_symbols.txt` (portefeuilles actuels)
2. **Partie ALÉATOIRE (40%)** : Complète avec sélection randomisée dans `popular_symbols.txt`

**Rationale :** Optimiser pour les stocks RÉELLEMENT tradés tout en maintenant diversité/opportunités

---

## ✅ Modifications Effectuées

### 1. **Signature de fonction** (ligne 133)
```python
def clean_sector_cap_groups(..., fixed_ratio: float = 0.6) -> Dict[...]:
```
- Ajout paramètre `fixed_ratio` (défaut 60% fixe, 40% aléatoire)

### 2. **Imports globaux** (lignes 31-38)
```python
from symbol_manager import (
    # ... imports existants ...
    get_popular_symbols_by_sector, get_all_popular_symbols  # NOUVEAUX
)
```

### 3. **Nouvelle logique de sélection** (lignes 158-235)

**ÉTAPE 1 : PARTIE FIXE (mes_symbols)**
```python
personal_symbols = get_symbols_by_sector_and_cap(
    sector=sector,
    cap_range=cap,
    list_type='personal'  # mes_symbols.txt
)
target_fixed_count = max(1, int(min_symbols * fixed_ratio))  # 60% de 6 = 3-4 symboles
fixed_core = personal_symbols[:target_fixed_count]
```

**ÉTAPE 2 : PARTIE ALÉATOIRE (popular_symbols)**
```python
if len(base) < min_symbols:
    # 2A. Popular MÊME secteur (randomisé)
    popular_same_sector = get_popular_symbols_by_sector(sector, max_count=100, exclude_symbols=exclude_set)
    random.shuffle(popular_same_sector)  # ⚡ RANDOMISATION
    added = popular_same_sector[:needed]
    
    # 2B. Fallback transsectoriel (randomisé)
    if len(base) < min_symbols:
        all_popular = get_all_popular_symbols(max_count=200, exclude_symbols=exclude_set)
        random.shuffle(all_popular)  # ⚡ RANDOMISATION
        added = all_popular[:needed]
```

**ÉTAPE 3 : RÉDUCTION (garde priorité fixe)**
```python
if len(base) > max_symbols:
    # GARDE TOUS les symboles fixes
    extra_symbols = [s for s in base if s not in fixed_core]
    random.shuffle(extra_symbols)
    keep_count = max_symbols - len(fixed_core)
    base = list(fixed_core) + extra_symbols[:keep_count]
```

---

## 📊 Comportement Attendu

### Exemple : Secteur Technology, Cap Range Large, min_symbols=6

**Input:**
- `mes_symbols.txt` (personal) : AAPL, MSFT, GOOGL (3 symboles Technology/Large)
- `popular_symbols.txt` : NVDA, META, TSLA, AMD, INTC, ... (100+ symboles)

**Output avec fixed_ratio=0.6:**
```
🔒 [Technology][Large] Partie FIXE: 3 symboles de mes_symbols
   → AAPL, MSFT, GOOGL (priorité absolue)

🎲 [Technology][Large] Ajout ALÉATOIRE: 3 symboles (même secteur)
   → NVDA, AMD, META (tirage aléatoire dans popular Technology)

✅ [Technology][Large] Final: 6 symboles (3 fixes + 3 aléatoires)
```

**Avantages :**
- ✅ Optimisation **colle aux portefeuilles réels**
- ✅ Diversité via **randomisation** (chaque run = combinaisons différentes)
- ✅ **Garde toujours** les symboles fixes même si réduction nécessaire

---

## 🔧 Paramètres Configurables

| Paramètre | Défaut | Description |
|-----------|--------|-------------|
| `fixed_ratio` | 0.6 | Proportion de symboles fixes (60% mes_symbols, 40% random) |
| `min_symbols` | 6 | Taille minimale d'un groupe secteur×cap |
| `max_symbols` | 15 | Taille maximale (réduction si dépassement) |

**Pour ajuster :** Modifier l'appel à `clean_sector_cap_groups()` (ligne ~850+) :
```python
cleaned = clean_sector_cap_groups(
    sector_cap_ranges,
    min_symbols=6,
    max_symbols=15,
    fixed_ratio=0.7  # 70% fixe, 30% aléatoire
)
```

---

## 🧪 Validation

### 1. Vérifier les fichiers sources
```bash
# Vérifier que mes_symbols.txt contient vos portefeuilles
cat src/mes_symbols.txt

# Vérifier la base popular_symbols
cat src/popular_symbols.txt
```

### 2. Tester avec symboles minimaux
```python
# Lancer optimisation sur 1 secteur
python src/optimisateur_hybride.py --sectors Technology --test-mode
```

### 3. Logs à surveiller
```
🔒 [Technology][Large] Partie FIXE: X symboles de mes_symbols
🎲 [Technology][Large] Ajout ALÉATOIRE: Y symboles (même secteur)
✅ [Technology][Large] Final: Z symboles (X fixes + Y aléatoires)
```

**Attendu :** 
- X ≈ 60% de min_symbols
- Y complète jusqu'à min_symbols
- X + Y ≤ max_symbols

---

## 🔗 Fichiers Liés

- **Modifié:** `optimisateur_hybride.py` (fonction `clean_sector_cap_groups`)
- **Dépendances:** `symbol_manager.py` (fonctions `get_symbols_by_sector_and_cap`, `get_popular_symbols_by_sector`, `get_all_popular_symbols`)
- **Données:** `mes_symbols.txt` (portefeuilles), `popular_symbols.txt` (univers)
- **Base SQLite:** `stock_analysis.db` (tables `symbols`, `symbol_lists`)

---

## 📝 Historique des Modifications

### Version 1.0 (2025-01-XX)
- ✅ Implémentation logique hybride FIXE + ALÉATOIRE
- ✅ Ajout paramètre `fixed_ratio` (default 0.6)
- ✅ Randomisation pour diversité (`random.shuffle`)
- ✅ Protection des symboles fixes lors réduction
- ✅ Logs détaillés (🔒 fixe, 🎲 aléatoire, ✅ final)

---

## 🎓 Contexte Technique

**Ancienne logique (3-tiers completion):**
```
IF group < min → Complete avec popular same sector
IF still < min → Merge avec autres cap_range same sector
IF still < min → Fallback transsectoriel
IF group > max → Reduce à max (FIFO)
```

**Nouvelle logique (FIXE + ALÉATOIRE):**
```
ALWAYS: Start avec personal (mes_symbols) → PARTIE FIXE
IF < min → Random selection dans popular → PARTIE ALÉATOIRE
IF > max → Reduce mais GARDE tous les fixes + random des extras
```

**Bénéfice clé :** Garantit que l'optimisation **travaille sur VOS stocks** en priorité

---

## ⚠️ Notes Importantes

1. **Cache :** Le cache (100 jours) stocke les résultats nettoyés. Pour tester la nouvelle logique immédiatement, supprimez `cache_data/cleaned_groups_cache.pkl`

2. **Randomisation :** Chaque exécution peut générer des combinaisons différentes pour la partie aléatoire. C'est **voulu** (exploration de l'espace de recherche).

3. **Si mes_symbols.txt vide :** Fallback automatique 100% sur popular_symbols (comportement gracieux)

4. **Sync SQLite :** S'assurer que `mes_symbols.txt` est bien synchro dans SQLite via `sync_txt_to_sqlite()`

---

**Validation finale :** ✅ Syntaxe Python vérifiée avec `py_compile`
