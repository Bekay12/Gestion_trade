# 📊 Gestion des Symboles - SQLite

## Vue d'ensemble

Le système de gestion des symboles utilise **SQLite** pour centraliser et organiser tous les symboles boursiers. Cela remplace l'approche précédente basée sur des fichiers `.txt` simples.

## Avantages

✅ **Métadonnées** - Secteur, capitalisation, date d'ajout  
✅ **Requêtes rapides** - Index sur symbol, secteur, cap_range, type de liste  
✅ **Synchronisation** - Fichiers txt -> SQLite automatique  
✅ **Flexibilité** - Filtrage par secteur, capitalisation, liste type  
✅ **Historique** - Suivi des symboles actifs/inactifs  

## Initialisation

```bash
python init_symbols.py
```

Cela va :
1. Créer la table `symbols` dans `stock_analysis.db`
2. Synchroniser `popular_symbols.txt` -> liste 'popular'
3. Synchroniser `mes_symbols.txt` -> liste 'personal'
4. Créer les index pour requêtes rapides

## Structure de la base de données

```sql
CREATE TABLE symbols (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT UNIQUE NOT NULL,
    sector TEXT,
    market_cap_range TEXT,  -- 'Small', 'Mid', 'Large', 'Giant', 'Unknown'
    market_cap_value REAL,  -- En milliards $
    list_type TEXT,  -- 'popular', 'personal', 'watchlist'
    added_date TIMESTAMP,
    last_checked TIMESTAMP,
    is_active BOOLEAN
)
```

## Utilisation

### 1. Charger les symboles d'une liste

```python
from symbol_manager import get_symbols_by_list_type

# Tous les symboles populaires
symbols = get_symbols_by_list_type('popular')

# Symboles personnels actifs uniquement
personal = get_symbols_by_list_type('personal', active_only=True)
```

### 2. Filtrer par secteur

```python
from symbol_manager import get_symbols_by_sector

tech = get_symbols_by_sector('Technology')
```

### 3. Filtrer par capitalisation

```python
from symbol_manager import get_symbols_by_cap_range

large_cap = get_symbols_by_cap_range('Large')
```

### 4. Combinaison secteur + capitalisation

```python
from symbol_manager import get_symbols_by_sector_and_cap

tech_large = get_symbols_by_sector_and_cap('Technology', 'Large')
```

### 5. Obtenir la liste de tous les secteurs

```python
from symbol_manager import get_all_sectors

sectors = get_all_sectors()
# ['Technology', 'Healthcare', 'Finance', ...]
```

### 6. Obtenir toutes les gammes de capitalisation

```python
from symbol_manager import get_all_cap_ranges

cap_ranges = get_all_cap_ranges()
# ['Small', 'Mid', 'Large', 'Giant', 'Unknown']
```

### 7. Compter les symboles

```python
from symbol_manager import get_symbol_count

total = get_symbol_count()  # Tous
popular = get_symbol_count('popular')  # Par liste type
```

### 8. Obtenir les infos d'un symbole

```python
from symbol_manager import get_symbol_info

info = get_symbol_info('AAPL')
print(info['sector'])  # 'Technology'
print(info['market_cap_range'])  # 'Giant'
print(info['market_cap_value'])  # ~3000 (milliards $)
```

### 9. Désactiver/Réactiver un symbole

```python
from symbol_manager import deactivate_symbol, activate_symbol

deactivate_symbol('AAPL')  # Cache sans supprimer
activate_symbol('AAPL')    # Réactive
```

### 10. Supprimer un symbole

```python
from symbol_manager import delete_symbol

delete_symbol('AAPL')  # Supprime complètement
```

## Intégration dans `qsi.py`

La fonction `load_symbols_from_txt()` utilise maintenant SQLite automatiquement :

```python
from qsi import load_symbols_from_txt

# Charge depuis SQLite (ou fallback txt si SQLite non disponible)
symbols = load_symbols_from_txt('popular_symbols.txt')
```

## Gammes de capitalisation

```
Small:  < 2 milliards $
Mid:    2-10 milliards $
Large:  10-200 milliards $
Giant:  > 200 milliards $
Unknown: Données manquantes
```

## Synchronisation automatique

Chaque fois que vous appelez `load_symbols_from_txt()`, les symboles du fichier `.txt` sont synchronisés vers SQLite (si disponible).

Pour une synchronisation manuelle :

```python
from symbol_manager import sync_txt_to_sqlite

sync_txt_to_sqlite('popular_symbols.txt', 'popular')
sync_txt_to_sqlite('mes_symbols.txt', 'personal')
```

## Statistiques actuelles

```
Total symboles:       511
- Populaires:         441
- Personnels:         70

Secteurs:             12
Gammes de cap:        5
```

## Maintenance

### Réinitialiser complètement

```python
from symbol_manager import init_symbols_table
init_symbols_table()  # Recrée la table avec index
```

### Voir la démo complète

```bash
python symbol_manager.py --demo
```

## Notes

- Les secteurs et cap_ranges sont obtenus dynamiquement via yfinance
- Les métadonnées sont **cached** pour éviter trop d'appels API
- Les symboles sont activés par défaut (is_active = 1)
- Les fichiers `.txt` peuvent toujours être modifiés manuellement, la synchro récupérera les changements

## Prochaines étapes

1. Ajouter des fonctions dans `optimisateur_hybride.py` pour générer les coefficients **par sector + cap_range**
2. Utiliser ces requêtes filtrées pour paralléliser l'optimisation
3. Ajouter une UI pour gérer les symboles (ajouter/retirer/filtrer)
