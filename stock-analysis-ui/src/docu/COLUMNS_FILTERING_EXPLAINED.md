# 🔍 COMPRENDRE LES COLONNES FILTRÉES

## ❓ Question: "Pourquoi il ne reste que quelques paramètres malgré le nombre important de colonnes?"

### Réponse Courte
Le PDF n'affichait avant que les colonnes qui avaient **une valeur non-vide ET non-zéro** pour ce symbole. Si une colonne était vide ("") ou égale à 0 ou "0" pour ce symbole, elle n'était pas affichée.

**Maintenant:** Toutes les colonnes de `clean_columns` sont affichées, même si la valeur est "N/A".

---

## 🔍 Comprendre le Filtrage des Colonnes

### Étape 1: Filtrage au Niveau du Tableau UI
**Où:** `main_window.py` → `_get_clean_columns_and_data()`

```python
# Cette fonction filtre les colonnes VIDES à TOUS LES NIVEAUX
for col in all_columns:
    has_valid_data = False
    for result in self.current_results:
        value = result.get(col, '')
        # Vérifier si TOUS les résultats ont cette colonne
        if value and value != '' and value != 0 and value != '0':
            has_valid_data = True
            break
    
    if has_valid_data:
        valid_columns.append(col)

# Résultat: valid_columns ne contient que les colonnes qui ont
# au moins UNE valeur non-vide/non-zéro sur TOUS les symboles
```

**Exemple:**
```
Si vous avez 50 symboles
Et une colonne "XYZ" n'a une valeur que pour 1 seul symbole
→ Elle sera incluse dans clean_columns

Mais si une colonne "ABC" n'a QUE des 0 ou des vides
→ Elle sera EXCLUE de clean_columns
```

### Étape 2: Filtrage au Niveau du PDF
**Avant la modification:** Chaque symbole ne montrait que ses colonnes avec valeur

```python
# ANCIEN CODE
if value and value != '' and value != 0 and value != '0':
    cols_with_data.append((col, formatted_value))

# Résultat: Pour le symbole AAPL
# Si AAPL a une colonne "PEG" = 0
# → La colonne PEG n'apparait PAS dans le tableau d'AAPL
```

**Après la modification:** Toutes les colonnes s'affichent

```python
# NOUVEAU CODE
for col in clean_columns:
    if col not in columns_to_skip:
        value = stock_data.get(col, 'N/A')
        # Affiche TOUJOURS la colonne, même si valeur = 'N/A'
        cols_with_data.append((col, formatted_value))

# Résultat: Pour le symbole AAPL
# Si AAPL a une colonne "PEG" = 0
# → La colonne PEG apparait avec la valeur 0
```

---

## 📊 Exemple Concret

### Scénario
Vous analysez 50 symboles avec ces colonnes:
```
ROE, PEG, EBITDA, Dividend, PriceSale, EarningsGrowth, ...
(supposons 28 colonnes total)
```

### Avant Modification (Problématique)
**step 1 - Filtre global:** 
```
Colonnes retirées (zéros/vides partout):
- some_column: 0 pour tous
- another_column: '' pour tous

Colonnes conservées: 28 colonnes
```

**Step 2 - Filtre par symbole:**
```
Symbol AAPL:
- ROE: 25.5 ✅
- PEG: 1.2 ✅
- EBITDA: 0 ❌ (caché car = 0)
- Dividend: '' ❌ (caché car vide)
- PriceSale: 2.1 ✅
...
→ Affiche que 15 colonnes (au lieu de 28)

Symbol MSFT:
- ROE: 18.3 ✅
- PEG: 1.8 ✅
- EBITDA: 150 ✅
- Dividend: 2.5 ✅
- PriceSale: 3.2 ✅
...
→ Affiche 22 colonnes (au lieu de 28)
```

**Résultat:** Chaque symbole affiche un nombre différent de colonnes → Tableau incohérent

### Après Modification (Correct)
**Step 1 - Filtre global:** (inchangé)
```
Colonnes conservées: 28 colonnes
```

**Step 2 - Affichage par symbole:** (NOUVEAU)
```
Symbol AAPL:
- ROE: 25.5 ✅
- PEG: 1.2 ✅
- EBITDA: N/A (affiche N/A au lieu de cacher) ✅
- Dividend: N/A ✅
- PriceSale: 2.1 ✅
...
→ Affiche TOUTES les 28 colonnes

Symbol MSFT:
- ROE: 18.3 ✅
- PEG: 1.8 ✅
- EBITDA: 150 ✅
- Dividend: 2.5 ✅
- PriceSale: 3.2 ✅
...
→ Affiche TOUTES les 28 colonnes
```

**Résultat:** Tous les symboles affichent exactement les mêmes colonnes → Tableau cohérent

---

## 🎯 Les Filtres qui S'appliquent

### Filtre 1: `columns_to_skip` (Hard-coded)
Ces colonnes sont TOUJOURS exclues du tableau des métriques:

```python
columns_to_skip = {
    'Signal',           # ACHAT/VENTE (déjà affiché ailleurs)
    'Score',            # Score (déjà affiché ailleurs)
    'Prix',             # Déjà visible
    'Tendance',         # Données de graphique
    'RSI',              # De graphique
    'Volume moyen',     # Non pertinent
    'Consensus',        # Redondant
    '_analysis_id',     # Technique
    'DomaineOriginal',  # Interne
    'ConsensusMean',    # Interne
    'Symbole'           # Affichée comme titre
}
```

### Filtre 2: clean_columns (du tableau UI)
Seules ces colonnes sont passées au PDF:
```python
clean_columns = [col for col in all_columns if has_valid_data(col)]
# Une colonne est incluse si elle a au MOINS UNE valeur non-zéro
# dans TOUS les symboles analysés
```

### Filtre 3: Affichage par symbole (ANCIEN)
Chaque symbole ne montrait que ses colonnes non-vides:
```
Avant: if value != '' and value != 0:
        afficher(value)
Après: afficher(value) # toujours, même si N/A
```

---

## 🔢 Exemple Numérique

### Données Brutes
```
Symbol | ROE | PEG | EBITDA | Dividend | PriceSale | Growth
AAPL   | 25  | 1.2 | 0      | N/A      | 2.1       | 8.5
MSFT   | 18  | 1.8 | 150    | 2.5      | 3.2       | 12.0
GOOGL  | 22  | 1.5 | 200    | N/A      | 5.0       | 15.0
```

### Après `_get_clean_columns_and_data()`
```
clean_columns = ['ROE', 'PEG', 'EBITDA', 'Dividend', 'PriceSale', 'Growth']
# EBITDA incluse car MSFT et GOOGL ont des valeurs non-zéro
# Dividend incluse car MSFT a une valeur
```

### Avant Correction (PDF pour AAPL)
```
Tableau AAPL:
ROE        25
PEG        1.2
PriceSale  2.1
Growth     8.5
# Manquent: EBITDA (0), Dividend (N/A)
# Affichage: 4 colonnes au lieu de 6
```

### Après Correction (PDF pour AAPL)
```
Tableau AAPL:
ROE        25      | PEG        1.2
EBITDA     N/A     | Dividend   N/A
PriceSale  2.1     | Growth     8.5
# Toutes les colonnes affichées!
# Affichage: 6 colonnes (complet)
```

---

## 🛠️ Comment C'est Implémenté

### Le Code Clé (Nouveau)

```python
# Afficher TOUTES les colonnes
cols_with_data = []
for col in clean_columns:  # ← Itère sur TOUTES les colonnes
    if col not in columns_to_skip:  # Sauf si dans skip list
        value = stock_data.get(col, 'N/A')  # Prend la valeur, ou 'N/A'
        # Formate la valeur (arrondit si float)
        formatted_value = format_value(value)
        cols_with_data.append((col, formatted_value))

# cols_with_data contient maintenant TOUTES les colonnes de clean_columns
# (excepté celles dans columns_to_skip)
```

### Comparaison
```python
# AVANT: Filtrait par "value != ''  and value != 0"
for col in clean_columns:
    value = stock_data.get(col, '')
    if value and value != '' and value != 0 and value != '0':  # ← Trop restrictif
        cols_with_data.append((col, formatted_value))

# APRÈS: Accepte tout, y compris vides (affichés comme N/A)
for col in clean_columns:
    value = stock_data.get(col, 'N/A')
    # Pas de condition! Ajoute TOUJOURS
    cols_with_data.append((col, formatted_value))
```

---

## ✅ Vérification

### Voir le nombre de colonnes dans le log

Quand vous générez un PDF, vérifiez le logging:

```
📊 GÉNÉRATION PDF - INFO DE DÉBUG
   Colonnes reçues: 28
   Colonnes à afficher: ['ROE', 'PEG', 'EBITDA', ..., 'Growth']
   Résultats: 50 symboles

   📊 AAPL: 25 colonnes total dans clean_columns ✅
   📊 MSFT: 25 colonnes total dans clean_columns ✅
   📊 GOOGL: 25 colonnes total dans clean_columns ✅
```

**Important:** Tous les symboles affichent le **même nombre de colonnes** maintenant!

---

## 🎓 Résumé

| Aspect | Avant | Après |
|--------|-------|-------|
| **Colonnes dans clean_columns** | 28 | 28 |
| **Colonnes affichées (AAPL)** | ~15 | 28 |
| **Colonnes affichées (MSFT)** | ~22 | 28 |
| **Cohérence par symbole** | Variable | Identique ✅ |
| **Valeurs manquantes** | Cachées | Affichées comme N/A ✅ |
| **Lignes tableau** | Variable | Fixe (28/4 colonnes) |

---

## 🚀 Prochaines Améliorations

- [ ] Option pour masquer les colonnes N/A si souhaité
- [ ] Tri des colonnes par importance
- [ ] Colonnes personnalisables par utilisateur
- [ ] Export des colonnes filtrées dans config

---

**Créé:** 25 février 2026  
**Mise à jour:** pdf_generator.py v2.0  
**Status:** ✅ Clarifié et Documenté
