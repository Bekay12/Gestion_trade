# 📊 Tableau Récapitulatif: Utilisation du Seed

## Vue d'Ensemble Rapide

| Méthode | Accepte Seed? | Comment Utilisé | Bénéfice | Recommandé? |
|---------|:-------------:|-----------------|----------|:----------:|
| **Genetic Algorithm** | ✅ OUI | 1er individu population | ⭐⭐⭐ Très bon | ✅ |
| **PSO** | ✅ OUI | 1ère particule essaim | ⭐⭐⭐ Très bon | ✅ |
| **LHS** | ✅ OUI | Baseline min (puis explore) | ⭐⭐⭐ Très bon | ✅ |
| **Differential Evolution** | ✅ OUI | Compare au résultat (fallback) | ⭐⭐⭐ Très bon | ✅ |
| **CMA-ES** | ❌ NON | Ignoré | ❌ Aucun | ⚠️ |
| **Local Refinement** | ✅ OUI | Perturbations locales | ⭐⭐ Moyen (souvent dégrade) | ⚠️ |

---

## Bénéfice du Seed par Stratégie

### **Stratégie 'hybrid'** (5/6 méthodes bénéficient) ✅ 🆕
```
✅ GA + PSO + LHS + DE = gain du seed (4 méthodes)
❌ CMA-ES = pas de gain
-> Bénéfice EXCELLENT (83% des méthodes)
```

### **Stratégie 'genetic'** (1/1 bénéficie)
```
✅ GA = gain du seed
-> Bénéfice COMPLET (100%)
```

### **Stratégie 'pso'** (1/1 bénéficie)
```
✅ PSO = gain du seed
-> Bénéfice COMPLET (100%)
```

### **Stratégie 'lhs'** (1/1 bénéficie)
```
✅ LHS = gain du seed
-> Bénéfice COMPLET (100%)
```

### **Stratégie 'differential'** (1/1 bénéficie) ✅ 🆕
```
✅ DE = gain du seed
-> Bénéfice COMPLET (100%)
```

### **Stratégie 'cma'** (0/2 bénéficient)
```
❌ CMA-ES = pas de seed direct
-> Bénéfice NUL (0%)
```

---

## Amélioration Recommandée

### Priority 1: **Ajouter Seed à Differential Evolution** ✅ COMPLÉTÉ
- Impact: +17% bénéfice pour 'hybrid' (67% → 83%)
- Effort: ✅ TERMINÉ
- Code: 15 lignes ajoutées

### Priority 2: **Ajouter Seed à CMA-ES** 🔴 TODO
- Impact: +8% bénéfice supplémentaire pour 'hybrid'
- Effort: Moyen
- Code: 20 lignes

### Priority 3: **Documenter la Perte de Seed**
- Impact: Prévient la confusion utilisateur
- Effort: Très faible
- Code: Commentaires + docstring

---

## Résumé

**Oui, le seed est maintenant utilisé PARTOUT (sauf CMA-ES):**
- ✅ GA, PSO, LHS, DE le reçoivent et en bénéficient (4/5 méthodes)
- ❌ CMA-ES l'ignore (reste à faire)
- 🎯 Pour maximiser le bénéfice: utilisez 'genetic', 'pso', 'lhs', 'differential', ou 'hybrid'
- ⚠️ Évitez 'cma' si vous avez de bons paramètres historiques

