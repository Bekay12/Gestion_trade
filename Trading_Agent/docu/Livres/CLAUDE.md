# docu/Livres/

PDF sources appartenant à l'utilisateur. **Aucun n'est un livrable** : ce sont
des entrées, jamais des sorties.

## Composition réelle du dossier

Le nom « Livres » est trompeur, et l'inventaire l'a montré :

| Fichier | Nature | Utilisé |
|---|---|---|
| `Academy_Germain_COMPLET.pdf` | Cours structuré, 459 p., 99 ch., français | **Oui, seule source retenue** |
| `l'investisseur intelligent.pdf` | Graham, 629 p. | Non — paradigme opposé au day trading intraday |
| 14 autres | Presse financière allemande et anglaise | Non — actualité boursière, aucun apport méthodologique |

Les magazines (Börse Online, Capital, Euro, Cash, GELDMASCHINE, teleskop,
FortuneEU) représentent ~1 800 pages sans rapport avec la méthode documentée.
Exclusion décidée avec l'utilisateur, pas par oubli.

## Règle de traitement

Ces documents se **synthétisent**, ils ne se recopient pas. Les notes de
[../synthese/](../synthese/) et [../methode/](../methode/) sont rédigées avec
leurs propres mots et leur propre structure. Ce qui en est repris directement :
les valeurs numériques, les seuils, les formules — des faits, non de
l'expression.

Ne jamais produire un fichier qui restitue de longs passages d'un de ces PDF.

## Extraction

`../_extraction/` contient un fichier par chapitre du cours, produit par
[../../scripts/youtube/split_course.py](../../scripts/youtube/split_course.py).
C'est un **cache de travail** régénérable, pas une publication :

```powershell
.venv\Scripts\python.exe scripts\youtube\split_course.py 2 3 5 7 8 9 11 12
```

Les arguments sont les numéros de Partie. Le noyau day-trading retenu couvre
les Parties 2, 3, 5, 7, 8, 9, 11 et 12 — 35 chapitres. Hors périmètre :
macroéconomie (P1), analyse graphique (P10), crypto (P13, 20 chapitres).

## Détail technique

Le cours ne porte pas de signet PDF : sa table des matières est du **texte
brut sur les trois premières pages**, d'où le passage par un analyseur dédié
plutôt que par `get_toc()`.

L'extraction est propre — zéro glyphe illisible sur les pages testées. Si des
accents apparaissent cassés, c'est la sortie console qui les mange, pas le PDF.

## Ne pas versionner

Ces fichiers sont volumineux et sous droits. Ils restent locaux.
