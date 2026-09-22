# Magasines

Magazines financiers en PDF, leur conversion en Markdown et le suivi des recommandations d'achat.

| Élément | Rôle |
|---|---|
| `PDFs/` | Numéros téléchargés (source). |
| `MARKDOWNS/` | Conversion MarkItDown, nommée `<Magazine>_<Titre>_<Date>.md`. |
| `pdf_to_markdown.py` + `titres.json` | Conversion des PDF ; titres de couverture éditables dans `titres.json`. |
| `Recommandations_achat_<début>_<fin>.md/.csv` | Recommandations d'achat extraites et vérifiées page par page (ticker, cours, objectif, stop, horizon, raison). |
| `tickers_recommandations_<début>_<fin>.txt` | Tickers à coller dans le champ symbole de la fenêtre principale. |
| `evaluer_recommandations.py` | Évaluation des recommandations N jours après parution. |
| `pipeline_recos/` | Chaîne d'extraction (pages texte, sorties du modèle local, corrections manuelles, tickers, tests). |

## Évaluer les recommandations

```bash
.venv_new/bin/python Magasines/evaluer_recommandations.py                   # à aujourd'hui
.venv_new/bin/python Magasines/evaluer_recommandations.py --asof 2027-03-20 # date donnée
.venv_new/bin/python Magasines/evaluer_recommandations.py --sans-reseau     # store + cache seuls
```

Sorties : `Evaluation_recommandations_<asof>.md/.csv`. Pour chaque recommandation : entrée à la
clôture du premier jour coté après la date d'entrée, échéance à 182 jours, objectif et stop en % du
cours imprimé (le premier touché l'emporte, le stop en cas d'égalité), écart à l'indice de la place
de cotation. Taux de réussite par magazine, type et horizon.

Cours : store Parquet de l'application d'abord, puis cache `pipeline_recos/cours/`, puis
yfinance par lots de 50 pour le reste ; un ticker encore incomplet est marqué « cours
incomplets » et exclu des taux (relancer plus tard si yfinance limite les requêtes).

Échéances des numéros du 19.06 au 19.09.2026 : de 2026-12-02 (Euro 07/2026) à 2027-03-19
(Börse Online 39/2026). Une évaluation avant ces dates est provisoire.

Tests hors ligne : `.venv_new/bin/python -m pytest Magasines/pipeline_recos/test_evaluer.py`
