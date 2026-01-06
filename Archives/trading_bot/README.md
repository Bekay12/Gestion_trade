# 🤖 Trading Bot - Analyse Technique

Bot d'analyse technique pour le trading d'actions avec double interface (Desktop PyQt5 + Web Streamlit).

## 📋 Table des Matières

- [Fonctionnalités](#fonctionnalités)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Déploiement Cloud](#déploiement-cloud)
- [Structure du Projet](#structure-du-projet)

## ✨ Fonctionnalités

- **Analyse Graphique**: Visualisation des indicateurs techniques (MACD, RSI, EMA, Bollinger Bands, ADX)
- **Détection de Signaux**: Identification automatique des signaux d'achat/vente
- **Optimisation de Paramètres**: Optimisation des seuils et coefficients
- **Double Interface**:
  - 🖥️ **PyQt5**: Interface desktop pour utilisation locale
  - 🌐 **Streamlit**: Interface web accessible depuis mobile/tablette/ordinateur

## 🚀 Installation

### Prérequis

- Python 3.8+
- pip

### Installation des dépendances

```bash
cd trading_bot
pip install -r requirements.txt
```

## 💻 Utilisation

### Interface Desktop (PyQt5)

Lancez l'interface graphique desktop:

```bash
python run_ui.py
```

### Interface Web (Streamlit)

Lancez l'interface web en local:

```bash
streamlit run streamlit_app.py
```

L'application sera accessible sur `http://localhost:8501`

### Ligne de Commande

#### Analyse de symboles

```bash
python main.py analysis --symbols test_symbols.txt --period 12mo
```

#### Optimisation de paramètres

```bash
python main.py optimization --symbols optimisation_symbols.txt
```

#### Affichage de graphiques

```bash
python main.py charts --symbols test_symbols.txt --period 1y
```

## ☁️ Déploiement Cloud

### Streamlit Cloud (Recommandé)

1. **Préparez votre repository GitHub**
   - Push votre code sur GitHub
   - Assurez-vous que `streamlit_app.py` et `requirements.txt` sont présents

2. **Déployez sur Streamlit Cloud**
   - Allez sur [streamlit.io](https://streamlit.io)
   - Connectez-vous avec votre compte GitHub
   - Cliquez sur "New app"
   - Sélectionnez votre repository
   - Branch: `main` (ou votre branche)
   - Main file path: `trading_bot/streamlit_app.py`
   - Cliquez sur "Deploy"

3. **Accédez à votre app**
   - Une URL publique sera générée (ex: `https://votre-app.streamlit.app`)
   - Accessible depuis n'importe quel appareil (mobile, tablette, PC)

### Autres Options de Déploiement

#### Heroku

1. Créez un fichier `Procfile`:
```
web: streamlit run trading_bot/streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
```

2. Créez un fichier `runtime.txt`:
```
python-3.9.16
```

3. Déployez:
```bash
heroku create votre-app-name
git push heroku main
```

#### Docker

1. Créez un `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY trading_bot/requirements.txt .
RUN pip install -r requirements.txt
COPY trading_bot/ .
EXPOSE 8501
CMD ["streamlit", "run", "streamlit_app.py"]
```

2. Build et run:
```bash
docker build -t trading-bot .
docker run -p 8501:8501 trading-bot
```

## 📁 Structure du Projet

```
trading_bot/
├── config/              # Configuration
│   └── settings.py      # Paramètres centralisés
├── data/                # Données
│   └── symbols/         # Fichiers de symboles
├── src/                 # Code source
│   ├── backtesting/     # Moteur de backtesting
│   ├── data/            # Récupération de données
│   ├── indicators/      # Indicateurs techniques
│   ├── optimization/    # Optimisation de paramètres
│   ├── signals/         # Génération de signaux
│   ├── utils/           # Utilitaires
│   └── visualization/   # Graphiques
├── logs/                # Fichiers de logs
├── results/             # Résultats d'analyses
├── tests/               # Tests
├── main.py              # Point d'entrée CLI
├── run_ui.py            # Interface PyQt5
├── streamlit_app.py     # Interface Streamlit (Web)
└── requirements.txt     # Dépendances

```

## 🎯 Fonctionnalités Détaillées

### Interface Streamlit (Web)

L'interface Streamlit offre:
- **📊 Analyse Graphique**: Analysez des symboles spécifiques avec visualisation
- **✅ Signaux Populaires**: Détection automatique de signaux sur une liste de symboles
- **📁 Gestion des Symboles**: Visualisez et gérez vos listes de symboles
- **📖 Documentation**: Guide d'utilisation intégré

### Interface PyQt5 (Desktop)

L'interface PyQt5 reste disponible pour:
- Utilisation locale sans connexion internet
- Intégration avec d'autres outils desktop
- Performance optimale sur PC

## ⚙️ Configuration

Les paramètres sont définis dans `config/settings.py`:
- **Indicateurs techniques**: MACD, RSI, EMA, Bollinger, ADX, Ichimoku
- **Seuils de trading**: Seuils d'achat/vente, filtres de volume
- **Optimisation**: Paramètres d'optimisation des stratégies

## 📊 Fichiers de Symboles

Les symboles sont organisés dans `data/symbols/`:
- `popular_symbols.txt`: Symboles populaires
- `mes_symbols.txt`: Vos symboles personnels
- `test_symbols.txt`: Symboles de test
- `optimisation_symbols.txt`: Symboles pour l'optimisation

Format: Un symbole par ligne (ex: AAPL, MSFT, GOOGL)

## 🔧 Développement

### Tests

```bash
pytest tests/
```

### Linting

```bash
flake8 src/
black src/
```

## 📝 Licence

Ce projet est privé et destiné à un usage personnel.

## 🤝 Support

Pour toute question ou problème:
1. Consultez les logs dans `logs/`
2. Vérifiez la configuration dans `config/settings.py`
3. Consultez la documentation intégrée dans l'interface Streamlit

---

**Note**: Les deux interfaces (PyQt5 et Streamlit) fonctionnent de manière indépendante et peuvent être utilisées simultanément selon vos besoins.
