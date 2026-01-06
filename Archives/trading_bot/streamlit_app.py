#!/usr/bin/env python3
"""
Streamlit interface for the trading bot.
Cloud-accessible web interface that provides the same functionality as the PyQt5 UI.
"""
import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
from io import BytesIO

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import config
from src.utils.logger import setup_logging, get_logger
from src.utils.file_manager import SymbolFileManager
from src.signals.signal_analyzer import signal_analyzer
from src.visualization.analysis_charts import analysis_charts

# Configure logging
setup_logging(config.logging)
logger = get_logger(__name__)

# Page configuration
st.set_page_config(
    page_title="Trading Bot - Analyse Technique",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'symbol_manager' not in st.session_state:
    st.session_state.symbol_manager = SymbolFileManager()
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Helper functions
def load_default_symbols():
    """Load default symbols from files."""
    popular = st.session_state.symbol_manager.load_symbols_from_txt("popular_symbols.txt")
    personal = st.session_state.symbol_manager.load_symbols_from_txt("mes_symbols.txt")
    return popular, personal

def parse_symbols(text: str):
    """Parse comma-separated symbols."""
    return [s.strip().upper() for s in text.split(',') if s.strip()]

def validate_period(period: str):
    """Validate period against valid periods."""
    valid_periods = config.data.valid_periods
    if period not in valid_periods:
        return False, f"Période invalide. Choisissez parmi: {', '.join(valid_periods)}"
    return True, period

# Main UI
def main():
    st.title("🤖 Trading Bot – Interface Web")
    st.markdown("---")
    
    # Load default symbols
    popular_symbols, personal_symbols = load_default_symbols()
    
    # Sidebar for settings
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Period selection
        period = st.selectbox(
            "Période d'analyse",
            options=config.data.valid_periods,
            index=config.data.valid_periods.index(config.trading.default_period)
            if config.trading.default_period in config.data.valid_periods else 0
        )
        
        st.markdown("---")
        st.info("💡 Cette interface web peut être accédée depuis n'importe quel appareil (téléphone, tablette, ordinateur)")
        
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Analyse Graphique",
        "✅ Signaux Populaires",
        "📁 Gestion des Symboles",
        "📖 Documentation"
    ])
    
    # Tab 1: Chart Analysis
    with tab1:
        st.header("📊 Analyse Graphique")
        st.write("Analysez les graphiques techniques pour des symboles spécifiques.")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            symbols_input = st.text_input(
                "Symboles à analyser (séparés par des virgules)",
                value="AAPL, MSFT, GOOGL",
                help="Entrez les symboles boursiers séparés par des virgules"
            )
        
        with col2:
            st.write("")  # Spacing
            st.write("")  # Spacing
            analyze_btn = st.button("🔍 Analyser", key="analyze_charts", type="primary")
        
        if analyze_btn:
            symbols = parse_symbols(symbols_input)
            if not symbols:
                st.error("⚠️ Veuillez entrer au moins un symbole à analyser.")
            else:
                with st.spinner(f"Analyse en cours de {len(symbols)} symbole(s)..."):
                    try:
                        analysis_charts.analyse_et_affiche(symbols, period)
                        
                        st.success(f"✅ Analyse graphique terminée pour {len(symbols)} symbole(s).")
                        st.info("📊 Les graphiques ont été générés. Dans la version actuelle, les graphiques s'affichent via matplotlib. Pour une intégration complète dans le navigateur, une mise à jour future permettra d'afficher les graphiques directement dans Streamlit.")
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        logger.error(f"Error in chart analysis: {e}")
    
    # Tab 2: Popular Signals
    with tab2:
        st.header("✅ Analyse des Signaux Populaires")
        st.write("Analysez les signaux d'achat/vente sur les symboles populaires.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            popular_input = st.text_area(
                "Symboles Populaires",
                value=", ".join(popular_symbols[:20]) if len(popular_symbols) > 20 else ", ".join(popular_symbols),
                height=100,
                help="Liste des symboles populaires à analyser"
            )
        
        with col2:
            personal_input = st.text_area(
                "Mes Symboles",
                value=", ".join(personal_symbols),
                height=100,
                help="Vos symboles personnels"
            )
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            display_charts = st.checkbox("Afficher les graphiques", value=True)
        with col2:
            save_csv = st.checkbox("Sauvegarder en CSV", value=True)
        
        analyze_signals_btn = st.button("🚀 Analyser les Signaux", type="primary", key="analyze_signals")
        
        if analyze_signals_btn:
            pop_symbols = parse_symbols(popular_input)
            per_symbols = parse_symbols(personal_input)
            
            if not pop_symbols:
                st.error("⚠️ Veuillez entrer des symboles populaires.")
            else:
                with st.spinner(f"Analyse des signaux en cours pour {len(pop_symbols)} symbole(s)..."):
                    try:
                        result = signal_analyzer.analyze_popular_signals(
                            pop_symbols,
                            per_symbols,
                            period=period,
                            display_charts=display_charts,
                            verbose=True,
                            save_csv=save_csv
                        )
                        
                        n_signals = len(result.get("signals", []))
                        st.success(f"✅ Analyse terminée – {n_signals} signal(s) détecté(s).")
                        
                        # Display results
                        if result.get("signals"):
                            st.subheader("📋 Signaux Détectés")
                            signals_df = pd.DataFrame(result["signals"])
                            st.dataframe(signals_df, use_container_width=True)
                            
                            # Download button for results
                            csv = signals_df.to_csv(index=False).encode('utf-8')
                            st.download_button(
                                label="⬇️ Télécharger les résultats (CSV)",
                                data=csv,
                                file_name=f"signals_{period}.csv",
                                mime="text/csv"
                            )
                        else:
                            st.warning("⚠️ Aucun signal détecté pour la période sélectionnée.")
                        
                        st.session_state.analysis_results = result
                        
                    except Exception as e:
                        st.error(f"❌ Erreur lors de l'analyse: {str(e)}")
                        logger.error(f"Error in signal analysis: {e}")
    
    # Tab 3: Symbol Management
    with tab3:
        st.header("📁 Gestion des Symboles")
        st.write("Gérez vos listes de symboles personnalisés.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📝 Symboles Populaires")
            st.text_area(
                "Liste actuelle",
                value="\n".join(popular_symbols),
                height=300,
                disabled=True,
                key="popular_display"
            )
            st.caption(f"Total: {len(popular_symbols)} symboles")
        
        with col2:
            st.subheader("⭐ Mes Symboles")
            st.text_area(
                "Liste actuelle",
                value="\n".join(personal_symbols),
                height=300,
                disabled=True,
                key="personal_display"
            )
            st.caption(f"Total: {len(personal_symbols)} symboles")
        
        st.markdown("---")
        st.subheader("ℹ️ À propos de la gestion des symboles")
        st.info("""
        Pour modifier vos listes de symboles:
        1. Éditez les fichiers dans `data/symbols/`
        2. `popular_symbols.txt` - Symboles populaires
        3. `mes_symbols.txt` - Vos symboles personnels
        
        Une fonctionnalité d'ajout interactif sera disponible dans une future mise à jour.
        """)
    
    # Tab 4: Documentation
    with tab4:
        st.header("📖 Documentation")
        
        st.markdown("""
        ### 🤖 Guide d'utilisation du Trading Bot
        
        #### 📊 Analyse Graphique
        Permet d'analyser les graphiques techniques pour des symboles spécifiques:
        - Entrez les symboles séparés par des virgules (ex: AAPL, MSFT, GOOGL)
        - Sélectionnez la période d'analyse dans la barre latérale
        - Cliquez sur "Analyser" pour générer les graphiques
        
        #### ✅ Signaux Populaires
        Analyse les signaux d'achat/vente sur une liste de symboles:
        - **Symboles Populaires**: Liste principale des actions à analyser
        - **Mes Symboles**: Votre liste personnelle de favoris
        - Options: Afficher les graphiques et sauvegarder les résultats en CSV
        - Les résultats affichent les signaux détectés et peuvent être téléchargés
        
        #### 📁 Gestion des Symboles
        Visualisez et gérez vos listes de symboles:
        - Voir les symboles populaires pré-configurés
        - Voir vos symboles personnels
        - Ajouter de nouveaux symboles à votre liste
        
        #### ⚙️ Configuration
        - **Période d'analyse**: Sélectionnez la période dans la barre latérale
        - Périodes disponibles: 1d, 5d, 1mo, 3mo, 6mo, 12mo, 1y, 2y, 5y, 10y, ytd, max
        
        #### 🌐 Accès depuis votre téléphone
        Cette interface Streamlit peut être déployée sur le cloud (Streamlit Cloud, Heroku, etc.)
        pour un accès depuis n'importe quel appareil:
        1. L'application fonctionne de manière indépendante de votre PC
        2. Accessible via une URL web depuis mobile, tablette, ou ordinateur
        3. L'interface PyQt5 reste disponible pour une utilisation locale
        
        #### 📝 Notes importantes
        - Les données sont récupérées en temps réel depuis Yahoo Finance
        - L'analyse utilise les mêmes algorithmes que l'interface PyQt5
        - Les résultats peuvent être téléchargés en CSV
        - Les logs sont disponibles dans le répertoire `logs/`
        
        #### 🚀 Déploiement Cloud
        Pour déployer sur Streamlit Cloud:
        1. Push le code sur GitHub
        2. Connectez-vous sur streamlit.io
        3. Déployez depuis votre repository
        4. L'app sera accessible via une URL publique
        """)
        
        st.markdown("---")
        st.info("💡 Pour toute question ou problème, consultez les logs dans le répertoire `logs/`")

if __name__ == "__main__":
    main()
