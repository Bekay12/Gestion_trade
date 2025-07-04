# qsi.py - Analyse technique unifiée pour les actions avec MACD et RSI
# Ce script télécharge les données boursières, calcule les indicateurs techniques et affiche
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import ta
import time
import csv
from matplotlib import dates as mdates

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_trading_signal(prices, volumes, variation_seuil=-20, volume_seuil=100000):
    """Détermine les signaux de trading avec validation des données"""
    # Correction : assure que prices et volumes sont bien des Series 1D
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    if len(prices) < 50:
        return "Données insuffisantes", None, None, None
    
    # Calcul des indicateurs
    macd, signal_line = calculate_macd(prices)
    #rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    
    # Validation des derniers points
    if len(macd) < 2 or len(rsi) < 1:
        return "Données récentes manquantes", None, None, None
    
    # Détection des signaux
    last_close = prices.iloc[-1]
    ema20 = prices.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]
    
    # Calcul de la variation sur 30 jours
    if len(prices) >= 30:
        variation_30j = (last_close - prices.iloc[-30]) / prices.iloc[-30] * 100
    else:
        variation_30j = None
    
    # Calcul du volume moyen sur 30 jours
    if len(volumes) >= 30:
        volume_mean = volumes.rolling(window=30).mean().iloc[-1]
        if isinstance(volume_mean, pd.Series):
            volume_mean = float(volume_mean.iloc[0])
        else:
            volume_mean = float(volume_mean)
    else:
        volume_mean = None
    
    if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1] and rsi.iloc[-1] < 70 and volume_mean > volume_seuil:
        signal = "ACHAT"
    elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1] and rsi.iloc[-1] > 30:
        signal = "VENTE"
    else:
        signal = "NEUTRE"
    
    return signal, last_close, last_close > ema20, rsi.iloc[-1], volume_mean

def download_stock_data(symbols, period):
    """Télécharge les données avec gestion des erreurs"""
    valid_data = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period, interval="1d", progress=False, timeout=10)
            
            if data.empty:
                print(f"⚠️ Aucune donnée pour {symbol}")
                continue
                
            # Vérification de la colonne Close
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                print(f"⚠️ Colonne 'Close' ou 'Volume' manquante pour {symbol}")
                continue

            valid_data[symbol] = {
                'Close': data['Close'],
                'Volume': data['Volume']}
            
            
        except Exception as e:
            print(f"🚨 Erreur sur {symbol}: {str(e)}")
    
    return valid_data

def backtest_signals(prices, volumes, montant=50):
    """
    Effectue un backtest sur la série de prix.
    Un 'trade' correspond ici à un cycle complet ACHAT puis VENTE (entrée puis sortie).
    Le gain est calculé pour chaque cycle achat-vente.
    """

    # Correction : refuse les scalaires et séries trop courtes
    if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
        return {
            "trades": 0,
            "gagnants": 0,
            "taux_reussite": 0,
            "gain_total": 0.0
        }
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()

    positions = []
    for i in range(1, len(prices)):
        signal, *_ = get_trading_signal(prices[:i], volumes[:i])
        if signal == "ACHAT":
            positions.append({"entry": prices.iloc[i], "entry_idx": i, "type": "buy"})
        elif signal == "VENTE" and positions and "exit" not in positions[-1]:
            positions[-1]["exit"] = prices.iloc[i]
            positions[-1]["exit_idx"] = i

    nb_trades = 0
    nb_gagnants = 0
    gain_total = 0.0

    for pos in positions:
        if "exit" in pos:
            nb_trades += 1
            entry = pos["entry"]
            exit = pos["exit"]
            rendement = (exit - entry) / entry
            gain = montant * rendement
            gain_total += gain
            if gain > 0:
                nb_gagnants += 1

    return {
        "trades": nb_trades,  # Un trade = un cycle achat-vente
        "gagnants": nb_gagnants,
        "taux_reussite": (nb_gagnants / nb_trades * 100) if nb_trades else 0,
        "gain_total": gain_total
    }

def plot_unified_chart(symbol, prices,volumes, ax):
    """Trace un graphique unifié avec prix, MACD et RSI intégré"""
    # Vérification du format des prix
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if not isinstance(prices, pd.Series):
        prices = pd.Series(prices, name=symbol)
    
    # Calcul des indicateurs
    ema20 = prices.ewm(span=20, adjust=False).mean()
    ema50 = prices.ewm(span=50, adjust=False).mean()
    sma50 = prices.rolling(window=50).mean() if len(prices) >= 50 else pd.Series()
    macd, signal_line = calculate_macd(prices)
    
    # Calcul du RSI avec vérification des données
    try:
        rsi = ta.momentum.RSIIndicator(close=prices, window=14).rsi()
    except Exception as e:
        print(f"⚠️ Erreur RSI pour {symbol}: {str(e)}")
        rsi = pd.Series(np.zeros(len(prices)), index=prices.index)
    
    # Tracé des prix sur l'axe principal
    color = 'tab:blue'
    ax.plot(prices.index, prices, label='Prix', color=color, linewidth=1.8)
    ax.plot(ema20.index, ema20, label='EMA20', linestyle='--', color='orange', linewidth=1.4)
    ax.plot(ema50.index, ema50, label='EMA50', linestyle='-.', color='purple', linewidth=1.4)
    
    if not sma50.empty:
        ax.plot(sma50.index, sma50, label='SMA50', linestyle=':', color='green', linewidth=1.4)
    
    ax.set_ylabel('Prix', color=color, fontsize=10)
    ax.tick_params(axis='y', labelcolor=color)
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # Création d'un axe secondaire pour MACD
    ax2 = ax.twinx()
    
    # Tracé MACD
    color = 'tab:purple'
    ax2.plot(macd.index, macd, label='MACD', color=color, linewidth=1.2)
    ax2.plot(signal_line.index, signal_line, label='Signal', color='tab:orange', linewidth=1.2)
    ax2.fill_between(
        macd.index, 0, macd - signal_line, 
        where=(macd - signal_line) >= 0, 
        facecolor='green', alpha=0.3, interpolate=True
    )
    ax2.fill_between(
        macd.index, 0, macd - signal_line, 
        where=(macd - signal_line) < 0, 
        facecolor='red', alpha=0.3, interpolate=True
    )
    ax2.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax2.set_ylabel('MACD', color=color, fontsize=10)
    ax2.tick_params(axis='y', labelcolor=color)
    
    # Tracé RSI en arrière-plan avec axvspan
    for i in range(1, len(prices)):
        start = prices.index[i-1]
        end = prices.index[i]
        rsi_val = rsi.iloc[i-1]
        
        if rsi_val > 70:
            color = 'lightcoral'
        elif rsi_val < 30:
            color = 'lightgreen'
        else:
            color = 'lightgray'
            
        ax.axvspan(start, end, facecolor=color, alpha=0.1, zorder=-1)
    
    # Ajout des légendes
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=9)
    
    # Ajout des signaux trading
    signal, last_price, trend, last_rsi, volume_moyen = get_trading_signal(prices,volumes)
    
    # Calcul de la progression en pourcentage
    if len(prices) > 1:
        progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
    else:
        progression = 0.0

    if last_price is not None:
        trend_symbol = "Haussière" if trend else "Baissière"
        rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
        signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'

        title = (
            f"{symbol} | Prix: {last_price:.2f} | Signal: {signal} | "
            f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | "
            f"Progression: {progression:+.2f}% | Vol. moyen: {volume_moyen:,.0f} (unité) "
        )
        ax.set_title(title, fontsize=12, fontweight='bold', color=signal_color)
    
    return ax2


def analyse_et_affiche(symbols, period="12mo"):
    """
    Télécharge les données pour les symboles donnés et affiche les graphiques d'analyse technique.
    """
    print("⏳ Téléchargement des données...")
    data = download_stock_data(symbols, period)

    if not data:
        print("❌ Aucune donnée valide disponible. Vérifiez les symboles ou la connexion internet.")
        return

    num_plots = len(data)
    fig, axes = plt.subplots(num_plots, 1, figsize=(14, 5 * num_plots), sharex=False)

    if num_plots == 1:
        axes = [axes]
    elif num_plots == 0:
        print("❌ Aucun symbole valide à afficher")
        return

    for i, (symbol, stock_data) in enumerate(data.items()):
        prices = stock_data['Close']
        volumes = stock_data['Volume']
        print(f"📊 Traitement de {symbol}...")
        plot_unified_chart(symbol, prices, volumes, axes[i])

    plt.tight_layout()
    plt.subplots_adjust(top=0.95, hspace=0.4)
    plt.show()

# ======================================================================
# CONFIGURATION PRINCIPALE
# ======================================================================

period = "12mo"  #variable globale pour la période d'analyse ne pas commenter 

# Exemple d'utilisation (décommente pour exécuter) :
# symbols = ["SYM", "LHA.DE", "INGA.AS", "FLEX", "ALDX"]

# analyse_et_affiche(symbols, period)

# ======================================================================
# SIGNEAUX POUR ACTIONS POPULAIRES (version simplifiée)
# ======================================================================
popular_symbols = list(dict.fromkeys([
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "TSLA", "META", "BRK-B", "JPM", "V", "BABA", "DIS", "NFLX", "PYPL", "INTC", "AMD", "CSCO", "WMT", "VZ", "KO", "SGMT",
    "PEP", "MRK", "T", "NKE", "XOM", "CVX", "ABT", "CRM", "IBM", "ORCL", "MCD", "TMDX", "BA", "CAT", "GS", "RTX", "MMM", "HON", "LMT", "SBUX", "ADBE", "SMSN.L",
    "EEM", "INTU", "NOW", "ZM", "SHOP", "SNAP", "PFE", "TGT", "CVS", "WFC", "RHM.DE", "SAP.DE", "BAS.DE", "ALV.DE", "BMW.DE", "VOW3.DE", "SMTC", "ZS", "ZTS",
    "DTE.DE", "DBK.DE", "LHA.DE", "FME.DE", "BAYN.DE", "LIN.DE", "ENR.DE", "VNA.DE", "1COV.DE", "FRE.DE", "HEN3.DE", "HEI.DE", "RWE.DE", "VOW.DE", "GLW", "TMO",
    "DHR", "ABB", "BAX", "MDT", "GE", "NOC", "GD", "HII", "TXT", "LHX", "TDY", "CARR", "OTIS", "JCI", "INOD", "BIDU", "JD", "PDD", "TCEHY", "NTES", "BILI",
    "XPEV", "LI", "NIO", "BYDDF", "GME", "AMC", "BB", "NOK", "RBLX", "PLTR", "FSLY", "CRWD", "OKTA", "Z", "DOCU", "PINS", "SPOT", "LYFT", "UBER", "SNOW", "TTWO",
    "VRSN", "WDAY", "2318.HK", "2382.HK", "2388.HK", "2628.HK", "3328.HK", "3988.HK", "9988.HK", "2319.HK", "0700.HK", "3690.HK", "ADSK", "02020.HK",
    "9618.HK", "1810.HK", "1211.HK", "1299.HK", "2313.HK", "2386.HK", "2623.HK", "2385.HK", "0005.HK", "0011.HK", "0027.HK", "0038.HK", "0066.HK", "0083.HK",
    "0101.HK", "0117.HK", "0120.HK", "LSEG.L", "VOD.L", "BP.L", "HSBA.L",  "GSK.L", "ULVR.L", "AZN.L", "RIO.L", "BATS.L", "ADYEN.AS", "TM", "MU", "GILT",
    "ASML.AS", "PHIA.AS", "INGA.AS", "MC.PA", "OR.PA", "AIR.PA", "BNP.PA", "SAN.PA", "ENGI.PA", "CAP.PA", "LVMH.PA", "WELL", "O", "VICI", "ETOR", "ABR",
    "PLD", "PSA", "AMT", "CCI", "DLR", "EXR", "EQR", "ESS", "AVB", "MAA", "UDR", "SBRA", "UNH", "HD", "MA", "PG", "LLY", "COST", "AVGO", "ABBV", "QCOM", 
    "DDOG", "CRL", "EXAS", "ILMN", "INCY", "MELI", "MRNA", "NTLA", "REGN", "ROKU", "QSI", "SYM", "IONQ", "QBTS", "RGTI", "SMCI", "TSM", "ALDX", "CSX", "LRCX", 
    "BIIB", "CDNS", "CTSH", "EA", "FTNT", "GILD", "IDXX", "MP", "MTCH", "MRVL", "PAYX", "PTON", "AAL", "UAL", "DAL", "LUV", "JBLU", "ALK", "FLEX", "CACI",  
    "CRIS", "CYTK", "EXEL", "FATE", "INSM", "KPTI", "NBIX", "NTRA", "PGEN", "RGEN", "SAGE", "SNY", "TGTX", "VYGR", "ARCT", "AXSM", "BMRN", "KTOS","BTC", "ETH",
    "LTC", "SOL", "DOGE", "LINK", "ATOM", "TRX", "COMP","VEEV", "LEN", "PHM", "DHI", "KBH", "TOL", "NVR", "RMAX", "BURL", "TJX", "ROST", "KSS", "GPS", "LB",
    "GLD", "SLV", "GDX", "GDXJ", "SPY", "QQQ", "IWM", "DIA", "XLF", "XLC", "XLI", "XLB", "XLC", "XLV", "XLI", "XLP", "XLY","XLK", "XBI", "XHB", "URBN", "ANF", "AEO", "XRT"
]))

mes_symbols = ["QSI", "GLD","SYM","INGA.AS", "FLEX", "ALDX", "TSM", "02020.HK", "ARCT", "CACI", "ERJ", "PYPL", "GLW", "MSFT",
               "TMDX", "GILT", "ENR.DE", "META", "AMD", "ASML.NV", "TBLA", "VOOG", "WELL", "SMSN.L", "BMRN", "GS", "BABA",
               "SMTC", "AFX.DE", "ABBN.ZU", "QCOM", "MP", "TM", "SGMT", "AMZN", "INOD", "SMCI", "GOOGL", "MU", "ETOR", 
               "DDOG", "OKTA", "AXSM", "EEM", "SPY", "HMY", "2318.HK", "RHM.DE", "NVDA", "QBTS", "SAP.DE", "V"]

# popular_symbols = list(set(mes_symbols))

def analyse_signaux_populaires(
    popular_symbols,
    mes_symbols,
    period="12mo",
    afficher_graphiques=True,
    chunk_size=15,
    verbose=True
):
    """
    Analyse les signaux pour les actions populaires, affiche les résultats, effectue le backtest,
    et affiche les graphiques pour les signaux fiables si demandé.
    Retourne un dictionnaire contenant les résultats principaux.
    """
    import matplotlib.pyplot as plt

    if verbose:
        print("\n🔍 Analyse des signaux pour actions populaires...")
    signals = []

    for i in range(0, len(popular_symbols), chunk_size):
        chunk = popular_symbols[i:i+chunk_size]
        if verbose:
            print(f"\n🔎 Lot {i//chunk_size + 1}: {', '.join(chunk)}")
        try:
            chunk_data = download_stock_data(chunk, period)
            for symbol, stock_data in chunk_data.items():
                prices = stock_data['Close']
                volumes = stock_data['Volume']
                if len(prices) < 50:
                    continue
                signal, last_price, trend, last_rsi, volume_mean = get_trading_signal(prices, volumes)
                if signal != "NEUTRE":
                    try:
                        info = yf.Ticker(symbol).info
                        domaine = info.get("sector", "Inconnu")
                    except Exception:
                        domaine = "Inconnu"
                    signals.append({
                        'Symbole': symbol,
                        'Signal': signal,
                        'Prix': last_price,
                        'Tendance': "Hausse" if trend else "Baisse",
                        'RSI': last_rsi,
                        'Domaine': domaine,
                        'Volume moyen': volume_mean
                    })
        except Exception as e:
            if verbose:
                print(f"⚠️ Erreur: {str(e)}")
        time.sleep(1)

    # Affichage des résultats
    signaux_tries = {"ACHAT": {"Hausse": [], "Baisse": []}, "VENTE": {"Hausse": [], "Baisse": []}}
    if signals:
        if verbose:
            print("\n" + "=" * 105)
            print("RÉSULTATS DES SIGNEAUX")
            print("=" * 105)
            print(f"{'Symbole':<8} {'Signal':<8} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Volume moyen':<15} {'Domaine':<24} Analyse")
            print("-" * 105)
        for s in signals:
            if s['Signal'] in signaux_tries and s['Tendance'] in signaux_tries[s['Signal']]:
                signaux_tries[s['Signal']][s['Tendance']].append(s)
        for signal_type in ["ACHAT", "VENTE"]:
            for tendance in ["Hausse", "Baisse"]:
                if signaux_tries[signal_type][tendance]:
                    signaux_tries[signal_type][tendance].sort(key=lambda x: x['Prix'])
                    if verbose:
                        print(f"\n--- Signal {signal_type} | Tendance {tendance} ---")
                        for s in signaux_tries[signal_type][tendance]:
                            analysis = ""
                            if s['Signal'] == "ACHAT":
                                analysis += "RSI bas" if s['RSI'] < 40 else ""
                                analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                            else:
                                analysis += "RSI élevé" if s['RSI'] > 60 else ""
                                analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""
                            print(f"{s['Symbole']:<8} {s['Signal']:<8} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Volume moyen']:<15,.0f} {s['Domaine']:<24} {analysis} ")
        if verbose:
            print("=" * 105)
    else:
        if verbose:
            print("\nℹ️ Aucun signal fort détecté parmi les actions populaires")
        return {}

    # Résumé du backtest sur les signaux détectés
    total_trades = 0
    total_gagnants = 0
    total_gain = 0.0
    cout_par_trade = 1.0
    backtest_results = []
    for s in signals:
        try:
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            if not isinstance(prices, (pd.Series, pd.DataFrame)) or len(prices) < 2:
                if verbose:
                    print(f"{s['Symbole']:<8} : Données insuffisantes pour le backtest")
                continue
            resultats = backtest_signals(prices, volumes, montant=50)
            backtest_results.append({
                "Symbole": s['Symbole'],
                "trades": resultats['trades'],
                "gagnants": resultats['gagnants'],
                "taux_reussite": resultats['taux_reussite'],
                "gain_total": resultats['gain_total']
            })
            total_trades += resultats['trades']
            total_gagnants += resultats['gagnants']
            total_gain += resultats['gain_total']
        except Exception as e:
            if verbose:
                print(f"{s['Symbole']:<8} : Erreur backtest ({e})")
    backtest_results.sort(key=lambda x: x['taux_reussite'], reverse=True)
    if verbose:
        for res in backtest_results:
            print(
                f"{res['Symbole']:<8} | Trades: {res['trades']:<2} | "
                f"Gagnants: {res['gagnants']:<2} | "
                f"Taux réussite: {res['taux_reussite']:.0f}% | "
                f"Gain total: {res['gain_total']:.2f} $"
            )
        cout_total_trades = total_trades * cout_par_trade
        total_investi_reel = len(backtest_results) * 50
        gain_total_reel = total_gain - cout_total_trades
        if total_trades > 0:
            taux_global = total_gagnants / total_trades * 100
            print("\n" + "="*105)
            print(f"🌍 Résultat global :")
            print(f"  - Taux de réussite = {taux_global:.1f}%")
            print(f"  - Nombre de trades = {total_trades}")
            print(f"  - Total investi réel = {total_investi_reel:.2f} $ (50 $ par action analysée)")
            print(f"  - Coût total des trades = {cout_total_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
            print(f"  - Gain total brut = {total_gain:.2f} $")
            print(f"  - Gain total net (après frais) = {gain_total_reel:.2f} $")
            print("="*105)
        else:
            print("\nAucun trade détecté pour le calcul global.")

    # Évaluation supplémentaire : stratégie filtrée
    filtres = [res for res in backtest_results if res['taux_reussite'] >= 60 and res['gain_total'] > 0]
    nb_actions_filtrees = len(filtres)
    total_trades_filtre = sum(res['trades'] for res in filtres)
    total_gagnants_filtre = sum(res['gagnants'] for res in filtres)
    total_gain_filtre = sum(res['gain_total'] for res in filtres)
    cout_total_trades_filtre = total_trades_filtre * cout_par_trade
    total_investi_filtre = nb_actions_filtrees * 50
    gain_total_reel_filtre = total_gain_filtre - cout_total_trades_filtre
    if verbose:
        print("\n" + "="*105)
        print("🔎 Évaluation si investissement SEULEMENT sur les actions à taux de réussite >= 60% ET gain total positif :")
        print(f"  - Nombre d'actions sélectionnées = {nb_actions_filtrees}")
        print(f"  - Nombre de trades = {total_trades_filtre}")
        print(f"  - Taux de réussite global = {(total_gagnants_filtre / total_trades_filtre * 100) if total_trades_filtre else 0:.1f}%")
        print(f"  - Total investi réel = {total_investi_filtre:.2f} $ (50 $ par action sélectionnée)")
        print(f"  - Coût total des trades = {cout_total_trades_filtre:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
        print(f"  - Gain total brut = {total_gain_filtre:.2f} $")
        print(f"  - Gain total net (après frais) = {gain_total_reel_filtre:.2f} $")
        print("="*105)

    # Tableau des signaux pour actions fiables (>=60% taux de réussite) ou non encore évaluables
    fiables_ou_non_eval = set()
    for res in backtest_results:
        if res['taux_reussite'] >= 60 or res['trades'] == 0:
            fiables_ou_non_eval.add(res['Symbole'])
    if verbose:
        print("\n" + "=" * 105)
        print("SIGNES UNIQUEMENT POUR ACTIONS FIABLES (>=60% taux de réussite) OU NON ÉVALUÉES")
        print("=" * 105)
        print(f"{'Symbole':<8} {'Signal':<8} {'Prix':<10} {'Tendance':<10} {'RSI':<6} {'Volume moyen':<15} {'Domaine':<24} Analyse")
        print("-" * 105)
    for signal_type in ["ACHAT", "VENTE"]:
        for tendance in ["Hausse", "Baisse"]:
            filtered = [
                s for s in signaux_tries[signal_type][tendance]
                if s['Symbole'] in fiables_ou_non_eval
            ]
            if filtered and verbose:
                print(f"\n--- Signal {signal_type} | Tendance {tendance} ---")
                for s in filtered:
                    analysis = ""
                    if s['Signal'] == "ACHAT":
                        analysis += "RSI bas" if s['RSI'] < 40 else ""
                        analysis += " + Tendance haussière" if s['Tendance'] == "Hausse" else ""
                    else:
                        analysis += "RSI élevé" if s['RSI'] > 60 else ""
                        analysis += " + Tendance baissière" if s['Tendance'] == "Baisse" else ""
                    print(f"{s['Symbole']:<8} {s['Signal']:<8} {s['Prix']:<10.2f} {s['Tendance']:<10} {s['RSI']:<6.1f} {s['Volume moyen']:<15,.0f} {s['Domaine']:<24} {analysis} ")
    if verbose:
        print("=" * 105)

    # Affichage des graphiques pour les 5 premiers signaux d'achat et de vente FIABLES
    top_achats_fiables = []
    top_ventes_fiables = []
    for signal_type in ["ACHAT", "VENTE"]:
        for tendance in ["Hausse", "Baisse"]:
            filtered = [
                s for s in signaux_tries[signal_type][tendance]
                if s['Symbole'] in fiables_ou_non_eval
            ]
            if signal_type == "ACHAT":
                top_achats_fiables.extend(filtered)
            else:
                top_ventes_fiables.extend(filtered)
    top_achats_fiables = top_achats_fiables[:5]
    top_ventes_fiables = top_ventes_fiables[:5]
    fiabilite_dict = {res['Symbole']: res['taux_reussite'] for res in backtest_results}

    if afficher_graphiques and top_achats_fiables:
        print("\nAffichage des graphiques pour les 5 premiers signaux d'ACHAT FIABLES détectés (sur une même figure)...")
        fig, axes = plt.subplots(len(top_achats_fiables), 1, figsize=(14, 5 * len(top_achats_fiables)), sharex=False)
        if len(top_achats_fiables) == 1:
            axes = [axes]
        for i, s in enumerate(top_achats_fiables):
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            plot_unified_chart(s['Symbole'], prices, volumes, axes[i])
            if len(prices) > 1:
                progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
                if isinstance(progression, pd.Series):
                    progression = float(progression.iloc[0])
            else:
                progression = 0.0
            signal, last_price, trend, last_rsi, volume_mean = get_trading_signal(prices, volumes)
            taux_fiabilite = fiabilite_dict.get(s['Symbole'], None)
            fiabilite_str = f" | Fiabilité: {taux_fiabilite:.0f}%" if taux_fiabilite is not None else ""
            if last_price is not None:
                trend_symbol = "Haussière" if trend else "Baissière"
                rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
                signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'
                special_marker = " ‼️" if s['Symbole'] in mes_symbols else ""
                title = (
                    f"{special_marker} {s['Symbole']} | Prix: {last_price:.2f} | Signal: {signal} {fiabilite_str} | "
                    f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | Vol. moyen: {volume_mean:,.0f} (unité) | "
                    f"Progression: {progression:+.2f}% {special_marker}"
                )
                axes[i].set_title(title, fontsize=12, fontweight='bold', color=signal_color)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        plt.show()

    if afficher_graphiques and top_ventes_fiables:
        print("\nAffichage des graphiques pour les 5 premiers signaux de VENTE FIABLES détectés (sur une même figure)...")
        fig, axes = plt.subplots(len(top_ventes_fiables), 1, figsize=(14, 5 * len(top_ventes_fiables)), sharex=False)
        if len(top_ventes_fiables) == 1:
            axes = [axes]
        for i, s in enumerate(top_ventes_fiables):
            stock_data = download_stock_data([s['Symbole']], period)[s['Symbole']]
            prices = stock_data['Close']
            volumes = stock_data['Volume']
            plot_unified_chart(s['Symbole'], prices, volumes, axes[i])
            if len(prices) > 1:
                progression = ((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0]) * 100
                if isinstance(progression, pd.Series):
                    progression = float(progression.iloc[0])
            else:
                progression = 0.0
            signal, last_price, trend, last_rsi, volume_mean = get_trading_signal(prices, volumes)
            taux_fiabilite = fiabilite_dict.get(s['Symbole'], None)
            fiabilite_str = f" | Fiabilité: {taux_fiabilite:.0f}%" if taux_fiabilite is not None else ""
            if last_price is not None:
                trend_symbol = "Haussière" if trend else "Baissière"
                rsi_status = "SURACH" if last_rsi > 70 else "SURVENTE" if last_rsi < 30 else "NEUTRE"
                signal_color = 'green' if signal == "ACHAT" else 'red' if signal == "VENTE" else 'black'
                special_marker = " ‼️" if s['Symbole'] in mes_symbols else ""
                title = (
                    f"{special_marker} {s['Symbole']} | Prix: {last_price:.2f} | Signal: {signal} {fiabilite_str} | "
                    f"Tendance: {trend_symbol} | RSI: {last_rsi:.1f} ({rsi_status}) | Vol. moyen: {s['Volume moyen']:.0f} (unité) | "
                    f"Progression: {progression:+.2f}% {special_marker}"
                )
                axes[i].set_title(title, fontsize=12, fontweight='bold', color=signal_color)
        plt.tight_layout()
        plt.subplots_adjust(top=0.95, hspace=0.4)
        plt.show()

    # Retourne les résultats pour usage ultérieur
    return {
        "signals": signals,
        "signaux_tries": signaux_tries,
        "backtest_results": backtest_results,
        "fiables_ou_non_eval": fiables_ou_non_eval,
        "top_achats_fiables": top_achats_fiables,
        "top_ventes_fiables": top_ventes_fiables
    }

# Pour utiliser la fonction sans exécution automatique :
resultats = analyse_signaux_populaires(popular_symbols, mes_symbols, period="12mo", afficher_graphiques=True)



# # ======================================================================
# # Évaluation dynamique : investissement uniquement si l'action est "fiable" au moment du signal
# # ======================================================================
# print("\n" + "="*105)
# print("🔎 Simulation dynamique : investissement SEULEMENT si l'action est déjà >50% réussite ET gain positif au moment du signal")
# print("="*105)
# print(f"{'Symbole':<8} {'Entrée':<10} {'Sortie':<10} {'Résultat':<8} {'Gain($)':<10} {'Taux%':<7} {'GainTot($)':<10}")

# total_dyn_trades = 0
# total_dyn_gagnants = 0
# total_dyn_gain = 0.0
# cout_total_dyn_trades = 0
# actions_dyn = set()

# for s in signals:
#     try:
#         prices = download_stock_data([s['Symbole']], period)[s['Symbole']]
#         positions = []
#         for i in range(50, len(prices)):  # Commence après 50 points pour avoir un historique suffisant
#             past_prices = prices[:i]
#             past_volumes = Volume[:i]
#             # On effectue le backtest sur les 50 derniers points
#             stats = backtest_signals(past_prices, montant=50)
#             # On vérifie la fiabilité à ce moment précis
#             # if stats['trades'] > 0 and stats['taux_reussite'] > 70 and stats['gain_total'] > 0:
#             if stats['trades'] > 0 and stats['taux_reussite'] > 60 and stats['gain_total'] > 0:
#                 signal, _, _, _ = get_trading_signal(past_prices, past_volumes)
#                 if signal == "ACHAT":
#                     entry = prices.iloc[i]
#                     entry_idx = i
#                     # Cherche la prochaine sortie (VENTE)
#                     for j in range(i+1, len(prices)):
#                         next_signal, _, _, _ = get_trading_signal(prices[:j], volumes[:j])
#                         if next_signal == "VENTE":
#                             exit = prices.iloc[j]
#                             rendement = (exit - entry) / entry
#                             gain = 50 * rendement
#                             total_dyn_trades += 1
#                             cout_total_dyn_trades += cout_par_trade
#                             total_dyn_gain += gain
#                             actions_dyn.add(s['Symbole'])
#                             if gain > 0:
#                                 total_dyn_gagnants += 1
#                             print(f"{s['Symbole']:<8} {entry:>10.2f} {exit:>10.2f} {'Gagnant' if gain>0 else 'Perdant':<8} {gain:>10.2f} {stats['taux_reussite']:>7.1f} {stats['gain_total']:>10.2f}")
#                             break  # Passe au prochain signal d'achat
#     except Exception as e:
#         print(f"{s['Symbole']:<8} : Erreur simulation dynamique ({e})")

# nb_dyn_actions = len(actions_dyn)
# total_investi_dyn = nb_dyn_actions * 50
# gain_total_reel_dyn = total_dyn_gain - cout_total_dyn_trades
# taux_dyn = (total_dyn_gagnants / total_dyn_trades * 100) if total_dyn_trades else 0

# print("="*105)
# print(f"  - Nombre d'actions sélectionnées dynamiquement = {nb_dyn_actions}")
# print(f"  - Nombre de trades dynamiques = {total_dyn_trades}")
# print(f"  - Taux de réussite global = {taux_dyn:.1f}%")
# print(f"  - Total investi réel = {total_investi_dyn:.2f} $ (50 $ par action sélectionnée)")
# print(f"  - Coût total des trades = {cout_total_dyn_trades:.2f} $ (à {cout_par_trade:.2f} $ par trade)")
# print(f"  - Gain total brut = {total_dyn_gain:.2f} $")
# print(f"  - Gain total net (après frais) = {gain_total_reel_dyn:.2f} $")
# print("="*105)



#todo :# 1. Ajouter une option pour sauvegarder les signaux dans un fichier CSV
# 2. Ajouter une option pour afficher les graphiques des 5 premiers signaux d'achat et de vente finalement détectés