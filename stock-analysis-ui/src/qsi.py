def download_stock_data(symbols, period):
    """Télécharge les données avec gestion des erreurs"""
    valid_data = {}
    
    for symbol in symbols:
        try:
            data = yf.download(symbol, period=period, interval="1d", progress=False, timeout=10)
            
            if data.empty:
                print(f"⚠️ Aucune donnée pour {symbol}")
                continue
                
            if 'Close' not in data.columns or 'Volume' not in data.columns:
                print(f"⚠️ Colonne 'Close' ou 'Volume' manquante pour {symbol}")
                continue
                
            valid_data[symbol] = {
                'Close': data['Close'],
                'Volume': data['Volume']}
            
        except Exception as e:
            print(f"🚨 Erreur sur {symbol}: {str(e)}")
    
    return valid_data

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calcule le MACD et sa ligne de signal"""
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def get_trading_signal(prices, volumes, variation_seuil=-20, volume_seuil=100000):
    """Détermine les signaux de trading avec validation des données"""
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()
    if len(prices) < 50:
        return "Données insuffisantes", None, None, None
    
    macd, signal_line = calculate_macd(prices)
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    
    if len(macd) < 2 or len(rsi) < 1:
        return "Données récentes manquantes", None, None, None
    
    last_close = prices.iloc[-1]
    ema20 = prices.ewm(span=20, adjust=False).mean().iloc[-1]
    ema50 = prices.ewm(span=50, adjust=False).mean().iloc[-1]
    
    if len(prices) >= 30:
        variation_30j = (last_close - prices.iloc[-30]) / prices.iloc[-30] * 100
    else:
        variation_30j = None
    
    if len(volumes) >= 20:
        volume_mean = volumes.rolling(window=20).mean().iloc[-1]
    else:
        volume_mean = None
    
    if macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1] and rsi.iloc[-1] < 70:
        signal = "ACHAT"
    elif macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1] and rsi.iloc[-1] > 30:
        signal = "VENTE"
    else:
        signal = "NEUTRE"
    
    return signal, last_close, last_close > ema20, rsi.iloc[-1]