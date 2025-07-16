# trading_ai_system.py

import yfinance as yf
import pandas as pd
import numpy as np
import ta
import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import requests
from bs4 import BeautifulSoup

# --------- Module : Téléchargement des données ---------
def download_stock_data(symbol, period="6mo"):
    data = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=True)
    if data.empty or 'Close' not in data.columns:
        return None, None
    return data['Close'].dropna().squeeze(), data['Volume'].dropna().squeeze()

# --------- Module : Indicateurs Techniques ---------
def calculate_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# --------- Module : Analyse approfondie des nouvelles ---------
def get_detailed_news_sentiment(symbol):
    try:
        url = f"https://finance.yahoo.com/quote/{symbol}/news?p={symbol}"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            return 0
        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('li', class_='js-stream-content')[:5]
        analyzer = SentimentIntensityAnalyzer()
        scores = []
        for article in articles:
            text = article.get_text()
            sentiment = analyzer.polarity_scores(text)['compound']
            scores.append(sentiment)
        return np.mean(scores) if scores else 0
    except:
        return 0

# --------- Module : Prédiction IA ---------
def load_model(path="light_model.pkl"):
    try:
        return joblib.load(path)
    except:
        return None

def predict_ml_score(model, features):
    if model is None:
        return 0.5
    try:
        X = pd.DataFrame([features])
        return float(model.predict_proba(X)[0][1])
    except:
        return 0.5

# --------- Module : Détection du signal ---------
def get_trading_signal(prices, volumes, model, symbol):
    if isinstance(prices, pd.DataFrame):
        prices = prices.squeeze()
    if isinstance(volumes, pd.DataFrame):
        volumes = volumes.squeeze()

    if prices.ndim > 1:
        prices = prices.iloc[:, 0]
    if volumes.ndim > 1:
        volumes = volumes.iloc[:, 0]

    if len(prices) < 50 or len(volumes) < 30:
        return "Données insuffisantes", None, None, None

    macd, signal_line = calculate_macd(prices)
    rsi = ta.momentum.RSIIndicator(close=prices, window=17).rsi()
    ema20 = prices.ewm(span=20).mean()
    ema50 = prices.ewm(span=50).mean()

    last_close = prices.iloc[-1]
    volume_mean = volumes.rolling(30).mean().iloc[-1]
    variation_30j = (last_close - prices.iloc[-30]) / prices.iloc[-30] * 100

    features = {
        'macd': macd.iloc[-1],
        'macd_diff': macd.iloc[-1] - signal_line.iloc[-1],
        'rsi': rsi.iloc[-1],
        'ema20_diff': last_close - ema20.iloc[-1],
        'ema50_diff': last_close - ema50.iloc[-1],
        'volume_mean': volume_mean,
        'variation_30j': variation_30j,
    }

    sentiment = get_detailed_news_sentiment(symbol)
    ml_score = predict_ml_score(model, features)

    is_macd_cross_up = macd.iloc[-2] < signal_line.iloc[-2] and macd.iloc[-1] > signal_line.iloc[-1]
    is_macd_cross_down = macd.iloc[-2] > signal_line.iloc[-2] and macd.iloc[-1] < signal_line.iloc[-1]
    is_volume_ok = volume_mean > 100000
    is_variation_ok = variation_30j > -20
    is_sentiment_ok = sentiment > 0.1
    is_ml_buy = ml_score > 0.6
    is_ml_sell = ml_score < 0.4

    if is_macd_cross_up and is_volume_ok and is_variation_ok and rsi.iloc[-1] < 75 and is_sentiment_ok and is_ml_buy:
        return "ACHAT", last_close, round(ml_score, 2), round(sentiment, 2)
    elif is_macd_cross_down and rsi.iloc[-1] > 30 and is_ml_sell:
        return "VENTE", last_close, round(ml_score, 2), round(sentiment, 2)
    else:
        return "NEUTRE", last_close, round(ml_score, 2), round(sentiment, 2)

# --------- Routine Principale ---------
if __name__ == "__main__":
    watchlist = ["SYM", "SMCI", "NVDA", "TSLA", "RHM.DE", "ENR.DE", "SAP.DE", "TBLA", "TM", "ERJ", "BMRN", "QSI", "WELL", "ALDX"]
    model = load_model()

    for symbol in watchlist:
        prices, volumes = download_stock_data(symbol)
        if prices is None:
            print(f"❌ Données manquantes pour {symbol}")
            continue

        signal, close, ml_score, sentiment = get_trading_signal(prices, volumes, model, symbol)
        print(f"\n🔍 {symbol} | Signal: {signal} | Close: {close:.2f} | IA: {ml_score} | Sentiment: {sentiment}")
