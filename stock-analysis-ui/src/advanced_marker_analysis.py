"""
Advanced Marker Discovery Tool
 
Analyzes technical and fundamental patterns in winning vs losing stocks
to identify predictive markers and entry signals.
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
from tqdm import tqdm


class AdvancedMarkerDiscovery:
    """Deep analysis of stock characteristics"""
    
    def __init__(self):
        self.cache = {}
    
    def analyze_stock(self, symbol: str, period: str = '2y') -> Dict:
        """
        Comprehensive analysis of a single stock.
        
        Returns dict with:
        - price metrics (momentum, RSI, Bollinger Bands)
        - volatility measures
        - trend strength
        - volume patterns
        - fundamental signals (if available)
        """
        
        if symbol in self.cache:
            return self.cache[symbol]
        
        try:
            # Download historical data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            info = ticker.info
            
            if len(hist) < 50:
                return None
            
            analysis = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'price_data': self._analyze_price_metrics(hist, info),
                'momentum': self._analyze_momentum(hist),
                'volatility': self._analyze_volatility(hist),
                'trend': self._analyze_trend(hist),
                'volume': self._analyze_volume(hist),
                'fundamentals': self._analyze_fundamentals(info),
            }
            
            self.cache[symbol] = analysis
            return analysis
            
        except Exception as e:
            print(f"   ⚠️ Error analyzing {symbol}: {e}")
            return None
    
    def _analyze_price_metrics(self, hist: pd.DataFrame, info: Dict) -> Dict:
        """Current price metrics"""
        close = hist['Close']
        high_52w = info.get('fiftyTwoWeekHigh', close.max())
        low_52w = info.get('fiftyTwoWeekLow', close.min())
        
        current = close.iloc[-1]
        
        return {
            'current_price': float(current),
            '52w_high': float(high_52w) if high_52w else None,
            '52w_low': float(low_52w) if low_52w else None,
            'from_52w_low_pct': float(((current - low_52w) / low_52w * 100)) if low_52w > 0 else 0,
            'from_52w_high_pct': float(((current - high_52w) / high_52w * 100)) if high_52w > 0 else 0,
            'avg_volume_3m': float(hist['Volume'].tail(60).mean()),
            'avg_volume_1y': float(hist['Volume'].tail(252).mean()),
        }
    
    def _analyze_momentum(self, hist: pd.DataFrame) -> Dict:
        """Momentum indicators"""
        close = hist['Close']
        
        # Simple momentum (% change)
        mom_1m = ((close.iloc[-1] - close.iloc[-21]) / close.iloc[-21] * 100) if len(close) > 21 else 0
        mom_3m = ((close.iloc[-1] - close.iloc[-60]) / close.iloc[-60] * 100) if len(close) > 60 else 0
        mom_1y = ((close.iloc[-1] - close.iloc[-252]) / close.iloc[-252] * 100) if len(close) > 252 else 0
        
        # RSI (Relative Strength Index)
        rsi = self._calculate_rsi(close, period=14)
        
        # MACD
        macd_line, signal_line, macd_hist = self._calculate_macd(close)
        
        return {
            'momentum_1m': float(mom_1m),
            'momentum_3m': float(mom_3m),
            'momentum_1y': float(mom_1y),
            'rsi_14': float(rsi[-1]) if rsi[-1] else 0,
            'macd_line': float(macd_line[-1]) if macd_line[-1] else 0,
            'macd_histogram': float(macd_hist[-1]) if macd_hist[-1] else 0,
            'macd_positive': bool(macd_hist[-1] > 0) if macd_hist[-1] else False,
        }
    
    def _analyze_volatility(self, hist: pd.DataFrame) -> Dict:
        """Volatility measures"""
        close = hist['Close']
        
        # Daily returns volatility
        returns = close.pct_change().dropna()
        vol_daily = returns.std() * 100
        vol_annual = vol_daily * np.sqrt(252)
        
        # Bollinger Bands
        sma20 = close.rolling(20).mean()
        bb_std = close.rolling(20).std()
        bb_upper = sma20 + (bb_std * 2)
        bb_lower = sma20 - (bb_std * 2)
        
        if len(bb_lower) > 0:
            current_price = close.iloc[-1]
            bb_position = (current_price - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1])
        else:
            bb_position = 0.5
        
        return {
            'volatility_daily': float(vol_daily),
            'volatility_annual': float(vol_annual),
            'bb_upper': float(bb_upper.iloc[-1]) if len(bb_upper) > 0 else 0,
            'bb_lower': float(bb_lower.iloc[-1]) if len(bb_lower) > 0 else 0,
            'bb_position': float(bb_position),  # 0=lower band, 1=upper band
            'bb_squeeze': bool(bb_std.iloc[-1] < bb_std.iloc[-50:].quantile(0.25)) if len(bb_std) > 50 else False,
        }
    
    def _analyze_trend(self, hist: pd.DataFrame) -> Dict:
        """Trend analysis"""
        close = hist['Close']
        
        # EMA (Exponential Moving Average)
        ema20 = close.ewm(span=20).mean()
        ema50 = close.ewm(span=50).mean()
        ema200 = close.ewm(span=200).mean()
        
        current = close.iloc[-1]
        
        # Trend signal
        ema_signal = 0
        if current > ema20.iloc[-1]:
            ema_signal += 1
        if current > ema50.iloc[-1]:
            ema_signal += 1
        if current > ema200.iloc[-1]:
            ema_signal += 1
        
        return {
            'price_vs_ema20': float((current / ema20.iloc[-1] - 1) * 100),
            'price_vs_ema50': float((current / ema50.iloc[-1] - 1) * 100),
            'price_vs_ema200': float((current / ema200.iloc[-1] - 1) * 100),
            'ema20_vs_ema50': float((ema20.iloc[-1] / ema50.iloc[-1] - 1) * 100),
            'ema_bullish_signal': int(ema_signal),  # 0-3 score
            'in_uptrend': bool(ema_signal >= 2),
        }
    
    def _analyze_volume(self, hist: pd.DataFrame) -> Dict:
        """Volume analysis"""
        volume = hist['Volume']
        close = hist['Close']
        
        # Volume trend
        vol_avg_20 = volume.tail(20).mean()
        vol_avg_200 = volume.tail(200).mean()
        vol_ratio = vol_avg_20 / vol_avg_200 if vol_avg_200 > 0 else 1
        
        # On Balance Volume (OBV)
        obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
        obv_trend = (obv.iloc[-1] - obv.iloc[-20]) / obv.iloc[-20] if obv.iloc[-20] != 0 else 0
        
        return {
            'volume_avg_20': float(vol_avg_20),
            'volume_avg_200': float(vol_avg_200),
            'volume_ratio_20_200': float(vol_ratio),
            'volume_increasing': bool(vol_ratio > 1.1),
            'obv_trend': float(obv_trend),
        }
    
    def _analyze_fundamentals(self, info: Dict) -> Dict:
        """Fundamental metrics"""
        return {
            'market_cap': info.get('marketCap', None),
            'pe_ratio': info.get('trailingPE', None),
            'pb_ratio': info.get('priceToBook', None),
            'dividend_yield': info.get('dividendYield', None),
            'profit_margin': info.get('profitMargins', None),
            'revenue_growth': info.get('revenueGrowth', None),
            'earnings_growth': info.get('earningsGrowth', None),
            'debt_to_equity': info.get('debtToEquity', None),
            'roe': info.get('returnOnEquity', None),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
        }
    
    def _calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_macd(self, prices, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist
    
    def compare_populations(self, winners: List[str], losers: List[str]) -> Dict:
        """
        Compare technical characteristics between two populations
        """
        
        print("\n[ANALYSIS] Analyzing winners...")
        winner_data = []
        for sym in tqdm(winners, desc="Winners"):
            analysis = self.analyze_stock(sym)
            if analysis:
                winner_data.append(analysis)
        
        print("\n[ANALYSIS] Analyzing losers...")
        loser_data = []
        for sym in tqdm(losers, desc="Losers"):
            analysis = self.analyze_stock(sym)
            if analysis:
                loser_data.append(analysis)
        
        # Compare metrics
        comparison = {
            'winners_count': len(winner_data),
            'losers_count': len(loser_data),
            'metrics': {},
        }
        
        # Extract all metrics and calculate statistics
        if winner_data and loser_data:
            metrics_to_compare = [
                ('momentum', 'momentum_1m'),
                ('momentum', 'momentum_3m'),
                ('momentum', 'momentum_1y'),
                ('momentum', 'rsi_14'),
                ('volatility', 'volatility_daily'),
                ('volatility', 'volatility_annual'),
                ('volatility', 'bb_position'),
                ('trend', 'ema_bullish_signal'),
                ('trend', 'price_vs_ema20'),
                ('volume', 'volume_ratio_20_200'),
            ]
            
            for section, metric in metrics_to_compare:
                winner_values = [
                    d[section][metric] for d in winner_data 
                    if section in d and metric in d[section] and d[section][metric] is not None
                ]
                loser_values = [
                    d[section][metric] for d in loser_data 
                    if section in d and metric in d[section] and d[section][metric] is not None
                ]
                
                if winner_values and loser_values:
                    comparison['metrics'][f"{section}_{metric}"] = {
                        'winners_mean': float(np.mean(winner_values)),
                        'losers_mean': float(np.mean(loser_values)),
                        'winners_median': float(np.median(winner_values)),
                        'losers_median': float(np.median(loser_values)),
                        'winners_std': float(np.std(winner_values)),
                        'losers_std': float(np.std(loser_values)),
                        'difference': float(np.mean(winner_values) - np.mean(loser_values)),
                    }
        
        return comparison
    
    def print_comparison(self, comparison: Dict):
        """Print formatted comparison"""
        
        print("\n" + "="*80)
        print("TECHNICAL MARKER COMPARISON: WINNERS vs LOSERS")
        print("="*80)
        
        print(f"\nSample size: {comparison['winners_count']} winners, {comparison['losers_count']} losers\n")
        
        for metric, stats in sorted(comparison['metrics'].items()):
            diff = stats['difference']
            symbol = "[+]" if diff > 0 else "[-]"
            
            print(f"{symbol} {metric}")
            print(f"    Winners:  {stats['winners_mean']:>10.2f}  (sigma={stats['winners_std']:.2f})")
            print(f"    Losers:   {stats['losers_mean']:>10.2f}  (sigma={stats['losers_std']:.2f})")
            print(f"    Delta:    {diff:>10.2f}  ({(diff/abs(stats['losers_mean']) if stats['losers_mean'] != 0 else 0)*100:+.1f}%)")
            print()


def main():
    """Main analysis"""
    
    # Winners from your data
    winners = ['CDTX', 'SEZL', 'OUST', 'OKLO', 'TRVI', 'RDDT', 'NVDA', 'AVGO', 'LAES', 'GEV']
    # Shuffle into top and bottom for comparison
    all_sorted = ['CDTX', 'SEZL', 'OUST', 'OKLO', 'TRVI', 'RDDT', 'NVDA', 'AVGO', 'LAES', 'GEV']
    winners_top = all_sorted[:5]
    winners_bottom = all_sorted[5:]
    
    analyzer = AdvancedMarkerDiscovery()
    comparison = analyzer.compare_populations(winners_top, winners_bottom)
    analyzer.print_comparison(comparison)
    
    # Save results
    with open('advanced_marker_analysis.json', 'w') as f:
        def default_handler(x):
            if isinstance(x, (np.integer, np.floating)):
                return float(x)
            elif isinstance(x, np.ndarray):
                return x.tolist()
            raise TypeError(f"Object of type {type(x).__name__} is not JSON serializable")
        
        json.dump(comparison, f, default=default_handler, indent=2)
    
    print("\n✓ Full analysis saved to advanced_marker_analysis.json")
    return comparison


if __name__ == '__main__':
    main()
