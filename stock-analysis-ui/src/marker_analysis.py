"""
Marker Analysis Tool - Analyze common patterns in winning stocks

Compares characteristics of high-performing stocks (winners) with lower performers
to identify common markers and patterns.
"""

import sqlite3
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Set
from pathlib import Path

# Project imports
import sys
import os
PROJECT_SRC = os.path.abspath(os.path.dirname(__file__))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

from qsi import download_stock_data


class MarkerAnalyzer:
    """Analyze markers (characteristics) of winning vs losing stocks"""
    
    def __init__(self, db_path: str = 'symbols.db'):
        self.db_path = db_path
        self.winners = []  # High performers
        self.losers = []   # Low performers
        
    def analyze_winners(self, 
                       symbols: List[str],
                       performance_data: Dict[str, float],  # symbol -> return_pct
                       threshold_pct: float = 50.0) -> Dict:
        """
        Analyze markers of winning stocks vs underperformers.
        
        Args:
            symbols: List of stock symbols to analyze
            performance_data: Dict mapping symbol to return percentage
            threshold_pct: Split threshold (top %ile vs bottom %ile)
        
        Returns:
            Dict with analysis results including:
            - winners: Top performing stocks and their characteristics
            - losers: Low performing stocks
            - common_markers: Shared characteristics of winners
            - differentiators: Unique markers of winners vs losers
        """
        
        # Split into winners and losers
        sorted_symbols = sorted(symbols, 
                               key=lambda s: performance_data.get(s, 0), 
                               reverse=True)
        
        split_idx = int(len(sorted_symbols) * (threshold_pct / 100))
        winners = sorted_symbols[:max(1, split_idx)]
        losers = sorted_symbols[max(split_idx, len(sorted_symbols)-split_idx):]
        
        self.winners = winners
        self.losers = losers
        
        print(f"\n📊 MARKER ANALYSIS")
        print(f"   Winners ({len(winners)}): {winners}")
        print(f"   Losers ({len(losers)}): {losers}")
        
        # Analyze characteristics
        result = {
            'winners': winners,
            'losers': losers,
            'common_markers': self._analyze_common_markers(winners),
            'differentiators': self._analyze_differentiators(winners, losers),
            'sector_analysis': self._analyze_sectors(winners, losers),
            'market_cap_analysis': self._analyze_market_caps(winners, losers),
            'price_patterns': self._analyze_price_patterns(winners, losers),
            'volatility_analysis': self._analyze_volatility(winners, losers),
        }
        
        return result
    
    def _analyze_common_markers(self, symbols: List[str]) -> Dict:
        """Analyze common characteristics of winning stocks"""
        markers = {
            'sectors': self._get_symbols_sectors(symbols),
            'market_caps': self._get_symbols_market_caps(symbols),
            'avg_volume': self._get_avg_volume(symbols),
            'price_ranges': self._get_price_ranges(symbols),
            'volatility': self._get_volatility(symbols),
        }
        return markers
    
    def _analyze_differentiators(self, winners: List[str], losers: List[str]) -> Dict:
        """Find markers that differentiate winners from losers"""
        
        diff = {}
        
        # Sector comparison
        winner_sectors = self._get_symbols_sectors(winners)
        loser_sectors = self._get_symbols_sectors(losers)
        diff['sectors'] = {
            'winners': winner_sectors,
            'losers': loser_sectors,
            'overrepresented_in_winners': set(winner_sectors.keys()) - set(loser_sectors.keys()),
        }
        
        # Market cap comparison
        winner_caps = self._get_symbols_market_caps(winners)
        loser_caps = self._get_symbols_market_caps(losers)
        diff['market_cap'] = {
            'winners_avg': np.mean([v for v in winner_caps.values() if v > 0]) if winner_caps else 0,
            'losers_avg': np.mean([v for v in loser_caps.values() if v > 0]) if loser_caps else 0,
        }
        
        # Volatility comparison
        winner_vol = self._get_volatility(winners)
        loser_vol = self._get_volatility(losers)
        diff['volatility'] = {
            'winners_avg': np.mean([v for v in winner_vol.values() if v > 0]) if winner_vol else 0,
            'losers_avg': np.mean([v for v in loser_vol.values() if v > 0]) if loser_vol else 0,
        }
        
        return diff
    
    def _analyze_sectors(self, winners: List[str], losers: List[str]) -> Dict:
        """Analyze sector distribution"""
        winner_sectors = self._get_symbols_sectors(winners)
        loser_sectors = self._get_symbols_sectors(losers)
        
        return {
            'winners': winner_sectors,
            'losers': loser_sectors,
            'winner_dominance': {
                sector: count for sector, count in winner_sectors.items()
                if loser_sectors.get(sector, 0) == 0
            }
        }
    
    def _analyze_market_caps(self, winners: List[str], losers: List[str]) -> Dict:
        """Analyze market cap preferences"""
        winner_caps = self._get_symbols_market_caps(winners)
        loser_caps = self._get_symbols_market_caps(losers)
        
        # Categorize by cap range
        def categorize_cap(cap_val):
            if cap_val >= 200e9:
                return "Mega (>$200B)"
            elif cap_val >= 10e9:
                return "Large ($10B-$200B)"
            elif cap_val >= 2e9:
                return "Mid ($2B-$10B)"
            else:
                return "Small (<$2B)"
        
        winner_ranges = {}
        loser_ranges = {}
        
        for sym, cap in winner_caps.items():
            if cap > 0:
                r = categorize_cap(cap)
                winner_ranges[r] = winner_ranges.get(r, 0) + 1
        
        for sym, cap in loser_caps.items():
            if cap > 0:
                r = categorize_cap(cap)
                loser_ranges[r] = loser_ranges.get(r, 0) + 1
        
        return {
            'winners_ranges': winner_ranges,
            'losers_ranges': loser_ranges,
        }
    
    def _analyze_price_patterns(self, winners: List[str], losers: List[str]) -> Dict:
        """Analyze price movement patterns"""
        print("   Analyzing price patterns...")
        
        # Get recent price data (last 6 months)
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        patterns = {
            'winners': {},
            'losers': {},
        }
        
        for sym in winners[:3]:  # Sample first 3 for speed
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    patterns['winners'][sym] = {
                        'recent_change': price_change,
                        'days_analyzed': len(hist),
                    }
            except Exception as e:
                pass
        
        for sym in losers[:3]:  # Sample first 3
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    price_change = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0]) * 100
                    patterns['losers'][sym] = {
                        'recent_change': price_change,
                        'days_analyzed': len(hist),
                    }
            except Exception as e:
                pass
        
        return patterns
    
    def _analyze_volatility(self, winners: List[str], losers: List[str]) -> Dict:
        """Analyze volatility patterns"""
        print("   Analyzing volatility...")
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        volatility = {
            'winners': {},
            'losers': {},
        }
        
        for sym in winners[:3]:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    daily_returns = hist['Close'].pct_change().dropna()
                    vol = daily_returns.std() * 100
                    volatility['winners'][sym] = vol
            except Exception:
                pass
        
        for sym in losers[:3]:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    daily_returns = hist['Close'].pct_change().dropna()
                    vol = daily_returns.std() * 100
                    volatility['losers'][sym] = vol
            except Exception:
                pass
        
        return volatility
    
    def _get_symbols_sectors(self, symbols: List[str]) -> Dict[str, int]:
        """Get sector distribution for symbols"""
        sectors = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for sym in symbols:
                cursor.execute("SELECT sector FROM symbols WHERE symbol=?", (sym,))
                row = cursor.fetchone()
                if row and row[0]:
                    sector = str(row[0]).strip()
                    sectors[sector] = sectors.get(sector, 0) + 1
            
            conn.close()
        except Exception as e:
            print(f"   ⚠️ Could not fetch sectors: {e}")
        
        return sectors
    
    def _get_symbols_market_caps(self, symbols: List[str]) -> Dict[str, float]:
        """Get market cap for symbols"""
        caps = {}
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            for sym in symbols:
                cursor.execute("SELECT market_cap_value FROM symbols WHERE symbol=?", (sym,))
                row = cursor.fetchone()
                if row and row[0]:
                    try:
                        caps[sym] = float(row[0])
                    except (ValueError, TypeError):
                        caps[sym] = 0
            
            conn.close()
        except Exception as e:
            print(f"   ⚠️ Could not fetch market caps: {e}")
        
        return caps
    
    def _get_avg_volume(self, symbols: List[str]) -> Dict[str, float]:
        """Get average trading volume"""
        volumes = {}
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=60)
        
        for sym in symbols[:3]:  # Sample
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 0 and 'Volume' in hist.columns:
                    volumes[sym] = hist['Volume'].mean()
            except Exception:
                pass
        
        return volumes
    
    def _get_price_ranges(self, symbols: List[str]) -> Dict[str, Tuple[float, float]]:
        """Get current price ranges"""
        ranges = {}
        
        for sym in symbols[:3]:  # Sample
            try:
                ticker = yf.Ticker(sym)
                info = ticker.info
                if 'fiftyTwoWeekHigh' in info and 'fiftyTwoWeekLow' in info:
                    ranges[sym] = (info['fiftyTwoWeekLow'], info['fiftyTwoWeekHigh'])
            except Exception:
                pass
        
        return ranges
    
    def _get_volatility(self, symbols: List[str]) -> Dict[str, float]:
        """Get volatility measures"""
        volatilities = {}
        
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=180)
        
        for sym in symbols:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(start=start_date, end=end_date)
                if len(hist) > 20:
                    daily_returns = hist['Close'].pct_change().dropna()
                    vol = daily_returns.std() * 100
                    volatilities[sym] = vol
            except Exception:
                pass
        
        return volatilities
    
    def print_report(self, analysis: Dict):
        """Print formatted analysis report"""
        
        print("\n" + "="*80)
        print("WINNING STOCKS MARKER ANALYSIS REPORT")
        print("="*80)
        
        # Winners vs Losers
        print(f"\n🏆 TOP PERFORMERS ({len(analysis['winners'])} stocks):")
        for sym in analysis['winners'][:5]:
            print(f"   • {sym}")
        if len(analysis['winners']) > 5:
            print(f"   ... and {len(analysis['winners']) - 5} more")
        
        print(f"\n📉 UNDERPERFORMERS ({len(analysis['losers'])} stocks):")
        for sym in analysis['losers'][:5]:
            print(f"   • {sym}")
        if len(analysis['losers']) > 5:
            print(f"   ... and {len(analysis['losers']) - 5} more")
        
        # Sector Analysis
        print(f"\n🏭 SECTOR ANALYSIS")
        print(f"   Winners dominate in: {analysis['differentiators']['sectors']['overrepresented_in_winners']}")
        print(f"   Winner sectors: {analysis['sector_analysis']['winners']}")
        print(f"   Loser sectors: {analysis['sector_analysis']['losers']}")
        
        # Market Cap Analysis
        print(f"\n💰 MARKET CAP ANALYSIS")
        print(f"   Winners cap ranges: {analysis['market_cap_analysis']['winners_ranges']}")
        print(f"   Losers cap ranges: {analysis['market_cap_analysis']['losers_ranges']}")
        print(f"   Winners avg cap: ${analysis['differentiators']['market_cap']['winners_avg']:,.0f}")
        print(f"   Losers avg cap: ${analysis['differentiators']['market_cap']['losers_avg']:,.0f}")
        
        # Volatility Analysis
        print(f"\n📈 VOLATILITY PATTERNS")
        print(f"   Winners avg volatility: {analysis['differentiators']['volatility']['winners_avg']:.2f}%")
        print(f"   Losers avg volatility: {analysis['differentiators']['volatility']['losers_avg']:.2f}%")
        
        # Price Patterns
        if analysis['price_patterns']['winners']:
            print(f"\n📊 RECENT PRICE PATTERNS (Winners)")
            for sym, data in analysis['price_patterns']['winners'].items():
                print(f"   {sym}: {data['recent_change']:+.1f}% over {data['days_analyzed']} days")
        
        print("\n" + "="*80)


def main():
    """Example usage"""
    
    # Example winners from your data
    winners_data = {
        'CDTX': 391.4,
        'SEZL': 328.0,
        'OUST': 282.7,
        'OKLO': 107.0,
        'TRVI': 101.3,
        'RDDT': 83.1,
        'NVDA': 82.8,
        'AVGO': 60.2,
        'LAES': 59.4,
        'GEV': 54.9,
    }
    
    # List of symbols to analyze (would come from your database)
    # This is a placeholder - in real usage, you'd load from your symbols database
    all_symbols = list(winners_data.keys())
    
    # Create analyzer
    analyzer = MarkerAnalyzer(db_path='symbols.db')
    
    # Run analysis
    analysis = analyzer.analyze_winners(
        symbols=all_symbols,
        performance_data=winners_data,
        threshold_pct=50  # Top 50% vs bottom 50%
    )
    
    # Print report
    analyzer.print_report(analysis)
    
    return analysis


if __name__ == '__main__':
    results = main()
    
    # Save results
    import json
    with open('marker_analysis_results.json', 'w') as f:
        # Convert numpy types to Python natives for JSON serialization
        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, set):
                return list(obj)
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        json.dump(results, f, default=convert, indent=2)
    
    print("\n✅ Results saved to marker_analysis_results.json")
