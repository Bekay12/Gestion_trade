"""
Marker-Based Filter Module
Integrates discovered winning stock markers into QSI trading signal system
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


class MarkerFilter:
    """Apply marker-based filters to stocks for trading signal gating"""
    
    def __init__(self, 
                 min_momentum_1y: float = 150.0,        # Must have 150%+ 1-year return
                 min_volatility_annual: float = 80.0,    # Must have 80%+ annual volatility
                 max_price_vs_ema20: float = 0.0,       # Price should be below 20-day EMA (pullback)
                 min_volume_ratio: float = 0.80,        # Volume ratio min (80%)
                 max_volume_ratio: float = 0.95,        # Volume ratio max (95%) - not spiking
                 min_momentum_3m: float = 0.0,          # Optional 3-month momentum
                 target_rsi_range: Tuple[float, float] = (30.0, 45.0),  # Target RSI range
    ):
        """Initialize marker filter with thresholds"""
        self.min_momentum_1y = min_momentum_1y
        self.min_volatility_annual = min_volatility_annual
        self.max_price_vs_ema20 = max_price_vs_ema20
        self.min_volume_ratio = min_volume_ratio
        self.max_volume_ratio = max_volume_ratio
        self.min_momentum_3m = min_momentum_3m
        self.target_rsi_range = target_rsi_range
    
    def evaluate_stock(self, symbol: str, analysis: Dict) -> Dict:
        """
        Evaluate if a stock matches winning stock markers.
        
        Args:
            symbol: Stock symbol
            analysis: Dict with 'momentum', 'volatility', 'trend', 'volume' keys
                     (from advanced_marker_analysis.py analyze_stock output)
        
        Returns:
            Dict with evaluation results:
            {
                'symbol': symbol,
                'passes_filter': bool,
                'score': float (0-100),  # How well it matches winning pattern
                'reasons': [list of pass/fail reasons],
                'marker_details': {individual marker results}
            }
        """
        
        reasons = []
        marker_scores = {}
        
        # Extract metrics
        momentum = analysis.get('momentum', {})
        volatility = analysis.get('volatility', {})
        trend = analysis.get('trend', {})
        volume = analysis.get('volume', {})
        
        # 1. Momentum check (CRITICAL)
        momentum_1y = momentum.get('momentum_1y', 0)
        momentum_1y_pass = momentum_1y >= self.min_momentum_1y
        marker_scores['momentum_1y'] = min(100, (momentum_1y / self.min_momentum_1y) * 50)
        
        if momentum_1y_pass:
            reasons.append(f"✓ 1Y Momentum: {momentum_1y:.1f}% (need {self.min_momentum_1y}%)")
        else:
            reasons.append(f"✗ 1Y Momentum: {momentum_1y:.1f}% (need {self.min_momentum_1y}%) [CRITICAL]")
        
        # 2. Volatility check (CRITICAL)
        vol_annual = volatility.get('volatility_annual', 0)
        vol_pass = vol_annual >= self.min_volatility_annual
        marker_scores['volatility'] = min(100, (vol_annual / self.min_volatility_annual) * 30)
        
        if vol_pass:
            reasons.append(f"✓ Annual Volatility: {vol_annual:.1f}% (need {self.min_volatility_annual}%)")
        else:
            reasons.append(f"✗ Annual Volatility: {vol_annual:.1f}% (need {self.min_volatility_annual}%) [CRITICAL]")
        
        # 3. Price vs EMA20 check (CRITICAL - pullback indicator)
        price_vs_ema20 = trend.get('price_vs_ema20', 0)
        pullback_pass = price_vs_ema20 <= self.max_price_vs_ema20
        
        # Scoring: more negative (deeper pullback) = better
        pullback_score = 0
        if price_vs_ema20 < self.max_price_vs_ema20:
            pullback_score = min(20, abs(price_vs_ema20))  # Deeper pullback = higher score
        marker_scores['pullback'] = pullback_score
        
        if pullback_pass:
            reasons.append(f"✓ Price vs EMA20: {price_vs_ema20:.2f}% (should be <{self.max_price_vs_ema20}%) [Pullback detected]")
        else:
            reasons.append(f"✗ Price vs EMA20: {price_vs_ema20:.2f}% (should be <{self.max_price_vs_ema20}%) [CRITICAL]")
        
        # 4. Momentum 3M (HIGH - confirms trend)
        momentum_3m = momentum.get('momentum_3m', 0)
        momentum_3m_pass = momentum_3m >= self.min_momentum_3m
        marker_scores['momentum_3m'] = max(0, min(10, momentum_3m / 5))  # Small boost
        
        if momentum_3m_pass:
            reasons.append(f"✓ 3M Momentum: {momentum_3m:.1f}% (confirms positive trend)")
        else:
            reasons.append(f"~ 3M Momentum: {momentum_3m:.1f}% (weak but OK)")
        
        # 5. Volume ratio (MEDIUM - entry accumulation)
        vol_ratio = volume.get('volume_ratio_20_200', 0.9)
        volume_pass = self.min_volume_ratio <= vol_ratio <= self.max_volume_ratio
        
        # Score: closer to middle of range = better
        target_mid = (self.min_volume_ratio + self.max_volume_ratio) / 2
        vol_dist = abs(vol_ratio - target_mid)
        marker_scores['volume'] = max(0, 10 - (vol_dist * 50))
        
        if volume_pass:
            reasons.append(f"✓ Volume Ratio: {vol_ratio:.2f} (optimal {self.min_volume_ratio:.2f}-{self.max_volume_ratio:.2f})")
        else:
            if vol_ratio < self.min_volume_ratio:
                reasons.append(f"~ Volume Ratio: {vol_ratio:.2f} (low, possible early entry)")
            else:
                reasons.append(f"~ Volume Ratio: {vol_ratio:.2f} (high, possible breakout start)")
        
        # 6. RSI check (MEDIUM - confirms not overbought)
        rsi = momentum.get('rsi_14', 50)
        rsi_optimal = self.target_rsi_range[0] <= rsi <= self.target_rsi_range[1]
        
        # Score RSI
        if rsi_optimal:
            marker_scores['rsi'] = 10
            reasons.append(f"✓ RSI-14: {rsi:.1f} (optimal {self.target_rsi_range[0]}-{self.target_rsi_range[1]})")
        elif rsi < self.target_rsi_range[0]:
            marker_scores['rsi'] = 8
            reasons.append(f"~ RSI-14: {rsi:.1f} (oversold - strong bounce potential)")
        else:
            marker_scores['rsi'] = 5
            reasons.append(f"~ RSI-14: {rsi:.1f} (overbought - risky entry)")
        
        # 7. Bollinger Bands position (MEDIUM - volatility at lower band)
        bb_pos = volatility.get('bb_position', 0.5)
        bb_optimal = bb_pos < 0.4  # Lower half of bands
        
        if bb_optimal:
            marker_scores['bb'] = 10
            reasons.append(f"✓ BB Position: {bb_pos:.2f} (near lower band - expansion potential)")
        elif bb_pos < 0.6:
            marker_scores['bb'] = 5
            reasons.append(f"~ BB Position: {bb_pos:.2f} (middle - neutral)")
        else:
            marker_scores['bb'] = 2
            reasons.append(f"~ BB Position: {bb_pos:.2f} (near upper band - caution)")
        
        # Calculate overall score
        # Critical filters: must all pass for trading
        critical_pass = momentum_1y_pass and vol_pass and pullback_pass
        
        # Calculate weighted score from markers
        total_score = sum(marker_scores.values())
        normalized_score = min(100, total_score)
        
        return {
            'symbol': symbol,
            'passes_filter': critical_pass,
            'score': normalized_score,
            'critical_pass': critical_pass,
            'reasons': reasons,
            'marker_details': {
                'momentum_1y': momentum_1y,
                'volatility_annual': vol_annual,
                'price_vs_ema20': price_vs_ema20,
                'momentum_3m': momentum_3m,
                'volume_ratio': vol_ratio,
                'rsi': rsi,
                'bb_position': bb_pos,
            },
            'marker_scores': marker_scores,
        }
    
    def batch_evaluate(self, symbols_analyses: Dict[str, Dict]) -> pd.DataFrame:
        """
        Evaluate multiple stocks and return as DataFrame.
        
        Args:
            symbols_analyses: Dict mapping symbol -> analysis dict
        
        Returns:
            DataFrame with columns: symbol, passes_filter, score, and reason
        """
        
        results = []
        for symbol, analysis in symbols_analyses.items():
            result = self.evaluate_stock(symbol, analysis)
            results.append({
                'symbol': symbol,
                'passes_filter': result['passes_filter'],
                'score': result['score'],
                'reason_summary': "; ".join(result['reasons'][:3]),  # First 3 reasons
                **result['marker_details'],
            })
        
        return pd.DataFrame(results).sort_values('score', ascending=False)


def create_marker_filter_for_qsi(enable_markers: bool = True, **kwargs) -> Optional[MarkerFilter]:
    """
    Factory function to create a MarkerFilter configured for QSI integration.
    
    Args:
        enable_markers: If False, returns None (no filtering)
        **kwargs: Override default thresholds
    
    Returns:
        MarkerFilter instance or None
    """
    
    if not enable_markers:
        return None
    
    return MarkerFilter(**kwargs)


def apply_marker_gate_to_signal(signal: Dict, symbol: str, analysis: Dict, 
                                marker_filter: Optional[MarkerFilter] = None) -> Dict:
    """
    Apply marker-based gating to a trading signal.
    
    If markers are enabled and signal doesn't meet winning stock criteria,
    suppress or downgrade the signal.
    
    Args:
        signal: Original trading signal dict
        symbol: Stock symbol
        analysis: Stock analysis dict (from advanced_marker_analysis)
        marker_filter: MarkerFilter instance or None
    
    Returns:
        Modified signal dict (may suppress if marker criteria not met)
    """
    
    if marker_filter is None:
        return signal  # No filtering
    
    evaluation = marker_filter.evaluate_stock(symbol, analysis)
    
    if not evaluation['critical_pass']:
        # Suppress signal if critical markers not met
        signal['marker_gated'] = True
        signal['gate_reason'] = "; ".join([r for r in evaluation['reasons'] if 'CRITICAL' in r])
        
        # Can optionally downgrade signal strength
        if 'strength' in signal:
            signal['strength'] *= 0.5  # Halve signal strength
    else:
        signal['marker_boost'] = evaluation['score'] / 100  # Boost signal by confidence
    
    return signal


if __name__ == '__main__':
    # Example usage
    print("Marker Filter Module - Ready to integrate with QSI\n")
    
    # Show default thresholds
    mf = MarkerFilter()
    print("Default Winning Stock Marker Thresholds:")
    print(f"  - 1-Year Momentum: {mf.min_momentum_1y}%+ ")
    print(f"  - Annual Volatility: {mf.min_volatility_annual}%+")
    print(f"  - Price vs 20-day EMA: < {mf.max_price_vs_ema20}%")
    print(f"  - Volume Ratio Range: {mf.min_volume_ratio:.2f} to {mf.max_volume_ratio:.2f}")
    print(f"  - RSI Target: {mf.target_rsi_range[0]:.0f}-{mf.target_rsi_range[1]:.0f}")
    print("\nReady to be integrated into get_trading_signal() in qsi.py")
