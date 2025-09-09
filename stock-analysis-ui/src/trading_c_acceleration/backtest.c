#include "backtest.h"
#include <stdlib.h>

double get_trading_signal_score(TechnicalIndicators *indicators, TradingCoefficients *coeffs, 
                               double *prices, double *volumes, int length, int current_idx) {
    if (current_idx < 50 || current_idx >= length-1 || !indicators || !coeffs) return 0.0;
    
    // Variables locales (exactement comme votre logique Python)
    double last_close = prices[current_idx];
    double last_rsi = indicators->rsi[current_idx];
    double prev_rsi = current_idx > 0 ? indicators->rsi[current_idx-1] : last_rsi;
    double last_macd = indicators->macd[current_idx];
    double prev_macd = current_idx > 0 ? indicators->macd[current_idx-1] : last_macd;
    double last_signal = indicators->signal_line[current_idx];
    double prev_signal = current_idx > 0 ? indicators->signal_line[current_idx-1] : last_signal;
    double last_ema20 = indicators->ema20[current_idx];
    double last_ema50 = indicators->ema50[current_idx];
    double last_ema200 = indicators->ema200[current_idx];
    double last_bb_percent = indicators->bb_percent[current_idx];
    
    // Calcul volume moyen sur 30 jours (identique à votre Python)
    double volume_mean = 0.0;
    int vol_window = (current_idx >= 30) ? 30 : current_idx;
    if (vol_window > 0) {
        for (int i = current_idx - vol_window; i <= current_idx; i++) {
            volume_mean += volumes[i];
        }
        volume_mean /= vol_window;
    }
    
    // Calcul des variations (identique à votre Python)
    double delta_rsi = last_rsi - prev_rsi;
    double variation_30j = 0.0;
    if (current_idx >= 30 && prices[current_idx-30] != 0.0) {
        variation_30j = ((last_close - prices[current_idx-30]) / prices[current_idx-30]) * 100.0;
    }
    
    // Conditions logiques (EXACTEMENT comme dans votre Python)
    int is_macd_cross_up = (prev_macd < prev_signal) && (last_macd > last_signal);
    int is_macd_cross_down = (prev_macd > prev_signal) && (last_macd < last_signal);
    int is_volume_ok = volume_mean > 100000;
    int is_variation_ok = variation_30j > -20.0;
    int ema_structure_up = (last_close > last_ema20) && (last_ema20 > last_ema50) && (last_ema50 > last_ema200);
    int ema_structure_down = (last_close < last_ema20) && (last_ema20 < last_ema50) && (last_ema50 < last_ema200);
    int rsi_cross_up = (prev_rsi < 30.0) && (last_rsi >= 30.0);
    int rsi_cross_mid = (prev_rsi < 50.0) && (last_rsi >= 50.0);
    int rsi_cross_down = (prev_rsi > 65.0) && (last_rsi <= 65.0);
    int rsi_ok = (last_rsi < 75.0) && (last_rsi > 40.0);
    int strong_uptrend = last_close > last_ema20;
    int strong_downtrend = last_close < last_ema20;
    int adx_strong_trend = indicators->adx > 25.0;
    
    // Multiplicateurs (EXACTEMENT comme dans votre code)
    double m1 = adx_strong_trend ? 1.5 : 1.0;
    double m2 = 1.0, m3 = 1.0, m4 = 1.0;
    
    // Calcul volume ratio pour m3
    double current_volume = volumes[current_idx];
    if (volume_mean > 0) {
        double volume_ratio = current_volume / volume_mean;
        if (volume_ratio > 1.5) m3 = 1.5;
        else if (volume_ratio < 0.5) m3 = 0.7;
    }
    
    // Calcul du score (IDENTIQUE à votre logique Python)
    double score = 0.0;
    
    // RSI signaux haussiers
    if (rsi_cross_up) score += coeffs->a1;
    if (delta_rsi > 3.0) score += m3 * coeffs->a2;
    if (rsi_cross_mid) score += coeffs->a3;
    
    // RSI signaux baissiers
    if (rsi_cross_down) score -= coeffs->a1;
    if (delta_rsi < -3.0) score -= m3 * coeffs->a2;
    
    if (rsi_ok) score += coeffs->a4;
    else score -= coeffs->a4;
    
    // EMA structure
    if (ema_structure_up) score += m1 * coeffs->a5;
    if (ema_structure_down) score -= m1 * coeffs->a5;
    
    // MACD crossovers
    if (is_macd_cross_up) score += coeffs->a6;
    if (is_macd_cross_down) score -= coeffs->a6;
    
    // Volume
    if (is_volume_ok) score += m2 * coeffs->a6;
    else score -= m2 * coeffs->a6;
    
    // Performance passée
    if (is_variation_ok) score += coeffs->a7;
    else score -= coeffs->a7;
    
    // Conditions supplémentaires
    if (strong_uptrend) score += m2 * coeffs->a5;
    if (last_bb_percent < 0.4) score += m3 * coeffs->a4;
    if (strong_downtrend) score -= m2 * coeffs->a5;
    if (last_bb_percent > 0.6) score -= m3 * coeffs->a4;
    
    // Conditions d'achat/vente renforcées (comme dans votre Python)
    int buy_conditions = (is_macd_cross_up || ema_structure_up) && 
                        (rsi_cross_up || rsi_cross_mid) && 
                        (last_rsi < 65) && 
                        (last_bb_percent < 0.7) && 
                        (strong_uptrend || adx_strong_trend) && 
                        is_volume_ok && 
                        (is_variation_ok || current_idx < 30);
                        
    int sell_conditions = (is_macd_cross_down || ema_structure_down) && 
                         (rsi_cross_down || last_rsi > 70) && 
                         (last_rsi > 35) && 
                         (last_bb_percent > 0.3) && 
                         (strong_downtrend || adx_strong_trend) && 
                         is_volume_ok;
    
    if (buy_conditions) score += coeffs->a8;
    if (sell_conditions) score -= coeffs->a8;
    
    // Volatilité (comme dans votre Python)
    if (current_idx > 0 && prices[current_idx-1] != 0.0) {
        double volatility = fabs((prices[current_idx] - prices[current_idx-1]) / prices[current_idx-1]);
        if (volatility > 0.05) m4 = 0.75;
    }
    score *= m4;
    
    return score;
}

BacktestResult backtest_symbol_c(double *prices, double *volumes, int length, 
                                TradingCoefficients *coeffs, double amount, double transaction_cost) {
    BacktestResult result = {0, 0, 0.0, 0.0, 0.0, 0.0};
    
    if (length < 50 || !prices || !volumes || !coeffs) return result;
    
    // Calcul de tous les indicateurs une seule fois (optimisation majeure !)
    TechnicalIndicators indicators;
    calculate_all_indicators(prices, volumes, length, &indicators);
    
    // Vérification que les indicateurs ont été calculés
    if (!indicators.macd) {
        return result;
    }
    
    // Structures pour la simulation
    int position_open = 0;
    double entry_price = 0.0;
    double *gains = malloc(length * sizeof(double));
    int gain_count = 0;
    double *portfolio_values = malloc(length * sizeof(double));
    int portfolio_count = 0;
    
    if (!gains || !portfolio_values) {
        free(gains);
        free(portfolio_values);
        free_indicators(&indicators);
        return result;
    }
    
    portfolio_values[portfolio_count++] = amount;
    
    // Simulation du backtest (IDENTIQUE à votre logique)
    for (int i = 50; i < length-1; i++) {
        double score = get_trading_signal_score(&indicators, coeffs, prices, volumes, length, i);
        
        // Signal d'achat
        if (score >= coeffs->buy_threshold && !position_open) {
            position_open = 1;
            entry_price = prices[i];
        }
        // Signal de vente
        else if (score <= coeffs->sell_threshold && position_open) {
            double exit_price = prices[i];
            double return_pct = (exit_price - entry_price) / entry_price;
            double gain = amount * return_pct * (1.0 - 2.0 * transaction_cost);
            
            gains[gain_count++] = gain;
            result.total_gain += gain;
            result.trades++;
            
            if (gain > 0) result.winners++;
            
            if (portfolio_count < length) {
                portfolio_values[portfolio_count] = portfolio_values[portfolio_count-1] + gain;
                portfolio_count++;
            }
            position_open = 0;
        }
    }
    
    // Calcul des métriques finales (IDENTIQUE à votre Python)
    if (result.trades > 0) {
        result.success_rate = (double)result.winners / result.trades * 100.0;
        result.average_gain = result.total_gain / result.trades;
        
        // Calcul du drawdown maximum
        double max_value = portfolio_values[0];
        double max_drawdown = 0.0;
        for (int i = 1; i < portfolio_count; i++) {
            if (portfolio_values[i] > max_value) {
                max_value = portfolio_values[i];
            } else {
                double drawdown = 0.0;
                if (max_value > 0.0) {
                    drawdown = (max_value - portfolio_values[i]) / max_value * 100.0;
                }
                if (drawdown > max_drawdown) {
                    max_drawdown = drawdown;
                }
            }
        }
        result.max_drawdown = max_drawdown;
    }
    
    // Nettoyage mémoire
    free_indicators(&indicators);
    free(gains);
    free(portfolio_values);
    
    return result;
}