#include "indicators.h"
#include <stdlib.h>

void calculate_ema(double *prices, double *result, int length, int span) {
    if (length <= 0 || span <= 0) return;
    
    double alpha = 2.0 / (span + 1.0);
    result[0] = prices[0];
    
    for (int i = 1; i < length; i++) {
        result[i] = alpha * prices[i] + (1.0 - alpha) * result[i-1];
    }
}

void calculate_macd(double *prices, double *macd, double *signal_line, int length) {
    if (length < 26) return;
    
    double *ema12 = malloc(length * sizeof(double));
    double *ema26 = malloc(length * sizeof(double));
    
    if (!ema12 || !ema26) {
        free(ema12);
        free(ema26);
        return;
    }
    
    calculate_ema(prices, ema12, length, 12);
    calculate_ema(prices, ema26, length, 26);
    
    // Calcul MACD
    for (int i = 0; i < length; i++) {
        macd[i] = ema12[i] - ema26[i];
    }
    
    // Signal line (EMA9 du MACD)
    calculate_ema(macd, signal_line, length, 9);
    
    free(ema12);
    free(ema26);
}

void calculate_rsi(double *prices, double *rsi, int length, int window) {
    if (length <= window || window <= 0) return;
    
    double *gains = malloc((length-1) * sizeof(double));
    double *losses = malloc((length-1) * sizeof(double));
    
    if (!gains || !losses) {
        free(gains);
        free(losses);
        return;
    }
    
    // Calcul des gains/pertes
    for (int i = 1; i < length; i++) {
        double change = prices[i] - prices[i-1];
        gains[i-1] = change > 0 ? change : 0.0;
        losses[i-1] = change < 0 ? -change : 0.0;
    }
    
    // Première moyenne
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = 0; i < window; i++) {
        avg_gain += gains[i];
        avg_loss += losses[i];
    }
    avg_gain /= window;
    avg_loss /= window;
    
    // Calcul RSI avec moyenne mobile exponentielle
    double alpha = 1.0 / window;
    for (int i = window; i < length-1; i++) {
        avg_gain = alpha * gains[i] + (1.0 - alpha) * avg_gain;
        avg_loss = alpha * losses[i] + (1.0 - alpha) * avg_loss;
        
        if (avg_loss == 0.0) {
            rsi[i+1] = 100.0;
        } else {
            double rs = avg_gain / avg_loss;
            rsi[i+1] = 100.0 - (100.0 / (1.0 + rs));
        }
    }
    
    // Valeurs par défaut pour le début
    for (int i = 0; i <= window; i++) {
        rsi[i] = 50.0;
    }
    
    free(gains);
    free(losses);
}

void calculate_bollinger_bands(double *prices, double *upper, double *lower, double *percent, int length) {
    int window = 20;
    double std_dev = 2.0;
    
    for (int i = window-1; i < length; i++) {
        // Calcul de la moyenne mobile
        double sma = 0.0;
        for (int j = i - window + 1; j <= i; j++) {
            sma += prices[j];
        }
        sma /= window;
        
        // Calcul de l'écart-type
        double variance = 0.0;
        for (int j = i - window + 1; j <= i; j++) {
            double diff = prices[j] - sma;
            variance += diff * diff;
        }
        variance /= window;
        double std = sqrt(variance);
        
        upper[i] = sma + (std_dev * std);
        lower[i] = sma - (std_dev * std);
        
        // Bollinger %B
        double band_width = upper[i] - lower[i];
        if (band_width != 0.0) {
            percent[i] = (prices[i] - lower[i]) / band_width;
        } else {
            percent[i] = 0.5;
        }
    }
    
    // Valeurs par défaut pour le début
    for (int i = 0; i < window-1; i++) {
        upper[i] = prices[i];
        lower[i] = prices[i];
        percent[i] = 0.5;
    }
}

double calculate_adx(double *high, double *low, double *close, int length) {
    if (length < 14) return 0.0;
    
    // Simplification: calcule la volatilité récente comme proxy pour ADX
    double volatility = 0.0;
    int start = length > 14 ? length - 14 : 1;
    
    for (int i = start; i < length-1; i++) {
        double change = fabs(close[i+1] - close[i]);
        if (close[i] != 0.0) {
            volatility += change / close[i];
        }
    }
    volatility /= (length - start - 1);
    
    // Facteur pour obtenir des valeurs ADX-like (0-100)
    return volatility * 5000.0 > 100.0 ? 100.0 : volatility * 5000.0;
}

void calculate_all_indicators(double *prices, double *volumes, int length, TechnicalIndicators *indicators) {
    if (!indicators || length <= 0) return;
    
    indicators->length = length;
    
    // Allocation mémoire
    indicators->macd = malloc(length * sizeof(double));
    indicators->signal_line = malloc(length * sizeof(double));
    indicators->rsi = malloc(length * sizeof(double));
    indicators->ema20 = malloc(length * sizeof(double));
    indicators->ema50 = malloc(length * sizeof(double));
    indicators->ema200 = malloc(length * sizeof(double));
    indicators->bb_upper = malloc(length * sizeof(double));
    indicators->bb_lower = malloc(length * sizeof(double));
    indicators->bb_percent = malloc(length * sizeof(double));
    
    // Vérification des allocations
    if (!indicators->macd || !indicators->signal_line || !indicators->rsi ||
        !indicators->ema20 || !indicators->ema50 || !indicators->ema200 ||
        !indicators->bb_upper || !indicators->bb_lower || !indicators->bb_percent) {
        free_indicators(indicators);
        return;
    }
    
    // Initialisation avec des valeurs par défaut
    for (int i = 0; i < length; i++) {
        indicators->macd[i] = 0.0;
        indicators->signal_line[i] = 0.0;
        indicators->rsi[i] = 50.0;
        indicators->ema20[i] = prices[i];
        indicators->ema50[i] = prices[i];
        indicators->ema200[i] = prices[i];
        indicators->bb_upper[i] = prices[i];
        indicators->bb_lower[i] = prices[i];
        indicators->bb_percent[i] = 0.5;
    }
    
    // Calculs des indicateurs
    calculate_ema(prices, indicators->ema20, length, 20);
    calculate_ema(prices, indicators->ema50, length, 50);
    calculate_ema(prices, indicators->ema200, length, 200);
    calculate_macd(prices, indicators->macd, indicators->signal_line, length);
    calculate_rsi(prices, indicators->rsi, length, 17); // RSI-17 comme dans votre code
    calculate_bollinger_bands(prices, indicators->bb_upper, indicators->bb_lower, indicators->bb_percent, length);
    indicators->adx = calculate_adx(prices, prices, prices, length); // Utilise close comme proxy
}

void free_indicators(TechnicalIndicators *indicators) {
    if (!indicators) return;
    
    if (indicators->macd) { free(indicators->macd); indicators->macd = NULL; }
    if (indicators->signal_line) { free(indicators->signal_line); indicators->signal_line = NULL; }
    if (indicators->rsi) { free(indicators->rsi); indicators->rsi = NULL; }
    if (indicators->ema20) { free(indicators->ema20); indicators->ema20 = NULL; }
    if (indicators->ema50) { free(indicators->ema50); indicators->ema50 = NULL; }
    if (indicators->ema200) { free(indicators->ema200); indicators->ema200 = NULL; }
    if (indicators->bb_upper) { free(indicators->bb_upper); indicators->bb_upper = NULL; }
    if (indicators->bb_lower) { free(indicators->bb_lower); indicators->bb_lower = NULL; }
    if (indicators->bb_percent) { free(indicators->bb_percent); indicators->bb_percent = NULL; }
}