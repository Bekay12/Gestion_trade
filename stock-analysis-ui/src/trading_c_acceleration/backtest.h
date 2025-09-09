#ifndef BACKTEST_H
#define BACKTEST_H

#include <Python.h>
#include "indicators.h"

// Structure pour les résultats de backtest
typedef struct {
    int trades;
    int winners;
    double success_rate;
    double total_gain;
    double average_gain;
    double max_drawdown;
} BacktestResult;

// Structure pour les coefficients de trading
typedef struct {
    double a1, a2, a3, a4, a5, a6, a7, a8;
    double buy_threshold;
    double sell_threshold;
} TradingCoefficients;

// Functions
double get_trading_signal_score(TechnicalIndicators *indicators, TradingCoefficients *coeffs, 
                               double *prices, double *volumes, int length, int current_idx);
BacktestResult backtest_symbol_c(double *prices, double *volumes, int length, 
                                TradingCoefficients *coeffs, double amount, double transaction_cost);

#endif