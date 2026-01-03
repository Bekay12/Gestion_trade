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

// Structure pour les coefficients de trading (étendue avec price/fundamentals features)
typedef struct {
    // 8 coefficients de base
    double a1, a2, a3, a4, a5, a6, a7, a8;
    double buy_threshold;
    double sell_threshold;
    
    // 🚀 Price features (a9, a10)
    int use_price_slope;      // 0 ou 1
    int use_price_acc;        // 0 ou 1
    double a_price_slope;     // a9
    double a_price_acc;       // a10
    double th_price_slope;    // seuil price slope
    double th_price_acc;      // seuil price acc
    
    // 🚀 Fundamentals features (a11-a15)
    int use_fundamentals;     // 0 ou 1
    double a_rev_growth;      // a11
    double a_eps_growth;      // a12
    double a_roe;             // a13
    double a_fcf_yield;       // a14
    double a_de_ratio;        // a15
    double th_rev_growth;
    double th_eps_growth;
    double th_roe;
    double th_fcf_yield;
    double th_de_ratio;
    
    // Métriques fondamentales (passées depuis Python)
    double fund_rev_growth;
    double fund_eps_growth;
    double fund_roe;
    double fund_fcf_yield;
    double fund_de_ratio;
} TradingCoefficients;

// Functions
double get_trading_signal_score(TechnicalIndicators *indicators, TradingCoefficients *coeffs, 
                               double *prices, double *volumes, int length, int current_idx);
BacktestResult backtest_symbol_c(double *prices, double *volumes, int length, 
                                TradingCoefficients *coeffs, double amount, double transaction_cost);

#endif