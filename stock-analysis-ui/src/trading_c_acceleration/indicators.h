#ifndef INDICATORS_H
#define INDICATORS_H

#include <Python.h>
#include <numpy/arrayobject.h>
#include <math.h>
#include <string.h>

// Structure pour les résultats d'indicateurs
typedef struct {
    double *macd;
    double *signal_line;
    double *rsi;
    double *ema20;
    double *ema50;
    double *ema200;
    double *bb_upper;
    double *bb_lower;
    double *bb_percent;
    double adx;
    int length;
} TechnicalIndicators;

// Functions
void calculate_ema(double *prices, double *result, int length, int span);
void calculate_macd(double *prices, double *macd, double *signal_line, int length);
void calculate_rsi(double *prices, double *rsi, int length, int window);
void calculate_bollinger_bands(double *prices, double *upper, double *lower, double *percent, int length);
double calculate_adx(double *high, double *low, double *close, int length);
void calculate_all_indicators(double *prices, double *volumes, int length, TechnicalIndicators *indicators);
void free_indicators(TechnicalIndicators *indicators);

#endif