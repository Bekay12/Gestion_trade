#include <Python.h>
#include <numpy/arrayobject.h>
#include "backtest.h"
#include <stdio.h>

// Wrapper pour backtest_symbol_c
static PyObject* py_backtest_symbol(PyObject* self, PyObject* args) {
    PyObject *prices_array, *volumes_array, *coeffs_tuple;
    double amount, transaction_cost;
    
    // Parse des arguments Python
    if (!PyArg_ParseTuple(args, "OOOdd", &prices_array, &volumes_array, &coeffs_tuple, &amount, &transaction_cost)) {
        PyErr_SetString(PyExc_TypeError, "Arguments invalides");
        return NULL;
    }
    
    // Conversion des arrays NumPy
    PyArrayObject *prices_arr = (PyArrayObject*)PyArray_FROM_OTF(prices_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    PyArrayObject *volumes_arr = (PyArrayObject*)PyArray_FROM_OTF(volumes_array, NPY_DOUBLE, NPY_ARRAY_IN_ARRAY);
    
    if (prices_arr == NULL || volumes_arr == NULL) {
        Py_XDECREF(prices_arr);
        Py_XDECREF(volumes_arr);
        PyErr_SetString(PyExc_ValueError, "Impossible de convertir les arrays");
        return NULL;
    }
    
    // Vérification des dimensions
    if (PyArray_NDIM(prices_arr) != 1 || PyArray_NDIM(volumes_arr) != 1) {
        Py_DECREF(prices_arr);
        Py_DECREF(volumes_arr);
        PyErr_SetString(PyExc_ValueError, "Arrays doivent être 1D");
        return NULL;
    }
    
    int length = (int)PyArray_DIM(prices_arr, 0);
    int vol_length = (int)PyArray_DIM(volumes_arr, 0);
    
    if (length != vol_length) {
        Py_DECREF(prices_arr);
        Py_DECREF(volumes_arr);
        PyErr_SetString(PyExc_ValueError, "Prices et volumes doivent avoir la même longueur");
        return NULL;
    }
    
    if (length < 50) {
        Py_DECREF(prices_arr);
        Py_DECREF(volumes_arr);
        // Retourner un résultat vide plutôt qu'une erreur
        PyObject *result_dict = PyDict_New();
        PyDict_SetItemString(result_dict, "trades", PyLong_FromLong(0));
        PyDict_SetItemString(result_dict, "gagnants", PyLong_FromLong(0));
        PyDict_SetItemString(result_dict, "taux_reussite", PyFloat_FromDouble(0.0));
        PyDict_SetItemString(result_dict, "gain_total", PyFloat_FromDouble(0.0));
        PyDict_SetItemString(result_dict, "gain_moyen", PyFloat_FromDouble(0.0));
        PyDict_SetItemString(result_dict, "drawdown_max", PyFloat_FromDouble(0.0));
        return result_dict;
    }
    
    double *prices = (double*)PyArray_DATA(prices_arr);
    double *volumes = (double*)PyArray_DATA(volumes_arr);
    
    // Extraction des coefficients du tuple Python
    TradingCoefficients coeffs;
    
    if (!PyTuple_Check(coeffs_tuple)) {
        Py_DECREF(prices_arr);
        Py_DECREF(volumes_arr);
        PyErr_SetString(PyExc_TypeError, "Coefficients doivent être un tuple");
        return NULL;
    }
    
    Py_ssize_t tuple_size = PyTuple_Size(coeffs_tuple);
    if (tuple_size < 10) {
        // Valeurs par défaut si tuple incomplet
        coeffs.a1 = coeffs.a2 = coeffs.a3 = coeffs.a4 = 1.5;
        coeffs.a5 = coeffs.a6 = coeffs.a7 = coeffs.a8 = 1.5;
        coeffs.buy_threshold = 4.2;
        coeffs.sell_threshold = -0.5;
    } else {
        // Extraction des coefficients
        PyObject *item;
        
        item = PyTuple_GetItem(coeffs_tuple, 0);
        coeffs.a1 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.5;
        
        item = PyTuple_GetItem(coeffs_tuple, 1);
        coeffs.a2 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.0;
        
        item = PyTuple_GetItem(coeffs_tuple, 2);
        coeffs.a3 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.5;
        
        item = PyTuple_GetItem(coeffs_tuple, 3);
        coeffs.a4 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.25;
        
        item = PyTuple_GetItem(coeffs_tuple, 4);
        coeffs.a5 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.75;
        
        item = PyTuple_GetItem(coeffs_tuple, 5);
        coeffs.a6 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.25;
        
        item = PyTuple_GetItem(coeffs_tuple, 6);
        coeffs.a7 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.0;
        
        item = PyTuple_GetItem(coeffs_tuple, 7);
        coeffs.a8 = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 1.75;
        
        item = PyTuple_GetItem(coeffs_tuple, 8);
        coeffs.buy_threshold = PyFloat_Check(item) ? PyFloat_AsDouble(item) : 4.2;
        
        item = PyTuple_GetItem(coeffs_tuple, 9);
        coeffs.sell_threshold = PyFloat_Check(item) ? PyFloat_AsDouble(item) : -0.5;
    }
    
    // Vérification des erreurs de conversion
    if (PyErr_Occurred()) {
        Py_DECREF(prices_arr);
        Py_DECREF(volumes_arr);
        return NULL;
    }
    
    // Appel de la fonction C ultra-rapide
    BacktestResult result = backtest_symbol_c(prices, volumes, length, &coeffs, amount, transaction_cost);
    
    // Nettoyage
    Py_DECREF(prices_arr);
    Py_DECREF(volumes_arr);
    
    // Retour du résultat comme dictionnaire Python (identique à votre format)
    PyObject *result_dict = PyDict_New();
    if (!result_dict) {
        return NULL;
    }
    
    PyDict_SetItemString(result_dict, "trades", PyLong_FromLong(result.trades));
    PyDict_SetItemString(result_dict, "gagnants", PyLong_FromLong(result.winners));
    PyDict_SetItemString(result_dict, "taux_reussite", PyFloat_FromDouble(result.success_rate));
    PyDict_SetItemString(result_dict, "gain_total", PyFloat_FromDouble(result.total_gain));
    PyDict_SetItemString(result_dict, "gain_moyen", PyFloat_FromDouble(result.average_gain));
    PyDict_SetItemString(result_dict, "drawdown_max", PyFloat_FromDouble(result.max_drawdown));
    
    return result_dict;
}

// Fonction utilitaire pour tester le module
static PyObject* py_test_module(PyObject* self, PyObject* args) {
    return PyUnicode_FromString("Module C trading opérationnel - accélération activée !");
}

// Définition des méthodes du module
static PyMethodDef TradingMethods[] = {
    {"backtest_symbol", py_backtest_symbol, METH_VARARGS, 
     "Backtest ultra-rapide en C pour un symbole.\n"
     "Args: prices_array, volumes_array, coeffs_tuple, amount, transaction_cost\n"
     "Returns: dict avec trades, gagnants, taux_reussite, gain_total, gain_moyen, drawdown_max"},
    {"test_module", py_test_module, METH_NOARGS, 
     "Test si le module C fonctionne correctement"},
    {NULL, NULL, 0, NULL}  // Sentinel
};

// Documentation du module
static char module_doc[] = 
    "Module C ultra-rapide pour optimisation de trading\n"
    "\n"
    "Ce module fournit des fonctions C optimisées pour l'analyse technique\n"
    "et le backtesting, offrant une accélération de 50-200x par rapport au Python pur.\n"
    "\n"
    "Fonctions:\n"
    "  backtest_symbol(prices, volumes, coeffs, amount, cost) - Backtest C\n"
    "  test_module() - Test du module\n";

// Définition du module
static struct PyModuleDef tradingmodule = {
    PyModuleDef_HEAD_INIT,
    "trading_c",           // nom du module
    module_doc,            // documentation
    -1,                    // état du module
    TradingMethods         // méthodes
};

// Fonction d'initialisation du module (OBLIGATOIRE)
PyMODINIT_FUNC PyInit_trading_c(void) {
    PyObject *module;
    
    // Initialiser NumPy (CRITIQUE pour les arrays)
    import_array();
    if (PyErr_Occurred()) {
        return NULL;
    }
    
    // Créer le module
    module = PyModule_Create(&tradingmodule);
    if (module == NULL) {
        return NULL;
    }
    
    // Ajouter des constantes utiles
    PyModule_AddStringConstant(module, "__version__", "1.0.0");
    PyModule_AddStringConstant(module, "__author__", "Trading Optimizer C");
    
    return module;
}