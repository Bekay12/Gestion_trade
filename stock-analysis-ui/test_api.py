#!/usr/bin/env python3
# ============================================================================
# TEST_API.PY - TESTER L'API REST
# Script pour tester tous les endpoints de l'API
# ============================================================================

import requests
import json
import time
from colorama import init, Fore, Style

# Initialize colorama for Windows
init()

# Configuration
BASE_URL = "http://localhost:5000"
API_URL = f"{BASE_URL}/api"

def print_success(message):
    print(f"{Fore.GREEN}✅ {message}{Style.RESET_ALL}")

def print_error(message):
    print(f"{Fore.RED}❌ {message}{Style.RESET_ALL}")

def print_info(message):
    print(f"{Fore.CYAN}ℹ️  {message}{Style.RESET_ALL}")

def print_warning(message):
    print(f"{Fore.YELLOW}⚠️  {message}{Style.RESET_ALL}")

def test_endpoint(method, url, description, data=None, expected_status=200):
    """Tester un endpoint et afficher le résultat"""
    try:
        print_info(f"Testing {method} {url}")
        
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, json=data, timeout=10)
        else:
            print_error(f"Méthode HTTP non supportée: {method}")
            return False
        
        # Vérifier status code
        if response.status_code == expected_status:
            print_success(f"{description} - Status {response.status_code}")
            
            # Afficher un aperçu de la réponse
            try:
                resp_json = response.json()
                if isinstance(resp_json, dict):
                    keys = list(resp_json.keys())[:5]
                    print(f"   Response keys: {keys}")
                elif isinstance(resp_json, list) and len(resp_json) > 0:
                    print(f"   Response: {len(resp_json)} items")
            except:
                print(f"   Response: {response.text[:100]}")
            
            return True
        else:
            print_error(f"{description} - Status {response.status_code} (expected {expected_status})")
            print(f"   Response: {response.text[:200]}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error(f"{description} - Connection error (is the server running?)")
        return False
    except requests.exceptions.Timeout:
        print_error(f"{description} - Timeout")
        return False
    except Exception as e:
        print_error(f"{description} - Error: {e}")
        return False

def main():
    print(f"""
    {Fore.CYAN}╔════════════════════════════════════════════╗
    ║  API TEST SUITE                            ║
    ║  Testing Stock Analysis API                ║
    ╚════════════════════════════════════════════╝{Style.RESET_ALL}
    """)
    
    results = {}
    
    # ====================================================================
    # 1. TEST HEALTH & STATUS ENDPOINTS
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("📊 HEALTH & STATUS ENDPOINTS")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    results['health'] = test_endpoint(
        "GET",
        f"{BASE_URL}/health",
        "Health check"
    )
    
    results['status'] = test_endpoint(
        "GET",
        f"{BASE_URL}/status",
        "Detailed status"
    )
    
    results['docs'] = test_endpoint(
        "GET",
        f"{API_URL}/docs",
        "API documentation"
    )
    
    # ====================================================================
    # 2. TEST SIGNALS ENDPOINTS
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("📈 SIGNALS ENDPOINTS")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    results['signals'] = test_endpoint(
        "GET",
        f"{API_URL}/signals?limit=5",
        "Get recent signals"
    )
    
    results['signals_filtered'] = test_endpoint(
        "GET",
        f"{API_URL}/signals?limit=10&min_reliability=50",
        "Get filtered signals (reliability >= 50)"
    )
    
    results['signal_aapl'] = test_endpoint(
        "GET",
        f"{API_URL}/signals/AAPL",
        "Get AAPL signals"
    )
    
    # ====================================================================
    # 3. TEST ANALYSIS ENDPOINTS
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("🔬 ANALYSIS ENDPOINTS")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    print_info("Ces tests nécessitent le téléchargement de données (peut prendre du temps)")
    print_warning("Les tests d'analyse sont optionnels pour validation rapide\n")
    
    # Test analyse (optionnel - peut être lent)
    # results['analyze'] = test_endpoint(
    #     "POST",
    #     f"{API_URL}/analyze",
    #     "Analyze AAPL",
    #     data={"symbol": "AAPL", "period": "3mo"}
    # )
    
    # results['analyze_batch'] = test_endpoint(
    #     "POST",
    #     f"{API_URL}/analyze-batch",
    #     "Analyze batch (3 symbols)",
    #     data={"symbols": ["AAPL", "MSFT", "GOOGL"], "period": "3mo"}
    # )
    
    # ====================================================================
    # 4. TEST STATS ENDPOINT
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("📊 STATISTICS ENDPOINTS")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    results['stats'] = test_endpoint(
        "GET",
        f"{API_URL}/stats",
        "Get statistics"
    )
    
    # ====================================================================
    # 5. TEST ERROR HANDLING
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("⚠️  ERROR HANDLING")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    results['404'] = test_endpoint(
        "GET",
        f"{API_URL}/nonexistent",
        "404 error handling",
        expected_status=404
    )
    
    results['bad_request'] = test_endpoint(
        "POST",
        f"{API_URL}/analyze",
        "400 bad request (missing symbol)",
        data={},
        expected_status=400
    )
    
    # ====================================================================
    # 6. RÉSUMÉ DES TESTS
    # ====================================================================
    print(f"\n{Fore.YELLOW}{'='*50}")
    print("📋 TEST SUMMARY")
    print(f"{'='*50}{Style.RESET_ALL}\n")
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    failed_tests = total_tests - passed_tests
    
    print(f"Total tests: {total_tests}")
    print_success(f"Passed: {passed_tests}")
    if failed_tests > 0:
        print_error(f"Failed: {failed_tests}")
    
    # Détails des échecs
    if failed_tests > 0:
        print(f"\n{Fore.RED}Failed tests:{Style.RESET_ALL}")
        for test, result in results.items():
            if not result:
                print(f"  ❌ {test}")
    
    # Statut final
    print(f"\n{'='*50}")
    if failed_tests == 0:
        print_success("🎉 ALL TESTS PASSED!")
        print_info("\nL'API est prête pour le déploiement!")
    else:
        print_warning(f"⚠️  {failed_tests} test(s) failed")
        print_info("\nVérifier les erreurs ci-dessus")
    
    print(f"{'='*50}\n")
    
    return failed_tests == 0

if __name__ == '__main__':
    print_info("Assurez-vous que l'API est démarrée:")
    print_info("  python src/api.py\n")
    
    input("Appuyez sur Entrée pour commencer les tests...")
    
    success = main()
    
    if not success:
        exit(1)
