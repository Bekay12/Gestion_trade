"""
sector_normalizer.py - Normalisation des noms de secteurs

Converts various sector names from yfinance into standardized forms
to match database and optimization parameters.
"""

import logging

logger = logging.getLogger(__name__)

# Mapping de normalisations pour les secteurs
# Couvre les variations de yfinance, Yahoo, et autres sources
SECTOR_NORMALIZATION_MAP = {
    # Healthcare
    'Health Care': 'Healthcare',
    'Healthcare': 'Healthcare',
    'Medical': 'Healthcare',
    'Pharmaceuticals': 'Healthcare',
    'Biotechnology': 'Healthcare',
    'Medical Devices': 'Healthcare',
    
    # Financials / Financial Services
    'Financials': 'Financial Services',
    'Financial Services': 'Financial Services',
    'Banks': 'Financial Services',
    'Insurance': 'Financial Services',
    'Real Estate': 'Real Estate',
    'Real Estate Investment Trusts': 'Real Estate',
    
    # Technology
    'Information Technology': 'Technology',
    'Technology': 'Technology',
    'Software': 'Technology',
    'Hardware': 'Technology',
    'Semiconductors': 'Technology',
    'Semiconductor Equipment': 'Technology',
    'Internet Software and Services': 'Technology',
    'IT Services': 'Technology',
    
    # Industrials — garder le nom DB exact "Industrials"
    'Industrials': 'Industrials',
    'Industrial': 'Industrials',
    'Machinery': 'Industrials',
    'Building Products': 'Industrials',
    'Aerospace & Defense': 'Industrials',
    'Aerospace and Defense': 'Industrials',
    'Defense': 'Industrials',
    'Transportation': 'Industrials',
    'Marine Shipping': 'Industrials',
    
    # Consumer Cyclical (discrétionnaire) — NE PAS mélanger avec Defensive
    'Consumer Cyclical': 'Consumer Cyclical',
    'Consumer Discretionary': 'Consumer Cyclical',
    'Retail': 'Consumer Cyclical',
    'Apparel': 'Consumer Cyclical',
    'Restaurants': 'Consumer Cyclical',
    'Consumer Non-Cyclical': 'Consumer Defensive',  # Non-cyclical = defensive
    
    # Consumer Defensive (staples) — profil très différent de Cyclical
    'Consumer Defensive': 'Consumer Defensive',
    'Consumer Staples': 'Consumer Defensive',
    'Household & Personal Products': 'Consumer Defensive',
    
    # Energy
    'Energy': 'Energy',
    'Oil & Gas': 'Energy',
    'Oil and Gas': 'Energy',
    'Utilities': 'Utilities',
    'Electric Utilities': 'Utilities',
    'Water Utilities': 'Utilities',
    
    # Materials — garder le nom DB exact "Basic Materials"
    'Basic Materials': 'Basic Materials',
    'Materials': 'Basic Materials',
    'Metals & Mining': 'Basic Materials',
    'Metals and Mining': 'Basic Materials',
    'Chemicals': 'Basic Materials',
    'Paper & Forest Products': 'Basic Materials',
    'Paper and Forest Products': 'Basic Materials',
    'Diversified Metals & Mining': 'Basic Materials',
    
    # Communication Services — garder le nom DB exact
    'Communication Services': 'Communication Services',
    'Telecommunications': 'Communication Services',
    'Media & Entertainment': 'Communication Services',
    'Media and Entertainment': 'Communication Services',
    'Publishing': 'Communication Services',
    
    # Other / Unknown
    'Unknown': 'Unknown',
    'Inconnu': 'Unknown',
    '': 'Unknown',
    None: 'Unknown',
}


def normalize_sector(sector: str, fallback: str = 'Unknown') -> str:
    """
    Normalise un nom de secteur depuis yfinance vers la forme standardisée.
    
    Args:
        sector: Nom du secteur depuis yfinance ou autre source
        fallback: Valeur retournée si normalization échoue (défaut: 'Unknown')
    
    Returns:
        Secteur normalisé (ex: 'Healthcare', 'Technology', 'Financial Services')
    
    Exemples:
        >>> normalize_sector('Health Care')
        'Healthcare'
        >>> normalize_sector('Information Technology')
        'Technology'
        >>> normalize_sector('UNKNOWN_SECTOR')
        'Unknown'
    """
    if sector is None:
        return fallback
    
    sector_str = str(sector).strip()
    
    # Recherche directe dans la map
    if sector_str in SECTOR_NORMALIZATION_MAP:
        result = SECTOR_NORMALIZATION_MAP[sector_str]
        if result:
            return result
    
    # Case-insensitive
    sector_lower = sector_str.lower()
    for key, value in SECTOR_NORMALIZATION_MAP.items():
        if key and key.lower() == sector_lower:
            return value
    
    # Partial match en dernier recours (ex: "Tech" -> "Technology")
    for key, value in SECTOR_NORMALIZATION_MAP.items():
        if key and key.lower() in sector_lower and len(key) > 3:
            logger.debug(f"🔄 Partial match: '{sector}' -> '{key}' -> '{value}'")
            return value
    
    # Fallback si rien ne trouve
    logger.warning(f"⚠️ Secteur non reconnu: '{sector}', utilisation du fallback: '{fallback}'")
    return fallback


def normalize_and_validate(sector: str, valid_sectors: list = None) -> tuple:
    """
    Normalise un secteur ET vérifie qu'il existe dans une liste valide.
    
    Args:
        sector: Secteur à normaliser
        valid_sectors: Liste des secteurs valides (pour DB ou optimisation)
                      Si None, retourne juste le normalisé
    
    Returns:
        Tuple (normalized_sector, is_valid, original_normalized)
        
    Exemples:
        >>> valid = ['Healthcare', 'Technology', 'Financial Services']
        >>> normalize_and_validate('Health Care', valid)
        ('Healthcare', True, 'Healthcare')
        >>> normalize_and_validate('UNKNOWN', valid)
        ('Healthcare', False, 'Unknown')  # Fallback au premier valide
    """
    normalized = normalize_sector(sector)
    
    if valid_sectors is None:
        return (normalized, True, normalized)
    
    is_valid = normalized in valid_sectors
    
    if not is_valid and normalized != 'Unknown':
        # Le secteur normalisé ne figure pas dans la liste valide
        # Peut indiquer un problème de mapping ou de DB
        logger.warning(f"⚠️ Secteur normalisé '{normalized}' pas dans valid_sectors: {valid_sectors}")
    
    return (normalized, is_valid, normalized)


# Fonction de debug
def get_normalization_report(sector: str) -> dict:
    """Retourne un rapport détaillé de la normalisation pour debug."""
    normalized = normalize_sector(sector)
    return {
        'input': sector,
        'output': normalized,
        'found_direct': sector in SECTOR_NORMALIZATION_MAP,
        'found_case_insensitive': any(
            key and key.lower() == (sector or '').lower() 
            for key in SECTOR_NORMALIZATION_MAP.keys()
        ),
        'map_keys': list(SECTOR_NORMALIZATION_MAP.keys()),
    }


if __name__ == '__main__':
    # Tests
    test_cases = [
        'Health Care',
        'Information Technology',
        'Financials',
        'UNKNOWN',
        '',
        None,
        'Oil & Gas',
    ]
    
    print("🧪 Tests de normalisation:")
    for test in test_cases:
        result = normalize_sector(test)
        print(f"  '{test}' -> '{result}'")
