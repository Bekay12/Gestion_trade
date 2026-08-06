# optimisateur_hybride_fixed.py
# Version optimisée avec limitation des décimales pour réduire l'espace de recherche

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import random
import sys
from pathlib import Path
_trading_accel_path = Path(__file__).parent / "trading_c_acceleration"
if _trading_accel_path.exists():
    _racine_src = str(_trading_accel_path.parent)
    if _racine_src not in sys.path:
        sys.path.insert(0, _racine_src)
from qsi import download_stock_data, load_symbols_from_txt, extract_best_parameters
# backtest_signals_c_extended n'est plus importe : il n'a aucun parametre de
# seuils, donc tout appel depuis ce module mesurerait des coefficients seuls.
from trading_c_acceleration.qsi_optimized import backtest_signals, backtest_signals_with_events
from core import optim_params as params
from core import optim_budget
from tqdm import tqdm
# import yfinance as yf  # Import paresseux - chargé seulement si nécessaire
from collections import deque, namedtuple
from core.cache import _BoundedCache, TA_CACHE
from scipy.optimize import differential_evolution
from scipy.stats import qmc
from typing import Dict, List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Metriques d'UNE configuration. Elles doivent voyager avec leur vecteur :
# la sauvegarde lisait auparavant `meilleur_trades`, qui appartenait a la
# meilleure configuration jamais vue, pas a celle qui etait ecrite en base.
Mesure = namedtuple('Mesure', 'score gain_moyen trades gagnants')

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except Exception:
    cp = None
    CUPY_AVAILABLE = False

# 🚀 PERFORMANCE: Détection automatique du nombre de workers
MAX_WORKERS = min(os.cpu_count() or 4, 12)  # Max 12 pour éviter surcharge

# Cout par trade, en MONTANT absolu et non en pourcentage : le moteur calcule
# profit = (close - entry) / entry * montant - transaction_cost. Sur une
# position de 50 $, 1.0 vaut 2 %, ce qui est realiste pour un petit ordre au
# detail ; les 0.02 que passait le CLI valaient deux centimes et flattaient
# mecaniquement les configurations qui multiplient les trades.
COUT_TRANSACTION_PAR_TRADE = 1.0

# Budget de reference, unique. Le menu et le lancement en derivent tous deux via
# optim_budget.budget_effectif(), qui affichaient auparavant 3 500 contre 30 000.
BUDGET_BASE = 30000

# Colonnes d'extras a creer si la base precede leur introduction.
EXTRAS_COLONNES_MIGRATION = tuple(
    spec for spec in (params.EXTRAS_PRIX + params.EXTRAS_FONDAMENTAUX)
    if spec.cle not in params.DRAPEAUX
)


class ComputeBackend:
    """Backend de calcul abstrait avec fallback CPU automatique."""

    def __init__(self, name: str):
        self.name = name

    @property
    def is_gpu(self) -> bool:
        return False

    def to_numpy(self, values) -> np.ndarray:
        if isinstance(values, pd.Series):
            arr = values.values
        else:
            arr = values
        return np.asarray(arr, dtype=np.float64)

    def precalculate_symbol_features(self, close, volume) -> Dict[str, pd.Series]:
        """Pré-calcule des features légères pour limiter les recalculs pandas."""
        close_s = close if isinstance(close, pd.Series) else pd.Series(self.to_numpy(close))
        volume_s = volume if isinstance(volume, pd.Series) else pd.Series(self.to_numpy(volume))
        return {
            'PRECALC_EMA20': close_s.ewm(span=20, adjust=False).mean(),
            'PRECALC_EMA50': close_s.ewm(span=50, adjust=False).mean(),
            'PRECALC_VOL30': volume_s.rolling(window=30).mean(),
        }


class CupyBackend(ComputeBackend):
    """Backend GPU CuPy (première version) avec fallback CPU interne."""

    def __init__(self):
        super().__init__('gpu-cupy')
        self._enabled = False
        if not CUPY_AVAILABLE:
            return
        try:
            _ = cp.cuda.runtime.getDeviceCount()
            self._enabled = True
        except Exception:
            self._enabled = False

    @property
    def is_gpu(self) -> bool:
        return bool(self._enabled)

    def to_numpy(self, values) -> np.ndarray:
        if isinstance(values, pd.Series):
            arr = values.values
        else:
            arr = values
        try:
            return cp.asnumpy(cp.asarray(arr, dtype=cp.float64))
        except Exception:
            return np.asarray(arr, dtype=np.float64)

    def precalculate_symbol_features(self, close, volume) -> Dict[str, pd.Series]:
        # EMA reste en pandas (stabilité/résultat identique), volume rolling est accéléré via CuPy.
        close_s = close if isinstance(close, pd.Series) else pd.Series(np.asarray(close, dtype=np.float64))
        volume_idx = volume.index if isinstance(volume, pd.Series) else close_s.index
        volume_np = np.asarray(volume.values if isinstance(volume, pd.Series) else volume, dtype=np.float64)

        ema20 = close_s.ewm(span=20, adjust=False).mean()
        ema50 = close_s.ewm(span=50, adjust=False).mean()

        try:
            vol_gpu = cp.asarray(volume_np, dtype=cp.float64)
            kernel = cp.ones(30, dtype=cp.float64) / 30.0
            vol30_valid = cp.convolve(vol_gpu, kernel, mode='valid')
            vol30_np = np.full(volume_np.shape, np.nan, dtype=np.float64)
            vol30_np[29:] = cp.asnumpy(vol30_valid)
            vol30 = pd.Series(vol30_np, index=volume_idx)
        except Exception:
            vol30 = pd.Series(volume_np, index=volume_idx).rolling(window=30).mean()

        return {
            'PRECALC_EMA20': ema20,
            'PRECALC_EMA50': ema50,
            'PRECALC_VOL30': vol30,
        }


def resolve_compute_backend(backend_preference: str = 'auto') -> ComputeBackend:
    pref = (backend_preference or 'auto').strip().lower()
    if pref in ('gpu', 'cupy', 'gpu-cupy'):
        gpu = CupyBackend()
        if gpu.is_gpu:
            print("⚡ Backend calcul: GPU (CuPy)")
            return gpu
        print("ℹ️ Backend GPU demandé mais indisponible, fallback CPU")
        return ComputeBackend('cpu')

    if pref == 'auto':
        gpu = CupyBackend()
        if gpu.is_gpu:
            print("⚡ Backend calcul: GPU (CuPy, auto)")
            return gpu
        print("🖥️ Backend calcul: CPU (auto)")
        return ComputeBackend('cpu')

    return ComputeBackend('cpu')

# Import du gestionnaire de symboles SQLite
try:
    from symbol_manager import (
        init_symbols_table, sync_txt_to_sqlite, get_symbols_by_list_type, get_all_sectors,
        get_all_cap_ranges, get_symbols_by_sector_and_cap, get_symbol_count,
        get_cleaned_group_cache, save_cleaned_group_cache,
        get_popular_symbols_by_sector, get_all_popular_symbols,
        get_symbols_by_sector  # Pour répartition proportionnelle
    )
    SYMBOL_MANAGER_AVAILABLE = True
except ImportError:
    print("⚠️ symbol_manager non disponible, utilisation de la méthode classique")
    SYMBOL_MANAGER_AVAILABLE = False

# 🔧 OPTIMISATION: Caching des secteurs (mémoire + disque)
# Chemin absolu venant de config, et creation du dossier a l'ECRITURE. Le
# chemin relatif et le mkdir a l'import fabriquaient un dossier cache_data/ la
# ou se trouvait le repertoire courant.
from config import CACHE_DIR

SECTOR_CACHE_FILE = Path(CACHE_DIR) / "sector_cache.json"
SECTOR_TTL_DAYS = 30
SECTOR_TTL_UNKNOWN_DAYS = 7

def _load_sector_cache():
    try:
        if SECTOR_CACHE_FILE.exists():
            with open(SECTOR_CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        print(f"⚠️ Impossible de charger le cache secteurs: {e}")
    return {}

def _save_sector_cache(cache):
    SECTOR_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(SECTOR_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f)
    except Exception as e:
        print(f"⚠️ Impossible d'écrire le cache secteurs: {e}")

def _is_sector_expired(entry):
    try:
        ts = entry.get("ts")
        if not ts:
            return True
        dt = datetime.fromisoformat(ts)
        ttl_days = SECTOR_TTL_UNKNOWN_DAYS if entry.get("sector") == "ℹ️Inconnu!!" else SECTOR_TTL_DAYS
        return (datetime.utcnow() - dt).days >= ttl_days
    except Exception:
        return True

_sector_cache = _load_sector_cache()

def get_sector(symbol, use_cache=True):
    """Récupère le secteur d'une action avec cache mémoire + disque."""
    if use_cache:
        entry = _sector_cache.get(symbol)
        if entry and not _is_sector_expired(entry):
            return entry.get("sector", "ℹ️Inconnu!!")

    try:
        import yfinance as yf  # Import paresseux - chargé seulement si nécessaire
        ticker = yf.Ticker(symbol)
        info = ticker.info
        sector = info.get('sector', 'ℹ️Inconnu!!')
        print(f"📋 {symbol}: Secteur = {sector}")
        _sector_cache[symbol] = {"sector": sector, "ts": datetime.utcnow().isoformat()}
        _save_sector_cache(_sector_cache)
        return sector
    except Exception as e:
        print(f"⚠️ Erreur pour {symbol}: {e}")
        return 'ℹ️Inconnu!!'


def classify_cap_range(symbol: str) -> str:
    """Classe la capitalisation en 4 catégories (Small, Mid, Large, Mega) ou Unknown."""
    try:
        import yfinance as yf  # Import paresseux
        ticker = yf.Ticker(symbol)
        market_cap = ticker.info.get('marketCap')
        if market_cap is None:
            return 'Unknown'

        market_cap_b = market_cap / 1e9
        if market_cap_b < 2:
            return 'Small'
        if market_cap_b < 10:
            return 'Mid'
        if market_cap_b < 200:
            return 'Large'
        return 'Mega'
    except Exception:
        return 'Unknown'

# ----------------------------
# Nettoyage paresseux des groupes secteur × cap_range (complément + réduction)
# ----------------------------
def clean_sector_cap_groups(sector_cap_ranges: Dict[str, Dict[str, List[str]]],
                            ttl_days: int = 10,
                            min_symbols: int = 7,
                            min_total: int = 400,
                            max_symbols: int = 12,
                            fixed_ratio: float = 0.6) -> Dict[str, Dict[str, List[str]]]:
    """Sélectionne les symboles pour l'optimisation avec répartition proportionnelle.

    Garanties :
    1. Tous les symboles de mes_symbols (personal) sont TOUJOURS inclus.
    2. Le total de symboles sélectionnés est >= min_total (défaut 400).
    3. La répartition par secteur est proportionnelle au dataset complet.
    4. Au sein de chaque secteur, le complément est pris aléatoirement parmi
       les populaires du même secteur.
    """
    import math

    # ── Étape 0 : Recenser le dataset complet par secteur ──
    dataset_per_sector: Dict[str, List[str]] = {}
    for sector, buckets in sector_cap_ranges.items():
        all_syms_sector = set()
        for cap, syms in buckets.items():
            all_syms_sector.update(syms)
        if all_syms_sector:
            dataset_per_sector[sector] = list(all_syms_sector)

    dataset_total = sum(len(s) for s in dataset_per_sector.values())
    if dataset_total == 0:
        return sector_cap_ranges

    # ── Étape 1 : Identifier TOUS les personal symbols ──
    all_personal: Dict[str, set] = {}  # sector -> set of personal
    all_personal_flat = set()
    for sector in dataset_per_sector:
        try:
            ps = get_symbols_by_sector(sector, list_type='personal', active_only=True)
            ps = [s.strip().upper() for s in ps if isinstance(s, str) and s.strip()]
        except Exception:
            ps = []
        if ps:
            all_personal[sector] = set(ps)
            all_personal_flat.update(ps)
    
    # Also include personal symbols not matching any existing sector
    try:
        all_mes = get_symbols_by_list_type('personal', active_only=True)
        all_mes = [s.strip().upper() for s in all_mes if isinstance(s, str) and s.strip()]
        orphan_personal = set(all_mes) - all_personal_flat
        if orphan_personal:
            # Assign orphans to 'Unknown' sector  
            all_personal.setdefault('Unknown', set()).update(orphan_personal)
            all_personal_flat.update(orphan_personal)
    except Exception:
        pass

    total_personal = len(all_personal_flat)
    print(f"      🔒 Personal (mes_symbols) forcés: {total_personal} symboles")

    # ── Étape 2 : Calculer la répartition proportionnelle ──
    remaining_budget = max(0, min_total - total_personal)
    
    # Proportional allocation per sector (based on dataset size minus personal already included)
    sector_allocation: Dict[str, int] = {}
    for sector, syms in dataset_per_sector.items():
        proportion = len(syms) / dataset_total
        personal_in_sector = len(all_personal.get(sector, set()))
        # Proportional share of the remaining budget
        alloc = max(0, math.floor(proportion * remaining_budget))
        sector_allocation[sector] = alloc
    
    # Distribute rounding remainder to largest sectors
    allocated = sum(sector_allocation.values())
    deficit = remaining_budget - allocated
    if deficit > 0:
        sorted_sectors = sorted(sector_allocation.keys(), 
                                key=lambda s: len(dataset_per_sector.get(s, [])), reverse=True)
        for i in range(deficit):
            sector_allocation[sorted_sectors[i % len(sorted_sectors)]] += 1

    print(f"      📊 Budget total: {min_total} (personal={total_personal} + proportionnel={remaining_budget})")
    
    # ── Étape 3 : Sélectionner les symboles par secteur ──
    cleaned: Dict[str, Dict[str, List[str]]] = {}
    grand_total = 0
    
    for sector, buckets in sector_cap_ranges.items():
        cleaned[sector] = {}
        
        # All available symbols in this sector (across all cap ranges)
        all_sector_syms = set()
        for cap, syms in buckets.items():
            all_sector_syms.update(syms)
        
        # Personal symbols for this sector
        personal_sector = all_personal.get(sector, set())
        
        # Target count for this sector = personal + proportional allocation
        target = len(personal_sector) + sector_allocation.get(sector, 0)
        
        # Build the selection: personal first, then fill proportionally
        selected = set(personal_sector)
        
        # Fill from popular symbols of this sector
        if len(selected) < target:
            needed = target - len(selected)
            try:
                popular_sector = get_popular_symbols_by_sector(
                    sector=sector,
                    exclude_symbols=selected
                )
                random.shuffle(popular_sector)
                added = popular_sector[:needed]
                selected.update(added)
            except Exception:
                pass
        
        # If still short, use any available symbols from the dataset in this sector
        if len(selected) < target:
            remaining = [s for s in all_sector_syms if s not in selected]
            random.shuffle(remaining)
            selected.update(remaining[:target - len(selected)])
        
        # Build a lookup: symbol -> cap_range (from original buckets)
        sym_to_cap: Dict[str, str] = {}
        for cap, syms in buckets.items():
            for s in syms:
                sym_to_cap[s] = cap
        
        # For symbols not in original buckets (added from popular), look up their cap_range
        new_syms = selected - set(sym_to_cap.keys())
        if new_syms:
            try:
                import sqlite3
                from config import DB_PATH
                conn = sqlite3.connect(DB_PATH)
                cursor = conn.cursor()
                placeholders = ','.join('?' for _ in new_syms)
                cursor.execute(
                    f'SELECT symbol, market_cap_range FROM symbols WHERE symbol IN ({placeholders})',
                    list(new_syms)
                )
                for row in cursor.fetchall():
                    sym_to_cap[row[0]] = row[1] if row[1] else 'Unknown'
                conn.close()
            except Exception:
                pass
            # Default to 'Unknown' for any still-unmapped symbols
            for s in new_syms:
                if s not in sym_to_cap:
                    sym_to_cap[s] = 'Unknown'
        
        # Distribute selected symbols into cap_range buckets
        cap_groups: Dict[str, List[str]] = {}
        for s in selected:
            cap = sym_to_cap.get(s, 'Unknown')
            cap_groups.setdefault(cap, []).append(s)
        
        # ── Compléter les cap_ranges < 7 avec des populaires du même secteur+cap ──
        already_used = set(selected)
        for cap in list(cap_groups.keys()):
            if len(cap_groups[cap]) < min_symbols:
                needed = min_symbols - len(cap_groups[cap])
                try:
                    pop_same_cap = get_symbols_by_sector_and_cap(
                        sector=sector, cap_range=cap, list_type='popular', active_only=True
                    )
                    candidates = [s for s in pop_same_cap if s not in already_used]
                    random.shuffle(candidates)
                    added = candidates[:needed]
                    if added:
                        cap_groups[cap].extend(added)
                        already_used.update(added)
                        print(f"      📥 [{sector}][{cap}] Complété avec {len(added)} populaires → {len(cap_groups[cap])} symboles")
                except Exception:
                    pass
        
        # ── Fusion des cap_ranges encore < 7 après complément ──
        # Ordre de fusion : Small → Mid → Large → Mega → Unknown
        CAP_ORDER = ['Small', 'Mid', 'Large', 'Mega', 'Unknown']
        # Add any cap_range not in CAP_ORDER at the end
        extra_caps = [c for c in cap_groups if c not in CAP_ORDER]
        merge_order = CAP_ORDER + extra_caps
        
        merged_groups: Dict[str, List[str]] = {}
        accumulator: List[str] = []
        accumulator_labels: List[str] = []
        
        for cap in merge_order:
            syms_in_cap = cap_groups.get(cap, [])
            if not syms_in_cap:
                continue
            accumulator.extend(syms_in_cap)
            accumulator_labels.append(cap)
            
            if len(accumulator) >= min_symbols:
                # Enough symbols — create the merged group
                merged_label = '+'.join(accumulator_labels)
                merged_groups[merged_label] = list(dict.fromkeys(accumulator))
                accumulator = []
                accumulator_labels = []
        
        # Handle leftover: merge with the last created group
        if accumulator:
            if merged_groups:
                # Merge into the last group
                last_key = list(merged_groups.keys())[-1]
                combined_label = last_key + '+' + '+'.join(accumulator_labels)
                merged_groups[combined_label] = list(dict.fromkeys(
                    merged_groups.pop(last_key) + accumulator
                ))
            else:
                # No groups yet — create one with whatever we have
                merged_label = '+'.join(accumulator_labels)
                merged_groups[merged_label] = list(dict.fromkeys(accumulator))
        
        # Log merges
        for label, syms_list in merged_groups.items():
            if '+' in label:
                print(f"      🔗 [{sector}] Fusionné {label}: {len(syms_list)} symboles")
        
        # Store merged groups
        for label, syms_list in merged_groups.items():
            cleaned[sector][label] = syms_list
            grand_total += len(syms_list)
        
        sector_total = sum(len(s) for s in cleaned[sector].values())
        if sector_total > 0:
            pct = len(dataset_per_sector.get(sector, [])) / dataset_total * 100
            print(f"      ✅ {sector}: {sector_total} symboles "
                  f"({len(personal_sector)} personal + {sector_total - len(personal_sector)} popular) "
                  f"[dataset: {pct:.0f}%]")
    
    print(f"      📊 TOTAL sélectionné: {grand_total} symboles (min={min_total})")
    
    return cleaned

# Drapeaux d'un ANCIEN vecteur, un par feature de prix, que la base historique
# porte encore. La base reelle a 545 lignes et aucune colonne
# `use_price_extras`, mais 185 lignes ou `use_price_slope` ou `use_price_acc`
# est non nul. Liste reprise telle quelle de l'ancien chemin de relecture, dont
# trois colonnes n'existent dans aucune base connue : elles se lisent alors
# comme absentes et ne declenchent rien.
DRAPEAUX_PRIX_HERITES = (
    'use_price_slope', 'use_price_acc', 'use_price_rsi_slope',
    'use_price_vol_slope', 'use_price_var5j',
)

# Extras de prix introduits APRES ces drapeaux. Une ligne heritee ne les porte
# pas ; l'ancien chemin de relecture les lisait a 0.0, qui est le poids neutre
# d'un coefficient. Les laisser prendre le milieu de leurs bornes fabriquerait
# une configuration qui n'a jamais tourne.
EXTRAS_PRIX_POSTERIEURS = ('a16', 'a17', 'a18', 'th16', 'th17', 'th18')


def _vecteur_depuis_ligne_historique(row, prix: bool = False,
                                     fond: bool = False) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Objectif:
        Relire une ligne de optimization_runs par le contrat unique, en
        restaurant d'abord la semantique des lignes anterieures a la colonne
        `use_price_extras`. Sans ce repli, 185 lignes de la base reelle
        perdraient silencieusement leurs features de prix au rejeu, donc
        changeraient de score et avec lui la decision de sauvegarde.

    Inputs:
        row (Mapping | sqlite3.Row): indexable par nom de colonne
        prix (bool), fond (bool): drapeaux de features du run COURANT

    Outputs:
        vecteur (np.ndarray): contraint, de la taille du vecteur courant

    Le repli ne touche qu'une ligne a la fois heritee et concernee. Une ligne
    moderne, qui porte une vraie colonne `use_price_extras`, passe inchangee,
    meme si elle vaut 0 et meme si un drapeau herite y traine a 1 : la colonne
    moderne fait foi. Le repli est aussi inerte quand le run courant tourne
    sans features de prix, car le vecteur n'a alors aucun emplacement ou le
    poser ; c'est exactement ce que faisait l'ancien code, dont tout le bloc
    d'extras etait sous `if use_price_features:`.
    --------------------------------------------------------------------------
    """
    def _colonne(nom):
        try:
            return row[nom]
        except (KeyError, IndexError, TypeError):
            return None

    if (prix
            and _colonne('use_price_extras') is None
            and any(_colonne(nom) for nom in DRAPEAUX_PRIX_HERITES)):
        ligne = dict(row)
        ligne['use_price_extras'] = 1
        for nom in EXTRAS_PRIX_POSTERIEURS:
            ligne[nom] = 0.0
        return params.depuis_colonnes(ligne, prix=prix, fond=fond)

    return params.depuis_colonnes(row, prix=prix, fond=fond)


def _vecteur_historique_depuis_champs(coeffs, seuils_moteur,
                                      seuil_achat: float, seuil_vente: float,
                                      extras_prix: Optional[dict] = None,
                                      extras_fondamentaux: Optional[dict] = None,
                                      prix: bool = False,
                                      fond: bool = False) -> np.ndarray:
    """
    --------------------------------------------------------------------------
    Objectif:
        Ranger des parametres historiques deja lus (coefficients, seuils du
        moteur, seuils globaux, extras) aux index que `params.indices()`
        declare. La relecture posait auparavant ces valeurs a la main, dans un
        ordre litteral de 8 coefficients, 4 seuils, 2 globaux, 11 extras de
        prix et 11 extras fondamentaux : c'etait la quatrieme description du
        vecteur, et la plus longue, donc celle qui se serait tue si l'ordre du
        contrat changeait.

    Inputs:
        coeffs (Sequence[float]): les 8 coefficients a1 a a8
        seuils_moteur (Sequence[float]): les 8 seuils dans l'ordre
            params.ORDRE_SEUILS_MOTEUR ; les seuils geles y sont ignores, car
            ils n'occupent aucun emplacement du vecteur de recherche
        seuil_achat (float), seuil_vente (float): seuils globaux
        extras_prix (dict | None): extras de prix, cles par SpecParam.cle
        extras_fondamentaux (dict | None): extras fondamentaux, meme convention
        prix (bool), fond (bool): drapeaux de features du run COURANT

    Outputs:
        vecteur (np.ndarray): non contraint et non arrondi, de la taille du
        vecteur courant. Le bridage reste au seul `params.contraindre`, applique
        a l'evaluation et a la sauvegarde.

    Leve ValueError si un emplacement du vecteur reste sans valeur, plutot que
    de laisser passer un zero silencieux.
    --------------------------------------------------------------------------
    """
    index_par_cle = params.indices(prix=prix, fond=fond)

    valeurs_par_cle: Dict[str, float] = {
        f'a{numero}': float(coeffs[numero - 1]) for numero in range(1, 9)
    }
    for position, cle in enumerate(params.ORDRE_SEUILS_MOTEUR):
        if cle in index_par_cle and position < len(seuils_moteur):
            valeurs_par_cle[cle] = float(seuils_moteur[position])
    valeurs_par_cle['seuil_achat'] = float(seuil_achat)
    valeurs_par_cle['seuil_vente'] = float(seuil_vente)
    for source in (extras_prix, extras_fondamentaux):
        if not source:
            continue
        for cle, valeur in source.items():
            if cle in index_par_cle:
                valeurs_par_cle[cle] = float(valeur)

    manquantes = sorted(cle for cle in index_par_cle
                        if cle not in valeurs_par_cle)
    if manquantes:
        raise ValueError(
            "parametres historiques absents du vecteur : "
            f"{', '.join(manquantes)}")

    vecteur = np.zeros(len(index_par_cle), dtype=float)
    for cle, index in index_par_cle.items():
        vecteur[index] = valeurs_par_cle[cle]
    return vecteur


# ----
# Top-level objective for SciPy DE (picklable on Windows)
def _de_objective(params_vecteur, optimizer):
    """Objective wrapper for differential_evolution.
    Uses a top-level function so it can be pickled when workers are used.
    """
    try:
        rounded = optimizer.round_params(params_vecteur)
        return -optimizer.evaluate_config(rounded)
    except Exception:
        # Penalize any evaluation failure to keep DE robust
        return 1000.0

class HybridOptimizer:
    """Optimiseur hybride utilisant plusieurs stratégies d'optimisation avec limitation des décimales"""
    
    def __init__(self, stock_data, domain, montant=50,
                 transaction_cost=COUT_TRANSACTION_PAR_TRADE, precision=2,
                 use_price_features: bool = False,
                 use_fundamentals_features: bool = False,
                 optimization_mode: str = 'gain_moyen',
                 compute_backend: Optional[ComputeBackend] = None,
                 seed: Optional[int] = None):
        self.stock_data = stock_data
        self.domain = domain
        self.montant = montant
        self.transaction_cost = transaction_cost
        self.evaluation_count = 0
        # Borne pour ne pas croitre indefiniment sur des dizaines de milliers
        # d'evaluations. Cle : tuple du vecteur contraint.
        self.mesures = _BoundedCache(maxsize=4096)
        self.precision = precision  # 🔧 NUOVO: Précision des paramètres (nombre de décimales)
        self.use_price_features = use_price_features
        self.use_fundamentals_features = use_fundamentals_features
        self.optimization_mode = optimization_mode  # 'gain_moyen' ou 'taux_reussite'
        self.compute_backend = compute_backend or ComputeBackend('cpu')

        # Executor réutilisable pour l'évaluation par symbole afin d'éviter
        # la création répétée de ThreadPoolExecutor (coûteux) à chaque
        # évaluation de configuration et réduire le risque de nested
        # thread oversubscription.
        try:
            symbol_workers = min(MAX_WORKERS, max(1, len(self.stock_data)))
        except Exception:
            symbol_workers = min(MAX_WORKERS, 4)
        self._symbol_executor = ThreadPoolExecutor(max_workers=symbol_workers)
        # Préparation des séries: garder des pandas Series (index temporel nécessaire au moteur C/Python).
        self.stock_series = {}
        for symbol, data in self.stock_data.items():
            try:
                close_s = data['Close'] if isinstance(data['Close'], pd.Series) else pd.Series(data['Close'])
                vol_s = data['Volume'] if isinstance(data['Volume'], pd.Series) else pd.Series(data['Volume'], index=close_s.index)
                self.stock_series[symbol] = {'Close': close_s, 'Volume': vol_s}
            except Exception:
                self.stock_series[symbol] = {'Close': data['Close'], 'Volume': data['Volume']}
        
        # Graine explicite : elle amorce numpy ET random, utilises par le GA,
        # le PSO et clean_sector_cap_groups. Un run peut ainsi etre rejoue.
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

        # Bornes derivees du contrat unique (core/optim_params.py). Elles
        # etaient auparavant decrites ici, puis re-decrites plus etroitement a
        # l'evaluation et a la sauvegarde, ce qui faisait diverger le vecteur
        # sauvegarde du vecteur evalue.
        self.bounds = params.bornes(prix=self.use_price_features,
                                    fond=self.use_fundamentals_features)
        
        # ✨ V2.0: Charger les paramètres optimisés existants comme point de départ
        self.optimized_coeffs_loaded = False
        self.initial_coeffs = None
        self.initial_thresholds = None
        self.meilleur_score = -float('inf')  # 🔧 Meilleur score trouvé (global)
        self.meilleur_trades = 0  # 🔧 Stocker le nombre de trades de la meilleure config
        self.meilleur_success = 0  # 🔧 Stocker le nombre de trades gagnants de la meilleure config
        # 🔧 Pénalité par trade/symbole: à gain égal, moins de trades = meilleur score
        # Ex: 0.02 * 6 trades/symbole = -0.12 sur le score. Assez pour départager,
        # trop petit pour dominer un vrai meilleur gain.
        self.trade_efficiency_penalty = 0.02

    @property
    def meilleur_success_rate(self):
        return (self.meilleur_success / self.meilleur_trades * 100) if self.meilleur_trades > 0 else 0.0
        
    def round_params(self, params_vecteur):
        """🔧 NOUVEAU: Arrondir les paramètres à la précision définie"""
        return np.round(params_vecteur, self.precision)
    
    def evaluate_config(self, params_vecteur):
        """Évalue une configuration. Le bridage passe par le contrat unique."""
        prix = self.use_price_features
        fond = self.use_fundamentals_features

        vecteur = params.contraindre(
            self.round_params(params_vecteur), prix=prix, fond=fond)
        param_key = tuple(vecteur)
        connue = self.mesures.get(param_key)
        if connue is not None:
            return connue.score

        coeffs = params.coefficients(vecteur, prix=prix, fond=fond)
        feature_thresholds = params.seuils_features(vecteur, prix=prix, fond=fond)
        seuil_achat, seuil_vente = params.globaux(vecteur, prix=prix, fond=fond)
        price_extras = params.dict_extras_prix(vecteur, prix=prix, fond=fond)
        fundamentals_extras = params.dict_extras_fondamentaux(
            vecteur, prix=prix, fond=fond)

        total_gain = 0.0
        total_trades = 0
        total_success = 0
        
        # 🚀 Helper function for parallel execution
        def evaluate_symbol(symbol):
            try:
                ser_data = self.stock_series.get(symbol)
                if not ser_data:
                    return 0.0, 0, 0
                # backtest_signals_c_extended n'a AUCUN parametre de seuils :
                # py_backtest_symbol ne prend que (prices, volumes, coeffs,
                # montant, cost). Les 4 seuils optimises y etaient donc perdus.
                # with_events les honore via domain_thresholds.
                #
                # COUT REEL, mesure le 2026-08-06 sur 1210 barres. Le "+1 %"
                # qu'annoncait ce commentaire n'etait vrai que sans module C :
                #   with_events (Python)            7,17 s
                #   c_extended avec C_ACCELERATION   0,0002 s
                # soit un facteur 37 000. En interface graphique la question ne
                # se pose pas, QSI_DISABLE_C_ACCELERATION=1 y ramenant de toute
                # facon c_extended sur ce meme chemin Python ; en ligne de
                # commande, ou le module C se charge, le surcout est entier.
                #
                # Il est assume ici : le moteur C ne peut PAS honorer les
                # seuils, donc le garder rendait 4 des 14 dimensions inertes a
                # l'optimisation tout en les ecrivant en base, ou elles
                # pilotaient les signaux reels. Justesse d'abord.
                #
                # Le surcout n'est pas une fatalite du Python : 96 % du temps
                # part dans get_trading_signal, appele une fois PAR BARRE, qui
                # recalcule tous les indicateurs sur tout l'historique (ADX
                # 47 %, RSI 7 %) et relit les parametres en base a chaque barre
                # (extract_best_parameters 14 %, soit 1160 requetes SQLite).
                # Sortir ces calculs de la boucle ne change aucun resultat.
                # C'est le defaut N2, objet du lot 2.
                result, _evenements = backtest_signals_with_events(
                    ser_data['Close'], ser_data['Volume'], "default",
                    self.montant, self.transaction_cost,
                    domain_coeffs={"default": coeffs},
                    domain_thresholds={"default": feature_thresholds},
                    seuil_achat=seuil_achat, seuil_vente=seuil_vente,
                    extra_params=price_extras,
                    fundamentals_extras=fundamentals_extras,
                    symbol_name=symbol,
                )
                return result['gain_total'], result['trades'], result.get('gagnants', 0)
            except Exception as e:
                return 0.0, 0, 0
        
        try:
            # ⚡⚡ PARALLÉLISATION: Évaluer tous les symboles en parallèle
            # Reuse a shared executor to avoid repeated creation and nested
            # thread oversubscription when the optimizer itself parallelises
            # populations or when SciPy uses workers.
            futures = [self._symbol_executor.submit(evaluate_symbol, symbol)
                       for symbol in self.stock_data.keys()]
            for future in as_completed(futures):
                gain, trades, success = future.result()
                total_gain += gain
                total_trades += trades
                total_success += success

            avg_gain = total_gain / len(self.stock_data) if self.stock_data else 0.0
            self.evaluation_count += 1

            # 🚦 Gardes: rejeter les configs sans trades pour éviter des scores incohérents
            if total_trades == 0:
                mesure = Mesure(score=-1e6, gain_moyen=avg_gain, trades=0, gagnants=0)
                self.mesures[param_key] = mesure
                return mesure.score

            # 🔧 Pénalité d'efficacité: à gain égal, préférer moins de trades
            # trades_per_symbol = moyenne de trades par symbole
            n_symbols = len(self.stock_data)
            trades_per_symbol = total_trades / n_symbols if n_symbols > 0 else 0
            score = avg_gain - self.trade_efficiency_penalty * trades_per_symbol

            # Mode taux_reussite : rejeter si success_rate < 50% ou gain <= 0
            if self.optimization_mode == 'taux_reussite':
                success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0
                if success_rate < 50.0 or avg_gain <= 0:
                    mesure = Mesure(score=-1e6, gain_moyen=avg_gain,
                                    trades=total_trades, gagnants=total_success)
                    self.mesures[param_key] = mesure
                    return mesure.score

            mesure = Mesure(score=score, gain_moyen=avg_gain,
                            trades=total_trades, gagnants=total_success)
            self.mesures[param_key] = mesure

            # meilleur_* ne sert plus qu'a l'affichage tqdm. Ne rien en deduire
            # pour la sauvegarde : sous workers=-1 ces compteurs restent dans
            # les sous-processus.
            if score > self.meilleur_score:
                self.meilleur_score = score
                self.meilleur_trades = total_trades
                self.meilleur_success = total_success

            return score

        except Exception as e:
            print(f"⚠️ evaluate_config error: {e}")  # Debug: show exceptions
            return -1000.0  # Pénalité pour configurations invalides

    def shutdown(self):
        """Shutdown any internal executors cleanly."""
        try:
            if hasattr(self, '_symbol_executor') and self._symbol_executor is not None:
                self._symbol_executor.shutdown(wait=True)
        except Exception:
            pass

    def mesure_de(self, vecteur) -> Mesure:
        """
        --------------------------------------------------------------------------
        Objectif:
            Rendre les metriques DU vecteur demande, en evaluant si elles
            manquent. C'est ce qui garantit que les `trades` ecrits en base
            appartiennent aux coefficients ecrits sur la meme ligne.

        Inputs:
            vecteur (Sequence[float]): vecteur de parametres

        Outputs:
            mesure (Mesure): score, gain moyen, trades, gagnants
        --------------------------------------------------------------------------
        """
        contraint = params.contraindre(
            self.round_params(vecteur),
            prix=self.use_price_features, fond=self.use_fundamentals_features)
        cle = tuple(contraint)
        connue = self.mesures.get(cle)
        if connue is None:
            score = self.evaluate_config(contraint)
            # evaluate_config() rearrondit et rebride en interne
            # (round_params puis contraindre appliques une deuxieme fois). Pres
            # d'une borne non alignee sur la grille de precision (ex. 0.25 a
            # precision=1), la deuxieme passe peut retomber sur une valeur
            # differente de `contraint` et donc ecrire sous une autre cle ; le
            # `except Exception` englobant d'evaluate_config peut aussi
            # court-circuiter les trois ecritures normales. Dans les deux cas
            # `self.mesures[cle]` n'existe pas : replier sur le score deja
            # obtenu plutot que de lever un KeyError qui ferait avorter la
            # sauvegarde apres tout le budget d'evaluations depense.
            connue = self.mesures.get(cle)
            if connue is None:
                connue = Mesure(score=score, gain_moyen=0.0, trades=0, gagnants=0)
        return connue

    def genetic_algorithm(self, population_size=50, generations=30, mutation_rate=0.15):
        """Algorithme génétique pour l'optimisation avec précision limitée"""
        print(f"🧬 Démarrage algorithme génétique (pop={population_size}, gen={generations}, précision={self.precision})")
        
        # Utiliser self.bounds (16 paramètres: 8 coefficients + 8 seuils individuels)
        bounds = self.bounds
        population = []
        for _ in range(population_size):
            individual = []
            for low, high in bounds:
                # 🔧 MODIFIÉ: Génération avec pas discret selon la précision
                if self.precision == 1:
                    step = 0.1
                elif self.precision == 2:
                    step = 0.05
                else:
                    step = 0.01
                
                # Génération par pas discrets
                n_steps = int((high - low) / step)
                random_step = np.random.randint(0, n_steps + 1)
                value = low + random_step * step
                individual.append(round(value, self.precision))
            population.append(np.array(individual))

        best_fitness = -float('inf')
        best_individual = None

        with tqdm(total=generations, desc="🧬 Évolution génétique", unit="gen") as pbar:
            for gen in range(generations):
                # Évaluation (parallélisée sur la population)
                # Utiliser un petit threadpool pour évaluer les individus en
                # parallèle; chaque évaluation elle-même parallélise par
                # symbole via `self._symbol_executor`.
                pop_workers = min(8, max(1, population_size))
                with ThreadPoolExecutor(max_workers=pop_workers) as pop_exec:
                    fitness_scores = list(pop_exec.map(self.evaluate_config, population))

                # Sélection des meilleurs
                fitness_indices = np.argsort(fitness_scores)[::-1]
                elite_size = population_size // 4
                elite = [population[i] for i in fitness_indices[:elite_size]]

                # Mise à jour du meilleur
                current_best = fitness_scores[fitness_indices[0]]
                if current_best > best_fitness:
                    best_fitness = current_best
                    best_individual = population[fitness_indices[0]].copy()

                # Nouvelle génération
                new_population = elite.copy()
                while len(new_population) < population_size:
                    # Sélection par tournoi
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Croisement
                    child1, child2 = self._crossover(parent1, parent2)

                    # Mutation
                    if np.random.random() < mutation_rate:
                        child1 = self._mutate(child1, bounds)
                    if np.random.random() < mutation_rate:
                        child2 = self._mutate(child2, bounds)

                    new_population.extend([child1, child2])

                population = new_population[:population_size]

                pbar.set_postfix({'Meilleur': f"{best_fitness:.3f} ({self.meilleur_success_rate:.1f}%)", 'Trades': self.meilleur_trades})
                pbar.update(1)

        return self.round_params(best_individual), best_fitness

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Sélection par tournoi"""
        indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in indices]
        winner_idx = indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2, alpha=0.3):
        """Croisement BLX-α avec arrondi"""
        child1 = np.zeros_like(parent1)
        child2 = np.zeros_like(parent2)

        for i in range(len(parent1)):
            min_val = min(parent1[i], parent2[i])
            max_val = max(parent1[i], parent2[i])
            range_val = max_val - min_val

            low = min_val - alpha * range_val
            high = max_val + alpha * range_val

            # 🔧 MODIFIÉ: Arrondir les enfants
            child1[i] = round(np.random.uniform(low, high), self.precision)
            child2[i] = round(np.random.uniform(low, high), self.precision)

        # BLX-alpha etend l'intervalle parental, donc les enfants peuvent sortir
        # des bornes. Sans ce bridage, le meilleur individu retourne par le GA
        # pouvait etre hors domaine et partir en base tel quel.
        prix = self.use_price_features
        fond = self.use_fundamentals_features
        return (params.contraindre(child1, prix=prix, fond=fond),
                params.contraindre(child2, prix=prix, fond=fond))

    def _mutate(self, individual, bounds, sigma=0.1):
        """Mutation gaussienne avec arrondi"""
        mutated = individual.copy()
        for i in range(len(individual)):
            if np.random.random() < 0.1:  # Probabilité de mutation par gène
                noise = np.random.normal(0, sigma * (bounds[i][1] - bounds[i][0]))
                new_value = individual[i] + noise
                # 🔧 MODIFIÉ: Arrondir et contraindre
                mutated[i] = round(np.clip(new_value, bounds[i][0], bounds[i][1]), self.precision)
        return mutated

    def differential_evolution_opt(self, enveloppe: int = 30000):
        """Évolution différentielle. L'enveloppe est un nombre d'évaluations."""
        bounds = self.bounds
        dimension = len(bounds)
        multiplicateur, max_iterations = optim_budget.plan_differential(enveloppe, dimension)
        prevu = optim_budget.evaluations_differential(multiplicateur, max_iterations, dimension)
        print(f"🔄 Évolution différentielle : population={multiplicateur * dimension}, "
              f"iterations={max_iterations}, évaluations prévues={prevu} "
              f"(enveloppe={enveloppe}, précision={self.precision})")

        with tqdm(total=max_iterations, desc="🔄 Évolution différentielle", unit="iter") as pbar:
            def callback(xk, convergence):
                pbar.set_postfix({
                    'Convergence': f"{convergence:.6f}",
                    'Score': f"{self.meilleur_score:.3f} ({self.meilleur_success_rate:.1f}%)",
                    'Trades': self.meilleur_trades,
                })
                pbar.update(1)

            result = differential_evolution(
                _de_objective,
                bounds,
                args=(self,),
                maxiter=max_iterations,
                popsize=multiplicateur,   # MULTIPLICATEUR, population = popsize * dimension
                mutation=(0.5, 1.5),
                recombination=0.7,
                callback=callback,
                polish=False,
                seed=self.seed,
                # workers=1 : sous multiprocessing, les mesures enregistrees par
                # evaluate_config restent dans les sous-processus, ce dont la
                # sauvegarde depend. La vitesse viendra du lot 2.
                workers=1,
            )

        return params.contraindre(
            result.x, prix=self.use_price_features,
            fond=self.use_fundamentals_features), -result.fun

    def latin_hypercube_sampling(self, n_samples=500):
        """Échantillonnage Latin Hypercube avec arrondi"""
        print(f"🎯 Latin Hypercube Sampling avec {n_samples} échantillons (précision={self.precision})")
        
        # 🔧 CORRIGÉ: Utiliser la dimension réelle des bounds au lieu de 18 hardcodé
        n_dimensions = len(self.bounds)
        sampler = qmc.LatinHypercube(d=n_dimensions, seed=self.seed)
        samples = sampler.random(n=n_samples)

        # Mise à l'échelle
        bounds = self.bounds
        l_bounds = [b[0] for b in bounds]
        u_bounds = [b[1] for b in bounds]
        scaled_samples = qmc.scale(samples, l_bounds, u_bounds)

        # 🔧 MODIFIÉ: Arrondir les échantillons
        scaled_samples = np.array([self.round_params(sample) for sample in scaled_samples])

        best_params = None
        best_score = -float('inf')

        with tqdm(total=n_samples, desc="🎯 LHS Exploration", unit="sample") as pbar:
            for sample in scaled_samples:
                score = self.evaluate_config(sample)
                if score > best_score:
                    best_score = score
                    best_params = sample.copy()

                pbar.set_postfix({'Meilleur': f"{best_score:.3f} ({self.meilleur_success_rate:.1f}%)", 'Trades': self.meilleur_trades})
                pbar.update(1)

        return best_params, best_score

    def particle_swarm_optimization(self, n_particles=30, max_iterations=50):
        """Optimisation par essaim particulaire (PSO) avec arrondi"""
        print(f"🐝 Particle Swarm Optimization (particles={n_particles}, iter={max_iterations}, précision={self.precision})")
        
        bounds = np.array(self.bounds)
        n_dims = len(self.bounds)  # 🔧 Dimension dynamique

        # Initialisation avec arrondi
        particles = np.random.uniform(bounds[:, 0], bounds[:, 1], (n_particles, n_dims))
        particles = np.array([self.round_params(p) for p in particles])  # 🔧 MODIFIÉ
        
        velocities = np.random.uniform(-1, 1, (n_particles, n_dims))
        personal_best_positions = particles.copy()
        personal_best_scores = np.array([self.evaluate_config(p) for p in particles])

        global_best_idx = np.argmax(personal_best_scores)
        global_best_position = personal_best_positions[global_best_idx].copy()
        global_best_score = personal_best_scores[global_best_idx]

        w = 0.7  # Inertie
        c1 = 1.4  # Coefficient cognitif
        c2 = 1.4  # Coefficient social

        with tqdm(total=max_iterations, desc="🐝 PSO", unit="iter") as pbar:
            for iteration in range(max_iterations):
                for i in range(n_particles):
                    # Mise à jour vitesse
                    r1, r2 = np.random.random(2)
                    velocities[i] = (w * velocities[i] +
                                   c1 * r1 * (personal_best_positions[i] - particles[i]) +
                                   c2 * r2 * (global_best_position - particles[i]))

                    # Mise à jour position avec arrondi
                    particles[i] += velocities[i]
                    particles[i] = np.clip(particles[i], bounds[:, 0], bounds[:, 1])
                    particles[i] = self.round_params(particles[i])  # 🔧 MODIFIÉ

                    # Évaluation
                    score = self.evaluate_config(particles[i])

                    # Mise à jour personnel
                    if score > personal_best_scores[i]:
                        personal_best_scores[i] = score
                        personal_best_positions[i] = particles[i].copy()

                    # Mise à jour global
                    if score > global_best_score:
                        global_best_score = score
                        global_best_position = particles[i].copy()

                pbar.set_postfix({'Meilleur': f"{global_best_score:.3f} ({self.meilleur_success_rate:.1f}%)", 'Trades': self.meilleur_trades})
                pbar.update(1)

        return global_best_position, global_best_score

    def replay_all_historical(self, domain, use_price_features=False, use_fundamentals_features=False):
        """Réévalue TOUS les sets historiques de la DB pour cette catégorie sur les données actuelles"""
        import sqlite3
        from config import OPTIMIZATION_DB_PATH

        # Extraire secteur / cap_range depuis le domain composite
        allowed_caps = {'Small', 'Mid', 'Large', 'Mega', 'Unknown'}
        sector = domain
        cap_range = None
        if '_' in domain:
            maybe_sector, maybe_cap = domain.rsplit('_', 1)
            if maybe_cap in allowed_caps:
                sector = maybe_sector
                cap_range = maybe_cap

        conn = None
        try:
            conn = sqlite3.connect(OPTIMIZATION_DB_PATH)
            conn.row_factory = sqlite3.Row
            cur = conn.cursor()

            # Charger TOUS les runs pour ce secteur+cap_range
            if cap_range:
                cur.execute(
                    "SELECT * FROM optimization_runs WHERE sector = ? AND COALESCE(market_cap_range, 'Unknown') = ? ORDER BY timestamp",
                    (sector, cap_range)
                )
            else:
                cur.execute(
                    "SELECT * FROM optimization_runs WHERE sector = ? ORDER BY timestamp",
                    (sector,)
                )

            rows = cur.fetchall()
        except Exception as e:
            print(f"⚠️ Impossible de charger l'historique: {e}")
            return None, -float('inf'), None
        finally:
            # Fermer meme si la requete a leve : la connexion fuyait sinon.
            if conn is not None:
                conn.close()

        if not rows:
            print(f"ℹ️ Aucun historique trouvé pour {domain}")
            return None, -float('inf'), None

        print(f"📚 Replay de {len(rows)} sets historiques pour {domain}...")

        best_params = None
        best_score = -float('inf')
        best_label = None

        # La ligne est relue par le contrat unique (core/optim_params.py). Les
        # deux listes de colonnes ecrites a la main, ici a l'envers de la
        # sauvegarde, etaient la troisieme description concurrente du vecteur.
        # `_vecteur_depuis_ligne_historique` y ajoute le seul savoir qui ne
        # relevait pas du contrat : la compatibilite des lignes anterieures a
        # la colonne `use_price_extras`.
        for row in rows:
            try:
                vecteur = _vecteur_depuis_ligne_historique(
                    row, prix=use_price_features, fond=use_fundamentals_features)
                score = self.evaluate_config(vecteur)
                if score > best_score:
                    best_score = score
                    best_params = vecteur.copy()
                    best_label = f"Historical ({row['timestamp']})"
            except Exception:
                continue

        if best_params is not None:
            sr = (self.meilleur_success / self.meilleur_trades * 100) if self.meilleur_trades > 0 else 0.0
            print(f"🏆 Meilleur historique: {best_label} score={best_score:.4f} ({sr:.1f}%)")
        else:
            print(f"ℹ️ Aucun set historique valide")

        return best_params, best_score, best_label

def optimize_sector_coefficients_hybrid(
    sector_symbols, domain,
    period='5y', strategy='hybrid',
    montant=50, transaction_cost=COUT_TRANSACTION_PAR_TRADE,
    initial_thresholds=(4.20, -0.5),
    budget_evaluations=1000,
    precision=2,  # 🔧 NOUVEAU: Paramètre de précision
    cap_range='Unknown',
    use_price_features=False,
    use_fundamentals_features=False,
    optimization_mode='gain_moyen',
    compute_backend_preference: str = 'auto',
    seed: Optional[int] = None
):
    """
    Optimisation hybride des coefficients sectoriels avec limitation des décimales
    
    Strategies disponibles:
    - 'genetic': Algorithmes génétiques
    - 'differential': Évolution différentielle  
    - 'pso': Particle Swarm Optimization
    - 'lhs': Latin Hypercube Sampling
    - 'hybrid': Combine plusieurs méthodes
    
    precision: Nombre de décimales pour les paramètres (1, 2, ou 3)
    cap_range: Segment de capitalisation associé au secteur
    """
    if not sector_symbols:
        print(f"🚫 Secteur {domain} vide, ignoré")
        return None, 0.0, 0.0, initial_thresholds, None

    # Téléchargement des données
    stock_data = download_stock_data(sector_symbols, period=period)
    if not stock_data:
        print(f"🚨 Aucune donnée téléchargée pour le secteur {domain}")
        return None, 0.0, 0.0, initial_thresholds, None

    compute_backend = resolve_compute_backend(compute_backend_preference)

    # 🔄 Pré-chargement des fondamentaux (évite les appels yfinance pendant l'optimisation)
    if use_fundamentals_features:
        try:
            from fundamentals_cache import get_fundamental_metrics
            print(f"📊 Pré-chargement des fondamentaux pour {len(sector_symbols)} symboles...")
            for sym in sector_symbols:
                try:
                    get_fundamental_metrics(sym, use_cache=True)
                except Exception as e:
                    print(f"  ⚠️ {sym}: {e}")
            print(f"✅ Fondamentaux pré-chargés")
        except Exception as e:
            print(f"⚠️ Erreur pré-chargement fondamentaux: {e}")

    # Pré-calcul léger des features pour chauffer les opérations pandas
    def precalculate_features(sd: Dict[str, Dict[str, pd.Series]], backend: ComputeBackend):
        try:
            for sym, dat in sd.items():
                close = dat.get('Close')
                volume = dat.get('Volume')
                if isinstance(close, pd.Series) and len(close) >= 50:
                    # Calculs légers pour chauffer les caches internes (CPU/GPU selon backend)
                    precalc = backend.precalculate_symbol_features(close, volume)
                    # Stockage facultatif pour analyse future (non utilisé par backtest)
                    dat['PRECALC_EMA20'] = precalc.get('PRECALC_EMA20')
                    dat['PRECALC_EMA50'] = precalc.get('PRECALC_EMA50')
                    dat['PRECALC_VOL30'] = precalc.get('PRECALC_VOL30')
        except Exception:
            pass

    precalculate_features(stock_data, compute_backend)

    for symbol, data in stock_data.items():
        print(f"📊 {symbol}: {len(data['Close'])} points de données")

    # Récupération des meilleurs paramètres historiques
    from config import OPTIMIZATION_DB_PATH
    db_path = OPTIMIZATION_DB_PATH
    best_params_per_sector = extract_best_parameters(db_path)

    csv_timestamp = None
    if domain in best_params_per_sector:
        csv_coeffs, csv_thresholds, csv_globals, csv_gain, csv_extras = best_params_per_sector[domain]
        # 🔧 Sécuriser en float (déjà des floats depuis SQLite, mais par sécurité)
        csv_coeffs = tuple(float(x) for x in csv_coeffs)
        csv_thresholds = tuple(float(x) for x in csv_thresholds)
        csv_globals = tuple(float(x) for x in csv_globals)
        if isinstance(csv_extras, dict):
            csv_timestamp = csv_extras.get('timestamp')
        print(f"📋 Paramètres historiques trouvés: coeffs={csv_coeffs}, seuils={csv_thresholds}, globaux={csv_globals}, gain={csv_gain:.2f}")
    else:
        csv_coeffs, csv_thresholds, csv_globals, csv_gain = None, initial_thresholds, (4.2, -0.5), -float('inf')

    hist_avg_gain = None  # 🔧 Pour mesurer l'amélioration vs l'historique
    hist_objective_score = None  # 🔧 Score historique avec pénalité trades (via evaluate_config)
    hist_total_trades = None
    hist_success_rate = None
    hist_params_vector = None
    hist_label = "Historical (re-eval)"

    # ♻️ Réévaluer les paramètres historiques sur les données ACTUELLES
    if csv_coeffs is not None and len(csv_coeffs) >= 8 and len(csv_thresholds) >= 8 and len(csv_globals) == 2:
        try:
            hist_coeffs = tuple(csv_coeffs[:8])
            hist_feature_thresholds = tuple(csv_thresholds[:8])
            hist_seuil_achat = float(csv_globals[0])
            hist_seuil_vente = float(csv_globals[1])
            hist_label = f"Historical ({csv_timestamp})" if csv_timestamp else "Historical (re-eval)"
            
            # 🔧 CORRIGÉ: Récupérer les extras depuis BEST_PARAM_EXTRAS (global)
            hist_extra_params = None
            hist_fundamentals_extras = None
            
            from qsi import BEST_PARAM_EXTRAS

            if domain in BEST_PARAM_EXTRAS:
                extras_dict = BEST_PARAM_EXTRAS[domain]
                # Vérifier si des extras de price existent dans le dictionnaire
                if 'use_price_extras' in extras_dict or any(k in extras_dict for k in (
                    'use_price_slope', 'use_price_acc', 'use_price_rsi_slope', 'use_price_vol_slope', 'use_price_var5j'
                )):
                    hist_extra_params = {
                        'use_price_extras': int(extras_dict.get('use_price_extras', 0) or int(
                            any(extras_dict.get(k, 0) for k in (
                                'use_price_slope', 'use_price_acc', 'use_price_rsi_slope',
                                'use_price_vol_slope', 'use_price_var5j'
                            ))
                        )),
                        'a_price_slope': float(extras_dict.get('a_price_slope', 0.0)),
                        'a_price_acc': float(extras_dict.get('a_price_acc', 0.0)),
                        'th_price_slope': float(extras_dict.get('th_price_slope', 0.0)),
                        'th_price_acc': float(extras_dict.get('th_price_acc', 0.0)),
                        'a_price_rsi_slope': float(extras_dict.get('a_price_rsi_slope', 0.0)),
                        'a_price_vol_slope': float(extras_dict.get('a_price_vol_slope', 0.0)),
                        'a_price_var5j': float(extras_dict.get('a_price_var5j', 0.0)),
                        'th_price_rsi_slope': float(extras_dict.get('th_price_rsi_slope', 0.0)),
                        'th_price_vol_slope': float(extras_dict.get('th_price_vol_slope', 0.0)),
                        'th_price_var5j': float(extras_dict.get('th_price_var5j', 0.0)),
                    }
                
                # Vérifier si des extras de fundamentals existent dans le dictionnaire
                if 'use_fundamentals' in extras_dict:
                    hist_fundamentals_extras = {
                        'use_fundamentals': int(extras_dict.get('use_fundamentals', 0)),
                        'a_rev_growth': float(extras_dict.get('a_rev_growth', 0.0)),
                        'a_eps_growth': float(extras_dict.get('a_eps_growth', 0.0)),
                        'a_roe': float(extras_dict.get('a_roe', 0.0)),
                        'a_fcf_yield': float(extras_dict.get('a_fcf_yield', 0.0)),
                        'a_de_ratio': float(extras_dict.get('a_de_ratio', 0.0)),
                        'th_rev_growth': float(extras_dict.get('th_rev_growth', 0.0)),
                        'th_eps_growth': float(extras_dict.get('th_eps_growth', 0.0)),
                        'th_roe': float(extras_dict.get('th_roe', 0.0)),
                        'th_fcf_yield': float(extras_dict.get('th_fcf_yield', 0.0)),
                        'th_de_ratio': float(extras_dict.get('th_de_ratio', 0.0)),
                    }

            # Respecter les flags du run courant : désactiver ou injecter des valeurs neutres
            if not use_price_features:
                hist_extra_params = None
            elif hist_extra_params is None:
                hist_extra_params = {
                    'use_price_extras': 0,
                    'a_price_slope': 0.0,
                    'a_price_acc': 0.0,
                    'th_price_slope': 0.0,
                    'th_price_acc': 0.0,
                    'a_price_rsi_slope': 0.0,
                    'a_price_vol_slope': 0.0,
                    'a_price_var5j': 0.0,
                    'th_price_rsi_slope': 0.0,
                    'th_price_vol_slope': 0.0,
                    'th_price_var5j': 0.0,
                }

            if not use_fundamentals_features:
                hist_fundamentals_extras = None
            elif hist_fundamentals_extras is None:
                hist_fundamentals_extras = {
                    'use_fundamentals': 0,
                    'a_rev_growth': 0.0,
                    'a_eps_growth': 0.0,
                    'a_roe': 0.0,
                    'a_fcf_yield': 0.0,
                    'a_de_ratio': 0.0,
                    'th_rev_growth': 0.0,
                    'th_eps_growth': 0.0,
                    'th_roe': 0.0,
                    'th_fcf_yield': 0.0,
                    'th_de_ratio': 0.0,
                }

            # Le vecteur historique derive du contrat unique
            # (core/optim_params.py) : coefficients, seuils encore optimisables,
            # globaux et extras se rangent aux index que params.indices()
            # declare. Sa mesure, elle, attend l'optimiseur : elle passe par
            # optimizer.mesure_de() plus bas, donc par le meme moteur et les
            # memes seuils que le score auquel elle est comparee.
            hist_params_vector = tuple(_vecteur_historique_depuis_champs(
                hist_coeffs, hist_feature_thresholds,
                hist_seuil_achat, hist_seuil_vente,
                extras_prix=hist_extra_params,
                extras_fondamentaux=hist_fundamentals_extras,
                prix=use_price_features, fond=use_fundamentals_features))

            if use_price_features or use_fundamentals_features:
                # Afficher l'état réel des features historiques
                price_active = hist_extra_params and hist_extra_params.get('use_price_extras', 0)
                fund_active = hist_fundamentals_extras and hist_fundamentals_extras.get('use_fundamentals', 0)
                status_msg = f"price={'✅' if price_active else '❌'}, fundamentals={'✅' if fund_active else '❌'}"
                print(f"   ℹ️  Paramètres étendus de 14 → {len(hist_params_vector)} ({status_msg})")
        except Exception as e:
            print(f"⚠️ Reconstruction des paramètres historiques impossible: {e}")

    # 🔧 MODIFIÉ: Initialisation de l'optimiseur avec précision
    optimizer = HybridOptimizer(
        stock_data,
        domain,
        montant,
        transaction_cost,
        precision,
        use_price_features,
        use_fundamentals_features,
        optimization_mode,
        compute_backend=compute_backend,
        seed=seed,
    )

    # Le pool de threads du symbol-executor doit etre libere meme si une
    # strategie ou la sauvegarde leve : ce bloc couvre tout, du calcul du
    # score historique a la constitution du resume.
    try:
        # 🔄 Le baseline historique est mesuré par le MÊME moteur que le score
        # auquel il est comparé. Une boucle backtest_signals_c_extended le
        # calculait auparavant à part ; ce moteur n'a aucun paramètre de seuils,
        # il ignore donc les 8 seuils, les 2 globaux et les deux dictionnaires
        # d'extras. La ligne de fin de groupe opposait ainsi un backtest
        # entièrement paramétré à un backtest de coefficients seuls.
        historical_candidate = None
        if hist_params_vector is not None:
            try:
                rounded_hist_vector = optimizer.round_params(np.array(hist_params_vector))
                mesure_hist = optimizer.mesure_de(rounded_hist_vector)
                hist_avg_gain = mesure_hist.gain_moyen
                hist_total_trades = mesure_hist.trades
                hist_success_rate = ((mesure_hist.gagnants / mesure_hist.trades * 100)
                                     if mesure_hist.trades > 0 else 0.0)

                # 🚦 Gardes: ignorer les historiques sans trades. Ils ne
                # concourent pas et ne servent pas de référence de sauvegarde.
                if mesure_hist.trades == 0:
                    print("   ⚠️  Historique ignoré (0 trade)")
                else:
                    # 🔧 Conserver pour comparaison de sauvegarde
                    hist_objective_score = mesure_hist.score
                    historical_candidate = (hist_label, tuple(rounded_hist_vector),
                                            mesure_hist.score)
                    print(f"   ✅ Score historique (objective aligné): {mesure_hist.score:.2f} | gain_moy={hist_avg_gain:.2f} | trades={hist_total_trades}")
            except Exception as e:
                print(f"⚠️ Normalisation du score historique impossible: {e}")

        # Une seule regle d'echelle : le budget recu est deja final.
        print(f"🚀 Optimisation hybride pour {domain}, stratégie '{strategy}', "
              f"précision {precision}, budget {budget_evaluations} évaluations")

        results = []
        if historical_candidate:
            results.append(historical_candidate)

        dimension = len(optimizer.bounds)
        part = optim_budget.REPARTITION_HYBRIDE if strategy == 'hybrid' else None

        def _enveloppe(nom: str) -> int:
            return int(budget_evaluations * part[nom]) if part else budget_evaluations

        if strategy == 'genetic':
            population, generations = optim_budget.plan_genetique(budget_evaluations)
            params_ga, score_ga = optimizer.genetic_algorithm(population, generations)
            results.append(('Genetic Algorithm', params_ga, score_ga))

        if strategy in ('hybrid', 'differential'):
            params_de, score_de = optimizer.differential_evolution_opt(_enveloppe('differential'))
            results.append(('Differential Evolution', params_de, score_de))

        if strategy in ('hybrid', 'pso'):
            particules, iterations = optim_budget.plan_pso(_enveloppe('pso'))
            params_pso, score_pso = optimizer.particle_swarm_optimization(particules, iterations)
            results.append(('PSO', params_pso, score_pso))

        if strategy in ('hybrid', 'lhs'):
            params_lhs, score_lhs = optimizer.latin_hypercube_sampling(
                optim_budget.plan_lhs(_enveloppe('lhs')))
            results.append(('Latin Hypercube', params_lhs, score_lhs))

        # Sélection du meilleur résultat
        best_method, best_params, best_score = max(results, key=lambda x: x[2])
        _sr = (optimizer.meilleur_success / optimizer.meilleur_trades * 100) if optimizer.meilleur_trades > 0 else 0.0
        print(f"🏆 Meilleure méthode: {best_method} avec score {best_score:.4f} ({_sr:.1f}%)")

        # Afficher aussi le meilleur candidat non-historique pour transparence
        non_hist_candidates = [r for r in results if r[0] != 'Historical (re-eval)']
        if non_hist_candidates:
            nh_method, nh_params, nh_score = max(non_hist_candidates, key=lambda x: x[2])
            nh_coeffs = tuple(float(x) for x in nh_params[:8])
            nh_th = tuple(float(nh_params[i]) for i in range(8, 12))
            nh_buy = float(nh_params[12])
            nh_sell = float(nh_params[13])
            print(f"🔎 Meilleur candidat Optimiseur: {nh_method} score={nh_score:.4f} ({_sr:.1f}%)")
            print(f"   coeffs={nh_coeffs}")
            print(f"   seuils: features={nh_th}, achat={nh_buy:.2f}, vente={nh_sell:.2f}")

        # Replay de TOUS les sets historiques pour cette catégorie
        if strategy == 'hybrid':
            hist_params, hist_score, hist_label = optimizer.replay_all_historical(
                domain, use_price_features=use_price_features,
                use_fundamentals_features=use_fundamentals_features
            )
            if hist_params is not None and hist_score > best_score:
                best_params = hist_params
                best_score = hist_score
                best_method = hist_label
                _sr = optimizer.meilleur_success_rate
                print(f"✨ Un set historique bat l'optimiseur: {hist_label} score={best_score:.4f} ({_sr:.1f}%)")

        # Coefficients, seuils et extras derivent tous du contrat unique
        # (core/optim_params.py). Le bloc precedent les reconstruisait a la main a
        # partir d'offsets fixes et les bridait une seconde fois, plus etroitement
        # que les bornes de recherche : le vecteur affiche puis sauvegarde n'etait
        # alors plus celui qui avait produit le score.
        best_coeffs = params.coefficients(best_params, prix=use_price_features,
                                          fond=use_fundamentals_features)
        best_feature_thresholds = params.seuils_features(
            best_params, prix=use_price_features, fond=use_fundamentals_features)
        best_seuil_achat, best_seuil_vente = params.globaux(
            best_params, prix=use_price_features, fond=use_fundamentals_features)
        extra_params = params.dict_extras_prix(
            best_params, prix=use_price_features, fond=use_fundamentals_features)
        fundamentals_extras = params.dict_extras_fondamentaux(
            best_params, prix=use_price_features, fond=use_fundamentals_features)
        all_thresholds = best_feature_thresholds + (best_seuil_achat, best_seuil_vente)

        # 🔧 DEBUG: Taille du vecteur de params
        print(f"   🔍 Taille du meilleur vecteur de params: {len(best_params)} (expected: {len(optimizer.bounds)})")

        # Metriques DU vecteur retenu, jamais celles de la meilleure configuration
        # jamais vue. Avec workers=-1 les compteurs de l'optimiseur restaient a zero
        # et strategy='differential' ne sauvegardait donc jamais rien.
        mesure_retenue = optimizer.mesure_de(best_params)
        total_trades = mesure_retenue.trades
        total_success = mesure_retenue.gagnants
        best_score = mesure_retenue.score
        success_rate = (total_success / total_trades * 100) if total_trades > 0 else 0.0

        # �📊 Rapport synthétique secteur
        if hist_avg_gain is not None:
            delta = best_score - hist_avg_gain
            delta_pct = (delta / abs(hist_avg_gain) * 100) if hist_avg_gain != 0 else None
            success_old_str = f"{hist_success_rate:.1f}%" if hist_success_rate is not None else "-"
            trades_old_str = hist_total_trades if hist_total_trades is not None else "-"
            delta_pct_str = f", {delta_pct:+.1f}%" if delta_pct is not None else ""
            print(f"✅ {domain}: gain {best_score:.2f} vs {hist_avg_gain:.2f} ({delta:+.2f}{delta_pct_str}); trades {total_trades} vs {trades_old_str}; success {success_rate:.1f}% vs {success_old_str} ; méthode {best_method}")
        else:
            print(f"✅ {domain}: gain {best_score:.2f}; trades {total_trades}; success {success_rate:.1f}% ; méthode {best_method}")

        print(f"   coeffs: {best_coeffs}")
        print(f"   seuils: features={best_feature_thresholds}, achat={best_seuil_achat:.2f}, vente={best_seuil_vente:.2f}")
    
        # 🔧 Afficher les extras si présents
        if extra_params:
            print(
                f"   📊 Price features: enabled={extra_params['use_price_extras']}"
            )
            print(
                f"      Poids: slope={extra_params['a_price_slope']:.1f}, acc={extra_params['a_price_acc']:.1f}, "
                f"rsi={extra_params['a_price_rsi_slope']:.1f}, vol={extra_params['a_price_vol_slope']:.1f}, var5j={extra_params['a_price_var5j']:.1f}"
            )
            print(
                f"      Seuils: slope={extra_params['th_price_slope']:.3f}, acc={extra_params['th_price_acc']:.3f}, "
                f"rsi={extra_params['th_price_rsi_slope']:.3f}, vol={extra_params['th_price_vol_slope']:.3f}, var5j={extra_params['th_price_var5j']:.2f}"
            )
    
        if fundamentals_extras:
            print(f"   📊 Fundamentals: use={fundamentals_extras['use_fundamentals']}")
            print(f"      Poids: rev={fundamentals_extras['a_rev_growth']:.1f}, eps={fundamentals_extras['a_eps_growth']:.1f}, roe={fundamentals_extras['a_roe']:.1f}, fcf={fundamentals_extras['a_fcf_yield']:.1f}, de={fundamentals_extras['a_de_ratio']:.1f}")
            print(f"      Seuils: rev={fundamentals_extras['th_rev_growth']:.1f}%, eps={fundamentals_extras['th_eps_growth']:.1f}%, roe={fundamentals_extras['th_roe']:.1f}%, fcf={fundamentals_extras['th_fcf_yield']:.1f}%, de={fundamentals_extras['th_de_ratio']:.1f}")

        # 🔧 Sauvegarder si le nouveau score surpasse le score historique RÉÉVALUÉ sur données actuelles
        # Comparaison avec hist_objective_score (incluant pénalité trades) pour cohérence
        save_epsilon = 0.01
        hist_ref = hist_objective_score if hist_objective_score is not None else hist_avg_gain
        score_is_better = (hist_ref is None) or (best_score > hist_ref + save_epsilon)
        no_trades = (total_trades == 0)
        should_save = score_is_better and not no_trades
    
        if should_save:
            save_optimization_results(
                domain, best_params, mesure_retenue, cap_range=cap_range,
                prix=use_price_features, fond=use_fundamentals_features,
                transaction_cost=transaction_cost, seed=seed)
            hist_str = f"{hist_ref:.2f}" if hist_ref is not None else "N/A"
            print(f"💾 Sauvegarde: nouveau score {best_score:.2f} ({success_rate:.1f}%) > historique réévalué {hist_str} (trades: {total_trades})")
        elif no_trades:
            print(f"ℹ️ Pas de sauvegarde: aucun trade généré (score {best_score:.2f} mais 0 trades)")
        else:
            print(f"ℹ️ Pas de sauvegarde: nouveau {best_score:.2f} ({success_rate:.1f}%) ≤ historique réévalué {hist_ref:.2f} (epsilon={save_epsilon})")

        summary = {
            'sector': domain,
            'cap_range': cap_range,
            'gain_new': best_score,
            'gain_old': hist_avg_gain,
            'trades_new': total_trades,
            'trades_old': hist_total_trades,
            'success_new': success_rate,
            'success_old': hist_success_rate
        }
    finally:
        optimizer.shutdown()
        # Frontiere naturelle de liberation du cache d'instantanes techniques.
        # Sa cle porte le nom du symbole et les groupes secteur x cap_range ne
        # partagent pas de symboles : les entrees du groupe qui s'acheve ne
        # resserviront jamais, et sans ce vidage l'etat stable d'un run complet
        # serait le plafond de 100 000 entrees (~196 Mo) tenu jusqu'a la sortie
        # du process. Voir core/cache.py, section « CYCLE DE VIE REEL ».
        TA_CACHE.clear()

    return best_coeffs, best_score, success_rate, all_thresholds, summary

def save_optimization_results(domain, vecteur, mesure, cap_range=None,
                              prix=False, fond=False,
                              transaction_cost=COUT_TRANSACTION_PAR_TRADE,
                              seed=None):
    """
    --------------------------------------------------------------------------
    Objectif:
        Ecrire une ligne de optimization_runs decrivant UNE configuration et
        les metriques de CETTE configuration.

    Inputs:
        domain (str): secteur ou cle composite secteur_capRange
        vecteur (Sequence[float]): vecteur de parametres, deja contraint
        mesure (Mesure): metriques du meme vecteur
        cap_range (str | None): segment de capitalisation
        prix (bool), fond (bool): drapeaux de features actives
        transaction_cost (float): cout absolu par trade utilise pour la mesure
        seed (int | None): graine du run, pour rejouabilite

    Outputs:
        None. Journalise l'echec sans le propager a l'appelant.
    --------------------------------------------------------------------------
    """
    from datetime import datetime
    import sqlite3
    from config import OPTIMIZATION_DB_PATH

    def _ensure_opt_runs_schema(conn):
        try:
            cur = conn.cursor()
            cur.execute("PRAGMA table_info(optimization_runs)")
            cols = {row[1] for row in cur.fetchall()}
            nouvelles = [('market_cap_range', 'TEXT')]
            nouvelles += [(spec.colonne_db, 'REAL')
                          for spec in EXTRAS_COLONNES_MIGRATION]
            nouvelles += [('use_price_extras', 'INTEGER DEFAULT 0'),
                          ('use_fundamentals', 'INTEGER DEFAULT 0'),
                          ('transaction_cost', 'REAL'),
                          ('seed', 'INTEGER')]
            for nom, decl in nouvelles:
                if nom not in cols:
                    try:
                        cur.execute(
                            f"ALTER TABLE optimization_runs ADD COLUMN {nom} {decl}")
                    except Exception:
                        pass
            conn.commit()
        except Exception:
            # Non fatal : la retrocompatibilite prime sur la migration.
            pass

    normalized_cap = cap_range or 'Unknown'
    normalized_sector = domain
    allowed_caps = {'Small', 'Mid', 'Large', 'Mega', 'Unknown'}
    if '_' in domain:
        maybe_sector, maybe_cap = domain.rsplit('_', 1)
        if maybe_cap in allowed_caps:
            normalized_sector = maybe_sector
            if normalized_cap == 'Unknown':
                normalized_cap = maybe_cap

    conn = None
    try:
        # Dans le try : params.vers_colonnes leve ValueError sur un vecteur de
        # mauvaise taille, et le docstring promet de ne rien propager.
        colonnes = params.vers_colonnes(vecteur, prix=prix, fond=fond)
        colonnes.update({
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'sector': normalized_sector,
            'market_cap_range': normalized_cap,
            'gain_moy': float(mesure.score),
            'success_rate': (mesure.gagnants / mesure.trades * 100) if mesure.trades else 0.0,
            'trades': int(mesure.trades),
            'transaction_cost': float(transaction_cost),
            'seed': int(seed) if seed is not None else None,
        })

        conn = sqlite3.connect(OPTIMIZATION_DB_PATH)
        _ensure_opt_runs_schema(conn)
        noms = list(colonnes)
        marqueurs = ', '.join('?' for _ in noms)
        # Frontiere de securite : les noms de colonnes viennent du contrat
        # core/optim_params.py et des cles ajoutees juste au-dessus, jamais
        # d'une entree utilisateur ou d'une ligne de base. Seule cette
        # provenance rend l'interpolation sure ; les VALEURS restent liees par
        # des marqueurs. Ne jamais alimenter `colonnes` depuis l'exterieur.
        conn.execute(
            f"INSERT OR REPLACE INTO optimization_runs ({', '.join(noms)}) "
            f"VALUES ({marqueurs})",
            [colonnes[nom] for nom in noms],
        )
        conn.commit()
        print(f"📝 Résultats sauvegardés pour {normalized_sector} ({normalized_cap})")
    except Exception as exc:
        print(f"⚠️ Erreur lors de la sauvegarde: {exc}")
    finally:
        if conn is not None:
            conn.close()

# Exemple d'utilisation
if __name__ == "__main__":
    print("\n" + "="*80)
    print("🚀 OPTIMISATEUR HYBRIDE - Génération de coefficients par secteur × cap_range")
    print("="*80)
    
    # Chargement des symboles - priorité à SQLite
    list_type = "optimization"
    if SYMBOL_MANAGER_AVAILABLE:
        print("\n1️⃣  Chargement des symboles depuis SQLite...")
        init_symbols_table()
        try:
            count = sync_txt_to_sqlite("optimisation_symbols.txt", list_type)
            # Les messages sont maintenant affichés par sync_txt_to_sqlite
        except Exception as e:
            print(f"   ⚠️ Impossible de synchroniser optimisation_symbols.txt: {e}")

        symbols = get_symbols_by_list_type(list_type, active_only=True)
        print(f"   ✅ {len(symbols)} symboles actifs chargés ({list_type})")
        
        # Obtenir tous les secteurs et cap_ranges disponibles
        sectors_available = get_all_sectors(list_type=list_type)
        cap_ranges_available = get_all_cap_ranges(list_type=list_type)
        
        print(f"\n2️⃣  Organisation des symboles:")
        print(f"   - Secteurs: {len(sectors_available)}")
        print(f"   - Gammes de cap: {len(cap_ranges_available)}")
        
        # Construction de sector_cap_ranges depuis SQLite
        sector_cap_ranges = {}
        total_combos = 0
        for sector in sectors_available:
            sector_cap_ranges[sector] = {}
            for cap_range in cap_ranges_available:
                syms = get_symbols_by_sector_and_cap(sector, cap_range, list_type, active_only=True)
                if syms:
                    sector_cap_ranges[sector][cap_range] = syms
                    total_combos += 1
                    print(f"   ✅ {sector} × {cap_range}: {len(syms)} symboles")
        
        print(f"\n   📊 Total: {total_combos} combinaisons secteur×cap_range avec symboles")
        
        # Résumé par secteur
        print(f"\n   📈 Détail par secteur:")
        sector_symbol_counts = {}
        for sector, cap_dict in sector_cap_ranges.items():
            total_syms = sum(len(syms) for syms in cap_dict.values())
            if total_syms > 0:
                sector_symbol_counts[sector] = total_syms
                print(f"      • {sector}: {total_syms} symboles ({len(cap_dict)} cap_range(s))")
        
        print(f"      Total secteurs: {len(sector_symbol_counts)}")

        # Nettoyage des groupes (complément + réduction)
        print("\n   🧹 Nettoyage des groupes (complément + réduction)...")
        sector_cap_ranges = clean_sector_cap_groups(sector_cap_ranges, ttl_days=0, min_total=400)

    else:
        print("\n⚠️ SQLite non disponible, utilisation de la méthode classique...")
        # Fallback: méthode originale
        symbols = list(dict.fromkeys(load_symbols_from_txt("optimisation_symbols.txt")))
        
        sectors = {
            "Technology": [],
            "Healthcare": [],
            "Financial Services": [],
            "Consumer Cyclical": [],
            "Industrials": [],
            "Energy": [],
            "Basic Materials": [],
            "Communication Services": [],
            "Consumer Defensive": [],
            "Utilities": [],
            "Real Estate": [],
            "ℹ️Inconnu!!": []
        }
        
        cap_buckets = ["Small", "Mid", "Large", "Mega", "Unknown"]
        sector_cap_ranges = {sec: {cap: [] for cap in cap_buckets} for sec in sectors.keys()}
        
        print(f"📋 Assignation des secteurs (cache yfinance utilisé)...")
        for symbol in symbols:
            sector = get_sector(symbol, use_cache=True)
            if sector not in sectors:
                sector = "ℹ️Inconnu!!"
            cap_range = classify_cap_range(symbol)
            
            sectors[sector].append(symbol)
            sector_cap_ranges.setdefault(sector, {cap: [] for cap in cap_buckets})
            sector_cap_ranges[sector].setdefault(cap_range, []).append(symbol)
        
        print("\n📋 Assignation secteur × cap range:")
        for sector, buckets in sector_cap_ranges.items():
            for cap_range, syms in buckets.items():
                if syms:
                    print(f"{sector} [{cap_range}]: {len(syms)} symboles")

        # Nettoyage des groupes (complément + réduction)
        print("\n   🧹 Nettoyage des groupes (complément + réduction)...")
        sector_cap_ranges = clean_sector_cap_groups(sector_cap_ranges, ttl_days=0, min_total=400)

    # ═══════════════════════════════════════════════════════════════════════
    # 3️⃣  CONFIGURATION DE L'OPTIMISATION (menu interactif)
    # ═══════════════════════════════════════════════════════════════════════
    # Vérifier si accélération C disponible
    try:
        from trading_c_acceleration.qsi_optimized import C_ACCELERATION
        accel_status = "⚡ Module C activé" if C_ACCELERATION else "🐍 Python vectorisé"
    except Exception:
        accel_status = "🐍 Python vectorisé"

    total_to_optimize = sum(1 for s in sector_cap_ranges.values() for cap, syms in s.items() if syms)
    total_symbols = sum(len(syms) for s in sector_cap_ranges.values() for syms in s.values())

    # Valeurs par défaut
    strategy = 'hybrid'
    precision = 2
    use_price_features = True
    use_fundamentals_features = True
    optimization_mode = 'taux_reussite'
    compute_backend_preference = 'auto'

    def _show_config():
        """Affiche la configuration courante"""
        param_count = 14
        if use_price_features:
            param_count += 11
        if use_fundamentals_features:
            param_count += 11
        budget_affiche = optim_budget.budget_effectif(BUDGET_BASE, precision)
        mode_label = "Gain moyen" if optimization_mode == 'gain_moyen' else "Taux réussite ≥50% + gain>0"

        print("\n" + "─" * 60)
        print("  ⚙️  CONFIGURATION ACTUELLE")
        print("─" * 60)
        print(f"  [1] Stratégie ............. {strategy}")
        print(f"  [2] Précision ............. {precision} décimale(s)")
        print(f"  [3] Price features ........ {'✅ Oui' if use_price_features else '❌ Non'}")
        print(f"  [4] Fundamentals features . {'✅ Oui' if use_fundamentals_features else '❌ Non'}")
        print(f"  [5] Mode d'optimisation ... {mode_label}")
        print(f"  [6] Backend calcul ........ {compute_backend_preference}")
        print("─" * 60)
        print(f"  📊 {total_symbols} symboles · {total_to_optimize} groupes · "
              f"{param_count} params · {budget_affiche} éval/groupe")
        print(f"  🖥️  {accel_status} · {min(MAX_WORKERS, total_symbols)} workers")
        print("─" * 60)
        print("  [o] Lancer l'optimisation")
        print("  [q] Quitter")
        print("─" * 60)

    while True:
        _show_config()
        choice = input("  Choix [1-6/o/q] : ").strip().lower()

        if choice == '1':
            options = {'1': 'hybrid', '2': 'differential', '3': 'genetic', '4': 'pso', '5': 'lhs'}
            print("\n  Stratégies disponibles:")
            print("    1. hybrid        (DE + PSO + LHS + replay historique)")
            print("    2. differential  (Évolution différentielle seule)")
            print("    3. genetic       (Algorithme génétique seul)")
            print("    4. pso           (Particle Swarm seul)")
            print("    5. lhs           (Latin Hypercube seul)")
            s = input("  Choix [1-5] : ").strip()
            if s in options:
                strategy = options[s]

        elif choice == '2':
            print("\n  Précision :")
            print("    1. Rapide   (1 décimale — espace réduit)")
            print("    2. Standard (2 décimales)")
            print("    3. Fine     (3 décimales — plus long)")
            p = input("  Choix [1-3] : ").strip()
            if p in ('1', '2', '3'):
                precision = int(p)

        elif choice == '3':
            use_price_features = not use_price_features
            print(f"  → Price features {'activés' if use_price_features else 'désactivés'}")

        elif choice == '4':
            use_fundamentals_features = not use_fundamentals_features
            print(f"  → Fundamentals features {'activés' if use_fundamentals_features else 'désactivés'}")

        elif choice == '5':
            print("\n  Mode d'optimisation :")
            print("    1. Gain moyen — maximise le gain (défaut)")
            print("    2. Taux de réussite — gain max avec ≥50% de trades gagnants et gain > 0")
            m = input("  Choix [1-2] : ").strip()
            if m == '2':
                optimization_mode = 'taux_reussite'
            else:
                optimization_mode = 'gain_moyen'

        elif choice == '6':
            print("\n  Backend calcul :")
            print("    1. auto (GPU CuPy si dispo, sinon CPU)")
            print("    2. cpu")
            print("    3. gpu (forcer CuPy, sinon fallback CPU)")
            b = input("  Choix [1-3] : ").strip()
            if b == '2':
                compute_backend_preference = 'cpu'
            elif b == '3':
                compute_backend_preference = 'gpu'
            else:
                compute_backend_preference = 'auto'

        elif choice == 'o':
            break

        elif choice == 'q':
            print("\n❌ Optimisation annulée")
            sys.exit(0)

    # Budget final
    budget_evaluations = optim_budget.budget_effectif(BUDGET_BASE, precision)

    graine = int(datetime.now().timestamp()) % 100000
    print(f"   Graine du run : {graine} (rejouable, stockée en base)")

    print(f"\n🚀 Lancement : {total_to_optimize} groupes · stratégie={strategy} · précision={precision} · mode={optimization_mode} · backend={compute_backend_preference}")
    print(f"   Budget: {budget_evaluations} éval/groupe\n")

    optimized_coeffs = {}
    sector_summaries = []

    for sector, buckets in sector_cap_ranges.items():
        for cap_range, sector_symbols in buckets.items():
            if not sector_symbols:
                continue

            combo_key = f"{sector}_{cap_range}"
            print(f"\n" + "="*160)
            print(f"🎯 OPTIMISATION {strategy.upper()} - {sector} / {cap_range}")
            print(f"="*160)

            coeffs, gain_total, success_rate, thresholds, summary = optimize_sector_coefficients_hybrid(
                sector_symbols, combo_key,
                period='5y',
                strategy=strategy,
                montant=50,
                transaction_cost=COUT_TRANSACTION_PAR_TRADE,
                budget_evaluations=budget_evaluations,
                precision=precision,  # 🔧 NOUVEAU: Paramètre de précision
                cap_range=cap_range,
                use_price_features=use_price_features,  # 🎯 Features étendues
                use_fundamentals_features=use_fundamentals_features,  # 🎯 Features étendues
                optimization_mode=optimization_mode,
                compute_backend_preference=compute_backend_preference,
                seed=graine,
            )

            if coeffs:
                optimized_coeffs[combo_key] = coeffs
            if summary:
                sector_summaries.append(summary)

    print("\n" + "="*80)
    print("🏆 DICTIONNAIRE FINAL OPTIMISÉ")
    print("="*80)
    print("domain_coeffs = {")
    for sector, coeffs in optimized_coeffs.items():
        print(f"    '{sector}': {coeffs},")
    print("}")
    print("="*80)

    # 📊 Comparaison globale (sectors avec historique disponible)
    comparables = [s for s in sector_summaries if s.get('gain_old') is not None]
    if comparables:
        print("\n📊 Bilan global vs historique (réévalué aujourd'hui):")
        total_old_gain = 0.0
        total_new_gain = 0.0
        total_old_trades = 0
        total_new_trades = 0
        for s in comparables:
            delta = s['gain_new'] - s['gain_old']
            delta_pct = (delta / abs(s['gain_old']) * 100) if s['gain_old'] != 0 else None
            delta_pct_str = f", {delta_pct:+.1f}%" if delta_pct is not None else ""
            trades_old_str = s['trades_old'] if s['trades_old'] is not None else "-"
            success_old_str = f"{s['success_old']:.1f}%" if s['success_old'] is not None else "-"
            label = f"{s['sector']} ({s.get('cap_range', 'Unknown')})"
            print(f" - {label}: gain {s['gain_new']:.2f} vs {s['gain_old']:.2f} ({delta:+.2f}{delta_pct_str}); trades {s['trades_new']} vs {trades_old_str}; success {s['success_new']:.1f}% vs {success_old_str}")
            total_old_gain += s['gain_old']
            total_new_gain += s['gain_new']
            total_old_trades += s['trades_old'] or 0
            total_new_trades += s['trades_new']

        n = len(comparables)
        avg_old = total_old_gain / n if n else 0
        avg_new = total_new_gain / n if n else 0
        delta_tot = avg_new - avg_old
        delta_tot_pct = (delta_tot / abs(avg_old) * 100) if avg_old != 0 else None
        print(f"\nRésumé moyen (sur {n} secteurs): gain {avg_new:.2f} vs {avg_old:.2f} ({delta_tot:+.2f}{'' if delta_tot_pct is None else f', {delta_tot_pct:+.1f}%'}); trades totaux {total_new_trades} vs {total_old_trades}")