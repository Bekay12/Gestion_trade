import argparse
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

# Ensure project src root is on sys.path for imports
import os
import sys
PROJECT_SRC = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_SRC not in sys.path:
    sys.path.insert(0, PROJECT_SRC)

# Project imports
from qsi import download_stock_data, get_trading_signal
from qsi_optimized import extract_best_parameters, backtest_signals_with_events

try:
    import symbol_manager
    from symbol_manager import get_symbols_by_list_type
    SYMBOLS_DB_AVAILABLE = True
except Exception:
    SYMBOLS_DB_AVAILABLE = False
    get_symbols_by_list_type = None


def _get_sector_for_symbol(symbol: str) -> str:
    """Fetch sector for a symbol from SQLite if available, else fallback to 'Unknown'."""
    if SYMBOLS_DB_AVAILABLE:
        try:
            conn = sqlite3.connect(symbol_manager.DB_PATH)
            cur = conn.cursor()
            cur.execute("SELECT sector FROM symbols WHERE symbol=?", (symbol,))
            row = cur.fetchone()
            conn.close()
            if row and row[0]:
                return str(row[0])
        except Exception:
            pass
    return "Unknown"


def _slice_by_date(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    series = series.copy()
    if not isinstance(series.index, (pd.DatetimeIndex, pd.PeriodIndex)):
        series.index = pd.to_datetime(series.index)
    return series.loc[(series.index >= start) & (series.index <= end)]


def compute_daily_reliability(close: pd.Series, volume: pd.Series, domain: str,
                              domain_coeffs: Tuple[float, ...], domain_thresholds: Tuple[float, ...],
                              seuil_achat: float, seuil_vente: float,
                              price_extras: Optional[Dict[str, float]] = None,
                              fundamentals_extras: Optional[Dict[str, float]] = None,
                              min_hold_days: int = 14,
                              volume_min: int = 100000,
                              trailing_months: int = 9,
                              up_to_date: Optional[pd.Timestamp] = None) -> float:
    """Compute trailing reliability (success rate %) on data available up to up_to_date,
    using a trailing window of trailing_months. Used for per-action gating."""
    try:
        c = close.copy()
        v = volume.copy()
        if not isinstance(c.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            c.index = pd.to_datetime(c.index)
        if not isinstance(v.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            v.index = pd.to_datetime(v.index)

        # Determine trailing window
        if up_to_date is None:
            up_to_date = c.index.max()
        window_start = up_to_date - pd.DateOffset(months=trailing_months)
        
        # Slice
        c_window = c.loc[(c.index >= window_start) & (c.index <= up_to_date)]
        v_window = v.loc[(v.index >= window_start) & (v.index <= up_to_date)]
        if len(c_window) < 30:
            return 0.0
        
        # Walk-forward within window
        days = c_window.index.sort_values()
        open_pos = None
        trades = 0
        winners = 0
        
        for day in days:
            prices = c.loc[c.index <= day]
            volumes = v.loc[v.index <= day]
            if len(prices) < 50:
                continue
            signal, *_ = get_trading_signal(
                prices, volumes, domain,
                domain_coeffs={domain: domain_coeffs},
                domain_thresholds={domain: domain_thresholds},
                price_extras=price_extras,
                volume_seuil=volume_min,
                seuil_achat=seuil_achat,
                seuil_vente=seuil_vente
            )
            last_price = float(prices.iloc[-1])
            
            if open_pos is None:
                if signal == "ACHAT":
                    open_pos = {'buy_date': day, 'buy_price': last_price}
            else:
                min_exit_date = open_pos['buy_date'] + timedelta(days=min_hold_days)
                if day < min_exit_date:
                    continue
                if signal == "VENTE":
                    pnl_pct = (last_price - open_pos['buy_price']) / open_pos['buy_price']
                    trades += 1
                    if pnl_pct > 0:
                        winners += 1
                    open_pos = None
        
        if open_pos is not None and len(c) > 0:
            end_price = float(c.iloc[-1])
            pnl_pct = (end_price - open_pos['buy_price']) / open_pos['buy_price']
            trades += 1
            if pnl_pct > 0:
                winners += 1
        
        rate = (winners / trades * 100.0) if trades > 0 else 0.0
        return rate
    except Exception:
        return 0.0


def compute_reliability_walkforward(close: pd.Series, volume: pd.Series, domain: str,
                                    domain_coeffs: Tuple[float, ...], domain_thresholds: Tuple[float, ...],
                                    seuil_achat: float, seuil_vente: float,
                                    price_extras: Optional[Dict[str, float]] = None,
                                    fundamentals_extras: Optional[Dict[str, float]] = None,
                                    min_hold_days: int = 14,
                                    volume_min: int = 100000) -> Tuple[int, int, float]:
    """Compute reliability by a walk-forward backtest on the training window,
    enforcing minimum holding period and avoiding lookahead.

    Returns (winners, trades, success_rate%).
    """
    try:
        # Ensure datetime index
        c = close.copy()
        v = volume.copy()
        if not isinstance(c.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            c.index = pd.to_datetime(c.index)
        if not isinstance(v.index, (pd.DatetimeIndex, pd.PeriodIndex)):
            v.index = pd.to_datetime(v.index)

        # Use available trading days in the window (faster than calendar days)
        days = c.index.sort_values()
        open_pos = None
        trades = 0
        winners = 0

        for day in days:
            prices = c.loc[c.index <= day]
            volumes = v.loc[v.index <= day]
            if len(prices) < 50:
                continue
            signal, score, _rsi, _vol_mean, _tendance, *_ = get_trading_signal(
                prices, volumes, domain,
                domain_coeffs={domain: domain_coeffs},
                domain_thresholds={domain: domain_thresholds},
                price_extras=price_extras,
                volume_seuil=volume_min,
                seuil_achat=seuil_achat,
                seuil_vente=seuil_vente
            )
            last_price = float(prices.iloc[-1])

            if open_pos is None:
                if signal == "ACHAT":
                    open_pos = {'buy_date': day, 'buy_price': last_price}
            else:
                min_exit_date = open_pos['buy_date'] + timedelta(days=min_hold_days)
                if day < min_exit_date:
                    continue
                if signal == "VENTE":
                    pnl_pct = (last_price - open_pos['buy_price']) / open_pos['buy_price']
                    trades += 1
                    if pnl_pct > 0:
                        winners += 1
                    open_pos = None

        # Close remaining position at end of window
        if open_pos is not None and len(c) > 0:
            end_price = float(c.iloc[-1])
            pnl_pct = (end_price - open_pos['buy_price']) / open_pos['buy_price']
            trades += 1
            if pnl_pct > 0:
                winners += 1

        rate = (winners / trades * 100.0) if trades > 0 else 0.0
        return winners, trades, rate
    except Exception:
        return 0, 0, 0.0


def walk_forward_simulation(symbols: List[str],
                            start_date: pd.Timestamp,
                            end_date: pd.Timestamp,
                            domain_params: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, float]]],
                            reliability_threshold: float,
                            min_hold_days: int,
                            trade_amount: float = 1000.0,
                            transaction_cost: float = 1.0,
                            price_extras_by_domain: Optional[Dict[str, Dict[str, float]]] = None,
                            fundamentals_extras_by_domain: Optional[Dict[str, Dict[str, float]]] = None,
                            volume_min: int = 100000,
                            train_months: int = 12,
                            reliability_walkforward: bool = True,
                            use_business_days: bool = False,
                            gate_by_daily_reliability: bool = False,
                            trailing_months: int = 9,
                            recalc_reliability_every: int = 5) -> Dict:
    """Simulate a realistic workflow: step-by-step trading during [start_date, end_date].

    Rules:
    - Only trade symbols with training success rate >= reliability_threshold
    - Min holding period after buy: min_hold_days
    - Use signals computed with data available up to the current day (no lookahead)
    - P&L computed per symbol: trade_amount scaled by percent change minus costs
    """
    period_months = max(24, train_months + 12)  # fetch enough history: training + simulation
    print(f"📥 Downloading data for {len(symbols)} symbols over {period_months} months...", flush=True)
    stock_data = download_stock_data(symbols, period=f"{period_months}mo")
    print(f"✅ Data downloaded: {sum(1 for _ in stock_data)} symbols with data", flush=True)

    # Prepare per-symbol domain
    symbol_domain: Dict[str, str] = {s: _get_sector_for_symbol(s) for s in symbols}

    # Prepare extras lookups
    price_extras_by_domain = price_extras_by_domain or {}
    fundamentals_extras_by_domain = fundamentals_extras_by_domain or {}

    # Training window: 12 months before start_date
    train_end = start_date - pd.Timedelta(days=1)
    train_start = start_date - pd.DateOffset(months=train_months)

    # Compute reliability and filter eligible symbols
    eligible: List[str] = []
    reliability_map: Dict[str, float] = {}

    print("🔎 Computing training reliability per symbol...", flush=True)
    pbar = tqdm(symbols, desc="Reliability", unit="sym")
    for sym in pbar:
        data = stock_data.get(sym)
        if not data:
            continue
        close = _slice_by_date(pd.Series(data['Close']), train_start, train_end)
        vol = _slice_by_date(pd.Series(data['Volume']), train_start, train_end)
        if len(close) < 60:
            # Not enough training data
            reliability_map[sym] = 0.0
            continue
        domain = symbol_domain.get(sym, 'Unknown')
        coeffs, thresholds, globals_ = domain_params.get(domain, ((1.0,)*8, (50.0,0.0,0.0,1.2,25.0,0.0,0.5,4.2), (4.2,-0.5)))
        prix_ex = price_extras_by_domain.get(domain)
        fund_ex = fundamentals_extras_by_domain.get(domain)
        if reliability_walkforward:
            winners, trades, rate = compute_reliability_walkforward(
                close, vol, domain, coeffs, thresholds, globals_[0], globals_[1], prix_ex, fund_ex,
                min_hold_days=min_hold_days, volume_min=volume_min
            )
        else:
            # Fallback to batch backtest if needed
            winners, trades, rate = compute_reliability_walkforward(
                close, vol, domain, coeffs, thresholds, globals_[0], globals_[1], prix_ex, fund_ex,
                min_hold_days=min_hold_days, volume_min=volume_min
            )
        reliability_map[sym] = rate
        pbar.set_postfix({'rate': f"{rate:.1f}%", 'trades': trades})
        if trades > 0 and rate >= reliability_threshold:
            eligible.append(sym)

    print(f"✅ Eligible symbols: {len(eligible)}/{len(symbols)} (threshold={reliability_threshold:.1f}%)", flush=True)

    # Cache for daily reliability computations (if gating enabled)
    reliability_cache: Dict[str, Tuple[pd.Timestamp, float]] = {}  # sym -> (cached_date, rate)

    # Walk-forward simulation
    total_profit = 0.0
    trade_log: List[Dict] = []
    open_positions: Dict[str, Dict] = {}

    # Iterate over calendar days in the simulation window
    sim_dates = (pd.date_range(start_date, end_date, freq='B') if use_business_days
                 else pd.date_range(start_date, end_date, freq='D'))
    print(f"🧪 Simulating {len(sim_dates)} days from {start_date.date()} to {end_date.date()}...", flush=True)
    trades_executed_total = 0
    pbar_days = tqdm(sim_dates, desc="Simulation", unit="day")
    for current_day in pbar_days:
        for sym in eligible:
            data = stock_data.get(sym)
            if not data:
                continue
            prices_full = pd.Series(data['Close'])
            volumes_full = pd.Series(data['Volume'])
            # Slice data up to current_day
            prices = _slice_by_date(prices_full, start_date, current_day)
            volumes = _slice_by_date(volumes_full, start_date, current_day)
            if len(prices) < 50:
                continue

            domain = symbol_domain.get(sym, 'Unknown')
            coeffs, thresholds, globals_ = domain_params.get(domain, ((1.0,)*8, (50.0,0.0,0.0,1.2,25.0,0.0,0.5,4.2), (4.2,-0.5)))
            prix_ex = price_extras_by_domain.get(domain)
            fund_ex = fundamentals_extras_by_domain.get(domain)

            # Compute signal using only past data
            signal, score, rsi, vol_mean, tendance, *_ = get_trading_signal(
                prices, volumes, domain,
                domain_coeffs={domain: coeffs},
                domain_thresholds={domain: thresholds},
                price_extras=prix_ex,
                symbol=sym,
                volume_seuil=volume_min,
                seuil_achat=globals_[0],
                seuil_vente=globals_[1]
            )

            last_price = float(prices.iloc[-1])

            pos = open_positions.get(sym)
            if pos is None:
                # Consider opening a position only on ACHAT
                if signal == "ACHAT":
                    # If daily reliability gating enabled, check trailing reliability first
                    if gate_by_daily_reliability:
                        # Check cache: recompute only if older than recalc_reliability_every business days
                        cached_info = reliability_cache.get(sym)
                        if cached_info is None or (current_day - cached_info[0]).days >= recalc_reliability_every:
                            # Compute trailing reliability up to yesterday
                            yesterday = current_day - pd.Timedelta(days=1)
                            trailing_start = yesterday - pd.DateOffset(months=trailing_months)
                            close_trail = _slice_by_date(prices_full, trailing_start, yesterday)
                            vol_trail = _slice_by_date(volumes_full, trailing_start, yesterday)
                            if len(close_trail) >= 30:
                                daily_rate = compute_daily_reliability(
                                    close_trail, vol_trail, domain, coeffs, thresholds, globals_[0], globals_[1],
                                    prix_ex, fund_ex, min_hold_days=min_hold_days, volume_min=volume_min
                                )
                                reliability_cache[sym] = (current_day, daily_rate)
                            else:
                                daily_rate = 0.0
                                reliability_cache[sym] = (current_day, 0.0)
                        else:
                            daily_rate = cached_info[1]
                        
                        if daily_rate < reliability_threshold:
                            # Skip buy: trailing reliability too low
                            continue
                    
                    open_positions[sym] = {
                        'buy_date': current_day,
                        'buy_price': last_price,
                        'amount': trade_amount
                    }
                continue
            else:
                # Check minimum holding period
                min_exit_date = pos['buy_date'] + timedelta(days=min_hold_days)
                if current_day < min_exit_date:
                    continue
                # Consider closing on VENTE
                if signal == "VENTE":
                    pnl_pct = (last_price - pos['buy_price']) / pos['buy_price']
                    profit = pos['amount'] * pnl_pct - transaction_cost * 2.0
                    total_profit += profit
                    trade_log.append({
                        'symbol': sym,
                        'buy_date': pos['buy_date'],
                        'sell_date': current_day,
                        'buy_price': pos['buy_price'],
                        'sell_price': last_price,
                        'pnl_pct': pnl_pct,
                        'profit': profit,
                        'score_on_sell': score,
                    })
                    open_positions.pop(sym, None)
                    trades_executed_total += 1
        # Update progress info per day
        pbar_days.set_postfix({'open': len(open_positions), 'trades': trades_executed_total})

    # Liquidate any remaining open positions at end_date
    for sym, pos in list(open_positions.items()):
        data = stock_data.get(sym)
        prices_full = pd.Series(data['Close']) if data else None
        if prices_full is None:
            continue
        prices = _slice_by_date(prices_full, start_date, end_date)
        if len(prices) == 0:
            continue
        last_price = float(prices.iloc[-1])
        pnl_pct = (last_price - pos['buy_price']) / pos['buy_price']
        profit = pos['amount'] * pnl_pct - transaction_cost  # one extra cost for closing
        total_profit += profit
        trade_log.append({
            'symbol': sym,
            'buy_date': pos['buy_date'],
            'sell_date': end_date,
            'buy_price': pos['buy_price'],
            'sell_price': last_price,
            'pnl_pct': pnl_pct,
            'profit': profit,
            'score_on_sell': None,
        })
        open_positions.pop(sym, None)

    return {
        'eligible_symbols': eligible,
        'reliability': reliability_map,
        'profit_total': total_profit,
        'trade_log': trade_log,
        'start_date': start_date,
        'end_date': end_date,
        'min_hold_days': min_hold_days,
        'reliability_threshold': reliability_threshold,
    }


def build_domain_params_from_db(db_path: str = 'signaux/optimization_hist.db') -> Tuple[Dict, Dict, Dict]:
    """Load per-sector optimized parameters.
    Returns:
      domain_params_map: domain -> (coeffs, thresholds, (seuil_achat, seuil_vente))
      price_extras_by_domain: domain -> dict of price feature extras
      fundamentals_extras_by_domain: domain -> dict of fundamentals extras
    """
    best_params = extract_best_parameters(db_path)
    domain_params_map: Dict[str, Tuple[Tuple[float, ...], Tuple[float, ...], Tuple[float, float]]] = {}

    # BEST_PARAM_EXTRAS is populated inside qsi.extract_best_parameters
    from qsi import BEST_PARAM_EXTRAS

    price_extras_by_domain: Dict[str, Dict[str, float]] = {}
    fundamentals_extras_by_domain: Dict[str, Dict[str, float]] = {}

    for domain, params_tuple in best_params.items():
        # Handle both 4-tuple (old) and 5-tuple (new with extras)
        if len(params_tuple) == 4:
            coeffs, thresholds, globals_thresholds, _gain = params_tuple
            _extras = {}
        elif len(params_tuple) == 5:
            coeffs, thresholds, globals_thresholds, _gain, _extras = params_tuple
        else:
            continue  # Skip invalid tuples
            
        domain_params_map[domain] = (tuple(coeffs[:8]), tuple(thresholds[:8]), (float(globals_thresholds[0]), float(globals_thresholds[1])))
        extras = BEST_PARAM_EXTRAS.get(domain, {})
        # Split extras
        pe = {
            'use_price_slope': int(extras.get('use_price_slope', 0) or 0),
            'use_price_acc': int(extras.get('use_price_acc', 0) or 0),
            'a_price_slope': float(extras.get('a_price_slope', 0.0) or 0.0),
            'a_price_acc': float(extras.get('a_price_acc', 0.0) or 0.0),
            'th_price_slope': float(extras.get('th_price_slope', 0.0) or 0.0),
            'th_price_acc': float(extras.get('th_price_acc', 0.0) or 0.0),
        }
        fe = {
            'use_fundamentals': int(extras.get('use_fundamentals', 0) or 0),
            'a_rev_growth': float(extras.get('a_rev_growth', 0.0) or 0.0),
            'a_eps_growth': float(extras.get('a_eps_growth', 0.0) or 0.0),
            'a_roe': float(extras.get('a_roe', 0.0) or 0.0),
            'a_fcf_yield': float(extras.get('a_fcf_yield', 0.0) or 0.0),
            'a_de_ratio': float(extras.get('a_de_ratio', 0.0) or 0.0),
            'th_rev_growth': float(extras.get('th_rev_growth', 0.0) or 0.0),
            'th_eps_growth': float(extras.get('th_eps_growth', 0.0) or 0.0),
            'th_roe': float(extras.get('th_roe', 0.0) or 0.0),
            'th_fcf_yield': float(extras.get('th_fcf_yield', 0.0) or 0.0),
            'th_de_ratio': float(extras.get('th_de_ratio', 0.0) or 0.0),
        }
        price_extras_by_domain[domain] = pe
        fundamentals_extras_by_domain[domain] = fe

    return domain_params_map, price_extras_by_domain, fundamentals_extras_by_domain


def parse_args():
    p = argparse.ArgumentParser(description="Realistic workflow validation (1-year walk-forward)")
    p.add_argument('--list-type', default='personal', help="Symbols list type in SQLite (popular, personal, optimization)")
    p.add_argument('--reliability', type=float, default=60.0, help="Minimum success rate on training window (e.g., 30,50,60,80)")
    p.add_argument('--min-hold-days', type=int, default=14, help="Minimum holding period after a buy signal")
    p.add_argument('--trade-amount', type=float, default=100.0, help="Amount invested per position")
    p.add_argument('--year', type=int, help="Simulation year (e.g., 2024). Ignored if --start/--end provided.")
    p.add_argument('--volume-min', type=int, default=100000, help="Minimum average volume threshold in signal (default 100000)")
    p.add_argument('--train-months', type=int, default=12, help="Training window length in months (default 12)")
    p.add_argument('--use-business-days', action='store_true', help="Simulate on business days only (faster)")
    p.add_argument('--reliability-walkforward', action='store_true', help="Compute training reliability via walk-forward (default on)")
    p.add_argument('--start', type=str, help="Simulation start date (YYYY-MM-DD). If omitted, defaults to 12 months before current month start when --end omitted.")
    p.add_argument('--end', type=str, help="Simulation end date (YYYY-MM-DD). If omitted, defaults to current month start.")
    p.add_argument('--gate-by-daily-reliability', action='store_true', help="Gate buy/sell by trailing reliability computed daily (match action-time filters)")
    p.add_argument('--trailing-months', type=int, default=12, help="Trailing window for per-action reliability (default 9 months)")
    p.add_argument('--recalc-reliability-every', type=int, default=5, help="Recalc trailing reliability every N business days (default 5, cache otherwise)")
    return p.parse_args()


def main():
    args = parse_args()

    # Load symbols
    if SYMBOLS_DB_AVAILABLE and get_symbols_by_list_type:
        symbols = get_symbols_by_list_type(args.list_type, active_only=True)
    else:
        # Fallback: load from txt
        from qsi import load_symbols_from_txt
        fname = 'popular_symbols.txt' if args.list_type == 'popular' else 'mes_symbols.txt'
        symbols = load_symbols_from_txt(fname, use_sqlite=False)

    if not symbols:
        print("🚫 No symbols to simulate.")
        return

    # Build domain params from DB
    domain_params, price_extras_by_domain, fundamentals_extras_by_domain = build_domain_params_from_db()

    # Determine simulation dates
    if args.start and args.end:
        start_date = pd.Timestamp(args.start)
        end_date = pd.Timestamp(args.end)
    elif args.year:
        year = int(args.year)
        start_date = pd.Timestamp(f"{year}-01-01")
        end_date = pd.Timestamp(f"{year}-12-31")
    else:
        # Default: use first day of current month as reference
        ref = pd.Timestamp.now().normalize().replace(day=1)
        end_date = ref
        start_date = ref - pd.DateOffset(months=12)

    print(f"🔎 Simulating from {start_date.date()} to {end_date.date()} for {len(symbols)} symbols (list={args.list_type})")
    print(f"   Filters: reliability >= {args.reliability:.1f}%, min_hold_days={args.min_hold_days}, volume_min={args.volume_min}")
    if args.gate_by_daily_reliability:
        print(f"   Daily Gating: trailing_months={args.trailing_months}, recalc_every={args.recalc_reliability_every} days")

    result = walk_forward_simulation(
        symbols,
        start_date,
        end_date,
        domain_params,
        reliability_threshold=args.reliability,
        min_hold_days=args.min_hold_days,
        trade_amount=args.trade_amount,
        transaction_cost=1.0,
        price_extras_by_domain=price_extras_by_domain,
        fundamentals_extras_by_domain=fundamentals_extras_by_domain,
        volume_min=args.volume_min,
        train_months=args.train_months,
        reliability_walkforward=args.reliability_walkforward or True,
        use_business_days=args.use_business_days,
        gate_by_daily_reliability=args.gate_by_daily_reliability,
        trailing_months=args.trailing_months,
        recalc_reliability_every=args.recalc_reliability_every
    )

    # Summary
    print("\n✅ Simulation complete")
    print(f"   Year: {result['start_date'].date()} → {result['end_date'].date()}")
    print(f"   Eligible symbols: {len(result['eligible_symbols'])}")
    print(f"   Total profit: {result['profit_total']:.2f}")
    print(f"   Trades executed: {len(result['trade_log'])}")

    # Show top 10 trades by profit
    trades_sorted = sorted(result['trade_log'], key=lambda x: x['profit'], reverse=True)
    for t in trades_sorted[:10]:
        print(f"   {t['symbol']}: {t['buy_date'].date()} → {t['sell_date'].date()} | Pct={t['pnl_pct']*100:.1f}% | Profit={t['profit']:.2f}")

    # Reliability stats
    elig = result['eligible_symbols']
    rel = result['reliability']
    avg_rel = np.mean([rel[s] for s in elig]) if elig else 0.0
    print(f"   Avg reliability of eligible: {avg_rel:.1f}%")


if __name__ == '__main__':
    main()
