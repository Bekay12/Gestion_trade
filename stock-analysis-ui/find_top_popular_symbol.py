"""Rank a set of popular symbols with the same comparison logic as the UI.

This script downloads 15 months of data, runs the same signal/backtest
pipeline used by the desktop UI, then applies the comparison-table ranking
to identify which symbol lands at the top.
"""

from __future__ import annotations

import argparse
import io
import os
import sys
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Tuple


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


os.environ.setdefault("QSI_DISABLE_C_ACCELERATION", "1")

from qsi import (  # noqa: E402
    extract_best_parameters,
    analyse_signaux_populaires,
    load_symbols_from_txt,
    resolve_symbol_scoring_context,
)


FALLBACK_SYMBOLS = [
    "CVS",
    "BW",
    "BWXT",
    "GLND",
    "NUCL",
    "AXTI",
    "NVTS",
    "BKSY",
    "SIDU",
    "FJET",
    "HMY",
    "AMD",
    "TSM",
    "SNOW",
    "CTRA",
    "RR",
    "TSCO",
    "NVDA",
    "GS",
    "MBOT",
    "ENR.DE",
    "KGC",
]


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def load_popular_symbols() -> List[str]:
    """Load the current popular-symbol base from SQLite, with a safe fallback."""
    try:
        symbols = load_symbols_from_txt("popular_symbols.txt", use_sqlite=True)
        if symbols:
            return list(dict.fromkeys(symbols))
    except Exception as exc:
        print(f"⚠️ Impossible de charger la base populaire depuis SQLite: {exc}")
    return FALLBACK_SYMBOLS[:]


def load_personal_symbols() -> List[str]:
    """Load personal symbols (mes_symbols)."""
    try:
        symbols = load_symbols_from_txt("mes_symbols.txt", use_sqlite=True)
        if symbols:
            return list(dict.fromkeys(symbols))
    except Exception as exc:
        print(f"⚠️ Impossible de charger mes symboles: {exc}")
    return []


def load_coko_symbols() -> List[str]:
    """Load coko symbols, preferring SQLite list_type='coko' when available."""
    try:
        from symbol_manager import get_symbols_by_list_type  # lazy import

        symbols = get_symbols_by_list_type("coko", active_only=True)
        if symbols:
            return list(dict.fromkeys(symbols))
    except Exception:
        pass

    try:
        file_path = SRC_DIR / "coko_symbols.txt"
        if file_path.exists():
            values = [line.strip().upper() for line in file_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            return list(dict.fromkeys(values))
    except Exception as exc:
        print(f"⚠️ Impossible de charger les symboles coko: {exc}")
    return []


def build_symbol_metrics(symbols: List[str], period: str = "15mo") -> Dict[str, Dict[str, float]]:
    """Compute the same UI metrics that feed the comparison table."""
    best_params = extract_best_parameters()

    # Silence noisy internal prints so we can control display format here.
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        result = analyse_signaux_populaires(
            symbols,
            [],
            period=period,
            afficher_graphiques=False,
            chunk_size=len(symbols) if symbols else 1,
            verbose=False,
            save_csv=False,
            plot_all=False,
            max_workers=min(8, len(symbols)) if symbols else 1,
            taux_reussite_min=0,
            min_holding_days=7,
        )

    signals = result.get("signals", []) if isinstance(result, dict) else []
    backtest_results = result.get("backtest_results", []) if isinstance(result, dict) else []
    backtest_map = {row["Symbole"]: row for row in backtest_results if isinstance(row, dict) and row.get("Symbole")}

    metrics: Dict[str, Dict[str, float]] = {}
    for signal in signals:
        try:
            symbol = signal.get("Symbole")
            if not symbol:
                continue

            score_context = resolve_symbol_scoring_context(
                symbol,
                domaine=signal.get("Domaine", "Inconnu"),
                cap_range=signal.get("CapRange", "Unknown"),
                best_params=best_params,
            )
            bt = backtest_map.get(symbol, {})
            score_value = safe_float(signal.get("Score", 0.0), 0.0)
            seuil_achat = safe_float(score_context["seuil_achat"], 4.2)
            seuil_vente = safe_float(score_context["seuil_vente"], -0.5)
            score_seuil = (
                score_value / seuil_achat
                if score_value >= 0 and seuil_achat
                else (score_value / seuil_vente if score_value < 0 and seuil_vente else 0.0)
            )

            metrics[symbol] = {
                "Symbole": symbol,
                "Signal": signal.get("Signal", "N/A"),
                "Score": score_value,
                "Prix": safe_float(signal.get("Prix", 0.0), 0.0),
                "Tendance": signal.get("Tendance", "N/A"),
                "RSI": safe_float(signal.get("RSI", 0.0), 0.0),
                "Volume moyen($)": safe_float(signal.get("Volume moyen", 0.0), 0.0),
                "Domaine": score_context["domaine"],
                "Cap Range": score_context["cap_range"],
                "Score/Seuil": score_seuil,
                "Fiabilité (%)": safe_float(bt.get("taux_reussite", signal.get("Fiabilite", 0.0)), 0.0),
                "Nb Trades": int(bt.get("trades", signal.get("NbTrades", 0)) or 0),
                "Gagnants": int(bt.get("gagnants", signal.get("Gagnants", 0)) or 0),
                "Rev. Growth (%)": safe_float(signal.get("Rev. Growth (%)", 0.0), 0.0),
                "EBITDA Yield (%)": safe_float(signal.get("EBITDA Yield (%)", 0.0), 0.0),
                "FCF Yield (%)": safe_float(signal.get("FCF Yield (%)", 0.0), 0.0),
                "D/E Ratio": safe_float(signal.get("D/E Ratio", 0.0), 0.0),
                "Market Cap (B$)": safe_float(signal.get("Market Cap (B$)", 0.0), 0.0),
                "ROE (%)": safe_float(signal.get("ROE (%)", 0.0), 0.0),
                "dPrice": safe_float(signal.get("dPrice", 0.0), 0.0),
                "Var5j (%)": safe_float(signal.get("Var5j (%)", 0.0), 0.0),
                "dRSI": safe_float(signal.get("dRSI", 0.0), 0.0),
                "dVolRel": safe_float(signal.get("dVolRel", 0.0), 0.0),
                "Gain total ($)": safe_float(bt.get("gain_total", signal.get("Gain total ($)", 0.0)), 0.0),
                "Gain moyen ($)": safe_float(bt.get("gain_moyen", signal.get("Gain moyen ($)", 0.0)), 0.0),
                "Consensus": signal.get("Consensus", "N/A"),
            }
        except Exception as exc:
            print(f"⚠️ {signal.get('Symbole', '?')}: impossible de calculer les métriques ({exc})")

    return metrics


def rank_symbols(metrics: Dict[str, Dict[str, float]]) -> Tuple[List[str], Dict[str, float]]:
    """Reproduce the UI comparison ranking used in the Comparisons tab."""
    criteria_config = [
        {"key": "Score", "order": "desc"},
        {"key": "Score/Seuil", "order": "desc"},
        {"key": "Fiabilité (%)", "order": "desc"},
        {"key": "Nb Trades", "order": "desc"},
        {"key": "Gagnants", "order": "desc"},
        {"key": "Rev. Growth (%)", "order": "desc"},
        {"key": "EBITDA Yield (%)", "order": "desc"},
        {"key": "FCF Yield (%)", "order": "desc"},
        {"key": "D/E Ratio", "order": "asc"},
        {"key": "Market Cap (B$)", "order": "desc"},
        {"key": "ROE (%)", "order": "desc"},
        {"key": "dPrice", "order": "desc"},
        {"key": "Var5j (%)", "order": "asc"},
        {"key": "dRSI", "order": "asc"},
        {"key": "dVolRel", "order": "desc"},
        {"key": "Gain total ($)", "order": "desc"},
        {"key": "Gain moyen ($)", "order": "desc"},
    ]

    selected_symbols = [sym for sym in metrics if sym in metrics]
    n_stocks = len(selected_symbols)
    m_criteria = len(criteria_config)
    max_points = max(n_stocks * m_criteria, 1)

    points_by_symbol = {sym: 0 for sym in selected_symbols}

    for criterion in criteria_config:
        key = criterion["key"]
        reverse = criterion["order"] == "desc"
        ranked = sorted(
            selected_symbols,
            key=lambda sym: safe_float(metrics[sym].get(key, 0.0), 0.0),
            reverse=reverse,
        )
        for rank_idx, sym in enumerate(ranked):
            points_by_symbol[sym] += (n_stocks - rank_idx)

    pertinence_scores = {
        sym: (points_by_symbol[sym] / max_points) * 100.0
        for sym in selected_symbols
    }

    sorted_symbols = sorted(
        selected_symbols,
        key=lambda sym: (
            pertinence_scores[sym],
            metrics[sym]["Score/Seuil"],
            metrics[sym]["Score"],
        ),
        reverse=True,
    )
    return sorted_symbols, pertinence_scores


def _format_int_with_sep(value: float) -> str:
    try:
        return f"{int(float(value)):,}"
    except Exception:
        return "0"


def print_header_with_score_seuil() -> None:
    print(
        f"{'Symbole':<8} {'Signal':<8} {'Score':<7} {'Prix':<10} {'Tendance':<10} "
        f"{'RSI':<6} {'Volume':<15} {'Domaine':<16} {'Cap Range':<10} {'Score/Seuil':<12}"
    )
    print("-" * 114)


def print_symbol_row(data: Dict[str, float]) -> None:
    print(
        f"{str(data.get('Symbole', '')):<8} "
        f"{str(data.get('Signal', 'N/A')):<8} "
        f"{safe_float(data.get('Score', 0.0), 0.0):<7.2f} "
        f"{safe_float(data.get('Prix', 0.0), 0.0):<10.2f} "
        f"{str(data.get('Tendance', 'N/A')):<10} "
        f"{safe_float(data.get('RSI', 0.0), 0.0):<6.1f} "
        f"{_format_int_with_sep(data.get('Volume moyen($)', 0.0)):<15} "
        f"{str(data.get('Domaine', 'Unknown')):<16} "
        f"{str(data.get('Cap Range', 'Unknown')):<10} "
        f"{safe_float(data.get('Score/Seuil', 0.0), 0.0):<12.3f}"
    )


def print_metrics_table(metrics: Dict[str, Dict[str, float]], title: str, max_rows: int = 10) -> None:
    if not metrics:
        print(f"\n{title}: aucune ligne exploitable.")
        return

    ordered = sorted(
        metrics.values(),
        key=lambda row: (safe_float(row.get("Score/Seuil", 0.0), 0.0), safe_float(row.get("Score", 0.0), 0.0)),
        reverse=True,
    )

    print(f"\n{title}")
    print_header_with_score_seuil()
    for row in ordered[: max(1, max_rows)]:
        print_symbol_row(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank popular symbols using the UI comparison logic.")
    parser.add_argument("--period", default="18mo", help="Analysis period (default: 18mo)")
    parser.add_argument(
        "--symbols",
        default="",
        help="Optional comma-separated symbol list. If omitted, use the popular-symbol base from SQLite.",
    )
    parser.add_argument("--top", type=int, default=5, help="Number of winning popular symbols to find")
    parser.add_argument("--batch-size", type=int, default=40, help="Popular symbols processed per iteration")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Optional cap on scanned candidates (0 = all). Useful to avoid very long runs.",
    )
    parser.add_argument(
        "--reference-source",
        choices=["custom", "mes", "coko"],
        default="custom",
        help="Reference list to outperform: custom, mes, or coko (default: custom).",
    )
    args = parser.parse_args()

    if args.symbols.strip():
        custom_symbols = [sym.strip().upper() for sym in args.symbols.split(",") if sym.strip()]
    else:
        custom_symbols = FALLBACK_SYMBOLS[:]

    popular_symbols = [sym.strip().upper() for sym in load_popular_symbols() if sym and sym.strip()]
    personal_symbols = [sym.strip().upper() for sym in load_personal_symbols() if sym and sym.strip()]
    coko_symbols = [sym.strip().upper() for sym in load_coko_symbols() if sym and sym.strip()]

    reference_map = {
        "custom": custom_symbols,
        "mes": personal_symbols,
        "coko": coko_symbols,
    }
    reference_symbols = list(
        dict.fromkeys([sym for sym in reference_map.get(args.reference_source, []) if sym and sym.strip()])
    )

    if not reference_symbols:
        print("Aucune base de comparaison (custom/mes/coko) disponible.")
        return 1

    candidates = [sym for sym in popular_symbols if sym not in set(reference_symbols)]
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]

    print(f"📚 Base populaire: {len(popular_symbols)} symboles")
    print(
        f"🧪 Référence comparaison ({args.reference_source}): {len(reference_symbols)} symboles "
        f"[custom={len(custom_symbols)}, mes={len(personal_symbols)}, coko={len(coko_symbols)}]"
    )
    print(f"🎯 Candidats populaires hors référence: {len(candidates)}")

    ref_metrics = build_symbol_metrics(reference_symbols, period=args.period)
    if not ref_metrics:
        print("Aucune donnée exploitable pour la base de comparaison.")
        return 1

    print_metrics_table(
        ref_metrics,
        title=f"Table référence ({args.reference_source}) - triée par Score/Seuil",
        max_rows=12,
    )

    target_count = max(1, int(args.top))
    batch_size = max(1, int(args.batch_size))
    winners: List[Tuple[str, float, Dict[str, float]]] = []  # rank #1 only
    secondary_top3: List[Tuple[str, int, float, Dict[str, float]]] = []  # ranks #2/#3
    seen_winners: set[str] = set()
    seen_secondary: set[str] = set()

    interrupted = False
    try:
        for start in range(0, len(candidates), batch_size):
            if len(winners) >= target_count:
                break

            batch = candidates[start : start + batch_size]
            batch_metrics = build_symbol_metrics(batch, period=args.period)

            for symbol in batch:
                if len(winners) >= target_count:
                    break
                if symbol not in batch_metrics:
                    continue

                cohort = dict(ref_metrics)
                cohort[symbol] = batch_metrics[symbol]
                ranked, pertinence = rank_symbols(cohort)

                if not ranked:
                    continue

                # Keep primary objective unchanged: only rank #1 fills winners and controls stop.
                if ranked[0] == symbol and symbol not in seen_winners:
                    winners.append((symbol, pertinence[symbol], batch_metrics[symbol]))
                    seen_winners.add(symbol)
                    continue

                # Secondary list: symbols that reached top 3 without being first.
                if symbol in ranked[:3] and symbol not in seen_secondary:
                    rank_pos = ranked.index(symbol) + 1
                    if rank_pos in (2, 3):
                        secondary_top3.append((symbol, rank_pos, pertinence[symbol], batch_metrics[symbol]))
                        seen_secondary.add(symbol)

            scanned = min(start + batch_size, len(candidates))
            found_symbols = ", ".join([w[0] for w in winners]) if winners else "aucun"
            print(
                f"🔎 Candidats scannés: {scanned}/{len(candidates)} | gagnants trouvés: {len(winners)}/{target_count} "
                f"| déjà trouvés: [{found_symbols}]"
            )
    except KeyboardInterrupt:
        interrupted = True
        print("\n⛔ Analyse interrompue par l'utilisateur. Affichage des résultats partiels...")

    if interrupted:
        print(
            f"ℹ️ Résultat partiel: {len(winners)} dominant(s) rang #1 et {len(secondary_top3)} symbole(s) rang #2/#3."
        )
    elif len(winners) < target_count:
        print(
            f"⚠️ Seulement {len(winners)} symbole(s) trouvé(s) qui dominent la base {args.reference_source} "
            f"sur la période {args.period}."
        )
    else:
        print(f"\n✅ Arrêt après {target_count} symboles gagnants (période {args.period}).")

    print("\n" + "="*120)
    print("Top symboles populaires DOMINANTS (Rang #1):")
    print("="*120)
    print_header_with_score_seuil()
    for symbol, pert, data in winners:
        row_data = dict(data)
        row_data["Symbole"] = symbol
        print_symbol_row(row_data)
        print(
            f"   -> pertinence={pert:.1f}% | "
            f"fiabilité={safe_float(data.get('Fiabilité (%)', 0.0), 0.0):.1f}% | "
            f"trades={int(data.get('Nb Trades', 0) or 0)}"
        )

    if secondary_top3:
        print("\n" + "="*120)
        print("Symboles populaires en TOP 3 (Rang #2-#3) :")
        print("="*120)
        print_header_with_score_seuil()
        for symbol, rank, pert, data in secondary_top3:
            row_data = dict(data)
            row_data["Symbole"] = symbol
            print_symbol_row(row_data)
            print(
                f"   -> rang=#{rank} | pertinence={pert:.1f}% | "
                f"fiabilité={safe_float(data.get('Fiabilité (%)', 0.0), 0.0):.1f}% | "
                f"trades={int(data.get('Nb Trades', 0) or 0)}"
            )

    if winners:
        top_symbol, top_pert, top_data = winners[0]
        print("\n" + "="*120)
        print("🏆 Meilleur symbole trouvé (Rang #1) :")
        print("="*120)
        print(
            f"{top_symbol} | pertinence={top_pert:.1f}% | "
            f"score={top_data['Score']:.3f} | score/seuil={top_data['Score/Seuil']:.3f} | "
            f"fiabilité={top_data['Fiabilité (%)']:.1f}%"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())