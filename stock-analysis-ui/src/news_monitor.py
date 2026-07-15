"""
news_monitor.py
---------------
Surveille les flux d'actualités financières, analyse l'impact sur le portfolio
via Ollama (LLM local), et log les alertes dans news_alerts.csv.

Usage :
    python news_monitor.py            # tourne en boucle continue
    python news_monitor.py --once     # une seule vérification
    python news_monitor.py --interval 600  # toutes les 10 minutes (défaut: 15 min)
"""

import argparse
import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import feedparser
import requests

# ─── Configuration ────────────────────────────────────────────────────────────

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "deepseek-coder-v2:16b"  # modèle le plus capable disponible

BASE_DIR = Path(__file__).parent
PORTFOLIO_FILE = BASE_DIR / "mes_symbols.txt"
ALERTS_CSV = BASE_DIR / "news_alerts.csv"
SEEN_IDS_FILE = BASE_DIR / ".news_seen_ids.json"

DEFAULT_INTERVAL_SEC = 900  # 15 minutes

# Flux RSS financiers (sans clé API)
RSS_FEEDS = {
    "Reuters Finance":   "https://feeds.reuters.com/reuters/businessNews",
    "Yahoo Finance":     "https://finance.yahoo.com/news/rssindex",
    "Seeking Alpha":     "https://seekingalpha.com/feed.xml",
    "MarketWatch":       "https://feeds.marketwatch.com/marketwatch/topstories/",
    "CNBC":              "https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=100003114",
    "FT Markets":        "https://www.ft.com/rss/home/uk",
    "Bloomberg Markets": "https://feeds.bloomberg.com/markets/news.rss",
}

# Correspondance secteur → tickers de votre portfolio
SECTOR_TICKERS = {
    "semiconductor": ["NVDA", "AVGO", "MU", "MRVL", "NXP", "ALAB"],
    "tech":          ["NVDA", "AVGO", "MU", "MRVL", "NXP", "ALAB", "SEMR"],
    "healthcare":    ["AGEN", "ALDX", "IMNM", "MBOT"],
    "biotech":       ["AGEN", "ALDX", "IMNM", "MBOT", "BCRX"],
    "energy":        ["ENR.DE", "H2O.DE"],
    "finance":       ["GS", "DB"],
    "gold":          ["KGC"],
    "crypto":        ["MSTR"],
    "agriculture":   ["CF"],
    "hospitality":   ["H"],
}

# ─── Fonctions utilitaires ─────────────────────────────────────────────────────

def load_portfolio() -> list[str]:
    if not PORTFOLIO_FILE.exists():
        return []
    return [line.strip() for line in PORTFOLIO_FILE.read_text().splitlines() if line.strip()]


def load_seen_ids() -> set:
    if SEEN_IDS_FILE.exists():
        return set(json.loads(SEEN_IDS_FILE.read_text()))
    return set()


def save_seen_ids(ids: set):
    # Garder uniquement les 2000 derniers IDs pour éviter la croissance infinie
    trimmed = list(ids)[-2000:]
    SEEN_IDS_FILE.write_text(json.dumps(trimmed))


def fetch_news(max_per_feed: int = 5) -> list[dict]:
    """Récupère les dernières actualités des flux RSS."""
    articles = []
    for source, url in RSS_FEEDS.items():
        try:
            feed = feedparser.parse(url)
            for entry in feed.entries[:max_per_feed]:
                article_id = entry.get("id") or entry.get("link", "")
                articles.append({
                    "id": article_id,
                    "source": source,
                    "title": entry.get("title", ""),
                    "summary": entry.get("summary", entry.get("description", ""))[:500],
                    "link": entry.get("link", ""),
                    "published": entry.get("published", ""),
                })
        except Exception as e:
            print(f"[WARN] Erreur flux {source}: {e}")
    return articles


def analyze_with_llm(article: dict, portfolio: list[str]) -> dict | None:
    """
    Envoie l'article à Ollama et demande une analyse d'impact sur le portfolio.
    Retourne None si l'article n'est pas pertinent.
    """
    prompt = f"""Tu es un analyste financier expert. Analyse cette actualité et évalue son impact potentiel sur les actions suivantes du portfolio.

ACTUALITÉ:
Titre: {article['title']}
Résumé: {article['summary']}
Source: {article['source']}

PORTFOLIO: {', '.join(portfolio)}

Réponds UNIQUEMENT en JSON valide avec cette structure exacte:
{{
  "pertinent": true/false,
  "resume_impact": "1-2 phrases max expliquant l'impact principal",
  "tickers_affectes": ["TICKER1", "TICKER2"],
  "secteurs_affectes": ["secteur1", "secteur2"],
  "sentiment": "positif" | "negatif" | "neutre",
  "intensite": "faible" | "modere" | "fort",
  "raisonnement": "explication courte (max 3 phrases)"
}}

Si l'actualité n'a AUCUN impact sur le portfolio, réponds avec pertinent: false et les autres champs vides.
Réponds uniquement avec le JSON, sans texte avant ou après."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1, "num_predict": 512},
            },
            timeout=120,
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # Extraire le JSON de la réponse
        start = raw.find("{")
        end = raw.rfind("}") + 1
        if start == -1 or end == 0:
            return None

        result = json.loads(raw[start:end])
        if not result.get("pertinent"):
            return None
        return result

    except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
        print(f"[WARN] Erreur LLM pour '{article['title'][:60]}...': {e}")
        return None


def save_alert(article: dict, analysis: dict):
    """Sauvegarde une alerte dans news_alerts.csv et affiche dans le terminal."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    tickers = "|".join(analysis.get("tickers_affectes", []))
    secteurs = "|".join(analysis.get("secteurs_affectes", []))

    row = {
        "date": now,
        "source": article["source"],
        "titre": article["title"],
        "sentiment": analysis.get("sentiment", ""),
        "intensite": analysis.get("intensite", ""),
        "tickers_affectes": tickers,
        "secteurs_affectes": secteurs,
        "resume_impact": analysis.get("resume_impact", ""),
        "raisonnement": analysis.get("raisonnement", ""),
        "lien": article["link"],
    }

    file_exists = ALERTS_CSV.exists()
    with open(ALERTS_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # Affichage terminal coloré
    sentiment_icon = {"positif": "↑", "negatif": "↓", "neutre": "→"}.get(
        analysis.get("sentiment", ""), "?"
    )
    intensite_icon = {"faible": "●", "modere": "●●", "fort": "●●●"}.get(
        analysis.get("intensite", ""), ""
    )
    print(f"\n{'='*70}")
    print(f"  {sentiment_icon} [{analysis.get('intensite','?').upper()}] {intensite_icon}  {article['source']}")
    print(f"  {article['title'][:70]}")
    print(f"  Impact: {analysis.get('resume_impact', '')}")
    print(f"  Tickers: {tickers or 'N/A'}  |  Secteurs: {secteurs or 'N/A'}")
    print(f"  {article['link']}")
    print(f"{'='*70}")


# ─── Boucle principale ─────────────────────────────────────────────────────────

def run_once():
    portfolio = load_portfolio()
    if not portfolio:
        print("[ERROR] Portfolio vide. Vérifiez mes_symbols.txt")
        return

    seen_ids = load_seen_ids()
    articles = fetch_news()
    new_articles = [a for a in articles if a["id"] not in seen_ids]

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {len(articles)} articles récupérés, "
          f"{len(new_articles)} nouveaux à analyser.")

    alerts_count = 0
    for article in new_articles:
        seen_ids.add(article["id"])
        if not article["title"].strip():
            continue

        print(f"  Analyse: {article['title'][:65]}...", end="\r")
        analysis = analyze_with_llm(article, portfolio)
        if analysis:
            save_alert(article, analysis)
            alerts_count += 1

    save_seen_ids(seen_ids)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {alerts_count} alerte(s) générée(s). "
          f"Résultats dans: {ALERTS_CSV}")


def run_loop(interval: int):
    print(f"[INFO] Surveillance démarrée. Vérification toutes les {interval//60} min.")
    print(f"[INFO] Portfolio: {load_portfolio()}")
    print(f"[INFO] Modèle LLM: {OLLAMA_MODEL}")
    print(f"[INFO] Alertes → {ALERTS_CSV}")
    print("[INFO] Ctrl+C pour arrêter.\n")

    while True:
        try:
            run_once()
            print(f"[INFO] Prochaine vérification dans {interval//60} min...\n")
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\n[INFO] Arrêt du moniteur.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")
            time.sleep(60)


# ─── Point d'entrée ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Moniteur d'actualités financières avec LLM local")
    parser.add_argument("--once", action="store_true", help="Exécuter une seule fois puis quitter")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC,
                        help="Intervalle en secondes entre les vérifications (défaut: 900)")
    parser.add_argument("--model", type=str, default=OLLAMA_MODEL,
                        help="Modèle Ollama à utiliser")
    args = parser.parse_args()

    OLLAMA_MODEL = args.model

    if args.once:
        run_once()
    else:
        run_loop(args.interval)
