#!/usr/bin/env python3
"""
Surveillance d'un screener Finviz (version gratuite).

Compare la composition actuelle du screener avec celle du dernier run
et notifie les entrees et les sorties.

Usage:
    python finviz_watch.py

Configuration par variables d'environnement (voir CONFIG ci-dessous).
"""

import json
import os
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------

# Votre screener. Remplacez par l'URL exacte copiee depuis votre navigateur.
FILTERS = ",".join([
    "cap_largeover",
    "fa_peg_u2",
    "fa_eps5years_pos",
    "fa_sales5years_o5",
    "fa_debteq_u1",
    "fa_grossmargin_o30",
    "ta_sma50_pa",
    "ta_highlow52w_b20h",  # a moins de 20% sous le plus haut 52 semaines
    "ta_beta_u1",
])
SCREENER_URL = os.environ.get(
    "FINVIZ_URL",
    f"https://finviz.com/screener.ashx?v=111&f={FILTERS}&o=ticker",
)

STATE_FILE = Path(os.environ.get("FINVIZ_STATE", "finviz_state.json"))

# Notification via ntfy.sh (gratuit, sans compte).
# Choisissez un nom de topic long et non devinable.
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")

# Notification par e-mail (SMTP). Laissez vide pour desactiver.
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_TO = os.environ.get("MAIL_TO", "")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


# --------------------------------------------------------------------------
# Recuperation
# --------------------------------------------------------------------------

class ScrapeError(RuntimeError):
    """Le parsing a echoue. On veut un echec bruyant, pas silencieux."""


def fetch_page(url: str, retries: int = 3) -> str:
    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as err:
            last_err = err
            time.sleep(5 * (attempt + 1))
    raise ScrapeError(f"Echec de la requete apres {retries} tentatives: {last_err}")


def parse_tickers(html: str) -> dict[str, str]:
    """Retourne {ticker: nom_societe}."""
    soup = BeautifulSoup(html, "html.parser")

    rows: dict[str, str] = {}
    # Ancrage principal : chaque cellule de ligne porte le ticker et le nom
    # de la societe en attributs data-boxover-*, independamment du habillage.
    for cell in soup.select("td[data-boxover-ticker]"):
        ticker = (cell.get("data-boxover-ticker") or "").strip()
        if not ticker or ticker in rows:
            continue
        rows[ticker] = (cell.get("data-boxover-company") or "").strip()

    if not rows:
        # Repli : les liens tickers de la table du screener. Finviz est passe
        # de quote.ashx?t=XXX a stock?t=XXX ; on accepte les deux formes.
        table = soup.select_one("table.screener_table")
        scope = table if table is not None else soup
        for link in scope.select('a.tab-link[href*="t="]'):
            ticker = link.get_text(strip=True)
            if not ticker or ticker in rows:
                continue
            # Le nom de la societe est dans la cellule suivante
            cell = link.find_parent("td")
            name = ""
            if cell is not None:
                nxt = cell.find_next_sibling("td")
                if nxt is not None:
                    name = nxt.get_text(strip=True)
            rows[ticker] = name

    if not rows:
        raise ScrapeError(
            "Aucun ticker trouve. La structure HTML de Finviz a probablement "
            "change, ou la requete a ete bloquee."
        )
    return rows


def fetch_all_tickers(base_url: str) -> dict[str, str]:
    """Gere la pagination Finviz (20 lignes par page avec v=111)."""
    all_rows: dict[str, str] = {}
    offset = 1
    while True:
        url = f"{base_url}&r={offset}"
        page = parse_tickers(fetch_page(url))
        new = {k: v for k, v in page.items() if k not in all_rows}
        all_rows.update(page)
        if not new or len(page) < 20:
            break
        offset += 20
        time.sleep(2)  # on reste poli
    return all_rows


# --------------------------------------------------------------------------
# Etat et diff
# --------------------------------------------------------------------------

def load_state() -> dict:
    if not STATE_FILE.exists():
        return {}
    return json.loads(STATE_FILE.read_text(encoding="utf-8"))


def save_state(rows: dict[str, str]) -> None:
    payload = {
        "updated": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tickers": rows,
    }
    STATE_FILE.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


# --------------------------------------------------------------------------
# Notification
# --------------------------------------------------------------------------

def notify(title: str, body: str) -> None:
    sent = False

    if NTFY_TOPIC:
        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=body.encode("utf-8"),
                headers={"Title": title, "Priority": "default"},
                timeout=15,
            )
            sent = True
        except requests.RequestException as err:
            print(f"[warn] ntfy a echoue: {err}", file=sys.stderr)

    if SMTP_HOST and MAIL_TO:
        import smtplib
        from email.message import EmailMessage

        msg = EmailMessage()
        msg["Subject"] = title
        msg["From"] = SMTP_USER
        msg["To"] = MAIL_TO
        msg.set_content(body)
        try:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as smtp:
                smtp.starttls()
                smtp.login(SMTP_USER, SMTP_PASS)
                smtp.send_message(msg)
            sent = True
        except Exception as err:  # noqa: BLE001
            print(f"[warn] SMTP a echoue: {err}", file=sys.stderr)

    if not sent:
        print(f"\n=== {title} ===\n{body}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> int:
    try:
        current = fetch_all_tickers(SCREENER_URL)
    except ScrapeError as err:
        # Echec bruyant: on notifie, sinon on croira qu'il n'y a aucun mouvement
        notify("Finviz watch: ERREUR", str(err))
        print(f"[error] {err}", file=sys.stderr)
        return 1

    state = load_state()
    previous = state.get("tickers")

    if previous is None:
        save_state(current)
        print(f"Etat initial enregistre: {len(current)} valeurs")
        print("  " + ", ".join(sorted(current)))
        return 0

    entered = sorted(set(current) - set(previous))
    exited = sorted(set(previous) - set(current))

    if not entered and not exited:
        print(f"Aucun mouvement ({len(current)} valeurs)")
        save_state(current)
        return 0

    lines = []
    if entered:
        lines.append("ENTREES:")
        lines += [f"  + {t}  {current[t]}" for t in entered]
    if exited:
        if lines:
            lines.append("")
        lines.append("SORTIES:")
        lines += [f"  - {t}  {previous[t]}" for t in exited]
    lines.append("")
    lines.append(f"Composition actuelle ({len(current)}): " + ", ".join(sorted(current)))

    body = "\n".join(lines)
    title = f"Finviz: {len(entered)} entree(s), {len(exited)} sortie(s)"
    notify(title, body)
    print(body)

    save_state(current)
    return 0


if __name__ == "__main__":
    sys.exit(main())