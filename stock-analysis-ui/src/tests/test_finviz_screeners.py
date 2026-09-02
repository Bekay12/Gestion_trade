"""
Verrouillage du parsing des tickers Finviz (core/finviz_screeners.py).

Depuis la refonte de son screener, Finviz place un avatar-lettre dans la cellule
Ticker, en plus du lien portant le vrai symbole :

    <td data-boxover-ticker="IESC">
      <a class="company-ticker"><img src=".../IESC.svg"/><span>I</span></a>
      <a class="tab-link">IESC</a>
    </td>

finvizfinance 1.3.0 lit `td.text`, qui concatène TOUT le texte de la cellule et
renvoie donc « IIESC ». Chaque symbole partait avec sa première lettre doublée et
échouait ensuite côté yfinance (« No history available for FFUTU »), en plus de
polluer le store de profils d'instruments.

Aucun accès réseau : `web_scrap` est remplacé par un HTML figé, copie de la
structure réellement observée le 2026-08-03.
"""
import pytest
from bs4 import BeautifulSoup

from core import finviz_screeners


def _cellule_ticker(symbole: str, *, avec_attribut: bool = True,
                    classe_avatar: str = "company-ticker") -> str:
    """Reproduit une cellule Ticker Finviz : avatar-lettre + lien du symbole."""
    attribut = f' data-boxover-ticker="{symbole}"' if avec_attribut else ""
    return (
        f'<td align="left"{attribut} height="10"><span class="flex items-center">'
        f'<a class="{classe_avatar}" href="stock?t={symbole}">'
        f'<img alt="{symbole} logo" src="https://logo.finviz.com/{symbole}.svg"/>'
        f'<span>{symbole[0]}</span></a>'
        f'<a class="tab-link" href="stock?t={symbole}">{symbole}</a></span></td>'
    )


def _page_finviz(lignes) -> BeautifulSoup:
    """Page screener minimale : sélecteur de pages + table `screener_table`.

    Entrees:
        lignes (list[tuple]): (symbole, nom, kwargs de _cellule_ticker)
    """
    entetes = ("No.", "Ticker", "Company", "Sector", "Industry", "Country",
               "Market Cap", "P/E", "Price", "Change", "Volume")
    th = "".join(f"<th>{h}</th>" for h in entetes)
    trs = []
    for rang, (symbole, nom, kwargs) in enumerate(lignes, start=1):
        trs.append(
            f"<tr><td>{rang}</td>"
            + _cellule_ticker(symbole, **kwargs)
            + f"<td><a class='tab-link'>{nom}</a></td>"
            + "<td>Technology</td><td>Software</td><td>USA</td>"
            + "<td>12.34B</td><td>25.10</td><td>101.50</td><td>5.32%</td><td>1500000</td></tr>"
        )
    html = (
        '<html><body><select id="pageSelect"><option>1</option></select>'
        '<table class="screener_table"><tr>' + th + "</tr>"
        + "".join(trs)
        + "</table></body></html>"
    )
    return BeautifulSoup(html, "lxml")


@pytest.fixture
def page_finviz(monkeypatch):
    """Remplace web_scrap par une page figée. Retourne un poseur de lignes."""
    import finvizfinance.screener.base as base

    def _poser(lignes):
        soup = _page_finviz(lignes)
        monkeypatch.setattr(base, "web_scrap", lambda url, params: soup)

    return _poser


def test_avatar_lettre_ignore_par_lattribut_du_td(page_finviz) -> None:
    """L'attribut data-boxover-ticker fait autorité : « IESC », pas « IIESC »."""
    page_finviz([("IESC", "IES Holdings Inc", {}),
                 ("AMZN", "Amazon.com Inc", {})])

    df = finviz_screeners.run_screen({"Beta": "Under 1"})

    assert df["Ticker"].tolist() == ["IESC", "AMZN"]


def test_avatar_lettre_ignore_sans_lattribut_du_td(page_finviz) -> None:
    """Sans l'attribut, l'avatar est retiré du DOM avant lecture du texte."""
    page_finviz([("NVDA", "NVIDIA Corp", {"avec_attribut": False}),
                 ("ULTA", "Ulta Beauty Inc", {"avec_attribut": False})])

    df = finviz_screeners.run_screen({"Beta": "Under 1"})

    assert df["Ticker"].tolist() == ["NVDA", "ULTA"]


def test_ticker_legitime_a_lettre_doublee_preserve(page_finviz) -> None:
    """AAPL ne doit pas être « réparé » en APL : la correction n'est pas
    heuristique. Ce cas est le piège de toute réparation par suppression du
    premier caractère."""
    page_finviz([("AAPL", "Apple Inc", {}),
                 ("MMM", "3M Co", {"avec_attribut": False}),
                 ("TTWO", "Take-Two Interactive", {})])

    df = finviz_screeners.run_screen({"Beta": "Under 1"})

    assert df["Ticker"].tolist() == ["AAPL", "MMM", "TTWO"]


def test_structure_inconnue_leve_une_erreur_explicite(page_finviz) -> None:
    """Si Finviz change encore de structure (ni attribut, ni classe connue), il
    faut échouer bruyamment : une liste de tickers corrompue coûte des requêtes
    yfinance inutiles et pollue le store de profils."""
    page_finviz([("FUTU", "Futu Holdings", {"avec_attribut": False, "classe_avatar": "logo-v3"}),
                 ("INTU", "Intuit Inc", {"avec_attribut": False, "classe_avatar": "logo-v3"}),
                 ("VEEV", "Veeva Systems", {"avec_attribut": False, "classe_avatar": "logo-v3"})])

    with pytest.raises(RuntimeError, match="structure"):
        finviz_screeners.run_screen({"Beta": "Under 1"})


def test_run_preset_expose_le_nom_de_lentreprise(page_finviz) -> None:
    """La colonne « Nom » du dialog vient de la réponse Finviz : 0 requête de plus."""
    page_finviz([("IESC", "IES Holdings Inc", {})])

    res = finviz_screeners.run_preset("gap_up")

    assert res["headers"][:2] == ["Symbole", "Nom"]
    assert res["rows"][0][:2] == ["IESC", "IES Holdings Inc"]
