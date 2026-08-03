"""
Verrouillage de la fonction ensure_instrument_profiles et de sa gestion des profils d'instruments.
"""

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import market_store
from market_store import (
    ensure_instrument_profiles,
    ensure_market_data_schema,
)


def test_returns_zero_when_disable_flag_set(monkeypatch, tmp_path) -> None:
    """Verrouille : avec QSI_DISABLE_PROFILE_FETCH=1, recuperes == 0 et ignores = manquants."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.setenv("QSI_DISABLE_PROFILE_FETCH", "1")

    ensure_market_data_schema()
    test_symbols = ["AAPL", "MSFT"]

    result = ensure_instrument_profiles(test_symbols, max_fetch=10, max_age_days=90)

    assert result["recuperes"] == 0
    assert result["manquants"] == len(test_symbols)
    assert result["ignores"] == len(test_symbols)


def test_respects_max_fetch_cap(monkeypatch, tmp_path) -> None:
    """Verrouille : avec max_fetch=3 et 10 symboles verrouillés, recuperes == 3 et ignores == 7."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()
    stale_date = (datetime.utcnow() - timedelta(days=200)).isoformat()

    test_symbols = [f"SYM{i}" for i in range(10)]
    for symbol in test_symbols:
        row = {
            "symbol": symbol,
            "name": f"Company {symbol}",
            "short_name": f"Com {symbol}",
            "long_name": f"Company {symbol}",
            "sector": "Technology",
            "industry": "Software",
            "country": "US",
            "exchange": "NASDAQ",
            "currency": "USD",
            "fx_rate_to_usd": 1.0,
            "values_in_usd": 1,
            "quote_type": "EQUITY",
            "shares_outstanding": 1000000.0,
            "market_cap": 1000000000.0,
            "market_cap_usd": 1000000000.0,
            "enterprise_value": 900000000.0,
            "enterprise_value_usd": 900000000.0,
            "first_seen_at": stale_date,
            "last_profile_refresh": stale_date,
            "source": "yfinance",
        }
        df = pd.DataFrame([row])
        instrument_path = tmp_path / "instruments" / f"symbol={symbol}" / "part0.parquet"
        instrument_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(instrument_path))

    mock_ticker_info = {
        "shortName": "Test Corp",
        "longName": "Test Corporation",
        "sector": "Technology",
        "industry": "Software",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "quoteType": "EQUITY",
        "marketCap": 2000000000.0,
        "enterpriseValue": 1800000000.0,
        "sharesOutstanding": 2000000.0,
    }

    def mock_ticker(symbol):
        return MagicMock(info=mock_ticker_info)

    with patch("market_store.yf.Ticker", side_effect=mock_ticker):
        result = ensure_instrument_profiles(test_symbols, max_fetch=3, max_age_days=90)

    assert result["recuperes"] == 3
    assert result["ignores"] == 7
    assert result["manquants"] == 10


def test_skips_symbols_already_present_and_fresh(monkeypatch, tmp_path) -> None:
    """Verrouille : un symbole frais (< max_age_days) n'est pas rechargé."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()
    fresh_date = datetime.utcnow().isoformat()
    test_symbol = "FRESH"

    row = {
        "symbol": test_symbol,
        "name": "Fresh Corp",
        "short_name": "Fresh",
        "long_name": "Fresh Corporation",
        "sector": "Technology",
        "industry": "Software",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "fx_rate_to_usd": 1.0,
        "values_in_usd": 1,
        "quote_type": "EQUITY",
        "shares_outstanding": 1000000.0,
        "market_cap": 1000000000.0,
        "market_cap_usd": 1000000000.0,
        "enterprise_value": 900000000.0,
        "enterprise_value_usd": 900000000.0,
        "first_seen_at": fresh_date,
        "last_profile_refresh": fresh_date,
        "source": "yfinance",
    }

    df = pd.DataFrame([row])
    instrument_path = tmp_path / "instruments" / f"symbol={test_symbol}" / "part0.parquet"
    instrument_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(instrument_path))

    call_count = 0
    def mock_ticker(symbol):
        nonlocal call_count
        call_count += 1
        return MagicMock(info={})

    with patch("market_store.yf.Ticker", side_effect=mock_ticker):
        result = ensure_instrument_profiles([test_symbol], max_age_days=90)

    assert call_count == 0
    assert result["recuperes"] == 0


def test_refetches_stale_profile(monkeypatch, tmp_path) -> None:
    """Verrouille : un symbole verrouillé (> max_age_days) est rechargé."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()
    stale_date = (datetime.utcnow() - timedelta(days=200)).isoformat()
    test_symbol = "STALE"

    row = {
        "symbol": test_symbol,
        "name": "Stale Corp",
        "short_name": "Stale",
        "long_name": "Stale Corporation",
        "sector": "Technology",
        "industry": "Software",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "fx_rate_to_usd": 1.0,
        "values_in_usd": 1,
        "quote_type": "EQUITY",
        "shares_outstanding": 1000000.0,
        "market_cap": 1000000000.0,
        "market_cap_usd": 1000000000.0,
        "enterprise_value": 900000000.0,
        "enterprise_value_usd": 900000000.0,
        "first_seen_at": stale_date,
        "last_profile_refresh": stale_date,
        "source": "yfinance",
    }

    df = pd.DataFrame([row])
    instrument_path = tmp_path / "instruments" / f"symbol={test_symbol}" / "part0.parquet"
    instrument_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(instrument_path))

    mock_ticker_info = {
        "shortName": "Stale Updated",
        "longName": "Stale Corporation Updated",
        "sector": "Technology",
        "industry": "Software",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "quoteType": "EQUITY",
        "marketCap": 2000000000.0,
        "enterpriseValue": 1800000000.0,
        "sharesOutstanding": 2000000.0,
    }

    def mock_ticker(symbol):
        return MagicMock(info=mock_ticker_info)

    with patch("market_store.yf.Ticker", side_effect=mock_ticker):
        result = ensure_instrument_profiles([test_symbol], max_age_days=90)

    assert result["recuperes"] == 1
    assert result["ignores"] == 0


def test_one_failing_symbol_does_not_stop_the_others(monkeypatch, tmp_path) -> None:
    """Verrouille : une exception sur un symbole n'arrête pas le traitement des autres."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()
    stale_date = (datetime.utcnow() - timedelta(days=200)).isoformat()
    test_symbols = ["FAIL", "GOOD1", "GOOD2"]

    for symbol in test_symbols:
        row = {
            "symbol": symbol,
            "name": f"Corp {symbol}",
            "short_name": f"Cor {symbol}",
            "long_name": f"Corporation {symbol}",
            "sector": "Technology",
            "industry": "Software",
            "country": "US",
            "exchange": "NASDAQ",
            "currency": "USD",
            "fx_rate_to_usd": 1.0,
            "values_in_usd": 1,
            "quote_type": "EQUITY",
            "shares_outstanding": 1000000.0,
            "market_cap": 1000000000.0,
            "market_cap_usd": 1000000000.0,
            "enterprise_value": 900000000.0,
            "enterprise_value_usd": 900000000.0,
            "first_seen_at": stale_date,
            "last_profile_refresh": stale_date,
            "source": "yfinance",
        }
        df = pd.DataFrame([row])
        instrument_path = tmp_path / "instruments" / f"symbol={symbol}" / "part0.parquet"
        instrument_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), str(instrument_path))

    mock_ticker_info = {
        "shortName": "Good Corp",
        "longName": "Good Corporation",
        "sector": "Technology",
        "industry": "Software",
        "country": "US",
        "exchange": "NASDAQ",
        "currency": "USD",
        "quoteType": "EQUITY",
        "marketCap": 2000000000.0,
        "enterpriseValue": 1800000000.0,
        "sharesOutstanding": 2000000.0,
    }

    def mock_ticker(symbol):
        if symbol == "FAIL":
            raise Exception("Network error")
        return MagicMock(info=mock_ticker_info)

    with patch("market_store.yf.Ticker", side_effect=mock_ticker):
        result = ensure_instrument_profiles(test_symbols, max_fetch=25, max_age_days=90)

    assert result["recuperes"] == 2
    assert result["ignores"] == 0
    # L'échec doit être compté, pas seulement absorbé : un compteur muet
    # rendrait indiscernable « rien à faire » de « tout a échoué ».
    assert result["echecs"] == 1
    # Et les deux symboles sains doivent avoir été réellement écrits.
    assert market_store._instrument_path("GOOD1").exists()
    assert market_store._instrument_path("GOOD2").exists()


def test_empty_symbol_list_is_a_noop(monkeypatch, tmp_path) -> None:
    """Verrouille : liste vide ne produit aucun appel yfinance."""
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()

    call_count = 0
    def mock_ticker(symbol):
        nonlocal call_count
        call_count += 1
        return MagicMock(info={})

    with patch("market_store.yf.Ticker", side_effect=mock_ticker):
        result = ensure_instrument_profiles([], max_fetch=25, max_age_days=90)

    assert call_count == 0
    assert result["recuperes"] == 0
    assert result["manquants"] == 0
    assert result["ignores"] == 0


def test_profil_sans_identite_nest_pas_ecrit(monkeypatch, tmp_path) -> None:
    """Verrouille : un `info` yfinance sans aucun nom ne devient pas un profil.

    Mesure du 2026-08-03 : les tickers Finviz corrompus (« FFUTU », « IINTU »,
    « ZZBAO »…) renvoyaient un `info` non vide mais sans identite (ni shortName
    ni longName), et 18 profils fantomes ont ete ecrits dans le store, avec
    `name` egal au ticker corrompu. Ils comptaient ensuite comme profils frais,
    donc n'etaient jamais rafraichis, et polluaient les colonnes Nom / Pays.
    """
    monkeypatch.setattr(market_store, "PARQUET_DIR", tmp_path)
    monkeypatch.delenv("QSI_DISABLE_PROFILE_FETCH", raising=False)

    ensure_market_data_schema()

    infos = {
        "REAL": {"shortName": "Real Corp", "exchange": "NASDAQ", "currency": "USD"},
        # Exactement la forme observee pour un ticker inexistant.
        "FFUTU": {"currency": "USD", "trailingPegRatio": None},
        "ZZS": {"exchange": "NMS", "currency": "USD", "quoteType": "EQUITY"},
        # Pseudo-fonds « YHD » au nom numerique, l'autre forme de reponse
        # yfinance sur un symbole inexistant (AABT -> « 164 »).
        "AABT": {"shortName": "164", "exchange": "YHD", "quoteType": "MUTUALFUND"},
    }

    with patch("market_store.yf.Ticker", side_effect=lambda s: MagicMock(info=infos[s])):
        result = ensure_instrument_profiles(list(infos), max_fetch=10, max_age_days=90)

    assert result["recuperes"] == 1
    assert result["echecs"] == 3
    assert market_store._instrument_path("REAL").exists()
    assert not market_store._instrument_path("FFUTU").exists()
    assert not market_store._instrument_path("ZZS").exists()
    assert not market_store._instrument_path("AABT").exists()
