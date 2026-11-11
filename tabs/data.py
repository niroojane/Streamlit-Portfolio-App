# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.

"""Data loaders of the apps, shown at the top of the Investment Universe tab.

Each loader stores the universe with ui.set_universe (prices + profile +
optional exposure table) and returns True once data is available.
"""
import datetime
import pandas as pd
import streamlit as st

from . import engine, ui


# =============================================================================
# Excel workbook (Streamlit_App)
# =============================================================================

@st.cache_data(show_spinner=False)
def _read_workbook(content: bytes):
    return engine.read_workbook(content)


def excel_loader() -> bool:
    uploaded = st.file_uploader(
        "Upload an Excel file with time series", type="xlsx",
        help="Sheet 1: dates in the first column, one column per asset (prices or daily returns). "
             "Optional sheet 2: exposure table (buckets in rows, assets in columns, in % or decimals).")
    if uploaded is None:
        if 'prices' not in st.session_state:
            st.info("Upload a workbook to start ⬆️")
            return False
        return True

    try:
        raw, exposure, messages = _read_workbook(uploaded.getvalue())
    except Exception as exc:
        st.error(f"Could not read the workbook: {exc}")
        return False
    for message in messages:
        st.caption(message)

    assets = list(raw.columns)
    default = assets.index('BTCUSDT') if 'BTCUSDT' in assets else 0
    benchmark = st.selectbox("Benchmark asset", assets, index=default,
                             help="Held buy-and-hold as the 'Benchmark' series of the backtest and risk tabs")
    profile = engine.multi_asset(benchmark)
    prices, messages = engine.clean_prices(raw, profile)
    ui.show_messages(messages)
    exposure = engine.Exposure.read_exposure(exposure, list(prices.columns)) if exposure is not None else None
    ui.set_universe(prices, profile, exposure)
    return True


# =============================================================================
# Binance universe (Minimum Variance App, crypto)
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def _market_cap():
    from src.Price_Endpoint import get_market_cap
    return get_market_cap()


@st.cache_data(ttl=3600, show_spinner=False)
def binance_prices(tickers: tuple, start: datetime.date, today: datetime.date) -> pd.DataFrame:
    """Daily closes from Binance's public market data (src.Price_Endpoint.get_closes);
    `today` only refreshes the cache once a day"""
    from src.Price_Endpoint import get_closes
    return get_closes(list(tickers), start)


def binance_loader() -> bool:
    """The Crypto App notebook's universe: [Number of Crypto][Starting Date][Get Prices],
    the market-cap table of the top N and one checkbox per ticker"""
    try:
        market_cap = _market_cap()
    except Exception as exc:
        st.error(f"Binance market caps unavailable: {exc}")
        return 'prices' in st.session_state
    # Tokenized stocks ('... (bStocks）' -- Binance closes it with a full-width parenthesis) are not crypto
    market_cap = market_cap[~market_cap['Long name'].str.contains('(bStocks', regex=False, na=False)]
    market_cap = market_cap.set_index('Ticker')

    col1, col2, col3 = st.columns([3, 2, 1], vertical_alignment="bottom")
    number = col1.slider("Number of Crypto", min_value=1, max_value=100, value=20)
    start = col2.date_input("Starting Date of Backtest", datetime.date(2020, 1, 1))
    get_prices = col3.button("Get Prices", type="primary", width='stretch')

    top = market_cap.iloc[:number]
    ui.table(top, decimals=2, height=320)
    st.markdown("### Selected Tickers")
    tickers = list(top.index)
    selected = []
    for row in range(0, len(tickers), 5):
        for column, ticker in zip(st.columns(5), tickers[row:row + 5]):
            if column.checkbox(ticker, value=True, key=f"ticker_{ticker}"):
                selected.append(ticker)

    if get_prices:
        if not selected:
            st.warning("Select at least one crypto")
        else:
            raw = ui.run("Loading prices...", binance_prices, tuple(selected), start, datetime.date.today())
            if raw is not None and not raw.empty:
                failed = raw.attrs.get('failed', [])
                prices, messages = engine.clean_prices(raw, engine.CRYPTO)
                ui.show_messages(messages + ([f"Not retrieved: {', '.join(failed)}"] if failed else []))
                ui.set_universe(prices, engine.CRYPTO)
                st.success(f"✅ Loaded prices for {prices.shape[1]} assets from {prices.index[0].date()} "
                           f"to {prices.index[-1].date()}")

    if 'prices' not in st.session_state:
        st.info("Click 'Get Prices' to load the universe ⬆️")
        return False
    return True
