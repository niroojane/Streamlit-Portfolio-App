# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.
"""
Every yfinance call used by the app lives here:

  search_yahoo            -- symbol search
  fetch_ticker_details    -- name, exchange, currency, volume, market cap... (cached)
  has_price_history       -- does yfinance return prices for this symbol?
  fetch_ticker / get_close-- daily adjusted closes
  get_currencies          -- quote currency of each ticker (cached)
  get_fx_rates            -- daily FX to a base currency, incl. minor units (GBp)
  to_base_currency        -- convert a price table to one currency
"""
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import yfinance as yf


# quoteType -> asset class used in the universe table (anything else: 'Other')
QUOTE_TYPE_CLASS = {
    'EQUITY': 'Equity', 'ETF': 'ETF', 'MUTUALFUND': 'Fund', 'INDEX': 'Index',
    'FUTURE': 'Commodities', 'CURRENCY': 'FX', 'CRYPTOCURRENCY': 'Crypto',
}

# Yahoo quotes some markets in the minor unit: London in pence ('GBp'/'GBX'),
# Johannesburg in cents ('ZAc'), Tel Aviv in agorot ('ILA').
MINOR_UNITS = {'GBp': ('GBP', 0.01), 'GBX': ('GBP', 0.01), 'ZAc': ('ZAR', 0.01), 'ILA': ('ILS', 0.01)}

# Quote currencies Yahoo reports wrongly, checked against other listings of the same asset.
# GOLD.PA (Amundi Physical Gold, Euronext Paris) is labelled USD but trades in EUR: its
# returns follow the EUR gold ETCs (e.g. 4GLD.DE), not the USD ones.
CURRENCY_OVERRIDES = {'GOLD.PA': 'EUR'}

# Predefined universes (asset class -> tickers) of the Multi Asset App, the HMM Regime App
# and the web app. UCITS: funds an EU investor can buy, USD lines on London (cleaner Yahoo
# data than the EUR lines on Xetra). US ETFs: the US equivalents the regime research was
# done on (more history), kept to reproduce it; most can't be bought from the EU (PRIIPs).
UCITS_UNIVERSE = {
    'Commodities': ['ISLN.L', 'GOLD.PA', 'BRNT.L', 'CMOD.L'],
    'Bonds': ['CBU0.L', 'CBU7.L'],
    'Credit': ['LQDE.L'],
    'T-Bills': ['IB01.L'],
    'Equity': ['CSPX.L', 'IUUS.L', 'IUIT.L', 'SMH.L', 'IUCS.L', 'IUCD.L', 'IUIS.L', 'WMAT.L', 'IUES.L',
               'GDIG.L', 'EXV5.DE', 'IUHC.L', 'IUFS.L'],
}
US_ETF_UNIVERSE = {
    'Commodities': ['ISLN.L', 'GOLD.PA', 'BZ=F', 'DBC'],
    'Bonds': ['IEF', 'IEI'],
    'Credit': ['LQD'],
    'T-Bills': ['BIL'],
    'Equity': ['^GSPC', 'IDU', 'XLK', 'SOXX', 'XLP', 'XLY', 'IYJ', 'MXI', 'IYE',
               'PICK', 'EXV5.DE', 'IUHC.L', 'IYG', 'XSW'],
}
UNIVERSES = {'UCITS ETFs': UCITS_UNIVERSE, 'US ETFs (research)': US_ETF_UNIVERSE}
DEFAULT_UNIVERSE = 'UCITS ETFs'

_details_cache = {}
_currency_cache = {}


# -----------------------------------------------------------------------------
# Search & details
# -----------------------------------------------------------------------------
def search_yahoo(query, max_results=10):
    """Yahoo Finance symbol search -> DataFrame(Symbol, Name, Type, Exchange).

    Uses yf.Search (yfinance >= 0.2.44). If that isn't available or finds
    nothing, the query itself is tried as a ticker.
    """
    query = str(query).strip()
    cols = ['Symbol', 'Name', 'Type', 'Exchange']
    if not query:
        return pd.DataFrame(columns=cols)
    rows = []
    try:
        quotes = yf.Search(query, max_results=max_results, news_count=0).quotes
        for q in quotes:
            if not q.get('symbol'):
                continue
            rows.append({'Symbol': q['symbol'],
                         'Name': q.get('longname') or q.get('shortname') or '',
                         'Type': q.get('quoteType', ''),
                         'Exchange': q.get('exchDisp') or q.get('exchange', '')})
    except Exception:
        pass
    if not rows:
        d = fetch_ticker_details([query.upper()]).iloc[0]
        if d.notna().sum() > 1:
            rows.append({'Symbol': query.upper(), 'Name': d.get('Name', ''),
                         'Type': d.get('Type', ''), 'Exchange': d.get('Exchange', '')})
    return pd.DataFrame(rows, columns=cols)


def _one_ticker_details(ticker):
    """Details for one ticker from yf.Ticker.info, falling back to fast_info."""
    t = yf.Ticker(ticker)
    try:
        info = t.info or {}
    except Exception:
        info = {}
    price = info.get('regularMarketPrice') or info.get('currentPrice') or info.get('previousClose')
    avg_vol = info.get('averageVolume') or info.get('averageDailyVolume3Month')
    expense = info.get('netExpenseRatio')
    if expense is None and info.get('annualReportExpenseRatio') is not None:
        expense = info['annualReportExpenseRatio'] * 100        # given as a fraction
    d = {
        'Name': info.get('longName') or info.get('shortName'),
        'Type': info.get('quoteType'),
        'Exchange': info.get('fullExchangeName') or info.get('exchange'),
        'Currency': info.get('currency'),
        'Sector / Category': info.get('sector') or info.get('category'),
        'Industry': info.get('industry'),
        'Price': price,
        'Avg Volume (3M)': avg_vol,
        'Avg Volume (10D)': info.get('averageDailyVolume10Day') or info.get('averageVolume10days'),
        'Market Cap': info.get('marketCap'),
        'Fund Size': info.get('totalAssets'),
        'Expense Ratio %': expense,
        'Dividend Yield %': info.get('dividendYield') or (info['yield'] * 100 if info.get('yield') else None),
        '52W High': info.get('fiftyTwoWeekHigh'),
        '52W Low': info.get('fiftyTwoWeekLow'),
    }
    if d['Price'] is None or d['Currency'] is None:
        # info can be rate-limited or empty for some symbols; fast_info is lighter
        try:
            fi = t.fast_info
            d['Price'] = d['Price'] or fi.get('lastPrice')
            d['Currency'] = d['Currency'] or fi.get('currency')
            d['Exchange'] = d['Exchange'] or fi.get('exchange')
            d['Type'] = d['Type'] or fi.get('quoteType')
            d['Market Cap'] = d['Market Cap'] or fi.get('marketCap')
            d['Avg Volume (3M)'] = d['Avg Volume (3M)'] or fi.get('threeMonthAverageVolume')
            d['Avg Volume (10D)'] = d['Avg Volume (10D)'] or fi.get('tenDayAverageVolume')
            d['52W High'] = d['52W High'] or fi.get('yearHigh')
            d['52W Low'] = d['52W Low'] or fi.get('yearLow')
        except Exception:
            pass
    d['Currency'] = CURRENCY_OVERRIDES.get(ticker, d['Currency'])
    if d['Price'] and d['Avg Volume (3M)']:
        d['Avg Daily Value'] = d['Price'] * d['Avg Volume (3M)']    # liquidity, in Currency
    if d['Currency']:
        _currency_cache[ticker] = d['Currency']
    return d


def fetch_ticker_details(tickers_, refresh=False):
    """Details table (one row per ticker) from yfinance, fetched in parallel and cached."""
    tickers_ = list(dict.fromkeys(tickers_))
    todo = [t for t in tickers_ if refresh or t not in _details_cache]
    if todo:
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            futures = {ex.submit(_one_ticker_details, t): t for t in todo}
            for f in as_completed(futures):
                try:
                    _details_cache[futures[f]] = f.result()
                except Exception:
                    _details_cache[futures[f]] = {}
    out = pd.DataFrame({t: _details_cache.get(t, {}) for t in tickers_}).T.reindex(tickers_)
    text_cols = ['Name', 'Type', 'Exchange', 'Currency', 'Sector / Category', 'Industry']
    for c in out.columns:
        if c not in text_cols:
            out[c] = pd.to_numeric(out[c], errors='coerce')
    return out.dropna(axis=1, how='all')


# -----------------------------------------------------------------------------
# Prices
# -----------------------------------------------------------------------------
def has_price_history(ticker, period='1mo'):
    """True if yfinance returns at least one recent price for `ticker`."""
    try:
        return not yf.Ticker(ticker).history(period=period, auto_adjust=True).empty
    except Exception:
        return False


def fetch_ticker(ticker, start, end=None):
    """Daily adjusted close of one ticker as a Series named `ticker`, or None.

    `Close` with auto_adjust=True is already split- and dividend-adjusted,
    so dividends are not added back. The exchange timezone is dropped and
    dates normalised, so tickers from different exchanges line up by day.
    `end` is inclusive.
    """
    end = end or datetime.date.today()
    try:
        hist = yf.Ticker(ticker).history(start=start, end=pd.Timestamp(end) + pd.Timedelta(days=1),
                                         interval='1d', auto_adjust=True)
        if hist.empty:
            return None
        s = hist['Close'].copy()
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
        s.index = s.index.normalize()
        s = s[~s.index.duplicated(keep='last')]
        return s.rename(ticker)
    except Exception:
        return None


def get_close(tickers, start=None, end=None, max_workers=10, ffill_limit=5, on_progress=None):
    """Adjusted closes for several tickers, one column each.

    Default window is the last year (computed at call time). Short gaps from
    different exchange holidays are forward-filled (`ffill_limit` days).
    `on_progress(ticker, ok)` is called as each download finishes.
    Tickers yfinance can't price are left out (see .attrs['failed']).
    """
    tickers = list(dict.fromkeys(tickers))
    end = end or datetime.date.today()
    start = start or (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
    series, failed = [], []
    if tickers:
        with ThreadPoolExecutor(max_workers=min(max_workers, len(tickers))) as executor:
            futures = {executor.submit(fetch_ticker, t, start, end): t for t in tickers}
            for future in as_completed(futures):
                t = futures[future]
                s = future.result()
                (series.append(s) if s is not None else failed.append(t))
                if on_progress:
                    on_progress(t, s is not None)
    data = pd.concat(series, axis=1).sort_index() if series else pd.DataFrame()
    if not data.empty and ffill_limit:
        data = data.ffill(limit=ffill_limit)
    data.attrs['failed'] = sorted(failed)
    return data


# -----------------------------------------------------------------------------
# Currencies
# -----------------------------------------------------------------------------
def major_currency(ccy):
    """'GBp' -> ('GBP', 0.01); 'EUR' -> ('EUR', 1.0)."""
    return MINOR_UNITS.get(ccy, (ccy, 1.0))


def get_currencies(tickers_):
    """Quote currency of each ticker as yfinance reports it (e.g. 'USD', 'EUR', 'GBp').

    Uses fast_info (cheap), cached; None where yfinance can't tell.
    CURRENCY_OVERRIDES win over what yfinance reports.
    """
    tickers_ = list(dict.fromkeys(tickers_))
    todo = [t for t in tickers_ if t not in _currency_cache]

    def one(t):
        if t in CURRENCY_OVERRIDES:
            return CURRENCY_OVERRIDES[t]
        try:
            return yf.Ticker(t).fast_info.get('currency')
        except Exception:
            return None

    if todo:
        with ThreadPoolExecutor(max_workers=min(8, len(todo))) as ex:
            for t, ccy in zip(todo, ex.map(one, todo)):
                if ccy:
                    _currency_cache[t] = ccy
    return pd.Series({t: _currency_cache.get(t) for t in tickers_}, dtype=object)


def get_fx_rates(currencies, base='USD', start=None, end=None):
    """Daily conversion rates: units of `base` per 1 unit of each currency.

    Handles minor units (GBp = GBP / 100). Returns a DataFrame with one column
    per requested currency (always including `base` = 1.0). Currencies
    yfinance has no rate for are listed in .attrs['missing'] and omitted.
    """
    end = end or datetime.date.today()
    start = start or (pd.Timestamp(end) - pd.DateOffset(years=1)).date()
    wanted = sorted({c for c in currencies if isinstance(c, str) and c})
    majors = sorted({major_currency(c)[0] for c in wanted} - {base})
    symbols = {f"{m}{base}=X": m for m in majors}
    raw = get_close(list(symbols), start=start, end=end) if symbols else pd.DataFrame()
    raw = raw.rename(columns=symbols)

    idx = raw.index if not raw.empty else pd.DatetimeIndex(pd.bdate_range(start, end))
    out = pd.DataFrame(index=idx)
    missing = []
    for c in set(wanted) | {base}:
        major, factor = major_currency(c)
        if major == base:
            out[c] = factor
        elif major in raw.columns:
            out[c] = raw[major] * factor
        else:
            missing.append(c)
    out = out.sort_index().ffill().bfill()
    out.attrs['missing'] = sorted(missing)
    return out


def to_base_currency(prices, currencies, fx):
    """Multiply each price column by its currency's daily rate from `fx`.

    `currencies`: ticker -> quote currency. Columns whose currency has no
    rate are returned unchanged and listed in .attrs['unconverted'].
    """
    out = prices.copy()
    rates = fx.reindex(fx.index.union(out.index)).sort_index().ffill().bfill().reindex(out.index)
    unconverted = []
    for c in out.columns:
        ccy = currencies.get(c)
        if ccy in rates.columns:
            out[c] = out[c] * rates[ccy]
        else:
            unconverted.append(c)
    out.attrs['unconverted'] = unconverted
    return out
