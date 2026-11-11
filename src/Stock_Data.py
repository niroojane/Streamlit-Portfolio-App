# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    info = {}
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
    if d['Price'] and d['Avg Volume (3M)']:
        d['Avg Daily Value'] = d['Price'] * d['Avg Volume (3M)']    # liquidity, in Currency
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


def fetch_ticker(ticker, start, end):
    try:
        yahoo_data = yf.Ticker(ticker)
        stock_price = yahoo_data.history(start=start, end=end, interval='1d').reset_index()

        stock_price['Date'] = stock_price['Date'].dt.tz_localize(None)
        stock_price = stock_price.set_index('Date')

        stock_price[ticker] = stock_price['Close'] + stock_price['Dividends'].shift(periods=-1)

        return stock_price[[ticker]]

    except Exception:
        return None


def get_close(
    tickers,
    start=datetime.date(datetime.date.today().year - 1,
                        datetime.date.today().month,
                        datetime.date.today().day),
    end=datetime.date.today(),
    max_workers=10
):
    data = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_ticker, ticker, start, end): ticker
            for ticker in tickers
        }

        for future in as_completed(futures):
            result = future.result()

            if result is not None:
                data = pd.concat([data, result], axis=1)

    return data



