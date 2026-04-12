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



