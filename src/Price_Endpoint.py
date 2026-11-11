# Copyright (c) 2025 Niroojane Selvam
# Licensed under the MIT License. See LICENSE file in the project root for full license information.


import requests
import pandas as pd
import datetime

def get_price(tickers_list,date=datetime.datetime.today(),interval="1d"):
    numeric_columns =  ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                                'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']
    price=pd.DataFrame()
    timestamp_sec=int((date-datetime.timedelta(1)).timestamp()*1000)
    url='https://api.binance.us/api/v3/klines?'
    
    for ticker in tickers_list:
    
        try:
            if ticker!='USDTUSDT':
    
                temp_url=f"https://api.binance.us/api/v3/klines?symbol={ticker}&interval={interval}&startTime={timestamp_sec}&limit=1000"
                re=requests.get(temp_url)
                data=pd.DataFrame(re.json())
                data.columns=numeric_columns
                data['Close Time']=pd.to_datetime(data['Close Time'], unit='ms')
                data=data.set_index('Close Time')
                temp=pd.DataFrame()
                temp[ticker]=data['Close'].astype(float)
                price=pd.concat([price,temp[ticker]],axis=1)
    
            else:
                    
                price[ticker]=1
    
        except Exception as e:
    
            print(ticker +" not retrieved")
    
            pass
    
    price.index=pd.to_datetime(price.index).strftime('%Y-%m-%d')

    return price



def get_market_cap(quote="USDT"):


    resp = requests.get("https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products")

    market_cap=pd.DataFrame(resp.json()['data'])
    market_cap=market_cap[market_cap['q']==quote]

    market_cap=market_cap[['an','qn','s','b','q','c','cs']]
    market_cap['c']=market_cap['c'].astype(float)

    market_cap.columns=['Long name','Quote Name','Ticker','Short Name','Quote Short Name','Close','Supply']
    market_cap['Market Cap']=market_cap['Close']*market_cap['Supply']

    market_cap=market_cap.sort_values(by='Market Cap',ascending=False)


    return market_cap


# ---------------------------------------------------------------------------
# Daily closes from Binance's public market data (no API key), as the Crypto App
# notebook's Binance.get_price_threading. data-api.binance.vision serves the
# binance.com order books from any region (api.binance.com is geo-blocked in the
# US, where Streamlit Cloud runs); binance.us is the last resort. Tickers none of
# them prices fall back to Yahoo ('BASE-USD').
# ---------------------------------------------------------------------------
import time
from concurrent.futures import ThreadPoolExecutor

KLINE_HOSTS = ('https://data-api.binance.vision', 'https://api.binance.com', 'https://api.binance.us')
CASH_TICKER = 'USDTUSDT'


def _klines(host, symbol, start_ms):
    """All daily klines of `symbol` from `start_ms`, 1000 per request"""
    rows = []
    while True:
        r = requests.get(f"{host}/api/v3/klines", timeout=15,
                         params={'symbol': symbol, 'interval': '1d', 'startTime': start_ms, 'limit': 1000})
        if r.status_code == 429:
            time.sleep(1)
            continue
        r.raise_for_status()
        batch = r.json()
        rows += batch
        if len(batch) < 1000:
            return rows
        start_ms = batch[-1][6] + 1          # after the last close time


def _closes(symbol, start_ms):
    for host in KLINE_HOSTS:
        try:
            rows = _klines(host, symbol, start_ms)
        except Exception:
            continue
        if rows:
            index = pd.to_datetime([row[6] for row in rows], unit='ms').normalize()
            return pd.Series([float(row[4]) for row in rows], index=index, name=symbol), host
    return None, None


def get_closes(tickers, start, quote='USDT'):
    """Daily closes (one column per ticker) from `start` (a date). attrs: 'failed', 'sources'."""
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    tickers = list(dict.fromkeys(tickers))
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = dict(zip(tickers, executor.map(
            lambda t: (None, 'cash') if t == CASH_TICKER else _closes(t, start_ms), tickers)))

    series = {t: s for t, (s, _) in results.items() if s is not None}
    missing = [t for t, (s, source) in results.items() if s is None and source != 'cash']
    sources = {t: source for t, (_, source) in results.items() if source}
    if missing:                               # last resort: Yahoo, BASE-USD
        from .Yahoo_API import get_close
        yahoo = {t: t[:-len(quote)] + '-USD' for t in missing if t.endswith(quote)}
        data = get_close(list(yahoo.values()), start=pd.Timestamp(start).date()) if yahoo else pd.DataFrame()
        for ticker, symbol in yahoo.items():
            if symbol in data.columns and data[symbol].notna().any():
                series[ticker] = data[symbol]
                sources[ticker] = 'yahoo'
    prices = pd.DataFrame(series).sort_index()
    prices = prices[~prices.index.duplicated(keep='last')]
    if CASH_TICKER in tickers:
        prices[CASH_TICKER] = 1.0
    prices = prices[[t for t in tickers if t in prices.columns]]
    prices.attrs['failed'] = [t for t in tickers if t not in prices.columns]
    prices.attrs['sources'] = sources
    return prices
