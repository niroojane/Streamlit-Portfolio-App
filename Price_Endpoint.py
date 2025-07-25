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
