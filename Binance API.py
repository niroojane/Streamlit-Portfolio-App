#!/usr/bin/env python
# coding: utf-8

# In[5]:


from binance.spot import Spot
#from binance.client import Client
import pandas as pd
import requests
import datetime


# In[3]:


def daterange(start_date, end_date,interval=30):
    for n in range(0,int((end_date - start_date).days),interval):
        yield start_date + datetime.timedelta(n)

class BinanceAPI:
    
    def __init__(self,binance_api_key,binance_api_secret):
        
        self.binance_api_key=binance_api_key
        self.binance_api_secret=binance_api_secret
        
        self.binance_api=Spot(self.binance_api_key,self.binance_api_secret)
        
        #self.binance_api_client=Client(self.binance_api_key,self.binance_api_secret)
        
    def get_market_cap(self,quote="USDT"):

        
        resp = requests.get("https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products")

        market_cap=pd.DataFrame(resp.json()['data'])
        market_cap=market_cap[market_cap['q']==quote]

        market_cap=market_cap[['an','qn','s','b','q','c','cs']]
        market_cap['c']=market_cap['c'].astype(float)

        market_cap.columns=['Long name','Quote Name','Ticker','Short Name','Quote Short Name','Close','Supply']
        market_cap['Market Cap']=market_cap['Close']*market_cap['Supply']

        market_cap=market_cap.sort_values(by='Market Cap',ascending=False)


        return market_cap


    def get_price(self,ticker_list,date=datetime.datetime.today()):
        
        timestamp_sec=int((date-datetime.timedelta(1)).timestamp()*1000)
        price=pd.DataFrame()

        for ticker in ticker_list:
            try:
                
                temp=pd.DataFrame()
                numeric_columns =  ['Open Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 
                                'Number of Trades', 'TB Base Volume', 'TB Quote Volume', 'Ignore']

                if ticker!='USDTUSDT':
                    
                    data = pd.DataFrame(self.binance_api.klines(ticker,"1d", startTime=timestamp_sec),columns=numeric_columns)
                    data['Close Time']=pd.to_datetime(data['Close Time'], unit='ms')
                    data=data.set_index('Close Time')
                    temp[ticker]=data['Close'].astype(float)
                    price=pd.concat([price,temp[ticker]],axis=1)
                    
                else:
                    
                    price[ticker]=1

            except Exception as e:

                print(ticker +" not retrieved")

                pass

        price.index=pd.to_datetime(price.index).strftime('%Y-%m-%d')

        return price

    def get_inventory(self):
        
        
        ptf=pd.DataFrame(self.binance_api.user_asset())
        ptf['Ticker']=ptf['asset']+"USDT"

        ticker=ptf['Ticker'].to_list()
        ptf=ptf.set_index('Ticker')

        price=self.get_price(ptf.index).T
        price.columns=['Price']

        data=pd.concat([price,ptf],axis=1)
        data['Price in USDT']=data['Price']*(data['free'].astype(float)+data['locked'].astype(float))

        inventory=pd.DataFrame(data['Price in USDT'])
        inventory['Weights']=inventory['Price in USDT']/inventory['Price in USDT'].sum()

        inventory.loc['Total']=inventory.sum()

        return inventory.sort_values(by='Weights',ascending=False)
    
    def get_positions_history(self,enddate=datetime.datetime.today()):
    
    #,startdate=datetime.datetime(2024,5,5)):
        
        dt = enddate
        startdate=enddate-datetime.timedelta(30)

        timestamp_sec = dt.timestamp()
        timestamp_end = int(timestamp_sec * 1000)
    
        snapshots=self.binance_api.account_snapshot(type='SPOT',limit=30,endTime=timestamp_end)
        all_key=snapshots['snapshotVos']  
        
        #snapshots=[]

        #for date in daterange(startdate,enddate):

        #    print(date)

        #    timestamp_sec = date.timestamp()
        #    timestamp_end = int(timestamp_sec * 1000)
        #    snapshot=self.binance_api.account_snapshot(type='SPOT',limit=30,startTime=timestamp_end)
        #    snapshots.extend(snapshot['snapshotVos'])
        
        #all_key=(snapshots)
        history={}
        tickers=set()

        for i in range(len(all_key)):

            history[all_key[i]['updateTime']]=all_key[i]['data']['balances']

        for key in history.keys():
            holding=history[key]

            for i in range(len(holding)):
                tickers.add(holding[i]['asset']+"USDT")

        tickers=list(tickers)
        holdings={}

        for key in history.keys():

            temp={}
            date_key=pd.to_datetime(key,unit='ms').strftime('%Y-%m-%d')

            for i in range(len(history[key])):

                temp[history[key][i]['asset']]=float(history[key][i]['free'])+float(history[key][i]['locked'])

            #holdings[key]=temp

            holdings[date_key]=temp

        quantities=pd.DataFrame(holdings).T
        #quantities.index=pd.to_datetime(quantities.index,unit="ms").strftime(date_format='%Y-%m-%d')
        #quantities.index=pd.to_datetime(quantities.index)
        quantities.columns=quantities.columns+'USDT'
        #quantities=quantities.astype(float)


        crypto=quantities.columns

        prices=self.get_price(crypto,startdate)
        
        positions=pd.DataFrame()
        
        for col in crypto:
            
            try:
                positions[col]=quantities[col]*prices.loc[quantities.index][col]
            except Exception as e:
            
                print(col)
                
        return positions,quantities
    

    def get_trades(self,symbols):
            
            trades={}
            history=pd.DataFrame()
            
            for symbol in symbols:
                trades[symbol]=self.binance_api.my_trades(symbol=symbol)
            
            for key in trades:
                
                history=pd.concat([history,pd.DataFrame(trades[key])])
                
            history['time']=pd.to_datetime(history['time'],unit='ms')
            
                
            return history


# In[ ]:




